from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import torch
from tensor_core import CounterRng, RngKey

from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.runtime.sampling import SamplingRuntime
from tensor_dslab.readout.requirements import require_representable_float
from tensor_dslab.readout.charge.config import CorrelatedAvalancheConfig
from tensor_dslab.readout.charge.runtime.effects.counts import (
    MAX_COUNT,
    checked_add,
    checked_rate_product,
    draw_ordered_categories,
    original_positions,
    require_count_domain,
    require_tensor_allocation,
)
from tensor_dslab.readout.charge.runtime.effects.delays import (
    AfterpulseRuntime,
    DelayRuntime,
    prepare_afterpulse_recovery,
    prepare_delay,
    prepare_exponential_delay,
)


@final
@dataclass(frozen=True, slots=True)
class CorrelatedAvalancheRuntime:
    direct_crosstalk: DelayRuntime | None
    delayed_crosstalk: DelayRuntime | None
    afterpulse: AfterpulseRuntime | None
    ledger_depth: int
    ledger_bound: float
    maximum_generations: int
    sample_count: int
    tensor_numel: int
    direct_mean: float | None
    direct_retained_rng_key: RngKey | None
    direct_overflow_rng_key: RngKey | None
    delayed_mean: float | None
    delayed_retained_rng_key: RngKey | None
    delayed_overflow_rng_key: RngKey | None
    afterpulse_probability: float | None
    afterpulse_rng_key: RngKey | None


@dataclass(frozen=True, slots=True)
class _CorrelatedAvalancheResult:
    S1: torch.Tensor
    S2: torch.Tensor
    final_frontier: torch.Tensor
    total_count: torch.Tensor
    direct_crosstalk_count: torch.Tensor | None
    direct_crosstalk_overflow_count: torch.Tensor | None
    delayed_crosstalk_count: torch.Tensor | None
    delayed_crosstalk_overflow_count: torch.Tensor | None
    afterpulse_count: torch.Tensor | None
    afterpulse_overflow_count: torch.Tensor | None
    afterpulse_charge: torch.Tensor | None
    afterpulse_overflow_charge: torch.Tensor | None
    afterpulse_charge_square_sum: torch.Tensor | None


def prepare_ledger_envelope(
    *,
    floating_dtype: torch.dtype,
    maximum_generations: int,
    retained_mechanisms: int,
    recovered_afterpulse: bool,
    sample_count: int,
) -> tuple[int, float]:
    precision = 24 if floating_dtype is torch.float32 else 53
    depth = (
        retained_mechanisms * maximum_generations + sample_count + 3
        if recovered_afterpulse
        else retained_mechanisms * maximum_generations + 1
    )
    if depth >= 1 << precision:
        raise ValueError("correlated-avalanche ledger depth exceeds the dtype domain")
    gamma = depth / ((1 << precision) - depth)
    zero = torch.tensor(0.0, dtype=floating_dtype, device="cpu")
    one = torch.tensor(1.0, dtype=floating_dtype, device="cpu")
    subnormal = float(torch.nextafter(zero, one))
    bound = MAX_COUNT * (1.0 + gamma) + depth * subnormal
    if not math.isfinite(bound):
        raise ValueError("correlated-avalanche ledger bound is nonfinite")
    return depth, bound


def prepare_correlated_avalanches(
    config: CorrelatedAvalancheConfig,
    *,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
    tensor_numel: int,
) -> CorrelatedAvalancheRuntime:
    maximum_generations = config.maximum_generations.value
    sample_count = sampling.sample_count
    if maximum_generations == 0:
        _, bound = prepare_ledger_envelope(
            floating_dtype=floating_dtype,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=sample_count,
        )
        return CorrelatedAvalancheRuntime(
            direct_crosstalk=None,
            delayed_crosstalk=None,
            afterpulse=None,
            ledger_depth=1,
            ledger_bound=bound,
            maximum_generations=0,
            sample_count=sample_count,
            tensor_numel=tensor_numel,
            direct_mean=None,
            direct_retained_rng_key=None,
            direct_overflow_rng_key=None,
            delayed_mean=None,
            delayed_retained_rng_key=None,
            delayed_overflow_rng_key=None,
            afterpulse_probability=None,
            afterpulse_rng_key=None,
        )

    direct = (
        None
        if config.direct_crosstalk is None
        or config.direct_crosstalk.mean_offspring_per_parent.value == 0.0
        else prepare_delay(config.direct_crosstalk.delay, sampling=sampling)
    )
    delayed = (
        None
        if config.delayed_crosstalk is None
        or config.delayed_crosstalk.mean_offspring_per_parent.value == 0.0
        else prepare_delay(config.delayed_crosstalk.delay, sampling=sampling)
    )
    afterpulse: AfterpulseRuntime | None = None
    if config.afterpulse is not None and config.afterpulse.probability.value != 0.0:
        mean_delay_ns = canonical_magnitude(config.afterpulse.mean_delay)
        delay = prepare_exponential_delay(
            mean_delay_ns,
            sampling=sampling,
        )
        recovery: tuple[float, ...] | None = None
        overflow_recovery: tuple[float, ...] | None = None
        if config.afterpulse.recovery is not None:
            time_constant_ns = canonical_magnitude(
                config.afterpulse.recovery.time_constant
            )
            recovery, overflow_recovery = prepare_afterpulse_recovery(
                mean_delay_ns,
                time_constant_ns,
                sampling=sampling,
                delay=delay,
            )
            recovery = tuple(
                require_representable_float(
                    value,
                    dtype=floating_dtype,
                    field="afterpulse recovery weight",
                )
                for value in recovery
            )
            overflow_recovery = tuple(
                require_representable_float(
                    value,
                    dtype=floating_dtype,
                    field="afterpulse recovery weight",
                )
                for value in overflow_recovery
            )
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (*recovery, *overflow_recovery)
            ):
                raise ValueError("afterpulse recovery is invalid in the Charge dtype")
        afterpulse = AfterpulseRuntime(delay, recovery, overflow_recovery)

    if direct is not None or delayed is not None:
        if maximum_generations * tensor_numel > 1 << 63:
            raise ValueError("crosstalk address lattice exceeds its domain")
    if afterpulse is not None and (
        maximum_generations
        * (sampling.sample_count + 1)
        * tensor_numel
        > 1 << 63
    ):
        raise ValueError("afterpulse address lattice exceeds its domain")

    retained_mechanisms = sum(
        (
            direct is not None and any(direct.probabilities),
            delayed is not None and any(delayed.probabilities),
            afterpulse is not None and any(afterpulse.delay.probabilities),
        )
    )
    recovered_afterpulse = afterpulse is not None and afterpulse.recovery is not None
    depth, bound = prepare_ledger_envelope(
        floating_dtype=floating_dtype,
        maximum_generations=maximum_generations,
        retained_mechanisms=retained_mechanisms,
        recovered_afterpulse=recovered_afterpulse,
        sample_count=sample_count,
    )
    return CorrelatedAvalancheRuntime(
        direct_crosstalk=direct,
        delayed_crosstalk=delayed,
        afterpulse=afterpulse,
        ledger_depth=depth,
        ledger_bound=bound,
        maximum_generations=maximum_generations,
        sample_count=sample_count,
        tensor_numel=tensor_numel,
        direct_mean=(
            None
            if direct is None or config.direct_crosstalk is None
            else config.direct_crosstalk.mean_offspring_per_parent.value
        ),
        direct_retained_rng_key=(
            None
            if direct is None or config.direct_crosstalk is None
            else config.direct_crosstalk.retained_rng_key
        ),
        direct_overflow_rng_key=(
            None
            if direct is None or config.direct_crosstalk is None
            else config.direct_crosstalk.overflow_rng_key
        ),
        delayed_mean=(
            None
            if delayed is None or config.delayed_crosstalk is None
            else config.delayed_crosstalk.mean_offspring_per_parent.value
        ),
        delayed_retained_rng_key=(
            None
            if delayed is None or config.delayed_crosstalk is None
            else config.delayed_crosstalk.retained_rng_key
        ),
        delayed_overflow_rng_key=(
            None
            if delayed is None or config.delayed_crosstalk is None
            else config.delayed_crosstalk.overflow_rng_key
        ),
        afterpulse_probability=(
            None
            if afterpulse is None or config.afterpulse is None
            else config.afterpulse.probability.value
        ),
        afterpulse_rng_key=(
            None
            if afterpulse is None or config.afterpulse is None
            else config.afterpulse.rng_key
        ),
    )


def _draw_crosstalk(
    frontier: torch.Tensor,
    *,
    positions: torch.Tensor,
    runtime: DelayRuntime,
    mean: float,
    retained_key: RngKey,
    overflow_key: RngKey,
    generation_index: int,
    tensor_numel: int,
    rng: CounterRng,
    field: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count = frontier.shape[-1]
    basis = torch.zeros_like(frontier, dtype=torch.float64)
    for destination in range(sample_count):
        accumulated = basis[..., destination]
        for source in range(destination + 1):
            probability = runtime.probabilities[destination - source]
            if probability == 0.0:
                continue
            contribution = frontier[..., source].to(torch.float64) * probability
            accumulated = accumulated + contribution
        basis[..., destination] = accumulated
    retained_rate = checked_rate_product(basis, mean, field=f"{field} retained")

    overflow_basis = torch.zeros_like(frontier, dtype=torch.float64)
    for source in range(sample_count):
        probability = runtime.right_tails[sample_count - source]
        if probability != 0.0:
            overflow_basis[..., source] = (
                frontier[..., source].to(torch.float64) * probability
            )
    overflow_rate = checked_rate_product(
        overflow_basis,
        mean,
        field=f"{field} overflow",
    )
    generation_positions = positions + generation_index * tensor_numel
    retained = rng.poisson(
        mean=retained_rate,
        key=retained_key,
        positions=generation_positions,
        quantum=0,
    )
    overflow = rng.poisson(
        mean=overflow_rate,
        key=overflow_key,
        positions=generation_positions,
        quantum=0,
    )
    return retained, overflow


def _draw_afterpulses(
    frontier: torch.Tensor,
    *,
    positions: torch.Tensor,
    runtime: AfterpulseRuntime,
    probability: float,
    rng_key: RngKey,
    generation_index: int,
    tensor_numel: int,
    floating_dtype: torch.dtype,
    rng: CounterRng,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_count = frontier.shape[-1]
    retained_count = torch.zeros_like(frontier)
    overflow_count = torch.zeros_like(frontier)
    retained_charge = torch.zeros_like(frontier, dtype=floating_dtype)
    overflow_charge = torch.zeros_like(frontier, dtype=floating_dtype)
    charge_square_sum = torch.zeros_like(frontier, dtype=floating_dtype)

    for source in range(sample_count):
        source_counts = frontier[..., source]
        source_positions = positions[..., source]
        shape = tuple(source_counts.shape)
        success_masses: list[torch.Tensor] = []
        failure_masses: list[torch.Tensor] = []
        category_positions: list[torch.Tensor] = []
        for offset in range(sample_count - source):
            success = probability * runtime.delay.probabilities[offset]
            later = (1.0 - probability) + probability * runtime.delay.right_tails[
                offset + 1
            ]
            success_masses.append(
                torch.full(
                    shape,
                    success,
                    dtype=torch.float64,
                    device=frontier.device,
                )
            )
            failure_masses.append(
                torch.full(
                    shape,
                    later,
                    dtype=torch.float64,
                    device=frontier.device,
                )
            )
            category_positions.append(
                (generation_index * (sample_count + 1) + offset) * tensor_numel
                + source_positions
            )

        first_outside = sample_count - source
        success_masses.append(
            torch.full(
                shape,
                probability * runtime.delay.right_tails[first_outside],
                dtype=torch.float64,
                device=frontier.device,
            )
        )
        failure_masses.append(
            torch.full(
                shape,
                1.0 - probability,
                dtype=torch.float64,
                device=frontier.device,
            )
        )
        category_positions.append(
            (generation_index * (sample_count + 1) + sample_count)
            * tensor_numel
            + source_positions
        )
        *drawn_categories, _stop = draw_ordered_categories(
            source_counts,
            success_masses=tuple(success_masses),
            failure_masses=tuple(failure_masses),
            positions=tuple(category_positions),
            rng=rng,
            key=rng_key,
            field="afterpulse",
        )
        retained_categories = drawn_categories[:-1]
        overflow = drawn_categories[-1]

        for offset, category in enumerate(retained_categories):
            destination = source + offset
            retained_count[..., destination] = checked_add(
                retained_count[..., destination],
                category,
                field="afterpulse retained count",
            )
            if runtime.recovery is not None:
                represented_weight = torch.tensor(
                    runtime.recovery[offset],
                    dtype=floating_dtype,
                    device=frontier.device,
                )
                category_float = category.to(floating_dtype)
                retained_charge[..., destination] = (
                    retained_charge[..., destination]
                    + category_float * represented_weight
                )
                charge_square_sum[..., destination] = (
                    charge_square_sum[..., destination]
                    + category_float
                    * (represented_weight * represented_weight)
                )

        overflow_count[..., source] = overflow
        if runtime.overflow_recovery is not None:
            overflow_charge[..., source] = overflow.to(floating_dtype) * torch.tensor(
                runtime.overflow_recovery[first_outside],
                dtype=floating_dtype,
                device=frontier.device,
            )

    if runtime.recovery is None:
        retained_charge = retained_count.to(floating_dtype)
        charge_square_sum = retained_charge
        overflow_charge = overflow_count.to(floating_dtype)

    for field, value in (
        ("afterpulse charge", retained_charge),
        ("afterpulse overflow charge", overflow_charge),
        ("afterpulse charge-square sum", charge_square_sum),
    ):
        if not bool(torch.all(torch.isfinite(value) & (value >= 0.0)).item()):
            raise RuntimeError(f"{field} is invalid")
    return (
        retained_count,
        overflow_count,
        retained_charge,
        overflow_charge,
        charge_square_sum,
    )


def _checked_ledger_add(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bound: float,
    field: str,
) -> torch.Tensor:
    if left.dtype is not right.dtype or left.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise TypeError(f"{field} requires one common floating dtype")
    result = left + right
    if not bool(torch.all(torch.isfinite(result) & (result >= 0.0)).item()):
        raise RuntimeError(f"{field} produced an invalid ledger value")
    bound_d = torch.tensor(bound, dtype=result.dtype, device=result.device)
    if float(bound_d) > bound:
        bound_d = torch.nextafter(bound_d, torch.zeros_like(bound_d))
    if bool(torch.any(result > bound_d).item()):
        raise RuntimeError(f"{field} exceeds the proved ledger bound")
    return result


def simulate_correlated_avalanches(
    seed_avalanches: torch.Tensor,
    *,
    sample_dimension: int,
    floating_dtype: torch.dtype,
    runtime: CorrelatedAvalancheRuntime,
    rng: CounterRng,
) -> _CorrelatedAvalancheResult:
    require_count_domain(seed_avalanches, field="correlated-avalanche roots")
    if sample_dimension < 0 or sample_dimension >= seed_avalanches.ndim:
        raise ValueError("sample_dimension is outside the root rank")
    if seed_avalanches.shape[sample_dimension] != runtime.sample_count:
        raise ValueError("sample dimension disagrees with the prepared runtime")
    if seed_avalanches.numel() != runtime.tensor_numel:
        raise ValueError("input size disagrees with the prepared runtime")

    tensor_numel = seed_avalanches.numel()
    sample_last = seed_avalanches.movedim(sample_dimension, -1)
    positions = original_positions(
        tuple(seed_avalanches.shape),
        sample_dimension=sample_dimension,
        device=seed_avalanches.device,
    )
    maximum_generations = runtime.maximum_generations

    S1 = sample_last.to(floating_dtype)
    S2 = sample_last.to(floating_dtype)
    total_count = sample_last.clone()
    frontier = sample_last
    if (
        runtime.direct_crosstalk is None
        and runtime.delayed_crosstalk is None
        and runtime.afterpulse is None
    ):
        final_frontier = (
            frontier.clone()
            if maximum_generations == 0
            else torch.zeros_like(frontier)
        )
        restore = lambda value: value.movedim(-1, sample_dimension)
        return _CorrelatedAvalancheResult(
            restore(S1),
            restore(S2),
            restore(final_frontier),
            restore(total_count),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    direct_count = (
        torch.zeros_like(sample_last)
        if runtime.direct_crosstalk is not None
        else None
    )
    direct_overflow = (
        torch.zeros_like(sample_last)
        if runtime.direct_crosstalk is not None
        else None
    )
    delayed_count = (
        torch.zeros_like(sample_last)
        if runtime.delayed_crosstalk is not None
        else None
    )
    delayed_overflow = (
        torch.zeros_like(sample_last)
        if runtime.delayed_crosstalk is not None
        else None
    )
    afterpulse_count = (
        torch.zeros_like(sample_last) if runtime.afterpulse is not None else None
    )
    afterpulse_overflow = (
        torch.zeros_like(sample_last) if runtime.afterpulse is not None else None
    )
    afterpulse_charge = (
        torch.zeros_like(sample_last, dtype=floating_dtype)
        if runtime.afterpulse is not None
        else None
    )
    afterpulse_overflow_charge = (
        torch.zeros_like(sample_last, dtype=floating_dtype)
        if runtime.afterpulse is not None
        else None
    )
    afterpulse_square_sum = (
        torch.zeros_like(sample_last, dtype=floating_dtype)
        if runtime.afterpulse is not None
        else None
    )

    for generation_index in range(maximum_generations):
        children = torch.zeros_like(sample_last)

        if runtime.direct_crosstalk is not None:
            if (
                runtime.direct_mean is None
                or runtime.direct_retained_rng_key is None
                or runtime.direct_overflow_rng_key is None
            ):
                raise RuntimeError("direct-crosstalk runtime is incomplete")
            new_count, new_overflow = _draw_crosstalk(
                frontier,
                positions=positions,
                runtime=runtime.direct_crosstalk,
                mean=runtime.direct_mean,
                retained_key=runtime.direct_retained_rng_key,
                overflow_key=runtime.direct_overflow_rng_key,
                generation_index=generation_index,
                tensor_numel=tensor_numel,
                rng=rng,
                field="direct crosstalk",
            )
            assert direct_count is not None and direct_overflow is not None
            direct_count = checked_add(
                direct_count,
                new_count,
                field="direct-crosstalk cumulative count",
            )
            direct_overflow = checked_add(
                direct_overflow,
                new_overflow,
                field="direct-crosstalk cumulative overflow",
            )
            direct_charge = new_count.to(floating_dtype)
            S1 = _checked_ledger_add(
                S1,
                direct_charge,
                bound=runtime.ledger_bound,
                field="direct-crosstalk S1",
            )
            S2 = _checked_ledger_add(
                S2,
                direct_charge,
                bound=runtime.ledger_bound,
                field="direct-crosstalk S2",
            )
            children = checked_add(
                children,
                new_count,
                field="direct-crosstalk children",
            )

        if runtime.delayed_crosstalk is not None:
            if (
                runtime.delayed_mean is None
                or runtime.delayed_retained_rng_key is None
                or runtime.delayed_overflow_rng_key is None
            ):
                raise RuntimeError("delayed-crosstalk runtime is incomplete")
            new_count, new_overflow = _draw_crosstalk(
                frontier,
                positions=positions,
                runtime=runtime.delayed_crosstalk,
                mean=runtime.delayed_mean,
                retained_key=runtime.delayed_retained_rng_key,
                overflow_key=runtime.delayed_overflow_rng_key,
                generation_index=generation_index,
                tensor_numel=tensor_numel,
                rng=rng,
                field="delayed crosstalk",
            )
            assert delayed_count is not None and delayed_overflow is not None
            delayed_count = checked_add(
                delayed_count,
                new_count,
                field="delayed-crosstalk cumulative count",
            )
            delayed_overflow = checked_add(
                delayed_overflow,
                new_overflow,
                field="delayed-crosstalk cumulative overflow",
            )
            delayed_charge = new_count.to(floating_dtype)
            S1 = _checked_ledger_add(
                S1,
                delayed_charge,
                bound=runtime.ledger_bound,
                field="delayed-crosstalk S1",
            )
            S2 = _checked_ledger_add(
                S2,
                delayed_charge,
                bound=runtime.ledger_bound,
                field="delayed-crosstalk S2",
            )
            children = checked_add(
                children,
                new_count,
                field="delayed-crosstalk children",
            )

        if runtime.afterpulse is not None:
            if (
                runtime.afterpulse_probability is None
                or runtime.afterpulse_rng_key is None
            ):
                raise RuntimeError("afterpulse runtime is incomplete")
            (
                new_count,
                new_overflow,
                new_charge,
                new_overflow_charge,
                new_square_sum,
            ) = _draw_afterpulses(
                frontier,
                positions=positions,
                runtime=runtime.afterpulse,
                probability=runtime.afterpulse_probability,
                rng_key=runtime.afterpulse_rng_key,
                generation_index=generation_index,
                tensor_numel=tensor_numel,
                floating_dtype=floating_dtype,
                rng=rng,
            )
            assert (
                afterpulse_count is not None
                and afterpulse_overflow is not None
                and afterpulse_charge is not None
                and afterpulse_overflow_charge is not None
                and afterpulse_square_sum is not None
            )
            afterpulse_count = checked_add(
                afterpulse_count,
                new_count,
                field="afterpulse cumulative count",
            )
            afterpulse_overflow = checked_add(
                afterpulse_overflow,
                new_overflow,
                field="afterpulse cumulative overflow",
            )
            afterpulse_charge = _checked_ledger_add(
                afterpulse_charge,
                new_charge,
                bound=runtime.ledger_bound,
                field="afterpulse cumulative charge",
            )
            afterpulse_overflow_charge = _checked_ledger_add(
                afterpulse_overflow_charge,
                new_overflow_charge,
                bound=runtime.ledger_bound,
                field="afterpulse cumulative overflow charge",
            )
            afterpulse_square_sum = _checked_ledger_add(
                afterpulse_square_sum,
                new_square_sum,
                bound=runtime.ledger_bound,
                field="afterpulse cumulative charge-square sum",
            )
            S1 = _checked_ledger_add(
                S1,
                new_charge,
                bound=runtime.ledger_bound,
                field="afterpulse S1",
            )
            S2 = _checked_ledger_add(
                S2,
                new_square_sum,
                bound=runtime.ledger_bound,
                field="afterpulse S2",
            )
            children = checked_add(
                children,
                new_count,
                field="afterpulse children",
            )

        frontier = children
        total_count = checked_add(
            total_count,
            frontier,
            field="correlated-avalanche total count",
        )

    reconstructed_count = sample_last.clone()
    for field, contribution in (
        ("direct crosstalk", direct_count),
        ("delayed crosstalk", delayed_count),
        ("afterpulse", afterpulse_count),
    ):
        if contribution is not None:
            reconstructed_count = checked_add(
                reconstructed_count,
                contribution,
                field=f"correlated-avalanche reconstructed {field} count",
            )
    if not torch.equal(reconstructed_count, total_count):
        raise RuntimeError("correlated-avalanche integer count identity failed")

    precision = 24 if floating_dtype is torch.float32 else 53
    gamma = runtime.ledger_depth / ((1 << precision) - runtime.ledger_depth)
    zero = torch.tensor(0.0, dtype=floating_dtype, device="cpu")
    one = torch.tensor(1.0, dtype=floating_dtype, device="cpu")
    subnormal = float(torch.nextafter(zero, one))
    total_reference = total_count.to(torch.float64)
    tolerance = total_reference * gamma + runtime.ledger_depth * subnormal
    if not bool(
        torch.all(
            (S2.to(torch.float64) <= S1.to(torch.float64) + 2.0 * tolerance)
            & (S1.to(torch.float64) <= total_reference + tolerance)
        ).item()
    ):
        raise RuntimeError("correlated-avalanche ledgers violate their ordering")
    restore = lambda value: value.movedim(-1, sample_dimension)
    return _CorrelatedAvalancheResult(
        restore(S1),
        restore(S2),
        restore(frontier),
        restore(total_count),
        None if direct_count is None else restore(direct_count),
        None if direct_overflow is None else restore(direct_overflow),
        None if delayed_count is None else restore(delayed_count),
        None if delayed_overflow is None else restore(delayed_overflow),
        None if afterpulse_count is None else restore(afterpulse_count),
        None if afterpulse_overflow is None else restore(afterpulse_overflow),
        None if afterpulse_charge is None else restore(afterpulse_charge),
        None
        if afterpulse_overflow_charge is None
        else restore(afterpulse_overflow_charge),
        None if afterpulse_square_sum is None else restore(afterpulse_square_sum),
    )
