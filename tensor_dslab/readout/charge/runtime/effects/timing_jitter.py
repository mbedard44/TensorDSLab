from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import torch
from tensor_core import CounterRng, RngKey

from tensor_dslab.readout.runtime.sampling import SamplingRuntime
from tensor_dslab.readout.charge.config import TimingJitterConfig
from tensor_dslab.readout.charge.runtime.effects.counts import (
    checked_add,
    draw_ordered_categories,
    original_positions,
    require_count_domain,
)


_MAX_SAMPLE_COUNT = 8192
_LOCAL_PROBABILITY_TOLERANCE = 1.0e-12
_COMPLETE_LAW_TOLERANCE = 1.0e-11


@final
@dataclass(frozen=True, slots=True)
class TimingJitterRuntime:
    probabilities: tuple[float, ...]
    left_tails: tuple[float, ...]
    rng_key: RngKey


def _log_jitter_g(value: float) -> float:
    log_phi = -0.5 * value * value - 0.5 * math.log(2.0 * math.pi)
    if value == 0.0:
        return log_phi
    if value < 8.0:
        represented = math.exp(log_phi) - 0.5 * value * math.erfc(
            value / math.sqrt(2.0)
        )
        if not math.isfinite(represented) or represented <= 0.0:
            raise ValueError("timing-jitter tail helper is invalid")
        return math.log(represented)

    squared = value * value
    term = 1.0 / squared
    total = term
    for order in range(2, 101):
        candidate = -(2 * order - 1) * term / squared
        if abs(candidate) >= abs(term):
            break
        advanced = total + candidate
        if advanced == total:
            break
        total = advanced
        term = candidate
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("timing-jitter asymptotic tail helper is invalid")
    return log_phi + math.log(total)


def prepare_timing_jitter(
    config: TimingJitterConfig,
    *,
    sampling: SamplingRuntime,
    tensor_numel: int,
) -> TimingJitterRuntime:
    sigma_ns = config.sigma_ns.value
    if sigma_ns == 0.0:
        raise ValueError("zero timing jitter uses the exact identity path")
    sample_count = sampling.sample_count
    if sample_count > _MAX_SAMPLE_COUNT:
        raise ValueError("active timing jitter supports at most 8192 samples")
    if sample_count * tensor_numel > 1 << 63:
        raise ValueError("timing-jitter address lattice exceeds its domain")
    period_ps = float(sampling.sample_period_ps)
    if not (
        period_ps * 2.0**-52 * 1.0e-3
        <= sigma_ns
        <= period_ps * 64.0 * 1.0e-3
    ):
        raise ValueError("timing-jitter ratio is outside its accepted domain")
    sigma_ps = sigma_ns * 1000.0
    ratio = sigma_ps / period_ps
    if not math.isfinite(ratio) or not 2.0**-52 <= ratio <= 64.0:
        raise ValueError("timing-jitter represented ratio is outside its domain")

    log_g = tuple(
        _log_jitter_g(index / ratio)
        for index in range(sample_count + 1)
    )
    left_tails: list[float] = []
    log_left_tails: list[float] = []
    log_ratio = math.log(ratio)
    for index in range(sample_count):
        difference = log_g[index + 1] - log_g[index]
        log_tail = log_ratio + log_g[index] + math.log(-math.expm1(difference))
        log_left_tails.append(log_tail)
        left_tails.append(math.exp(log_tail))

    q_zero = math.erf(1.0 / (math.sqrt(2.0) * ratio)) + (
        ratio
        * math.sqrt(2.0 / math.pi)
        * math.expm1(-1.0 / (2.0 * ratio * ratio))
    )
    probabilities = [q_zero]
    for offset in range(1, sample_count):
        previous = left_tails[offset - 1]
        following = left_tails[offset]
        if previous == 0.0:
            probability = 0.0
        elif following == 0.0:
            probability = previous
        else:
            probability = math.exp(
                log_left_tails[offset - 1]
                + math.log(
                    -math.expm1(
                        log_left_tails[offset] - log_left_tails[offset - 1]
                    )
                )
            )
        probabilities.append(probability)

    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in (*probabilities, *left_tails)
    ):
        raise ValueError("timing-jitter preparation produced an invalid probability")
    if any(later > earlier for earlier, later in zip(left_tails, left_tails[1:])):
        raise ValueError("timing-jitter left tail must be nonincreasing")
    if abs((q_zero + 2.0 * left_tails[0]) - 1.0) > (
        _LOCAL_PROBABILITY_TOLERANCE
    ):
        raise ValueError("timing-jitter central identity failed")
    if abs(
        math.fsum((q_zero, *(2.0 * p for p in probabilities[1:]), 2.0 * left_tails[-1]))
        - 1.0
    ) > _COMPLETE_LAW_TOLERANCE:
        raise ValueError("timing-jitter complete-law identity failed")
    return TimingJitterRuntime(
        tuple(probabilities),
        tuple(left_tails),
        config.rng_key,
    )


def simulate_timing_jitter(
    counts: torch.Tensor,
    *,
    sample_dimension: int,
    runtime: TimingJitterRuntime,
    rng: CounterRng,
) -> torch.Tensor:
    if type(runtime) is not TimingJitterRuntime:
        raise TypeError("runtime must be exactly TimingJitterRuntime")
    require_count_domain(counts, field="timing-jitter input")
    if type(sample_dimension) is not int:
        raise TypeError("sample_dimension must be exactly an integer")
    if sample_dimension < 0 or sample_dimension >= counts.ndim:
        raise ValueError("sample_dimension is outside the count rank")
    sample_count = len(runtime.probabilities)
    if counts.shape[sample_dimension] != sample_count:
        raise ValueError("sample dimension disagrees with the prepared runtime")
    if not bool(torch.any(counts != 0).item()):
        return counts.clone()

    total_count = counts.numel()
    sample_last = counts.movedim(sample_dimension, -1)
    remaining = sample_last.clone()
    result = torch.zeros_like(sample_last)
    positions = original_positions(
        tuple(counts.shape),
        sample_dimension=sample_dimension,
        device=counts.device,
    )

    for target in range(sample_count):
        destination = result[..., target]
        for source in range(sample_count):
            offset = target - source
            if offset < 0:
                distance = -offset
                success = runtime.probabilities[distance]
                later = (
                    1.0
                    - runtime.left_tails[distance - 1]
                    + runtime.left_tails[source]
                )
            elif offset == 0:
                success = runtime.probabilities[0]
                later = runtime.left_tails[source] + runtime.left_tails[0]
            else:
                success = runtime.probabilities[offset]
                later = runtime.left_tails[source] + runtime.left_tails[offset]
            source_remaining = remaining[..., source]
            shape = tuple(source_remaining.shape)
            success_mass = torch.full(
                shape,
                success,
                dtype=torch.float64,
                device=counts.device,
            )
            later_mass = torch.full(
                shape,
                later,
                dtype=torch.float64,
                device=counts.device,
            )
            category, source_remainder = draw_ordered_categories(
                source_remaining,
                success_masses=(success_mass,),
                failure_masses=(later_mass,),
                positions=(positions[..., source] + target * total_count,),
                rng=rng,
                key=runtime.rng_key,
                field="timing jitter",
            )
            remaining[..., source] = source_remainder
            destination = checked_add(
                destination,
                category,
                field="timing-jitter destination",
            )
        result[..., target] = destination
    return result.movedim(-1, sample_dimension)
