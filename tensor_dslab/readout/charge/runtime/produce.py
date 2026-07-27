"""Private tensor and addressed-distribution execution for Charge."""

import itertools

import torch
from tensor_core import (
    CounterRng,
    GaussianDistribution,
    MultinomialDistribution,
    PoissonDistribution,
    RngElements,
)

from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.charge.runtime.branching import accumulate_branching
from tensor_dslab.readout.charge.runtime.counts import checked_add
from tensor_dslab.readout.charge.runtime.prepare import (
    ChargeRuntime,
    TimingJitterRuntime,
)
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.runtime.addresses import (
    charge_smearing_address,
    dark_count_address,
    timing_jitter_address,
)
from tensor_dslab.readout.runtime.keys import (
    AFTERPULSE_RNG_KEY,
    CHARGE_SMEARING_RNG_KEY,
    DARK_COUNT_RNG_KEY,
    DELAYED_CROSSTALK_RNG_KEY,
    DIRECT_CROSSTALK_RNG_KEY,
    TIMING_JITTER_RNG_KEY,
)


def _selected(
    tensor: torch.Tensor,
    elements: RngElements,
    fixed: dict[int, int],
) -> tuple[torch.Tensor, RngElements]:
    result = tensor
    selected = elements
    for dimension in sorted(fixed, reverse=True):
        result = result.select(dimension, fixed[dimension])
        selected = selected.select(dimension, fixed[dimension])
    return result, selected


def _apply_timing_jitter(
    source: torch.Tensor,
    *,
    sample_dimension: int,
    runtime: TimingJitterRuntime,
    rng: CounterRng,
    elements: RngElements,
) -> torch.Tensor:
    result = torch.zeros_like(source)
    condition_shape = runtime.probabilities.shape[:-1]
    condition_indices = itertools.product(
        *(range(size) for size in condition_shape)
    )
    for condition_index in condition_indices:
        fixed = dict(zip(runtime.conditioning_dimensions, condition_index))
        sample_indices = (
            (fixed[sample_dimension],)
            if sample_dimension in fixed
            else range(source.shape[sample_dimension])
        )
        probabilities = runtime.probabilities[
            (*condition_index, slice(None))
        ]
        for source_sample in sample_indices:
            selected_fixed = dict(fixed)
            selected_fixed[sample_dimension] = source_sample
            counts, selected_elements = _selected(
                source,
                elements,
                selected_fixed,
            )
            allocations = MultinomialDistribution(
                counts=counts,
                probabilities=probabilities,
                completion_probability=0.0,
            ).draw(
                rng=rng,
                address=timing_jitter_address(
                    selected_elements,
                    key=TIMING_JITTER_RNG_KEY,
                    kernel_shape=(len(runtime.sample_offsets),),
                ),
            )
            for operation_index, offset in enumerate(runtime.sample_offsets):
                destination = source_sample + offset
                if not 0 <= destination < source.shape[sample_dimension]:
                    continue
                target: list[int | slice] = [slice(None)] * source.ndim
                for dimension, index in fixed.items():
                    target[dimension] = index
                target[sample_dimension] = destination
                existing = result[tuple(target)]
                result[tuple(target)] = checked_add(
                    existing,
                    allocations[operation_index],
                    field="timing jitter",
                )
    return result


def produce_charge(
    photoelectrons: Photoelectrons,
    *,
    runtime: ChargeRuntime,
    rng: CounterRng,
) -> Charge:
    """Produce one fresh aggregate charge tensor."""

    source = photoelectrons.tensor
    stochastic = (
        runtime.dark_count_mean is not None
        or runtime.timing_jitter is not None
        or runtime.correlated_avalanche_generations > 0
        or runtime.smearing_width is not None
    )
    elements = (
        RngElements.from_shape(tuple(source.shape), device=source.device)
        if stochastic
        else None
    )
    counts = source
    if runtime.dark_count_mean is not None:
        assert elements is not None
        dark = PoissonDistribution(mean=runtime.dark_count_mean).draw(
            rng=rng,
            address=dark_count_address(elements, key=DARK_COUNT_RNG_KEY),
        )
        counts = checked_add(counts, dark, field="dark counts")
    if runtime.timing_jitter is not None:
        assert elements is not None
        counts = _apply_timing_jitter(
            counts,
            sample_dimension=runtime.sampling.sample_dimension,
            runtime=runtime.timing_jitter,
            rng=rng,
            elements=elements,
        )
    if runtime.correlated_avalanche_generations:
        assert elements is not None
        counts = accumulate_branching(
            counts,
            generations=runtime.correlated_avalanche_generations,
            direct=runtime.direct_crosstalk,
            delayed=runtime.delayed_crosstalk,
            afterpulse=runtime.afterpulse,
            rng=rng,
            elements=elements,
            direct_key=DIRECT_CROSSTALK_RNG_KEY,
            delayed_key=DELAYED_CROSSTALK_RNG_KEY,
            afterpulse_key=AFTERPULSE_RNG_KEY,
        )

    values = counts.to(dtype=runtime.floating_dtype)
    if runtime.smearing_width is not None:
        assert elements is not None
        scale = runtime.smearing_width * torch.sqrt(values)
        values = GaussianDistribution(
            mean=values,
            standard_deviation=scale,
            dtype=runtime.floating_dtype,
        ).draw(
            rng=rng,
            address=charge_smearing_address(
                elements,
                key=CHARGE_SMEARING_RNG_KEY,
            ),
        )
        values = torch.clamp_min(values, 0)
    return Charge(tensor=values, axes=photoelectrons.axes)
