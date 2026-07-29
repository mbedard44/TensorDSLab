"""Execute Charge tensor arithmetic and addressed distributions."""

import itertools

import torch
from tensor_core import (
    CounterRng,
    GaussianDistribution,
    MultinomialDistribution,
    PoissonDistribution,
    RngElements,
)

from tensor_dslab.charge.config import ChargeConfig
from tensor_dslab.charge.runtime.branching import accumulate_branching
from tensor_dslab.charge.runtime.counts import MAX_COUNT, checked_add
from tensor_dslab.charge.runtime.random import (
    AFTERPULSE_KEY,
    CHARGE_SMEARING_KEY,
    DARK_COUNT_KEY,
    DELAYED_CROSSTALK_KEY,
    DIRECT_CROSSTALK_KEY,
    point_address,
    timing_address,
)
from tensor_dslab.common.alignment import require_prepared_sources


def _broadcast(kernel, dimensions: tuple[int, ...], shape: tuple[int, ...]) -> torch.Tensor:
    conditioning_rank = kernel.conditioning_rank
    result_shape = [1] * len(shape)
    for index, dimension in enumerate(dimensions[:conditioning_rank]):
        result_shape[dimension] = kernel.tensor.shape[index]
    return kernel.tensor.to(torch.float64).reshape(result_shape)


def _selected(tensor: torch.Tensor, elements: RngElements, fixed: dict[int, int]):
    result = tensor
    selected = elements
    for dimension in sorted(fixed, reverse=True):
        result = result.select(dimension, fixed[dimension])
        selected = selected.select(dimension, fixed[dimension])
    return result, selected


def _timing_jitter(
    source: torch.Tensor,
    *,
    config: ChargeConfig,
    rng: CounterRng,
    elements: RngElements,
) -> torch.Tensor:
    kernel = config.kernels.timing_jitter
    assert kernel is not None
    dimensions = config._kernel_dimensions[0]
    assert dimensions is not None
    conditioning_dimensions = dimensions[: kernel.conditioning_rank]
    target_dimension = dimensions[-1]
    result = torch.zeros_like(source)
    condition_shape = kernel.tensor.shape[: kernel.conditioning_rank]
    for condition_index in itertools.product(*(range(size) for size in condition_shape)):
        fixed = dict(zip(conditioning_dimensions, condition_index))
        source_indices = (
            (fixed[target_dimension],)
            if target_dimension in fixed
            else range(source.shape[target_dimension])
        )
        probabilities = kernel.tensor[(*condition_index, slice(None))]
        for source_index in source_indices:
            selected_fixed = dict(fixed)
            selected_fixed[target_dimension] = source_index
            counts, selected_elements = _selected(source, elements, selected_fixed)
            allocations = MultinomialDistribution(
                counts=counts,
                probabilities=probabilities,
                completion_probability=0.0,
            ).draw(
                rng=rng,
                address=timing_address(
                    selected_elements, shape=kernel.operation_shape
                ),
            )
            for operation_index in itertools.product(
                *(range(size) for size in kernel.operation_shape)
            ):
                offset = kernel.operation_axes[0].coordinate_at(operation_index[0])
                destination = source_index + offset
                if not 0 <= destination < source.shape[target_dimension]:
                    continue
                target: list[int | slice] = [slice(None)] * source.ndim
                for dimension, index in fixed.items():
                    target[dimension] = index
                target[target_dimension] = destination
                result[tuple(target)] = checked_add(
                    result[tuple(target)],
                    allocations[operation_index],
                    field="timing jitter",
                )
    return result


def produce_charge(*, sources: tuple, config: ChargeConfig, rng: CounterRng) -> torch.Tensor:
    """Return one fresh Charge tensor from an exact prepared Config."""

    if not config._is_prepared or config._working_dtype is None:
        raise ValueError("ChargeConfig must be prepared")
    require_prepared_sources(sources, source_specs=config._source_specs)
    counts = torch.zeros(config.spec.shape, dtype=torch.int64, device=config.spec.device)
    for source, dimensions, scale in zip(sources, config._source_dimensions, config._source_scales):
        ordered = source.tensor.permute(dimensions)
        if bool((ordered < 0).any()) or bool((ordered > MAX_COUNT).any()):
            raise ValueError("Charge source exceeds the count domain")
        converted_values = ordered.to(torch.float64) * scale
        if (
            not bool(torch.isfinite(converted_values).all())
            or bool((converted_values != torch.trunc(converted_values)).any())
            or bool((converted_values < 0).any())
            or bool((converted_values > MAX_COUNT).any())
        ):
            raise ValueError(
                "Charge source conversion must preserve exact count values"
            )
        converted = converted_values.to(torch.int64)
        counts = checked_add(counts, converted, field="Charge source accumulation")
    stochastic = bool(config.kernels.members)
    elements = RngElements.from_shape(config.spec.shape, device=config.spec.device) if stochastic else None
    dark = config.kernels.dark_count_rate
    if dark is not None:
        assert elements is not None and config._temporal_step_seconds is not None
        mean = torch.broadcast_to(
            _broadcast(dark, config._kernel_dimensions[4], config.spec.shape),  # type: ignore[arg-type]
            config.spec.shape,
        ) * config._temporal_step_seconds
        drawn = PoissonDistribution(mean=mean).draw(
            rng=rng, address=point_address(elements, key=DARK_COUNT_KEY)
        )
        counts = checked_add(counts, drawn, field="dark counts")
    if config.kernels.timing_jitter is not None:
        assert elements is not None
        counts = _timing_jitter(counts, config=config, rng=rng, elements=elements)
    if config.correlated_avalanche_generations.value:
        assert elements is not None
        mechanisms = tuple(
            (kernel, dimensions, key, name)
            for kernel, dimensions, key, name in (
                (config.kernels.direct_crosstalk, config._kernel_dimensions[1], DIRECT_CROSSTALK_KEY, "direct crosstalk"),
                (config.kernels.delayed_crosstalk, config._kernel_dimensions[2], DELAYED_CROSSTALK_KEY, "delayed crosstalk"),
                (config.kernels.afterpulse, config._kernel_dimensions[3], AFTERPULSE_KEY, "afterpulse"),
            )
            if kernel is not None and dimensions is not None
        )
        counts = accumulate_branching(
            counts,
            mechanisms=mechanisms,
            generations=config.correlated_avalanche_generations.value,
            rng=rng,
            elements=elements,
        )
    values = counts.to(config._working_dtype)
    smearing = config.kernels.smearing_width
    if smearing is not None:
        assert elements is not None
        width = torch.broadcast_to(
            _broadcast(smearing, config._kernel_dimensions[5], config.spec.shape),  # type: ignore[arg-type]
            config.spec.shape,
        ).to(config._working_dtype)
        values = GaussianDistribution(
            mean=values,
            standard_deviation=width * torch.sqrt(values),
            dtype=config._working_dtype,
        ).draw(
            rng=rng,
            address=point_address(elements, key=CHARGE_SMEARING_KEY),
        )
        values = torch.clamp_min(values, 0)
    return values.to(config.spec.dtype).contiguous()
