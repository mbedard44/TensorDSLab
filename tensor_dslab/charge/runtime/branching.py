"""Private fixed-generation collapsed-rate Poisson branching."""

import itertools

import torch
from tensor_core import CounterRng, PoissonDistribution, RngElements, RngKey

from tensor_dslab.charge.runtime.counts import checked_add
from tensor_dslab.charge.runtime.random import branching_address


def _retained_rate(
    frontier: torch.Tensor,
    *,
    kernel,
    dimensions: tuple[int, ...],
    field: str,
) -> torch.Tensor:
    conditioning_rank = kernel.conditioning_rank
    conditioning_dimensions = dimensions[:conditioning_rank]
    target_dimensions = dimensions[conditioning_rank:]
    mean = torch.zeros_like(frontier, dtype=torch.float64)
    for operation_index in itertools.product(
        *(range(axis.size) for axis in kernel.operation_axes)
    ):
        offsets = tuple(
            axis.coordinate_at(index)
            for axis, index in zip(kernel.operation_axes, operation_index)
        )
        coefficient = kernel.tensor[(..., *operation_index)].to(torch.float64)
        view = [1] * frontier.ndim
        for index, dimension in enumerate(conditioning_dimensions):
            view[dimension] = coefficient.shape[index]
        coefficient = coefficient.reshape(view)
        source_slices: list[slice] = [slice(None)] * frontier.ndim
        target_slices: list[slice] = [slice(None)] * frontier.ndim
        for dimension, offset in zip(target_dimensions, offsets):
            size = frontier.shape[dimension]
            if abs(offset) >= size:
                break
            if offset >= 0:
                source_slices[dimension] = slice(0, size - offset)
                target_slices[dimension] = slice(offset, size)
            else:
                source_slices[dimension] = slice(-offset, size)
                target_slices[dimension] = slice(0, size + offset)
        else:
            coefficient_slices: list[slice] = [slice(None)] * frontier.ndim
            for dimension in conditioning_dimensions:
                coefficient_slices[dimension] = source_slices[dimension]
            updated = mean[tuple(target_slices)] + (
                frontier[tuple(source_slices)].to(torch.float64)
                * coefficient[tuple(coefficient_slices)]
            )
            if not bool(torch.isfinite(updated).all()) or bool((updated > 1.0e8).any()):
                raise RuntimeError(f"{field} destination mean exceeds Poisson domain")
            mean[tuple(target_slices)] = updated
    return mean


def accumulate_branching(
    seed: torch.Tensor,
    *,
    mechanisms: tuple[tuple[object, tuple[int, ...], RngKey, str], ...],
    generations: int,
    rng: CounterRng,
    elements: RngElements,
) -> torch.Tensor:
    total = seed
    frontier = seed
    for generation in range(generations):
        children = torch.zeros_like(frontier)
        for kernel, dimensions, key, field in mechanisms:
            mean = _retained_rate(
                frontier, kernel=kernel, dimensions=dimensions, field=field
            )
            produced = PoissonDistribution(mean=mean).draw(
                rng=rng,
                address=branching_address(
                    elements,
                    key=key,
                    generations=generations,
                    generation=generation,
                ),
            )
            children = checked_add(children, produced, field=field)
        total = checked_add(total, children, field="aggregate avalanche count")
        frontier = children
    return total
