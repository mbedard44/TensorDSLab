"""Private fixed-generation collapsed-rate Poisson branching."""

import torch
from tensor_core import CounterRng, PoissonDistribution, RngElements, RngKey

from tensor_dslab.readout.charge.runtime.counts import checked_add
from tensor_dslab.readout.charge.runtime.prepare import BranchingRuntime
from tensor_dslab.readout.runtime.addresses import branching_generation_address


def _condition_view(
    coefficient: torch.Tensor,
    *,
    dimensions: tuple[int, ...],
    rank: int,
) -> torch.Tensor:
    shape = [1] * rank
    for source_dimension, target_dimension in enumerate(dimensions):
        shape[target_dimension] = coefficient.shape[source_dimension]
    return coefficient.reshape(shape)


def _retained_rate(
    frontier: torch.Tensor,
    *,
    runtime: BranchingRuntime,
    field: str,
) -> torch.Tensor:
    shape = tuple(frontier.shape)
    mean = torch.zeros(shape, dtype=torch.float64, device=frontier.device)
    for operation_index, offsets in enumerate(runtime.offsets):
        coefficient = runtime.intensities.reshape(
            *runtime.intensities.shape[: len(runtime.conditioning_dimensions)],
            -1,
        )[..., operation_index]
        aligned = torch.broadcast_to(
            _condition_view(
                coefficient,
                dimensions=runtime.conditioning_dimensions,
                rank=frontier.ndim,
            ),
            shape,
        )
        source_slices: list[slice] = [slice(None)] * frontier.ndim
        target_slices: list[slice] = [slice(None)] * frontier.ndim
        retained = True
        for dimension, offset in zip(runtime.target_dimensions, offsets):
            size = shape[dimension]
            if abs(offset) >= size:
                retained = False
                break
            if offset >= 0:
                source_slices[dimension] = slice(0, size - offset)
                target_slices[dimension] = slice(offset, size)
            else:
                source_slices[dimension] = slice(-offset, size)
                target_slices[dimension] = slice(0, size + offset)
        if not retained:
            continue
        contribution = (
            frontier[tuple(source_slices)].to(dtype=torch.float64)
            * aligned[tuple(source_slices)]
        )
        target = mean[tuple(target_slices)]
        updated = target + contribution
        if not bool(
            torch.all(
                torch.isfinite(updated)
                & (updated >= 0)
                & (updated <= 1.0e8)
            )
        ):
            raise RuntimeError(f"{field} destination mean exceeds Poisson domain")
        mean[tuple(target_slices)] = updated
    return mean


def draw_branching(
    frontier: torch.Tensor,
    *,
    runtime: BranchingRuntime,
    rng: CounterRng,
    elements: RngElements,
    key: RngKey,
    maximum_generations: int,
    generation_index: int,
    field: str,
) -> torch.Tensor:
    """Draw one complete retained child frontier from collapsed destination means."""

    mean = _retained_rate(frontier, runtime=runtime, field=field)
    distribution = PoissonDistribution(mean=mean)
    return distribution.draw(
        rng=rng,
        address=branching_generation_address(
            elements,
            key=key,
            maximum_generations=maximum_generations,
            generation_index=generation_index,
        ),
    )


def accumulate_branching(
    seed: torch.Tensor,
    *,
    generations: int,
    direct: BranchingRuntime | None,
    delayed: BranchingRuntime | None,
    afterpulse: BranchingRuntime | None,
    rng: CounterRng,
    elements: RngElements,
    direct_key: RngKey,
    delayed_key: RngKey,
    afterpulse_key: RngKey,
) -> torch.Tensor:
    """Execute fixed rounds with independent mechanisms sharing one frontier."""

    total = seed
    frontier = seed
    for generation_index in range(generations):
        children = torch.zeros_like(frontier)
        for runtime, key, field in (
            (direct, direct_key, "direct crosstalk"),
            (delayed, delayed_key, "delayed crosstalk"),
            (afterpulse, afterpulse_key, "afterpulse"),
        ):
            if runtime is None:
                continue
            produced = draw_branching(
                frontier,
                runtime=runtime,
                rng=rng,
                elements=elements,
                key=key,
                maximum_generations=generations,
                generation_index=generation_index,
                field=field,
            )
            children = checked_add(children, produced, field=field)
        total = checked_add(total, children, field="aggregate avalanche count")
        frontier = children
    return total
