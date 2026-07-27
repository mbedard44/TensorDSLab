"""Private compilation of physical charge kernels into execution facts."""

from dataclasses import dataclass
import itertools
from typing import final

import torch
from tensor_core import OffsetAxis
from tensor_core.random.validation import require_count_tensor
from tensor_core.tensor.validation import (
    require_shape_span,
    require_tensor_allocation,
)

from tensor_dslab.common import SampleAxis
from tensor_dslab.readout.charge.config import ChargeConfig
from tensor_dslab.readout.charge.kernel import (
    Afterpulse,
    DarkCountRate,
    DelayedCrosstalk,
    DirectCrosstalk,
    SmearingWidth,
    TimingJitter,
)
from tensor_dslab.readout.charge.runtime.counts import MAX_COUNT
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.runtime.kernel import align_quantity_kernel
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


@final
@dataclass(frozen=True, slots=True)
class TimingJitterRuntime:
    probabilities: torch.Tensor
    conditioning_dimensions: tuple[int, ...]
    sample_offsets: tuple[int, ...]


@final
@dataclass(frozen=True, slots=True)
class BranchingRuntime:
    intensities: torch.Tensor
    conditioning_dimensions: tuple[int, ...]
    target_dimensions: tuple[int, ...]
    offsets: tuple[tuple[int, ...], ...]


@final
@dataclass(frozen=True, slots=True)
class ChargeRuntime:
    sampling: SamplingRuntime
    floating_dtype: torch.dtype
    correlated_avalanche_generations: int
    dark_count_mean: torch.Tensor | None
    timing_jitter: TimingJitterRuntime | None
    direct_crosstalk: BranchingRuntime | None
    delayed_crosstalk: BranchingRuntime | None
    afterpulse: BranchingRuntime | None
    smearing_width: torch.Tensor | None


def _broadcast_conditioning(
    tensor: torch.Tensor,
    *,
    conditioning_dimensions: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    shape = [1] * len(target_shape)
    for source_dimension, target_dimension in enumerate(
        conditioning_dimensions
    ):
        shape[target_dimension] = tensor.shape[source_dimension]
    return torch.broadcast_to(tensor.reshape(shape), target_shape)


def _prepare_branching(
    kernel: DirectCrosstalk | DelayedCrosstalk | Afterpulse,
    *,
    photoelectrons: Photoelectrons,
) -> BranchingRuntime:
    intensities, conditioning_dimensions = align_quantity_kernel(
        kernel,
        field=photoelectrons,
        dtype=torch.float64,
    )
    target_dimensions = tuple(
        photoelectrons.dimension_of(axis.relative_to)
        for axis in kernel.operation_axes
    )
    offsets = tuple(
        tuple(values)
        for values in itertools.product(
            *(axis.offsets for axis in kernel.operation_axes)
        )
    )
    return BranchingRuntime(
        intensities=intensities,
        conditioning_dimensions=conditioning_dimensions,
        target_dimensions=target_dimensions,
        offsets=offsets,
    )


def prepare_charge(
    config: ChargeConfig,
    *,
    photoelectrons: Photoelectrons,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
) -> ChargeRuntime:
    """Compile one admitted ChargeConfig for one concrete source geometry."""

    source = photoelectrons.tensor
    require_count_tensor(source, "Photoelectrons source")
    if bool((source > MAX_COUNT).any()):
        raise ValueError("Photoelectrons source exceeds the Charge count ceiling")
    require_shape_span(
        (config.correlated_avalanche_generations.value,),
        "branching generation address",
        upper=1 << 63,
    )
    shape = tuple(source.shape)
    require_tensor_allocation(
        shape,
        "Charge output",
        element_size=torch.empty((), dtype=floating_dtype).element_size(),
        upper=1 << 63,
    )
    dark_count_mean: torch.Tensor | None = None
    if config.dark_counts is not None:
        rate, dimensions = align_quantity_kernel(
            config.dark_counts,
            field=photoelectrons,
            dtype=torch.float64,
        )
        exposure_seconds = sampling.sample_period_ps * 1.0e-12
        dark_count_mean = _broadcast_conditioning(
            rate * exposure_seconds,
            conditioning_dimensions=dimensions,
            target_shape=shape,
        ).contiguous()
        if bool((dark_count_mean > 1.0e8).any()):
            raise ValueError("dark-count mean exceeds the Poisson domain")

    timing_jitter: TimingJitterRuntime | None = None
    if config.timing_jitter is not None:
        probabilities, dimensions = align_quantity_kernel(
            config.timing_jitter,
            field=photoelectrons,
            dtype=torch.float64,
        )
        axis = config.timing_jitter.operation_axes[0]
        timing_jitter = TimingJitterRuntime(
            probabilities=probabilities,
            conditioning_dimensions=dimensions,
            sample_offsets=axis.offsets,
        )

    direct = (
        None
        if config.direct_crosstalk is None
        else _prepare_branching(
            config.direct_crosstalk,
            photoelectrons=photoelectrons,
        )
    )
    delayed = (
        None
        if config.delayed_crosstalk is None
        else _prepare_branching(
            config.delayed_crosstalk,
            photoelectrons=photoelectrons,
        )
    )
    afterpulse = (
        None
        if config.afterpulse is None
        else _prepare_branching(
            config.afterpulse,
            photoelectrons=photoelectrons,
        )
    )

    smearing_width: torch.Tensor | None = None
    if config.smearing_width is not None:
        width, dimensions = align_quantity_kernel(
            config.smearing_width,
            field=photoelectrons,
            dtype=torch.float64,
        )
        smearing_width = _broadcast_conditioning(
            width,
            conditioning_dimensions=dimensions,
            target_shape=shape,
        ).to(dtype=floating_dtype)

    return ChargeRuntime(
        sampling=sampling,
        floating_dtype=floating_dtype,
        correlated_avalanche_generations=config.correlated_avalanche_generations.value,
        dark_count_mean=dark_count_mean,
        timing_jitter=timing_jitter,
        direct_crosstalk=direct,
        delayed_crosstalk=delayed,
        afterpulse=afterpulse,
        smearing_width=smearing_width,
    )
