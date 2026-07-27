"""Private compilation of physical charge kernels into execution facts."""

from dataclasses import dataclass
import itertools
from typing import final

import torch
from tensor_core import OffsetAxis, TensorAxis
from tensor_core.random.validation import require_count_tensor
from tensor_core.tensor.validation import (
    require_kernel_dimensions,
    require_shape_span,
    require_tensor_allocation,
)

from tensor_dslab.common import SampleAxis
from tensor_dslab.common.kernel import QuantityKernel
from tensor_dslab.readout.charge.config import ChargeConfig
from tensor_dslab.readout.charge.kernel import (
    Afterpulse,
    DarkCountRate,
    DelayedCrosstalk,
    DirectCrosstalk,
    SmearingWidth,
    TimingJitter,
)
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.runtime.sampling import SamplingRuntime
from tensor_dslab.readout.charge.runtime.counts import MAX_COUNT


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


def _aligned_magnitude(
    kernel: QuantityKernel,
    *,
    photoelectrons: Photoelectrons,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    require_kernel_dimensions(photoelectrons, kernel)
    tensor = kernel.tensor
    dimensions: list[int] = []
    for kernel_dimension, axis in enumerate(kernel.conditioning_axes):
        role = type(axis)
        try:
            target = photoelectrons.axis(role)
            target_dimension = photoelectrons.dimension_of(role)
        except KeyError as error:
            raise ValueError(
                f"{type(kernel).__name__} conditioning role is absent "
                "from the readout geometry"
            ) from error
        if len(axis.coordinates) != len(target.coordinates):
            raise ValueError(
                f"{type(kernel).__name__} conditioning coordinates "
                "must match the readout axis"
            )
        if len(set(axis.coordinates)) != len(axis.coordinates):
            raise ValueError(
                f"{type(kernel).__name__} conditioning coordinates "
                "must be unique"
            )
        try:
            indices = tuple(
                axis.index_of(coordinate) for coordinate in target.coordinates
            )
        except KeyError as error:
            raise ValueError(
                f"{type(kernel).__name__} conditioning coordinates "
                "must match the readout axis"
            ) from error
        tensor = tensor.index_select(
            kernel_dimension,
            torch.tensor(indices, dtype=torch.int64),
        )
        dimensions.append(target_dimension)

    order = tuple(
        sorted(range(len(dimensions)), key=lambda index: dimensions[index])
    )
    if order != tuple(range(len(order))):
        tensor = tensor.permute(
            *order,
            *range(len(order), tensor.ndim),
        )
    return (
        tensor.to(device=device, dtype=torch.float64).contiguous(),
        tuple(dimensions[index] for index in order),
    )


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
    device: torch.device,
) -> BranchingRuntime:
    intensities, conditioning_dimensions = _aligned_magnitude(
        kernel,
        photoelectrons=photoelectrons,
        device=device,
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
    device = source.device

    dark_count_mean: torch.Tensor | None = None
    if config.dark_counts is not None:
        rate, dimensions = _aligned_magnitude(
            config.dark_counts,
            photoelectrons=photoelectrons,
            device=device,
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
        probabilities, dimensions = _aligned_magnitude(
            config.timing_jitter,
            photoelectrons=photoelectrons,
            device=device,
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
            device=device,
        )
    )
    delayed = (
        None
        if config.delayed_crosstalk is None
        else _prepare_branching(
            config.delayed_crosstalk,
            photoelectrons=photoelectrons,
            device=device,
        )
    )
    afterpulse = (
        None
        if config.afterpulse is None
        else _prepare_branching(
            config.afterpulse,
            photoelectrons=photoelectrons,
            device=device,
        )
    )

    smearing_width: torch.Tensor | None = None
    if config.smearing_width is not None:
        width, dimensions = _aligned_magnitude(
            config.smearing_width,
            photoelectrons=photoelectrons,
            device=device,
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
