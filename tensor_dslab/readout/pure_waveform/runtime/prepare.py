"""Private compilation of a literal Pulse into execution facts."""

from dataclasses import dataclass
from typing import final

import torch
from tensor_core.tensor.validation import require_kernel_dimensions

from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.pure_waveform.config import PureWaveformConfig
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


@final
@dataclass(frozen=True, slots=True)
class PureWaveformRuntime:
    sampling: SamplingRuntime
    coefficients: torch.Tensor
    conditioning_dimensions: tuple[int, ...]
    sample_offsets: tuple[int, ...]


def prepare_pure_waveform(
    config: PureWaveformConfig,
    *,
    source: Photoelectrons,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> PureWaveformRuntime:
    """Align and materialize one pulse without retaining public kernel state."""

    pulse = config.pulse
    require_kernel_dimensions(source, pulse)
    tensor = pulse.tensor
    dimensions: list[int] = []
    for kernel_dimension, axis in enumerate(pulse.conditioning_axes):
        role = type(axis)
        try:
            target = source.axis(role)
            target_dimension = source.dimension_of(role)
        except KeyError as error:
            raise ValueError(
                "Pulse conditioning role is absent from readout geometry"
            ) from error
        if len(axis.coordinates) != len(target.coordinates):
            raise ValueError("Pulse conditioning coordinates must match")
        try:
            indices = tuple(
                axis.index_of(coordinate) for coordinate in target.coordinates
            )
        except KeyError as error:
            raise ValueError("Pulse conditioning coordinates must match") from error
        tensor = tensor.index_select(
            kernel_dimension,
            torch.tensor(indices, dtype=torch.int64),
        )
        dimensions.append(target_dimension)
    order = tuple(
        sorted(range(len(dimensions)), key=lambda index: dimensions[index])
    )
    if order != tuple(range(len(order))):
        tensor = tensor.permute(*order, len(order))
    return PureWaveformRuntime(
        sampling=sampling,
        coefficients=tensor.to(
            device=device,
            dtype=floating_dtype,
        ).contiguous(),
        conditioning_dimensions=tuple(dimensions[index] for index in order),
        sample_offsets=pulse.operation_axes[0].offsets,
    )
