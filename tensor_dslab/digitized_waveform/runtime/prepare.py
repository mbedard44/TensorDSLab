"""Prepare one DigitizedWaveform Config for staged execution."""

from dataclasses import replace

import torch

from tensor_dslab.common.alignment import (
    prepare_kernel,
    prepare_sources,
)
from tensor_dslab.common.requirements.capacity import require_tensor_capacity
from tensor_dslab.common.units import unit_registry
from tensor_dslab.digitized_waveform.config import DigitizedWaveformConfig


def _broadcast(
    tensor: torch.Tensor,
    dimensions: tuple[int, ...],
    *,
    shape: tuple[int, ...],
) -> torch.Tensor:
    target = [1] * len(shape)
    for source_dimension, target_dimension in enumerate(dimensions):
        target[target_dimension] = tensor.shape[source_dimension]
    return tensor.reshape(target)


def prepare_digitized_waveform(*, source_specs: tuple, config: DigitizedWaveformConfig) -> DigitizedWaveformConfig:
    minimum = config.kernels.input_minimum
    maximum = config.kernels.input_maximum
    gain = config.kernels.analog_gain
    bit_depth = config.kernels.bit_depth
    source_dimensions, scales, dtype = prepare_sources(
        source_specs,
        target_spec=config.spec,
        minimum_count=1,
        maximum_count=1,
        unit_target=minimum.spec.unit,
    )
    dtype = torch.promote_types(dtype, torch.float32)
    for kernel in (minimum, maximum, gain):
        dtype = torch.promote_types(dtype, kernel.dtype)
    require_tensor_capacity(
        config.spec.shape,
        dtype=dtype,
        field="DigitizedWaveform floating workspace",
    )
    require_tensor_capacity(
        config.spec.shape,
        dtype=torch.int64,
        field="DigitizedWaveform integer workspace",
    )
    prepared_members = []
    prepared_dimensions = []
    for kernel, target_unit in (
        (bit_depth, None),
        (minimum, minimum.spec.unit),
        (maximum, minimum.spec.unit),
        (gain, unit_registry.Unit("")),
    ):
        prepared_kernel, aligned_dimensions = prepare_kernel(
            kernel,
            target_axes=config.spec.axes,
            target_device=config.spec.device,
            target_unit=target_unit,
        )
        prepared_members.append(prepared_kernel)
        prepared_dimensions.append(aligned_dimensions)
    prepared_kernels = type(config.kernels)(members=tuple(prepared_members))
    minimum_values = _broadcast(
        prepared_kernels.input_minimum.tensor,
        prepared_dimensions[1],
        shape=config.spec.shape,
    )
    maximum_values = _broadcast(
        prepared_kernels.input_maximum.tensor,
        prepared_dimensions[2],
        shape=config.spec.shape,
    )
    if bool(
        (minimum_values >= maximum_values).any()
    ):
        raise ValueError("InputMinimum must be below InputMaximum")
    kdims = tuple(prepared_dimensions)
    prepared = replace(
        config,
        kernels=prepared_kernels,
    )
    for name, value in (
        ("_is_prepared", True),
        ("_source_specs", source_specs),
        ("_source_dimensions", source_dimensions),
        ("_source_scales", scales),
        ("_working_dtype", dtype),
        ("_kernel_dimensions", kdims),
    ):
        object.__setattr__(prepared, name, value)
    return prepared
