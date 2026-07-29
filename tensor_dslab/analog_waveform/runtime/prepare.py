"""Prepare one AnalogWaveform Config for staged execution."""

from dataclasses import replace

import torch

from tensor_dslab.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.common.alignment import (
    prepare_kernel,
    prepare_sources,
    require_allocation,
)


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


def prepare_analog_waveform(*, source_specs: tuple, config: AnalogWaveformConfig) -> AnalogWaveformConfig:
    source_dimensions, scales, dtype = prepare_sources(
        source_specs, target_spec=config.spec, minimum_count=1
    )
    dtype = torch.promote_types(dtype, torch.float32)
    require_allocation(
        config.spec.shape,
        dtype=dtype,
        field="AnalogWaveform workspace",
    )
    prepared_members = []
    kernel_dimensions_result: list[tuple[int, ...] | None] = []
    for kernel in (config.kernels.minimum, config.kernels.maximum):
        if kernel is None:
            kernel_dimensions_result.append(None)
            continue
        kernel, aligned_dimensions = prepare_kernel(
            kernel,
            target_axes=config.spec.axes,
            target_device=config.spec.device,
            target_unit=config.spec.unit,
        )
        prepared_members.append(kernel)
        kernel_dimensions_result.append(aligned_dimensions)
        dtype = torch.promote_types(dtype, kernel.dtype)
    prepared_kernels = type(config.kernels)(members=tuple(prepared_members))
    if prepared_kernels.minimum is not None and prepared_kernels.maximum is not None:
        minimum = _broadcast(
            prepared_kernels.minimum.tensor,
            kernel_dimensions_result[0],  # type: ignore[arg-type]
            shape=config.spec.shape,
        )
        maximum = _broadcast(
            prepared_kernels.maximum.tensor,
            kernel_dimensions_result[1],  # type: ignore[arg-type]
            shape=config.spec.shape,
        )
        if bool((minimum >= maximum).any()):
            raise ValueError("AnalogMinimum must be below AnalogMaximum")
    prepared = replace(config, kernels=prepared_kernels)
    for name, value in (
        ("_is_prepared", True),
        ("_source_specs", source_specs),
        ("_source_dimensions", source_dimensions),
        ("_source_scales", scales),
        ("_working_dtype", dtype),
        ("_kernel_dimensions", tuple(kernel_dimensions_result)),
    ):
        object.__setattr__(prepared, name, value)
    return prepared
