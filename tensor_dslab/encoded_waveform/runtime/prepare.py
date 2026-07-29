"""Prepare one EncodedWaveform Config for staged raw-ZLE execution."""

from dataclasses import replace
import math
from typing import Any, cast

import torch
from tensor_core import RegularCoordinates, TensorKernel

from tensor_dslab.common import TimeAxis
from tensor_dslab.common.alignment import prepare_kernel
from tensor_dslab.common.requirements.capacity import require_tensor_capacity
from tensor_dslab.digitized_waveform.field import DigitizedWaveformSpec
from tensor_dslab.encoded_waveform.config import EncodedWaveformConfig
from tensor_dslab.encoded_waveform.kernel import EncodedWaveformKernels


def _lane_values(
    tensor: torch.Tensor,
    dimensions: tuple[int, ...],
    *,
    shape: tuple[int, ...],
    time_dimension: int,
) -> torch.Tensor:
    """Broadcast one aligned policy Kernel across non-Time lanes."""

    non_time_dimensions = tuple(
        dimension
        for dimension in range(len(shape))
        if dimension != time_dimension
    )
    target = [1] * len(non_time_dimensions)
    for source_dimension, output_dimension in enumerate(dimensions):
        target[non_time_dimensions.index(output_dimension)] = tensor.shape[
            source_dimension
        ]
    non_time_shape = tuple(shape[dimension] for dimension in non_time_dimensions)
    return tensor.reshape(target).expand(non_time_shape).reshape(-1)


def prepare_encoded_waveform(
    *,
    source_specs: tuple,
    config: EncodedWaveformConfig,
) -> EncodedWaveformConfig:
    """Validate and compile one raw-ZLE Product punchcard."""

    if type(config) is not EncodedWaveformConfig:
        raise TypeError("config must be exactly EncodedWaveformConfig")
    if type(source_specs) is not tuple or len(source_specs) != 1:
        raise ValueError("EncodedWaveform requires exactly one source Spec")
    source_spec = source_specs[0]
    if type(source_spec) is not DigitizedWaveformSpec:
        raise TypeError(
            "EncodedWaveform source Spec must be exactly "
            "DigitizedWaveformSpec"
        )
    if source_spec.unit != config.spec.unit:
        raise ValueError(
            "EncodedWaveform and DigitizedWaveform units must be equal"
        )
    if source_spec.axes != config.spec.axes:
        raise ValueError(
            "EncodedWaveform and DigitizedWaveform axes must be equal"
        )
    if source_spec.shape != config.spec.shape:
        raise ValueError(
            "EncodedWaveform and DigitizedWaveform shapes must be equal"
        )
    if source_spec.device != config.spec.device:
        raise ValueError(
            "EncodedWaveform and DigitizedWaveform devices must be equal"
        )
    if source_spec.dtype is not config.spec.dtype:
        raise TypeError(
            "EncodedWaveform and DigitizedWaveform dtypes must be equal"
        )

    time_dimensions = tuple(
        index
        for index, axis in enumerate(config.spec.axes)
        if type(axis) is TimeAxis
    )
    if len(time_dimensions) != 1:
        raise ValueError("EncodedWaveformSpec must contain exactly one TimeAxis")
    time_dimension = time_dimensions[0]
    time_axis = cast(TimeAxis[Any], config.spec.axes[time_dimension])
    if type(time_axis.coordinates) is not RegularCoordinates:
        raise TypeError(
            "EncodedWaveform TimeAxis coordinates must be exactly "
            "RegularCoordinates"
        )
    coordinates = cast(RegularCoordinates, time_axis.coordinates)
    if coordinates.step != 1:
        raise ValueError("EncodedWaveform TimeAxis coordinates must have step 1")

    kernels = (
        config.kernels.trigger_threshold_code,
        config.kernels.release_threshold_code,
        config.kernels.required_time_over_samples,
        config.kernels.pre_trigger_samples,
        config.kernels.post_trigger_samples,
    )
    prepared_members: list[TensorKernel[Any]] = []
    prepared_dimensions: list[tuple[int, ...]] = []
    for kernel in kernels:
        prepared_kernel, dimensions = prepare_kernel(
            kernel,
            target_axes=config.spec.axes,
            target_device=config.spec.device,
            target_unit=None,
        )
        prepared_members.append(prepared_kernel)
        prepared_dimensions.append(dimensions)

    prepared_kernels = EncodedWaveformKernels(
        members=tuple(prepared_members)
    )
    trigger = _lane_values(
        prepared_kernels.trigger_threshold_code.tensor,
        prepared_dimensions[0],
        shape=config.spec.shape,
        time_dimension=time_dimension,
    )
    release = _lane_values(
        prepared_kernels.release_threshold_code.tensor,
        prepared_dimensions[1],
        shape=config.spec.shape,
        time_dimension=time_dimension,
    )
    if bool((release < trigger).any()):
        raise ValueError(
            "ReleaseThresholdCode must be greater than or equal to "
            "TriggerThresholdCode"
        )

    time_count = config.spec.shape[time_dimension]
    lane_count = math.prod(
        extent
        for dimension, extent in enumerate(config.spec.shape)
        if dimension != time_dimension
    )
    require_tensor_capacity(
        config.spec.shape,
        dtype=config.spec.dtype,
        field="EncodedWaveform output",
    )
    require_tensor_capacity(
        (lane_count, time_count),
        dtype=torch.bool,
        field="EncodedWaveform Boolean workspace",
    )
    require_tensor_capacity(
        (lane_count, time_count + 1),
        dtype=torch.int64,
        field="EncodedWaveform transition workspace",
    )
    require_tensor_capacity(
        (lane_count * (time_count + 1),),
        dtype=torch.int64,
        field="EncodedWaveform component workspace",
    )

    prepared = replace(config, kernels=prepared_kernels)
    for name, value in (
        ("_is_prepared", True),
        ("_source_specs", source_specs),
        ("_working_dtype", config.spec.dtype),
        ("_time_dimension", time_dimension),
        ("_kernel_dimensions", tuple(prepared_dimensions)),
    ):
        object.__setattr__(prepared, name, value)
    return prepared
