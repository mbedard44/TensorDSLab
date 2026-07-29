"""Prepare one NoiseWaveform Config for staged execution."""

from dataclasses import replace
import math

import pint
import torch
from tensor_core import GaussianDistribution, RegularCoordinates
from typing import cast

from tensor_dslab.common import FrequencyAxis, TimeAxis
from tensor_dslab.common.alignment import (
    prepare_kernel,
    prepare_sources,
    require_address_capacity,
    require_allocation,
)
from tensor_dslab.common.units import unit_registry
from tensor_dslab.noise_waveform.config import NoiseWaveformConfig


def prepare_noise_waveform(*, source_specs: tuple, config: NoiseWaveformConfig) -> NoiseWaveformConfig:
    dimensions, scales, dtype = prepare_sources(
        source_specs, target_spec=config.spec, minimum_count=0, maximum_count=0
    )
    dtype = torch.promote_types(dtype, torch.float32)
    temporal_dimension = None
    temporal_step_seconds = None
    kdims: tuple[tuple[int, ...] | None, ...] = (None, None)
    white = config.kernels.white_noise_rms
    psd = config.kernels.power_spectral_density
    prepared_members = []
    if white is not None:
        white, white_dimensions = prepare_kernel(
            white,
            target_axes=config.spec.axes,
            target_device=config.spec.device,
            target_unit=config.spec.unit,
        )
        prepared_members.append(white)
        dtype = torch.promote_types(dtype, white.dtype)
        GaussianDistribution(
            mean=0.0,
            standard_deviation=white.tensor.to(dtype),
            dtype=dtype,
        )
        kdims = (white_dimensions, None)
    if psd is not None:
        psd, psd_dimensions = prepare_kernel(
            psd,
            target_axes=config.spec.axes,
            target_device=config.spec.device,
            target_unit=cast(pint.Unit, config.spec.unit**2),
            include_operations=False,
        )
        prepared_members.append(psd)
        dtype = torch.promote_types(dtype, psd.dtype)
        try:
            temporal_dimension = config.spec.dimension_of(TimeAxis)
        except KeyError as error:
            raise ValueError("PSD noise requires one TimeAxis") from error
        time_axis = cast(TimeAxis, config.spec.axes[temporal_dimension])
        if type(time_axis.coordinates) is not RegularCoordinates or time_axis.coordinates.step != 1:
            raise ValueError("PSD noise requires a unit-step regular TimeAxis")
        temporal_step_seconds = float(
            unit_registry.Quantity(time_axis.coordinate_scale, time_axis.unit).to("s").magnitude
        )
        frequency_axis = cast(FrequencyAxis, psd.operation_axes[0])
        if frequency_axis.size != time_axis.size // 2 + 1:
            raise ValueError("PSD frequency count does not match TimeAxis")
        spacing = float(
            unit_registry.Quantity(frequency_axis.coordinate_scale, frequency_axis.unit)
            .to("Hz")
            .magnitude
        )
        expected = 1.0 / (time_axis.size * temporal_step_seconds)
        if not math.isclose(spacing, expected, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("PSD frequency spacing does not match TimeAxis")
        kdims = (
            None,
            psd_dimensions,
        )
        non_temporal_shape = (
            config.spec.shape[:temporal_dimension]
            + config.spec.shape[temporal_dimension + 1 :]
        )
        complex_dtype = (
            torch.complex64 if dtype is torch.float32 else torch.complex128
        )
        require_allocation(
            (*non_temporal_shape, frequency_axis.size),
            dtype=complex_dtype,
            field="PSD coefficient workspace",
        )
        row_count = math.prod(non_temporal_shape)
        require_address_capacity(
            (row_count, frequency_axis.size),
            address_shape=(),
            field="PSD noise address",
        )
        require_allocation(
            (row_count, frequency_axis.size - 1, 2),
            dtype=dtype,
            field="PSD Gaussian workspace",
        )
    require_allocation(
        config.spec.shape,
        dtype=dtype,
        field="NoiseWaveform workspace",
    )
    if white is not None:
        require_address_capacity(
            config.spec.shape,
            address_shape=(),
            field="white-noise address",
        )
    prepared = replace(
        config,
        kernels=type(config.kernels)(members=tuple(prepared_members)),
    )
    for name, value in (
        ("_is_prepared", True),
        ("_source_specs", source_specs),
        ("_source_dimensions", dimensions),
        ("_source_scales", scales),
        ("_working_dtype", dtype),
        ("_kernel_dimensions", kdims),
        ("_temporal_dimension", temporal_dimension),
        ("_temporal_step_seconds", temporal_step_seconds),
    ):
        object.__setattr__(prepared, name, value)
    return prepared
