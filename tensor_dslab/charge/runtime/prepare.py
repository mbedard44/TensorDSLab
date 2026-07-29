"""Prepare one Charge Config for staged stochastic execution."""

from dataclasses import replace

import torch
from tensor_core import GaussianDistribution, OffsetAxis, RegularCoordinates
from typing import cast

from tensor_dslab.charge.config import ChargeConfig
from tensor_dslab.common import TimeAxis
from tensor_dslab.common.alignment import (
    prepare_kernel,
    prepare_sources,
)
from tensor_dslab.common.requirements.capacity import (
    require_address_capacity,
    require_tensor_capacity,
)
from tensor_dslab.common.units import unit_registry


def prepare_charge(*, source_specs: tuple, config: ChargeConfig) -> ChargeConfig:
    source_dimensions, scales, dtype = prepare_sources(
        source_specs,
        target_spec=config.spec,
        minimum_count=1,
    )
    if any(source.dtype is not torch.int64 for source in source_specs):
        raise TypeError("Charge sources must use torch.int64")
    dtype = torch.promote_types(dtype, torch.float64)
    require_tensor_capacity(
        config.spec.shape,
        dtype=torch.int64,
        field="Charge count workspace",
    )
    require_tensor_capacity(
        config.spec.shape,
        dtype=dtype,
        field="Charge floating workspace",
    )
    kernels_and_units = (
        (config.kernels.timing_jitter, unit_registry.Unit("")),
        (config.kernels.direct_crosstalk, unit_registry.Unit("")),
        (config.kernels.delayed_crosstalk, unit_registry.Unit("")),
        (config.kernels.afterpulse, unit_registry.Unit("")),
        (
            config.kernels.dark_count_rate,
            unit_registry.Unit("avalanche / s"),
        ),
        (config.kernels.smearing_width, unit_registry.Unit("")),
    )
    prepared_members = []
    kdims: list[tuple[int, ...] | None] = []
    for kernel, target_unit in kernels_and_units:
        if kernel is None:
            kdims.append(None)
            continue
        prepared_kernel, aligned_dimensions = prepare_kernel(
            kernel,
            target_axes=config.spec.axes,
            target_device=config.spec.device,
            target_unit=target_unit,
        )
        prepared_members.append(prepared_kernel)
        kdims.append(aligned_dimensions)
        if kernel is config.kernels.timing_jitter:
            require_tensor_capacity(
                (*config.spec.shape, *kernel.operation_shape),
                dtype=torch.int64,
                field="TimingJitter category workspace",
            )
    needs_time = (
        config.kernels.timing_jitter is not None
        or config.kernels.delayed_crosstalk is not None
        or config.kernels.afterpulse is not None
        or config.kernels.dark_count_rate is not None
        or (
            config.kernels.direct_crosstalk is not None
            and any(
                cast(OffsetAxis, axis).relative_to is TimeAxis
                for axis in config.kernels.direct_crosstalk.operation_axes
            )
        )
    )
    temporal_dimension = None
    temporal_step_seconds = None
    if needs_time:
        try:
            temporal_dimension = config.spec.dimension_of(TimeAxis)
        except KeyError as error:
            raise ValueError("enabled Charge mechanisms require TimeAxis") from error
        axis = config.spec.axes[temporal_dimension]
        if type(axis.coordinates) is RegularCoordinates and axis.coordinates.step != 1:
            raise ValueError("regular Charge TimeAxis requires coordinate step one")
        temporal_step_seconds = float(
            unit_registry.Quantity(axis.coordinate_scale, axis.unit).to("s").magnitude
        )
    prepared_kernels = type(config.kernels)(members=tuple(prepared_members))
    prepared_dark = prepared_kernels.dark_count_rate
    if prepared_dark is not None:
        assert temporal_step_seconds is not None
        dark_mean = prepared_dark.tensor * temporal_step_seconds
        if not bool(torch.isfinite(dark_mean).all()) or bool(
            (dark_mean > 1.0e8).any()
        ):
            raise ValueError("dark-count mean exceeds the Poisson domain")
    prepared_smearing = prepared_kernels.smearing_width
    if prepared_smearing is not None:
        maximum = torch.full_like(
            prepared_smearing.tensor,
            float((1 << 53) - 1),
            dtype=dtype,
        )
        width = prepared_smearing.tensor.to(dtype)
        GaussianDistribution(
            mean=maximum,
            standard_deviation=width * torch.sqrt(maximum),
            dtype=dtype,
        )
    if config.kernels.members:
        require_address_capacity(
            config.spec.shape,
            address_shape=(),
            field="Charge point address",
        )
    if config.kernels.timing_jitter is not None:
        require_address_capacity(
            config.spec.shape,
            address_shape=config.kernels.timing_jitter.operation_shape,
            field="TimingJitter address",
        )
    if config.correlated_avalanche_generations.value:
        require_address_capacity(
            config.spec.shape,
            address_shape=(
                config.correlated_avalanche_generations.value,
            ),
            field="Charge branching address",
        )
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
        ("_kernel_dimensions", tuple(kdims)),
        ("_temporal_dimension", temporal_dimension),
        ("_temporal_step_seconds", temporal_step_seconds),
    ):
        object.__setattr__(prepared, name, value)
    return prepared
