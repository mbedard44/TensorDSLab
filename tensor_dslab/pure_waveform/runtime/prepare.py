"""Prepare one PureWaveform Config for exact staged execution."""

from dataclasses import replace

import torch

from tensor_dslab.common.alignment import (
    prepare_kernel,
    prepare_sources,
)
from tensor_dslab.common.requirements.capacity import require_tensor_capacity
from tensor_dslab.pure_waveform.config import PureWaveformConfig


def prepare_pure_waveform(*, source_specs: tuple, config: PureWaveformConfig) -> PureWaveformConfig:
    """Return a fresh prepared same-type pulse-convolution Config."""

    pulse = config.kernels.pulse_response
    target_source_unit = config.spec.unit / pulse.spec.unit
    dimensions, scales, dtype = prepare_sources(
        source_specs,
        target_spec=config.spec,
        minimum_count=1,
        unit_target=target_source_unit,
    )
    dtype = torch.promote_types(torch.promote_types(dtype, pulse.dtype), torch.float32)
    require_tensor_capacity(
        config.spec.shape,
        dtype=dtype,
        field="PureWaveform workspace",
    )
    prepared_pulse, kdims = prepare_kernel(
        pulse,
        target_axes=config.spec.axes,
        target_device=config.spec.device,
        target_unit=pulse.spec.unit,
    )
    prepared = replace(
        config,
        kernels=type(config.kernels)(members=(prepared_pulse,)),
    )
    object.__setattr__(prepared, "_is_prepared", True)
    object.__setattr__(prepared, "_source_specs", source_specs)
    object.__setattr__(prepared, "_source_dimensions", dimensions)
    object.__setattr__(prepared, "_source_scales", scales)
    object.__setattr__(prepared, "_working_dtype", dtype)
    object.__setattr__(prepared, "_kernel_dimensions", (kdims,))
    return prepared
