from __future__ import annotations

import torch
from tensor_core import (
    require_same_axes,
    require_same_device,
    require_same_dtype,
)

from tensor_dslab.readout._requirements import _require_representable_float
from tensor_dslab.readout.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.pure_waveform.field import PureWaveform


def _produce_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    config: AnalogWaveformConfig,
) -> AnalogWaveform:
    require_same_axes(pure, noise)
    if pure.shape != noise.shape:
        raise ValueError("pure and noise waveforms must have the same shape")
    require_same_device(pure, noise)
    require_same_dtype(pure, noise)

    minimum: float | None = None
    maximum: float | None = None
    if config.saturation is not None:
        if config.saturation.minimum_mv is not None:
            minimum = _require_representable_float(
                config.saturation.minimum_mv.value,
                dtype=pure.tensor.dtype,
                field="analog saturation minimum",
            )
        if config.saturation.maximum_mv is not None:
            maximum = _require_representable_float(
                config.saturation.maximum_mv.value,
                dtype=pure.tensor.dtype,
                field="analog saturation maximum",
            )
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(
                "analog saturation bounds collapse in the waveform dtype"
            )

    minimum_tensor = (
        None
        if minimum is None
        else torch.tensor(
            minimum,
            dtype=pure.tensor.dtype,
            device=pure.tensor.device,
        )
    )
    maximum_tensor = (
        None
        if maximum is None
        else torch.tensor(
            maximum,
            dtype=pure.tensor.dtype,
            device=pure.tensor.device,
        )
    )

    values = torch.add(pure.tensor, noise.tensor)
    if minimum_tensor is not None or maximum_tensor is not None:
        values = torch.clamp(values, min=minimum_tensor, max=maximum_tensor)
    return AnalogWaveform(tensor=values, axes=pure.axes)
