from __future__ import annotations

import math

import torch
from tensor_core import require_same_axes, require_same_device

from tensor_dslab.readout.analog_waveform.types import (
    AnalogWaveform,
    AnalogWaveformConfig,
)
from tensor_dslab.readout.noise_waveform import NoiseWaveform
from tensor_dslab.readout.pure_waveform import PureWaveform


def _round_finite_bound(
    value: float,
    *,
    dtype: torch.dtype,
    field: str,
) -> float:
    rounded = float(torch.tensor(value, dtype=dtype, device="cpu"))
    if not math.isfinite(rounded):
        raise ValueError(f"{field} is not finite in the waveform dtype")
    return rounded


def _product_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    config: AnalogWaveformConfig,
) -> AnalogWaveform:
    require_same_axes(pure, noise)
    if pure.shape != noise.shape:
        raise ValueError("pure and noise waveforms must have the same shape")
    require_same_device(pure, noise)
    if pure.tensor.dtype is not noise.tensor.dtype:
        raise ValueError("pure and noise waveforms must have the same dtype")

    minimum: float | None = None
    maximum: float | None = None
    if config.saturation is not None:
        if config.saturation.minimum_mv is not None:
            minimum = _round_finite_bound(
                config.saturation.minimum_mv.value,
                dtype=pure.tensor.dtype,
                field="analog saturation minimum",
            )
        if config.saturation.maximum_mv is not None:
            maximum = _round_finite_bound(
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
