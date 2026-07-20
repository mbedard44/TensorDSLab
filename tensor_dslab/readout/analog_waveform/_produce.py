from __future__ import annotations

from dataclasses import dataclass

import torch
from tensor_core import (
    require_same_axes,
    require_same_device,
    require_same_dtype,
)

from tensor_dslab.readout._requirements import _require_representable_float
from tensor_dslab.readout.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.readout.analog_waveform.field import (
    AnalogWaveform,
    _require_valid_values,
)
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.pure_waveform.field import PureWaveform


@dataclass(frozen=True, slots=True)
class _AnalogWaveformPlan:
    minimum: torch.Tensor | None
    maximum: torch.Tensor | None


def _prepare_analog_waveform(
    *,
    config: AnalogWaveformConfig,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> _AnalogWaveformPlan:
    if type(config) is not AnalogWaveformConfig:
        raise TypeError("config must be exactly AnalogWaveformConfig")
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")

    minimum: float | None = None
    maximum: float | None = None
    if config.saturation is not None:
        if config.saturation.minimum_mv is not None:
            minimum = _require_representable_float(
                config.saturation.minimum_mv.value,
                dtype=floating_dtype,
                field="analog saturation minimum",
            )
        if config.saturation.maximum_mv is not None:
            maximum = _require_representable_float(
                config.saturation.maximum_mv.value,
                dtype=floating_dtype,
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
            dtype=floating_dtype,
            device=device,
        )
    )
    maximum_tensor = (
        None
        if maximum is None
        else torch.tensor(
            maximum,
            dtype=floating_dtype,
            device=device,
        )
    )
    return _AnalogWaveformPlan(
        minimum=minimum_tensor,
        maximum=maximum_tensor,
    )


def _produce_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    plan: _AnalogWaveformPlan,
) -> AnalogWaveform:
    require_same_axes(pure, noise)
    if pure.shape != noise.shape:
        raise ValueError("pure and noise waveforms must have the same shape")
    require_same_device(pure, noise)
    require_same_dtype(pure, noise)

    values = torch.add(pure.tensor, noise.tensor)
    if plan.minimum is not None or plan.maximum is not None:
        values = torch.clamp(values, min=plan.minimum, max=plan.maximum)
    result = AnalogWaveform(tensor=values, axes=pure.axes)
    _require_valid_values(result)
    return result
