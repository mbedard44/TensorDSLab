from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from tensor_dslab.readout.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.readout.requirements import require_representable_float


@final
@dataclass(frozen=True, slots=True)
class AnalogWaveformRuntime:
    minimum: torch.Tensor | None
    maximum: torch.Tensor | None


def prepare_analog_waveform(
    config: AnalogWaveformConfig,
    *,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> AnalogWaveformRuntime:
    if type(config) is not AnalogWaveformConfig:
        raise TypeError("config must be exactly AnalogWaveformConfig")
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")

    minimum: float | None = None
    maximum: float | None = None
    if config.saturation is not None:
        if config.saturation.minimum_mv is not None:
            minimum = require_representable_float(
                config.saturation.minimum_mv.value,
                dtype=floating_dtype,
                field="analog saturation minimum",
            )
        if config.saturation.maximum_mv is not None:
            maximum = require_representable_float(
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
    return AnalogWaveformRuntime(
        minimum=minimum_tensor,
        maximum=maximum_tensor,
    )
