from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from tensor_core import require_representable_float

from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.analog_waveform.config import AnalogWaveformConfig


@final
@dataclass(frozen=True, slots=True)
class AnalogWaveformRuntime:
    minimum_mv: torch.Tensor | None
    maximum_mv: torch.Tensor | None


def prepare_analog_waveform(
    config: AnalogWaveformConfig,
    *,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> AnalogWaveformRuntime:
    minimum: float | None = None
    maximum: float | None = None
    if config.saturation is not None:
        if config.saturation.minimum is not None:
            minimum_mv = canonical_magnitude(config.saturation.minimum)
            minimum = require_representable_float(
                minimum_mv,
                dtype=floating_dtype,
                field="analog saturation minimum",
            )
        if config.saturation.maximum is not None:
            maximum_mv = canonical_magnitude(config.saturation.maximum)
            maximum = require_representable_float(
                maximum_mv,
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
        minimum_mv=minimum_tensor,
        maximum_mv=maximum_tensor,
    )
