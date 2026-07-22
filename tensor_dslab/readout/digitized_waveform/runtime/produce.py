from __future__ import annotations

import torch

from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform
from tensor_dslab.readout.digitized_waveform.runtime.prepare import (
    DigitizedWaveformRuntime,
)


def produce_digitized_waveform(
    analog: AnalogWaveform,
    *,
    runtime: DigitizedWaveformRuntime,
) -> DigitizedWaveform:
    interior = torch.clamp(
        torch.add(
            torch.mul(analog.tensor, runtime.slope),
            runtime.intercept,
        ),
        min=runtime.zero,
        max=runtime.maximum,
    )
    code_float = torch.where(
        analog.tensor <= runtime.lower_input,
        runtime.zero,
        torch.where(
            analog.tensor >= runtime.upper_input,
            runtime.maximum,
            interior,
        ),
    )
    values = code_float.to(dtype=torch.int32)
    return DigitizedWaveform(tensor=values, axes=analog.axes)
