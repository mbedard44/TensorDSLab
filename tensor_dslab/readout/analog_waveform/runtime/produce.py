"""Private tensor execution for analog waveform products."""

import torch

from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.analog_waveform.runtime.prepare import (
    AnalogWaveformRuntime,
)
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.pure_waveform.field import PureWaveform


def produce_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    runtime: AnalogWaveformRuntime,
) -> AnalogWaveform:
    values = torch.add(pure.tensor, noise.tensor)
    if runtime.minimum_mv is not None or runtime.maximum_mv is not None:
        values = torch.clamp(
            values,
            min=runtime.minimum_mv,
            max=runtime.maximum_mv,
        )
    return AnalogWaveform(tensor=values, axes=pure.axes)
