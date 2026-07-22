from __future__ import annotations

import torch
from tensor_core import require_same_axes, require_same_device, require_same_dtype

from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.pure_waveform.field import PureWaveform


def validate_analog_waveform(
    analog: AnalogWaveform,
    *,
    pure: PureWaveform,
    noise: NoiseWaveform,
) -> None:
    if not bool(torch.all(torch.isfinite(analog.tensor)).item()):
        raise ValueError("AnalogWaveform values must be finite")
    require_same_axes(pure, noise)
    if (
        analog.axes is not pure.axes
        or analog.shape != pure.shape
        or analog.shape != noise.shape
    ):
        raise ValueError("AnalogWaveform must preserve prerequisite axes and shape")
    require_same_dtype(pure, noise, analog)
    require_same_device(pure, noise, analog)
    addresses = {
        analog.tensor.untyped_storage().data_ptr(),
        pure.tensor.untyped_storage().data_ptr(),
        noise.tensor.untyped_storage().data_ptr(),
    }
    if len(addresses) != 3:
        raise ValueError("AnalogWaveform must have fresh storage")
