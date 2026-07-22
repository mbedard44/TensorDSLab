from __future__ import annotations

import torch

from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.noise_waveform.runtime.prepare import NoiseWaveformRuntime
from tensor_dslab.readout.photoelectrons.field import Photoelectrons


def validate_noise_waveform(
    noise: NoiseWaveform,
    *,
    source: Photoelectrons,
    runtime: NoiseWaveformRuntime,
) -> None:
    if not bool(torch.all(torch.isfinite(noise.tensor)).item()):
        raise ValueError("NoiseWaveform values must be finite")
    if noise.axes is not source.axes or noise.shape != source.shape:
        raise ValueError("NoiseWaveform must preserve source axes and shape")
    if noise.tensor.dtype is not runtime.floating_dtype:
        raise ValueError("NoiseWaveform must use the prepared floating dtype")
    if noise.tensor.device != source.tensor.device:
        raise ValueError("NoiseWaveform must preserve source device")
    if (
        noise.tensor.untyped_storage().data_ptr()
        == source.tensor.untyped_storage().data_ptr()
    ):
        raise ValueError("NoiseWaveform must have fresh storage")
