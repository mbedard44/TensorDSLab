"""Private completed-product validation for digitized waveform products."""

import torch

from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform


def validate_digitized_waveform(
    digitized: DigitizedWaveform,
    *,
    source: AnalogWaveform,
    maximum_code: int,
) -> None:
    if bool(torch.any(digitized.tensor < 0).item()):
        raise ValueError("DigitizedWaveform values must be nonnegative")
    if bool(torch.any(digitized.tensor > maximum_code).item()):
        raise ValueError("DigitizedWaveform values exceed configured bit depth")
    if digitized.axes is not source.axes or digitized.shape != source.shape:
        raise ValueError("DigitizedWaveform must preserve source axes and shape")
    if digitized.tensor.dtype is not torch.int32:
        raise ValueError("DigitizedWaveform must use torch.int32")
    if digitized.tensor.device != source.tensor.device:
        raise ValueError("DigitizedWaveform must preserve source device")
    if (
        digitized.tensor.untyped_storage().data_ptr()
        == source.tensor.untyped_storage().data_ptr()
    ):
        raise ValueError("DigitizedWaveform must have fresh storage")
