"""Private completed-product validation for pure waveform products."""

import torch

from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.pure_waveform.field import PureWaveform


def validate_pure_waveform(pure: PureWaveform, *, source: Charge) -> None:
    if not bool(torch.all(torch.isfinite(pure.tensor)).item()):
        raise ValueError("PureWaveform values must be finite")
    if pure.axes is not source.axes or pure.shape != source.shape:
        raise ValueError("PureWaveform must preserve Charge axes and shape")
    if pure.tensor.dtype is not source.tensor.dtype:
        raise ValueError("PureWaveform must preserve Charge dtype")
    if pure.tensor.device != source.tensor.device:
        raise ValueError("PureWaveform must preserve Charge device")
    if (
        pure.tensor.untyped_storage().data_ptr()
        == source.tensor.untyped_storage().data_ptr()
    ):
        raise ValueError("PureWaveform must have fresh storage")
