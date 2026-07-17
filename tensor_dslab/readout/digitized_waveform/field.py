from __future__ import annotations

from typing import final

import torch
from tensor_core import TensorField

from tensor_dslab.readout._requirements import (
    _require_dtype,
    _require_readout_structure,
)
from tensor_dslab.readout.digitized_waveform.config import (
    DigitizedWaveformConfig,
)


@final
class DigitizedWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_dtype(self, torch.int32)


def _require_valid_values(
    field: DigitizedWaveform,
    config: DigitizedWaveformConfig,
) -> None:
    maximum_code = (1 << config.bit_depth.value) - 1
    if bool(torch.any(field.tensor < 0).item()):
        raise ValueError("DigitizedWaveform values must be nonnegative")
    if bool(torch.any(field.tensor > maximum_code).item()):
        raise ValueError("DigitizedWaveform values exceed configured bit depth")
