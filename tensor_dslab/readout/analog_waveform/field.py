from __future__ import annotations

from typing import final

import torch
from tensor_core import TensorField, require_field_dtype

from tensor_dslab.readout.runtime.requirements import (
    require_readout_structure,
)


@final
class AnalogWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_field_dtype(self, torch.float32, torch.float64)
