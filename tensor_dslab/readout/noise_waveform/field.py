from __future__ import annotations

from typing import final

import torch
from tensor_core import TensorField

from tensor_dslab.readout._requirements import (
    _require_floating_dtype,
    _require_readout_structure,
)


@final
class NoiseWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_floating_dtype(self)


def _require_valid_values(field: NoiseWaveform) -> None:
    if not bool(torch.all(torch.isfinite(field.tensor)).item()):
        raise ValueError("NoiseWaveform values must be finite")
