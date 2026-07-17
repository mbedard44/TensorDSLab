from __future__ import annotations

from typing import final

import torch
from tensor_core import TensorField

from tensor_dslab.readout._requirements import (
    _require_dtype,
    _require_readout_structure,
)


@final
class Photoelectrons(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_dtype(self, torch.int64)


def _require_valid_values(field: Photoelectrons) -> None:
    if bool(torch.any(field.tensor < 0).item()):
        raise ValueError("Photoelectrons values must be nonnegative")
