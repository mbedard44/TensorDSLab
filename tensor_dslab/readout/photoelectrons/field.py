from __future__ import annotations

from typing import final

from tensor_core import TensorField

import torch

from tensor_dslab.readout.requirements import (
    require_dtype,
    require_readout_structure,
)


@final
class Photoelectrons(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_dtype(self, torch.int64)
