from __future__ import annotations

from typing import final

from tensor_core import TensorField

from tensor_dslab.readout.requirements import (
    require_floating_dtype,
    require_readout_structure,
)


@final
class PureWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_floating_dtype(self)
