"""TensorCore semantic field leaf for digitized waveform products."""

from typing import final, override

import torch
from tensor_core import TensorField
from tensor_core.tensor.validation import require_field_dtype

from tensor_dslab.readout.runtime.requirements import (
    require_readout_structure,
)


@final
class DigitizedWaveform(TensorField):
    """Represent ADC codes derived from an analog waveform."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_readout_structure(self)
        require_field_dtype(self, torch.int32)
