from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from tensor_core import FiniteFloat, NonnegativeFloat, PositiveInteger, TensorField

from tensor_dslab.readout._requirements import (
    _require_dtype,
    _require_exact,
    _require_readout_structure,
)


@final
class DigitizedWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_dtype(self, torch.int32)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DigitizedWaveformConfig:
    bit_depth: PositiveInteger
    input_min_mv: FiniteFloat
    input_max_mv: FiniteFloat
    analog_gain_db: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.bit_depth,
            PositiveInteger,
            "DigitizedWaveformConfig.bit_depth",
        )
        _require_exact(
            self.input_min_mv,
            FiniteFloat,
            "DigitizedWaveformConfig.input_min_mv",
        )
        _require_exact(
            self.input_max_mv,
            FiniteFloat,
            "DigitizedWaveformConfig.input_max_mv",
        )
        _require_exact(
            self.analog_gain_db,
            NonnegativeFloat,
            "DigitizedWaveformConfig.analog_gain_db",
        )
        if self.bit_depth.value > 16:
            raise ValueError("bit_depth must be between 1 and 16")
        if self.input_min_mv.value >= self.input_max_mv.value:
            raise ValueError("ADC input minimum must be below maximum")
        if self.analog_gain_db.value > 40.0:
            raise ValueError("analog_gain_db must be between 0 and 40")


def _require_valid_values(
    field: DigitizedWaveform,
    config: DigitizedWaveformConfig,
) -> None:
    maximum_code = (1 << config.bit_depth.value) - 1
    if bool(torch.any(field.tensor < 0).item()):
        raise ValueError("DigitizedWaveform values must be nonnegative")
    if bool(torch.any(field.tensor > maximum_code).item()):
        raise ValueError("DigitizedWaveform values exceed configured bit depth")
