from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_core import FiniteFloat, NonnegativeFloat, PositiveInteger

from tensor_dslab.readout._requirements import _require_exact


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
