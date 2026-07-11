from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tensor_core import (
    FiniteFloat,
    NonnegativeInteger,
    PositiveFloat,
    PositiveInteger,
)


class AdcQuantization(StrEnum):
    TRUNCATE = "truncate"


@dataclass(frozen=True, slots=True)
class SampleGrid:
    sample_period_ns: PositiveFloat
    origin_ns: FiniteFloat
    sample_offset: NonnegativeInteger

    def __post_init__(self) -> None:
        if type(self.sample_period_ns) is not PositiveFloat:
            raise TypeError("SampleGrid.sample_period_ns must be PositiveFloat")
        if type(self.origin_ns) is not FiniteFloat:
            raise TypeError("SampleGrid.origin_ns must be FiniteFloat")
        if type(self.sample_offset) is not NonnegativeInteger:
            raise TypeError("SampleGrid.sample_offset must be NonnegativeInteger")


@dataclass(frozen=True, slots=True)
class DigitizedWaveformSpec:
    bit_depth: PositiveInteger
    voltage_pp_mv: PositiveFloat
    voltage_offset_mv: FiniteFloat
    analog_gain_db: FiniteFloat
    quantization: AdcQuantization

    def __post_init__(self) -> None:
        if type(self.bit_depth) is not PositiveInteger:
            raise TypeError("DigitizedWaveformSpec.bit_depth must be PositiveInteger")
        if type(self.voltage_pp_mv) is not PositiveFloat:
            raise TypeError("DigitizedWaveformSpec.voltage_pp_mv must be PositiveFloat")
        if type(self.voltage_offset_mv) is not FiniteFloat:
            raise TypeError("DigitizedWaveformSpec.voltage_offset_mv must be FiniteFloat")
        if type(self.analog_gain_db) is not FiniteFloat:
            raise TypeError("DigitizedWaveformSpec.analog_gain_db must be FiniteFloat")
        if type(self.quantization) is not AdcQuantization:
            raise TypeError("DigitizedWaveformSpec.quantization must be AdcQuantization")
        if self.bit_depth.value > 16:
            raise ValueError("DigitizedWaveformSpec.bit_depth must be at most 16")
        if not 0.0 <= self.analog_gain_db.value <= 40.0:
            raise ValueError(
                "DigitizedWaveformSpec.analog_gain_db must be between 0.0 and 40.0"
            )

    @property
    def adc_min(self) -> int:
        return 0

    @property
    def adc_max(self) -> int:
        return (1 << self.bit_depth.value) - 1
