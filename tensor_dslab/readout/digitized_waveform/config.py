"""Public configuration records for digitized waveform."""

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import FiniteFloat, NonnegativeFloat, PositiveInteger

from tensor_dslab.common.units import (
    _canonicalize_quantity_fields,
    canonical_magnitude,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DigitizedWaveformConfig:
    """Configure the ADC range, depth, and analog gain."""

    bit_depth: PositiveInteger
    input_minimum: Quantity
    input_maximum: Quantity
    analog_gain_db: NonnegativeFloat
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("input_minimum", "mV", FiniteFloat),
                ("input_maximum", "mV", FiniteFloat),
            ),
        )
        if self.bit_depth.value > 16:
            raise ValueError("bit_depth must be between 1 and 16")
        if canonical_magnitude(self.input_minimum) >= canonical_magnitude(
            self.input_maximum
        ):
            raise ValueError("ADC input minimum must be below maximum")
        if self.analog_gain_db.value > 40.0:
            raise ValueError("analog_gain_db must be between 0 and 40")
