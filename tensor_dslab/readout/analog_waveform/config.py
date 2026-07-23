from __future__ import annotations

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import FiniteFloat

from tensor_dslab.common.units import (
    _canonical_quantity,
    canonical_magnitude,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogSaturationConfig:
    minimum: Quantity | None = None
    maximum: Quantity | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        if self.minimum is not None:
            object.__setattr__(
                self,
                "minimum",
                _canonical_quantity(
                    self.minimum,
                    unit="mV",
                    field="AnalogSaturationConfig.minimum",
                    constraint=FiniteFloat,
                ),
            )
        if self.maximum is not None:
            object.__setattr__(
                self,
                "maximum",
                _canonical_quantity(
                    self.maximum,
                    unit="mV",
                    field="AnalogSaturationConfig.maximum",
                    constraint=FiniteFloat,
                ),
            )
        if self.minimum is None and self.maximum is None:
            raise ValueError("analog saturation requires at least one bound")
        if (
            self.minimum is not None
            and self.maximum is not None
            and canonical_magnitude(self.minimum)
            >= canonical_magnitude(self.maximum)
        ):
            raise ValueError("analog saturation minimum must be below maximum")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogWaveformConfig:
    saturation: AnalogSaturationConfig | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]
