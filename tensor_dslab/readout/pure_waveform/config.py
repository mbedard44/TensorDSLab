from __future__ import annotations

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import FiniteFloat, PositiveFloat

from tensor_dslab.common.units import (
    _canonical_quantity,
    canonical_magnitude,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TpcFebSnrPulseConfig:
    fast_time_constant: Quantity
    slow_time_constant: Quantity
    support_time: Quantity
    peak_voltage_per_photoelectron: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fast_time_constant",
            _canonical_quantity(
                self.fast_time_constant,
                unit="ns",
                field="TpcFebSnrPulseConfig.fast_time_constant",
                constraint=PositiveFloat,
            ),
        )
        object.__setattr__(
            self,
            "slow_time_constant",
            _canonical_quantity(
                self.slow_time_constant,
                unit="ns",
                field="TpcFebSnrPulseConfig.slow_time_constant",
                constraint=PositiveFloat,
            ),
        )
        object.__setattr__(
            self,
            "support_time",
            _canonical_quantity(
                self.support_time,
                unit="ns",
                field="TpcFebSnrPulseConfig.support_time",
                constraint=PositiveFloat,
            ),
        )
        object.__setattr__(
            self,
            "peak_voltage_per_photoelectron",
            _canonical_quantity(
                self.peak_voltage_per_photoelectron,
                unit="mV",
                field="TpcFebSnrPulseConfig.peak_voltage_per_photoelectron",
                constraint=FiniteFloat,
            ),
        )
        if canonical_magnitude(self.slow_time_constant) <= canonical_magnitude(
            self.fast_time_constant
        ):
            raise ValueError("slow time constant must exceed fast time constant")
        if canonical_magnitude(self.peak_voltage_per_photoelectron) == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class VetoPduPulseConfig:
    gaussian_center: Quantity
    gaussian_width: Quantity
    edge_offset_1: Quantity
    edge_width_1: Quantity
    edge_offset_2: Quantity
    edge_width_2: Quantity
    support_time: Quantity
    peak_voltage_per_photoelectron: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        for name, constraint in (
            ("gaussian_center", FiniteFloat),
            ("gaussian_width", PositiveFloat),
            ("edge_offset_1", FiniteFloat),
            ("edge_width_1", PositiveFloat),
            ("edge_offset_2", FiniteFloat),
            ("edge_width_2", PositiveFloat),
            ("support_time", PositiveFloat),
            ("peak_voltage_per_photoelectron", FiniteFloat),
        ):
            object.__setattr__(
                self,
                name,
                _canonical_quantity(
                    getattr(self, name),
                    unit="mV" if name == "peak_voltage_per_photoelectron" else "ns",
                    field=f"VetoPduPulseConfig.{name}",
                    constraint=constraint,
                ),
            )
        if canonical_magnitude(self.peak_voltage_per_photoelectron) == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PureWaveformConfig:
    model: TpcFebSnrPulseConfig | VetoPduPulseConfig
    __hash__ = None  # pyright: ignore[reportAssignmentType]
