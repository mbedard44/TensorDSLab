"""Public configuration records for pure waveform."""

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import FiniteFloat, PositiveFloat

from tensor_dslab.common.units import (
    _canonicalize_quantity_fields,
    canonical_magnitude,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TpcFebSnrPulseConfig:
    """Configure the calibrated TPC FEB SNR pulse model."""

    fast_time_constant: Quantity
    slow_time_constant: Quantity
    support_time: Quantity
    peak_voltage_per_photoelectron: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("fast_time_constant", "ns", PositiveFloat),
                ("slow_time_constant", "ns", PositiveFloat),
                ("support_time", "ns", PositiveFloat),
                ("peak_voltage_per_photoelectron", "mV", PositiveFloat),
            ),
        )
        if canonical_magnitude(self.slow_time_constant) <= canonical_magnitude(
            self.fast_time_constant
        ):
            raise ValueError("slow time constant must exceed fast time constant")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class VetoPduPulseConfig:
    """Configure the calibrated veto PDU pulse model."""

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
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("gaussian_center", "ns", FiniteFloat),
                ("gaussian_width", "ns", PositiveFloat),
                ("edge_offset_1", "ns", FiniteFloat),
                ("edge_width_1", "ns", PositiveFloat),
                ("edge_offset_2", "ns", FiniteFloat),
                ("edge_width_2", "ns", PositiveFloat),
                ("support_time", "ns", PositiveFloat),
                ("peak_voltage_per_photoelectron", "mV", PositiveFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PureWaveformConfig:
    """Select one accepted pure-waveform pulse model."""

    model: TpcFebSnrPulseConfig | VetoPduPulseConfig
    __hash__ = None  # pyright: ignore[reportAssignmentType]
