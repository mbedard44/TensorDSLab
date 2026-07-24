"""Public configuration records for noise waveform."""

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import NonnegativeFloat, PositiveFloat

from tensor_dslab.common.units import (
    _canonicalize_quantity_fields,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroNoiseConfig:
    """Select the exact all-zero noise algorithm."""

    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class WhiteNoiseConfig:
    rms: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("rms", "mV", PositiveFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PsdNoiseConfig:
    frequency_left_edges: Quantity
    frequency_stop: Quantity
    power_density: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("frequency_stop", "Hz", PositiveFloat),
            ),
            vector_fields=(
                ("frequency_left_edges", "Hz", NonnegativeFloat),
                ("power_density", "mV ** 2 / Hz", NonnegativeFloat),
            ),
        )
        if len(self.frequency_left_edges) == 0:
            raise ValueError("a PSD requires at least one frequency bin")
        if len(self.frequency_left_edges) != len(self.power_density):
            raise ValueError("PSD left-edge and density counts must match")
        if self.frequency_left_edges[0] != 0.0 * self.frequency_left_edges.units:
            raise ValueError("PSD frequency coverage must start at zero")
        if any(
            right <= left
            for left, right in zip(
                self.frequency_left_edges,
                self.frequency_left_edges[1:],
            )
        ):
            raise ValueError("PSD frequency left edges must be strictly increasing")
        if self.frequency_left_edges[-1] >= self.frequency_stop:
            raise ValueError("PSD frequency stop must exceed its final left edge")
        if not any(
            density > 0.0 * self.power_density.units
            for density in self.power_density
        ):
            raise ValueError("use ZeroNoiseConfig for an all-zero PSD")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NoiseWaveformConfig:
    model: ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig
    __hash__ = None  # pyright: ignore[reportAssignmentType]
