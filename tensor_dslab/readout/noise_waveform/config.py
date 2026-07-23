from __future__ import annotations

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import NonnegativeFloat, PositiveFloat, RngKey

from tensor_dslab.common.units import (
    _canonical_quantity,
    canonical_magnitude,
)


_RNG_NAMESPACE = 0x54445331


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroNoiseConfig:
    """Select the exact all-zero noise algorithm."""

    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class WhiteNoiseConfig:
    rms: Quantity
    rng_key: RngKey = RngKey(namespace=_RNG_NAMESPACE, stream=0x0000_0001)
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rms",
            _canonical_quantity(
                self.rms,
                unit="mV",
                field="WhiteNoiseConfig.rms",
                constraint=PositiveFloat,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PsdNoiseConfig:
    frequency_left_edges: tuple[Quantity, ...]
    frequency_stop: Quantity
    power_density: tuple[Quantity, ...]
    rng_key: RngKey = RngKey(namespace=_RNG_NAMESPACE, stream=0x0000_0002)
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        if type(self.frequency_left_edges) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.frequency_left_edges must be a tuple"
            )
        if type(self.power_density) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.power_density must be a tuple"
            )
        if not self.frequency_left_edges:
            raise ValueError("a PSD requires at least one frequency bin")
        if len(self.frequency_left_edges) != len(self.power_density):
            raise ValueError("PSD left-edge and density counts must match")
        canonical_edges = tuple(
            _canonical_quantity(
                edge,
                unit="Hz",
                field=f"PsdNoiseConfig.frequency_left_edges[{index}]",
                constraint=NonnegativeFloat,
            )
            for index, edge in enumerate(self.frequency_left_edges)
        )
        object.__setattr__(self, "frequency_left_edges", canonical_edges)
        object.__setattr__(
            self,
            "frequency_stop",
            _canonical_quantity(
                self.frequency_stop,
                unit="Hz",
                field="PsdNoiseConfig.frequency_stop",
                constraint=PositiveFloat,
            ),
        )
        canonical_density = tuple(
            _canonical_quantity(
                density,
                unit="mV ** 2 / Hz",
                field=f"PsdNoiseConfig.power_density[{index}]",
                constraint=NonnegativeFloat,
            )
            for index, density in enumerate(self.power_density)
        )
        object.__setattr__(self, "power_density", canonical_density)
        if canonical_magnitude(self.frequency_left_edges[0]) != 0.0:
            raise ValueError("PSD frequency coverage must start at zero")
        if any(
            canonical_magnitude(right) <= canonical_magnitude(left)
            for left, right in zip(
                self.frequency_left_edges,
                self.frequency_left_edges[1:],
            )
        ):
            raise ValueError("PSD frequency left edges must be strictly increasing")
        if (
            canonical_magnitude(self.frequency_left_edges[-1])
            >= canonical_magnitude(self.frequency_stop)
        ):
            raise ValueError("PSD frequency stop must exceed its final left edge")
        if not any(
            canonical_magnitude(density) > 0.0
            for density in self.power_density
        ):
            raise ValueError("use ZeroNoiseConfig for an all-zero PSD")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NoiseWaveformConfig:
    model: ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig
    __hash__ = None  # pyright: ignore[reportAssignmentType]
