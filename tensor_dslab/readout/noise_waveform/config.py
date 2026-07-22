from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_core import NonnegativeFloat, PositiveFloat, RngKey

from tensor_dslab.readout.requirements import (
    require_exact,
    require_one_of_exact,
)


_RNG_NAMESPACE = 0x54445331


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroNoiseConfig:
    """Select the exact all-zero noise algorithm."""


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class WhiteNoiseConfig:
    rms_mv: PositiveFloat
    rng_key: RngKey = RngKey(namespace=_RNG_NAMESPACE, stream=0x0000_0001)

    def __post_init__(self) -> None:
        require_exact(self.rms_mv, PositiveFloat, "WhiteNoiseConfig.rms_mv")
        require_exact(self.rng_key, RngKey, "WhiteNoiseConfig.rng_key")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PsdNoiseConfig:
    frequency_left_edges_hz: tuple[NonnegativeFloat, ...]
    frequency_stop_hz: PositiveFloat
    power_density_mv2_per_hz: tuple[NonnegativeFloat, ...]
    rng_key: RngKey = RngKey(namespace=_RNG_NAMESPACE, stream=0x0000_0002)

    def __post_init__(self) -> None:
        if type(self.frequency_left_edges_hz) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.frequency_left_edges_hz must be a tuple"
            )
        if type(self.power_density_mv2_per_hz) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.power_density_mv2_per_hz must be a tuple"
            )
        if not self.frequency_left_edges_hz:
            raise ValueError("a PSD requires at least one frequency bin")
        if len(self.frequency_left_edges_hz) != len(
            self.power_density_mv2_per_hz
        ):
            raise ValueError("PSD left-edge and density counts must match")
        for edge in self.frequency_left_edges_hz:
            require_exact(
                edge,
                NonnegativeFloat,
                "PsdNoiseConfig.frequency_left_edges_hz",
            )
        require_exact(
            self.frequency_stop_hz,
            PositiveFloat,
            "PsdNoiseConfig.frequency_stop_hz",
        )
        if self.frequency_left_edges_hz[0].value != 0.0:
            raise ValueError("PSD frequency coverage must start at zero")
        if any(
            right.value <= left.value
            for left, right in zip(
                self.frequency_left_edges_hz,
                self.frequency_left_edges_hz[1:],
            )
        ):
            raise ValueError("PSD frequency left edges must be strictly increasing")
        if (
            self.frequency_left_edges_hz[-1].value
            >= self.frequency_stop_hz.value
        ):
            raise ValueError("PSD frequency stop must exceed its final left edge")
        for density in self.power_density_mv2_per_hz:
            require_exact(
                density,
                NonnegativeFloat,
                "PsdNoiseConfig.power_density_mv2_per_hz",
            )
        if not any(
            density.value > 0.0 for density in self.power_density_mv2_per_hz
        ):
            raise ValueError("use ZeroNoiseConfig for an all-zero PSD")
        require_exact(self.rng_key, RngKey, "PsdNoiseConfig.rng_key")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NoiseWaveformConfig:
    model: ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig

    def __post_init__(self) -> None:
        require_one_of_exact(
            self.model,
            (ZeroNoiseConfig, WhiteNoiseConfig, PsdNoiseConfig),
            "NoiseWaveformConfig.model",
        )
