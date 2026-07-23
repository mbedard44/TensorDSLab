from __future__ import annotations

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import (
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    Probability,
    RngKey,
)

from tensor_dslab.common.units import _canonical_quantity


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TimingJitterConfig:
    sigma: Quantity
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_0008)
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sigma",
            _canonical_quantity(
                self.sigma,
                unit="ns",
                field="TimingJitterConfig.sigma",
                constraint=NonnegativeFloat,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DarkCountConfig:
    rate: Quantity
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_0003)
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rate",
            _canonical_quantity(
                self.rate,
                unit="Hz",
                field="DarkCountConfig.rate",
                constraint=NonnegativeFloat,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FixedDelayConfig:
    delay: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delay",
            _canonical_quantity(
                self.delay,
                unit="ns",
                field="FixedDelayConfig.delay",
                constraint=NonnegativeFloat,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExponentialDelayConfig:
    mean_delay: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean_delay",
            _canonical_quantity(
                self.mean_delay,
                unit="ns",
                field="ExponentialDelayConfig.mean_delay",
                constraint=PositiveFloat,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig
    retained_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0004,
    )
    overflow_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0005,
    )
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        if self.retained_rng_key == self.overflow_rng_key:
            raise ValueError(
                "DirectCrosstalkConfig retained and overflow RNG keys must differ"
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DelayedCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig
    retained_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0006,
    )
    overflow_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0007,
    )
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        if self.retained_rng_key == self.overflow_rng_key:
            raise ValueError(
                "DelayedCrosstalkConfig retained and overflow RNG keys must differ"
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseRecoveryConfig:
    time_constant: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_constant",
            _canonical_quantity(
                self.time_constant,
                unit="ns",
                field="AfterpulseRecoveryConfig.time_constant",
                constraint=PositiveFloat,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseConfig:
    probability: Probability
    mean_delay: Quantity
    recovery: AfterpulseRecoveryConfig | None = None
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_0009)
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean_delay",
            _canonical_quantity(
                self.mean_delay,
                unit="ns",
                field="AfterpulseConfig.mean_delay",
                constraint=PositiveFloat,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CorrelatedAvalancheConfig:
    maximum_generations: NonnegativeInteger
    direct_crosstalk: DirectCrosstalkConfig | None = None
    delayed_crosstalk: DelayedCrosstalkConfig | None = None
    afterpulse: AfterpulseConfig | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeSmearingConfig:
    relative_sigma: NonnegativeFloat
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_000A)
    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeConfig:
    dark_count: DarkCountConfig | None = None
    timing_jitter: TimingJitterConfig | None = None
    correlated_avalanches: CorrelatedAvalancheConfig | None = None
    smearing: ChargeSmearingConfig | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]
