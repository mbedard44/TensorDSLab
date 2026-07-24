"""Public configuration records for charge."""

from dataclasses import dataclass
from typing import final

from pint import Quantity
from tensor_core import (
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    Probability,
)

from tensor_dslab.common.units import _canonicalize_quantity_fields


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TimingJitterConfig:
    """Configure Gaussian photoelectron timing jitter."""

    sigma: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("sigma", "ns", NonnegativeFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DarkCountConfig:
    """Configure independent dark-count generation."""

    rate: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("rate", "Hz", NonnegativeFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FixedDelayConfig:
    """Configure one fixed phase-marginalized crosstalk delay."""

    delay: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("delay", "ns", NonnegativeFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExponentialDelayConfig:
    """Configure one exponential phase-marginalized crosstalk delay."""

    mean_delay: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("mean_delay", "ns", PositiveFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectCrosstalkConfig:
    """Configure direct prompt crosstalk branching."""

    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig
    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DelayedCrosstalkConfig:
    """Configure delayed crosstalk branching and delay law."""

    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig
    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseRecoveryConfig:
    """Configure afterpulse recovery as a function of delay."""

    time_constant: Quantity
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("time_constant", "ns", PositiveFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseConfig:
    """Configure afterpulse branching, delay, and optional recovery."""

    probability: Probability
    mean_delay: Quantity
    recovery: AfterpulseRecoveryConfig | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        _canonicalize_quantity_fields(
            self,
            scalar_fields=(
                ("mean_delay", "ns", PositiveFloat),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CorrelatedAvalancheConfig:
    """Compose the accepted correlated-avalanche mechanisms."""

    maximum_generations: NonnegativeInteger
    direct_crosstalk: DirectCrosstalkConfig | None = None
    delayed_crosstalk: DelayedCrosstalkConfig | None = None
    afterpulse: AfterpulseConfig | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeSmearingConfig:
    """Configure relative Gaussian charge-response smearing."""

    relative_sigma: NonnegativeFloat
    __hash__ = None  # pyright: ignore[reportAssignmentType]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeConfig:
    """Compose the enabled aggregate charge effects."""

    dark_count: DarkCountConfig | None = None
    timing_jitter: TimingJitterConfig | None = None
    correlated_avalanches: CorrelatedAvalancheConfig | None = None
    smearing: ChargeSmearingConfig | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]
