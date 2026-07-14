from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from tensor_core import (
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    Probability,
    TensorField,
)

from tensor_dslab.readout._requirements import (
    _require_exact,
    _require_floating_dtype,
    _require_one_of_exact,
    _require_optional_exact,
    _require_readout_structure,
)


@final
class Charge(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_floating_dtype(self)


def _require_valid_values(field: Charge) -> None:
    if not bool(torch.all(torch.isfinite(field.tensor)).item()):
        raise ValueError("Charge values must be finite")
    if bool(torch.any(field.tensor < 0).item()):
        raise ValueError("Charge values must be nonnegative")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TimingJitterConfig:
    sigma_ns: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.sigma_ns,
            NonnegativeFloat,
            "TimingJitterConfig.sigma_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DarkCountConfig:
    rate_hz: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(self.rate_hz, NonnegativeFloat, "DarkCountConfig.rate_hz")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FixedDelayConfig:
    delay_ns: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.delay_ns,
            NonnegativeFloat,
            "FixedDelayConfig.delay_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExponentialDelayConfig:
    mean_delay_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_delay_ns,
            PositiveFloat,
            "ExponentialDelayConfig.mean_delay_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NormalDelayConfig:
    location_ns: NonnegativeFloat
    sigma_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.location_ns,
            NonnegativeFloat,
            "NormalDelayConfig.location_ns",
        )
        _require_exact(
            self.sigma_ns,
            PositiveFloat,
            "NormalDelayConfig.sigma_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig | NormalDelayConfig

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DirectCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig, NormalDelayConfig),
            "DirectCrosstalkConfig.delay",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DelayedCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig | NormalDelayConfig

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DelayedCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig, NormalDelayConfig),
            "DelayedCrosstalkConfig.delay",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseRecoveryConfig:
    time_constant_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.time_constant_ns,
            PositiveFloat,
            "AfterpulseRecoveryConfig.time_constant_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseConfig:
    probability: Probability
    mean_delay_ns: PositiveFloat
    recovery: AfterpulseRecoveryConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(
            self.probability,
            Probability,
            "AfterpulseConfig.probability",
        )
        _require_exact(
            self.mean_delay_ns,
            PositiveFloat,
            "AfterpulseConfig.mean_delay_ns",
        )
        _require_optional_exact(
            self.recovery,
            AfterpulseRecoveryConfig,
            "AfterpulseConfig.recovery",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CorrelatedAvalancheConfig:
    maximum_generations: NonnegativeInteger
    direct_crosstalk: DirectCrosstalkConfig | None = None
    delayed_crosstalk: DelayedCrosstalkConfig | None = None
    afterpulse: AfterpulseConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(
            self.maximum_generations,
            NonnegativeInteger,
            "CorrelatedAvalancheConfig.maximum_generations",
        )
        _require_optional_exact(
            self.direct_crosstalk,
            DirectCrosstalkConfig,
            "CorrelatedAvalancheConfig.direct_crosstalk",
        )
        _require_optional_exact(
            self.delayed_crosstalk,
            DelayedCrosstalkConfig,
            "CorrelatedAvalancheConfig.delayed_crosstalk",
        )
        _require_optional_exact(
            self.afterpulse,
            AfterpulseConfig,
            "CorrelatedAvalancheConfig.afterpulse",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeSmearingConfig:
    relative_sigma: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.relative_sigma,
            NonnegativeFloat,
            "ChargeSmearingConfig.relative_sigma",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeConfig:
    dark_count: DarkCountConfig | None = None
    timing_jitter: TimingJitterConfig | None = None
    correlated_avalanches: CorrelatedAvalancheConfig | None = None
    smearing: ChargeSmearingConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_exact(
            self.dark_count,
            DarkCountConfig,
            "ChargeConfig.dark_count",
        )
        _require_optional_exact(
            self.timing_jitter,
            TimingJitterConfig,
            "ChargeConfig.timing_jitter",
        )
        _require_optional_exact(
            self.correlated_avalanches,
            CorrelatedAvalancheConfig,
            "ChargeConfig.correlated_avalanches",
        )
        _require_optional_exact(
            self.smearing,
            ChargeSmearingConfig,
            "ChargeConfig.smearing",
        )
