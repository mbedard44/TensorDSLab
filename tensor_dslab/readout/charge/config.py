from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_core import (
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    Probability,
    RngKey,
)

from tensor_dslab.readout._requirements import (
    _require_exact,
    _require_one_of_exact,
    _require_optional_exact,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TimingJitterConfig:
    sigma_ns: NonnegativeFloat
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_0008)

    def __post_init__(self) -> None:
        _require_exact(
            self.sigma_ns,
            NonnegativeFloat,
            "TimingJitterConfig.sigma_ns",
        )
        _require_exact(self.rng_key, RngKey, "TimingJitterConfig.rng_key")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DarkCountConfig:
    rate_hz: NonnegativeFloat
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_0003)

    def __post_init__(self) -> None:
        _require_exact(self.rate_hz, NonnegativeFloat, "DarkCountConfig.rate_hz")
        _require_exact(self.rng_key, RngKey, "DarkCountConfig.rng_key")


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

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DirectCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig),
            "DirectCrosstalkConfig.delay",
        )
        _require_exact(
            self.retained_rng_key,
            RngKey,
            "DirectCrosstalkConfig.retained_rng_key",
        )
        _require_exact(
            self.overflow_rng_key,
            RngKey,
            "DirectCrosstalkConfig.overflow_rng_key",
        )
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

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DelayedCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig),
            "DelayedCrosstalkConfig.delay",
        )
        _require_exact(
            self.retained_rng_key,
            RngKey,
            "DelayedCrosstalkConfig.retained_rng_key",
        )
        _require_exact(
            self.overflow_rng_key,
            RngKey,
            "DelayedCrosstalkConfig.overflow_rng_key",
        )
        if self.retained_rng_key == self.overflow_rng_key:
            raise ValueError(
                "DelayedCrosstalkConfig retained and overflow RNG keys must differ"
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
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_0009)

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
        _require_exact(self.rng_key, RngKey, "AfterpulseConfig.rng_key")


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
    rng_key: RngKey = RngKey(namespace=0x54445331, stream=0x0000_000A)

    def __post_init__(self) -> None:
        _require_exact(
            self.relative_sigma,
            NonnegativeFloat,
            "ChargeSmearingConfig.relative_sigma",
        )
        _require_exact(self.rng_key, RngKey, "ChargeSmearingConfig.rng_key")


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
