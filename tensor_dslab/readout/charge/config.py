"""Public configuration record for charge."""

from dataclasses import dataclass
from typing import final

from tensor_core import NonnegativeInteger

from tensor_dslab.readout.charge.kernel import (
    Afterpulse,
    DarkCountRate,
    DelayedCrosstalk,
    DirectCrosstalk,
    SmearingWidth,
    TimingJitter,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeConfig:
    """Compose physical charge kernels and fixed branching depth."""

    correlated_avalanche_generations: NonnegativeInteger
    timing_jitter: TimingJitter | None = None
    direct_crosstalk: DirectCrosstalk | None = None
    delayed_crosstalk: DelayedCrosstalk | None = None
    afterpulse: Afterpulse | None = None
    dark_counts: DarkCountRate | None = None
    smearing_width: SmearingWidth | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        enabled = any(
            kernel is not None
            for kernel in (
                self.direct_crosstalk,
                self.delayed_crosstalk,
                self.afterpulse,
            )
        )
        if enabled != (self.correlated_avalanche_generations.value > 0):
            raise ValueError(
                "branching kernels are present exactly when "
                "correlated_avalanche_generations is positive"
            )
