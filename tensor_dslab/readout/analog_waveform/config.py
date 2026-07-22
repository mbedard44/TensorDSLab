from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_core import FiniteFloat

from tensor_dslab.readout.requirements import require_optional_exact


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogSaturationConfig:
    minimum_mv: FiniteFloat | None = None
    maximum_mv: FiniteFloat | None = None

    def __post_init__(self) -> None:
        require_optional_exact(
            self.minimum_mv,
            FiniteFloat,
            "AnalogSaturationConfig.minimum_mv",
        )
        require_optional_exact(
            self.maximum_mv,
            FiniteFloat,
            "AnalogSaturationConfig.maximum_mv",
        )
        if self.minimum_mv is None and self.maximum_mv is None:
            raise ValueError("analog saturation requires at least one bound")
        if (
            self.minimum_mv is not None
            and self.maximum_mv is not None
            and self.minimum_mv.value >= self.maximum_mv.value
        ):
            raise ValueError("analog saturation minimum must be below maximum")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogWaveformConfig:
    saturation: AnalogSaturationConfig | None = None

    def __post_init__(self) -> None:
        require_optional_exact(
            self.saturation,
            AnalogSaturationConfig,
            "AnalogWaveformConfig.saturation",
        )
