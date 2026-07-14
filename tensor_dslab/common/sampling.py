from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_core import PositiveInteger

from tensor_dslab.common.axes import SampleAxis


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingConfig:
    sample_period_ps: PositiveInteger
    sample_count: PositiveInteger

    def __post_init__(self) -> None:
        if type(self.sample_period_ps) is not PositiveInteger:
            raise TypeError(
                "SamplingConfig.sample_period_ps must be exactly PositiveInteger"
            )
        if type(self.sample_count) is not PositiveInteger:
            raise TypeError(
                "SamplingConfig.sample_count must be exactly PositiveInteger"
            )
        if self.sample_count.value < 2:
            raise ValueError("SamplingConfig.sample_count must be at least 2")
        if self.window_stop_ps > (1 << 63) - 1:
            raise ValueError("SamplingConfig.window_stop_ps exceeds int64")

    @property
    def window_stop_ps(self) -> int:
        return self.sample_period_ps.value * self.sample_count.value

    def build_axis(self) -> SampleAxis:
        period_ps = self.sample_period_ps.value
        return SampleAxis(
            coordinates=tuple(
                f"{index * period_ps}ps"
                for index in range(self.sample_count.value)
            )
        )
