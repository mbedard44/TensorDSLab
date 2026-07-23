from __future__ import annotations

from typing import final

from tensor_core import CountAxis, LabelAxis, RegularAxis


@final
class ExampleAxis(CountAxis):
    __slots__ = ()

    def _require(self) -> None:
        if self.count == 0:
            raise ValueError("ExampleAxis must be nonempty")


@final
class ChannelAxis(LabelAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.labels:
            raise ValueError("ChannelAxis must be nonempty")


@final
class SampleAxis(RegularAxis):
    __slots__ = ()

    def _require(self) -> None:
        if self.start < 0:
            raise ValueError("SampleAxis start must be nonnegative")
        if self.step < 0:
            raise ValueError("SampleAxis step must be positive")
        if self.count < 2:
            raise ValueError("SampleAxis requires at least two samples")
        if self.start + self.step * self.count > (1 << 63) - 1:
            raise ValueError("SampleAxis exclusive stop exceeds int64")

    @property
    def start_ps(self) -> int:
        return self.start

    @property
    def sample_period_ps(self) -> int:
        return self.step

    @property
    def stop_ps(self) -> int:
        return self.start + self.step * self.count
