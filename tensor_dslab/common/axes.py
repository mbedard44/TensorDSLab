from __future__ import annotations

from typing import final

from tensor_core import TensorAxis


@final
class ExampleAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
            raise ValueError("ExampleAxis must be nonempty")


@final
class ChannelAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
            raise ValueError("ChannelAxis must be nonempty")


@final
class SampleAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if len(self.coordinates) < 2:
            raise ValueError("SampleAxis requires at least two timestamps")

        times_ps: list[int] = []
        for coordinate in self.coordinates:
            if not coordinate.endswith("ps"):
                raise ValueError("SampleAxis timestamps must end in 'ps'")
            magnitude = coordinate[:-2]
            if not (
                magnitude == "0"
                or (
                    magnitude
                    and magnitude[0] != "0"
                    and magnitude.isascii()
                    and magnitude.isdigit()
                )
            ):
                raise ValueError("noncanonical SampleAxis timestamp")
            time_ps = int(magnitude)
            if time_ps > (1 << 63) - 1:
                raise ValueError("SampleAxis timestamp exceeds int64")
            times_ps.append(time_ps)

        period_ps = times_ps[1] - times_ps[0]
        if period_ps <= 0:
            raise ValueError("SampleAxis timestamps must increase")
        if any(
            right - left != period_ps
            for left, right in zip(times_ps, times_ps[1:])
        ):
            raise ValueError("SampleAxis timestamps must be uniformly spaced")
        if times_ps[-1] + period_ps > (1 << 63) - 1:
            raise ValueError("SampleAxis exclusive stop exceeds int64")

    @property
    def start_ps(self) -> int:
        return int(self.coordinates[0][:-2])

    @property
    def sample_period_ps(self) -> int:
        return int(self.coordinates[1][:-2]) - self.start_ps

    @property
    def stop_ps(self) -> int:
        return int(self.coordinates[-1][:-2]) + self.sample_period_ps
