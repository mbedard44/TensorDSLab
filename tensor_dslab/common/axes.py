"""TensorDSLab semantic axes and sampling-coordinate policy."""

import math
from typing import final, override

import pint
from pint import Quantity
from tensor_core import CountAxis, LabelAxis, RegularAxis

from tensor_dslab.common.units import _integer_quantity


@final
class ExampleAxis(CountAxis):
    """Identify nonempty examples by zero-based local ordinal."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if self.count == 0:
            raise ValueError("ExampleAxis must be nonempty")


@final
class ChannelAxis(LabelAxis):
    """Identify readout channels by unique string labels."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if not self.labels:
            raise ValueError("ChannelAxis must be nonempty")


@final
class SampleAxis(RegularAxis):
    """Represent integer-picosecond sampling coordinates."""

    __slots__ = ()

    @classmethod
    def from_period(cls, *, period: Quantity, count: int) -> SampleAxis:
        if not isinstance(period, pint.Quantity):
            raise TypeError("period must be a Pint Quantity")
        if type(period.magnitude) not in (int, float):
            raise TypeError("period magnitude must be exactly int or float")
        try:
            converted = period.to("ps")
        except (pint.PintError, OverflowError) as error:
            raise ValueError("period must be convertible to picoseconds") from error
        magnitude = converted.magnitude
        if type(magnitude) is int:
            step = magnitude
        elif type(magnitude) is float:
            if not math.isfinite(magnitude) or abs(magnitude) > 2**53:
                raise ValueError("period must be integer-representable in picoseconds")
            step = round(magnitude)
            if abs(magnitude - step) > math.ulp(magnitude):
                raise ValueError("period must be within one ULP of integer picoseconds")
        else:
            raise TypeError("converted period magnitude must be exactly int or float")
        return cls(start=0, step=step, count=count)

    @override
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
    def start_time(self) -> Quantity:
        return _integer_quantity(self.start, unit="ps")

    @property
    def sample_period(self) -> Quantity:
        return _integer_quantity(self.step, unit="ps")

    def time_at(self, index: int) -> Quantity:
        return _integer_quantity(self.coordinate_at(index), unit="ps")

    @property
    def stop_time(self) -> Quantity:
        return _integer_quantity(self.start + self.step * self.count, unit="ps")
