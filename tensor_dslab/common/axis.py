"""TensorDSLab semantic axes and physical coordinate representations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import cast, final, override

import pint
from tensor_core import (
    Coordinates,
    CountCoordinates,
    LabelCoordinates,
    OffsetCoordinates,
    RegularCoordinates,
    TensorAxis,
)

from tensor_dslab.common.units import _normalize_unit, unit_registry


def require_supported_coordinates[CoordinateT: (int, str)](
    coordinates: Coordinates[CoordinateT],
) -> None:
    """Admit only the exact supported coordinate representations."""

    if type(coordinates) not in (
        CountCoordinates,
        LabelCoordinates,
        RegularCoordinates,
        OffsetCoordinates,
    ):
        raise TypeError(
            "coordinates must be exactly CountCoordinates, LabelCoordinates, "
            "RegularCoordinates, or OffsetCoordinates"
        )


def require_supported_integer_coordinates(
    coordinates: Coordinates[int],
) -> None:
    """Admit only the exact supported integer coordinate representations."""

    if type(coordinates) not in (
        CountCoordinates,
        RegularCoordinates,
        OffsetCoordinates,
    ):
        raise TypeError(
            "QuantityAxis.coordinates must be exactly CountCoordinates, "
            "RegularCoordinates, or OffsetCoordinates"
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class QuantityAxis[
    CoordinatesT: Coordinates[int],
](TensorAxis[int], ABC):
    """Compose exact integer coordinates with one physical scale and unit."""

    coordinates: CoordinatesT
    coordinate_scale: float = 1.0
    unit: pint.Unit

    @final
    @override
    def _require(self) -> None:
        require_supported_integer_coordinates(self.coordinates)
        if type(self.coordinate_scale) is not float:
            raise TypeError("coordinate_scale must be exactly float")
        if not math.isfinite(self.coordinate_scale) or self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be finite and positive")
        object.__setattr__(self, "unit", _normalize_unit(self.unit))
        self._require_quantity_axis()

    @abstractmethod
    def _require_quantity_axis(self) -> None:
        """Enforce the concrete physical semantic-axis contract."""

    def quantity_at(self, index: int) -> pint.Quantity:
        """Return the physical quantity at one strict axis index."""

        return cast(
            pint.Quantity,
            unit_registry.Quantity(
                self.coordinate_at(index) * self.coordinate_scale,
                self.unit,
            ),
        )

    def quantity_of(self, magnitude: int) -> pint.Quantity:
        """Return the physical quantity for one integer coordinate magnitude."""

        if type(magnitude) is not int:
            raise TypeError("magnitude must be exactly int")
        return cast(
            pint.Quantity,
            unit_registry.Quantity(
                magnitude * self.coordinate_scale,
                self.unit,
            ),
        )


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class ExampleAxis[
    CoordinateT: (int, str),
](TensorAxis[CoordinateT]):
    """Identify examples independently of coordinate representation."""

    coordinates: Coordinates[CoordinateT]

    @override
    def _require(self) -> None:
        require_supported_coordinates(self.coordinates)


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class ChannelAxis[
    CoordinateT: (int, str),
](TensorAxis[CoordinateT]):
    """Identify detector channels independently of representation."""

    coordinates: Coordinates[CoordinateT]

    @override
    def _require(self) -> None:
        require_supported_coordinates(self.coordinates)


@final
class TimeAxis[
    CoordinatesT: Coordinates[int],
](QuantityAxis[CoordinatesT]):
    """Represent one physical time coordinate role."""

    __slots__ = ()

    @override
    def _require_quantity_axis(self) -> None:
        try:
            unit_registry.Quantity(1.0, self.unit).to("s")
        except pint.PintError as error:
            raise ValueError("TimeAxis unit must be time-compatible") from error


@final
class FrequencyAxis[
    CoordinatesT: Coordinates[int],
](QuantityAxis[CoordinatesT]):
    """Represent one physical frequency coordinate role."""

    __slots__ = ()

    @override
    def _require_quantity_axis(self) -> None:
        try:
            unit_registry.Quantity(1.0, self.unit).to("Hz")
        except pint.PintError as error:
            raise ValueError(
                "FrequencyAxis unit must be frequency-compatible"
            ) from error
