"""TensorDSLab semantic axes and physical coordinate representations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, final, override

import pint
from tensor_core import (
    Coordinates,
    TensorAxis,
)

from tensor_dslab.common.requirements.axis import (
    require_coordinate_scale,
    require_supported_coordinates,
    require_supported_integer_coordinates,
)
from tensor_dslab.common.requirements.unit import require_unit_compatible
from tensor_dslab.common.units import normalize_unit, unit_registry


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
        require_coordinate_scale(self.coordinate_scale)
        object.__setattr__(self, "unit", normalize_unit(self.unit))
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
        require_unit_compatible(self.unit, target="s", field="TimeAxis.unit")


@final
class FrequencyAxis[
    CoordinatesT: Coordinates[int],
](QuantityAxis[CoordinatesT]):
    """Represent one physical frequency coordinate role."""

    __slots__ = ()

    @override
    def _require_quantity_axis(self) -> None:
        require_unit_compatible(
            self.unit,
            target="Hz",
            field="FrequencyAxis.unit",
        )
