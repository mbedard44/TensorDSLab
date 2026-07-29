"""Requirements for semantic-axis coordinate representations."""

import math
from typing import cast

from tensor_core import (
    Coordinates,
    CountCoordinates,
    LabelCoordinates,
    OffsetCoordinates,
    RegularCoordinates,
)


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
    """Admit only exact supported integer coordinate representations."""

    if type(coordinates) not in (
        CountCoordinates,
        RegularCoordinates,
        OffsetCoordinates,
    ):
        raise TypeError(
            "QuantityAxis.coordinates must be exactly CountCoordinates, "
            "RegularCoordinates, or OffsetCoordinates"
        )


def require_coordinate_scale(coordinate_scale: object) -> None:
    """Require one exact finite positive binary64 coordinate scale."""

    if type(coordinate_scale) is not float:
        raise TypeError("coordinate_scale must be exactly float")
    if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be finite and positive")


def require_regular_coordinates(
    coordinates: Coordinates[int],
    *,
    start: int,
    step: int,
) -> None:
    """Require exact regular integer coordinates with fixed start and step."""

    if type(start) is not int:
        raise TypeError("start must be exactly int")
    if type(step) is not int:
        raise TypeError("step must be exactly int")
    if type(coordinates) is not RegularCoordinates:
        raise TypeError("coordinates must be exactly RegularCoordinates")
    regular = cast(RegularCoordinates, coordinates)
    if regular.start != start:
        raise ValueError(f"coordinates must start at {start}")
    if regular.step != step:
        raise ValueError(f"coordinates must have step {step}")
