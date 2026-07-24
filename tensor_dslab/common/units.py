"""TensorDSLab-owned Pint registry and canonical quantity boundaries."""

import tokenize
from typing import cast

import numpy as np
import pint
from pint import Quantity
from tensor_core import FiniteFloat, Scalar


type _QuantityField = tuple[str, str, type[Scalar[float]]]
_REGISTRY = pint.UnitRegistry(cache_folder=None)
_PARSER_ERRORS = (
    pint.PintError,
    ValueError,
    TypeError,
    ArithmeticError,
    AssertionError,
    tokenize.TokenError,
)


def _require_unit(unit: object) -> pint.Unit:
    if type(unit) is not str:
        raise TypeError("unit must be exactly str")
    if not unit.strip():
        raise ValueError("unit must be nonempty")
    try:
        return _REGISTRY.parse_units(unit)
    except _PARSER_ERRORS as error:
        raise ValueError("unit must be a valid unscaled unit expression") from error


def _finite_magnitude(magnitude: object, *, field: str) -> float:
    if type(magnitude) not in (int, float):
        raise TypeError(f"{field} must be exactly int or float")
    try:
        return FiniteFloat.require(magnitude, field)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error


def quantity(magnitude: int | float, unit: str) -> Quantity:
    """Return one copied scalar quantity in canonical package units."""

    parsed_unit = _require_unit(unit)
    normalized = _finite_magnitude(magnitude, field="quantity magnitude")
    return cast(Quantity, _REGISTRY.Quantity(normalized, parsed_unit))


def quantities(
    magnitudes: tuple[int | float, ...],
    unit: str,
) -> Quantity:
    """Return one copied immutable vector quantity in canonical package units."""

    if type(magnitudes) is not tuple:
        raise TypeError("magnitudes must be exactly tuple")
    parsed_unit = _require_unit(unit)
    normalized = np.array(
        tuple(
            _finite_magnitude(
                magnitude,
                field=f"quantity magnitude[{index}]",
            )
            for index, magnitude in enumerate(magnitudes)
        ),
        dtype=np.float64,
    )
    normalized.setflags(write=False)
    return cast(Quantity, _REGISTRY.Quantity(normalized, parsed_unit))


def _canonical_quantity(
    value: object,
    *,
    unit: str,
    field: str,
    constraint: type[Scalar[float]],
    vector: bool = False,
) -> Quantity:
    if not isinstance(value, pint.Quantity):
        raise TypeError(f"{field} must be a Pint Quantity")
    try:
        converted = value.to(unit)
    except (pint.PintError, OverflowError) as error:
        raise ValueError(f"{field} must be convertible to {unit}") from error

    if not vector:
        if type(converted.magnitude) not in (int, float):
            raise TypeError(f"{field} converted magnitude must be exactly int or float")
        try:
            normalized = constraint.require(converted.magnitude, field)
        except OverflowError as error:
            raise ValueError(f"{field} must be finite") from error
        return cast(Quantity, _REGISTRY.Quantity(normalized, unit))

    magnitude = converted.magnitude
    if type(magnitude) is not np.ndarray:
        raise TypeError(f"{field} converted magnitude must be exactly a NumPy array")
    if magnitude.ndim != 1:
        raise ValueError(f"{field} converted magnitude must be one-dimensional")
    elements: list[float] = []
    for index, item in enumerate(magnitude):
        try:
            primitive = item.item()
        except AttributeError as error:
            raise TypeError(
                f"{field}[{index}] must be exactly int or float"
            ) from error
        if type(primitive) not in (int, float):
            raise TypeError(f"{field}[{index}] must be exactly int or float")
        try:
            elements.append(constraint.require(primitive, f"{field}[{index}]"))
        except OverflowError as error:
            raise ValueError(f"{field}[{index}] must be finite") from error
    canonical = np.array(elements, dtype=np.float64)
    canonical.setflags(write=False)
    return cast(Quantity, _REGISTRY.Quantity(canonical, unit))


def _canonicalize_quantity_fields(
    config: object,
    *,
    scalar_fields: tuple[_QuantityField, ...] = (),
    vector_fields: tuple[_QuantityField, ...] = (),
) -> None:
    owner = type(config).__name__
    for vector, fields in (
        (False, scalar_fields),
        (True, vector_fields),
    ):
        for name, unit, constraint in fields:
            parameter = getattr(config, name)
            if parameter is None:
                continue
            canonical = _canonical_quantity(
                parameter,
                unit=unit,
                field=f"{owner}.{name}",
                constraint=constraint,
                vector=vector,
            )
            object.__setattr__(config, name, canonical)


def canonical_magnitude(value: Quantity) -> float:
    return cast(float, value.magnitude)


def canonical_magnitudes(value: Quantity) -> tuple[float, ...]:
    magnitude = cast(np.ndarray, value.magnitude)
    return tuple(float(item) for item in magnitude)


def _integer_quantity(magnitude: int, *, unit: str) -> Quantity:
    if type(magnitude) is not int:
        raise TypeError("integer quantity magnitude must be exactly int")
    return cast(Quantity, _REGISTRY.Quantity(magnitude, unit))
