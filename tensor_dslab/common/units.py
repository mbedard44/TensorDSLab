from __future__ import annotations

import tokenize
from typing import cast

import pint
from pint import Quantity
from tensor_core import FiniteFloat, Scalar


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
    parsed_unit = _require_unit(unit)
    normalized = _finite_magnitude(magnitude, field="quantity magnitude")
    return cast(Quantity, _REGISTRY.Quantity(normalized, parsed_unit))


def quantities(
    magnitudes: tuple[int | float, ...],
    unit: str,
) -> tuple[Quantity, ...]:
    if type(magnitudes) is not tuple:
        raise TypeError("magnitudes must be exactly tuple")
    parsed_unit = _require_unit(unit)
    return tuple(
        cast(
            Quantity,
            _REGISTRY.Quantity(
                _finite_magnitude(
                    magnitude,
                    field=f"quantity magnitude[{index}]",
                ),
                parsed_unit,
            ),
        )
        for index, magnitude in enumerate(magnitudes)
    )


def _canonical_quantity(
    value: object,
    *,
    unit: str,
    field: str,
    constraint: type[Scalar[float]],
) -> Quantity:
    if not isinstance(value, pint.Quantity):
        raise TypeError(f"{field} must be a Pint Quantity")
    if type(value.magnitude) not in (int, float):
        raise TypeError(f"{field} magnitude must be exactly int or float")
    try:
        converted = value.to(unit)
    except (pint.PintError, OverflowError) as error:
        raise ValueError(f"{field} must be convertible to {unit}") from error
    if type(converted.magnitude) not in (int, float):
        raise TypeError(f"{field} converted magnitude must be exactly int or float")
    try:
        normalized = constraint.require(converted.magnitude, field)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    return cast(Quantity, _REGISTRY.Quantity(normalized, unit))


def canonical_magnitude(value: Quantity) -> float:
    return cast(float, value.magnitude)


def _integer_quantity(magnitude: int, *, unit: str) -> Quantity:
    if type(magnitude) is not int:
        raise TypeError("integer quantity magnitude must be exactly int")
    return cast(Quantity, _REGISTRY.Quantity(magnitude, unit))
