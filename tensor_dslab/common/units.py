"""TensorDSLab-owned Pint registry and canonical quantity boundaries."""

import tokenize
from typing import cast

import numpy as np
import pint
from pint import Quantity
import torch
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
    magnitudes: tuple[int | float, ...] | torch.Tensor,
    unit: str,
) -> Quantity:
    """Return one copied immutable tensor quantity in canonical package units."""

    parsed_unit = _require_unit(unit)
    if type(magnitudes) is tuple:
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
    elif type(magnitudes) is torch.Tensor:
        if magnitudes.device.type != "cpu":
            raise ValueError("magnitudes tensor must be on CPU")
        if magnitudes.layout is not torch.strided:
            raise ValueError("magnitudes tensor must use torch.strided layout")
        if magnitudes.requires_grad:
            raise ValueError("magnitudes tensor must not require gradients")
        if not magnitudes.dtype.is_floating_point and magnitudes.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("magnitudes tensor must have a real numeric dtype")
        converted = magnitudes.to(dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(converted).all()):
            raise ValueError("magnitudes tensor values must be finite")
        normalized = converted.numpy().copy()
    else:
        raise TypeError("magnitudes must be exactly tuple or torch.Tensor")
    normalized.setflags(write=False)
    return cast(Quantity, _REGISTRY.Quantity(normalized, parsed_unit))


def _canonical_tensor_quantity(
    value: object,
    *,
    unit: str,
    field: str,
) -> tuple[torch.Tensor, pint.Unit]:
    if not isinstance(value, pint.Quantity):
        raise TypeError(f"{field} must be a Pint Quantity")
    try:
        converted = value.to(unit)
    except (pint.PintError, OverflowError) as error:
        raise ValueError(f"{field} must be convertible to {unit}") from error
    magnitude = converted.magnitude
    if type(magnitude) in (int, float):
        normalized = torch.tensor(
            _finite_magnitude(magnitude, field=field),
            dtype=torch.float64,
        )
    elif type(magnitude) is np.ndarray:
        if magnitude.dtype.kind not in "iuf":
            raise TypeError(f"{field} magnitude must have a real numeric dtype")
        try:
            normalized = torch.from_numpy(
                np.array(magnitude, dtype=np.float64, copy=True)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{field} magnitude cannot be represented") from error
    elif type(magnitude) is torch.Tensor:
        if magnitude.device.type != "cpu":
            raise ValueError(f"{field} magnitude tensor must be on CPU")
        if magnitude.layout is not torch.strided:
            raise ValueError(f"{field} magnitude tensor must use torch.strided layout")
        if magnitude.requires_grad:
            raise ValueError(f"{field} magnitude tensor must not require gradients")
        normalized = magnitude.to(dtype=torch.float64).contiguous().clone()
    else:
        raise TypeError(
            f"{field} magnitude must be a scalar, NumPy array, or CPU torch.Tensor"
        )
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError(f"{field} magnitude values must be finite")
    return normalized.contiguous(), _require_unit(unit)


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
