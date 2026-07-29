"""TensorDSLab's one Pint registry and scalar quantity boundary."""

from typing import cast

import pint
from tensor_core import FiniteFloat


unit_registry = pint.UnitRegistry(cache_folder=None)
unit_registry.define("avalanche = [avalanche]")
_UNIT_TYPE = type(unit_registry.Unit(""))


def _normalize_unit(unit: object) -> pint.Unit:
    if not isinstance(unit, pint.Unit):
        raise TypeError("unit must be exactly a Pint Unit")
    if getattr(unit, "_REGISTRY", None) is not unit_registry:
        raise ValueError("unit must belong to tensor_dslab.unit_registry")
    if type(unit) is not _UNIT_TYPE:
        raise TypeError("unit must be exactly a Pint Unit")
    return cast(pint.Unit, unit)


def quantity(magnitude: int | float, unit: str) -> pint.Quantity:
    """Return one finite scalar quantity owned by the package registry."""

    if type(unit) is not str:
        raise TypeError("unit must be exactly str")
    if not unit.strip():
        raise ValueError("unit must be nonempty")
    try:
        parsed = unit_registry.parse_units(unit)
    except (pint.PintError, ValueError, TypeError) as error:
        raise ValueError("unit must be a valid unscaled unit expression") from error
    try:
        value = FiniteFloat.require(magnitude, "quantity magnitude")
    except OverflowError as error:
        raise ValueError("quantity magnitude must be finite") from error
    return cast(pint.Quantity, unit_registry.Quantity(value, parsed))
