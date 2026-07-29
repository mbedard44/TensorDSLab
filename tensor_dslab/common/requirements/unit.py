"""Requirements for already-normalized package-owned units."""

import pint

from tensor_dslab.common.units import _UNIT_TYPE, unit_registry


def require_unit_compatible(
    unit: pint.Unit,
    *,
    target: str,
    field: str,
) -> None:
    """Require an exact package-owned Unit compatible with one target."""

    if not isinstance(unit, pint.Unit):
        raise TypeError(f"{field} must be exactly a Pint Unit")
    if getattr(unit, "_REGISTRY", None) is not unit_registry:
        raise ValueError(f"{field} must belong to tensor_dslab.unit_registry")
    if type(unit) is not _UNIT_TYPE:
        raise TypeError(f"{field} must be exactly a Pint Unit")
    try:
        unit_registry.Quantity(1.0, unit).to(target)
    except pint.PintError as error:
        raise ValueError(f"{field} must be compatible with {target}") from error
