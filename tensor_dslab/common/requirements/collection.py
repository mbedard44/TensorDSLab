"""Requirements for exact semantic collection membership."""

from typing import Any

from tensor_core import TensorCollection


def require_admitted_member_types(
    collection: TensorCollection[Any],
    *,
    admitted: tuple[type, ...],
) -> None:
    """Require every collection member to have an admitted exact type."""

    for member in collection.members.values():
        if type(member) not in admitted:
            names = ", ".join(member_type.__name__ for member_type in admitted)
            raise TypeError(f"members must be exactly one of: {names}")


def require_exact_member_types(
    collection: TensorCollection[Any],
    *,
    required: tuple[type, ...],
) -> None:
    """Require one exact set of collection member types."""

    if collection.member_types != frozenset(required):
        names = ", ".join(member_type.__name__ for member_type in required)
        raise ValueError(f"collection must contain exactly: {names}")


def require_member_count(
    collection: TensorCollection[Any],
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> None:
    """Require a collection member count within one inclusive interval."""

    if type(minimum) is not int:
        raise TypeError("minimum must be exactly int")
    if maximum is not None and type(maximum) is not int:
        raise TypeError("maximum must be exactly int or None")
    count = len(collection.members)
    if count < minimum or (maximum is not None and count > maximum):
        if maximum is None:
            raise ValueError(
                f"collection must contain at least {minimum} members"
            )
        raise ValueError(
            f"collection must contain between {minimum} and {maximum} members"
        )
