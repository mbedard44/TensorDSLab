"""Requirements for represented tensor dtype and value domains."""

from typing import Protocol

import torch


class _DtypeBearing(Protocol):
    @property
    def dtype(self) -> torch.dtype:
        ...


class _TensorBearing(_DtypeBearing, Protocol):
    @property
    def tensor(self) -> torch.Tensor:
        ...


def require_exact_dtype(
    value: _DtypeBearing,
    dtype: torch.dtype,
) -> None:
    """Require one exact represented dtype."""

    if value.dtype is not dtype:
        raise TypeError(f"{type(value).__name__} dtype must be {dtype}")


def require_dtype_in(
    value: _DtypeBearing,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    """Require a represented dtype from one exact admitted set."""

    if not any(value.dtype is dtype for dtype in dtypes):
        names = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(
            f"{type(value).__name__} dtype must be one of: {names}"
        )


def require_floating_dtype(
    value: _DtypeBearing,
) -> None:
    """Require a Torch floating dtype."""

    if not value.dtype.is_floating_point:
        raise TypeError(f"{type(value).__name__} dtype must be floating")


def require_signed_integer_dtype(
    value: _DtypeBearing,
) -> None:
    """Require an exact supported signed integer dtype."""

    require_dtype_in(
        value,
        dtypes=(torch.int8, torch.int16, torch.int32, torch.int64),
    )


def require_finite(value: _TensorBearing) -> None:
    """Require every represented value to be finite."""

    if not bool(torch.isfinite(value.tensor).all()):
        raise ValueError(f"{type(value).__name__} values must be finite")


def require_nonnegative(value: _TensorBearing) -> None:
    """Require every represented value to be nonnegative."""

    if bool((value.tensor < 0).any()):
        raise ValueError(
            f"{type(value).__name__} values must be nonnegative"
        )


def require_positive(value: _TensorBearing) -> None:
    """Require every represented value to be strictly positive."""

    if bool((value.tensor <= 0).any()):
        raise ValueError(f"{type(value).__name__} values must be positive")


def require_values_between(
    value: _TensorBearing,
    *,
    minimum: int | float,
    maximum: int | float,
) -> None:
    """Require every represented value within one inclusive interval."""

    if bool(((value.tensor < minimum) | (value.tensor > maximum)).any()):
        raise ValueError(
            f"{type(value).__name__} values must be between "
            f"{minimum} and {maximum}"
        )
