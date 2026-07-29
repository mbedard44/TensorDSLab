"""Requirements for represented tensor dtype and value domains."""

from typing import Protocol

import torch


class _DtypeBearing(Protocol):
    """Expose represented dtype metadata to tensor requirements."""

    @property
    def dtype(self) -> torch.dtype:
        ...


class _TensorBearing(_DtypeBearing, Protocol):
    """Expose represented tensor storage to value-domain requirements."""

    @property
    def tensor(self) -> torch.Tensor:
        ...


class _SuppressionSpec(_DtypeBearing, Protocol):
    """Expose one encoded sentinel and represented dtype."""

    @property
    def suppression_code(self) -> int:
        ...


class _EncodedTensor(_TensorBearing, Protocol):
    """Expose one encoded tensor and its sentinel-bearing Spec."""

    @property
    def spec(self) -> _SuppressionSpec:
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


def require_negative_representable_suppression_code(
    value: _SuppressionSpec,
) -> None:
    """Require one explicit negative sentinel representable by the Spec dtype."""

    suppression_code = value.suppression_code
    if type(suppression_code) is not int:
        raise TypeError("suppression_code must be exactly int")
    if suppression_code >= 0:
        raise ValueError("suppression_code must be negative")
    limits = torch.iinfo(value.dtype)
    if suppression_code < limits.min or suppression_code > limits.max:
        raise ValueError(
            "suppression_code must be representable by the Spec dtype"
        )


def require_encoded_values(value: _EncodedTensor) -> None:
    """Require every encoded value to be retained data or the exact sentinel."""

    suppression_code = value.spec.suppression_code
    invalid = (value.tensor < 0) & (value.tensor != suppression_code)
    if bool(invalid.any()):
        raise ValueError(
            f"{type(value).__name__} values must be nonnegative or equal "
            "the suppression code"
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
