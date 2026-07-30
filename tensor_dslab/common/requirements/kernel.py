"""Requirements for exact Kernel Specs and operation geometry."""

import math
from typing import Any, cast

import torch
from tensor_core import OffsetAxis, TensorKernel, TensorKernelSpec


def require_exact_kernel_spec(
    kernel: TensorKernel[Any],
    spec_type: type,
) -> None:
    """Require a Kernel to carry one exact semantic Spec type."""

    if type(kernel.spec) is not spec_type:
        raise TypeError(
            f"{type(kernel).__name__}.spec must be exactly {spec_type.__name__}"
        )


def require_no_operation_axes(
    spec: TensorKernelSpec[Any, Any],
) -> None:
    """Require a scalar or conditioning-only kernel geometry."""

    if spec.operation_axes:
        raise TypeError(
            f"{type(spec).__name__} must not have operation axes"
        )


def require_no_conditioning_axis_type(
    spec: TensorKernelSpec[Any, Any],
    axis_type: type,
) -> None:
    """Reject one exact semantic role from Kernel conditioning geometry."""

    if any(type(axis) is axis_type for axis in spec.conditioning_axes):
        raise TypeError(
            f"{type(spec).__name__} must not condition on "
            f"{axis_type.__name__}"
        )


def require_operation_axis_count(
    spec: TensorKernelSpec[Any, Any],
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    """Require an operation-axis count within one inclusive interval."""

    count = len(spec.operation_axes)
    if count < minimum or (maximum is not None and count > maximum):
        if maximum is None:
            raise ValueError(
                f"{type(spec).__name__} must have at least {minimum} operation axes"
            )
        raise ValueError(
            f"{type(spec).__name__} must have between {minimum} and "
            f"{maximum} operation axes"
        )


def require_operation_axes_type(
    spec: TensorKernelSpec[Any, Any],
    axis_type: type,
) -> None:
    """Require every operation axis to have one exact representation type."""

    if any(type(axis) is not axis_type for axis in spec.operation_axes):
        raise TypeError(
            f"{type(spec).__name__} operation axes must be exactly "
            f"{axis_type.__name__}"
        )


def require_nonempty_operation_extents(
    spec: TensorKernelSpec[Any, Any],
) -> None:
    """Require every operation axis to contain at least one coordinate."""

    if any(axis.size == 0 for axis in spec.operation_axes):
        raise ValueError(
            f"{type(spec).__name__} operation axes must be nonempty"
        )


def require_operation_target_count(
    spec: TensorKernelSpec[Any, Any],
    *,
    relative_to: type,
    count: int,
) -> None:
    """Require an exact count of operation axes targeting one semantic role."""

    found = sum(
        cast(OffsetAxis, axis).relative_to is relative_to
        for axis in spec.operation_axes
        if type(axis) is OffsetAxis
    )
    if found != count:
        raise ValueError(
            f"{type(spec).__name__} must target {relative_to.__name__} "
            f"exactly {count} times"
        )


def require_offset_bounds(
    spec: TensorKernelSpec[Any, Any],
    *,
    relative_to: type,
    minimum: int,
    inclusive: bool,
) -> None:
    """Require lower bounds for offsets targeting one exact semantic role."""

    offsets = (
        offset
        for axis in spec.operation_axes
        if type(axis) is OffsetAxis
        and cast(OffsetAxis, axis).relative_to is relative_to
        for offset in cast(OffsetAxis, axis).coordinates.offsets
    )
    invalid = (
        any(offset < minimum for offset in offsets)
        if inclusive
        else any(offset <= minimum for offset in offsets)
    )
    if invalid:
        relation = "at least" if inclusive else "greater than"
        raise ValueError(
            f"{type(spec).__name__} offsets targeting {relative_to.__name__} must be "
            f"{relation} {minimum}"
        )


def require_operation_row_total(
    kernel: TensorKernel[Any],
    *,
    exact: float | None = None,
    maximum: float | None = None,
    tolerance: float,
) -> None:
    """Require each operation row to have an exact or maximum binary64 total."""

    if (exact is None) == (maximum is None):
        raise ValueError("exactly one of exact or maximum must be supplied")
    operation_count = kernel.spec.operation_element_count
    if operation_count == 0:
        return
    cpu = kernel.tensor.detach().to(device="cpu", dtype=torch.float64)
    rows = cpu.reshape(-1, operation_count).tolist()
    totals = tuple(math.fsum(float(value) for value in row) for row in rows)
    if exact is not None:
        if any(abs(total - exact) > tolerance for total in totals):
            raise ValueError(
                f"{type(kernel).__name__} operation rows must total {exact}"
            )
    elif maximum is not None and any(
        total > maximum + tolerance for total in totals
    ):
        raise ValueError(
            f"{type(kernel).__name__} operation rows must not exceed {maximum}"
        )
