"""Requirements for tensor-allocation and RNG-address capacity."""

import torch
from tensor_core.tensor.validation import (
    require_shape_span as _require_shape_span,
    require_tensor_allocation as _require_tensor_allocation,
)


_INT63_LIMIT = 1 << 63


def require_tensor_capacity(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    field: str,
) -> None:
    """Preflight one exact tensor element and byte span."""

    _require_tensor_allocation(
        shape,
        field,
        element_size=dtype.itemsize,
        upper=_INT63_LIMIT,
    )


def require_address_capacity(
    element_shape: tuple[int, ...],
    *,
    address_shape: tuple[int, ...],
    field: str,
) -> None:
    """Preflight one exact RngElements and RngAddress domain."""

    _require_shape_span(
        element_shape,
        f"{field} elements",
        upper=_INT63_LIMIT,
    )
    _require_shape_span(
        (*address_shape, *element_shape),
        field,
        upper=_INT63_LIMIT + 1,
    )
