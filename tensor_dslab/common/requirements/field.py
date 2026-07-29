"""Requirements for exact Field Specs and fresh Product storage."""

from typing import Any

from tensor_core import TensorField, TensorKernel


def require_exact_field_spec(
    field: TensorField[Any],
    spec_type: type,
) -> None:
    """Require a Field to carry one exact semantic Spec type."""

    if type(field.spec) is not spec_type:
        raise TypeError(
            f"{type(field).__name__}.spec must be exactly {spec_type.__name__}"
        )


def require_fresh_product(
    product: TensorField[Any],
    *,
    sources: tuple[TensorField[Any], ...],
    kernels: tuple[TensorKernel[Any], ...],
) -> None:
    """Require contiguous storage disjoint from every live input tensor."""

    if not product.tensor.is_contiguous():
        raise ValueError("generated Product tensor must be contiguous")
    product_storage = product.tensor.untyped_storage()
    input_storages = (
        *(source.tensor.untyped_storage() for source in sources),
        *(kernel.tensor.untyped_storage() for kernel in kernels),
    )
    if any(product_storage == storage for storage in input_storages):
        raise ValueError(
            "generated Product storage must be source- and kernel-disjoint"
        )
