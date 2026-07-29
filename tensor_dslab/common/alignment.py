"""Private semantic source and kernel alignment mechanics."""

from dataclasses import replace
from typing import Any, cast

import pint
import torch
from tensor_core import OffsetAxis, TensorField, TensorKernel
from tensor_core.tensor.validation import (
    require_shape_span,
    require_tensor_allocation,
)

from tensor_dslab.common.field import QuantityFieldSpec
from tensor_dslab.common.units import unit_registry


_INT63_LIMIT = 1 << 63


def require_allocation(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    field: str,
) -> None:
    """Preflight one exact tensor element and byte span."""

    require_tensor_allocation(
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

    require_shape_span(
        element_shape,
        f"{field} elements",
        upper=_INT63_LIMIT,
    )
    require_shape_span(
        (*address_shape, *element_shape),
        field,
        upper=_INT63_LIMIT + 1,
    )


def align_source(
    source: TensorField[Any],
    *,
    target_axes: tuple,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Permute and scale one admitted source into target axis order."""

    dimensions = tuple(source.dimension_of(type(axis)) for axis in target_axes)
    tensor = source.tensor.permute(dimensions)
    return (tensor.to(dtype=dtype) * scale).contiguous()


def prepare_sources(
    source_specs: tuple[QuantityFieldSpec[Any], ...],
    *,
    target_spec: QuantityFieldSpec[Any],
    minimum_count: int,
    maximum_count: int | None = None,
    unit_target=None,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[float, ...],
    torch.dtype,
]:
    """Validate and compile ordered source-to-output alignment."""

    if type(source_specs) is not tuple:
        raise TypeError("source_specs must be exactly a tuple")
    if len(source_specs) < minimum_count or (
        maximum_count is not None and len(source_specs) > maximum_count
    ):
        raise ValueError("source_specs has the wrong product-specific count")
    target_roles = tuple(type(axis) for axis in target_spec.axes)
    dimensions: list[tuple[int, ...]] = []
    scales: list[float] = []
    working_dtype = target_spec.dtype
    for index, source in enumerate(source_specs):
        if not isinstance(source, QuantityFieldSpec):
            raise TypeError(f"source_specs[{index}] must be a QuantityFieldSpec")
        source_roles = tuple(type(axis) for axis in source.axes)
        if set(source_roles) != set(target_roles):
            raise ValueError(f"source_specs[{index}] semantic roles do not match output")
        permutation = tuple(source.dimension_of(role) for role in target_roles)
        for target_axis, source_dimension in zip(target_spec.axes, permutation):
            if source.axes[source_dimension] != target_axis:
                raise ValueError(
                    f"source_specs[{index}] coordinates do not match output"
                )
        if source.device != target_spec.device:
            raise ValueError(f"source_specs[{index}] device does not match output")
        try:
            scale = unit_registry.Quantity(1.0, source.unit).to(
                target_spec.unit if unit_target is None else unit_target
            ).magnitude
        except Exception as error:
            raise ValueError(f"source_specs[{index}] unit is incompatible") from error
        dimensions.append(permutation)
        scales.append(float(scale))
        working_dtype = torch.promote_types(working_dtype, source.dtype)
    return tuple(dimensions), tuple(scales), working_dtype


def kernel_dimensions(
    kernel: TensorKernel[Any],
    *,
    target_axes: tuple,
    include_operations: bool = True,
) -> tuple[int, ...]:
    """Compile exact conditioning and operation-target dimensions."""

    result: list[int] = []
    for axis in kernel.conditioning_axes:
        dimension = next(
            (index for index, target in enumerate(target_axes) if type(target) is type(axis)),
            None,
        )
        if dimension is None or target_axes[dimension] != axis:
            raise ValueError(
                f"{type(kernel).__name__} conditioning axis does not match output"
            )
        result.append(dimension)
    if include_operations:
        for axis in kernel.operation_axes:
            dimension = next(
                (
                    index
                    for index, target in enumerate(target_axes)
                    if type(target) is cast(OffsetAxis, axis).relative_to
                ),
                None,
            )
            if dimension is None:
                raise ValueError(
                    f"{type(kernel).__name__} targets a role absent from output"
                )
            result.append(dimension)
    return tuple(result)


def require_prepared_sources(
    sources: tuple[TensorField[Any], ...],
    *,
    source_specs: tuple[QuantityFieldSpec[Any], ...],
) -> None:
    """Bind staged execution to the exact prepared ordered source Specs."""

    if type(sources) is not tuple or len(sources) != len(source_specs):
        raise ValueError("sources do not match the prepared source count")
    for index, (source, prepared) in enumerate(zip(sources, source_specs)):
        if not isinstance(source, TensorField):
            raise TypeError(f"sources[{index}] must be a TensorField")
        if not isinstance(source.spec, QuantityFieldSpec):
            raise TypeError(f"sources[{index}].spec must be a QuantityFieldSpec")
        if source.spec != prepared:
            raise ValueError(f"sources[{index}].spec differs from prepared provenance")


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


def prepare_kernel(
    kernel: TensorKernel[Any],
    *,
    target_axes: tuple,
    target_device: torch.device,
    target_unit: pint.Unit | None = None,
    include_operations: bool = True,
) -> tuple[TensorKernel[Any], tuple[int, ...]]:
    """Return one same-type kernel aligned to exact output conditioning axes."""

    tensor = kernel.tensor
    dimensions: list[int] = []
    conditioning_axes: list[Any] = []
    for dimension, axis in enumerate(kernel.conditioning_axes):
        target_dimension = next(
            (
                index
                for index, target in enumerate(target_axes)
                if type(target) is type(axis)
            ),
            None,
        )
        if target_dimension is None:
            raise ValueError(
                f"{type(kernel).__name__} conditions on a role absent from output"
            )
        target = target_axes[target_dimension]
        try:
            indices = tuple(
                axis.index_of(target.coordinate_at(i)) for i in range(target.size)
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"{type(kernel).__name__} conditioning coordinates do not cover output"
            ) from error
        tensor = tensor.index_select(
            dimension,
            torch.tensor(indices, dtype=torch.int64, device=tensor.device),
        )
        dimensions.append(target_dimension)
        conditioning_axes.append(target)
    order = tuple(sorted(range(len(dimensions)), key=dimensions.__getitem__))
    if order != tuple(range(len(order))):
        tensor = tensor.permute(*order, *range(len(order), tensor.ndim))
    conditioning_axes = [conditioning_axes[index] for index in order]
    scale = 1.0
    spec_updates: dict[str, Any] = {
        "conditioning_axes": tuple(conditioning_axes),
        "device": target_device,
    }
    if target_unit is not None:
        if not hasattr(kernel.spec, "unit"):
            raise TypeError(f"{type(kernel).__name__} has no quantity unit")
        try:
            scale = float(
                unit_registry.Quantity(1.0, kernel.spec.unit)
                .to(target_unit)
                .magnitude
            )
        except Exception as error:
            raise ValueError(
                f"{type(kernel).__name__} unit is incompatible"
            ) from error
        spec_updates["unit"] = target_unit
    prepared_spec = replace(kernel.spec, **spec_updates)
    require_allocation(
        tuple(prepared_spec.shape),
        dtype=prepared_spec.dtype,
        field=f"{type(kernel).__name__} prepared kernel",
    )
    prepared_tensor = tensor.to(
        device=target_device,
        dtype=kernel.dtype,
    )
    if target_unit is not None and scale != 1.0:
        prepared_tensor = prepared_tensor * scale
    prepared_tensor = prepared_tensor.contiguous()
    prepared = type(kernel)(tensor=prepared_tensor, spec=prepared_spec)
    return (
        prepared,
        kernel_dimensions(
            prepared,
            target_axes=target_axes,
            include_operations=include_operations,
        ),
    )
