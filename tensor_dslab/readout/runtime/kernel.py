"""Private physical-kernel alignment actions."""

import torch
from tensor_core import TensorField
from tensor_core.tensor.validation import require_kernel_dimensions

from tensor_dslab.common.kernel import QuantityKernel


def align_quantity_kernel(
    kernel: QuantityKernel,
    *,
    field: TensorField,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Align one physical kernel to a semantic execution field."""

    kernel_name = type(kernel).__name__
    try:
        require_kernel_dimensions(field, kernel)
    except ValueError as error:
        raise ValueError(
            f"{kernel_name} requires a semantic role absent "
            "from the execution field"
        ) from error

    tensor = kernel.tensor
    dimensions: list[int] = []
    for kernel_dimension, axis in enumerate(kernel.conditioning_axes):
        role = type(axis)
        target = field.axis(role)
        target_dimension = field.dimension_of(role)
        if len(axis.coordinates) != len(target.coordinates):
            raise ValueError(
                f"{kernel_name} conditioning coordinates do not exactly "
                "correspond to the execution field axis"
            )
        try:
            indices = tuple(
                axis.index_of(coordinate)
                for coordinate in target.coordinates
            )
        except KeyError as error:
            raise ValueError(
                f"{kernel_name} conditioning coordinates do not exactly "
                "correspond to the execution field axis"
            ) from error
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"{kernel_name} conditioning coordinates do not exactly "
                "correspond to the execution field axis"
            )
        tensor = tensor.index_select(
            kernel_dimension,
            torch.tensor(
                indices,
                dtype=torch.int64,
                device=tensor.device,
            ),
        )
        dimensions.append(target_dimension)

    order = tuple(
        sorted(range(len(dimensions)), key=lambda index: dimensions[index])
    )
    if order != tuple(range(len(order))):
        tensor = tensor.permute(
            *order,
            *range(len(order), tensor.ndim),
        )
    return (
        tensor.to(
            device=field.tensor.device,
            dtype=dtype,
        ).contiguous(),
        tuple(dimensions[index] for index in order),
    )
