from __future__ import annotations

import math

import torch
from tensor_core import TensorField

from tensor_dslab.common import ChannelAxis, ExampleAxis, SampleAxis


def require_readout_structure(field: TensorField) -> None:
    axis_types = tuple(type(axis) for axis in field.axes)
    accepted = frozenset({ExampleAxis, ChannelAxis, SampleAxis})
    if len(axis_types) != 3 or frozenset(axis_types) != accepted:
        raise ValueError(
            "readout fields require exactly example, channel, and sample axes"
        )
    if field.tensor.layout is not torch.strided:
        raise ValueError("readout fields require dense strided tensors")


def require_dtype(field: TensorField, expected: torch.dtype) -> None:
    if field.tensor.dtype is not expected:
        raise ValueError(f"field tensor dtype must be exactly {expected}")


def require_floating_dtype(field: TensorField) -> None:
    if field.tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError("field tensor dtype must be torch.float32 or torch.float64")


def require_representable_float(
    value: float | int,
    *,
    dtype: torch.dtype,
    field: str,
) -> float:
    if type(value) not in (float, int):
        raise TypeError(f"{field} must be a Python float or non-boolean int")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be exactly torch.float32 or torch.float64")
    try:
        represented = float(torch.tensor(value, dtype=dtype, device="cpu"))
    except OverflowError as error:
        raise ValueError(f"{field} is not finite in {dtype}") from error
    if not math.isfinite(represented):
        raise ValueError(f"{field} is not finite in {dtype}")
    return represented
