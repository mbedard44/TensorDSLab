from __future__ import annotations

import torch
from tensor_core import TensorField

from tensor_dslab.common import (
    ChannelAxis,
    ExampleAxis,
    SampleAxis,
    SamplingConfig,
)


def _require_readout_structure(field: TensorField) -> None:
    axis_types = tuple(type(axis) for axis in field.axes)
    accepted = frozenset({ExampleAxis, ChannelAxis, SampleAxis})
    if len(axis_types) != 3 or frozenset(axis_types) != accepted:
        raise ValueError(
            "readout fields require exactly example, channel, and sample axes"
        )
    if field.tensor.layout is not torch.strided:
        raise ValueError("readout fields require dense strided tensors")


def _require_dtype(field: TensorField, expected: torch.dtype) -> None:
    if field.tensor.dtype is not expected:
        raise ValueError(f"field tensor dtype must be exactly {expected}")


def _require_floating_dtype(field: TensorField) -> None:
    if field.tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError("field tensor dtype must be torch.float32 or torch.float64")


def _require_sampling(
    field: TensorField,
    sampling: SamplingConfig,
) -> None:
    _require_exact(
        sampling,
        SamplingConfig,
        "_require_sampling.sampling",
    )
    sample_axis = field.axis(SampleAxis)
    if sample_axis.size != sampling.sample_count.value:
        raise ValueError("sample-axis size must agree with SamplingConfig")
    if sample_axis.start_ps != 0:
        raise ValueError("sample-axis start must be zero")
    if sample_axis.sample_period_ps != sampling.sample_period_ps.value:
        raise ValueError("sample-axis period must agree with SamplingConfig")


def _require_exact(
    value: object,
    expected: type[object],
    field: str,
) -> None:
    if type(value) is not expected:
        raise TypeError(f"{field} must be exactly {expected.__name__}")


def _require_optional_exact(
    value: object | None,
    expected: type[object],
    field: str,
) -> None:
    if value is not None:
        _require_exact(value, expected, field)


def _require_one_of_exact(
    value: object,
    expected: tuple[type[object], ...],
    field: str,
) -> None:
    if type(value) not in expected:
        names = ", ".join(item.__name__ for item in expected)
        raise TypeError(f"{field} must be exactly one of: {names}")
