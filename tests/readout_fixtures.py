from __future__ import annotations

from typing import TypeVar, final

import torch
from tensor_core import TensorAxis, TensorField

from tensor_dslab.common import ChannelAxis, ExampleAxis, SampleAxis
from tensor_dslab.readout import (
    AnalogWaveform,
    Charge,
    DigitizedWaveform,
    NoiseWaveform,
    Photoelectrons,
    PureWaveform,
    ReadoutCollection,
)


FieldT = TypeVar("FieldT", bound=TensorField)


@final
class OtherAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
            raise ValueError("OtherAxis must be nonempty")


@final
class ForeignField(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        return


DEFAULT_AXIS_ORDER: tuple[type[TensorAxis], ...] = (
    ExampleAxis,
    ChannelAxis,
    SampleAxis,
)
ALTERNATE_AXIS_ORDER: tuple[type[TensorAxis], ...] = (
    SampleAxis,
    ExampleAxis,
    ChannelAxis,
)
PRODUCT_TYPES: tuple[type[TensorField], ...] = (
    Photoelectrons,
    Charge,
    PureWaveform,
    NoiseWaveform,
    AnalogWaveform,
    DigitizedWaveform,
)
FLOATING_PRODUCT_TYPES: tuple[type[TensorField], ...] = (
    Charge,
    PureWaveform,
    NoiseWaveform,
    AnalogWaveform,
)


def make_axes(
    order: tuple[type[TensorAxis], ...] = DEFAULT_AXIS_ORDER,
    *,
    example_coordinates: tuple[str, ...] = ("example-0", "example-1"),
    channel_coordinates: tuple[str, ...] = ("channel-0", "channel-1"),
    sample_coordinates: tuple[str, ...] = ("0ps", "2000ps", "4000ps", "6000ps"),
) -> tuple[TensorAxis, ...]:
    axes: list[TensorAxis] = []
    for axis_type in order:
        if axis_type is ExampleAxis:
            axes.append(ExampleAxis(coordinates=example_coordinates))
        elif axis_type is ChannelAxis:
            axes.append(ChannelAxis(coordinates=channel_coordinates))
        elif axis_type is SampleAxis:
            axes.append(SampleAxis(coordinates=sample_coordinates))
        elif axis_type is OtherAxis:
            axes.append(OtherAxis(coordinates=("other-0", "other-1")))
        else:
            raise ValueError(f"unsupported fixture axis type: {axis_type}")
    return tuple(axes)


def product_dtype(
    field_type: type[TensorField],
    *,
    floating_dtype: torch.dtype = torch.float32,
) -> torch.dtype:
    if field_type is Photoelectrons:
        return torch.int64
    if field_type is DigitizedWaveform:
        return torch.int32
    return floating_dtype


def make_tensor(
    axes: tuple[TensorAxis, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
) -> torch.Tensor:
    shape = tuple(axis.size for axis in axes)
    if noncontiguous:
        return torch.zeros((*shape, 2), dtype=dtype, device=device)[..., 0]
    return torch.zeros(shape, dtype=dtype, device=device)


def make_product(
    field_type: type[FieldT],
    *,
    axes: tuple[TensorAxis, ...] | None = None,
    dtype: torch.dtype | None = None,
    floating_dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    tensor: torch.Tensor | None = None,
    noncontiguous: bool = False,
) -> FieldT:
    exact_axes = make_axes() if axes is None else axes
    exact_dtype = (
        product_dtype(field_type, floating_dtype=floating_dtype)
        if dtype is None
        else dtype
    )
    exact_tensor = (
        make_tensor(
            exact_axes,
            dtype=exact_dtype,
            device=device,
            noncontiguous=noncontiguous,
        )
        if tensor is None
        else tensor
    )
    return field_type(tensor=exact_tensor, axes=exact_axes)


def make_collection(
    field_types: tuple[type[TensorField], ...] = PRODUCT_TYPES,
    *,
    axes: tuple[TensorAxis, ...] | None = None,
    floating_dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> ReadoutCollection:
    exact_axes = make_axes() if axes is None else axes
    return ReadoutCollection(
        fields=tuple(
            make_product(
                field_type,
                axes=exact_axes,
                floating_dtype=floating_dtype,
                device=device,
            )
            for field_type in field_types
        )
    )
