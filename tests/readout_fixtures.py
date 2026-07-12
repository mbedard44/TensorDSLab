from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from tensor_core import (
    FiniteFloat,
    Id,
    IdSequence,
    NonnegativeInteger,
    PositiveFloat,
    PositiveInteger,
    TensorAxes,
    TensorAxis,
    TensorAxisId,
    TensorField,
    TensorFieldId,
    TensorLayout,
    build_id_axis,
    build_tensor_layout,
)

from tensor_dslab.common import ChannelId, ExampleId
from tensor_dslab.readout import (
    AdcQuantization,
    DigitizedWaveformSpec,
    READOUT_ANALOG_WAVEFORM_FIELD_ID,
    CHANNEL_AXIS_ID,
    READOUT_CHARGE_FIELD_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    EXAMPLE_AXIS_ID,
    READOUT_FIELD_IDS,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_PHOTOELECTRONS_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
    SAMPLE_AXIS_ID,
    ReadoutCollection,
    SampleGrid,
)

AxisOrder: TypeAlias = tuple[TensorAxisId, ...]

EXTRA_AXIS_ID = TensorAxisId("variant")


@dataclass(frozen=True, slots=True)
class ExtraId(Id):
    pass


@dataclass(frozen=True, slots=True)
class OtherId(Id):
    pass


DEFAULT_AXIS_ORDER: AxisOrder = (
    EXAMPLE_AXIS_ID,
    CHANNEL_AXIS_ID,
    SAMPLE_AXIS_ID,
)

ALTERNATE_AXIS_ORDER: AxisOrder = (
    SAMPLE_AXIS_ID,
    EXAMPLE_AXIS_ID,
    CHANNEL_AXIS_ID,
)


def make_sample_grid() -> SampleGrid:
    return SampleGrid(
        sample_period_ns=PositiveFloat(2.0),
        origin_ns=FiniteFloat(-3.0),
        sample_offset=NonnegativeInteger(7),
    )


def make_digitized_spec(
    *,
    bit_depth: int = 12,
    analog_gain_db: float = 20.0,
) -> DigitizedWaveformSpec:
    return DigitizedWaveformSpec(
        bit_depth=PositiveInteger(bit_depth),
        voltage_pp_mv=PositiveFloat(2000.0),
        voltage_offset_mv=FiniteFloat(0.0),
        analog_gain_db=FiniteFloat(analog_gain_db),
        quantization=AdcQuantization.TRUNCATE,
    )


def make_layout(
    axis_order: AxisOrder = DEFAULT_AXIS_ORDER,
    *,
    example_coordinate_type: type[Id] = ExampleId,
    channel_coordinate_type: type[Id] = ChannelId,
    example_count_only: bool = False,
    channel_count_only: bool = False,
    sample_id_backed: bool = False,
) -> TensorLayout:
    axes: list[TensorAxis] = []
    for axis_id in axis_order:
        if axis_id == EXAMPLE_AXIS_ID:
            if example_count_only:
                axis = TensorAxis(id=axis_id, size=2)
            else:
                axis = build_id_axis(
                    axis_id,
                    IdSequence(
                        (
                            example_coordinate_type("example-0"),
                            example_coordinate_type("example-1"),
                        )
                    ),
                )
        elif axis_id == CHANNEL_AXIS_ID:
            if channel_count_only:
                axis = TensorAxis(id=axis_id, size=2)
            else:
                axis = build_id_axis(
                    axis_id,
                    IdSequence(
                        (
                            channel_coordinate_type("channel-0"),
                            channel_coordinate_type("channel-1"),
                        )
                    ),
                )
        elif axis_id == SAMPLE_AXIS_ID:
            if sample_id_backed:
                axis = build_id_axis(
                    axis_id,
                    IdSequence(tuple(OtherId(f"sample-{index}") for index in range(4))),
                )
            else:
                axis = TensorAxis(id=axis_id, size=4)
        elif axis_id == EXTRA_AXIS_ID:
            axis = build_id_axis(
                axis_id,
                IdSequence((ExtraId("variant-0"), ExtraId("variant-1"))),
            )
        else:
            axis = TensorAxis(id=axis_id, size=3)
        axes.append(axis)
    return build_tensor_layout(TensorAxes(tuple(axes)))


def field_dtype(field_id: TensorFieldId, floating_dtype: torch.dtype) -> torch.dtype:
    if field_id == READOUT_PHOTOELECTRONS_FIELD_ID:
        return torch.int64
    if field_id == READOUT_DIGITIZED_WAVEFORM_FIELD_ID:
        return torch.int32
    return floating_dtype


def make_tensor(
    layout: TensorLayout,
    *,
    dtype: torch.dtype,
    device: torch.device,
    noncontiguous: bool = False,
    expanded: bool = False,
) -> torch.Tensor:
    shape = tuple(axis.size for axis in layout.axes.axes)
    if any(size is None for size in shape):
        raise ValueError("fixture layouts require fixed sizes")
    exact_shape = tuple(int(size) for size in shape)
    if expanded:
        base = torch.zeros((1, *exact_shape[1:]), dtype=dtype, device=device)
        return base.expand(exact_shape)
    if noncontiguous:
        base = torch.zeros((*exact_shape, 2), dtype=dtype, device=device)
        return base[..., 0]
    return torch.zeros(exact_shape, dtype=dtype, device=device)


_DEFAULT_SPEC = object()


def make_collection(
    field_ids: tuple[TensorFieldId, ...] = READOUT_FIELD_IDS.ids,
    *,
    layout: TensorLayout | None = None,
    floating_dtype: torch.dtype = torch.float32,
    dtype_overrides: dict[TensorFieldId, torch.dtype] | None = None,
    tensor_overrides: dict[TensorFieldId, torch.Tensor] | None = None,
    noncontiguous_field_ids: frozenset[TensorFieldId] = frozenset(),
    expanded_field_ids: frozenset[TensorFieldId] = frozenset(),
    device: torch.device = torch.device("cpu"),
    shared_axes: IdSequence | None = None,
    sample_grid: SampleGrid | None = None,
    digitized_waveform_spec: object = _DEFAULT_SPEC,
) -> ReadoutCollection:
    layout = make_layout() if layout is None else layout
    dtype_overrides = {} if dtype_overrides is None else dtype_overrides
    tensor_overrides = {} if tensor_overrides is None else tensor_overrides
    fields: dict[TensorFieldId, TensorField] = {}
    for field_id in field_ids:
        dtype = dtype_overrides.get(field_id, field_dtype(field_id, floating_dtype))
        tensor = tensor_overrides.get(
            field_id,
            make_tensor(
                layout,
                dtype=dtype,
                device=device,
                noncontiguous=field_id in noncontiguous_field_ids,
                expanded=field_id in expanded_field_ids,
            ),
        )
        fields[field_id] = TensorField(
            id=field_id,
            tensor=tensor,
            layout=layout,
            metadata={"field": field_id.value},
        )
    if shared_axes is None:
        shared_axes = IdSequence(tuple(axis.id for axis in layout.axes.axes))
    if digitized_waveform_spec is _DEFAULT_SPEC:
        digitized_waveform_spec = (
            make_digitized_spec()
            if READOUT_DIGITIZED_WAVEFORM_FIELD_ID in field_ids
            else None
        )
    return ReadoutCollection(
        fields=fields,
        shared_axes=shared_axes,
        metadata={"source": "fixture"},
        sample_grid=make_sample_grid() if sample_grid is None else sample_grid,
        digitized_waveform_spec=digitized_waveform_spec,  # type: ignore[arg-type]
    )


def storage_pointer(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()
