from __future__ import annotations

import torch
from tensor_core import IdSequence

from tensor_dslab.common import ChannelId, ExampleId
from tensor_dslab.readout.ids import (
    READOUT_ANALOG_WAVEFORM_FIELD_ID,
    READOUT_CHANNEL_AXIS_ID,
    READOUT_CHARGE_FIELD_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_EXAMPLE_AXIS_ID,
    READOUT_FIELD_IDS,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_PHOTOELECTRONS_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
    READOUT_SAMPLE_AXIS_ID,
)
from tensor_dslab.readout.types import DigitizedWaveformSpec, SampleGrid
from tensor_dslab.readout.tensors import ReadoutCollection


_FLOATING_FIELD_IDS = (
    READOUT_CHARGE_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_ANALOG_WAVEFORM_FIELD_ID,
)


def _tensor_boolean(value: torch.Tensor, field: str) -> bool:
    try:
        return bool(value.item())
    except (NotImplementedError, RuntimeError) as error:
        raise ValueError(
            f"{field} values must be readable for validation"
        ) from error


def _require_all_finite(tensor: torch.Tensor, field: str) -> None:
    if not _tensor_boolean(torch.isfinite(tensor).all(), field):
        raise ValueError(f"{field} values must be finite")


def _require_all_nonnegative(tensor: torch.Tensor, field: str) -> None:
    if _tensor_boolean(torch.any(tensor < 0), field):
        raise ValueError(f"{field} values must be nonnegative")


def require_valid_readout_collection(collection: ReadoutCollection) -> None:
    if type(collection) is not ReadoutCollection:
        raise TypeError("collection must be exactly ReadoutCollection")
    if type(collection.sample_grid) is not SampleGrid:
        raise TypeError("ReadoutCollection.sample_grid must be SampleGrid")

    present_field_ids = tuple(collection.fields)
    canonical_present_field_ids = tuple(
        field_id for field_id in READOUT_FIELD_IDS.ids if field_id in collection.fields
    )
    if present_field_ids != canonical_present_field_ids:
        raise ValueError(
            "ReadoutCollection fields must be a recognized nonempty canonical subset"
        )

    reference_field = next(iter(collection.fields.values()))
    reference_layout = reference_field.layout
    expected_shared_axes = IdSequence(
        tuple(axis.id for axis in reference_layout.axes.axes)
    )
    if collection.shared_axes != expected_shared_axes:
        raise ValueError(
            "ReadoutCollection.shared_axes must contain every layout axis in order"
        )

    for field in collection.fields.values():
        if field.layout != reference_layout:
            raise ValueError("ReadoutCollection fields must have one exact layout")
        if field.tensor.device != reference_field.tensor.device:
            raise ValueError("ReadoutCollection fields must have one exact device")
        if field.tensor.layout != torch.strided:
            raise ValueError("ReadoutCollection tensors must use torch.strided layout")

    axes = reference_layout.axes
    try:
        example_axis = axes.axes[axes.index(READOUT_EXAMPLE_AXIS_ID)]
        channel_axis = axes.axes[axes.index(READOUT_CHANNEL_AXIS_ID)]
        sample_axis = axes.axes[axes.index(READOUT_SAMPLE_AXIS_ID)]
    except ValueError as error:
        raise ValueError(
            "ReadoutCollection layout must include example, channel, and sample axes"
        ) from error

    if example_axis.coordinates is None or any(
        type(coordinate) is not ExampleId
        for coordinate in example_axis.coordinates.ids
    ):
        raise ValueError(
            "ReadoutCollection example axis must be ID-backed by exact ExampleId values"
        )
    if channel_axis.coordinates is None or any(
        type(coordinate) is not ChannelId
        for coordinate in channel_axis.coordinates.ids
    ):
        raise ValueError(
            "ReadoutCollection channel axis must be ID-backed by exact ChannelId values"
        )
    if sample_axis.coordinates is not None:
        raise ValueError("ReadoutCollection sample axis must be count-only")

    floating_dtype: torch.dtype | None = None
    for field_id in _FLOATING_FIELD_IDS:
        if field_id not in collection.fields:
            continue
        dtype = collection.tensor(field_id).dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "ReadoutCollection floating fields must use torch.float32 or torch.float64"
            )
        if floating_dtype is None:
            floating_dtype = dtype
        elif dtype != floating_dtype:
            raise ValueError(
                "ReadoutCollection floating fields must have one exact dtype"
            )

    if (
        READOUT_PHOTOELECTRONS_FIELD_ID in collection.fields
        and collection.tensor(READOUT_PHOTOELECTRONS_FIELD_ID).dtype != torch.int64
    ):
        raise ValueError("readout.photoelectrons must use torch.int64")
    if (
        READOUT_DIGITIZED_WAVEFORM_FIELD_ID in collection.fields
        and collection.tensor(READOUT_DIGITIZED_WAVEFORM_FIELD_ID).dtype
        != torch.int32
    ):
        raise ValueError("readout.waveform.digitized must use torch.int32")

    has_digitized = READOUT_DIGITIZED_WAVEFORM_FIELD_ID in collection.fields
    if has_digitized:
        if collection.digitized_waveform_spec is None:
            raise ValueError(
                "ReadoutCollection.digitized_waveform_spec is required when digitized is present"
            )
        if type(collection.digitized_waveform_spec) is not DigitizedWaveformSpec:
            raise TypeError(
                "ReadoutCollection.digitized_waveform_spec must be DigitizedWaveformSpec when digitized is present"
            )
    elif collection.digitized_waveform_spec is not None:
        raise ValueError(
            "ReadoutCollection.digitized_waveform_spec requires the digitized field"
        )

    if READOUT_PHOTOELECTRONS_FIELD_ID in collection.fields:
        _require_all_nonnegative(
            collection.tensor(READOUT_PHOTOELECTRONS_FIELD_ID),
            READOUT_PHOTOELECTRONS_FIELD_ID.value,
        )
    if READOUT_CHARGE_FIELD_ID in collection.fields:
        charge = collection.tensor(READOUT_CHARGE_FIELD_ID)
        _require_all_finite(charge, READOUT_CHARGE_FIELD_ID.value)
        _require_all_nonnegative(charge, READOUT_CHARGE_FIELD_ID.value)
    for field_id in (
        READOUT_PURE_WAVEFORM_FIELD_ID,
        READOUT_NOISE_WAVEFORM_FIELD_ID,
        READOUT_ANALOG_WAVEFORM_FIELD_ID,
    ):
        if field_id in collection.fields:
            _require_all_finite(collection.tensor(field_id), field_id.value)
    if has_digitized:
        digitized = collection.tensor(READOUT_DIGITIZED_WAVEFORM_FIELD_ID)
        _require_all_nonnegative(
            digitized, READOUT_DIGITIZED_WAVEFORM_FIELD_ID.value
        )
        spec = collection.digitized_waveform_spec
        if spec is None:
            raise ValueError("digitized waveform spec is required")
        if _tensor_boolean(
            torch.any(digitized > spec.adc_max),
            READOUT_DIGITIZED_WAVEFORM_FIELD_ID.value,
        ):
            raise ValueError(
                "readout.waveform.digitized values must not exceed adc_max"
            )
