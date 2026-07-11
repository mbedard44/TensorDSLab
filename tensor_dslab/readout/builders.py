from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
from tensor_core import TensorField, TensorFieldId

from tensor_dslab.readout.ids import (
    READOUT_ANALOG_WAVEFORM_FIELD_ID,
    READOUT_CHARGE_FIELD_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_FIELD_IDS,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_PHOTOELECTRONS_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
)
from tensor_dslab.readout.tensors import ReadoutCollection
from tensor_dslab.readout.types import DigitizedWaveformSpec
from tensor_dslab.readout.validation import require_valid_readout_collection


_STALE_DESCENDANTS_BY_TARGET = MappingProxyType(
    {
        READOUT_PHOTOELECTRONS_FIELD_ID: frozenset(
            (
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            )
        ),
        READOUT_CHARGE_FIELD_ID: frozenset(
            (
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            )
        ),
        READOUT_PURE_WAVEFORM_FIELD_ID: frozenset(
            (
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            )
        ),
        READOUT_NOISE_WAVEFORM_FIELD_ID: frozenset(
            (
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            )
        ),
        READOUT_ANALOG_WAVEFORM_FIELD_ID: frozenset(
            (READOUT_DIGITIZED_WAVEFORM_FIELD_ID,)
        ),
        READOUT_DIGITIZED_WAVEFORM_FIELD_ID: frozenset(),
    }
)

_REQUIRED_SOURCE_FIELDS_BY_TARGET = MappingProxyType(
    {
        READOUT_PHOTOELECTRONS_FIELD_ID: frozenset(
            (READOUT_PHOTOELECTRONS_FIELD_ID,)
        ),
        READOUT_CHARGE_FIELD_ID: frozenset(
            (READOUT_PHOTOELECTRONS_FIELD_ID,)
        ),
        READOUT_PURE_WAVEFORM_FIELD_ID: frozenset((READOUT_CHARGE_FIELD_ID,)),
        READOUT_NOISE_WAVEFORM_FIELD_ID: frozenset(),
        READOUT_ANALOG_WAVEFORM_FIELD_ID: frozenset(
            (
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            )
        ),
        READOUT_DIGITIZED_WAVEFORM_FIELD_ID: frozenset(
            (READOUT_ANALOG_WAVEFORM_FIELD_ID,)
        ),
    }
)

_FLOATING_FIELD_IDS = frozenset(
    (
        READOUT_CHARGE_FIELD_ID,
        READOUT_PURE_WAVEFORM_FIELD_ID,
        READOUT_NOISE_WAVEFORM_FIELD_ID,
        READOUT_ANALOG_WAVEFORM_FIELD_ID,
    )
)


def _require_target_dtype(
    target_field_id: TensorFieldId,
    target_dtype: torch.dtype,
    retained_fields: dict[TensorFieldId, TensorField],
) -> None:
    if type(target_dtype) is not torch.dtype:
        raise TypeError("target_dtype must be torch.dtype")
    if target_field_id == READOUT_PHOTOELECTRONS_FIELD_ID:
        if target_dtype != torch.int64:
            raise ValueError("photoelectron target_dtype must be torch.int64")
        return
    if target_field_id == READOUT_DIGITIZED_WAVEFORM_FIELD_ID:
        if target_dtype != torch.int32:
            raise ValueError("digitized target_dtype must be torch.int32")
        return
    if target_dtype not in (torch.float32, torch.float64):
        raise ValueError("floating target_dtype must be torch.float32 or torch.float64")
    retained_floating_dtypes = {
        field.tensor.dtype
        for field_id, field in retained_fields.items()
        if field_id in _FLOATING_FIELD_IDS
    }
    if retained_floating_dtypes and retained_floating_dtypes != {target_dtype}:
        raise ValueError("floating target_dtype must match retained floating fields")


def _zeros_field(
    source: ReadoutCollection,
    *,
    field_id: TensorFieldId,
    dtype: torch.dtype,
) -> TensorField:
    shape = tuple(
        source.layout.axes.size(axis.id) for axis in source.layout.axes.axes
    )
    tensor = torch.zeros(shape, dtype=dtype, device=source.device)
    return TensorField(id=field_id, tensor=tensor, layout=source.layout, metadata={})


def _canonical_fields(
    fields: Mapping[TensorFieldId, TensorField],
) -> dict[TensorFieldId, TensorField]:
    return {
        field_id: fields[field_id]
        for field_id in READOUT_FIELD_IDS.ids
        if field_id in fields
    }


def _plan_fields_for_target(
    fields: Mapping[TensorFieldId, TensorField],
    *,
    target_field_id: TensorFieldId,
    target_field: TensorField | None = None,
) -> dict[TensorFieldId, TensorField]:
    removed = _STALE_DESCENDANTS_BY_TARGET[target_field_id] | {target_field_id}
    planned = {
        field_id: field
        for field_id, field in fields.items()
        if field_id not in removed
    }
    if target_field is not None:
        if target_field.id != target_field_id:
            raise ValueError("target_field.id must match target_field_id")
        planned[target_field_id] = target_field
    return _canonical_fields(planned)


def build_readout_result_buffer(
    source: ReadoutCollection,
    *,
    target_field_id: TensorFieldId,
    target_dtype: torch.dtype,
    digitized_waveform_spec: DigitizedWaveformSpec | None = None,
) -> ReadoutCollection:
    require_valid_readout_collection(source)
    if type(target_field_id) is not TensorFieldId:
        raise TypeError("target_field_id must be TensorFieldId")
    if target_field_id not in _STALE_DESCENDANTS_BY_TARGET:
        raise ValueError("target_field_id must be a recognized readout field")

    required_source_fields = _REQUIRED_SOURCE_FIELDS_BY_TARGET[target_field_id]
    missing = required_source_fields.difference(source.fields)
    if missing:
        names = ", ".join(sorted(field_id.value for field_id in missing))
        raise ValueError(f"source is missing required fields: {names}")

    if target_field_id == READOUT_DIGITIZED_WAVEFORM_FIELD_ID:
        if digitized_waveform_spec is None:
            raise ValueError("digitized target requires digitized_waveform_spec")
        if type(digitized_waveform_spec) is not DigitizedWaveformSpec:
            raise TypeError("digitized target requires DigitizedWaveformSpec")
    elif digitized_waveform_spec is not None:
        raise ValueError(
            "digitized_waveform_spec is accepted only for the digitized target"
        )

    retained_fields = _plan_fields_for_target(
        source.fields,
        target_field_id=target_field_id,
    )
    _require_target_dtype(target_field_id, target_dtype, retained_fields)
    target_field = _zeros_field(
        source,
        field_id=target_field_id,
        dtype=target_dtype,
    )
    fields = _plan_fields_for_target(
        source.fields,
        target_field_id=target_field_id,
        target_field=target_field,
    )

    result_digitized_spec = (
        digitized_waveform_spec
        if target_field_id == READOUT_DIGITIZED_WAVEFORM_FIELD_ID
        else (
            source.digitized_waveform_spec
            if READOUT_DIGITIZED_WAVEFORM_FIELD_ID in fields
            else None
        )
    )
    return ReadoutCollection(
        fields=fields,
        shared_axes=source.shared_axes,
        metadata=source.metadata,
        sample_grid=source.sample_grid,
        digitized_waveform_spec=result_digitized_spec,
    )


def build_readout_output_buffer(
    source: ReadoutCollection,
    *,
    floating_dtype: torch.dtype,
    replace_photoelectrons: bool,
    digitized_waveform_spec: DigitizedWaveformSpec | None = None,
) -> ReadoutCollection:
    require_valid_readout_collection(source)
    if READOUT_PHOTOELECTRONS_FIELD_ID not in source.fields:
        raise ValueError("source must contain readout.photoelectrons")
    if type(floating_dtype) is not torch.dtype:
        raise TypeError("floating_dtype must be torch.dtype")
    if floating_dtype not in (torch.float32, torch.float64):
        raise ValueError("floating_dtype must be torch.float32 or torch.float64")
    if type(replace_photoelectrons) is not bool:
        raise TypeError("replace_photoelectrons must be bool")
    if (
        digitized_waveform_spec is not None
        and type(digitized_waveform_spec) is not DigitizedWaveformSpec
    ):
        raise TypeError("digitized_waveform_spec must be DigitizedWaveformSpec")

    fields: dict[TensorFieldId, TensorField] = {}
    if replace_photoelectrons:
        photoelectron_field = _zeros_field(
            source,
            field_id=READOUT_PHOTOELECTRONS_FIELD_ID,
            dtype=torch.int64,
        )
    else:
        photoelectron_field = source.field(
            READOUT_PHOTOELECTRONS_FIELD_ID
        )
    fields = _plan_fields_for_target(
        fields,
        target_field_id=READOUT_PHOTOELECTRONS_FIELD_ID,
        target_field=photoelectron_field,
    )
    for field_id in (
        READOUT_CHARGE_FIELD_ID,
        READOUT_PURE_WAVEFORM_FIELD_ID,
        READOUT_NOISE_WAVEFORM_FIELD_ID,
        READOUT_ANALOG_WAVEFORM_FIELD_ID,
    ):
        target_field = _zeros_field(
            source,
            field_id=field_id,
            dtype=floating_dtype,
        )
        fields = _plan_fields_for_target(
            fields,
            target_field_id=field_id,
            target_field=target_field,
        )
    if digitized_waveform_spec is not None:
        target_field = _zeros_field(
            source,
            field_id=READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            dtype=torch.int32,
        )
        fields = _plan_fields_for_target(
            fields,
            target_field_id=READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            target_field=target_field,
        )

    return ReadoutCollection(
        fields=fields,
        shared_axes=source.shared_axes,
        metadata=source.metadata,
        sample_grid=source.sample_grid,
        digitized_waveform_spec=digitized_waveform_spec,
    )
