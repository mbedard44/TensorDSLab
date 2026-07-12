from __future__ import annotations

import torch
from tensor_core import (
    FiniteFloat,
    NonnegativeInteger,
    TensorAxisSelection,
    TensorCollection,
    TensorFieldSelection,
)

from tensor_dslab.readout.ids import (
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_FIELD_IDS,
    SAMPLE_AXIS_ID,
)
from tensor_dslab.readout.types import (
    DigitizedWaveformSpec,
    ReadoutCollection,
    SampleGrid,
)


def _reconstruct_readout_collection(
    collection: TensorCollection,
    *,
    sample_grid: SampleGrid,
    digitized_waveform_spec: DigitizedWaveformSpec | None = None,
) -> ReadoutCollection:
    if type(collection) is not TensorCollection:
        raise TypeError("collection must be exactly TensorCollection")
    return ReadoutCollection(
        fields=collection.fields,
        shared_axes=collection.shared_axes,
        metadata=collection.metadata,
        sample_grid=sample_grid,
        digitized_waveform_spec=digitized_waveform_spec,
    )


def project_readout_fields(
    collection: ReadoutCollection,
    selection: TensorFieldSelection,
) -> ReadoutCollection:
    from tensor_dslab.readout.validation import require_valid_readout_collection

    require_valid_readout_collection(collection)
    if type(selection) is not TensorFieldSelection:
        raise TypeError("selection must be TensorFieldSelection")
    requested = selection.ids.ids
    canonical_requested = tuple(
        field_id for field_id in READOUT_FIELD_IDS.ids if field_id in requested
    )
    if requested != canonical_requested:
        raise ValueError("readout field projection must use canonical field order")

    selected = collection.select_fields(selection)
    digitized_waveform_spec = (
        collection.digitized_waveform_spec
        if READOUT_DIGITIZED_WAVEFORM_FIELD_ID in selected.fields
        else None
    )
    return _reconstruct_readout_collection(
        selected,
        sample_grid=collection.sample_grid,
        digitized_waveform_spec=digitized_waveform_spec,
    )


def select_readout_indices(
    collection: ReadoutCollection,
    selection: TensorAxisSelection,
) -> ReadoutCollection:
    from tensor_dslab.readout.validation import require_valid_readout_collection

    require_valid_readout_collection(collection)
    if type(selection) is not TensorAxisSelection:
        raise TypeError("selection must be TensorAxisSelection")

    sample_grid = collection.sample_grid
    if selection.axis_id == SAMPLE_AXIS_ID:
        indices = selection.indices
        expected = tuple(range(indices[0], indices[0] + len(indices)))
        if indices != expected:
            raise ValueError(
                "sample selection must be contiguous, increasing, and unit-stride"
            )

    selected = collection.select_indices(selection)
    if selection.axis_id == SAMPLE_AXIS_ID:
        first_index = selection.indices[0]
        sample_grid = SampleGrid(
            sample_period_ns=collection.sample_grid.sample_period_ns,
            origin_ns=FiniteFloat(
                collection.sample_grid.origin_ns.value
                + first_index * collection.sample_grid.sample_period_ns.value
            ),
            sample_offset=NonnegativeInteger(
                collection.sample_grid.sample_offset.value + first_index
            ),
        )

    return _reconstruct_readout_collection(
        selected,
        sample_grid=sample_grid,
        digitized_waveform_spec=collection.digitized_waveform_spec,
    )


def move_readout_collection(
    collection: ReadoutCollection,
    *,
    device: torch.device,
) -> ReadoutCollection:
    from tensor_dslab.readout.validation import require_valid_readout_collection

    require_valid_readout_collection(collection)
    if type(device) is not torch.device:
        raise TypeError("device must be torch.device")
    moved = collection.to(device=device)
    return _reconstruct_readout_collection(
        moved,
        sample_grid=collection.sample_grid,
        digitized_waveform_spec=collection.digitized_waveform_spec,
    )
