from __future__ import annotations

from unittest import mock
import unittest

import torch
from tensor_core import (
    IdSequence,
    TensorAxisSelection,
    TensorCollection,
    TensorFieldSelection,
)

from tensor_dslab.readout import (
    READOUT_CHANNEL_AXIS_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_EXAMPLE_AXIS_ID,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_PHOTOELECTRONS_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
    READOUT_SAMPLE_AXIS_ID,
    ReadoutCollection,
    move_readout_collection,
    project_readout_fields,
    select_readout_indices,
)
from tests.readout_fixtures import (
    EXTRA_AXIS_ID,
    make_collection,
    make_layout,
)


class ReadoutTensorOperationTest(unittest.TestCase):
    def test_tensorcore_projection_returns_base_collection(self) -> None:
        source = make_collection()
        selection = TensorFieldSelection(
            IdSequence((READOUT_PHOTOELECTRONS_FIELD_ID,))
        )
        selected = source.select_fields(selection)
        self.assertIs(type(selected), TensorCollection)
        self.assertIs(
            selected.field(READOUT_PHOTOELECTRONS_FIELD_ID),
            source.field(READOUT_PHOTOELECTRONS_FIELD_ID),
        )

    def test_domain_projection_reconstructs_and_shares_field_records(self) -> None:
        source = make_collection()
        selection = TensorFieldSelection(
            IdSequence(
                (
                    READOUT_PHOTOELECTRONS_FIELD_ID,
                    READOUT_NOISE_WAVEFORM_FIELD_ID,
                )
            )
        )
        selected = project_readout_fields(source, selection)
        self.assertIs(type(selected), ReadoutCollection)
        self.assertEqual(tuple(selected.fields), selection.ids.ids)
        self.assertIs(selected.sample_grid, source.sample_grid)
        for field_id in selection.ids.ids:
            self.assertIs(selected.field(field_id), source.field(field_id))

    def test_noncanonical_model_projection_remains_tensorcore_owned(self) -> None:
        source = make_collection()
        selection = TensorFieldSelection(
            IdSequence(
                (
                    READOUT_NOISE_WAVEFORM_FIELD_ID,
                    READOUT_PURE_WAVEFORM_FIELD_ID,
                )
            )
        )
        selected = source.select_fields(selection)
        self.assertIs(type(selected), TensorCollection)
        self.assertEqual(tuple(selected.fields), selection.ids.ids)
        with self.assertRaises(ValueError):
            project_readout_fields(source, selection)

    def test_projection_retains_or_drops_digitized_spec_with_its_field(self) -> None:
        source = make_collection()
        digitized = project_readout_fields(
            source,
            TensorFieldSelection(IdSequence((READOUT_DIGITIZED_WAVEFORM_FIELD_ID,))),
        )
        self.assertIs(digitized.digitized_waveform_spec, source.digitized_waveform_spec)
        nondigitized = project_readout_fields(
            source,
            TensorFieldSelection(IdSequence((READOUT_PHOTOELECTRONS_FIELD_ID,))),
        )
        self.assertIsNone(nondigitized.digitized_waveform_spec)

    def test_example_channel_and_extra_selection_preserve_sample_grid(self) -> None:
        layout = make_layout(
            (
                READOUT_EXAMPLE_AXIS_ID,
                EXTRA_AXIS_ID,
                READOUT_CHANNEL_AXIS_ID,
                READOUT_SAMPLE_AXIS_ID,
            )
        )
        for axis_id in (
            READOUT_EXAMPLE_AXIS_ID,
            READOUT_CHANNEL_AXIS_ID,
            EXTRA_AXIS_ID,
        ):
            source = make_collection(layout=layout)
            selected = select_readout_indices(
                source,
                TensorAxisSelection(axis_id=axis_id, indices=(1,)),
            )
            self.assertIs(selected.sample_grid, source.sample_grid)
            self.assertIs(selected.digitized_waveform_spec, source.digitized_waveform_spec)

    def test_contiguous_sample_selection_advances_origin_and_offset(self) -> None:
        source = make_collection()
        selected = select_readout_indices(
            source,
            TensorAxisSelection(
                axis_id=READOUT_SAMPLE_AXIS_ID,
                indices=(1, 2, 3),
            ),
        )
        self.assertIs(selected.sample_grid.sample_period_ns, source.sample_grid.sample_period_ns)
        self.assertEqual(selected.sample_grid.origin_ns.value, -1.0)
        self.assertEqual(selected.sample_grid.sample_offset.value, 8)
        self.assertEqual(selected.layout.axes.size(READOUT_SAMPLE_AXIS_ID), 3)
        self.assertIs(selected.digitized_waveform_spec, source.digitized_waveform_spec)

    def test_irregular_sample_selection_fails_before_tensor_allocation(self) -> None:
        source = make_collection()
        for indices in ((0, 2), (2, 1)):
            with self.subTest(indices=indices), mock.patch("torch.index_select") as operation:
                with self.assertRaises(ValueError):
                    select_readout_indices(
                        source,
                        TensorAxisSelection(
                            axis_id=READOUT_SAMPLE_AXIS_ID,
                            indices=indices,
                        ),
                    )
                operation.assert_not_called()

    def test_move_reconstructs_on_exact_device_without_dtype_cast(self) -> None:
        source = make_collection()
        moved = move_readout_collection(source, device=torch.device("cpu"))
        self.assertIs(type(moved), ReadoutCollection)
        self.assertEqual(moved.device, torch.device("cpu"))
        self.assertIs(moved.sample_grid, source.sample_grid)
        self.assertIs(moved.digitized_waveform_spec, source.digitized_waveform_spec)
        for field_id, source_field in source.fields.items():
            self.assertEqual(moved.tensor(field_id).dtype, source_field.tensor.dtype)
            self.assertTrue(torch.equal(moved.tensor(field_id), source_field.tensor))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_projection_selection_and_movement(self) -> None:
        source = make_collection(device=torch.device("cuda"))
        projected = project_readout_fields(
            source,
            TensorFieldSelection(IdSequence((READOUT_PHOTOELECTRONS_FIELD_ID,))),
        )
        selected = select_readout_indices(
            projected,
            TensorAxisSelection(READOUT_SAMPLE_AXIS_ID, (0, 1)),
        )
        moved = move_readout_collection(selected, device=torch.device("cpu"))
        self.assertEqual(moved.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
