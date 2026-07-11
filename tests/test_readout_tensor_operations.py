from __future__ import annotations

from unittest import mock
import unittest

import torch
from tensor_core import (
    IdSequence,
    TensorAxisSelection,
    TensorCollection,
    TensorField,
    TensorFieldId,
    TensorFieldSelection,
    TensorLayout,
)

from tensor_dslab.readout import (
    READOUT_CHANNEL_AXIS_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_EXAMPLE_AXIS_ID,
    READOUT_FIELD_IDS,
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
    field_dtype,
    make_collection,
    make_layout,
)


def _make_distinctive_collection(*, layout: TensorLayout) -> ReadoutCollection:
    exact_shape = tuple(layout.axes.size(axis.id) for axis in layout.axes.axes)
    element_count = torch.empty(exact_shape).numel()
    tensor_overrides: dict[TensorFieldId, torch.Tensor] = {}
    for field_index, field_id in enumerate(READOUT_FIELD_IDS.ids):
        dtype = field_dtype(field_id, torch.float32)
        tensor = torch.arange(
            1,
            element_count + 1,
            dtype=dtype,
        ).reshape(exact_shape)
        tensor = tensor + field_index * (element_count + 1)
        if tensor.is_floating_point():
            tensor.requires_grad_()
        tensor_overrides[field_id] = tensor
    return make_collection(layout=layout, tensor_overrides=tensor_overrides)


def _direct_selection(
    tensor: torch.Tensor,
    *,
    dimension: int,
    indices: tuple[int, ...],
) -> torch.Tensor:
    index: list[slice | list[int]] = [slice(None)] * tensor.ndim
    index[dimension] = list(indices)
    return tensor[tuple(index)]


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
        original_select_fields = TensorCollection.select_fields
        delegated_results: list[TensorCollection] = []

        def capture_projection(
            base_collection: TensorCollection,
            base_selection: TensorFieldSelection,
        ) -> TensorCollection:
            selected_base = original_select_fields(base_collection, base_selection)
            result = TensorCollection(
                fields={
                    field_id: TensorField(
                        id=field.id,
                        tensor=field.tensor,
                        layout=field.layout,
                        metadata={"delegated": field_id.value},
                    )
                    for field_id, field in selected_base.fields.items()
                },
                shared_axes=selected_base.shared_axes,
                metadata=selected_base.metadata,
            )
            delegated_results.append(result)
            return result

        with mock.patch.object(
            TensorCollection,
            "select_fields",
            autospec=True,
            side_effect=capture_projection,
        ) as operation:
            selected = project_readout_fields(source, selection)
        operation.assert_called_once_with(source, selection)
        self.assertEqual(len(delegated_results), 1)
        delegated = delegated_results[0]
        self.assertIs(type(delegated), TensorCollection)

        self.assertIs(type(selected), ReadoutCollection)
        self.assertEqual(tuple(selected.fields), selection.ids.ids)
        self.assertIs(selected.sample_grid, source.sample_grid)
        for field_id in selection.ids.ids:
            delegated_field = delegated.field(field_id)
            self.assertIs(selected.field(field_id), delegated_field)
            self.assertIsNot(delegated_field, source.field(field_id))
            self.assertIs(
                delegated_field.tensor,
                source.field(field_id).tensor,
            )
            self.assertEqual(delegated_field.metadata, {"delegated": field_id.value})

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

    def test_axis_selection_delegates_and_preserves_exact_semantics(self) -> None:
        axis_orders = (
            (
                "sample-last",
                (
                    READOUT_EXAMPLE_AXIS_ID,
                    EXTRA_AXIS_ID,
                    READOUT_CHANNEL_AXIS_ID,
                    READOUT_SAMPLE_AXIS_ID,
                ),
            ),
            (
                "sample-first",
                (
                    READOUT_SAMPLE_AXIS_ID,
                    READOUT_CHANNEL_AXIS_ID,
                    EXTRA_AXIS_ID,
                    READOUT_EXAMPLE_AXIS_ID,
                ),
            ),
        )
        selections = (
            (READOUT_EXAMPLE_AXIS_ID, (1,)),
            (READOUT_CHANNEL_AXIS_ID, (1,)),
            (EXTRA_AXIS_ID, (1,)),
            (READOUT_SAMPLE_AXIS_ID, (1, 2, 3)),
        )

        original_select_indices = TensorCollection.select_indices
        for order_name, axis_order in axis_orders:
            layout = make_layout(axis_order)
            for axis_id, indices in selections:
                with self.subTest(order=order_name, axis=axis_id.value):
                    source = _make_distinctive_collection(layout=layout)
                    selection = TensorAxisSelection(axis_id=axis_id, indices=indices)
                    delegated_results: list[TensorCollection] = []

                    def capture_selection(
                        base_collection: TensorCollection,
                        base_selection: TensorAxisSelection,
                    ) -> TensorCollection:
                        result = original_select_indices(
                            base_collection,
                            base_selection,
                        )
                        delegated_results.append(result)
                        return result

                    with mock.patch.object(
                        TensorCollection,
                        "select_indices",
                        autospec=True,
                        side_effect=capture_selection,
                    ) as operation:
                        selected = select_readout_indices(source, selection)
                    operation.assert_called_once_with(source, selection)
                    self.assertEqual(len(delegated_results), 1)
                    delegated = delegated_results[0]
                    self.assertIs(type(delegated), TensorCollection)

                    self.assertIs(type(selected), ReadoutCollection)
                    self.assertEqual(selected.shared_axes, source.shared_axes)
                    self.assertEqual(selected.metadata, source.metadata)
                    self.assertIs(
                        selected.digitized_waveform_spec,
                        source.digitized_waveform_spec,
                    )

                    source_dimension = source.layout.axes.index(axis_id)
                    expected_shape = list(
                        source.tensor(READOUT_PHOTOELECTRONS_FIELD_ID).shape
                    )
                    expected_shape[source_dimension] = len(indices)
                    self.assertEqual(
                        tuple(axis.id for axis in selected.layout.axes.axes),
                        axis_order,
                    )
                    self.assertEqual(
                        tuple(axis.size for axis in selected.layout.axes.axes),
                        tuple(expected_shape),
                    )

                    for source_axis, selected_axis in zip(
                        source.layout.axes.axes,
                        selected.layout.axes.axes,
                        strict=True,
                    ):
                        if source_axis.id == axis_id:
                            expected_coordinates = (
                                None
                                if source_axis.coordinates is None
                                else tuple(
                                    source_axis.coordinates.ids[index]
                                    for index in indices
                                )
                            )
                            self.assertEqual(selected_axis.size, len(indices))
                            self.assertEqual(
                                None
                                if selected_axis.coordinates is None
                                else selected_axis.coordinates.ids,
                                expected_coordinates,
                            )
                        else:
                            self.assertEqual(selected_axis, source_axis)

                    expected_index_by_axis = {
                        axis.id: {
                            coordinate: index
                            for index, coordinate in enumerate(axis.coordinates.ids)
                        }
                        for axis in selected.layout.axes.axes
                        if axis.coordinates is not None
                    }
                    actual_index_by_axis = {
                        mapped_axis_id: dict(coordinate_map)
                        for mapped_axis_id, coordinate_map in (
                            selected.layout.index_by_axis.items()
                        )
                    }
                    self.assertEqual(actual_index_by_axis, expected_index_by_axis)
                    if axis_id == READOUT_SAMPLE_AXIS_ID:
                        self.assertNotIn(
                            READOUT_SAMPLE_AXIS_ID,
                            selected.layout.index_by_axis,
                        )

                    for field_id, source_field in source.fields.items():
                        selected_field = selected.field(field_id)
                        self.assertIs(selected_field, delegated.field(field_id))
                        expected_tensor = _direct_selection(
                            source_field.tensor,
                            dimension=source_dimension,
                            indices=indices,
                        )
                        self.assertEqual(selected_field.layout, selected.layout)
                        self.assertEqual(selected_field.metadata, source_field.metadata)
                        self.assertEqual(
                            selected_field.tensor.shape,
                            tuple(expected_shape),
                        )
                        self.assertEqual(
                            selected_field.tensor.dtype,
                            source_field.tensor.dtype,
                        )
                        self.assertEqual(
                            selected_field.tensor.requires_grad,
                            source_field.tensor.requires_grad,
                        )
                        self.assertTrue(
                            torch.equal(selected_field.tensor, expected_tensor)
                        )
                        self.assertGreater(
                            torch.count_nonzero(selected_field.tensor).item(),
                            0,
                        )
                        if source_field.tensor.requires_grad:
                            self.assertIsNotNone(selected_field.tensor.grad_fn)

                    if axis_id == READOUT_SAMPLE_AXIS_ID:
                        self.assertIs(
                            selected.sample_grid.sample_period_ns,
                            source.sample_grid.sample_period_ns,
                        )
                        self.assertEqual(selected.sample_grid.origin_ns.value, -1.0)
                        self.assertEqual(selected.sample_grid.sample_offset.value, 8)
                    else:
                        self.assertIs(selected.sample_grid, source.sample_grid)

                    source_autograd_tensor = source.tensor(
                        READOUT_PURE_WAVEFORM_FIELD_ID
                    )
                    selected.tensor(READOUT_PURE_WAVEFORM_FIELD_ID).sum().backward()
                    expected_gradient = torch.zeros_like(source_autograd_tensor)
                    gradient_index: list[slice | list[int]] = [
                        slice(None)
                    ] * source_autograd_tensor.ndim
                    gradient_index[source_dimension] = list(indices)
                    expected_gradient[tuple(gradient_index)] = 1
                    source_gradient = source_autograd_tensor.grad
                    self.assertIsNotNone(source_gradient)
                    assert source_gradient is not None
                    self.assertTrue(torch.equal(source_gradient, expected_gradient))

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
        axis_orders = (
            (
                READOUT_EXAMPLE_AXIS_ID,
                EXTRA_AXIS_ID,
                READOUT_CHANNEL_AXIS_ID,
                READOUT_SAMPLE_AXIS_ID,
            ),
            (
                READOUT_SAMPLE_AXIS_ID,
                READOUT_CHANNEL_AXIS_ID,
                EXTRA_AXIS_ID,
                READOUT_EXAMPLE_AXIS_ID,
            ),
        )
        original_to = TensorCollection.to
        for axis_order in axis_orders:
            with self.subTest(axis_order=tuple(axis.value for axis in axis_order)):
                source = _make_distinctive_collection(layout=make_layout(axis_order))
                device = torch.device("cpu")
                delegated_results: list[TensorCollection] = []

                def capture_movement(
                    base_collection: TensorCollection,
                    *,
                    device: torch.device,
                ) -> TensorCollection:
                    result = original_to(base_collection, device=device)
                    delegated_results.append(result)
                    return result

                with mock.patch.object(
                    TensorCollection,
                    "to",
                    autospec=True,
                    side_effect=capture_movement,
                ) as operation:
                    moved = move_readout_collection(source, device=device)
                operation.assert_called_once_with(source, device=device)
                self.assertEqual(len(delegated_results), 1)
                delegated = delegated_results[0]
                self.assertIs(type(delegated), TensorCollection)

                self.assertIs(type(moved), ReadoutCollection)
                self.assertIsNot(moved, source)
                self.assertEqual(moved.device, device)
                self.assertEqual(moved.shared_axes, source.shared_axes)
                self.assertEqual(moved.metadata, source.metadata)
                self.assertIs(moved.sample_grid, source.sample_grid)
                self.assertIs(
                    moved.digitized_waveform_spec,
                    source.digitized_waveform_spec,
                )
                for field_id, source_field in source.fields.items():
                    moved_field = moved.field(field_id)
                    self.assertIs(moved_field, delegated.field(field_id))
                    self.assertIsNot(moved_field, source_field)
                    self.assertEqual(moved_field.layout, source_field.layout)
                    self.assertEqual(moved_field.metadata, source_field.metadata)
                    self.assertEqual(
                        moved_field.tensor.shape,
                        source_field.tensor.shape,
                    )
                    self.assertEqual(
                        moved_field.tensor.dtype,
                        source_field.tensor.dtype,
                    )
                    self.assertEqual(
                        moved_field.tensor.requires_grad,
                        source_field.tensor.requires_grad,
                    )
                    self.assertEqual(
                        moved_field.tensor.is_leaf,
                        source_field.tensor.is_leaf,
                    )
                    self.assertIs(
                        moved_field.tensor.grad_fn,
                        source_field.tensor.grad_fn,
                    )
                    self.assertTrue(
                        torch.equal(moved_field.tensor, source_field.tensor)
                    )
                    self.assertGreater(
                        torch.count_nonzero(moved_field.tensor).item(),
                        0,
                    )

                source_autograd_tensor = source.tensor(
                    READOUT_PURE_WAVEFORM_FIELD_ID
                )
                moved.tensor(READOUT_PURE_WAVEFORM_FIELD_ID).sum().backward()
                source_gradient = source_autograd_tensor.grad
                self.assertIsNotNone(source_gradient)
                assert source_gradient is not None
                self.assertTrue(
                    torch.equal(
                        source_gradient,
                        torch.ones_like(source_autograd_tensor),
                    )
                )

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
