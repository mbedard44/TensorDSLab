from __future__ import annotations

import unittest

import torch
from tensor_core import TensorFieldId

from tensor_dslab.readout import (
    READOUT_ANALOG_WAVEFORM_FIELD_ID,
    READOUT_CHARGE_FIELD_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_FIELD_IDS,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_PHOTOELECTRONS_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
    ReadoutCollection,
    build_readout_output_buffer,
    build_readout_result_buffer,
)
from tests.readout_fixtures import (
    ALTERNATE_AXIS_ORDER,
    make_collection,
    make_digitized_spec,
    make_layout,
    storage_pointer,
)


TARGET_DTYPES = {
    READOUT_PHOTOELECTRONS_FIELD_ID: torch.int64,
    READOUT_CHARGE_FIELD_ID: torch.float32,
    READOUT_PURE_WAVEFORM_FIELD_ID: torch.float32,
    READOUT_NOISE_WAVEFORM_FIELD_ID: torch.float32,
    READOUT_ANALOG_WAVEFORM_FIELD_ID: torch.float32,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID: torch.int32,
}


def make_distinctive_collection(
    field_ids: tuple[TensorFieldId, ...] = READOUT_FIELD_IDS.ids,
    *,
    floating_dtype: torch.dtype = torch.float32,
    storage_kind: str = "contiguous",
) -> ReadoutCollection:
    if storage_kind not in ("contiguous", "noncontiguous", "expanded"):
        raise ValueError("unknown storage_kind")
    layout = make_layout()
    shape = tuple(layout.axes.size(axis.id) for axis in layout.axes.axes)
    tensor_overrides: dict[TensorFieldId, torch.Tensor] = {}
    for field_index, field_id in enumerate(field_ids):
        dtype = (
            TARGET_DTYPES[field_id]
            if field_id
            in (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            )
            else floating_dtype
        )
        value_offset = (field_index + 1) * 100
        if storage_kind == "expanded":
            base = torch.full((1, *shape[1:]), value_offset, dtype=dtype)
            tensor = base.expand(shape)
        else:
            storage_shape = (
                (*shape, 2) if storage_kind == "noncontiguous" else shape
            )
            base = torch.arange(
                1,
                torch.tensor(storage_shape).prod().item() + 1,
                dtype=dtype,
            ).reshape(storage_shape) + value_offset
            tensor = base[..., 0] if storage_kind == "noncontiguous" else base
        tensor_overrides[field_id] = tensor
    return make_collection(
        field_ids,
        layout=layout,
        floating_dtype=floating_dtype,
        tensor_overrides=tensor_overrides,
    )


def make_atomic_buffer(source, target_field_id):
    kwargs = {}
    if target_field_id == READOUT_DIGITIZED_WAVEFORM_FIELD_ID:
        kwargs["digitized_waveform_spec"] = make_digitized_spec()
    return build_readout_result_buffer(
        source,
        target_field_id=target_field_id,
        target_dtype=TARGET_DTYPES[target_field_id],
        **kwargs,
    )


class ReadoutOutputPreparationTest(unittest.TestCase):
    def test_atomic_output_requires_its_exact_source_fields(self) -> None:
        cases = (
            (READOUT_PHOTOELECTRONS_FIELD_ID, (READOUT_NOISE_WAVEFORM_FIELD_ID,)),
            (READOUT_CHARGE_FIELD_ID, (READOUT_NOISE_WAVEFORM_FIELD_ID,)),
            (READOUT_PURE_WAVEFORM_FIELD_ID, (READOUT_PHOTOELECTRONS_FIELD_ID,)),
            (READOUT_ANALOG_WAVEFORM_FIELD_ID, (READOUT_PURE_WAVEFORM_FIELD_ID,)),
            (
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
                (READOUT_PURE_WAVEFORM_FIELD_ID, READOUT_NOISE_WAVEFORM_FIELD_ID),
            ),
        )
        for target, source_fields in cases:
            with self.subTest(target=target), self.assertRaises(ValueError):
                make_atomic_buffer(make_collection(source_fields), target)
        noise = make_atomic_buffer(
            make_collection((READOUT_PHOTOELECTRONS_FIELD_ID,)),
            READOUT_NOISE_WAVEFORM_FIELD_ID,
        )
        self.assertIn(READOUT_NOISE_WAVEFORM_FIELD_ID, noise.fields)

    def test_each_target_uses_exact_descendant_invalidation_table(self) -> None:
        source = make_collection()
        expected = {
            READOUT_PHOTOELECTRONS_FIELD_ID: (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            ),
            READOUT_CHARGE_FIELD_ID: (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            ),
            READOUT_PURE_WAVEFORM_FIELD_ID: (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            ),
            READOUT_NOISE_WAVEFORM_FIELD_ID: (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            ),
            READOUT_ANALOG_WAVEFORM_FIELD_ID: (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
            ),
            READOUT_DIGITIZED_WAVEFORM_FIELD_ID: tuple(source.fields),
        }
        for target, field_ids in expected.items():
            with self.subTest(target=target):
                result = make_atomic_buffer(source, target)
                self.assertEqual(tuple(result.fields), field_ids)
                self.assertIsNone(
                    result.digitized_waveform_spec
                    if target != READOUT_DIGITIZED_WAVEFORM_FIELD_ID
                    else None
                )

        stale = {
            READOUT_PHOTOELECTRONS_FIELD_ID: {
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            },
            READOUT_CHARGE_FIELD_ID: {
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            },
            READOUT_PURE_WAVEFORM_FIELD_ID: {
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            },
            READOUT_NOISE_WAVEFORM_FIELD_ID: {
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            },
            READOUT_ANALOG_WAVEFORM_FIELD_ID: {
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            },
            READOUT_DIGITIZED_WAVEFORM_FIELD_ID: set(),
        }
        requirements = {
            READOUT_PHOTOELECTRONS_FIELD_ID: {READOUT_PHOTOELECTRONS_FIELD_ID},
            READOUT_CHARGE_FIELD_ID: {READOUT_PHOTOELECTRONS_FIELD_ID},
            READOUT_PURE_WAVEFORM_FIELD_ID: {READOUT_CHARGE_FIELD_ID},
            READOUT_NOISE_WAVEFORM_FIELD_ID: set(),
            READOUT_ANALOG_WAVEFORM_FIELD_ID: {
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            },
            READOUT_DIGITIZED_WAVEFORM_FIELD_ID: {
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
            },
        }
        checked = 0
        canonical = READOUT_FIELD_IDS.ids
        for mask in range(1, 1 << len(canonical)):
            subset = tuple(
                field_id
                for index, field_id in enumerate(canonical)
                if mask & (1 << index)
            )
            source_subset = make_distinctive_collection(subset)
            source_records = dict(source_subset.fields)
            source_values = {
                field_id: field.tensor.clone()
                for field_id, field in source_subset.fields.items()
            }
            source_storage = {
                storage_pointer(field.tensor)
                for field in source_subset.fields.values()
            }
            for target in canonical:
                if not requirements[target].issubset(source_subset.fields):
                    continue
                result = make_atomic_buffer(source_subset, target)
                expected_set = (set(subset) - stale[target] - {target}) | {target}
                expected_order = tuple(
                    field_id for field_id in canonical if field_id in expected_set
                )
                self.assertEqual(tuple(result.fields), expected_order)
                for field_id in expected_order:
                    if field_id == target:
                        self.assertIsNot(
                            result.field(field_id),
                            source_records.get(field_id),
                        )
                    else:
                        self.assertIs(result.field(field_id), source_records[field_id])
                self.assertNotIn(
                    storage_pointer(result.tensor(target)),
                    source_storage,
                )
                self.assertEqual(result.field(target).metadata, {})
                for field_id, source_field in source_subset.fields.items():
                    self.assertIs(source_field, source_records[field_id])
                    self.assertTrue(
                        torch.equal(source_field.tensor, source_values[field_id])
                    )
                checked += 1
        self.assertEqual(checked, 207)

    def test_atomic_output_shares_every_unaffected_record_for_all_targets(self) -> None:
        source = make_distinctive_collection()
        original_records = dict(source.fields)
        original_values = {
            field_id: field.tensor.clone() for field_id, field in source.fields.items()
        }
        for target in READOUT_FIELD_IDS.ids:
            with self.subTest(target=target):
                result = make_atomic_buffer(source, target)
                for field_id, field in result.fields.items():
                    if field_id == target:
                        self.assertIsNot(field, original_records[field_id])
                    else:
                        self.assertIs(field, original_records[field_id])
                for field_id, field in source.fields.items():
                    self.assertIs(field, original_records[field_id])
                    self.assertTrue(
                        torch.equal(field.tensor, original_values[field_id])
                    )

    def test_atomic_target_is_distinct_zero_initialized_and_role_typed(self) -> None:
        source = make_collection()
        for target, dtype in TARGET_DTYPES.items():
            with self.subTest(target=target):
                result = make_atomic_buffer(source, target)
                tensor = result.tensor(target)
                self.assertEqual(tensor.dtype, dtype)
                self.assertEqual(torch.count_nonzero(tensor).item(), 0)
                self.assertIsNot(tensor, source.tensor(target))
                self.assertEqual(result.field(target).metadata, {})
        with self.assertRaises(TypeError):
            build_readout_result_buffer(
                source,
                target_field_id=READOUT_CHARGE_FIELD_ID,
                target_dtype="float32",  # type: ignore[arg-type]
            )
        mixed_source = make_collection(
            (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            ),
            floating_dtype=torch.float64,
        )
        with self.assertRaises(ValueError):
            build_readout_result_buffer(
                mixed_source,
                target_field_id=READOUT_CHARGE_FIELD_ID,
                target_dtype=torch.float32,
            )

    def test_generated_targets_are_contiguous_without_normalizing_retained_fields(self) -> None:
        source = make_collection(
            (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            ),
            noncontiguous_field_ids=frozenset(
                (
                    READOUT_PHOTOELECTRONS_FIELD_ID,
                    READOUT_NOISE_WAVEFORM_FIELD_ID,
                )
            ),
        )
        result = make_atomic_buffer(source, READOUT_CHARGE_FIELD_ID)
        self.assertTrue(result.tensor(READOUT_CHARGE_FIELD_ID).is_contiguous())
        for retained in (
            READOUT_PHOTOELECTRONS_FIELD_ID,
            READOUT_NOISE_WAVEFORM_FIELD_ID,
        ):
            self.assertIs(result.field(retained), source.field(retained))
            self.assertFalse(result.tensor(retained).is_contiguous())

    def test_every_atomic_target_is_fresh_for_noncontiguous_and_expanded_sources(
        self,
    ) -> None:
        for storage_kind in ("noncontiguous", "expanded"):
            source = make_distinctive_collection(storage_kind=storage_kind)
            if storage_kind == "noncontiguous":
                self.assertTrue(
                    all(
                        not field.tensor.is_contiguous()
                        for field in source.fields.values()
                    )
                )
            else:
                self.assertTrue(
                    all(
                        field.tensor.stride()[0] == 0
                        for field in source.fields.values()
                    )
                )
            source_storage = {
                storage_pointer(field.tensor) for field in source.fields.values()
            }
            source_values = {
                field_id: field.tensor.clone()
                for field_id, field in source.fields.items()
            }
            for target in READOUT_FIELD_IDS.ids:
                with self.subTest(storage_kind=storage_kind, target=target):
                    result = make_atomic_buffer(source, target)
                    target_tensor = result.tensor(target)
                    self.assertTrue(target_tensor.is_contiguous())
                    self.assertNotIn(storage_pointer(target_tensor), source_storage)
                    for field_id, field in source.fields.items():
                        self.assertTrue(
                            torch.equal(field.tensor, source_values[field_id])
                        )

    def test_atomic_digitized_target_installs_exact_spec(self) -> None:
        source = make_collection(
            (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
            )
        )
        spec = make_digitized_spec(bit_depth=8)
        result = build_readout_result_buffer(
            source,
            target_field_id=READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            target_dtype=torch.int32,
            digitized_waveform_spec=spec,
        )
        self.assertIs(result.digitized_waveform_spec, spec)
        with self.assertRaises(ValueError):
            build_readout_result_buffer(
                source,
                target_field_id=READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
                target_dtype=torch.int32,
            )
        with self.assertRaises(ValueError):
            build_readout_result_buffer(
                source,
                target_field_id=READOUT_ANALOG_WAVEFORM_FIELD_ID,
                target_dtype=torch.float32,
                digitized_waveform_spec=spec,
            )

    def test_full_output_has_exact_required_and_optional_schema(self) -> None:
        source = make_collection((READOUT_PHOTOELECTRONS_FIELD_ID,))
        without_digitized = build_readout_output_buffer(
            source,
            floating_dtype=torch.float32,
            replace_photoelectrons=False,
        )
        self.assertEqual(
            tuple(without_digitized.fields),
            (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
            ),
        )
        spec = make_digitized_spec()
        with_digitized = build_readout_output_buffer(
            source,
            floating_dtype=torch.float64,
            replace_photoelectrons=True,
            digitized_waveform_spec=spec,
        )
        self.assertEqual(
            tuple(with_digitized.fields),
            (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            ),
        )
        expected_dtypes = {
            READOUT_PHOTOELECTRONS_FIELD_ID: torch.int64,
            READOUT_CHARGE_FIELD_ID: torch.float64,
            READOUT_PURE_WAVEFORM_FIELD_ID: torch.float64,
            READOUT_NOISE_WAVEFORM_FIELD_ID: torch.float64,
            READOUT_ANALOG_WAVEFORM_FIELD_ID: torch.float64,
            READOUT_DIGITIZED_WAVEFORM_FIELD_ID: torch.int32,
        }
        for field_id, dtype in expected_dtypes.items():
            self.assertEqual(with_digitized.tensor(field_id).dtype, dtype)
            self.assertEqual(with_digitized.field(field_id).metadata, {})
        self.assertIs(with_digitized.digitized_waveform_spec, spec)
        with self.assertRaises(ValueError):
            build_readout_output_buffer(
                make_collection((READOUT_NOISE_WAVEFORM_FIELD_ID,)),
                floating_dtype=torch.float32,
                replace_photoelectrons=False,
            )
        alternate_source = make_collection(
            (READOUT_PHOTOELECTRONS_FIELD_ID,),
            layout=make_layout(ALTERNATE_AXIS_ORDER),
        )
        alternate_output = build_readout_output_buffer(
            alternate_source,
            floating_dtype=torch.float32,
            replace_photoelectrons=True,
        )
        self.assertEqual(alternate_output.layout, alternate_source.layout)
        self.assertEqual(
            alternate_output.tensor(READOUT_CHARGE_FIELD_ID).shape,
            alternate_source.tensor(READOUT_PHOTOELECTRONS_FIELD_ID).shape,
        )
        self.assertEqual(alternate_output.sample_dimension, 0)

    def test_full_output_ownership_for_both_replacement_modes_and_source_strides(
        self,
    ) -> None:
        for storage_kind in ("contiguous", "noncontiguous", "expanded"):
            source = make_distinctive_collection(storage_kind=storage_kind)
            source_records = dict(source.fields)
            source_values = {
                field_id: field.tensor.clone()
                for field_id, field in source.fields.items()
            }
            source_storage = {
                storage_pointer(field.tensor) for field in source.fields.values()
            }
            for replace_photoelectrons in (False, True):
                with self.subTest(
                    storage_kind=storage_kind,
                    replace_photoelectrons=replace_photoelectrons,
                ):
                    result = build_readout_output_buffer(
                        source,
                        floating_dtype=torch.float64,
                        replace_photoelectrons=replace_photoelectrons,
                        digitized_waveform_spec=make_digitized_spec(),
                    )
                    generated_ids = set(result.fields)
                    if replace_photoelectrons:
                        self.assertIsNot(
                            result.field(READOUT_PHOTOELECTRONS_FIELD_ID),
                            source_records[READOUT_PHOTOELECTRONS_FIELD_ID],
                        )
                    else:
                        self.assertIs(
                            result.field(READOUT_PHOTOELECTRONS_FIELD_ID),
                            source_records[READOUT_PHOTOELECTRONS_FIELD_ID],
                        )
                        generated_ids.remove(READOUT_PHOTOELECTRONS_FIELD_ID)

                    generated_storage = []
                    for field_id in generated_ids:
                        field = result.field(field_id)
                        self.assertIsNot(field, source_records[field_id])
                        self.assertTrue(field.tensor.is_contiguous())
                        pointer = storage_pointer(field.tensor)
                        self.assertNotIn(pointer, source_storage)
                        generated_storage.append(pointer)
                    self.assertEqual(
                        len(generated_storage), len(set(generated_storage))
                    )
                    for field_id, field in source.fields.items():
                        self.assertIs(field, source_records[field_id])
                        self.assertTrue(
                            torch.equal(field.tensor, source_values[field_id])
                        )

    def test_full_generated_fields_are_distinct_zero_initialized_storage(self) -> None:
        source = make_distinctive_collection()
        result = build_readout_output_buffer(
            source,
            floating_dtype=torch.float64,
            replace_photoelectrons=True,
            digitized_waveform_spec=make_digitized_spec(),
        )
        source_storage = {
            storage_pointer(field.tensor) for field in source.fields.values()
        }
        pointers = []
        for field in result.fields.values():
            self.assertEqual(torch.count_nonzero(field.tensor).item(), 0)
            self.assertTrue(field.tensor.is_contiguous())
            pointer = storage_pointer(field.tensor)
            self.assertNotIn(pointer, source_storage)
            pointers.append(pointer)
        self.assertEqual(len(pointers), len(set(pointers)))

    def test_generated_writable_storage_is_internally_nonoverlapping(self) -> None:
        source = make_collection((READOUT_PHOTOELECTRONS_FIELD_ID,))
        result = build_readout_output_buffer(
            source,
            floating_dtype=torch.float32,
            replace_photoelectrons=True,
            digitized_waveform_spec=make_digitized_spec(),
        )
        for tensor in (field.tensor for field in result.fields.values()):
            self.assertTrue(tensor.is_contiguous())
            self.assertEqual(tensor.numel() * tensor.element_size(), tensor.untyped_storage().nbytes())

    def test_outputs_preserve_collection_metadata_and_use_empty_new_field_metadata(self) -> None:
        source = make_collection()
        atomic = make_atomic_buffer(source, READOUT_CHARGE_FIELD_ID)
        full = build_readout_output_buffer(
            source,
            floating_dtype=torch.float32,
            replace_photoelectrons=False,
        )
        self.assertEqual(atomic.metadata, source.metadata)
        self.assertEqual(full.metadata, source.metadata)
        self.assertIs(atomic.sample_grid, source.sample_grid)
        self.assertIs(full.sample_grid, source.sample_grid)
        self.assertEqual(atomic.field(READOUT_CHARGE_FIELD_ID).metadata, {})
        self.assertEqual(full.field(READOUT_CHARGE_FIELD_ID).metadata, {})
        self.assertEqual(
            full.field(READOUT_PHOTOELECTRONS_FIELD_ID).metadata,
            source.field(READOUT_PHOTOELECTRONS_FIELD_ID).metadata,
        )

    def test_preparation_never_mutates_source(self) -> None:
        source = make_distinctive_collection()
        original_fields = dict(source.fields)
        original_values = {
            field_id: field.tensor.clone()
            for field_id, field in source.fields.items()
        }
        make_atomic_buffer(source, READOUT_CHARGE_FIELD_ID)
        build_readout_output_buffer(
            source,
            floating_dtype=torch.float32,
            replace_photoelectrons=True,
            digitized_waveform_spec=make_digitized_spec(),
        )
        self.assertEqual(tuple(source.fields), tuple(original_fields))
        for field_id, field in source.fields.items():
            self.assertIs(field, original_fields[field_id])
            self.assertTrue(torch.equal(field.tensor, original_values[field_id]))

    def test_preparation_rejects_invalid_role_inputs(self) -> None:
        source = make_collection()
        with self.assertRaises(ValueError):
            build_readout_result_buffer(
                source,
                target_field_id=TensorFieldId("readout.unknown"),
                target_dtype=torch.float32,
            )
        with self.assertRaises(ValueError):
            build_readout_result_buffer(
                source,
                target_field_id=READOUT_PHOTOELECTRONS_FIELD_ID,
                target_dtype=torch.int32,
            )
        with self.assertRaises(TypeError):
            build_readout_output_buffer(
                source,
                floating_dtype=torch.float32,
                replace_photoelectrons=1,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            build_readout_output_buffer(
                source,
                floating_dtype=torch.float16,
                replace_photoelectrons=False,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_output_preparation(self) -> None:
        source = make_collection(device=torch.device("cuda"))
        result = build_readout_output_buffer(
            source,
            floating_dtype=torch.float32,
            replace_photoelectrons=True,
            digitized_waveform_spec=make_digitized_spec(),
        )
        self.assertTrue(all(field.tensor.device.type == "cuda" for field in result.fields.values()))


if __name__ == "__main__":
    unittest.main()
