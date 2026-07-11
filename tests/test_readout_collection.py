from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import torch
from tensor_core import (
    FiniteFloat,
    IdSequence,
    NonnegativeInteger,
    PositiveFloat,
    PositiveInteger,
    TensorAxisId,
    TensorField,
    TensorFieldId,
)

from tensor_dslab.readout import (
    AdcQuantization,
    DigitizedWaveformSpec,
    READOUT_ANALOG_WAVEFORM_FIELD_ID,
    READOUT_CHANNEL_AXIS_ID,
    READOUT_CHARGE_FIELD_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_EXAMPLE_AXIS_ID,
    READOUT_FIELD_IDS,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_PHOTOELECTRONS_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
    READOUT_REQUIRED_AXIS_IDS,
    READOUT_SAMPLE_AXIS_ID,
    ReadoutCollection,
    SampleGrid,
    require_valid_readout_collection,
)
from tests.readout_fixtures import (
    ALTERNATE_AXIS_ORDER,
    DEFAULT_AXIS_ORDER,
    OtherId,
    make_collection,
    make_digitized_spec,
    make_layout,
    make_sample_grid,
    make_tensor,
)


class ReadoutCollectionTest(unittest.TestCase):
    def test_field_constants_have_exact_values_and_canonical_order(self) -> None:
        expected = tuple(
            TensorFieldId(value)
            for value in (
                "readout.photoelectrons",
                "readout.charge",
                "readout.waveform.pure",
                "readout.waveform.noise",
                "readout.waveform.analog",
                "readout.waveform.digitized",
            )
        )
        self.assertEqual(
            (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                READOUT_CHARGE_FIELD_ID,
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
                READOUT_ANALOG_WAVEFORM_FIELD_ID,
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
            ),
            expected,
        )
        self.assertEqual(READOUT_FIELD_IDS.ids, expected)

    def test_accepts_all_sixty_three_nonempty_canonical_field_subsets(self) -> None:
        field_ids = READOUT_FIELD_IDS.ids
        accepted = 0
        for mask in range(1, 1 << len(field_ids)):
            subset = tuple(
                field_id
                for index, field_id in enumerate(field_ids)
                if mask & (1 << index)
            )
            collection = make_collection(subset)
            self.assertEqual(tuple(collection.fields), subset)
            require_valid_readout_collection(collection)
            accepted += 1
        self.assertEqual(accepted, 63)

    def test_rejects_empty_unknown_and_noncanonical_field_sets(self) -> None:
        layout = make_layout()
        shared_axes = IdSequence(tuple(axis.id for axis in layout.axes.axes))
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields={},
                shared_axes=shared_axes,
                metadata={},
                sample_grid=make_sample_grid(),
            )

        unknown = TensorFieldId("readout.unknown")
        unknown_field = TensorField(
            id=unknown,
            tensor=make_tensor(
                layout,
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
            layout=layout,
        )
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields={unknown: unknown_field},
                shared_axes=shared_axes,
                metadata={},
                sample_grid=make_sample_grid(),
            )

        with self.assertRaises(ValueError):
            make_collection(
                (
                    READOUT_PURE_WAVEFORM_FIELD_ID,
                    READOUT_PHOTOELECTRONS_FIELD_ID,
                )
            )

    def test_rejects_runtime_readout_collection_subclasses(self) -> None:
        derived_readout_collection = type(
            "DerivedReadoutCollection",
            (ReadoutCollection,),
            {},
        )
        source = make_collection((READOUT_PHOTOELECTRONS_FIELD_ID,))
        with self.assertRaises(TypeError):
            derived_readout_collection(
                fields=source.fields,
                shared_axes=source.shared_axes,
                metadata=source.metadata,
                sample_grid=source.sample_grid,
            )

    def test_axis_constants_have_exact_values_and_use_value_equality(self) -> None:
        self.assertEqual(READOUT_EXAMPLE_AXIS_ID, TensorAxisId("example"))
        self.assertEqual(READOUT_CHANNEL_AXIS_ID, TensorAxisId("channel"))
        self.assertEqual(READOUT_SAMPLE_AXIS_ID, TensorAxisId("sample"))
        self.assertIsNot(READOUT_SAMPLE_AXIS_ID, TensorAxisId("sample"))
        self.assertEqual(
            READOUT_REQUIRED_AXIS_IDS.ids,
            (
                READOUT_EXAMPLE_AXIS_ID,
                READOUT_CHANNEL_AXIS_ID,
                READOUT_SAMPLE_AXIS_ID,
            ),
        )
        collection = make_collection((READOUT_PHOTOELECTRONS_FIELD_ID,))
        self.assertEqual(collection.layout.axes.index(TensorAxisId("sample")), 2)

    def test_requires_example_channel_and_sample_axes_in_any_order(self) -> None:
        for order in (DEFAULT_AXIS_ORDER, ALTERNATE_AXIS_ORDER):
            collection = make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=make_layout(order),
            )
            self.assertEqual(
                collection.sample_dimension,
                order.index(READOUT_SAMPLE_AXIS_ID),
            )
        missing_sample_layout = make_layout(
            (READOUT_EXAMPLE_AXIS_ID, READOUT_CHANNEL_AXIS_ID)
        )
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=missing_sample_layout,
            )

    def test_requires_exact_example_and_channel_coordinate_classes(self) -> None:
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=make_layout(example_coordinate_type=OtherId),
            )
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=make_layout(channel_coordinate_type=OtherId),
            )
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=make_layout(example_count_only=True),
            )
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=make_layout(channel_count_only=True),
            )

    def test_requires_count_only_sample_axis(self) -> None:
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=make_layout(sample_id_backed=True),
            )

    def test_requires_all_layout_axes_in_shared_axes_layout_order(self) -> None:
        layout = make_layout()
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=layout,
                shared_axes=IdSequence(
                    (READOUT_EXAMPLE_AXIS_ID, READOUT_CHANNEL_AXIS_ID)
                ),
            )
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=layout,
                shared_axes=IdSequence(
                    (
                        READOUT_CHANNEL_AXIS_ID,
                        READOUT_EXAMPLE_AXIS_ID,
                        READOUT_SAMPLE_AXIS_ID,
                    )
                ),
            )
        count_extra_axis_id = TensorAxisId("count-extra")
        collection = make_collection(
            (READOUT_PHOTOELECTRONS_FIELD_ID,),
            layout=make_layout(
                (
                    READOUT_EXAMPLE_AXIS_ID,
                    count_extra_axis_id,
                    READOUT_CHANNEL_AXIS_ID,
                    READOUT_SAMPLE_AXIS_ID,
                )
            ),
        )
        self.assertEqual(
            collection.shared_axes.ids,
            (
                READOUT_EXAMPLE_AXIS_ID,
                count_extra_axis_id,
                READOUT_CHANNEL_AXIS_ID,
                READOUT_SAMPLE_AXIS_ID,
            ),
        )

    def test_requires_structurally_equal_layouts_and_one_device(self) -> None:
        layout = make_layout()
        equal_layout = make_layout()
        self.assertEqual(layout, equal_layout)
        self.assertIsNot(layout, equal_layout)
        equal_photo = TensorField(
            id=READOUT_PHOTOELECTRONS_FIELD_ID,
            tensor=torch.zeros((2, 2, 4), dtype=torch.int64),
            layout=layout,
        )
        equal_noise = TensorField(
            id=READOUT_NOISE_WAVEFORM_FIELD_ID,
            tensor=torch.zeros((2, 2, 4), dtype=torch.float32),
            layout=equal_layout,
        )
        structurally_equal = ReadoutCollection(
            fields={
                READOUT_PHOTOELECTRONS_FIELD_ID: equal_photo,
                READOUT_NOISE_WAVEFORM_FIELD_ID: equal_noise,
            },
            shared_axes=IdSequence(tuple(axis.id for axis in layout.axes.axes)),
            metadata={},
            sample_grid=make_sample_grid(),
        )
        self.assertEqual(structurally_equal.layout, equal_layout)

        reordered_layout = make_layout(
            (
                READOUT_CHANNEL_AXIS_ID,
                READOUT_EXAMPLE_AXIS_ID,
                READOUT_SAMPLE_AXIS_ID,
            )
        )
        photo = TensorField(
            id=READOUT_PHOTOELECTRONS_FIELD_ID,
            tensor=torch.zeros((2, 2, 4), dtype=torch.int64),
            layout=layout,
        )
        noise = TensorField(
            id=READOUT_NOISE_WAVEFORM_FIELD_ID,
            tensor=torch.zeros((2, 2, 4), dtype=torch.float32),
            layout=reordered_layout,
        )
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields={
                    READOUT_PHOTOELECTRONS_FIELD_ID: photo,
                    READOUT_NOISE_WAVEFORM_FIELD_ID: noise,
                },
                shared_axes=IdSequence(tuple(axis.id for axis in layout.axes.axes)),
                metadata={},
                sample_grid=make_sample_grid(),
            )

        meta_noise = TensorField(
            id=READOUT_NOISE_WAVEFORM_FIELD_ID,
            tensor=torch.zeros((2, 2, 4), dtype=torch.float32, device="meta"),
            layout=layout,
        )
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields={
                    READOUT_PHOTOELECTRONS_FIELD_ID: photo,
                    READOUT_NOISE_WAVEFORM_FIELD_ID: meta_noise,
                },
                shared_axes=IdSequence(tuple(axis.id for axis in layout.axes.axes)),
                metadata={},
                sample_grid=make_sample_grid(),
            )

    def test_accepts_noncontiguous_strided_and_rejects_sparse_layout(self) -> None:
        collection = make_collection(
            (READOUT_PHOTOELECTRONS_FIELD_ID,),
            noncontiguous_field_ids=frozenset((READOUT_PHOTOELECTRONS_FIELD_ID,)),
        )
        self.assertFalse(
            collection.tensor(READOUT_PHOTOELECTRONS_FIELD_ID).is_contiguous()
        )
        self.assertEqual(
            collection.tensor(READOUT_PHOTOELECTRONS_FIELD_ID).layout,
            torch.strided,
        )

        layout = make_layout()
        sparse_field = TensorField(
            id=READOUT_PHOTOELECTRONS_FIELD_ID,
            tensor=torch.zeros((2, 2, 4), dtype=torch.int64).to_sparse(),
            layout=layout,
        )
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields={READOUT_PHOTOELECTRONS_FIELD_ID: sparse_field},
                shared_axes=IdSequence(tuple(axis.id for axis in layout.axes.axes)),
                metadata={},
                sample_grid=make_sample_grid(),
            )

    def test_accepts_expanded_read_only_semantic_source_storage(self) -> None:
        collection = make_collection(
            (READOUT_PHOTOELECTRONS_FIELD_ID,),
            expanded_field_ids=frozenset((READOUT_PHOTOELECTRONS_FIELD_ID,)),
        )
        tensor = collection.tensor(READOUT_PHOTOELECTRONS_FIELD_ID)
        self.assertEqual(tensor.stride()[0], 0)
        self.assertEqual(tensor.layout, torch.strided)

    def test_enforces_exact_role_dtypes_and_one_float_dtype(self) -> None:
        invalid_cases = (
            (
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                {READOUT_PHOTOELECTRONS_FIELD_ID: torch.int32},
            ),
            (
                (READOUT_PURE_WAVEFORM_FIELD_ID,),
                {READOUT_PURE_WAVEFORM_FIELD_ID: torch.float16},
            ),
            (
                (READOUT_DIGITIZED_WAVEFORM_FIELD_ID,),
                {READOUT_DIGITIZED_WAVEFORM_FIELD_ID: torch.int64},
            ),
        )
        for field_ids, overrides in invalid_cases:
            with self.subTest(field_ids=field_ids), self.assertRaises(ValueError):
                make_collection(field_ids, dtype_overrides=overrides)
        with self.assertRaises(ValueError):
            make_collection(
                (
                    READOUT_PURE_WAVEFORM_FIELD_ID,
                    READOUT_NOISE_WAVEFORM_FIELD_ID,
                ),
                dtype_overrides={READOUT_NOISE_WAVEFORM_FIELD_ID: torch.float64},
            )
        make_collection(
            (
                READOUT_PURE_WAVEFORM_FIELD_ID,
                READOUT_NOISE_WAVEFORM_FIELD_ID,
            ),
            floating_dtype=torch.float64,
        )

    def test_enforces_field_value_domains(self) -> None:
        layout = make_layout()
        shape = (2, 2, 4)
        cases = (
            (
                READOUT_PHOTOELECTRONS_FIELD_ID,
                torch.full(shape, -1, dtype=torch.int64),
            ),
            (
                READOUT_FIELD_IDS.ids[1],
                torch.full(shape, -1.0, dtype=torch.float32),
            ),
            (
                READOUT_FIELD_IDS.ids[1],
                torch.full(shape, float("nan"), dtype=torch.float32),
            ),
            (
                READOUT_PURE_WAVEFORM_FIELD_ID,
                torch.full(shape, float("inf"), dtype=torch.float32),
            ),
            (
                READOUT_NOISE_WAVEFORM_FIELD_ID,
                torch.full(shape, float("nan"), dtype=torch.float32),
            ),
            (
                READOUT_FIELD_IDS.ids[4],
                torch.full(shape, float("-inf"), dtype=torch.float32),
            ),
            (
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
                torch.full(shape, -1, dtype=torch.int32),
            ),
            (
                READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
                torch.full(shape, 4096, dtype=torch.int32),
            ),
        )
        for field_id, tensor in cases:
            with self.subTest(field_id=field_id), self.assertRaises(ValueError):
                make_collection(
                    (field_id,),
                    layout=layout,
                    tensor_overrides={field_id: tensor},
                )
        with self.assertRaisesRegex(ValueError, "readable for validation"):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                device=torch.device("meta"),
            )

    def test_accepts_signed_finite_waveforms_for_both_float_dtypes(self) -> None:
        field_ids = (
            READOUT_PURE_WAVEFORM_FIELD_ID,
            READOUT_NOISE_WAVEFORM_FIELD_ID,
            READOUT_ANALOG_WAVEFORM_FIELD_ID,
        )
        for dtype in (torch.float32, torch.float64):
            tensors = {
                field_id: torch.linspace(
                    -float(index + 1),
                    float(index + 2),
                    steps=16,
                    dtype=dtype,
                ).reshape(2, 2, 4)
                for index, field_id in enumerate(field_ids)
            }
            with self.subTest(dtype=dtype):
                collection = make_collection(
                    field_ids,
                    floating_dtype=dtype,
                    tensor_overrides=tensors,
                )
                for field_id in field_ids:
                    self.assertEqual(collection.tensor(field_id).dtype, dtype)
                    self.assertTrue(
                        torch.equal(collection.tensor(field_id), tensors[field_id])
                    )
                    self.assertLess(collection.tensor(field_id).min().item(), 0.0)
                    self.assertGreater(collection.tensor(field_id).max().item(), 0.0)

    def test_accepts_inclusive_digitized_adc_boundaries(self) -> None:
        for bit_depth, adc_max in ((12, 4095), (16, 65535)):
            tensor = torch.tensor(
                [0, adc_max] * 8,
                dtype=torch.int32,
            ).reshape(2, 2, 4)
            with self.subTest(bit_depth=bit_depth):
                collection = make_collection(
                    (READOUT_DIGITIZED_WAVEFORM_FIELD_ID,),
                    tensor_overrides={READOUT_DIGITIZED_WAVEFORM_FIELD_ID: tensor},
                    digitized_waveform_spec=make_digitized_spec(bit_depth=bit_depth),
                )
                result = collection.tensor(READOUT_DIGITIZED_WAVEFORM_FIELD_ID)
                self.assertEqual(result.dtype, torch.int32)
                self.assertEqual(result.min().item(), 0)
                self.assertEqual(result.max().item(), adc_max)

    def test_requires_digitized_spec_exactly_with_digitized_field(self) -> None:
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_DIGITIZED_WAVEFORM_FIELD_ID,),
                digitized_waveform_spec=None,
            )
        with self.assertRaises(ValueError):
            make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                digitized_waveform_spec=make_digitized_spec(),
            )
        with self.assertRaises(TypeError):
            SampleGrid(
                sample_period_ns=1.0,  # type: ignore[arg-type]
                origin_ns=FiniteFloat(0.0),
                sample_offset=NonnegativeInteger(0),
            )
        with self.assertRaises(TypeError):
            DigitizedWaveformSpec(
                bit_depth=12,  # type: ignore[arg-type]
                voltage_pp_mv=PositiveFloat(1.0),
                voltage_offset_mv=FiniteFloat(0.0),
                analog_gain_db=FiniteFloat(0.0),
                quantization=AdcQuantization.TRUNCATE,
            )
        with self.assertRaises(TypeError):
            DigitizedWaveformSpec(
                bit_depth=PositiveInteger(12),
                voltage_pp_mv=PositiveFloat(1.0),
                voltage_offset_mv=FiniteFloat(0.0),
                analog_gain_db=FiniteFloat(0.0),
                quantization="truncate",  # type: ignore[arg-type]
            )

    def test_digitized_spec_requires_one_to_sixteen_bits(self) -> None:
        for bit_depth in (1, 16):
            self.assertEqual(make_digitized_spec(bit_depth=bit_depth).bit_depth.value, bit_depth)
        with self.assertRaises(ValueError):
            make_digitized_spec(bit_depth=17)
        with self.assertRaises(ValueError):
            PositiveInteger(0)

    def test_digitized_spec_derives_int32_safe_adc_bounds(self) -> None:
        self.assertEqual(make_digitized_spec(bit_depth=1).adc_min, 0)
        self.assertEqual(make_digitized_spec(bit_depth=1).adc_max, 1)
        self.assertEqual(make_digitized_spec(bit_depth=16).adc_max, 65535)

    def test_digitized_spec_requires_zero_to_forty_db_gain(self) -> None:
        for gain in (0.0, 40.0):
            self.assertEqual(make_digitized_spec(analog_gain_db=gain).analog_gain_db.value, gain)
        for gain in (-0.001, 40.001):
            with self.assertRaises(ValueError):
                make_digitized_spec(analog_gain_db=gain)

    def test_sidecars_and_tensorcore_mappings_are_frozen(self) -> None:
        collection = make_collection()
        with self.assertRaises(FrozenInstanceError):
            collection.sample_grid.origin_ns = FiniteFloat(1.0)  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            collection.digitized_waveform_spec.bit_depth = PositiveInteger(8)  # type: ignore[union-attr,misc]
        with self.assertRaises(TypeError):
            collection.fields[READOUT_PHOTOELECTRONS_FIELD_ID] = collection.field(  # type: ignore[index]
                READOUT_PHOTOELECTRONS_FIELD_ID
            )
        with self.assertRaises(TypeError):
            collection.metadata["new"] = "value"  # type: ignore[index]

    def test_dimension_properties_follow_two_axis_orders(self) -> None:
        for order in (DEFAULT_AXIS_ORDER, ALTERNATE_AXIS_ORDER):
            collection = make_collection(
                (READOUT_PHOTOELECTRONS_FIELD_ID,),
                layout=make_layout(order),
            )
            self.assertEqual(collection.example_dimension, order.index(READOUT_EXAMPLE_AXIS_ID))
            self.assertEqual(collection.channel_dimension, order.index(READOUT_CHANNEL_AXIS_ID))
            self.assertEqual(collection.sample_dimension, order.index(READOUT_SAMPLE_AXIS_ID))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_collection_construction(self) -> None:
        collection = make_collection(device=torch.device("cuda"))
        self.assertEqual(collection.device.type, "cuda")
        require_valid_readout_collection(collection)


if __name__ == "__main__":
    unittest.main()
