from itertools import combinations
import unittest
from unittest.mock import patch

import torch
from tensor_core import TensorField
from tensor_core.tensor.validation import require_same_dtype

from tensor_dslab import (
    AnalogWaveform,
    Charge,
    DigitizedWaveform,
    NoiseWaveform,
    Photoelectrons,
    PureWaveform,
    ReadoutCollection,
)
from tests.readout_fixtures import (
    ForeignField,
    PRODUCT_TYPES,
    make_axes,
    make_collection,
    make_product,
)


class ReadoutCollectionTest(unittest.TestCase):
    def test_accepts_every_nonempty_product_subset(self) -> None:
        accepted = 0
        for count in range(1, len(PRODUCT_TYPES) + 1):
            for subset in combinations(PRODUCT_TYPES, count):
                with self.subTest(subset=tuple(item.__name__ for item in subset)):
                    collection = make_collection(subset)
                    self.assertEqual(collection.field_types, frozenset(subset))
                    for field_type in subset:
                        self.assertIs(type(collection.field(field_type)), field_type)
                    accepted += 1
        self.assertEqual(accepted, 63)

    def test_schema_is_one_unordered_class_owned_frozenset(self) -> None:
        self.assertEqual(
            ReadoutCollection.accepted_field_types(),
            frozenset(
                {
                    Photoelectrons,
                    Charge,
                    PureWaveform,
                    NoiseWaveform,
                    AnalogWaveform,
                    DigitizedWaveform,
                }
            ),
        )
        self.assertIs(type(ReadoutCollection.accepted_field_types()), frozenset)

    def test_rejects_empty_foreign_and_duplicate_membership(self) -> None:
        with self.assertRaises(ValueError):
            ReadoutCollection(fields=())

        axes = make_axes()
        foreign = ForeignField(
            tensor=torch.zeros(tuple(axis.size for axis in axes)),
            axes=axes,
        )
        with self.assertRaises(ValueError):
            ReadoutCollection(fields=(foreign,))

        photoelectrons = make_product(Photoelectrons, axes=axes)
        with self.assertRaises(ValueError):
            ReadoutCollection(fields=(photoelectrons, photoelectrons))
        with self.assertRaises(TypeError):
            ReadoutCollection(fields=(object(),))  # type: ignore[arg-type]

    def test_membership_order_has_no_semantic_lookup_effect(self) -> None:
        axes = make_axes()
        charge = make_product(Charge, axes=axes)
        noise = make_product(NoiseWaveform, axes=axes)
        forward = ReadoutCollection(fields=(charge, noise))
        reverse = ReadoutCollection(fields=(noise, charge))
        self.assertEqual(forward.field_types, reverse.field_types)
        self.assertIs(forward.field(Charge), reverse.field(Charge))
        self.assertIs(forward.field(NoiseWaveform), reverse.field(NoiseWaveform))

    def test_requires_equal_ordered_axes(self) -> None:
        axes = make_axes()
        equal_but_distinct_axes = make_axes()
        photoelectrons = make_product(Photoelectrons, axes=axes)
        noise = make_product(NoiseWaveform, axes=equal_but_distinct_axes)
        accepted = ReadoutCollection(fields=(photoelectrons, noise))
        self.assertIs(accepted.field(Photoelectrons).axes, axes)
        self.assertIs(accepted.field(NoiseWaveform).axes, equal_but_distinct_axes)

        mismatched_coordinates = make_axes(
            channel_labels=("other-0", "other-1")
        )
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields=(
                    photoelectrons,
                    make_product(NoiseWaveform, axes=mismatched_coordinates),
                )
            )

        reordered = (axes[2], axes[0], axes[1])
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields=(
                    photoelectrons,
                    make_product(NoiseWaveform, axes=reordered),
                )
            )

    def test_requires_one_device_without_moving_fields(self) -> None:
        axes = make_axes()
        photoelectrons = make_product(Photoelectrons, axes=axes, device="cpu")
        meta_noise = make_product(NoiseWaveform, axes=axes, device="meta")
        with self.assertRaises(ValueError):
            ReadoutCollection(fields=(photoelectrons, meta_noise))
        self.assertEqual(photoelectrons.tensor.device.type, "cpu")
        self.assertEqual(meta_noise.tensor.device.type, "meta")

    def test_requires_one_common_floating_dtype(self) -> None:
        axes = make_axes()
        with self.assertRaises(ValueError):
            ReadoutCollection(
                fields=(
                    make_product(PureWaveform, axes=axes, dtype=torch.float32),
                    make_product(NoiseWaveform, axes=axes, dtype=torch.float64),
                )
            )

        mixed_roles = ReadoutCollection(
            fields=(
                make_product(Photoelectrons, axes=axes),
                make_product(Charge, axes=axes, dtype=torch.float64),
                make_product(DigitizedWaveform, axes=axes),
            )
        )
        self.assertEqual(mixed_roles.tensor(Photoelectrons).dtype, torch.int64)
        self.assertEqual(mixed_roles.tensor(Charge).dtype, torch.float64)
        self.assertEqual(mixed_roles.tensor(DigitizedWaveform).dtype, torch.int32)

    def test_delegates_dtype_check_for_only_present_floating_fields(self) -> None:
        axes = make_axes()
        photoelectrons = make_product(Photoelectrons, axes=axes)
        charge = make_product(Charge, axes=axes, dtype=torch.float64)
        digitized = make_product(DigitizedWaveform, axes=axes)
        pure = make_product(PureWaveform, axes=axes, dtype=torch.float64)
        noise = make_product(NoiseWaveform, axes=axes, dtype=torch.float64)
        analog = make_product(AnalogWaveform, axes=axes, dtype=torch.float64)

        with patch(
            "tensor_dslab.readout.collection.require_same_dtype",
            wraps=require_same_dtype,
        ) as delegated:
            collection = ReadoutCollection(
                fields=(
                    photoelectrons,
                    charge,
                    digitized,
                    pure,
                    noise,
                    analog,
                )
            )

        delegated.assert_called_once_with(charge, pure, noise, analog)
        self.assertIs(collection.field(Photoelectrons), photoelectrons)
        self.assertIs(collection.field(DigitizedWaveform), digitized)

    def test_retains_exact_field_and_tensor_references(self) -> None:
        axes = make_axes()
        charge = make_product(Charge, axes=axes)
        analog = make_product(AnalogWaveform, axes=axes)
        collection = ReadoutCollection(fields=(charge, analog))
        self.assertIs(collection.field(Charge), charge)
        self.assertIs(collection.field(AnalogWaveform), analog)
        self.assertIs(collection.tensor(Charge), charge.tensor)
        self.assertIs(collection.tensor(AnalogWaveform), analog.tensor)

    def test_exact_typed_lookup_and_missing_behavior(self) -> None:
        collection = make_collection((Photoelectrons, Charge))
        self.assertIs(type(collection.field(Photoelectrons)), Photoelectrons)
        self.assertIs(type(collection.field(Charge)), Charge)
        with self.assertRaises(KeyError):
            collection.field(PureWaveform)

    def test_membership_is_immutable_and_equality_is_identity(self) -> None:
        first = make_collection((Photoelectrons,))
        second = make_collection((Photoelectrons,))
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        with self.assertRaises(TypeError):
            first.fields[Charge] = make_product(Charge)  # type: ignore[index]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_collection_constructs_on_cuda_without_movement(self) -> None:
        collection = make_collection(device="cuda")
        self.assertEqual(
            {field.tensor.device.type for field in collection.fields.values()},
            {"cuda"},
        )


if __name__ == "__main__":
    unittest.main()
