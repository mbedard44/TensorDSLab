from __future__ import annotations

import unittest

import torch
from tensor_core import TensorAxis, TensorField

from tensor_dslab.common import ChannelAxis, ExampleAxis, SampleAxis
from tensor_dslab.readout import (
    AnalogWaveform,
    Charge,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    NoiseWaveform,
    Photoelectrons,
    PureWaveform,
)
from tensor_dslab.readout._requirements import (
    _require_dtype,
    _require_exact,
    _require_floating_dtype,
    _require_one_of_exact,
    _require_optional_exact,
)
from tensor_dslab.readout.analog_waveform.types import (
    _require_valid_values as require_valid_analog,
)
from tensor_dslab.readout.charge.types import (
    _require_valid_values as require_valid_charge,
)
from tensor_dslab.readout.digitized_waveform.types import (
    _require_valid_values as require_valid_digitized,
)
from tensor_dslab.readout.noise_waveform.types import (
    _require_valid_values as require_valid_noise,
)
from tensor_dslab.readout.photoelectrons.types import (
    _require_valid_values as require_valid_photoelectrons,
)
from tensor_dslab.readout.pure_waveform.types import (
    _require_valid_values as require_valid_pure,
)
from tensor_core import FiniteFloat, NonnegativeFloat, PositiveInteger
from tests.readout_fixtures import (
    ALTERNATE_AXIS_ORDER,
    DEFAULT_AXIS_ORDER,
    FLOATING_PRODUCT_TYPES,
    OtherAxis,
    PRODUCT_TYPES,
    make_axes,
    make_product,
    make_tensor,
)


class ReadoutProductTypesTest(unittest.TestCase):
    def test_every_product_accepts_both_representative_axis_orders(self) -> None:
        for order in (DEFAULT_AXIS_ORDER, ALTERNATE_AXIS_ORDER):
            axes = make_axes(order)
            for field_type in PRODUCT_TYPES:
                with self.subTest(order=order, field=field_type.__name__):
                    field = make_product(field_type, axes=axes)
                    self.assertEqual(field.axes, axes)
                    self.assertEqual(field.shape, tuple(axis.size for axis in axes))
                    self.assertEqual(field.dimension_of(SampleAxis), order.index(SampleAxis))
                    self.assertIs(field.axis(SampleAxis), axes[order.index(SampleAxis)])

    def test_products_require_exact_three_readout_axes(self) -> None:
        invalid_orders = (
            (ExampleAxis, ChannelAxis),
            (ExampleAxis, ChannelAxis, OtherAxis),
            (ExampleAxis, ChannelAxis, SampleAxis, OtherAxis),
        )
        for order in invalid_orders:
            axes = make_axes(order)
            tensor = torch.zeros(tuple(axis.size for axis in axes), dtype=torch.int64)
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    Photoelectrons(tensor=tensor, axes=axes)

        duplicate_axes = (
            ExampleAxis(coordinates=("e0",)),
            ExampleAxis(coordinates=("e1",)),
            SampleAxis(coordinates=("0ps", "1ps")),
        )
        with self.assertRaises(ValueError):
            Photoelectrons(
                tensor=torch.zeros((1, 1, 2), dtype=torch.int64),
                axes=duplicate_axes,
            )

    def test_tensor_shape_must_match_axis_sizes(self) -> None:
        axes = make_axes()
        with self.assertRaises(ValueError):
            Photoelectrons(
                tensor=torch.zeros((2, 2, 3), dtype=torch.int64),
                axes=axes,
            )

    def test_products_accept_noncontiguous_strided_tensor_without_copy(self) -> None:
        axes = make_axes()
        for field_type in PRODUCT_TYPES:
            with self.subTest(field=field_type.__name__):
                field = make_product(field_type, axes=axes, noncontiguous=True)
                self.assertEqual(field.tensor.layout, torch.strided)
                self.assertFalse(field.tensor.is_contiguous())

        tensor = make_tensor(axes, dtype=torch.int64, noncontiguous=True)
        field = Photoelectrons(tensor=tensor, axes=axes)
        self.assertIs(field.tensor, tensor)
        self.assertIs(field.axes, axes)

    def test_products_reject_sparse_tensor_layout(self) -> None:
        axes = make_axes()
        indices = torch.empty((3, 0), dtype=torch.int64)
        values = torch.empty((0,), dtype=torch.float32)
        with torch.sparse.check_sparse_tensor_invariants():
            sparse = torch.sparse_coo_tensor(
                indices,
                values,
                size=tuple(axis.size for axis in axes),
            )
        with self.assertRaises(ValueError):
            PureWaveform(tensor=sparse, axes=axes)

    def test_photoelectrons_and_digitized_require_exact_integer_dtypes(self) -> None:
        axes = make_axes()
        for dtype in (torch.int32, torch.float32, torch.float64):
            with self.subTest(field="Photoelectrons", dtype=dtype):
                with self.assertRaises(ValueError):
                    make_product(Photoelectrons, axes=axes, dtype=dtype)
        for dtype in (torch.int64, torch.float32, torch.float64):
            with self.subTest(field="DigitizedWaveform", dtype=dtype):
                with self.assertRaises(ValueError):
                    make_product(DigitizedWaveform, axes=axes, dtype=dtype)

    def test_floating_products_accept_only_float32_or_float64(self) -> None:
        axes = make_axes()
        for field_type in FLOATING_PRODUCT_TYPES:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(field=field_type.__name__, dtype=dtype):
                    self.assertEqual(
                        make_product(field_type, axes=axes, dtype=dtype).tensor.dtype,
                        dtype,
                    )
            for dtype in (
                torch.float16,
                torch.bfloat16,
                torch.int32,
                torch.int64,
            ):
                with self.subTest(field=field_type.__name__, dtype=dtype):
                    with self.assertRaises(ValueError):
                        make_product(field_type, axes=axes, dtype=dtype)

    def test_constructor_does_not_scan_scientific_values(self) -> None:
        axes = make_axes(
            example_coordinates=("e0",),
            channel_coordinates=("c0",),
            sample_coordinates=("0ps", "1ps"),
        )
        tensor = torch.tensor([[[-1, 0]]], dtype=torch.int64)
        field = Photoelectrons(tensor=tensor, axes=axes)
        self.assertIs(field.tensor, tensor)
        with self.assertRaises(ValueError):
            require_valid_photoelectrons(field)

    def test_product_deep_validators_accept_valid_values(self) -> None:
        self.assertIsNone(require_valid_photoelectrons(make_product(Photoelectrons)))
        self.assertIsNone(require_valid_charge(make_product(Charge)))
        self.assertIsNone(require_valid_pure(make_product(PureWaveform)))
        self.assertIsNone(require_valid_noise(make_product(NoiseWaveform)))
        self.assertIsNone(require_valid_analog(make_product(AnalogWaveform)))

        digitized = make_product(DigitizedWaveform)
        config = DigitizedWaveformConfig(
            bit_depth=PositiveInteger(12),
            input_min_mv=FiniteFloat(-1000.0),
            input_max_mv=FiniteFloat(1000.0),
            analog_gain_db=NonnegativeFloat(0.0),
        )
        self.assertIsNone(require_valid_digitized(digitized, config))

    def test_product_deep_validators_reject_invalid_values(self) -> None:
        axes = make_axes(
            example_coordinates=("e0",),
            channel_coordinates=("c0",),
            sample_coordinates=("0ps", "1ps"),
        )
        with self.assertRaises(ValueError):
            require_valid_charge(
                Charge(
                    tensor=torch.tensor([[[0.0, -1.0]]]),
                    axes=axes,
                )
            )
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(field=Charge.__name__, value=bad):
                with self.assertRaises(ValueError):
                    require_valid_charge(
                        Charge(tensor=torch.tensor([[[0.0, bad]]]), axes=axes)
                    )
            with self.subTest(field=PureWaveform.__name__, value=bad):
                with self.assertRaises(ValueError):
                    require_valid_pure(
                        PureWaveform(
                            tensor=torch.tensor([[[0.0, bad]]]), axes=axes
                        )
                    )
            with self.subTest(field=NoiseWaveform.__name__, value=bad):
                with self.assertRaises(ValueError):
                    require_valid_noise(
                        NoiseWaveform(
                            tensor=torch.tensor([[[0.0, bad]]]), axes=axes
                        )
                    )
            with self.subTest(field=AnalogWaveform.__name__, value=bad):
                with self.assertRaises(ValueError):
                    require_valid_analog(
                        AnalogWaveform(
                            tensor=torch.tensor([[[0.0, bad]]]), axes=axes
                        )
                    )

        config = DigitizedWaveformConfig(
            bit_depth=PositiveInteger(2),
            input_min_mv=FiniteFloat(-1.0),
            input_max_mv=FiniteFloat(1.0),
            analog_gain_db=NonnegativeFloat(0.0),
        )
        for values in ((-1, 0), (0, 4)):
            with self.subTest(values=values):
                field = DigitizedWaveform(
                    tensor=torch.tensor([[list(values)]], dtype=torch.int32),
                    axes=axes,
                )
                with self.assertRaises(ValueError):
                    require_valid_digitized(field, config)

    def test_shared_private_requirements_have_exact_relationship_behavior(self) -> None:
        photoelectrons = make_product(Photoelectrons)
        _require_dtype(photoelectrons, torch.int64)
        with self.assertRaises(ValueError):
            _require_dtype(photoelectrons, torch.int32)
        charge = make_product(Charge)
        _require_floating_dtype(charge)

        _require_exact(photoelectrons, Photoelectrons, "field")
        with self.assertRaises(TypeError):
            _require_exact(photoelectrons, Charge, "field")
        _require_optional_exact(None, Charge, "field")
        _require_optional_exact(charge, Charge, "field")
        with self.assertRaises(TypeError):
            _require_optional_exact(photoelectrons, Charge, "field")
        _require_one_of_exact(charge, (Charge, PureWaveform), "field")
        with self.assertRaises(TypeError):
            _require_one_of_exact(photoelectrons, (Charge, PureWaveform), "field")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_products_construct_on_cuda_without_movement(self) -> None:
        axes = make_axes()
        for field_type in PRODUCT_TYPES:
            with self.subTest(field=field_type.__name__):
                field = make_product(field_type, axes=axes, device="cuda")
                self.assertEqual(field.tensor.device.type, "cuda")
                self.assertIs(field.axes, axes)


if __name__ == "__main__":
    unittest.main()
