import unittest

import torch
from tensor_core import (
    TensorAxis,
    TensorField,
)

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
from tensor_dslab.readout.analog_waveform.runtime.validate import (
    validate_analog_waveform as require_valid_analog,
)
from tensor_dslab.readout.charge.runtime.prepare import ChargeRuntime
from tensor_dslab.readout.charge.runtime.validate import (
    validate_charge as require_valid_charge,
)
from tensor_dslab.readout.digitized_waveform.runtime.validate import (
    validate_digitized_waveform as require_valid_digitized,
)
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    NoiseWaveformRuntime,
    ZeroNoiseRuntime,
)
from tensor_dslab.readout.noise_waveform.runtime.validate import (
    validate_noise_waveform as require_valid_noise,
)
from tensor_dslab.readout.photoelectrons.runtime.validate import (
    validate_photoelectrons as require_valid_photoelectrons,
)
from tensor_dslab.readout.pure_waveform.runtime.validate import (
    validate_pure_waveform as require_valid_pure,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime
from tensor_core import FiniteFloat, NonnegativeFloat, PositiveInteger
from tensor_core.tensor.validation import (
    require_field_dtype,
    require_representable_float,
)
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


def _sampling_runtime(field: TensorField) -> SamplingRuntime:
    return SamplingRuntime(
        sample_count=field.axis(SampleAxis).size,
        sample_period_ps=1,
        sample_dimension=field.dimension_of(SampleAxis),
    )


def _charge_runtime(source: Photoelectrons, dtype: torch.dtype) -> ChargeRuntime:
    return ChargeRuntime(
        sampling=_sampling_runtime(source),
        floating_dtype=dtype,
        dark=None,
        timing_jitter=None,
        correlated_avalanches=None,
        smearing=None,
    )


def _noise_runtime(
    source: Photoelectrons,
    dtype: torch.dtype,
) -> NoiseWaveformRuntime:
    return NoiseWaveformRuntime(
        shape=source.shape,
        device=source.tensor.device,
        floating_dtype=dtype,
        sampling=_sampling_runtime(source),
        model=ZeroNoiseRuntime(),
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

    def test_productsrequire_exact_three_readout_axes(self) -> None:
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
            ExampleAxis(count=1),
            ExampleAxis(count=2),
            SampleAxis(start=0, step=1, count=2),
        )
        with self.assertRaises(ValueError):
            Photoelectrons(
                tensor=torch.zeros((1, 2, 2), dtype=torch.int64),
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

    def test_photoelectrons_and_digitizedrequire_exact_integer_dtypes(self) -> None:
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
            example_count=1,
            channel_labels=("c0",),
            sample_step=1,
            sample_count=2,
        )
        tensor = torch.tensor([[[-1, 0]]], dtype=torch.int64)
        field = Photoelectrons(tensor=tensor, axes=axes)
        self.assertIs(field.tensor, tensor)
        with self.assertRaises(ValueError):
            require_valid_photoelectrons(field)

    def test_product_deep_validators_accept_valid_values(self) -> None:
        photoelectrons = make_product(Photoelectrons)
        charge = make_product(Charge, axes=photoelectrons.axes)
        pure = make_product(PureWaveform, axes=photoelectrons.axes)
        noise = make_product(NoiseWaveform, axes=photoelectrons.axes)
        analog = make_product(AnalogWaveform, axes=photoelectrons.axes)
        digitized = make_product(DigitizedWaveform, axes=photoelectrons.axes)
        charge_runtime = _charge_runtime(photoelectrons, charge.tensor.dtype)
        noise_runtime = _noise_runtime(photoelectrons, noise.tensor.dtype)

        self.assertIsNone(require_valid_photoelectrons(photoelectrons))
        self.assertIsNone(
            require_valid_charge(
                charge,
                source=photoelectrons,
                runtime=charge_runtime,
            )
        )
        self.assertIsNone(require_valid_pure(pure, source=charge))
        self.assertIsNone(
            require_valid_noise(
                noise,
                source=photoelectrons,
                runtime=noise_runtime,
            )
        )
        self.assertIsNone(
            require_valid_analog(analog, pure=pure, noise=noise)
        )
        self.assertIsNone(
            require_valid_digitized(
                digitized,
                source=analog,
                maximum_code=(1 << 12) - 1,
            )
        )

    def test_product_deep_validators_reject_invalid_values(self) -> None:
        axes = make_axes(
            example_count=1,
            channel_labels=("c0",),
            sample_step=1,
            sample_count=2,
        )
        source = Photoelectrons(
            tensor=torch.zeros((1, 1, 2), dtype=torch.int64),
            axes=axes,
        )
        charge_runtime = _charge_runtime(source, torch.float32)
        with self.assertRaises(RuntimeError):
            require_valid_charge(
                Charge(
                    tensor=torch.tensor([[[0.0, -1.0]]]),
                    axes=axes,
                ),
                source=source,
                runtime=charge_runtime,
            )
        valid_charge = Charge(tensor=torch.zeros((1, 1, 2)), axes=axes)
        valid_pure = PureWaveform(tensor=torch.zeros((1, 1, 2)), axes=axes)
        valid_noise = NoiseWaveform(tensor=torch.zeros((1, 1, 2)), axes=axes)
        noise_runtime = _noise_runtime(source, torch.float32)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(field=Charge.__name__, value=bad):
                with self.assertRaises(RuntimeError):
                    require_valid_charge(
                        Charge(tensor=torch.tensor([[[0.0, bad]]]), axes=axes),
                        source=source,
                        runtime=charge_runtime,
                    )
            with self.subTest(field=PureWaveform.__name__, value=bad):
                with self.assertRaises(ValueError):
                    require_valid_pure(
                        PureWaveform(
                            tensor=torch.tensor([[[0.0, bad]]]), axes=axes
                        ),
                        source=valid_charge,
                    )
            with self.subTest(field=NoiseWaveform.__name__, value=bad):
                with self.assertRaises(ValueError):
                    require_valid_noise(
                        NoiseWaveform(
                            tensor=torch.tensor([[[0.0, bad]]]), axes=axes
                        ),
                        source=source,
                        runtime=noise_runtime,
                    )
            with self.subTest(field=AnalogWaveform.__name__, value=bad):
                with self.assertRaises(ValueError):
                    require_valid_analog(
                        AnalogWaveform(
                            tensor=torch.tensor([[[0.0, bad]]]), axes=axes
                        ),
                        pure=valid_pure,
                        noise=valid_noise,
                    )

        source_analog = AnalogWaveform(
            tensor=torch.zeros((1, 1, 2)),
            axes=axes,
        )
        for values in ((-1, 0), (0, 4)):
            with self.subTest(values=values):
                field = DigitizedWaveform(
                    tensor=torch.tensor([[list(values)]], dtype=torch.int32),
                    axes=axes,
                )
                with self.assertRaises(ValueError):
                    require_valid_digitized(
                        field,
                        source=source_analog,
                        maximum_code=3,
                    )

    def test_shared_private_requirements_have_exact_relationship_behavior(self) -> None:
        photoelectrons = make_product(Photoelectrons)
        require_field_dtype(photoelectrons, torch.int64)
        with self.assertRaises(ValueError):
            require_field_dtype(photoelectrons, torch.int32)
        charge = make_product(Charge)
        require_field_dtype(charge, torch.float32, torch.float64)

        represented = require_representable_float(
            0.1,
            dtype=torch.float32,
            field="scalar",
        )
        self.assertEqual(
            represented,
            float(torch.tensor(0.1, dtype=torch.float32)),
        )
        self.assertEqual(
            require_representable_float(
                7,
                dtype=torch.float64,
                field="scalar",
            ),
            7.0,
        )
        self.assertEqual(
            require_representable_float(
                16_777_217,
                dtype=torch.float32,
                field="scalar",
            ),
            16_777_216.0,
        )
        for dtype in (torch.float32, torch.float64):
            maximum = float(torch.finfo(dtype).max)
            with self.subTest(dtype=dtype, boundary="maximum"):
                self.assertEqual(
                    require_representable_float(
                        maximum,
                        dtype=dtype,
                        field="scalar",
                    ),
                    maximum,
                )
                self.assertEqual(
                    require_representable_float(
                        -maximum,
                        dtype=dtype,
                        field="scalar",
                    ),
                    -maximum,
                )
        for value in (-1.0, 0.0):
            with self.subTest(value=value, policy="caller-owned"):
                self.assertEqual(
                    require_representable_float(
                        value,
                        dtype=torch.float64,
                        field="scalar",
                    ),
                    value,
                )
        for malformed in (True, "1.0", None):
            with self.subTest(malformed=malformed):
                with self.assertRaises(TypeError):
                    require_representable_float(
                        malformed,  # pyright: ignore[reportArgumentType]
                        dtype=torch.float32,
                        field="scalar",
                    )
        for dtype in (torch.float16, torch.int64, "torch.float32"):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    require_representable_float(
                        1.0,
                        dtype=dtype,  # type: ignore[arg-type]
                        field="scalar",
                    )
        for value, dtype in (
            (float("inf"), torch.float64),
            (1.0e40, torch.float32),
            (10**1000, torch.float64),
        ):
            with self.subTest(value=value, dtype=dtype):
                with self.assertRaises(ValueError):
                    require_representable_float(
                        value,
                        dtype=dtype,
                        field="scalar",
                    )

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
