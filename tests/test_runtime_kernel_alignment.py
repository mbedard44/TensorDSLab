"""Focused private physical-kernel alignment evidence."""

from typing import cast, final, override
import unittest
from unittest import mock

import torch
from tensor_core import LabelAxis, NonnegativeInteger, OffsetAxis

from tensor_dslab import (
    ChannelAxis,
    ChargeConfig,
    DarkCountRate,
    DirectCrosstalk,
    ExampleAxis,
    Photoelectrons,
    Pulse,
    PureWaveformConfig,
    SampleAxis,
    quantities,
    quantity,
)
from tensor_dslab.readout.charge.runtime.prepare import prepare_charge
from tensor_dslab.readout.pure_waveform.runtime.prepare import (
    prepare_pure_waveform,
)
from tensor_dslab.readout.runtime.kernel import align_quantity_kernel
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


@final
class ForeignAxis(LabelAxis):
    __slots__ = ()

    @override
    def _require(self) -> None:
        if not self.labels:
            raise ValueError("ForeignAxis must be nonempty")


def _source(
    axes: tuple,
    *,
    device: torch.device | str = "cpu",
) -> Photoelectrons:
    return Photoelectrons(
        tensor=torch.zeros(
            tuple(axis.size for axis in axes),
            dtype=torch.int64,
            device=device,
        ),
        axes=axes,
    )


class RuntimeKernelAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example_axis = ExampleAxis(count=2)
        self.channel_axis = ChannelAxis(labels=("left", "right"))
        self.sample_axis = SampleAxis(start=0, step=2000, count=4)
        self.source = _source(
            (self.example_axis, self.channel_axis, self.sample_axis)
        )

    def test_global_scalar_and_requested_dtypes(self) -> None:
        kernel = DarkCountRate(
            quantity=quantity(2.5, "kHz"),
            conditioning_axes=(),
            operation_axes=(),
        )
        source_snapshot = self.source.tensor.clone()
        kernel_snapshot = kernel.tensor.clone()

        aligned64, dimensions64 = align_quantity_kernel(
            kernel,
            field=self.source,
            dtype=torch.float64,
        )
        aligned32, dimensions32 = align_quantity_kernel(
            kernel,
            field=self.source,
            dtype=torch.float32,
        )

        self.assertEqual(dimensions64, ())
        self.assertEqual(dimensions32, ())
        self.assertEqual(aligned64.dtype, torch.float64)
        self.assertEqual(aligned32.dtype, torch.float32)
        self.assertEqual(aligned64.device, self.source.tensor.device)
        self.assertEqual(aligned32.device, self.source.tensor.device)
        self.assertTrue(aligned64.is_contiguous())
        self.assertTrue(aligned32.is_contiguous())
        self.assertFalse(aligned64.requires_grad)
        self.assertFalse(aligned32.requires_grad)
        self.assertEqual(float(aligned64), 2500.0)
        self.assertEqual(float(aligned32), 2500.0)
        self.assertNotEqual(aligned32.data_ptr(), kernel.tensor.data_ptr())
        self.assertTrue(torch.equal(self.source.tensor, source_snapshot))
        self.assertTrue(torch.equal(kernel.tensor, kernel_snapshot))

    def test_one_example_and_one_channel_conditioning(self) -> None:
        example_kernel = DarkCountRate(
            quantity=quantities((10.0, 20.0), "Hz"),
            conditioning_axes=(ExampleAxis(count=2),),
            operation_axes=(),
        )
        channel_kernel = DarkCountRate(
            quantity=quantities((30.0, 40.0), "Hz"),
            conditioning_axes=(
                ChannelAxis(labels=("right", "left")),
            ),
            operation_axes=(),
        )

        examples, example_dimensions = align_quantity_kernel(
            example_kernel,
            field=self.source,
            dtype=torch.float64,
        )
        channels, channel_dimensions = align_quantity_kernel(
            channel_kernel,
            field=self.source,
            dtype=torch.float64,
        )

        self.assertEqual(example_dimensions, (0,))
        self.assertEqual(channel_dimensions, (1,))
        self.assertEqual(examples.tolist(), [10.0, 20.0])
        self.assertEqual(channels.tolist(), [40.0, 30.0])

    def test_coordinate_and_conditioning_dimension_permutations_preserve_operations(
        self,
    ) -> None:
        source = _source(
            (self.sample_axis, self.example_axis, self.channel_axis)
        )
        operation_axes = (
            OffsetAxis(relative_to=ChannelAxis, offsets=(0, 1)),
            OffsetAxis(relative_to=SampleAxis, offsets=(0, 2, 3)),
        )
        raw = torch.arange(24, dtype=torch.float64).reshape(2, 2, 2, 3)
        kernel = DirectCrosstalk(
            quantity=quantities(raw, "dimensionless"),
            conditioning_axes=(
                ChannelAxis(labels=("right", "left")),
                ExampleAxis(count=2),
            ),
            operation_axes=operation_axes,
        )
        kernel_snapshot = kernel.tensor.clone()

        aligned, dimensions = align_quantity_kernel(
            kernel,
            field=source,
            dtype=torch.float64,
        )
        expected = raw.index_select(
            0,
            torch.tensor((1, 0), dtype=torch.int64),
        ).permute(1, 0, 2, 3)

        self.assertEqual(dimensions, (1, 2))
        self.assertEqual(aligned.shape, (2, 2, 2, 3))
        self.assertTrue(torch.equal(aligned, expected))
        self.assertEqual(kernel.operation_axes, operation_axes)
        self.assertTrue(torch.equal(kernel.tensor, kernel_snapshot))
        self.assertTrue(aligned.is_contiguous())

    def test_one_operation_axis_remains_trailing_and_unchanged(self) -> None:
        pulse = Pulse(
            quantity=quantities(
                torch.tensor(
                    (
                        (-1.0, -2.0, -3.0),
                        (-4.0, -5.0, -6.0),
                    ),
                    dtype=torch.float64,
                ),
                "mV",
            ),
            conditioning_axes=(ExampleAxis(count=2),),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0, 1, 3)),
            ),
        )
        aligned, dimensions = align_quantity_kernel(
            pulse,
            field=self.source,
            dtype=torch.float32,
        )

        self.assertEqual(dimensions, (0,))
        self.assertEqual(aligned.dtype, torch.float32)
        self.assertEqual(aligned.shape, (2, 3))
        self.assertEqual(
            aligned.tolist(),
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
        )
        self.assertEqual(pulse.operation_axes[0].offsets, (0, 1, 3))

    def test_missing_role_and_coordinate_mismatch_fail_before_conversion(
        self,
    ) -> None:
        invalid_dtype = cast(torch.dtype, object())
        missing = DarkCountRate(
            quantity=quantities((1.0, 2.0), "Hz"),
            conditioning_axes=(ForeignAxis(labels=("a", "b")),),
            operation_axes=(),
        )
        mismatched = DarkCountRate(
            quantity=quantities((1.0, 2.0), "Hz"),
            conditioning_axes=(
                ChannelAxis(labels=("left", "other")),
            ),
            operation_axes=(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "DarkCountRate.*semantic role absent",
        ):
            align_quantity_kernel(
                missing,
                field=self.source,
                dtype=invalid_dtype,
            )
        with self.assertRaisesRegex(
            ValueError,
            "DarkCountRate.*coordinates.*execution field axis",
        ):
            align_quantity_kernel(
                mismatched,
                field=self.source,
                dtype=invalid_dtype,
            )

    def test_charge_preparation_delegates_to_shared_alignment(self) -> None:
        rate = DarkCountRate(
            quantity=quantity(100.0, "Hz"),
            conditioning_axes=(),
            operation_axes=(),
        )
        config = ChargeConfig(
            correlated_avalanche_generations=NonnegativeInteger(0),
            dark_counts=rate,
        )
        sampling = SamplingRuntime(
            sample_count=4,
            sample_period_ps=2000,
            sample_dimension=2,
        )

        with mock.patch(
            "tensor_dslab.readout.charge.runtime.prepare."
            "align_quantity_kernel",
            wraps=align_quantity_kernel,
        ) as delegated:
            runtime = prepare_charge(
                config,
                photoelectrons=self.source,
                sampling=sampling,
                floating_dtype=torch.float32,
            )

        delegated.assert_called_once_with(
            rate,
            field=self.source,
            dtype=torch.float64,
        )
        self.assertIsNotNone(runtime.dark_count_mean)

    def test_pure_preparation_delegates_to_shared_alignment(self) -> None:
        pulse = Pulse(
            quantity=quantities((-1.0, -0.5), "mV"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0, 1)),
            ),
        )
        config = PureWaveformConfig(pulse=pulse)
        sampling = SamplingRuntime(
            sample_count=4,
            sample_period_ps=2000,
            sample_dimension=2,
        )

        with mock.patch(
            "tensor_dslab.readout.pure_waveform.runtime.prepare."
            "align_quantity_kernel",
            wraps=align_quantity_kernel,
        ) as delegated:
            runtime = prepare_pure_waveform(
                config,
                source=self.source,
                sampling=sampling,
                floating_dtype=torch.float32,
                device=self.source.tensor.device,
            )

        delegated.assert_called_once_with(
            pulse,
            field=self.source,
            dtype=torch.float32,
        )
        self.assertEqual(runtime.coefficients.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
