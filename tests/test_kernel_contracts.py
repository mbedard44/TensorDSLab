import unittest

import torch
from tensor_core import (
    CountCoordinates,
    LabelCoordinates,
    OffsetCoordinates,
)

from tensor_dslab import (
    AnalogMinimumSpec,
    AnalogWaveform,
    AnalogWaveformConfig,
    AnalogWaveformKernels,
    AnalogWaveformSpec,
    AnalogMaximum,
    AnalogMinimum,
    BitDepth,
    ChannelAxis,
    ExampleAxis,
    InputMaximum,
    InputMinimum,
    ChargeKernels,
    PulseResponse,
    WhiteNoiseRms,
    TimeAxis,
    unit_registry,
)
from tests._product_support import (
    analog_config,
    axes,
    digitized_config,
    pure_config,
)


class KernelContractTests(unittest.TestCase):
    def test_exact_typed_collection_properties(self) -> None:
        self.assertIsInstance(
            pure_config().kernels.pulse_response, PulseResponse
        )
        self.assertIsInstance(analog_config().kernels.minimum, AnalogMinimum)
        self.assertIsInstance(analog_config().kernels.maximum, AnalogMaximum)
        digitizer = digitized_config().kernels
        self.assertIsInstance(digitizer.bit_depth, BitDepth)
        self.assertIsInstance(digitizer.input_minimum, InputMinimum)
        self.assertIsInstance(digitizer.input_maximum, InputMaximum)
        with self.assertRaises(TypeError):
            ChargeKernels(members=(pure_config().kernels.pulse_response,))

    def test_preparation_reorders_conditioning_coordinates_and_dimensions(
        self,
    ) -> None:
        output_axes = axes()
        channel = ChannelAxis(
            coordinates=LabelCoordinates(labels=("b", "a"))
        )
        example = ExampleAxis(
            coordinates=OffsetCoordinates(offsets=(1, 0))
        )
        spec = AnalogMinimumSpec(
            conditioning_axes=(channel, example),
            operation_axes=(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit=unit_registry.Unit("V"),
        )
        original = AnalogMinimum(
            tensor=torch.tensor(
                [[-0.011, -0.001], [-0.010, 0.0]],
                dtype=torch.float32,
            ),
            spec=spec,
        )
        config = AnalogWaveformConfig(
            spec=AnalogWaveformSpec(
                axes=output_axes,
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("mV"),
            ),
            kernels=AnalogWaveformKernels(members=(original,)),
        )
        prepared = AnalogWaveform.prepare(
            source_specs=(analog_config().spec,),
            config=config,
        )
        aligned = prepared.kernels.minimum
        assert aligned is not None
        self.assertIsNot(aligned, original)
        self.assertEqual(
            aligned.conditioning_axes,
            (output_axes[0], output_axes[1]),
        )
        self.assertEqual(prepared._kernel_dimensions, ((0, 1), None))
        torch.testing.assert_close(
            aligned.tensor,
            torch.tensor([[0.0, -1.0], [-10.0, -11.0]]),
            rtol=0,
            atol=1.0e-6,
        )
        self.assertEqual(aligned.spec.unit, unit_registry.Unit("mV"))
        self.assertEqual(original.conditioning_axes, (channel, example))
        torch.testing.assert_close(
            original.tensor,
            torch.tensor([[-0.011, -0.001], [-0.010, 0.0]]),
            rtol=0,
            atol=0,
        )
        self.assertNotEqual(
            aligned.tensor.untyped_storage().data_ptr(),
            original.tensor.untyped_storage().data_ptr(),
        )

    def test_exact_kernel_specs_movement_and_defensive_snapshot(self) -> None:
        pulse = pure_config().kernels.pulse_response
        with self.assertRaises(TypeError):
            pulse.to(dtype=torch.int64)
        minimum = analog_config().kernels.minimum
        assert minimum is not None
        with self.assertRaises(TypeError):
            WhiteNoiseRms(tensor=minimum.tensor, spec=minimum.spec)  # type: ignore[arg-type]

        source_tensor = torch.tensor([-1.0, -0.5])
        copied = PulseResponse(tensor=source_tensor, spec=pulse.spec)
        source_tensor.zero_()
        torch.testing.assert_close(
            copied.tensor,
            torch.tensor([-1.0, -0.5]),
            rtol=0,
            atol=0,
        )
