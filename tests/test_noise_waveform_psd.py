import unittest
from typing import override

import torch
from tensor_core import CountCoordinates, RegularCoordinates, Threefry4x32

from tensor_dslab import (
    ExampleAxis,
    FrequencyAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    NoiseWaveformKernels,
    NoiseWaveformSpec,
    PowerSpectralDensity,
    PowerSpectralDensitySpec,
    QuantityAxis,
    TimeAxis,
    unit_registry,
)


class SpectralAxis(QuantityAxis[RegularCoordinates]):
    """Represent a distinct frequency-dimensional semantic role."""

    __slots__ = ()

    @override
    def _require_quantity_axis(self) -> None:
        unit_registry.Quantity(1.0, self.unit).to("Hz")


class NoisePsdTests(unittest.TestCase):
    def test_psd_requires_exact_frequency_axis_semantics(self) -> None:
        coordinates = RegularCoordinates(start=0, step=1, count=3)
        genuine = FrequencyAxis(
            coordinates=coordinates,
            coordinate_scale=100.0,
            unit=unit_registry.Unit("MHz"),
        )
        impostor = SpectralAxis(
            coordinates=coordinates,
            coordinate_scale=100.0,
            unit=unit_registry.Unit("MHz"),
        )
        self.assertIsNot(type(impostor), type(genuine))
        self.assertIs(impostor.coordinates, genuine.coordinates)
        self.assertEqual(impostor.coordinate_scale, genuine.coordinate_scale)
        self.assertEqual(impostor.unit, genuine.unit)

        with self.assertRaisesRegex(
            TypeError,
            "requires regular FrequencyAxis",
        ):
            PowerSpectralDensitySpec(
                conditioning_axes=(),
                operation_axes=(impostor,),  # type: ignore[arg-type]
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("mV ** 2"),
            )

    def test_psd_requires_zero_dc_and_positive_non_dc(self) -> None:
        axis = FrequencyAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=3),
            coordinate_scale=100.0,
            unit=unit_registry.Unit("MHz"),
        )
        spec = PowerSpectralDensitySpec(
            conditioning_axes=(),
            operation_axes=(axis,),
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit=unit_registry.Unit("mV ** 2"),
        )
        PowerSpectralDensity(
            tensor=torch.tensor([0.0, 1.0, 2.0]), spec=spec
        )
        with self.assertRaises(ValueError):
            PowerSpectralDensity(
                tensor=torch.tensor([1.0, 1.0, 2.0]), spec=spec
            )
        empty_axis = FrequencyAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=0),
            coordinate_scale=100.0,
            unit=unit_registry.Unit("MHz"),
        )
        empty_spec = PowerSpectralDensitySpec(
            conditioning_axes=(),
            operation_axes=(empty_axis,),
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit=unit_registry.Unit("mV ** 2"),
        )
        with self.assertRaises(ValueError):
            PowerSpectralDensity(
                tensor=torch.empty((0,), dtype=torch.float32),
                spec=empty_spec,
            )

    def test_psd_execution_replays_and_requires_reciprocal_geometry(self) -> None:
        example = ExampleAxis(coordinates=CountCoordinates(count=4000))
        time = TimeAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=5),
            coordinate_scale=2.0,
            unit=unit_registry.Unit("ns"),
        )
        frequency = FrequencyAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=3),
            coordinate_scale=100.0,
            unit=unit_registry.Unit("MHz"),
        )
        psd = PowerSpectralDensity(
            tensor=torch.tensor([0.0, 1.0, 1.0]),
            spec=PowerSpectralDensitySpec(
                conditioning_axes=(),
                operation_axes=(frequency,),
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("mV ** 2"),
            ),
        )
        config = NoiseWaveformConfig(
            spec=NoiseWaveformSpec(
                axes=(example, time),
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("mV"),
            ),
            kernels=NoiseWaveformKernels(members=(psd,)),
        )
        left = NoiseWaveform.create(
            sources=(),
            config=config,
            rng=Threefry4x32(seed=19),
        )
        right = NoiseWaveform.create(
            sources=(),
            config=config,
            rng=Threefry4x32(seed=19),
        )
        self.assertTrue(torch.equal(left.tensor, right.tensor))
        torch.testing.assert_close(
            left.tensor.mean(dim=1),
            torch.zeros(example.size),
            rtol=0,
            atol=2.0e-7,
        )
        self.assertGreater(float(left.tensor.var()), 1.8)
        self.assertLess(float(left.tensor.var()), 2.2)

        bad_frequency = FrequencyAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=3),
            coordinate_scale=90.0,
            unit=unit_registry.Unit("MHz"),
        )
        bad_psd = PowerSpectralDensity(
            tensor=torch.tensor([0.0, 1.0, 1.0]),
            spec=PowerSpectralDensitySpec(
                conditioning_axes=(),
                operation_axes=(bad_frequency,),
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("mV ** 2"),
            ),
        )
        with self.assertRaises(ValueError):
            NoiseWaveform.prepare(
                source_specs=(),
                config=NoiseWaveformConfig(
                    spec=config.spec,
                    kernels=NoiseWaveformKernels(members=(bad_psd,)),
                ),
            )
