import unittest

import torch
from tensor_core import (
    CountCoordinates,
    NonnegativeInteger,
    OffsetAxis,
    OffsetCoordinates,
    RegularCoordinates,
    Threefry4x32,
)

from tensor_dslab import (
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    ExampleAxis,
    Photoelectrons,
    PhotoelectronsSpec,
    TimeAxis,
    TimingJitter,
    TimingJitterSpec,
    unit_registry,
)


class TimingJitterTests(unittest.TestCase):
    def test_complete_probability_law(self) -> None:
        axis = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(-1, 0, 1)),
            relative_to=TimeAxis,
        )
        spec = TimingJitterSpec(
            conditioning_axes=(),
            operation_axes=(axis,),
            device=torch.device("cpu"),
            dtype=torch.float64,
            unit=unit_registry.Unit(""),
        )
        kernel = TimingJitter(
            tensor=torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64),
            spec=spec,
        )
        self.assertEqual(kernel.operation_shape, (3,))

    def test_incomplete_probability_law_rejected(self) -> None:
        axis = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(0, 1)), relative_to=TimeAxis
        )
        spec = TimingJitterSpec(
            conditioning_axes=(),
            operation_axes=(axis,),
            device=torch.device("cpu"),
            dtype=torch.float64,
            unit=unit_registry.Unit(""),
        )
        with self.assertRaises(ValueError):
            TimingJitter(
                tensor=torch.tensor([0.2, 0.2], dtype=torch.float64), spec=spec
            )

    def test_execution_matches_complete_multinomial_law_and_replays(self) -> None:
        example = ExampleAxis(coordinates=CountCoordinates(count=4000))
        time = TimeAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=3),
            coordinate_scale=1.0,
            unit=unit_registry.Unit("ns"),
        )
        source_spec = PhotoelectronsSpec(
            axes=(example, time),
            device=torch.device("cpu"),
            dtype=torch.int64,
            unit=unit_registry.Unit("avalanche"),
        )
        counts = torch.zeros(source_spec.shape, dtype=torch.int64)
        counts[:, 1] = 100
        source = Photoelectrons(tensor=counts, spec=source_spec)
        offset = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(-1, 0, 1)),
            relative_to=TimeAxis,
        )
        jitter = TimingJitter(
            tensor=torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64),
            spec=TimingJitterSpec(
                conditioning_axes=(),
                operation_axes=(offset,),
                device=torch.device("cpu"),
                dtype=torch.float64,
                unit=unit_registry.Unit(""),
            ),
        )
        config = ChargeConfig(
            spec=ChargeSpec(
                axes=(example, time),
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("avalanche"),
            ),
            kernels=ChargeKernels(members=(jitter,)),
            correlated_avalanche_generations=NonnegativeInteger(value=0),
        )
        left = Charge.create(
            sources=(source,),
            config=config,
            rng=Threefry4x32(seed=41),
        )
        right = Charge.create(
            sources=(source,),
            config=config,
            rng=Threefry4x32(seed=41),
        )
        self.assertTrue(torch.equal(left.tensor, right.tensor))
        self.assertTrue(bool((left.tensor.sum(dim=1) == 100).all()))
        observed = left.tensor.to(torch.float64).mean(dim=0)
        expected = torch.tensor([25.0, 50.0, 25.0], dtype=torch.float64)
        standard_error = torch.sqrt(
            100.0
            * torch.tensor([0.25, 0.5, 0.25])
            * torch.tensor([0.75, 0.5, 0.75])
            / 4000
        )
        self.assertTrue(bool((torch.abs(observed - expected) < 6 * standard_error).all()))
