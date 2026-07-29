import unittest

import torch
from tensor_core import (
    CountCoordinates,
    LabelCoordinates,
    NonnegativeInteger,
    OffsetAxis,
    OffsetCoordinates,
    RegularCoordinates,
    Threefry4x32,
)

from tensor_dslab import (
    Afterpulse,
    AfterpulseSpec,
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    ChannelAxis,
    DelayedCrosstalk,
    DelayedCrosstalkSpec,
    DirectCrosstalk,
    DirectCrosstalkSpec,
    ExampleAxis,
    Photoelectrons,
    PhotoelectronsSpec,
    TimeAxis,
    unit_registry,
)


class CorrelatedAvalancheTests(unittest.TestCase):
    def test_delayed_offsets_are_strictly_positive(self) -> None:
        axis = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(1, 2)), relative_to=TimeAxis
        )
        spec = DelayedCrosstalkSpec(
            conditioning_axes=(),
            operation_axes=(axis,),
            device=torch.device("cpu"),
            dtype=torch.float64,
            unit=unit_registry.Unit(""),
        )
        DelayedCrosstalk(
            tensor=torch.tensor([0.1, 0.2], dtype=torch.float64), spec=spec
        )
        bad_axis = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(0, 1)), relative_to=TimeAxis
        )
        bad_spec = DelayedCrosstalkSpec(
            conditioning_axes=(),
            operation_axes=(bad_axis,),
            device=torch.device("cpu"),
            dtype=torch.float64,
            unit=unit_registry.Unit(""),
        )
        with self.assertRaises(ValueError):
            DelayedCrosstalk(
                tensor=torch.tensor([0.1, 0.2], dtype=torch.float64),
                spec=bad_spec,
            )
        empty_spec = DelayedCrosstalkSpec(
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(
                    coordinates=OffsetCoordinates(offsets=()),
                    relative_to=TimeAxis,
                ),
            ),
            device=torch.device("cpu"),
            dtype=torch.float64,
            unit=unit_registry.Unit(""),
        )
        with self.assertRaises(ValueError):
            DelayedCrosstalk(
                tensor=torch.empty((0,), dtype=torch.float64),
                spec=empty_spec,
            )

    def test_fixed_generation_afterpulse_law_and_window_discard(self) -> None:
        example = ExampleAxis(coordinates=CountCoordinates(count=30000))
        time = TimeAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=4),
            coordinate_scale=1.0,
            unit=unit_registry.Unit("ns"),
        )
        axes = (example, time)
        source_spec = PhotoelectronsSpec(
            axes=axes,
            device=torch.device("cpu"),
            dtype=torch.int64,
            unit=unit_registry.Unit("avalanche"),
        )
        counts = torch.zeros(source_spec.shape, dtype=torch.int64)
        counts[:, 0] = 1
        source = Photoelectrons(tensor=counts, spec=source_spec)
        operation = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(1,)),
            relative_to=TimeAxis,
        )
        probability = 0.4
        afterpulse = Afterpulse(
            tensor=torch.tensor([probability], dtype=torch.float64),
            spec=AfterpulseSpec(
                conditioning_axes=(),
                operation_axes=(operation,),
                device=torch.device("cpu"),
                dtype=torch.float64,
                unit=unit_registry.Unit(""),
            ),
        )
        config = ChargeConfig(
            spec=ChargeSpec(
                axes=axes,
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("avalanche"),
            ),
            kernels=ChargeKernels(members=(afterpulse,)),
            correlated_avalanche_generations=NonnegativeInteger(value=2),
        )
        left = Charge.create(
            sources=(source,),
            config=config,
            rng=Threefry4x32(seed=73),
        )
        right = Charge.create(
            sources=(source,),
            config=config,
            rng=Threefry4x32(seed=73),
        )
        self.assertTrue(torch.equal(left.tensor, right.tensor))
        observed = left.tensor.to(torch.float64).mean(dim=0)
        expected = torch.tensor(
            [1.0, probability, probability**2, 0.0],
            dtype=torch.float64,
        )
        standard_error = torch.tensor(
            [
                0.0,
                (probability / 30000) ** 0.5,
                (probability**2 * (1.0 + probability) / 30000) ** 0.5,
                0.0,
            ],
            dtype=torch.float64,
        )
        self.assertTrue(
            bool(
                (
                    torch.abs(observed - expected)
                    <= 7 * standard_error + 1.0e-12
                ).all()
            )
        )

        edge_counts = torch.zeros(source_spec.shape, dtype=torch.int64)
        edge_counts[:, -1] = 1
        edge = Photoelectrons(tensor=edge_counts, spec=source_spec)
        edge_result = Charge.create(
            sources=(edge,),
            config=config,
            rng=Threefry4x32(seed=73),
        )
        self.assertTrue(
            torch.equal(edge_result.tensor, edge_counts.to(torch.float32))
        )

        channel = ChannelAxis(
            coordinates=LabelCoordinates(labels=("left", "right"))
        )
        spatial_axes = (example, channel, time.window(start_index=0, count=1))
        spatial_source_spec = PhotoelectronsSpec(
            axes=spatial_axes,
            device=torch.device("cpu"),
            dtype=torch.int64,
            unit=unit_registry.Unit("avalanche"),
        )
        spatial_counts = torch.zeros(
            spatial_source_spec.shape,
            dtype=torch.int64,
        )
        spatial_counts[:, 1, 0] = 1
        spatial_source = Photoelectrons(
            tensor=spatial_counts,
            spec=spatial_source_spec,
        )
        spatial_probability = 0.3
        direct = DirectCrosstalk(
            tensor=torch.tensor(
                [spatial_probability],
                dtype=torch.float64,
            ),
            spec=DirectCrosstalkSpec(
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(
                        coordinates=OffsetCoordinates(offsets=(-1,)),
                        relative_to=ChannelAxis,
                    ),
                ),
                device=torch.device("cpu"),
                dtype=torch.float64,
                unit=unit_registry.Unit(""),
            ),
        )
        spatial_result = Charge.create(
            sources=(spatial_source,),
            config=ChargeConfig(
                spec=ChargeSpec(
                    axes=spatial_axes,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    unit=unit_registry.Unit("avalanche"),
                ),
                kernels=ChargeKernels(members=(direct,)),
                correlated_avalanche_generations=NonnegativeInteger(value=1),
            ),
            rng=Threefry4x32(seed=91),
        )
        observed_spatial = spatial_result.tensor.to(torch.float64).mean(dim=0)
        expected_spatial = torch.tensor(
            [[spatial_probability], [1.0]],
            dtype=torch.float64,
        )
        spatial_error = (spatial_probability / 30000) ** 0.5
        self.assertTrue(
            bool(
                (
                    torch.abs(observed_spatial - expected_spatial)
                    <= 7 * spatial_error + 1.0e-12
                ).all()
            )
        )
