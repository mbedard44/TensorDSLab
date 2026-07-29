import unittest

import pint
import torch
from tensor_core import (
    CountCoordinates,
    LabelCoordinates,
    OffsetCoordinates,
    RegularCoordinates,
)

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    FrequencyAxis,
    PhotoelectronsSpec,
    TimeAxis,
    quantity,
    unit_registry,
)


class QuantityRepresentationTests(unittest.TestCase):
    def test_semantic_axes_compose_coordinates(self) -> None:
        example = ExampleAxis(coordinates=CountCoordinates(count=3))
        channel = ChannelAxis(
            coordinates=LabelCoordinates(labels=("x", "y"))
        )
        self.assertEqual(example.coordinate_at(2), 2)
        self.assertEqual(channel.coordinate_at(1), "y")

    def test_quantity_axis_scale_and_window(self) -> None:
        axis = TimeAxis(
            coordinates=RegularCoordinates(start=-2, step=1, count=4),
            coordinate_scale=0.5,
            unit=unit_registry.Unit("ns"),
        )
        self.assertEqual(axis.quantity_at(2).to("ns").magnitude, 0.0)
        window = axis.window(start_index=1, count=2)
        self.assertIs(type(window), TimeAxis)
        self.assertEqual(window.coordinate_scale, 0.5)
        self.assertEqual(window.unit, axis.unit)
        self.assertNotEqual(
            axis,
            TimeAxis(
                coordinates=axis.coordinates,
                coordinate_scale=1.0,
                unit=axis.unit,
            ),
        )
        self.assertEqual(
            axis.quantity_of(2).to("ns").magnitude,
            1.0,
        )
        for invalid in (True, 1.0):
            with self.assertRaises(TypeError):
                axis.quantity_of(invalid)  # type: ignore[arg-type]

    def test_quantity_scale_and_registry_admission_are_exact(self) -> None:
        coordinates = CountCoordinates(count=2)
        for invalid in (1, True):
            with self.assertRaises(TypeError):
                TimeAxis(
                    coordinates=coordinates,
                    coordinate_scale=invalid,  # type: ignore[arg-type]
                    unit=unit_registry.Unit("ns"),
                )
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                TimeAxis(
                    coordinates=coordinates,
                    coordinate_scale=invalid,
                    unit=unit_registry.Unit("ns"),
                )
        foreign = pint.UnitRegistry(cache_folder=None)
        with self.assertRaises(ValueError):
            TimeAxis(
                coordinates=coordinates,
                unit=foreign.Unit("ns"),
            )
        value = quantity(2.5, "ns")
        self.assertIs(value._REGISTRY, unit_registry)

    def test_quantity_axes_require_supported_integer_coordinates(self) -> None:
        for coordinates in (
            CountCoordinates(count=2),
            RegularCoordinates(start=-1, step=2, count=2),
            OffsetCoordinates(offsets=(-2, 3)),
        ):
            with self.subTest(coordinates=type(coordinates).__name__):
                time = TimeAxis(
                    coordinates=coordinates,
                    unit=unit_registry.Unit("ns"),
                )
                frequency = FrequencyAxis(
                    coordinates=coordinates,
                    unit=unit_registry.Unit("Hz"),
                )
                self.assertIs(time.coordinates, coordinates)
                self.assertIs(frequency.coordinates, coordinates)

        for axis_type, wrong_unit in (
            (TimeAxis, "Hz"),
            (FrequencyAxis, "ns"),
        ):
            with self.subTest(axis=axis_type.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "QuantityAxis.coordinates",
                ):
                    axis_type(
                        coordinates=LabelCoordinates(labels=("one", "two")),
                        unit=unit_registry.Unit(wrong_unit),
                    )  # type: ignore[arg-type]

    def test_quantity_spec_movement_preserves_semantic_subtype_and_unit(
        self,
    ) -> None:
        spec = PhotoelectronsSpec(
            axes=(ExampleAxis(coordinates=CountCoordinates(count=2)),),
            device=torch.device("cpu"),
            dtype=torch.int64,
            unit=unit_registry.Unit("avalanche"),
        )
        moved = spec.to(device=torch.device("meta"))
        self.assertIs(type(moved), PhotoelectronsSpec)
        self.assertEqual(moved.unit, spec.unit)
        self.assertEqual(moved.axes, spec.axes)

    def test_frequency_rejects_time_unit(self) -> None:
        with self.assertRaises(ValueError):
            FrequencyAxis(
                coordinates=CountCoordinates(count=2),
                unit=unit_registry.Unit("ns"),
            )
