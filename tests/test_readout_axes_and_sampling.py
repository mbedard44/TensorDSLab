from __future__ import annotations

from inspect import Parameter, signature
from typing import cast
import unittest
from unittest.mock import PropertyMock, patch

import torch
from tensor_core import CountAxis, LabelAxis, RegularAxis

from tensor_dslab.common import ChannelAxis, ExampleAxis, SampleAxis
from tensor_dslab.readout import Photoelectrons
from tensor_dslab.readout.runtime.sampling import prepare_sampling


class ReadoutAxesAndSamplingTest(unittest.TestCase):
    def test_axes_are_exact_final_fieldless_representation_leaves(self) -> None:
        expected = (
            (ExampleAxis, CountAxis, ("count",)),
            (ChannelAxis, LabelAxis, ("labels",)),
            (SampleAxis, RegularAxis, ("start", "step", "count")),
        )
        for axis_type, root, parameters in expected:
            with self.subTest(axis=axis_type.__name__):
                self.assertEqual(axis_type.__bases__, (root,))
                self.assertEqual(axis_type.__slots__, ())
                self.assertTrue(getattr(axis_type, "__final__", False))
                self.assertNotIn("__dataclass_fields__", axis_type.__dict__)
                self.assertNotIn("__dataclass_params__", axis_type.__dict__)
                constructor = signature(axis_type)
                self.assertEqual(tuple(constructor.parameters), parameters)
                self.assertTrue(
                    all(
                        parameter.kind is Parameter.KEYWORD_ONLY
                        for parameter in constructor.parameters.values()
                    )
                )

        axes = (
            ExampleAxis(count=2),
            ChannelAxis(labels=("channel-0", "channel-1")),
            SampleAxis(start=0, step=2_000, count=4),
        )
        self.assertTrue(all(not hasattr(axis, "__dict__") for axis in axes))

    def test_compact_axis_construction_and_coordinates_are_exact(self) -> None:
        examples = ExampleAxis(count=2)
        labels = ("tile-0", "tile-1")
        channels = ChannelAxis(labels=labels)
        samples = SampleAxis(start=0, step=2_000, count=4)

        self.assertEqual(examples.coordinates, range(2))
        self.assertIsInstance(examples.coordinates, range)
        self.assertIs(channels.labels, labels)
        self.assertIs(channels.coordinates, labels)
        self.assertEqual(samples.coordinates, range(0, 8_000, 2_000))
        self.assertIsInstance(samples.coordinates, range)
        self.assertEqual(samples.size, 4)

        large = ExampleAxis(count=1 << 40)
        self.assertIsInstance(large.coordinates, range)
        self.assertEqual(len(large.coordinates), 1 << 40)
        large_samples = SampleAxis(start=0, step=1, count=1 << 40)
        self.assertIsInstance(large_samples.coordinates, range)
        self.assertEqual(len(large_samples.coordinates), 1 << 40)

    def test_axis_lookup_types_values_and_error_categories(self) -> None:
        examples = ExampleAxis(count=2)
        channels = ChannelAxis(labels=("tile-0", "tile-1"))
        samples = SampleAxis(start=4_000, step=2_000, count=3)

        self.assertIs(type(examples.coordinate_at(1)), int)
        self.assertEqual(examples.coordinate_at(1), 1)
        self.assertIs(type(examples.index_of(1)), int)
        self.assertEqual(channels.coordinate_at(1), "tile-1")
        self.assertEqual(channels.index_of("tile-1"), 1)
        self.assertEqual(samples.coordinate_at(2), 8_000)
        self.assertEqual(samples.index_of(8_000), 2)

        for axis, bad_index in ((examples, True), (channels, 2), (samples, -1)):
            with self.subTest(axis=type(axis).__name__, index=bad_index):
                with self.assertRaises(
                    TypeError if type(bad_index) is not int else IndexError
                ):
                    axis.coordinate_at(bad_index)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            examples.index_of(True)
        with self.assertRaises(KeyError):
            examples.index_of(2)
        with self.assertRaises(TypeError):
            channels.index_of(1)  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            channels.index_of("missing")
        with self.assertRaises(TypeError):
            samples.index_of(4_000.0)  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            samples.index_of(5_000)

    def test_example_and_channel_axis_domains(self) -> None:
        with self.assertRaises(ValueError):
            ExampleAxis(count=0)
        with self.assertRaises(ValueError):
            ChannelAxis(labels=())

        for invalid in (True, 1.0, "1"):
            with self.subTest(axis="example", value=invalid):
                with self.assertRaises(TypeError):
                    ExampleAxis(count=invalid)  # type: ignore[arg-type]
        for invalid in (["channel"], ("",), ("channel", "channel"), (1,)):
            with self.subTest(axis="channel", value=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    ChannelAxis(labels=invalid)  # type: ignore[arg-type]

        class EmptyCountAxis(CountAxis):
            __slots__ = ()

            def _require(self) -> None:
                return

        self.assertEqual(EmptyCountAxis(count=0).size, 0)

    def test_sample_axis_domain_properties_and_adjacent_boundaries(self) -> None:
        axis = SampleAxis(start=4_000, step=2_000, count=3)
        self.assertEqual(axis.start_ps, 4_000)
        self.assertEqual(axis.sample_period_ps, 2_000)
        self.assertEqual(axis.stop_ps, 10_000)
        self.assertEqual(axis.coordinates, range(4_000, 10_000, 2_000))

        invalid_cases: tuple[dict[str, int], ...] = (
            {"start": -1, "step": 1, "count": 2},
            {"start": 0, "step": 0, "count": 2},
            {"start": 0, "step": -1, "count": 2},
            {"start": 0, "step": 1, "count": 1},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SampleAxis(**kwargs)
        for field in ("start", "step", "count"):
            kwargs = {"start": 0, "step": 1, "count": 2}
            kwargs[field] = cast(int, True)
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    SampleAxis(**kwargs)

        maximum = (1 << 63) - 1
        valid = SampleAxis(start=maximum - 2, step=1, count=2)
        self.assertEqual(valid.stop_ps, maximum)
        with self.assertRaises(ValueError):
            SampleAxis(start=maximum - 1, step=1, count=2)
        with self.assertRaises(ValueError):
            SampleAxis(start=0, step=(maximum // 2) + 1, count=2)

    def test_axis_equality_and_hash_include_exact_type_and_state(self) -> None:
        class OtherCountAxis(CountAxis):
            __slots__ = ()

            def _require(self) -> None:
                return

        self.assertEqual(ExampleAxis(count=2), ExampleAxis(count=2))
        self.assertEqual(
            hash(ExampleAxis(count=2)),
            hash(ExampleAxis(count=2)),
        )
        self.assertNotEqual(ExampleAxis(count=2), ExampleAxis(count=3))
        self.assertEqual(
            ChannelAxis(labels=("a", "b")),
            ChannelAxis(labels=("a", "b")),
        )
        self.assertNotEqual(
            ChannelAxis(labels=("a", "b")),
            ChannelAxis(labels=("b", "a")),
        )
        self.assertEqual(
            SampleAxis(start=0, step=2, count=3),
            SampleAxis(start=0, step=2, count=3),
        )
        self.assertNotEqual(
            SampleAxis(start=0, step=2, count=3),
            SampleAxis(start=2, step=2, count=3),
        )
        self.assertNotEqual(
            ExampleAxis(count=2),
            OtherCountAxis(count=2),
        )

    def test_sampling_runtime_is_derived_from_source_axis_and_order(self) -> None:
        for axes in (
            (
                ExampleAxis(count=2),
                ChannelAxis(labels=("a", "b")),
                SampleAxis(start=0, step=2_000, count=4),
            ),
            (
                SampleAxis(start=0, step=4_000, count=3),
                ExampleAxis(count=2),
                ChannelAxis(labels=("a", "b")),
            ),
        ):
            source = Photoelectrons(
                tensor=torch.zeros(
                    tuple(axis.size for axis in axes),
                    dtype=torch.int64,
                ),
                axes=axes,
            )
            runtime = prepare_sampling(source)
            sample_axis = source.axis(SampleAxis)
            self.assertEqual(runtime.sample_count, sample_axis.count)
            self.assertEqual(runtime.sample_period_ps, sample_axis.step)
            self.assertEqual(
                runtime.sample_dimension,
                source.dimension_of(SampleAxis),
            )
            self.assertIs(source.axis(SampleAxis), sample_axis)

    def test_sampling_preparation_never_materializes_coordinates(self) -> None:
        axes = (
            ExampleAxis(count=1),
            ChannelAxis(labels=("a",)),
            SampleAxis(start=0, step=2_000, count=4),
        )
        source = Photoelectrons(
            tensor=torch.zeros((1, 1, 4), dtype=torch.int64),
            axes=axes,
        )
        with patch.object(
            RegularAxis,
            "coordinates",
            new_callable=PropertyMock,
            side_effect=AssertionError("sampling materialized coordinates"),
        ) as coordinates:
            runtime = prepare_sampling(source)
        coordinates.assert_not_called()
        self.assertEqual(runtime.sample_count, 4)
        self.assertEqual(runtime.sample_period_ps, 2_000)
        self.assertEqual(runtime.sample_dimension, 2)

    def test_nonzero_start_axis_is_valid_but_complete_readout_rejects_it(self) -> None:
        axes = (
            ExampleAxis(count=1),
            ChannelAxis(labels=("a",)),
            SampleAxis(start=2_000, step=2_000, count=2),
        )
        source = Photoelectrons(
            tensor=torch.zeros((1, 1, 2), dtype=torch.int64),
            axes=axes,
        )
        self.assertEqual(source.axis(SampleAxis).start, 2_000)
        with self.assertRaisesRegex(ValueError, "start must be zero"):
            prepare_sampling(source)


if __name__ == "__main__":
    unittest.main()
