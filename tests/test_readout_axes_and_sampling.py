from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast
import unittest

from tensor_core import PositiveInteger

from tensor_dslab.common import (
    ChannelAxis,
    ExampleAxis,
    SampleAxis,
    SamplingConfig,
)


class ReadoutAxesAndSamplingTest(unittest.TestCase):
    def test_example_and_channel_axes_are_nonempty_exact_string_axes(self) -> None:
        examples = ExampleAxis(coordinates=("event-10", "event-11"))
        channels = ChannelAxis(coordinates=("tile-0", "tile-1"))
        self.assertEqual(examples.size, 2)
        self.assertEqual(examples.coordinate_at(1), "event-11")
        self.assertEqual(channels.index_of("tile-1"), 1)

        for axis_type in (ExampleAxis, ChannelAxis):
            with self.subTest(axis=axis_type.__name__, case="empty"):
                with self.assertRaises(ValueError):
                    axis_type(coordinates=())
            with self.subTest(axis=axis_type.__name__, case="list"):
                with self.assertRaises(TypeError):
                    axis_type(coordinates=["x"])  # type: ignore[arg-type]
            with self.subTest(axis=axis_type.__name__, case="non-string"):
                with self.assertRaises(TypeError):
                    axis_type(coordinates=(1,))  # type: ignore[arg-type]
            with self.subTest(axis=axis_type.__name__, case="empty-string"):
                with self.assertRaises(ValueError):
                    axis_type(coordinates=("",))
            with self.subTest(axis=axis_type.__name__, case="duplicate"):
                with self.assertRaises(ValueError):
                    axis_type(coordinates=("x", "x"))

    def test_sample_axis_accepts_canonical_uniform_left_edges(self) -> None:
        axis = SampleAxis(coordinates=("0ps", "2000ps", "4000ps", "6000ps"))
        self.assertEqual(axis.start_ps, 0)
        self.assertEqual(axis.sample_period_ps, 2000)
        self.assertEqual(axis.stop_ps, 8000)
        self.assertEqual(axis.size, 4)
        self.assertEqual(axis.coordinates[-1], "6000ps")
        self.assertNotIn("8000ps", axis.coordinates)

        subaxis = SampleAxis(coordinates=("4000ps", "6000ps", "8000ps"))
        self.assertEqual(subaxis.start_ps, 4000)
        self.assertEqual(subaxis.sample_period_ps, 2000)
        self.assertEqual(subaxis.stop_ps, 10000)

    def test_sample_axis_rejects_noncanonical_timestamp_grammar(self) -> None:
        bad_coordinates = (
            ("0ps",),
            ("0", "1ps"),
            ("+0ps", "+1ps"),
            ("-1ps", "0ps"),
            ("00ps", "01ps"),
            ("0PS", "1PS"),
            ("0ns", "1ns"),
            ("0.0ps", "1.0ps"),
            ("0e0ps", "1e0ps"),
            (" 0ps", "1ps"),
            ("0ps ", "1ps "),
            ("０ps", "１ps"),
        )
        for coordinates in bad_coordinates:
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    SampleAxis(coordinates=coordinates)

    def test_sample_axis_rejects_nonincreasing_or_nonuniform_times(self) -> None:
        for coordinates in (
            ("0ps", "0ps"),
            ("2ps", "1ps"),
            ("0ps", "2ps", "5ps"),
        ):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    SampleAxis(coordinates=coordinates)

    def test_sample_axis_signed_int64_boundaries(self) -> None:
        maximum = (1 << 63) - 1
        valid = SampleAxis(
            coordinates=(f"{maximum - 2}ps", f"{maximum - 1}ps")
        )
        self.assertEqual(valid.stop_ps, maximum)

        with self.assertRaises(ValueError):
            SampleAxis(coordinates=(f"{maximum - 1}ps", f"{maximum}ps"))
        with self.assertRaises(ValueError):
            SampleAxis(coordinates=("0ps", f"{maximum + 1}ps"))

    def test_sampling_config_builds_exact_zero_start_axis(self) -> None:
        config = SamplingConfig(
            sample_period_ps=PositiveInteger(2000),
            sample_count=PositiveInteger(4),
        )
        self.assertEqual(config.window_stop_ps, 8000)
        axis = config.build_axis()
        self.assertIs(type(axis), SampleAxis)
        self.assertEqual(axis.coordinates, ("0ps", "2000ps", "4000ps", "6000ps"))
        self.assertEqual(axis.sample_period_ps, config.sample_period_ps.value)
        self.assertEqual(axis.size, config.sample_count.value)
        self.assertEqual(axis.stop_ps, config.window_stop_ps)

    def test_sampling_config_requires_exact_wrappers_and_at_least_two(self) -> None:
        with self.assertRaises(TypeError):
            SamplingConfig(
                sample_period_ps=cast(PositiveInteger, 2000),
                sample_count=PositiveInteger(2),
            )
        with self.assertRaises(TypeError):
            SamplingConfig(
                sample_period_ps=PositiveInteger(2000),
                sample_count=cast(PositiveInteger, 2),
            )
        with self.assertRaises(ValueError):
            SamplingConfig(
                sample_period_ps=PositiveInteger(2000),
                sample_count=PositiveInteger(1),
            )

    def test_sampling_config_enforces_exclusive_stop_int64_bound(self) -> None:
        maximum = (1 << 63) - 1
        valid = SamplingConfig(
            sample_period_ps=PositiveInteger(maximum // 2),
            sample_count=PositiveInteger(2),
        )
        self.assertLessEqual(valid.window_stop_ps, maximum)
        with self.assertRaises(ValueError):
            SamplingConfig(
                sample_period_ps=PositiveInteger((maximum // 2) + 1),
                sample_count=PositiveInteger(2),
            )

    def test_sampling_config_is_keyword_only_frozen_and_slotted(self) -> None:
        with self.assertRaises(TypeError):
            SamplingConfig(PositiveInteger(1), PositiveInteger(2))  # type: ignore[misc]
        config = SamplingConfig(
            sample_period_ps=PositiveInteger(1),
            sample_count=PositiveInteger(2),
        )
        with self.assertRaises(FrozenInstanceError):
            config.sample_count = PositiveInteger(3)  # type: ignore[misc]
        self.assertFalse(hasattr(config, "__dict__"))


if __name__ == "__main__":
    unittest.main()
