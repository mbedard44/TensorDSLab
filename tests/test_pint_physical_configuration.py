"""Pint boundary evidence for scalar and kernel quantities."""

import unittest

import numpy as np
import torch
from tensor_core import OffsetAxis

from tensor_dslab import (
    DarkCountRate,
    Pulse,
    SampleAxis,
    SmearingWidth,
    quantities,
    quantity,
)


class PintPhysicalBoundaryTest(unittest.TestCase):
    def test_scalar_helper_is_copied_and_canonicalizable(self) -> None:
        rate = DarkCountRate(
            quantity=quantity(100, "kHz"),
            conditioning_axes=(),
            operation_axes=(),
        )
        self.assertEqual(float(rate.tensor), 100_000.0)
        self.assertEqual(str(rate.quantity.units), "hertz")

    def test_tuple_and_arbitrary_rank_tensor_helpers(self) -> None:
        vector = quantities((1, 2, 3), "mV")
        matrix_source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        matrix = quantities(matrix_source, "mV")
        matrix_source.zero_()
        self.assertEqual(vector.magnitude.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(matrix.magnitude.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_incompatible_unit_and_nonfinite_values_reject(self) -> None:
        with self.assertRaises(ValueError):
            SmearingWidth(
                quantity=quantity(1, "mV"),
                conditioning_axes=(),
                operation_axes=(),
            )
        with self.assertRaises(ValueError):
            quantities(torch.tensor([float("nan")]), "mV")

    def test_pulse_owns_one_float64_snapshot(self) -> None:
        source = torch.tensor([-1.0, -2.0], dtype=torch.float32)
        pulse = Pulse(
            quantity=quantities(source, "mV"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0, 1)),
            ),
        )
        source.zero_()
        self.assertEqual(pulse.tensor.tolist(), [-1.0, -2.0])
        self.assertEqual(pulse.tensor.dtype, torch.float64)


for _magnitude in range(1, 13):
    def _unit_case(
        self: PintPhysicalBoundaryTest,
        magnitude: int = _magnitude,
    ) -> None:
        width = SmearingWidth(
            quantity=quantity(magnitude, "percent"),
            conditioning_axes=(),
            operation_axes=(),
        )
        self.assertAlmostEqual(float(width.tensor), magnitude / 100.0)

    setattr(PintPhysicalBoundaryTest, f"test_percent_case_{_magnitude:02d}", _unit_case)
