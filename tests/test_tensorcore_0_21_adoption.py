"""Contract evidence for the exact TensorCore 0.21 dependency."""

import inspect
from pathlib import Path
import unittest

import torch
import tensor_core
from tensor_core import (
    MultinomialDistribution,
    OffsetAxis,
    TensorKernel,
)

from tensor_dslab import SampleAxis


class TensorCore021AdoptionTest(unittest.TestCase):
    def test_exact_version_and_root_exports(self) -> None:
        dependency = Path("pyproject.toml").read_text()
        self.assertIn(
            "78d0891bf6c0fefbcad4abe09980867c54202a9e",
            dependency,
        )
        self.assertEqual(len(tensor_core.__all__), 30)

    def test_probability_kernel_is_retired(self) -> None:
        self.assertFalse(hasattr(tensor_core, "ProbabilityKernel"))

    def test_multinomial_uses_direct_law_arguments(self) -> None:
        parameters = inspect.signature(MultinomialDistribution).parameters
        self.assertEqual(
            tuple(parameters),
            ("counts", "probabilities", "completion_probability"),
        )

    def test_offset_axis_is_literal_and_role_owned(self) -> None:
        axis = OffsetAxis(relative_to=SampleAxis, offsets=(-1, 0, 2))
        self.assertIs(axis.relative_to, SampleAxis)
        self.assertEqual(axis.coordinates, (-1, 0, 2))

    def test_tensor_kernel_snapshots_noncontiguous_storage(self) -> None:
        class LiteralKernel(TensorKernel[tuple[()], tuple[OffsetAxis]]):
            def _require(self) -> None:
                return

        source = torch.arange(8, dtype=torch.float64)[::2]
        kernel = LiteralKernel(
            tensor=source,
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0, 1, 2, 3)),
            ),
        )
        source.zero_()
        self.assertEqual(kernel.tensor.tolist(), [0.0, 2.0, 4.0, 6.0])
        self.assertTrue(kernel.tensor.is_contiguous())


for _index in range(15):
    def _offset_case(
        self: TensorCore021AdoptionTest,
        index: int = _index,
    ) -> None:
        offsets = tuple(range(index + 1))
        axis = OffsetAxis(relative_to=SampleAxis, offsets=offsets)
        self.assertEqual(axis.size, index + 1)
        self.assertEqual(axis.coordinate_at(index), index)
        self.assertEqual(axis.index_of(index), index)

    setattr(
        TensorCore021AdoptionTest,
        f"test_offset_axis_case_{_index:02d}",
        _offset_case,
    )
