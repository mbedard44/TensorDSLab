import unittest

import torch
from tensor_core import CountCoordinates, CounterRng, Threefry4x32
from typing import override

from tensor_dslab import (
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    NoiseWaveformKernels,
    NoiseWaveformSpec,
    unit_registry,
)
from tests._product_support import noise_config


class _FailingRng(CounterRng):
    @override
    def _generate_block(
        self,
        *,
        key,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        raise AssertionError("the exact-zero branch must not request words")


class NoiseBranchTests(unittest.TestCase):
    def test_zero_branch_draws_no_nonzero_values(self) -> None:
        result = NoiseWaveform.create(
            sources=(), config=noise_config(), rng=_FailingRng(seed=3)
        )
        self.assertTrue(torch.equal(result.tensor, torch.zeros_like(result.tensor)))
        empty = NoiseWaveform.create(
            sources=(),
            config=NoiseWaveformConfig(
                spec=NoiseWaveformSpec(
                    axes=(
                        ExampleAxis(
                            coordinates=CountCoordinates(count=0)
                        ),
                    ),
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    unit=unit_registry.Unit("mV"),
                ),
                kernels=NoiseWaveformKernels(members=()),
            ),
            rng=_FailingRng(seed=3),
        )
        self.assertEqual(empty.tensor.numel(), 0)

    def test_white_branch_replays(self) -> None:
        left = NoiseWaveform.create(
            sources=(), config=noise_config(white=True), rng=Threefry4x32(seed=4)
        )
        right = NoiseWaveform.create(
            sources=(), config=noise_config(white=True), rng=Threefry4x32(seed=4)
        )
        self.assertTrue(torch.equal(left.tensor, right.tensor))
