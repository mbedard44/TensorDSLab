import unittest

import torch
from tensor_core import Threefry4x32

from tensor_dslab import NoiseWaveform
from tests._product_support import noise_config


class NoiseStatisticsTests(unittest.TestCase):
    def test_white_noise_is_finite_and_nonconstant(self) -> None:
        result = NoiseWaveform.create(
            sources=(), config=noise_config(white=True), rng=Threefry4x32(seed=9)
        )
        self.assertTrue(bool(torch.isfinite(result.tensor).all()))
        self.assertGreater(float(result.tensor.std()), 0.0)
