import unittest

import torch

from tensor_dslab import AnalogWaveform, NoiseWaveform, PureWaveform
from tests._product_support import analog_config, noise_config, pure_config


class AnalogWaveformTests(unittest.TestCase):
    def test_addition_and_saturation(self) -> None:
        pure = PureWaveform(
            tensor=torch.full(pure_config().spec.shape, 3.0),
            spec=pure_config().spec,
        )
        noise = NoiseWaveform(
            tensor=torch.full(noise_config().spec.shape, -0.5),
            spec=noise_config().spec,
        )
        result = AnalogWaveform.create(
            sources=(pure, noise), config=analog_config()
        )
        self.assertTrue(torch.equal(result.tensor, torch.full_like(result.tensor, 2.0)))

    def test_composition_preserves_autograd_without_aliasing(self) -> None:
        pure_tensor = torch.full(
            pure_config().spec.shape,
            0.5,
            requires_grad=True,
        )
        noise_tensor = torch.full(
            noise_config().spec.shape,
            -0.25,
            requires_grad=True,
        )
        pure = PureWaveform(tensor=pure_tensor, spec=pure_config().spec)
        noise = NoiseWaveform(tensor=noise_tensor, spec=noise_config().spec)
        result = AnalogWaveform.create(
            sources=(pure, noise),
            config=analog_config(),
        )
        result.tensor.sum().backward()
        assert pure_tensor.grad is not None and noise_tensor.grad is not None
        self.assertTrue(torch.equal(pure_tensor.grad, torch.ones_like(pure_tensor)))
        self.assertTrue(torch.equal(noise_tensor.grad, torch.ones_like(noise_tensor)))
        for source_tensor in (pure_tensor, noise_tensor):
            self.assertNotEqual(
                result.tensor.untyped_storage().data_ptr(),
                source_tensor.untyped_storage().data_ptr(),
            )
