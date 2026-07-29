import unittest

import torch
from tensor_core import OffsetAxis, OffsetCoordinates, Threefry4x32

from tensor_dslab import (
    Charge,
    PulseResponse,
    PulseResponseSpec,
    PureWaveform,
    PureWaveformConfig,
    PureWaveformKernels,
    TimeAxis,
)
from tests._product_support import charge_config, pure_config, source


class PureWaveformTests(unittest.TestCase):
    def test_literal_pulse_convolution(self) -> None:
        charge = Charge.create(
            sources=(source(),), config=charge_config(), rng=Threefry4x32(seed=0)
        )
        result = PureWaveform.create(sources=(charge,), config=pure_config())
        expected = -charge.tensor
        expected = expected + torch.nn.functional.pad(
            -0.5 * charge.tensor[..., :-1], (1, 0)
        )
        self.assertTrue(torch.equal(result.tensor, expected))

        negative_pulse = PulseResponse(
            tensor=torch.tensor([2.0], dtype=torch.float32),
            spec=PulseResponseSpec(
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(
                        coordinates=OffsetCoordinates(offsets=(-1,)),
                        relative_to=TimeAxis,
                    ),
                ),
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=pure_config().kernels.pulse_response.spec.unit,
            ),
        )
        negative_result = PureWaveform.create(
            sources=(charge,),
            config=PureWaveformConfig(
                spec=pure_config().spec,
                kernels=PureWaveformKernels(members=(negative_pulse,)),
            ),
        )
        negative_expected = torch.zeros_like(charge.tensor)
        negative_expected[..., :-1] = 2.0 * charge.tensor[..., 1:]
        self.assertTrue(torch.equal(negative_result.tensor, negative_expected))

    def test_literal_pulse_preserves_autograd_and_fresh_storage(self) -> None:
        config = charge_config()
        values = torch.ones(
            config.spec.shape,
            dtype=torch.float32,
            requires_grad=True,
        )
        charge = Charge(tensor=values, spec=config.spec)
        result = PureWaveform.create(
            sources=(charge,),
            config=pure_config(),
        )
        self.assertTrue(result.tensor.requires_grad)
        self.assertNotEqual(
            result.tensor.untyped_storage().data_ptr(),
            values.untyped_storage().data_ptr(),
        )
        result.tensor.sum().backward()
        assert values.grad is not None
        expected = torch.full_like(values, -1.5)
        expected[..., -1] = -1.0
        torch.testing.assert_close(values.grad, expected, rtol=0, atol=0)
