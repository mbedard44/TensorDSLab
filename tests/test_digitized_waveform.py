import unittest

import torch

from tensor_dslab import AnalogWaveform, DigitizedWaveform
from tests._product_support import analog_config, digitized_config


class DigitizedWaveformTests(unittest.TestCase):
    def test_linear_adc_endpoints(self) -> None:
        values = torch.zeros(analog_config().spec.shape)
        values[..., 0] = -2.0
        values[..., -1] = 2.0
        source = AnalogWaveform(tensor=values, spec=analog_config().spec)
        result = DigitizedWaveform.create(
            sources=(source,), config=digitized_config()
        )
        self.assertTrue(bool((result.tensor[..., 0] == 0).all()))
        self.assertTrue(bool((result.tensor[..., -1] == 4095).all()))

        prepared = DigitizedWaveform.prepare(
            source_specs=(source.spec,),
            config=digitized_config(),
        )
        self.assertIs(prepared.kernels.bit_depth.dtype, torch.int16)
        self.assertEqual(int(prepared.kernels.bit_depth.tensor), 12)
