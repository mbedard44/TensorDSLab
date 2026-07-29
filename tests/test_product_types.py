import unittest

import torch

from tensor_dslab import (
    AnalogWaveform,
    Charge,
    DigitizedWaveform,
    NoiseWaveform,
    Photoelectrons,
    PureWaveform,
)
from tests._product_support import (
    analog_config,
    charge_config,
    digitized_config,
    noise_config,
    pure_config,
    source,
)


class ProductTypeTests(unittest.TestCase):
    def test_products_are_fieldless_semantic_leaves(self) -> None:
        fixtures = (
            source(),
            Charge(tensor=torch.zeros(charge_config().spec.shape), spec=charge_config().spec),
            PureWaveform(tensor=torch.zeros(pure_config().spec.shape), spec=pure_config().spec),
            NoiseWaveform(tensor=torch.zeros(noise_config().spec.shape), spec=noise_config().spec),
            AnalogWaveform(tensor=torch.zeros(analog_config().spec.shape), spec=analog_config().spec),
            DigitizedWaveform(tensor=torch.zeros(digitized_config().spec.shape, dtype=torch.int32), spec=digitized_config().spec),
        )
        self.assertEqual(
            tuple(type(value) for value in fixtures),
            (Photoelectrons, Charge, PureWaveform, NoiseWaveform, AnalogWaveform, DigitizedWaveform),
        )
        for value in fixtures:
            self.assertFalse(hasattr(value, "__dict__"))

    def test_exact_product_specs_and_most_derived_movement(self) -> None:
        source_field = source()
        with self.assertRaises(TypeError):
            Charge(tensor=source_field.tensor.clone(), spec=source_field.spec)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            charge_config().spec.to(dtype=torch.int64)
