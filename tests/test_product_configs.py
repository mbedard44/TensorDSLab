import unittest
from typing import Any, override

import torch
from tensor_core import TensorAxis, TensorFieldSpec
from tensor_dslab import (
    AnalogWaveform,
    AnalogWaveformConfig,
    Charge,
    ChargeConfig,
    DigitizedWaveform,
    NoiseWaveform,
    NoiseWaveformKernels,
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


class UnitlessSourceSpec[
    AxesT: tuple[TensorAxis[Any], ...],
](TensorFieldSpec[AxesT]):
    """Represent an ordinary nonquantity semantic field specification."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        pass


class ProductConfigTests(unittest.TestCase):
    def test_preparation_rejects_nonquantity_source_spec(self) -> None:
        config = analog_config()
        source_spec = UnitlessSourceSpec(
            axes=config.spec.axes,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(source_spec.axes, config.spec.axes)
        self.assertEqual(source_spec.device, config.spec.device)
        self.assertIs(source_spec.dtype, config.spec.dtype)
        self.assertFalse(hasattr(source_spec, "unit"))

        with self.assertRaisesRegex(
            TypeError,
            "must be a QuantityFieldSpec",
        ):
            AnalogWaveform.prepare(
                source_specs=(source_spec,),  # type: ignore[arg-type]
                config=config,
            )

    def test_unprepared_private_state_is_exact(self) -> None:
        for config, expected in (
            (charge_config(), 6),
            (pure_config(), 1),
            (noise_config(), 2),
            (analog_config(), 2),
            (digitized_config(), 4),
        ):
            with self.subTest(config=type(config).__name__):
                self.assertFalse(config._is_prepared)
                self.assertEqual(config._source_specs, ())
                self.assertEqual(len(config._kernel_dimensions), expected)
                with self.assertRaises(TypeError):
                    hash(config)

    def test_collections_are_identity_values(self) -> None:
        left = noise_config().kernels
        right = noise_config().kernels
        self.assertIsNot(left, right)
        self.assertNotEqual(left, right)

    def test_configs_require_exact_product_specs_and_kernel_collections(
        self,
    ) -> None:
        charge = charge_config()
        with self.assertRaises(TypeError):
            ChargeConfig(
                spec=noise_config().spec,  # type: ignore[arg-type]
                kernels=charge.kernels,
                correlated_avalanche_generations=charge.correlated_avalanche_generations,
            )
        with self.assertRaises(TypeError):
            ChargeConfig(
                spec=charge.spec,
                kernels=NoiseWaveformKernels(members=()),  # type: ignore[arg-type]
                correlated_avalanche_generations=charge.correlated_avalanche_generations,
            )
        with self.assertRaises(TypeError):
            AnalogWaveformConfig(
                spec=analog_config().spec,
                kernels=1.0,  # type: ignore[arg-type]
            )

    def test_every_preparation_is_fresh_and_retains_source_spec_identity(
        self,
    ) -> None:
        source_specs = (
            source().spec,
            charge_config().spec,
            pure_config().spec,
            noise_config().spec,
            analog_config().spec,
        )
        cases = (
            (Charge, (source_specs[0],), charge_config()),
            (PureWaveform, (source_specs[1],), pure_config()),
            (NoiseWaveform, (), noise_config()),
            (
                AnalogWaveform,
                (source_specs[2], source_specs[3]),
                analog_config(),
            ),
            (DigitizedWaveform, (source_specs[4],), digitized_config()),
        )
        for product_type, admitted, config in cases:
            with self.subTest(product=product_type.__name__):
                prepared = product_type.prepare(
                    source_specs=admitted,
                    config=config,
                )
                self.assertIs(type(prepared), type(config))
                self.assertIsNot(prepared, config)
                self.assertTrue(prepared._is_prepared)
                self.assertIs(prepared._source_specs, admitted)
                for retained, supplied in zip(prepared._source_specs, admitted):
                    self.assertIs(retained, supplied)
