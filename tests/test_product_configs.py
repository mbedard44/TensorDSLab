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
        photoelectrons_spec = source().spec
        charge_spec = charge_config().spec
        pure_spec = pure_config().spec
        noise_spec = noise_config().spec
        analog_spec = analog_config().spec

        charge_sources = (photoelectrons_spec,)
        charge = charge_config()
        prepared_charge = Charge.prepare(
            source_specs=charge_sources,
            config=charge,
        )
        self.assertIs(type(prepared_charge), type(charge))
        self.assertIsNot(prepared_charge, charge)
        self.assertTrue(prepared_charge._is_prepared)
        self.assertIs(prepared_charge._source_specs, charge_sources)
        self.assertIs(prepared_charge._source_specs[0], photoelectrons_spec)

        pure_sources = (charge_spec,)
        pure = pure_config()
        prepared_pure = PureWaveform.prepare(
            source_specs=pure_sources,
            config=pure,
        )
        self.assertIs(type(prepared_pure), type(pure))
        self.assertIsNot(prepared_pure, pure)
        self.assertTrue(prepared_pure._is_prepared)
        self.assertIs(prepared_pure._source_specs, pure_sources)
        self.assertIs(prepared_pure._source_specs[0], charge_spec)

        noise_sources = ()
        noise = noise_config()
        prepared_noise = NoiseWaveform.prepare(
            source_specs=noise_sources,
            config=noise,
        )
        self.assertIs(type(prepared_noise), type(noise))
        self.assertIsNot(prepared_noise, noise)
        self.assertTrue(prepared_noise._is_prepared)
        self.assertIs(prepared_noise._source_specs, noise_sources)

        analog_sources = (pure_spec, noise_spec)
        analog = analog_config()
        prepared_analog = AnalogWaveform.prepare(
            source_specs=analog_sources,
            config=analog,
        )
        self.assertIs(type(prepared_analog), type(analog))
        self.assertIsNot(prepared_analog, analog)
        self.assertTrue(prepared_analog._is_prepared)
        self.assertIs(prepared_analog._source_specs, analog_sources)
        self.assertIs(prepared_analog._source_specs[0], pure_spec)
        self.assertIs(prepared_analog._source_specs[1], noise_spec)

        digitized_sources = (analog_spec,)
        digitized = digitized_config()
        prepared_digitized = DigitizedWaveform.prepare(
            source_specs=digitized_sources,
            config=digitized,
        )
        self.assertIs(type(prepared_digitized), type(digitized))
        self.assertIsNot(prepared_digitized, digitized)
        self.assertTrue(prepared_digitized._is_prepared)
        self.assertIs(prepared_digitized._source_specs, digitized_sources)
        self.assertIs(prepared_digitized._source_specs[0], analog_spec)
