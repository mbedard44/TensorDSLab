from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from inspect import Parameter, signature
from typing import Any, cast
import unittest

from tensor_core import (
    FiniteFloat,
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    PositiveInteger,
    Probability,
)

from tensor_dslab import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    AnalogSaturationConfig,
    AnalogWaveformConfig,
    ChargeConfig,
    ChargeSmearingConfig,
    CorrelatedAvalancheConfig,
    DarkCountConfig,
    DelayedCrosstalkConfig,
    DigitizedWaveformConfig,
    DirectCrosstalkConfig,
    ExponentialDelayConfig,
    FixedDelayConfig,
    NoiseWaveformConfig,
    PsdNoiseConfig,
    PureWaveformConfig,
    ReadoutConfig,
    SamplingConfig,
    TimingJitterConfig,
    TpcFebSnrPulseConfig,
    VetoPduPulseConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)


def make_tpc_config() -> TpcFebSnrPulseConfig:
    return TpcFebSnrPulseConfig(
        fast_time_constant_ns=PositiveFloat(83.0),
        slow_time_constant_ns=PositiveFloat(383.0),
        support_time_ns=PositiveFloat(3000.0),
        peak_voltage_mv_per_pe=FiniteFloat(-7.0),
    )


def make_veto_config() -> VetoPduPulseConfig:
    return VetoPduPulseConfig(
        gaussian_center_ns=FiniteFloat(232.89),
        gaussian_width_ns=PositiveFloat(507.72),
        edge_offset_1_ns=FiniteFloat(-81.92),
        edge_width_1_ns=PositiveFloat(147.28),
        edge_offset_2_ns=FiniteFloat(-176.50),
        edge_width_2_ns=PositiveFloat(45.69),
        support_time_ns=PositiveFloat(2020.27),
        peak_voltage_mv_per_pe=FiniteFloat(-14.5912372),
    )


def make_psd_config() -> PsdNoiseConfig:
    return PsdNoiseConfig(
        frequency_left_edges_hz=(
            NonnegativeFloat(0.0),
            NonnegativeFloat(1.0),
        ),
        frequency_stop_hz=PositiveFloat(2.0),
        power_density_mv2_per_hz=(
            NonnegativeFloat(0.0),
            NonnegativeFloat(1.0),
        ),
    )


def make_digitized_config(
    *,
    bit_depth: int = 12,
    gain_db: float = 20.0,
) -> DigitizedWaveformConfig:
    return DigitizedWaveformConfig(
        bit_depth=PositiveInteger(bit_depth),
        input_min_mv=FiniteFloat(-1000.0),
        input_max_mv=FiniteFloat(1000.0),
        analog_gain_db=NonnegativeFloat(gain_db),
    )


def make_all_valid_configs() -> tuple[object, ...]:
    fixed = FixedDelayConfig(delay_ns=NonnegativeFloat(0.0))
    exponential = ExponentialDelayConfig(mean_delay_ns=PositiveFloat(10.0))
    direct = DirectCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(0.3),
        delay=fixed,
    )
    delayed = DelayedCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(0.1),
        delay=exponential,
    )
    recovery = AfterpulseRecoveryConfig(time_constant_ns=PositiveFloat(100.0))
    afterpulse = AfterpulseConfig(
        probability=Probability(0.2),
        mean_delay_ns=PositiveFloat(1000.0),
        recovery=recovery,
    )
    correlated = CorrelatedAvalancheConfig(
        maximum_generations=NonnegativeInteger(1),
        direct_crosstalk=direct,
        delayed_crosstalk=delayed,
        afterpulse=afterpulse,
    )
    charge = ChargeConfig(
        dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(0.0)),
        timing_jitter=TimingJitterConfig(sigma_ns=NonnegativeFloat(0.0)),
        correlated_avalanches=correlated,
        smearing=ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.0)),
    )
    tpc = make_tpc_config()
    pure = PureWaveformConfig(model=tpc)
    zero = ZeroNoiseConfig()
    white = WhiteNoiseConfig(rms_mv=PositiveFloat(1.0))
    psd = make_psd_config()
    noise = NoiseWaveformConfig(model=psd)
    saturation = AnalogSaturationConfig(
        minimum_mv=FiniteFloat(-1000.0),
        maximum_mv=FiniteFloat(1000.0),
    )
    analog = AnalogWaveformConfig(saturation=saturation)
    digitized = make_digitized_config()
    sampling = SamplingConfig(
        sample_period_ps=PositiveInteger(2000),
        sample_count=PositiveInteger(4),
    )
    readout = ReadoutConfig(
        sampling=sampling,
        charge=charge,
        pure_waveform=pure,
        noise_waveform=noise,
        analog_waveform=analog,
        digitized_waveform=digitized,
    )
    return (
        TimingJitterConfig(sigma_ns=NonnegativeFloat(0.0)),
        DarkCountConfig(rate_hz=NonnegativeFloat(0.0)),
        fixed,
        exponential,
        direct,
        delayed,
        recovery,
        afterpulse,
        correlated,
        ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.0)),
        charge,
        tpc,
        make_veto_config(),
        pure,
        zero,
        white,
        psd,
        noise,
        saturation,
        AnalogWaveformConfig(),
        analog,
        digitized,
        sampling,
        readout,
    )


class ReadoutConfigsTest(unittest.TestCase):
    def test_every_config_is_final_frozen_slotted_and_keyword_only(self) -> None:
        for candidate in make_all_valid_configs():
            config = cast(Any, candidate)
            config_type = type(config)
            with self.subTest(config=config_type.__name__):
                self.assertTrue(is_dataclass(config))
                self.assertTrue(getattr(config_type, "__final__", False))
                self.assertTrue(config_type.__dataclass_params__.frozen)
                self.assertFalse(hasattr(config, "__dict__"))
                for parameter in signature(config_type).parameters.values():
                    self.assertIs(parameter.kind, Parameter.KEYWORD_ONLY)
                first = fields(config)[0] if fields(config) else None
                if first is None:
                    with self.assertRaises((TypeError, AttributeError)):
                        setattr(config, "added", object())
                else:
                    with self.assertRaises(FrozenInstanceError):
                        setattr(config, first.name, object())

    def test_every_config_component_requires_its_exact_declared_class(self) -> None:
        for candidate in make_all_valid_configs():
            config = cast(Any, candidate)
            for component in fields(config):
                with self.subTest(config=type(config).__name__, field=component.name):
                    with self.assertRaises(TypeError):
                        replace(config, **{component.name: object()})

    def test_charge_configs_accept_mvp_delay_models_and_optional_composition(self) -> None:
        delay_models = (
            FixedDelayConfig(delay_ns=NonnegativeFloat(0.0)),
            ExponentialDelayConfig(mean_delay_ns=PositiveFloat(1.0)),
        )
        for delay in delay_models:
            self.assertIs(
                DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.0),
                    delay=delay,
                ).delay,
                delay,
            )
            self.assertIs(
                DelayedCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.0),
                    delay=delay,
                ).delay,
                delay,
            )

        self.assertIsNone(AfterpulseConfig(
            probability=Probability(0.0),
            mean_delay_ns=PositiveFloat(1.0),
        ).recovery)
        self.assertEqual(
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(0)
            ).maximum_generations.value,
            0,
        )
        self.assertEqual(ChargeConfig(), ChargeConfig())

    def test_pulse_model_relationships_and_exact_union(self) -> None:
        tpc = make_tpc_config()
        veto = make_veto_config()
        self.assertIs(PureWaveformConfig(model=tpc).model, tpc)
        self.assertIs(PureWaveformConfig(model=veto).model, veto)

        with self.assertRaises(ValueError):
            TpcFebSnrPulseConfig(
                fast_time_constant_ns=PositiveFloat(10.0),
                slow_time_constant_ns=PositiveFloat(10.0),
                support_time_ns=PositiveFloat(100.0),
                peak_voltage_mv_per_pe=FiniteFloat(-1.0),
            )
        with self.assertRaises(ValueError):
            TpcFebSnrPulseConfig(
                fast_time_constant_ns=PositiveFloat(10.0),
                slow_time_constant_ns=PositiveFloat(20.0),
                support_time_ns=PositiveFloat(100.0),
                peak_voltage_mv_per_pe=FiniteFloat(0.0),
            )
        with self.assertRaises(ValueError):
            replace(veto, peak_voltage_mv_per_pe=FiniteFloat(0.0))
        with self.assertRaises(TypeError):
            PureWaveformConfig(model=object())  # type: ignore[arg-type]

    def test_noise_model_exact_union(self) -> None:
        models = (
            ZeroNoiseConfig(),
            WhiteNoiseConfig(rms_mv=PositiveFloat(1.0)),
            make_psd_config(),
        )
        for model in models:
            self.assertIs(NoiseWaveformConfig(model=model).model, model)
        with self.assertRaises(TypeError):
            NoiseWaveformConfig(model=object())  # type: ignore[arg-type]

    def test_psd_structure_edges(self) -> None:
        edge0 = NonnegativeFloat(0.0)
        edge1 = NonnegativeFloat(1.0)
        density0 = NonnegativeFloat(0.0)
        density1 = NonnegativeFloat(1.0)

        bad_kwargs: tuple[dict[str, object], ...] = (
            {
                "frequency_left_edges_hz": [],
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (density1,),
            },
            {
                "frequency_left_edges_hz": (edge0,),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": [density1],
            },
            {
                "frequency_left_edges_hz": (),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (),
            },
            {
                "frequency_left_edges_hz": (edge0, edge1),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (density1,),
            },
            {
                "frequency_left_edges_hz": (edge1,),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (density1,),
            },
            {
                "frequency_left_edges_hz": (edge0, edge0),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (density0, density1),
            },
            {
                "frequency_left_edges_hz": (edge0, edge1),
                "frequency_stop_hz": PositiveFloat(1.0),
                "power_density_mv2_per_hz": (density0, density1),
            },
            {
                "frequency_left_edges_hz": (edge0, edge1),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (density0, density0),
            },
            {
                "frequency_left_edges_hz": (edge0, object()),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (density0, density1),
            },
            {
                "frequency_left_edges_hz": (edge0, edge1),
                "frequency_stop_hz": PositiveFloat(2.0),
                "power_density_mv2_per_hz": (density0, object()),
            },
        )
        for kwargs in bad_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    PsdNoiseConfig(**kwargs)  # type: ignore[arg-type]

    def test_analog_saturation_boundaries(self) -> None:
        self.assertIsNotNone(
            AnalogSaturationConfig(minimum_mv=FiniteFloat(-1.0)).minimum_mv
        )
        self.assertIsNotNone(
            AnalogSaturationConfig(maximum_mv=FiniteFloat(1.0)).maximum_mv
        )
        with self.assertRaises(ValueError):
            AnalogSaturationConfig()
        for lower, upper in ((1.0, 1.0), (2.0, 1.0)):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    AnalogSaturationConfig(
                        minimum_mv=FiniteFloat(lower),
                        maximum_mv=FiniteFloat(upper),
                    )

    def test_digitized_config_range_boundaries(self) -> None:
        for bit_depth in (1, 16):
            self.assertEqual(make_digitized_config(bit_depth=bit_depth).bit_depth.value, bit_depth)
        for gain in (0.0, 40.0):
            self.assertEqual(make_digitized_config(gain_db=gain).analog_gain_db.value, gain)
        with self.assertRaises(ValueError):
            make_digitized_config(bit_depth=17)
        with self.assertRaises(ValueError):
            make_digitized_config(gain_db=40.0001)
        with self.assertRaises(ValueError):
            replace(
                make_digitized_config(),
                input_min_mv=FiniteFloat(1.0),
                input_max_mv=FiniteFloat(1.0),
            )

    def test_readout_config_requires_sampling_and_accepts_optional_products(self) -> None:
        sampling = SamplingConfig(
            sample_period_ps=PositiveInteger(1),
            sample_count=PositiveInteger(2),
        )
        minimal = ReadoutConfig(sampling=sampling)
        self.assertIs(minimal.sampling, sampling)
        self.assertIsNone(minimal.charge)
        with self.assertRaises(TypeError):
            ReadoutConfig(sampling=object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ReadoutConfig(
                sampling=sampling,
                charge=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
