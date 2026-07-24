import ast
from dataclasses import fields, is_dataclass, replace
import math
from pathlib import Path
import tokenize
from typing import Any, cast, ClassVar, override
import unittest
from unittest.mock import patch

import numpy as np
import pint
from pint import Quantity
import torch
from tensor_core import (
    CounterRng,
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveInteger,
    Probability,
    RngKey,
    Threefry4x32,
)

import tensor_dslab
import tensor_dslab.common as common
from tensor_dslab import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    AnalogSaturationConfig,
    AnalogWaveformConfig,
    ChannelAxis,
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    CorrelatedAvalancheConfig,
    DarkCountConfig,
    DelayedCrosstalkConfig,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    DirectCrosstalkConfig,
    ExampleAxis,
    ExponentialDelayConfig,
    FixedDelayConfig,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    TimingJitterConfig,
    TpcFebSnrPulseConfig,
    VetoPduPulseConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
    quantities,
    quantity,
    simulate_readout,
)
from tensor_dslab.common import units as unit_boundary
from tensor_dslab.readout.runtime.prepare import prepare_readout
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


def _ns(value: int | float) -> Quantity:
    return quantity(value, "ns")


def _hz(value: int | float) -> Quantity:
    return quantity(value, "Hz")


def _mv(value: int | float) -> Quantity:
    return quantity(value, "mV")


def _density(value: int | float) -> Quantity:
    return quantity(value, "mV ** 2 / Hz")


class _RecordingRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[list[tuple[RngKey, torch.Tensor, int, int]]] = []

    @override
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls.append((key, positions.clone(), quantum, block))
        return torch.zeros(
            positions.shape + (4,),
            dtype=torch.int64,
            device=positions.device,
        )

    @classmethod
    def reset(cls) -> None:
        cls.calls = []


def _valid_configs() -> tuple[Any, ...]:
    fixed = FixedDelayConfig(delay=_ns(0.0))
    exponential = ExponentialDelayConfig(mean_delay=_ns(10.0))
    direct = DirectCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(0.1),
        delay=fixed,
    )
    delayed = DelayedCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(0.1),
        delay=exponential,
    )
    recovery = AfterpulseRecoveryConfig(time_constant=_ns(20.0))
    afterpulse = AfterpulseConfig(
        probability=Probability(0.1),
        mean_delay=_ns(10.0),
        recovery=recovery,
    )
    correlated = CorrelatedAvalancheConfig(
        maximum_generations=NonnegativeInteger(1),
        direct_crosstalk=direct,
        delayed_crosstalk=delayed,
        afterpulse=afterpulse,
    )
    tpc = TpcFebSnrPulseConfig(
        fast_time_constant=_ns(1.0),
        slow_time_constant=_ns(2.0),
        support_time=_ns(6.0),
        peak_voltage_per_photoelectron=_mv(1.0),
    )
    veto = VetoPduPulseConfig(
        gaussian_center=_ns(0.0),
        gaussian_width=_ns(1.0),
        edge_offset_1=_ns(-1.0),
        edge_width_1=_ns(1.0),
        edge_offset_2=_ns(1.0),
        edge_width_2=_ns(1.0),
        support_time=_ns(6.0),
        peak_voltage_per_photoelectron=_mv(1.0),
    )
    zero = ZeroNoiseConfig()
    white = WhiteNoiseConfig(rms=_mv(1.0))
    psd = PsdNoiseConfig(
        frequency_left_edges=quantities((0.0, 1.0e8), "Hz"),
        frequency_stop=_hz(3.0e8),
        power_density=quantities((1.0e-9, 2.0e-9), "mV ** 2 / Hz"),
    )
    saturation = AnalogSaturationConfig(minimum=_mv(-2.0), maximum=_mv(2.0))
    analog = AnalogWaveformConfig(saturation=saturation)
    digitized = DigitizedWaveformConfig(
        bit_depth=PositiveInteger(12),
        input_minimum=_mv(-20.0),
        input_maximum=_mv(20.0),
        analog_gain_db=NonnegativeFloat(0.0),
    )
    charge = ChargeConfig(
        dark_count=DarkCountConfig(rate=_hz(1.0e6)),
        timing_jitter=TimingJitterConfig(sigma=_ns(1.0)),
        correlated_avalanches=correlated,
        smearing=ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.1)),
    )
    pure = PureWaveformConfig(model=tpc)
    noise = NoiseWaveformConfig(model=zero)
    readout = ReadoutConfig(
        charge=charge,
        pure_waveform=pure,
        noise_waveform=noise,
        analog_waveform=analog,
        digitized_waveform=digitized,
    )
    return (
        TimingJitterConfig(sigma=_ns(1.0)),
        DarkCountConfig(rate=_hz(1.0)),
        fixed,
        exponential,
        direct,
        delayed,
        recovery,
        afterpulse,
        correlated,
        ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.1)),
        charge,
        tpc,
        veto,
        pure,
        zero,
        white,
        psd,
        noise,
        saturation,
        analog,
        digitized,
        readout,
    )


class PintPhysicalConfigurationTest(unittest.TestCase):
    def test_public_helpers_use_one_private_registry_and_strict_inputs(self) -> None:
        application_registry = pint.get_application_registry()
        first = quantity(2, "ns")
        second = quantity(2.0, "ns")
        group = quantities((1, 2.0), "mV")
        self.assertIs(first._REGISTRY, second._REGISTRY)
        self.assertIs(group._REGISTRY, first._REGISTRY)
        self.assertIs(pint.get_application_registry(), application_registry)
        self.assertEqual(first.magnitude, 2.0)
        self.assertIs(type(first.magnitude), float)
        self.assertAlmostEqual(first.to("ps").magnitude, 2_000.0)
        self.assertIs(type(group.magnitude), np.ndarray)
        self.assertIs(group.magnitude.dtype, np.dtype(np.float64))
        self.assertFalse(group.magnitude.flags.writeable)
        np.testing.assert_array_equal(group.magnitude, np.array((1.0, 2.0)))

        for invalid in (True, "1", None, object()):
            with self.subTest(magnitude=invalid):
                with self.assertRaises(TypeError):
                    quantity(cast(Any, invalid), "ns")
        with self.assertRaises(ValueError) as raised:
            quantity(1 << 20_000, "ns")
        self.assertIsInstance(raised.exception.__cause__, OverflowError)
        for invalid in ([], [1.0], (value for value in (1.0,))):
            with self.subTest(container=type(invalid).__name__):
                with self.assertRaises(TypeError):
                    quantities(cast(Any, invalid), "ns")
        empty = quantities((), "ns")
        self.assertEqual(empty.magnitude.shape, (0,))
        self.assertFalse(empty.magnitude.flags.writeable)
        with self.assertRaises(ValueError):
            quantities((), "")

        self.assertEqual(
            unit_boundary._PARSER_ERRORS,
            (
                pint.PintError,
                ValueError,
                TypeError,
                ArithmeticError,
                AssertionError,
                tokenize.TokenError,
            ),
        )
        for invalid in ("", " ", "2 ns", "not_a_unit", "ns +"):
            with self.subTest(unit=invalid):
                with self.assertRaises(ValueError):
                    quantity(1, invalid)
        with self.assertRaises(TypeError):
            quantity(1, cast(Any, 1))

    def test_external_scalars_are_copied_and_non_scalars_are_rejected(self) -> None:
        package_source = quantity(2, "ns")
        package_first = TimingJitterConfig(sigma=package_source)
        package_second = TimingJitterConfig(sigma=package_source)
        self.assertIsNot(package_first.sigma, package_source)
        self.assertIsNot(package_second.sigma, package_source)
        self.assertIsNot(package_first.sigma, package_second.sigma)
        self.assertIs(package_first.sigma._REGISTRY, package_source._REGISTRY)

        external = pint.UnitRegistry(cache_folder=None)
        source = cast(Quantity, external.Quantity(0.002, "us"))
        config = TimingJitterConfig(sigma=source)
        self.assertIsNot(config.sigma, source)
        self.assertIsNot(config.sigma._REGISTRY, external)
        self.assertEqual(config.sigma.magnitude, source.to("ns").magnitude)
        self.assertEqual(str(config.sigma.units), "nanosecond")

        for invalid in (
            external.Quantity([1.0], "ns"),
            object(),
        ):
            with self.subTest(value=type(invalid).__name__):
                with self.assertRaises(TypeError):
                    TimingJitterConfig(sigma=cast(Any, invalid))
        with self.assertRaises(ValueError):
            TimingJitterConfig(
                sigma=cast(Quantity, external.Quantity(1.0, "kg"))
            )
        with self.assertRaises(ValueError):
            TimingJitterConfig(
                sigma=cast(
                    Quantity,
                    external.Quantity(float("nan"), "ns"),
                )
            )

        measurement_factory = getattr(external, "Measurement", None)
        if measurement_factory is not None:
            try:
                measurement = measurement_factory(1.0, 0.1, "ns")
            except (ImportError, RuntimeError):
                measurement = None
            if measurement is not None:
                with self.assertRaises(TypeError):
                    TimingJitterConfig(sigma=cast(Any, measurement))

    def test_all_26_physical_fields_are_canonical_and_strict(self) -> None:
        tpc = _valid_configs()[11]
        veto = _valid_configs()[12]
        psd = _valid_configs()[16]
        saturation = _valid_configs()[18]
        digitized = _valid_configs()[20]
        cases: tuple[tuple[Any, str, str], ...] = (
            (TimingJitterConfig(sigma=_ns(1)), "sigma", "ns"),
            (DarkCountConfig(rate=_hz(1)), "rate", "Hz"),
            (FixedDelayConfig(delay=_ns(1)), "delay", "ns"),
            (ExponentialDelayConfig(mean_delay=_ns(1)), "mean_delay", "ns"),
            (
                AfterpulseRecoveryConfig(time_constant=_ns(1)),
                "time_constant",
                "ns",
            ),
            (
                AfterpulseConfig(
                    probability=Probability(0.1),
                    mean_delay=_ns(1),
                ),
                "mean_delay",
                "ns",
            ),
            (tpc, "fast_time_constant", "ns"),
            (tpc, "slow_time_constant", "ns"),
            (tpc, "support_time", "ns"),
            (tpc, "peak_voltage_per_photoelectron", "mV"),
            (veto, "gaussian_center", "ns"),
            (veto, "gaussian_width", "ns"),
            (veto, "edge_offset_1", "ns"),
            (veto, "edge_width_1", "ns"),
            (veto, "edge_offset_2", "ns"),
            (veto, "edge_width_2", "ns"),
            (veto, "support_time", "ns"),
            (veto, "peak_voltage_per_photoelectron", "mV"),
            (WhiteNoiseConfig(rms=_mv(1)), "rms", "mV"),
            (psd, "frequency_left_edges", "Hz"),
            (psd, "frequency_stop", "Hz"),
            (psd, "power_density", "mV ** 2 / Hz"),
            (saturation, "minimum", "mV"),
            (saturation, "maximum", "mV"),
            (digitized, "input_minimum", "mV"),
            (digitized, "input_maximum", "mV"),
        )
        self.assertEqual(len(cases), 26)
        external = pint.UnitRegistry(cache_folder=None)
        vector_fields = {"frequency_left_edges", "power_density"}
        for config, field_name, unit in cases:
            with self.subTest(config=type(config).__name__, field=field_name):
                value = getattr(config, field_name)
                self.assertIsInstance(value, Quantity)
                self.assertEqual(value.units, quantity(1, unit).units)
                if field_name in vector_fields:
                    self.assertIs(type(value.magnitude), np.ndarray)
                    self.assertEqual(value.magnitude.ndim, 1)
                    self.assertIs(value.magnitude.dtype, np.dtype(np.float64))
                    self.assertFalse(value.magnitude.flags.writeable)
                    raw: object = (1.0, 1.0)
                    wrong = external.Quantity(np.array((1.0, 1.0)), "kg")
                    nonscalar = external.Quantity(1.0, unit)
                    nonfinite = external.Quantity(
                        np.array((1.0, float("nan"))),
                        unit,
                    )
                else:
                    self.assertIs(type(value.magnitude), float)
                    raw = 1.0
                    wrong = external.Quantity(1.0, "kg")
                    nonscalar = external.Quantity([1.0], unit)
                    nonfinite = external.Quantity(float("nan"), unit)
                with self.assertRaises(TypeError):
                    replace(config, **{field_name: raw})
                with self.assertRaises(ValueError):
                    replace(config, **{field_name: wrong})
                with self.assertRaises(TypeError):
                    replace(config, **{field_name: nonscalar})
                with self.assertRaises(ValueError):
                    replace(config, **{field_name: nonfinite})

    def test_sign_and_order_boundaries_remain_exact(self) -> None:
        tpc = cast(TpcFebSnrPulseConfig, _valid_configs()[11])
        veto = cast(VetoPduPulseConfig, _valid_configs()[12])
        psd = cast(PsdNoiseConfig, _valid_configs()[16])
        nonnegative = (
            (TimingJitterConfig(sigma=_ns(0)), "sigma", _ns(-1)),
            (DarkCountConfig(rate=_hz(0)), "rate", _hz(-1)),
            (FixedDelayConfig(delay=_ns(0)), "delay", _ns(-1)),
            (
                psd,
                "frequency_left_edges",
                quantities(
                    (-1, float(psd.frequency_left_edges.magnitude[1])),
                    "Hz",
                ),
            ),
            (
                psd,
                "power_density",
                quantities(
                    (-1, float(psd.power_density.magnitude[1])),
                    "mV ** 2 / Hz",
                ),
            ),
        )
        positive = (
            (ExponentialDelayConfig(mean_delay=_ns(1)), "mean_delay", _ns(0)),
            (
                AfterpulseRecoveryConfig(time_constant=_ns(1)),
                "time_constant",
                _ns(0),
            ),
            (
                AfterpulseConfig(
                    probability=Probability(0.1),
                    mean_delay=_ns(1),
                ),
                "mean_delay",
                _ns(0),
            ),
            (tpc, "fast_time_constant", _ns(0)),
            (tpc, "slow_time_constant", _ns(0)),
            (tpc, "support_time", _ns(0)),
            (tpc, "peak_voltage_per_photoelectron", _mv(0)),
            (tpc, "peak_voltage_per_photoelectron", _mv(-1)),
            (veto, "gaussian_width", _ns(0)),
            (veto, "edge_width_1", _ns(0)),
            (veto, "edge_width_2", _ns(0)),
            (veto, "support_time", _ns(0)),
            (veto, "peak_voltage_per_photoelectron", _mv(0)),
            (veto, "peak_voltage_per_photoelectron", _mv(-1)),
            (WhiteNoiseConfig(rms=_mv(1)), "rms", _mv(0)),
            (psd, "frequency_stop", _hz(0)),
        )
        for config, field_name, invalid in (*nonnegative, *positive):
            with self.subTest(config=type(config).__name__, field=field_name):
                with self.assertRaises(ValueError):
                    replace(config, **{field_name: invalid})
        with self.assertRaises(ValueError):
            TpcFebSnrPulseConfig(
                fast_time_constant=_ns(2),
                slow_time_constant=_ns(1),
                support_time=_ns(6),
                peak_voltage_per_photoelectron=_mv(1),
            )
        with self.assertRaises(ValueError):
            AnalogSaturationConfig(minimum=_mv(1), maximum=_mv(1))
        with self.assertRaises(ValueError):
            DigitizedWaveformConfig(
                bit_depth=PositiveInteger(12),
                input_minimum=_mv(1),
                input_maximum=_mv(1),
                analog_gain_db=NonnegativeFloat(0),
            )
        with self.assertRaises(ValueError):
            TpcFebSnrPulseConfig(
                fast_time_constant=_ns(1),
                slow_time_constant=_ns(2),
                support_time=_ns(6),
                peak_voltage_per_photoelectron=_mv(0),
            )
        with self.assertRaises(ValueError):
            replace(veto, peak_voltage_per_photoelectron=_mv(0))

    def test_all_22_public_configs_are_explicitly_unhashable(self) -> None:
        configs = _valid_configs()
        self.assertEqual(len(configs), 22)
        for config in configs:
            with self.subTest(config=type(config).__name__):
                self.assertIs(type(config).__hash__, None)
                with self.assertRaises(TypeError):
                    hash(config)

    def test_sample_axis_period_boundary_and_fresh_integer_accessors(self) -> None:
        for period, expected in (
            (quantity(2, "ns"), 2_000),
            (quantity(0.5, "ns"), 500),
            (quantity(2_000, "ps"), 2_000),
        ):
            with self.subTest(period=period):
                axis = SampleAxis.from_period(period=period, count=3)
                self.assertEqual(axis, SampleAxis(start=0, step=expected, count=3))
        external = pint.UnitRegistry(cache_folder=None)
        self.assertEqual(
            SampleAxis.from_period(
                period=cast(Quantity, external.Quantity(2, "ns")),
                count=3,
            ).step,
            2_000,
        )
        one_ulp = math.nextafter(1_000.0, math.inf)
        two_ulps = math.nextafter(one_ulp, math.inf)
        self.assertEqual(
            SampleAxis.from_period(
                period=cast(Quantity, external.Quantity(one_ulp, "ps")),
                count=3,
            ).step,
            1_000,
        )
        for invalid in (
            external.Quantity(two_ulps, "ps"),
            external.Quantity(1.25, "ps"),
            external.Quantity(float(2**54), "ps"),
        ):
            with self.subTest(period=invalid):
                with self.assertRaises(ValueError):
                    SampleAxis.from_period(
                        period=cast(Quantity, invalid),
                        count=3,
                    )

        large = SampleAxis(start=2**53 + 1, step=2, count=3)
        accessors = (
            (large.start_time, 2**53 + 1),
            (large.sample_period, 2),
            (large.time_at(2), 2**53 + 5),
            (large.stop_time, 2**53 + 7),
        )
        for value, expected in accessors:
            self.assertIs(type(value.magnitude), int)
            self.assertEqual(value.magnitude, expected)
            self.assertEqual(str(value.units), "picosecond")
        self.assertIsNot(large.start_time, large.start_time)
        self.assertIsNot(large.sample_period, large.sample_period)
        self.assertIsNot(large.time_at(1), large.time_at(1))
        self.assertIsNot(large.stop_time, large.stop_time)
        for retired in ("start_ps", "sample_period_ps", "stop_ps"):
            self.assertFalse(hasattr(large, retired))
        round_trip = SampleAxis.from_period(
            period=large.sample_period,
            count=large.count,
        )
        self.assertEqual(round_trip.step, large.step)

    def test_runtime_is_recursively_quantity_and_config_free(self) -> None:
        axes = (
            ExampleAxis(count=1),
            ChannelAxis(labels=("channel-0",)),
            SampleAxis(start=0, step=2_000, count=4),
        )
        source = Photoelectrons(
            tensor=torch.ones((1, 1, 4), dtype=torch.int64),
            axes=axes,
        )
        readout = cast(ReadoutConfig, _valid_configs()[-1])
        self.assertIsInstance(readout, ReadoutConfig)
        _, runtime = prepare_readout(
            source,
            products=(DigitizedWaveform,),
            config=readout,
            rng=Threefry4x32(seed=17),
            floating_dtype=torch.float32,
        )

        seen: set[int] = set()

        def visit(value: Any) -> None:
            if id(value) in seen:
                return
            seen.add(id(value))
            self.assertNotIsInstance(value, (pint.Quantity, pint.UnitRegistry))
            self.assertFalse(type(value).__name__.endswith("Config"))
            if isinstance(value, str):
                self.assertNotIn(
                    value,
                    {"ns", "ps", "Hz", "mV", "mV ** 2 / Hz"},
                )
            if is_dataclass(value) and not isinstance(value, type):
                for field in fields(value):
                    visit(getattr(value, field.name))
            elif isinstance(value, (tuple, list, frozenset)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(key)
                    visit(item)

        visit(runtime)

    def test_active_preparation_extracts_each_physical_operand_once(self) -> None:
        import tensor_dslab.readout.analog_waveform.runtime.prepare as analog
        import tensor_dslab.readout.charge.runtime.effects.correlated_avalanches as correlated
        import tensor_dslab.readout.charge.runtime.effects.dark_counts as dark
        import tensor_dslab.readout.charge.runtime.effects.delays as delays
        import tensor_dslab.readout.charge.runtime.effects.timing_jitter as jitter
        import tensor_dslab.readout.digitized_waveform.runtime.prepare as digitized
        import tensor_dslab.readout.noise_waveform.runtime.prepare as noise
        import tensor_dslab.readout.pure_waveform.runtime.prepare as pure

        sampling = SamplingRuntime(
            sample_dimension=2,
            sample_period_ps=2_000,
            sample_count=4,
        )
        with patch.object(
            jitter,
            "canonical_magnitude",
            wraps=unit_boundary.canonical_magnitude,
        ) as extracted:
            runtime = jitter.prepare_timing_jitter(
                TimingJitterConfig(sigma=_ns(1)),
                sampling=sampling,
                tensor_numel=4,
            )
            self.assertIsNotNone(runtime)
            self.assertEqual(extracted.call_count, 1)
        with patch.object(
            dark,
            "canonical_magnitude",
            wraps=unit_boundary.canonical_magnitude,
        ) as extracted:
            dark.prepare_dark_counts(
                DarkCountConfig(rate=_hz(1.0e6)),
                sampling=sampling,
            )
            self.assertEqual(extracted.call_count, 1)
        for delay_config in (
            FixedDelayConfig(delay=_ns(1)),
            ExponentialDelayConfig(mean_delay=_ns(1)),
        ):
            with self.subTest(delay=type(delay_config).__name__):
                with patch.object(
                    delays,
                    "canonical_magnitude",
                    wraps=unit_boundary.canonical_magnitude,
                ) as extracted:
                    delays.prepare_delay(delay_config, sampling=sampling)
                    self.assertEqual(extracted.call_count, 1)

        afterpulse = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            afterpulse=AfterpulseConfig(
                probability=Probability(0.1),
                mean_delay=_ns(2),
                recovery=AfterpulseRecoveryConfig(time_constant=_ns(4)),
            ),
        )
        with patch.object(
            correlated,
            "canonical_magnitude",
            wraps=unit_boundary.canonical_magnitude,
        ) as extracted:
            correlated.prepare_correlated_avalanches(
                afterpulse,
                sampling=sampling,
                floating_dtype=torch.float64,
                tensor_numel=4,
            )
            self.assertEqual(extracted.call_count, 2)

        tpc = TpcFebSnrPulseConfig(
            fast_time_constant=_ns(1),
            slow_time_constant=_ns(2),
            support_time=_ns(6),
            peak_voltage_per_photoelectron=_mv(1),
        )
        veto = VetoPduPulseConfig(
            gaussian_center=_ns(0),
            gaussian_width=_ns(1),
            edge_offset_1=_ns(-1),
            edge_width_1=_ns(1),
            edge_offset_2=_ns(1),
            edge_width_2=_ns(1),
            support_time=_ns(6),
            peak_voltage_per_photoelectron=_mv(1),
        )
        for model, expected in ((tpc, 4), (veto, 8)):
            with self.subTest(pulse=type(model).__name__):
                with patch.object(
                    pure,
                    "canonical_magnitude",
                    wraps=unit_boundary.canonical_magnitude,
                ) as extracted:
                    pure.prepare_pure_waveform(
                        PureWaveformConfig(model=model),
                        sampling=sampling,
                        floating_dtype=torch.float64,
                        device=torch.device("cpu"),
                    )
                    self.assertEqual(extracted.call_count, expected)

        with patch.object(
            noise,
            "canonical_magnitude",
            wraps=unit_boundary.canonical_magnitude,
        ) as extracted:
            noise.prepare_noise_waveform(
                NoiseWaveformConfig(model=WhiteNoiseConfig(rms=_mv(1))),
                sampling=sampling,
                shape=(1, 1, 4),
                floating_dtype=torch.float64,
                device=torch.device("cpu"),
            )
            self.assertEqual(extracted.call_count, 1)
        with (
            patch.object(
                noise,
                "canonical_magnitude",
                wraps=unit_boundary.canonical_magnitude,
            ) as scalar_extracted,
            patch.object(
                noise,
                "canonical_magnitudes",
                wraps=unit_boundary.canonical_magnitudes,
            ) as vectors_extracted,
        ):
            noise.prepare_noise_waveform(
                NoiseWaveformConfig(
                    model=PsdNoiseConfig(
                        frequency_left_edges=quantities((0, 1.0e8), "Hz"),
                        frequency_stop=_hz(3.0e8),
                        power_density=quantities(
                            (1.0e-9, 2.0e-9),
                            "mV ** 2 / Hz",
                        ),
                    )
                ),
                sampling=sampling,
                shape=(1, 1, 4),
                floating_dtype=torch.float64,
                device=torch.device("cpu"),
            )
            self.assertEqual(scalar_extracted.call_count, 1)
            self.assertEqual(vectors_extracted.call_count, 2)

        with patch.object(
            analog,
            "canonical_magnitude",
            wraps=unit_boundary.canonical_magnitude,
        ) as extracted:
            analog.prepare_analog_waveform(
                AnalogWaveformConfig(
                    saturation=AnalogSaturationConfig(
                        minimum=_mv(-1),
                        maximum=_mv(1),
                    )
                ),
                floating_dtype=torch.float64,
                device=torch.device("cpu"),
            )
            self.assertEqual(extracted.call_count, 2)
        with patch.object(
            digitized,
            "canonical_magnitude",
            wraps=unit_boundary.canonical_magnitude,
        ) as extracted:
            digitized.prepare_digitized_waveform(
                DigitizedWaveformConfig(
                    bit_depth=PositiveInteger(12),
                    input_minimum=_mv(-20),
                    input_maximum=_mv(20),
                    analog_gain_db=NonnegativeFloat(0),
                ),
                floating_dtype=torch.float64,
                device=torch.device("cpu"),
            )
            self.assertEqual(extracted.call_count, 2)

    def test_equivalent_units_preserve_operands_rng_calls_and_products(self) -> None:
        axes = (
            ExampleAxis(count=1),
            ChannelAxis(labels=("channel-0",)),
            SampleAxis(start=0, step=2_000, count=4),
        )
        source = Photoelectrons(
            tensor=torch.ones((1, 1, 4), dtype=torch.int64),
            axes=axes,
        )

        rate_configs = (
            ReadoutConfig(
                charge=ChargeConfig(
                    dark_count=DarkCountConfig(rate=quantity(1, "MHz"))
                )
            ),
            ReadoutConfig(
                charge=ChargeConfig(
                    dark_count=DarkCountConfig(
                        rate=quantity(1_000_000, "Hz")
                    )
                )
            ),
        )
        noise_configs = (
            ReadoutConfig(
                noise_waveform=NoiseWaveformConfig(
                    model=WhiteNoiseConfig(rms=quantity(1, "V"))
                )
            ),
            ReadoutConfig(
                noise_waveform=NoiseWaveformConfig(
                    model=WhiteNoiseConfig(rms=quantity(1_000, "mV"))
                )
            ),
        )

        for configs, product in (
            (rate_configs, Charge),
            (noise_configs, NoiseWaveform),
        ):
            runtimes = tuple(
                prepare_readout(
                    source,
                    products=(product,),
                    config=config,
                    rng=_RecordingRng(seed=17),
                    floating_dtype=torch.float64,
                )[1]
                for config in configs
            )
            if product is Charge:
                assert runtimes[0].charge is not None
                assert runtimes[1].charge is not None
                self.assertEqual(
                    runtimes[0].charge.dark,
                    runtimes[1].charge.dark,
                )
            else:
                assert runtimes[0].noise_waveform is not None
                assert runtimes[1].noise_waveform is not None
                self.assertEqual(
                    runtimes[0].noise_waveform.model,
                    runtimes[1].noise_waveform.model,
                )

            tensors: list[torch.Tensor] = []
            call_sets: list[tuple[tuple[RngKey, torch.Tensor, int, int], ...]] = []
            for config in configs:
                _RecordingRng.reset()
                result = simulate_readout(
                    source,
                    products=(product,),
                    config=config,
                    rng=_RecordingRng(seed=17),
                    floating_dtype=torch.float64,
                )
                tensors.append(result.field(product).tensor)
                call_sets.append(tuple(_RecordingRng.calls))
            self.assertTrue(torch.equal(tensors[0], tensors[1]))
            self.assertEqual(len(call_sets[0]), len(call_sets[1]))
            self.assertGreater(len(call_sets[0]), 0)
            for left, right in zip(call_sets[0], call_sets[1], strict=True):
                self.assertEqual(left[0], right[0])
                self.assertTrue(torch.equal(left[1], right[1]))
                self.assertEqual(left[2:], right[2:])

        two_ns = TimingJitterConfig(sigma=quantity(2, "ns"))
        two_thousand_ps = TimingJitterConfig(sigma=quantity(2_000, "ps"))
        self.assertEqual(two_ns.sigma.magnitude.hex(), "0x1.0000000000000p+1")
        self.assertEqual(
            two_thousand_ps.sigma.magnitude.hex(),
            "0x1.ffffffffffffep+0",
        )
        self.assertNotEqual(two_ns, two_thousand_ps)
        self.assertEqual(
            SampleAxis.from_period(period=quantity(2, "ns"), count=4),
            SampleAxis.from_period(period=quantity(2_000, "ps"), count=4),
        )

    def test_exports_retired_names_and_static_unit_boundary_are_exact(self) -> None:
        from tensor_dslab.readout.analog_waveform.runtime.prepare import (
            AnalogWaveformRuntime,
        )
        from tensor_dslab.readout.digitized_waveform.runtime.prepare import (
            DigitizedWaveformRuntime,
        )
        from tensor_dslab.readout.noise_waveform.runtime.prepare import (
            PsdNoiseRuntime,
            WhiteNoiseRuntime,
        )

        self.assertEqual(len(tensor_dslab.__all__), 35)
        self.assertEqual(len(common.__all__), 5)
        self.assertIs(tensor_dslab.quantity, common.quantity)
        self.assertIs(tensor_dslab.quantities, common.quantities)
        self.assertEqual(
            tuple(WhiteNoiseRuntime.__dataclass_fields__),
            ("rng_key", "represented_rms_mv"),
        )
        self.assertEqual(
            tuple(PsdNoiseRuntime.__dataclass_fields__),
            ("rng_key", "represented_powers_mv2"),
        )
        self.assertEqual(
            tuple(AnalogWaveformRuntime.__dataclass_fields__),
            ("minimum_mv", "maximum_mv"),
        )
        self.assertEqual(
            tuple(DigitizedWaveformRuntime.__dataclass_fields__),
            (
                "maximum_code",
                "zero",
                "maximum",
                "slope_per_mv",
                "intercept",
                "lower_input_mv",
                "upper_input_mv",
            ),
        )
        for retired in (
            "sigma_ns",
            "rate_hz",
            "delay_ns",
            "mean_delay_ns",
            "time_constant_ns",
            "rms_mv",
            "frequency_left_edges_hz",
            "frequency_stop_hz",
            "power_density_mv2_per_hz",
            "input_min_mv",
            "input_max_mv",
        ):
            for config in _valid_configs():
                self.assertNotIn(retired, config.__dataclass_fields__)

        source = Path("tensor_dslab/common/units.py").read_text()
        tree = ast.parse(source)
        registries = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "UnitRegistry"
        ]
        self.assertEqual(len(registries), 1)
        self.assertNotIn("set_application_registry", source)
        retired_requirements = {
            "require_exact",
            "require_optional_exact",
            "require_one_of_exact",
        }
        requirements_tree = ast.parse(
            Path("tensor_dslab/readout/runtime/requirements.py").read_text()
        )
        self.assertFalse(
            retired_requirements
            & {
                node.name
                for node in requirements_tree.body
                if isinstance(node, ast.FunctionDef)
            }
        )
        for path in (
            Path("tensor_dslab/readout/config.py"),
            *Path("tensor_dslab/readout").glob("*/config.py"),
        ):
            module = ast.parse(path.read_text())
            calls = {
                node.func.id
                for node in ast.walk(module)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
            }
            self.assertFalse(retired_requirements & calls, path)
        for path in Path("tensor_dslab/readout").rglob("*.py"):
            if path.name not in ("produce.py", "validate.py"):
                continue
            module = ast.parse(path.read_text())
            imports = (
                node
                for node in ast.walk(module)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            for imported in imports:
                if isinstance(imported, ast.Import):
                    names = tuple(alias.name for alias in imported.names)
                else:
                    names = (imported.module or "",)
                self.assertFalse(
                    any(name == "pint" or name.startswith("pint.") for name in names),
                    path,
                )
                self.assertNotIn("tensor_dslab.common.units", names, path)

        charge_prepare = Path(
            "tensor_dslab/readout/charge/runtime/prepare.py"
        ).read_text()
        self.assertNotIn(".sigma", charge_prepare)
        for path, function_name in (
            (
                "tensor_dslab/readout/charge/runtime/prepare.py",
                "prepare_charge",
            ),
            (
                "tensor_dslab/readout/pure_waveform/runtime/prepare.py",
                "prepare_pure_waveform",
            ),
            (
                "tensor_dslab/readout/noise_waveform/runtime/prepare.py",
                "prepare_noise_waveform",
            ),
            (
                "tensor_dslab/readout/analog_waveform/runtime/prepare.py",
                "prepare_analog_waveform",
            ),
            (
                "tensor_dslab/readout/digitized_waveform/runtime/prepare.py",
                "prepare_digitized_waveform",
            ),
        ):
            module = ast.parse(Path(path).read_text())
            function = next(
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef)
                and node.name == function_name
            )
            text = ast.unparse(function)
            self.assertNotIn("type(config)", text, path)
            self.assertNotIn(
                "floating_dtype not in (torch.float32, torch.float64)",
                text,
                path,
            )
            self.assertNotIn("device.type not in", text, path)

        for path, function_name in (
            (
                "tensor_dslab/readout/charge/runtime/effects/dark_counts.py",
                "simulate_dark_counts",
            ),
            (
                "tensor_dslab/readout/charge/runtime/effects/timing_jitter.py",
                "simulate_timing_jitter",
            ),
            (
                "tensor_dslab/readout/charge/runtime/effects/"
                "correlated_avalanches.py",
                "simulate_correlated_avalanches",
            ),
            (
                "tensor_dslab/readout/charge/runtime/effects/smearing.py",
                "simulate_charge_smearing",
            ),
        ):
            module = ast.parse(Path(path).read_text())
            function = next(
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef)
                and node.name == function_name
            )
            text = ast.unparse(function)
            self.assertNotIn("type(runtime)", text, path)
            self.assertNotIn("type(sample_dimension)", text, path)
