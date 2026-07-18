from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
import importlib.util
from inspect import Parameter, signature
from typing import ClassVar
import unittest
from unittest import mock

import torch
from tensor_core import (
    CounterRng,
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    PositiveInteger,
    Probability,
    RngKey,
    Threefry4x32,
    logical_positions,
)

import tensor_dslab
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
    DigitizedWaveformConfig,
    DirectCrosstalkConfig,
    ExampleAxis,
    ExponentialDelayConfig,
    FixedDelayConfig,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    SamplingConfig,
    TimingJitterConfig,
    TpcFebSnrPulseConfig,
    VetoPduPulseConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.analog_waveform._produce import (
    _produce_analog_waveform,
)
from tensor_dslab.readout.charge._produce import _produce_charge
from tensor_dslab.readout.digitized_waveform._produce import (
    _produce_digitized_waveform,
)
from tensor_dslab.readout.noise_waveform._produce import (
    _produce_noise_waveform,
)
from tensor_dslab.readout.pure_waveform._produce import (
    _produce_pure_waveform,
)


_NAMESPACE = 0x54445331
_SEED = 0x0123456789ABCDEF


def _hex_bits(values: torch.Tensor) -> tuple[str, ...]:
    if values.dtype is torch.float32:
        integers = values.contiguous().view(torch.int32).reshape(-1)
        return tuple(f"{int(value) & 0xFFFFFFFF:08x}" for value in integers)
    if values.dtype is torch.float64:
        integers = values.contiguous().view(torch.int64).reshape(-1)
        return tuple(f"{int(value) & 0xFFFFFFFFFFFFFFFF:016x}" for value in integers)
    raise TypeError("bit-pattern fixture requires float32 or float64")


def _source() -> Photoelectrons:
    axes = (
        ExampleAxis(coordinates=("event-0",)),
        ChannelAxis(coordinates=("channel-0",)),
        SampleAxis(coordinates=("0ps", "2000ps", "4000ps", "6000ps")),
    )
    return Photoelectrons(
        tensor=torch.tensor([[[3, 0, 1, 2]]], dtype=torch.int64),
        axes=axes,
    )


def _sampling() -> SamplingConfig:
    return SamplingConfig(
        sample_period_ps=PositiveInteger(2000),
        sample_count=PositiveInteger(4),
    )


class _RecordingRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[list[tuple[RngKey, torch.Tensor, int, int]]] = []

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


class _FailingRng(CounterRng):
    __slots__ = ()

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        raise AssertionError(
            f"unexpected RNG request: {key=}, {quantum=}, {block=}"
        )


class RngOwnershipMigrationTest(unittest.TestCase):
    def test_exact_config_owned_key_defaults_overrides_and_identity(self) -> None:
        fixed = FixedDelayConfig(delay_ns=NonnegativeFloat(0.0))
        direct = DirectCrosstalkConfig(
            mean_offspring_per_parent=NonnegativeFloat(0.1),
            delay=fixed,
        )
        delayed = DelayedCrosstalkConfig(
            mean_offspring_per_parent=NonnegativeFloat(0.1),
            delay=fixed,
        )
        keyed = (
            (
                WhiteNoiseConfig(rms_mv=PositiveFloat(1.0)),
                "rng_key",
                1,
            ),
            (
                PsdNoiseConfig(
                    frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                    frequency_stop_hz=PositiveFloat(1.0),
                    power_density_mv2_per_hz=(NonnegativeFloat(1.0),),
                ),
                "rng_key",
                2,
            ),
            (
                DarkCountConfig(rate_hz=NonnegativeFloat(0.0)),
                "rng_key",
                3,
            ),
            (direct, "retained_rng_key", 4),
            (direct, "overflow_rng_key", 5),
            (delayed, "retained_rng_key", 6),
            (delayed, "overflow_rng_key", 7),
            (
                TimingJitterConfig(sigma_ns=NonnegativeFloat(0.0)),
                "rng_key",
                8,
            ),
            (
                AfterpulseConfig(
                    probability=Probability(0.0),
                    mean_delay_ns=PositiveFloat(1.0),
                ),
                "rng_key",
                9,
            ),
            (
                ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(0.0)
                ),
                "rng_key",
                10,
            ),
        )
        for index, (baseline, field_name, stream) in enumerate(keyed):
            with self.subTest(config=type(baseline).__name__, field=field_name):
                default = getattr(baseline, field_name)
                self.assertEqual(
                    (default.namespace, default.stream),
                    (_NAMESPACE, stream),
                )

                override = RngKey(namespace=0x1357_2468, stream=101 + index)
                other = RngKey(namespace=0x1357_2468, stream=201 + index)
                first = replace(baseline, **{field_name: override})
                second = replace(baseline, **{field_name: override})
                changed = replace(baseline, **{field_name: other})
                self.assertIs(getattr(first, field_name), override)
                self.assertEqual(first, second)
                self.assertNotEqual(first, changed)
                self.assertIn(f"{field_name}={override!r}", repr(first))
                with self.assertRaises(TypeError):
                    replace(baseline, **{field_name: 1})

        for config in (direct, delayed):
            with self.subTest(config=type(config).__name__):
                retained = RngKey(namespace=7, stream=11)
                overflow = RngKey(namespace=7, stream=11)
                self.assertIsNot(retained, overflow)
                self.assertEqual(retained, overflow)
                with self.assertRaises(ValueError):
                    replace(
                        config,
                        retained_rng_key=retained,
                        overflow_rng_key=overflow,
                    )

    def test_all_config_owned_keys_reach_the_exact_public_distribution(self) -> None:
        keys = tuple(
            RngKey(namespace=0x2468_1357, stream=101 + index)
            for index in range(10)
        )
        (
            white_key,
            psd_key,
            dark_key,
            direct_retained_key,
            direct_overflow_key,
            delayed_retained_key,
            delayed_overflow_key,
            jitter_key,
            afterpulse_key,
            smearing_key,
        ) = keys
        source = _source()
        sampling = _sampling()

        def noise(config: NoiseWaveformConfig) -> Callable[[CounterRng], object]:
            return lambda rng: _produce_noise_waveform(
                source,
                sampling=sampling,
                config=config,
                rng=rng,
                floating_dtype=torch.float32,
            )

        def charge(config: ChargeConfig) -> Callable[[CounterRng], object]:
            return lambda rng: _produce_charge(
                source,
                sampling=sampling,
                config=config,
                rng=rng,
                floating_dtype=torch.float32,
            )

        cases: tuple[
            tuple[
                str,
                str,
                tuple[RngKey, ...],
                Callable[[CounterRng], object],
            ],
            ...,
        ] = (
            (
                "white noise",
                "gaussian",
                (white_key,),
                noise(
                    NoiseWaveformConfig(
                        model=WhiteNoiseConfig(
                            rms_mv=PositiveFloat(1.0),
                            rng_key=white_key,
                        )
                    )
                ),
            ),
            (
                "PSD noise",
                "gaussian",
                (psd_key,),
                noise(
                    NoiseWaveformConfig(
                        model=PsdNoiseConfig(
                            frequency_left_edges_hz=(
                                NonnegativeFloat(0.0),
                                NonnegativeFloat(100_000_000.0),
                            ),
                            frequency_stop_hz=PositiveFloat(250_000_000.0),
                            power_density_mv2_per_hz=(
                                NonnegativeFloat(1.0e-8),
                                NonnegativeFloat(2.0e-8),
                            ),
                            rng_key=psd_key,
                        )
                    )
                ),
            ),
            (
                "dark counts",
                "poisson",
                (dark_key,),
                charge(
                    ChargeConfig(
                        dark_count=DarkCountConfig(
                            rate_hz=NonnegativeFloat(5.0e8),
                            rng_key=dark_key,
                        )
                    )
                ),
            ),
            (
                "direct crosstalk",
                "poisson",
                (direct_retained_key, direct_overflow_key),
                charge(
                    ChargeConfig(
                        correlated_avalanches=CorrelatedAvalancheConfig(
                            maximum_generations=NonnegativeInteger(1),
                            direct_crosstalk=DirectCrosstalkConfig(
                                mean_offspring_per_parent=NonnegativeFloat(0.1),
                                delay=FixedDelayConfig(
                                    delay_ns=NonnegativeFloat(0.0)
                                ),
                                retained_rng_key=direct_retained_key,
                                overflow_rng_key=direct_overflow_key,
                            ),
                        )
                    )
                ),
            ),
            (
                "delayed crosstalk",
                "poisson",
                (delayed_retained_key, delayed_overflow_key),
                charge(
                    ChargeConfig(
                        correlated_avalanches=CorrelatedAvalancheConfig(
                            maximum_generations=NonnegativeInteger(1),
                            delayed_crosstalk=DelayedCrosstalkConfig(
                                mean_offspring_per_parent=NonnegativeFloat(0.1),
                                delay=FixedDelayConfig(
                                    delay_ns=NonnegativeFloat(0.0)
                                ),
                                retained_rng_key=delayed_retained_key,
                                overflow_rng_key=delayed_overflow_key,
                            ),
                        )
                    )
                ),
            ),
            (
                "timing jitter",
                "binomial",
                (jitter_key,),
                charge(
                    ChargeConfig(
                        timing_jitter=TimingJitterConfig(
                            sigma_ns=NonnegativeFloat(1.0),
                            rng_key=jitter_key,
                        )
                    )
                ),
            ),
            (
                "afterpulse",
                "binomial",
                (afterpulse_key,),
                charge(
                    ChargeConfig(
                        correlated_avalanches=CorrelatedAvalancheConfig(
                            maximum_generations=NonnegativeInteger(1),
                            afterpulse=AfterpulseConfig(
                                probability=Probability(0.5),
                                mean_delay_ns=PositiveFloat(1.0),
                                rng_key=afterpulse_key,
                            ),
                        )
                    )
                ),
            ),
            (
                "charge smearing",
                "gaussian",
                (smearing_key,),
                charge(
                    ChargeConfig(
                        smearing=ChargeSmearingConfig(
                            relative_sigma=NonnegativeFloat(0.1),
                            rng_key=smearing_key,
                        )
                    )
                ),
            ),
        )

        for name, distribution_name, expected_keys, invoke in cases:
            with self.subTest(role=name):
                rng = Threefry4x32(seed=_SEED)
                original = getattr(CounterRng, distribution_name)
                with mock.patch.object(
                    CounterRng,
                    distribution_name,
                    autospec=True,
                    side_effect=original,
                ) as distribution:
                    invoke(rng)
                calls = distribution.call_args_list
                self.assertTrue(calls)
                self.assertTrue(all(call.args[0] is rng for call in calls))
                observed_keys = tuple(call.kwargs["key"] for call in calls)
                if len(expected_keys) == 1:
                    self.assertTrue(
                        all(key is expected_keys[0] for key in observed_keys)
                    )
                else:
                    self.assertEqual(len(observed_keys), len(expected_keys))
                    self.assertTrue(
                        all(
                            actual is expected
                            for actual, expected in zip(
                                observed_keys,
                                expected_keys,
                                strict=True,
                            )
                        )
                    )

    def test_complete_public_config_inventory_has_exact_key_ownership(self) -> None:
        no_key_fields = {
            SamplingConfig: ("sample_period_ps", "sample_count"),
            ReadoutConfig: (
                "sampling",
                "charge",
                "pure_waveform",
                "noise_waveform",
                "analog_waveform",
                "digitized_waveform",
            ),
            FixedDelayConfig: ("delay_ns",),
            ExponentialDelayConfig: ("mean_delay_ns",),
            AfterpulseRecoveryConfig: ("time_constant_ns",),
            CorrelatedAvalancheConfig: (
                "maximum_generations",
                "direct_crosstalk",
                "delayed_crosstalk",
                "afterpulse",
            ),
            ChargeConfig: (
                "dark_count",
                "timing_jitter",
                "correlated_avalanches",
                "smearing",
            ),
            ZeroNoiseConfig: (),
            NoiseWaveformConfig: ("model",),
            TpcFebSnrPulseConfig: (
                "fast_time_constant_ns",
                "slow_time_constant_ns",
                "support_time_ns",
                "peak_voltage_mv_per_pe",
            ),
            VetoPduPulseConfig: (
                "gaussian_center_ns",
                "gaussian_width_ns",
                "edge_offset_1_ns",
                "edge_width_1_ns",
                "edge_offset_2_ns",
                "edge_width_2_ns",
                "support_time_ns",
                "peak_voltage_mv_per_pe",
            ),
            PureWaveformConfig: ("model",),
            AnalogSaturationConfig: ("minimum_mv", "maximum_mv"),
            AnalogWaveformConfig: ("saturation",),
            DigitizedWaveformConfig: (
                "bit_depth",
                "input_min_mv",
                "input_max_mv",
                "analog_gain_db",
            ),
        }
        keyed_fields = {
            WhiteNoiseConfig: ("rms_mv", "rng_key"),
            PsdNoiseConfig: (
                "frequency_left_edges_hz",
                "frequency_stop_hz",
                "power_density_mv2_per_hz",
                "rng_key",
            ),
            TimingJitterConfig: ("sigma_ns", "rng_key"),
            DarkCountConfig: ("rate_hz", "rng_key"),
            DirectCrosstalkConfig: (
                "mean_offspring_per_parent",
                "delay",
                "retained_rng_key",
                "overflow_rng_key",
            ),
            DelayedCrosstalkConfig: (
                "mean_offspring_per_parent",
                "delay",
                "retained_rng_key",
                "overflow_rng_key",
            ),
            AfterpulseConfig: (
                "probability",
                "mean_delay_ns",
                "recovery",
                "rng_key",
            ),
            ChargeSmearingConfig: ("relative_sigma", "rng_key"),
        }
        expected_public_names = {
            config_type.__name__
            for config_type in (*no_key_fields, *keyed_fields)
        }
        actual_public_names = {
            name
            for name in tensor_dslab.__all__
            if name.endswith("Config")
        }
        self.assertTrue(set(keyed_fields).isdisjoint(no_key_fields))
        self.assertEqual(actual_public_names, expected_public_names)
        for name in actual_public_names:
            self.assertEqual(getattr(tensor_dslab, name).__name__, name)
        for config_type, expected_fields in (
            *no_key_fields.items(),
            *keyed_fields.items(),
        ):
            with self.subTest(config=config_type.__name__):
                self.assertEqual(
                    tuple(field.name for field in fields(config_type)),
                    expected_fields,
                )

    def test_public_tensorcore_distribution_continuity(self) -> None:
        rng = Threefry4x32(seed=_SEED)
        positions = torch.tensor(
            [0, 1, 2, 4_294_967_299],
            dtype=torch.int64,
        )
        self.assertTrue(
            torch.equal(
                positions,
                logical_positions((4,), device="cpu").index_copy(
                    0,
                    torch.tensor([3]),
                    torch.tensor([4_294_967_299]),
                ),
            )
        )

        key1 = RngKey(namespace=_NAMESPACE, stream=1)
        expected_uniform = {
            (torch.float32, True): (
                "3ecdc482", "3f7ace57", "3f28f330", "3ed8ba00"
            ),
            (torch.float32, False): (
                "3ecdc482", "3f7ace57", "3f28f331", "3ed8ba02"
            ),
            (torch.float64, True): (
                "3fd9b8905cff7a8c", "3fef59caf40108e5",
                "3fe51e6600fe5534", "3fdb17402dca4490",
            ),
            (torch.float64, False): (
                "3fd9b8905cff7a8e", "3fef59caf40108e5",
                "3fe51e6600fe5535", "3fdb17402dca4492",
            ),
        }
        for (dtype, include_zero), expected in expected_uniform.items():
            actual = rng.uniform(
                key=key1,
                positions=positions,
                dtype=dtype,
                quantum=0,
                ordinal=0,
                count=1,
                include_zero=include_zero,
            )
            self.assertEqual(_hex_bits(actual), expected)

        expected_gaussian = {
            torch.float32: (
                "3f81741c", "3e184d06", "3f25bf13", "3f06517b"
            ),
            torch.float64: (
                "3fcf75d99582d78a", "3f89fe5e8e452724",
                "3fe178009eac0f2d", "bfdc45b21a9f07da",
            ),
        }
        for dtype, expected in expected_gaussian.items():
            actual = rng.gaussian(
                mean=0.0,
                standard_deviation=0.75,
                key=key1,
                positions=positions,
                dtype=dtype,
                quantum=0,
                ordinal=0,
                count=1,
            )
            self.assertEqual(_hex_bits(actual), expected)

        key2 = RngKey(namespace=_NAMESPACE, stream=2)
        expected_pair = {
            torch.float32: (
                "bfe0b1d4", "bf186554", "bf01b88f", "4018c5a8",
                "bef8a544", "3f1953a0", "3f9c0e27", "3fb37fb6",
            ),
            torch.float64: (
                "bff73a00f4cae65c", "bff2713778ea1903",
                "c002d5befcdee864", "3fe488e59e06900e",
                "bfe2b75e6f86914b", "3fe01435868a425e",
                "3ffabc480d2e0237", "bfea0354fa54a537",
            ),
        }
        for dtype, expected in expected_pair.items():
            actual = rng.gaussian(
                mean=0.0,
                standard_deviation=1.0,
                key=key2,
                positions=positions,
                dtype=dtype,
                quantum=0,
                ordinal=0,
                count=2,
            )
            self.assertEqual(_hex_bits(actual), expected)

        means = torch.tensor([0.0, 0.75, 9.5, 25.0], dtype=torch.float64)
        expected_poisson = {
            3: (0, 2, 12, 30),
            4: (0, 0, 8, 25),
            5: (0, 1, 7, 29),
            6: (0, 2, 11, 17),
            7: (0, 4, 4, 26),
        }
        for stream, expected in expected_poisson.items():
            actual = rng.poisson(
                mean=means,
                key=RngKey(namespace=_NAMESPACE, stream=stream),
                positions=positions,
                quantum=0,
            )
            self.assertEqual(tuple(int(value) for value in actual), expected)

        counts = torch.tensor([0, 3, 20, 100], dtype=torch.int64)
        success = torch.tensor([0.0, 0.25, 0.9, 0.2], dtype=torch.float64)
        failure = torch.tensor([0.0, 0.75, 0.1, 0.8], dtype=torch.float64)
        for stream, expected in ((8, (0, 1, 19, 17)), (9, (0, 1, 16, 23))):
            actual = rng.binomial(
                counts=counts,
                success_mass=success,
                failure_mass=failure,
                key=RngKey(namespace=_NAMESPACE, stream=stream),
                positions=positions,
                quantum=0,
            )
            self.assertEqual(tuple(int(value) for value in actual), expected)

        gaussian_means = (0.25, -1.0, 3.5, 0.0)
        gaussian_scales = (0.5, 0.25, 1.5, 2.0)
        expected_stream10 = {
            torch.float32: (
                "3e936e95", "bfab1848", "400b2629", "406d6d29"
            ),
            torch.float64: (
                "3fe0d430c98c26d9", "bff0364ad8f2cf8c",
                "400aefb6b143ae4d", "4001eb62399da026",
            ),
        }
        for dtype, expected in expected_stream10.items():
            actual = rng.gaussian(
                mean=torch.tensor(gaussian_means, dtype=dtype),
                standard_deviation=torch.tensor(gaussian_scales, dtype=dtype),
                key=RngKey(namespace=_NAMESPACE, stream=10),
                positions=positions,
                dtype=dtype,
                quantum=0,
                ordinal=0,
                count=1,
            )
            self.assertEqual(_hex_bits(actual), expected)

    def test_public_tensorcore_zero_dimension_address_span(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "logical_positions shape span must be less than 2\\*\\*63",
        ):
            logical_positions((0, 1 << 62, 2), device="cpu")

        positions = logical_positions((0, 1 << 62), device="cpu")
        self.assertEqual(tuple(positions.shape), (0, 1 << 62))
        self.assertEqual(positions.numel(), 0)
        with self.assertRaisesRegex(
            ValueError,
            "result shape span must be less than 2\\*\\*63",
        ):
            _FailingRng(seed=_SEED).gaussian(
                mean=0.0,
                standard_deviation=1.0,
                key=RngKey(namespace=_NAMESPACE, stream=1),
                positions=positions,
                dtype=torch.float32,
                quantum=0,
                ordinal=0,
                count=2,
            )

    def test_public_producer_signatures_and_draw_free_branches(self) -> None:
        for producer in (_produce_noise_waveform, _produce_charge):
            parameters = signature(producer).parameters
            self.assertIn("rng", parameters)
            self.assertNotIn("seed", parameters)
            self.assertIs(parameters["rng"].kind, Parameter.KEYWORD_ONLY)
            self.assertIs(parameters["rng"].default, Parameter.empty)
        for producer in (
            _produce_pure_waveform,
            _produce_analog_waveform,
            _produce_digitized_waveform,
        ):
            parameters = signature(producer).parameters
            self.assertNotIn("rng", parameters)
            self.assertNotIn("seed", parameters)

        source = _source()
        rng = _FailingRng(seed=0)
        zero = _produce_noise_waveform(
            source,
            sampling=_sampling(),
            config=NoiseWaveformConfig(model=ZeroNoiseConfig()),
            rng=rng,
            floating_dtype=torch.float32,
        )
        charge = _produce_charge(
            source,
            sampling=_sampling(),
            config=ChargeConfig(),
            rng=rng,
            floating_dtype=torch.float32,
        )
        self.assertTrue(torch.equal(zero.tensor, torch.zeros_like(zero.tensor)))
        self.assertTrue(
            torch.equal(charge.tensor, source.tensor.to(dtype=torch.float32))
        )

    def test_exact_zero_charge_branches_are_fresh_and_draw_free(self) -> None:
        source = _source()
        original = source.tensor.clone()
        configs = (
            (
                "dark_count",
                ChargeConfig(
                    dark_count=DarkCountConfig(
                        rate_hz=NonnegativeFloat(0.0)
                    )
                ),
            ),
            (
                "timing_jitter",
                ChargeConfig(
                    timing_jitter=TimingJitterConfig(
                        sigma_ns=NonnegativeFloat(0.0)
                    )
                ),
            ),
            (
                "smearing",
                ChargeConfig(
                    smearing=ChargeSmearingConfig(
                        relative_sigma=NonnegativeFloat(0.0)
                    )
                ),
            ),
        )
        for branch, config in configs:
            for floating_dtype in (torch.float32, torch.float64):
                with self.subTest(
                    branch=branch,
                    floating_dtype=floating_dtype,
                ):
                    result = _produce_charge(
                        source,
                        sampling=_sampling(),
                        config=config,
                        rng=_FailingRng(seed=_SEED),
                        floating_dtype=floating_dtype,
                    )
                    self.assertIs(type(result), Charge)
                    self.assertIs(result.tensor.dtype, floating_dtype)
                    self.assertTrue(
                        torch.equal(
                            result.tensor,
                            source.tensor.to(dtype=floating_dtype),
                        )
                    )
                    self.assertIsNot(result.tensor, source.tensor)
                    self.assertNotEqual(
                        result.tensor.untyped_storage().data_ptr(),
                        source.tensor.untyped_storage().data_ptr(),
                    )
                    self.assertTrue(torch.equal(source.tensor, original))

    def test_completed_noise_and_charge_eager_cpu_continuity(self) -> None:
        source = _source()
        rng = Threefry4x32(seed=_SEED)
        white_config = NoiseWaveformConfig(
            model=WhiteNoiseConfig(rms_mv=PositiveFloat(0.75))
        )
        psd_config = NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges_hz=(
                    NonnegativeFloat(0.0),
                    NonnegativeFloat(100_000_000.0),
                ),
                frequency_stop_hz=PositiveFloat(250_000_000.0),
                power_density_mv2_per_hz=(
                    NonnegativeFloat(1.0e-8),
                    NonnegativeFloat(2.0e-8),
                ),
            )
        )
        expected_white = {
            torch.float32: ("3f81741c", "3e184d06", "3f25bf13", "3e499154"),
            torch.float64: (
                "3fcf75d99582d78a", "3f89fe5e8e452724",
                "3fe178009eac0f2d", "3f851d1a3e211f04",
            ),
        }
        expected_psd = {
            torch.float32: ("bfa40c51", "c03bf3d0", "3e486878", "4080b9b9"),
            torch.float64: (
                "c01058090a5b3d75", "bfd20487b1837bec",
                "4006399d2ea79a9c", "3ff96e0bb87e9f95",
            ),
        }
        for dtype in (torch.float32, torch.float64):
            white = _produce_noise_waveform(
                source,
                sampling=_sampling(),
                config=white_config,
                rng=rng,
                floating_dtype=dtype,
            )
            psd = _produce_noise_waveform(
                source,
                sampling=_sampling(),
                config=psd_config,
                rng=rng,
                floating_dtype=dtype,
            )
            self.assertEqual(_hex_bits(white.tensor), expected_white[dtype])
            self.assertEqual(_hex_bits(psd.tensor), expected_psd[dtype])

        charge_config = ChargeConfig(
            dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(5.0e8)),
            timing_jitter=TimingJitterConfig(sigma_ns=NonnegativeFloat(1.0)),
            correlated_avalanches=CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(2),
                direct_crosstalk=DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.6),
                    delay=ExponentialDelayConfig(
                        mean_delay_ns=PositiveFloat(2.5)
                    ),
                ),
                delayed_crosstalk=DelayedCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.4),
                    delay=ExponentialDelayConfig(
                        mean_delay_ns=PositiveFloat(4.0)
                    ),
                ),
                afterpulse=AfterpulseConfig(
                    probability=Probability(0.35),
                    mean_delay_ns=PositiveFloat(3.0),
                    recovery=AfterpulseRecoveryConfig(
                        time_constant_ns=PositiveFloat(5.0)
                    ),
                ),
            ),
            smearing=ChargeSmearingConfig(
                relative_sigma=NonnegativeFloat(0.1)
            ),
        )
        expected_charge = {
            torch.float32: ("40a08b0b", "40b571a7", "40f01acc", "4161c37e"),
            torch.float64: (
                "40147e5936eabbbe", "4017f2b37dd7adc7",
                "401edf582a1ee0b5", "402c390c96702ab5",
            ),
        }
        for dtype in (torch.float32, torch.float64):
            charge = _produce_charge(
                source,
                sampling=_sampling(),
                config=charge_config,
                rng=rng,
                floating_dtype=dtype,
            )
            self.assertEqual(_hex_bits(charge.tensor), expected_charge[dtype])

    def test_recording_hook_observes_config_key_and_retired_modules_are_absent(
        self,
    ) -> None:
        _RecordingRng.calls = []
        source = _source()
        key = RngKey(namespace=19, stream=23)
        _produce_noise_waveform(
            source,
            sampling=_sampling(),
            config=NoiseWaveformConfig(
                model=WhiteNoiseConfig(
                    rms_mv=PositiveFloat(1.0),
                    rng_key=key,
                )
            ),
            rng=_RecordingRng(seed=0),
            floating_dtype=torch.float32,
        )
        self.assertTrue(_RecordingRng.calls)
        self.assertTrue(all(call[0] == key for call in _RecordingRng.calls))
        self.assertTrue(all(call[2] == 0 for call in _RecordingRng.calls))

        retired = (
            "tensor_dslab.readout._random",
            "tensor_dslab.readout._rng",
            "tensor_dslab.readout.types",
            "tensor_dslab.readout.photoelectrons.types",
            "tensor_dslab.readout.charge.types",
            "tensor_dslab.readout.pure_waveform.types",
            "tensor_dslab.readout.noise_waveform.types",
            "tensor_dslab.readout.analog_waveform.types",
            "tensor_dslab.readout.digitized_waveform.types",
        )
        self.assertEqual(
            tuple(name for name in retired if importlib.util.find_spec(name)),
            (),
        )


if __name__ == "__main__":
    unittest.main()
