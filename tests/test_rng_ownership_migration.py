from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
import importlib.util
from inspect import Parameter, signature
import platform
import sys
from typing import Any, ClassVar
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
)
from tensor_dslab.readout.analog_waveform.runtime.produce import (
    produce_analog_waveform as _produce_analog_waveform_prepared,
)
from tensor_dslab.readout.charge.runtime.prepare import prepare_charge
from tensor_dslab.readout.charge.runtime.produce import (
    produce_charge as _produce_charge_prepared,
)
from tensor_dslab.readout.charge.runtime.validate import validate_charge
from tensor_dslab.readout.digitized_waveform.runtime.produce import (
    produce_digitized_waveform as _produce_digitized_waveform_prepared,
)
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    prepare_noise_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.produce import (
    produce_noise_waveform as _produce_noise_waveform_prepared,
)
from tensor_dslab.readout.noise_waveform.runtime.validate import (
    validate_noise_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.produce import (
    produce_pure_waveform as _produce_pure_waveform_prepared,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime, prepare_sampling


_NAMESPACE = 0x54445331
_SEED = 0x0123456789ABCDEF


def _is_maintenance_2_reference_stack() -> bool:
    return (
        sys.version_info[:3] == (3, 13, 11)
        and str(torch.__version__) == "2.12.1"
        and torch.version.cuda is None
        and platform.system() == "Darwin"
        and platform.mac_ver()[0] == "15.7.4"
        and platform.machine() == "arm64"
    )


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
        ExampleAxis(count=1),
        ChannelAxis(labels=("channel-0",)),
        SampleAxis(start=0, step=2_000, count=4),
    )
    return Photoelectrons(
        tensor=torch.tensor([[[3, 0, 1, 2]]], dtype=torch.int64),
        axes=axes,
    )


def _sampling() -> SamplingRuntime:
    return SamplingRuntime(
        sample_count=4,
        sample_period_ps=2_000,
        sample_dimension=2,
    )


def _produce_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingRuntime,
    config: ChargeConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> Charge:
    sampling_runtime = prepare_sampling(photoelectrons)
    self_consistent = (
        sampling_runtime.sample_count == sampling.sample_count
        and sampling_runtime.sample_period_ps == sampling.sample_period_ps
    )
    if not self_consistent:
        raise AssertionError("test source and sampling runtime diverged")
    runtime = prepare_charge(
        config,
        photoelectrons=photoelectrons,
        sampling=sampling_runtime,
        floating_dtype=floating_dtype,
    )
    result = _produce_charge_prepared(photoelectrons, runtime=runtime, rng=rng)
    validate_charge(result, source=photoelectrons, runtime=runtime)
    return result


def _produce_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingRuntime,
    config: NoiseWaveformConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> NoiseWaveform:
    sampling_runtime = prepare_sampling(photoelectrons)
    self_consistent = (
        sampling_runtime.sample_count == sampling.sample_count
        and sampling_runtime.sample_period_ps == sampling.sample_period_ps
    )
    if not self_consistent:
        raise AssertionError("test source and sampling runtime diverged")
    runtime = prepare_noise_waveform(
        config,
        sampling=sampling_runtime,
        shape=photoelectrons.shape,
        floating_dtype=floating_dtype,
        device=photoelectrons.tensor.device,
    )
    result = _produce_noise_waveform_prepared(
        photoelectrons,
        runtime=runtime,
        rng=rng,
    )
    validate_noise_waveform(result, source=photoelectrons, runtime=runtime)
    return result


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
    def _replay_public_request(
        self,
        *,
        method_name: str,
        request: Callable[[CounterRng], torch.Tensor],
        identity_arguments: dict[str, object],
        value_arguments: dict[str, object],
        assert_inputs_unchanged: Callable[[], None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_rng = Threefry4x32(seed=_SEED)
        second_rng = Threefry4x32(seed=_SEED)
        self.assertIsNot(first_rng, second_rng)
        self.assertEqual(first_rng, second_rng)
        self.assertIs(type(first_rng), Threefry4x32)
        self.assertIs(type(second_rng), Threefry4x32)

        original: Any = getattr(CounterRng, method_name)
        returned: list[torch.Tensor] = []

        def record_return(*args: object, **kwargs: object) -> torch.Tensor:
            result: torch.Tensor = original(*args, **kwargs)
            returned.append(result)
            return result

        with mock.patch.object(
            CounterRng,
            method_name,
            autospec=True,
            side_effect=record_return,
        ) as distribution:
            first = request(first_rng)
            assert_inputs_unchanged()
            second = request(second_rng)
            assert_inputs_unchanged()

        self.assertEqual(distribution.call_count, 2)
        calls = distribution.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(returned), 2)
        self.assertIs(first, returned[0])
        self.assertIs(second, returned[1])
        for call, expected_rng in zip(
            calls,
            (first_rng, second_rng),
            strict=True,
        ):
            self.assertEqual(len(call.args), 1)
            self.assertIs(call.args[0], expected_rng)
            self.assertEqual(
                set(call.kwargs),
                set(identity_arguments) | set(value_arguments),
            )
            for name, expected in identity_arguments.items():
                self.assertIs(call.kwargs[name], expected)
            for name, expected in value_arguments.items():
                self.assertEqual(call.kwargs[name], expected)

        self.assertIsNot(first, second)
        self.assertNotEqual(
            first.untyped_storage().data_ptr(),
            second.untyped_storage().data_ptr(),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.reshape(-1).tolist(), second.reshape(-1).tolist())
        self.assertEqual(first_rng.seed, _SEED)
        self.assertEqual(second_rng.seed, _SEED)
        self.assertEqual(first_rng, second_rng)
        return first, second

    def _replay_completed_product(
        self,
        *,
        prepared_name: str,
        invoke: Callable[[CounterRng], NoiseWaveform | Charge],
        source: Photoelectrons,
        sampling: SamplingRuntime,
        config: NoiseWaveformConfig | ChargeConfig,
        field_type: type[NoiseWaveform] | type[Charge],
        floating_dtype: torch.dtype,
    ) -> tuple[NoiseWaveform | Charge, NoiseWaveform | Charge]:
        source_tensor = source.tensor
        source_storage = source_tensor.untyped_storage().data_ptr()
        source_values = source_tensor.clone()
        source_axes = source.axes
        sampling_repr = repr(sampling)
        config_repr = repr(config)
        first_rng = Threefry4x32(seed=_SEED)
        second_rng = Threefry4x32(seed=_SEED)
        self.assertIsNot(first_rng, second_rng)
        self.assertEqual(first_rng, second_rng)

        original: Any = globals()[prepared_name]
        returned: list[NoiseWaveform | Charge] = []

        def record_return(
            *args: object,
            **kwargs: object,
        ) -> NoiseWaveform | Charge:
            result: NoiseWaveform | Charge = original(*args, **kwargs)
            returned.append(result)
            return result

        def assert_inputs_unchanged() -> None:
            self.assertIs(source.tensor, source_tensor)
            self.assertEqual(
                source.tensor.untyped_storage().data_ptr(),
                source_storage,
            )
            self.assertTrue(torch.equal(source.tensor, source_values))
            self.assertIs(source.axes, source_axes)
            self.assertEqual(repr(sampling), sampling_repr)
            self.assertEqual(repr(config), config_repr)

        with mock.patch(
            f"{__name__}.{prepared_name}",
            autospec=True,
            side_effect=record_return,
        ) as producer:
            first = invoke(first_rng)
            assert_inputs_unchanged()
            second = invoke(second_rng)
            assert_inputs_unchanged()

        self.assertEqual(producer.call_count, 2)
        calls = producer.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(returned), 2)
        self.assertIs(first, returned[0])
        self.assertIs(second, returned[1])
        self.assertIs(calls[0].args[0], source)
        self.assertIs(calls[1].args[0], source)
        self.assertIs(calls[0].kwargs["rng"], first_rng)
        self.assertIs(calls[1].kwargs["rng"], second_rng)
        self.assertIsNot(calls[0].kwargs["runtime"], calls[1].kwargs["runtime"])

        self.assertIs(type(first), field_type)
        self.assertIs(type(second), field_type)
        self.assertIsNot(first, second)
        self.assertIs(first.tensor.dtype, floating_dtype)
        self.assertIs(second.tensor.dtype, floating_dtype)
        self.assertEqual(tuple(first.tensor.shape), tuple(source.tensor.shape))
        self.assertEqual(tuple(second.tensor.shape), tuple(source.tensor.shape))
        self.assertEqual(first.tensor.device, source.tensor.device)
        self.assertEqual(second.tensor.device, source.tensor.device)
        self.assertIs(first.axes, source.axes)
        self.assertIs(second.axes, source.axes)
        self.assertIsNot(first.tensor, second.tensor)
        self.assertNotEqual(
            first.tensor.untyped_storage().data_ptr(),
            second.tensor.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            first.tensor.untyped_storage().data_ptr(),
            source_storage,
        )
        self.assertNotEqual(
            second.tensor.untyped_storage().data_ptr(),
            source_storage,
        )
        self.assertTrue(torch.equal(first.tensor, second.tensor))
        self.assertEqual(
            _hex_bits(first.tensor),
            _hex_bits(second.tensor),
        )
        self.assertTrue(bool(torch.isfinite(first.tensor).all().item()))
        self.assertTrue(bool(torch.isfinite(second.tensor).all().item()))
        self.assertFalse(first.tensor.requires_grad)
        self.assertFalse(second.tensor.requires_grad)
        self.assertEqual(first_rng.seed, _SEED)
        self.assertEqual(second_rng.seed, _SEED)
        self.assertEqual(first_rng, second_rng)
        return first, second

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
            ReadoutConfig: (
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
        positions_snapshot = positions.clone()

        def assert_positions_unchanged() -> None:
            self.assertTrue(torch.equal(positions, positions_snapshot))

        def assert_float_pair(
            first: torch.Tensor,
            second: torch.Tensor,
            *,
            dtype: torch.dtype,
            shape: tuple[int, ...],
        ) -> None:
            for result in (first, second):
                self.assertIs(type(result), torch.Tensor)
                self.assertIs(result.dtype, dtype)
                self.assertEqual(tuple(result.shape), shape)
                self.assertEqual(result.device, torch.device("cpu"))
                self.assertIs(result.layout, torch.strided)
                self.assertTrue(result.is_contiguous())
                self.assertFalse(result.requires_grad)
                self.assertTrue(bool(torch.isfinite(result).all().item()))

        def assert_count_pair(
            first: torch.Tensor,
            second: torch.Tensor,
            *,
            upper: torch.Tensor | int,
        ) -> None:
            for result in (first, second):
                self.assertIs(type(result), torch.Tensor)
                self.assertIs(result.dtype, torch.int64)
                self.assertEqual(tuple(result.shape), tuple(positions.shape))
                self.assertEqual(result.device, torch.device("cpu"))
                self.assertIs(result.layout, torch.strided)
                self.assertTrue(result.is_contiguous())
                self.assertFalse(result.requires_grad)
                self.assertTrue(bool(torch.all(result >= 0).item()))
                self.assertTrue(bool(torch.all(result <= upper).item()))

        reference_stack = _is_maintenance_2_reference_stack()

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
            first, second = self._replay_public_request(
                method_name="uniform",
                request=lambda rng: rng.uniform(
                    key=key1,
                    positions=positions,
                    dtype=dtype,
                    quantum=0,
                    ordinal=0,
                    count=1,
                    include_zero=include_zero,
                ),
                identity_arguments={
                    "key": key1,
                    "positions": positions,
                    "dtype": dtype,
                },
                value_arguments={
                    "quantum": 0,
                    "ordinal": 0,
                    "count": 1,
                    "include_zero": include_zero,
                },
                assert_inputs_unchanged=assert_positions_unchanged,
            )
            assert_float_pair(
                first,
                second,
                dtype=dtype,
                shape=tuple(positions.shape),
            )
            for actual in (first, second):
                self.assertTrue(bool(torch.all(actual < 1.0).item()))
                comparison = actual >= 0.0 if include_zero else actual > 0.0
                self.assertTrue(bool(torch.all(comparison).item()))
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
            first, second = self._replay_public_request(
                method_name="gaussian",
                request=lambda rng: rng.gaussian(
                    mean=0.0,
                    standard_deviation=0.75,
                    key=key1,
                    positions=positions,
                    dtype=dtype,
                    quantum=0,
                    ordinal=0,
                    count=1,
                ),
                identity_arguments={
                    "key": key1,
                    "positions": positions,
                    "dtype": dtype,
                },
                value_arguments={
                    "mean": 0.0,
                    "standard_deviation": 0.75,
                    "quantum": 0,
                    "ordinal": 0,
                    "count": 1,
                },
                assert_inputs_unchanged=assert_positions_unchanged,
            )
            assert_float_pair(
                first,
                second,
                dtype=dtype,
                shape=tuple(positions.shape),
            )
            if reference_stack:
                self.assertEqual(_hex_bits(first), expected)
                self.assertEqual(_hex_bits(second), expected)

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
            first, second = self._replay_public_request(
                method_name="gaussian",
                request=lambda rng: rng.gaussian(
                    mean=0.0,
                    standard_deviation=1.0,
                    key=key2,
                    positions=positions,
                    dtype=dtype,
                    quantum=0,
                    ordinal=0,
                    count=2,
                ),
                identity_arguments={
                    "key": key2,
                    "positions": positions,
                    "dtype": dtype,
                },
                value_arguments={
                    "mean": 0.0,
                    "standard_deviation": 1.0,
                    "quantum": 0,
                    "ordinal": 0,
                    "count": 2,
                },
                assert_inputs_unchanged=assert_positions_unchanged,
            )
            assert_float_pair(
                first,
                second,
                dtype=dtype,
                shape=tuple(positions.shape) + (2,),
            )
            if reference_stack:
                self.assertEqual(_hex_bits(first), expected)
                self.assertEqual(_hex_bits(second), expected)

        means = torch.tensor([0.0, 0.75, 9.5, 25.0], dtype=torch.float64)
        means_snapshot = means.clone()

        def assert_poisson_inputs_unchanged() -> None:
            assert_positions_unchanged()
            self.assertTrue(torch.equal(means, means_snapshot))

        expected_poisson = {
            3: (0, 2, 12, 30),
            4: (0, 0, 8, 25),
            5: (0, 1, 7, 29),
            6: (0, 2, 11, 17),
            7: (0, 4, 4, 26),
        }
        for stream, expected in expected_poisson.items():
            key = RngKey(namespace=_NAMESPACE, stream=stream)
            first, second = self._replay_public_request(
                method_name="poisson",
                request=lambda rng: rng.poisson(
                    mean=means,
                    key=key,
                    positions=positions,
                    quantum=0,
                ),
                identity_arguments={
                    "mean": means,
                    "key": key,
                    "positions": positions,
                },
                value_arguments={"quantum": 0},
                assert_inputs_unchanged=assert_poisson_inputs_unchanged,
            )
            assert_count_pair(first, second, upper=(1 << 53) - 1)
            if reference_stack:
                self.assertEqual(
                    tuple(int(value) for value in first),
                    expected,
                )
                self.assertEqual(
                    tuple(int(value) for value in second),
                    expected,
                )

        counts = torch.tensor([0, 3, 20, 100], dtype=torch.int64)
        success = torch.tensor([0.0, 0.25, 0.9, 0.2], dtype=torch.float64)
        failure = torch.tensor([0.0, 0.75, 0.1, 0.8], dtype=torch.float64)
        counts_snapshot = counts.clone()
        success_snapshot = success.clone()
        failure_snapshot = failure.clone()

        def assert_binomial_inputs_unchanged() -> None:
            assert_positions_unchanged()
            self.assertTrue(torch.equal(counts, counts_snapshot))
            self.assertTrue(torch.equal(success, success_snapshot))
            self.assertTrue(torch.equal(failure, failure_snapshot))

        for stream, expected in ((8, (0, 1, 19, 17)), (9, (0, 1, 16, 23))):
            key = RngKey(namespace=_NAMESPACE, stream=stream)
            first, second = self._replay_public_request(
                method_name="binomial",
                request=lambda rng: rng.binomial(
                    counts=counts,
                    success_mass=success,
                    failure_mass=failure,
                    key=key,
                    positions=positions,
                    quantum=0,
                ),
                identity_arguments={
                    "counts": counts,
                    "success_mass": success,
                    "failure_mass": failure,
                    "key": key,
                    "positions": positions,
                },
                value_arguments={"quantum": 0},
                assert_inputs_unchanged=assert_binomial_inputs_unchanged,
            )
            assert_count_pair(first, second, upper=counts)
            if reference_stack:
                self.assertEqual(
                    tuple(int(value) for value in first),
                    expected,
                )
                self.assertEqual(
                    tuple(int(value) for value in second),
                    expected,
                )

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
            represented_means = torch.tensor(gaussian_means, dtype=dtype)
            represented_scales = torch.tensor(gaussian_scales, dtype=dtype)
            means_before = represented_means.clone()
            scales_before = represented_scales.clone()
            key = RngKey(namespace=_NAMESPACE, stream=10)

            def assert_gaussian_inputs_unchanged() -> None:
                assert_positions_unchanged()
                self.assertTrue(torch.equal(represented_means, means_before))
                self.assertTrue(torch.equal(represented_scales, scales_before))

            first, second = self._replay_public_request(
                method_name="gaussian",
                request=lambda rng: rng.gaussian(
                    mean=represented_means,
                    standard_deviation=represented_scales,
                    key=key,
                    positions=positions,
                    dtype=dtype,
                    quantum=0,
                    ordinal=0,
                    count=1,
                ),
                identity_arguments={
                    "mean": represented_means,
                    "standard_deviation": represented_scales,
                    "key": key,
                    "positions": positions,
                    "dtype": dtype,
                },
                value_arguments={
                    "quantum": 0,
                    "ordinal": 0,
                    "count": 1,
                },
                assert_inputs_unchanged=assert_gaussian_inputs_unchanged,
            )
            assert_float_pair(
                first,
                second,
                dtype=dtype,
                shape=tuple(positions.shape),
            )
            if reference_stack:
                self.assertEqual(_hex_bits(first), expected)
                self.assertEqual(_hex_bits(second), expected)

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

    def test_runtime_producer_signatures_and_draw_free_branches(self) -> None:
        for producer in (
            _produce_noise_waveform_prepared,
            _produce_charge_prepared,
        ):
            parameters = signature(producer).parameters
            self.assertIn("rng", parameters)
            self.assertIn("runtime", parameters)
            self.assertNotIn("seed", parameters)
            self.assertNotIn("config", parameters)
            self.assertNotIn("sampling", parameters)
            self.assertNotIn("floating_dtype", parameters)
            self.assertIs(parameters["rng"].kind, Parameter.KEYWORD_ONLY)
            self.assertIs(parameters["rng"].default, Parameter.empty)
        for producer in (
            _produce_pure_waveform_prepared,
            _produce_analog_waveform_prepared,
            _produce_digitized_waveform_prepared,
        ):
            parameters = signature(producer).parameters
            self.assertIn("runtime", parameters)
            self.assertNotIn("rng", parameters)
            self.assertNotIn("seed", parameters)
            self.assertNotIn("config", parameters)
            self.assertNotIn("sampling", parameters)
            self.assertNotIn("floating_dtype", parameters)

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
        sampling = _sampling()
        reference_stack = _is_maintenance_2_reference_stack()
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
            first_white, second_white = self._replay_completed_product(
                prepared_name="_produce_noise_waveform_prepared",
                invoke=lambda rng: _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=white_config,
                    rng=rng,
                    floating_dtype=dtype,
                ),
                source=source,
                sampling=sampling,
                config=white_config,
                field_type=NoiseWaveform,
                floating_dtype=dtype,
            )
            first_psd, second_psd = self._replay_completed_product(
                prepared_name="_produce_noise_waveform_prepared",
                invoke=lambda rng: _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=psd_config,
                    rng=rng,
                    floating_dtype=dtype,
                ),
                source=source,
                sampling=sampling,
                config=psd_config,
                field_type=NoiseWaveform,
                floating_dtype=dtype,
            )
            if reference_stack:
                self.assertEqual(
                    _hex_bits(first_white.tensor),
                    expected_white[dtype],
                )
                self.assertEqual(
                    _hex_bits(second_white.tensor),
                    expected_white[dtype],
                )
                self.assertEqual(
                    _hex_bits(first_psd.tensor),
                    expected_psd[dtype],
                )
                self.assertEqual(
                    _hex_bits(second_psd.tensor),
                    expected_psd[dtype],
                )

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
            first_charge, second_charge = self._replay_completed_product(
                prepared_name="_produce_charge_prepared",
                invoke=lambda rng: _produce_charge(
                    source,
                    sampling=sampling,
                    config=charge_config,
                    rng=rng,
                    floating_dtype=dtype,
                ),
                source=source,
                sampling=sampling,
                config=charge_config,
                field_type=Charge,
                floating_dtype=dtype,
            )
            if reference_stack:
                self.assertEqual(
                    _hex_bits(first_charge.tensor),
                    expected_charge[dtype],
                )
                self.assertEqual(
                    _hex_bits(second_charge.tensor),
                    expected_charge[dtype],
                )

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
