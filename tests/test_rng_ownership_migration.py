from collections.abc import Callable
from dataclasses import fields
import importlib.util
from inspect import Parameter, signature
import platform
import sys
from typing import Any, ClassVar, override
import unittest
from unittest import mock

import torch
from tensor_core import (
    CounterRng,
    NonnegativeFloat,
    NonnegativeInteger,
    Probability,
    RngKey,
    RngPositions,
    Threefry4x32,
)

import tensor_dslab
import tensor_dslab.readout as readout
from tensor_dslab import (
    quantities,
    quantity,
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    AnalogSaturationConfig,
    AnalogWaveformConfig,
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    ChannelAxis,
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
    PsdNoiseRuntime,
    WhiteNoiseRuntime,
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
from tensor_dslab.readout.runtime import keys as fixed_keys
from tensor_dslab.readout.runtime.sampling import SamplingRuntime, prepare_sampling


_NAMESPACE = 0x54445331
_SEED = 0x0123456789ABCDEF
_EXPECTED_KEYS = (
    fixed_keys.WHITE_NOISE_RNG_KEY,
    fixed_keys.PSD_NOISE_RNG_KEY,
    fixed_keys.DARK_COUNT_RNG_KEY,
    fixed_keys.DIRECT_CROSSTALK_RETAINED_RNG_KEY,
    fixed_keys.DIRECT_CROSSTALK_OVERFLOW_RNG_KEY,
    fixed_keys.DELAYED_CROSSTALK_RETAINED_RNG_KEY,
    fixed_keys.DELAYED_CROSSTALK_OVERFLOW_RNG_KEY,
    fixed_keys.TIMING_JITTER_RNG_KEY,
    fixed_keys.AFTERPULSE_RNG_KEY,
    fixed_keys.CHARGE_SMEARING_RNG_KEY,
)


def _ns(value: int | float):
    return quantity(value, "ns")


def _hz(value: int | float):
    return quantity(value, "Hz")


def _mv(value: int | float):
    return quantity(value, "mV")


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
        return tuple(
            f"{int(value) & 0xFFFFFFFFFFFFFFFF:016x}" for value in integers
        )
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
    source: Photoelectrons,
    *,
    config: ChargeConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> Charge:
    sampling = prepare_sampling(source)
    runtime = prepare_charge(
        config,
        photoelectrons=source,
        sampling=sampling,
        floating_dtype=floating_dtype,
    )
    result = _produce_charge_prepared(source, runtime=runtime, rng=rng)
    validate_charge(result, source=source, runtime=runtime)
    return result


def _produce_noise(
    source: Photoelectrons,
    *,
    config: NoiseWaveformConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> NoiseWaveform:
    sampling = prepare_sampling(source)
    runtime = prepare_noise_waveform(
        config,
        sampling=sampling,
        shape=source.shape,
        floating_dtype=floating_dtype,
        device=source.tensor.device,
    )
    result = _produce_noise_waveform_prepared(source, runtime=runtime, rng=rng)
    validate_noise_waveform(result, source=source, runtime=runtime)
    return result


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


class _FailingRng(CounterRng):
    __slots__ = ()

    @override
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
    def _replay_product(
        self,
        invoke: Callable[[CounterRng], NoiseWaveform | Charge],
        *,
        source: Photoelectrons,
        dtype: torch.dtype,
    ) -> tuple[NoiseWaveform | Charge, NoiseWaveform | Charge]:
        source_values = source.tensor.clone()
        source_storage = source.tensor.untyped_storage().data_ptr()
        first_rng = Threefry4x32(seed=_SEED)
        second_rng = Threefry4x32(seed=_SEED)
        self.assertIsNot(first_rng, second_rng)
        self.assertEqual(first_rng, second_rng)
        first = invoke(first_rng)
        second = invoke(second_rng)
        for result in (first, second):
            self.assertIs(result.tensor.dtype, dtype)
            self.assertIs(result.axes, source.axes)
            self.assertTrue(bool(torch.isfinite(result.tensor).all().item()))
            self.assertNotEqual(
                result.tensor.untyped_storage().data_ptr(),
                source_storage,
            )
        self.assertIsNot(first, second)
        self.assertNotEqual(
            first.tensor.untyped_storage().data_ptr(),
            second.tensor.untyped_storage().data_ptr(),
        )
        self.assertTrue(torch.equal(first.tensor, second.tensor))
        self.assertTrue(torch.equal(source.tensor, source_values))
        return first, second

    def test_fixed_key_table_is_exact_unique_and_export_private(self) -> None:
        self.assertEqual(
            tuple((key.namespace, key.stream) for key in _EXPECTED_KEYS),
            tuple((_NAMESPACE, stream) for stream in range(1, 11)),
        )
        self.assertEqual(len(set(_EXPECTED_KEYS)), 10)
        for module in (tensor_dslab, readout):
            for name in (
                "RNG_NAMESPACE",
                *(
                    name
                    for name in vars(fixed_keys)
                    if name.endswith("_RNG_KEY")
                ),
            ):
                self.assertNotIn(name, module.__all__)
                self.assertFalse(hasattr(module, name))

    def test_public_config_inventory_has_no_rng_key_surface(self) -> None:
        config_types = tuple(
            getattr(tensor_dslab, name)
            for name in tensor_dslab.__all__
            if name.endswith("Config")
        )
        self.assertEqual(len(config_types), 22)
        for config_type in config_types:
            with self.subTest(config=config_type.__name__):
                field_names = tuple(field.name for field in fields(config_type))
                parameters = signature(config_type).parameters
                for retired in (
                    "rng_key",
                    "retained_rng_key",
                    "overflow_rng_key",
                ):
                    self.assertNotIn(retired, field_names)
                    self.assertNotIn(retired, parameters)

    def test_preparation_uses_exact_fixed_keys_and_disabled_none(self) -> None:
        source = _source()
        sampling = prepare_sampling(source)
        white = prepare_noise_waveform(
            NoiseWaveformConfig(model=WhiteNoiseConfig(rms=_mv(1.0))),
            sampling=sampling,
            shape=source.shape,
            floating_dtype=torch.float32,
            device=source.tensor.device,
        )
        self.assertIs(type(white.model), WhiteNoiseRuntime)
        assert type(white.model) is WhiteNoiseRuntime
        self.assertIs(white.model.rng_key, fixed_keys.WHITE_NOISE_RNG_KEY)
        psd = prepare_noise_waveform(
            NoiseWaveformConfig(
                model=PsdNoiseConfig(
                    frequency_left_edges=quantities(
                        (0.0, 100_000_000.0),
                        "Hz",
                    ),
                    frequency_stop=_hz(250_000_000.0),
                    power_density=quantities(
                        (1.0e-8, 2.0e-8),
                        "mV ** 2 / Hz",
                    ),
                )
            ),
            sampling=sampling,
            shape=source.shape,
            floating_dtype=torch.float32,
            device=source.tensor.device,
        )
        self.assertIs(type(psd.model), PsdNoiseRuntime)
        assert type(psd.model) is PsdNoiseRuntime
        self.assertIs(psd.model.rng_key, fixed_keys.PSD_NOISE_RNG_KEY)

        config = ChargeConfig(
            dark_count=DarkCountConfig(rate=_hz(5.0e8)),
            timing_jitter=TimingJitterConfig(sigma=_ns(1.0)),
            correlated_avalanches=CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                direct_crosstalk=DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.6),
                    delay=FixedDelayConfig(delay=_ns(0.0)),
                ),
                delayed_crosstalk=DelayedCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.4),
                    delay=FixedDelayConfig(delay=_ns(0.0)),
                ),
                afterpulse=AfterpulseConfig(
                    probability=Probability(0.35),
                    mean_delay=_ns(3.0),
                ),
            ),
            smearing=ChargeSmearingConfig(
                relative_sigma=NonnegativeFloat(0.1)
            ),
        )
        runtime = prepare_charge(
            config,
            photoelectrons=source,
            sampling=sampling,
            floating_dtype=torch.float32,
        )
        assert runtime.dark is not None
        assert runtime.timing_jitter is not None
        assert runtime.correlated_avalanches is not None
        assert runtime.smearing is not None
        correlated = runtime.correlated_avalanches
        self.assertIs(runtime.dark.rng_key, fixed_keys.DARK_COUNT_RNG_KEY)
        self.assertIs(
            runtime.timing_jitter.rng_key,
            fixed_keys.TIMING_JITTER_RNG_KEY,
        )
        self.assertIs(
            correlated.direct_retained_rng_key,
            fixed_keys.DIRECT_CROSSTALK_RETAINED_RNG_KEY,
        )
        self.assertIs(
            correlated.direct_overflow_rng_key,
            fixed_keys.DIRECT_CROSSTALK_OVERFLOW_RNG_KEY,
        )
        self.assertIs(
            correlated.delayed_retained_rng_key,
            fixed_keys.DELAYED_CROSSTALK_RETAINED_RNG_KEY,
        )
        self.assertIs(
            correlated.delayed_overflow_rng_key,
            fixed_keys.DELAYED_CROSSTALK_OVERFLOW_RNG_KEY,
        )
        self.assertIs(
            correlated.afterpulse_rng_key,
            fixed_keys.AFTERPULSE_RNG_KEY,
        )
        self.assertIs(
            runtime.smearing.rng_key,
            fixed_keys.CHARGE_SMEARING_RNG_KEY,
        )
        disabled = prepare_charge(
            ChargeConfig(),
            photoelectrons=source,
            sampling=sampling,
            floating_dtype=torch.float32,
        )
        self.assertIsNone(disabled.dark)
        self.assertIsNone(disabled.timing_jitter)
        self.assertIsNone(disabled.correlated_avalanches)
        self.assertIsNone(disabled.smearing)

    def test_rng_positions_snapshot_transforms_and_raw_hook_are_exact(self) -> None:
        raw = torch.tensor((0, 1, 2, 4_294_967_299), dtype=torch.int64)
        snapshot = RngPositions.from_tensor(raw)
        raw.fill_(99)
        transformed = snapshot.movedim(0, 0).slice(0, 1, 4).offset(7)
        self.assertIs(type(transformed), RngPositions)
        self.assertEqual(transformed.shape, (3,))
        _RecordingRng.calls = []
        _RecordingRng(seed=0).uniform(
            key=fixed_keys.WHITE_NOISE_RNG_KEY,
            positions=transformed,
            dtype=torch.float64,
        )
        self.assertEqual(len(_RecordingRng.calls), 1)
        self.assertTrue(
            torch.equal(
                _RecordingRng.calls[0][1],
                torch.tensor((8, 9, 4_294_967_306), dtype=torch.int64),
            )
        )

    def test_public_tensorcore_distribution_replay_and_raw_rejection(self) -> None:
        raw = torch.tensor((0, 1, 2, 4_294_967_299), dtype=torch.int64)
        positions = RngPositions.from_tensor(raw)
        requests = (
            lambda rng: rng.uniform(
                key=fixed_keys.WHITE_NOISE_RNG_KEY,
                positions=positions,
                dtype=torch.float64,
            ),
            lambda rng: rng.gaussian(
                mean=0.0,
                standard_deviation=0.75,
                key=fixed_keys.WHITE_NOISE_RNG_KEY,
                positions=positions,
                dtype=torch.float64,
            ),
            lambda rng: rng.poisson(
                mean=torch.tensor(
                    (0.0, 0.75, 9.5, 25.0),
                    dtype=torch.float64,
                ),
                key=fixed_keys.DARK_COUNT_RNG_KEY,
                positions=positions,
            ),
            lambda rng: rng.binomial(
                counts=torch.tensor((0, 3, 20, 100), dtype=torch.int64),
                success_mass=torch.tensor(
                    (0.0, 0.25, 0.9, 0.2),
                    dtype=torch.float64,
                ),
                failure_mass=torch.tensor(
                    (0.0, 0.75, 0.1, 0.8),
                    dtype=torch.float64,
                ),
                key=fixed_keys.TIMING_JITTER_RNG_KEY,
                positions=positions,
            ),
        )
        for request in requests:
            with self.subTest(request=request):
                first = request(Threefry4x32(seed=_SEED))
                second = request(Threefry4x32(seed=_SEED))
                self.assertTrue(torch.equal(first, second))
                self.assertNotEqual(
                    first.untyped_storage().data_ptr(),
                    second.untyped_storage().data_ptr(),
                )
        with self.assertRaises(TypeError):
            Threefry4x32(seed=0).uniform(
                key=fixed_keys.WHITE_NOISE_RNG_KEY,
                positions=raw,  # type: ignore[arg-type]
                dtype=torch.float32,
            )

    def test_public_tensorcore_zero_dimension_address_span(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "RngPositions.from_shape shape span must be less than 2\\*\\*63",
        ):
            RngPositions.from_shape((0, 1 << 62, 2), device="cpu")
        positions = RngPositions.from_shape((0, 1 << 62), device="cpu")
        self.assertEqual(positions.shape, (0, 1 << 62))
        with self.assertRaisesRegex(
            ValueError,
            "result shape span must be less than 2\\*\\*63",
        ):
            _FailingRng(seed=_SEED).gaussian(
                mean=0.0,
                standard_deviation=1.0,
                key=fixed_keys.WHITE_NOISE_RNG_KEY,
                positions=positions,
                dtype=torch.float32,
                count=2,
            )

    def test_runtime_producer_signatures_and_exact_zero_draws(self) -> None:
        for producer in (
            _produce_noise_waveform_prepared,
            _produce_charge_prepared,
        ):
            parameters = signature(producer).parameters
            self.assertIn("rng", parameters)
            self.assertIn("runtime", parameters)
            self.assertNotIn("config", parameters)
            self.assertIs(parameters["rng"].kind, Parameter.KEYWORD_ONLY)
        for producer in (
            _produce_pure_waveform_prepared,
            _produce_analog_waveform_prepared,
            _produce_digitized_waveform_prepared,
        ):
            self.assertNotIn("rng", signature(producer).parameters)

        source = _source()
        original = source.tensor.clone()
        configs = (
            ChargeConfig(dark_count=DarkCountConfig(rate=_hz(0.0))),
            ChargeConfig(timing_jitter=TimingJitterConfig(sigma=_ns(0.0))),
            ChargeConfig(
                smearing=ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(0.0)
                )
            ),
        )
        for config in configs:
            for dtype in (torch.float32, torch.float64):
                result = _produce_charge(
                    source,
                    config=config,
                    rng=_FailingRng(seed=_SEED),
                    floating_dtype=dtype,
                )
                self.assertTrue(
                    torch.equal(result.tensor, source.tensor.to(dtype=dtype))
                )
                self.assertNotEqual(
                    result.tensor.untyped_storage().data_ptr(),
                    source.tensor.untyped_storage().data_ptr(),
                )
        self.assertTrue(torch.equal(source.tensor, original))

    def test_completed_noise_and_charge_eager_cpu_continuity(self) -> None:
        source = _source()
        reference_stack = _is_maintenance_2_reference_stack()
        white_config = NoiseWaveformConfig(
            model=WhiteNoiseConfig(rms=_mv(0.75))
        )
        psd_config = NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges=quantities(
                    (0.0, 100_000_000.0),
                    "Hz",
                ),
                frequency_stop=_hz(250_000_000.0),
                power_density=quantities(
                    (1.0e-8, 2.0e-8),
                    "mV ** 2 / Hz",
                ),
            )
        )
        expected_white = {
            torch.float32: ("3f81741c", "3e184d06", "3f25bf13", "3e499154"),
            torch.float64: (
                "3fcf75d99582d78a",
                "3f89fe5e8e452724",
                "3fe178009eac0f2d",
                "3f851d1a3e211f04",
            ),
        }
        expected_psd = {
            torch.float32: ("bfa40c51", "c03bf3d0", "3e486878", "4080b9b9"),
            torch.float64: (
                "c01058090a5b3d75",
                "bfd20487b1837bec",
                "4006399d2ea79a9c",
                "3ff96e0bb87e9f95",
            ),
        }
        for dtype in (torch.float32, torch.float64):
            white_first, white_second = self._replay_product(
                lambda rng: _produce_noise(
                    source,
                    config=white_config,
                    rng=rng,
                    floating_dtype=dtype,
                ),
                source=source,
                dtype=dtype,
            )
            psd_first, psd_second = self._replay_product(
                lambda rng: _produce_noise(
                    source,
                    config=psd_config,
                    rng=rng,
                    floating_dtype=dtype,
                ),
                source=source,
                dtype=dtype,
            )
            if reference_stack:
                self.assertEqual(_hex_bits(white_first.tensor), expected_white[dtype])
                self.assertEqual(_hex_bits(white_second.tensor), expected_white[dtype])
                self.assertEqual(_hex_bits(psd_first.tensor), expected_psd[dtype])
                self.assertEqual(_hex_bits(psd_second.tensor), expected_psd[dtype])

        charge_config = ChargeConfig(
            dark_count=DarkCountConfig(rate=_hz(5.0e8)),
            timing_jitter=TimingJitterConfig(sigma=_ns(1.0)),
            correlated_avalanches=CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(2),
                direct_crosstalk=DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.6),
                    delay=ExponentialDelayConfig(mean_delay=_ns(2.5)),
                ),
                delayed_crosstalk=DelayedCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.4),
                    delay=ExponentialDelayConfig(mean_delay=_ns(4.0)),
                ),
                afterpulse=AfterpulseConfig(
                    probability=Probability(0.35),
                    mean_delay=_ns(3.0),
                    recovery=AfterpulseRecoveryConfig(
                        time_constant=_ns(5.0)
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
                "40147e5936eabbbe",
                "4017f2b37dd7adc7",
                "401edf582a1ee0b5",
                "402c390c96702ab5",
            ),
        }
        for dtype in (torch.float32, torch.float64):
            first, second = self._replay_product(
                lambda rng: _produce_charge(
                    source,
                    config=charge_config,
                    rng=rng,
                    floating_dtype=dtype,
                ),
                source=source,
                dtype=dtype,
            )
            if reference_stack:
                self.assertEqual(_hex_bits(first.tensor), expected_charge[dtype])
                self.assertEqual(_hex_bits(second.tensor), expected_charge[dtype])

    def test_recording_hook_uses_fixed_key_and_retired_modules_are_absent(
        self,
    ) -> None:
        _RecordingRng.calls = []
        _produce_noise(
            _source(),
            config=NoiseWaveformConfig(
                model=WhiteNoiseConfig(rms=_mv(1.0))
            ),
            rng=_RecordingRng(seed=0),
            floating_dtype=torch.float32,
        )
        self.assertTrue(_RecordingRng.calls)
        self.assertTrue(
            all(
                call[0] is fixed_keys.WHITE_NOISE_RNG_KEY
                for call in _RecordingRng.calls
            )
        )
        self.assertTrue(all(call[2] == 0 for call in _RecordingRng.calls))
        retired = (
            "tensor_dslab.readout._random",
            "tensor_dslab.readout._rng",
            "tensor_dslab.readout.requirements",
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
