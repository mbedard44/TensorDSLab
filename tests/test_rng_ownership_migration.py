from __future__ import annotations

from dataclasses import fields
import importlib.util
from inspect import Parameter, signature
from typing import ClassVar
import unittest

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

from tensor_dslab import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    ChannelAxis,
    ChargeConfig,
    ChargeSmearingConfig,
    CorrelatedAvalancheConfig,
    DarkCountConfig,
    DelayedCrosstalkConfig,
    DirectCrosstalkConfig,
    ExampleAxis,
    ExponentialDelayConfig,
    FixedDelayConfig,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    SampleAxis,
    SamplingConfig,
    TimingJitterConfig,
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
            (WhiteNoiseConfig(rms_mv=PositiveFloat(1.0)).rng_key, 1),
            (
                PsdNoiseConfig(
                    frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                    frequency_stop_hz=PositiveFloat(1.0),
                    power_density_mv2_per_hz=(NonnegativeFloat(1.0),),
                ).rng_key,
                2,
            ),
            (DarkCountConfig(rate_hz=NonnegativeFloat(0.0)).rng_key, 3),
            (direct.retained_rng_key, 4),
            (direct.overflow_rng_key, 5),
            (delayed.retained_rng_key, 6),
            (delayed.overflow_rng_key, 7),
            (TimingJitterConfig(sigma_ns=NonnegativeFloat(0.0)).rng_key, 8),
            (
                AfterpulseConfig(
                    probability=Probability(0.0),
                    mean_delay_ns=PositiveFloat(1.0),
                ).rng_key,
                9,
            ),
            (
                ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(0.0)
                ).rng_key,
                10,
            ),
        )
        self.assertEqual(
            tuple((key.namespace, key.stream) for key, _ in keyed),
            tuple((_NAMESPACE, stream) for _, stream in keyed),
        )
        self.assertNotEqual(direct.retained_rng_key, direct.overflow_rng_key)
        self.assertNotEqual(delayed.retained_rng_key, delayed.overflow_rng_key)

        override = RngKey(namespace=7, stream=11)
        first = WhiteNoiseConfig(rms_mv=PositiveFloat(1.0), rng_key=override)
        second = WhiteNoiseConfig(rms_mv=PositiveFloat(1.0), rng_key=override)
        self.assertEqual(first, second)
        self.assertIn("rng_key=RngKey(namespace=7, stream=11)", repr(first))
        self.assertIs(first.rng_key, override)
        with self.assertRaises(TypeError):
            WhiteNoiseConfig(rms_mv=PositiveFloat(1.0), rng_key=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            DirectCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.1),
                delay=fixed,
                retained_rng_key=override,
                overflow_rng_key=override,
            )

        for config_type in (
            FixedDelayConfig,
            ExponentialDelayConfig,
            AfterpulseRecoveryConfig,
            CorrelatedAvalancheConfig,
            ChargeConfig,
            NoiseWaveformConfig,
            ZeroNoiseConfig,
        ):
            self.assertFalse(
                any("rng_key" in field.name for field in fields(config_type)),
                config_type.__name__,
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
