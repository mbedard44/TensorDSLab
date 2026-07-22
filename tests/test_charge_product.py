from __future__ import annotations

from fractions import Fraction
import itertools
import math
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

from tensor_dslab import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    ChannelAxis,
    CorrelatedAvalancheConfig,
    DarkCountConfig,
    DelayedCrosstalkConfig,
    DirectCrosstalkConfig,
    ExampleAxis,
    ExponentialDelayConfig,
    FixedDelayConfig,
    Photoelectrons,
    SampleAxis,
    SamplingConfig,
    TimingJitterConfig,
)
import tensor_dslab.readout.charge.runtime.produce as charge_producer
from tensor_dslab.readout.charge.runtime.effects import (
    correlated_avalanches as correlated_effect,
)
from tensor_dslab.readout.charge.runtime.effects import counts as count_effect
from tensor_dslab.readout.charge.runtime.effects import dark_counts as dark_effect
from tensor_dslab.readout.charge.runtime.effects import smearing as smearing_effect
from tensor_dslab.readout.charge.runtime.prepare import prepare_charge
from tensor_dslab.readout.charge.runtime.validate import validate_charge
from tensor_dslab.readout.runtime.sampling import SamplingRuntime, prepare_sampling


def _produce_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> Charge:
    sampling_runtime = prepare_sampling(photoelectrons, config=sampling)
    runtime = prepare_charge(
        config,
        photoelectrons=photoelectrons,
        sampling=sampling_runtime,
        floating_dtype=floating_dtype,
    )
    result = charge_producer.produce_charge(
        photoelectrons,
        runtime=runtime,
        rng=rng,
    )
    validate_charge(result, source=photoelectrons, runtime=runtime)
    return result


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


class _FixedBlockRng(CounterRng):
    __slots__ = ()

    words: ClassVar[tuple[int, int, int, int]] = (0, 0, 0, 0)
    calls: ClassVar[int] = 0

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls += 1
        return (
            torch.tensor(
                type(self).words,
                dtype=torch.int64,
                device=positions.device,
            )
            .expand(tuple(positions.shape) + (4,))
            .clone()
        )

    @classmethod
    def use(cls, words: tuple[int, int, int, int]) -> None:
        cls.words = words
        cls.calls = 0


def _sampling() -> SamplingConfig:
    return SamplingConfig(
        sample_period_ps=PositiveInteger(2000),
        sample_count=PositiveInteger(4),
    )


def _sampling_runtime(sampling: SamplingConfig) -> SamplingRuntime:
    return SamplingRuntime(
        sample_count=sampling.sample_count.value,
        sample_period_ps=sampling.sample_period_ps.value,
        sample_dimension=2,
    )


def _field(
    values: torch.Tensor,
    *,
    sample_first: bool = False,
) -> Photoelectrons:
    sampling = _sampling()
    example = ExampleAxis(coordinates=tuple(f"e{i}" for i in range(2)))
    channel = ChannelAxis(coordinates=tuple(f"c{i}" for i in range(2)))
    sample = sampling.build_axis()
    axes = (sample, example, channel) if sample_first else (example, channel, sample)
    return Photoelectrons(tensor=values, axes=axes)


def _ensemble_field(
    values: torch.Tensor,
    *,
    sampling: SamplingConfig,
) -> Photoelectrons:
    example = ExampleAxis(
        coordinates=tuple(f"e{i}" for i in range(values.shape[0]))
    )
    channel = ChannelAxis(
        coordinates=tuple(f"c{i}" for i in range(values.shape[1]))
    )
    return Photoelectrons(
        tensor=values,
        axes=(example, channel, sampling.build_axis()),
    )


def _config(
    dark: bool,
    jitter: bool,
    correlated: bool,
    smearing: bool,
) -> ChargeConfig:
    return ChargeConfig(
        dark_count=(
            DarkCountConfig(rate_hz=NonnegativeFloat(1.0e10)) if dark else None
        ),
        timing_jitter=(
            TimingJitterConfig(sigma_ns=NonnegativeFloat(1.0))
            if jitter
            else None
        ),
        correlated_avalanches=(
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                direct_crosstalk=DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.2),
                    delay=FixedDelayConfig(delay_ns=NonnegativeFloat(0.0)),
                ),
            )
            if correlated
            else None
        ),
        smearing=(
            ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.1))
            if smearing
            else None
        ),
    )


def _assert_statistic(
    test: unittest.TestCase,
    *,
    name: str,
    observed: float,
    target: float,
    standard_error: float,
    dtype: torch.dtype,
    accumulation_length: int,
) -> None:
    delta = (
        64.0
        * torch.finfo(dtype).eps
        * max(1, math.ceil(math.log2(accumulation_length)))
        * abs(target)
    )
    bound = 8.0 * standard_error + delta
    test.assertLessEqual(
        abs(observed - target),
        bound,
        msg=(
            f"{name}: observed={observed!r}, target={target!r}, "
            f"SE={standard_error!r}, delta={delta!r}, bound={bound!r}"
        ),
    )


class ChargeProductStructureTest(unittest.TestCase):
    def test_all_sixteen_stage_combinations_are_valid(self) -> None:
        source_values = torch.zeros((2, 2, 4), dtype=torch.int64)
        source_values[0, 0, 0] = 8
        source = _field(source_values)
        for flags in itertools.product((False, True), repeat=4):
            with self.subTest(flags=flags):
                result = _produce_charge(
                    source,
                    sampling=_sampling(),
                    config=_config(*flags),
                    rng=Threefry4x32(seed=1234),
                    floating_dtype=torch.float32,
                )
                self.assertIs(type(result), Charge)
                self.assertEqual(result.tensor.shape, source.tensor.shape)
                self.assertIs(result.axes, source.axes)
                self.assertTrue(all(a is b for a, b in zip(result.axes, source.axes)))
                self.assertTrue(bool(torch.all(torch.isfinite(result.tensor)).item()))
                self.assertTrue(bool(torch.all(result.tensor >= 0.0).item()))

    def test_arbitrary_axis_order_noncontiguous_source_and_both_dtypes(self) -> None:
        base = torch.zeros((4, 2, 2, 2), dtype=torch.int64)
        values = base[..., 0]
        values[0, 0, 0] = 4
        self.assertFalse(values.is_contiguous())
        source = _field(values, sample_first=True)
        original = source.tensor.clone()
        for dtype in (torch.float32, torch.float64):
            result = _produce_charge(
                source,
                sampling=_sampling(),
                config=_config(True, True, True, True),
                rng=Threefry4x32(seed=99),
                floating_dtype=dtype,
            )
            self.assertEqual(result.tensor.dtype, dtype)
            self.assertEqual(result.tensor.shape, source.tensor.shape)
            self.assertIs(result.axes, source.axes)
            self.assertNotEqual(
                result.tensor.untyped_storage().data_ptr(),
                source.tensor.untyped_storage().data_ptr(),
            )
        self.assertTrue(torch.equal(source.tensor, original))

    def test_direct_truth_path_is_fresh_exact_and_draw_free(self) -> None:
        values = torch.arange(16, dtype=torch.int64).reshape(2, 2, 4)
        source = _field(values)
        state = torch.random.get_rng_state().clone()
        result = _produce_charge(
            source,
            sampling=_sampling(),
            config=ChargeConfig(),
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float64,
        )
        self.assertTrue(torch.equal(result.tensor, values.to(torch.float64)))
        self.assertNotEqual(
            result.tensor.untyped_storage().data_ptr(),
            source.tensor.untyped_storage().data_ptr(),
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))


class ChargeProductPreflightTest(unittest.TestCase):
    def test_zero_population_paths_request_no_words(
        self,
    ) -> None:
        zeros = _field(torch.zeros((2, 2, 4), dtype=torch.int64))
        draw_free = ChargeConfig(
            timing_jitter=TimingJitterConfig(sigma_ns=NonnegativeFloat(1.0)),
            correlated_avalanches=CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(2),
                direct_crosstalk=DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(1.0),
                    delay=FixedDelayConfig(delay_ns=NonnegativeFloat(0.0)),
                ),
            ),
        )
        result = _produce_charge(
            zeros,
            sampling=_sampling(),
            config=draw_free,
            rng=_FailingRng(seed=(1 << 64) - 1),
            floating_dtype=torch.float32,
        )
        self.assertTrue(torch.equal(result.tensor, torch.zeros_like(result.tensor)))

    def test_source_andchecked_add_count_boundaries(self) -> None:
        source_values = torch.zeros((2, 2, 4), dtype=torch.int64)
        source_values[0, 0, 0] = (1 << 53) - 2
        source = _field(source_values)
        one = torch.zeros_like(source.tensor)
        one[0, 0, 0] = 1
        two = one * 2
        accepted_counts = count_effect.checked_add(
            source.tensor,
            one,
            field="test count result",
        )
        accepted = _produce_charge(
            _field(accepted_counts),
            sampling=_sampling(),
            config=ChargeConfig(),
            rng=_FailingRng(seed=1),
            floating_dtype=torch.float64,
        )
        self.assertEqual(accepted.tensor[0, 0, 0].item(), float((1 << 53) - 1))
        with self.assertRaises(RuntimeError):
            count_effect.checked_add(
                source.tensor,
                two,
                field="test count result",
            )

        invalid_values = torch.zeros((2, 2, 4), dtype=torch.int64)
        invalid_values[0, 0, 0] = 1 << 53
        invalid = _field(invalid_values)
        with self.assertRaises(ValueError):
            _produce_charge(
                invalid,
                sampling=_sampling(),
                config=ChargeConfig(),
                rng=_FailingRng(seed=0),
                floating_dtype=torch.float64,
            )

        complete_grid = torch.full(
            (2, 2, 4),
            (1 << 53) - 1,
            dtype=torch.int64,
        )
        aggregate_above_cell_ceiling = _produce_charge(
            _field(complete_grid),
            sampling=_sampling(),
            config=ChargeConfig(),
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float64,
        )
        self.assertTrue(
            torch.equal(
                aggregate_above_cell_ceiling.tensor,
                complete_grid.to(torch.float64),
            )
        )

    def test_dark_mean_exact_rational_endpoint_and_adjacent_rejection(
        self,
    ) -> None:
        period = 2677300530967072003
        endpoint_rate = 37.35105523020191
        above_rate = math.nextafter(endpoint_rate, math.inf)
        sampling = SamplingConfig(
            sample_period_ps=PositiveInteger(period),
            sample_count=PositiveInteger(2),
        )
        numerator, denominator = endpoint_rate.as_integer_ratio()
        exact_endpoint = Fraction(
            numerator * period,
            denominator * 10**12,
        )
        above_numerator, above_denominator = above_rate.as_integer_ratio()
        exact_above = Fraction(
            above_numerator * period,
            above_denominator * 10**12,
        )
        exact_ceiling = Fraction(100_000_000)
        self.assertLess(exact_endpoint, exact_ceiling)
        self.assertEqual(float(exact_endpoint), 100_000_000.0)
        self.assertGreater(exact_above, exact_ceiling)
        self.assertEqual(math.nextafter(endpoint_rate, math.inf), above_rate)

        endpoint_config = DarkCountConfig(
            rate_hz=NonnegativeFloat(endpoint_rate)
        )
        self.assertEqual(
            dark_effect._prepare_dark_mean(
                endpoint_config,
                sampling=_sampling_runtime(sampling),
            ),
            100_000_000.0,
        )
        source = _ensemble_field(
            torch.zeros((1, 1, 2), dtype=torch.int64),
            sampling=sampling,
        )
        original = source.tensor.clone()
        state = torch.random.get_rng_state().clone()
        accepted = _produce_charge(
            source,
            sampling=sampling,
            config=ChargeConfig(dark_count=endpoint_config),
            rng=Threefry4x32(seed=0),
            floating_dtype=torch.float64,
        )
        self.assertTrue(bool(torch.all(accepted.tensor > 0.0).item()))
        self.assertTrue(torch.equal(source.tensor, original))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

        invalid_config = ChargeConfig(
            dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(above_rate))
        )
        with self.assertRaisesRegex(
            ValueError,
            "dark-count mean exceeds the accepted Poisson domain",
        ):
            _produce_charge(
                source,
                sampling=sampling,
                config=invalid_config,
                rng=_FailingRng(seed=0),
                floating_dtype=torch.float64,
            )
        self.assertTrue(torch.equal(source.tensor, original))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_preflight_failure_precedes_rng_and_preserves_source_and_global_rng(self) -> None:
        values = torch.zeros((2, 2, 4), dtype=torch.int64)
        values[0, 0, 0] = 1
        source = _field(values)
        original = source.tensor.clone()
        state = torch.random.get_rng_state().clone()
        bad = ChargeConfig(
            timing_jitter=TimingJitterConfig(sigma_ns=NonnegativeFloat(200.0)),
            dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(1.0e10)),
        )
        with self.assertRaises(ValueError):
            _produce_charge(
                source,
                sampling=_sampling(),
                config=bad,
                rng=_FailingRng(seed=1),
                floating_dtype=torch.float32,
            )
        self.assertTrue(torch.equal(source.tensor, original))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))


class DarkCountStatisticalTest(unittest.TestCase):
    def test_dark_poisson_mean_variance_zero_pmf_and_tail(self) -> None:
        per_seed = 1 << 16
        seeds = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
        sampling = SamplingConfig(
            sample_period_ps=PositiveInteger(2000),
            sample_count=PositiveInteger(2),
        )
        config = DarkCountConfig(rate_hz=NonnegativeFloat(2.0e9))
        observations: list[torch.Tensor] = []
        for seed in seeds:
            counts = torch.zeros((per_seed, 1, 2), dtype=torch.int64)
            dark = dark_effect.simulate_dark_counts(
                counts,
                runtime=dark_effect.prepare_dark_counts(
                    config,
                    sampling=_sampling_runtime(sampling),
                ),
                rng=Threefry4x32(seed=seed),
            )
            observations.append(dark[:, 0, 0].to(torch.float64))

        values = torch.cat(observations)
        total = 1 << 18
        self.assertEqual(values.numel(), total)
        poisson_mean = 4.0
        probabilities = [math.exp(-poisson_mean)]
        for count in range(1, 8):
            probabilities.append(
                probabilities[-1] * poisson_mean / float(count)
            )
        targets = (
            (
                "mean",
                float(torch.mean(values)),
                poisson_mean,
                math.sqrt(poisson_mean / total),
            ),
            (
                "centered variance",
                float(torch.mean((values - poisson_mean) ** 2)),
                poisson_mean,
                math.sqrt(
                    (poisson_mean + 2.0 * poisson_mean * poisson_mean)
                    / total
                ),
            ),
            (
                "zero probability",
                float(torch.mean((values == 0.0).to(torch.float64))),
                probabilities[0],
                math.sqrt(
                    probabilities[0] * (1.0 - probabilities[0]) / total
                ),
            ),
            (
                "PMF at four",
                float(torch.mean((values == 4.0).to(torch.float64))),
                probabilities[4],
                math.sqrt(
                    probabilities[4] * (1.0 - probabilities[4]) / total
                ),
            ),
            (
                "tail at eight",
                float(torch.mean((values >= 8.0).to(torch.float64))),
                1.0 - math.fsum(probabilities),
                math.sqrt(
                    (1.0 - math.fsum(probabilities))
                    * math.fsum(probabilities)
                    / total
                ),
            ),
        )
        for name, observed, target, standard_error in targets:
            if name in ("zero probability", "PMF at four", "tail at eight"):
                self.assertGreaterEqual(total * target, 256.0, name)
                self.assertGreaterEqual(total * (1.0 - target), 256.0, name)
            _assert_statistic(
                self,
                name=name,
                observed=observed,
                target=target,
                standard_error=standard_error,
                dtype=torch.float64,
                accumulation_length=total,
            )


class ChargeSmearingTest(unittest.TestCase):
    def test_finite_envelope_has_exact_adjacent_dtype_boundaries(self) -> None:
        cases = (
            (
                torch.float32,
                "0x1.f61fea0000000p+98",
                "0x1.f61fec0000000p+98",
                "0x1.fffffc0000000p+127",
                (0, 0x80000000, 0, 0),
            ),
            (
                torch.float64,
                "0x1.51e4a059b7cf4p+994",
                "0x1.51e4a059b7cf5p+994",
                "0x1.ffffffffffff9p+1023",
                (0, 0, 0x80000000, 0),
            ),
        )
        for (
            dtype,
            accepted_hex,
            rejected_hex,
            positive_hex,
            negative_words,
        ) in cases:
            with self.subTest(dtype=dtype):
                _, ledger_bound = correlated_effect.prepare_ledger_envelope(
                    floating_dtype=dtype,
                    maximum_generations=0,
                    retained_mechanisms=0,
                    recovered_afterpulse=False,
                    sample_count=4,
                )
                maximum_ledger = torch.tensor(
                    float.fromhex("0x1.0000000000000p+53"),
                    dtype=dtype,
                )
                self.assertLessEqual(float(maximum_ledger), ledger_bound)
                self.assertGreater(
                    float(
                        torch.nextafter(
                            maximum_ledger,
                            torch.tensor(math.inf, dtype=dtype),
                        )
                    ),
                    ledger_bound,
                )
                if dtype is torch.float32:
                    self.assertGreater(
                        float(torch.tensor(ledger_bound, dtype=dtype)),
                        ledger_bound,
                    )
                accepted = float.fromhex(accepted_hex)
                rejected = float.fromhex(rejected_hex)
                represented = smearing_effect._prepare_smearing_sigma(
                    ChargeSmearingConfig(
                        relative_sigma=NonnegativeFloat(accepted)
                    ),
                    floating_dtype=dtype,
                    ledger_bound=ledger_bound,
                    device=maximum_ledger.device,
                )
                self.assertEqual(represented, accepted)
                with self.assertRaisesRegex(ValueError, "finite envelope"):
                    smearing_effect._prepare_smearing_sigma(
                        ChargeSmearingConfig(
                            relative_sigma=NonnegativeFloat(rejected)
                        ),
                        floating_dtype=dtype,
                        ledger_bound=ledger_bound,
                        device=maximum_ledger.device,
                    )

                ledgers = maximum_ledger.reshape(1)
                for words, expected_hex in (
                    ((0, 0, 0, 0), positive_hex),
                    (negative_words, "0x0.0p+0"),
                ):
                    with self.subTest(dtype=dtype, words=words):
                        _FixedBlockRng.use(words)
                        result = smearing_effect.simulate_charge_smearing(
                            ledgers,
                            ledgers,
                            runtime=smearing_effect.prepare_charge_smearing(
                                ChargeSmearingConfig(
                                    relative_sigma=NonnegativeFloat(accepted)
                                ),
                                floating_dtype=dtype,
                                ledger_bound=ledger_bound,
                                device=ledgers.device,
                            ),
                            rng=_FixedBlockRng(seed=1),
                        )
                        self.assertEqual(_FixedBlockRng.calls, 2)
                        self.assertTrue(
                            bool(torch.all(torch.isfinite(result)).item())
                        )
                        self.assertTrue(bool(torch.all(result >= 0.0).item()))
                        self.assertEqual(float(result[0]).hex(), expected_hex)

        for requested in (2.0**-150, 2.0**128):
            with self.subTest(float32_requested=requested):
                _, ledger_bound = correlated_effect.prepare_ledger_envelope(
                    floating_dtype=torch.float32,
                    maximum_generations=0,
                    retained_mechanisms=0,
                    recovered_afterpulse=False,
                    sample_count=4,
                )
                with self.assertRaisesRegex(ValueError, "finite"):
                    smearing_effect._prepare_smearing_sigma(
                        ChargeSmearingConfig(
                            relative_sigma=NonnegativeFloat(requested)
                        ),
                        floating_dtype=torch.float32,
                        ledger_bound=ledger_bound,
                        device=torch.device("cpu"),
                    )

    def test_contextual_envelope_rejects_before_any_charge_effect(self) -> None:
        _, ledger_bound = correlated_effect.prepare_ledger_envelope(
            floating_dtype=torch.float32,
            maximum_generations=23,
            retained_mechanisms=1,
            recovered_afterpulse=False,
            sample_count=4,
        )
        accepted = float.fromhex("0x1.f61fd20000000p+98")
        rejected = float.fromhex("0x1.f61fd40000000p+98")
        represented = smearing_effect._prepare_smearing_sigma(
            ChargeSmearingConfig(relative_sigma=NonnegativeFloat(accepted)),
            floating_dtype=torch.float32,
            ledger_bound=ledger_bound,
            device=torch.device("cpu"),
        )
        self.assertEqual(represented, accepted)
        with self.assertRaisesRegex(ValueError, "finite envelope"):
            smearing_effect._prepare_smearing_sigma(
                ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(rejected)
                ),
                floating_dtype=torch.float32,
                ledger_bound=ledger_bound,
                device=torch.device("cpu"),
            )

        source = _field(torch.ones((2, 2, 4), dtype=torch.int64))
        original = source.tensor.clone()
        _RecordingRng.calls = []
        config = ChargeConfig(
            dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(1.0e10)),
            timing_jitter=TimingJitterConfig(sigma_ns=NonnegativeFloat(1.0)),
            correlated_avalanches=CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(23),
                direct_crosstalk=DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.2),
                    delay=FixedDelayConfig(delay_ns=NonnegativeFloat(0.0)),
                ),
            ),
            smearing=ChargeSmearingConfig(
                relative_sigma=NonnegativeFloat(rejected)
            ),
        )
        with (
            mock.patch.object(
                charge_producer,
                "simulate_dark_counts",
                side_effect=AssertionError("dark-count effect reached"),
            ) as dark,
            mock.patch.object(
                charge_producer,
                "simulate_timing_jitter",
                side_effect=AssertionError("timing-jitter effect reached"),
            ) as jitter,
            mock.patch.object(
                charge_producer,
                "simulate_correlated_avalanches",
                side_effect=AssertionError("correlated-avalanche effect reached"),
            ) as correlated,
            self.assertRaisesRegex(ValueError, "finite envelope"),
        ):
            _produce_charge(
                source,
                sampling=_sampling(),
                config=config,
                rng=_RecordingRng(seed=5),
                floating_dtype=torch.float32,
            )
        dark.assert_not_called()
        jitter.assert_not_called()
        correlated.assert_not_called()
        self.assertEqual(_RecordingRng.calls, [])
        self.assertTrue(torch.equal(source.tensor, original))

    def test_zero_s2_still_owns_a_position_and_is_observationally_inert(self) -> None:
        source = _field(torch.zeros((2, 2, 4), dtype=torch.int64))
        _RecordingRng.calls = []
        smearing = ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.1))
        result = _produce_charge(
            source,
            sampling=_sampling(),
            config=ChargeConfig(smearing=smearing),
            rng=_RecordingRng(seed=5),
            floating_dtype=torch.float32,
        )
        self.assertTrue(_RecordingRng.calls)
        self.assertTrue(
            all(call[0] == smearing.rng_key for call in _RecordingRng.calls)
        )
        expected_positions = logical_positions(
            tuple(source.tensor.shape),
            device=source.tensor.device,
        )
        self.assertTrue(
            all(
                torch.equal(call[1], expected_positions)
                for call in _RecordingRng.calls
            )
        )
        self.assertTrue(torch.equal(result.tensor, torch.zeros_like(result.tensor)))

    def test_repeatability_stream_isolation_and_global_rng_immutability(self) -> None:
        values = torch.ones((2, 2, 4), dtype=torch.int64) * 10
        source = _field(values)
        config = ChargeConfig(
            smearing=ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.2))
        )
        state = torch.random.get_rng_state().clone()
        first = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            rng=Threefry4x32(seed=0x0123456789ABCDEF),
            floating_dtype=torch.float64,
        )
        repeated = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            rng=Threefry4x32(seed=0x0123456789ABCDEF),
            floating_dtype=torch.float64,
        )
        other = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            rng=Threefry4x32(seed=1),
            floating_dtype=torch.float64,
        )
        self.assertTrue(torch.equal(first.tensor, repeated.tensor))
        self.assertFalse(torch.equal(first.tensor, other.tensor))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_completed_dark_plus_smearing_charge_matches_independent_mixture(
        self,
    ) -> None:
        per_seed = 1 << 14
        seeds = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
        sampling = SamplingConfig(
            sample_period_ps=PositiveInteger(2000),
            sample_count=PositiveInteger(2),
        )
        observations: list[torch.Tensor] = []
        for seed in seeds:
            source = _ensemble_field(
                torch.zeros((per_seed // 2, 1, 2), dtype=torch.int64),
                sampling=sampling,
            )
            charge = _produce_charge(
                source,
                sampling=sampling,
                config=ChargeConfig(
                    dark_count=DarkCountConfig(
                        rate_hz=NonnegativeFloat(2.0e9)
                    ),
                    smearing=ChargeSmearingConfig(
                        relative_sigma=NonnegativeFloat(1.0)
                    ),
                ),
                rng=Threefry4x32(seed=seed),
                floating_dtype=torch.float64,
            )
            observations.append(charge.tensor.reshape(-1))

        values = torch.cat(observations)
        self.assertEqual(values.numel(), 1 << 16)
        observed_mean = float(torch.mean(values))
        observed_second = float(torch.mean(values * values))
        observed_zero = float(torch.mean((values == 0.0).to(torch.float64)))
        targets = (
            (observed_mean, 4.0249212050092185, 0.010889198244378121),
            (observed_second, 23.970898209067027, 0.11959098010595064),
            (observed_zero, 0.057102939492073235, 0.0009064031540966277),
        )
        epsilon = torch.finfo(torch.float64).eps
        for observed, target, standard_error in targets:
            delta = 64.0 * epsilon * abs(target)
            self.assertLessEqual(
                abs(observed - target),
                8.0 * standard_error + delta,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaChargeProductTest(unittest.TestCase):
    def test_smearing_compatibility_guard_uses_cuda_dtype_arithmetic(self) -> None:
        _, k_zero_bound = correlated_effect.prepare_ledger_envelope(
            floating_dtype=torch.float32,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=4,
        )
        accepted = float.fromhex("0x1.f61fea0000000p+98")
        self.assertEqual(
            smearing_effect._prepare_smearing_sigma(
                ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(accepted)
                ),
                floating_dtype=torch.float32,
                ledger_bound=k_zero_bound,
                device=torch.device("cuda"),
            ),
            accepted,
        )

        _, contextual_bound = correlated_effect.prepare_ledger_envelope(
            floating_dtype=torch.float32,
            maximum_generations=23,
            retained_mechanisms=1,
            recovered_afterpulse=False,
            sample_count=4,
        )
        with self.assertRaisesRegex(ValueError, "finite envelope"):
            smearing_effect._prepare_smearing_sigma(
                ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(
                        float.fromhex("0x1.f61fd40000000p+98")
                    )
                ),
                floating_dtype=torch.float32,
                ledger_bound=contextual_bound,
                device=torch.device("cuda"),
            )

    def test_representative_noncontiguous_sample_first_product_is_repeatable(
        self,
    ) -> None:
        base = torch.zeros((4, 2, 2, 2), dtype=torch.int64, device="cuda")
        values = base[..., 0]
        values[0, 0, 0] = 32
        values[3, 1, 1] = 16
        self.assertFalse(values.is_contiguous())
        source = _field(values, sample_first=True)
        original = source.tensor.clone()
        config = ChargeConfig(
            dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(1.0e9)),
            timing_jitter=TimingJitterConfig(sigma_ns=NonnegativeFloat(0.5)),
            correlated_avalanches=CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(2),
                direct_crosstalk=DirectCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.3),
                    delay=FixedDelayConfig(delay_ns=NonnegativeFloat(1.0)),
                ),
                delayed_crosstalk=DelayedCrosstalkConfig(
                    mean_offspring_per_parent=NonnegativeFloat(0.2),
                    delay=ExponentialDelayConfig(
                        mean_delay_ns=PositiveFloat(4.0)
                    ),
                ),
                afterpulse=AfterpulseConfig(
                    probability=Probability(0.25),
                    mean_delay_ns=PositiveFloat(10.0),
                    recovery=AfterpulseRecoveryConfig(
                        time_constant_ns=PositiveFloat(20.0)
                    ),
                ),
            ),
            smearing=ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.1)),
        )
        first = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            rng=Threefry4x32(seed=0x0123456789ABCDEF),
            floating_dtype=torch.float32,
        )
        repeated = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            rng=Threefry4x32(seed=0x0123456789ABCDEF),
            floating_dtype=torch.float32,
        )
        double = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            rng=Threefry4x32(seed=0x0123456789ABCDEF),
            floating_dtype=torch.float64,
        )
        self.assertEqual(first.tensor.device.type, "cuda")
        self.assertEqual(double.tensor.device.type, "cuda")
        self.assertIs(first.axes, source.axes)
        self.assertTrue(torch.equal(first.tensor, repeated.tensor))
        self.assertTrue(torch.equal(source.tensor, original))
        self.assertNotEqual(
            first.tensor.untyped_storage().data_ptr(),
            source.tensor.untyped_storage().data_ptr(),
        )

    def test_completed_dark_plus_smearing_statistics(self) -> None:
        per_seed = 1 << 14
        sampling = SamplingConfig(
            sample_period_ps=PositiveInteger(2000),
            sample_count=PositiveInteger(2),
        )
        observations: list[torch.Tensor] = []
        for seed in (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF):
            source = _ensemble_field(
                torch.zeros(
                    (per_seed // 2, 1, 2),
                    dtype=torch.int64,
                    device="cuda",
                ),
                sampling=sampling,
            )
            charge = _produce_charge(
                source,
                sampling=sampling,
                config=ChargeConfig(
                    dark_count=DarkCountConfig(
                        rate_hz=NonnegativeFloat(2.0e9)
                    ),
                    smearing=ChargeSmearingConfig(
                        relative_sigma=NonnegativeFloat(1.0)
                    ),
                ),
                rng=Threefry4x32(seed=seed),
                floating_dtype=torch.float64,
            )
            observations.append(charge.tensor.reshape(-1))
        values = torch.cat(observations)
        self.assertEqual(values.numel(), 1 << 16)
        targets = (
            (float(torch.mean(values)), 4.0249212050092185, 0.010889198244378121),
            (
                float(torch.mean(values * values)),
                23.970898209067027,
                0.11959098010595064,
            ),
            (
                float(torch.mean((values == 0.0).to(torch.float64))),
                0.057102939492073235,
                0.0009064031540966277,
            ),
        )
        epsilon = torch.finfo(torch.float64).eps
        for observed, target, standard_error in targets:
            self.assertLessEqual(
                abs(observed - target),
                8.0 * standard_error + 64.0 * epsilon * abs(target),
            )


if __name__ == "__main__":
    unittest.main()
