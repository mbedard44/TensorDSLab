from __future__ import annotations

import itertools
import math
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    PositiveInteger,
    Probability,
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
from tensor_dslab.readout.charge import _produce as charge_produce
from tensor_dslab.readout.charge._produce import _produce_charge


def _sampling() -> SamplingConfig:
    return SamplingConfig(
        sample_period_ps=PositiveInteger(2000),
        sample_count=PositiveInteger(4),
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
                    seed=1234 if any(flags) else None,
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
                seed=99,
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
        with patch(
            "tensor_dslab.readout.charge._produce._standard_normal_pair",
            side_effect=AssertionError("truth path must not draw"),
        ), patch(
            "tensor_dslab.readout.charge._produce._sample_poisson",
            side_effect=AssertionError("truth path must not draw"),
        ):
            result = _produce_charge(
                source,
                sampling=_sampling(),
                config=ChargeConfig(),
                seed=None,
                floating_dtype=torch.float64,
            )
        self.assertTrue(torch.equal(result.tensor, values.to(torch.float64)))
        self.assertNotEqual(
            result.tensor.untyped_storage().data_ptr(),
            source.tensor.untyped_storage().data_ptr(),
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))


class ChargeProductPreflightTest(unittest.TestCase):
    def test_seed_is_required_only_for_a_path_that_can_draw(self) -> None:
        nonzero = torch.zeros((2, 2, 4), dtype=torch.int64)
        nonzero[0, 0, 0] = 1
        source = _field(nonzero)
        for config in (
            _config(True, False, False, False),
            _config(False, True, False, False),
            _config(False, False, True, False),
            _config(False, False, False, True),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    _produce_charge(
                        source,
                        sampling=_sampling(),
                        config=config,
                        seed=None,
                        floating_dtype=torch.float32,
                    )

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
            seed=None,
            floating_dtype=torch.float32,
        )
        self.assertTrue(torch.equal(result.tensor, torch.zeros_like(result.tensor)))

    def test_source_and_checked_add_count_boundaries(self) -> None:
        source_values = torch.zeros((2, 2, 4), dtype=torch.int64)
        source_values[0, 0, 0] = (1 << 53) - 2
        source = _field(source_values)
        one = torch.zeros_like(source.tensor)
        one[0, 0, 0] = 1
        two = one * 2
        dark = ChargeConfig(
            dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(1.0))
        )
        with patch.object(charge_produce, "_sample_poisson", return_value=one):
            accepted = _produce_charge(
                source,
                sampling=_sampling(),
                config=dark,
                seed=1,
                floating_dtype=torch.float64,
            )
        self.assertEqual(accepted.tensor[0, 0, 0].item(), float((1 << 53) - 1))
        with patch.object(charge_produce, "_sample_poisson", return_value=two):
            with self.assertRaises(RuntimeError):
                _produce_charge(
                    source,
                    sampling=_sampling(),
                    config=dark,
                    seed=1,
                    floating_dtype=torch.float64,
                )

        invalid_values = torch.zeros((2, 2, 4), dtype=torch.int64)
        invalid_values[0, 0, 0] = 1 << 53
        invalid = _field(invalid_values)
        with self.assertRaises(ValueError):
            _produce_charge(
                invalid,
                sampling=_sampling(),
                config=ChargeConfig(),
                seed=None,
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
            seed=None,
            floating_dtype=torch.float64,
        )
        self.assertTrue(
            torch.equal(
                aggregate_above_cell_ceiling.tensor,
                complete_grid.to(torch.float64),
            )
        )

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
        with patch(
            "tensor_dslab.readout.charge._produce._sample_poisson",
            side_effect=AssertionError("preflight must precede RNG"),
        ):
            with self.assertRaises(ValueError):
                _produce_charge(
                    source,
                    sampling=_sampling(),
                    config=bad,
                    seed=1,
                    floating_dtype=torch.float32,
                )
        self.assertTrue(torch.equal(source.tensor, original))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))


class ChargeSmearingTest(unittest.TestCase):
    def test_finite_envelope_has_exact_adjacent_dtype_boundaries(self) -> None:
        cases = (
            (
                torch.float32,
                "0x1.f61fea0000000p+98",
                "0x1.f61fec0000000p+98",
            ),
            (
                torch.float64,
                "0x1.51e4a059b7cf4p+994",
                "0x1.51e4a059b7cf5p+994",
            ),
        )
        for dtype, accepted_hex, rejected_hex in cases:
            with self.subTest(dtype=dtype):
                _, ledger_bound = charge_produce._ledger_envelope(
                    floating_dtype=dtype,
                    maximum_generations=0,
                    retained_mechanisms=0,
                    recovered_afterpulse=False,
                    sample_count=4,
                )
                accepted = float.fromhex(accepted_hex)
                rejected = float.fromhex(rejected_hex)
                represented = charge_produce._prepare_smearing_sigma(
                    ChargeSmearingConfig(
                        relative_sigma=NonnegativeFloat(accepted)
                    ),
                    floating_dtype=dtype,
                    ledger_bound=ledger_bound,
                )
                self.assertEqual(represented, accepted)
                with self.assertRaisesRegex(ValueError, "finite envelope"):
                    charge_produce._prepare_smearing_sigma(
                        ChargeSmearingConfig(
                            relative_sigma=NonnegativeFloat(rejected)
                        ),
                        floating_dtype=dtype,
                        ledger_bound=ledger_bound,
                    )

                precision = 24 if dtype is torch.float32 else 53
                maximum_normal = math.nextafter(
                    math.sqrt(-2.0 * math.log(2.0**-precision)),
                    math.inf,
                )
                ledgers = torch.full((2,), ledger_bound, dtype=dtype)
                standards = torch.tensor(
                    (maximum_normal, -maximum_normal),
                    dtype=dtype,
                )
                with patch.object(
                    charge_produce,
                    "_standard_normal_pair",
                    return_value=(standards, torch.zeros_like(standards)),
                ):
                    result = charge_produce._simulate_charge_smearing(
                        ledgers,
                        ledgers,
                        config=ChargeSmearingConfig(
                            relative_sigma=NonnegativeFloat(accepted)
                        ),
                        seed=1,
                    )
                self.assertTrue(bool(torch.all(torch.isfinite(result)).item()))
                self.assertGreater(float(result[0]), 0.0)
                self.assertEqual(float(result[1]), 0.0)

        for requested in (2.0**-150, 2.0**128):
            with self.subTest(float32_requested=requested):
                _, ledger_bound = charge_produce._ledger_envelope(
                    floating_dtype=torch.float32,
                    maximum_generations=0,
                    retained_mechanisms=0,
                    recovered_afterpulse=False,
                    sample_count=4,
                )
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    charge_produce._prepare_smearing_sigma(
                        ChargeSmearingConfig(
                            relative_sigma=NonnegativeFloat(requested)
                        ),
                        floating_dtype=torch.float32,
                        ledger_bound=ledger_bound,
                    )

    def test_zero_s2_still_owns_a_position_and_is_observationally_inert(self) -> None:
        source = _field(torch.zeros((2, 2, 4), dtype=torch.int64))
        seen: list[torch.Tensor] = []
        original = charge_produce._standard_normal_pair

        def record(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            positions = kwargs["logical_positions"]
            assert type(positions) is torch.Tensor
            seen.append(positions.clone())
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(charge_produce, "_standard_normal_pair", side_effect=record):
            result = _produce_charge(
                source,
                sampling=_sampling(),
                config=ChargeConfig(
                    smearing=ChargeSmearingConfig(
                        relative_sigma=NonnegativeFloat(0.1)
                    )
                ),
                seed=5,
                floating_dtype=torch.float32,
            )
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].shape, source.tensor.shape)
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
            seed=0x0123456789ABCDEF,
            floating_dtype=torch.float64,
        )
        repeated = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            seed=0x0123456789ABCDEF,
            floating_dtype=torch.float64,
        )
        other = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            seed=1,
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
                seed=seed,
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
            seed=0x0123456789ABCDEF,
            floating_dtype=torch.float32,
        )
        repeated = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            seed=0x0123456789ABCDEF,
            floating_dtype=torch.float32,
        )
        double = _produce_charge(
            source,
            sampling=_sampling(),
            config=config,
            seed=0x0123456789ABCDEF,
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
                seed=seed,
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
