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
    CorrelatedAvalancheConfig,
    DelayedCrosstalkConfig,
    DirectCrosstalkConfig,
    ExponentialDelayConfig,
    FixedDelayConfig,
    SamplingConfig,
)
from tensor_dslab.readout._random import _RngStream
from tensor_dslab.readout.charge import _produce as charge_produce
from tensor_dslab.readout.charge._produce import _simulate_correlated_avalanches


def _sampling(*, count: int = 4) -> SamplingConfig:
    return SamplingConfig(
        sample_period_ps=PositiveInteger(2000),
        sample_count=PositiveInteger(count),
    )


def _direct(*, mean: float = 0.3, delay_ns: float = 0.0) -> DirectCrosstalkConfig:
    return DirectCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(mean),
        delay=FixedDelayConfig(delay_ns=NonnegativeFloat(delay_ns)),
    )


def _delayed(*, mean: float = 0.2, delay_ns: float = 2.0) -> DelayedCrosstalkConfig:
    return DelayedCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(mean),
        delay=FixedDelayConfig(delay_ns=NonnegativeFloat(delay_ns)),
    )


def _afterpulse(*, probability: float = 0.25, recovery: bool = False) -> AfterpulseConfig:
    return AfterpulseConfig(
        probability=Probability(probability),
        mean_delay_ns=PositiveFloat(10.0),
        recovery=(
            AfterpulseRecoveryConfig(time_constant_ns=PositiveFloat(20.0))
            if recovery
            else None
        ),
    )


def _simulate(
    roots: torch.Tensor,
    config: CorrelatedAvalancheConfig,
    *,
    dtype: torch.dtype = torch.float64,
    seed: int | None = 1234,
) -> charge_produce._CorrelatedAvalancheResult:
    return _simulate_correlated_avalanches(
        roots,
        sample_dimension=roots.ndim - 1,
        sampling=_sampling(count=roots.shape[-1]),
        floating_dtype=dtype,
        config=config,
        seed=seed,
    )


class CorrelatedAvalancheIdentityTest(unittest.TestCase):
    def test_k_zero_and_ineffective_modes_are_draw_free(self) -> None:
        roots = torch.tensor((3, 0, 1, 0), dtype=torch.int64).reshape(1, 1, 4)
        configured = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(0),
            direct_crosstalk=_direct(mean=1.0, delay_ns=1.0e100),
            delayed_crosstalk=_delayed(mean=1.0, delay_ns=1.0e100),
            afterpulse=_afterpulse(probability=1.0, recovery=True),
        )
        with patch(
            "tensor_dslab.readout.charge._produce._sample_poisson",
            side_effect=AssertionError("K=0 must not sample"),
        ), patch(
            "tensor_dslab.readout.charge._produce._sample_conditional_binomial",
            side_effect=AssertionError("K=0 must not sample"),
        ):
            result = _simulate(roots, configured, seed=None)
        self.assertTrue(torch.equal(result.final_frontier, roots))
        self.assertTrue(torch.equal(result.total_count, roots))
        self.assertTrue(torch.equal(result.S1, roots.to(torch.float64)))
        self.assertTrue(torch.equal(result.S2, roots.to(torch.float64)))
        self.assertIsNone(result.direct_crosstalk_count)
        self.assertIsNone(result.delayed_crosstalk_count)
        self.assertIsNone(result.afterpulse_count)

        ineffective = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(10),
            direct_crosstalk=_direct(mean=0.0, delay_ns=1.0e100),
            delayed_crosstalk=_delayed(mean=0.0, delay_ns=1.0e100),
            afterpulse=_afterpulse(probability=0.0, recovery=True),
        )
        result = _simulate(roots, ineffective, seed=None)
        self.assertTrue(torch.equal(result.final_frontier, torch.zeros_like(roots)))
        self.assertTrue(torch.equal(result.total_count, roots))

    def test_contextual_identity_bypasses_inactive_numerical_and_address_gates(
        self,
    ) -> None:
        inactive_delay = ExponentialDelayConfig(
            mean_delay_ns=PositiveFloat(1.0e100)
        )
        inactive_recovery = AfterpulseRecoveryConfig(
            time_constant_ns=PositiveFloat(1.0e100)
        )
        k_zero = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(0),
            direct_crosstalk=DirectCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(1.0),
                delay=inactive_delay,
            ),
            delayed_crosstalk=DelayedCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(1.0),
                delay=inactive_delay,
            ),
            afterpulse=AfterpulseConfig(
                probability=Probability(1.0),
                mean_delay_ns=PositiveFloat(1.0e100),
                recovery=inactive_recovery,
            ),
        )
        roots = torch.zeros((1, 1, 8193), dtype=torch.int64)
        roots[0, 0, 0] = 1
        with patch.object(
            charge_produce,
            "_prepare_delay",
            side_effect=AssertionError("K=0 must not prepare crosstalk"),
        ), patch.object(
            charge_produce,
            "_prepare_exponential_delay",
            side_effect=AssertionError("K=0 must not prepare afterpulsing"),
        ), patch.object(
            charge_produce,
            "_sample_poisson",
            side_effect=AssertionError("K=0 must not sample"),
        ), patch.object(
            charge_produce,
            "_sample_conditional_binomial",
            side_effect=AssertionError("K=0 must not sample"),
        ):
            result = _simulate_correlated_avalanches(
                roots,
                sample_dimension=2,
                sampling=_sampling(count=8193),
                floating_dtype=torch.float32,
                config=k_zero,
                seed=None,
            )
        self.assertTrue(torch.equal(result.total_count, roots))

        zero_effect = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1 << 63),
            direct_crosstalk=DirectCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.0),
                delay=inactive_delay,
            ),
            delayed_crosstalk=DelayedCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.0),
                delay=inactive_delay,
            ),
            afterpulse=AfterpulseConfig(
                probability=Probability(0.0),
                mean_delay_ns=PositiveFloat(1.0e100),
                recovery=inactive_recovery,
            ),
        )
        small_roots = torch.tensor((1, 0), dtype=torch.int64).reshape(1, 1, 2)
        with patch.object(
            charge_produce,
            "_prepare_delay",
            side_effect=AssertionError("zero crosstalk must not prepare delay"),
        ), patch.object(
            charge_produce,
            "_prepare_exponential_delay",
            side_effect=AssertionError("zero afterpulsing must not prepare delay"),
        ), patch.object(
            charge_produce,
            "_prepare_afterpulse_recovery",
            side_effect=AssertionError("zero afterpulsing must not prepare recovery"),
        ), patch.object(
            charge_produce,
            "_sample_poisson",
            side_effect=AssertionError("zero effects must not sample"),
        ), patch.object(
            charge_produce,
            "_sample_conditional_binomial",
            side_effect=AssertionError("zero effects must not sample"),
        ):
            result = _simulate_correlated_avalanches(
                small_roots,
                sample_dimension=2,
                sampling=_sampling(count=2),
                floating_dtype=torch.float64,
                config=zero_effect,
                seed=None,
            )
        self.assertTrue(torch.equal(result.total_count, small_roots))

    def test_all_eight_structural_mechanism_combinations(self) -> None:
        roots = torch.tensor((8, 0, 0, 0), dtype=torch.int64).reshape(1, 1, 4)
        for direct_on, delayed_on, afterpulse_on in itertools.product((False, True), repeat=3):
            with self.subTest(
                direct=direct_on,
                delayed=delayed_on,
                afterpulse=afterpulse_on,
            ):
                config = CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(1),
                    direct_crosstalk=_direct() if direct_on else None,
                    delayed_crosstalk=_delayed() if delayed_on else None,
                    afterpulse=_afterpulse() if afterpulse_on else None,
                )
                result = _simulate(
                    roots,
                    config,
                    seed=1234 if any((direct_on, delayed_on, afterpulse_on)) else None,
                )
                contributions = roots.clone()
                for value in (
                    result.direct_crosstalk_count,
                    result.delayed_crosstalk_count,
                    result.afterpulse_count,
                ):
                    if value is not None:
                        contributions = contributions + value
                self.assertTrue(torch.equal(result.total_count, contributions))
                self.assertEqual(result.direct_crosstalk_count is not None, direct_on)
                self.assertEqual(result.delayed_crosstalk_count is not None, delayed_on)
                self.assertEqual(result.afterpulse_count is not None, afterpulse_on)


class CorrelatedAvalancheMechanismTest(unittest.TestCase):
    def test_mechanism_order_streams_and_separate_ct_draws(self) -> None:
        roots = torch.tensor((1000, 0, 0, 0), dtype=torch.int64).reshape(1, 1, 4)
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            direct_crosstalk=_direct(mean=0.5),
            delayed_crosstalk=_delayed(mean=0.5, delay_ns=0.0),
            afterpulse=_afterpulse(probability=0.5),
        )
        streams: list[_RngStream] = []
        original = charge_produce._sample_poisson

        def record(*args: object, **kwargs: object) -> torch.Tensor:
            stream = kwargs["stream"]
            assert type(stream) is _RngStream
            streams.append(stream)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(charge_produce, "_sample_poisson", side_effect=record):
            result = _simulate(roots, config)
        self.assertEqual(
            streams,
            [
                _RngStream.CHARGE_DIRECT_CROSSTALK,
                _RngStream.CHARGE_DIRECT_CROSSTALK_OVERFLOW,
                _RngStream.CHARGE_DELAYED_CROSSTALK,
                _RngStream.CHARGE_DELAYED_CROSSTALK_OVERFLOW,
            ],
        )
        assert result.direct_crosstalk_count is not None
        assert result.delayed_crosstalk_count is not None
        self.assertFalse(
            torch.equal(result.direct_crosstalk_count, result.delayed_crosstalk_count)
        )

    def test_absorbing_overflow_never_enters_ledgers_or_frontier(self) -> None:
        roots = torch.tensor((1000, 0, 0, 0), dtype=torch.int64).reshape(1, 1, 4)
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            direct_crosstalk=_direct(mean=1.0, delay_ns=8.0),
        )
        result = _simulate(roots, config)
        assert result.direct_crosstalk_count is not None
        assert result.direct_crosstalk_overflow_count is not None
        self.assertEqual(int(result.direct_crosstalk_count.sum()), 0)
        self.assertGreater(int(result.direct_crosstalk_overflow_count.sum()), 0)
        self.assertTrue(torch.equal(result.final_frontier, torch.zeros_like(roots)))
        self.assertTrue(torch.equal(result.total_count, roots))
        self.assertTrue(torch.equal(result.S1, roots.to(torch.float64)))

    def test_recovery_weights_charge_only_and_preserves_count_history_across_dtypes(self) -> None:
        roots = torch.tensor((32, 0, 0, 0), dtype=torch.int64).reshape(1, 1, 4)
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(2),
            direct_crosstalk=_direct(),
            afterpulse=_afterpulse(probability=0.5, recovery=True),
        )
        single = _simulate(roots, config, dtype=torch.float32, seed=77)
        double = _simulate(roots, config, dtype=torch.float64, seed=77)
        for name in (
            "final_frontier",
            "total_count",
            "direct_crosstalk_count",
            "direct_crosstalk_overflow_count",
            "afterpulse_count",
            "afterpulse_overflow_count",
        ):
            self.assertTrue(torch.equal(getattr(single, name), getattr(double, name)), name)
        assert single.afterpulse_charge is not None
        assert single.afterpulse_charge_square_sum is not None
        self.assertTrue(
            bool(
                torch.all(
                    single.afterpulse_charge_square_sum
                    <= single.afterpulse_charge
                ).item()
            )
        )
        self.assertTrue(bool(torch.all(single.S2 <= single.S1).item()))
        self.assertTrue(bool(torch.all(single.S1 <= single.total_count.to(torch.float32)).item()))

    def test_unit_response_aggregates_counts_before_float32_ledger_conversion(self) -> None:
        large = (1 << 24) + 1
        roots = torch.tensor((large, 1, 1, 1), dtype=torch.int64).reshape(1, 1, 4)
        scheduled = [torch.zeros((1, 1), dtype=torch.int64) for _ in range(14)]
        scheduled[3] = torch.full((1, 1), large, dtype=torch.int64)
        scheduled[7] = torch.ones((1, 1), dtype=torch.int64)
        scheduled[10] = torch.ones((1, 1), dtype=torch.int64)
        scheduled[12] = torch.ones((1, 1), dtype=torch.int64)

        with patch.object(
            charge_produce,
            "_sample_conditional_binomial",
            side_effect=scheduled,
        ):
            result = _simulate(
                roots,
                CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(1),
                    afterpulse=_afterpulse(probability=0.5),
                ),
                dtype=torch.float32,
            )

        assert result.afterpulse_count is not None
        assert result.afterpulse_charge is not None
        assert result.afterpulse_charge_square_sum is not None
        represented_count = result.afterpulse_count.to(torch.float32)
        self.assertTrue(torch.equal(result.afterpulse_charge, represented_count))
        self.assertTrue(
            torch.equal(result.afterpulse_charge_square_sum, represented_count)
        )

        depth = 2
        gamma = depth / ((1 << 24) - depth)
        reference = result.total_count.to(torch.float64)
        tolerance = gamma * reference + depth * (2.0**-149)
        self.assertTrue(
            bool(torch.all(torch.abs(result.S1.to(torch.float64) - reference) <= tolerance))
        )
        self.assertTrue(
            bool(torch.all(torch.abs(result.S2.to(torch.float64) - reference) <= tolerance))
        )

    def test_recovery_square_sum_uses_count_times_prepared_weight_squared(
        self,
    ) -> None:
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            afterpulse=_afterpulse(probability=0.5, recovery=True),
        )
        for dtype, count in ((torch.float32, 100), (torch.float64, 3)):
            with self.subTest(dtype=dtype):
                roots = torch.tensor(
                    (count, 0, 0, 0),
                    dtype=torch.int64,
                ).reshape(1, 1, 4)
                scheduled = [
                    torch.zeros((1, 1), dtype=torch.int64)
                    for _ in range(14)
                ]
                scheduled[0] = torch.full(
                    (1, 1),
                    count,
                    dtype=torch.int64,
                )
                with patch.object(
                    charge_produce,
                    "_sample_conditional_binomial",
                    side_effect=scheduled,
                ):
                    result = _simulate(roots, config, dtype=dtype)

                plan = charge_produce._prepare_correlated_plan(
                    config,
                    sampling=_sampling(),
                    floating_dtype=dtype,
                    tensor_numel=roots.numel(),
                )
                assert plan.afterpulse is not None
                assert plan.afterpulse.recovery is not None
                represented_count = torch.tensor(float(count), dtype=dtype)
                represented_weight = torch.tensor(
                    plan.afterpulse.recovery[0],
                    dtype=dtype,
                )
                expected = represented_count * (
                    represented_weight * represented_weight
                )
                retired_reassociation = (
                    represented_count * represented_weight
                ) * represented_weight
                self.assertNotEqual(expected.item(), retired_reassociation.item())
                assert result.afterpulse_charge_square_sum is not None
                self.assertEqual(
                    result.afterpulse_charge_square_sum[0, 0, 0].item(),
                    expected.item(),
                )

    def test_exact_integer_count_identity_is_checked_before_return(self) -> None:
        roots = torch.tensor((1, 0, 0, 0), dtype=torch.int64).reshape(1, 1, 4)
        original = charge_produce._checked_add

        def corrupt_total(
            left: torch.Tensor,
            right: torch.Tensor,
            *,
            field: str,
        ) -> torch.Tensor:
            result = original(left, right, field=field)
            if field == "correlated-avalanche total count":
                return result + 1
            return result

        with patch.object(charge_produce, "_checked_add", side_effect=corrupt_total):
            with self.assertRaisesRegex(RuntimeError, "integer count identity"):
                _simulate(
                    roots,
                    CorrelatedAvalancheConfig(
                        maximum_generations=NonnegativeInteger(1),
                        direct_crosstalk=_direct(mean=0.1),
                    ),
                )


class CorrelatedAvalancheEnvelopeTest(unittest.TestCase):
    def test_exact_address_product_boundaries_without_lattice_materialization(
        self,
    ) -> None:
        crosstalk = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(2),
            direct_crosstalk=_direct(mean=0.1),
        )
        accepted = charge_produce._prepare_correlated_plan(
            crosstalk,
            sampling=_sampling(count=2),
            floating_dtype=torch.float64,
            tensor_numel=1 << 62,
        )
        self.assertIsNotNone(accepted.direct_crosstalk)
        with self.assertRaisesRegex(ValueError, "crosstalk address lattice"):
            charge_produce._prepare_correlated_plan(
                crosstalk,
                sampling=_sampling(count=2),
                floating_dtype=torch.float64,
                tensor_numel=(1 << 62) + 1,
            )

        afterpulse = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(2),
            afterpulse=_afterpulse(probability=0.25),
        )
        accepted = charge_produce._prepare_correlated_plan(
            afterpulse,
            sampling=_sampling(count=3),
            floating_dtype=torch.float64,
            tensor_numel=1 << 60,
        )
        self.assertIsNotNone(accepted.afterpulse)
        with self.assertRaisesRegex(ValueError, "afterpulse address lattice"):
            charge_produce._prepare_correlated_plan(
                afterpulse,
                sampling=_sampling(count=3),
                floating_dtype=torch.float64,
                tensor_numel=(1 << 60) + 1,
            )

    def test_allocation_byte_and_element_products_have_exact_boundaries(self) -> None:
        self.assertEqual(
            charge_produce._require_tensor_allocation(
                ((1 << 63) - 1,),
                element_size=1,
                field="test",
            ),
            (1 << 63) - 1,
        )
        with self.assertRaises(ValueError):
            charge_produce._require_tensor_allocation(
                (1 << 63,),
                element_size=1,
                field="test",
            )
        self.assertEqual(
            charge_produce._require_tensor_allocation(
                ((1 << 60) - 1,),
                element_size=8,
                field="test",
            ),
            (1 << 60) - 1,
        )
        with self.assertRaises(ValueError):
            charge_produce._require_tensor_allocation(
                (1 << 60,),
                element_size=8,
                field="test",
            )

    def test_ledger_depth_boundaries_match_the_independent_equation(self) -> None:
        for dtype, precision in ((torch.float32, 24), (torch.float64, 53)):
            with self.subTest(dtype=dtype, recovered=False):
                maximum_generations = (1 << precision) - 2
                depth, bound = charge_produce._ledger_envelope(
                    floating_dtype=dtype,
                    maximum_generations=maximum_generations,
                    retained_mechanisms=1,
                    recovered_afterpulse=False,
                    sample_count=2,
                )
                self.assertEqual(depth, (1 << precision) - 1)
                gamma = depth / ((1 << precision) - depth)
                zero = torch.tensor(0.0, dtype=dtype)
                one = torch.tensor(1.0, dtype=dtype)
                subnormal = float(torch.nextafter(zero, one))
                expected = ((1 << 53) - 1) * (1.0 + gamma) + depth * subnormal
                self.assertEqual(bound, expected)
                with self.assertRaisesRegex(ValueError, "ledger depth"):
                    charge_produce._ledger_envelope(
                        floating_dtype=dtype,
                        maximum_generations=maximum_generations + 1,
                        retained_mechanisms=1,
                        recovered_afterpulse=False,
                        sample_count=2,
                    )

            with self.subTest(dtype=dtype, recovered=True):
                maximum_generations = (1 << precision) - 6
                depth, _ = charge_produce._ledger_envelope(
                    floating_dtype=dtype,
                    maximum_generations=maximum_generations,
                    retained_mechanisms=1,
                    recovered_afterpulse=True,
                    sample_count=2,
                )
                self.assertEqual(depth, (1 << precision) - 1)
                with self.assertRaisesRegex(ValueError, "ledger depth"):
                    charge_produce._ledger_envelope(
                        floating_dtype=dtype,
                        maximum_generations=maximum_generations + 1,
                        retained_mechanisms=1,
                        recovered_afterpulse=True,
                        sample_count=2,
                    )


class CorrelatedAvalancheStatisticalTest(unittest.TestCase):
    def test_direct_one_generation_q32_poisson_moments(self) -> None:
        per_seed = 1 << 14
        seeds = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
        mean = 0.3
        observations = []
        for seed in seeds:
            roots = torch.zeros((per_seed, 1, 2), dtype=torch.int64)
            roots[:, 0, 0] = 32
            result = _simulate(
                roots,
                CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(1),
                    direct_crosstalk=_direct(mean=mean),
                ),
                seed=seed,
            )
            assert result.direct_crosstalk_count is not None
            observations.append(result.direct_crosstalk_count[:, 0, 0].to(torch.float64))
        values = torch.cat(observations)
        target = 32.0 * mean
        total = values.numel()
        observed_mean = float(torch.mean(values))
        observed_variance = float(torch.var(values, correction=0))
        self.assertLessEqual(abs(observed_mean - target), 8.0 * math.sqrt(target / total))
        self.assertLessEqual(
            abs(observed_variance - target),
            8.0 * target * math.sqrt(2.0 / total),
        )

    def test_three_generation_small_grid_matches_branching_recurrence(self) -> None:
        per_seed = 1 << 14
        seeds = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
        mean = 0.2
        observations: list[torch.Tensor] = []
        for seed in seeds:
            roots = torch.zeros((per_seed, 1, 2), dtype=torch.int64)
            roots[:, 0, 0] = 1
            result = _simulate(
                roots,
                CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(3),
                    direct_crosstalk=_direct(mean=mean),
                ),
                seed=seed,
            )
            observations.append(result.final_frontier[:, 0, 0].to(torch.float64))

        values = torch.cat(observations)
        self.assertEqual(values.numel(), 1 << 16)
        generation_mean = 1.0
        generation_variance = 0.0
        for _ in range(3):
            generation_variance = (
                mean * generation_mean + mean * mean * generation_variance
            )
            generation_mean = mean * generation_mean
        observed_mean = float(torch.mean(values))
        observed_variance = float(torch.var(values, correction=0))
        self.assertLessEqual(
            abs(observed_mean - generation_mean),
            8.0 * math.sqrt(generation_variance / values.numel()),
        )
        self.assertLessEqual(
            abs(observed_variance - generation_variance),
            8.0 * generation_variance * math.sqrt(2.0 / values.numel()),
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaCorrelatedAvalancheTest(unittest.TestCase):
    def test_fixed_and_exponential_crosstalk_with_recovered_afterpulses(self) -> None:
        roots = torch.zeros((2, 1, 8), dtype=torch.int64, device="cuda")
        roots[0, 0, 0] = 1024
        roots[1, 0, -1] = 4096
        original_roots = roots.clone()
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(2),
            direct_crosstalk=_direct(mean=0.3, delay_ns=1.0),
            delayed_crosstalk=DelayedCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.2),
                delay=ExponentialDelayConfig(
                    mean_delay_ns=PositiveFloat(4.0),
                ),
            ),
            afterpulse=_afterpulse(probability=0.35, recovery=True),
        )

        single = _simulate(roots, config, dtype=torch.float32, seed=0x12345678)
        repeated = _simulate(roots, config, dtype=torch.float32, seed=0x12345678)
        double = _simulate(roots, config, dtype=torch.float64, seed=0x12345678)
        result_fields = (
            "S1",
            "S2",
            "final_frontier",
            "total_count",
            "direct_crosstalk_count",
            "direct_crosstalk_overflow_count",
            "delayed_crosstalk_count",
            "delayed_crosstalk_overflow_count",
            "afterpulse_count",
            "afterpulse_overflow_count",
            "afterpulse_charge",
            "afterpulse_overflow_charge",
            "afterpulse_charge_square_sum",
        )
        for name in result_fields:
            observed = getattr(single, name)
            repeated_observed = getattr(repeated, name)
            self.assertIsNotNone(observed, name)
            assert observed is not None
            assert repeated_observed is not None
            self.assertEqual(observed.device.type, "cuda", name)
            self.assertTrue(torch.equal(observed, repeated_observed), name)

        integer_history = (
            "final_frontier",
            "total_count",
            "direct_crosstalk_count",
            "direct_crosstalk_overflow_count",
            "delayed_crosstalk_count",
            "delayed_crosstalk_overflow_count",
            "afterpulse_count",
            "afterpulse_overflow_count",
        )
        for name in integer_history:
            single_value = getattr(single, name)
            double_value = getattr(double, name)
            assert single_value is not None
            assert double_value is not None
            self.assertEqual(double_value.device.type, "cuda", name)
            self.assertTrue(torch.equal(single_value, double_value), name)

        assert single.direct_crosstalk_count is not None
        assert single.direct_crosstalk_overflow_count is not None
        assert single.delayed_crosstalk_count is not None
        assert single.delayed_crosstalk_overflow_count is not None
        assert single.afterpulse_count is not None
        assert single.afterpulse_overflow_count is not None
        assert single.afterpulse_charge is not None
        assert single.afterpulse_overflow_charge is not None
        self.assertGreater(int(single.direct_crosstalk_count.sum()), 0)
        self.assertGreater(int(single.direct_crosstalk_overflow_count.sum()), 0)
        self.assertGreater(int(single.delayed_crosstalk_count.sum()), 0)
        self.assertGreater(int(single.delayed_crosstalk_overflow_count.sum()), 0)
        self.assertGreater(int(single.afterpulse_count.sum()), 0)
        self.assertGreater(int(single.afterpulse_overflow_count.sum()), 0)
        self.assertGreater(float(single.afterpulse_charge.sum()), 0.0)
        self.assertLess(
            float(single.afterpulse_charge.sum()),
            float(single.afterpulse_count.sum()),
        )
        self.assertGreater(float(single.afterpulse_overflow_charge.sum()), 0.0)
        self.assertLessEqual(
            float(single.afterpulse_overflow_charge.sum()),
            float(single.afterpulse_overflow_count.sum()),
        )
        self.assertEqual(single.S1.dtype, torch.float32)
        self.assertEqual(double.S1.dtype, torch.float64)
        self.assertTrue(torch.equal(roots, original_roots))

if __name__ == "__main__":
    unittest.main()
