import itertools
import math
from typing import ClassVar, override
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    CounterRng,
    NonnegativeFloat,
    NonnegativeInteger,
    Probability,
    RngElements,
    RngKey,
    Threefry4x32,
)
from tensor_core.tensor.validation import require_tensor_allocation

from tensor_dslab import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    CorrelatedAvalancheConfig,
    DelayedCrosstalkConfig,
    DirectCrosstalkConfig,
    ExponentialDelayConfig,
    FixedDelayConfig,
    quantity,
)
from tensor_dslab.readout.charge.runtime.effects import (
    correlated_avalanches as correlated_effect,
)
from tensor_dslab.readout.charge.runtime.effects import counts as count_effect
from tensor_dslab.readout.charge.runtime.effects.correlated_avalanches import (
    _draw_crosstalk,
    prepare_correlated_avalanches,
    prepare_ledger_envelope,
    simulate_correlated_avalanches,
)
from tensor_dslab.readout.runtime.keys import (
    AFTERPULSE_RNG_KEY,
    DELAYED_CROSSTALK_RETAINED_RNG_KEY,
    DIRECT_CROSSTALK_RETAINED_RNG_KEY,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


def _ns(value: int | float):
    return quantity(value, "ns")


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
        return Threefry4x32(seed=self.seed)._generate_block(
            key=key,
            positions=positions,
            quantum=quantum,
            block=block,
        )


def _sampling(*, count: int = 4) -> SamplingRuntime:
    return SamplingRuntime(
        sample_period_ps=2000,
        sample_count=count,
        sample_dimension=2,
    )


def _direct(
    *,
    mean: float = 0.3,
    delay: float = 0.0,
) -> DirectCrosstalkConfig:
    return DirectCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(mean),
        delay=FixedDelayConfig(delay=_ns(delay)),
    )


def _delayed(
    *,
    mean: float = 0.2,
    delay: float = 2.0,
) -> DelayedCrosstalkConfig:
    return DelayedCrosstalkConfig(
        mean_offspring_per_parent=NonnegativeFloat(mean),
        delay=FixedDelayConfig(delay=_ns(delay)),
    )


def _afterpulse(
    *,
    probability: float = 0.25,
    recovery: bool = False,
) -> AfterpulseConfig:
    return AfterpulseConfig(
        probability=Probability(probability),
        mean_delay=_ns(10.0),
        recovery=(
            AfterpulseRecoveryConfig(time_constant=_ns(20.0))
            if recovery
            else None
        ),
    )


def _runtime(
    roots: torch.Tensor,
    config: CorrelatedAvalancheConfig,
    *,
    dtype: torch.dtype = torch.float64,
) -> correlated_effect.CorrelatedAvalancheRuntime:
    return prepare_correlated_avalanches(
        config,
        sampling=_sampling(count=roots.shape[-1]),
        floating_dtype=dtype,
        tensor_numel=roots.numel(),
        device=roots.device,
    )


def _simulate(
    roots: torch.Tensor,
    config: CorrelatedAvalancheConfig,
    *,
    dtype: torch.dtype = torch.float64,
    rng: CounterRng | None = None,
) -> correlated_effect._CorrelatedAvalancheResult:
    runtime = _runtime(roots, config, dtype=dtype)
    return simulate_correlated_avalanches(
        roots,
        sample_dimension=roots.ndim - 1,
        floating_dtype=dtype,
        runtime=runtime,
        rng=Threefry4x32(seed=1234) if rng is None else rng,
        elements=RngElements.from_shape(tuple(roots.shape), device=roots.device),
    )


def _assert_statistic(
    case: unittest.TestCase,
    observed: float,
    target: float,
    standard_error: float,
) -> None:
    case.assertLessEqual(
        abs(observed - target),
        8.0 * standard_error
        + 64.0 * torch.finfo(torch.float64).eps * max(1.0, abs(target)),
    )


class CorrelatedAvalancheIdentityTest(unittest.TestCase):
    def test_k_zero_and_ineffective_modes_are_draw_free(self) -> None:
        roots = torch.tensor([[[2, 0, 1, 0]]], dtype=torch.int64)
        configs = (
            CorrelatedAvalancheConfig(maximum_generations=NonnegativeInteger(0)),
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(3),
                direct_crosstalk=_direct(mean=0.0),
                delayed_crosstalk=_delayed(mean=0.0),
                afterpulse=_afterpulse(probability=0.0),
            ),
        )
        for config in configs:
            result = _simulate(roots, config, rng=_FailingRng(seed=0))
            self.assertTrue(torch.equal(result.S1, roots.to(torch.float64)))
            self.assertTrue(torch.equal(result.S2, roots.to(torch.float64)))
            self.assertTrue(torch.equal(result.total_count, roots))
            self.assertIsNone(result.direct_crosstalk_count)
            self.assertIsNone(result.delayed_crosstalk_count)
            self.assertIsNone(result.afterpulse_count)

    def test_contextual_identity_bypasses_inactive_numerical_and_address_gates(
        self,
    ) -> None:
        roots = torch.zeros((1, 1, 2), dtype=torch.int64)
        inactive = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(0),
            direct_crosstalk=DirectCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(1.0e308),
                delay=ExponentialDelayConfig(mean_delay=_ns(1.0e308)),
            ),
        )
        result = _simulate(roots, inactive, rng=_FailingRng(seed=0))
        self.assertTrue(torch.equal(result.total_count, roots))

    def test_all_eight_structural_mechanism_combinations(self) -> None:
        roots = torch.tensor([[[1, 0, 0, 0]]], dtype=torch.int64)
        for direct, delayed, afterpulse in itertools.product((False, True), repeat=3):
            config = CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(2),
                direct_crosstalk=_direct() if direct else None,
                delayed_crosstalk=_delayed() if delayed else None,
                afterpulse=_afterpulse() if afterpulse else None,
            )
            result = _simulate(roots, config)
            self.assertTrue(torch.all(result.S1 >= 0))
            self.assertTrue(torch.all(result.S2 >= 0))
            self.assertEqual(result.S1.shape, roots.shape)
            self.assertEqual(result.direct_crosstalk_count is not None, direct)
            self.assertEqual(result.delayed_crosstalk_count is not None, delayed)
            self.assertEqual(result.afterpulse_count is not None, afterpulse)

    def test_mechanism_order_keys_quanta_and_separate_crosstalk_draws(self) -> None:
        roots = torch.ones((1, 1, 4), dtype=torch.int64)
        _RecordingRng.calls = []
        _simulate(
            roots,
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                direct_crosstalk=_direct(mean=0.2),
                delayed_crosstalk=_delayed(mean=0.2),
                afterpulse=_afterpulse(probability=0.2),
            ),
            rng=_RecordingRng(seed=0),
        )
        keys = tuple(call[0] for call in _RecordingRng.calls)
        quanta = tuple(call[2] for call in _RecordingRng.calls)
        self.assertIn(DIRECT_CROSSTALK_RETAINED_RNG_KEY, keys)
        self.assertIn(DELAYED_CROSSTALK_RETAINED_RNG_KEY, keys)
        self.assertIn(AFTERPULSE_RNG_KEY, keys)
        self.assertIn(0, quanta)
        self.assertIn(1, quanta)
        first_direct = keys.index(DIRECT_CROSSTALK_RETAINED_RNG_KEY)
        first_delayed = keys.index(DELAYED_CROSSTALK_RETAINED_RNG_KEY)
        first_afterpulse = keys.index(AFTERPULSE_RNG_KEY)
        self.assertLess(first_direct, first_delayed)
        self.assertLess(first_delayed, first_afterpulse)

    def test_finite_window_discards_tail_without_overflow_state(self) -> None:
        roots = torch.zeros((4096, 1, 4), dtype=torch.int64)
        roots[:, 0, -1] = 10
        result = _simulate(
            roots,
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                delayed_crosstalk=_delayed(mean=2.0, delay=2.0),
            ),
        )
        self.assertEqual(
            tuple(result.__dataclass_fields__),
            (
                "S1",
                "S2",
                "final_frontier",
                "total_count",
                "direct_crosstalk_count",
                "delayed_crosstalk_count",
                "afterpulse_count",
                "afterpulse_charge",
                "afterpulse_charge_square_sum",
            ),
        )
        assert result.delayed_crosstalk_count is not None
        self.assertEqual(int(result.delayed_crosstalk_count.sum()), 0)
        self.assertEqual(int(result.total_count.sum()), int(roots.sum()))

    def test_recovery_weights_charge_only_and_preserves_count_history_across_dtypes(
        self,
    ) -> None:
        roots = torch.full((1024, 1, 4), 3, dtype=torch.int64)
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            afterpulse=_afterpulse(probability=0.7, recovery=True),
        )
        float32 = _simulate(roots, config, dtype=torch.float32)
        float64 = _simulate(roots, config, dtype=torch.float64)
        assert float32.afterpulse_count is not None
        assert float64.afterpulse_count is not None
        self.assertTrue(
            torch.equal(float32.afterpulse_count, float64.afterpulse_count)
        )
        assert float64.afterpulse_charge is not None
        self.assertTrue(
            torch.all(
                float64.afterpulse_charge
                <= float64.afterpulse_count.to(torch.float64)
            )
        )

    def test_unit_response_aggregates_counts_before_float32_ledger_conversion(self) -> None:
        roots = torch.tensor([[[1 << 24, 0, 0, 0]]], dtype=torch.int64)
        result = _simulate(
            roots,
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                direct_crosstalk=_direct(mean=0.01),
            ),
            dtype=torch.float32,
        )
        self.assertEqual(result.total_count.dtype, torch.int64)
        self.assertEqual(result.S1.dtype, torch.float32)
        self.assertTrue(torch.all(result.S1 >= roots.to(torch.float32)))

    def test_recovery_square_sum_uses_count_times_prepared_weight_squared(
        self,
    ) -> None:
        roots = torch.full((2048, 1, 4), 4, dtype=torch.int64)
        result = _simulate(
            roots,
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                afterpulse=_afterpulse(probability=0.8, recovery=True),
            ),
        )
        assert result.afterpulse_charge is not None
        assert result.afterpulse_charge_square_sum is not None
        self.assertTrue(
            torch.all(
                result.afterpulse_charge_square_sum
                <= result.afterpulse_charge
            )
        )
        self.assertTrue(torch.any(result.afterpulse_charge > 0))

    def test_exact_integer_count_identity_is_checked_before_return(self) -> None:
        roots = torch.ones((1, 1, 4), dtype=torch.int64)
        with patch.object(
            correlated_effect,
            "checked_add",
            side_effect=lambda left, right, *, field: (
                left + right + 1
                if "total count" in field
                else left + right
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "integer count identity"):
                _simulate(
                    roots,
                    CorrelatedAvalancheConfig(
                        maximum_generations=NonnegativeInteger(1),
                        direct_crosstalk=_direct(mean=0.5),
                    ),
                )

    def test_proved_bound_uses_greatest_represented_ledger_at_most_real_bound(
        self,
    ) -> None:
        for dtype in (torch.float32, torch.float64):
            depth, bound = prepare_ledger_envelope(
                floating_dtype=dtype,
                maximum_generations=23,
                retained_mechanisms=3,
                recovered_afterpulse=True,
                sample_count=24,
            )
            represented = torch.tensor(bound, dtype=dtype)
            if float(represented) > bound:
                represented = torch.nextafter(
                    represented,
                    torch.zeros_like(represented),
                )
            self.assertLessEqual(float(represented), bound)
            self.assertGreater(depth, 1)

    def test_exact_address_product_boundaries_without_lattice_materialization(
        self,
    ) -> None:
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            direct_crosstalk=_direct(),
        )
        runtime = prepare_correlated_avalanches(
            config,
            sampling=_sampling(count=2),
            floating_dtype=torch.float64,
            tensor_numel=1 << 63,
            device=torch.device("cpu"),
        )
        self.assertEqual(runtime.maximum_generations, 1)
        with self.assertRaisesRegex(ValueError, "address lattice"):
            prepare_correlated_avalanches(
                config,
                sampling=_sampling(count=2),
                floating_dtype=torch.float64,
                tensor_numel=(1 << 63) + 1,
                device=torch.device("cpu"),
            )

    def test_allocation_byte_and_element_products_have_exact_boundaries(self) -> None:
        require_tensor_allocation(
            (0, 1 << 62),
            "empty",
            element_size=8,
            upper=1 << 63,
        )
        with self.assertRaises(ValueError):
            require_tensor_allocation(
                (2, 1 << 62),
                "oversized",
                element_size=8,
                upper=1 << 63,
            )

    def test_ledger_depth_boundaries_match_the_independent_equation(self) -> None:
        for dtype, precision in ((torch.float32, 24), (torch.float64, 53)):
            accepted = (1 << precision) - 1
            depth, _ = prepare_ledger_envelope(
                floating_dtype=dtype,
                maximum_generations=accepted - 1,
                retained_mechanisms=1,
                recovered_afterpulse=False,
                sample_count=2,
            )
            self.assertEqual(depth, accepted)
            with self.assertRaisesRegex(ValueError, "ledger depth"):
                prepare_ledger_envelope(
                    floating_dtype=dtype,
                    maximum_generations=accepted,
                    retained_mechanisms=1,
                    recovered_afterpulse=False,
                    sample_count=2,
                )


class CorrelatedAvalancheStatisticalTest(unittest.TestCase):
    def test_direct_q32_three_generation_branching_moments_and_covariances(
        self,
    ) -> None:
        sample_size = 1 << 15
        roots = torch.zeros((sample_size, 1, 2), dtype=torch.int64)
        roots[:, 0, 0] = 4
        result = _simulate(
            roots,
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                direct_crosstalk=_direct(mean=0.5),
            ),
        )
        assert result.direct_crosstalk_count is not None
        values = result.direct_crosstalk_count[:, 0, 0].to(torch.float64)
        _assert_statistic(
            self,
            float(values.mean()),
            2.0,
            math.sqrt(2.0 / sample_size),
        )
        _assert_statistic(
            self,
            float(values.var(correction=0)),
            2.0,
            math.sqrt((2.0 + 8.0) / sample_size),
        )

    def test_delayed_q32_destination_rate_and_boundary_moments(self) -> None:
        sample_size = 1 << 15
        roots = torch.zeros((sample_size, 1, 4), dtype=torch.int64)
        roots[:, 0, 0] = 4
        result = _simulate(
            roots,
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                delayed_crosstalk=_delayed(mean=0.5, delay=2.0),
            ),
        )
        assert result.delayed_crosstalk_count is not None
        destination = result.delayed_crosstalk_count[:, 0, 1].to(torch.float64)
        _assert_statistic(
            self,
            float(destination.mean()),
            2.0,
            math.sqrt(2.0 / sample_size),
        )
        self.assertEqual(
            int(result.delayed_crosstalk_count[:, 0, 0].sum()),
            0,
        )

    def test_afterpulse_occurrence_delay_recovery_and_boundary_statistics(
        self,
    ) -> None:
        sample_size = 1 << 16
        sample_period_ns = 2.0
        root_pattern = torch.tensor(
            (4, 3, 2, 1, 5, 2),
            dtype=torch.int64,
        )
        roots = root_pattern.reshape(1, 1, -1).expand(
            sample_size,
            1,
            -1,
        ).clone()
        afterpulse = AfterpulseConfig(
            probability=Probability(0.35),
            mean_delay=_ns(5.0),
            recovery=AfterpulseRecoveryConfig(
                time_constant=_ns(8.0),
            ),
        )
        maximum_generations = 3
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(maximum_generations),
            afterpulse=afterpulse,
        )
        result = _simulate(
            roots,
            config,
            rng=Threefry4x32(seed=7_321),
        )
        replay = _simulate(
            roots,
            config,
            rng=Threefry4x32(seed=7_321),
        )
        for field in result.__dataclass_fields__:
            first = getattr(result, field)
            second = getattr(replay, field)
            if first is None:
                self.assertIsNone(second)
            else:
                assert second is not None
                self.assertTrue(torch.equal(first, second), field)

        occurrence_probability = afterpulse.probability.value
        mean_delay_ns = float(afterpulse.mean_delay.magnitude)
        assert afterpulse.recovery is not None
        recovery_time_ns = float(
            afterpulse.recovery.time_constant.magnitude
        )
        delay_rate = 1.0 / mean_delay_ns
        recovery_rate = 1.0 / recovery_time_ns
        sample_count = root_pattern.numel()

        def phase_marginal_probability(
            rate: float,
            offset: int,
        ) -> float:
            def primitive(
                time_ns: float,
                *,
                constant: float,
                slope: float,
            ) -> float:
                return -math.exp(-rate * time_ns) * (
                    constant
                    + slope * time_ns
                    + slope / rate
                )

            if offset == 0:
                return primitive(
                    sample_period_ns,
                    constant=1.0,
                    slope=-1.0 / sample_period_ns,
                ) - primitive(
                    0.0,
                    constant=1.0,
                    slope=-1.0 / sample_period_ns,
                )
            left = (offset - 1) * sample_period_ns
            center = offset * sample_period_ns
            right = (offset + 1) * sample_period_ns
            rising = primitive(
                center,
                constant=-(offset - 1),
                slope=1.0 / sample_period_ns,
            ) - primitive(
                left,
                constant=-(offset - 1),
                slope=1.0 / sample_period_ns,
            )
            falling = primitive(
                right,
                constant=offset + 1,
                slope=-1.0 / sample_period_ns,
            ) - primitive(
                center,
                constant=offset + 1,
                slope=-1.0 / sample_period_ns,
            )
            return rising + falling

        delay_probabilities = tuple(
            phase_marginal_probability(delay_rate, offset)
            for offset in range(sample_count)
        )
        combined_rate = delay_rate + recovery_rate
        recovery_weights = tuple(
            1.0
            - (
                delay_rate
                / combined_rate
                * phase_marginal_probability(combined_rate, offset)
                / delay_probabilities[offset]
            )
            for offset in range(sample_count)
        )
        self.assertTrue(
            all(0.0 < probability < 1.0 for probability in delay_probabilities)
        )
        self.assertLess(math.fsum(delay_probabilities), 1.0)
        self.assertTrue(
            all(0.0 < weight < 1.0 for weight in recovery_weights)
        )

        count_matrix = torch.zeros(
            (sample_count, sample_count),
            dtype=torch.float64,
        )
        charge_matrix = torch.zeros_like(count_matrix)
        square_matrix = torch.zeros_like(count_matrix)
        for source in range(sample_count):
            for destination in range(source, sample_count):
                offset = destination - source
                probability = (
                    occurrence_probability * delay_probabilities[offset]
                )
                weight = recovery_weights[offset]
                count_matrix[source, destination] = probability
                charge_matrix[source, destination] = probability * weight
                square_matrix[source, destination] = (
                    probability * weight * weight
                )
        self.assertTrue(
            torch.equal(
                torch.tril(count_matrix, diagonal=-1),
                torch.zeros_like(count_matrix),
            )
        )
        discarded_probability = (
            occurrence_probability - count_matrix.sum(dim=1)
        )
        self.assertTrue(torch.all(discarded_probability > 0.0))
        self.assertGreater(
            float(discarded_probability[-1]),
            float(discarded_probability[0]),
        )

        expected_roots = root_pattern.to(torch.float64)
        expected_frontier = expected_roots
        expected_afterpulse_count = torch.zeros_like(expected_roots)
        expected_afterpulse_charge = torch.zeros_like(expected_roots)
        expected_afterpulse_square_sum = torch.zeros_like(expected_roots)
        for _ in range(maximum_generations):
            expected_afterpulse_count = (
                expected_afterpulse_count
                + expected_frontier @ count_matrix
            )
            expected_afterpulse_charge = (
                expected_afterpulse_charge
                + expected_frontier @ charge_matrix
            )
            expected_afterpulse_square_sum = (
                expected_afterpulse_square_sum
                + expected_frontier @ square_matrix
            )
            expected_frontier = expected_frontier @ count_matrix
        expected_total_count = expected_roots + expected_afterpulse_count
        expected_S1 = expected_roots + expected_afterpulse_charge
        expected_S2 = expected_roots + expected_afterpulse_square_sum

        assert result.afterpulse_count is not None
        assert result.afterpulse_charge is not None
        assert result.afterpulse_charge_square_sum is not None
        observed_and_expected = (
            (
                "afterpulse_count",
                result.afterpulse_count,
                expected_afterpulse_count,
            ),
            (
                "afterpulse_charge",
                result.afterpulse_charge,
                expected_afterpulse_charge,
            ),
            (
                "afterpulse_charge_square_sum",
                result.afterpulse_charge_square_sum,
                expected_afterpulse_square_sum,
            ),
            (
                "final_frontier",
                result.final_frontier,
                expected_frontier,
            ),
            (
                "total_count",
                result.total_count,
                expected_total_count,
            ),
            ("S1", result.S1, expected_S1),
            ("S2", result.S2, expected_S2),
        )
        for field, observed, expected in observed_and_expected:
            values = observed[:, 0].to(torch.float64)
            mean = values.mean(dim=0)
            empirical_error = (
                values.std(dim=0, correction=1) / math.sqrt(sample_size)
            )
            conservative_error = torch.sqrt(
                torch.clamp(expected.abs(), min=1.0) / sample_size
            )
            standard_error = torch.maximum(
                empirical_error,
                conservative_error,
            )
            for destination in range(sample_count):
                with self.subTest(field=field, destination=destination):
                    _assert_statistic(
                        self,
                        float(mean[destination]),
                        float(expected[destination]),
                        float(standard_error[destination]),
                    )

    def test_fixed_and_exponential_crosstalk_with_recovered_afterpulses(self) -> None:
        roots = torch.tensor([[[5, 0, 0, 0]]], dtype=torch.int64)
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(3),
            direct_crosstalk=_direct(mean=0.2, delay=0.0),
            delayed_crosstalk=DelayedCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.2),
                delay=ExponentialDelayConfig(mean_delay=_ns(4.0)),
            ),
            afterpulse=_afterpulse(probability=0.2, recovery=True),
        )
        first = _simulate(roots, config, rng=Threefry4x32(seed=99))
        repeated = _simulate(roots, config, rng=Threefry4x32(seed=99))
        self.assertTrue(torch.equal(first.S1, repeated.S1))
        self.assertTrue(torch.equal(first.S2, repeated.S2))
        self.assertTrue(torch.equal(first.total_count, repeated.total_count))

    def test_destination_rate_accepts_one_e8_without_total_source_restriction(
        self,
    ) -> None:
        roots = torch.tensor([[[100_000_000, 0]]], dtype=torch.int64)
        runtime = _runtime(
            roots,
            CorrelatedAvalancheConfig(
                maximum_generations=NonnegativeInteger(1),
                direct_crosstalk=_direct(mean=1.0),
            ),
        )
        assert runtime.direct_crosstalk is not None
        assert runtime.direct_mean is not None
        assert runtime.direct_retained_rng_key is not None
        with self.assertRaises(AssertionError):
            _draw_crosstalk(
                roots,
                elements=RngElements.from_shape(tuple(roots.shape), device="cpu"),
                runtime=runtime.direct_crosstalk,
                mean=runtime.direct_mean,
                retained_key=runtime.direct_retained_rng_key,
                generation_index=0,
                maximum_generations=1,
                rng=_FailingRng(seed=0),
                field="direct crosstalk",
            )
        with self.assertRaisesRegex(RuntimeError, "Poisson"):
            _draw_crosstalk(
                roots + torch.tensor([[[1, 0]]], dtype=torch.int64),
                elements=RngElements.from_shape(tuple(roots.shape), device="cpu"),
                runtime=runtime.direct_crosstalk,
                mean=runtime.direct_mean,
                retained_key=runtime.direct_retained_rng_key,
                generation_index=0,
                maximum_generations=1,
                rng=_FailingRng(seed=0),
                field="direct crosstalk",
            )


if __name__ == "__main__":
    unittest.main()
