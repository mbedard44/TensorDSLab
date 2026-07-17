from __future__ import annotations

from decimal import Decimal, localcontext
import itertools
import math
from typing import ClassVar
import unittest
from unittest.mock import patch

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
from tensor_dslab.readout.charge.effects import (
    _correlated_avalanches as correlated_effect,
)
from tensor_dslab.readout.charge.effects import _counts as count_effect
from tensor_dslab.readout.charge.effects._correlated_avalanches import (
    _simulate_correlated_avalanches,
)


_STATISTICAL_SEEDS = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)


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


def _assert_statistic(
    case: unittest.TestCase,
    *,
    name: str,
    observed: float,
    target: float,
    standard_error: float,
    accumulation_length: int = 1,
    scale: float | None = None,
) -> None:
    reference_scale = target if scale is None else scale
    delta = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(1, math.ceil(math.log2(accumulation_length)))
        * abs(reference_scale)
    )
    bound = 8.0 * standard_error + delta
    case.assertLessEqual(
        abs(observed - target),
        bound,
        (
            f"{name}: observed={observed:.17g}, target={target:.17g}, "
            f"SE={standard_error:.17g}, delta={delta:.17g}, "
            f"bound={bound:.17g}"
        ),
    )


def _poisson_probabilities(mean: float, maximum: int = 64) -> tuple[float, ...]:
    probabilities = [math.exp(-mean)]
    for value in range(1, maximum + 1):
        probabilities.append(probabilities[-1] * mean / value)
    return tuple(probabilities)


def _branching_target_moments(
    roots: int,
    offspring_mean: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    dict[tuple[int, int], float],
]:
    means = (
        roots * offspring_mean,
        roots * offspring_mean**2,
        roots * offspring_mean**3,
    )
    if offspring_mean == 1.0:
        variances = (float(roots), float(2 * roots), float(3 * roots))
    else:
        variances = tuple(
            roots
            * offspring_mean**generation
            * (offspring_mean**generation - 1.0)
            / (offspring_mean - 1.0)
            for generation in range(1, 4)
        )

    fourth_moments = [0.0, 0.0, 0.0]
    cross_fourth_moments = {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0}
    total_probability = 0.0
    first = _poisson_probabilities(roots * offspring_mean)
    for z1, probability1 in enumerate(first):
        second = _poisson_probabilities(z1 * offspring_mean)
        for z2, probability2 in enumerate(second):
            third = _poisson_probabilities(z2 * offspring_mean)
            prefix = probability1 * probability2
            for z3, probability3 in enumerate(third):
                probability = prefix * probability3
                total_probability += probability
                centered = (
                    z1 - means[0],
                    z2 - means[1],
                    z3 - means[2],
                )
                for index, value in enumerate(centered):
                    fourth_moments[index] += probability * value**4
                for pair in cross_fourth_moments:
                    left, right = pair
                    cross_fourth_moments[pair] += (
                        probability * centered[left] ** 2 * centered[right] ** 2
                    )
    if abs(total_probability - 1.0) > 1.0e-12:
        raise AssertionError(
            "independent branching oracle omitted material probability mass: "
            f"mass={total_probability:.17g}"
        )
    return (
        means,
        (variances[0], variances[1], variances[2]),
        (fourth_moments[0], fourth_moments[1], fourth_moments[2]),
        cross_fourth_moments,
    )


def _assert_branching_statistics(
    case: unittest.TestCase,
    *,
    name: str,
    generations: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    roots: int,
    offspring_mean: float,
) -> None:
    means, variances, fourth_moments, cross_fourth_moments = (
        _branching_target_moments(roots, offspring_mean)
    )
    sample_size = generations[0].numel()
    for index, values in enumerate(generations):
        observed_mean = float(torch.mean(values))
        observed_centered_second = float(
            torch.mean((values - means[index]) ** 2)
        )
        _assert_statistic(
            case,
            name=f"{name} generation {index + 1} mean",
            observed=observed_mean,
            target=means[index],
            standard_error=math.sqrt(variances[index] / sample_size),
            accumulation_length=sample_size,
        )
        variance_standard_error = math.sqrt(
            (fourth_moments[index] - variances[index] ** 2) / sample_size
        )
        _assert_statistic(
            case,
            name=f"{name} generation {index + 1} centered variance",
            observed=observed_centered_second,
            target=variances[index],
            standard_error=variance_standard_error,
            accumulation_length=sample_size,
        )

    for left, right in ((0, 1), (0, 2), (1, 2)):
        target = offspring_mean ** (right - left) * variances[left]
        observed = float(
            torch.mean(
                (generations[left] - means[left])
                * (generations[right] - means[right])
            )
        )
        standard_error = math.sqrt(
            (cross_fourth_moments[(left, right)] - target**2) / sample_size
        )
        _assert_statistic(
            case,
            name=f"{name} generations {left + 1},{right + 1} covariance",
            observed=observed,
            target=target,
            standard_error=standard_error,
            accumulation_length=sample_size,
        )


def _afterpulse_reference_law(
    inverse_delay_ratio: float,
    inverse_recovery_ratio: float,
    *,
    sample_count: int,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    with localcontext() as context:
        context.prec = 80
        x = Decimal(str(inverse_delay_ratio))
        y = Decimal(str(inverse_recovery_ratio))

        def exponential_law(
            inverse_ratio: Decimal,
        ) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
            interval_mass = Decimal(1) - (-inverse_ratio).exp()
            tail_scale = interval_mass / inverse_ratio
            probabilities = [Decimal(1) - tail_scale]
            right_tails = [Decimal(1)]
            for offset in range(1, sample_count + 1):
                tail = tail_scale * (-(offset - 1) * inverse_ratio).exp()
                right_tails.append(tail)
                if offset < sample_count:
                    probabilities.append(tail * interval_mass)
            return tuple(probabilities), tuple(right_tails)

        delay_probabilities, delay_tails = exponential_law(x)
        effective_probabilities, effective_tails = exponential_law(x + y)
        second_effective_probabilities, second_effective_tails = (
            exponential_law(x + 2 * y)
        )
        recovery_scale = x / (x + y)
        second_recovery_scale = x / (x + 2 * y)
        recovery = tuple(
            (delay - recovery_scale * effective) / delay
            for delay, effective in zip(
                delay_probabilities,
                effective_probabilities,
                strict=True,
            )
        )
        overflow_recovery = (Decimal(0),) + tuple(
            (delay_tails[first_outside] - recovery_scale * effective_tails[first_outside])
            / delay_tails[first_outside]
            for first_outside in range(1, sample_count + 1)
        )
        conditional_second_moments = tuple(
            (
                delay
                - 2 * recovery_scale * effective
                + second_recovery_scale * second_effective
            )
            / delay
            for delay, effective, second_effective in zip(
                delay_probabilities,
                effective_probabilities,
                second_effective_probabilities,
                strict=True,
            )
        )
        overflow_conditional_second = (Decimal(0),) + tuple(
            (
                delay_tails[first_outside]
                - 2 * recovery_scale * effective_tails[first_outside]
                + second_recovery_scale
                * second_effective_tails[first_outside]
            )
            / delay_tails[first_outside]
            for first_outside in range(1, sample_count + 1)
        )
        return (
            tuple(float(value) for value in delay_probabilities),
            tuple(float(value) for value in delay_tails),
            tuple(float(value) for value in recovery),
            tuple(float(value) for value in overflow_recovery),
            tuple(float(value) for value in conditional_second_moments),
            tuple(float(value) for value in overflow_conditional_second),
        )


def _category_sum_moments(
    probabilities: tuple[float, ...],
    values: tuple[float, ...],
    *,
    count: int,
) -> tuple[float, float, float]:
    one_mean = math.fsum(
        probability * value
        for probability, value in zip(probabilities, values, strict=True)
    )
    one_variance = math.fsum(
        probability * (value - one_mean) ** 2
        for probability, value in zip(probabilities, values, strict=True)
    )
    one_fourth = math.fsum(
        probability * (value - one_mean) ** 4
        for probability, value in zip(probabilities, values, strict=True)
    )
    variance = count * one_variance
    fourth = count * one_fourth + 3 * count * (count - 1) * one_variance**2
    return count * one_mean, variance, fourth


def _category_sum_cross_moment(
    probabilities: tuple[float, ...],
    left_values: tuple[float, ...],
    right_values: tuple[float, ...],
    *,
    count: int,
) -> tuple[float, float]:
    left_mean = math.fsum(
        probability * value
        for probability, value in zip(probabilities, left_values, strict=True)
    )
    right_mean = math.fsum(
        probability * value
        for probability, value in zip(probabilities, right_values, strict=True)
    )
    left_variance = math.fsum(
        probability * (value - left_mean) ** 2
        for probability, value in zip(probabilities, left_values, strict=True)
    )
    right_variance = math.fsum(
        probability * (value - right_mean) ** 2
        for probability, value in zip(probabilities, right_values, strict=True)
    )
    one_covariance = math.fsum(
        probability * (left - left_mean) * (right - right_mean)
        for probability, left, right in zip(
            probabilities,
            left_values,
            right_values,
            strict=True,
        )
    )
    one_cross_fourth = math.fsum(
        probability
        * (left - left_mean) ** 2
        * (right - right_mean) ** 2
        for probability, left, right in zip(
            probabilities,
            left_values,
            right_values,
            strict=True,
        )
    )
    covariance = count * one_covariance
    cross_fourth = (
        count * one_cross_fourth
        + count * (count - 1) * left_variance * right_variance
        + 2 * count * (count - 1) * one_covariance**2
    )
    return covariance, cross_fourth


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
    rng: CounterRng | None = None,
) -> correlated_effect._CorrelatedAvalancheResult:
    return _simulate_correlated_avalanches(
        roots,
        sample_dimension=roots.ndim - 1,
        sampling=_sampling(count=roots.shape[-1]),
        floating_dtype=dtype,
        config=config,
        rng=Threefry4x32(seed=1234) if rng is None else rng,
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
        result = _simulate(roots, configured, rng=_FailingRng(seed=0))
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
        result = _simulate(roots, ineffective, rng=_FailingRng(seed=0))
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
            correlated_effect,
            "_prepare_delay",
            side_effect=AssertionError("K=0 must not prepare crosstalk"),
        ), patch.object(
            correlated_effect,
            "_prepare_exponential_delay",
            side_effect=AssertionError("K=0 must not prepare afterpulsing"),
        ):
            result = _simulate_correlated_avalanches(
                roots,
                sample_dimension=2,
                sampling=_sampling(count=8193),
                floating_dtype=torch.float32,
                config=k_zero,
                rng=_FailingRng(seed=0),
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
            correlated_effect,
            "_prepare_delay",
            side_effect=AssertionError("zero crosstalk must not prepare delay"),
        ), patch.object(
            correlated_effect,
            "_prepare_exponential_delay",
            side_effect=AssertionError("zero afterpulsing must not prepare delay"),
        ), patch.object(
            correlated_effect,
            "_prepare_afterpulse_recovery",
            side_effect=AssertionError("zero afterpulsing must not prepare recovery"),
        ):
            result = _simulate_correlated_avalanches(
                small_roots,
                sample_dimension=2,
                sampling=_sampling(count=2),
                floating_dtype=torch.float64,
                config=zero_effect,
                rng=_FailingRng(seed=0),
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
                    rng=(
                        Threefry4x32(seed=1234)
                        if any((direct_on, delayed_on, afterpulse_on))
                        else _FailingRng(seed=0)
                    ),
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
        roots = torch.tensor((1, 0, 0, 0), dtype=torch.int64).reshape(1, 1, 4)
        config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            direct_crosstalk=DirectCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.5),
                delay=ExponentialDelayConfig(
                    mean_delay_ns=PositiveFloat(2.0)
                ),
            ),
            delayed_crosstalk=DelayedCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.5),
                delay=ExponentialDelayConfig(
                    mean_delay_ns=PositiveFloat(2.0)
                ),
            ),
            afterpulse=_afterpulse(probability=0.5),
        )
        _RecordingRng.calls = []
        result = _simulate_correlated_avalanches(
            roots,
            sample_dimension=2,
            sampling=_sampling(),
            floating_dtype=torch.float64,
            config=config,
            rng=_RecordingRng(seed=0),
        )
        streams: list[int] = []
        for key, _, quantum, block in _RecordingRng.calls:
            self.assertEqual(key.namespace, 0x54445331)
            self.assertEqual(quantum, 0)
            self.assertEqual(block, 0)
            if not streams or streams[-1] != key.stream:
                streams.append(key.stream)
        self.assertEqual(
            streams,
            [4, 5, 6, 7, 9],
        )
        assert result.direct_crosstalk_count is not None
        assert result.delayed_crosstalk_count is not None

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
        single = _simulate(
            roots,
            config,
            dtype=torch.float32,
            rng=Threefry4x32(seed=77),
        )
        double = _simulate(
            roots,
            config,
            dtype=torch.float64,
            rng=Threefry4x32(seed=77),
        )
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
        call_index = 0

        def schedule_to_last_sample(
            counts: torch.Tensor,
            **kwargs: object,
        ) -> tuple[torch.Tensor, ...]:
            nonlocal call_index
            success_masses = kwargs["success_masses"]
            assert type(success_masses) is tuple
            categories = [torch.zeros_like(counts) for _ in success_masses]
            destination_offset = 3 - call_index
            categories[destination_offset] = counts.clone()
            call_index += 1
            return (*categories, torch.zeros_like(counts))

        with patch.object(
            correlated_effect,
            "_draw_ordered_categories",
            side_effect=schedule_to_last_sample,
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
                def retain_first_category(
                    counts: torch.Tensor,
                    **kwargs: object,
                ) -> tuple[torch.Tensor, ...]:
                    success_masses = kwargs["success_masses"]
                    assert type(success_masses) is tuple
                    categories = [
                        torch.zeros_like(counts) for _ in success_masses
                    ]
                    categories[0] = counts.clone()
                    return (*categories, torch.zeros_like(counts))

                with patch.object(
                    correlated_effect,
                    "_draw_ordered_categories",
                    side_effect=retain_first_category,
                ):
                    result = _simulate(roots, config, dtype=dtype)

                plan = correlated_effect._prepare_correlated_plan(
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
        original = correlated_effect._checked_add

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

        with patch.object(correlated_effect, "_checked_add", side_effect=corrupt_total):
            with self.assertRaisesRegex(RuntimeError, "integer count identity"):
                _simulate(
                    roots,
                    CorrelatedAvalancheConfig(
                        maximum_generations=NonnegativeInteger(1),
                        direct_crosstalk=_direct(mean=0.1),
                    ),
                )


class CorrelatedAvalancheEnvelopeTest(unittest.TestCase):
    def test_proved_bound_uses_greatest_represented_ledger_at_most_real_bound(
        self,
    ) -> None:
        _, bound = correlated_effect._ledger_envelope(
            floating_dtype=torch.float32,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=4,
        )
        bound_d = torch.tensor(
            float.fromhex("0x1.0000000000000p+53"),
            dtype=torch.float32,
        )
        above = torch.nextafter(
            bound_d,
            torch.tensor(math.inf, dtype=torch.float32),
        )
        self.assertLessEqual(float(bound_d), bound)
        self.assertGreater(float(above), bound)
        accepted = correlated_effect._checked_ledger_add(
            bound_d.reshape(1),
            torch.zeros(1, dtype=torch.float32),
            bound=bound,
            field="test ledger",
        )
        self.assertTrue(torch.equal(accepted, bound_d.reshape(1)))
        with self.assertRaisesRegex(RuntimeError, "proved ledger bound"):
            correlated_effect._checked_ledger_add(
                above.reshape(1),
                torch.zeros(1, dtype=torch.float32),
                bound=bound,
                field="test ledger",
            )

    def test_exact_address_product_boundaries_without_lattice_materialization(
        self,
    ) -> None:
        crosstalk = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(2),
            direct_crosstalk=_direct(mean=0.1),
        )
        accepted = correlated_effect._prepare_correlated_plan(
            crosstalk,
            sampling=_sampling(count=2),
            floating_dtype=torch.float64,
            tensor_numel=1 << 62,
        )
        self.assertIsNotNone(accepted.direct_crosstalk)
        with self.assertRaisesRegex(ValueError, "crosstalk address lattice"):
            correlated_effect._prepare_correlated_plan(
                crosstalk,
                sampling=_sampling(count=2),
                floating_dtype=torch.float64,
                tensor_numel=(1 << 62) + 1,
            )

        afterpulse = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(2),
            afterpulse=_afterpulse(probability=0.25),
        )
        accepted = correlated_effect._prepare_correlated_plan(
            afterpulse,
            sampling=_sampling(count=3),
            floating_dtype=torch.float64,
            tensor_numel=1 << 60,
        )
        self.assertIsNotNone(accepted.afterpulse)
        with self.assertRaisesRegex(ValueError, "afterpulse address lattice"):
            correlated_effect._prepare_correlated_plan(
                afterpulse,
                sampling=_sampling(count=3),
                floating_dtype=torch.float64,
                tensor_numel=(1 << 60) + 1,
            )

    def test_allocation_byte_and_element_products_have_exact_boundaries(self) -> None:
        self.assertEqual(
            count_effect._require_tensor_allocation(
                ((1 << 63) - 1,),
                element_size=1,
                field="test",
            ),
            (1 << 63) - 1,
        )
        with self.assertRaises(ValueError):
            count_effect._require_tensor_allocation(
                (1 << 63,),
                element_size=1,
                field="test",
            )
        self.assertEqual(
            count_effect._require_tensor_allocation(
                ((1 << 60) - 1,),
                element_size=8,
                field="test",
            ),
            (1 << 60) - 1,
        )
        with self.assertRaises(ValueError):
            count_effect._require_tensor_allocation(
                (1 << 60,),
                element_size=8,
                field="test",
            )

    def test_ledger_depth_boundaries_match_the_independent_equation(self) -> None:
        for dtype, precision in ((torch.float32, 24), (torch.float64, 53)):
            with self.subTest(dtype=dtype, recovered=False):
                maximum_generations = (1 << precision) - 2
                depth, bound = correlated_effect._ledger_envelope(
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
                    correlated_effect._ledger_envelope(
                        floating_dtype=dtype,
                        maximum_generations=maximum_generations + 1,
                        retained_mechanisms=1,
                        recovered_afterpulse=False,
                        sample_count=2,
                    )

            with self.subTest(dtype=dtype, recovered=True):
                maximum_generations = (1 << precision) - 6
                depth, _ = correlated_effect._ledger_envelope(
                    floating_dtype=dtype,
                    maximum_generations=maximum_generations,
                    retained_mechanisms=1,
                    recovered_afterpulse=True,
                    sample_count=2,
                )
                self.assertEqual(depth, (1 << precision) - 1)
                with self.assertRaisesRegex(ValueError, "ledger depth"):
                    correlated_effect._ledger_envelope(
                        floating_dtype=dtype,
                        maximum_generations=maximum_generations + 1,
                        retained_mechanisms=1,
                        recovered_afterpulse=True,
                        sample_count=2,
                    )


class CorrelatedAvalancheStatisticalTest(unittest.TestCase):
    def test_direct_q32_three_generation_branching_moments_and_covariances(
        self,
    ) -> None:
        per_seed = 1 << 14
        mean = 0.2
        generation_observations: list[list[torch.Tensor]] = [[], [], []]
        for seed in _STATISTICAL_SEEDS:
            roots = torch.zeros((per_seed, 1, 2), dtype=torch.int64)
            roots[:, 0, 0] = 32
            previous_cumulative: torch.Tensor | None = None
            for generation in range(1, 4):
                result = _simulate(
                    roots,
                    CorrelatedAvalancheConfig(
                        maximum_generations=NonnegativeInteger(generation),
                        direct_crosstalk=_direct(mean=mean),
                    ),
                    rng=Threefry4x32(seed=seed),
                )
                assert result.direct_crosstalk_count is not None
                assert result.direct_crosstalk_overflow_count is not None
                frontier = result.final_frontier[:, 0, 0]
                cumulative = result.direct_crosstalk_count[:, 0, 0]
                self.assertEqual(int(result.final_frontier[:, :, 1].sum()), 0)
                self.assertEqual(int(result.direct_crosstalk_overflow_count.sum()), 0)
                self.assertTrue(
                    torch.equal(result.total_count[:, 0, 0], roots[:, 0, 0] + cumulative)
                )
                if previous_cumulative is not None:
                    self.assertTrue(
                        torch.equal(cumulative - frontier, previous_cumulative)
                    )
                previous_cumulative = cumulative
                generation_observations[generation - 1].append(
                    frontier.to(torch.float64)
                )

        concatenated = tuple(
            torch.cat(observations) for observations in generation_observations
        )
        generations = (concatenated[0], concatenated[1], concatenated[2])
        self.assertTrue(all(values.numel() == 1 << 16 for values in generations))
        _assert_branching_statistics(
            self,
            name="direct crosstalk",
            generations=generations,
            roots=32,
            offspring_mean=mean,
        )

    def test_delayed_q32_three_generation_positions_overflow_and_moments(
        self,
    ) -> None:
        per_seed = 1 << 14
        mean = 0.2
        generation_observations: list[list[torch.Tensor]] = [[], [], []]
        for seed in _STATISTICAL_SEEDS:
            roots = torch.zeros((per_seed, 1, 3), dtype=torch.int64)
            roots[:, 0, 0] = 32
            result = _simulate(
                roots,
                CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(3),
                    delayed_crosstalk=_delayed(mean=mean, delay_ns=2.0),
                ),
                rng=Threefry4x32(seed=seed),
            )
            assert result.delayed_crosstalk_count is not None
            assert result.delayed_crosstalk_overflow_count is not None
            retained = result.delayed_crosstalk_count
            overflow = result.delayed_crosstalk_overflow_count
            self.assertEqual(int(retained[:, :, 0].sum()), 0)
            self.assertEqual(int(overflow[:, :, :2].sum()), 0)
            self.assertEqual(int(result.final_frontier.sum()), 0)
            self.assertTrue(
                torch.equal(result.total_count, roots + retained)
            )
            generation_observations[0].append(retained[:, 0, 1].to(torch.float64))
            generation_observations[1].append(retained[:, 0, 2].to(torch.float64))
            generation_observations[2].append(overflow[:, 0, 2].to(torch.float64))

        concatenated = tuple(
            torch.cat(observations) for observations in generation_observations
        )
        generations = (concatenated[0], concatenated[1], concatenated[2])
        self.assertTrue(all(values.numel() == 1 << 16 for values in generations))
        _assert_branching_statistics(
            self,
            name="delayed crosstalk",
            generations=generations,
            roots=32,
            offspring_mean=mean,
        )

    def test_afterpulse_q32_retained_overflow_stop_and_recovery_statistics(
        self,
    ) -> None:
        per_seed = 1 << 14
        parent_count = 32
        probability = 0.25
        inverse_delay_ratio = 0.2
        inverse_recovery_ratio = 0.1
        (
            delay_probabilities,
            delay_tails,
            recovery,
            overflow_recovery_by_first_outside,
            conditional_second_moments,
            overflow_conditional_second_by_first_outside,
        ) = _afterpulse_reference_law(
            inverse_delay_ratio,
            inverse_recovery_ratio,
            sample_count=4,
        )
        overflow_recovery = overflow_recovery_by_first_outside[2]
        overflow_conditional_second = (
            overflow_conditional_second_by_first_outside[2]
        )
        outcome_probabilities = (
            probability * delay_probabilities[0],
            probability * delay_probabilities[1],
            probability * delay_tails[2],
            1.0 - probability,
        )
        self.assertAlmostEqual(math.fsum(outcome_probabilities), 1.0, places=15)

        omitted_within_category_variances = (
            conditional_second_moments[0] - recovery[0] ** 2,
            conditional_second_moments[1] - recovery[1] ** 2,
            overflow_conditional_second - overflow_recovery**2,
        )
        self.assertTrue(
            all(value > 0.0 for value in omitted_within_category_variances)
        )

        unit_afterpulse = AfterpulseConfig(
            probability=Probability(probability),
            mean_delay_ns=PositiveFloat(10.0),
        )
        recovered_afterpulse = AfterpulseConfig(
            probability=Probability(probability),
            mean_delay_ns=PositiveFloat(10.0),
            recovery=AfterpulseRecoveryConfig(
                time_constant_ns=PositiveFloat(20.0)
            ),
        )
        recovered_config = CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(1),
            afterpulse=recovered_afterpulse,
        )
        prepared = correlated_effect._prepare_correlated_plan(
            recovered_config,
            sampling=_sampling(),
            floating_dtype=torch.float64,
            tensor_numel=4,
        )
        assert prepared.afterpulse is not None
        assert prepared.afterpulse.recovery is not None
        assert prepared.afterpulse.overflow_recovery is not None
        local_differences = tuple(
            abs(observed - reference)
            for observed, reference in zip(
                prepared.afterpulse.delay.probabilities,
                delay_probabilities,
                strict=True,
            )
        ) + tuple(
            abs(observed - reference)
            for observed, reference in zip(
                prepared.afterpulse.delay.right_tails,
                delay_tails,
                strict=True,
            )
        ) + tuple(
            abs(observed - reference)
            for observed, reference in zip(
                prepared.afterpulse.recovery,
                recovery,
                strict=True,
            )
        ) + tuple(
            abs(observed - reference)
            for observed, reference in zip(
                prepared.afterpulse.overflow_recovery,
                overflow_recovery_by_first_outside,
                strict=True,
            )
        )
        self.assertLessEqual(max(local_differences), 1.0e-12)
        self.assertLessEqual(math.fsum(local_differences), 1.0e-11)

        observations: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "offset 0 count",
                "offset 1 count",
                "retained count",
                "overflow count",
                "stop count",
                "retained charge",
                "retained charge-square sum",
                "overflow charge",
            )
        }
        for seed in _STATISTICAL_SEEDS:
            roots = torch.zeros((per_seed, 1, 4), dtype=torch.int64)
            roots[:, 0, 2] = parent_count
            unit = _simulate(
                roots,
                CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(1),
                    afterpulse=unit_afterpulse,
                ),
                rng=Threefry4x32(seed=seed),
            )
            recovered = _simulate(
                roots,
                recovered_config,
                rng=Threefry4x32(seed=seed),
            )
            for field in (
                "final_frontier",
                "total_count",
                "afterpulse_count",
                "afterpulse_overflow_count",
            ):
                self.assertTrue(
                    torch.equal(getattr(unit, field), getattr(recovered, field)),
                    field,
                )
            assert recovered.afterpulse_count is not None
            assert recovered.afterpulse_overflow_count is not None
            assert recovered.afterpulse_charge is not None
            assert recovered.afterpulse_overflow_charge is not None
            assert recovered.afterpulse_charge_square_sum is not None
            assert unit.afterpulse_charge is not None
            assert unit.afterpulse_charge_square_sum is not None

            retained_count = recovered.afterpulse_count
            overflow_count = recovered.afterpulse_overflow_count
            self.assertEqual(int(retained_count[:, :, :2].sum()), 0)
            self.assertEqual(int(overflow_count[:, :, :2].sum()), 0)
            self.assertEqual(int(overflow_count[:, :, 3].sum()), 0)
            self.assertTrue(torch.equal(recovered.final_frontier, retained_count))
            self.assertTrue(torch.equal(recovered.total_count, roots + retained_count))
            self.assertTrue(
                torch.equal(unit.afterpulse_charge, retained_count.to(torch.float64))
            )
            self.assertTrue(
                torch.equal(
                    unit.afterpulse_charge_square_sum,
                    retained_count.to(torch.float64),
                )
            )

            offset_zero = retained_count[:, 0, 2]
            offset_one = retained_count[:, 0, 3]
            overflow = overflow_count[:, 0, 2]
            stopped = parent_count - offset_zero - offset_one - overflow
            self.assertTrue(bool(torch.all(stopped >= 0).item()))
            self.assertTrue(
                torch.equal(
                    offset_zero + offset_one + overflow + stopped,
                    torch.full_like(stopped, parent_count),
                )
            )

            recovered_charge = recovered.afterpulse_charge
            recovered_square_sum = recovered.afterpulse_charge_square_sum
            recovered_overflow_charge = recovered.afterpulse_overflow_charge
            ledger_depth = 4
            gamma = ledger_depth / ((1 << 53) - ledger_depth)
            eta = float(
                torch.nextafter(
                    torch.tensor(0.0, dtype=torch.float64),
                    torch.tensor(1.0, dtype=torch.float64),
                )
            )
            ledger_cases = (
                (
                    "offset 0 charge",
                    recovered_charge[:, 0, 2],
                    offset_zero.to(torch.float64) * recovery[0],
                    offset_zero,
                ),
                (
                    "offset 1 charge",
                    recovered_charge[:, 0, 3],
                    offset_one.to(torch.float64) * recovery[1],
                    offset_one,
                ),
                (
                    "offset 0 charge-square sum",
                    recovered_square_sum[:, 0, 2],
                    offset_zero.to(torch.float64) * (recovery[0] ** 2),
                    offset_zero,
                ),
                (
                    "offset 1 charge-square sum",
                    recovered_square_sum[:, 0, 3],
                    offset_one.to(torch.float64) * (recovery[1] ** 2),
                    offset_one,
                ),
                (
                    "overflow charge",
                    recovered_overflow_charge[:, 0, 2],
                    overflow.to(torch.float64) * overflow_recovery,
                    overflow,
                ),
            )
            for name, observed, reference, counts in ledger_cases:
                tolerance = counts.to(torch.float64) * gamma + ledger_depth * eta
                maximum_error = float(torch.max(torch.abs(observed - reference)))
                maximum_tolerance = float(torch.max(tolerance))
                self.assertTrue(
                    bool(torch.all(torch.abs(observed - reference) <= tolerance).item()),
                    (
                        f"{name}: maximum_error={maximum_error:.17g}, "
                        f"maximum_tolerance={maximum_tolerance:.17g}"
                    ),
                )

            retained = offset_zero + offset_one
            charge = recovered_charge[:, 0, 2] + recovered_charge[:, 0, 3]
            square_sum = (
                recovered_square_sum[:, 0, 2]
                + recovered_square_sum[:, 0, 3]
            )
            observations["offset 0 count"].append(offset_zero.to(torch.float64))
            observations["offset 1 count"].append(offset_one.to(torch.float64))
            observations["retained count"].append(retained.to(torch.float64))
            observations["overflow count"].append(overflow.to(torch.float64))
            observations["stop count"].append(stopped.to(torch.float64))
            observations["retained charge"].append(charge)
            observations["retained charge-square sum"].append(square_sum)
            observations["overflow charge"].append(
                recovered_overflow_charge[:, 0, 2]
            )

        values = {name: torch.cat(parts) for name, parts in observations.items()}
        self.assertTrue(all(value.numel() == 1 << 16 for value in values.values()))
        outcome_values = {
            "offset 0 count": (1.0, 0.0, 0.0, 0.0),
            "offset 1 count": (0.0, 1.0, 0.0, 0.0),
            "retained count": (1.0, 1.0, 0.0, 0.0),
            "overflow count": (0.0, 0.0, 1.0, 0.0),
            "stop count": (0.0, 0.0, 0.0, 1.0),
            "retained charge": (recovery[0], recovery[1], 0.0, 0.0),
            "retained charge-square sum": (
                recovery[0] ** 2,
                recovery[1] ** 2,
                0.0,
                0.0,
            ),
            "overflow charge": (0.0, 0.0, overflow_recovery, 0.0),
        }
        sample_size = 1 << 16
        targets: dict[str, float] = {}
        for name, category_values in outcome_values.items():
            target, variance, fourth = _category_sum_moments(
                outcome_probabilities,
                category_values,
                count=parent_count,
            )
            targets[name] = target
            observed_mean = float(torch.mean(values[name]))
            observed_centered_second = float(
                torch.mean((values[name] - target) ** 2)
            )
            _assert_statistic(
                self,
                name=f"afterpulse {name} mean",
                observed=observed_mean,
                target=target,
                standard_error=math.sqrt(variance / sample_size),
                accumulation_length=sample_size,
            )
            _assert_statistic(
                self,
                name=f"afterpulse {name} centered variance",
                observed=observed_centered_second,
                target=variance,
                standard_error=math.sqrt((fourth - variance**2) / sample_size),
                accumulation_length=sample_size,
            )

        for left, right in (
            ("offset 0 count", "offset 1 count"),
            ("retained count", "retained charge"),
            ("retained count", "overflow count"),
            ("retained charge", "overflow charge"),
        ):
            target, cross_fourth = _category_sum_cross_moment(
                outcome_probabilities,
                outcome_values[left],
                outcome_values[right],
                count=parent_count,
            )
            observed = float(
                torch.mean(
                    (values[left] - targets[left])
                    * (values[right] - targets[right])
                )
            )
            _assert_statistic(
                self,
                name=f"afterpulse {left}/{right} covariance",
                observed=observed,
                target=target,
                standard_error=math.sqrt(
                    (cross_fourth - target**2) / sample_size
                ),
                accumulation_length=sample_size,
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

        single = _simulate(
            roots,
            config,
            dtype=torch.float32,
            rng=Threefry4x32(seed=0x12345678),
        )
        repeated = _simulate(
            roots,
            config,
            dtype=torch.float32,
            rng=Threefry4x32(seed=0x12345678),
        )
        double = _simulate(
            roots,
            config,
            dtype=torch.float64,
            rng=Threefry4x32(seed=0x12345678),
        )
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
