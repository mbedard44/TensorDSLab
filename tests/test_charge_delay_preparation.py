from __future__ import annotations

from decimal import Decimal, localcontext
import math
import unittest

from tensor_core import PositiveFloat, PositiveInteger, Probability

from tensor_dslab import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    SamplingConfig,
)
from tensor_dslab.readout.charge.effects._delays import (
    _prepare_afterpulse_recovery,
    _prepare_exponential_delay,
    _prepare_fixed_delay,
)


def _sampling(*, period_ps: int = 2000, count: int = 8) -> SamplingConfig:
    return SamplingConfig(
        sample_period_ps=PositiveInteger(period_ps),
        sample_count=PositiveInteger(count),
    )


def _decimal_exponential_law(
    inverse_mean: Decimal,
    *,
    sample_count: int,
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    one = Decimal(1)
    survival_step = (-inverse_mean).exp()
    exponential_mass = one - survival_step
    phase_factor = exponential_mass / inverse_mean
    tail_values = [one, phase_factor]
    for _ in range(2, sample_count + 1):
        tail_values.append(tail_values[-1] * survival_step)
    tails = tuple(tail_values)
    probabilities = (one - phase_factor,) + tuple(
        tails[offset] * exponential_mass
        for offset in range(1, sample_count)
    )
    return probabilities, tails


class FixedDelayPreparationTest(unittest.TestCase):
    def test_left_edge_boundaries_and_exact_two_point_mass(self) -> None:
        sampling = _sampling(count=4)
        zero = _prepare_fixed_delay(0.0, sampling=sampling)
        self.assertEqual(zero.probabilities, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(zero.right_tails, (1.0, 0.0, 0.0, 0.0, 0.0))

        exact = _prepare_fixed_delay(2.0, sampling=sampling)
        self.assertEqual(exact.probabilities, (0.0, 1.0, 0.0, 0.0))
        below = _prepare_fixed_delay(math.nextafter(2.0, 0.0), sampling=sampling)
        above = _prepare_fixed_delay(math.nextafter(2.0, math.inf), sampling=sampling)
        self.assertGreater(below.probabilities[0], 0.0)
        self.assertGreater(below.probabilities[1], 0.0)
        self.assertGreater(above.probabilities[1], 0.0)
        self.assertGreater(above.probabilities[2], 0.0)
        self.assertEqual(math.fsum(below.probabilities), 1.0)
        self.assertEqual(math.fsum(above.probabilities), 1.0)

        quarter = _prepare_fixed_delay(0.5, sampling=sampling)
        self.assertEqual(quarter.probabilities, (0.75, 0.25, 0.0, 0.0))
        self.assertEqual(quarter.right_tails, (1.0, 0.25, 0.0, 0.0, 0.0))

        two_and_quarter = _prepare_fixed_delay(4.5, sampling=sampling)
        self.assertEqual(
            two_and_quarter.probabilities,
            (0.0, 0.0, 0.75, 0.25),
        )
        self.assertEqual(
            two_and_quarter.right_tails,
            (1.0, 1.0, 1.0, 0.25, 0.0),
        )
        self.assertEqual(two_and_quarter.right_tails[4], 0.0)
        self.assertEqual(two_and_quarter.right_tails[3], 0.25)
        self.assertEqual(two_and_quarter.right_tails[2], 1.0)

        with self.assertRaisesRegex(ValueError, "became deterministic"):
            _prepare_fixed_delay(math.ulp(0.0), sampling=sampling)

    def test_all_overflow_is_analytic(self) -> None:
        sampling = _sampling(count=4)
        below = _prepare_fixed_delay(
            math.nextafter(8.0, 0.0),
            sampling=sampling,
        )
        self.assertGreater(below.probabilities[-1], 0.0)
        self.assertLess(below.right_tails[-1], 1.0)
        for delay_ns in (8.0, math.nextafter(8.0, math.inf), 1.0e308):
            with self.subTest(delay_ns=delay_ns):
                plan = _prepare_fixed_delay(delay_ns, sampling=sampling)
                self.assertEqual(plan.probabilities, (0.0, 0.0, 0.0, 0.0))
                self.assertEqual(
                    plan.right_tails,
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                )


class ExponentialDelayPreparationTest(unittest.TestCase):
    def _assert_high_precision_law(
        self,
        *,
        ratio: float,
        sample_count: int,
    ) -> None:
        sampling = _sampling(count=sample_count)
        mean_ns = sampling.sample_period_ps.value * 1.0e-3 * ratio
        plan = _prepare_exponential_delay(mean_ns, sampling=sampling)
        with localcontext() as context:
            context.prec = 110
            inverse_mean = Decimal(sampling.sample_period_ps.value) / (
                Decimal.from_float(mean_ns) * Decimal(1000)
            )
            expected_probability, expected_tail = _decimal_exponential_law(
                inverse_mean,
                sample_count=sample_count,
            )

        complete_error = Decimal(0)
        for actual, expected in zip(plan.probabilities, expected_probability):
            error = abs(Decimal.from_float(actual) - expected)
            complete_error += error
            self.assertLessEqual(error, Decimal("1e-12"))
        for actual, expected in zip(plan.right_tails, expected_tail):
            error = abs(Decimal.from_float(actual) - expected)
            complete_error += error
            self.assertLessEqual(error, Decimal("1e-12"))
        self.assertLessEqual(complete_error, Decimal("1e-11"))
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(
                    plan.right_tails,
                    plan.right_tails[1:],
                )
            )
        )
        self.assertLessEqual(
            abs(plan.probabilities[0] + plan.right_tails[1] - 1.0),
            1.0e-12,
        )
        for offset in range(1, sample_count):
            self.assertLessEqual(
                abs(
                    plan.probabilities[offset]
                    + plan.right_tails[offset + 1]
                    - plan.right_tails[offset]
                ),
                1.0e-12,
            )
        self.assertLessEqual(
            abs(math.fsum((*plan.probabilities, plan.right_tails[-1])) - 1.0),
            1.0e-11,
        )

    def test_complete_declared_ratio_and_sample_count_matrix(self) -> None:
        ratios = (
            2.0**-52,
            2.0**-40,
            1.0e-6,
            0.1,
            0.5,
            1.0,
            2.0,
            16.0,
            2.0**40,
            2.0**52,
        )
        for ratio in ratios:
            for sample_count in (2, 3, 8, 64, 512, 8192):
                with self.subTest(ratio=ratio, sample_count=sample_count):
                    self._assert_high_precision_law(
                        ratio=ratio,
                        sample_count=sample_count,
                    )

    def test_central_mass_branch_sides_and_natural_underflow(self) -> None:
        for ratio in (
            math.nextafter(2.0, 0.0),
            2.0,
            math.nextafter(2.0, math.inf),
        ):
            with self.subTest(ratio=ratio):
                self._assert_high_precision_law(ratio=ratio, sample_count=8)

        underflow = _prepare_exponential_delay(
            2.0 * 2.0**-52,
            sampling=_sampling(count=8),
        )
        self.assertGreater(underflow.right_tails[1], 0.0)
        self.assertGreater(underflow.probabilities[1], 0.0)
        self.assertTrue(all(value == 0.0 for value in underflow.right_tails[2:]))
        self.assertTrue(all(value == 0.0 for value in underflow.probabilities[2:]))

    def test_ratio_and_sample_count_domains(self) -> None:
        sampling = _sampling()
        period_ns = sampling.sample_period_ps.value * 1.0e-3
        for ratio in (2.0**-52, 1.0, 2.0**52):
            with self.subTest(ratio=ratio):
                plan = _prepare_exponential_delay(
                    period_ns * ratio,
                    sampling=sampling,
                )
                self.assertEqual(len(plan.probabilities), 8)
        with self.assertRaises(ValueError):
            _prepare_exponential_delay(
                period_ns * math.nextafter(2.0**-52, 0.0),
                sampling=sampling,
            )
        with self.assertRaises(ValueError):
            _prepare_exponential_delay(
                period_ns * math.nextafter(2.0**52, math.inf),
                sampling=sampling,
            )
        with self.assertRaises(ValueError):
            _prepare_exponential_delay(10.0, sampling=_sampling(count=8193))


class AfterpulseRecoveryPreparationTest(unittest.TestCase):
    def test_matches_110_digit_law_in_all_difference_branches(self) -> None:
        sampling = _sampling(count=8)
        branch_cases = (
            ("series", 10.0, 20.0),
            ("midpoint", 2.0, 2_097_152.0),
            ("general", 2.0, 4.0),
            ("lower-endpoint", 2.0**53, 2.0**53),
            ("upper-endpoint", 2.0**-51, 2.0**-51),
        )
        local_tolerance = Decimal("1e-12")
        complete_tolerance = Decimal("1e-11")

        for branch, mean_delay_ns, recovery_ns in branch_cases:
            with self.subTest(branch=branch):
                recovery_config = AfterpulseRecoveryConfig(
                    time_constant_ns=PositiveFloat(recovery_ns)
                )
                afterpulse = AfterpulseConfig(
                    probability=Probability(0.4),
                    mean_delay_ns=PositiveFloat(mean_delay_ns),
                    recovery=recovery_config,
                )
                delay = _prepare_exponential_delay(
                    mean_delay_ns,
                    sampling=sampling,
                )
                recovery, overflow = _prepare_afterpulse_recovery(
                    afterpulse,
                    recovery_config,
                    sampling=sampling,
                    delay=delay,
                )

                period = float(sampling.sample_period_ps.value)
                x_binary64 = period / (mean_delay_ns * 1000.0)
                y_binary64 = period / (recovery_ns * 1000.0)
                if branch == "series":
                    self.assertLessEqual(x_binary64 + y_binary64, 0.5)
                elif branch == "midpoint":
                    self.assertGreater(x_binary64 + y_binary64, 0.5)
                    self.assertLessEqual(
                        y_binary64,
                        2.0**-16 * max(1.0, x_binary64),
                    )
                elif branch == "general":
                    self.assertGreater(x_binary64 + y_binary64, 0.5)
                    self.assertGreater(
                        y_binary64,
                        2.0**-16 * max(1.0, x_binary64),
                    )
                elif branch == "lower-endpoint":
                    self.assertEqual(x_binary64 + y_binary64, 2.0**-51)
                else:
                    self.assertEqual(branch, "upper-endpoint")
                    self.assertEqual(x_binary64 + y_binary64, 2.0**53)

                with localcontext() as context:
                    context.prec = 110
                    period_ps = Decimal(sampling.sample_period_ps.value)
                    x = period_ps / (
                        Decimal.from_float(mean_delay_ns) * Decimal(1000)
                    )
                    y = period_ps / (
                        Decimal.from_float(recovery_ns) * Decimal(1000)
                    )
                    combined = x + y
                    scaling = x / combined
                    q_x, r_x = _decimal_exponential_law(
                        x,
                        sample_count=sampling.sample_count.value,
                    )
                    q_combined, r_combined = _decimal_exponential_law(
                        combined,
                        sample_count=sampling.sample_count.value,
                    )

                    complete_errors: list[Decimal] = []
                    recovered_masses: list[Decimal] = []
                    for offset, actual_weight in enumerate(recovery):
                        if q_x[offset] == 0:
                            self.assertEqual(actual_weight, 0.0)
                            self.assertEqual(delay.probabilities[offset], 0.0)
                            self.assertEqual(q_combined[offset], 0)
                            continue
                        expected_mass = (
                            q_x[offset] - scaling * q_combined[offset]
                        )
                        expected_weight = expected_mass / q_x[offset]
                        represented_probability = Decimal.from_float(
                            delay.probabilities[offset]
                        )
                        represented_weight = Decimal.from_float(actual_weight)
                        represented_mass = (
                            represented_probability * represented_weight
                        )
                        self.assertLessEqual(
                            abs(represented_weight - expected_weight),
                            local_tolerance,
                        )
                        self.assertLessEqual(
                            abs(represented_mass - expected_mass),
                            local_tolerance,
                        )
                        self.assertLessEqual(
                            abs(
                                represented_mass
                                + scaling * q_combined[offset]
                                - q_x[offset]
                            ),
                            local_tolerance,
                        )
                        complete_errors.append(
                            abs(represented_mass - expected_mass)
                        )
                        recovered_masses.append(represented_mass)

                    for first_outside in range(
                        1,
                        sampling.sample_count.value + 1,
                    ):
                        if r_x[first_outside] == 0:
                            self.assertEqual(overflow[first_outside], 0.0)
                            self.assertEqual(
                                delay.right_tails[first_outside],
                                0.0,
                            )
                            self.assertEqual(r_combined[first_outside], 0)
                            continue
                        expected_tail_mass = (
                            r_x[first_outside]
                            - scaling * r_combined[first_outside]
                        )
                        expected_tail_weight = (
                            expected_tail_mass / r_x[first_outside]
                        )
                        represented_tail = Decimal.from_float(
                            delay.right_tails[first_outside]
                        )
                        represented_weight = Decimal.from_float(
                            overflow[first_outside]
                        )
                        represented_tail_mass = (
                            represented_tail * represented_weight
                        )
                        self.assertLessEqual(
                            abs(represented_weight - expected_tail_weight),
                            local_tolerance,
                        )
                        self.assertLessEqual(
                            abs(represented_tail_mass - expected_tail_mass),
                            local_tolerance,
                        )
                        self.assertLessEqual(
                            abs(
                                represented_tail_mass
                                + scaling * r_combined[first_outside]
                                - r_x[first_outside]
                            ),
                            local_tolerance,
                        )

                    final_tail_mass = (
                        Decimal.from_float(delay.right_tails[-1])
                        * Decimal.from_float(overflow[-1])
                    )
                    expected_final_tail_mass = (
                        r_x[-1] - scaling * r_combined[-1]
                    )
                    complete_errors.append(
                        abs(final_tail_mass - expected_final_tail_mass)
                    )
                    self.assertLessEqual(
                        sum(complete_errors, Decimal(0)),
                        complete_tolerance,
                    )
                    self.assertLessEqual(
                        abs(
                            sum(recovered_masses, Decimal(0))
                            + final_tail_mass
                            - (Decimal(1) - scaling)
                        ),
                        complete_tolerance,
                    )

    def test_recovery_is_bounded_delay_conditioned_and_tail_aware(self) -> None:
        sampling = _sampling(count=8)
        recovery_config = AfterpulseRecoveryConfig(
            time_constant_ns=PositiveFloat(20.0)
        )
        afterpulse = AfterpulseConfig(
            probability=Probability(0.4),
            mean_delay_ns=PositiveFloat(10.0),
            recovery=recovery_config,
        )
        delay = _prepare_exponential_delay(10.0, sampling=sampling)
        recovery, overflow = _prepare_afterpulse_recovery(
            afterpulse,
            recovery_config,
            sampling=sampling,
            delay=delay,
        )
        self.assertEqual(len(recovery), sampling.sample_count.value)
        self.assertEqual(len(overflow), sampling.sample_count.value + 1)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in recovery))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in overflow))
        self.assertTrue(all(later >= earlier for earlier, later in zip(recovery, recovery[1:])))
        self.assertTrue(all(later >= earlier for earlier, later in zip(overflow[1:], overflow[2:])))
        self.assertNotAlmostEqual(recovery[1], 1.0 - math.exp(-2.0 / 20.0))


if __name__ == "__main__":
    unittest.main()
