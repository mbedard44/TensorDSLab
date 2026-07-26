from decimal import Decimal, localcontext
import math
import unittest

import torch
from tensor_core import ProbabilityKernel, Probability, TensorKernel

from tensor_dslab import AfterpulseConfig, AfterpulseRecoveryConfig, quantity
from tensor_dslab.common import SampleAxis
from tensor_dslab.readout.charge.runtime.effects.delays import (
    AfterpulseRecoveryKernel,
    DelayProbabilityKernel,
    DelayRuntime,
    _prepare_fixed_delay,
    prepare_afterpulse_recovery,
    prepare_exponential_delay,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


def _sampling(*, period_ps: int = 2000, count: int = 8) -> SamplingRuntime:
    return SamplingRuntime(
        sample_count=count,
        sample_period_ps=period_ps,
        sample_dimension=2,
    )


def _fixed(value: float, *, sampling: SamplingRuntime) -> DelayRuntime:
    return _prepare_fixed_delay(value, sampling=sampling, device=torch.device("cpu"))


def _exponential(value: float, *, sampling: SamplingRuntime) -> DelayRuntime:
    return prepare_exponential_delay(
        value,
        sampling=sampling,
        device=torch.device("cpu"),
    )


def _probabilities(runtime: DelayRuntime) -> tuple[float, ...]:
    return tuple(float(value) for value in runtime.kernel.tensor.tolist())


def _decimal_exponential_law(
    inverse_mean: Decimal,
    *,
    sample_count: int,
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    one = Decimal(1)
    survival_step = (-inverse_mean).exp()
    exponential_mass = one - survival_step
    phase_factor = exponential_mass / inverse_mean
    tails = [one, phase_factor]
    for _ in range(2, sample_count + 1):
        tails.append(tails[-1] * survival_step)
    probabilities = (one - phase_factor,) + tuple(
        tails[offset] * exponential_mass
        for offset in range(1, sample_count)
    )
    return probabilities, tuple(tails)


class FixedDelayPreparationTest(unittest.TestCase):
    def test_left_edge_boundaries_and_exact_two_point_mass(self) -> None:
        sampling = _sampling(count=4)
        expected = {
            0.0: (1.0, 0.0, 0.0, 0.0),
            0.5: (0.75, 0.25, 0.0, 0.0),
            2.0: (0.0, 1.0, 0.0, 0.0),
            4.5: (0.0, 0.0, 0.75, 0.25),
        }
        for delay, probabilities in expected.items():
            with self.subTest(delay=delay):
                runtime = _fixed(delay, sampling=sampling)
                self.assertIs(type(runtime.kernel), DelayProbabilityKernel)
                self.assertIsInstance(runtime.kernel, ProbabilityKernel)
                self.assertEqual(runtime.kernel.axis_types, (SampleAxis,))
                self.assertEqual(_probabilities(runtime), probabilities)
                self.assertEqual(runtime.kernel.tensor.dtype, torch.float64)
                self.assertEqual(runtime.kernel.tensor.device.type, "cpu")

        below = _fixed(math.nextafter(2.0, 0.0), sampling=sampling)
        above = _fixed(math.nextafter(2.0, math.inf), sampling=sampling)
        below_values = _probabilities(below)
        above_values = _probabilities(above)
        self.assertGreater(below_values[0], 0.0)
        self.assertGreater(below_values[1], 0.0)
        self.assertGreater(above_values[1], 0.0)
        self.assertGreater(above_values[2], 0.0)
        self.assertEqual(math.fsum(below_values), 1.0)
        self.assertEqual(math.fsum(above_values), 1.0)
        with self.assertRaisesRegex(ValueError, "became deterministic"):
            _fixed(math.ulp(0.0), sampling=sampling)

    def test_all_overflow_is_analytic(self) -> None:
        sampling = _sampling(count=4)
        below = _fixed(math.nextafter(8.0, 0.0), sampling=sampling)
        self.assertGreater(_probabilities(below)[-1], 0.0)
        self.assertLess(below.right_tails[-1], 1.0)
        for delay in (8.0, math.nextafter(8.0, math.inf), 1.0e308):
            with self.subTest(delay=delay):
                runtime = _fixed(delay, sampling=sampling)
                self.assertEqual(_probabilities(runtime), (0.0,) * 4)
                self.assertEqual(runtime.right_tails, (1.0,) * 5)


class ExponentialDelayPreparationTest(unittest.TestCase):
    def _assert_high_precision_law(
        self,
        *,
        ratio: float,
        sample_count: int,
    ) -> None:
        sampling = _sampling(count=sample_count)
        mean_ns = sampling.sample_period_ps * 1.0e-3 * ratio
        runtime = _exponential(mean_ns, sampling=sampling)
        actual_probabilities = _probabilities(runtime)
        with localcontext() as context:
            context.prec = 110
            inverse_mean = Decimal(sampling.sample_period_ps) / (
                Decimal.from_float(mean_ns) * Decimal(1000)
            )
            expected_probabilities, expected_tails = _decimal_exponential_law(
                inverse_mean,
                sample_count=sample_count,
            )
        complete_error = Decimal(0)
        for actual, expected in zip(actual_probabilities, expected_probabilities):
            error = abs(Decimal.from_float(actual) - expected)
            complete_error += error
            self.assertLessEqual(error, Decimal("1e-12"))
        for actual, expected in zip(runtime.right_tails, expected_tails):
            error = abs(Decimal.from_float(actual) - expected)
            complete_error += error
            self.assertLessEqual(error, Decimal("1e-12"))
        self.assertLessEqual(complete_error, Decimal("1e-11"))
        self.assertLessEqual(
            abs(math.fsum((*actual_probabilities, runtime.right_tails[-1])) - 1.0),
            1.0e-11,
        )

    def test_complete_declared_ratio_and_sample_count_matrix(self) -> None:
        for ratio in (
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
        ):
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
        runtime = _exponential(
            2.0 * 2.0**-52,
            sampling=_sampling(count=8),
        )
        probabilities = _probabilities(runtime)
        self.assertGreater(runtime.right_tails[1], 0.0)
        self.assertGreater(probabilities[1], 0.0)
        self.assertTrue(all(value == 0.0 for value in runtime.right_tails[2:]))
        self.assertTrue(all(value == 0.0 for value in probabilities[2:]))

    def test_ratio_and_sample_count_domains(self) -> None:
        sampling = _sampling()
        period_ns = sampling.sample_period_ps * 1.0e-3
        for ratio in (2.0**-52, 1.0, 2.0**52):
            self.assertEqual(
                _exponential(period_ns * ratio, sampling=sampling).kernel.shape,
                (8,),
            )
        with self.assertRaises(ValueError):
            _exponential(
                period_ns * math.nextafter(2.0**-52, 0.0),
                sampling=sampling,
            )
        with self.assertRaises(ValueError):
            _exponential(
                period_ns * math.nextafter(2.0**52, math.inf),
                sampling=sampling,
            )
        with self.assertRaises(ValueError):
            _exponential(10.0, sampling=_sampling(count=8193))


class AfterpulseRecoveryPreparationTest(unittest.TestCase):
    def _prepare(
        self,
        *,
        mean_delay: float,
        recovery_ns: float,
    ) -> tuple[DelayRuntime, tuple[float, ...]]:
        sampling = _sampling(count=8)
        delay = _exponential(mean_delay, sampling=sampling)
        recovery = prepare_afterpulse_recovery(
            mean_delay,
            recovery_ns,
            sampling=sampling,
            delay=delay,
            device=torch.device("cpu"),
        )
        return delay, recovery

    def test_matches_110_digit_law_in_all_difference_branches(self) -> None:
        for _, mean_delay, recovery_ns in (
            ("series", 10.0, 20.0),
            ("midpoint", 2.0, 2_097_152.0),
            ("general", 2.0, 4.0),
            ("lower-endpoint", 2.0**53, 2.0**53),
            ("upper-endpoint", 2.0**-51, 2.0**-51),
        ):
            with self.subTest(mean_delay=mean_delay, recovery_ns=recovery_ns):
                delay, recovery = self._prepare(
                    mean_delay=mean_delay,
                    recovery_ns=recovery_ns,
                )
                probabilities = _probabilities(delay)
                with localcontext() as context:
                    context.prec = 110
                    x = Decimal(2000) / (
                        Decimal.from_float(mean_delay) * Decimal(1000)
                    )
                    y = Decimal(2000) / (
                        Decimal.from_float(recovery_ns) * Decimal(1000)
                    )
                    scaling = x / (x + y)
                    q_x, _ = _decimal_exponential_law(x, sample_count=8)
                    q_combined, _ = _decimal_exponential_law(
                        x + y,
                        sample_count=8,
                    )
                for index, weight in enumerate(recovery):
                    self.assertGreaterEqual(weight, 0.0)
                    self.assertLessEqual(weight, 1.0)
                    if q_x[index] == 0:
                        self.assertEqual(weight, 0.0)
                    else:
                        expected = (q_x[index] - scaling * q_combined[index]) / q_x[
                            index
                        ]
                        self.assertLessEqual(
                            abs(Decimal.from_float(weight) - expected),
                            Decimal("1e-12"),
                        )
                    self.assertLessEqual(
                        abs(
                            Decimal.from_float(probabilities[index] * weight)
                            - (q_x[index] - scaling * q_combined[index])
                        ),
                        Decimal("1e-12"),
                    )

    def test_recovery_is_bounded_delay_conditioned_and_tail_aware(self) -> None:
        _, recovery = self._prepare(mean_delay=10.0, recovery_ns=20.0)
        kernel = AfterpulseRecoveryKernel(
            tensor=torch.tensor(recovery, dtype=torch.float64),
            axis_types=(SampleAxis,),
        )
        self.assertIsInstance(kernel, TensorKernel)
        self.assertEqual(kernel.shape, (8,))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in recovery))
        self.assertTrue(
            all(later >= earlier for earlier, later in zip(recovery, recovery[1:]))
        )
        self.assertNotAlmostEqual(recovery[1], 1.0 - math.exp(-2.0 / 20.0))


if __name__ == "__main__":
    unittest.main()
