import math
import unittest
from typing import override
from unittest.mock import patch

import torch
from tensor_core import (
    CounterRng,
    MultinomialDistribution,
    NonnegativeFloat,
    PositiveInteger,
    RngElements,
    RngKey,
    Threefry4x32,
)

from tensor_dslab import TimingJitterConfig, quantity
from tensor_dslab.readout.charge.runtime.effects import (
    timing_jitter,
)
from tensor_dslab.readout.charge.runtime.effects.timing_jitter import (
    TimingJitterRuntime,
    prepare_timing_jitter as _prepare_timing_jitter,
    simulate_timing_jitter as simulate_timing_jitter_prepared,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


class _PositionRecordingRng(CounterRng):
    __slots__ = ()

    calls: list[torch.Tensor] = []

    @override
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        del key, quantum, block
        type(self).calls.append(positions.clone())
        return torch.zeros(
            positions.shape + (4,),
            dtype=torch.int64,
            device=positions.device,
        )


def _ns(value: int | float):
    return quantity(value, "ns")


def _hz(value: int | float):
    return quantity(value, "Hz")


def _mv(value: int | float):
    return quantity(value, "mV")


def _density(value: int | float):
    return quantity(value, "mV ** 2 / Hz")


def _sampling(*, count: int = 4) -> SamplingRuntime:
    return SamplingRuntime(
        sample_period_ps=2000,
        sample_count=count,
        sample_dimension=2,
    )


def prepare_timing_jitter(
    config: TimingJitterConfig,
    *,
    sampling: SamplingRuntime,
    tensor_numel: int,
) -> TimingJitterRuntime:
    runtime = _prepare_timing_jitter(
        config,
        sampling=sampling,
        tensor_numel=tensor_numel,
        device=torch.device("cpu"),
    )
    if runtime is None:
        raise AssertionError("active timing jitter must prepare a runtime")
    return runtime


def simulate_timing_jitter(
    counts: torch.Tensor,
    *,
    sample_dimension: int,
    sampling: SamplingRuntime,
    config: TimingJitterConfig,
    rng: CounterRng,
) -> torch.Tensor:
    if config.sigma.magnitude == 0.0:
        return counts
    plan = prepare_timing_jitter(
        config,
        sampling=sampling,
        tensor_numel=counts.numel(),
    )
    if plan is None:
        raise AssertionError("active timing jitter must prepare a runtime")
    return simulate_timing_jitter_prepared(
        counts,
        sample_dimension=sample_dimension,
        runtime=plan,
        rng=rng,
        elements=RngElements.from_shape(
            tuple(counts.shape),
            device=counts.device,
        ),
    )


def _positive_probabilities(runtime: TimingJitterRuntime) -> tuple[float, ...]:
    sample_count = (runtime.kernel.shape[0] + 1) // 2
    values = runtime.kernel.tensor.tolist()
    return tuple(float(value) for value in values[sample_count - 1 :])


def _direct_probability(offset: int, ratio: float) -> float:
    def phi(value: float) -> float:
        return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)

    def cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def h(value: float) -> float:
        return value * cdf(value) + phi(value)

    def x_cdf(value: float) -> float:
        return ratio * (h(value / ratio) - h((value - 1.0) / ratio))

    return x_cdf(float(offset + 1)) - x_cdf(float(offset))


def _assert_statistic(
    case: unittest.TestCase,
    *,
    name: str,
    observed: float,
    target: float,
    standard_error: float,
    accumulation_length: int,
) -> None:
    delta = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(1, math.ceil(math.log2(accumulation_length)))
        * abs(target)
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


class TimingJitterPreparationTest(unittest.TestCase):
    def test_symmetric_complete_law_matches_independent_equation(self) -> None:
        sampling = _sampling(count=8)
        config = TimingJitterConfig(sigma=_ns(1.0))
        plan = prepare_timing_jitter(
            config,
            sampling=sampling,
            tensor_numel=16,
        )
        ratio = 0.5
        probabilities = _positive_probabilities(plan)
        for offset, actual in enumerate(probabilities):
            expected = _direct_probability(offset, ratio)
            self.assertLessEqual(abs(actual - expected), 1.0e-12)
        self.assertLessEqual(
            abs(
                math.fsum(
                    (
                        probabilities[0],
                        *(2.0 * value for value in probabilities[1:]),
                        plan.completion_probability,
                    )
                )
                - 1.0
            ),
            1.0e-11,
        )

    def test_ratio_sample_and_address_domains(self) -> None:
        sampling = _sampling()
        period_ns = 2.0
        for ratio in (2.0**-52, 0.5, 64.0):
            with self.subTest(ratio=ratio):
                plan = prepare_timing_jitter(
                    TimingJitterConfig(
                        sigma=_ns(period_ns * ratio)
                    ),
                    sampling=sampling,
                    tensor_numel=16,
                )
                self.assertEqual(plan.kernel.shape, (7,))
        with self.assertRaises(ValueError):
            prepare_timing_jitter(
                TimingJitterConfig(
                    sigma=_ns(
                        period_ns * math.nextafter(2.0**-52, 0.0)
                    )
                ),
                sampling=sampling,
                tensor_numel=16,
            )
        with self.assertRaises(ValueError):
            prepare_timing_jitter(
                TimingJitterConfig(
                    sigma=_ns(
                        period_ns * math.nextafter(64.0, math.inf)
                    )
                ),
                sampling=sampling,
                tensor_numel=16,
            )
        with self.assertRaises(ValueError):
            prepare_timing_jitter(
                TimingJitterConfig(sigma=_ns(1.0)),
                sampling=_sampling(count=8193),
                tensor_numel=8193,
            )

        maximum_sampling = _sampling(count=8192)
        for ratio in (2.0**-52, 64.0):
            with self.subTest(maximum_sample_ratio=ratio):
                plan = prepare_timing_jitter(
                    TimingJitterConfig(
                        sigma=_ns(period_ns * ratio)
                    ),
                    sampling=maximum_sampling,
                    tensor_numel=1,
                )
                self.assertEqual(plan.kernel.shape, (16383,))

        exact_capacity = (1 << 63) // 15
        exact_address = prepare_timing_jitter(
            TimingJitterConfig(sigma=_ns(1.0)),
            sampling=_sampling(count=8),
            tensor_numel=exact_capacity,
        )
        self.assertEqual(exact_address.kernel.shape, (15,))
        with self.assertRaisesRegex(ValueError, "address lattice"):
            prepare_timing_jitter(
                TimingJitterConfig(sigma=_ns(1.0)),
                sampling=_sampling(count=8),
                tensor_numel=exact_capacity + 1,
            )

    def test_endpoint_fixtures_and_log_tail_split_match_frozen_oracles(self) -> None:
        fixtures = (
            (
                2.0**-52,
                (
                    0.9999999999999998228340379207456410082483788961918465,
                    8.8582981039627179495875810551904076745235146176357e-17,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    8.8582981039627179495875810551904076745235146176357e-17,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            ),
            (
                64.0,
                (
                    0.0062333463140330019042464216655415788495652174345428,
                    0.0062325854848973917446062117016167532988127170227628,
                    0.0062303035546364434070046442200417827681507995426254,
                    0.0062265021940078734367639245964154655690166021556618,
                    0.0062211841853426047350627339390288964392389086760097,
                    0.0062143534191509225533259520500356784361788269419818,
                    0.0062060148893792097196563207700064534119242376381550,
                    0.0061961746873253565423553322423747882444020218698756,
                ),
                (
                    0.4968833268429834990478767891672292105752173912827286,
                    0.4906507413580861073032705774656124572764046742599658,
                    0.4844204378034496638962659332455706745082538747173404,
                    0.4781939356094417904595020086491552089392372725616786,
                    0.4719727514240991857244392747101263124999983638856689,
                    0.4657583980049482631711133226600906340638195369436872,
                    0.4595523831155690534514570018900841806518952993055322,
                    0.4533562084282436969091016696477093924074932774356566,
                ),
            ),
        )
        for ratio, expected_probabilities, expected_tails in fixtures:
            with self.subTest(ratio=ratio):
                plan = prepare_timing_jitter(
                    TimingJitterConfig(
                        sigma=_ns(2.0 * ratio)
                    ),
                    sampling=_sampling(count=8),
                    tensor_numel=16,
                )
                complete_error = 0.0
                probabilities = _positive_probabilities(plan)
                for actual, expected in zip(probabilities, expected_probabilities):
                    error = abs(actual - expected)
                    complete_error += error
                    self.assertLessEqual(error, 1.0e-12)
                self.assertLessEqual(
                    abs(plan.completion_probability / 2.0 - expected_tails[-1]),
                    1.0e-12,
                )
                self.assertLessEqual(complete_error, 1.0e-11)

        expected_logs = (
            (8.0, -37.122364261692632999988189714292255861667532593724),
            (64.0, -2057.23743649695208852708542818071566313308576519734),
        )
        for value, expected in expected_logs:
            self.assertLessEqual(
                abs(timing_jitter._log_jitter_g(value) - expected),
                1.0e-12,
            )
        around_split = tuple(
            timing_jitter._log_jitter_g(value)
            for value in (
                math.nextafter(8.0, 0.0),
                8.0,
                math.nextafter(8.0, math.inf),
            )
        )
        self.assertLessEqual(max(around_split) - min(around_split), 1.0e-12)


class TimingJitterSimulationTest(unittest.TestCase):
    def test_zero_identity_and_all_source_counts_are_conserved_or_dropped(self) -> None:
        sampling = _sampling()
        counts = torch.tensor((3, 5, 7, 11), dtype=torch.int64).reshape(1, 1, 4)
        identity = simulate_timing_jitter(
            counts,
            sample_dimension=2,
            sampling=sampling,
            config=TimingJitterConfig(sigma=_ns(0.0)),
            rng=Threefry4x32(seed=0),
        )
        self.assertIs(identity, counts)
        jittered = simulate_timing_jitter(
            counts,
            sample_dimension=2,
            sampling=sampling,
            config=TimingJitterConfig(sigma=_ns(1.0)),
            rng=Threefry4x32(seed=1234),
        )
        self.assertTrue(bool(torch.all(jittered >= 0).item()))
        self.assertLessEqual(int(jittered.sum()), int(counts.sum()))
        self.assertTrue(torch.equal(counts, torch.tensor((3, 5, 7, 11), dtype=torch.int64).reshape(1, 1, 4)))

    def test_category_calls_use_kernel_addresses_and_original_elements(self) -> None:
        sampling = _sampling()
        counts = torch.ones((1, 1, 4), dtype=torch.int64)
        _PositionRecordingRng.calls = []
        result = simulate_timing_jitter(
            counts,
            sample_dimension=2,
            sampling=sampling,
            config=TimingJitterConfig(sigma=_ns(1.0)),
            rng=_PositionRecordingRng(seed=9),
        )
        self.assertEqual(result.shape, counts.shape)
        calls = _PositionRecordingRng.calls
        flattened = [int(call.item()) for call in calls]
        self.assertEqual(len(flattened), 16)
        self.assertEqual(
            flattened,
            [
                source + 4 * category
                for source in range(4)
                for category in range(4)
            ],
        )
        self.assertEqual(len(set(flattened)), len(flattened))

    def test_one_parent_ensemble_matches_prepared_categories(self) -> None:
        sampling = _sampling()
        config = TimingJitterConfig(sigma=_ns(1.0))
        per_seed = 1 << 16
        seeds = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
        retained = []
        for seed in seeds:
            counts = torch.zeros((per_seed, 1, 4), dtype=torch.int64)
            counts[:, 0, 1] = 1
            retained.append(
                simulate_timing_jitter(
                    counts,
                    sample_dimension=2,
                    sampling=sampling,
                    config=config,
                    rng=Threefry4x32(seed=seed),
                )[:, 0, :]
            )
        values = torch.cat(retained, dim=0).to(torch.float64)
        expected = (
            _direct_probability(1, 0.5),
            _direct_probability(0, 0.5),
            _direct_probability(1, 0.5),
            _direct_probability(2, 0.5),
        )
        total = values.shape[0]
        self.assertEqual(total, 1 << 18)
        for target, probability in enumerate(expected):
            target_values = values[:, target]
            observed = float(torch.mean(target_values))
            standard_error = math.sqrt(
                probability * (1.0 - probability) / total
            )
            _assert_statistic(
                self,
                name=f"timing-jitter target {target} mean",
                observed=observed,
                target=probability,
                standard_error=standard_error,
                accumulation_length=total,
            )

            target_variance = probability * (1.0 - probability)
            observed_variance = float(
                torch.mean((target_values - probability) ** 2)
            )
            fourth = (
                probability * (1.0 - probability) ** 4
                + (1.0 - probability) * probability**4
            )
            variance_standard_error = math.sqrt(
                max(0.0, fourth - target_variance**2) / total
            )
            _assert_statistic(
                self,
                name=f"timing-jitter target {target} centered variance",
                observed=observed_variance,
                target=target_variance,
                standard_error=variance_standard_error,
                accumulation_length=total,
            )

        for first in range(len(expected)):
            for second in range(first + 1, len(expected)):
                first_probability = expected[first]
                second_probability = expected[second]
                centered_product = (
                    (values[:, first] - first_probability)
                    * (values[:, second] - second_probability)
                )
                observed_covariance = float(torch.mean(centered_product))
                target_covariance = -first_probability * second_probability
                second_moment = (
                    first_probability
                    * (1.0 - first_probability) ** 2
                    * second_probability**2
                    + second_probability
                    * first_probability**2
                    * (1.0 - second_probability) ** 2
                    + (1.0 - first_probability - second_probability)
                    * first_probability**2
                    * second_probability**2
                )
                covariance_standard_error = math.sqrt(
                    max(0.0, second_moment - target_covariance**2) / total
                )
                _assert_statistic(
                    self,
                    name=f"timing-jitter targets {first},{second} covariance",
                    observed=observed_covariance,
                    target=target_covariance,
                    standard_error=covariance_standard_error,
                    accumulation_length=total,
                )

        retained_count = torch.sum(values, dim=1)
        observed_drop = float(torch.mean(1.0 - retained_count))
        expected_drop = 1.0 - math.fsum(expected)
        drop_standard_error = math.sqrt(
            expected_drop * (1.0 - expected_drop) / total
        )
        _assert_statistic(
            self,
            name="timing-jitter drop probability",
            observed=observed_drop,
            target=expected_drop,
            standard_error=drop_standard_error,
            accumulation_length=total,
        )

        offsets = torch.tensor((-1.0, 0.0, 1.0, 2.0), dtype=torch.float64)
        displacement = values @ offsets
        expected_displacement = math.fsum(
            offset * probability
            for offset, probability in zip((-1.0, 0.0, 1.0, 2.0), expected)
        )
        expected_displacement_square = math.fsum(
            offset * offset * probability
            for offset, probability in zip((-1.0, 0.0, 1.0, 2.0), expected)
        )
        displacement_variance = (
            expected_displacement_square - expected_displacement**2
        )
        observed_displacement = float(torch.mean(displacement))
        observed_displacement_variance = float(
            torch.mean((displacement - expected_displacement) ** 2)
        )
        _assert_statistic(
            self,
            name="timing-jitter displacement mean",
            observed=observed_displacement,
            target=expected_displacement,
            standard_error=math.sqrt(displacement_variance / total),
            accumulation_length=total,
        )
        fourth_displacement = math.fsum(
            probability * (offset - expected_displacement) ** 4
            for offset, probability in zip((-1.0, 0.0, 1.0, 2.0), expected)
        ) + expected_drop * expected_displacement**4
        displacement_variance_standard_error = math.sqrt(
            max(0.0, fourth_displacement - displacement_variance**2) / total
        )
        _assert_statistic(
            self,
            name="timing-jitter displacement centered variance",
            observed=observed_displacement_variance,
            target=displacement_variance,
            standard_error=displacement_variance_standard_error,
            accumulation_length=total,
        )


if __name__ == "__main__":
    unittest.main()
