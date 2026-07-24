from __future__ import annotations

import math
from typing import ClassVar
import unittest

import torch
from tensor_core import CounterRng, RngKey, RngPositions, Threefry4x32
from tensor_core.validation import require_tensor_allocation
from tensor_core.validation.random import require_count_tensor

from tensor_dslab.readout.charge.runtime.effects.counts import (
    MAX_COUNT,
    checked_add,
    checked_subtract,
    draw_ordered_categories,
    original_positions,
)


_KEY = RngKey(namespace=0x54445331, stream=0x0000_0009)
_STATISTICAL_SEEDS = (
    0,
    1,
    0x0123456789ABCDEF,
    0xFFFFFFFFFFFFFFFF,
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


def _statistical_delta(*, scale: float, length: int) -> float:
    return (
        64.0
        * torch.finfo(torch.float64).eps
        * max(1, math.ceil(math.log2(length)))
        * abs(scale)
    )


def _multinomial_cross_fourth_moment(
    count: int,
    first_probability: float,
    second_probability: float,
) -> float:
    n = float(count)
    p = first_probability
    q = second_probability
    falling_2 = n * (n - 1.0)
    falling_3 = falling_2 * (n - 2.0)
    falling_4 = falling_3 * (n - 3.0)
    mean_x = n * p
    mean_y = n * q
    raw_xy = falling_2 * p * q
    raw_x2 = falling_2 * p * p + mean_x
    raw_y2 = falling_2 * q * q + mean_y
    raw_x2y = falling_3 * p * p * q + raw_xy
    raw_xy2 = falling_3 * p * q * q + raw_xy
    raw_x2y2 = (
        falling_4 * p * p * q * q
        + falling_3 * (p * p * q + p * q * q)
        + raw_xy
    )
    return (
        raw_x2y2
        - 2.0 * mean_y * raw_x2y
        + mean_y * mean_y * raw_x2
        - 2.0 * mean_x * raw_xy2
        + 4.0 * mean_x * mean_y * raw_xy
        - 2.0 * mean_x * mean_y * mean_y * mean_x
        + mean_x * mean_x * raw_y2
        - 2.0 * mean_x * mean_x * mean_y * mean_y
        + mean_x * mean_x * mean_y * mean_y
    )


def _draw_q32(
    *,
    seed: int,
    examples: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    total = torch.full((examples,), 32, dtype=torch.int64, device=device)
    basis = RngPositions.from_shape(tuple(total.shape), device=total.device)
    masses = ((0.10, 0.90), (0.15, 0.75), (0.20, 0.55))
    categories = draw_ordered_categories(
        total,
        success_masses=tuple(
            torch.full(total.shape, current, dtype=torch.float64, device=device)
            for current, _ in masses
        ),
        failure_masses=tuple(
            torch.full(total.shape, later, dtype=torch.float64, device=device)
            for _, later in masses
        ),
        positions=tuple(
            basis.offset(category * total.numel())
            for category in range(len(masses))
        ),
        rng=Threefry4x32(seed=seed),
        key=_KEY,
        field="Q32 multinomial",
    )
    return torch.stack(categories, dim=1)


class ChargeCountDomainTest(unittest.TestCase):
    def test_count_domain_andchecked_add_subtract_boundaries(self) -> None:
        accepted = torch.tensor((0, MAX_COUNT), dtype=torch.int64)
        require_count_tensor(accepted, "accepted")
        with self.assertRaises(TypeError):
            require_count_tensor(accepted.to(torch.int32), "dtype")
        with self.assertRaises(ValueError):
            require_count_tensor(torch.tensor((-1,), dtype=torch.int64), "low")
        with self.assertRaises(ValueError):
            require_count_tensor(
                torch.tensor((MAX_COUNT + 1,), dtype=torch.int64),
                "high",
            )

        left = torch.tensor((0, MAX_COUNT - 1), dtype=torch.int64)
        right = torch.ones(2, dtype=torch.int64)
        self.assertTrue(
            torch.equal(
                checked_add(left, right, field="endpoint"),
                torch.tensor((1, MAX_COUNT), dtype=torch.int64),
            )
        )
        with self.assertRaises(RuntimeError):
            checked_add(
                torch.tensor((MAX_COUNT,), dtype=torch.int64),
                torch.ones(1, dtype=torch.int64),
                field="overflow",
            )
        self.assertTrue(
            torch.equal(
                checked_subtract(right, right, field="remainder"),
                torch.zeros_like(right),
            )
        )
        with self.assertRaises(RuntimeError):
            checked_subtract(
                torch.zeros(1, dtype=torch.int64),
                torch.ones(1, dtype=torch.int64),
                field="underflow",
            )

    def test_allocation_and_original_position_boundaries(self) -> None:
        self.assertEqual(
            require_tensor_allocation(
                (2, 3, 4),
                "small",
                element_size=8,
                upper=1 << 63,
            ),
            24,
        )
        with self.assertRaises(ValueError):
            require_tensor_allocation(
                (1 << 62,),
                "bytes",
                element_size=8,
                upper=1 << 63,
            )
        with self.assertRaises(ValueError):
            require_tensor_allocation(
                (1 << 62, 2),
                "elements",
                element_size=1,
                upper=1 << 63,
            )

        _RecordingRng.calls = []
        actual = original_positions(
            (2, 3, 4),
            sample_dimension=1,
            device=torch.device("cpu"),
        )
        self.assertIs(type(actual), RngPositions)
        self.assertEqual(actual.shape, (2, 4, 3))
        _RecordingRng(seed=0).uniform(
            key=_KEY,
            positions=actual,
            dtype=torch.float64,
            quantum=0,
            ordinal=0,
            count=1,
        )
        self.assertTrue(
            torch.equal(
                _RecordingRng.calls[0][1],
                torch.arange(24, dtype=torch.int64)
                .reshape(2, 3, 4)
                .movedim(1, -1),
            )
        )


class OrderedMultinomialOrchestrationTest(unittest.TestCase):
    def test_zero_one_and_no_count_categories_are_draw_free(self) -> None:
        positions = (RngPositions.from_shape((8,), device="cpu"),)
        failing = _FailingRng(seed=0)
        zeros = torch.zeros(8, dtype=torch.int64)
        zero_category, zero_remainder = draw_ordered_categories(
            zeros,
            success_masses=(0.25,),
            failure_masses=(0.75,),
            positions=positions,
            rng=failing,
            key=_KEY,
            field="no counts",
        )
        self.assertTrue(torch.equal(zero_category, zeros))
        self.assertTrue(torch.equal(zero_remainder, zeros))

        counts = torch.full((8,), 32, dtype=torch.int64)
        absent, unchanged = draw_ordered_categories(
            counts,
            success_masses=(0.0,),
            failure_masses=(1.0,),
            positions=positions,
            rng=failing,
            key=_KEY,
            field="zero mass",
        )
        self.assertTrue(torch.equal(absent, torch.zeros_like(counts)))
        self.assertTrue(torch.equal(unchanged, counts))
        all_success, exhausted = draw_ordered_categories(
            counts,
            success_masses=(1.0,),
            failure_masses=(0.0,),
            positions=positions,
            rng=failing,
            key=_KEY,
            field="unit mass",
        )
        self.assertTrue(torch.equal(all_success, counts))
        self.assertTrue(torch.equal(exhausted, torch.zeros_like(counts)))

    def test_category_order_key_positions_and_final_remainder(self) -> None:
        _RecordingRng.calls = []
        counts = torch.full((5,), 5, dtype=torch.int64)
        basis = torch.tensor((11, 13, 17, 19, 23), dtype=torch.int64)
        positions = tuple(
            RngPositions.from_tensor(basis).offset(category * 100)
            for category in range(3)
        )
        categories = draw_ordered_categories(
            counts,
            success_masses=(0.2, 0.3, 0.1),
            failure_masses=(0.8, 0.5, 0.4),
            positions=positions,
            rng=_RecordingRng(seed=7),
            key=_KEY,
            field="recorded multinomial",
        )
        self.assertTrue(torch.equal(sum(categories[1:], categories[0]), counts))
        self.assertEqual(len(_RecordingRng.calls), 3)
        self.assertEqual(tuple(call[0] for call in _RecordingRng.calls), (_KEY,) * 3)
        self.assertEqual(tuple(call[2] for call in _RecordingRng.calls), (0, 0, 0))
        self.assertEqual(tuple(call[3] for call in _RecordingRng.calls), (0, 0, 0))
        for actual, expected in zip(
            (call[1] for call in _RecordingRng.calls),
            (basis + category * 100 for category in range(3)),
            strict=True,
        ):
            self.assertTrue(torch.equal(actual, expected))

    def test_q32_joint_ensemble_preserves_frozen_moments_and_covariances(
        self,
    ) -> None:
        examples_per_seed = 1 << 14
        ensemble = torch.cat(
            tuple(
                _draw_q32(seed=seed, examples=examples_per_seed)
                for seed in _STATISTICAL_SEEDS
            ),
            dim=0,
        ).to(torch.float64)
        total_examples = ensemble.shape[0]
        self.assertEqual(total_examples, 1 << 16)
        self.assertTrue(
            torch.equal(
                torch.sum(ensemble, dim=1),
                torch.full((total_examples,), 32.0, dtype=torch.float64),
            )
        )

        probabilities = (0.10, 0.15, 0.20, 0.55)
        for category, probability in enumerate(probabilities):
            values = ensemble[:, category]
            target_mean = 32.0 * probability
            target_variance = 32.0 * probability * (1.0 - probability)
            mean_standard_error = math.sqrt(target_variance / total_examples)
            mean_bound = 8.0 * mean_standard_error + _statistical_delta(
                scale=target_mean,
                length=total_examples,
            )
            self.assertLessEqual(
                abs(float(torch.mean(values)) - target_mean),
                mean_bound,
            )

            fourth_moment = (
                3.0 * target_variance * target_variance
                + target_variance
                * (1.0 - 6.0 * probability * (1.0 - probability))
            )
            variance_standard_error = math.sqrt(
                (fourth_moment - target_variance * target_variance)
                / total_examples
            )
            variance_bound = 8.0 * variance_standard_error + _statistical_delta(
                scale=target_variance,
                length=total_examples,
            )
            observed_variance = float(torch.mean((values - target_mean) ** 2))
            self.assertLessEqual(
                abs(observed_variance - target_variance),
                variance_bound,
            )

        for first in range(len(probabilities)):
            for second in range(first + 1, len(probabilities)):
                first_probability = probabilities[first]
                second_probability = probabilities[second]
                first_mean = 32.0 * first_probability
                second_mean = 32.0 * second_probability
                target_covariance = -32.0 * first_probability * second_probability
                observed_covariance = float(
                    torch.mean(
                        (ensemble[:, first] - first_mean)
                        * (ensemble[:, second] - second_mean)
                    )
                )
                cross_fourth = _multinomial_cross_fourth_moment(
                    32,
                    first_probability,
                    second_probability,
                )
                standard_error = math.sqrt(
                    (cross_fourth - target_covariance * target_covariance)
                    / total_examples
                )
                bound = 8.0 * standard_error + _statistical_delta(
                    scale=target_covariance,
                    length=total_examples,
                )
                self.assertLessEqual(
                    abs(observed_covariance - target_covariance),
                    bound,
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaOrderedMultinomialOrchestrationTest(unittest.TestCase):
    def test_q32_conservation_on_cuda(self) -> None:
        ensemble = _draw_q32(seed=0x0123456789ABCDEF, examples=4096, device="cuda")
        self.assertEqual(ensemble.device.type, "cuda")
        self.assertTrue(
            torch.equal(
                torch.sum(ensemble, dim=1),
                torch.full((4096,), 32, dtype=torch.int64, device="cuda"),
            )
        )


if __name__ == "__main__":
    unittest.main()
