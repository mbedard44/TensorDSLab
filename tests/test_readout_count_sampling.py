from __future__ import annotations

import math
import unittest
from collections.abc import Callable, Sequence
from decimal import Decimal, localcontext
from unittest.mock import patch

import torch

from tensor_dslab.readout._random import (
    _RngStream,
    _random_block,
    _sample_conditional_binomial,
    _sample_poisson,
)


_WordBlock = tuple[int, int, int, int]
_MIDPOINT_WORDS = (0x80000000, 0x00000000)
_MINIMUM_WORDS = (0x00000000, 0x00000000)
_MAXIMUM_WORDS = (0xFFFFFFFF, 0xFFFFFFFF)
_QUICK_ACCEPT_BLOCK = (*_MIDPOINT_WORDS, *_MINIMUM_WORDS)
_FULL_ACCEPT_BLOCK = (*_MIDPOINT_WORDS, *_MIDPOINT_WORDS)
_FULL_REJECT_BLOCK = (
    *_MIDPOINT_WORDS,
    0xFD70A3D7,
    0x0A3D6000,
)
_PTRS_UNCERTAIN_BLOCK = (
    *_MIDPOINT_WORDS,
    0xE6FD9A35,
    0xD24D6000,
)
_BTRS_UNCERTAIN_BLOCK = (
    *_MIDPOINT_WORDS,
    0xE7884283,
    0x06389000,
)
_BTRS_CANCELLATION_BLOCK = (
    *_MIDPOINT_WORDS,
    0xF3333333,
    0x33332000,
)
_STIRLING_CORRECTIONS = (
    0.0810614667953272,
    0.0413406959554092,
    0.0276779256849983,
    0.02079067210376509,
    0.0166446911898211,
    0.0138761288230707,
    0.0118967099458917,
    0.0104112652619720,
    0.00925546218271273,
    0.00833056343336287,
)

_STATISTICAL_SEEDS = (
    0,
    1,
    0x0123456789ABCDEF,
    0xFFFFFFFFFFFFFFFF,
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
    """Return E[(X-E[X])**2 * (Y-E[Y])**2] for two categories."""

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


def _uniform_closed_open_oracle(words: tuple[int, int]) -> float:
    mantissa = words[0] * (1 << 21) + (words[1] >> 11)
    return float(mantissa) * 2.0**-53


def _uniform_open_open_oracle(words: tuple[int, int]) -> float:
    mantissa = words[0] * (1 << 20) + (words[1] >> 12)
    return (0.5 + float(mantissa)) * 2.0**-52


def _poisson_inversion_oracle(mean: float, block: _WordBlock) -> int:
    uniform = _uniform_closed_open_oracle(block[:2])
    probability = math.exp(-mean)
    cumulative = probability
    for count in range(64):
        if uniform < cumulative:
            return count
        if count == 63:
            raise RuntimeError("oracle Poisson inversion exhausted")
        probability = probability * mean / float(count + 1)
        cumulative = cumulative + probability
    raise AssertionError("unreachable Poisson inversion state")


def _binomial_inversion_oracle(
    n: int,
    probability: float,
    block: _WordBlock,
) -> int:
    uniform = _uniform_closed_open_oracle(block[:2])
    q = 1.0 - probability
    term = math.exp(float(n) * math.log1p(-probability))
    cumulative = term
    for count in range(min(n, 63) + 1):
        if uniform < cumulative:
            return count
        if count == min(n, 63):
            raise RuntimeError("oracle Binomial inversion exhausted")
        term = term * (
            (float(n - count) / float(count + 1))
            * (probability / q)
        )
        cumulative = cumulative + term
    raise AssertionError("unreachable Binomial inversion state")


def _stirling_correction_oracle(value: int) -> float:
    if value < 10:
        return _STIRLING_CORRECTIONS[value]
    x = float(value + 1)
    x2 = x * x
    inner = (1.0 / 360.0) - ((1.0 / 1260.0) / x2)
    return ((1.0 / 12.0) - (inner / x2)) / x


def _ptrs_oracle(mean: float, blocks: Sequence[_WordBlock]) -> tuple[int, int]:
    sqrt_mean = math.sqrt(mean)
    log_mean = math.log(mean)
    b = 0.931 + 2.53 * sqrt_mean
    a = -0.059 + 0.02483 * b
    inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
    rectangle = 0.9277 - 3.6224 / (b - 2.0)
    for attempt, block in enumerate(blocks):
        uniform = _uniform_open_open_oracle(block[:2])
        variate = _uniform_open_open_oracle(block[2:])
        u = uniform - 0.5
        u_s = 0.5 - abs(u)
        proposal_float = math.floor(
            (2.0 * a / u_s + b) * u + mean + 0.43
        )
        if proposal_float < 0:
            continue
        proposal = int(proposal_float)
        if u_s >= 0.07 and variate <= rectangle:
            return proposal, attempt
        if u_s < 0.013 and variate > u_s:
            continue
        left = (
            math.log(variate)
            + math.log(inverse_alpha)
            - math.log(a / (u_s * u_s) + b)
        )
        right = (
            -mean
            + float(proposal) * log_mean
            - math.lgamma(float(proposal) + 1.0)
        )
        if left <= right:
            return proposal, attempt
    raise RuntimeError("oracle PTRS exhausted")


def _ptrs_float_sides(
    mean: float,
    block: _WordBlock,
    proposal: int,
) -> tuple[float, float]:
    sqrt_mean = math.sqrt(mean)
    b = 0.931 + 2.53 * sqrt_mean
    a = -0.059 + 0.02483 * b
    inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
    uniform = _uniform_open_open_oracle(block[:2])
    variate = _uniform_open_open_oracle(block[2:])
    u_s = 0.5 - abs(uniform - 0.5)
    left = (
        math.log(variate)
        + math.log(inverse_alpha)
        - math.log(a / (u_s * u_s) + b)
    )
    right = (
        -mean
        + float(proposal) * math.log(mean)
        - math.lgamma(float(proposal) + 1.0)
    )
    return left, right


def _btrs_parameters(n: int, probability: float) -> tuple[float, ...]:
    n_float = float(n)
    s = math.sqrt(n_float * probability * (1.0 - probability))
    b = 1.15 + 2.53 * s
    a = -0.0873 + 0.0248 * b + 0.01 * probability
    c = n_float * probability + 0.5
    rectangle = 0.92 - 4.2 / b
    ratio = probability / (1.0 - probability)
    alpha = (2.83 + 5.1 / b) * s
    mode = math.floor((n_float + 1.0) * probability)
    return n_float, s, b, a, c, rectangle, ratio, alpha, float(mode)


def _btrs_upper_oracle(
    n: int,
    probability: float,
    proposal: int,
) -> float:
    n_float, _, _, _, _, _, ratio, _, mode_float = _btrs_parameters(
        n,
        probability,
    )
    mode = int(mode_float)
    displacement = proposal - mode
    displacement_float = float(displacement)
    proposal_float = float(proposal)
    log_left = math.log1p(
        displacement_float / (n_float - proposal_float + 1.0)
    )
    log_right = math.log1p(
        -displacement_float / (proposal_float + 1.0)
    )
    log_ratio = math.log(
        ratio
        * (n_float - proposal_float + 1.0)
        / (proposal_float + 1.0)
    )
    main = (
        (n_float - float(mode) + 0.5) * log_left
        + (float(mode) + 0.5) * log_right
    ) + displacement_float * log_ratio
    correction = (
        (
            _stirling_correction_oracle(mode)
            + _stirling_correction_oracle(n - mode)
        )
        - _stirling_correction_oracle(proposal)
    ) - _stirling_correction_oracle(n - proposal)
    return main + correction


def _btrs_oracle(
    n: int,
    probability: float,
    blocks: Sequence[_WordBlock],
) -> tuple[int, int]:
    n_float, _, b, a, c, rectangle, _, alpha, _ = _btrs_parameters(
        n,
        probability,
    )
    for attempt, block in enumerate(blocks):
        uniform = _uniform_open_open_oracle(block[:2])
        variate = _uniform_open_open_oracle(block[2:])
        u = uniform - 0.5
        u_s = 0.5 - abs(u)
        proposal_float = math.floor((2.0 * a / u_s + b) * u + c)
        if proposal_float < 0 or proposal_float > n_float:
            continue
        proposal = int(proposal_float)
        if u_s >= 0.07 and variate <= rectangle:
            return proposal, attempt
        left = math.log(
            variate * alpha / (a / (u_s * u_s) + b)
        )
        if left <= _btrs_upper_oracle(n, probability, proposal):
            return proposal, attempt
    raise RuntimeError("oracle BTRS exhausted")


def _btrs_float_sides(
    n: int,
    probability: float,
    block: _WordBlock,
    proposal: int,
) -> tuple[float, float]:
    _, _, b, a, _, _, _, alpha, _ = _btrs_parameters(n, probability)
    uniform = _uniform_open_open_oracle(block[:2])
    variate = _uniform_open_open_oracle(block[2:])
    u_s = 0.5 - abs(uniform - 0.5)
    left = math.log(variate * alpha / (a / (u_s * u_s) + b))
    return left, _btrs_upper_oracle(n, probability, proposal)


def _fixed_blocks(
    blocks: Sequence[_WordBlock],
    observed_blocks: list[int],
) -> Callable[..., torch.Tensor]:
    def draw(
        *,
        seed: int,
        stream: _RngStream,
        logical_positions: torch.Tensor,
        source_quantum: int = 0,
        block: int = 0,
    ) -> torch.Tensor:
        del seed, stream, source_quantum
        observed_blocks.append(block)
        fixture = torch.tensor(
            blocks[block],
            dtype=torch.int64,
            device=logical_positions.device,
        )
        return fixture.expand((*logical_positions.shape, 4)).clone()

    return draw


def _decimal_open_open(words: tuple[int, int]) -> Decimal:
    mantissa = words[0] * (1 << 20) + (words[1] >> 12)
    return (Decimal("0.5") + Decimal(mantissa)) / Decimal(1 << 52)


def _decimal_stirling(value: int) -> Decimal:
    if value < 10:
        return Decimal.from_float(_STIRLING_CORRECTIONS[value])
    x = Decimal(value + 1)
    x2 = x * x
    inner = Decimal.from_float(1.0 / 360.0) - (
        Decimal.from_float(1.0 / 1260.0) / x2
    )
    return (Decimal.from_float(1.0 / 12.0) - inner / x2) / x


def _decimal_ptrs_sides(
    mean: float,
    block: _WordBlock,
    proposal: int,
) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = 100
        mean_decimal = Decimal.from_float(mean)
        sqrt_mean = mean_decimal.sqrt()
        b = Decimal.from_float(0.931) + Decimal.from_float(2.53) * sqrt_mean
        a = Decimal.from_float(-0.059) + Decimal.from_float(0.02483) * b
        inverse_alpha = Decimal.from_float(1.1239) + (
            Decimal.from_float(1.1328) / (b - Decimal.from_float(3.4))
        )
        uniform = _decimal_open_open(block[:2])
        variate = _decimal_open_open(block[2:])
        u_s = Decimal("0.5") - abs(uniform - Decimal("0.5"))
        left = (
            variate.ln()
            + inverse_alpha.ln()
            - (a / (u_s * u_s) + b).ln()
        )
        factorial = Decimal(math.factorial(proposal))
        right = (
            -mean_decimal
            + Decimal(proposal) * mean_decimal.ln()
            - factorial.ln()
        )
        return +left, +right


def _decimal_btrs_upper(
    n: int,
    probability: float,
    proposal: int,
) -> Decimal:
    with localcontext() as context:
        context.prec = 100
        n_decimal = Decimal(n)
        probability_decimal = Decimal.from_float(probability)
        ratio = probability_decimal / (Decimal(1) - probability_decimal)
        mode = math.floor((float(n) + 1.0) * probability)
        displacement = proposal - mode
        proposal_decimal = Decimal(proposal)
        displacement_decimal = Decimal(displacement)
        log_left = (
            Decimal(1)
            + displacement_decimal
            / (n_decimal - proposal_decimal + Decimal(1))
        ).ln()
        log_right = (
            Decimal(1)
            - displacement_decimal / (proposal_decimal + Decimal(1))
        ).ln()
        log_ratio = (
            ratio
            * (n_decimal - proposal_decimal + Decimal(1))
            / (proposal_decimal + Decimal(1))
        ).ln()
        main = (
            (n_decimal - Decimal(mode) + Decimal("0.5")) * log_left
            + (Decimal(mode) + Decimal("0.5")) * log_right
        ) + displacement_decimal * log_ratio
        correction = (
            (_decimal_stirling(mode) + _decimal_stirling(n - mode))
            - _decimal_stirling(proposal)
        ) - _decimal_stirling(n - proposal)
        return +(main + correction)


def _decimal_btrs_left(
    n: int,
    probability: float,
    block: _WordBlock,
) -> Decimal:
    with localcontext() as context:
        context.prec = 100
        n_decimal = Decimal(n)
        probability_decimal = Decimal.from_float(probability)
        s = (
            n_decimal
            * probability_decimal
            * (Decimal(1) - probability_decimal)
        ).sqrt()
        b = Decimal.from_float(1.15) + Decimal.from_float(2.53) * s
        a = (
            Decimal.from_float(-0.0873)
            + Decimal.from_float(0.0248) * b
            + Decimal.from_float(0.01) * probability_decimal
        )
        alpha = (
            Decimal.from_float(2.83) + Decimal.from_float(5.1) / b
        ) * s
        uniform = _decimal_open_open(block[:2])
        variate = _decimal_open_open(block[2:])
        u_s = Decimal("0.5") - abs(uniform - Decimal("0.5"))
        return +(
            variate * alpha / (a / (u_s * u_s) + b)
        ).ln()


def _log_allowance(reference: Decimal) -> float:
    return 1.0e-6 + 64.0 * math.ulp(1.0) * max(1.0, abs(float(reference)))


class PoissonSamplingTest(unittest.TestCase):
    def test_zero_is_exact_draw_free_and_explicit_positions_are_valid(self) -> None:
        positions = torch.tensor((0, 1), dtype=torch.int64)
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=AssertionError("zero mean must not request a word"),
        ):
            result = _sample_poisson(
                torch.zeros(2, dtype=torch.float64),
                shape=(2,),
                seed=0,
                stream=_RngStream.CHARGE_DARK_COUNTS,
                device="cpu",
                logical_positions=positions,
            )
        self.assertTrue(torch.equal(result, torch.zeros(2, dtype=torch.int64)))

    def test_repeatability_branch_crossover_and_position_preservation(self) -> None:
        means = torch.tensor((0.0, 0.5, 9.999, 10.0, 100.0), dtype=torch.float64)
        positions = torch.tensor((8, 3, 99, 5, 7), dtype=torch.int64)
        first = _sample_poisson(
            means,
            shape=(5,),
            seed=0x0123456789ABCDEF,
            stream=_RngStream.CHARGE_DARK_COUNTS,
            device="cpu",
            logical_positions=positions,
        )
        repeated = _sample_poisson(
            means,
            shape=(5,),
            seed=0x0123456789ABCDEF,
            stream=_RngStream.CHARGE_DARK_COUNTS,
            device="cpu",
            logical_positions=positions,
        )
        self.assertTrue(torch.equal(first, repeated))
        self.assertEqual(first.dtype, torch.int64)
        self.assertEqual(first[0].item(), 0)
        permuted = _sample_poisson(
            means[[4, 1, 2, 3, 0]],
            shape=(5,),
            seed=0x0123456789ABCDEF,
            stream=_RngStream.CHARGE_DARK_COUNTS,
            device="cpu",
            logical_positions=positions[[4, 1, 2, 3, 0]],
        )
        self.assertEqual(permuted[0].item(), first[4].item())
        self.assertEqual(permuted[1].item(), first[1].item())

    def test_parameter_and_representation_boundaries(self) -> None:
        accepted = _sample_poisson(
            1.0e8,
            shape=(4,),
            seed=1,
            stream=_RngStream.CHARGE_DIRECT_CROSSTALK,
            device="cpu",
        )
        self.assertTrue(bool(torch.all(accepted >= 0).item()))
        for mean in (-1.0, math.inf, math.nan, math.nextafter(1.0e8, math.inf)):
            with self.subTest(mean=mean):
                with self.assertRaises(ValueError):
                    _sample_poisson(
                        mean,
                        shape=(1,),
                        seed=1,
                        stream=_RngStream.CHARGE_DARK_COUNTS,
                        device="cpu",
                    )
        with self.assertRaises(TypeError):
            _sample_poisson(
                torch.ones(1, dtype=torch.float32),
                shape=(1,),
                seed=1,
                stream=_RngStream.CHARGE_DARK_COUNTS,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            _sample_poisson(
                torch.ones(2, dtype=torch.float64),
                shape=(1,),
                seed=1,
                stream=_RngStream.CHARGE_DARK_COUNTS,
                device="cpu",
            )

    def test_fixed_word_inversion_oracle_crossover_and_exhaustion(self) -> None:
        successful = (
            (0.5, (*_MINIMUM_WORDS, 0, 0)),
            (0.5, (*_MIDPOINT_WORDS, 0, 0)),
            (0.5, (*_MAXIMUM_WORDS, 0, 0)),
            (math.nextafter(10.0, 0.0), (*_MIDPOINT_WORDS, 0, 0)),
        )
        for mean, block in successful:
            with self.subTest(mean=mean, block=block[:2]):
                observed_blocks: list[int] = []
                expected = _poisson_inversion_oracle(mean, block)
                with patch(
                    "tensor_dslab.readout._random._random_block",
                    side_effect=_fixed_blocks((block,), observed_blocks),
                ):
                    sampled = _sample_poisson(
                        mean,
                        shape=(1,),
                        seed=1234,
                        stream=_RngStream.CHARGE_DARK_COUNTS,
                        device="cpu",
                        logical_positions=torch.tensor((37,), dtype=torch.int64),
                    )
                self.assertEqual(sampled.item(), expected)
                self.assertEqual(observed_blocks, [0])

        maximum_block = (*_MAXIMUM_WORDS, 0, 0)
        observed_blocks = []
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=_fixed_blocks((maximum_block,), observed_blocks),
        ):
            with self.assertRaisesRegex(RuntimeError, "64 terms"):
                _sample_poisson(
                    math.nextafter(10.0, 0.0),
                    shape=(1,),
                    seed=1234,
                    stream=_RngStream.CHARGE_DARK_COUNTS,
                    device="cpu",
                )
        self.assertEqual(observed_blocks, [0])

    def test_poisson_inversion_terms_match_100_digit_oracle(self) -> None:
        mean = math.nextafter(10.0, 0.0)
        probability = math.exp(-mean)
        cumulative = probability
        with localcontext() as context:
            context.prec = 100
            decimal_mean = Decimal.from_float(mean)
            decimal_probability = (-decimal_mean).exp()
            decimal_cumulative = decimal_probability
            for count in range(64):
                self.assertLessEqual(
                    abs(Decimal.from_float(probability) - decimal_probability),
                    Decimal("1e-12"),
                )
                self.assertLessEqual(
                    abs(Decimal.from_float(cumulative) - decimal_cumulative),
                    Decimal("1e-12"),
                )
                if count == 63:
                    break
                probability = probability * mean / float(count + 1)
                cumulative = cumulative + probability
                decimal_probability = (
                    decimal_probability
                    * decimal_mean
                    / Decimal(count + 1)
                )
                decimal_cumulative += decimal_probability

    def test_fixed_word_ptrs_paths_and_position_preservation(self) -> None:
        cases = (
            (10.0, (_QUICK_ACCEPT_BLOCK,)),
            (10.0, (_FULL_ACCEPT_BLOCK,)),
            (10.0, ((0, 0, 0, 0), _QUICK_ACCEPT_BLOCK)),
            (123.5, (_FULL_REJECT_BLOCK, _QUICK_ACCEPT_BLOCK)),
            (1.0e8, (_BTRS_CANCELLATION_BLOCK,)),
        )
        for mean, blocks in cases:
            with self.subTest(mean=mean, blocks=len(blocks)):
                expected, accepted_attempt = _ptrs_oracle(mean, blocks)
                observed_blocks: list[int] = []
                positions = torch.tensor((91,), dtype=torch.int64)
                with patch(
                    "tensor_dslab.readout._random._random_block",
                    side_effect=_fixed_blocks(blocks, observed_blocks),
                ):
                    sampled = _sample_poisson(
                        mean,
                        shape=(1,),
                        seed=99,
                        stream=_RngStream.CHARGE_DIRECT_CROSSTALK,
                        device="cpu",
                        logical_positions=positions,
                    )
                self.assertEqual(sampled.item(), expected)
                self.assertEqual(observed_blocks, list(range(accepted_attempt + 1)))

    def test_ptrs_100_digit_log_gate_and_uncertainty_fixture(self) -> None:
        proposal, _ = _ptrs_oracle(10.0, (_FULL_ACCEPT_BLOCK,))
        represented = _ptrs_float_sides(10.0, _FULL_ACCEPT_BLOCK, proposal)
        reference = _decimal_ptrs_sides(10.0, _FULL_ACCEPT_BLOCK, proposal)
        for actual, expected in zip(represented, reference):
            self.assertLessEqual(
                abs(actual - float(expected)),
                _log_allowance(expected),
            )
        self.assertGreater(
            abs(float(reference[0] - reference[1])),
            _log_allowance(reference[0]) + _log_allowance(reference[1]),
        )
        self.assertEqual(represented[0] <= represented[1], reference[0] <= reference[1])

        uncertain_reference = _decimal_ptrs_sides(
            10.0,
            _PTRS_UNCERTAIN_BLOCK,
            10,
        )
        self.assertLessEqual(
            abs(float(uncertain_reference[0] - uncertain_reference[1])),
            _log_allowance(uncertain_reference[0])
            + _log_allowance(uncertain_reference[1]),
        )
        observed_blocks: list[int] = []
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=_fixed_blocks(
                (_PTRS_UNCERTAIN_BLOCK, _QUICK_ACCEPT_BLOCK),
                observed_blocks,
            ),
        ):
            sampled = _sample_poisson(
                10.0,
                shape=(1,),
                seed=0,
                stream=_RngStream.CHARGE_DARK_COUNTS,
                device="cpu",
            )
        self.assertEqual(sampled.item(), 10)
        self.assertEqual(observed_blocks, [0, 1])

    def test_ptrs_exhausts_exactly_64_fixed_attempt_blocks(self) -> None:
        blocks = tuple((0, 0, 0, 0) for _ in range(64))
        observed_blocks: list[int] = []
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=_fixed_blocks(blocks, observed_blocks),
        ):
            with self.assertRaisesRegex(RuntimeError, "64 attempts"):
                _sample_poisson(
                    10.0,
                    shape=(1,),
                    seed=0,
                    stream=_RngStream.CHARGE_DARK_COUNTS,
                    device="cpu",
                )
        self.assertEqual(observed_blocks, list(range(64)))

    def test_frozen_scalar_moments_across_four_seeds(self) -> None:
        sample_per_seed = 1 << 16
        seeds = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
        for mean in (1.0, 9.5, 10.0, 100.0):
            parts = tuple(
                _sample_poisson(
                    mean,
                    shape=(sample_per_seed,),
                    seed=seed,
                    stream=_RngStream.CHARGE_DARK_COUNTS,
                    device="cpu",
                ).to(torch.float64)
                for seed in seeds
            )
            values = torch.cat(parts)
            observed_mean = float(torch.mean(values))
            observed_variance = float(torch.var(values, correction=0))
            total = values.numel()
            mean_standard_error = math.sqrt(mean / total)
            variance_standard_error = mean * math.sqrt(2.0 / total)
            self.assertLessEqual(abs(observed_mean - mean), 8.0 * mean_standard_error)
            self.assertLessEqual(
                abs(observed_variance - mean),
                8.0 * variance_standard_error,
            )

    def test_frozen_lambda_four_law_and_independent_superposition(self) -> None:
        examples_per_seed = 1 << 16
        positions = torch.arange(examples_per_seed, dtype=torch.int64)
        direct_parts: list[torch.Tensor] = []
        first_parts: list[torch.Tensor] = []
        second_parts: list[torch.Tensor] = []
        for seed in _STATISTICAL_SEEDS:
            direct_parts.append(
                _sample_poisson(
                    4.0,
                    shape=(examples_per_seed,),
                    seed=seed,
                    stream=_RngStream.CHARGE_DARK_COUNTS,
                    device="cpu",
                    logical_positions=positions,
                )
            )
            first_parts.append(
                _sample_poisson(
                    1.5,
                    shape=(examples_per_seed,),
                    seed=seed,
                    stream=_RngStream.CHARGE_DIRECT_CROSSTALK,
                    device="cpu",
                    logical_positions=positions,
                )
            )
            second_parts.append(
                _sample_poisson(
                    2.5,
                    shape=(examples_per_seed,),
                    seed=seed,
                    stream=_RngStream.CHARGE_DELAYED_CROSSTALK,
                    device="cpu",
                    logical_positions=positions,
                )
            )

        direct = torch.cat(direct_parts).to(torch.float64)
        first = torch.cat(first_parts).to(torch.float64)
        second = torch.cat(second_parts).to(torch.float64)
        superposed = first + second
        total_examples = direct.numel()
        self.assertEqual(total_examples, 1 << 18)
        self.assertEqual(first.numel(), total_examples)
        self.assertEqual(second.numel(), total_examples)

        def assert_gate(
            name: str,
            observed: float,
            target: float,
            standard_error: float,
        ) -> None:
            delta = _statistical_delta(
                scale=target,
                length=total_examples,
            )
            bound = 8.0 * standard_error + delta
            self.assertLessEqual(
                abs(observed - target),
                bound,
                msg=(
                    f"{name}: observed={observed!r}, target={target!r}, "
                    f"SE={standard_error!r}, delta={delta!r}, "
                    f"bound={bound!r}"
                ),
            )

        def assert_poisson_moments(
            name: str,
            values: torch.Tensor,
            mean: float,
        ) -> None:
            observed_mean = float(torch.mean(values))
            assert_gate(
                f"{name} mean",
                observed_mean,
                mean,
                math.sqrt(mean / total_examples),
            )
            observed_centered_variance = float(
                torch.mean((values - mean) ** 2)
            )
            centered_variance_standard_error = math.sqrt(
                (mean + 2.0 * mean * mean) / total_examples
            )
            assert_gate(
                f"{name} centered variance",
                observed_centered_variance,
                mean,
                centered_variance_standard_error,
            )

        assert_poisson_moments("direct lambda=4", direct, 4.0)

        probability_zero = math.exp(-4.0)
        probability_four = (
            math.exp(-4.0) * 4.0**4 / math.factorial(4)
        )
        probability_tail = 1.0 - math.fsum(
            math.exp(-4.0) * 4.0**value / math.factorial(value)
            for value in range(8)
        )
        frequency_cases = (
            ("direct P(X=0)", direct == 0.0, probability_zero),
            ("direct P(X=4)", direct == 4.0, probability_four),
            ("direct P(X>=8)", direct >= 8.0, probability_tail),
        )
        for name, mask, probability in frequency_cases:
            self.assertGreaterEqual(total_examples * probability, 256.0)
            self.assertGreaterEqual(
                total_examples * (1.0 - probability),
                256.0,
            )
            assert_gate(
                name,
                float(torch.mean(mask.to(torch.float64))),
                probability,
                math.sqrt(
                    probability * (1.0 - probability) / total_examples
                ),
            )

        assert_poisson_moments("first lambda=1.5", first, 1.5)
        assert_poisson_moments("second lambda=2.5", second, 2.5)
        assert_poisson_moments("superposed lambda=4", superposed, 4.0)

        observed_covariance = float(
            torch.mean((first - 1.5) * (second - 2.5))
        )
        assert_gate(
            "component covariance",
            observed_covariance,
            0.0,
            math.sqrt(1.5 * 2.5 / total_examples),
        )

        joint_zero_probability = math.exp(-1.5) * math.exp(-2.5)
        self.assertGreaterEqual(
            total_examples * joint_zero_probability,
            256.0,
        )
        self.assertGreaterEqual(
            total_examples * (1.0 - joint_zero_probability),
            256.0,
        )
        assert_gate(
            "independent joint P(X=0,Y=0)",
            float(torch.mean(((first == 0.0) & (second == 0.0)).to(torch.float64))),
            joint_zero_probability,
            math.sqrt(
                joint_zero_probability
                * (1.0 - joint_zero_probability)
                / total_examples
            ),
        )


class ConditionalBinomialSamplingTest(unittest.TestCase):
    def _sample(
        self,
        counts: torch.Tensor,
        success: float,
        later: float,
        *,
        seed: int,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        exact_positions = (
            torch.arange(counts.numel(), dtype=torch.int64).reshape(counts.shape)
            if positions is None
            else positions
        )
        return _sample_conditional_binomial(
            counts,
            torch.full(counts.shape, success, dtype=torch.float64),
            torch.full(counts.shape, later, dtype=torch.float64),
            seed=seed,
            stream=_RngStream.CHARGE_AFTERPULSES,
            logical_positions=exact_positions,
        )

    def test_exact_boundaries_reflection_and_mixed_support(self) -> None:
        counts = torch.tensor((0, 1, 5, 32), dtype=torch.int64)
        self.assertTrue(torch.equal(self._sample(counts, 0.0, 1.0, seed=0), torch.zeros_like(counts)))
        self.assertTrue(torch.equal(self._sample(counts, 1.0, 0.0, seed=0), counts))
        reflected = self._sample(counts, 0.9, 0.1, seed=7)
        unreflected = self._sample(counts, 0.1, 0.9, seed=7)
        self.assertTrue(torch.equal(reflected, counts - unreflected))
        mixed = self._sample(counts, 0.5, 0.5, seed=13)
        self.assertTrue(bool(torch.all((mixed >= 0) & (mixed <= counts)).item()))

    def test_aggregate_multinomial_conserves_every_count(self) -> None:
        total = torch.full((1024,), 32, dtype=torch.int64)
        remaining = total.clone()
        positions = torch.arange(total.numel(), dtype=torch.int64)
        categories: list[torch.Tensor] = []
        for category, (success, later) in enumerate(
            ((0.2, 0.8), (0.3, 0.5), (0.1, 0.4))
        ):
            sampled = _sample_conditional_binomial(
                remaining,
                torch.full(total.shape, success, dtype=torch.float64),
                torch.full(total.shape, later, dtype=torch.float64),
                seed=1234,
                stream=_RngStream.CHARGE_AFTERPULSES,
                logical_positions=positions + category * total.numel(),
            )
            categories.append(sampled)
            remaining = remaining - sampled
        categories.append(remaining)
        self.assertTrue(torch.equal(sum(categories[1:], categories[0]), total))

    def test_frozen_q32_aggregate_multinomial_joint_ensemble(self) -> None:
        count = 32
        examples_per_seed = 1 << 14
        conditional_masses = (
            (0.10, 0.90),
            (0.15, 0.75),
            (0.20, 0.55),
        )
        category_probabilities = (0.10, 0.15, 0.20, 0.55)
        position_basis = torch.arange(examples_per_seed, dtype=torch.int64)
        seed_ensembles: list[torch.Tensor] = []

        original_sampler = _sample_conditional_binomial
        with patch(
            f"{__name__}._sample_conditional_binomial",
            wraps=original_sampler,
        ) as aggregate_sampler:
            for seed in _STATISTICAL_SEEDS:
                total = torch.full(
                    (examples_per_seed,),
                    count,
                    dtype=torch.int64,
                )
                remaining = total.clone()
                categories: list[torch.Tensor] = []
                for category, (success, later) in enumerate(conditional_masses):
                    sampled = _sample_conditional_binomial(
                        remaining,
                        torch.full(
                            total.shape,
                            success,
                            dtype=torch.float64,
                        ),
                        torch.full(
                            total.shape,
                            later,
                            dtype=torch.float64,
                        ),
                        seed=seed,
                        stream=_RngStream.CHARGE_AFTERPULSES,
                        logical_positions=(
                            position_basis + category * examples_per_seed
                        ),
                    )
                    categories.append(sampled)
                    remaining = remaining - sampled
                categories.append(remaining)
                ensemble = torch.stack(categories, dim=1)
                self.assertTrue(
                    torch.equal(
                        torch.sum(ensemble, dim=1),
                        total,
                    )
                )
                seed_ensembles.append(ensemble)

        self.assertEqual(
            aggregate_sampler.call_count,
            len(_STATISTICAL_SEEDS) * len(conditional_masses),
        )
        for call_index, sampler_call in enumerate(
            aggregate_sampler.call_args_list
        ):
            seed_index, category = divmod(
                call_index,
                len(conditional_masses),
            )
            counts, success_mass, later_mass = sampler_call.args
            expected_success, expected_later = conditional_masses[category]
            self.assertEqual(counts.shape, (examples_per_seed,))
            self.assertEqual(counts.numel(), examples_per_seed)
            self.assertLessEqual(int(torch.max(counts)), count)
            self.assertTrue(
                bool(torch.all(success_mass == expected_success).item())
            )
            self.assertTrue(
                bool(torch.all(later_mass == expected_later).item())
            )
            self.assertEqual(
                sampler_call.kwargs["seed"],
                _STATISTICAL_SEEDS[seed_index],
            )
            self.assertTrue(
                torch.equal(
                    sampler_call.kwargs["logical_positions"],
                    position_basis + category * examples_per_seed,
                )
            )

        no_counts = torch.zeros(8, dtype=torch.int64)
        unit_counts = torch.full((8,), count, dtype=torch.int64)
        identity_positions = torch.arange(8, dtype=torch.int64)
        with patch("tensor_dslab.readout._random._random_block") as raw_words:
            self.assertTrue(
                torch.equal(
                    original_sampler(
                        no_counts,
                        torch.full((8,), 0.25, dtype=torch.float64),
                        torch.full((8,), 0.75, dtype=torch.float64),
                        seed=0,
                        stream=_RngStream.CHARGE_AFTERPULSES,
                        logical_positions=identity_positions,
                    ),
                    no_counts,
                )
            )
            self.assertTrue(
                torch.equal(
                    original_sampler(
                        unit_counts,
                        torch.zeros(8, dtype=torch.float64),
                        torch.ones(8, dtype=torch.float64),
                        seed=0,
                        stream=_RngStream.CHARGE_AFTERPULSES,
                        logical_positions=identity_positions,
                    ),
                    torch.zeros_like(unit_counts),
                )
            )
            self.assertTrue(
                torch.equal(
                    original_sampler(
                        unit_counts,
                        torch.ones(8, dtype=torch.float64),
                        torch.zeros(8, dtype=torch.float64),
                        seed=0,
                        stream=_RngStream.CHARGE_AFTERPULSES,
                        logical_positions=identity_positions,
                    ),
                    unit_counts,
                )
            )
            raw_words.assert_not_called()

        ensemble = torch.cat(seed_ensembles, dim=0).to(torch.float64)
        total_examples = len(_STATISTICAL_SEEDS) * examples_per_seed
        self.assertEqual(total_examples, 1 << 16)
        self.assertEqual(ensemble.shape, (total_examples, 4))

        for category, probability in enumerate(category_probabilities):
            values = ensemble[:, category]
            target_mean = count * probability
            target_variance = count * probability * (1.0 - probability)
            observed_mean = float(torch.mean(values))
            mean_standard_error = math.sqrt(target_variance / total_examples)
            mean_delta = _statistical_delta(
                scale=target_mean,
                length=total_examples,
            )
            mean_bound = 8.0 * mean_standard_error + mean_delta
            self.assertLessEqual(
                abs(observed_mean - target_mean),
                mean_bound,
                msg=(
                    f"category {category} mean: observed={observed_mean!r}, "
                    f"target={target_mean!r}, SE={mean_standard_error!r}, "
                    f"delta={mean_delta!r}, bound={mean_bound!r}"
                ),
            )

            observed_variance = float(
                torch.mean((values - target_mean) ** 2)
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
            variance_delta = _statistical_delta(
                scale=target_variance,
                length=total_examples,
            )
            variance_bound = 8.0 * variance_standard_error + variance_delta
            self.assertLessEqual(
                abs(observed_variance - target_variance),
                variance_bound,
                msg=(
                    f"category {category} variance: "
                    f"observed={observed_variance!r}, "
                    f"target={target_variance!r}, "
                    f"SE={variance_standard_error!r}, "
                    f"delta={variance_delta!r}, bound={variance_bound!r}"
                ),
            )

        for first in range(len(category_probabilities)):
            for second in range(first + 1, len(category_probabilities)):
                first_probability = category_probabilities[first]
                second_probability = category_probabilities[second]
                first_mean = count * first_probability
                second_mean = count * second_probability
                target_covariance = (
                    -count * first_probability * second_probability
                )
                observed_covariance = float(
                    torch.mean(
                        (ensemble[:, first] - first_mean)
                        * (ensemble[:, second] - second_mean)
                    )
                )
                cross_fourth = _multinomial_cross_fourth_moment(
                    count,
                    first_probability,
                    second_probability,
                )
                covariance_standard_error = math.sqrt(
                    (cross_fourth - target_covariance * target_covariance)
                    / total_examples
                )
                covariance_delta = _statistical_delta(
                    scale=target_covariance,
                    length=total_examples,
                )
                covariance_bound = (
                    8.0 * covariance_standard_error + covariance_delta
                )
                self.assertLessEqual(
                    abs(observed_covariance - target_covariance),
                    covariance_bound,
                    msg=(
                        f"categories ({first}, {second}) covariance: "
                        f"observed={observed_covariance!r}, "
                        f"target={target_covariance!r}, "
                        f"SE={covariance_standard_error!r}, "
                        f"delta={covariance_delta!r}, "
                        f"bound={covariance_bound!r}"
                    ),
                )

    def test_fixed_word_inversion_oracle_crossover_reflection_and_guard(self) -> None:
        cases = (
            (1, 0.5, (*_MINIMUM_WORDS, 0, 0)),
            (1, 0.5, (*_MAXIMUM_WORDS, 0, 0)),
            (100, math.nextafter(0.1, 0.0), (*_MIDPOINT_WORDS, 0, 0)),
            (
                (1 << 53) - 1,
                9.0 / float((1 << 53) - 1),
                (*_MIDPOINT_WORDS, 0, 0),
            ),
        )
        for n, probability, block in cases:
            with self.subTest(n=n, probability=probability):
                expected = _binomial_inversion_oracle(n, probability, block)
                observed_blocks: list[int] = []
                with patch(
                    "tensor_dslab.readout._random._random_block",
                    side_effect=_fixed_blocks((block,), observed_blocks),
                ):
                    sampled = self._sample(
                        torch.tensor((n,), dtype=torch.int64),
                        probability,
                        1.0 - probability,
                        seed=0,
                        positions=torch.tensor((73,), dtype=torch.int64),
                    )
                self.assertEqual(sampled.item(), expected)
                self.assertEqual(observed_blocks, [0])

        n = 100
        probability = math.nextafter(0.1, 0.0)
        expected = _binomial_inversion_oracle(
            n,
            probability,
            (*_MIDPOINT_WORDS, 0, 0),
        )
        observed_blocks = []
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=_fixed_blocks(
                ((*_MIDPOINT_WORDS, 0, 0),),
                observed_blocks,
            ),
        ):
            reflected = self._sample(
                torch.tensor((n,), dtype=torch.int64),
                1.0 - probability,
                probability,
                seed=0,
            )
        self.assertEqual(reflected.item(), n - expected)

        with patch(
            "tensor_dslab.readout._random._random_block",
            return_value=torch.zeros((1, 4), dtype=torch.int64),
        ), patch(
            "tensor_dslab.readout._random._uniform_closed_open",
            return_value=torch.ones(1, dtype=torch.float64),
        ):
            with self.assertRaisesRegex(RuntimeError, "64 terms"):
                self._sample(
                    torch.tensor((100,), dtype=torch.int64),
                    0.05,
                    0.95,
                    seed=0,
                )

    def test_binomial_inversion_terms_match_100_digit_oracle(self) -> None:
        n = (1 << 53) - 1
        probability = 9.0 / float(n)
        q = 1.0 - probability
        term = math.exp(float(n) * math.log1p(-probability))
        cumulative = term
        with localcontext() as context:
            context.prec = 100
            decimal_probability = Decimal.from_float(probability)
            decimal_q = Decimal(1) - decimal_probability
            decimal_term = (Decimal(n) * decimal_q.ln()).exp()
            decimal_cumulative = decimal_term
            for count in range(64):
                self.assertLessEqual(
                    abs(Decimal.from_float(term) - decimal_term),
                    Decimal("1e-12"),
                )
                self.assertLessEqual(
                    abs(Decimal.from_float(cumulative) - decimal_cumulative),
                    Decimal("1e-12"),
                )
                if count == 63:
                    break
                term = term * (
                    (float(n - count) / float(count + 1))
                    * (probability / q)
                )
                cumulative = cumulative + term
                decimal_term = decimal_term * (
                    Decimal(n - count)
                    / Decimal(count + 1)
                    * (decimal_probability / decimal_q)
                )
                decimal_cumulative += decimal_term

    def test_fixed_word_btrs_paths_crossover_and_reflection(self) -> None:
        maximum = (1 << 53) - 1
        cases = (
            (100, 0.1, (_QUICK_ACCEPT_BLOCK,)),
            (100, 0.1, (_FULL_ACCEPT_BLOCK,)),
            (100, 0.1, ((0, 0, 0, 0), _QUICK_ACCEPT_BLOCK)),
            (247, 0.05, (_FULL_REJECT_BLOCK, _QUICK_ACCEPT_BLOCK)),
            (maximum, 10.0 / float(maximum), (_FULL_ACCEPT_BLOCK,)),
            (maximum, 0.5, (_BTRS_CANCELLATION_BLOCK,)),
        )
        for n, probability, blocks in cases:
            with self.subTest(n=n, probability=probability, blocks=len(blocks)):
                expected, accepted_attempt = _btrs_oracle(n, probability, blocks)
                observed_blocks: list[int] = []
                with patch(
                    "tensor_dslab.readout._random._random_block",
                    side_effect=_fixed_blocks(blocks, observed_blocks),
                ):
                    sampled = self._sample(
                        torch.tensor((n,), dtype=torch.int64),
                        probability,
                        1.0 - probability,
                        seed=0,
                        positions=torch.tensor((29,), dtype=torch.int64),
                    )
                self.assertEqual(sampled.item(), expected)
                self.assertEqual(observed_blocks, list(range(accepted_attempt + 1)))

        expected, _ = _btrs_oracle(100, 0.1, (_FULL_ACCEPT_BLOCK,))
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=_fixed_blocks((_FULL_ACCEPT_BLOCK,), []),
        ):
            reflected = self._sample(
                torch.tensor((100,), dtype=torch.int64),
                0.9,
                0.1,
                seed=0,
            )
        self.assertEqual(reflected.item(), 100 - expected)

    def test_btrs_100_digit_gates_cancellation_and_uncertainty(self) -> None:
        for n, probability, block in (
            (100, 0.1, _FULL_ACCEPT_BLOCK),
            ((1 << 53) - 1, 0.5, _BTRS_CANCELLATION_BLOCK),
        ):
            proposal, _ = _btrs_oracle(n, probability, (block,))
            represented = _btrs_float_sides(n, probability, block, proposal)
            reference = (
                _decimal_btrs_left(n, probability, block),
                _decimal_btrs_upper(n, probability, proposal),
            )
            for actual, expected in zip(represented, reference):
                self.assertLessEqual(
                    abs(actual - float(expected)),
                    _log_allowance(expected),
                )
            separation = abs(float(reference[0] - reference[1]))
            if separation > (
                _log_allowance(reference[0]) + _log_allowance(reference[1])
            ):
                self.assertEqual(
                    represented[0] <= represented[1],
                    reference[0] <= reference[1],
                )

        maximum = (1 << 53) - 1
        probability = 0.25
        _, _, _, _, _, _, ratio, _, mode_float = _btrs_parameters(
            maximum,
            probability,
        )
        mode = int(mode_float)
        proposal = mode + 1
        represented_upper = _btrs_upper_oracle(maximum, probability, proposal)
        reference_upper = _decimal_btrs_upper(maximum, probability, proposal)
        self.assertLessEqual(
            abs(represented_upper - float(reference_upper)),
            _log_allowance(reference_upper),
        )
        maximum_float = float(maximum)
        old_grouping = (
            (float(mode) + 0.5)
            * math.log(
                (float(mode) + 1.0)
                / (ratio * (maximum_float - float(mode) + 1.0))
            )
            + _stirling_correction_oracle(mode)
            + _stirling_correction_oracle(maximum - mode)
            + (maximum_float + 1.0)
            * math.log(
                (maximum_float - float(mode) + 1.0)
                / (maximum_float - float(proposal) + 1.0)
            )
            + (float(proposal) + 0.5)
            * math.log(
                ratio
                * (maximum_float - float(proposal) + 1.0)
                / (float(proposal) + 1.0)
            )
            - _stirling_correction_oracle(proposal)
            - _stirling_correction_oracle(maximum - proposal)
        )
        self.assertGreater(abs(old_grouping - float(reference_upper)), 0.1)

        uncertain_reference = (
            _decimal_btrs_left(100, 0.1, _BTRS_UNCERTAIN_BLOCK),
            _decimal_btrs_upper(100, 0.1, 10),
        )
        self.assertLessEqual(
            abs(float(uncertain_reference[0] - uncertain_reference[1])),
            _log_allowance(uncertain_reference[0])
            + _log_allowance(uncertain_reference[1]),
        )
        observed_blocks: list[int] = []
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=_fixed_blocks(
                (_BTRS_UNCERTAIN_BLOCK, _QUICK_ACCEPT_BLOCK),
                observed_blocks,
            ),
        ):
            sampled = self._sample(
                torch.tensor((100,), dtype=torch.int64),
                0.1,
                0.9,
                seed=0,
            )
        self.assertEqual(sampled.item(), 10)
        self.assertEqual(observed_blocks, [0, 1])

    def test_btrs_complete_support_mixed_log_bound_gate(self) -> None:
        maximum = (1 << 53) - 1
        for probability in (10.0 / float(maximum), 0.1, 0.5):
            n_float, s, _, _, _, _, _, _, mode_float = _btrs_parameters(
                maximum,
                probability,
            )
            mode = int(mode_float)
            candidates = {
                0,
                maximum,
                mode,
                max(0, int(math.floor(mode_float - 25.0 * s))),
                min(maximum, int(math.ceil(mode_float + 25.0 * s))),
            }
            self.assertEqual(n_float, float(maximum))
            for proposal in candidates:
                with self.subTest(probability=probability, proposal=proposal):
                    represented = _btrs_upper_oracle(
                        maximum,
                        probability,
                        proposal,
                    )
                    reference = _decimal_btrs_upper(
                        maximum,
                        probability,
                        proposal,
                    )
                    self.assertLessEqual(
                        abs(represented - float(reference)),
                        _log_allowance(reference),
                    )

    def test_btrs_exhausts_exactly_64_fixed_attempt_blocks(self) -> None:
        blocks = tuple((0, 0, 0, 0) for _ in range(64))
        observed_blocks: list[int] = []
        with patch(
            "tensor_dslab.readout._random._random_block",
            side_effect=_fixed_blocks(blocks, observed_blocks),
        ):
            with self.assertRaisesRegex(RuntimeError, "64 attempts"):
                self._sample(
                    torch.tensor((100,), dtype=torch.int64),
                    0.1,
                    0.9,
                    seed=0,
                )
        self.assertEqual(observed_blocks, list(range(64)))

    def test_frozen_q32_moments_across_four_seeds(self) -> None:
        sample_per_seed = 1 << 14
        seeds = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
        probability = 0.25
        values = torch.cat(
            tuple(
                self._sample(
                    torch.full((sample_per_seed,), 32, dtype=torch.int64),
                    probability,
                    1.0 - probability,
                    seed=seed,
                ).to(torch.float64)
                for seed in seeds
            )
        )
        target_mean = 32.0 * probability
        target_variance = 32.0 * probability * (1.0 - probability)
        total = values.numel()
        observed_mean = float(torch.mean(values))
        observed_variance = float(torch.var(values, correction=0))
        self.assertLessEqual(
            abs(observed_mean - target_mean),
            8.0 * math.sqrt(target_variance / total),
        )
        self.assertLessEqual(
            abs(observed_variance - target_variance),
            8.0 * target_variance * math.sqrt(2.0 / total),
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaCountSamplingTest(unittest.TestCase):
    def test_charge_stream_raw_words_match_cpu_exactly(self) -> None:
        positions = torch.tensor(
            (0, 1, (1 << 32) - 1, 1 << 32, (1 << 48) + 17),
            dtype=torch.int64,
        )
        charge_streams = tuple(
            stream
            for stream in _RngStream
            if stream.name.startswith("CHARGE_")
        )
        self.assertEqual(len(charge_streams), 8)
        for stream in charge_streams:
            with self.subTest(stream=stream.name):
                cpu = _random_block(
                    seed=0xFEDCBA9876543210,
                    stream=stream,
                    logical_positions=positions,
                    source_quantum=0xA5A5A5A5,
                    block=0x10203040,
                )
                cuda = _random_block(
                    seed=0xFEDCBA9876543210,
                    stream=stream,
                    logical_positions=positions.to("cuda"),
                    source_quantum=0xA5A5A5A5,
                    block=0x10203040,
                )
                self.assertEqual(cuda.device.type, "cuda")
                self.assertTrue(torch.equal(cuda.cpu(), cpu))

    def test_cuda_poisson_and_binomial_are_repeatable(self) -> None:
        device = torch.device("cuda")
        means = torch.tensor(
            (0.5, math.nextafter(10.0, 0.0), 10.0, 100.0),
            dtype=torch.float64,
            device=device,
        )
        positions = torch.tensor((91, 17, 1 << 32, (1 << 48) + 3), device=device)
        poisson = _sample_poisson(
            means,
            shape=(4,),
            seed=0x0123456789ABCDEF,
            stream=_RngStream.CHARGE_DARK_COUNTS,
            device=device,
            logical_positions=positions,
        )
        repeated_poisson = _sample_poisson(
            means,
            shape=(4,),
            seed=0x0123456789ABCDEF,
            stream=_RngStream.CHARGE_DARK_COUNTS,
            device=device,
            logical_positions=positions,
        )
        self.assertEqual(poisson.device, device)
        self.assertTrue(torch.equal(poisson, repeated_poisson))

        counts = torch.tensor((1, 32, 100, 4096), dtype=torch.int64, device=device)
        success = torch.full((4,), 0.25, dtype=torch.float64, device=device)
        later = torch.full((4,), 0.75, dtype=torch.float64, device=device)
        binomial = _sample_conditional_binomial(
            counts,
            success,
            later,
            seed=0x0123456789ABCDEF,
            stream=_RngStream.CHARGE_AFTERPULSES,
            logical_positions=positions,
        )
        repeated_binomial = _sample_conditional_binomial(
            counts,
            success,
            later,
            seed=0x0123456789ABCDEF,
            stream=_RngStream.CHARGE_AFTERPULSES,
            logical_positions=positions,
        )
        self.assertEqual(binomial.device, device)
        self.assertTrue(torch.equal(binomial, repeated_binomial))
        self.assertTrue(bool(torch.all((binomial >= 0) & (binomial <= counts)).item()))


if __name__ == "__main__":
    unittest.main()
