from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import torch

from tensor_dslab.common import SampleAxis, SamplingConfig
from tensor_dslab.readout._random import (
    _RngStream,
    _logical_positions,
    _require_seed,
    _sample_conditional_binomial,
    _sample_poisson,
    _standard_normal_pair,
)
from tensor_dslab.readout._requirements import _require_sampling
from tensor_dslab.readout.charge.types import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    CorrelatedAvalancheConfig,
    DarkCountConfig,
    DirectCrosstalkConfig,
    DelayedCrosstalkConfig,
    ExponentialDelayConfig,
    FixedDelayConfig,
    TimingJitterConfig,
    _require_valid_values,
)
from tensor_dslab.readout.photoelectrons import Photoelectrons


_MAX_COUNT = (1 << 53) - 1
_MAX_POISSON_MEAN = 1.0e8
_MAX_SAMPLE_COUNT = 8192
_LOCAL_PROBABILITY_TOLERANCE = 1.0e-12
_COMPLETE_LAW_TOLERANCE = 1.0e-11


@dataclass(frozen=True, slots=True)
class _TimingJitterPlan:
    probabilities: tuple[float, ...]
    left_tails: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _DelayPlan:
    probabilities: tuple[float, ...]
    right_tails: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _AfterpulsePlan:
    delay: _DelayPlan
    recovery: tuple[float, ...] | None
    overflow_recovery: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class _CorrelatedAvalanchePlan:
    direct_crosstalk: _DelayPlan | None
    delayed_crosstalk: _DelayPlan | None
    afterpulse: _AfterpulsePlan | None
    ledger_depth: int
    ledger_bound: float


@dataclass(frozen=True, slots=True)
class _CorrelatedAvalancheResult:
    S1: torch.Tensor
    S2: torch.Tensor
    final_frontier: torch.Tensor
    total_count: torch.Tensor
    direct_crosstalk_count: torch.Tensor | None
    direct_crosstalk_overflow_count: torch.Tensor | None
    delayed_crosstalk_count: torch.Tensor | None
    delayed_crosstalk_overflow_count: torch.Tensor | None
    afterpulse_count: torch.Tensor | None
    afterpulse_overflow_count: torch.Tensor | None
    afterpulse_charge: torch.Tensor | None
    afterpulse_overflow_charge: torch.Tensor | None
    afterpulse_charge_square_sum: torch.Tensor | None


def _require_count_domain(counts: torch.Tensor, *, field: str) -> None:
    if counts.dtype is not torch.int64:
        raise TypeError(f"{field} must have dtype torch.int64")
    if bool(torch.any(counts < 0).item()) or bool(
        torch.any(counts > _MAX_COUNT).item()
    ):
        raise ValueError(f"{field} exceeds the accepted Charge count domain")


def _checked_add(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    field: str,
) -> torch.Tensor:
    if left.dtype is not torch.int64 or right.dtype is not torch.int64:
        raise TypeError(f"{field} count additions require torch.int64")
    if left.shape != right.shape or left.device != right.device:
        raise ValueError(f"{field} count additions require equal representations")
    if bool(torch.any(right < 0).item()):
        raise RuntimeError(f"{field} received a negative count contribution")
    if bool(torch.any(right > (_MAX_COUNT - left)).item()):
        raise RuntimeError(f"{field} exceeds the Charge count ceiling")
    return left + right


def _require_tensor_allocation(
    shape: tuple[int, ...],
    *,
    element_size: int,
    field: str,
) -> int:
    count = math.prod(shape)
    if count <= 0 or count > (1 << 63) - 1:
        raise ValueError(f"{field} element count exceeds the accepted range")
    if count * element_size > (1 << 63) - 1:
        raise ValueError(f"{field} byte count exceeds the accepted range")
    return count


def _q_exp_zero(x: float) -> float:
    if x <= 0.5:
        coefficients = tuple(
            float(Fraction((-1) ** (degree + 1), math.factorial(degree + 1)))
            for degree in range(1, 21)
        )
        value = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            value = coefficient + x * value
        return x * value
    return 1.0 - (-math.expm1(-x)) / x


def _prepare_exponential_from_inverse(
    x: float,
    *,
    sample_count: int,
) -> _DelayPlan:
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError("exponential inverse ratio must be finite and positive")
    a = -math.expm1(-x)
    c = a / x
    q_zero = _q_exp_zero(x)
    if not all(math.isfinite(value) for value in (a, c, q_zero)):
        raise ValueError("exponential preparation produced a nonfinite value")
    if not 0.0 <= q_zero <= 1.0 or not 0.0 <= c <= 1.0:
        raise ValueError("exponential preparation produced an invalid probability")

    log_c = math.log(c)
    log_a = math.log(a)
    tails = [1.0]
    probabilities = [q_zero]
    for offset in range(1, sample_count + 1):
        log_tail = log_c - (offset - 1) * x
        tail = math.exp(log_tail)
        tails.append(tail)
        if offset < sample_count:
            probabilities.append(math.exp(log_tail + log_a))

    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in (*tails, *probabilities)
    ):
        raise ValueError("exponential law contains an invalid represented value")
    if any(later > earlier for earlier, later in zip(tails, tails[1:])):
        raise ValueError("exponential right tail must be nonincreasing")
    if abs((q_zero + tails[1]) - 1.0) > _LOCAL_PROBABILITY_TOLERANCE:
        raise ValueError("exponential central probability identity failed")
    for offset, probability in enumerate(probabilities[1:], start=1):
        if abs((probability + tails[offset + 1]) - tails[offset]) > (
            _LOCAL_PROBABILITY_TOLERANCE
        ):
            raise ValueError("exponential tail telescoping identity failed")
    if abs(math.fsum((*probabilities, tails[sample_count])) - 1.0) > (
        _COMPLETE_LAW_TOLERANCE
    ):
        raise ValueError("exponential complete-law identity failed")
    return _DelayPlan(tuple(probabilities), tuple(tails))


def _prepare_exponential_delay(
    mean_delay_ns: float,
    *,
    sampling: SamplingConfig,
) -> _DelayPlan:
    sample_count = sampling.sample_count.value
    if sample_count > _MAX_SAMPLE_COUNT:
        raise ValueError("active exponential delay supports at most 8192 samples")
    mean_numerator, mean_denominator = mean_delay_ns.as_integer_ratio()
    period = sampling.sample_period_ps.value
    if mean_numerator * 1000 * (1 << 52) < mean_denominator * period:
        raise ValueError("exponential delay ratio is below the accepted domain")
    if mean_numerator * 1000 > mean_denominator * period * (1 << 52):
        raise ValueError("exponential delay ratio exceeds the accepted domain")
    mean_ps = mean_delay_ns * 1000.0
    ratio = mean_ps / float(period)
    if not math.isfinite(ratio) or not 2.0**-52 <= ratio <= 2.0**52:
        raise ValueError("exponential delay ratio is outside its represented domain")
    return _prepare_exponential_from_inverse(1.0 / ratio, sample_count=sample_count)


def _prepare_fixed_delay(
    delay_ns: float,
    *,
    sampling: SamplingConfig,
) -> _DelayPlan:
    sample_count = sampling.sample_count.value
    numerator, denominator = delay_ns.as_integer_ratio()
    period = sampling.sample_period_ps.value
    if numerator * 1000 >= denominator * sampling.window_stop_ps:
        return _DelayPlan(
            probabilities=(0.0,) * sample_count,
            right_tails=(1.0,) * (sample_count + 1),
        )
    offset, remainder = divmod(
        numerator * 1000,
        denominator * period,
    )
    fraction = float(Fraction(remainder, denominator * period))
    if remainder != 0 and not 0.0 < fraction < 1.0:
        raise ValueError("fixed-delay fractional mass became deterministic")
    first = 1.0 - fraction
    if math.fsum((first, fraction)) != 1.0:
        raise ValueError("fixed-delay represented masses do not sum exactly")

    probabilities = [0.0] * sample_count
    probabilities[offset] = first
    if fraction != 0.0 and offset + 1 < sample_count:
        probabilities[offset + 1] = fraction
    tails: list[float] = []
    for first_offset in range(sample_count + 1):
        if first_offset <= offset:
            tails.append(1.0)
        elif fraction != 0.0 and first_offset == offset + 1:
            tails.append(fraction)
        else:
            tails.append(0.0)
    return _DelayPlan(tuple(probabilities), tuple(tails))


def _prepare_delay(
    config: FixedDelayConfig | ExponentialDelayConfig,
    *,
    sampling: SamplingConfig,
) -> _DelayPlan:
    if type(config) is FixedDelayConfig:
        return _prepare_fixed_delay(config.delay_ns.value, sampling=sampling)
    if type(config) is ExponentialDelayConfig:
        return _prepare_exponential_delay(
            config.mean_delay_ns.value,
            sampling=sampling,
        )
    raise TypeError("crosstalk delay model is not recognized")


_F_SERIES = {
    1: Fraction(-1, 2),
    2: Fraction(1, 24),
    4: Fraction(-1, 2880),
    6: Fraction(1, 181440),
    8: Fraction(-1, 9676800),
    10: Fraction(1, 479001600),
    12: Fraction(-691, 15692092416000),
    14: Fraction(1, 1046139494400),
}
_G_SERIES = {
    1: Fraction(-1, 3),
    2: Fraction(1, 36),
    3: Fraction(-1, 810),
    4: Fraction(-1, 12960),
    5: Fraction(1, 68040),
    6: Fraction(-1, 12247200),
    7: Fraction(-1, 6123600),
    8: Fraction(13, 1175731200),
    9: Fraction(307, 218245104000),
    10: Fraction(-479, 2036954304000),
    11: Fraction(-167, 39720608928000),
    12: Fraction(100921, 28598838428160000),
    13: Fraction(-109, 649973600640000),
    14: Fraction(-3391, 85796515284480000),
}


def _series_difference(
    x: float,
    y: float,
    coefficients: dict[int, Fraction],
) -> float:
    z = x + y
    divided_power_difference = 1.0
    x_power = 1.0
    terms: list[float] = []
    for degree in range(1, 15):
        if degree in coefficients:
            terms.append(
                float(coefficients[degree]) * divided_power_difference
            )
        if degree < 14:
            x_power = x_power * x
            divided_power_difference = z * divided_power_difference + x_power
    return y * math.fsum(terms)


def _f_prime(value: float) -> float:
    return math.exp(-value) / (-math.expm1(-value)) - 1.0 / value


def _g_prime(value: float) -> float:
    a = -math.expm1(-value)
    q_zero = _q_exp_zero(value)
    derivative = (a - value * math.exp(-value)) / (value * value)
    return derivative / q_zero - 1.0 / value


def _f_difference(x: float, y: float) -> float:
    if x + y <= 0.5:
        return _series_difference(x, y, _F_SERIES)
    if y <= 2.0**-16 * max(1.0, x):
        return y * _f_prime(x + y / 2.0)
    return math.log1p(
        math.exp(-x) * (-math.expm1(-y)) / (-math.expm1(-x))
    ) - math.log1p(y / x)


def _g_difference(x: float, y: float) -> float:
    if x + y <= 0.5:
        return _series_difference(x, y, _G_SERIES)
    if y <= 2.0**-16 * max(1.0, x):
        return y * _g_prime(x + y / 2.0)
    return math.log(_q_exp_zero(x + y) / _q_exp_zero(x)) - math.log1p(y / x)


def _prepare_afterpulse_recovery(
    afterpulse: AfterpulseConfig,
    recovery: AfterpulseRecoveryConfig,
    *,
    sampling: SamplingConfig,
    delay: _DelayPlan,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    period = sampling.sample_period_ps.value
    recovery_value = recovery.time_constant_ns.value
    numerator, denominator = recovery_value.as_integer_ratio()
    if numerator * 1000 * (1 << 52) < denominator * period:
        raise ValueError("afterpulse recovery ratio is below the accepted domain")
    if numerator * 1000 > denominator * period * (1 << 52):
        raise ValueError("afterpulse recovery ratio exceeds the accepted domain")

    delay_ps = afterpulse.mean_delay_ns.value * 1000.0
    recovery_ps = recovery_value * 1000.0
    x = float(period) / delay_ps
    y = float(period) / recovery_ps
    combined = x + y
    if not all(math.isfinite(value) and value > 0.0 for value in (x, y, combined)):
        raise ValueError("afterpulse recovery inverse ratios must be finite")
    if not 2.0**-51 <= combined <= 2.0**53:
        raise ValueError("effective recovery inverse ratio is outside its domain")

    effective = _prepare_exponential_from_inverse(
        combined,
        sample_count=sampling.sample_count.value,
    )
    c = x / combined
    f_difference = _f_difference(x, y)
    recovery_weights: list[float] = []
    for offset, probability in enumerate(delay.probabilities):
        if probability == 0.0:
            recovery_weights.append(0.0)
            continue
        ell = (
            _g_difference(x, y)
            if offset == 0
            else 2.0 * f_difference - (offset - 1) * y
        )
        if not math.isfinite(ell) or ell > 0.0:
            raise ValueError("afterpulse recovery category log ratio is invalid")
        weight = -math.expm1(ell)
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("afterpulse recovery category is invalid")
        recovery_mass = probability * weight
        if not 0.0 <= recovery_mass <= probability:
            raise ValueError("afterpulse recovery mass is invalid")
        if abs(
            (recovery_mass + c * effective.probabilities[offset]) - probability
        ) > _LOCAL_PROBABILITY_TOLERANCE:
            raise ValueError("afterpulse recovery category identity failed")
        recovery_weights.append(weight)

    overflow_weights = [0.0]
    for first_outside in range(1, sampling.sample_count.value + 1):
        probability = delay.right_tails[first_outside]
        if probability == 0.0:
            overflow_weights.append(0.0)
            continue
        ell = (
            -math.log1p(y / x)
            + f_difference
            - (first_outside - 1) * y
        )
        if not math.isfinite(ell) or ell > 0.0:
            raise ValueError("afterpulse overflow recovery log ratio is invalid")
        weight = -math.expm1(ell)
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("afterpulse overflow recovery is invalid")
        recovery_mass = probability * weight
        if abs(
            (recovery_mass + c * effective.right_tails[first_outside])
            - probability
        ) > _LOCAL_PROBABILITY_TOLERANCE:
            raise ValueError("afterpulse overflow recovery identity failed")
        overflow_weights.append(weight)
    return tuple(recovery_weights), tuple(overflow_weights)


def _log_jitter_g(value: float) -> float:
    log_phi = -0.5 * value * value - 0.5 * math.log(2.0 * math.pi)
    if value == 0.0:
        return log_phi
    if value < 8.0:
        represented = math.exp(log_phi) - 0.5 * value * math.erfc(
            value / math.sqrt(2.0)
        )
        if not math.isfinite(represented) or represented <= 0.0:
            raise ValueError("timing-jitter tail helper is invalid")
        return math.log(represented)

    squared = value * value
    term = 1.0 / squared
    total = term
    for order in range(2, 101):
        candidate = -(2 * order - 1) * term / squared
        if abs(candidate) >= abs(term):
            break
        advanced = total + candidate
        if advanced == total:
            break
        total = advanced
        term = candidate
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("timing-jitter asymptotic tail helper is invalid")
    return log_phi + math.log(total)


def _prepare_timing_jitter(
    config: TimingJitterConfig,
    *,
    sampling: SamplingConfig,
    tensor_numel: int,
) -> _TimingJitterPlan:
    sigma_ns = config.sigma_ns.value
    if sigma_ns == 0.0:
        raise ValueError("zero timing jitter uses the exact identity path")
    sample_count = sampling.sample_count.value
    if sample_count > _MAX_SAMPLE_COUNT:
        raise ValueError("active timing jitter supports at most 8192 samples")
    if sample_count * tensor_numel > 1 << 63:
        raise ValueError("timing-jitter address lattice exceeds its domain")
    period_ps = float(sampling.sample_period_ps.value)
    if not (
        period_ps * 2.0**-52 * 1.0e-3
        <= sigma_ns
        <= period_ps * 64.0 * 1.0e-3
    ):
        raise ValueError("timing-jitter ratio is outside its accepted domain")
    sigma_ps = sigma_ns * 1000.0
    ratio = sigma_ps / period_ps
    if not math.isfinite(ratio) or not 2.0**-52 <= ratio <= 64.0:
        raise ValueError("timing-jitter represented ratio is outside its domain")

    log_g = tuple(
        _log_jitter_g(index / ratio)
        for index in range(sample_count + 1)
    )
    left_tails: list[float] = []
    log_left_tails: list[float] = []
    log_ratio = math.log(ratio)
    for index in range(sample_count):
        difference = log_g[index + 1] - log_g[index]
        log_tail = log_ratio + log_g[index] + math.log(-math.expm1(difference))
        log_left_tails.append(log_tail)
        left_tails.append(math.exp(log_tail))

    q_zero = math.erf(1.0 / (math.sqrt(2.0) * ratio)) + (
        ratio
        * math.sqrt(2.0 / math.pi)
        * math.expm1(-1.0 / (2.0 * ratio * ratio))
    )
    probabilities = [q_zero]
    for offset in range(1, sample_count):
        previous = left_tails[offset - 1]
        following = left_tails[offset]
        if previous == 0.0:
            probability = 0.0
        elif following == 0.0:
            probability = previous
        else:
            probability = math.exp(
                log_left_tails[offset - 1]
                + math.log(
                    -math.expm1(
                        log_left_tails[offset] - log_left_tails[offset - 1]
                    )
                )
            )
        probabilities.append(probability)

    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in (*probabilities, *left_tails)
    ):
        raise ValueError("timing-jitter preparation produced an invalid probability")
    if any(later > earlier for earlier, later in zip(left_tails, left_tails[1:])):
        raise ValueError("timing-jitter left tail must be nonincreasing")
    if abs((q_zero + 2.0 * left_tails[0]) - 1.0) > (
        _LOCAL_PROBABILITY_TOLERANCE
    ):
        raise ValueError("timing-jitter central identity failed")
    if abs(
        math.fsum((q_zero, *(2.0 * p for p in probabilities[1:]), 2.0 * left_tails[-1]))
        - 1.0
    ) > _COMPLETE_LAW_TOLERANCE:
        raise ValueError("timing-jitter complete-law identity failed")
    return _TimingJitterPlan(tuple(probabilities), tuple(left_tails))


def _prepare_dark_mean(
    config: DarkCountConfig,
    *,
    sampling: SamplingConfig,
) -> float:
    rate = config.rate_hz.value
    if rate == 0.0:
        return 0.0
    numerator, denominator = rate.as_integer_ratio()
    if numerator * sampling.sample_period_ps.value > denominator * 10**20:
        raise ValueError("dark-count mean exceeds the accepted Poisson domain")
    mean = float(
        Fraction(
            numerator * sampling.sample_period_ps.value,
            denominator * 10**12,
        )
    )
    if not math.isfinite(mean) or not 0.0 < mean <= _MAX_POISSON_MEAN:
        raise ValueError("dark-count mean is outside the accepted Poisson domain")
    return mean


def _ledger_envelope(
    *,
    floating_dtype: torch.dtype,
    maximum_generations: int,
    retained_mechanisms: int,
    recovered_afterpulse: bool,
    sample_count: int,
) -> tuple[int, float]:
    precision = 24 if floating_dtype is torch.float32 else 53
    depth = (
        retained_mechanisms * maximum_generations + sample_count + 3
        if recovered_afterpulse
        else retained_mechanisms * maximum_generations + 1
    )
    if depth >= 1 << precision:
        raise ValueError("correlated-avalanche ledger depth exceeds the dtype domain")
    gamma = depth / ((1 << precision) - depth)
    zero = torch.tensor(0.0, dtype=floating_dtype, device="cpu")
    one = torch.tensor(1.0, dtype=floating_dtype, device="cpu")
    subnormal = float(torch.nextafter(zero, one))
    bound = _MAX_COUNT * (1.0 + gamma) + depth * subnormal
    if not math.isfinite(bound):
        raise ValueError("correlated-avalanche ledger bound is nonfinite")
    return depth, bound


def _prepare_correlated_plan(
    config: CorrelatedAvalancheConfig,
    *,
    sampling: SamplingConfig,
    floating_dtype: torch.dtype,
    tensor_numel: int,
) -> _CorrelatedAvalanchePlan:
    maximum_generations = config.maximum_generations.value
    if maximum_generations == 0:
        _, bound = _ledger_envelope(
            floating_dtype=floating_dtype,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=sampling.sample_count.value,
        )
        return _CorrelatedAvalanchePlan(None, None, None, 1, bound)

    direct = (
        None
        if config.direct_crosstalk is None
        or config.direct_crosstalk.mean_offspring_per_parent.value == 0.0
        else _prepare_delay(config.direct_crosstalk.delay, sampling=sampling)
    )
    delayed = (
        None
        if config.delayed_crosstalk is None
        or config.delayed_crosstalk.mean_offspring_per_parent.value == 0.0
        else _prepare_delay(config.delayed_crosstalk.delay, sampling=sampling)
    )
    afterpulse: _AfterpulsePlan | None = None
    if config.afterpulse is not None and config.afterpulse.probability.value != 0.0:
        delay = _prepare_exponential_delay(
            config.afterpulse.mean_delay_ns.value,
            sampling=sampling,
        )
        recovery: tuple[float, ...] | None = None
        overflow_recovery: tuple[float, ...] | None = None
        if config.afterpulse.recovery is not None:
            recovery, overflow_recovery = _prepare_afterpulse_recovery(
                config.afterpulse,
                config.afterpulse.recovery,
                sampling=sampling,
                delay=delay,
            )
            recovery = tuple(
                float(torch.tensor(value, dtype=floating_dtype, device="cpu"))
                for value in recovery
            )
            overflow_recovery = tuple(
                float(torch.tensor(value, dtype=floating_dtype, device="cpu"))
                for value in overflow_recovery
            )
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (*recovery, *overflow_recovery)
            ):
                raise ValueError("afterpulse recovery is invalid in the Charge dtype")
        afterpulse = _AfterpulsePlan(delay, recovery, overflow_recovery)

    if direct is not None or delayed is not None:
        if maximum_generations * tensor_numel > 1 << 63:
            raise ValueError("crosstalk address lattice exceeds its domain")
    if afterpulse is not None and (
        maximum_generations
        * (sampling.sample_count.value + 1)
        * tensor_numel
        > 1 << 63
    ):
        raise ValueError("afterpulse address lattice exceeds its domain")

    retained_mechanisms = sum(
        (
            direct is not None and any(direct.probabilities),
            delayed is not None and any(delayed.probabilities),
            afterpulse is not None and any(afterpulse.delay.probabilities),
        )
    )
    recovered_afterpulse = afterpulse is not None and afterpulse.recovery is not None
    depth, bound = _ledger_envelope(
        floating_dtype=floating_dtype,
        maximum_generations=maximum_generations,
        retained_mechanisms=retained_mechanisms,
        recovered_afterpulse=recovered_afterpulse,
        sample_count=sampling.sample_count.value,
    )
    return _CorrelatedAvalanchePlan(
        direct,
        delayed,
        afterpulse,
        depth,
        bound,
    )


def _prepare_smearing_sigma(
    config: ChargeSmearingConfig,
    *,
    floating_dtype: torch.dtype,
    ledger_bound: float,
) -> float:
    requested = config.relative_sigma.value
    if requested == 0.0:
        return 0.0
    represented = float(torch.tensor(requested, dtype=floating_dtype, device="cpu"))
    if not math.isfinite(represented) or represented <= 0.0:
        raise ValueError("charge-smearing width is not positive and finite in the dtype")
    precision = 24 if floating_dtype is torch.float32 else 53
    maximum_normal = math.sqrt(-2.0 * math.log(2.0**-precision))
    maximum_normal = math.nextafter(maximum_normal, math.inf)
    square_root = math.nextafter(math.sqrt(ledger_bound), math.inf)
    scale = math.nextafter(represented * square_root, math.inf)
    excursion = math.nextafter(maximum_normal * scale, math.inf)
    maximum = math.nextafter(ledger_bound + excursion, math.inf)
    if not math.isfinite(maximum) or maximum > torch.finfo(floating_dtype).max:
        raise ValueError("charge-smearing finite envelope exceeds the dtype")
    return represented


def _original_positions(
    shape: tuple[int, ...],
    *,
    sample_dimension: int,
    device: torch.device,
) -> torch.Tensor:
    return _logical_positions(shape, device=device).movedim(sample_dimension, -1)


def _simulate_dark_counts(
    counts: torch.Tensor,
    *,
    sampling: SamplingConfig,
    config: DarkCountConfig,
    seed: int | None,
) -> torch.Tensor:
    if type(config) is not DarkCountConfig:
        raise TypeError("config must be exactly DarkCountConfig")
    _require_count_domain(counts, field="dark-count input")
    mean = _prepare_dark_mean(config, sampling=sampling)
    if mean == 0.0:
        return counts
    if seed is None:
        raise ValueError("dark counts require a seed")
    sampled = _sample_poisson(
        mean,
        shape=tuple(counts.shape),
        seed=seed,
        stream=_RngStream.CHARGE_DARK_COUNTS,
        device=counts.device,
    )
    return _checked_add(counts, sampled, field="dark-count result")


def _simulate_timing_jitter(
    counts: torch.Tensor,
    *,
    sample_dimension: int,
    sampling: SamplingConfig,
    config: TimingJitterConfig,
    seed: int | None,
) -> torch.Tensor:
    if type(config) is not TimingJitterConfig:
        raise TypeError("config must be exactly TimingJitterConfig")
    _require_count_domain(counts, field="timing-jitter input")
    if type(sample_dimension) is not int:
        raise TypeError("sample_dimension must be exactly an integer")
    if sample_dimension < 0 or sample_dimension >= counts.ndim:
        raise ValueError("sample_dimension is outside the count rank")
    if counts.shape[sample_dimension] != sampling.sample_count.value:
        raise ValueError("sample dimension disagrees with SamplingConfig")
    if config.sigma_ns.value == 0.0:
        return counts
    plan = _prepare_timing_jitter(
        config,
        sampling=sampling,
        tensor_numel=counts.numel(),
    )
    if not bool(torch.any(counts != 0).item()):
        return counts.clone()
    if seed is None:
        raise ValueError("effective timing jitter requires a seed")

    sample_count = sampling.sample_count.value
    total_count = counts.numel()
    sample_last = counts.movedim(sample_dimension, -1)
    remaining = sample_last.clone()
    result = torch.zeros_like(sample_last)
    positions = _original_positions(
        tuple(counts.shape),
        sample_dimension=sample_dimension,
        device=counts.device,
    )

    for target in range(sample_count):
        destination = result[..., target]
        for source in range(sample_count):
            offset = target - source
            if offset < 0:
                distance = -offset
                success = plan.probabilities[distance]
                later = 1.0 - plan.left_tails[distance - 1] + plan.left_tails[source]
            elif offset == 0:
                success = plan.probabilities[0]
                later = plan.left_tails[source] + plan.left_tails[0]
            else:
                success = plan.probabilities[offset]
                later = plan.left_tails[source] + plan.left_tails[offset]
            source_remaining = remaining[..., source]
            shape = tuple(source_remaining.shape)
            success_mass = torch.full(
                shape,
                success,
                dtype=torch.float64,
                device=counts.device,
            )
            later_mass = torch.full(
                shape,
                later,
                dtype=torch.float64,
                device=counts.device,
            )
            category = _sample_conditional_binomial(
                source_remaining,
                success_mass,
                later_mass,
                seed=seed,
                stream=_RngStream.CHARGE_TIMING_JITTER,
                logical_positions=positions[..., source] + target * total_count,
            )
            if bool(torch.any(category > source_remaining).item()):
                raise RuntimeError("timing-jitter category exceeds its remainder")
            remaining[..., source] = source_remaining - category
            destination = _checked_add(
                destination,
                category,
                field="timing-jitter destination",
            )
        result[..., target] = destination
    return result.movedim(-1, sample_dimension)


def _checked_rate_product(
    basis: torch.Tensor,
    mean: float,
    *,
    field: str,
) -> torch.Tensor:
    if basis.dtype is not torch.float64:
        raise TypeError(f"{field} basis must have dtype torch.float64")
    if not bool(torch.all(torch.isfinite(basis) & (basis >= 0.0)).item()):
        raise RuntimeError(f"{field} basis is invalid")
    threshold = _MAX_POISSON_MEAN / mean
    if math.isinf(threshold):
        if not math.isfinite(mean) or mean <= 0.0:
            raise RuntimeError(f"{field} mean is invalid")
    elif bool(torch.any(basis > threshold).item()):
        raise RuntimeError(f"{field} rate exceeds the Poisson ceiling")
    rate = basis * mean
    if not bool(
        torch.all(
            torch.isfinite(rate)
            & (rate >= 0.0)
            & (rate <= _MAX_POISSON_MEAN)
        ).item()
    ):
        raise RuntimeError(f"{field} rate exceeds the Poisson domain")
    return rate


def _draw_crosstalk(
    frontier: torch.Tensor,
    *,
    positions: torch.Tensor,
    plan: _DelayPlan,
    mean: float,
    retained_stream: _RngStream,
    overflow_stream: _RngStream,
    generation_index: int,
    tensor_numel: int,
    seed: int | None,
    field: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count = frontier.shape[-1]
    basis = torch.zeros_like(frontier, dtype=torch.float64)
    for destination in range(sample_count):
        accumulated = basis[..., destination]
        for source in range(destination + 1):
            probability = plan.probabilities[destination - source]
            if probability == 0.0:
                continue
            contribution = frontier[..., source].to(torch.float64) * probability
            accumulated = accumulated + contribution
        basis[..., destination] = accumulated
    retained_rate = _checked_rate_product(basis, mean, field=f"{field} retained")

    overflow_basis = torch.zeros_like(frontier, dtype=torch.float64)
    for source in range(sample_count):
        probability = plan.right_tails[sample_count - source]
        if probability != 0.0:
            overflow_basis[..., source] = (
                frontier[..., source].to(torch.float64) * probability
            )
    overflow_rate = _checked_rate_product(
        overflow_basis,
        mean,
        field=f"{field} overflow",
    )
    if bool(torch.any((retained_rate > 0.0) | (overflow_rate > 0.0)).item()):
        if seed is None:
            raise ValueError(f"effective {field} requires a seed")
        sampler_seed = seed
    else:
        sampler_seed = 0
    generation_positions = positions + generation_index * tensor_numel
    retained = _sample_poisson(
        retained_rate,
        shape=tuple(frontier.shape),
        seed=sampler_seed,
        stream=retained_stream,
        device=frontier.device,
        logical_positions=generation_positions,
    )
    overflow = _sample_poisson(
        overflow_rate,
        shape=tuple(frontier.shape),
        seed=sampler_seed,
        stream=overflow_stream,
        device=frontier.device,
        logical_positions=generation_positions,
    )
    return retained, overflow


def _draw_afterpulses(
    frontier: torch.Tensor,
    *,
    positions: torch.Tensor,
    plan: _AfterpulsePlan,
    config: AfterpulseConfig,
    generation_index: int,
    tensor_numel: int,
    floating_dtype: torch.dtype,
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_count = frontier.shape[-1]
    retained_count = torch.zeros_like(frontier)
    overflow_count = torch.zeros_like(frontier)
    retained_charge = torch.zeros_like(frontier, dtype=floating_dtype)
    overflow_charge = torch.zeros_like(frontier, dtype=floating_dtype)
    charge_square_sum = torch.zeros_like(frontier, dtype=floating_dtype)
    probability = config.probability.value

    for source in range(sample_count):
        remaining = frontier[..., source].clone()
        source_positions = positions[..., source]
        for offset in range(sample_count - source):
            success = probability * plan.delay.probabilities[offset]
            later = (1.0 - probability) + probability * plan.delay.right_tails[
                offset + 1
            ]
            shape = tuple(remaining.shape)
            success_mass = torch.full(
                shape,
                success,
                dtype=torch.float64,
                device=frontier.device,
            )
            later_mass = torch.full(
                shape,
                later,
                dtype=torch.float64,
                device=frontier.device,
            )
            if bool(torch.any(remaining != 0).item()) and success != 0.0 and later != 0.0:
                if seed is None:
                    raise ValueError("effective afterpulsing requires a seed")
                sampler_seed = seed
            else:
                sampler_seed = 0
            category_position = (
                (generation_index * (sample_count + 1) + offset) * tensor_numel
                + source_positions
            )
            category = _sample_conditional_binomial(
                remaining,
                success_mass,
                later_mass,
                seed=sampler_seed,
                stream=_RngStream.CHARGE_AFTERPULSES,
                logical_positions=category_position,
            )
            if bool(torch.any(category > remaining).item()):
                raise RuntimeError("afterpulse category exceeds its remainder")
            remaining = remaining - category
            destination = source + offset
            retained_count[..., destination] = _checked_add(
                retained_count[..., destination],
                category,
                field="afterpulse retained count",
            )
            if plan.recovery is not None:
                represented_weight = torch.tensor(
                    plan.recovery[offset],
                    dtype=floating_dtype,
                    device=frontier.device,
                )
                category_float = category.to(floating_dtype)
                retained_charge[..., destination] = (
                    retained_charge[..., destination]
                    + category_float * represented_weight
                )
                charge_square_sum[..., destination] = (
                    charge_square_sum[..., destination]
                    + category_float
                    * (represented_weight * represented_weight)
                )

        first_outside = sample_count - source
        success = probability * plan.delay.right_tails[first_outside]
        later = 1.0 - probability
        shape = tuple(remaining.shape)
        if bool(torch.any(remaining != 0).item()) and success != 0.0 and later != 0.0:
            if seed is None:
                raise ValueError("effective afterpulse overflow requires a seed")
            sampler_seed = seed
        else:
            sampler_seed = 0
        overflow = _sample_conditional_binomial(
            remaining,
            torch.full(
                shape,
                success,
                dtype=torch.float64,
                device=frontier.device,
            ),
            torch.full(
                shape,
                later,
                dtype=torch.float64,
                device=frontier.device,
            ),
            seed=sampler_seed,
            stream=_RngStream.CHARGE_AFTERPULSES,
            logical_positions=(
                (generation_index * (sample_count + 1) + sample_count)
                * tensor_numel
                + source_positions
            ),
        )
        if bool(torch.any(overflow > remaining).item()):
            raise RuntimeError("afterpulse overflow exceeds its remainder")
        overflow_count[..., source] = overflow
        if plan.overflow_recovery is not None:
            overflow_charge[..., source] = overflow.to(floating_dtype) * torch.tensor(
                plan.overflow_recovery[first_outside],
                dtype=floating_dtype,
                device=frontier.device,
            )

    if plan.recovery is None:
        retained_charge = retained_count.to(floating_dtype)
        charge_square_sum = retained_charge
        overflow_charge = overflow_count.to(floating_dtype)

    for field, value in (
        ("afterpulse charge", retained_charge),
        ("afterpulse overflow charge", overflow_charge),
        ("afterpulse charge-square sum", charge_square_sum),
    ):
        if not bool(torch.all(torch.isfinite(value) & (value >= 0.0)).item()):
            raise RuntimeError(f"{field} is invalid")
    return (
        retained_count,
        overflow_count,
        retained_charge,
        overflow_charge,
        charge_square_sum,
    )


def _checked_ledger_add(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bound: float,
    field: str,
) -> torch.Tensor:
    if left.dtype is not right.dtype or left.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{field} requires one common floating dtype")
    result = left + right
    if not bool(torch.all(torch.isfinite(result) & (result >= 0.0)).item()):
        raise RuntimeError(f"{field} produced an invalid ledger value")
    if bool(torch.any(result > bound).item()):
        raise RuntimeError(f"{field} exceeds the proved ledger bound")
    return result


def _simulate_correlated_avalanches(
    seed_avalanches: torch.Tensor,
    *,
    sample_dimension: int,
    sampling: SamplingConfig,
    floating_dtype: torch.dtype,
    config: CorrelatedAvalancheConfig,
    seed: int | None,
) -> _CorrelatedAvalancheResult:
    if type(config) is not CorrelatedAvalancheConfig:
        raise TypeError("config must be exactly CorrelatedAvalancheConfig")
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")
    _require_count_domain(seed_avalanches, field="correlated-avalanche roots")
    if type(sample_dimension) is not int:
        raise TypeError("sample_dimension must be exactly an integer")
    if sample_dimension < 0 or sample_dimension >= seed_avalanches.ndim:
        raise ValueError("sample_dimension is outside the root rank")
    if seed_avalanches.shape[sample_dimension] != sampling.sample_count.value:
        raise ValueError("sample dimension disagrees with SamplingConfig")

    tensor_numel = seed_avalanches.numel()
    plan = _prepare_correlated_plan(
        config,
        sampling=sampling,
        floating_dtype=floating_dtype,
        tensor_numel=tensor_numel,
    )
    sample_last = seed_avalanches.movedim(sample_dimension, -1)
    positions = _original_positions(
        tuple(seed_avalanches.shape),
        sample_dimension=sample_dimension,
        device=seed_avalanches.device,
    )
    maximum_generations = config.maximum_generations.value

    S1 = sample_last.to(floating_dtype)
    S2 = sample_last.to(floating_dtype)
    total_count = sample_last.clone()
    frontier = sample_last
    if (
        plan.direct_crosstalk is None
        and plan.delayed_crosstalk is None
        and plan.afterpulse is None
    ):
        final_frontier = (
            frontier.clone()
            if maximum_generations == 0
            else torch.zeros_like(frontier)
        )
        restore = lambda value: value.movedim(-1, sample_dimension)
        return _CorrelatedAvalancheResult(
            restore(S1),
            restore(S2),
            restore(final_frontier),
            restore(total_count),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    direct_count = (
        torch.zeros_like(sample_last)
        if plan.direct_crosstalk is not None
        else None
    )
    direct_overflow = (
        torch.zeros_like(sample_last)
        if plan.direct_crosstalk is not None
        else None
    )
    delayed_count = (
        torch.zeros_like(sample_last)
        if plan.delayed_crosstalk is not None
        else None
    )
    delayed_overflow = (
        torch.zeros_like(sample_last)
        if plan.delayed_crosstalk is not None
        else None
    )
    afterpulse_count = (
        torch.zeros_like(sample_last) if plan.afterpulse is not None else None
    )
    afterpulse_overflow = (
        torch.zeros_like(sample_last) if plan.afterpulse is not None else None
    )
    afterpulse_charge = (
        torch.zeros_like(sample_last, dtype=floating_dtype)
        if plan.afterpulse is not None
        else None
    )
    afterpulse_overflow_charge = (
        torch.zeros_like(sample_last, dtype=floating_dtype)
        if plan.afterpulse is not None
        else None
    )
    afterpulse_square_sum = (
        torch.zeros_like(sample_last, dtype=floating_dtype)
        if plan.afterpulse is not None
        else None
    )

    for generation_index in range(maximum_generations):
        children = torch.zeros_like(sample_last)

        if plan.direct_crosstalk is not None:
            direct_config = config.direct_crosstalk
            if type(direct_config) is not DirectCrosstalkConfig:
                raise RuntimeError("prepared direct-crosstalk config disappeared")
            new_count, new_overflow = _draw_crosstalk(
                frontier,
                positions=positions,
                plan=plan.direct_crosstalk,
                mean=direct_config.mean_offspring_per_parent.value,
                retained_stream=_RngStream.CHARGE_DIRECT_CROSSTALK,
                overflow_stream=_RngStream.CHARGE_DIRECT_CROSSTALK_OVERFLOW,
                generation_index=generation_index,
                tensor_numel=tensor_numel,
                seed=seed,
                field="direct crosstalk",
            )
            assert direct_count is not None and direct_overflow is not None
            direct_count = _checked_add(
                direct_count,
                new_count,
                field="direct-crosstalk cumulative count",
            )
            direct_overflow = _checked_add(
                direct_overflow,
                new_overflow,
                field="direct-crosstalk cumulative overflow",
            )
            direct_charge = new_count.to(floating_dtype)
            S1 = _checked_ledger_add(
                S1,
                direct_charge,
                bound=plan.ledger_bound,
                field="direct-crosstalk S1",
            )
            S2 = _checked_ledger_add(
                S2,
                direct_charge,
                bound=plan.ledger_bound,
                field="direct-crosstalk S2",
            )
            children = _checked_add(
                children,
                new_count,
                field="direct-crosstalk children",
            )

        if plan.delayed_crosstalk is not None:
            delayed_config = config.delayed_crosstalk
            if type(delayed_config) is not DelayedCrosstalkConfig:
                raise RuntimeError("prepared delayed-crosstalk config disappeared")
            new_count, new_overflow = _draw_crosstalk(
                frontier,
                positions=positions,
                plan=plan.delayed_crosstalk,
                mean=delayed_config.mean_offspring_per_parent.value,
                retained_stream=_RngStream.CHARGE_DELAYED_CROSSTALK,
                overflow_stream=_RngStream.CHARGE_DELAYED_CROSSTALK_OVERFLOW,
                generation_index=generation_index,
                tensor_numel=tensor_numel,
                seed=seed,
                field="delayed crosstalk",
            )
            assert delayed_count is not None and delayed_overflow is not None
            delayed_count = _checked_add(
                delayed_count,
                new_count,
                field="delayed-crosstalk cumulative count",
            )
            delayed_overflow = _checked_add(
                delayed_overflow,
                new_overflow,
                field="delayed-crosstalk cumulative overflow",
            )
            delayed_charge = new_count.to(floating_dtype)
            S1 = _checked_ledger_add(
                S1,
                delayed_charge,
                bound=plan.ledger_bound,
                field="delayed-crosstalk S1",
            )
            S2 = _checked_ledger_add(
                S2,
                delayed_charge,
                bound=plan.ledger_bound,
                field="delayed-crosstalk S2",
            )
            children = _checked_add(
                children,
                new_count,
                field="delayed-crosstalk children",
            )

        if plan.afterpulse is not None:
            afterpulse_config = config.afterpulse
            if type(afterpulse_config) is not AfterpulseConfig:
                raise RuntimeError("prepared afterpulse config disappeared")
            (
                new_count,
                new_overflow,
                new_charge,
                new_overflow_charge,
                new_square_sum,
            ) = _draw_afterpulses(
                frontier,
                positions=positions,
                plan=plan.afterpulse,
                config=afterpulse_config,
                generation_index=generation_index,
                tensor_numel=tensor_numel,
                floating_dtype=floating_dtype,
                seed=seed,
            )
            assert (
                afterpulse_count is not None
                and afterpulse_overflow is not None
                and afterpulse_charge is not None
                and afterpulse_overflow_charge is not None
                and afterpulse_square_sum is not None
            )
            afterpulse_count = _checked_add(
                afterpulse_count,
                new_count,
                field="afterpulse cumulative count",
            )
            afterpulse_overflow = _checked_add(
                afterpulse_overflow,
                new_overflow,
                field="afterpulse cumulative overflow",
            )
            afterpulse_charge = _checked_ledger_add(
                afterpulse_charge,
                new_charge,
                bound=plan.ledger_bound,
                field="afterpulse cumulative charge",
            )
            afterpulse_overflow_charge = _checked_ledger_add(
                afterpulse_overflow_charge,
                new_overflow_charge,
                bound=plan.ledger_bound,
                field="afterpulse cumulative overflow charge",
            )
            afterpulse_square_sum = _checked_ledger_add(
                afterpulse_square_sum,
                new_square_sum,
                bound=plan.ledger_bound,
                field="afterpulse cumulative charge-square sum",
            )
            S1 = _checked_ledger_add(
                S1,
                new_charge,
                bound=plan.ledger_bound,
                field="afterpulse S1",
            )
            S2 = _checked_ledger_add(
                S2,
                new_square_sum,
                bound=plan.ledger_bound,
                field="afterpulse S2",
            )
            children = _checked_add(
                children,
                new_count,
                field="afterpulse children",
            )

        frontier = children
        total_count = _checked_add(
            total_count,
            frontier,
            field="correlated-avalanche total count",
        )

    reconstructed_count = sample_last.clone()
    for field, contribution in (
        ("direct crosstalk", direct_count),
        ("delayed crosstalk", delayed_count),
        ("afterpulse", afterpulse_count),
    ):
        if contribution is not None:
            reconstructed_count = _checked_add(
                reconstructed_count,
                contribution,
                field=f"correlated-avalanche reconstructed {field} count",
            )
    if not torch.equal(reconstructed_count, total_count):
        raise RuntimeError("correlated-avalanche integer count identity failed")

    precision = 24 if floating_dtype is torch.float32 else 53
    gamma = plan.ledger_depth / ((1 << precision) - plan.ledger_depth)
    zero = torch.tensor(0.0, dtype=floating_dtype, device="cpu")
    one = torch.tensor(1.0, dtype=floating_dtype, device="cpu")
    subnormal = float(torch.nextafter(zero, one))
    total_reference = total_count.to(torch.float64)
    tolerance = total_reference * gamma + plan.ledger_depth * subnormal
    if not bool(
        torch.all(
            (S2.to(torch.float64) <= S1.to(torch.float64) + 2.0 * tolerance)
            & (S1.to(torch.float64) <= total_reference + tolerance)
        ).item()
    ):
        raise RuntimeError("correlated-avalanche ledgers violate their ordering")
    restore = lambda value: value.movedim(-1, sample_dimension)
    return _CorrelatedAvalancheResult(
        restore(S1),
        restore(S2),
        restore(frontier),
        restore(total_count),
        None if direct_count is None else restore(direct_count),
        None if direct_overflow is None else restore(direct_overflow),
        None if delayed_count is None else restore(delayed_count),
        None if delayed_overflow is None else restore(delayed_overflow),
        None if afterpulse_count is None else restore(afterpulse_count),
        None if afterpulse_overflow is None else restore(afterpulse_overflow),
        None if afterpulse_charge is None else restore(afterpulse_charge),
        None
        if afterpulse_overflow_charge is None
        else restore(afterpulse_overflow_charge),
        None if afterpulse_square_sum is None else restore(afterpulse_square_sum),
    )


def _simulate_charge_smearing(
    charge_pe: torch.Tensor,
    charge_square_sum: torch.Tensor,
    *,
    config: ChargeSmearingConfig,
    seed: int | None,
) -> torch.Tensor:
    if type(config) is not ChargeSmearingConfig:
        raise TypeError("config must be exactly ChargeSmearingConfig")
    if charge_pe.dtype not in (torch.float32, torch.float64):
        raise TypeError("charge_pe must use a supported floating dtype")
    if charge_square_sum.dtype is not charge_pe.dtype:
        raise ValueError("charge ledgers must have the same dtype")
    if charge_square_sum.shape != charge_pe.shape:
        raise ValueError("charge ledgers must have the same shape")
    if charge_square_sum.device != charge_pe.device:
        raise ValueError("charge ledgers must be on the same device")
    for field, value in (("S1", charge_pe), ("S2", charge_square_sum)):
        if not bool(torch.all(torch.isfinite(value) & (value >= 0.0)).item()):
            raise ValueError(f"charge-smearing {field} must be finite and nonnegative")
    if config.relative_sigma.value == 0.0:
        return charge_pe
    if seed is None:
        raise ValueError("effective charge smearing requires a seed")
    represented_sigma = float(
        torch.tensor(
            config.relative_sigma.value,
            dtype=charge_pe.dtype,
            device="cpu",
        )
    )
    if not math.isfinite(represented_sigma) or represented_sigma <= 0.0:
        raise ValueError("charge-smearing width is invalid in the Charge dtype")
    positions = _logical_positions(tuple(charge_pe.shape), device=charge_pe.device)
    standard, _ = _standard_normal_pair(
        seed=seed,
        stream=_RngStream.CHARGE_SMEARING,
        logical_positions=positions,
        dtype=charge_pe.dtype,
    )
    sigma = torch.tensor(
        represented_sigma,
        dtype=charge_pe.dtype,
        device=charge_pe.device,
    )
    zero = torch.tensor(0.0, dtype=charge_pe.dtype, device=charge_pe.device)
    with torch.autocast(device_type=charge_pe.device.type, enabled=False):
        scale = sigma * torch.sqrt(charge_square_sum)
        excursion = scale * standard
        draw = charge_pe + excursion
        result = torch.maximum(draw, zero)
    for field, value in (
        ("scale", scale),
        ("excursion", excursion),
        ("draw", draw),
        ("result", result),
    ):
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise RuntimeError(f"charge-smearing {field} is nonfinite")
    return result


def _produce_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    seed: int | None,
    floating_dtype: torch.dtype,
) -> Charge:
    if type(photoelectrons) is not Photoelectrons:
        raise TypeError("photoelectrons must be exactly Photoelectrons")
    if type(config) is not ChargeConfig:
        raise TypeError("config must be exactly ChargeConfig")
    _require_sampling(photoelectrons, sampling)
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")
    device = photoelectrons.tensor.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Charge production supports only CPU and CUDA")
    if seed is not None:
        _require_seed(seed)

    source = photoelectrons.tensor
    _require_count_domain(source, field="Photoelectrons source")
    shape = tuple(source.shape)
    tensor_numel = _require_tensor_allocation(
        shape,
        element_size=source.element_size(),
        field="Charge source",
    )
    _require_tensor_allocation(
        shape,
        element_size=torch.empty((), dtype=floating_dtype).element_size(),
        field="Charge output",
    )
    sample_dimension = photoelectrons.dimension_of(SampleAxis)

    dark_mean = 0.0
    if config.dark_count is not None:
        dark_mean = _prepare_dark_mean(config.dark_count, sampling=sampling)

    jitter_plan: _TimingJitterPlan | None = None
    if config.timing_jitter is not None and config.timing_jitter.sigma_ns.value != 0.0:
        jitter_plan = _prepare_timing_jitter(
            config.timing_jitter,
            sampling=sampling,
            tensor_numel=tensor_numel,
        )

    if config.correlated_avalanches is None:
        _, ledger_bound = _ledger_envelope(
            floating_dtype=floating_dtype,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=sampling.sample_count.value,
        )
        correlated_plan: _CorrelatedAvalanchePlan | None = None
    else:
        correlated_plan = _prepare_correlated_plan(
            config.correlated_avalanches,
            sampling=sampling,
            floating_dtype=floating_dtype,
            tensor_numel=tensor_numel,
        )
        ledger_bound = correlated_plan.ledger_bound

    smearing_sigma = 0.0
    if config.smearing is not None and config.smearing.relative_sigma.value != 0.0:
        smearing_sigma = _prepare_smearing_sigma(
            config.smearing,
            floating_dtype=floating_dtype,
            ledger_bound=ledger_bound,
        )

    source_nonzero = bool(torch.any(source != 0).item())
    correlated_can_draw = (
        correlated_plan is not None
        and (
            correlated_plan.direct_crosstalk is not None
            or correlated_plan.delayed_crosstalk is not None
            or correlated_plan.afterpulse is not None
        )
        and (source_nonzero or dark_mean > 0.0)
    )
    jitter_can_draw = jitter_plan is not None and (source_nonzero or dark_mean > 0.0)
    if (
        dark_mean > 0.0
        or jitter_can_draw
        or correlated_can_draw
        or smearing_sigma > 0.0
    ) and seed is None:
        raise ValueError("this Charge configuration requires a seed")

    charge = source
    charge_square_sum: torch.Tensor | None = None
    if config.dark_count is not None and dark_mean != 0.0:
        charge = _simulate_dark_counts(
            charge,
            sampling=sampling,
            config=config.dark_count,
            seed=seed,
        )
    if config.timing_jitter is not None and jitter_plan is not None:
        charge = _simulate_timing_jitter(
            charge,
            sample_dimension=sample_dimension,
            sampling=sampling,
            config=config.timing_jitter,
            seed=seed,
        )
    if config.correlated_avalanches is not None:
        correlated = _simulate_correlated_avalanches(
            charge,
            sample_dimension=sample_dimension,
            sampling=sampling,
            floating_dtype=floating_dtype,
            config=config.correlated_avalanches,
            seed=seed,
        )
        charge = correlated.S1
        charge_square_sum = correlated.S2
    if config.smearing is not None and smearing_sigma != 0.0:
        charge = charge.to(dtype=floating_dtype)
        charge = _simulate_charge_smearing(
            charge,
            charge if charge_square_sum is None else charge_square_sum,
            config=config.smearing,
            seed=seed,
        )

    values = charge.to(dtype=floating_dtype)
    if not bool(torch.all(torch.isfinite(values) & (values >= 0.0)).item()):
        raise RuntimeError("Charge production produced an invalid terminal value")
    if values.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
        raise RuntimeError("Charge output must have fresh storage")
    result = Charge(tensor=values, axes=photoelectrons.axes)
    _require_valid_values(result)
    return result
