"""Private phase-marginalized delay-law preparation."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import final

from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.runtime.sampling import SamplingRuntime
from tensor_dslab.readout.charge.config import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    ExponentialDelayConfig,
    FixedDelayConfig,
)


_MAX_SAMPLE_COUNT = 8192
_LOCAL_PROBABILITY_TOLERANCE = 1.0e-12
_COMPLETE_LAW_TOLERANCE = 1.0e-11


@final
@dataclass(frozen=True, slots=True)
class DelayRuntime:
    probabilities: tuple[float, ...]
    right_tails: tuple[float, ...]


@final
@dataclass(frozen=True, slots=True)
class AfterpulseRuntime:
    delay: DelayRuntime
    recovery: tuple[float, ...] | None
    overflow_recovery: tuple[float, ...] | None


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
) -> DelayRuntime:
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
    return DelayRuntime(tuple(probabilities), tuple(tails))


def prepare_exponential_delay(
    mean_delay_ns: float,
    *,
    sampling: SamplingRuntime,
) -> DelayRuntime:
    sample_count = sampling.sample_count
    if sample_count > _MAX_SAMPLE_COUNT:
        raise ValueError("active exponential delay supports at most 8192 samples")
    mean_numerator, mean_denominator = mean_delay_ns.as_integer_ratio()
    period = sampling.sample_period_ps
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
    sampling: SamplingRuntime,
) -> DelayRuntime:
    sample_count = sampling.sample_count
    numerator, denominator = delay_ns.as_integer_ratio()
    period = sampling.sample_period_ps
    if numerator * 1000 >= denominator * (
        sampling.sample_count * sampling.sample_period_ps
    ):
        return DelayRuntime(
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
    return DelayRuntime(tuple(probabilities), tuple(tails))


def prepare_delay(
    config: FixedDelayConfig | ExponentialDelayConfig,
    *,
    sampling: SamplingRuntime,
) -> DelayRuntime:
    if type(config) is FixedDelayConfig:
        return _prepare_fixed_delay(
            canonical_magnitude(config.delay),
            sampling=sampling,
        )
    if type(config) is ExponentialDelayConfig:
        return prepare_exponential_delay(
            canonical_magnitude(config.mean_delay),
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


def prepare_afterpulse_recovery(
    mean_delay_ns: float,
    time_constant_ns: float,
    *,
    sampling: SamplingRuntime,
    delay: DelayRuntime,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    period = sampling.sample_period_ps
    numerator, denominator = time_constant_ns.as_integer_ratio()
    if numerator * 1000 * (1 << 52) < denominator * period:
        raise ValueError("afterpulse recovery ratio is below the accepted domain")
    if numerator * 1000 > denominator * period * (1 << 52):
        raise ValueError("afterpulse recovery ratio exceeds the accepted domain")

    delay_ps = mean_delay_ns * 1000.0
    recovery_ps = time_constant_ns * 1000.0
    x = float(period) / delay_ps
    y = float(period) / recovery_ps
    combined = x + y
    if not all(math.isfinite(value) and value > 0.0 for value in (x, y, combined)):
        raise ValueError("afterpulse recovery inverse ratios must be finite")
    if not 2.0**-51 <= combined <= 2.0**53:
        raise ValueError("effective recovery inverse ratio is outside its domain")

    effective = _prepare_exponential_from_inverse(
        combined,
        sample_count=sampling.sample_count,
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
    for first_outside in range(1, sampling.sample_count + 1):
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
