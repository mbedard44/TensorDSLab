"""Private timing-jitter law preparation and execution."""

import math
from dataclasses import dataclass
from typing import final

import torch
from tensor_core import (
    CounterRng,
    MultinomialDistribution,
    ProbabilityKernel,
    RngElements,
    RngKey,
)
from tensor_core.random.validation import require_count_tensor

from tensor_dslab.common.axes import SampleAxis
from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.charge.config import TimingJitterConfig
from tensor_dslab.readout.charge.runtime.effects.counts import (
    checked_add,
)
from tensor_dslab.readout.runtime.addresses import timing_jitter_address
from tensor_dslab.readout.runtime.keys import TIMING_JITTER_RNG_KEY
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


_MAX_SAMPLE_COUNT = 8192
_LOCAL_PROBABILITY_TOLERANCE = 1.0e-12
_COMPLETE_LAW_TOLERANCE = 1.0e-11


@final
class TimingJitterProbabilityKernel(
    ProbabilityKernel[tuple[type[SampleAxis]]]
):
    __slots__ = ()

    def _require(self) -> None:
        if self.axis_types != (SampleAxis,):
            raise ValueError("timing-jitter kernel must use the SampleAxis role")
        if len(self.shape) != 1 or self.shape[0] < 3 or self.shape[0] % 2 == 0:
            raise ValueError(
                "timing-jitter kernel must have one odd displacement dimension"
            )


@final
@dataclass(frozen=True, slots=True)
class TimingJitterRuntime:
    kernel: TimingJitterProbabilityKernel
    completion_probability: float
    rng_key: RngKey


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


def prepare_timing_jitter(
    config: TimingJitterConfig,
    *,
    sampling: SamplingRuntime,
    tensor_numel: int,
    device: torch.device,
) -> TimingJitterRuntime | None:
    sigma_ns = canonical_magnitude(config.sigma)
    if sigma_ns == 0.0:
        return None
    sample_count = sampling.sample_count
    if sample_count > _MAX_SAMPLE_COUNT:
        raise ValueError("active timing jitter supports at most 8192 samples")
    if (2 * sample_count - 1) * tensor_numel > 1 << 63:
        raise ValueError("timing-jitter address lattice exceeds its domain")
    period_ps = float(sampling.sample_period_ps)
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
    represented = (
        *reversed(probabilities[1:]),
        probabilities[0],
        *probabilities[1:],
    )
    return TimingJitterRuntime(
        kernel=TimingJitterProbabilityKernel(
            tensor=torch.tensor(
                represented,
                dtype=torch.float64,
                device=device,
            ),
            axis_types=(SampleAxis,),
        ),
        completion_probability=2.0 * left_tails[-1],
        rng_key=TIMING_JITTER_RNG_KEY,
    )


def simulate_timing_jitter(
    counts: torch.Tensor,
    *,
    sample_dimension: int,
    runtime: TimingJitterRuntime,
    rng: CounterRng,
    elements: RngElements,
) -> torch.Tensor:
    require_count_tensor(counts, "timing-jitter input")
    if sample_dimension < 0 or sample_dimension >= counts.ndim:
        raise ValueError("sample_dimension is outside the count rank")
    sample_count = (runtime.kernel.shape[0] + 1) // 2
    if counts.shape[sample_dimension] != sample_count:
        raise ValueError("sample dimension disagrees with the prepared runtime")
    if not bool(torch.any(counts != 0).item()):
        return counts.clone()

    sample_last = counts.movedim(sample_dimension, -1)
    result = torch.zeros_like(sample_last)
    sample_last_elements = elements.movedim(sample_dimension, -1)
    displacement_origin = sample_count - 1
    for source in range(sample_count):
        source_counts = sample_last[..., source]
        source_elements = sample_last_elements.select(-1, source)
        allocation = MultinomialDistribution(
            counts=source_counts,
            kernel=runtime.kernel,
            completion_probability=runtime.completion_probability,
        ).draw(
            rng=rng,
            address=timing_jitter_address(
                source_elements,
                key=runtime.rng_key,
                kernel_shape=runtime.kernel.shape,
            ),
        )
        for kernel_index in range(runtime.kernel.shape[0]):
            destination_index = source + kernel_index - displacement_origin
            if 0 <= destination_index < sample_count:
                result[..., destination_index] = checked_add(
                    result[..., destination_index],
                    allocation[kernel_index],
                    field="timing-jitter destination",
                )
    return result.movedim(-1, sample_dimension)
