from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import final

import torch
from tensor_core import CounterRng, RngKey, RngPositions
from tensor_core.validation.random import require_count_tensor

from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.runtime.sampling import SamplingRuntime
from tensor_dslab.readout.charge.config import DarkCountConfig
from tensor_dslab.readout.charge.runtime.effects.counts import (
    MAX_POISSON_MEAN,
    checked_add,
)
from tensor_dslab.readout.runtime.keys import DARK_COUNT_RNG_KEY


@final
@dataclass(frozen=True, slots=True)
class DarkCountRuntime:
    mean: float
    rng_key: RngKey


def _prepare_dark_mean(
    rate_hz: float,
    *,
    sampling: SamplingRuntime,
) -> float:
    if rate_hz == 0.0:
        return 0.0
    numerator, denominator = rate_hz.as_integer_ratio()
    if numerator * sampling.sample_period_ps > denominator * 10**20:
        raise ValueError("dark-count mean exceeds the accepted Poisson domain")
    mean = float(
        Fraction(
            numerator * sampling.sample_period_ps,
            denominator * 10**12,
        )
    )
    if not math.isfinite(mean) or not 0.0 < mean <= MAX_POISSON_MEAN:
        raise ValueError("dark-count mean is outside the accepted Poisson domain")
    return mean


def prepare_dark_counts(
    config: DarkCountConfig,
    *,
    sampling: SamplingRuntime,
) -> DarkCountRuntime:
    return DarkCountRuntime(
        mean=_prepare_dark_mean(
            canonical_magnitude(config.rate),
            sampling=sampling,
        ),
        rng_key=DARK_COUNT_RNG_KEY,
    )


def simulate_dark_counts(
    counts: torch.Tensor,
    *,
    runtime: DarkCountRuntime,
    rng: CounterRng,
) -> torch.Tensor:
    require_count_tensor(counts, "dark-count input")
    if runtime.mean == 0.0:
        return counts
    positions = RngPositions.from_shape(tuple(counts.shape), device=counts.device)
    sampled = rng.poisson(
        mean=runtime.mean,
        key=runtime.rng_key,
        positions=positions,
        quantum=0,
    )
    return checked_add(counts, sampled, field="dark-count result")
