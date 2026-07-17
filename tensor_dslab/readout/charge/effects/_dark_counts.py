from __future__ import annotations

import math
from fractions import Fraction

import torch
from tensor_core import CounterRng, logical_positions

from tensor_dslab.common import SamplingConfig
from tensor_dslab.readout.charge.config import DarkCountConfig
from tensor_dslab.readout.charge.effects._counts import (
    _MAX_POISSON_MEAN,
    _checked_add,
    _require_count_domain,
)


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


def _simulate_dark_counts(
    counts: torch.Tensor,
    *,
    sampling: SamplingConfig,
    config: DarkCountConfig,
    rng: CounterRng,
) -> torch.Tensor:
    if type(config) is not DarkCountConfig:
        raise TypeError("config must be exactly DarkCountConfig")
    _require_count_domain(counts, field="dark-count input")
    mean = _prepare_dark_mean(config, sampling=sampling)
    if mean == 0.0:
        return counts
    positions = logical_positions(tuple(counts.shape), device=counts.device)
    sampled = rng.poisson(
        mean=mean,
        key=config.rng_key,
        positions=positions,
        quantum=0,
    )
    return _checked_add(counts, sampled, field="dark-count result")
