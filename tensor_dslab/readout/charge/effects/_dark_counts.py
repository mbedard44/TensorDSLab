from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from tensor_core import CounterRng, RngKey, logical_positions

from tensor_dslab.common import SamplingConfig
from tensor_dslab.readout.charge.config import DarkCountConfig
from tensor_dslab.readout.charge.effects._counts import (
    _MAX_POISSON_MEAN,
    _checked_add,
    _require_count_domain,
)


@dataclass(frozen=True, slots=True)
class _DarkCountPlan:
    mean: float
    rng_key: RngKey


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


def _prepare_dark_counts(
    config: DarkCountConfig,
    *,
    sampling: SamplingConfig,
) -> _DarkCountPlan:
    return _DarkCountPlan(
        mean=_prepare_dark_mean(config, sampling=sampling),
        rng_key=config.rng_key,
    )


def _simulate_dark_counts(
    counts: torch.Tensor,
    *,
    plan: _DarkCountPlan,
    rng: CounterRng,
) -> torch.Tensor:
    if type(plan) is not _DarkCountPlan:
        raise TypeError("plan must be exactly _DarkCountPlan")
    _require_count_domain(counts, field="dark-count input")
    if plan.mean == 0.0:
        return counts
    positions = logical_positions(tuple(counts.shape), device=counts.device)
    sampled = rng.poisson(
        mean=plan.mean,
        key=plan.rng_key,
        positions=positions,
        quantum=0,
    )
    return _checked_add(counts, sampled, field="dark-count result")
