from __future__ import annotations

import math

import torch
from tensor_core import CounterRng, RngKey, RngPositions
from tensor_core.validation.random import require_count_tensor


MAX_COUNT = (1 << 53) - 1
MAX_POISSON_MEAN = 1.0e8


def checked_add(
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
    if bool(torch.any(right > (MAX_COUNT - left)).item()):
        raise RuntimeError(f"{field} exceeds the Charge count ceiling")
    return left + right


def checked_subtract(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    field: str,
) -> torch.Tensor:
    if left.dtype is not torch.int64 or right.dtype is not torch.int64:
        raise TypeError(f"{field} count subtractions require torch.int64")
    if left.shape != right.shape or left.device != right.device:
        raise ValueError(f"{field} count subtractions require equal representations")
    if bool(torch.any(right < 0).item()):
        raise RuntimeError(f"{field} received a negative count contribution")
    if bool(torch.any(right > left).item()):
        raise RuntimeError(f"{field} category exceeds its remainder")
    return left - right


def original_positions(
    shape: tuple[int, ...],
    *,
    sample_dimension: int,
    device: torch.device,
) -> RngPositions:
    return RngPositions.from_shape(shape, device=device).movedim(
        sample_dimension,
        -1,
    )


def checked_rate_product(
    basis: torch.Tensor,
    mean: float,
    *,
    field: str,
) -> torch.Tensor:
    if basis.dtype is not torch.float64:
        raise TypeError(f"{field} basis must have dtype torch.float64")
    if not bool(torch.all(torch.isfinite(basis) & (basis >= 0.0)).item()):
        raise RuntimeError(f"{field} basis is invalid")
    threshold = MAX_POISSON_MEAN / mean
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
            & (rate <= MAX_POISSON_MEAN)
        ).item()
    ):
        raise RuntimeError(f"{field} rate exceeds the Poisson domain")
    return rate


def draw_ordered_categories(
    counts: torch.Tensor,
    *,
    success_masses: tuple[float | torch.Tensor, ...],
    failure_masses: tuple[float | torch.Tensor, ...],
    positions: tuple[RngPositions, ...],
    rng: CounterRng,
    key: RngKey,
    field: str,
) -> tuple[torch.Tensor, ...]:
    if not (
        len(success_masses) == len(failure_masses) == len(positions)
    ):
        raise ValueError(f"{field} category plans must have equal lengths")
    require_count_tensor(counts, f"{field} input")
    remaining = counts.clone()
    categories: list[torch.Tensor] = []
    for success_mass, failure_mass, category_positions in zip(
        success_masses,
        failure_masses,
        positions,
        strict=True,
    ):
        category = rng.binomial(
            counts=remaining,
            success_mass=success_mass,
            failure_mass=failure_mass,
            key=key,
            positions=category_positions,
            quantum=0,
        )
        remaining = checked_subtract(
            remaining,
            category,
            field=field,
        )
        categories.append(category)
    return (*categories, remaining)
