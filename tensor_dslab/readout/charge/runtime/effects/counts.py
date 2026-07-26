"""Private charge count-domain arithmetic."""

import math

import torch


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
