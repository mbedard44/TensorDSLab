"""Private charge count-domain arithmetic."""

import torch


MAX_COUNT = (1 << 53) - 1


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
