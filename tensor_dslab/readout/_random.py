from __future__ import annotations

import math
from enum import Enum, unique

import torch


_UINT32_MASK = (1 << 32) - 1
_THREEFRY_PARITY = 0x1BD11BDA
_DOMAIN_TAG = 0x54445331
_ROTATIONS = (
    (10, 26),
    (11, 21),
    (13, 27),
    (23, 5),
    (6, 20),
    (17, 11),
    (25, 10),
    (18, 20),
)


@unique
class _RngStream(Enum):
    NOISE_WHITE = 0x0000_0001
    NOISE_PSD_COEFFICIENT = 0x0000_0002


def _require_bounded_integer(
    value: object,
    *,
    field: str,
    upper_exclusive: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be exactly an integer")
    if value < 0 or value >= upper_exclusive:
        raise ValueError(f"{field} is outside its accepted range")
    return value


def _require_seed(seed: object) -> int:
    return _require_bounded_integer(
        seed,
        field="seed",
        upper_exclusive=1 << 64,
    )


def _address_words(
    *,
    seed: int,
    stream: int,
    logical_position: int,
    source_quantum: int,
    raw_word_ordinal: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    checked_seed = _require_seed(seed)
    checked_stream = _require_bounded_integer(
        stream,
        field="stream",
        upper_exclusive=1 << 32,
    )
    checked_position = _require_bounded_integer(
        logical_position,
        field="logical_position",
        upper_exclusive=1 << 63,
    )
    checked_quantum = _require_bounded_integer(
        source_quantum,
        field="source_quantum",
        upper_exclusive=1 << 32,
    )
    checked_ordinal = _require_bounded_integer(
        raw_word_ordinal,
        field="raw_word_ordinal",
        upper_exclusive=1 << 34,
    )
    block, lane = divmod(checked_ordinal, 4)
    key = (
        checked_seed & _UINT32_MASK,
        (checked_seed >> 32) & _UINT32_MASK,
        checked_stream,
        _DOMAIN_TAG,
    )
    counter = (
        checked_position & _UINT32_MASK,
        (checked_position >> 32) & _UINT32_MASK,
        checked_quantum,
        block,
    )
    return key, counter, lane


def _rotate_left_32(value: torch.Tensor, distance: int) -> torch.Tensor:
    return (
        ((value << distance) & _UINT32_MASK)
        | (value >> (32 - distance))
    ) & _UINT32_MASK


def _mix(
    left: torch.Tensor,
    right: torch.Tensor,
    rotation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixed_left = (left + right) & _UINT32_MASK
    mixed_right = _rotate_left_32(right, rotation) ^ mixed_left
    return mixed_left, mixed_right


def _threefry4x32(
    counter: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    if counter.dtype is not torch.int64 or counter.shape[-1:] != (4,):
        raise TypeError("counter must be an int64 tensor with final size four")
    if key.dtype is not torch.int64 or key.shape[-1:] != (4,):
        raise TypeError("key must be an int64 tensor with final size four")
    if key.device != counter.device:
        raise ValueError("key and counter must be on the same device")

    key_0, key_1, key_2, key_3 = key.unbind(dim=-1)
    key_4 = (
        key_0 ^ key_1 ^ key_2 ^ key_3 ^ _THREEFRY_PARITY
    ) & _UINT32_MASK
    schedule = (key_0, key_1, key_2, key_3, key_4)

    value_0, value_1, value_2, value_3 = counter.unbind(dim=-1)
    value_0 = (value_0 + key_0) & _UINT32_MASK
    value_1 = (value_1 + key_1) & _UINT32_MASK
    value_2 = (value_2 + key_2) & _UINT32_MASK
    value_3 = (value_3 + key_3) & _UINT32_MASK

    for round_index in range(20):
        rotation_0, rotation_1 = _ROTATIONS[round_index % len(_ROTATIONS)]
        if round_index % 2 == 0:
            value_0, value_1 = _mix(value_0, value_1, rotation_0)
            value_2, value_3 = _mix(value_2, value_3, rotation_1)
        else:
            value_0, value_3 = _mix(value_0, value_3, rotation_0)
            value_2, value_1 = _mix(value_2, value_1, rotation_1)

        if (round_index + 1) % 4 == 0:
            injection = (round_index + 1) // 4
            value_0 = (value_0 + schedule[injection % 5]) & _UINT32_MASK
            value_1 = (value_1 + schedule[(injection + 1) % 5]) & _UINT32_MASK
            value_2 = (value_2 + schedule[(injection + 2) % 5]) & _UINT32_MASK
            value_3 = (
                value_3 + schedule[(injection + 3) % 5] + injection
            ) & _UINT32_MASK

    return torch.stack((value_0, value_1, value_2, value_3), dim=-1)


def _require_stream(stream: object) -> _RngStream:
    if type(stream) is not _RngStream:
        raise TypeError("stream must be exactly _RngStream")
    return stream


def _random_block(
    *,
    seed: int,
    stream: _RngStream,
    logical_positions: torch.Tensor,
    source_quantum: int = 0,
    block: int = 0,
) -> torch.Tensor:
    checked_seed = _require_seed(seed)
    checked_stream = _require_stream(stream)
    checked_quantum = _require_bounded_integer(
        source_quantum,
        field="source_quantum",
        upper_exclusive=1 << 32,
    )
    checked_block = _require_bounded_integer(
        block,
        field="block",
        upper_exclusive=1 << 32,
    )
    if logical_positions.dtype is not torch.int64:
        raise TypeError("logical_positions must have dtype torch.int64")

    key = torch.tensor(
        (
            checked_seed & _UINT32_MASK,
            (checked_seed >> 32) & _UINT32_MASK,
            checked_stream.value,
            _DOMAIN_TAG,
        ),
        dtype=torch.int64,
        device=logical_positions.device,
    )
    counter = torch.stack(
        (
            logical_positions & _UINT32_MASK,
            logical_positions >> 32,
            torch.full_like(logical_positions, checked_quantum),
            torch.full_like(logical_positions, checked_block),
        ),
        dim=-1,
    )
    return _threefry4x32(counter, key)


def _raw_word(
    *,
    seed: int,
    stream: _RngStream,
    logical_position: int,
    source_quantum: int = 0,
    raw_word_ordinal: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    key, counter, lane = _address_words(
        seed=seed,
        stream=_require_stream(stream).value,
        logical_position=logical_position,
        source_quantum=source_quantum,
        raw_word_ordinal=raw_word_ordinal,
    )
    key_tensor = torch.tensor(key, dtype=torch.int64, device=device)
    counter_tensor = torch.tensor(counter, dtype=torch.int64, device=device)
    return _threefry4x32(counter_tensor, key_tensor)[lane]


def _logical_positions(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    if type(shape) is not tuple:
        raise TypeError("shape must be exactly a tuple")
    count = 1
    for size in shape:
        if type(size) is not int:
            raise TypeError("every shape size must be exactly an integer")
        if size < 0:
            raise ValueError("shape sizes must be nonnegative")
        count *= size
    if count > 1 << 63:
        raise ValueError("logical position count exceeds the accepted range")
    return torch.arange(count, dtype=torch.int64, device=device).reshape(shape)


def _uniform_closed_open(
    word_0: torch.Tensor,
    *,
    dtype: torch.dtype,
    word_1: torch.Tensor | None = None,
) -> torch.Tensor:
    if word_0.dtype is not torch.int64:
        raise TypeError("uniform words must have dtype torch.int64")
    if dtype is torch.float32:
        if word_1 is not None:
            raise TypeError("float32 uniform conversion consumes one word")
        mantissa = word_0 >> 8
        scale = torch.tensor(2.0**-24, dtype=dtype, device=word_0.device)
    elif dtype is torch.float64:
        if word_1 is None or word_1.dtype is not torch.int64:
            raise TypeError("float64 uniform conversion consumes two int64 words")
        if word_1.device != word_0.device:
            raise ValueError("uniform words must be on the same device")
        mantissa = word_0 * (1 << 21) + (word_1 >> 11)
        scale = torch.tensor(2.0**-53, dtype=dtype, device=word_0.device)
    else:
        raise TypeError("uniform dtype must be torch.float32 or torch.float64")
    with torch.autocast(device_type=word_0.device.type, enabled=False):
        return mantissa.to(dtype=dtype) * scale


def _uniform_open_open(
    word_0: torch.Tensor,
    *,
    dtype: torch.dtype,
    word_1: torch.Tensor | None = None,
) -> torch.Tensor:
    if word_0.dtype is not torch.int64:
        raise TypeError("uniform words must have dtype torch.int64")
    if dtype is torch.float32:
        if word_1 is not None:
            raise TypeError("float32 uniform conversion consumes one word")
        mantissa = word_0 >> 9
        scale = torch.tensor(2.0**-23, dtype=dtype, device=word_0.device)
    elif dtype is torch.float64:
        if word_1 is None or word_1.dtype is not torch.int64:
            raise TypeError("float64 uniform conversion consumes two int64 words")
        if word_1.device != word_0.device:
            raise ValueError("uniform words must be on the same device")
        mantissa = word_0 * (1 << 20) + (word_1 >> 12)
        scale = torch.tensor(2.0**-52, dtype=dtype, device=word_0.device)
    else:
        raise TypeError("uniform dtype must be torch.float32 or torch.float64")
    half = torch.tensor(0.5, dtype=dtype, device=word_0.device)
    with torch.autocast(device_type=word_0.device.type, enabled=False):
        return (half + mantissa.to(dtype=dtype)) * scale


def _standard_normal_pair(
    *,
    seed: int,
    stream: _RngStream,
    logical_positions: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    words = _random_block(
        seed=seed,
        stream=stream,
        logical_positions=logical_positions,
    )
    if dtype is torch.float32:
        radius_uniform = _uniform_open_open(words[..., 0], dtype=dtype)
        angle_uniform = _uniform_closed_open(words[..., 1], dtype=dtype)
    elif dtype is torch.float64:
        radius_uniform = _uniform_open_open(
            words[..., 0],
            word_1=words[..., 1],
            dtype=dtype,
        )
        angle_uniform = _uniform_closed_open(
            words[..., 2],
            word_1=words[..., 3],
            dtype=dtype,
        )
    else:
        raise TypeError("normal dtype must be torch.float32 or torch.float64")

    tau = torch.tensor(math.tau, dtype=dtype, device=words.device)
    minus_two = torch.tensor(-2.0, dtype=dtype, device=words.device)
    with torch.autocast(device_type=words.device.type, enabled=False):
        radius = torch.sqrt(torch.log(radius_uniform) * minus_two)
        angle = tau * angle_uniform
        return radius * torch.cos(angle), radius * torch.sin(angle)
