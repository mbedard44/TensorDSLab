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
_MAX_CHARGE_COUNT = (1 << 53) - 1
_MAX_POISSON_MEAN = 1.0e8
_STIRLING_CORRECTIONS = (
    0.0810614667953272,
    0.0413406959554092,
    0.0276779256849983,
    0.02079067210376509,
    0.0166446911898211,
    0.0138761288230707,
    0.0118967099458917,
    0.0104112652619720,
    0.00925546218271273,
    0.00833056343336287,
)


@unique
class _RngStream(Enum):
    NOISE_WHITE = 0x0000_0001
    NOISE_PSD_COEFFICIENT = 0x0000_0002
    CHARGE_DARK_COUNTS = 0x0000_0003
    CHARGE_DIRECT_CROSSTALK = 0x0000_0004
    CHARGE_DIRECT_CROSSTALK_OVERFLOW = 0x0000_0005
    CHARGE_DELAYED_CROSSTALK = 0x0000_0006
    CHARGE_DELAYED_CROSSTALK_OVERFLOW = 0x0000_0007
    CHARGE_TIMING_JITTER = 0x0000_0008
    CHARGE_AFTERPULSES = 0x0000_0009
    CHARGE_SMEARING = 0x0000_000A


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


def _require_count_shape(shape: object) -> tuple[int, ...]:
    if type(shape) is not tuple:
        raise TypeError("shape must be exactly a tuple")
    checked: list[int] = []
    count = 1
    for size in shape:
        if type(size) is not int:
            raise TypeError("every shape size must be exactly an integer")
        if size < 0:
            raise ValueError("shape sizes must be nonnegative")
        count *= size
        if count > (1 << 63) - 1:
            raise ValueError("shape element count exceeds the accepted range")
        checked.append(size)
    return tuple(checked)


def _require_logical_positions(
    logical_positions: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if logical_positions is None:
        return _logical_positions(shape, device=device)
    if logical_positions.dtype is not torch.int64:
        raise TypeError("logical_positions must have dtype torch.int64")
    if logical_positions.shape != shape:
        raise ValueError("logical_positions must have the exact output shape")
    if logical_positions.device != device:
        raise ValueError("logical_positions must be on the output device")
    if bool(torch.any(logical_positions < 0).item()):
        raise ValueError("logical_positions must be nonnegative")
    # A nonnegative int64 value is already strictly below 2**63.  Comparing
    # an int64 tensor with the unrepresentable Python integer 2**63 would wrap
    # the scalar in Torch and reject every valid position.
    return logical_positions


def _prepare_poisson_mean(
    mean: float | torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if type(mean) is float:
        means = torch.full(shape, mean, dtype=torch.float64, device=device)
    elif type(mean) is torch.Tensor:
        if mean.dtype is not torch.float64:
            raise TypeError("Poisson mean tensor must have dtype torch.float64")
        if mean.shape != shape:
            raise ValueError("Poisson mean tensor must have the exact output shape")
        if mean.device != device:
            raise ValueError("Poisson mean tensor must be on the output device")
        means = mean
    else:
        raise TypeError("Poisson mean must be exactly float or torch.Tensor")
    if not bool(torch.all(torch.isfinite(means)).item()):
        raise ValueError("Poisson means must be finite")
    if bool(torch.any(means < 0.0).item()):
        raise ValueError("Poisson means must be nonnegative")
    if bool(torch.any(means > _MAX_POISSON_MEAN).item()):
        raise ValueError("Poisson means exceed the accepted ceiling")
    return means


def _sample_poisson(
    mean: float | torch.Tensor,
    *,
    shape: tuple[int, ...],
    seed: int,
    stream: _RngStream,
    device: torch.device | str,
    logical_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample independent Poisson counts at fixed positional RNG addresses."""

    checked_shape = _require_count_shape(shape)
    checked_device = torch.device(device)
    checked_stream = _require_stream(stream)
    means = _prepare_poisson_mean(
        mean,
        shape=checked_shape,
        device=checked_device,
    )
    positions = _require_logical_positions(
        logical_positions,
        shape=checked_shape,
        device=checked_device,
    )
    active = means > 0.0
    if not bool(torch.any(active).item()):
        return torch.zeros(checked_shape, dtype=torch.int64, device=checked_device)
    checked_seed = _require_seed(seed)

    flat_means = means.reshape(-1)
    flat_positions = positions.reshape(-1)
    output = torch.zeros(
        flat_means.shape,
        dtype=torch.int64,
        device=checked_device,
    )

    inversion_mask = (flat_means > 0.0) & (flat_means < 10.0)
    if bool(torch.any(inversion_mask).item()):
        indices = torch.nonzero(inversion_mask, as_tuple=False).flatten()
        selected_mean = flat_means[indices]
        selected_positions = flat_positions[indices]
        words = _random_block(
            seed=checked_seed,
            stream=checked_stream,
            logical_positions=selected_positions,
        )
        uniform = _uniform_closed_open(
            words[..., 0],
            word_1=words[..., 1],
            dtype=torch.float64,
        )
        probability = torch.exp(-selected_mean)
        cumulative = probability.clone()
        unresolved = torch.ones_like(indices, dtype=torch.bool)
        selected_output = torch.zeros_like(indices, dtype=torch.int64)
        for count in range(64):
            accepted = unresolved & (uniform < cumulative)
            selected_output[accepted] = count
            unresolved = unresolved & ~accepted
            if not bool(torch.any(unresolved).item()):
                break
            if count == 63:
                raise RuntimeError("Poisson inversion exhausted its 64 terms")
            count_float = float(count + 1)
            probability = probability * selected_mean / count_float
            cumulative = cumulative + probability
        output[indices] = selected_output

    ptrs_mask = flat_means >= 10.0
    if bool(torch.any(ptrs_mask).item()):
        indices = torch.nonzero(ptrs_mask, as_tuple=False).flatten()
        selected_mean = flat_means[indices]
        selected_positions = flat_positions[indices]
        sqrt_mean = torch.sqrt(selected_mean)
        log_mean = torch.log(selected_mean)
        b = 0.931 + 2.53 * sqrt_mean
        a = -0.059 + 0.02483 * b
        inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
        v_rectangle = 0.9277 - 3.6224 / (b - 2.0)
        unresolved = torch.ones_like(indices, dtype=torch.bool)
        selected_output = torch.zeros_like(indices, dtype=torch.int64)
        for attempt in range(64):
            local = torch.nonzero(unresolved, as_tuple=False).flatten()
            words = _random_block(
                seed=checked_seed,
                stream=checked_stream,
                logical_positions=selected_positions[local],
                block=attempt,
            )
            uniform = _uniform_open_open(
                words[..., 0],
                word_1=words[..., 1],
                dtype=torch.float64,
            )
            variate = _uniform_open_open(
                words[..., 2],
                word_1=words[..., 3],
                dtype=torch.float64,
            )
            u = uniform - 0.5
            u_s = 0.5 - torch.abs(u)
            proposal_float = torch.floor(
                (2.0 * a[local] / u_s + b[local]) * u
                + selected_mean[local]
                + 0.43
            )
            too_large = (
                torch.isfinite(proposal_float)
                & (proposal_float >= 0.0)
                & (proposal_float > float(_MAX_CHARGE_COUNT))
            )
            if bool(torch.any(too_large).item()):
                raise RuntimeError("Poisson proposal exceeds the Charge count ceiling")
            supported = (
                torch.isfinite(proposal_float)
                & (proposal_float >= 0.0)
                & (proposal_float <= float(_MAX_CHARGE_COUNT))
            )
            proposal = torch.zeros_like(local, dtype=torch.int64)
            proposal[supported] = proposal_float[supported].to(torch.int64)
            quick_accept = (
                supported
                & (u_s >= 0.07)
                & (variate <= v_rectangle[local])
            )
            quick_reject = (~supported) | (
                (u_s < 0.013) & (variate > u_s)
            )
            full = supported & ~quick_accept & ~quick_reject
            full_accept = torch.zeros_like(full)
            if bool(torch.any(full).item()):
                full_local = torch.nonzero(full, as_tuple=False).flatten()
                k = proposal[full_local]
                left = (
                    torch.log(variate[full_local])
                    + torch.log(inverse_alpha[local[full_local]])
                    - torch.log(
                        a[local[full_local]]
                        / (u_s[full_local] * u_s[full_local])
                        + b[local[full_local]]
                    )
                )
                right = (
                    -selected_mean[local[full_local]]
                    + k.to(torch.float64) * log_mean[local[full_local]]
                    - torch.lgamma(k.to(torch.float64) + 1.0)
                )
                full_accept[full_local] = left <= right
            accepted = quick_accept | full_accept
            if bool(torch.any(accepted).item()):
                accepted_local = local[accepted]
                selected_output[accepted_local] = proposal[accepted]
                unresolved[accepted_local] = False
            if not bool(torch.any(unresolved).item()):
                break
            if attempt == 63:
                raise RuntimeError("Poisson PTRS exhausted its 64 attempts")
        output[indices] = selected_output

    return output.reshape(checked_shape)


def _stirling_correction(values: torch.Tensor) -> torch.Tensor:
    if values.dtype is not torch.int64:
        raise TypeError("Stirling arguments must have dtype torch.int64")
    result = torch.empty_like(values, dtype=torch.float64)
    small = values < 10
    if bool(torch.any(small).item()):
        table = torch.tensor(
            _STIRLING_CORRECTIONS,
            dtype=torch.float64,
            device=values.device,
        )
        result[small] = table[values[small]]
    if bool(torch.any(~small).item()):
        x = values[~small].to(torch.float64) + 1.0
        x2 = x * x
        inner = (1.0 / 360.0) - ((1.0 / 1260.0) / x2)
        result[~small] = ((1.0 / 12.0) - (inner / x2)) / x
    return result


def _sample_conditional_binomial(
    counts: torch.Tensor,
    success_mass: torch.Tensor,
    later_mass: torch.Tensor,
    *,
    seed: int,
    stream: _RngStream,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    """Sample one stable aggregate-multinomial conditional category."""

    if counts.dtype is not torch.int64:
        raise TypeError("counts must have dtype torch.int64")
    if success_mass.dtype is not torch.float64 or later_mass.dtype is not torch.float64:
        raise TypeError("conditional masses must have dtype torch.float64")
    if success_mass.shape != counts.shape or later_mass.shape != counts.shape:
        raise ValueError("conditional masses must have the exact count shape")
    if success_mass.device != counts.device or later_mass.device != counts.device:
        raise ValueError("conditional masses must be on the count device")
    positions = _require_logical_positions(
        logical_positions,
        shape=tuple(counts.shape),
        device=counts.device,
    )
    if bool(torch.any(counts < 0).item()) or bool(
        torch.any(counts > _MAX_CHARGE_COUNT).item()
    ):
        raise ValueError("counts exceed the accepted Charge count domain")
    for name, mass in (("success", success_mass), ("later", later_mass)):
        if not bool(torch.all(torch.isfinite(mass)).item()):
            raise ValueError(f"{name} masses must be finite")
        if bool(torch.any((mass < 0.0) | (mass > 1.0)).item()):
            raise ValueError(f"{name} masses must lie in [0, 1]")

    flat_counts = counts.reshape(-1)
    flat_success = success_mass.reshape(-1)
    flat_later = later_mass.reshape(-1)
    flat_positions = positions.reshape(-1)
    output = torch.zeros_like(flat_counts)

    both_zero = (flat_success == 0.0) & (flat_later == 0.0)
    if bool(torch.any(both_zero & (flat_counts != 0)).item()):
        raise ValueError("zero conditional mass cannot own a remaining count")
    deterministic_all = (flat_later == 0.0) & (flat_success > 0.0)
    output[deterministic_all] = flat_counts[deterministic_all]
    active = (
        (flat_counts > 0)
        & (flat_success > 0.0)
        & (flat_later > 0.0)
    )
    if not bool(torch.any(active).item()):
        return output.reshape(counts.shape)
    checked_seed = _require_seed(seed)
    checked_stream = _require_stream(stream)

    indices = torch.nonzero(active, as_tuple=False).flatten()
    n = flat_counts[indices]
    a_mass = flat_success[indices]
    b_mass = flat_later[indices]
    total = a_mass + b_mass
    if not bool(torch.all(torch.isfinite(total) & (total > 0.0)).item()):
        raise ValueError("conditional mass totals must be finite and positive")
    complement = b_mass < a_mass
    p = torch.minimum(a_mass, b_mass) / total
    if bool(torch.any((p <= 0.0) | (p > 0.5)).item()):
        raise ValueError("reduced conditional probabilities must lie in (0, 0.5]")
    selected = torch.zeros_like(n)

    inversion = n.to(torch.float64) * p < 10.0
    if bool(torch.any(inversion).item()):
        local = torch.nonzero(inversion, as_tuple=False).flatten()
        local_n = n[local]
        local_p = p[local]
        words = _random_block(
            seed=checked_seed,
            stream=checked_stream,
            logical_positions=flat_positions[indices[local]],
        )
        uniform = _uniform_closed_open(
            words[..., 0],
            word_1=words[..., 1],
            dtype=torch.float64,
        )
        q = 1.0 - local_p
        probability = torch.exp(local_n.to(torch.float64) * torch.log1p(-local_p))
        cumulative = probability.clone()
        unresolved = torch.ones_like(local, dtype=torch.bool)
        local_output = torch.zeros_like(local_n)
        for count in range(64):
            inside_support = local_n >= count
            accepted = unresolved & inside_support & (uniform < cumulative)
            local_output[accepted] = count
            unresolved = unresolved & ~accepted
            if not bool(torch.any(unresolved).item()):
                break
            if count == 63 or bool(
                torch.any(unresolved & (local_n <= count)).item()
            ):
                raise RuntimeError("Binomial inversion exhausted its 64 terms")
            advanced_probability = probability.clone()
            advanced_probability[unresolved] = probability[unresolved] * (
                (local_n[unresolved].to(torch.float64) - float(count))
                / float(count + 1)
            ) * (local_p[unresolved] / q[unresolved])
            probability = advanced_probability
            advanced_cumulative = cumulative.clone()
            advanced_cumulative[unresolved] = (
                cumulative[unresolved] + probability[unresolved]
            )
            cumulative = advanced_cumulative
        selected[local] = local_output

    btrs = ~inversion
    if bool(torch.any(btrs).item()):
        local = torch.nonzero(btrs, as_tuple=False).flatten()
        local_n = n[local]
        local_p = p[local]
        n_float = local_n.to(torch.float64)
        s = torch.sqrt(n_float * local_p * (1.0 - local_p))
        b = 1.15 + 2.53 * s
        a = -0.0873 + 0.0248 * b + 0.01 * local_p
        c = n_float * local_p + 0.5
        v_r = 0.92 - 4.2 / b
        r = local_p / (1.0 - local_p)
        alpha = (2.83 + 5.1 / b) * s
        m = torch.floor((n_float + 1.0) * local_p).to(torch.int64)
        unresolved = torch.ones_like(local, dtype=torch.bool)
        local_output = torch.zeros_like(local_n)
        for attempt in range(64):
            pending = torch.nonzero(unresolved, as_tuple=False).flatten()
            words = _random_block(
                seed=checked_seed,
                stream=checked_stream,
                logical_positions=flat_positions[indices[local[pending]]],
                block=attempt,
            )
            u_0 = _uniform_open_open(
                words[..., 0],
                word_1=words[..., 1],
                dtype=torch.float64,
            )
            v = _uniform_open_open(
                words[..., 2],
                word_1=words[..., 3],
                dtype=torch.float64,
            )
            u = u_0 - 0.5
            u_s = 0.5 - torch.abs(u)
            proposal_float = torch.floor(
                (2.0 * a[pending] / u_s + b[pending]) * u + c[pending]
            )
            supported = (
                torch.isfinite(proposal_float)
                & (proposal_float >= 0.0)
                & (proposal_float <= n_float[pending])
            )
            proposal = torch.zeros_like(pending, dtype=torch.int64)
            proposal[supported] = proposal_float[supported].to(torch.int64)
            quick_accept = supported & (u_s >= 0.07) & (v <= v_r[pending])
            full = supported & ~quick_accept
            full_accept = torch.zeros_like(full)
            if bool(torch.any(full).item()):
                full_pending = torch.nonzero(full, as_tuple=False).flatten()
                k = proposal[full_pending]
                chosen = pending[full_pending]
                d = k - m[chosen]
                left = torch.log(
                    v[full_pending]
                    * alpha[chosen]
                    / (a[chosen] / (u_s[full_pending] * u_s[full_pending]) + b[chosen])
                )
                k_float = k.to(torch.float64)
                d_float = d.to(torch.float64)
                log_left = torch.log1p(
                    d_float / (n_float[chosen] - k_float + 1.0)
                )
                log_right = torch.log1p(-d_float / (k_float + 1.0))
                log_ratio = torch.log(
                    r[chosen]
                    * (n_float[chosen] - k_float + 1.0)
                    / (k_float + 1.0)
                )
                main = (
                    (n_float[chosen] - m[chosen].to(torch.float64) + 0.5)
                    * log_left
                    + (m[chosen].to(torch.float64) + 0.5) * log_right
                ) + d_float * log_ratio
                correction = (
                    _stirling_correction(m[chosen])
                    + _stirling_correction(local_n[chosen] - m[chosen])
                    - _stirling_correction(k)
                    - _stirling_correction(local_n[chosen] - k)
                )
                full_accept[full_pending] = left <= main + correction
            accepted = quick_accept | full_accept
            if bool(torch.any(accepted).item()):
                accepted_pending = pending[accepted]
                local_output[accepted_pending] = proposal[accepted]
                unresolved[accepted_pending] = False
            if not bool(torch.any(unresolved).item()):
                break
            if attempt == 63:
                raise RuntimeError("Binomial BTRS exhausted its 64 attempts")
        selected[local] = local_output

    selected = torch.where(complement, n - selected, selected)
    output[indices] = selected
    return output.reshape(counts.shape)
