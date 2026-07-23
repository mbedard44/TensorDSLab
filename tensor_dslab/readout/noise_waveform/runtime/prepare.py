from __future__ import annotations

from dataclasses import dataclass
import math
from typing import final

import torch
from tensor_core import RngKey

from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.requirements import require_representable_float
from tensor_dslab.readout.noise_waveform.config import (
    NoiseWaveformConfig,
    PsdNoiseConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


@final
@dataclass(frozen=True, slots=True)
class ZeroNoiseRuntime:
    pass


@final
@dataclass(frozen=True, slots=True)
class WhiteNoiseRuntime:
    rng_key: RngKey
    represented_rms_mv: float


@final
@dataclass(frozen=True, slots=True)
class PsdNoiseRuntime:
    rng_key: RngKey
    represented_powers_mv2: torch.Tensor


@final
@dataclass(frozen=True, slots=True)
class NoiseWaveformRuntime:
    shape: tuple[int, ...]
    device: torch.device
    floating_dtype: torch.dtype
    sampling: SamplingRuntime
    model: ZeroNoiseRuntime | WhiteNoiseRuntime | PsdNoiseRuntime
    rng_roles: tuple[tuple[str, RngKey], ...]


def _require_position_count(count: int, *, field: str) -> None:
    if count < 0 or count >= 1 << 63:
        raise ValueError(f"{field} exceeds the accepted logical-position range")


def _prepare_white_rms(value: float, *, dtype: torch.dtype) -> float:
    represented = require_representable_float(
        value,
        dtype=dtype,
        field="white-noise RMS",
    )
    finfo = torch.finfo(dtype)
    guard = 8.0 if dtype is torch.float32 else 16.0
    bound = guard * represented
    if represented < finfo.tiny:
        raise ValueError("white-noise RMS must remain in the normal dtype range")
    if not math.isfinite(bound) or bound > finfo.max:
        raise ValueError("white-noise RMS exceeds the finite output domain")
    return represented


def _prepare_psd_powers(
    frequency_left_edges_hz: tuple[float, ...],
    frequency_stop_hz: float,
    power_density_mv2_per_hz: tuple[float, ...],
    *,
    sampling: SamplingRuntime,
    dtype: torch.dtype,
) -> tuple[float, ...]:
    sample_count = sampling.sample_count
    try:
        sample_rate_hz = 1.0e12 / sampling.sample_period_ps
        spacing_hz = sample_rate_hz / sample_count
        nyquist_hz = sample_rate_hz / 2.0
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("sampling frequencies cannot be represented") from error
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (sample_rate_hz, spacing_hz, nyquist_hz)
    ):
        raise ValueError("sampling frequencies must be finite and positive")
    if frequency_stop_hz < nyquist_hz:
        raise ValueError("PSD frequency coverage must reach Nyquist")

    source_left = frequency_left_edges_hz
    source_right = source_left[1:] + (frequency_stop_hz,)
    source_density = power_density_mv2_per_hz

    frequency_count = sample_count // 2 + 1
    target_left = (0.0,) + tuple(
        (index - 0.5) * spacing_hz for index in range(1, frequency_count)
    )
    target_right = target_left[1:] + (nyquist_hz,)

    integrated: list[float] = []
    for left, right in zip(target_left, target_right):
        contributions = (
            density
            * max(
                0.0,
                min(source_stop, right) - max(source_start, left),
            )
            for source_start, source_stop, density in zip(
                source_left,
                source_right,
                source_density,
            )
        )
        power = math.fsum(contributions)
        if not math.isfinite(power) or power < 0.0:
            raise ValueError("PSD integration produced an invalid power")
        integrated.append(power)

    represented = (0.0,) + tuple(
        require_representable_float(
            power,
            dtype=dtype,
            field=f"PSD power[{index}]",
        )
        for index, power in enumerate(integrated[1:], start=1)
    )
    if not any(power > 0.0 for power in represented[1:]):
        raise ValueError("PSD has no retained representable power")

    guard = 8.0 if dtype is torch.float32 else 16.0
    accumulation = sample_count * guard * math.fsum(
        math.sqrt(power) for power in represented[1:]
    )
    if not math.isfinite(accumulation) or accumulation > torch.finfo(dtype).max:
        raise ValueError("PSD exceeds the finite inverse-transform domain")
    return represented


def prepare_noise_waveform(
    config: NoiseWaveformConfig,
    *,
    sampling: SamplingRuntime,
    shape: tuple[int, ...],
    floating_dtype: torch.dtype,
    device: torch.device,
) -> NoiseWaveformRuntime:
    output_count = math.prod(shape)
    _require_position_count(output_count, field="output")
    model = config.model

    if type(model) is ZeroNoiseConfig:
        return NoiseWaveformRuntime(
            shape=shape,
            device=device,
            floating_dtype=floating_dtype,
            sampling=sampling,
            model=ZeroNoiseRuntime(),
            rng_roles=(),
        )
    elif type(model) is WhiteNoiseConfig:
        represented_rms = _prepare_white_rms(
            canonical_magnitude(model.rms),
            dtype=floating_dtype,
        )
        return NoiseWaveformRuntime(
            shape=shape,
            device=device,
            floating_dtype=floating_dtype,
            sampling=sampling,
            model=WhiteNoiseRuntime(
                rng_key=model.rng_key,
                represented_rms_mv=represented_rms,
            ),
            rng_roles=(("noise.white", model.rng_key),),
        )
    elif type(model) is PsdNoiseConfig:
        frequency_left_edges_hz = tuple(
            canonical_magnitude(value) for value in model.frequency_left_edges
        )
        frequency_stop_hz = canonical_magnitude(model.frequency_stop)
        power_density_mv2_per_hz = tuple(
            canonical_magnitude(value) for value in model.power_density
        )
        represented_power_values = _prepare_psd_powers(
            frequency_left_edges_hz,
            frequency_stop_hz,
            power_density_mv2_per_hz,
            sampling=sampling,
            dtype=floating_dtype,
        )
        row_count = output_count // sampling.sample_count
        coefficient_position_count = row_count * len(represented_power_values)
        _require_position_count(
            coefficient_position_count,
            field="PSD coefficient",
        )
        represented_powers = torch.tensor(
            represented_power_values,
            dtype=floating_dtype,
            device=device,
        )
        return NoiseWaveformRuntime(
            shape=shape,
            device=device,
            floating_dtype=floating_dtype,
            sampling=sampling,
            model=PsdNoiseRuntime(
                rng_key=model.rng_key,
                represented_powers_mv2=represented_powers,
            ),
            rng_roles=(("noise.psd", model.rng_key),),
        )
    else:
        raise TypeError("NoiseWaveformConfig.model is not recognized")
