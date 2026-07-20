from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from tensor_core import CounterRng, RngKey, logical_positions

from tensor_dslab.common import SampleAxis, SamplingConfig
from tensor_dslab.readout._requirements import (
    _require_representable_float,
    _require_sampling,
)
from tensor_dslab.readout.noise_waveform.config import (
    NoiseWaveformConfig,
    PsdNoiseConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.noise_waveform.field import (
    NoiseWaveform,
    _require_valid_values,
)
from tensor_dslab.readout.photoelectrons import Photoelectrons


@dataclass(frozen=True, slots=True)
class _ZeroNoisePlan:
    pass


@dataclass(frozen=True, slots=True)
class _WhiteNoisePlan:
    rng_key: RngKey
    represented_rms: float


@dataclass(frozen=True, slots=True)
class _PsdNoisePlan:
    rng_key: RngKey
    represented_powers: torch.Tensor
    sample_dimension: int


@dataclass(frozen=True, slots=True)
class _NoiseWaveformPlan:
    shape: tuple[int, ...]
    device: torch.device
    floating_dtype: torch.dtype
    model: _ZeroNoisePlan | _WhiteNoisePlan | _PsdNoisePlan
    rng_roles: tuple[tuple[str, RngKey], ...]


def _require_position_count(count: int, *, field: str) -> None:
    if count < 0 or count >= 1 << 63:
        raise ValueError(f"{field} exceeds the accepted logical-position range")


def _prepare_white_rms(value: float, *, dtype: torch.dtype) -> float:
    represented = _require_representable_float(
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
    config: PsdNoiseConfig,
    *,
    sampling: SamplingConfig,
    dtype: torch.dtype,
) -> tuple[float, ...]:
    sample_count = sampling.sample_count.value
    try:
        sample_rate_hz = 1.0e12 / sampling.sample_period_ps.value
        spacing_hz = sample_rate_hz / sample_count
        nyquist_hz = sample_rate_hz / 2.0
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("sampling frequencies cannot be represented") from error
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (sample_rate_hz, spacing_hz, nyquist_hz)
    ):
        raise ValueError("sampling frequencies must be finite and positive")
    if config.frequency_stop_hz.value < nyquist_hz:
        raise ValueError("PSD frequency coverage must reach Nyquist")

    source_left = tuple(edge.value for edge in config.frequency_left_edges_hz)
    source_right = source_left[1:] + (config.frequency_stop_hz.value,)
    source_density = tuple(
        density.value for density in config.power_density_mv2_per_hz
    )

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
        _require_representable_float(
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


def _white_noise(
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    rng: CounterRng,
    rng_key: RngKey,
    represented_rms: float,
) -> torch.Tensor:
    positions = logical_positions(shape, device=device)
    return rng.gaussian(
        mean=0.0,
        standard_deviation=represented_rms,
        key=rng_key,
        positions=positions,
        dtype=dtype,
        quantum=0,
        ordinal=0,
        count=1,
    )


def _psd_noise(
    *,
    shape: tuple[int, ...],
    sample_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    rng: CounterRng,
    rng_key: RngKey,
    represented_powers: torch.Tensor,
) -> torch.Tensor:
    sample_count = shape[sample_dimension]
    frequency_count = sample_count // 2 + 1
    non_sample_shape = shape[:sample_dimension] + shape[sample_dimension + 1 :]
    row_count = math.prod(non_sample_shape)

    positions = logical_positions(
        (row_count, frequency_count),
        device=device,
    )[:, 1:]
    normals = rng.gaussian(
        mean=0.0,
        standard_deviation=1.0,
        key=rng_key,
        positions=positions,
        dtype=dtype,
        quantum=0,
        ordinal=0,
        count=2,
    )
    normal_real = normals[..., 0]
    normal_imaginary = normals[..., 1]

    complex_dtype = torch.complex64 if dtype is torch.float32 else torch.complex128
    dc = torch.zeros((row_count, 1), dtype=complex_dtype, device=device)
    interior_count = (sample_count - 1) // 2
    with torch.autocast(device_type=device.type, enabled=False):
        interior_scale = (
            torch.tensor(sample_count / 2.0, dtype=dtype, device=device)
            * torch.sqrt(represented_powers[1 : interior_count + 1])
        )
        interior = torch.complex(
            normal_real[:, :interior_count] * interior_scale,
            normal_imaginary[:, :interior_count] * interior_scale,
        )
        if sample_count % 2 == 0:
            nyquist_scale = (
                torch.tensor(float(sample_count), dtype=dtype, device=device)
                * torch.sqrt(represented_powers[-1])
            )
            nyquist_real = normal_real[:, -1:] * nyquist_scale
            nyquist = torch.complex(
                nyquist_real,
                torch.zeros_like(nyquist_real),
            )
            coefficients = torch.cat((dc, interior, nyquist), dim=-1)
        else:
            coefficients = torch.cat((dc, interior), dim=-1)
        sample_last = torch.fft.irfft(
            coefficients,
            n=sample_count,
            dim=-1,
            norm="backward",
        )

    sample_last_shape = non_sample_shape + (sample_count,)
    return sample_last.reshape(sample_last_shape).movedim(-1, sample_dimension)


def _prepare_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: NoiseWaveformConfig,
    floating_dtype: torch.dtype,
) -> _NoiseWaveformPlan:
    if type(photoelectrons) is not Photoelectrons:
        raise TypeError("photoelectrons must be exactly Photoelectrons")
    if type(config) is not NoiseWaveformConfig:
        raise TypeError("config must be exactly NoiseWaveformConfig")
    _require_sampling(photoelectrons, sampling)
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")

    device = photoelectrons.tensor.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("noise production supports only CPU and CUDA")
    shape = photoelectrons.shape
    output_count = math.prod(shape)
    _require_position_count(output_count, field="output")
    model = config.model

    if type(model) is ZeroNoiseConfig:
        return _NoiseWaveformPlan(
            shape=shape,
            device=device,
            floating_dtype=floating_dtype,
            model=_ZeroNoisePlan(),
            rng_roles=(),
        )
    elif type(model) is WhiteNoiseConfig:
        represented_rms = _prepare_white_rms(
            model.rms_mv.value,
            dtype=floating_dtype,
        )
        return _NoiseWaveformPlan(
            shape=shape,
            device=device,
            floating_dtype=floating_dtype,
            model=_WhiteNoisePlan(
                rng_key=model.rng_key,
                represented_rms=represented_rms,
            ),
            rng_roles=(("noise.white", model.rng_key),),
        )
    elif type(model) is PsdNoiseConfig:
        represented_power_values = _prepare_psd_powers(
            model,
            sampling=sampling,
            dtype=floating_dtype,
        )
        sample_dimension = photoelectrons.dimension_of(SampleAxis)
        row_count = output_count // sampling.sample_count.value
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
        return _NoiseWaveformPlan(
            shape=shape,
            device=device,
            floating_dtype=floating_dtype,
            model=_PsdNoisePlan(
                rng_key=model.rng_key,
                represented_powers=represented_powers,
                sample_dimension=sample_dimension,
            ),
            rng_roles=(("noise.psd", model.rng_key),),
        )
    else:
        raise TypeError("NoiseWaveformConfig.model is not recognized")


def _produce_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    plan: _NoiseWaveformPlan,
    rng: CounterRng,
) -> NoiseWaveform:
    if type(photoelectrons) is not Photoelectrons:
        raise TypeError("photoelectrons must be exactly Photoelectrons")
    if not isinstance(rng, CounterRng):
        raise TypeError("rng must be a CounterRng")

    model = plan.model
    if type(model) is _ZeroNoisePlan:
        values = torch.zeros(
            plan.shape,
            dtype=plan.floating_dtype,
            device=plan.device,
        )
    elif type(model) is _WhiteNoisePlan:
        values = _white_noise(
            shape=plan.shape,
            device=plan.device,
            dtype=plan.floating_dtype,
            rng=rng,
            rng_key=model.rng_key,
            represented_rms=model.represented_rms,
        )
    elif type(model) is _PsdNoisePlan:
        values = _psd_noise(
            shape=plan.shape,
            sample_dimension=model.sample_dimension,
            device=plan.device,
            dtype=plan.floating_dtype,
            rng=rng,
            rng_key=model.rng_key,
            represented_powers=model.represented_powers,
        )
    else:
        raise RuntimeError("noise plan model is not recognized")

    result = NoiseWaveform(tensor=values, axes=photoelectrons.axes)
    _require_valid_values(result)
    return result
