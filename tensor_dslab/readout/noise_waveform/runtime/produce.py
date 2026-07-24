"""Private tensor execution for noise waveform products."""

import math

import torch
from tensor_core import CounterRng, RngKey, RngPositions

from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    NoiseWaveformRuntime,
    PsdNoiseRuntime,
    WhiteNoiseRuntime,
    ZeroNoiseRuntime,
)
from tensor_dslab.readout.photoelectrons.field import Photoelectrons


def _white_noise(
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    rng: CounterRng,
    rng_key: RngKey,
    represented_rms: float,
) -> torch.Tensor:
    positions = RngPositions.from_shape(shape, device=device)
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

    positions = RngPositions.from_shape(
        (row_count, frequency_count),
        device=device,
    ).slice(1, 1, None)
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


def produce_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    runtime: NoiseWaveformRuntime,
    rng: CounterRng,
) -> NoiseWaveform:
    model = runtime.model
    if type(model) is ZeroNoiseRuntime:
        values = torch.zeros(
            runtime.shape,
            dtype=runtime.floating_dtype,
            device=runtime.device,
        )
    elif type(model) is WhiteNoiseRuntime:
        values = _white_noise(
            shape=runtime.shape,
            device=runtime.device,
            dtype=runtime.floating_dtype,
            rng=rng,
            rng_key=model.rng_key,
            represented_rms=model.represented_rms_mv,
        )
    elif type(model) is PsdNoiseRuntime:
        values = _psd_noise(
            shape=runtime.shape,
            sample_dimension=runtime.sampling.sample_dimension,
            device=runtime.device,
            dtype=runtime.floating_dtype,
            rng=rng,
            rng_key=model.rng_key,
            represented_powers=model.represented_powers_mv2,
        )
    else:
        raise RuntimeError("noise runtime model is not recognized")
    return NoiseWaveform(tensor=values, axes=photoelectrons.axes)
