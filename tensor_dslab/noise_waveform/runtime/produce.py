"""Execute exact-zero, white, and prepared-PSD noise laws."""

import math

import torch
from tensor_core import CounterRng, GaussianDistribution, RngElements

from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.noise_waveform.config import NoiseWaveformConfig
from tensor_dslab.noise_waveform.runtime.random import (
    psd_noise_address,
    white_noise_address,
)


def _broadcast_kernel(kernel, dimensions: tuple[int, ...], *, config: NoiseWaveformConfig) -> torch.Tensor:
    shape = [1] * len(config.spec.shape)
    for index, dimension in enumerate(dimensions):
        shape[dimension] = kernel.tensor.shape[index]
    return kernel.tensor.to(config._working_dtype).reshape(shape)


def produce_noise_waveform(*, sources: tuple, config: NoiseWaveformConfig, rng: CounterRng) -> torch.Tensor:
    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="NoiseWaveformConfig",
    )
    assert config._working_dtype is not None
    require_prepared_sources(sources, source_specs=config._source_specs)
    shape = config.spec.shape
    device = config.spec.device
    white = config.kernels.white_noise_rms
    psd = config.kernels.power_spectral_density
    if white is None and psd is None:
        return torch.zeros(shape, dtype=config.spec.dtype, device=device)
    if white is not None:
        rms = _broadcast_kernel(white, config._kernel_dimensions[0], config=config)  # type: ignore[arg-type]
        values = GaussianDistribution(
            mean=0.0,
            standard_deviation=torch.broadcast_to(rms, shape),
            dtype=config._working_dtype,
        ).draw(
            rng=rng,
            address=white_noise_address(RngElements.from_shape(shape, device=device)),
        )
        return values.to(config.spec.dtype).contiguous()
    assert psd is not None and config._temporal_dimension is not None
    sample_dimension = config._temporal_dimension
    sample_count = shape[sample_dimension]
    frequency_count = sample_count // 2 + 1
    non_sample_shape = shape[:sample_dimension] + shape[sample_dimension + 1 :]
    row_count = math.prod(non_sample_shape)
    powers = psd.tensor.to(config._working_dtype)
    target_shape = [1] * len(non_sample_shape) + [frequency_count]
    if psd.conditioning_axes:
        for source_dimension, target_dimension in enumerate(config._kernel_dimensions[1]):  # type: ignore[union-attr]
            adjusted = target_dimension if target_dimension < sample_dimension else target_dimension - 1
            target_shape[adjusted] = powers.shape[source_dimension]
    powers = torch.broadcast_to(
        powers.reshape(target_shape),
        (*non_sample_shape, frequency_count),
    )
    powers = powers.reshape(row_count, frequency_count)
    elements = RngElements.from_shape((row_count, frequency_count), device=device).slice(1, 1, None)
    normals = GaussianDistribution(
        mean=0.0,
        standard_deviation=1.0,
        dtype=config._working_dtype,
        count=2,
    ).draw(rng=rng, address=psd_noise_address(elements))
    complex_dtype = torch.complex64 if config._working_dtype is torch.float32 else torch.complex128
    dc = torch.zeros((row_count, 1), dtype=complex_dtype, device=device)
    interior_count = (sample_count - 1) // 2
    interior_scale = torch.tensor(sample_count / 2.0, dtype=config._working_dtype, device=device) * torch.sqrt(powers[:, 1 : interior_count + 1])
    interior = torch.complex(normals[:, :interior_count, 0] * interior_scale, normals[:, :interior_count, 1] * interior_scale)
    coefficients = [dc, interior]
    if sample_count % 2 == 0:
        nyquist_scale = torch.tensor(float(sample_count), dtype=config._working_dtype, device=device) * torch.sqrt(powers[:, -1:])
        nyquist_real = normals[:, -1:, 0] * nyquist_scale
        coefficients.append(torch.complex(nyquist_real, torch.zeros_like(nyquist_real)))
    values = torch.fft.irfft(torch.cat(coefficients, dim=-1), n=sample_count, dim=-1)
    return values.reshape((*non_sample_shape, sample_count)).movedim(-1, sample_dimension).to(config.spec.dtype).contiguous()
