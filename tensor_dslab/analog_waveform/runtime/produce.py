"""Execute ordered analog composition and saturation."""

import torch

from tensor_dslab.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.common.alignment import require_prepared_sources


def _aligned(kernel, dimensions: tuple[int, ...], *, config: AnalogWaveformConfig) -> torch.Tensor:
    shape = [1] * len(config.spec.shape)
    for source_dimension, target_dimension in enumerate(dimensions):
        shape[target_dimension] = kernel.tensor.shape[source_dimension]
    return kernel.tensor.to(config._working_dtype).reshape(shape)


def produce_analog_waveform(*, sources: tuple, config: AnalogWaveformConfig) -> torch.Tensor:
    if not config._is_prepared or config._working_dtype is None:
        raise ValueError("AnalogWaveformConfig must be prepared")
    require_prepared_sources(sources, source_specs=config._source_specs)
    result = torch.zeros(config.spec.shape, dtype=config._working_dtype, device=config.spec.device)
    for source, dimensions, scale in zip(sources, config._source_dimensions, config._source_scales):
        result = result + source.tensor.permute(dimensions).to(config._working_dtype) * scale
    if config.kernels.minimum is not None:
        result = torch.maximum(
            result,
            _aligned(config.kernels.minimum, config._kernel_dimensions[0], config=config),  # type: ignore[arg-type]
        )
    if config.kernels.maximum is not None:
        result = torch.minimum(
            result,
            _aligned(config.kernels.maximum, config._kernel_dimensions[1], config=config),  # type: ignore[arg-type]
        )
    return result.to(config.spec.dtype).contiguous()
