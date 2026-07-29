"""Execute one pointwise linear ADC transformation."""

import torch

from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.digitized_waveform.config import DigitizedWaveformConfig


def _aligned(kernel, dimensions: tuple[int, ...], *, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    target = [1] * len(shape)
    for index, dimension in enumerate(dimensions):
        target[dimension] = kernel.tensor.shape[index]
    return kernel.tensor.to(dtype).reshape(target)


def produce_digitized_waveform(*, sources: tuple, config: DigitizedWaveformConfig) -> torch.Tensor:
    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="DigitizedWaveformConfig",
    )
    assert config._working_dtype is not None
    require_prepared_sources(sources, source_specs=config._source_specs)
    source = sources[0].tensor.permute(config._source_dimensions[0]).to(config._working_dtype)
    source = source * config._source_scales[0]
    bit_depth = _aligned(config.kernels.bit_depth, config._kernel_dimensions[0], shape=config.spec.shape, dtype=torch.int64)  # type: ignore[arg-type]
    minimum = _aligned(config.kernels.input_minimum, config._kernel_dimensions[1], shape=config.spec.shape, dtype=config._working_dtype)  # type: ignore[arg-type]
    maximum = _aligned(config.kernels.input_maximum, config._kernel_dimensions[2], shape=config.spec.shape, dtype=config._working_dtype)  # type: ignore[arg-type]
    gain = _aligned(config.kernels.analog_gain, config._kernel_dimensions[3], shape=config.spec.shape, dtype=config._working_dtype)  # type: ignore[arg-type]
    if bool((minimum >= maximum).any()):
        raise ValueError("InputMinimum must be below InputMaximum")
    maximum_code_integer = torch.bitwise_left_shift(
        torch.ones_like(bit_depth), bit_depth
    ) - 1
    maximum_code = maximum_code_integer.to(config._working_dtype)
    transformed = source * gain
    code = (transformed - minimum) * maximum_code / (maximum - minimum)
    code = torch.minimum(torch.maximum(code, torch.zeros_like(code)), maximum_code)
    return code.to(dtype=config.spec.dtype).contiguous()
