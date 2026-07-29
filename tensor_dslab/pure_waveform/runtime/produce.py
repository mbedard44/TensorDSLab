"""Execute deterministic literal pulse convolution."""

import itertools

import torch

from tensor_dslab.common.alignment import require_prepared_sources
from tensor_dslab.pure_waveform.config import PureWaveformConfig


def produce_pure_waveform(*, sources: tuple, config: PureWaveformConfig) -> torch.Tensor:
    """Return one fresh tensor from a prepared PureWaveform Config."""

    if not config._is_prepared or config._working_dtype is None:
        raise ValueError("PureWaveformConfig must be prepared")
    require_prepared_sources(sources, source_specs=config._source_specs)
    combined = torch.zeros(config.spec.shape, dtype=config._working_dtype, device=config.spec.device)
    for source, dimensions, scale in zip(
        sources, config._source_dimensions, config._source_scales
    ):
        combined = combined + source.tensor.permute(dimensions).to(config._working_dtype) * scale
    pulse = config.kernels.pulse_response
    conditioning_rank = pulse.conditioning_rank
    kernel_dimensions = config._kernel_dimensions[0]
    assert kernel_dimensions is not None
    conditioning_dimensions = kernel_dimensions[:conditioning_rank]
    operation_dimensions = kernel_dimensions[conditioning_rank:]
    result = torch.zeros_like(combined)
    for operation_index in itertools.product(
        *(range(axis.size) for axis in pulse.operation_axes)
    ):
        offsets = tuple(
            axis.coordinate_at(index)
            for axis, index in zip(pulse.operation_axes, operation_index)
        )
        coefficient = pulse.tensor[(..., *operation_index)].to(config._working_dtype)
        view = [1] * combined.ndim
        for index, dimension in enumerate(conditioning_dimensions):
            view[dimension] = coefficient.shape[index]
        coefficient = coefficient.reshape(view)
        source_slices: list[slice] = [slice(None)] * combined.ndim
        target_slices: list[slice] = [slice(None)] * combined.ndim
        valid = True
        for dimension, offset in zip(operation_dimensions, offsets):
            count = combined.shape[dimension]
            if abs(offset) >= count:
                valid = False
                break
            if offset >= 0:
                source_slices[dimension] = slice(0, count - offset)
                target_slices[dimension] = slice(offset, count)
            else:
                source_slices[dimension] = slice(-offset, count)
                target_slices[dimension] = slice(0, count + offset)
        if valid:
            coefficient_slices: list[slice] = [slice(None)] * combined.ndim
            for dimension in conditioning_dimensions:
                coefficient_slices[dimension] = source_slices[dimension]
            result[tuple(target_slices)] = (
                result[tuple(target_slices)]
                + combined[tuple(source_slices)]
                * coefficient[tuple(coefficient_slices)]
            )
    return result.to(dtype=config.spec.dtype).contiguous()
