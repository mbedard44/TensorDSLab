"""Private deterministic literal pulse convolution."""

import torch

from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.pure_waveform.field import PureWaveform
from tensor_dslab.readout.pure_waveform.runtime.prepare import PureWaveformRuntime


def produce_pure_waveform(
    charge: Charge,
    *,
    runtime: PureWaveformRuntime,
) -> PureWaveform:
    """Convolve Charge with one prepared signed finite Pulse."""

    result = torch.zeros_like(charge.tensor)
    sample_dimension = runtime.sampling.sample_dimension
    sample_count = charge.shape[sample_dimension]
    flat = runtime.coefficients.reshape(
        *runtime.coefficients.shape[: len(runtime.conditioning_dimensions)],
        -1,
    )
    for operation_index, offset in enumerate(runtime.sample_offsets):
        if offset >= sample_count:
            continue
        coefficient = flat[..., operation_index]
        view_shape = [1] * charge.tensor.ndim
        for source_dimension, target_dimension in enumerate(
            runtime.conditioning_dimensions
        ):
            view_shape[target_dimension] = coefficient.shape[source_dimension]
        aligned = coefficient.reshape(tuple(view_shape))
        source_slices = [slice(None)] * charge.tensor.ndim
        target_slices = [slice(None)] * charge.tensor.ndim
        source_slices[sample_dimension] = slice(0, sample_count - offset)
        target_slices[sample_dimension] = slice(offset, sample_count)
        result[tuple(target_slices)] = (
            result[tuple(target_slices)]
            + charge.tensor[tuple(source_slices)]
            * aligned[tuple(source_slices)]
        )
    return PureWaveform(tensor=result, axes=charge.axes)
