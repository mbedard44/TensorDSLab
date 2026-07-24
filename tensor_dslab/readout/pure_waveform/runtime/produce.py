"""Private tensor execution for pure waveform products."""

import torch
from torch.nn import functional

from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.pure_waveform.field import PureWaveform
from tensor_dslab.readout.pure_waveform.runtime.prepare import PureWaveformRuntime


def produce_pure_waveform(
    charge: Charge,
    *,
    runtime: PureWaveformRuntime,
) -> PureWaveform:
    sample_dimension = runtime.sampling.sample_dimension
    sample_last = charge.tensor.movedim(sample_dimension, -1)
    sample_count = sample_last.shape[-1]
    rows = sample_last.reshape(-1, 1, sample_count)
    coefficient_count = runtime.kernel.shape[-1]
    with torch.autocast(device_type=charge.tensor.device.type, enabled=False):
        padded = functional.pad(rows, (coefficient_count - 1, 0))
        convolved = functional.conv1d(padded, runtime.kernel)
    values = convolved.reshape(sample_last.shape).movedim(-1, sample_dimension)
    return PureWaveform(tensor=values, axes=charge.axes)
