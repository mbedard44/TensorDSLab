"""Private compilation of a literal Pulse into execution facts."""

from dataclasses import dataclass
from typing import final

import torch

from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.pure_waveform.config import PureWaveformConfig
from tensor_dslab.readout.runtime.kernel import align_quantity_kernel
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


@final
@dataclass(frozen=True, slots=True)
class PureWaveformRuntime:
    sampling: SamplingRuntime
    coefficients: torch.Tensor
    conditioning_dimensions: tuple[int, ...]
    sample_offsets: tuple[int, ...]


def prepare_pure_waveform(
    config: PureWaveformConfig,
    *,
    source: Photoelectrons,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> PureWaveformRuntime:
    """Align and materialize one pulse without retaining public kernel state."""

    pulse = config.pulse
    coefficients, dimensions = align_quantity_kernel(
        pulse,
        field=source,
        dtype=floating_dtype,
    )
    return PureWaveformRuntime(
        sampling=sampling,
        coefficients=coefficients,
        conditioning_dimensions=dimensions,
        sample_offsets=pulse.operation_axes[0].offsets,
    )
