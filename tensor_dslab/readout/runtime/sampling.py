from __future__ import annotations

from dataclasses import dataclass
from typing import cast, final

from tensor_dslab.common import SampleAxis, SamplingConfig
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.requirements import require_exact


@final
@dataclass(frozen=True, slots=True)
class SamplingRuntime:
    sample_count: int
    sample_period_ps: int
    sample_dimension: int


def prepare_sampling(
    photoelectrons: Photoelectrons,
    *,
    config: SamplingConfig,
) -> SamplingRuntime:
    require_exact(config, SamplingConfig, "prepare_sampling.config")
    sample_dimension = photoelectrons.dimension_of(SampleAxis)
    sample_axis = cast(SampleAxis, photoelectrons.axes[sample_dimension])
    if sample_axis.size != config.sample_count.value:
        raise ValueError("sample-axis size must agree with SamplingConfig")
    if sample_axis.start_ps != 0:
        raise ValueError("sample-axis start must be zero")
    if sample_axis.sample_period_ps != config.sample_period_ps.value:
        raise ValueError("sample-axis period must agree with SamplingConfig")
    return SamplingRuntime(
        sample_count=config.sample_count.value,
        sample_period_ps=config.sample_period_ps.value,
        sample_dimension=sample_dimension,
    )
