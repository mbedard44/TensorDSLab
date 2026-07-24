"""Private source-owned sampling runtime preparation."""

from dataclasses import dataclass
from typing import final

from tensor_dslab.common import SampleAxis
from tensor_dslab.readout.photoelectrons.field import Photoelectrons


@final
@dataclass(frozen=True, slots=True)
class SamplingRuntime:
    sample_count: int
    sample_period_ps: int
    sample_dimension: int


def prepare_sampling(
    photoelectrons: Photoelectrons,
) -> SamplingRuntime:
    sample_dimension = photoelectrons.dimension_of(SampleAxis)
    sample_axis = photoelectrons.axis(SampleAxis)
    if sample_axis.start != 0:
        raise ValueError("sample-axis start must be zero")
    return SamplingRuntime(
        sample_count=sample_axis.count,
        sample_period_ps=sample_axis.step,
        sample_dimension=sample_dimension,
    )
