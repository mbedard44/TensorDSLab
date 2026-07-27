"""Public physical pulse kernel."""

from typing import final, override

from tensor_core import OffsetAxis

from tensor_dslab.common import SampleAxis
from tensor_dslab.common.kernel import QuantityKernel


@final
class Pulse[ConditioningAxesT: tuple](QuantityKernel):
    """Represent signed millivolt response coefficients per PE-equivalent."""

    canonical_unit = "mV"

    @override
    def _require(self) -> None:
        if len(self.operation_axes) != 1:
            raise ValueError("Pulse requires exactly one operation axis")
        axis = self.operation_axes[0]
        if type(axis) is not OffsetAxis or axis.relative_to is not SampleAxis:
            raise ValueError("Pulse operation axis must target SampleAxis")
        if not axis.offsets:
            raise ValueError("Pulse operation axis must be nonempty")
        if any(offset < 0 for offset in axis.offsets):
            raise ValueError("Pulse sample offsets must be nonnegative")
