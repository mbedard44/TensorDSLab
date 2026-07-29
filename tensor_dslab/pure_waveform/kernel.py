"""Physical pulse-response Spec, kernel, and collection."""

from typing import Any, final, override

from tensor_core import OffsetAxis, TensorCollection, TensorKernel

from tensor_dslab.common import QuantityKernelSpec
from tensor_dslab.common.requirements.collection import (
    require_exact_member_types,
)
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_nonempty_operation_extents,
    require_operation_axes_type,
    require_operation_axis_count,
)
from tensor_dslab.common.requirements.tensor import (
    require_finite,
    require_floating_dtype,
)


@final
class PulseResponseSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    """Specify one literal physical pulse-response kernel."""

    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_floating_dtype(self)
        require_operation_axis_count(self, minimum=1)
        require_operation_axes_type(self, OffsetAxis)
        require_nonempty_operation_extents(self)


@final
class PulseResponse(TensorKernel[PulseResponseSpec[Any, Any]]):
    """Represent finite signed convolution coefficients."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, PulseResponseSpec)
        require_finite(self)


@final
class PureWaveformKernels(TensorCollection[TensorKernel[Any]]):
    """Hold the exact pulse-response coefficient set."""

    __slots__ = ()

    def _require(self) -> None:
        require_exact_member_types(self, required=(PulseResponse,))

    @property
    def pulse_response(self) -> PulseResponse:
        return self.member(PulseResponse)
