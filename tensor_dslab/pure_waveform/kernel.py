"""Physical pulse-response specification and kernel."""

from typing import Any, final, override

import torch
from tensor_core import OffsetAxis, TensorKernel

from tensor_dslab.common import QuantityKernelSpec


@final
class PulseResponseSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    """Specify one literal physical pulse-response kernel."""

    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if not self.operation_axes or any(
            type(axis) is not OffsetAxis for axis in self.operation_axes
        ):
            raise ValueError("PulseResponseSpec requires OffsetAxis operations")
        if any(axis.size == 0 for axis in self.operation_axes):
            raise ValueError("PulseResponseSpec operation axes must be nonempty")


@final
class PulseResponse(TensorKernel[PulseResponseSpec[Any, Any]]):
    """Represent finite signed convolution coefficients."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not PulseResponseSpec:
            raise TypeError("PulseResponse requires exact PulseResponseSpec")
        if not self.dtype.is_floating_point:
            raise TypeError("PulseResponse dtype must be floating")
        if not bool(torch.isfinite(self.tensor).all()):
            raise ValueError("PulseResponse values must be finite")
