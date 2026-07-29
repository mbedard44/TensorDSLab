"""Physical saturation coefficient Specs, kernels, and collection."""

from typing import Any, final, override

from tensor_core import TensorCollection, TensorKernel

from tensor_dslab.common import QuantityKernelSpec
from tensor_dslab.common.requirements.collection import (
    require_admitted_member_types,
)
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_no_operation_axes,
)
from tensor_dslab.common.requirements.tensor import (
    require_finite,
    require_floating_dtype,
)


@final
class AnalogMinimumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_floating_dtype(self)
        require_no_operation_axes(self)


@final
class AnalogMaximumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_floating_dtype(self)
        require_no_operation_axes(self)


@final
class AnalogMinimum(TensorKernel[AnalogMinimumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, AnalogMinimumSpec)
        require_finite(self)


@final
class AnalogMaximum(TensorKernel[AnalogMaximumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, AnalogMaximumSpec)
        require_finite(self)


@final
class AnalogWaveformKernels(TensorCollection[TensorKernel[Any]]):
    """Hold the optional exact analog-saturation bounds."""

    __slots__ = ()

    def _require(self) -> None:
        require_admitted_member_types(
            self,
            admitted=(AnalogMinimum, AnalogMaximum),
        )

    @property
    def minimum(self) -> AnalogMinimum | None:
        return self.members.get(AnalogMinimum)  # type: ignore[return-value]

    @property
    def maximum(self) -> AnalogMaximum | None:
        return self.members.get(AnalogMaximum)  # type: ignore[return-value]
