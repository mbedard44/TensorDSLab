"""Physical saturation coefficient Specs and kernels."""

from typing import Any, final, override

import torch
from tensor_core import TensorKernel

from tensor_dslab.common import QuantityKernelSpec


@final
class AnalogMinimumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if self.operation_axes:
            raise ValueError("AnalogMinimumSpec has no operation axes")


@final
class AnalogMaximumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if self.operation_axes:
            raise ValueError("AnalogMaximumSpec has no operation axes")


def _require_finite(
    kernel: TensorKernel[Any],
    *,
    spec_type: type,
) -> None:
    if type(kernel.spec) is not spec_type:
        raise TypeError(
            f"{type(kernel).__name__} requires exact {spec_type.__name__}"
        )
    if not kernel.dtype.is_floating_point:
        raise TypeError(f"{type(kernel).__name__} dtype must be floating")
    if not bool(torch.isfinite(kernel.tensor).all()):
        raise ValueError(f"{type(kernel).__name__} values must be finite")


@final
class AnalogMinimum(TensorKernel[AnalogMinimumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        _require_finite(self, spec_type=AnalogMinimumSpec)


@final
class AnalogMaximum(TensorKernel[AnalogMaximumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        _require_finite(self, spec_type=AnalogMaximumSpec)
