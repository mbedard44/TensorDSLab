"""Digitizer coefficient Specs and kernels."""

from typing import Any, final, override

import torch
from tensor_core import TensorKernel, TensorKernelSpec

from tensor_dslab.common import QuantityKernelSpec


@final
class InputMinimumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if self.operation_axes:
            raise ValueError("InputMinimumSpec has no operation axes")


@final
class InputMaximumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if self.operation_axes:
            raise ValueError("InputMaximumSpec has no operation axes")


@final
class AnalogGainSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if self.operation_axes:
            raise ValueError("AnalogGainSpec has no operation axes")


@final
class BitDepthSpec[C: tuple, O: tuple](TensorKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        if self.operation_axes:
            raise ValueError("BitDepthSpec has no operation axes")
        if self.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("BitDepthSpec requires a signed integer dtype")


def _require_finite(
    kernel: TensorKernel[Any],
    *,
    spec_type: type,
    positive: bool = False,
) -> None:
    if type(kernel.spec) is not spec_type:
        raise TypeError(
            f"{type(kernel).__name__} requires exact {spec_type.__name__}"
        )
    if not kernel.dtype.is_floating_point:
        raise TypeError(f"{type(kernel).__name__} dtype must be floating")
    if not bool(torch.isfinite(kernel.tensor).all()):
        raise ValueError(f"{type(kernel).__name__} values must be finite")
    if positive and bool((kernel.tensor <= 0).any()):
        raise ValueError(f"{type(kernel).__name__} values must be positive")


@final
class BitDepth(TensorKernel[BitDepthSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not BitDepthSpec:
            raise TypeError("BitDepth requires exact BitDepthSpec")
        if bool((self.tensor < 1).any()) or bool((self.tensor > 16).any()):
            raise ValueError("BitDepth values must be in [1, 16]")


@final
class InputMinimum(TensorKernel[InputMinimumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        _require_finite(self, spec_type=InputMinimumSpec)


@final
class InputMaximum(TensorKernel[InputMaximumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        _require_finite(self, spec_type=InputMaximumSpec)


@final
class AnalogGain(TensorKernel[AnalogGainSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        _require_finite(
            self,
            spec_type=AnalogGainSpec,
            positive=True,
        )
