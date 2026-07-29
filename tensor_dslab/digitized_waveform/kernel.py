"""Digitizer coefficient Specs, kernels, and collection."""

from typing import Any, final, override

from tensor_core import TensorCollection, TensorKernel, TensorKernelSpec

from tensor_dslab.common import QuantityKernelSpec
from tensor_dslab.common.requirements.collection import (
    require_exact_member_types,
)
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_no_operation_axes,
)
from tensor_dslab.common.requirements.tensor import (
    require_finite,
    require_floating_dtype,
    require_positive,
    require_signed_integer_dtype,
    require_values_between,
)
from tensor_dslab.common.requirements.unit import require_unit_compatible


@final
class InputMinimumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_floating_dtype(self)
        require_no_operation_axes(self)


@final
class InputMaximumSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_floating_dtype(self)
        require_no_operation_axes(self)


@final
class AnalogGainSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_floating_dtype(self)
        require_unit_compatible(
            self.unit,
            target="",
            field="AnalogGainSpec.unit",
        )
        require_no_operation_axes(self)


@final
class BitDepthSpec[C: tuple, O: tuple](TensorKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_signed_integer_dtype(self)
        require_no_operation_axes(self)


@final
class BitDepth(TensorKernel[BitDepthSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, BitDepthSpec)
        require_values_between(self, minimum=1, maximum=16)


@final
class InputMinimum(TensorKernel[InputMinimumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, InputMinimumSpec)
        require_finite(self)


@final
class InputMaximum(TensorKernel[InputMaximumSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, InputMaximumSpec)
        require_finite(self)


@final
class AnalogGain(TensorKernel[AnalogGainSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, AnalogGainSpec)
        require_finite(self)
        require_positive(self)


@final
class DigitizedWaveformKernels(TensorCollection[TensorKernel[Any]]):
    """Hold the exact digitizer coefficient set."""

    __slots__ = ()

    def _require(self) -> None:
        require_exact_member_types(
            self,
            required=(BitDepth, InputMinimum, InputMaximum, AnalogGain),
        )

    @property
    def bit_depth(self) -> BitDepth:
        return self.member(BitDepth)

    @property
    def input_minimum(self) -> InputMinimum:
        return self.member(InputMinimum)

    @property
    def input_maximum(self) -> InputMaximum:
        return self.member(InputMaximum)

    @property
    def analog_gain(self) -> AnalogGain:
        return self.member(AnalogGain)
