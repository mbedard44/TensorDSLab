"""Physical coefficient Specs, kernels, and collection for Charge."""

from typing import Any, final, override

import torch
from tensor_core import OffsetAxis, TensorCollection, TensorKernel

from tensor_dslab.common import QuantityKernelSpec, TimeAxis
from tensor_dslab.common.requirements.collection import (
    require_admitted_member_types,
)
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_no_operation_axes,
    require_nonempty_operation_extents,
    require_offset_bounds,
    require_operation_axes_type,
    require_operation_axis_count,
    require_operation_row_total,
    require_operation_target_count,
)
from tensor_dslab.common.requirements.tensor import (
    require_exact_dtype,
    require_finite,
    require_nonnegative,
)
from tensor_dslab.common.requirements.unit import require_unit_compatible


@final
class TimingJitterSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_exact_dtype(self, torch.float64)
        require_unit_compatible(
            self.unit,
            target="",
            field="TimingJitterSpec.unit",
        )
        require_operation_axis_count(self, minimum=1, maximum=1)
        require_operation_axes_type(self, OffsetAxis)
        require_nonempty_operation_extents(self)
        require_operation_target_count(
            self,
            relative_to=TimeAxis,
            count=1,
        )


@final
class DirectCrosstalkSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_exact_dtype(self, torch.float64)
        require_unit_compatible(
            self.unit,
            target="",
            field="DirectCrosstalkSpec.unit",
        )
        require_operation_axis_count(self, minimum=1)
        require_operation_axes_type(self, OffsetAxis)
        require_nonempty_operation_extents(self)
        require_offset_bounds(
            self,
            relative_to=TimeAxis,
            minimum=0,
            inclusive=True,
        )


@final
class DelayedCrosstalkSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_exact_dtype(self, torch.float64)
        require_unit_compatible(
            self.unit,
            target="",
            field="DelayedCrosstalkSpec.unit",
        )
        require_operation_axis_count(self, minimum=1)
        require_operation_axes_type(self, OffsetAxis)
        require_nonempty_operation_extents(self)
        require_operation_target_count(
            self,
            relative_to=TimeAxis,
            count=1,
        )
        require_offset_bounds(
            self,
            relative_to=TimeAxis,
            minimum=0,
            inclusive=False,
        )


@final
class AfterpulseSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_exact_dtype(self, torch.float64)
        require_unit_compatible(
            self.unit,
            target="",
            field="AfterpulseSpec.unit",
        )
        require_operation_axis_count(self, minimum=1, maximum=1)
        require_operation_axes_type(self, OffsetAxis)
        require_nonempty_operation_extents(self)
        require_operation_target_count(
            self,
            relative_to=TimeAxis,
            count=1,
        )
        require_offset_bounds(
            self,
            relative_to=TimeAxis,
            minimum=0,
            inclusive=False,
        )


@final
class DarkCountRateSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_exact_dtype(self, torch.float64)
        require_unit_compatible(
            self.unit,
            target="avalanche / s",
            field="DarkCountRateSpec.unit",
        )
        require_no_operation_axes(self)


@final
class SmearingWidthSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_exact_dtype(self, torch.float64)
        require_unit_compatible(
            self.unit,
            target="",
            field="SmearingWidthSpec.unit",
        )
        require_no_operation_axes(self)


@final
class TimingJitter(TensorKernel[TimingJitterSpec[Any, Any]]):
    """Represent a complete temporal displacement probability law."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, TimingJitterSpec)
        require_finite(self)
        require_nonnegative(self)
        require_operation_row_total(
            self,
            exact=1.0,
            tolerance=1.0e-11,
        )


@final
class DirectCrosstalk(TensorKernel[DirectCrosstalkSpec[Any, Any]]):
    """Represent prompt expected offspring intensities."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, DirectCrosstalkSpec)
        require_finite(self)
        require_nonnegative(self)
        require_operation_row_total(
            self,
            maximum=1.0,
            tolerance=1.0e-11,
        )


@final
class DelayedCrosstalk(TensorKernel[DelayedCrosstalkSpec[Any, Any]]):
    """Represent delayed expected offspring intensities."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, DelayedCrosstalkSpec)
        require_finite(self)
        require_nonnegative(self)
        require_operation_row_total(
            self,
            maximum=1.0,
            tolerance=1.0e-11,
        )


@final
class Afterpulse(TensorKernel[AfterpulseSpec[Any, Any]]):
    """Represent expected full-charge afterpulse intensities."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, AfterpulseSpec)
        require_finite(self)
        require_nonnegative(self)
        require_operation_row_total(
            self,
            maximum=1.0,
            tolerance=1.0e-11,
        )


@final
class DarkCountRate(TensorKernel[DarkCountRateSpec[Any, Any]]):
    """Represent nonnegative avalanche rates."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, DarkCountRateSpec)
        require_finite(self)
        require_nonnegative(self)


@final
class SmearingWidth(TensorKernel[SmearingWidthSpec[Any, Any]]):
    """Represent relative Gaussian charge-response widths."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, SmearingWidthSpec)
        require_finite(self)
        require_nonnegative(self)


@final
class ChargeKernels(TensorCollection[TensorKernel[Any]]):
    """Hold the optional physical coefficient set for Charge."""

    __slots__ = ()

    def _require(self) -> None:
        require_admitted_member_types(
            self,
            admitted=(
                TimingJitter,
                DirectCrosstalk,
                DelayedCrosstalk,
                Afterpulse,
                DarkCountRate,
                SmearingWidth,
            ),
        )

    def _optional[T](self, member_type: type[T]) -> T | None:
        return self.members.get(member_type)  # type: ignore[return-value]

    @property
    def timing_jitter(self) -> TimingJitter | None:
        return self._optional(TimingJitter)

    @property
    def direct_crosstalk(self) -> DirectCrosstalk | None:
        return self._optional(DirectCrosstalk)

    @property
    def delayed_crosstalk(self) -> DelayedCrosstalk | None:
        return self._optional(DelayedCrosstalk)

    @property
    def afterpulse(self) -> Afterpulse | None:
        return self._optional(Afterpulse)

    @property
    def dark_count_rate(self) -> DarkCountRate | None:
        return self._optional(DarkCountRate)

    @property
    def smearing_width(self) -> SmearingWidth | None:
        return self._optional(SmearingWidth)
