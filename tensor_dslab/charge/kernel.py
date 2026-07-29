"""Physical coefficient Specs and kernels for Charge."""

import math
from typing import Any, cast, final, override

import torch
from tensor_core import OffsetAxis, TensorKernel

from tensor_dslab.common import QuantityKernelSpec, TimeAxis


@final
class TimingJitterSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        pass


@final
class DirectCrosstalkSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        pass


@final
class DelayedCrosstalkSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        pass


@final
class AfterpulseSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        pass


@final
class DarkCountRateSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        pass


@final
class SmearingWidthSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        pass


def _require_unit(kernel: TensorKernel[Any], target: str, field: str) -> None:
    try:
        (1 * kernel.spec.unit).to(target)
    except Exception as error:
        raise ValueError(f"{field} unit is incompatible with {target}") from error


def _offsets(axis: OffsetAxis) -> tuple[int, ...]:
    return axis.coordinates.offsets


def _rows(kernel: TensorKernel[Any]) -> tuple[tuple[float, ...], ...]:
    operation_count = kernel.spec.operation_element_count
    if operation_count == 0:
        return ()
    cpu = kernel.tensor.detach().to(device="cpu", dtype=torch.float64)
    return tuple(
        tuple(float(value) for value in row)
        for row in cpu.reshape(-1, operation_count).tolist()
    )


def _require_probability_kernel(
    kernel: TensorKernel[Any],
    *,
    complete: bool,
    delayed: bool,
    exactly_one_time: bool,
    allow_negative_time: bool = False,
) -> None:
    if not kernel.operation_axes or any(
        type(axis) is not OffsetAxis for axis in kernel.operation_axes
    ):
        raise ValueError(f"{type(kernel).__name__} requires OffsetAxis operations")
    if any(axis.size == 0 for axis in kernel.operation_axes):
        raise ValueError(
            f"{type(kernel).__name__} operation axes must be nonempty"
        )
    operation_axes = tuple(cast(OffsetAxis, axis) for axis in kernel.operation_axes)
    time_axes = [
        axis for axis in operation_axes if axis.relative_to is TimeAxis
    ]
    if exactly_one_time and len(time_axes) != 1:
        raise ValueError(f"{type(kernel).__name__} requires one TimeAxis target")
    if delayed and (
        len(time_axes) != 1 or any(offset <= 0 for offset in _offsets(time_axes[0]))
    ):
        raise ValueError(f"{type(kernel).__name__} time offsets must be positive")
    if not delayed and not allow_negative_time and any(
        offset < 0
        for axis in time_axes
        for offset in _offsets(axis)
    ):
        raise ValueError(f"{type(kernel).__name__} time offsets must be nonnegative")
    for row in _rows(kernel):
        if any(not math.isfinite(value) or value < 0 for value in row):
            raise ValueError(f"{type(kernel).__name__} values must be finite and nonnegative")
        total = math.fsum(row)
        if complete and abs(total - 1.0) > 1.0e-11:
            raise ValueError("TimingJitter rows must sum to one")
        if not complete and total > 1.0 + 1.0e-11:
            raise ValueError(f"{type(kernel).__name__} row total exceeds one")


@final
class TimingJitter(TensorKernel[TimingJitterSpec[Any, Any]]):
    """Represent a complete temporal displacement probability law."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not TimingJitterSpec:
            raise TypeError("TimingJitter requires exact TimingJitterSpec")
        _require_unit(self, "", "TimingJitter")
        if self.dtype is not torch.float64 or len(self.operation_axes) != 1:
            raise TypeError("TimingJitter requires binary64 and one operation axis")
        _require_probability_kernel(
            self,
            complete=True,
            delayed=False,
            exactly_one_time=True,
            allow_negative_time=True,
        )


@final
class DirectCrosstalk(TensorKernel[DirectCrosstalkSpec[Any, Any]]):
    """Represent prompt expected offspring intensities."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not DirectCrosstalkSpec:
            raise TypeError(
                "DirectCrosstalk requires exact DirectCrosstalkSpec"
            )
        _require_unit(self, "", "DirectCrosstalk")
        if self.dtype is not torch.float64:
            raise TypeError("DirectCrosstalk requires torch.float64")
        _require_probability_kernel(
            self, complete=False, delayed=False, exactly_one_time=False
        )


@final
class DelayedCrosstalk(TensorKernel[DelayedCrosstalkSpec[Any, Any]]):
    """Represent delayed expected offspring intensities."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not DelayedCrosstalkSpec:
            raise TypeError(
                "DelayedCrosstalk requires exact DelayedCrosstalkSpec"
            )
        _require_unit(self, "", "DelayedCrosstalk")
        if self.dtype is not torch.float64:
            raise TypeError("DelayedCrosstalk requires torch.float64")
        _require_probability_kernel(
            self, complete=False, delayed=True, exactly_one_time=True
        )


@final
class Afterpulse(TensorKernel[AfterpulseSpec[Any, Any]]):
    """Represent expected full-charge afterpulse intensities."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not AfterpulseSpec:
            raise TypeError("Afterpulse requires exact AfterpulseSpec")
        _require_unit(self, "", "Afterpulse")
        if self.dtype is not torch.float64 or len(self.operation_axes) != 1:
            raise TypeError("Afterpulse requires binary64 and one operation axis")
        _require_probability_kernel(
            self, complete=False, delayed=True, exactly_one_time=True
        )


@final
class DarkCountRate(TensorKernel[DarkCountRateSpec[Any, Any]]):
    """Represent nonnegative avalanche rates."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not DarkCountRateSpec:
            raise TypeError("DarkCountRate requires exact DarkCountRateSpec")
        _require_unit(self, "avalanche / s", "DarkCountRate")
        if self.dtype is not torch.float64 or self.operation_axes:
            raise TypeError("DarkCountRate requires binary64 and no operations")
        if not bool(torch.isfinite(self.tensor).all()) or bool((self.tensor < 0).any()):
            raise ValueError("DarkCountRate values must be finite and nonnegative")


@final
class SmearingWidth(TensorKernel[SmearingWidthSpec[Any, Any]]):
    """Represent relative Gaussian charge-response widths."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not SmearingWidthSpec:
            raise TypeError("SmearingWidth requires exact SmearingWidthSpec")
        _require_unit(self, "", "SmearingWidth")
        if self.dtype is not torch.float64 or self.operation_axes:
            raise TypeError("SmearingWidth requires binary64 and no operations")
        if not bool(torch.isfinite(self.tensor).all()) or bool((self.tensor < 0).any()):
            raise ValueError("SmearingWidth values must be finite and nonnegative")
