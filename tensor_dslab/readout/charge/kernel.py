"""Public physical kernels for charge simulation."""

import math
from typing import final, override

import torch
from tensor_core import OffsetAxis

from tensor_dslab.common import ExampleAxis, SampleAxis
from tensor_dslab.common.kernel import QuantityKernel


def _require_nonnegative(kernel: QuantityKernel, field: str) -> None:
    if bool((kernel.tensor < 0).any()):
        raise ValueError(f"{field} values must be nonnegative")


def _require_sample_operation(
    kernel: QuantityKernel,
    *,
    positive: bool,
) -> OffsetAxis:
    if len(kernel.operation_axes) != 1:
        raise ValueError(f"{type(kernel).__name__} requires one operation axis")
    axis = kernel.operation_axes[0]
    if type(axis) is not OffsetAxis or axis.relative_to is not SampleAxis:
        raise ValueError(
            f"{type(kernel).__name__} operation axis must target SampleAxis"
        )
    if not axis.offsets:
        raise ValueError(f"{type(kernel).__name__} operation axis must be nonempty")
    if positive and any(offset <= 0 for offset in axis.offsets):
        raise ValueError(
            f"{type(kernel).__name__} sample offsets must be positive"
        )
    return axis


def _require_branching(kernel: QuantityKernel, *, delayed: bool) -> None:
    _require_nonnegative(kernel, type(kernel).__name__)
    if not kernel.operation_axes:
        raise ValueError(f"{type(kernel).__name__} operation geometry must be nonempty")
    if any(type(axis) is not OffsetAxis for axis in kernel.operation_axes):
        raise TypeError(
            f"{type(kernel).__name__} operation axes must be OffsetAxis values"
        )
    if any(axis.relative_to is ExampleAxis for axis in kernel.operation_axes):
        raise ValueError(f"{type(kernel).__name__} cannot target ExampleAxis")
    sample_axes = [
        axis
        for axis in kernel.operation_axes
        if axis.relative_to is SampleAxis
    ]
    if len(sample_axes) != 1:
        raise ValueError(f"{type(kernel).__name__} requires one SampleAxis target")
    minimum = 1 if delayed else 0
    if any(offset < minimum for offset in sample_axes[0].offsets):
        qualifier = "positive" if delayed else "nonnegative"
        raise ValueError(
            f"{type(kernel).__name__} sample offsets must be {qualifier}"
        )


@final
class DarkCountRate[ConditioningAxesT: tuple](QuantityKernel):
    """Represent finite nonnegative dark-count rates."""

    canonical_unit = "Hz"

    @override
    def _require(self) -> None:
        if self.operation_axes:
            raise ValueError("DarkCountRate has no operation axes")
        _require_nonnegative(self, "DarkCountRate")


@final
class TimingJitter[ConditioningAxesT: tuple](QuantityKernel):
    """Represent a complete finite sample-offset probability law."""

    canonical_unit = "dimensionless"

    @override
    def _require(self) -> None:
        _require_nonnegative(self, "TimingJitter")
        _require_sample_operation(self, positive=False)
        totals = self.tensor.sum(dim=-1)
        if bool((torch.abs(totals - 1.0) > 1.0e-11).any()):
            raise ValueError("TimingJitter probabilities must form a complete law")


@final
class DirectCrosstalk[ConditioningAxesT: tuple, OperationAxesT: tuple](
    QuantityKernel
):
    """Represent expected prompt crosstalk offspring by destination offset."""

    canonical_unit = "dimensionless"

    @override
    def _require(self) -> None:
        _require_branching(self, delayed=False)


@final
class DelayedCrosstalk[ConditioningAxesT: tuple, OperationAxesT: tuple](
    QuantityKernel
):
    """Represent expected delayed crosstalk offspring by destination offset."""

    canonical_unit = "dimensionless"

    @override
    def _require(self) -> None:
        _require_branching(self, delayed=True)


@final
class Afterpulse[ConditioningAxesT: tuple](QuantityKernel):
    """Represent expected full-charge afterpulses by positive sample offset."""

    canonical_unit = "dimensionless"

    @override
    def _require(self) -> None:
        _require_nonnegative(self, "Afterpulse")
        _require_sample_operation(self, positive=True)


@final
class SmearingWidth[ConditioningAxesT: tuple](QuantityKernel):
    """Represent relative one-photoelectron Gaussian response width."""

    canonical_unit = "dimensionless"

    @override
    def _require(self) -> None:
        if self.operation_axes:
            raise ValueError("SmearingWidth has no operation axes")
        _require_nonnegative(self, "SmearingWidth")
