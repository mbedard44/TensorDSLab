"""Public physical tensor-kernel representation."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import pint
from pint import Quantity
import torch
from tensor_core import TensorAxis, TensorKernel

from tensor_dslab.common.units import _REGISTRY, _canonical_tensor_quantity


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    init=False,
    kw_only=True,
)
class QuantityKernel[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](
    TensorKernel[ConditioningAxesT, OperationAxesT],
    ABC,
):
    """Own a canonical physical quantity over literal semantic kernel axes."""

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]
    canonical_unit: ClassVar[str]
    _unit: pint.Unit

    def __init__(
        self,
        *,
        quantity: Quantity,
        conditioning_axes: ConditioningAxesT,
        operation_axes: OperationAxesT,
    ) -> None:
        magnitude, unit = _canonical_tensor_quantity(
            quantity,
            unit=self.canonical_unit,
            field=f"{type(self).__name__}.quantity",
        )
        object.__setattr__(self, "_unit", unit)
        TensorKernel.__init__(
            self,
            tensor=magnitude,
            conditioning_axes=conditioning_axes,
            operation_axes=operation_axes,
        )

    @property
    def quantity(self) -> Quantity:
        """Return a read-only canonical Quantity view of the owned magnitude."""

        magnitude = self.tensor.numpy()
        magnitude.setflags(write=False)
        return cast(Quantity, _REGISTRY.Quantity(magnitude, self._unit))
