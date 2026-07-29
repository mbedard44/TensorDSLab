"""Quantity-aware semantic tensor-kernel specifications."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, final, override

import pint
from tensor_core import TensorAxis, TensorKernelSpec

from tensor_dslab.common.units import normalize_unit


@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class QuantityKernelSpec[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](
    TensorKernelSpec[ConditioningAxesT, OperationAxesT],
    ABC,
):
    """Describe one literal physical tensor kernel."""

    unit: pint.Unit

    @final
    @override
    def _require(self) -> None:
        object.__setattr__(self, "unit", normalize_unit(self.unit))
        self._require_quantity_kernel_spec()

    @abstractmethod
    def _require_quantity_kernel_spec(self) -> None:
        """Enforce the concrete physical coefficient contract."""
