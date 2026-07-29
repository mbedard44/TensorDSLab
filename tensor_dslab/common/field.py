"""Quantity-aware semantic tensor-field specifications."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, final, override

import pint
from tensor_core import TensorAxis, TensorFieldSpec

from tensor_dslab.common.units import _normalize_unit


@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class QuantityFieldSpec[
    AxesT: tuple[TensorAxis[Any], ...],
](TensorFieldSpec[AxesT], ABC):
    """Describe one physical tensor field without materializing a tensor."""

    unit: pint.Unit

    @final
    @override
    def _require(self) -> None:
        object.__setattr__(self, "unit", _normalize_unit(self.unit))
        self._require_quantity_field_spec()

    @abstractmethod
    def _require_quantity_field_spec(self) -> None:
        """Enforce the concrete physical product specification."""
