"""Semantic specification and field for source photoelectrons."""

from __future__ import annotations

from typing import Any, Self, final, override

import torch
from tensor_core import TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.field import require_exact_field_spec
from tensor_dslab.common.requirements.tensor import (
    require_exact_dtype,
    require_values_between,
)
from tensor_dslab.common.requirements.unit import require_unit_compatible


@final
class PhotoelectronsSpec[AxesT: tuple](QuantityFieldSpec[AxesT]):
    """Specify one exact represented photoelectron-count field."""

    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        require_exact_dtype(self, torch.int64)
        require_unit_compatible(
            self.unit,
            target="avalanche",
            field="PhotoelectronsSpec.unit",
        )


@final
class Photoelectrons(TensorField[PhotoelectronsSpec[Any]]):
    """Represent already-produced nonnegative photoelectron counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_field_spec(self, PhotoelectronsSpec)
        require_values_between(
            self,
            minimum=0,
            maximum=(1 << 53) - 1,
        )

    @classmethod
    def validate(cls, *, product: Self) -> None:
        """Validate one caller-constructed source field."""

        from tensor_dslab.photoelectrons.runtime.validate import (
            validate_photoelectrons,
        )

        validate_photoelectrons(product=product)
