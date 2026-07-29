"""Semantic specification and field for source photoelectrons."""

from __future__ import annotations

from typing import Any, Self, final, override

import torch
from tensor_core import TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.units import unit_registry


@final
class PhotoelectronsSpec[AxesT: tuple](QuantityFieldSpec[AxesT]):
    """Specify one exact represented photoelectron-count field."""

    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        if self.dtype is not torch.int64:
            raise TypeError("PhotoelectronsSpec dtype must be torch.int64")
        try:
            unit_registry.Quantity(1.0, self.unit).to("avalanche")
        except Exception as error:
            raise ValueError(
                "PhotoelectronsSpec unit must be avalanche-compatible"
            ) from error


@final
class Photoelectrons(TensorField[PhotoelectronsSpec[Any]]):
    """Represent already-produced nonnegative photoelectron counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not PhotoelectronsSpec:
            raise TypeError("Photoelectrons requires exact PhotoelectronsSpec")
        from tensor_dslab.photoelectrons.runtime.validate import (
            validate_photoelectrons,
        )

        validate_photoelectrons(product=self)

    @classmethod
    def validate(cls, *, product: Self) -> None:
        """Validate one caller-constructed source field."""

        from tensor_dslab.photoelectrons.runtime.validate import (
            validate_photoelectrons,
        )

        validate_photoelectrons(product=product)
