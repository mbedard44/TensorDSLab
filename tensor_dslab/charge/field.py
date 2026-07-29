"""Semantic specification and Product lifecycle for Charge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, final, override

import torch
from tensor_core import CounterRng, TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.field import require_exact_field_spec
from tensor_dslab.common.requirements.tensor import (
    require_dtype_in,
    require_finite,
    require_nonnegative,
)
from tensor_dslab.common.requirements.unit import require_unit_compatible

if TYPE_CHECKING:
    from tensor_dslab.charge.config import ChargeConfig


@final
class ChargeSpec[AxesT: tuple](QuantityFieldSpec[AxesT]):
    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        require_dtype_in(self, (torch.float32, torch.float64))
        require_unit_compatible(
            self.unit,
            target="avalanche",
            field="ChargeSpec.unit",
        )


@final
class Charge(TensorField[ChargeSpec[Any]]):
    """Represent aggregate nonnegative detector charge."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_field_spec(self, ChargeSpec)
        require_finite(self)
        require_nonnegative(self)

    @classmethod
    def prepare(
        cls,
        *,
        source_specs: tuple[QuantityFieldSpec[Any], ...],
        config: ChargeConfig,
    ) -> ChargeConfig:
        from tensor_dslab.charge.runtime.prepare import prepare_charge
        return prepare_charge(source_specs=source_specs, config=config)

    @classmethod
    def produce(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: ChargeConfig,
        rng: CounterRng,
    ) -> Self:
        from tensor_dslab.charge.runtime.produce import produce_charge
        tensor = produce_charge(sources=sources, config=config, rng=rng)
        return cls(tensor=tensor, spec=config.spec)

    @classmethod
    def validate(
        cls,
        *,
        product: Self,
        sources: tuple[TensorField[Any], ...],
        config: ChargeConfig,
    ) -> None:
        from tensor_dslab.charge.runtime.validate import validate_charge
        validate_charge(product=product, sources=sources, config=config)

    @classmethod
    def create(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: ChargeConfig,
        rng: CounterRng,
    ) -> Self:
        prepared = cls.prepare(
            source_specs=tuple(source.spec for source in sources), config=config
        )
        product = cls.produce(sources=sources, config=prepared, rng=rng)
        cls.validate(product=product, sources=sources, config=prepared)
        return product
