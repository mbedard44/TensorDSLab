"""Semantic specification and Product lifecycle for Charge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, final, override

import torch
from tensor_core import CounterRng, TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.units import unit_registry

if TYPE_CHECKING:
    from tensor_dslab.charge.config import ChargeConfig


@final
class ChargeSpec[AxesT: tuple](QuantityFieldSpec[AxesT]):
    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError("ChargeSpec dtype must be torch.float32 or torch.float64")
        try:
            unit_registry.Quantity(1.0, self.unit).to("avalanche")
        except Exception as error:
            raise ValueError("ChargeSpec unit must be avalanche-compatible") from error


@final
class Charge(TensorField[ChargeSpec[Any]]):
    """Represent aggregate nonnegative detector charge."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not ChargeSpec:
            raise TypeError("Charge requires exact ChargeSpec")
        if not bool(torch.isfinite(self.tensor).all()) or bool((self.tensor < 0).any()):
            raise ValueError("Charge values must be finite and nonnegative")

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
