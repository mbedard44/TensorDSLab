"""Semantic specification and Product lifecycle for DigitizedWaveform."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, final, override

import torch
from tensor_core import TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.units import unit_registry

if TYPE_CHECKING:
    from tensor_dslab.digitized_waveform.config import DigitizedWaveformConfig


@final
class DigitizedWaveformSpec[AxesT: tuple](QuantityFieldSpec[AxesT]):
    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        if self.dtype is not torch.int32:
            raise TypeError("DigitizedWaveformSpec dtype must be torch.int32")
        try:
            unit_registry.Quantity(1.0, self.unit).to("")
        except Exception as error:
            raise ValueError("DigitizedWaveformSpec unit must be dimensionless") from error


@final
class DigitizedWaveform(TensorField[DigitizedWaveformSpec[Any]]):
    """Represent nonnegative integer ADC codes."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not DigitizedWaveformSpec:
            raise TypeError(
                "DigitizedWaveform requires exact DigitizedWaveformSpec"
            )
        if bool((self.tensor < 0).any()):
            raise ValueError("DigitizedWaveform values must be nonnegative")

    @classmethod
    def prepare(
        cls,
        *,
        source_specs: tuple[QuantityFieldSpec[Any], ...],
        config: DigitizedWaveformConfig,
    ) -> DigitizedWaveformConfig:
        from tensor_dslab.digitized_waveform.runtime.prepare import prepare_digitized_waveform
        return prepare_digitized_waveform(source_specs=source_specs, config=config)

    @classmethod
    def produce(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: DigitizedWaveformConfig,
    ) -> Self:
        from tensor_dslab.digitized_waveform.runtime.produce import produce_digitized_waveform
        return cls(tensor=produce_digitized_waveform(sources=sources, config=config), spec=config.spec)

    @classmethod
    def validate(
        cls,
        *,
        product: Self,
        sources: tuple[TensorField[Any], ...],
        config: DigitizedWaveformConfig,
    ) -> None:
        from tensor_dslab.digitized_waveform.runtime.validate import validate_digitized_waveform
        validate_digitized_waveform(product=product, sources=sources, config=config)

    @classmethod
    def create(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: DigitizedWaveformConfig,
    ) -> Self:
        prepared = cls.prepare(source_specs=tuple(source.spec for source in sources), config=config)
        product = cls.produce(sources=sources, config=prepared)
        cls.validate(product=product, sources=sources, config=prepared)
        return product
