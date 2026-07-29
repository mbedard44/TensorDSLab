"""Semantic specification and Product lifecycle for PureWaveform."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, final, override

import torch
from tensor_core import TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.field import require_exact_field_spec
from tensor_dslab.common.requirements.tensor import (
    require_dtype_in,
    require_finite,
)

if TYPE_CHECKING:
    from tensor_dslab.pure_waveform.config import PureWaveformConfig


@final
class PureWaveformSpec[AxesT: tuple](QuantityFieldSpec[AxesT]):
    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        require_dtype_in(self, (torch.float32, torch.float64))


@final
class PureWaveform(TensorField[PureWaveformSpec[Any]]):
    """Represent a deterministic physical pulse response."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_field_spec(self, PureWaveformSpec)
        require_finite(self)

    @classmethod
    def prepare(
        cls,
        *,
        source_specs: tuple[QuantityFieldSpec[Any], ...],
        config: PureWaveformConfig,
    ) -> PureWaveformConfig:
        from tensor_dslab.pure_waveform.runtime.prepare import prepare_pure_waveform
        return prepare_pure_waveform(source_specs=source_specs, config=config)

    @classmethod
    def produce(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: PureWaveformConfig,
    ) -> Self:
        from tensor_dslab.pure_waveform.runtime.produce import produce_pure_waveform
        return cls(tensor=produce_pure_waveform(sources=sources, config=config), spec=config.spec)

    @classmethod
    def validate(
        cls,
        *,
        product: Self,
        sources: tuple[TensorField[Any], ...],
        config: PureWaveformConfig,
    ) -> None:
        from tensor_dslab.pure_waveform.runtime.validate import validate_pure_waveform
        validate_pure_waveform(product=product, sources=sources, config=config)

    @classmethod
    def create(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: PureWaveformConfig,
    ) -> Self:
        prepared = cls.prepare(source_specs=tuple(source.spec for source in sources), config=config)
        product = cls.produce(sources=sources, config=prepared)
        cls.validate(product=product, sources=sources, config=prepared)
        return product
