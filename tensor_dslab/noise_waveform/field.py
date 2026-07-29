"""Semantic specification and Product lifecycle for NoiseWaveform."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, final, override

import torch
from tensor_core import CounterRng, TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.field import require_exact_field_spec
from tensor_dslab.common.requirements.tensor import (
    require_dtype_in,
    require_finite,
)

if TYPE_CHECKING:
    from tensor_dslab.noise_waveform.config import NoiseWaveformConfig


@final
class NoiseWaveformSpec[AxesT: tuple](QuantityFieldSpec[AxesT]):
    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        require_dtype_in(self, (torch.float32, torch.float64))


@final
class NoiseWaveform(TensorField[NoiseWaveformSpec[Any]]):
    """Represent a finite stochastic electronic-noise waveform."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_field_spec(self, NoiseWaveformSpec)
        require_finite(self)

    @classmethod
    def prepare(
        cls,
        *,
        source_specs: tuple[QuantityFieldSpec[Any], ...],
        config: NoiseWaveformConfig,
    ) -> NoiseWaveformConfig:
        from tensor_dslab.noise_waveform.runtime.prepare import prepare_noise_waveform
        return prepare_noise_waveform(source_specs=source_specs, config=config)

    @classmethod
    def produce(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: NoiseWaveformConfig,
        rng: CounterRng,
    ) -> Self:
        from tensor_dslab.noise_waveform.runtime.produce import produce_noise_waveform
        return cls(tensor=produce_noise_waveform(sources=sources, config=config, rng=rng), spec=config.spec)

    @classmethod
    def validate(
        cls,
        *,
        product: Self,
        sources: tuple[TensorField[Any], ...],
        config: NoiseWaveformConfig,
    ) -> None:
        from tensor_dslab.noise_waveform.runtime.validate import validate_noise_waveform
        validate_noise_waveform(product=product, sources=sources, config=config)

    @classmethod
    def create(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: NoiseWaveformConfig,
        rng: CounterRng,
    ) -> Self:
        prepared = cls.prepare(source_specs=tuple(source.spec for source in sources), config=config)
        product = cls.produce(sources=sources, config=prepared, rng=rng)
        cls.validate(product=product, sources=sources, config=prepared)
        return product
