"""Semantic specification and Product lifecycle for EncodedWaveform."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, final, override

from tensor_core import TensorAxis, TensorField

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.field import require_exact_field_spec
from tensor_dslab.common.requirements.tensor import (
    require_encoded_values,
    require_negative_representable_suppression_code,
    require_signed_integer_dtype,
)
from tensor_dslab.common.requirements.unit import require_unit_compatible

if TYPE_CHECKING:
    from tensor_dslab.encoded_waveform.config import EncodedWaveformConfig


def _require_source_specs(source_specs: object) -> None:
    """Require the public prepared-source representation before semantics."""

    if type(source_specs) is not tuple:
        raise TypeError("source_specs must be exactly tuple")
    for index, source_spec in enumerate(source_specs):
        if not isinstance(source_spec, QuantityFieldSpec):
            raise TypeError(
                f"source_specs[{index}] must be a QuantityFieldSpec"
            )


def _require_sources(sources: object) -> None:
    """Require the public source representation before lifecycle effects."""

    if type(sources) is not tuple:
        raise TypeError("sources must be exactly tuple")
    for index, source in enumerate(sources):
        if not isinstance(source, TensorField):
            raise TypeError(f"sources[{index}] must be a TensorField")


def _require_config(config: object) -> None:
    """Require the exact public Config representation before lifecycle effects."""

    from tensor_dslab.encoded_waveform.config import EncodedWaveformConfig

    if type(config) is not EncodedWaveformConfig:
        raise TypeError("config must be exactly EncodedWaveformConfig")


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class EncodedWaveformSpec[
    AxesT: tuple[TensorAxis[Any], ...],
](QuantityFieldSpec[AxesT]):
    """Describe dense retained ADC codes with one explicit suppression value."""

    suppression_code: int

    @override
    def _require_quantity_field_spec(self) -> None:
        require_signed_integer_dtype(self)
        require_unit_compatible(
            self.unit,
            target="",
            field="EncodedWaveformSpec.unit",
        )
        require_negative_representable_suppression_code(self)


@final
class EncodedWaveform(TensorField[EncodedWaveformSpec[Any]]):
    """Represent retained ADC codes and explicit suppressed samples."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_field_spec(self, EncodedWaveformSpec)
        require_encoded_values(self)

    @classmethod
    def prepare(
        cls,
        *,
        source_specs: tuple[QuantityFieldSpec[Any], ...],
        config: EncodedWaveformConfig,
    ) -> EncodedWaveformConfig:
        _require_source_specs(source_specs)
        _require_config(config)
        from tensor_dslab.encoded_waveform.runtime.prepare import (
            prepare_encoded_waveform,
        )

        return prepare_encoded_waveform(
            source_specs=source_specs,
            config=config,
        )

    @classmethod
    def produce(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: EncodedWaveformConfig,
    ) -> Self:
        _require_sources(sources)
        _require_config(config)
        from tensor_dslab.encoded_waveform.runtime.produce import (
            produce_encoded_waveform,
        )

        return cls(
            tensor=produce_encoded_waveform(sources=sources, config=config),
            spec=config.spec,
        )

    @classmethod
    def validate(
        cls,
        *,
        product: Self,
        sources: tuple[TensorField[Any], ...],
        config: EncodedWaveformConfig,
    ) -> None:
        _require_sources(sources)
        _require_config(config)
        from tensor_dslab.encoded_waveform.runtime.validate import (
            validate_encoded_waveform,
        )

        validate_encoded_waveform(
            product=product,
            sources=sources,
            config=config,
        )

    @classmethod
    def create(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: EncodedWaveformConfig,
    ) -> Self:
        _require_sources(sources)
        _require_config(config)
        prepared = cls.prepare(
            source_specs=tuple(source.spec for source in sources),
            config=config,
        )
        product = cls.produce(sources=sources, config=prepared)
        cls.validate(product=product, sources=sources, config=prepared)
        return product
