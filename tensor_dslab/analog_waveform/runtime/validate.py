"""Validate completed AnalogWaveform relationships."""

from tensor_dslab.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.analog_waveform.field import AnalogWaveform
from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.common.requirements.field import require_fresh_product


def validate_analog_waveform(*, product: AnalogWaveform, sources: tuple, config: AnalogWaveformConfig) -> None:
    if type(product) is not AnalogWaveform:
        raise TypeError("product must be exact AnalogWaveform")
    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="AnalogWaveformConfig",
    )
    if product.spec is not config.spec:
        raise ValueError("AnalogWaveform must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
