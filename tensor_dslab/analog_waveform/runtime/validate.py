"""Validate completed AnalogWaveform relationships."""

from tensor_dslab.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.analog_waveform.field import AnalogWaveform
from tensor_dslab.common.alignment import (
    require_fresh_product,
    require_prepared_sources,
)


def validate_analog_waveform(*, product: AnalogWaveform, sources: tuple, config: AnalogWaveformConfig) -> None:
    if type(product) is not AnalogWaveform:
        raise TypeError("product must be exact AnalogWaveform")
    if not config._is_prepared or product.spec is not config.spec:
        raise ValueError("AnalogWaveform must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
