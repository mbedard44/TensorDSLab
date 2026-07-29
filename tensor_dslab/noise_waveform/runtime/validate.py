"""Validate completed NoiseWaveform relationships."""

from tensor_dslab.common.alignment import (
    require_fresh_product,
    require_prepared_sources,
)
from tensor_dslab.noise_waveform.config import NoiseWaveformConfig
from tensor_dslab.noise_waveform.field import NoiseWaveform


def validate_noise_waveform(*, product: NoiseWaveform, sources: tuple, config: NoiseWaveformConfig) -> None:
    if type(product) is not NoiseWaveform:
        raise TypeError("product must be exact NoiseWaveform")
    if not config._is_prepared or product.spec is not config.spec:
        raise ValueError("NoiseWaveform must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
