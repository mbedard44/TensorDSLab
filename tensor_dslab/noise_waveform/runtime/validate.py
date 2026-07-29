"""Validate completed NoiseWaveform relationships."""

from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.common.requirements.field import require_fresh_product
from tensor_dslab.noise_waveform.config import NoiseWaveformConfig
from tensor_dslab.noise_waveform.field import NoiseWaveform


def validate_noise_waveform(*, product: NoiseWaveform, sources: tuple, config: NoiseWaveformConfig) -> None:
    if type(product) is not NoiseWaveform:
        raise TypeError("product must be exact NoiseWaveform")
    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="NoiseWaveformConfig",
    )
    if product.spec is not config.spec:
        raise ValueError("NoiseWaveform must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
