"""Validate completed PureWaveform relationships."""

from tensor_dslab.common.alignment import (
    require_fresh_product,
    require_prepared_sources,
)
from tensor_dslab.pure_waveform.config import PureWaveformConfig
from tensor_dslab.pure_waveform.field import PureWaveform


def validate_pure_waveform(*, product: PureWaveform, sources: tuple, config: PureWaveformConfig) -> None:
    if type(product) is not PureWaveform:
        raise TypeError("product must be exact PureWaveform")
    if not config._is_prepared or product.spec is not config.spec:
        raise ValueError("PureWaveform must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
