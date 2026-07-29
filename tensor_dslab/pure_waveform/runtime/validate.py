"""Validate completed PureWaveform relationships."""

from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.common.requirements.field import require_fresh_product
from tensor_dslab.pure_waveform.config import PureWaveformConfig
from tensor_dslab.pure_waveform.field import PureWaveform


def validate_pure_waveform(*, product: PureWaveform, sources: tuple, config: PureWaveformConfig) -> None:
    if type(product) is not PureWaveform:
        raise TypeError("product must be exact PureWaveform")
    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="PureWaveformConfig",
    )
    if product.spec is not config.spec:
        raise ValueError("PureWaveform must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
