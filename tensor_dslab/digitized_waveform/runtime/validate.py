"""Validate completed DigitizedWaveform relationships."""

import torch

from tensor_dslab.common.alignment import (
    require_fresh_product,
    require_prepared_sources,
)
from tensor_dslab.digitized_waveform.config import DigitizedWaveformConfig
from tensor_dslab.digitized_waveform.field import DigitizedWaveform


def validate_digitized_waveform(*, product: DigitizedWaveform, sources: tuple, config: DigitizedWaveformConfig) -> None:
    if type(product) is not DigitizedWaveform:
        raise TypeError("product must be exact DigitizedWaveform")
    if not config._is_prepared or product.spec is not config.spec:
        raise ValueError("DigitizedWaveform must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    shape = [1] * len(config.spec.shape)
    dimensions = config._kernel_dimensions[0]
    assert dimensions is not None
    for source_dimension, target_dimension in enumerate(dimensions):
        shape[target_dimension] = config.kernels.bit_depth.tensor.shape[
            source_dimension
        ]
    maximum_code = (
        1
        << config.kernels.bit_depth.tensor.to(torch.int64).reshape(shape)
    ) - 1
    if bool((product.tensor > maximum_code).any()):
        raise ValueError("DigitizedWaveform values exceed configured bit depth")
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
