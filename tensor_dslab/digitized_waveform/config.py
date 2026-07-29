"""Digitizer coefficient collection and configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.config import require_config_components
from tensor_dslab.digitized_waveform.field import DigitizedWaveformSpec
from tensor_dslab.digitized_waveform.kernel import DigitizedWaveformKernels


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class DigitizedWaveformConfig:
    """Describe one pointwise linear ADC transformation."""

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    spec: DigitizedWaveformSpec[Any]
    kernels: DigitizedWaveformKernels
    _is_prepared: bool = field(default=False, init=False, repr=False)
    _source_specs: tuple[QuantityFieldSpec[Any], ...] = field(
        default=(), init=False, repr=False
    )
    _source_dimensions: tuple[tuple[int, ...], ...] = field(
        default=(), init=False, repr=False
    )
    _source_scales: tuple[float, ...] = field(default=(), init=False, repr=False)
    _working_dtype: torch.dtype | None = field(
        default=None, init=False, repr=False
    )
    _kernel_dimensions: tuple[tuple[int, ...] | None, ...] = field(
        default=(None, None, None, None), init=False, repr=False
    )

    def __post_init__(self) -> None:
        require_config_components(
            spec=self.spec,
            kernels=self.kernels,
            spec_type=DigitizedWaveformSpec,
            kernels_type=DigitizedWaveformKernels,
            field="DigitizedWaveformConfig",
        )
