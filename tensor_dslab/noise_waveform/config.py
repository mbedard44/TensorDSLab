"""Noise-waveform coefficient collection and configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.config import require_config_components
from tensor_dslab.noise_waveform.field import NoiseWaveformSpec
from tensor_dslab.noise_waveform.kernel import NoiseWaveformKernels


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class NoiseWaveformConfig:
    """Describe one exact zero, white, or PSD noise transformation."""

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    spec: NoiseWaveformSpec[Any]
    kernels: NoiseWaveformKernels
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
        default=(None, None), init=False, repr=False
    )
    _temporal_dimension: int | None = field(default=None, init=False, repr=False)
    _temporal_step_seconds: float | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        require_config_components(
            spec=self.spec,
            kernels=self.kernels,
            spec_type=NoiseWaveformSpec,
            kernels_type=NoiseWaveformKernels,
            field="NoiseWaveformConfig",
        )
