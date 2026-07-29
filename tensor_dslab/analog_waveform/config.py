"""Analog-waveform coefficient collection and configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch

from tensor_dslab.analog_waveform.field import AnalogWaveformSpec
from tensor_dslab.analog_waveform.kernel import AnalogWaveformKernels
from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.config import require_config_components


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class AnalogWaveformConfig:
    """Describe ordered physical-field addition and saturation."""

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    spec: AnalogWaveformSpec[Any]
    kernels: AnalogWaveformKernels
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

    def __post_init__(self) -> None:
        require_config_components(
            spec=self.spec,
            kernels=self.kernels,
            spec_type=AnalogWaveformSpec,
            kernels_type=AnalogWaveformKernels,
            field="AnalogWaveformConfig",
        )
