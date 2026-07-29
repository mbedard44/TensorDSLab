"""Noise-waveform coefficient collection and configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch
from tensor_core import TensorCollection, TensorKernel

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.noise_waveform.field import NoiseWaveformSpec
from tensor_dslab.noise_waveform.kernel import (
    PowerSpectralDensity,
    WhiteNoiseRms,
)


@final
class NoiseWaveformKernels(TensorCollection[TensorKernel[Any]]):
    __slots__ = ()

    def _require(self) -> None:
        admitted = {WhiteNoiseRms, PowerSpectralDensity}
        if any(type(member) not in admitted for member in self.members.values()):
            raise TypeError("NoiseWaveformKernels contains an unsupported member")
        if len(self.members) > 1:
            raise ValueError("NoiseWaveformKernels admits at most one branch")

    @property
    def white_noise_rms(self) -> WhiteNoiseRms | None:
        return self.members.get(WhiteNoiseRms)  # type: ignore[return-value]

    @property
    def power_spectral_density(self) -> PowerSpectralDensity | None:
        return self.members.get(PowerSpectralDensity)  # type: ignore[return-value]


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
        if type(self.spec) is not NoiseWaveformSpec:
            raise TypeError(
                "NoiseWaveformConfig.spec must be exact NoiseWaveformSpec"
            )
        if type(self.kernels) is not NoiseWaveformKernels:
            raise TypeError(
                "NoiseWaveformConfig.kernels must be exact NoiseWaveformKernels"
            )
