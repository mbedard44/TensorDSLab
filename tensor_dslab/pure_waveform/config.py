"""Pure-waveform coefficient collection and configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch
from tensor_core import TensorCollection, TensorKernel

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.pure_waveform.field import PureWaveformSpec
from tensor_dslab.pure_waveform.kernel import PulseResponse


@final
class PureWaveformKernels(TensorCollection[TensorKernel[Any]]):
    __slots__ = ()

    def _require(self) -> None:
        if self.member_types != frozenset((PulseResponse,)):
            raise ValueError("PureWaveformKernels requires exactly PulseResponse")

    @property
    def pulse_response(self) -> PulseResponse:
        return self.member(PulseResponse)


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class PureWaveformConfig:
    """Describe one deterministic pulse convolution."""

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    spec: PureWaveformSpec[Any]
    kernels: PureWaveformKernels
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
        default=(None,), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if type(self.spec) is not PureWaveformSpec:
            raise TypeError(
                "PureWaveformConfig.spec must be exact PureWaveformSpec"
            )
        if type(self.kernels) is not PureWaveformKernels:
            raise TypeError(
                "PureWaveformConfig.kernels must be exact PureWaveformKernels"
            )
