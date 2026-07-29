"""Analog-waveform coefficient collection and configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch
from tensor_core import TensorCollection, TensorKernel

from tensor_dslab.analog_waveform.field import AnalogWaveformSpec
from tensor_dslab.analog_waveform.kernel import AnalogMaximum, AnalogMinimum
from tensor_dslab.common import QuantityFieldSpec


@final
class AnalogWaveformKernels(TensorCollection[TensorKernel[Any]]):
    __slots__ = ()

    def _require(self) -> None:
        if any(
            type(member) not in (AnalogMinimum, AnalogMaximum)
            for member in self.members.values()
        ):
            raise TypeError("AnalogWaveformKernels contains an unsupported member")

    @property
    def minimum(self) -> AnalogMinimum | None:
        return self.members.get(AnalogMinimum)  # type: ignore[return-value]

    @property
    def maximum(self) -> AnalogMaximum | None:
        return self.members.get(AnalogMaximum)  # type: ignore[return-value]


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
        if type(self.spec) is not AnalogWaveformSpec:
            raise TypeError(
                "AnalogWaveformConfig.spec must be exact AnalogWaveformSpec"
            )
        if type(self.kernels) is not AnalogWaveformKernels:
            raise TypeError(
                "AnalogWaveformConfig.kernels must be exact AnalogWaveformKernels"
            )
