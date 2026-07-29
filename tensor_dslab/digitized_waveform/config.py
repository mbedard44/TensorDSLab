"""Digitizer coefficient collection and configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch
from tensor_core import TensorCollection, TensorKernel

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.digitized_waveform.field import DigitizedWaveformSpec
from tensor_dslab.digitized_waveform.kernel import (
    AnalogGain,
    BitDepth,
    InputMaximum,
    InputMinimum,
)


@final
class DigitizedWaveformKernels(TensorCollection[TensorKernel[Any]]):
    __slots__ = ()

    def _require(self) -> None:
        required = {BitDepth, InputMinimum, InputMaximum, AnalogGain}
        if self.member_types != frozenset(required):
            raise ValueError(
                "DigitizedWaveformKernels requires bit depth, input bounds, and gain"
            )

    @property
    def bit_depth(self) -> BitDepth:
        return self.member(BitDepth)

    @property
    def input_minimum(self) -> InputMinimum:
        return self.member(InputMinimum)

    @property
    def input_maximum(self) -> InputMaximum:
        return self.member(InputMaximum)

    @property
    def analog_gain(self) -> AnalogGain:
        return self.member(AnalogGain)


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
        if type(self.spec) is not DigitizedWaveformSpec:
            raise TypeError(
                "DigitizedWaveformConfig.spec must be exact DigitizedWaveformSpec"
            )
        if type(self.kernels) is not DigitizedWaveformKernels:
            raise TypeError(
                "DigitizedWaveformConfig.kernels must be exact DigitizedWaveformKernels"
            )
