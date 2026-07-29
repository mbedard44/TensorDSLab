"""Physical coefficient Specs and kernels for noise waveforms."""

from typing import Any, final, override

import torch
from tensor_core import RegularCoordinates, TensorKernel

from tensor_dslab.common import FrequencyAxis, QuantityKernelSpec


@final
class WhiteNoiseRmsSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if self.operation_axes:
            raise ValueError("WhiteNoiseRmsSpec has no operation axes")


@final
class PowerSpectralDensitySpec[C: tuple](
    QuantityKernelSpec[
        C,
        tuple[FrequencyAxis[RegularCoordinates]],
    ]
):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        if len(self.operation_axes) != 1:
            raise ValueError("PowerSpectralDensitySpec requires one frequency axis")
        axis = self.operation_axes[0]
        if type(axis) is not FrequencyAxis or type(axis.coordinates) is not RegularCoordinates:
            raise TypeError("PowerSpectralDensitySpec requires regular FrequencyAxis")
        if axis.coordinates.start != 0 or axis.coordinates.step != 1:
            raise ValueError("PSD frequency coordinates require start 0 and step 1")


@final
class WhiteNoiseRms(TensorKernel[WhiteNoiseRmsSpec[Any, Any]]):
    """Represent strictly positive white-noise RMS magnitudes."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not WhiteNoiseRmsSpec:
            raise TypeError("WhiteNoiseRms requires exact WhiteNoiseRmsSpec")
        if not self.dtype.is_floating_point:
            raise TypeError("WhiteNoiseRms dtype must be floating")
        if not bool(torch.isfinite(self.tensor).all()) or bool((self.tensor <= 0).any()):
            raise ValueError("WhiteNoiseRms values must be finite and positive")


@final
class PowerSpectralDensity(TensorKernel[PowerSpectralDensitySpec[Any]]):
    """Represent prepared nonnegative per-bin noise powers."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        if type(self.spec) is not PowerSpectralDensitySpec:
            raise TypeError(
                "PowerSpectralDensity requires exact PowerSpectralDensitySpec"
            )
        if self.operation_axes[0].size < 2:
            raise ValueError(
                "PowerSpectralDensity requires DC and non-DC bins"
            )
        if not self.dtype.is_floating_point:
            raise TypeError("PowerSpectralDensity dtype must be floating")
        if not bool(torch.isfinite(self.tensor).all()) or bool((self.tensor < 0).any()):
            raise ValueError("PowerSpectralDensity values must be finite and nonnegative")
        if bool((self.tensor[..., 0] != 0).any()):
            raise ValueError("PowerSpectralDensity DC power must be zero")
        if not bool((self.tensor[..., 1:] > 0).any()):
            raise ValueError("PowerSpectralDensity requires positive non-DC power")
