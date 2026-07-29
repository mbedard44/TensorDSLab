"""Physical coefficient Specs, kernels, and collection for noise waveforms."""

from typing import Any, final, override

from tensor_core import (
    RegularCoordinates,
    TensorCollection,
    TensorKernel,
)

from tensor_dslab.common import FrequencyAxis, QuantityKernelSpec
from tensor_dslab.common.requirements.axis import require_regular_coordinates
from tensor_dslab.common.requirements.collection import (
    require_admitted_member_types,
    require_member_count,
)
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_no_operation_axes,
    require_operation_axis_count,
)
from tensor_dslab.common.requirements.tensor import (
    require_finite,
    require_floating_dtype,
    require_nonnegative,
    require_positive,
)


@final
class WhiteNoiseRmsSpec[C: tuple, O: tuple](QuantityKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require_quantity_kernel_spec(self) -> None:
        require_floating_dtype(self)
        require_no_operation_axes(self)


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
        require_floating_dtype(self)
        require_operation_axis_count(self, minimum=1, maximum=1)
        axis = self.operation_axes[0]
        if type(axis) is not FrequencyAxis:
            raise TypeError(
                "PowerSpectralDensitySpec requires regular FrequencyAxis"
            )
        require_regular_coordinates(
            axis.coordinates,
            start=0,
            step=1,
        )


@final
class WhiteNoiseRms(TensorKernel[WhiteNoiseRmsSpec[Any, Any]]):
    """Represent strictly positive white-noise RMS magnitudes."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, WhiteNoiseRmsSpec)
        require_finite(self)
        require_positive(self)


@final
class PowerSpectralDensity(TensorKernel[PowerSpectralDensitySpec[Any]]):
    """Represent prepared nonnegative per-bin noise powers."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, PowerSpectralDensitySpec)
        if self.operation_axes[0].size < 2:
            raise ValueError(
                "PowerSpectralDensity requires DC and non-DC bins"
            )
        require_finite(self)
        require_nonnegative(self)
        if bool((self.tensor[..., 0] != 0).any()):
            raise ValueError("PowerSpectralDensity DC power must be zero")
        if not bool((self.tensor[..., 1:] > 0).any()):
            raise ValueError(
                "PowerSpectralDensity requires positive non-DC power"
            )


@final
class NoiseWaveformKernels(TensorCollection[TensorKernel[Any]]):
    """Hold at most one exact stochastic noise coefficient branch."""

    __slots__ = ()

    def _require(self) -> None:
        require_admitted_member_types(
            self,
            admitted=(WhiteNoiseRms, PowerSpectralDensity),
        )
        require_member_count(self, maximum=1)

    @property
    def white_noise_rms(self) -> WhiteNoiseRms | None:
        return self.members.get(WhiteNoiseRms)  # type: ignore[return-value]

    @property
    def power_spectral_density(self) -> PowerSpectralDensity | None:
        return self.members.get(PowerSpectralDensity)  # type: ignore[return-value]
