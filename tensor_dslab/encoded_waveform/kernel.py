"""Raw-ZLE policy Specs, Kernels, and exact collection."""

from typing import Any, final, override

import torch
from tensor_core import TensorCollection, TensorKernel, TensorKernelSpec

from tensor_dslab.common import TimeAxis
from tensor_dslab.common.requirements.collection import (
    require_exact_member_types,
)
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_no_conditioning_axis_type,
    require_no_operation_axes,
)
from tensor_dslab.common.requirements.tensor import (
    require_exact_dtype,
    require_nonnegative,
    require_positive,
)


@final
class TriggerThresholdCodeSpec[C: tuple, O: tuple](TensorKernelSpec[C, O]):
    """Describe literal inclusive trigger ADC-code thresholds."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_dtype(self, torch.int64)
        require_no_operation_axes(self)
        require_no_conditioning_axis_type(self, TimeAxis)


@final
class ReleaseThresholdCodeSpec[C: tuple, O: tuple](TensorKernelSpec[C, O]):
    """Describe literal inclusive release ADC-code thresholds."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_dtype(self, torch.int64)
        require_no_operation_axes(self)
        require_no_conditioning_axis_type(self, TimeAxis)


@final
class RequiredTimeOverSamplesSpec[C: tuple, O: tuple](
    TensorKernelSpec[C, O]
):
    """Describe consecutive trigger-qualification sample counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_dtype(self, torch.int64)
        require_no_operation_axes(self)
        require_no_conditioning_axis_type(self, TimeAxis)


@final
class PreTriggerSamplesSpec[C: tuple, O: tuple](TensorKernelSpec[C, O]):
    """Describe pre-trigger retained sample counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_dtype(self, torch.int64)
        require_no_operation_axes(self)
        require_no_conditioning_axis_type(self, TimeAxis)


@final
class PostTriggerSamplesSpec[C: tuple, O: tuple](TensorKernelSpec[C, O]):
    """Describe post-release retained sample counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_dtype(self, torch.int64)
        require_no_operation_axes(self)
        require_no_conditioning_axis_type(self, TimeAxis)


@final
class TriggerThresholdCode(
    TensorKernel[TriggerThresholdCodeSpec[Any, Any]]
):
    """Carry inclusive trigger ADC-code thresholds."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, TriggerThresholdCodeSpec)
        require_nonnegative(self)


@final
class ReleaseThresholdCode(
    TensorKernel[ReleaseThresholdCodeSpec[Any, Any]]
):
    """Carry inclusive release ADC-code thresholds."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, ReleaseThresholdCodeSpec)
        require_nonnegative(self)


@final
class RequiredTimeOverSamples(
    TensorKernel[RequiredTimeOverSamplesSpec[Any, Any]]
):
    """Carry positive trigger-qualification sample counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, RequiredTimeOverSamplesSpec)
        require_positive(self)


@final
class PreTriggerSamples(TensorKernel[PreTriggerSamplesSpec[Any, Any]]):
    """Carry nonnegative pre-trigger sample counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, PreTriggerSamplesSpec)
        require_nonnegative(self)


@final
class PostTriggerSamples(TensorKernel[PostTriggerSamplesSpec[Any, Any]]):
    """Carry nonnegative post-release sample counts."""

    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_kernel_spec(self, PostTriggerSamplesSpec)
        require_nonnegative(self)


@final
class EncodedWaveformKernels(TensorCollection[TensorKernel[Any]]):
    """Hold the exact raw-ZLE coefficient set."""

    __slots__ = ()

    def _require(self) -> None:
        require_exact_member_types(
            self,
            required=(
                TriggerThresholdCode,
                ReleaseThresholdCode,
                RequiredTimeOverSamples,
                PreTriggerSamples,
                PostTriggerSamples,
            ),
        )

    @property
    def trigger_threshold_code(self) -> TriggerThresholdCode:
        return self.member(TriggerThresholdCode)

    @property
    def release_threshold_code(self) -> ReleaseThresholdCode:
        return self.member(ReleaseThresholdCode)

    @property
    def required_time_over_samples(self) -> RequiredTimeOverSamples:
        return self.member(RequiredTimeOverSamples)

    @property
    def pre_trigger_samples(self) -> PreTriggerSamples:
        return self.member(PreTriggerSamples)

    @property
    def post_trigger_samples(self) -> PostTriggerSamples:
        return self.member(PostTriggerSamples)
