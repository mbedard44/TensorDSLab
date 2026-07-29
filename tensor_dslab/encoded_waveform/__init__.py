"""Public encoded-waveform Product surface."""

from tensor_dslab.encoded_waveform.config import EncodedWaveformConfig
from tensor_dslab.encoded_waveform.field import (
    EncodedWaveform,
    EncodedWaveformSpec,
)
from tensor_dslab.encoded_waveform.kernel import (
    EncodedWaveformKernels,
    PostTriggerSamples,
    PostTriggerSamplesSpec,
    PreTriggerSamples,
    PreTriggerSamplesSpec,
    ReleaseThresholdCode,
    ReleaseThresholdCodeSpec,
    RequiredTimeOverSamples,
    RequiredTimeOverSamplesSpec,
    TriggerThresholdCode,
    TriggerThresholdCodeSpec,
)

__all__ = (
    "EncodedWaveform",
    "EncodedWaveformConfig",
    "EncodedWaveformKernels",
    "EncodedWaveformSpec",
    "PostTriggerSamples",
    "PostTriggerSamplesSpec",
    "PreTriggerSamples",
    "PreTriggerSamplesSpec",
    "ReleaseThresholdCode",
    "ReleaseThresholdCodeSpec",
    "RequiredTimeOverSamples",
    "RequiredTimeOverSamplesSpec",
    "TriggerThresholdCode",
    "TriggerThresholdCodeSpec",
)
