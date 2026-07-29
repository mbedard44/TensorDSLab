"""Public digitized-waveform product surface."""

from tensor_dslab.digitized_waveform.config import (
    DigitizedWaveformConfig,
    DigitizedWaveformKernels,
)
from tensor_dslab.digitized_waveform.field import (
    DigitizedWaveform,
    DigitizedWaveformSpec,
)
from tensor_dslab.digitized_waveform.kernel import (
    AnalogGain,
    AnalogGainSpec,
    BitDepth,
    BitDepthSpec,
    InputMaximum,
    InputMaximumSpec,
    InputMinimum,
    InputMinimumSpec,
)

__all__ = (
    "AnalogGain",
    "AnalogGainSpec",
    "BitDepth",
    "BitDepthSpec",
    "DigitizedWaveform",
    "DigitizedWaveformConfig",
    "DigitizedWaveformKernels",
    "DigitizedWaveformSpec",
    "InputMaximum",
    "InputMaximumSpec",
    "InputMinimum",
    "InputMinimumSpec",
)
