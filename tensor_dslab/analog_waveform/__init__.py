"""Public analog-waveform product surface."""

from tensor_dslab.analog_waveform.config import (
    AnalogWaveformConfig,
    AnalogWaveformKernels,
)
from tensor_dslab.analog_waveform.field import AnalogWaveform, AnalogWaveformSpec
from tensor_dslab.analog_waveform.kernel import (
    AnalogMaximum,
    AnalogMaximumSpec,
    AnalogMinimum,
    AnalogMinimumSpec,
)

__all__ = (
    "AnalogMaximum",
    "AnalogMaximumSpec",
    "AnalogMinimum",
    "AnalogMinimumSpec",
    "AnalogWaveform",
    "AnalogWaveformConfig",
    "AnalogWaveformKernels",
    "AnalogWaveformSpec",
)
