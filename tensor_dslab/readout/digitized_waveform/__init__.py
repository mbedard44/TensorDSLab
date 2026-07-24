"""Public digitized waveform product facade."""

from tensor_dslab.readout.digitized_waveform.config import DigitizedWaveformConfig
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform

__all__ = (
    "DigitizedWaveform",
    "DigitizedWaveformConfig",
)
