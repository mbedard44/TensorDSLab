"""Public configuration records for readout orchestration."""

from dataclasses import dataclass
from typing import final

from tensor_dslab.readout.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.readout.charge.config import ChargeConfig
from tensor_dslab.readout.digitized_waveform.config import (
    DigitizedWaveformConfig,
)
from tensor_dslab.readout.noise_waveform.config import NoiseWaveformConfig
from tensor_dslab.readout.pure_waveform.config import PureWaveformConfig


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutConfig:
    """Select configurations for the requested generated readout products."""

    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None
    __hash__ = None  # pyright: ignore[reportAssignmentType]
