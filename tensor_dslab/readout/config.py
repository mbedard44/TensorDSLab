from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_dslab.readout.requirements import (
    require_optional_exact,
)
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
    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None

    def __post_init__(self) -> None:
        require_optional_exact(
            self.charge,
            ChargeConfig,
            "ReadoutConfig.charge",
        )
        require_optional_exact(
            self.pure_waveform,
            PureWaveformConfig,
            "ReadoutConfig.pure_waveform",
        )
        require_optional_exact(
            self.noise_waveform,
            NoiseWaveformConfig,
            "ReadoutConfig.noise_waveform",
        )
        require_optional_exact(
            self.analog_waveform,
            AnalogWaveformConfig,
            "ReadoutConfig.analog_waveform",
        )
        require_optional_exact(
            self.digitized_waveform,
            DigitizedWaveformConfig,
            "ReadoutConfig.digitized_waveform",
        )
