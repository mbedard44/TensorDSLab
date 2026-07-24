"""Public provisional readout configuration profiles."""

from tensor_core import NonnegativeFloat, PositiveInteger

from tensor_dslab.common import quantities, quantity
from tensor_dslab.readout.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.readout.charge.config import ChargeConfig, DarkCountConfig
from tensor_dslab.readout.config import ReadoutConfig
from tensor_dslab.readout.digitized_waveform.config import (
    DigitizedWaveformConfig,
)
from tensor_dslab.readout.noise_waveform.config import (
    NoiseWaveformConfig,
    PsdNoiseConfig,
)
from tensor_dslab.readout.pure_waveform.config import (
    PureWaveformConfig,
    VetoPduPulseConfig,
)

__all__ = ("ds20k_veto",)


def ds20k_veto() -> ReadoutConfig:
    """Return a fresh provisional DS20k Veto demonstration profile."""

    frequency_left_edges = (
        0.0,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        62.5,
    )
    power_density = (
        4.0e-8,
        7.0e-8,
        6.0e-8,
        3.0e-8,
        7.0e-9,
        1.0e-9,
        2.0e-10,
        5.0e-11,
        0.0,
    )

    return ReadoutConfig(
        charge=ChargeConfig(
            dark_count=DarkCountConfig(
                rate=quantity(100.0, "kHz"),
            ),
        ),
        pure_waveform=PureWaveformConfig(
            model=VetoPduPulseConfig(
                gaussian_center=quantity(232.89, "ns"),
                gaussian_width=quantity(507.72, "ns"),
                edge_offset_1=quantity(-81.92, "ns"),
                edge_width_1=quantity(147.28, "ns"),
                edge_offset_2=quantity(-176.50, "ns"),
                edge_width_2=quantity(45.69, "ns"),
                support_time=quantity(2020.27, "ns"),
                peak_voltage_per_photoelectron=quantity(
                    14.5912372,
                    "mV",
                ),
            )
        ),
        noise_waveform=NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges=quantities(
                    frequency_left_edges,
                    "MHz",
                ),
                frequency_stop=quantity(250.0, "MHz"),
                power_density=quantities(
                    power_density,
                    "mV ** 2 / Hz",
                ),
            ),
        ),
        analog_waveform=AnalogWaveformConfig(),
        digitized_waveform=DigitizedWaveformConfig(
            bit_depth=PositiveInteger(12),
            input_minimum=quantity(-20.0, "mV"),
            input_maximum=quantity(2.0, "mV"),
            analog_gain_db=NonnegativeFloat(0.0),
        ),
    )
