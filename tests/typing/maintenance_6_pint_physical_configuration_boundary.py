from typing import assert_type

from pint import Quantity
from tensor_core import (
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveInteger,
    Probability,
)

from tensor_dslab import (
    AfterpulseConfig,
    AfterpulseRecoveryConfig,
    AnalogSaturationConfig,
    AnalogWaveformConfig,
    ChargeConfig,
    ChargeSmearingConfig,
    CorrelatedAvalancheConfig,
    DarkCountConfig,
    DelayedCrosstalkConfig,
    DigitizedWaveformConfig,
    DirectCrosstalkConfig,
    ExponentialDelayConfig,
    FixedDelayConfig,
    NoiseWaveformConfig,
    PsdNoiseConfig,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    TimingJitterConfig,
    TpcFebSnrPulseConfig,
    VetoPduPulseConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
    quantities,
    quantity,
)


time = quantity(2, "ns")
voltage = quantity(1, "mV")
frequency = quantity(1.0e6, "Hz")
density = quantity(1.0e-9, "mV ** 2 / Hz")
assert_type(time, Quantity)
assert_type(quantities((1, 2.0), "ns"), Quantity)

axis = SampleAxis.from_period(period=time, count=4)
assert_type(axis, SampleAxis)
assert_type(axis.start_time, Quantity)
assert_type(axis.sample_period, Quantity)
assert_type(axis.time_at(1), Quantity)
assert_type(axis.stop_time, Quantity)

fixed = FixedDelayConfig(delay=time)
exponential = ExponentialDelayConfig(mean_delay=time)
direct = DirectCrosstalkConfig(
    mean_offspring_per_parent=NonnegativeFloat(0.1),
    delay=fixed,
)
delayed = DelayedCrosstalkConfig(
    mean_offspring_per_parent=NonnegativeFloat(0.1),
    delay=exponential,
)
recovery = AfterpulseRecoveryConfig(time_constant=time)
afterpulse = AfterpulseConfig(
    probability=Probability(0.1),
    mean_delay=time,
    recovery=recovery,
)
correlated = CorrelatedAvalancheConfig(
    maximum_generations=NonnegativeInteger(1),
    direct_crosstalk=direct,
    delayed_crosstalk=delayed,
    afterpulse=afterpulse,
)
charge = ChargeConfig(
    dark_count=DarkCountConfig(rate=frequency),
    timing_jitter=TimingJitterConfig(sigma=time),
    correlated_avalanches=correlated,
    smearing=ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.1)),
)

tpc = TpcFebSnrPulseConfig(
    fast_time_constant=quantity(1, "ns"),
    slow_time_constant=quantity(2, "ns"),
    support_time=quantity(6, "ns"),
    peak_voltage_per_photoelectron=voltage,
)
veto = VetoPduPulseConfig(
    gaussian_center=quantity(0, "ns"),
    gaussian_width=quantity(1, "ns"),
    edge_offset_1=quantity(-1, "ns"),
    edge_width_1=quantity(1, "ns"),
    edge_offset_2=quantity(1, "ns"),
    edge_width_2=quantity(1, "ns"),
    support_time=quantity(6, "ns"),
    peak_voltage_per_photoelectron=voltage,
)
pure = PureWaveformConfig(model=tpc)
assert_type(PureWaveformConfig(model=veto), PureWaveformConfig)

white = WhiteNoiseConfig(rms=quantity(1, "mV"))
psd = PsdNoiseConfig(
    frequency_left_edges=quantities((0, 1.0e6), "Hz"),
    frequency_stop=quantity(3.0e8, "Hz"),
    power_density=quantities((1.0e-9, 1.0e-9), "mV ** 2 / Hz"),
)
noise = NoiseWaveformConfig(model=white)
assert_type(NoiseWaveformConfig(model=psd), NoiseWaveformConfig)
assert_type(NoiseWaveformConfig(model=ZeroNoiseConfig()), NoiseWaveformConfig)

analog = AnalogWaveformConfig(
    saturation=AnalogSaturationConfig(
        minimum=quantity(-20, "mV"),
        maximum=quantity(20, "mV"),
    )
)
digitized = DigitizedWaveformConfig(
    bit_depth=PositiveInteger(12),
    input_minimum=quantity(-20, "mV"),
    input_maximum=quantity(20, "mV"),
    analog_gain_db=NonnegativeFloat(0),
)
readout = ReadoutConfig(
    charge=charge,
    pure_waveform=pure,
    noise_waveform=noise,
    analog_waveform=analog,
    digitized_waveform=digitized,
)
assert_type(readout, ReadoutConfig)
assert charge.dark_count is not None
assert_type(charge.dark_count.rate, Quantity)
assert_type(tpc.fast_time_constant, Quantity)
assert_type(white.rms, Quantity)
assert_type(psd.frequency_left_edges, Quantity)
assert analog.saturation is not None
assert_type(analog.saturation.minimum, Quantity | None)
assert_type(digitized.input_minimum, Quantity)
