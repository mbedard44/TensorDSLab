from __future__ import annotations

from collections.abc import Iterable
from typing import assert_type

import torch
from tensor_core import (
    FiniteFloat,
    NonnegativeFloat,
    PositiveFloat,
    PositiveInteger,
    TensorField,
    Threefry4x32,
)

from tensor_dslab import (
    AnalogWaveform,
    AnalogWaveformConfig,
    ChargeConfig,
    ChannelAxis,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    ExampleAxis,
    NoiseWaveformConfig,
    Photoelectrons,
    PureWaveformConfig,
    ReadoutCollection,
    ReadoutConfig,
    SampleAxis,
    TpcFebSnrPulseConfig,
    ZeroNoiseConfig,
    simulate_readout,
)


photoelectrons = Photoelectrons(
    tensor=torch.ones((1, 1, 4), dtype=torch.int64),
    axes=(
        ExampleAxis(count=1),
        ChannelAxis(labels=("channel-0",)),
        SampleAxis(start=0, step=2_000, count=4),
    ),
)
config = ReadoutConfig(
    charge=ChargeConfig(),
    pure_waveform=PureWaveformConfig(
        model=TpcFebSnrPulseConfig(
            fast_time_constant_ns=PositiveFloat(83.0),
            slow_time_constant_ns=PositiveFloat(383.0),
            support_time_ns=PositiveFloat(32.0),
            peak_voltage_mv_per_pe=FiniteFloat(-7.0),
        )
    ),
    noise_waveform=NoiseWaveformConfig(model=ZeroNoiseConfig()),
    analog_waveform=AnalogWaveformConfig(),
    digitized_waveform=DigitizedWaveformConfig(
        bit_depth=PositiveInteger(12),
        input_min_mv=FiniteFloat(-20.0),
        input_max_mv=FiniteFloat(2.0),
        analog_gain_db=NonnegativeFloat(0.0),
    ),
)
products: Iterable[type[TensorField]] = (AnalogWaveform, DigitizedWaveform)
readout: ReadoutCollection = simulate_readout(
    photoelectrons,
    products=products,
    config=config,
    rng=Threefry4x32(seed=17),
    floating_dtype=torch.float32,
)
assert_type(readout, ReadoutCollection)
analog: AnalogWaveform = readout.field(AnalogWaveform)
digitized: DigitizedWaveform = readout.field(DigitizedWaveform)
assert_type(analog, AnalogWaveform)
assert_type(digitized, DigitizedWaveform)
