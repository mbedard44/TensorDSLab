from __future__ import annotations

from typing import assert_type

import torch
from tensor_core import FiniteFloat, NonnegativeFloat, PositiveFloat, PositiveInteger

from tensor_dslab import (
    AnalogWaveform,
    AnalogWaveformConfig,
    ChannelAxis,
    Charge,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    ExampleAxis,
    NoiseWaveform,
    PureWaveform,
    PureWaveformConfig,
    SamplingConfig,
    TpcFebSnrPulseConfig,
)
from tensor_dslab.readout.analog_waveform._produce import (
    _prepare_analog_waveform,
    _produce_analog_waveform,
)
from tensor_dslab.readout.digitized_waveform._produce import (
    _prepare_digitized_waveform,
    _produce_digitized_waveform,
)
from tensor_dslab.readout.pure_waveform._produce import (
    _prepare_pure_waveform,
    _produce_pure_waveform,
)


sampling = SamplingConfig(
    sample_period_ps=PositiveInteger(2_000),
    sample_count=PositiveInteger(4),
)
axes = (
    ExampleAxis(coordinates=("example-0",)),
    ChannelAxis(coordinates=("channel-0",)),
    sampling.build_axis(),
)

charge = Charge(
    tensor=torch.ones((1, 1, 4), dtype=torch.float64),
    axes=axes,
)
pure_config = PureWaveformConfig(
    model=TpcFebSnrPulseConfig(
        fast_time_constant_ns=PositiveFloat(83.0),
        slow_time_constant_ns=PositiveFloat(383.0),
        support_time_ns=PositiveFloat(3_000.0),
        peak_voltage_mv_per_pe=FiniteFloat(-7.0),
    )
)
pure_plan = _prepare_pure_waveform(
    charge,
    sampling=sampling,
    config=pure_config,
    floating_dtype=charge.tensor.dtype,
    device=charge.tensor.device,
)
pure = _produce_pure_waveform(charge, plan=pure_plan)
assert_type(pure, PureWaveform)

noise = NoiseWaveform(
    tensor=torch.zeros((1, 1, 4), dtype=torch.float64),
    axes=axes,
)
analog_plan = _prepare_analog_waveform(
    config=AnalogWaveformConfig(),
    floating_dtype=pure.tensor.dtype,
    device=pure.tensor.device,
)
analog = _produce_analog_waveform(
    pure,
    noise,
    plan=analog_plan,
)
assert_type(analog, AnalogWaveform)

digitized_config = DigitizedWaveformConfig(
        bit_depth=PositiveInteger(12),
        input_min_mv=FiniteFloat(-20.0),
        input_max_mv=FiniteFloat(2.0),
        analog_gain_db=NonnegativeFloat(0.0),
)
digitized_plan = _prepare_digitized_waveform(
    config=digitized_config,
    floating_dtype=analog.tensor.dtype,
    device=analog.tensor.device,
)
digitized = _produce_digitized_waveform(analog, plan=digitized_plan)
assert_type(digitized, DigitizedWaveform)
