from __future__ import annotations

from typing import assert_type

import torch
from tensor_core import (
    FiniteFloat,
    NonnegativeFloat,
    PositiveFloat,
    PositiveInteger,
)

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
    SampleAxis,
    TpcFebSnrPulseConfig,
)
from tensor_dslab.readout.analog_waveform.runtime.prepare import (
    prepare_analog_waveform,
)
from tensor_dslab.readout.analog_waveform.runtime.produce import (
    produce_analog_waveform,
)
from tensor_dslab.readout.digitized_waveform.runtime.prepare import (
    prepare_digitized_waveform,
)
from tensor_dslab.readout.digitized_waveform.runtime.produce import (
    produce_digitized_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.prepare import (
    prepare_pure_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.produce import (
    produce_pure_waveform,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


axes = (
    ExampleAxis(count=1),
    ChannelAxis(labels=("channel-0",)),
    SampleAxis(start=0, step=2_000, count=4),
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
sampling_runtime = SamplingRuntime(
    sample_count=4,
    sample_period_ps=2_000,
    sample_dimension=2,
)
pure_runtime = prepare_pure_waveform(
    pure_config,
    sampling=sampling_runtime,
    floating_dtype=charge.tensor.dtype,
    device=charge.tensor.device,
)
pure = produce_pure_waveform(charge, runtime=pure_runtime)
assert_type(pure, PureWaveform)

noise = NoiseWaveform(
    tensor=torch.zeros((1, 1, 4), dtype=torch.float64),
    axes=axes,
)
analog_runtime = prepare_analog_waveform(
    config=AnalogWaveformConfig(),
    floating_dtype=pure.tensor.dtype,
    device=pure.tensor.device,
)
analog = produce_analog_waveform(
    pure,
    noise,
    runtime=analog_runtime,
)
assert_type(analog, AnalogWaveform)

digitized_config = DigitizedWaveformConfig(
    bit_depth=PositiveInteger(12),
    input_min_mv=FiniteFloat(-20.0),
    input_max_mv=FiniteFloat(2.0),
    analog_gain_db=NonnegativeFloat(0.0),
)
digitized_runtime = prepare_digitized_waveform(
    config=digitized_config,
    floating_dtype=analog.tensor.dtype,
    device=analog.tensor.device,
)
digitized = produce_digitized_waveform(analog, runtime=digitized_runtime)
assert_type(digitized, DigitizedWaveform)
