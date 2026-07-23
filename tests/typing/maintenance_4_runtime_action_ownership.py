from __future__ import annotations

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
    Charge,
    ChargeConfig,
    ChannelAxis,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PureWaveform,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    TpcFebSnrPulseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.analog_waveform.runtime.produce import (
    produce_analog_waveform,
)
from tensor_dslab.readout.analog_waveform.runtime.validate import (
    validate_analog_waveform,
)
from tensor_dslab.readout.charge.runtime.produce import produce_charge
from tensor_dslab.readout.charge.runtime.validate import validate_charge
from tensor_dslab.readout.digitized_waveform.runtime.produce import (
    produce_digitized_waveform,
)
from tensor_dslab.readout.digitized_waveform.runtime.validate import (
    validate_digitized_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.produce import (
    produce_noise_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.validate import (
    validate_noise_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.produce import (
    produce_pure_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.validate import (
    validate_pure_waveform,
)
from tensor_dslab.readout.runtime.prepare import ReadoutRuntime, prepare_readout


source = Photoelectrons(
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
            fast_time_constant_ns=PositiveFloat(1.0),
            slow_time_constant_ns=PositiveFloat(2.0),
            support_time_ns=PositiveFloat(6.0),
            peak_voltage_mv_per_pe=FiniteFloat(-1.0),
        )
    ),
    noise_waveform=NoiseWaveformConfig(model=ZeroNoiseConfig()),
    analog_waveform=AnalogWaveformConfig(),
    digitized_waveform=DigitizedWaveformConfig(
        bit_depth=PositiveInteger(12),
        input_min_mv=FiniteFloat(-20.0),
        input_max_mv=FiniteFloat(20.0),
        analog_gain_db=NonnegativeFloat(0.0),
    ),
)
rng = Threefry4x32(seed=17)
requested, runtime = prepare_readout(
    source,
    products=(DigitizedWaveform,),
    config=config,
    rng=rng,
    floating_dtype=torch.float32,
)
assert_type(requested, frozenset[type[TensorField]])
assert_type(runtime, ReadoutRuntime)

assert runtime.charge is not None
charge = produce_charge(source, runtime=runtime.charge, rng=rng)
assert_type(charge, Charge)
validate_charge(charge, source=source, runtime=runtime.charge)

assert runtime.pure_waveform is not None
pure = produce_pure_waveform(charge, runtime=runtime.pure_waveform)
assert_type(pure, PureWaveform)
validate_pure_waveform(pure, source=charge)

assert runtime.noise_waveform is not None
noise = produce_noise_waveform(source, runtime=runtime.noise_waveform, rng=rng)
assert_type(noise, NoiseWaveform)
validate_noise_waveform(noise, source=source, runtime=runtime.noise_waveform)

assert runtime.analog_waveform is not None
analog = produce_analog_waveform(pure, noise, runtime=runtime.analog_waveform)
assert_type(analog, AnalogWaveform)
validate_analog_waveform(analog, pure=pure, noise=noise)

assert runtime.digitized_waveform is not None
digitized = produce_digitized_waveform(
    analog,
    runtime=runtime.digitized_waveform,
)
assert_type(digitized, DigitizedWaveform)
validate_digitized_waveform(
    digitized,
    source=analog,
    maximum_code=runtime.digitized_waveform.maximum_code,
)
