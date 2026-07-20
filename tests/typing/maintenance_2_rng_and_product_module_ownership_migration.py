from __future__ import annotations

from typing import assert_type

import torch
from tensor_core import (
    NonnegativeFloat,
    PositiveFloat,
    PositiveInteger,
    RngKey,
    Threefry4x32,
)

from tensor_dslab import (
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    ChannelAxis,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    SamplingConfig,
    WhiteNoiseConfig,
)
from tensor_dslab.readout.charge._produce import _prepare_charge, _produce_charge
from tensor_dslab.readout.noise_waveform._produce import (
    _prepare_noise_waveform,
    _produce_noise_waveform,
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
source = Photoelectrons(
    tensor=torch.ones((1, 1, 4), dtype=torch.int64),
    axes=axes,
)
rng = Threefry4x32(seed=17)
key = RngKey(namespace=0x54445331, stream=101)

noise_plan = _prepare_noise_waveform(
    source,
    sampling=sampling,
    config=NoiseWaveformConfig(
        model=WhiteNoiseConfig(
            rms_mv=PositiveFloat(1.0),
            rng_key=key,
        )
    ),
    floating_dtype=torch.float32,
)
noise = _produce_noise_waveform(source, plan=noise_plan, rng=rng)
assert_type(noise, NoiseWaveform)
assert_type(noise.tensor, torch.Tensor)

charge_plan = _prepare_charge(
    source,
    sampling=sampling,
    config=ChargeConfig(
        smearing=ChargeSmearingConfig(
            relative_sigma=NonnegativeFloat(0.1),
            rng_key=key,
        )
    ),
    floating_dtype=torch.float64,
)
charge = _produce_charge(source, plan=charge_plan, rng=rng)
assert_type(charge, Charge)
assert_type(charge.tensor, torch.Tensor)
