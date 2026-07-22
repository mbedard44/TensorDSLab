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
from tensor_dslab.readout.charge.runtime.prepare import prepare_charge
from tensor_dslab.readout.charge.runtime.produce import produce_charge
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    prepare_noise_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.produce import (
    produce_noise_waveform,
)
from tensor_dslab.readout.runtime.sampling import prepare_sampling


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

sampling_runtime = prepare_sampling(source, config=sampling)
noise_runtime = prepare_noise_waveform(
    NoiseWaveformConfig(
        model=WhiteNoiseConfig(
            rms_mv=PositiveFloat(1.0),
            rng_key=key,
        )
    ),
    sampling=sampling_runtime,
    shape=source.shape,
    floating_dtype=torch.float32,
    device=source.tensor.device,
)
noise = produce_noise_waveform(source, runtime=noise_runtime, rng=rng)
assert_type(noise, NoiseWaveform)
assert_type(noise.tensor, torch.Tensor)

charge_runtime = prepare_charge(
    ChargeConfig(
        smearing=ChargeSmearingConfig(
            relative_sigma=NonnegativeFloat(0.1),
            rng_key=key,
        )
    ),
    photoelectrons=source,
    sampling=sampling_runtime,
    floating_dtype=torch.float64,
)
charge = produce_charge(source, runtime=charge_runtime, rng=rng)
assert_type(charge, Charge)
assert_type(charge.tensor, torch.Tensor)
