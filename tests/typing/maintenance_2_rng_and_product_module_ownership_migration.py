from typing import assert_type

import torch
from tensor_core import (
    NonnegativeFloat,
    RngKey,
    RngPositions,
    Threefry4x32,
)

from tensor_dslab import (
    quantity,
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    ChannelAxis,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    SampleAxis,
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
from tensor_dslab.readout.runtime.sampling import (
    SamplingRuntime,
    prepare_sampling,
)
from tensor_dslab.readout.runtime.keys import WHITE_NOISE_RNG_KEY

def _mv(value: int | float):
    return quantity(value, "mV")


axes = (
    ExampleAxis(count=1),
    ChannelAxis(labels=("channel-0",)),
    SampleAxis(start=0, step=2_000, count=4),
)
source = Photoelectrons(
    tensor=torch.ones((1, 1, 4), dtype=torch.int64),
    axes=axes,
)
rng = Threefry4x32(seed=17)
assert_type(WHITE_NOISE_RNG_KEY, RngKey)
assert_type(RngPositions.from_shape((1, 1, 4), device="cpu"), RngPositions)

sampling_runtime = prepare_sampling(source)
assert_type(sampling_runtime, SamplingRuntime)
noise_runtime = prepare_noise_waveform(
    NoiseWaveformConfig(
        model=WhiteNoiseConfig(
            rms=_mv(1.0),
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
        )
    ),
    photoelectrons=source,
    sampling=sampling_runtime,
    floating_dtype=torch.float64,
)
charge = produce_charge(source, runtime=charge_runtime, rng=rng)
assert_type(charge, Charge)
assert_type(charge.tensor, torch.Tensor)
