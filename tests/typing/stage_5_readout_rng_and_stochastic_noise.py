from __future__ import annotations

from typing import assert_type

import torch
from tensor_core import NonnegativeFloat, PositiveFloat, PositiveInteger

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    SamplingConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.noise_waveform._product import (
    _product_noise_waveform,
)


sampling = SamplingConfig(
    sample_period_ps=PositiveInteger(1_000),
    sample_count=PositiveInteger(4),
)
axes = (
    ExampleAxis(coordinates=("example-0",)),
    ChannelAxis(coordinates=("channel-0",)),
    sampling.build_axis(),
)
photoelectrons = Photoelectrons(
    tensor=torch.zeros((1, 1, 4), dtype=torch.int64),
    axes=axes,
)

zero = _product_noise_waveform(
    photoelectrons,
    sampling=sampling,
    config=NoiseWaveformConfig(model=ZeroNoiseConfig()),
    seed=None,
    floating_dtype=torch.float32,
)
assert_type(zero, NoiseWaveform)

white = _product_noise_waveform(
    photoelectrons,
    sampling=sampling,
    config=NoiseWaveformConfig(
        model=WhiteNoiseConfig(rms_mv=PositiveFloat(1.0))
    ),
    seed=0,
    floating_dtype=torch.float64,
)
assert_type(white, NoiseWaveform)

psd = _product_noise_waveform(
    photoelectrons,
    sampling=sampling,
    config=NoiseWaveformConfig(
        model=PsdNoiseConfig(
            frequency_left_edges_hz=(NonnegativeFloat(0.0),),
            frequency_stop_hz=PositiveFloat(500_000_000.0),
            power_density_mv2_per_hz=(NonnegativeFloat(2.0e-9),),
        )
    ),
    seed=(1 << 64) - 1,
    floating_dtype=torch.float32,
)
assert_type(psd, NoiseWaveform)
