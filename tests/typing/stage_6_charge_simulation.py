from __future__ import annotations

from typing import assert_type

import torch
from tensor_core import (
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveInteger,
)

from tensor_dslab import (
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    ChannelAxis,
    CorrelatedAvalancheConfig,
    DirectCrosstalkConfig,
    ExampleAxis,
    FixedDelayConfig,
    Photoelectrons,
    SamplingConfig,
)
from tensor_dslab.readout.charge._produce import _produce_charge


sampling = SamplingConfig(
    sample_period_ps=PositiveInteger(2_000),
    sample_count=PositiveInteger(4),
)
axes = (
    ExampleAxis(coordinates=("example-0",)),
    ChannelAxis(coordinates=("channel-0",)),
    sampling.build_axis(),
)
photoelectrons = Photoelectrons(
    tensor=torch.ones((1, 1, 4), dtype=torch.int64),
    axes=axes,
)

deterministic = _produce_charge(
    photoelectrons,
    sampling=sampling,
    config=ChargeConfig(),
    seed=None,
    floating_dtype=torch.float32,
)
assert_type(deterministic, Charge)

stochastic = _produce_charge(
    photoelectrons,
    sampling=sampling,
    config=ChargeConfig(
        correlated_avalanches=CorrelatedAvalancheConfig(
            maximum_generations=NonnegativeInteger(2),
            direct_crosstalk=DirectCrosstalkConfig(
                mean_offspring_per_parent=NonnegativeFloat(0.3),
                delay=FixedDelayConfig(delay_ns=NonnegativeFloat(0.0)),
            ),
        ),
        smearing=ChargeSmearingConfig(relative_sigma=NonnegativeFloat(0.1)),
    ),
    seed=0,
    floating_dtype=torch.float64,
)
assert_type(stochastic, Charge)
