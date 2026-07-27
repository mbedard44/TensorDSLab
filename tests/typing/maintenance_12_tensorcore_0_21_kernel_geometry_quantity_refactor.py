# pyright: standard
"""Positive typing contracts for Maintenance 12."""

from typing import assert_type

import torch
from tensor_core import NonnegativeInteger, OffsetAxis

from tensor_dslab import (
    ChargeConfig,
    DarkCountRate,
    Pulse,
    PureWaveformConfig,
    QuantityKernel,
    SampleAxis,
    quantities,
    quantity,
)

sample_axis = SampleAxis(start=0, step=2, count=8)
rate = DarkCountRate(
    quantity=quantity(1, "Hz"),
    conditioning_axes=(),
    operation_axes=(),
)
pulse = Pulse(
    quantity=quantities((-1.0,), "mV"),
    conditioning_axes=(),
    operation_axes=(OffsetAxis(relative_to=SampleAxis, offsets=(0,)),),
)
assert_type(rate, DarkCountRate)
assert_type(pulse, Pulse)
assert_type(
    ChargeConfig(
        correlated_avalanche_generations=NonnegativeInteger(0),
        dark_counts=rate,
    ),
    ChargeConfig,
)
assert_type(PureWaveformConfig(pulse=pulse), PureWaveformConfig)
assert_type(pulse.tensor, torch.Tensor)
