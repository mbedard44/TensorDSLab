# pyright: standard
"""Current deterministic waveform Config typing."""

from typing import assert_type

from tensor_core import OffsetAxis
from tensor_dslab import Pulse, PureWaveformConfig, SampleAxis, quantities

pulse = Pulse(
    quantity=quantities((-1.0,), "mV"),
    conditioning_axes=(),
    operation_axes=(OffsetAxis(relative_to=SampleAxis, offsets=(0,)),),
)
assert_type(pulse, Pulse)
assert_type(PureWaveformConfig(pulse=pulse), PureWaveformConfig)
