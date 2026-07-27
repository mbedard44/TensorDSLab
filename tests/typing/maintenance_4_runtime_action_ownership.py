# pyright: standard
"""Current Runtime action typing continuity."""

from typing import assert_type

from tensor_dslab.readout.charge.runtime.prepare import ChargeRuntime
from tensor_dslab.readout.pure_waveform.runtime.prepare import PureWaveformRuntime

assert_type(ChargeRuntime, type[ChargeRuntime])
assert_type(PureWaveformRuntime, type[PureWaveformRuntime])
