# pyright: standard
"""Current provisional profile typing contract."""

from typing import assert_type

from tensor_dslab import ReadoutConfig, SampleAxis
from tensor_dslab.readout.profiles import ds20k_veto

axis = SampleAxis(start=0, step=2, count=8)
assert_type(ds20k_veto(sample_axis=axis), ReadoutConfig)
