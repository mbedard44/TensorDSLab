# pyright: standard
"""Current semantic leaf typing continuity."""

from typing import assert_type

from tensor_dslab import ChannelAxis, ExampleAxis, SampleAxis

assert_type(ExampleAxis(count=1), ExampleAxis)
assert_type(ChannelAxis(labels=("a",)), ChannelAxis)
assert_type(SampleAxis(start=0, step=2, count=4), SampleAxis)
