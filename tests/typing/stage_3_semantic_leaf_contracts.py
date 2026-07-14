from __future__ import annotations

from typing import assert_type

import torch

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    Photoelectrons,
    ReadoutCollection,
    SampleAxis,
)


sample_axis = SampleAxis(coordinates=("0ps", "2000ps"))
assert_type(sample_axis, SampleAxis)

example_axis = ExampleAxis(coordinates=("e0",))
assert_type(example_axis, ExampleAxis)

channel_axis = ChannelAxis(coordinates=("c0",))
assert_type(channel_axis, ChannelAxis)

photoelectrons = Photoelectrons(
    tensor=torch.zeros((1, 1, 2), dtype=torch.int64),
    axes=(example_axis, channel_axis, sample_axis),
)
assert_type(photoelectrons, Photoelectrons)
assert_type(photoelectrons.axis(SampleAxis), SampleAxis)
assert_type(photoelectrons.dimension_of(SampleAxis), int)

readout = ReadoutCollection(fields=(photoelectrons,))
assert_type(readout, ReadoutCollection)
assert_type(readout.field(Photoelectrons), Photoelectrons)
assert_type(readout.tensor(Photoelectrons), torch.Tensor)
