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


sample_axis = SampleAxis(start=0, step=2_000, count=2)
assert_type(sample_axis, SampleAxis)
assert_type(sample_axis.coordinate_at(1), int)
assert_type(sample_axis.index_of(2_000), int)

example_axis = ExampleAxis(count=1)
assert_type(example_axis, ExampleAxis)
assert_type(example_axis.coordinate_at(0), int)
assert_type(example_axis.index_of(0), int)

channel_axis = ChannelAxis(labels=("c0",))
assert_type(channel_axis, ChannelAxis)
assert_type(channel_axis.coordinate_at(0), str)
assert_type(channel_axis.index_of("c0"), int)

photoelectrons = Photoelectrons(
    tensor=torch.zeros((1, 1, 2), dtype=torch.int64),
    axes=(example_axis, channel_axis, sample_axis),
)
assert_type(photoelectrons, Photoelectrons)
assert_type(photoelectrons.axis(SampleAxis), SampleAxis)
assert_type(photoelectrons.axis(ExampleAxis), ExampleAxis)
assert_type(photoelectrons.axis(ChannelAxis), ChannelAxis)
assert_type(photoelectrons.dimension_of(SampleAxis), int)
assert_type(photoelectrons.coordinate_at(ExampleAxis, index=0), int)
assert_type(photoelectrons.coordinate_at(ChannelAxis, index=0), str)
assert_type(photoelectrons.coordinate_at(SampleAxis, index=0), int)

readout = ReadoutCollection(fields=(photoelectrons,))
assert_type(readout, ReadoutCollection)
assert_type(readout.field(Photoelectrons), Photoelectrons)
assert_type(readout.tensor(Photoelectrons), torch.Tensor)
