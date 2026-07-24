"""Private readout-domain structural requirements."""

import torch
from tensor_core import TensorField
from tensor_core.tensor.validation import require_field_layout

from tensor_dslab.common import ChannelAxis, ExampleAxis, SampleAxis


def require_readout_structure(field: TensorField) -> None:
    axis_types = tuple(type(axis) for axis in field.axes)
    accepted = frozenset({ExampleAxis, ChannelAxis, SampleAxis})
    if len(axis_types) != 3 or frozenset(axis_types) != accepted:
        raise ValueError(
            "readout fields require exactly example, channel, and sample axes"
        )
    require_field_layout(field, torch.strided)
