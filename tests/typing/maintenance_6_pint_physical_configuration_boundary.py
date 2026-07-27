# pyright: standard
"""Current Pint construction typing boundary."""

from typing import assert_type

from pint import Quantity
import torch

from tensor_dslab import quantities, quantity

assert_type(quantity(1, "ns"), Quantity)
assert_type(quantities((1, 2), "ns"), Quantity)
assert_type(quantities(torch.ones((2, 3)), "ns"), Quantity)
