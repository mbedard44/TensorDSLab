"""Positive static contracts for Maintenance 15."""

from typing import Any, assert_type

import torch
from tensor_core import CountCoordinates, NonnegativeInteger

from tensor_dslab import (
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    ExampleAxis,
    Photoelectrons,
    PhotoelectronsSpec,
    QuantityFieldSpec,
    unit_registry,
)

axis = ExampleAxis(coordinates=CountCoordinates(count=2))
source_spec = PhotoelectronsSpec(
    axes=(axis,),
    device=torch.device("cpu"),
    dtype=torch.int64,
    unit=unit_registry.Unit("avalanche"),
)
source = Photoelectrons(tensor=torch.ones(2, dtype=torch.int64), spec=source_spec)
charge_spec = ChargeSpec(
    axes=(axis,),
    device=torch.device("cpu"),
    dtype=torch.float32,
    unit=unit_registry.Unit("avalanche"),
)
config = ChargeConfig(
    spec=charge_spec,
    kernels=ChargeKernels(members=()),
    correlated_avalanche_generations=NonnegativeInteger(value=0),
)
assert_type(source, Photoelectrons)
assert_type(config, ChargeConfig)
assert_type(Charge.prepare(source_specs=(source_spec,), config=config), ChargeConfig)
assert_type(source.spec, PhotoelectronsSpec[Any])
assert_type(charge_spec, ChargeSpec[tuple[ExampleAxis[int]]])
quantity_spec: QuantityFieldSpec[Any] = source_spec
