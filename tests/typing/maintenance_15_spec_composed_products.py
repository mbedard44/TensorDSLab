"""Positive static contracts for Maintenance 15."""

from typing import Any, assert_type

import torch
from tensor_core import CountCoordinates, NonnegativeInteger

from tensor_dslab import (
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    EncodedWaveform,
    EncodedWaveformConfig,
    EncodedWaveformKernels,
    EncodedWaveformSpec,
    ExampleAxis,
    Photoelectrons,
    PhotoelectronsSpec,
    PostTriggerSamples,
    PostTriggerSamplesSpec,
    PreTriggerSamples,
    PreTriggerSamplesSpec,
    QuantityFieldSpec,
    ReleaseThresholdCode,
    ReleaseThresholdCodeSpec,
    RequiredTimeOverSamples,
    RequiredTimeOverSamplesSpec,
    TriggerThresholdCode,
    TriggerThresholdCodeSpec,
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

encoded_spec = EncodedWaveformSpec(
    axes=(axis,),
    device=torch.device("cpu"),
    dtype=torch.int32,
    unit=unit_registry.Unit(""),
    suppression_code=-1,
)
trigger_spec = TriggerThresholdCodeSpec(
    conditioning_axes=(),
    operation_axes=(),
    device=torch.device("cpu"),
    dtype=torch.int64,
)
release_spec = ReleaseThresholdCodeSpec(
    conditioning_axes=(),
    operation_axes=(),
    device=torch.device("cpu"),
    dtype=torch.int64,
)
required_spec = RequiredTimeOverSamplesSpec(
    conditioning_axes=(),
    operation_axes=(),
    device=torch.device("cpu"),
    dtype=torch.int64,
)
pre_spec = PreTriggerSamplesSpec(
    conditioning_axes=(),
    operation_axes=(),
    device=torch.device("cpu"),
    dtype=torch.int64,
)
post_spec = PostTriggerSamplesSpec(
    conditioning_axes=(),
    operation_axes=(),
    device=torch.device("cpu"),
    dtype=torch.int64,
)
encoded_config = EncodedWaveformConfig(
    spec=encoded_spec,
    kernels=EncodedWaveformKernels(
        members=(
            TriggerThresholdCode(tensor=torch.tensor(1), spec=trigger_spec),
            ReleaseThresholdCode(tensor=torch.tensor(2), spec=release_spec),
            RequiredTimeOverSamples(
                tensor=torch.tensor(1),
                spec=required_spec,
            ),
            PreTriggerSamples(tensor=torch.tensor(0), spec=pre_spec),
            PostTriggerSamples(tensor=torch.tensor(0), spec=post_spec),
        )
    ),
)
assert_type(encoded_spec, EncodedWaveformSpec[tuple[ExampleAxis[int]]])
assert_type(encoded_config, EncodedWaveformConfig)
assert_type(
    EncodedWaveform.prepare(
        source_specs=(),
        config=encoded_config,
    ),
    EncodedWaveformConfig,
)
