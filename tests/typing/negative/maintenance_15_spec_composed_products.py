"""The twelve intentional Maintenance 15 public typing failures."""

import torch
from tensor_core import (
    CountCoordinates,
    LabelCoordinates,
    NonnegativeInteger,
    OffsetAxis,
    OffsetCoordinates,
    RegularCoordinates,
    Threefry4x32,
)

from tensor_dslab import (
    AnalogMinimumSpec,
    BitDepth,
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    ExampleAxis,
    InputMinimumSpec,
    NoiseWaveformKernels,
    Photoelectrons,
    PhotoelectronsSpec,
    PowerSpectralDensitySpec,
    PulseResponse,
    PulseResponseSpec,
    PureWaveform,
    PureWaveformConfig,
    PureWaveformKernels,
    PureWaveformSpec,
    TimeAxis,
    WhiteNoiseRms,
    unit_registry,
)
from tensor_dslab import ReadoutConfig


device = torch.device("cpu")
example = ExampleAxis(coordinates=CountCoordinates(count=2))
time = TimeAxis(
    coordinates=RegularCoordinates(start=0, step=1, count=2),
    unit=unit_registry.Unit("ns"),
)
source_spec = PhotoelectronsSpec(
    axes=(example, time),
    device=device,
    dtype=torch.int64,
    unit=unit_registry.Unit("avalanche"),
)
source = Photoelectrons(
    tensor=torch.ones(source_spec.shape, dtype=torch.int64),
    spec=source_spec,
)
charge_spec = ChargeSpec(
    axes=(example, time),
    device=device,
    dtype=torch.float32,
    unit=unit_registry.Unit("avalanche"),
)
charge_config = ChargeConfig(
    spec=charge_spec,
    kernels=ChargeKernels(members=()),
    correlated_avalanche_generations=NonnegativeInteger(value=0),
)
offset = OffsetAxis(
    coordinates=OffsetCoordinates(offsets=(0,)),
    relative_to=TimeAxis,
)
pulse_spec = PulseResponseSpec(
    conditioning_axes=(),
    operation_axes=(offset,),
    device=device,
    dtype=torch.float32,
    unit=unit_registry.Unit("mV / avalanche"),
)
pulse = PulseResponse(tensor=torch.ones(1), spec=pulse_spec)
pure_config = PureWaveformConfig(
    spec=PureWaveformSpec(
        axes=(example, time),
        device=device,
        dtype=torch.float32,
        unit=unit_registry.Unit("mV"),
    ),
    kernels=PureWaveformKernels(members=(pulse,)),
)
rng = Threefry4x32(seed=0)


TimeAxis(
    coordinates=LabelCoordinates(labels=("t",)),
    unit=unit_registry.Unit("ns"),
)
Charge(tensor=torch.ones(source_spec.shape), spec=source_spec)
ChargeConfig(
    spec=charge_spec,
    kernels=NoiseWaveformKernels(members=()),
    correlated_avalanche_generations=NonnegativeInteger(value=0),
)
Charge.produce(sources=(1,), config=charge_config, rng=rng)
Charge.create(sources=(source,), config=charge_config)
PureWaveform.create(sources=(source,), config=pure_config, rng=rng)
WhiteNoiseRms(
    tensor=torch.tensor(1.0),
    spec=AnalogMinimumSpec(
        conditioning_axes=(),
        operation_axes=(),
        device=device,
        dtype=torch.float32,
        unit=unit_registry.Unit("mV"),
    ),
)
ChargeKernels(members=(object(),))
BitDepth(
    tensor=torch.tensor(12, dtype=torch.int16),
    spec=InputMinimumSpec(
        conditioning_axes=(),
        operation_axes=(),
        device=device,
        dtype=torch.float32,
        unit=unit_registry.Unit("mV"),
    ),
)
PowerSpectralDensitySpec(
    conditioning_axes=(),
    operation_axes=(time,),
    device=device,
    dtype=torch.float32,
    unit=unit_registry.Unit("mV ** 2"),
)
ChargeKernels(members=()).to(device=device, dtype=torch.float32)
