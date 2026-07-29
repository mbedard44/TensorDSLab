"""Shared deterministic fixtures for the Maintenance 15 Product tests."""

import torch
from tensor_core import (
    CountCoordinates,
    LabelCoordinates,
    NonnegativeInteger,
    OffsetAxis,
    OffsetCoordinates,
    RegularCoordinates,
)

from tensor_dslab import (
    AnalogGain,
    AnalogGainSpec,
    AnalogMaximum,
    AnalogMaximumSpec,
    AnalogMinimum,
    AnalogMinimumSpec,
    AnalogWaveformConfig,
    AnalogWaveformKernels,
    AnalogWaveformSpec,
    BitDepth,
    BitDepthSpec,
    ChannelAxis,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    DigitizedWaveformConfig,
    DigitizedWaveformKernels,
    DigitizedWaveformSpec,
    ExampleAxis,
    InputMaximum,
    InputMaximumSpec,
    InputMinimum,
    InputMinimumSpec,
    NoiseWaveformConfig,
    NoiseWaveformKernels,
    NoiseWaveformSpec,
    Photoelectrons,
    PhotoelectronsSpec,
    PulseResponse,
    PulseResponseSpec,
    PureWaveformConfig,
    PureWaveformKernels,
    PureWaveformSpec,
    TimeAxis,
    WhiteNoiseRms,
    WhiteNoiseRmsSpec,
    unit_registry,
)


CPU = torch.device("cpu")
U = unit_registry


def axes() -> tuple:
    return (
        ExampleAxis(coordinates=CountCoordinates(count=2)),
        ChannelAxis(coordinates=LabelCoordinates(labels=("a", "b"))),
        TimeAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=5),
            coordinate_scale=2.0,
            unit=U.Unit("ns"),
        ),
    )


def source() -> Photoelectrons:
    spec = PhotoelectronsSpec(
        axes=axes(),
        device=CPU,
        dtype=torch.int64,
        unit=U.Unit("avalanche"),
    )
    values = torch.zeros(spec.shape, dtype=torch.int64)
    values[0, 0, 1] = 2
    values[1, 1, 3] = 1
    return Photoelectrons(tensor=values, spec=spec)


def charge_config() -> ChargeConfig:
    spec = ChargeSpec(
        axes=axes(),
        device=CPU,
        dtype=torch.float32,
        unit=U.Unit("avalanche"),
    )
    return ChargeConfig(
        spec=spec,
        kernels=ChargeKernels(members=()),
        correlated_avalanche_generations=NonnegativeInteger(value=0),
    )


def pure_config() -> PureWaveformConfig:
    spec = PureWaveformSpec(
        axes=axes(), device=CPU, dtype=torch.float32, unit=U.Unit("mV")
    )
    operation = OffsetAxis(
        coordinates=OffsetCoordinates(offsets=(0, 1)),
        relative_to=TimeAxis,
    )
    pulse_spec = PulseResponseSpec(
        conditioning_axes=(),
        operation_axes=(operation,),
        device=CPU,
        dtype=torch.float32,
        unit=U.Unit("mV / avalanche"),
    )
    pulse = PulseResponse(
        tensor=torch.tensor([-1.0, -0.5], dtype=torch.float32),
        spec=pulse_spec,
    )
    return PureWaveformConfig(
        spec=spec, kernels=PureWaveformKernels(members=(pulse,))
    )


def noise_config(*, white: bool = False) -> NoiseWaveformConfig:
    spec = NoiseWaveformSpec(
        axes=axes(), device=CPU, dtype=torch.float32, unit=U.Unit("mV")
    )
    members = ()
    if white:
        rms_spec = WhiteNoiseRmsSpec(
            conditioning_axes=(),
            operation_axes=(),
            device=CPU,
            dtype=torch.float32,
            unit=U.Unit("mV"),
        )
        members = (
            WhiteNoiseRms(tensor=torch.tensor(0.2), spec=rms_spec),
        )
    return NoiseWaveformConfig(
        spec=spec, kernels=NoiseWaveformKernels(members=members)
    )


def analog_config() -> AnalogWaveformConfig:
    spec = AnalogWaveformSpec(
        axes=axes(), device=CPU, dtype=torch.float32, unit=U.Unit("mV")
    )
    minimum_spec = AnalogMinimumSpec(
        conditioning_axes=(),
        operation_axes=(),
        device=CPU,
        dtype=torch.float32,
        unit=U.Unit("mV"),
    )
    maximum_spec = AnalogMaximumSpec(
        conditioning_axes=(),
        operation_axes=(),
        device=CPU,
        dtype=torch.float32,
        unit=U.Unit("mV"),
    )
    return AnalogWaveformConfig(
        spec=spec,
        kernels=AnalogWaveformKernels(
            members=(
                AnalogMinimum(tensor=torch.tensor(-2.0), spec=minimum_spec),
                AnalogMaximum(tensor=torch.tensor(2.0), spec=maximum_spec),
            )
        ),
    )


def digitized_config() -> DigitizedWaveformConfig:
    spec = DigitizedWaveformSpec(
        axes=axes(), device=CPU, dtype=torch.int32, unit=U.Unit("")
    )
    bit = BitDepth(
        tensor=torch.tensor(12, dtype=torch.int16),
        spec=BitDepthSpec(
            conditioning_axes=(),
            operation_axes=(),
            device=CPU,
            dtype=torch.int16,
        ),
    )
    minimum = InputMinimum(
        tensor=torch.tensor(-2.0),
        spec=InputMinimumSpec(
            conditioning_axes=(),
            operation_axes=(),
            device=CPU,
            dtype=torch.float32,
            unit=U.Unit("mV"),
        ),
    )
    maximum = InputMaximum(
        tensor=torch.tensor(2.0),
        spec=InputMaximumSpec(
            conditioning_axes=(),
            operation_axes=(),
            device=CPU,
            dtype=torch.float32,
            unit=U.Unit("mV"),
        ),
    )
    gain = AnalogGain(
        tensor=torch.tensor(1.0),
        spec=AnalogGainSpec(
            conditioning_axes=(),
            operation_axes=(),
            device=CPU,
            dtype=torch.float32,
            unit=U.Unit(""),
        ),
    )
    return DigitizedWaveformConfig(
        spec=spec,
        kernels=DigitizedWaveformKernels(
            members=(bit, minimum, maximum, gain)
        ),
    )
