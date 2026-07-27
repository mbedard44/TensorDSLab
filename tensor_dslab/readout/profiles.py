"""Public provisional readout configuration profiles."""

import math

import torch
from tensor_core import NonnegativeFloat, NonnegativeInteger, OffsetAxis, PositiveInteger

from tensor_dslab.common import (
    ChannelAxis,
    ExampleAxis,
    SampleAxis,
    quantities,
    quantity,
)
from tensor_dslab.readout.analog_waveform.config import AnalogWaveformConfig
from tensor_dslab.readout.charge.config import ChargeConfig
from tensor_dslab.readout.charge.kernel import DarkCountRate
from tensor_dslab.readout.config import ReadoutConfig
from tensor_dslab.readout.digitized_waveform.config import DigitizedWaveformConfig
from tensor_dslab.readout.noise_waveform.config import NoiseWaveformConfig, PsdNoiseConfig
from tensor_dslab.readout.pure_waveform.config import PureWaveformConfig
from tensor_dslab.readout.pure_waveform.kernel import Pulse

__all__ = ("ds20k_veto",)


def _require_available_conditioning_axes(
    config: ReadoutConfig,
    *,
    available_roles: frozenset[type],
) -> None:
    charge = config.charge
    kernels = (
        ()
        if charge is None
        else (
            charge.timing_jitter,
            charge.direct_crosstalk,
            charge.delayed_crosstalk,
            charge.afterpulse,
            charge.dark_counts,
            charge.smearing_width,
        )
    )
    pulse = None if config.pure_waveform is None else config.pure_waveform.pulse
    for kernel in (*kernels, pulse):
        if kernel is None:
            continue
        unavailable = {
            type(axis)
            for axis in kernel.conditioning_axes
            if type(axis) not in available_roles
        }
        if unavailable:
            names = ", ".join(sorted(role.__name__ for role in unavailable))
            raise ValueError(
                f"profile kernel conditioning roles were not supplied: {names}"
            )


def _veto_pulse(sample_axis: SampleAxis) -> Pulse:
    period_ns = sample_axis.step / 1000.0
    support_ns = 2020.27
    full_coefficient_count = math.ceil(support_ns / period_ns)

    def raw(time_ns: float) -> float:
        x = time_ns - 232.89
        gaussian = math.exp(-(x**2) / (2.0 * 507.72**2)) / math.sqrt(
            2.0 * math.pi * 507.72**2
        )
        first = 1.0 + math.erf(
            (x - (-81.92)) / (math.sqrt(2.0) * 147.28)
        )
        second = 1.0 + math.erf(
            (x - (-176.50)) / (math.sqrt(2.0) * 45.69)
        )
        return gaussian * first * second

    values = tuple(
        raw(index * period_ns) for index in range(full_coefficient_count)
    )
    normalization = max(abs(value) for value in values)
    coefficient_count = min(sample_axis.count, full_coefficient_count)
    coefficients = torch.tensor(
        tuple(
            value / normalization * -14.5912372
            for value in values[:coefficient_count]
        ),
        dtype=torch.float64,
    )
    return Pulse(
        quantity=quantities(coefficients, "mV"),
        conditioning_axes=(),
        operation_axes=(
            OffsetAxis(
                relative_to=SampleAxis,
                offsets=tuple(range(coefficient_count)),
            ),
        ),
    )


def ds20k_veto(
    *,
    sample_axis: SampleAxis,
    channel_axis: ChannelAxis | None = None,
    example_axis: ExampleAxis | None = None,
) -> ReadoutConfig:
    """Return a fresh provisional DS20k Veto profile for supplied geometry."""

    frequency_left_edges = (
        0.0,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        62.5,
    )
    power_density = (
        4.0e-8,
        7.0e-8,
        6.0e-8,
        3.0e-8,
        7.0e-9,
        1.0e-9,
        2.0e-10,
        5.0e-11,
        0.0,
    )
    config = ReadoutConfig(
        charge=ChargeConfig(
            correlated_avalanche_generations=NonnegativeInteger(0),
            dark_counts=DarkCountRate(
                quantity=quantity(100.0, "kHz"),
                conditioning_axes=(),
                operation_axes=(),
            ),
        ),
        pure_waveform=PureWaveformConfig(pulse=_veto_pulse(sample_axis)),
        noise_waveform=NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges=quantities(frequency_left_edges, "MHz"),
                frequency_stop=quantity(250.0, "MHz"),
                power_density=quantities(power_density, "mV ** 2 / Hz"),
            )
        ),
        analog_waveform=AnalogWaveformConfig(),
        digitized_waveform=DigitizedWaveformConfig(
            bit_depth=PositiveInteger(16),
            input_minimum=quantity(-3900.0, "mV"),
            input_maximum=quantity(100.0, "mV"),
            analog_gain_db=NonnegativeFloat(3.5218),
        ),
    )
    available_roles: set[type] = {SampleAxis}
    if channel_axis is not None:
        available_roles.add(ChannelAxis)
    if example_axis is not None:
        available_roles.add(ExampleAxis)
    _require_available_conditioning_axes(
        config,
        available_roles=frozenset(available_roles),
    )
    return config
