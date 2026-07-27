from collections.abc import Iterable
from itertools import permutations
import math
from typing import Any, override
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    CounterRng,
    GaussianDistribution,
    NonnegativeFloat,
    PositiveFloat,
    RngAddress,
    RngElements,
    RngKey,
    TensorAxis,
    Threefry4x32,
)

from tensor_dslab import (
    quantities,
    quantity,
    ChannelAxis,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    SampleAxis,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    _prepare_psd_powers as _prepare_psd_powers_prepared,
    prepare_noise_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.produce import (
    produce_noise_waveform as _produce_noise_waveform_prepared,
)
from tensor_dslab.readout.noise_waveform.runtime.validate import (
    validate_noise_waveform,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime, prepare_sampling
from tensor_dslab.readout.runtime.keys import (
    PSD_NOISE_RNG_KEY,
    WHITE_NOISE_RNG_KEY,
)


SEEDS = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
AXIS_ORDERS = tuple(
    permutations((ExampleAxis, ChannelAxis, SampleAxis))
)


def _ns(value: int | float):
    return quantity(value, "ns")


def _hz(value: int | float):
    return quantity(value, "Hz")


def _mv(value: int | float):
    return quantity(value, "mV")


def _density(value: int | float):
    return quantity(value, "mV ** 2 / Hz")


def _hzs(values: tuple[int | float, ...]):
    return quantities(values, "Hz")


def _densities(values: tuple[int | float, ...]):
    return quantities(values, "mV ** 2 / Hz")


class _FailingRng(CounterRng):
    __slots__ = ()

    @override
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        raise AssertionError(
            f"unexpected RNG request: {key=}, {quantum=}, {block=}"
        )


def _sampling(*, count: int, period_ps: int = 1_000) -> SamplingRuntime:
    return SamplingRuntime(
        sample_count=count,
        sample_period_ps=period_ps,
        sample_dimension=2,
    )


def _prepare_psd_powers(
    config: PsdNoiseConfig,
    *,
    sampling: SamplingRuntime,
    dtype: torch.dtype,
) -> tuple[float, ...]:
    return _prepare_psd_powers_prepared(
        tuple(float(value.magnitude) for value in config.frequency_left_edges),
        float(config.frequency_stop.magnitude),
        tuple(float(value.magnitude) for value in config.power_density),
        sampling=sampling,
        dtype=dtype,
    )


def _produce_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingRuntime,
    config: NoiseWaveformConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> NoiseWaveform:
    sampling_runtime = prepare_sampling(photoelectrons)
    if (
        sampling_runtime.sample_count != sampling.sample_count
        or sampling_runtime.sample_period_ps != sampling.sample_period_ps
    ):
        raise AssertionError("test source and sampling runtime diverged")
    runtime = prepare_noise_waveform(
        config,
        sampling=sampling_runtime,
        shape=photoelectrons.shape,
        floating_dtype=floating_dtype,
        device=photoelectrons.tensor.device,
    )
    result = _produce_noise_waveform_prepared(
        photoelectrons,
        runtime=runtime,
        rng=rng,
    )
    validate_noise_waveform(result, source=photoelectrons, runtime=runtime)
    return result


def _axes(
    sampling: SamplingRuntime,
    *,
    order: tuple[type[TensorAxis[Any]], ...] = (
        ExampleAxis,
        ChannelAxis,
        SampleAxis,
    ),
    examples: int = 2,
    channels: int = 3,
    label_prefix: str = "original",
) -> tuple[TensorAxis[Any], ...]:
    available: dict[type[TensorAxis[Any]], TensorAxis[Any]] = {
        ExampleAxis: ExampleAxis(count=examples),
        ChannelAxis: ChannelAxis(
            labels=tuple(
                f"{label_prefix}-channel-{index}" for index in range(channels)
            )
        ),
        SampleAxis: SampleAxis(
            start=0,
            step=sampling.sample_period_ps,
            count=sampling.sample_count,
        ),
    }
    return tuple(available[axis_type] for axis_type in order)


def _photoelectrons(
    sampling: SamplingRuntime,
    *,
    order: tuple[type[TensorAxis[Any]], ...] = (
        ExampleAxis,
        ChannelAxis,
        SampleAxis,
    ),
    examples: int = 2,
    channels: int = 3,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
    label_prefix: str = "original",
    fill_offset: int = 0,
) -> Photoelectrons:
    axes = _axes(
        sampling,
        order=order,
        examples=examples,
        channels=channels,
        label_prefix=label_prefix,
    )
    shape = tuple(axis.size for axis in axes)
    values = torch.arange(
        math.prod(shape),
        dtype=torch.int64,
        device=device,
    ).reshape(shape)
    values = values + fill_offset
    if noncontiguous:
        backing = torch.empty((*shape, 2), dtype=torch.int64, device=device)
        view = backing[..., 0]
        view.copy_(values)
        values = view
    return Photoelectrons(tensor=values, axes=axes)


def _zero_config() -> NoiseWaveformConfig:
    return NoiseWaveformConfig(model=ZeroNoiseConfig())


def _white_config(rms: float = 1.0) -> NoiseWaveformConfig:
    return NoiseWaveformConfig(
        model=WhiteNoiseConfig(rms=_mv(rms))
    )


def _flat_psd_config(
    *,
    density: float = 2.0e-9,
    stop_hz: float = 500_000_000.0,
) -> NoiseWaveformConfig:
    return NoiseWaveformConfig(
        model=PsdNoiseConfig(
            frequency_left_edges=_hzs((0.0,)),
            frequency_stop=_hz(stop_hz),
            power_density=_densities((density,)),
        )
    )


def _sample_last(field: NoiseWaveform) -> torch.Tensor:
    return field.tensor.movedim(field.dimension_of(SampleAxis), -1)


def _psd_normals(
    model: PsdNoiseConfig,
    *,
    seed: int,
    row_count: int,
    frequency_count: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    elements = RngElements.from_shape(
        (row_count, frequency_count),
        device=device,
    ).slice(1, 1, None)
    return GaussianDistribution(
        mean=0.0,
        standard_deviation=1.0,
        dtype=dtype,
        ordinal=0,
        count=2,
    ).draw(
        rng=Threefry4x32(seed=seed),
        address=RngAddress.root(
            key=PSD_NOISE_RNG_KEY,
            elements=elements,
            shape=(),
        ),
    )


def _independent_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()


def _round(value: float, dtype: torch.dtype) -> float:
    return float(torch.tensor(value, dtype=dtype))


def _reference_psd_powers(
    config: PsdNoiseConfig,
    *,
    sampling: SamplingRuntime,
    dtype: torch.dtype,
) -> tuple[float, ...]:
    sample_count = sampling.sample_count
    sample_rate = 1.0e12 / sampling.sample_period_ps
    spacing = sample_rate / sample_count
    nyquist = sample_rate / 2.0
    frequency_count = sample_count // 2 + 1
    target_left = (0.0,) + tuple(
        (index - 0.5) * spacing for index in range(1, frequency_count)
    )
    target_right = target_left[1:] + (nyquist,)
    source_left = tuple(item.magnitude for item in config.frequency_left_edges)
    source_right = source_left[1:] + (config.frequency_stop.magnitude,)
    density = tuple(item.magnitude for item in config.power_density)
    integrated = tuple(
        math.fsum(
            source_power
            * max(
                0.0,
                min(source_stop, target_stop)
                - max(source_start, target_start),
            )
            for source_start, source_stop, source_power in zip(
                source_left,
                source_right,
                density,
            )
        )
        for target_start, target_stop in zip(target_left, target_right)
    )
    return (0.0,) + tuple(_round(power, dtype) for power in integrated[1:])


def _delta(dtype: torch.dtype, scale: float, length: int) -> float:
    return (
        64
        * torch.finfo(dtype).eps
        * max(1, math.ceil(math.log2(length)))
        * abs(scale)
    )
