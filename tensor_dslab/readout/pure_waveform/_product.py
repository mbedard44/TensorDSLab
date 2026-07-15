from __future__ import annotations

import math

import torch
from torch.nn import functional

from tensor_dslab.common import SampleAxis, SamplingConfig
from tensor_dslab.readout._requirements import _require_sampling
from tensor_dslab.readout.charge import Charge
from tensor_dslab.readout.pure_waveform.types import (
    PureWaveform,
    PureWaveformConfig,
    TpcFebSnrPulseConfig,
    VetoPduPulseConfig,
)


def _template_sample_count(
    *,
    support_time_ns: float,
    sample_period_ns: float,
) -> int:
    if (
        not math.isfinite(support_time_ns)
        or support_time_ns <= 0.0
        or not math.isfinite(sample_period_ns)
        or sample_period_ns <= 0.0
    ):
        raise ValueError("pulse support cannot be represented at this sampling")

    maximum_count = (1 << 63) - 1
    if maximum_count * sample_period_ns < support_time_ns:
        raise ValueError("pulse template sample count is outside int64")

    # The first excluded left edge is the sample count. Binary search keeps
    # the exact binary64 comparison while bounding even extreme preflight.
    included_index = 0
    excluded_index = maximum_count
    while included_index + 1 < excluded_index:
        candidate = (included_index + excluded_index) // 2
        if candidate * sample_period_ns < support_time_ns:
            included_index = candidate
        else:
            excluded_index = candidate
    return excluded_index


def _tpc_raw(t_ns: float, config: TpcFebSnrPulseConfig) -> float:
    return math.exp(-t_ns / config.slow_time_constant_ns.value) - math.exp(
        -t_ns / config.fast_time_constant_ns.value
    )


def _veto_raw(t_ns: float, config: VetoPduPulseConfig) -> float:
    x = t_ns - config.gaussian_center_ns.value
    gaussian_width_ns = config.gaussian_width_ns.value
    gaussian = math.exp(
        -(x**2) / (2.0 * gaussian_width_ns**2)
    ) / math.sqrt(2.0 * math.pi * gaussian_width_ns**2)
    first_edge = 1.0 + math.erf(
        (x - config.edge_offset_1_ns.value)
        / (math.sqrt(2.0) * config.edge_width_1_ns.value)
    )
    second_edge = 1.0 + math.erf(
        (x - config.edge_offset_2_ns.value)
        / (math.sqrt(2.0) * config.edge_width_2_ns.value)
    )
    return gaussian * first_edge * second_edge


def _round_finite_scalar(
    value: float,
    *,
    dtype: torch.dtype,
    field: str,
) -> float:
    rounded = float(torch.tensor(value, dtype=dtype, device="cpu"))
    if not math.isfinite(rounded):
        raise ValueError(f"{field} is not finite in the Charge dtype")
    return rounded


def _product_pure_waveform(
    charge: Charge,
    *,
    sampling: SamplingConfig,
    config: PureWaveformConfig,
) -> PureWaveform:
    _require_sampling(charge, sampling)

    try:
        sample_period_ns = sampling.sample_period_ps.value / 1000.0
    except OverflowError as error:
        raise ValueError("sample period cannot be represented in binary64") from error
    if not math.isfinite(sample_period_ns) or sample_period_ns <= 0.0:
        raise ValueError("sample period must be finite and positive in binary64")

    model = config.model
    if type(model) is TpcFebSnrPulseConfig:
        support_time_ns = model.support_time_ns.value
        raw_at = lambda time_ns: _tpc_raw(time_ns, model)
    elif type(model) is VetoPduPulseConfig:
        support_time_ns = model.support_time_ns.value
        raw_at = lambda time_ns: _veto_raw(time_ns, model)
    else:
        raise TypeError("PureWaveformConfig.model is not recognized")

    template_sample_count = _template_sample_count(
        support_time_ns=support_time_ns,
        sample_period_ns=sample_period_ns,
    )
    try:
        raw = [
            raw_at(index * sample_period_ns)
            for index in range(template_sample_count)
        ]
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise ValueError("pulse template evaluation failed in binary64") from error
    if not raw or not all(math.isfinite(value) for value in raw):
        raise ValueError("pulse template must be nonempty and finite")

    normalization = max(abs(value) for value in raw)
    if not math.isfinite(normalization) or normalization == 0.0:
        raise ValueError("pulse template sampled extremum must be finite and nonzero")

    peak_voltage_mv_per_pe = model.peak_voltage_mv_per_pe.value
    rounded_peak = _round_finite_scalar(
        peak_voltage_mv_per_pe,
        dtype=charge.tensor.dtype,
        field="pulse normalized extremum",
    )
    if rounded_peak == 0.0:
        raise ValueError("pulse normalized extremum vanishes in the Charge dtype")

    coefficient_count = min(template_sample_count, sampling.sample_count.value)
    rounded_coefficients = [
        _round_finite_scalar(
            value / normalization * peak_voltage_mv_per_pe,
            dtype=charge.tensor.dtype,
            field=f"pulse coefficient[{index}]",
        )
        for index, value in enumerate(raw[:coefficient_count])
    ]
    coefficients = torch.tensor(
        rounded_coefficients,
        dtype=charge.tensor.dtype,
        device=charge.tensor.device,
    )

    sample_dimension = charge.dimension_of(SampleAxis)
    sample_last = charge.tensor.movedim(sample_dimension, -1)
    sample_count = sample_last.shape[-1]
    rows = sample_last.reshape(-1, 1, sample_count)
    kernel = coefficients.flip(0).reshape(1, 1, coefficient_count)
    with torch.autocast(device_type=charge.tensor.device.type, enabled=False):
        padded = functional.pad(rows, (coefficient_count - 1, 0))
        convolved = functional.conv1d(padded, kernel)
    values = convolved.reshape(sample_last.shape).movedim(-1, sample_dimension)
    return PureWaveform(tensor=values, axes=charge.axes)
