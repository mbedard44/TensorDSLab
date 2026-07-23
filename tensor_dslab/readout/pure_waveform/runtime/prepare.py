from __future__ import annotations

from dataclasses import dataclass
import math
from typing import final

import torch

from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.requirements import require_representable_float
from tensor_dslab.readout.pure_waveform.config import (
    PureWaveformConfig,
    TpcFebSnrPulseConfig,
    VetoPduPulseConfig,
)
from tensor_dslab.readout.runtime.sampling import (
    SamplingRuntime,
)


@final
@dataclass(frozen=True, slots=True)
class PureWaveformRuntime:
    sampling: SamplingRuntime
    kernel: torch.Tensor


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


def _tpc_raw(
    t_ns: float,
    *,
    fast_time_constant_ns: float,
    slow_time_constant_ns: float,
) -> float:
    return math.exp(-t_ns / slow_time_constant_ns) - math.exp(
        -t_ns / fast_time_constant_ns
    )


def _veto_raw(
    t_ns: float,
    *,
    gaussian_center_ns: float,
    gaussian_width_ns: float,
    edge_offset_1_ns: float,
    edge_width_1_ns: float,
    edge_offset_2_ns: float,
    edge_width_2_ns: float,
) -> float:
    x = t_ns - gaussian_center_ns
    gaussian = math.exp(
        -(x**2) / (2.0 * gaussian_width_ns**2)
    ) / math.sqrt(2.0 * math.pi * gaussian_width_ns**2)
    first_edge = 1.0 + math.erf(
        (x - edge_offset_1_ns) / (math.sqrt(2.0) * edge_width_1_ns)
    )
    second_edge = 1.0 + math.erf(
        (x - edge_offset_2_ns) / (math.sqrt(2.0) * edge_width_2_ns)
    )
    return gaussian * first_edge * second_edge


def prepare_pure_waveform(
    config: PureWaveformConfig,
    *,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> PureWaveformRuntime:
    try:
        sample_period_ns = sampling.sample_period_ps / 1000.0
    except OverflowError as error:
        raise ValueError("sample period cannot be represented in binary64") from error
    if not math.isfinite(sample_period_ns) or sample_period_ns <= 0.0:
        raise ValueError("sample period must be finite and positive in binary64")

    model = config.model
    if type(model) is TpcFebSnrPulseConfig:
        fast_time_constant_ns = canonical_magnitude(model.fast_time_constant)
        slow_time_constant_ns = canonical_magnitude(model.slow_time_constant)
        support_time_ns = canonical_magnitude(model.support_time)
        peak_voltage_mv_per_pe = canonical_magnitude(
            model.peak_voltage_per_photoelectron
        )
        raw_at = lambda time_ns: _tpc_raw(
            time_ns,
            fast_time_constant_ns=fast_time_constant_ns,
            slow_time_constant_ns=slow_time_constant_ns,
        )
    elif type(model) is VetoPduPulseConfig:
        gaussian_center_ns = canonical_magnitude(model.gaussian_center)
        gaussian_width_ns = canonical_magnitude(model.gaussian_width)
        edge_offset_1_ns = canonical_magnitude(model.edge_offset_1)
        edge_width_1_ns = canonical_magnitude(model.edge_width_1)
        edge_offset_2_ns = canonical_magnitude(model.edge_offset_2)
        edge_width_2_ns = canonical_magnitude(model.edge_width_2)
        support_time_ns = canonical_magnitude(model.support_time)
        peak_voltage_mv_per_pe = canonical_magnitude(
            model.peak_voltage_per_photoelectron
        )
        raw_at = lambda time_ns: _veto_raw(
            time_ns,
            gaussian_center_ns=gaussian_center_ns,
            gaussian_width_ns=gaussian_width_ns,
            edge_offset_1_ns=edge_offset_1_ns,
            edge_width_1_ns=edge_width_1_ns,
            edge_offset_2_ns=edge_offset_2_ns,
            edge_width_2_ns=edge_width_2_ns,
        )
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

    rounded_peak = require_representable_float(
        peak_voltage_mv_per_pe,
        dtype=floating_dtype,
        field="pulse normalized extremum",
    )
    if rounded_peak == 0.0:
        raise ValueError("pulse normalized extremum vanishes in the Charge dtype")

    coefficient_count = min(template_sample_count, sampling.sample_count)
    rounded_coefficients = [
        require_representable_float(
            value / normalization * peak_voltage_mv_per_pe,
            dtype=floating_dtype,
            field=f"pulse coefficient[{index}]",
        )
        for index, value in enumerate(raw[:coefficient_count])
    ]
    coefficients = torch.tensor(
        rounded_coefficients,
        dtype=floating_dtype,
        device=device,
    )
    kernel = coefficients.flip(0).reshape(1, 1, coefficient_count)
    return PureWaveformRuntime(
        sampling=sampling,
        kernel=kernel,
    )
