from __future__ import annotations

from dataclasses import dataclass
import math
from typing import final

import torch

from tensor_dslab.readout.digitized_waveform.config import (
    DigitizedWaveformConfig,
)
from tensor_dslab.readout.requirements import require_representable_float


@final
@dataclass(frozen=True, slots=True)
class DigitizedWaveformRuntime:
    maximum_code: int
    zero: torch.Tensor
    maximum: torch.Tensor
    slope: torch.Tensor
    intercept: torch.Tensor
    lower_input: torch.Tensor
    upper_input: torch.Tensor


def prepare_digitized_waveform(
    config: DigitizedWaveformConfig,
    *,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> DigitizedWaveformRuntime:
    if type(config) is not DigitizedWaveformConfig:
        raise TypeError("config must be exactly DigitizedWaveformConfig")
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")

    maximum_code = (1 << config.bit_depth.value) - 1
    try:
        gain = 10.0 ** (config.analog_gain_db.value / 20.0)
        span = config.input_max_mv.value - config.input_min_mv.value
        slope = gain * maximum_code / span
        intercept = -config.input_min_mv.value * maximum_code / span
        lower_input_mv = config.input_min_mv.value / gain
        upper_input_mv = config.input_max_mv.value / gain
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("ADC transfer cannot be represented in binary64") from error

    derived = (
        gain,
        span,
        slope,
        intercept,
        lower_input_mv,
        upper_input_mv,
    )
    if not all(math.isfinite(value) for value in derived):
        raise ValueError("ADC transfer constants must be finite in binary64")
    if span <= 0.0 or slope <= 0.0:
        raise ValueError("ADC span and slope must be positive in binary64")

    rounded_maximum_code = require_representable_float(
        maximum_code,
        dtype=floating_dtype,
        field="ADC maximum code",
    )
    rounded_gain = require_representable_float(
        gain,
        dtype=floating_dtype,
        field="ADC gain",
    )
    rounded_span = require_representable_float(
        span,
        dtype=floating_dtype,
        field="ADC span",
    )
    rounded_slope = require_representable_float(
        slope,
        dtype=floating_dtype,
        field="ADC slope",
    )
    rounded_intercept = require_representable_float(
        intercept,
        dtype=floating_dtype,
        field="ADC intercept",
    )
    rounded_lower_input = require_representable_float(
        lower_input_mv,
        dtype=floating_dtype,
        field="ADC lower input threshold",
    )
    rounded_upper_input = require_representable_float(
        upper_input_mv,
        dtype=floating_dtype,
        field="ADC upper input threshold",
    )
    if rounded_maximum_code != maximum_code:
        raise ValueError("ADC maximum code is not exact in the waveform dtype")
    if rounded_gain <= 0.0 or rounded_span <= 0.0 or rounded_slope <= 0.0:
        raise ValueError("ADC gain, span, and slope must remain positive")
    if rounded_lower_input >= rounded_upper_input:
        raise ValueError("ADC input thresholds collapse in the waveform dtype")

    scalar_values = (
        0.0,
        rounded_maximum_code,
        rounded_slope,
        rounded_intercept,
        rounded_lower_input,
        rounded_upper_input,
    )
    (
        zero,
        maximum,
        slope_tensor,
        intercept_tensor,
        lower_input,
        upper_input,
    ) = tuple(
        torch.tensor(value, dtype=floating_dtype, device=device)
        for value in scalar_values
    )
    return DigitizedWaveformRuntime(
        maximum_code=maximum_code,
        zero=zero,
        maximum=maximum,
        slope=slope_tensor,
        intercept=intercept_tensor,
        lower_input=lower_input,
        upper_input=upper_input,
    )
