"""Private preparation of trusted digitized waveform runtime facts."""

from dataclasses import dataclass
import math
from typing import final

import torch
from tensor_core.tensor.validation import require_representable_float

from tensor_dslab.common.units import canonical_magnitude
from tensor_dslab.readout.digitized_waveform.config import (
    DigitizedWaveformConfig,
)


@final
@dataclass(frozen=True, slots=True)
class DigitizedWaveformRuntime:
    maximum_code: int
    zero: torch.Tensor
    maximum: torch.Tensor
    slope_per_mv: torch.Tensor
    intercept: torch.Tensor
    lower_input_mv: torch.Tensor
    upper_input_mv: torch.Tensor


def prepare_digitized_waveform(
    config: DigitizedWaveformConfig,
    *,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> DigitizedWaveformRuntime:
    maximum_code = (1 << config.bit_depth.value) - 1
    input_minimum_mv = canonical_magnitude(config.input_minimum)
    input_maximum_mv = canonical_magnitude(config.input_maximum)
    try:
        gain = 10.0 ** (config.analog_gain_db.value / 20.0)
        span = input_maximum_mv - input_minimum_mv
        slope = gain * maximum_code / span
        intercept = -input_minimum_mv * maximum_code / span
        lower_input_mv = input_minimum_mv / gain
        upper_input_mv = input_maximum_mv / gain
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
        slope_per_mv=slope_tensor,
        intercept=intercept_tensor,
        lower_input_mv=lower_input,
        upper_input_mv=upper_input,
    )
