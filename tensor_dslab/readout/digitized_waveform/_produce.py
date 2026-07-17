from __future__ import annotations

import math

import torch

from tensor_dslab.readout._requirements import _require_representable_float
from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.digitized_waveform.config import (
    DigitizedWaveformConfig,
)
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform


def _produce_digitized_waveform(
    analog: AnalogWaveform,
    *,
    config: DigitizedWaveformConfig,
) -> DigitizedWaveform:
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

    dtype = analog.tensor.dtype
    rounded_maximum_code = _require_representable_float(
        maximum_code,
        dtype=dtype,
        field="ADC maximum code",
    )
    rounded_gain = _require_representable_float(
        gain,
        dtype=dtype,
        field="ADC gain",
    )
    rounded_span = _require_representable_float(
        span,
        dtype=dtype,
        field="ADC span",
    )
    rounded_slope = _require_representable_float(
        slope,
        dtype=dtype,
        field="ADC slope",
    )
    rounded_intercept = _require_representable_float(
        intercept,
        dtype=dtype,
        field="ADC intercept",
    )
    rounded_lower_input = _require_representable_float(
        lower_input_mv,
        dtype=dtype,
        field="ADC lower input threshold",
    )
    rounded_upper_input = _require_representable_float(
        upper_input_mv,
        dtype=dtype,
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
        torch.tensor(value, dtype=dtype, device=analog.tensor.device)
        for value in scalar_values
    )

    interior = torch.clamp(
        torch.add(
            torch.mul(analog.tensor, slope_tensor),
            intercept_tensor,
        ),
        min=zero,
        max=maximum,
    )
    code_float = torch.where(
        analog.tensor <= lower_input,
        zero,
        torch.where(analog.tensor >= upper_input, maximum, interior),
    )
    values = code_float.to(dtype=torch.int32)
    return DigitizedWaveform(tensor=values, axes=analog.axes)
