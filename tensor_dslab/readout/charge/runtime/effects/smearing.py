"""Private charge-smearing preparation and execution."""

import math
from dataclasses import dataclass
from typing import final

import torch
from tensor_core import CounterRng, GaussianDistribution, RngElements, RngKey
from tensor_core.tensor.validation import require_representable_float

from tensor_dslab.readout.charge.config import ChargeSmearingConfig
from tensor_dslab.readout.runtime.keys import CHARGE_SMEARING_RNG_KEY
from tensor_dslab.readout.runtime.addresses import charge_smearing_address


@final
@dataclass(frozen=True, slots=True)
class ChargeSmearingRuntime:
    represented_sigma: float
    rng_key: RngKey


def _prepare_smearing_sigma(
    config: ChargeSmearingConfig,
    *,
    floating_dtype: torch.dtype,
    ledger_bound: float,
    device: torch.device,
) -> float:
    requested = config.relative_sigma.value
    if requested == 0.0:
        return 0.0
    represented = require_representable_float(
        requested,
        dtype=floating_dtype,
        field="ChargeSmearingConfig.relative_sigma",
    )
    if not math.isfinite(represented) or represented <= 0.0:
        raise ValueError(
            "charge-smearing width is not positive and finite in the dtype"
        )
    precision = 24 if floating_dtype is torch.float32 else 53
    maximum_normal = math.sqrt(-2.0 * math.log(2.0**-precision))
    maximum_normal = math.nextafter(maximum_normal, math.inf)
    square_root = math.nextafter(math.sqrt(ledger_bound), math.inf)
    scale = math.nextafter(represented * square_root, math.inf)
    excursion = math.nextafter(maximum_normal * scale, math.inf)
    maximum = math.nextafter(ledger_bound + excursion, math.inf)
    if not math.isfinite(maximum) or maximum > torch.finfo(floating_dtype).max:
        raise ValueError("charge-smearing finite envelope exceeds the dtype")

    compatibility_radius = (
        float.fromhex("0x1.7128ac8de74b6p+2")
        if floating_dtype is torch.float32
        else float.fromhex("0x1.124b2800eda49p+3")
    )
    zero = torch.tensor(0.0, dtype=floating_dtype, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        maximum_ledger = torch.tensor(
            ledger_bound,
            dtype=floating_dtype,
            device=device,
        )
        if float(maximum_ledger) > ledger_bound:
            maximum_ledger = torch.nextafter(maximum_ledger, zero)
        represented_sigma = torch.tensor(
            represented,
            dtype=floating_dtype,
            device=device,
        )
        prepared_scale = represented_sigma * torch.sqrt(maximum_ledger)
        compatibility_envelope = maximum_ledger.to(
            torch.float64
        ) + prepared_scale.to(torch.float64) * torch.tensor(
            compatibility_radius,
            dtype=torch.float64,
            device=device,
        )
    if (
        not bool(torch.isfinite(maximum_ledger).item())
        or not bool(torch.isfinite(prepared_scale).item())
        or not bool(torch.isfinite(compatibility_envelope).item())
        or float(compatibility_envelope) > torch.finfo(floating_dtype).max
    ):
        raise ValueError("charge-smearing finite envelope exceeds the dtype")
    return represented


def prepare_charge_smearing(
    config: ChargeSmearingConfig,
    *,
    floating_dtype: torch.dtype,
    ledger_bound: float,
    device: torch.device,
) -> ChargeSmearingRuntime:
    return ChargeSmearingRuntime(
        represented_sigma=_prepare_smearing_sigma(
            config,
            floating_dtype=floating_dtype,
            ledger_bound=ledger_bound,
            device=device,
        ),
        rng_key=CHARGE_SMEARING_RNG_KEY,
    )


def simulate_charge_smearing(
    charge_pe: torch.Tensor,
    charge_square_sum: torch.Tensor,
    *,
    runtime: ChargeSmearingRuntime,
    rng: CounterRng,
    elements: RngElements,
) -> torch.Tensor:
    if charge_pe.dtype not in (torch.float32, torch.float64):
        raise TypeError("charge_pe must use a supported floating dtype")
    if charge_square_sum.dtype is not charge_pe.dtype:
        raise ValueError("charge ledgers must have the same dtype")
    if charge_square_sum.shape != charge_pe.shape:
        raise ValueError("charge ledgers must have the same shape")
    if charge_square_sum.device != charge_pe.device:
        raise ValueError("charge ledgers must be on the same device")
    for field, value in (("S1", charge_pe), ("S2", charge_square_sum)):
        if not bool(torch.all(torch.isfinite(value) & (value >= 0.0)).item()):
            raise ValueError(f"charge-smearing {field} must be finite and nonnegative")
    if runtime.represented_sigma == 0.0:
        return charge_pe
    if runtime.represented_sigma <= 0.0:
        raise ValueError("charge-smearing width is invalid in the Charge dtype")
    sigma = torch.tensor(
        runtime.represented_sigma,
        dtype=charge_pe.dtype,
        device=charge_pe.device,
    )
    zero = torch.tensor(0.0, dtype=charge_pe.dtype, device=charge_pe.device)
    with torch.autocast(device_type=charge_pe.device.type, enabled=False):
        scale = sigma * torch.sqrt(charge_square_sum)
    draw = GaussianDistribution(
        mean=charge_pe,
        standard_deviation=scale,
        dtype=charge_pe.dtype,
        ordinal=0,
        count=1,
    ).draw(
        rng=rng,
        address=charge_smearing_address(elements, key=runtime.rng_key),
    )
    with torch.autocast(device_type=charge_pe.device.type, enabled=False):
        result = torch.maximum(draw, zero)
    for field, value in (
        ("scale", scale),
        ("draw", draw),
        ("result", result),
    ):
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise RuntimeError(f"charge-smearing {field} is nonfinite")
    return result
