"""Private preparation of trusted charge runtime facts."""

from dataclasses import dataclass
from typing import final

import torch
from tensor_core.tensor.validation import require_tensor_allocation
from tensor_core.random.validation import require_count_tensor

from tensor_dslab.readout.charge.config import ChargeConfig
from tensor_dslab.readout.charge.runtime.effects.correlated_avalanches import (
    CorrelatedAvalancheRuntime,
    prepare_correlated_avalanches,
    prepare_ledger_envelope,
)
from tensor_dslab.readout.charge.runtime.effects.dark_counts import (
    DarkCountRuntime,
    prepare_dark_counts,
)
from tensor_dslab.readout.charge.runtime.effects.smearing import (
    ChargeSmearingRuntime,
    prepare_charge_smearing,
)
from tensor_dslab.readout.charge.runtime.effects.timing_jitter import (
    TimingJitterRuntime,
    prepare_timing_jitter,
)
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


@final
@dataclass(frozen=True, slots=True)
class ChargeRuntime:
    sampling: SamplingRuntime
    floating_dtype: torch.dtype
    dark: DarkCountRuntime | None
    timing_jitter: TimingJitterRuntime | None
    correlated_avalanches: CorrelatedAvalancheRuntime | None
    smearing: ChargeSmearingRuntime | None


def prepare_charge(
    config: ChargeConfig,
    *,
    photoelectrons: Photoelectrons,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
) -> ChargeRuntime:
    device = photoelectrons.tensor.device

    source = photoelectrons.tensor
    require_count_tensor(source, "Photoelectrons source")
    shape = tuple(source.shape)
    tensor_numel = require_tensor_allocation(
        shape,
        "Charge source",
        element_size=source.element_size(),
        upper=1 << 63,
    )
    require_tensor_allocation(
        shape,
        "Charge output",
        element_size=torch.empty((), dtype=floating_dtype).element_size(),
        upper=1 << 63,
    )

    dark = (
        None
        if config.dark_count is None
        else prepare_dark_counts(config.dark_count, sampling=sampling)
    )
    timing_jitter = (
        None
        if config.timing_jitter is None
        else prepare_timing_jitter(
            config.timing_jitter,
            sampling=sampling,
            tensor_numel=tensor_numel,
        )
    )

    if config.correlated_avalanches is None:
        _, ledger_bound = prepare_ledger_envelope(
            floating_dtype=floating_dtype,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=sampling.sample_count,
        )
        correlated_avalanches: CorrelatedAvalancheRuntime | None = None
    else:
        correlated_avalanches = prepare_correlated_avalanches(
            config.correlated_avalanches,
            sampling=sampling,
            floating_dtype=floating_dtype,
            tensor_numel=tensor_numel,
        )
        ledger_bound = correlated_avalanches.ledger_bound

    smearing = (
        None
        if config.smearing is None
        else prepare_charge_smearing(
            config.smearing,
            floating_dtype=floating_dtype,
            ledger_bound=ledger_bound,
            device=device,
        )
    )
    return ChargeRuntime(
        sampling=sampling,
        floating_dtype=floating_dtype,
        dark=dark,
        timing_jitter=timing_jitter,
        correlated_avalanches=correlated_avalanches,
        smearing=smearing,
    )
