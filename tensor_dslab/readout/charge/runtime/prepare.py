from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from tensor_core import RngKey

from tensor_dslab.readout.charge.config import ChargeConfig
from tensor_dslab.readout.charge.runtime.effects.correlated_avalanches import (
    CorrelatedAvalancheRuntime,
    prepare_correlated_avalanches,
    prepare_ledger_envelope,
)
from tensor_dslab.readout.charge.runtime.effects.counts import (
    require_count_domain,
    require_tensor_allocation,
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
    rng_roles: tuple[tuple[str, RngKey], ...]
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
    if type(config) is not ChargeConfig:
        raise TypeError("config must be exactly ChargeConfig")
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")
    device = photoelectrons.tensor.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Charge production supports only CPU and CUDA")

    source = photoelectrons.tensor
    require_count_domain(source, field="Photoelectrons source")
    shape = tuple(source.shape)
    tensor_numel = require_tensor_allocation(
        shape,
        element_size=source.element_size(),
        field="Charge source",
    )
    require_tensor_allocation(
        shape,
        element_size=torch.empty((), dtype=floating_dtype).element_size(),
        field="Charge output",
    )
    rng_roles: list[tuple[str, RngKey]] = []
    if config.dark_count is not None:
        rng_roles.append(("dark count", config.dark_count.rng_key))
    if config.timing_jitter is not None:
        rng_roles.append(("timing jitter", config.timing_jitter.rng_key))
    correlated_config = config.correlated_avalanches
    if correlated_config is not None:
        if correlated_config.direct_crosstalk is not None:
            rng_roles.extend(
                (
                    (
                        "direct crosstalk retained",
                        correlated_config.direct_crosstalk.retained_rng_key,
                    ),
                    (
                        "direct crosstalk overflow",
                        correlated_config.direct_crosstalk.overflow_rng_key,
                    ),
                )
            )
        if correlated_config.delayed_crosstalk is not None:
            rng_roles.extend(
                (
                    (
                        "delayed crosstalk retained",
                        correlated_config.delayed_crosstalk.retained_rng_key,
                    ),
                    (
                        "delayed crosstalk overflow",
                        correlated_config.delayed_crosstalk.overflow_rng_key,
                    ),
                )
            )
        if correlated_config.afterpulse is not None:
            rng_roles.append(("afterpulse", correlated_config.afterpulse.rng_key))
    if config.smearing is not None:
        rng_roles.append(("charge smearing", config.smearing.rng_key))

    dark = (
        None
        if config.dark_count is None
        else prepare_dark_counts(config.dark_count, sampling=sampling)
    )
    timing_jitter = (
        None
        if config.timing_jitter is None
        or config.timing_jitter.sigma_ns.value == 0.0
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
        rng_roles=tuple(rng_roles),
        dark=dark,
        timing_jitter=timing_jitter,
        correlated_avalanches=correlated_avalanches,
        smearing=smearing,
    )
