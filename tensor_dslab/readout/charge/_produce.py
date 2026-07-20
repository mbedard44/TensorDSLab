from __future__ import annotations

from dataclasses import dataclass

import torch
from tensor_core import CounterRng, RngKey

from tensor_dslab.common import SampleAxis, SamplingConfig
from tensor_dslab.readout._requirements import _require_sampling
from tensor_dslab.readout.charge.config import ChargeConfig
from tensor_dslab.readout.charge.effects._correlated_avalanches import (
    _CorrelatedAvalanchePlan,
    _ledger_envelope,
    _prepare_correlated_plan,
    _simulate_correlated_avalanches,
)
from tensor_dslab.readout.charge.effects._counts import (
    _require_count_domain,
    _require_tensor_allocation,
)
from tensor_dslab.readout.charge.effects._dark_counts import (
    _DarkCountPlan,
    _prepare_dark_counts,
    _simulate_dark_counts,
)
from tensor_dslab.readout.charge.effects._smearing import (
    _ChargeSmearingPlan,
    _prepare_charge_smearing,
    _simulate_charge_smearing,
)
from tensor_dslab.readout.charge.effects._timing_jitter import (
    _TimingJitterPlan,
    _prepare_timing_jitter,
    _simulate_timing_jitter,
)
from tensor_dslab.readout.charge.field import Charge, _require_valid_values
from tensor_dslab.readout.photoelectrons import Photoelectrons


@dataclass(frozen=True, slots=True)
class _ChargePlan:
    sample_dimension: int
    floating_dtype: torch.dtype
    rng_roles: tuple[tuple[str, RngKey], ...]
    dark: _DarkCountPlan | None
    timing_jitter: _TimingJitterPlan | None
    correlated_avalanches: _CorrelatedAvalanchePlan | None
    smearing: _ChargeSmearingPlan | None


def _prepare_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    floating_dtype: torch.dtype,
) -> _ChargePlan:
    if type(photoelectrons) is not Photoelectrons:
        raise TypeError("photoelectrons must be exactly Photoelectrons")
    if type(config) is not ChargeConfig:
        raise TypeError("config must be exactly ChargeConfig")
    _require_sampling(photoelectrons, sampling)
    if floating_dtype not in (torch.float32, torch.float64):
        raise TypeError("floating_dtype must be torch.float32 or torch.float64")
    device = photoelectrons.tensor.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Charge production supports only CPU and CUDA")

    source = photoelectrons.tensor
    _require_count_domain(source, field="Photoelectrons source")
    shape = tuple(source.shape)
    tensor_numel = _require_tensor_allocation(
        shape,
        element_size=source.element_size(),
        field="Charge source",
    )
    _require_tensor_allocation(
        shape,
        element_size=torch.empty((), dtype=floating_dtype).element_size(),
        field="Charge output",
    )
    sample_dimension = photoelectrons.dimension_of(SampleAxis)

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
        else _prepare_dark_counts(config.dark_count, sampling=sampling)
    )
    timing_jitter = (
        None
        if config.timing_jitter is None
        or config.timing_jitter.sigma_ns.value == 0.0
        else _prepare_timing_jitter(
            config.timing_jitter,
            sampling=sampling,
            tensor_numel=tensor_numel,
        )
    )

    if config.correlated_avalanches is None:
        _, ledger_bound = _ledger_envelope(
            floating_dtype=floating_dtype,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=sampling.sample_count.value,
        )
        correlated_avalanches: _CorrelatedAvalanchePlan | None = None
    else:
        correlated_avalanches = _prepare_correlated_plan(
            config.correlated_avalanches,
            sampling=sampling,
            floating_dtype=floating_dtype,
            tensor_numel=tensor_numel,
        )
        ledger_bound = correlated_avalanches.ledger_bound

    smearing = (
        None
        if config.smearing is None
        else _prepare_charge_smearing(
            config.smearing,
            floating_dtype=floating_dtype,
            ledger_bound=ledger_bound,
            device=device,
        )
    )
    return _ChargePlan(
        sample_dimension=sample_dimension,
        floating_dtype=floating_dtype,
        rng_roles=tuple(rng_roles),
        dark=dark,
        timing_jitter=timing_jitter,
        correlated_avalanches=correlated_avalanches,
        smearing=smearing,
    )


def _produce_charge(
    photoelectrons: Photoelectrons,
    *,
    plan: _ChargePlan,
    rng: CounterRng,
) -> Charge:
    if type(photoelectrons) is not Photoelectrons:
        raise TypeError("photoelectrons must be exactly Photoelectrons")
    if type(plan) is not _ChargePlan:
        raise TypeError("plan must be exactly _ChargePlan")
    if not isinstance(rng, CounterRng):
        raise TypeError("rng must be a CounterRng")

    source = photoelectrons.tensor
    charge = source
    charge_square_sum: torch.Tensor | None = None
    if plan.dark is not None and plan.dark.mean != 0.0:
        charge = _simulate_dark_counts(
            charge,
            plan=plan.dark,
            rng=rng,
        )
    if plan.timing_jitter is not None:
        charge = _simulate_timing_jitter(
            charge,
            sample_dimension=plan.sample_dimension,
            plan=plan.timing_jitter,
            rng=rng,
        )
    if plan.correlated_avalanches is not None:
        correlated = _simulate_correlated_avalanches(
            charge,
            sample_dimension=plan.sample_dimension,
            floating_dtype=plan.floating_dtype,
            plan=plan.correlated_avalanches,
            rng=rng,
        )
        charge = correlated.S1
        charge_square_sum = correlated.S2
    if plan.smearing is not None and plan.smearing.represented_sigma != 0.0:
        charge = charge.to(dtype=plan.floating_dtype)
        charge = _simulate_charge_smearing(
            charge,
            charge if charge_square_sum is None else charge_square_sum,
            plan=plan.smearing,
            rng=rng,
        )

    values = charge.to(dtype=plan.floating_dtype)
    if not bool(torch.all(torch.isfinite(values) & (values >= 0.0)).item()):
        raise RuntimeError("Charge production produced an invalid terminal value")
    if values.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
        raise RuntimeError("Charge output must have fresh storage")
    result = Charge(tensor=values, axes=photoelectrons.axes)
    _require_valid_values(result)
    return result
