from __future__ import annotations

import torch
from tensor_core import CounterRng

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
    _prepare_dark_mean,
    _simulate_dark_counts,
)
from tensor_dslab.readout.charge.effects._smearing import (
    _prepare_smearing_sigma,
    _simulate_charge_smearing,
)
from tensor_dslab.readout.charge.effects._timing_jitter import (
    _TimingJitterPlan,
    _prepare_timing_jitter,
    _simulate_timing_jitter,
)
from tensor_dslab.readout.charge.field import Charge, _require_valid_values
from tensor_dslab.readout.photoelectrons import Photoelectrons


def _produce_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> Charge:
    if type(photoelectrons) is not Photoelectrons:
        raise TypeError("photoelectrons must be exactly Photoelectrons")
    if type(config) is not ChargeConfig:
        raise TypeError("config must be exactly ChargeConfig")
    if not isinstance(rng, CounterRng):
        raise TypeError("rng must be a CounterRng")
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

    dark_mean = 0.0
    if config.dark_count is not None:
        dark_mean = _prepare_dark_mean(config.dark_count, sampling=sampling)

    jitter_plan: _TimingJitterPlan | None = None
    if config.timing_jitter is not None and config.timing_jitter.sigma_ns.value != 0.0:
        jitter_plan = _prepare_timing_jitter(
            config.timing_jitter,
            sampling=sampling,
            tensor_numel=tensor_numel,
        )

    if config.correlated_avalanches is None:
        _, ledger_bound = _ledger_envelope(
            floating_dtype=floating_dtype,
            maximum_generations=0,
            retained_mechanisms=0,
            recovered_afterpulse=False,
            sample_count=sampling.sample_count.value,
        )
        correlated_plan: _CorrelatedAvalanchePlan | None = None
    else:
        correlated_plan = _prepare_correlated_plan(
            config.correlated_avalanches,
            sampling=sampling,
            floating_dtype=floating_dtype,
            tensor_numel=tensor_numel,
        )
        ledger_bound = correlated_plan.ledger_bound

    smearing_sigma = 0.0
    if config.smearing is not None and config.smearing.relative_sigma.value != 0.0:
        smearing_sigma = _prepare_smearing_sigma(
            config.smearing,
            floating_dtype=floating_dtype,
            ledger_bound=ledger_bound,
            device=device,
        )

    charge = source
    charge_square_sum: torch.Tensor | None = None
    if config.dark_count is not None and dark_mean != 0.0:
        charge = _simulate_dark_counts(
            charge,
            sampling=sampling,
            config=config.dark_count,
            rng=rng,
        )
    if config.timing_jitter is not None and jitter_plan is not None:
        charge = _simulate_timing_jitter(
            charge,
            sample_dimension=sample_dimension,
            sampling=sampling,
            config=config.timing_jitter,
            rng=rng,
        )
    if config.correlated_avalanches is not None:
        correlated = _simulate_correlated_avalanches(
            charge,
            sample_dimension=sample_dimension,
            sampling=sampling,
            floating_dtype=floating_dtype,
            config=config.correlated_avalanches,
            rng=rng,
        )
        charge = correlated.S1
        charge_square_sum = correlated.S2
    if config.smearing is not None and smearing_sigma != 0.0:
        charge = charge.to(dtype=floating_dtype)
        charge = _simulate_charge_smearing(
            charge,
            charge if charge_square_sum is None else charge_square_sum,
            config=config.smearing,
            rng=rng,
        )

    values = charge.to(dtype=floating_dtype)
    if not bool(torch.all(torch.isfinite(values) & (values >= 0.0)).item()):
        raise RuntimeError("Charge production produced an invalid terminal value")
    if values.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
        raise RuntimeError("Charge output must have fresh storage")
    result = Charge(tensor=values, axes=photoelectrons.axes)
    _require_valid_values(result)
    return result
