"""Private tensor execution for charge products."""

import torch
from tensor_core import CounterRng

from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.charge.runtime.effects.correlated_avalanches import (
    simulate_correlated_avalanches,
)
from tensor_dslab.readout.charge.runtime.effects.dark_counts import (
    simulate_dark_counts,
)
from tensor_dslab.readout.charge.runtime.effects.smearing import (
    simulate_charge_smearing,
)
from tensor_dslab.readout.charge.runtime.effects.timing_jitter import (
    simulate_timing_jitter,
)
from tensor_dslab.readout.charge.runtime.prepare import ChargeRuntime
from tensor_dslab.readout.photoelectrons.field import Photoelectrons


def produce_charge(
    photoelectrons: Photoelectrons,
    *,
    runtime: ChargeRuntime,
    rng: CounterRng,
) -> Charge:
    source = photoelectrons.tensor
    charge = source
    charge_square_sum: torch.Tensor | None = None
    if runtime.dark is not None and runtime.dark.mean != 0.0:
        charge = simulate_dark_counts(
            charge,
            runtime=runtime.dark,
            rng=rng,
        )
    if runtime.timing_jitter is not None:
        charge = simulate_timing_jitter(
            charge,
            sample_dimension=runtime.sampling.sample_dimension,
            runtime=runtime.timing_jitter,
            rng=rng,
        )
    if runtime.correlated_avalanches is not None:
        correlated = simulate_correlated_avalanches(
            charge,
            sample_dimension=runtime.sampling.sample_dimension,
            floating_dtype=runtime.floating_dtype,
            runtime=runtime.correlated_avalanches,
            rng=rng,
        )
        charge = correlated.S1
        charge_square_sum = correlated.S2
    if runtime.smearing is not None and runtime.smearing.represented_sigma != 0.0:
        charge = charge.to(dtype=runtime.floating_dtype)
        charge = simulate_charge_smearing(
            charge,
            charge if charge_square_sum is None else charge_square_sum,
            runtime=runtime.smearing,
            rng=rng,
        )

    values = charge.to(dtype=runtime.floating_dtype)
    return Charge(tensor=values, axes=photoelectrons.axes)
