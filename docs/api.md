# Public API

TensorDSLab's supported public surface is deliberately small. Ordinary product
types, Config types, semantic axes, physical-quantity helpers, and
`simulate_readout(...)` are exported from `tensor_dslab`. The
`tensor_dslab.readout` facade exposes the corresponding readout-owned surface.
Private product Runtime actions, validators, producers, requirements, and RNG
role keys are implementation details.

## Literal Physical Kernel Candidate

Maintenance 12 is the active fixed-commit API rebaseline to exact TensorCore
`0.21.0`. It adds public `QuantityKernel`, `DarkCountRate`, `TimingJitter`,
`DirectCrosstalk`, `DelayedCrosstalk`, `Afterpulse`, `SmearingWidth`, and
`Pulse` leaves. `ChargeConfig` accepts those literal physical kernels directly;
`PureWaveformConfig` accepts one `Pulse`. Kernel quantities are canonical,
copied CPU `float64` snapshots, while prepared Runtime facts are plain Torch
tensors.

The provisional profile now receives its available geometry explicitly:

```python
config = ds20k_veto(
    sample_axis=sample_axis,
    channel_axis=channel_axis,
    example_axis=example_axis,
)
```

Omitted optional axes remain valid only when no profile kernel is conditioned
on that role. The profile is still illustrative, not an approved run
calibration. These bytes are candidate documentation while absent from
`main`; if present unchanged on `main`, Review's fast-forward has completed,
but final Design acceptance remains pending until the Maintenance 12 work
order and index say **Merged / Closed**.

## Addressed Random Execution

Maintenance 11 is **Merged / Closed** through exact Review-cleared and
fast-forwarded Candidate 2 `a527042701ac56f368f26248381244fdfcfb7fd3`,
tree `5c76122b25d17b9fe0b796618613d7bff0b102c1`, against exact published
TensorCore `0.19.0` containing commit
`ed17f4b637258f0a7f4544f235648b747f17fa44`. The public TensorDSLab API is
unchanged: callers still provide a `CounterRng` to `simulate_readout(...)`,
and its seed remains the realization control. At that closed historical
baseline, stochastic products used public TensorCore `RngElements`,
`RngAddress`, Distribution, and ProbabilityKernel objects. Maintenance 12
removes `ProbabilityKernel` and uses direct addressed distribution laws over
literal `TensorKernel` geometry. Role keys and address construction remain
private package policy, not caller customization surfaces.

The CPU-only [addressed-randomness notebook](../demos/random.ipynb) is an
educational inspection of the public TensorCore address model and the
TensorDSLab delayed-crosstalk mapping. Its private role-key import is clearly
marked unsupported; applications must not depend on that module or duplicate
its numeric values. The notebook does not add a TensorDSLab RNG facade,
calibration, CUDA, performance, release, or deployment contract.

## Provisional DS20k Veto Profile

Maintenance 9 is **Merged / Closed** through exact Review-cleared target
`2a04942229ab06d2cfc17ab7a5fd09afaf4e3c58` and adds one precise-module public
factory:

```python
from tensor_dslab.readout.profiles import ds20k_veto

config = ds20k_veto(sample_axis=sample_axis)
```

`ds20k_veto(...)` requires the available `SampleAxis` and optionally accepts
the available `ChannelAxis` and `ExampleAxis`. It returns a fresh
`ReadoutConfig` tree with fresh copied Pint quantities and physical kernels on
every call. The factory performs profile-construction tensor allocation but no
simulation, RNG activity, file or network access, or device inspection. It is
exported only from
`tensor_dslab.readout.profiles`; it is not re-exported from `tensor_dslab` or
`tensor_dslab.readout`.

The returned Config is a **provisional demonstration profile, not an approved
run calibration**. Its Veto pulse magnitudes retain audited donor fixtures for
the existing numerical-parity comparison boundary. Its dark-count, PSD,
analog, and digitization settings are illustrative choices. The provisional
digitizer uses `16` bits, an input interval of `[-3900, 100] mV`, and
`3.5218 dB` analog gain; those IV-DSLab-like values are not an approved
hardware or run calibration. Calling the factory does not select source axes,
channel identities, a sample grid, input data, products, dtype, device, RNG
seed, or retention policy.

Use the profile explicitly with the ordinary public simulation API:

```python
import torch
from tensor_core import Threefry4x32
from tensor_dslab import (
    AnalogWaveform,
    DigitizedWaveform,
    Photoelectrons,
    SampleAxis,
    simulate_readout,
)
from tensor_dslab.readout.profiles import ds20k_veto

def run_readout(photoelectrons: Photoelectrons):
    """Run selected products from an already-binned source field."""

    sample_axis = next(
        axis for axis in photoelectrons.axes if type(axis) is SampleAxis
    )
    return simulate_readout(
        photoelectrons,
        products=(AnalogWaveform, DigitizedWaveform),
        config=ds20k_veto(sample_axis=sample_axis),
        rng=Threefry4x32(seed=17),
        floating_dtype=torch.float32,
    )
```

The executable [readout demonstration](../demos/readout.py) and
[notebook](../demos/readout.ipynb) construct their `Photoelectrons` input and
sampling axes separately, compare manual and profile Config construction, and
retain only the requested products. They use a `2 ns`, `5000`-sample CPU grid
with explicit `1`, `2`, `3`, and `4` PE source deposits at `200`, `2600`,
`5000`, and `7400 ns`. The seeded readout may add separate dark-count Charge
events. The source pattern, labels, grid, seed, plot selection, and requested
products are demonstration choices, not hidden profile state.

Create the ordinary project/demo environment from the repository root before
running either form:

```bash
./create_environment.sh
conda activate tensor_dslab
python demos/readout.py
```

The complete value-by-value comparison boundary and classification are
recorded in [IV-DSLab Parity](parity.md#maintenance-9-provisional-ds20k-veto-profile).
Promoting or replacing any provisional value as a production calibration is a
scientific/API decision owned by Design; it is not an ordinary maintenance
edit.
