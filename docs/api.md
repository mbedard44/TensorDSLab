# Public API

TensorDSLab's supported public surface is deliberately small. Ordinary product
types, Config types, semantic axes, physical-quantity helpers, and
`simulate_readout(...)` are exported from `tensor_dslab`. The
`tensor_dslab.readout` facade exposes the corresponding readout-owned surface.
Private product Runtime actions, validators, producers, requirements, and RNG
role keys are implementation details.

## Provisional DS20k Veto Profile

Maintenance 9 is **Merged / Closed** through exact Review-cleared target
`2a04942229ab06d2cfc17ab7a5fd09afaf4e3c58` and adds one precise-module public
factory:

```python
from tensor_dslab.readout.profiles import ds20k_veto

config = ds20k_veto()
```

`ds20k_veto()` takes no arguments and returns a fresh `ReadoutConfig` tree with
fresh copied Pint quantities on every call. The factory performs no tensor
allocation, preparation, simulation, RNG activity, file or network access, or
environment/device inspection. It is exported only from
`tensor_dslab.readout.profiles`; it is not re-exported from `tensor_dslab` or
`tensor_dslab.readout`.

The returned Config is a **provisional demonstration profile, not an approved
run calibration**. Its Veto pulse magnitudes retain audited donor fixtures for
the existing numerical-parity comparison boundary. Its dark-count, PSD,
analog, and digitization settings are illustrative choices. Calling the
factory does not select source axes, channel identities, a sample grid, input
data, products, dtype, device, RNG seed, or retention policy.

Use the profile explicitly with the ordinary public simulation API:

```python
import torch
from tensor_core import Threefry4x32
from tensor_dslab import (
    AnalogWaveform,
    DigitizedWaveform,
    Photoelectrons,
    simulate_readout,
)
from tensor_dslab.readout.profiles import ds20k_veto

def run_readout(photoelectrons: Photoelectrons):
    """Run selected products from an already-binned source field."""

    return simulate_readout(
        photoelectrons,
        products=(AnalogWaveform, DigitizedWaveform),
        config=ds20k_veto(),
        rng=Threefry4x32(seed=17),
        floating_dtype=torch.float32,
    )
```

The executable [readout demonstration](../demos/readout.py) and
[notebook](../demos/readout.ipynb) construct their `Photoelectrons` input and
sampling axes separately, compare manual and profile Config construction, and
retain only the requested products. Their source pattern, labels, grid, seed,
plot selection, and requested products are demonstration choices, not hidden
profile state.

The complete value-by-value comparison boundary and classification are
recorded in [IV-DSLab Parity](parity.md#maintenance-9-provisional-ds20k-veto-profile).
Promoting or replacing any provisional value as a production calibration is a
scientific/API decision owned by Design; it is not an ordinary maintenance
edit.
