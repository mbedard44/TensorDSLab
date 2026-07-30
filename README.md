# TensorDSLab

TensorDSLab is a clean-slate, tensor-native detector data-lab parts bin. Its
intended ecosystem data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared semantic tensor substrate. TensorDSLab owns reusable
detector products and scientific transformations; an application owns source
selection, workflow composition, retention, IO, and orchestration.

## Maintenance 18 Product Surface

The current package targets exact TensorCore `0.22.0` commit
`19bfae35fbc773b55cac7bcd659dda57c4dee6d6`. Four semantic axes compose
TensorCore Coordinates:

- `ExampleAxis` and `ChannelAxis` identify application roles;
- `TimeAxis` and `FrequencyAxis` add a positive binary64 coordinate scale and
  package-registry Pint Unit.

Seven direct tensor Products carry exact semantic quantity Specs:

- `Photoelectrons`
- `Charge`
- `PureWaveform`
- `NoiseWaveform`
- `AnalogWaveform`
- `DigitizedWaveform`
- `EncodedWaveform`

Generated Products expose Product-specific `create`, `prepare`, `produce`, and
`validate` classmethods. Preparation returns a fresh same-type Config
punchcard with immutable source-Spec provenance and aligned semantic Kernels.
Production accepts only sources whose Specs remain positionally structurally
equal to that provenance. It performs no Pint interpretation, coordinate
discovery, source movement, or dtype-policy selection.

`EncodedWaveform` is the deterministic terminal DAQ/readout Product. It keeps
exact DigitizedWaveform codes on configured raw-ZLE support and uses one
explicit negative Spec-owned suppression code elsewhere. It adds no record,
IO, reconstruction, or application-workflow abstraction.

There is intentionally no generic readout package, `ReadoutConfig`,
`ReadoutCollection`, `simulate_readout`, or embedded detector profile.
Applications compose the direct Products they need.

## Readout Quickstart

The [application-neutral readout notebook](demos/readout.ipynb) builds two
independent examples over three sensors by hand from the public Product APIs,
including EncodedWaveform as the seventh separate transformation. Each
Product has its own `3 x 2` grid, and the seven illustrative figures are
committed so readers can inspect the complete story before running a kernel.
Rerunning the notebook on the recorded eager-CPU stack deterministically
refreshes the same demonstration. The snapshot is not a package-owned
workflow, durable artifact, detector profile, calibration, or cross-platform
PNG promise.

```python
import torch
from tensor_core import CountCoordinates, NonnegativeInteger, Threefry4x32
from tensor_dslab import (
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    ExampleAxis,
    Photoelectrons,
    PhotoelectronsSpec,
    unit_registry,
)

axis = ExampleAxis(coordinates=CountCoordinates(count=4))
source_spec = PhotoelectronsSpec(
    axes=(axis,),
    device=torch.device("cpu"),
    dtype=torch.int64,
    unit=unit_registry.Unit("avalanche"),
)
source = Photoelectrons(
    tensor=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    spec=source_spec,
)
config = ChargeConfig(
    spec=ChargeSpec(
        axes=(axis,),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unit=unit_registry.Unit("avalanche"),
    ),
    kernels=ChargeKernels(members=()),
    correlated_avalanche_generations=NonnegativeInteger(value=0),
)
charge = Charge.create(
    sources=(source,),
    config=config,
    rng=Threefry4x32(seed=0),
)
```

TensorDSLab remains pre-deployment. The current package evidence is local
CPU-only and makes no accelerator, compatibility, release, calibration, or
production-readiness claim.
