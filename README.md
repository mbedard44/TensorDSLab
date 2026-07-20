# TensorDSLab

TensorDSLab is a clean-slate, tensor-native detector data-lab package. Its
intended ecosystem data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared semantic tensor substrate rather than another
pipeline stage. TensorDSLab owns downstream readout and future reconstruction
meaning; it does not parse native G4DS files or own TensorG4DS deposit
clustering, TensorML training, durable IO, or campaign orchestration.

## Implemented Foundation

The package currently provides the semantic foundation introduced against
TensorCore `0.7` and now installed against exact TensorCore `0.9.0` for
post-binned readout:

- `ExampleAxis`, `ChannelAxis`, and regular timestamp-backed `SampleAxis`;
- `SamplingConfig` for zero-start, integer-picosecond left-edge samples;
- `Photoelectrons`, `Charge`, `PureWaveform`, `NoiseWaveform`,
  `AnalogWaveform`, and `DigitizedWaveform` fields;
- product-owned immutable scientific configuration records; and
- `ReadoutCollection`, an immutable completed result containing any nonempty
  unordered subset of the six products.

The exact concrete axis, field, and collection classes carry in-process
semantic identity. There are no parallel axis IDs, field IDs, layout objects,
product registries, or collection sidecars.

```python
import torch
from tensor_core import PositiveInteger

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    Photoelectrons,
    ReadoutCollection,
    SamplingConfig,
)

sampling = SamplingConfig(
    sample_period_ps=PositiveInteger(2_000),
    sample_count=PositiveInteger(4),
)
axes = (
    ExampleAxis(coordinates=("event-0",)),
    ChannelAxis(coordinates=("tile-0",)),
    sampling.build_axis(),
)
photoelectrons = Photoelectrons(
    tensor=torch.zeros((1, 1, 4), dtype=torch.int64),
    axes=axes,
)
readout = ReadoutCollection(fields=(photoelectrons,))

assert readout.field(Photoelectrons) is photoelectrons
```

`Photoelectrons` is an already-produced dense, binned photon-origin truth
input. TensorDSLab does not yet construct it from TensorG4DS data.

Private implementation seams now produce complete `Charge`, `PureWaveform`,
`NoiseWaveform`, `AnalogWaveform`, and `DigitizedWaveform` values. Charge
production includes its configured dark-count, timing-jitter, fixed-generation
correlated-avalanche, recovery-ledger, and smearing submodels; noise supports
zero, white, and caller-supplied PSD models. These seams remain private.
Maintenance 2 is Merged / Closed through exact implementation candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. It moves generic RNG and
distribution mechanics to exact TensorCore `0.9.0`, uses config-owned
stochastic keys, and records eager-CPU evidence only because CUDA was
unavailable. Public orchestration is not implemented. Ordinary users should
continue to import only the documented package-root types and configs.

The historical [TensorCore consumer proposal](docs/implementation/proposed_tensorcore_counter_rng_and_distributions.md)
is now fulfilled by published TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The
[TensorDSLab Maintenance 2 work order](docs/implementation/maintenance_2_rng_and_product_module_ownership_migration.md)
selects that exact dependency. The closed implementation pins it and completes
the ownership migration. CUDA was unavailable, so the recorded evidence makes
no GPU or cross-backend claim.

## Explicit Exclusions

This foundation does not yet implement `simulate_readout(...)`, public atomic
product transforms, the accepted `CounterRng`-based simulation boundary, PE
binning, TensorG4DS or TensorML adapters, IO,
caches, `out=`, workspaces, movement/selection helpers, or an allocation-free
execution path. It makes no GPU-execution, release, deployment,
backward-compatibility, conformance, or broad cross-package compatibility
claim. The focused
[Stage 7 work order](docs/implementation/stage_7_public_readout_orchestration.md)
is Design-complete / Undispatched.

Start with [the documentation overview](docs/overview.md) and the
[rebuild architecture](docs/architecture/rebuild.md). Local tests run from the
project root with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
