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
- product-owned immutable scientific configuration records;
- `ReadoutCollection`, an immutable completed result containing any nonempty
  unordered subset of the six products; and
- `simulate_readout(...)`, the one public request-aware readout operation.

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
unavailable. Stage 7 is Merged / Closed through exact Review-cleared candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177`; it implements complete
whole-request preparation, execute-once prerequisite planning, exact requested
retention, and the public `simulate_readout(...)` export. Ordinary users should
import the documented package-root API rather than private product producers.

[Maintenance 3 Environment-Qualified Stochastic Continuity](docs/implementation/maintenance_3_environment_qualified_stochastic_continuity.md)
is Merged / Closed through exact Review-cleared candidate
`dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0` and its Design closeout. It
qualifies completed stochastic literals by numerical stack and changes no
production, dependency, RNG, or scientific contract.

[Maintenance 4 Runtime Action Ownership](docs/implementation/maintenance_4_runtime_action_ownership.md)
is **Merged / Closed** through exact Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda`. It reorganizes each generated
product behind a non-exported `runtime/` package with explicit `prepare_*`,
`produce_*`, and `validate_*` actions and concrete prepared `*Runtime` records.
Whole-request
preparation remains before RNG or product execution, and
`simulate_readout(...)` continues to run `produce -> validate ->
descendant` before constructing the final collection. Runtime modules remain
ordinary importable Python implementation details, but no runtime name is a
facade export or carries a compatibility promise. Maintenance 4 changes no
public API, product meaning, scientific equation, stochastic address, result
law, dependency, or supported
device boundary.

The historical [TensorCore consumer proposal](docs/implementation/proposed_tensorcore_counter_rng_and_distributions.md)
is now fulfilled by published TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The
[TensorDSLab Maintenance 2 work order](docs/implementation/maintenance_2_rng_and_product_module_ownership_migration.md)
selects that exact dependency. The closed implementation pins it and completes
the ownership migration. CUDA was unavailable, so the recorded evidence makes
no GPU or cross-backend claim.

## Explicit Exclusions

This package does not yet implement PE binning, TensorG4DS or TensorML
adapters, IO, caches, `PureWaveformRenderer`, public atomic product transforms,
`out=`, workspaces, movement/selection helpers, or an allocation-free execution
path. It makes no GPU-execution, release, deployment,
backward-compatibility, conformance, or broad cross-package compatibility
claim. The focused
[Stage 7 work order](docs/implementation/stage_7_public_readout_orchestration.md)
is Merged / Closed; its accepted evidence is eager CPU-only because CUDA was
unavailable.

Start with [the documentation overview](docs/overview.md) and the
[rebuild architecture](docs/architecture/rebuild.md). Local tests run from the
project root with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
