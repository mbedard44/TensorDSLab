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

## Readout Foundation

The package provides the post-binned readout foundation introduced against
TensorCore `0.7`, reorganized through Maintenance 4, and migrated by the closed
[Maintenance 5 migration](docs/implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md)
to exact published TensorCore `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b` and fixes the current target
surface:

- identity-free zero-based `ExampleAxis`, string-label `ChannelAxis`, and
  compact regular integer-picosecond `SampleAxis`;
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

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    Photoelectrons,
    ReadoutCollection,
    SampleAxis,
)

axes = (
    ExampleAxis(count=1),
    ChannelAxis(labels=("tile-0",)),
    SampleAxis(start=0, step=2_000, count=4),
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

Maintenance 5 removes `SamplingConfig` and derives private sampling execution
facts directly from the source `SampleAxis`. It is a clean pre-deployment API
replacement: there is no legacy constructor, alias, or dual axis model.
Example coordinates are local ordinals and do not claim durable event identity;
channel labels retain detector identity. The complete readout boundary—not
`SampleAxis` construction generally—requires example-local `start == 0`.

[Maintenance 6 Pint Physical Configuration Boundary](docs/implementation/maintenance_6_pint_physical_configuration_boundary.md)
is **Merged / Closed** through exact Review-cleared target
`0257fb477ee04556ebbe26351123ae610b5d7925`. Its implemented API lets
collaborators configure physical values with scalar Pint quantities while
preserving a unit-free execution core:

```text
Pint Quantity -> Config -> prepare_* -> plain Runtime -> produce_* -> validate_*
```

The target retains exact TensorCore `0.13.0`, uses its public
`Scalar.require(...)` normalization instead of duplicating numeric rules, and
keeps quantities out of tensors, RNG addressing, Runtime records, producers,
validators, and collections. Complete local Validation and independent Review
cleared the exact final bytes with `13` unavailable-CUDA skips and no
accelerator claim. Maintenance 7 now closes the separate TensorCore `0.15.0`
adoption; local `main` remains unpushed pending later authorized gates.

[Maintenance 7 TensorCore 0.15 Adoption](docs/implementation/maintenance_7_tensorcore_0_15_adoption.md)
is **Merged / Closed** through exact Review-cleared and fast-forwarded target
`205182f0c7a4359cece79211ad22b47b522c34e3`, tree
`4c9f0ed2700b5683debb6e658ff2ec832e3d6acf`; immutable production Candidate 1
is `68c2f62c2ce354dd6c92fde28b020c0ce71881d6`. It replaces
raw logical-position tensors with TensorCore `RngPositions`, retires matching
local generic validators in favor of `tensor_core.validation`, and gives the
unchanged readout RNG namespace and ten role keys one non-exported source.
Public Configs expose no key overrides. The maintenance also makes both pulse
Configs accept positive voltage-amplitude magnitudes while preparation applies
the fixed DS20k negative polarity once; calibrated rendered results remain
exact. Public field names, Pint ownership, fixed keys and streams, and
same-stack stochastic results remain unchanged. Complete local Review evidence
passed with `13` conditional CUDA skips, so no accelerator claim follows.

[Maintenance 8](docs/implementation/maintenance_8_python314_tensorcore_0_16_modernization.md)
is **Merged / Closed** through exact Review-cleared target
`e5cc70adddaed357298e3e3bc4d95df78d3a55b7`. Its exact dependency is the
published TensorCore `0.16.0` containing commit
`e05324699892a8bcea024375720bfae1ed9569cc` plus the Python `3.14.6`,
PyTorch `2.13.0`, domain-validation-import, typing-syntax, and intentional
docstring modernization. It preserves the complete TensorDSLab scientific,
Pint, RNG, product, and public API behavior. CUDA and push remain separately
authorized.

[Maintenance 9](docs/implementation/maintenance_9_ds20k_veto_profile_and_public_readout_demos.md)
is **Merged / Closed** through exact Review-cleared target
`2a04942229ab06d2cfc17ab7a5fd09afaf4e3c58`. It adds a provisional
`ds20k_veto()` Config profile and executable CPU script/notebook demonstrations
of manual versus profile construction and
`PureWaveform + NoiseWaveform = AnalogWaveform -> DigitizedWaveform`. The
profile is illustrative rather than calibrated, and the accepted local
evidence makes no accelerator or push claim.

[Maintenance 11](docs/implementation/maintenance_11_tensorcore_0_19_addressed_distributions.md)
is **Merged / Closed** through exact Review-cleared and fast-forwarded
Candidate 2 `a527042701ac56f368f26248381244fdfcfb7fd3`, tree
`5c76122b25d17b9fe0b796618613d7bff0b102c1`. It migrates to exact TensorCore
`0.19.0` containing commit
`ed17f4b637258f0a7f4544f235648b747f17fa44` and replaces the previous
position-plus-method RNG calls with public addressed `RngElements`,
`RngAddress`, Distribution, and ProbabilityKernel objects while retaining the
public TensorDSLab facades and downstream physical laws. The CPU-only
[addressed-randomness notebook](demos/random.ipynb) shows exact address
metadata, raw words, one delayed-crosstalk collapsed-rate Poisson draw,
repeatability, chunk invariance, and global Torch RNG isolation. Its private
role-key import is explicitly unsupported.

Complete source/archive evidence passed at `232/221/11`; the `11`
TensorDSLab and two TensorCore unavailable-CUDA skips authorize no integrated
CUDA, performance, compatibility, release, publication, deployment, or
production-readiness claim.

The historical [TensorCore consumer proposal](docs/implementation/proposed_tensorcore_counter_rng_and_distributions.md)
is now fulfilled by published TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The
[TensorDSLab Maintenance 2 work order](docs/implementation/maintenance_2_rng_and_product_module_ownership_migration.md)
selects that exact dependency. The closed implementation pins it and completes
the ownership migration. Its original evidence was CPU-only; later
Maintenance 3 and 4 exact-stack full-A100 runs established functional CUDA
evidence without creating a broad backend or performance claim.

## Explicit Exclusions

This package does not yet implement PE binning, TensorG4DS or TensorML
adapters, IO, caches, `PureWaveformRenderer`, public atomic product transforms,
`out=`, workspaces, movement/selection helpers, or an allocation-free execution
path. It makes no GPU-performance or broad-backend, release, deployment,
backward-compatibility, conformance, or broad cross-package compatibility
claim. Exact Maintenance 3 and 4 full-A100 functional evidence does not imply
GPU-performance or broad-backend qualification. The focused
[Stage 7 work order](docs/implementation/stage_7_public_readout_orchestration.md)
is Merged / Closed; its accepted evidence is eager CPU-only because CUDA was
unavailable.

Start with [the documentation overview](docs/overview.md) and the
[rebuild architecture](docs/architecture/rebuild.md). Local tests run from the
project root with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
