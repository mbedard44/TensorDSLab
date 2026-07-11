# TensorDSLab

TensorDSLab is a clean-slate, tensor-native detector data-lab package.

Its intended ecosystem data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the common tensor substrate rather than a pipeline stage.
TensorDSLab consumes a future accepted TensorG4DS public product; it does not
parse native G4DS files or own low-level simulation analysis such as deposit
clustering. The exact upstream product and adapter are not implemented yet.
The production integration target keeps tensor payloads on one explicit GPU
across TensorG4DS, TensorDSLab, and TensorML, without an implicit CPU,
NumPy, serialization, or reload boundary. New semantic transforms may still
create new tensors on that same device; same-device handoff does not imply
shared layouts or zero-copy computation.

The package should define post-TensorG4DS detector/readout, reconstruction,
and semantic tensor-product contracts while using TensorCore as the generic
tensor identity, layout, field, collection, selection, batching, movement,
validation, and pure operation backbone.

The first MVP direction starts at already-binned photon-origin primary
photoelectrons and focuses on the post-binned readout path: aggregate SiPM
charge simulation, waveform fields, analog waveform composition, and optional
digitization. Native G4DS parsing stays upstream; the typed TensorG4DS handoff,
photoelectron binning, durable cache, and external integration boundaries come
later.

The accepted primary tensor handoff is one
`ReadoutCollection(TensorCollection)`, not one class per charge or waveform
field. Each structurally immutable partial snapshot contains any nonempty
subset of these fields in canonical filtered order:

```text
readout.photoelectrons
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.analog
readout.waveform.digitized
```

Fields share one exact layout, device, and `torch.strided` tensor layout;
noncontiguous strided tensors remain valid collection structure.
That is the semantic contract, not the warmed memory profile. Functional use
may explicitly allocate to normalize arbitrary order/strides. The strict
`out + workspace` profile instead requires contiguous participating tensors
with the sample axis last; it rejects rather than permuting, making contiguous,
casting, moving, or allocating fallback storage inside the warmed call.
`tensor_dslab.common` owns and exports the shared stable `ExampleId` and
`ChannelId` coordinate types. `tensor_dslab.readout` owns and exports
readout-specific axis and field constants, including the three required
semantic axis identities:

```python
READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")
READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")
READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")
```

They may occur in any layout order and are located by axis-ID equality/index,
not fixed dimensions. Example is ID-backed by exact `ExampleId` coordinates,
channel by exact `ChannelId` coordinates, and sample is count-only.
Every extra axis is shared too. A typed `SampleGrid` retains regular sample
facts; there is no caller-defined axis-role sidecar. Photoelectrons use
`torch.int64`, present floating roles use one common `torch.float32` or
`torch.float64` dtype, and digitized ADC codes use `torch.int32`. A conditional
`DigitizedWaveformSpec` keeps transfer facts for bit depths 1 through 16,
inclusive `[0, 40]` dB gain, and truncation policy with the digitized field.
Photoelectrons are binned primary PE seeds; only timing jitter replaces them.
One public `simulate_charge` transform keeps intermediate avalanche counts
private while producing the
floating aggregate PE-equivalent `readout.charge` response per readout
channel/sample, not SI coulombs or an individual-SPAD output. Pure and noise
are components at one shared analog reference plane, and their composition
produces the analog waveform that is subsequently digitized.
Transforms add or replace one target field in a new snapshot, invalidate all
materialized descendants reachable from that target, and structurally share
retained fields without mutating their tensor payloads. A future
`ReadoutExample` may be a thin provenance/context wrapper; it is not the primary
tensor handoff.

Callers treat materialized fields as read-only value state. Manual in-place
tensor edits bypass dependency invalidation and are outside the public
contract. Execution has three layers: independently callable atomic free
transforms, caller-owned scratch-only `ReadoutWorkspace`, and the local
fixed-chain `build_readout_collection(...)` domain builder. Atomic transforms
retain the functional/autograd `out=None` path and exact-target `out=` path.
Public `out` is caller-owned and valid; its preparation factory zero-initializes
new payloads, while reuse may contain the prior valid result because execution
fully overwrites it. Workspace scratch is separate and may be uninitialized
only under write-before-read. A workspace is valid only with `out`; supplying
`out` selects the non-autograd simulation path.

The builder's full result contains photoelectrons, charge, pure, noise, analog,
and optional digitized fields; disabled timing may structurally share source
photoelectrons. With neither `out` nor workspace, it uses the functional
allocating path.

The warmed steady-state hot path uses a caller-prepared exact `out` plus an
exact-match workspace and performs no TensorDSLab-managed tensor-storage
allocation. Its workspace signature stores ordered layout axis IDs and sizes;
sample-last position and contiguous strides follow from that ordered shape,
while required-axis positions are derived by equality/index rather than
retained in a role sidecar. Different leading-axis orders use different
workspaces. Returned collections never reference workspace scratch. Reusing
an `out` authorizes its complete overwrite, so callers use output banks when
prior results remain in flight. This allocation-free claim does not cover
Python, PyTorch-internal, or vendor-library allocations. The builder performs
local readout composition only; it does not own source IO or Projects/dag
orchestration.

Later stochastic transforms use layout-order-independent logical coordinates:
required example-axis ID/coordinate, required channel-axis ID/coordinate,
other ID-backed shared axes ordered lexically by `axis_id.value` and paired
with their coordinates, then
`SampleGrid.sample_offset + local_sample_index`. Extra count-only axes remain
structurally valid, but stochastic use rejects one that has no accepted stable
global-offset rule.

## Current Status

The Stage 1 post-binned readout architecture is Design-complete. Stage 2 was
dispatched on 2026-07-11 from exact clean package baseline
`d097cb3cdde185c6814116e886e7844ea3f55178` through persistent package-owned
Implementation, Validation, and Review roles. Before the Review-owned clean
fast-forward, the accepted `main` baseline remains documentation-only. If this
record is read on `main`, that merge gate has completed and the Stage 2 package,
metadata, exact TensorCore dependency, and test suite are accepted there. Stage
2 accepts no cache schema.

The Stage 2 feature-branch line contains the scoped production candidate:
the installable package foundation, typed readout collection and sidecars,
TensorCore-delegating reconstruction helpers, field-scoped and full-output
buffer preparation, and focused construction tests. Fixed-commit Validation
has cleared a candidate on that line; every later Review-fix commit returns
through Validation before Review recheck. A feature-branch checkout remains
candidate evidence and does not itself imply Review clearance or merge.

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`, bound to
exact candidate `d634401a853915edeb4f83df4a4943b3553deced`. Conformance remains
`Not evaluated`, Coordination remains `Deferred`, and Profile B remains
`Disabled`. This active-development, pre-deployment record makes no
deployability, backward-compatibility, or broad compatibility claim. See the
[Package Governance index](docs/governance/index.md).

Start with:

- [Agent Workflow](AGENTS.md)
- [Contributing](CONTRIBUTING.md)
- [Package Governance](docs/governance/index.md)
- [Overview](docs/overview.md)
- [Design](docs/design.md)
- [IV-DSLab Parity](docs/parity.md)
- [TensorCore Integration](docs/architecture/tensors.md)
- [Post-Binned Readout Architecture](docs/architecture/readout.md)
- [Implementation Stages](docs/implementation/index.md)
- [Stage 2 Work Order](docs/implementation/stage_2_package_and_readout_collection_foundation.md)

## Intended Package Shape

```text
TensorDSLab/                  # project/display folder
  tensor_dslab/              # Python import package
    common/
    detector/                # optional post-TensorG4DS semantics, when accepted
    readout/
    reconstruction/
    caches/
    executables/             # future integration surface
    operations/              # future integration surface
    recipes/                 # future integration surface
```

When production code is accepted, local smoke and test commands should run from
the project root with `PYTHONPATH=.`. Semantic packages remain directly below
`tensor_dslab`.
