# Overview

This is the quick architecture map for TensorDSLab design, implementation,
validation, and review threads.

## Project Identity

TensorDSLab is a clean-slate, tensor-native detector data-lab package. It
should consume accepted TensorG4DS tensor-native products and produce typed
readout and future reconstruction products, with durable cache contracts
deferred until a stage accepts durable outputs.

The intended chain is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is the intended data flow, not an import graph or a claim that its
deferred adapters already exist. TensorCore is the shared substrate. Native
G4DS parsing and low-level analysis such as deposit clustering stay upstream
of TensorDSLab. A future TensorDSLab-owned bridge may consume an exact public
TensorG4DS type once that contract exists.

TensorCore provides the generic tensor substrate. TensorDSLab owns its
post-TensorG4DS semantic mapping, readout collections and fields, product
labels, cache semantics, validators, and domain-specific transforms.
Downstream ML packages own source adaptation for model training, batching
policy, models, objectives,
metrics, checkpoints, training, and evaluation, but TensorDSLab should not
design around those downstream integration contracts until its local
collection model is stable.

## Current Local Focus

The Stage 1 post-binned readout architecture is Design-complete, and the
repository remains in a documentation-only phase owned by the Design thread:

```text
AGENTS.md / CONTRIBUTING.md
  -> overview, design, decisions, parity, validation
  -> architecture/tensors.md and architecture/readout.md
  -> implementation-stage work orders
  -> future package code
```

No production package, tests, cache schema, DAG surface, or copied donor code
has been accepted yet. The Stage 2 package-and-collection work order is
Design-complete but has not been dispatched. The synchronized Design
documentation is committed, while Stage 2's exact production-dispatch base and
role handoff remain unset.

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`, bound to
exact candidate `d634401a853915edeb4f83df4a4943b3553deced`. Conformance remains
`Not evaluated`, Coordination remains `Deferred`, and Profile B remains
`Disabled`. TensorDSLab is in active development and pre-deployment; adoption
makes no deployability, backward-compatibility, or broad compatibility claim.
The [Package Governance index](governance/index.md) records the decision,
adopted scope, and separate state boundaries.

The first accepted MVP direction is the post-binned tensor-native readout path:
start from already-binned photon-origin primary photoelectrons, then simulate
the aggregate SiPM charge response, waveform products, analog waveform
composition, and optional digitization around TensorCore records. Native G4DS
parsing remains upstream. The typed TensorG4DS handoff, detector-window
construction, photoelectron binning, IO, cache, DAG, and TensorML adapter
boundaries come later.

TensorCore is the dense tensor spine. The first readout surface is one semantic
`ReadoutCollection(TensorCollection)` subclass with a typed `SampleGrid`, a
conditional `DigitizedWaveformSpec`, and free transform functions. Any nonempty
recognized field subset is a valid structurally immutable, partially
materialized snapshot. TensorDSLab defers concrete rank and axis order to
runtime layouts while keeping field roles, exact required axis identities,
sample-grid facts, and stochastic coordinates explicit. Caller-defined
semantic axis-role sidecars, per-product collection subclasses, and a generic
`Product` base are not accepted.

`tensor_dslab.common` owns and exports the shared stable `ExampleId` and
`ChannelId` coordinate types. `tensor_dslab.readout` owns and exports
readout-specific axis and field constants, including exactly these required
axis identities:

```python
READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")
READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")
READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")
```

The axes may appear in any layout order and are located by axis-ID
equality/index. Example is ID-backed by exact `ExampleId` coordinates, channel
by exact `ChannelId` coordinates, and sample is count-only. Every extra
layout axis is shared as well; `SampleGrid` retains regular sample period,
origin, and containing-grid offset.

The local post-binned product flow is:

```text
readout.photoelectrons
  -> timing jitter
  -> simulate_charge
  -> readout.charge
  -> readout.waveform.pure

collection layout/sample grid -> readout.waveform.noise
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog
  -> readout.waveform.digitized
```

Photoelectrons are binned primary PE seeds, and only timing jitter replaces
that recognized field. `simulate_charge` performs dark-count, crosstalk,
afterpulse, and smearing behavior internally; intermediate avalanche-count
values are private runtime scratch whose storage may be workspace-reused. The
charge result is a floating aggregate PE-equivalent response per readout
channel/sample, not SI coulombs or an individual-SPAD output. Pure and noise
waveforms are signal-only and noise-only components at one shared analog
reference plane rather than sequential hardware products.

Fields use that canonical topological order filtered to those present. Every
present field has the same ordered layout, common device, and `torch.strided`
tensor layout; noncontiguous strided tensors remain valid structure.
That semantic flexibility is separate from warmed execution: the
`out + workspace` readout profile requires contiguous participating tensors
with sample last and rejects hidden normalization or fallback allocation.
Photoelectrons use `torch.int64`; charge, pure, noise, and analog roles share
one common `torch.float32` or `torch.float64` dtype; and ADC codes use
`torch.int32`. Transforms return new snapshots, share retained field records,
and centrally remove transitive descendants when an upstream field is added or
replaced.

Later stochastic transforms derive tensor-order-independent identity from the
required example-axis ID/coordinate, required channel-axis ID/coordinate,
other ID-backed shared axes ordered lexically by `axis_id.value` and paired
with their coordinates, and finally
`SampleGrid.sample_offset + local_sample_index`. Extra count-only axes are
valid collection structure but are rejected by stochastic transforms without
an accepted stable global-offset rule.

Execution has three layers:

```text
atomic free transforms
  -> optional caller-owned ReadoutWorkspace scratch
  -> build_readout_collection(...) fixed-chain domain composition
```

Atomic transforms retain functional `out=None` and exact-target `out=` modes.
Public caller-owned `out` remains valid and never aliases workspace scratch;
its factory zero-initializes new payloads, while reuse may contain the prior
valid result because targets are fully overwritten. `ReadoutWorkspace` is
runtime-only scratch, exact to one ordered sequence of layout axis IDs and
sizes, device, role-dtype set, algorithm scratch signature, and CPU/default or
CUDA stream. Required-axis positions are derived by equality/index; no
semantic-role sidecar enters the signature. It is never a product, field,
sidecar, config, ID, cache record, or returned-storage owner. One nonreentrant
same-stream lease may be active at a time, and private uninitialized scratch is
allowed only with write-before-read.

`build_readout_collection(...)` owns the configured full local chain and its
scratch schedule while leaving every low-level transform available. Its full
result contains photoelectrons, charge, pure, noise, analog, and optional
digitized fields; disabled timing may structurally share source
photoelectrons. With neither `out` nor workspace it is a functional allocating
path. A workspace without `out` is invalid; supplying `out` selects a
non-autograd simulation path, and the warmed allocation-free hot path requires
exact caller-prepared `out` plus workspace, sample-last layout, contiguous
participating storage, and an exact destination/device/dtype/algorithm/stream/
lease signature. Different leading-axis orders use different workspaces.
Returned collections never alias workspace scratch. Reusing an output
authorizes overwrite, so overlapping consumers require caller-managed output
banks. Allocation-free here means no warmed steady-state
TensorDSLab-managed tensor-storage allocation, not zero Python, PyTorch, or
vendor-library allocation.

## Intended Product Flow

TensorDSLab should preserve this data-flow and ownership rule unless a future
Design decision changes it:

```text
G4DS native products
  -> TensorG4DS typed tensor-native products
  -> deferred TensorG4DS-to-TensorDSLab bridge
       -> explicit provenance and coordinate mapping
       -> detector-window/readout-grid construction
       -> photon-origin PE binning
  -> ReadoutCollection{"readout.photoelectrons"}
       -> optional ReadoutExample provenance/context wrapper
       -> TensorDSLab readout and future reconstruction tensor/product views
  -> deferred explicit TensorML field-selection/model boundary
```

This is a product-flow and ownership rule, not a Python import graph or a
campaign scheduling policy. Local fixed-chain
composition in `build_readout_collection(...)` belongs to TensorDSLab;
Projects/dag orchestration, durable cache compatibility, compaction, and
downstream handoff are later concerns.

`ReadoutExample` is not a second tensor handoff and does not own duplicate
field payloads.

The production cross-package target retains payload tensors on one explicit
GPU and forbids implicit host staging, NumPy conversion, or serialization at
the TensorG4DS/TensorDSLab/TensorML boundaries. The future upstream bridge
constructs new TensorDSLab semantics on that device; it does not cast a
TensorG4DS collection into a `ReadoutCollection` or equate TensorG4DS
`EventId` provenance with TensorDSLab `ExampleId`.

## Package Shape

Use the ecosystem naming convention:

```text
TensorDSLab/
  tensor_dslab/
    common/
    detector/          # optional post-TensorG4DS semantics, when accepted
    readout/
    reconstruction/
    caches/
    executables/       # future integration surface
    operations/        # future integration surface
    recipes/           # future integration surface
```

The outer project/display folder is `TensorDSLab`; the Python import package
should be `tensor_dslab` when package code is accepted. Do not create a flat
TitleCase import package; keep semantic subpackages directly below the import
root.

## Documentation Map

- [Design](design.md): architecture thesis, ownership boundaries, and
  non-goals.
- [Decisions](decisions.md): accepted and open design decisions.
- [IV-DSLab Parity](parity.md): comparison classes, donor evidence,
  assumptions, fixture provenance, and intentional divergences.
- [Validation](validation.md): validation philosophy and early expectations.
- [Package Governance](governance/index.md): adoption decision and declaration,
  TensorDSLab overlay, complete semantic rule map, state boundaries, and
  closeout record.
- [TensorCore Integration](architecture/tensors.md): TensorCore consumer,
  semantic collection, field ordering, layout, output/workspace, placement,
  lifetime, reconstruction, and coordination contracts.
- [Post-Binned Readout](architecture/readout.md): readout fields, transform
  order, scientific behavior, RNG, workspace, builder, allocation, lifetime,
  device, and validation contracts.
- [Implementation Stages](implementation/index.md): staged work orders and
  scope limits.
- [Stage 2 Work Order](implementation/stage_2_package_and_readout_collection_foundation.md):
  package and `ReadoutCollection` foundation contract; not dispatched.

Additional architecture docs should be added only when they carry real
contracts, such as detector products, reconstruction products, or caches.

Implementation threads should treat these docs, `CONTRIBUTING.md`, and
`AGENTS.md` as the source of truth. If implementation uncovers a contradiction,
route it back to Design before changing architecture.
