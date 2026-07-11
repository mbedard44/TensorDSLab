# Design

## Core Thesis

TensorDSLab should define detector data-lab products around a tensor-native
backbone without turning generic tensor mechanics into local copies.

Shared TensorDSLab modules should use domain vocabulary for products and cache
boundaries:

- post-TensorG4DS provenance mapping and row identity;
- detector response, channelization, window, and binning records only when a
  focused integration stage accepts them;
- `ReadoutCollection`, its semantic fields, and optional readout context;
- future reconstruction examples and products;
- product labels;
- cache manifests and cache rows;
- TensorCore-backed semantic collections and fields.

Generic tensor vocabulary belongs to TensorCore. Model, objective, training,
evaluation, metric, and checkpoint vocabulary belongs downstream unless a
future Design decision accepts a TensorDSLab-specific boundary.

## Build Philosophy

Define the MVP early, but build the system from typed in-memory products
outward. The first production priority is not cache IO, DAG compatibility, or
downstream handoff; it is the post-binned TensorCore-backed readout field
graph:

```text
ReadoutCollection{"readout.photoelectrons"}
  -> fixed-grid readout field updates
  -> charge and waveform field snapshots
```

The typed TensorG4DS bridge, any post-TensorG4DS detector grouping record, an
optional thin `ReadoutExample`, and the full detector-to-readout integration
remain future domain surfaces. They should be added only after the local
post-binned collection is stable enough to compose and the exact upstream
TensorG4DS public contract exists. `ReadoutCollection` remains the direct
tensor handoff through that future integration.

IO boundaries should follow the product model, not lead it. Durable cache
schemas, table/array codecs, manifests, compaction, executable doors,
operation specs, recipe fragments, and downstream adapters are deferred until
the in-memory contracts are stable enough to deserve persistence or external
integration.

## Post-Binned Readout MVP

The first production MVP should focus on the post-binned tensor-native readout
path. TensorDSLab should start from already-binned photon-origin primary
photoelectrons. Native G4DS parsing and low-level deposit clustering stay
upstream in the G4DS/TensorG4DS side of the boundary. TensorDSLab defers the
typed TensorG4DS handoff, provenance mapping, event placement,
detector-window construction, photoelectron binning, durable IO, cache
compatibility, DAG integration, and TensorML adapters.

The initial readout chain should be:

```text
"readout.photoelectrons"
  -> timing jitter
  -> simulate_charge
       -> private dark-count avalanche grid
       -> private frozen-snapshot crosstalk and afterpulse contributions
       -> private aggregate charge smearing
  -> "readout.charge"
  -> "readout.waveform.pure"

layout/sample grid -> "readout.waveform.noise"
pure + noise -> "readout.waveform.analog"
analog -> optional "readout.waveform.digitized"
```

`readout.photoelectrons` represents binned photon-origin primary PE seeds. Only
timing jitter replaces it. One public `simulate_charge` transform consumes the
photoelectrons, performs dark counts, frozen-snapshot crosstalk and afterpulse
contributions, and charge smearing internally, and adds or replaces
`readout.charge`. Intermediate avalanche-count grids are private ephemeral
tensors rather than recognized product fields.

The resulting charge is a floating aggregate PE-equivalent response per
readout channel and sample. It is not an SI-coulomb quantity and does not claim
an explicit individual-SPAD output. Pure and noise waveforms are signal-only
and noise-only components at the same analog reference plane, not sequential
hardware products. Their composition is the analog waveform presented to the
digitization transfer.

The old fixed-grid behavior remains useful as a semantic reference, especially
the internal operation order and the rule that crosstalk and afterpulses use the
same post-dark-count source snapshot. TensorDSLab should not inherit a fixed
tensor rank, singleton batch convention, or package layout from that reference.

## TensorCore Spine And Domain Semantics

TensorCore is the dense tensor spine. TensorDSLab gives detector and readout
meaning to that spine through accepted domain surfaces.

TensorCore owns what the dense tensor record is:

```text
TensorAxis / TensorAxes / TensorLayout
TensorField / TensorCollection
TensorAxisSelection / TensorFieldSelection
```

TensorDSLab owns what the record means in the detector data-lab process:

```text
readout.photoelectrons field role
readout.charge field role
readout.waveform.pure field role
readout.waveform.noise field role
readout.waveform.analog field role
readout.waveform.digitized field role
future detector / reconstruction roles
```

TensorDSLab should defer concrete tensor shape to scripts, runtime builders,
and TensorCore layouts. It should not defer tensor semantics. Domain surfaces
and configs should identify the required product role, field role, and
sample-grid period/origin/offset needed by the operation. Readout uses exact
public `example`, `channel`, and `sample` axis IDs while leaving their tensor
dimension order and optional extra axes to the TensorCore layout provided at
runtime.

The primary readout tensor product is one concrete semantic
`ReadoutCollection(TensorCollection)` with a typed `SampleGrid` and a
conditional field-specific `DigitizedWaveformSpec`. Axis meaning comes from
the canonical public axis IDs, not a separate role sidecar. The collection is
a structurally immutable partial snapshot, not one class per charge or
waveform field. Transforms treat every retained tensor as read-only even though
PyTorch tensor storage is not intrinsically immutable. Physics transforms
remain free functions. There is no generic `Product` base, semantic
`TensorField` subclass, per-field semantic collection subclass, or
ToyProduct-like hierarchy.

## Readout Collection Snapshot Contract

`ReadoutCollection` recognizes exactly these semantic TensorCore field IDs in
this canonical topological order:

```text
readout.photoelectrons
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.analog
readout.waveform.digitized
```

A collection contains any nonempty subset. Its field order is always the
canonical order filtered to the fields present; callers do not define a second
ordering policy. Every present field has the same exact `TensorLayout`, device,
and PyTorch `torch.strided` layout; noncontiguous strided tensors are
valid. The required axes are exactly
`READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")`,
`READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")`, and
`READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")`. They are shared and may
occur in any layout order. Example and channel use exact `ExampleId` and
`ChannelId` coordinates; sample is count-only. `ChannelId` belongs in
`tensor_dslab.common` because readout and future reconstruction reuse the same
channel coordinate identity. All accepted extra axes are common to every
present field. `SampleGrid` carries sample period, origin, and containing-grid
offset. Mixed numerical domains are allowed:
photoelectrons use `torch.int64`, every present floating field
(`readout.charge`, pure, noise, and analog) shares either `torch.float32` or
`torch.float64`, and digitized waveform uses `torch.int32`.

The digitized field additionally requires `DigitizedWaveformSpec`, a typed
field-specific transfer record preserving bit depth, voltage range and offset,
analog gain, and quantization policy. The first contract accepts bit depths
from 1 through 16, inclusive gain from 0 through 40 dB, and
`AdcQuantization.TRUNCATE`. The gain range intentionally corrects the donor's
impossible out-of-range conjunction.

Collection construction is placement-neutral: tensors may already reside on
any PyTorch device when all fields share that exact device, but construction
does not assert that every later kernel supports it. CPU behavior is mandatory
and CUDA construction/collection checks are conditional on available hardware.
Later transform work orders own their device support claims.

The semantic dependency graph is:

```text
readout.photoelectrons -> readout.charge
readout.charge -> readout.waveform.pure
layout + sample grid -> readout.waveform.noise
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog
readout.waveform.analog -> readout.waveform.digitized
```

A transform returns a new collection snapshot by adding or replacing exactly
one target field. Either transform-driven addition or replacement transitively
removes every materialized descendant reachable from that target because it
would otherwise be stale. Fields that remain valid are structurally shared as
the same frozen `TensorField` records; their tensor storage is not copied and
is read-only under the transform contract. Projection or explicit field
removal is different from a semantic update and may retain descendant fields
because it does not change any retained value.

Callers also treat every materialized field tensor as read-only. A manual
in-place PyTorch edit bypasses dependency invalidation and is outside the
`ReadoutCollection` value-object contract. The only public write is to the
fresh, nonaliasing target tensor of an explicit `out` call.

`ReadoutCollection` is the primary tensor handoff. A future `ReadoutExample`
may be a thin provenance/context wrapper around it, but it should not replace
the collection with a second product graph or a loose product tuple.

Stochastic readout keys use one canonical payload order independent of tensor
dimension order: example coordinate, channel coordinate, every other
ID-backed shared axis and coordinate ordered lexically by `axis_id.value`, then
`SampleGrid.sample_offset + local_sample_index`. Additional count-only axes are
valid structural dimensions, but stochastic transforms reject them until each
has a typed containing-grid offset contract. Workspace and RNG code derive
canonical-axis positions from the ordered layout IDs rather than storing a
second role mapping.

## Readout Execution Layers

Readout construction has three explicit layers. They share scientific field
semantics but have different allocation and lifetime responsibilities.

### Semantic Layout Versus Execution Layout

Collections define meaning. Execution profiles define memory arrangement.
Domain handoffs are explicit materialization boundaries.

The semantic `ReadoutCollection` contract remains flexible: required axes may
appear in any order and read-only payloads may be noncontiguous
`torch.strided` tensors. TensorLayout is the sole semantic axis-order source of
truth. General construction does not reject internally overlapping or expanded
read-only source views solely because of storage layout. TensorDSLab does not
add an execution-ready collection subclass, axis-order/stride sidecar, or
runtime storage flag to semantic identity.

The warmed readout execution profile is narrower because its kernels operate
primarily along time. For `out + workspace`:

- `READOUT_SAMPLE_AXIS_ID` is the last dimension;
- every participating source, generated public target, and private scratch
  tensor is contiguous;
- every writable tensor is internally nonoverlapping and storage-disjoint from
  sources and other live outputs; and
- ordered axes/sizes, shape, device, role dtypes, enabled algorithms,
  destination schema, stream, and exclusive lease match the prepared
  signature.

Preflight rejects every mismatch before RNG consumption or writes. It never
permutes, calls `.contiguous()`, clones, casts, moves, performs a reshape-copy,
or falls back to allocation inside the warmed call. A contiguous
`(..., sample)` tensor lets kernels flatten the arbitrary leading axes with a
view and use length-last convolution, FFT, and scatter without hidden
materialization. Different leading-axis orders are valid but require distinct
workspaces. Contiguous strides follow from ordered shape, so the MVP workspace
signature needs no arbitrary stride tuple.

The functional path may perform explicit allocating normalization while
preserving accepted autograd. Ordinary `out` without a workspace may allocate
documented scratch or normalization and makes no allocation-free claim. A
source that is not sample-last and contiguous must be explicitly prepared once
outside a repeated warmed loop. A future allocation-free noncontiguous profile
requires its own kernel matrix, exact stride/storage contract, and
memory-instrumented evidence.

### Atomic Free Transforms

Atomic transforms remain the smallest public scientific operations. They know
their source fields, config, RNG inputs, one target field, and optional runtime
destinations; they do not know the full-chain schedule.

They support three execution modes:

```text
out=None
  -> allocating functional path
  -> new target field and collection snapshot
  -> accepted deterministic autograd behavior is preserved

out=destination, workspace omitted
  -> exact caller-owned target is reused
  -> internal scratch allocation remains permitted
  -> simulation path, not an autograd guarantee

out=destination, compatible workspace lease supplied
  -> exact caller-owned target plus preallocated scratch are reused
  -> prepared steady-state tensor-storage hot path
  -> simulation path, not an autograd guarantee
```

Every `out` destination is still an already-valid `ReadoutCollection` with the
exact post-invalidation schema. Its factory zero-initializes new targets in
contiguous storage using the collection's existing semantic axis order; it
does not reorder or materialize retained fields. A reused destination may
contain its prior valid value. Retained fields
are exact shared source records; only the fresh target is writable, and it
is internally nonoverlapping and does not alias source, retained, other live
output, or workspace storage. Every transform performs
all schema, config, alias, workspace, stream, and gradient preflight before RNG
consumption or target writes, then completely overwrites its target on
successful completion. Atomic here means preflight-safe and complete-write; it
does not promise rollback after an asynchronous backend failure.

### Reusable Readout Workspace

`ReadoutWorkspace` is a public advanced runtime resource and owns private
scratch only. It is not a TensorCore field or collection, a sidecar, scientific
config, ID, product label, cache artifact, or durable value. A returned
`ReadoutCollection` never references workspace storage.

The first workspace is caller-owned, non-resizable, non-reentrant, and bound to
one exact execution signature:

- ordered axis IDs and sizes, with positions derived from that order and the
  canonical readout IDs;
- exact device and accelerator index;
- photoelectron, common-floating, digitized, and derived complex dtypes;
- enabled algorithm families and every option that changes scratch geometry;
- exact configured destination schema; and
- one synchronous CPU execution domain or one exact CUDA stream.

Warmed use additionally requires sample-last axis order and contiguous
participating storage. No arbitrary stride tuple is stored because contiguous
strides derive from the ordered sizes.

Numeric config changes that do not change scratch geometry may reuse the same
signature. A mismatch fails before writes; the builder never silently grows,
shrinks, moves, casts, or replaces a supplied workspace. The MVP has no hidden
global, thread-local, device-local, or LRU workspace cache. One call holds an
exclusive lease. Same-stream sequential reuse relies on stream ordering;
concurrent, nested, reentrant, or different-stream use is rejected. Use one
workspace per concurrent worker or stream.

Private scratch may be allocated uninitialized during workspace preparation
because it is never a public collection value. Each scheduled kernel must
prove write-before-read for every scratch generation. The charge path requires
at least the logical roles `count_source`, `count_total`, and a reusable
secondary-contribution slot so the frozen post-dark source remains live while
crosstalk and afterpulse contributions are formed. Two-slot ping-pong is not a
general promise; the exact physical inventory and possible fusion require
implementation evidence.

Ping-pong within one private count-domain storage class swaps references
between exact contiguous A/B buffers; it does not permute or copy data.
Integer count, floating charge/waveform, and integer ADC boundaries use
separate buffers because their dtype and semantic storage classes differ.

### Full-Chain Readout Builder

`build_readout_collection(...)` is the ordinary local domain entry point. It
owns enabled-operation order, exact stage-result schemas, scratch-slot
liveness, target selection, and final collection assembly. This is in-process
readout composition, not source loading, device movement, cache IO, or
Projects/dag campaign orchestration. Low-level atomic transforms remain public
for focused tests and advanced callers.

Non-final surface sketch:

```python
workspace = ReadoutWorkspace.allocate(source, config, stream=stream)
destination = build_readout_output_buffer(
    source,
    floating_dtype=floating_dtype,
    replace_photoelectrons=timing_enabled,
    digitized_waveform_spec=digitized_waveform_spec,
)

result = build_readout_collection(
    source,
    config,
    rng=rng,
    workspace=workspace,
    out=destination,
)
```

The first builder requires `readout.photoelectrons` and recomputes the complete
configured chain: optional timing jitter, charge, pure, noise, analog, and
optional digitized waveform. The canonical result contains photoelectrons,
charge, pure, noise, and analog, plus digitized exactly when configured. When
timing is disabled, photoelectrons may be the exact source record; otherwise
every produced field has distinct public target storage. Existing derived
source fields are not opportunistically trusted.

The builder composes one-target atomic operations through stage-specific exact
collection views over the prepared public fields and workspace scratch. It
schedules each final compatible write directly into its public destination and
does not clone result tensors during final assembly. Public output fields that
coexist in the result never ping-pong through one payload; automatic buffer
reuse is limited to private intermediates whose last reader has completed.

### Output Lifetime And Allocation Claim

With `out=None` and no workspace, the builder uses the allocating functional
paths and returns ordinarily owned results. A workspace is accepted only with
a supplied destination; workspace-without-`out` is rejected. Supplying `out`
selects the non-autograd simulation path and rejects gradient-sensitive use
before mutation. A supplied destination is caller-owned and remains stable
after return until the caller explicitly submits it as writable `out` again.
That reuse authorizes overwrite and ends the previous result's stable snapshot
lifetime. Overlapped producers and consumers therefore use caller-managed
output banks; the workspace never doubles as an output pool.

The phrase *steady-state TensorDSLab-managed tensor-storage allocation-free*
applies only after warm-up when both an exact output destination and compatible
workspace are supplied for an unchanged execution signature. It means every
TensorDSLab-owned public target and named scratch tensor comes from prepared
storage. It does not promise zero lightweight Python records or views, zero
allocator bookkeeping, or zero opaque PyTorch/CUDA library planning and
scratch. Stronger backend-wide zero-allocation or CUDA Graph claims require a
focused, instrumented work order.

Readout's sample-last profile is not a universal TensorDSLab layout. A future
Readout-to-Reconstruction bridge should select required fields, validate the
exact `ChannelId` set, reorder by stable axis IDs/coordinates, explicitly
materialize contiguous reconstruction-preferred storage once, and construct a
reconstruction-owned semantic input before entering that domain's execution
profile. Cross-channel reconstruction may prefer channel-last.

## Ownership Boundaries

### G4DS And TensorG4DS Boundary

G4DS owns detector simulation execution and native simulation output.
TensorG4DS owns the G4DS-facing tensor-native low-level analysis boundary,
including accepted deposit semantics, truth-level deposit clustering,
assignment/summary products, and its own event/deposit/cluster identities.
TensorDSLab does not parse native G4DS files, reproduce TensorG4DS algorithms,
or relabel a generic collection as if it proved those upstream semantics.

The long-term ecosystem data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is not a Python import graph. TensorCore is the shared tensor substrate.
TensorG4DS must not import TensorDSLab; core `tensor_dslab.common` and
`tensor_dslab.readout` remain independent of TensorG4DS. A later
TensorDSLab-owned leaf adapter may import an exact, versioned public TensorG4DS
type after both repositories accept that integration contract.

TensorG4DS currently has no implemented or frozen GPU output API. The eventual
production boundary must require already-placed TensorCore/Torch payloads and
preserve one explicit accelerator device without implicit `.cpu()`, NumPy/list
conversion, serialization/reload, movement, casting, or detachment. The bridge
may compute new tensors on the same device; direct handoff does not mean that
deposit/cluster and example/channel/sample layouts are interchangeable or that
detector response is zero-copy. TensorCore IDs, layouts, and small semantic
records may remain host-side metadata.

The current TensorG4DS semantic unit contract is centimetres for positions,
nanoseconds for times, and keV for deposited energies, although its exact field
schema and unit encoding remain open. The future bridge must validate the
accepted versioned units and perform any required conversion as an explicit
documented on-device detector transform.

### TensorDSLab Boundary

TensorDSLab owns the post-TensorG4DS middle link:

- explicit mapping from accepted TensorG4DS event provenance to one or more
  TensorDSLab `ExampleId` values;
- detector response, photon/electron yield and transport, sensor/channel
  mapping, window/grid construction, and PE binning when their focused stages
  are accepted;
- one primary `ReadoutCollection` tensor handoff and future optional thin
  `ReadoutExample` provenance/context composition;
- future reconstruction example construction and products;
- semantic tensor products and domain transforms over TensorCore primitives;
- future durable cache writing, loading, validation, and deterministic
  storage-level compaction after in-memory contracts are accepted;
- future integration surfaces after local contracts are accepted.

When the future TensorG4DS bridge is accepted, it uses an exact nominal public
upstream type rather than a loose protocol or product tuple. It validates
upstream semantics, units, relationships, layouts, dtype, and device before
constructing new TensorDSLab-owned axes and values. `ReadoutCollection` remains
the primary tensor handoff, and `ReadoutExample` remains an optional thin
wrapper:

```text
TensorG4DS EventId + deposit/cluster products
  -> explicit provenance/coordinate mapping and detector response
  -> ReadoutCollection{"readout.photoelectrons"}
ReadoutCollection -> optional ReadoutExample provenance/context
ReadoutCollection -> future reconstruction tensor/product views
```

TensorG4DS `EventId`, `DepositId`, and `ClusterId` remain upstream identities.
An event may produce zero, one, or multiple TensorDSLab examples, so `EventId`
is provenance and must never be cast to or silently reused as `ExampleId`.
`ChannelId` remains TensorDSLab-owned; channelization is not inferred from an
upstream tensor index or matching string.

This discrete upstream bridge carries no end-to-end autograd promise. It must
not detach silently; the first clustering/detector-response integration should
reject gradient-sensitive inputs unless a focused differentiable contract is
accepted. That does not weaken the existing functional-autograd contract for
accepted deterministic waveform transforms later in the readout chain.

### TensorCore Boundary

TensorCore owns generic tensor identity, axes, layouts, fields, collections,
selections, batching, movement, validation, and pure tensor operations.
TensorDSLab should import those surfaces from `tensor_core` instead of copying
or mirroring them.

Imports should use the public `tensor_core` package root. TensorDSLab may
subclass the open `Id` and `TensorCollection` extension points, but it must not
subclass sealed TensorCore primitives or re-export generic TensorCore helpers
through a local common package.

TensorDSLab domain IDs may appear as TensorCore coordinates when they subclass
TensorCore `Id`. TensorCore should not import TensorDSLab or own domain
concepts.

### TensorML Boundary

TensorML and other downstream ML packages own source adaptation for model training, split
planning, batching policy, models, objectives, metrics, training, evaluation,
checkpoints, and ML-specific artifacts.

TensorDSLab should not design its first stages around downstream package
requirements. Future consumers should be able to depend on stable typed
products or accepted adapters, but those adapters are deferred until the local
product graph is stable.

For future TensorML use, a model-facing `TensorFieldSelection` order is a
positional argument ABI. The `ReadoutCollection` class alone does not declare a
model schema, and generic TensorCore selection and batching return base
`TensorCollection` records rather than preserving the semantic subclass. A
future adapter may project and, where needed, reconstruct a semantic
collection. This stage does not request TensorML `input_fields` or
`output_fields` changes.

### Projects/dag Boundary

Projects/dag owns campaign fanout and fanin, scheduling, dispatch, retries,
repair, cancellation, status, concrete DAG objects, scheduler-visible grouping,
and cross-shard orchestration.

`build_readout_collection(...)` is not an exception to this boundary. It is a
single-process domain builder that orders the fixed local readout transforms
over already-present tensors; it owns no campaign discovery, scheduling,
retry, repair, or cross-shard policy.

TensorDSLab may later expose DAG-compatible operation specs and recipe
fragments, but local product and cache contracts should be accepted first.
TensorDSLab owns a future deterministic storage-compaction primitive;
Projects/dag owns discovering/scheduling inputs and campaign or cross-shard
fan-in execution.

## Product Labels And Readout Tensor Fields

The first `ReadoutCollection` recognizes these exact TensorCore field IDs:

```text
readout.photoelectrons
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.analog
readout.waveform.digitized
```

These are tensor-local semantic field identities. Producer-owned durable
product labels remain a separate namespace and require an explicit bridge even
when a label uses the same spelling. `readout.photoelectrons` is also an
accepted durable producer label, but accepting the label does not by itself
accept a durable cache format or compatibility contract.

The first tensor-native readout work should make axes, layouts, coordinates,
and indices explicit. It should not smuggle product labels, channel identity,
sample positions, or row identity through implicit array positions.

## Initial Package Direction

The likely future package tree is a roadmap, not permission to create
placeholder modules:

```text
tensor_dslab/
  common/            # shared IDs such as ExampleId/ChannelId, only when real
  detector/          # optional post-TensorG4DS semantics, only when real
  readout/           # ReadoutCollection, sample/ADC records, and field transforms
  reconstruction/    # future reconstruction products, reusing common ChannelId
  caches/            # future durable cache bridge, after in-memory contracts
  executables/       # future task doors, only when accepted
  operations/        # future operation specs, only when accepted
  recipes/           # future recipe fragments, only when accepted
```

Domain packages live directly under `tensor_dslab`; there is no intermediate
`tensor_dslab.domain` namespace. Required domains are those accepted by a
concrete implementation stage. A missing folder is better than a decorative
folder.

## Parts-Bin Policy

Historical predecessor code is parts-bin material only. It can inform
scientific behavior, cache semantics, algorithms, tests, and naming lessons,
but it is not binding architecture.

Reference locations:

```text
/Users/mbedard/Projects/TensorCore
/Users/mbedard/Projects/TensorML
/Users/mbedard/Projects/dslab
/Users/mbedard/Projects/iv-dslab-main_db_PB
```

Reference precedence:

1. TensorDSLab decisions and architecture pages are authoritative for accepted
   product, scientific, and cache behavior.
2. [IV-DSLab Parity](parity.md) governs donor comparison classifications,
   assumptions, evidence, and intentional divergences without overriding the
   target architecture.
3. TensorCore is authoritative for generic tensor contracts.
4. TensorML is a workflow, documentation, and tensor-spine style reference,
   not a detector data-lab domain template.
5. DSLab and IV-DSLab are scientific behavior, fixture, and cautionary-example
   donors only.

Promote only the reviewed concept that fits the tensor-native design. Do not
inherit donor architecture automatically. Adopting a donor concept does not
by itself establish exact, numerical, distributional, or statistical parity.

## Current Non-Goals

- No production package code outside the dispatched Stage 2 structural
  foundation.
- No package metadata or dependency surface beyond the exact Stage 2 work
  order.
- No copied donor code.
- No native G4DS reader, TensorG4DS dependency, or upstream integration adapter
  in the current post-binned stages.
- No tests beyond the dispatched Stage 2 structural contract until a later
  focused production work order or explicit validation-only stage accepts
  them.
- No cache schema commitment.
- No durable IO, manifest, compaction, or cache compatibility requirements
  before the in-memory product model is accepted.
- No DAG operation specs, recipe fragments, executable doors, local DAG
  factories, or downstream adapter surfaces.
- No downstream model, training, evaluation, objective, metric, checkpoint, or
  artifact surfaces.
- No semantic collection subclass per recognized readout field.
- No TensorML model-schema or `input_fields` / `output_fields` API change.
- No local fork of TensorCore concepts.
- No execution-ready collection subclass, storage-layout sidecar, or one
  universal axis order across readout, reconstruction, and TensorML.
