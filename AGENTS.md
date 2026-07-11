# Agent Workflow

This repository uses role-separated Codex ownership. Design owns architecture,
decisions, validation expectations, and future work orders directly. Stage 2
activated the first production Implementation/Validation/Review loop on
2026-07-11. Documentation-only Design work outside that dispatched branch may
still remain in Design unless the user requests independent Validation or
Review.

The production workflow is:

```text
Design -> Implementation + Validation -> Review -> Implementation fixes -> Review recheck
```

Before production implementation starts, establish one persistent thread per
role for each active TensorDSLab workspace:

- Design
- Implementation
- Validation
- Review

Tasks and stages are passed through handoffs, not represented as new permanent
threads. Documentation-only Design work does not need to simulate the
Implementation/Validation/Review loop. Once code is in scope, the role split
keeps architecture, implementation, behavioral validation, and independent
critique separate enough that each thread can do its job without blurring
ownership.

When a stage spans multiple repositories, such as TensorDSLab, TensorG4DS,
TensorCore, TensorML, G4DS/g4ds11, or Projects/dag, keep each workspace's role
threads explicit in the handoff. A role from one repository must not silently
own implementation, validation, review, or merge work in another repository.
Any exception must be explicit in the handoff and accepted by the user and
every affected package Design authority.

Agents should also follow `CONTRIBUTING.md`, which defines repository-wide
engineering standards. Start with `docs/overview.md` for the documentation map.
Design work orders should cite the relevant `CONTRIBUTING.md` standards when a
stage touches TensorCore layout, TensorDSLab product semantics, in-memory
product relationships, durable cache shape, validation boundaries, public
typing, IDs versus indices, artifacts, or future integration boundaries.
They should also cite `docs/parity.md` when promoting donor behavior, changing
a comparison boundary, or accepting a statistical approximation or intentional
divergence.
Validation and Review should treat violations of accepted `CONTRIBUTING.md`
standards as real findings, not style-only comments.

## Governance Authority And State

TensorDSLab Design owns this package's architecture, public contracts,
ownership boundaries, accepted dependencies, documentation, work orders,
governance adoption, conformance findings, routing, and deviations. A
cross-package proposal binds TensorDSLab only after every affected package
Design authority ratifies the same immutable proposal. Coordination agreement,
Moderator synthesis, tests, work orders, and similarity among package documents
are evidence; none creates package architecture.

Package sources take precedence for TensorDSLab architecture and contracts. If
package and cross-package sources disagree, stop the affected work or routing,
identify the conflicting records, return the contradiction to every affected
Design authority, and resume only from an explicit resolution and synchronized
baseline. `AGENTS.md` governs roles, handoffs, routing, work-order gates, and
verification responsibilities. `CONTRIBUTING.md` governs engineering quality,
API design, typing, validation, testing, documentation, and code style.

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`, bound to
exact accepted candidate `d634401a853915edeb4f83df4a4943b3553deced`. The
current package states are:

```text
package_adoption_state: Adopted
conformance_finding: Not evaluated
coordination_status: Deferred
registry_storage_profile: Disabled
```

These states are independent. Package adoption does not constitute conformance,
routing activation, Coordination activation, production authority, or Stage 2
dispatch.

Design may operate alone during documentation-only maturity. Design,
Implementation, Validation, and Review are persistent logical roles per
workspace after activation. Production dispatch requires every execution role
named by the work order to be Active and verified. A dormant, stale, missing,
or discrepant route does not authorize dispatch; procedural routing returns to
Design.

Coordination is an optional representation role and remains Deferred.
TensorDSLab Design is its procedural fallback. Coordination may represent only
an accepted package position and may not ratify architecture, command Design,
dispatch implementation, replace D/I/V/R, edit production or tests by virtue
of the role, or own merge authority. Later activation requires a concrete
recurring cross-package need, an accepted charter and Design-return path, an
adopted routing/privacy procedure, verified route and fallback, no routing
discrepancy, and explicit Design and user authorization.

The Ecosystem Moderator is neutral and procedural. It may distribute
authorized packets, collect package positions, synthesize agreements and
objections, and maintain authorized procedural records. It may not represent
TensorDSLab, vote or break ties, ratify architecture, command Design, dispatch
package execution roles, modify this repository, broaden package ownership,
conceal objections, or infer consent from silence.

TensorDSLab Design owns package routing and discrepancy resolution. Stable
logical package, workspace, role, and work-order keys are primary. Raw platform
route identifiers are optional private attributes and must not appear in
committed package records. Profile B is disabled and not instantiated: do not
create `.agents`, an ignore rule, a committed route table, a private live-route
store, or a Moderator cache because the common core was ratified, a candidate
was prepared, or package adoption was issued. A discrepancy pauses only the
affected routing and returns to Design. Profile B requires a later focused
Design decision covering the private path, ignore policy,
permissions/operators, sharing,
replacement/history/deletion, verification, and discrepancy procedure.

TensorDSLab is in active development and pre-deployment. It makes no
deployability, release-readiness, backward-compatibility, or broad
cross-package compatibility claim. Later compatibility evidence is limited to
exact named commits, environment, device/backend, and execution mode. The
same-device and no-silent-host-materialization Design constraints are not proof
of an implemented or compatible package handoff.

## Project Mode

TensorDSLab is a clean-slate, tensor-native detector data-lab package. It
consumes accepted TensorG4DS tensor-native products and turns them into typed
readout and future reconstruction products, while using TensorCore as the
generic tensor identity, layout, field, collection, selection, batching,
movement, validation, and pure operation backbone.

The intended chain is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is the intended data flow, not an import graph or a claim that every
boundary is implemented. TensorCore is the shared substrate across the three
tensor packages. TensorDSLab must not parse native G4DS files or implement
TensorG4DS low-level analysis such as deposit clustering. A later focused
integration stage may add a narrow TensorDSLab-owned adapter over an accepted
public TensorG4DS product.

TensorDSLab owns its post-TensorG4DS detector/readout semantics, readout and
future reconstruction products, and future durable cache contracts. It should
not own native G4DS ingestion, TensorG4DS deposit/cluster products or
algorithms, generic TensorCore primitives, TensorML
model/training/evaluation surfaces, checkpoint policy, metric reporting, or
campaign orchestration.

The first accepted MVP direction is the post-binned tensor-native readout
path: already-binned photon-origin primary photoelectrons, the aggregate SiPM
charge response, waveform products, analog waveform composition, and optional
digitization. Native G4DS parsing is permanently upstream of TensorDSLab.
Defer the typed TensorG4DS handoff, detector-window construction,
photoelectron binning, IO, cache compatibility, DAG compatibility, and
TensorML integration until the post-binned contract is stable.

Historical predecessor code, if consulted outside this repository, is parts-bin
material only. It may provide scientific facts, algorithms, fixtures, tests,
and cautionary examples, but it does not define current architecture by
default. Do not copy old package layouts, helper framework shape,
compatibility baggage, or DAG-facing mechanics into TensorDSLab by default.
Promote only reviewed behavior that fits the tensor-native design and is
recorded in TensorDSLab docs. Every promoted donor behavior must name its
comparison boundary and parity classification or intentional divergence in
`docs/parity.md`.

TensorML is a style and workflow reference, not a detector data-lab domain
template. Replace TensorML process semantics with TensorDSLab product and cache
semantics when adapting docs or patterns. TensorCore is the source of truth for
generic tensor vocabulary and contracts.

The accepted `main` baseline remains documentation-only while Stage 2 executes
through its dispatched package-owned production loop. Stage 1 post-binned
readout architecture and the Stage 2 package-and-collection work order are
Design-complete. Only Stage 2's focused work order accepts package modules and
tests; it does not accept cache schemas, DAG surfaces, downstream integration
surfaces, or copied donor code. A feature-branch candidate is not an accepted
package surface until fixed-commit Validation, independent Review, and clean
merge complete.

If implementation reveals a concrete contradiction in the accepted design, stop
and send the issue back to Design. Do not silently widen architecture, create
placeholder package trees, add DAG-facing surfaces, rename public concepts,
fork TensorCore, or copy donor code into production modules inside an
implementation thread.

## Package Shape And Imports

Use the ecosystem naming convention:

```text
Project/display folder: TensorDSLab
Python import package:  tensor_dslab
```

The checkout root is the project folder. When production code is accepted, the
`tensor_dslab/` directory should be the Python import package. Do not create a
flat TitleCase Python package that imports as `TensorDSLab`.

Keep semantic subpackages flat beneath that import root:

```text
tensor_dslab/
  common/
  detector/          # optional post-TensorG4DS semantics, only when accepted
  readout/
  reconstruction/
  caches/
  executables/       # future integration surface, only when accepted
  operations/        # future integration surface, only when accepted
  recipes/           # future integration surface, only when accepted
```

Runtime commands launched from the project root should use the project root on
`PYTHONPATH` so absolute `tensor_dslab.*` imports resolve:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

Production imports should stay absolute, such as:

```python
from tensor_core import TensorAxisId, TensorLayout
from tensor_dslab.common import ChannelId, ExampleId
from tensor_dslab.readout import READOUT_CHANNEL_AXIS_ID
```

Do not rewrite imports to relative forms to satisfy editor-only diagnostics.
Editor analysis tools should mirror the runtime path by including the project
root on their analysis path.

Do not create placeholder modules to reserve architecture. Add a module only
when there is a real TensorDSLab concept, behavior, or contract to house.

## TensorCore Boundary

TensorDSLab should use TensorCore directly for generic tensor contracts:

- `Id`, `TensorAxisId`, and `TensorFieldId`;
- `IdSequence` and generic constrained scalars such as `PositiveInteger`,
  `NonnegativeInteger`, `FiniteFloat`, `PositiveFloat`, `NonnegativeFloat`, and
  `Probability`;
- `TensorAxis`, `TensorAxes`, `TensorLayout`, `TensorField`, and
  `TensorCollection`;
- tensor selections such as `TensorFieldSelection` and
  `TensorAxisSelection`;
- generic builders, validators, immutable mapping helpers, batching helpers,
  movement, reduction, selection, detachment, addition, and other pure tensor
  operations.

Production imports must come from the public `tensor_core` package root, not
TensorCore implementation modules. Do not re-export generic TensorCore helpers
through `tensor_dslab.common`.

TensorCore's intentional downstream extension points are `Id` and
`TensorCollection`. TensorDSLab may subclass `Id` for domain coordinates and
defines one primary readout collection, `ReadoutCollection`, directly from
`TensorCollection`. Do not subclass sealed TensorCore primitives such as
axis/field IDs, scalar wrappers, axes, layouts, `TensorField`, selections, or
`BatchConfig`.

TensorDSLab owns domain IDs, semantic collection and field contracts, builders,
validators, and domain transforms. Future cache records and loaders belong in
TensorDSLab only after in-memory product contracts are accepted. Domain IDs
may appear as TensorCore coordinates when they subclass TensorCore `Id`, but
they should not become TensorCore-owned primitives.

TensorCore is the dense tensor spine. TensorDSLab gives TensorCore records
detector/readout product meaning instead of recreating generic tensor
mechanics. Runtime scripts and builders may choose concrete TensorCore axis
order, but TensorDSLab fixes required readout-axis identities and makes field
roles, sample-grid facts, and stochastic coordinate inputs explicit. The first
readout products are recognized fields inside one direct
`ReadoutCollection(TensorCollection)` subclass, with free transform functions,
a typed `SampleGrid`, and a conditional typed `DigitizedWaveformSpec`. Do not
create caller-defined semantic axis-role mappings, one single-field collection
subclass per product, a generic `Product` base, or a ToyProduct-like hierarchy.

Every valid `ReadoutCollection` is a nonempty, structurally immutable,
partially materialized snapshot. Transforms treat retained tensor payloads as
read-only, and callers must not mutate existing field tensors in place. Only
the fresh target tensor of an atomic `out=` call, or the prepared generated
field set held exclusively by `build_readout_collection(...)`, may be written.
Mutable private scratch belongs only to a separate runtime
`ReadoutWorkspace`; it is never a collection field or sidecar. The collection's
present fields are an ordered subsequence of this canonical schema:

```text
readout.photoelectrons
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.analog
readout.waveform.digitized
```

`tensor_dslab.common` owns and exports the shared stable coordinate types
`ExampleId` and `ChannelId`. `tensor_dslab.readout` owns and exports the
readout-specific axis and field constants, including exactly these required
semantic axis identities:

```python
READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")
READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")
READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")
```

All present fields in one snapshot use the same exact ordered layout and
device, and every tensor uses `torch.strided` layout; noncontiguous strided
tensors remain valid collection structure. The three required
axes occur exactly once and may appear in any layout order; locate them by
`TensorAxisId` value equality and `TensorAxes.index(...)`, never a fixed tensor
dimension or object identity. The example axis is ID-backed by exact
`ExampleId` coordinates, the channel axis by exact `ChannelId`
coordinates, and the sample axis is count-only. Photoelectrons use exactly
`torch.int64`; charge, pure, noise, and analog use one common
`torch.float32` or `torch.float64` dtype; and digitized ADC counts use exactly
`torch.int32`. If the accepted
layout has additional axes, every field carries them and the collection
declares them shared too. `shared_axes` lists every common-layout axis ID in
layout order. TensorDSLab constructors, not TensorCore, enforce these stronger
invariants. `SampleGrid` remains the typed source of regular sample-grid facts;
there is no `ReadoutAxisRoles` sidecar.

Collection semantics and warmed execution storage are separate contracts.
Arbitrary axis order and noncontiguous `torch.strided` values remain valid
read-only collection structure; the semantic constructor does not reject an
expanded or internally overlapping source merely because of its storage
arrangement. TensorLayout is the only semantic axis-order source of truth. Do
not add an `ExecutionReadyReadoutCollection`, axis-order sidecar, stride
metadata, or runtime policy to collection identity.

The MVP warmed `out + workspace` readout profile is stricter:

- `READOUT_SAMPLE_AXIS_ID` is the last tensor dimension;
- every participating source, generated public target, and scratch tensor is
  contiguous;
- every writable target and scratch tensor is internally nonoverlapping and
  storage-disjoint from sources and other live outputs; and
- the ordered axes/sizes, shape, device, role dtypes, algorithms, destination
  schema, bound stream, and exclusive lease match exactly.

Preflight rejects a mismatch before RNG consumption or writes. It must not
permute, call `.contiguous()`, clone, cast, move, use a reshape that copies, or
fall back to allocating storage inside the warmed call. Leading-axis order
remains flexible, but each order is a different workspace signature. Because
contiguous strides derive from ordered shape, the MVP signature needs no
arbitrary stride tuple. A future allocation-free noncontiguous profile requires
a focused Design decision and memory-instrumented evidence.

The digitized field conditionally owns a typed `DigitizedWaveformSpec` sidecar
with bit depth from 1 through 16, voltage transfer, inclusive `[0, 40]` dB
analog gain, and `AdcQuantization.TRUNCATE`. It is
required exactly when `readout.waveform.digitized` is present so projected or
loaded ADC counts remain interpretable and bounds-checkable. Invalidation or
projection that removes the field removes the spec too.

TensorCore terminology is strict:

- a coordinate is a stable `Id` value associated with an ID-backed axis;
- an index is a zero-based integer tensor position along an axis;
- a layout is ordered axes plus coordinate-to-index maps for ID-backed axes.

Coordinates and indices are never interchangeable. Do not persist transient
tensor, table, or array indices as durable identity. Diagnostics, caches, and
reports should prefer semantic IDs when an axis is ID-backed.

`IdSequence` preserves caller order and rejects empty, duplicate, base-`Id`, or
mixed-concrete-class values. TensorDSLab builders must not sort or infer ID
order and should validate the exact domain coordinate class for ID-backed axes.

Later coordinate-addressed stochastic transforms derive logical identity
independently of tensor dimension order: required example-axis ID and
coordinate first, required channel-axis ID and coordinate second, then every
other ID-backed shared axis ordered lexically by `axis_id.value` and paired
with its coordinate, followed by
`SampleGrid.sample_offset + local_sample_index`. An operation-local
draw/counter coordinate follows when needed. Extra count-only axes are valid
collection structure, but a stochastic transform must reject one that lacks an
accepted stable global-offset rule.

TensorCore contract changes require explicit Design acceptance in the
TensorCore workspace. TensorDSLab should not fork TensorCore, keep local
compatibility shims for retired TensorCore names, or broaden TensorCore public
surfaces from a TensorDSLab implementation stage.

## Product Relationships And Boundaries

TensorDSLab should preserve this data-flow and ownership rule unless Design
accepts a focused change:

```text
G4DS native products
  -> TensorG4DS typed tensor-native products
  -> deferred TensorG4DS-to-TensorDSLab bridge
       -> explicit provenance and coordinate mapping
       -> detector-window/readout-grid construction
       -> photon-origin PE binning
  -> ReadoutCollection{"readout.photoelectrons"}
  -> TensorDSLab readout and future reconstruction tensor views
  -> deferred explicit TensorML field-selection/model boundary
```

The external chain is data flow, not package dependency flow. Core readout and
common modules depend only on TensorCore. A future downstream-owned bridge may
import an exact accepted public TensorG4DS type; TensorG4DS must never import
TensorDSLab to construct downstream identities. The bridge is a semantic
transformation, not a subclass cast or an assumption that TensorG4DS and
TensorDSLab layouts are interchangeable.

The production integration target keeps tensor payloads resident on one
explicit accelerator device across TensorG4DS, TensorDSLab, and TensorML.
Boundary code must not silently call `.cpu()`, `.numpy()`, serialize/reload, or
otherwise use host materialization as the package handoff. New computations
may allocate new tensors on that same device, and TensorCore IDs, layouts, and
other small semantic records may remain ordinary host-side metadata. Device
movement is always explicit. Because TensorG4DS has not yet frozen a public
GPU output contract, the exact accepted input type, dtype/layout matrix, and
device-preservation tests belong to the future integration work order; they do
not add a TensorG4DS dependency to the post-binned Stage 2 foundation.

The discrete TensorG4DS bridge carries no end-to-end autograd promise and must
not detach silently. Its first work order should reject gradient-sensitive
inputs unless Design accepts a separate differentiable detector surface. This
does not weaken functional autograd for accepted deterministic waveform
transforms later in TensorDSLab.

The primary readout tensor handoff is `ReadoutCollection`, not a loose product
tuple or a required dataclass adapter. A future `ReadoutExample` may be a thin
provenance/context record containing a collection; it is not the tensor spine.
TensorG4DS `EventId` values and native G4DS event values are upstream
provenance, not TensorDSLab row identity. `tensor_dslab.common` owns and
exports opaque stable `ExampleId` and `ChannelId` coordinates. A later bridge
stage must define explicit provenance-to-example and channel-coordinate
mapping rather than equating IDs or guessing identity from transient indices.

Producer product labels such as `readout.photoelectrons`, `readout.charge`,
`readout.waveform.pure`,
`readout.waveform.noise`, `readout.waveform.analog`,
`readout.waveform.digitized`, and future
reconstruction labels are durable TensorDSLab product labels. TensorCore
`TensorFieldId` values identify runtime collection fields. A durable label may
use the same string as its corresponding field ID, but the namespaces and
types remain explicit and are never interchangeable.

Consumer-facing adapters are deferred. TensorDSLab should first make the local
typed product graph coherent enough that future consumers can depend on it
without parsing raw `.fil`, table, array, manifest, or private representation
details.

The first readout operations should work from the already-binned primary-PE
field in a `ReadoutCollection`:

```text
readout.photoelectrons
  -> timing jitter
  -> simulate_charge
       -> private dark-count avalanche grid
       -> private frozen-snapshot crosstalk and afterpulse contributions
       -> private aggregate charge smearing
  -> readout.charge
  -> readout.waveform.pure

shared layout/sample grid -> readout.waveform.noise
pure + noise -> readout.waveform.analog -> readout.waveform.digitized
```

`readout.photoelectrons` contains binned photon-origin primary PE seeds. Only
timing jitter replaces that field. One public `simulate_charge` transform
consumes it, performs dark counts, frozen-snapshot crosstalk and afterpulses,
and charge smearing internally, and adds or replaces `readout.charge`.
Intermediate avalanche-count grids are private runtime scratch values, not
recognized fields or durable products; their storage may come from a caller-
owned compatible workspace. Do not expose an apparently sequential public API
that lets one generated contribution feed another by accident.

`readout.charge` is the finite floating aggregate PE-equivalent response per
readout channel and sample. It is not an SI-coulomb measurement and does not
claim an explicit individual-SPAD output. Pure and noise waveforms are
signal-only and noise-only components at one shared analog reference plane;
they are not sequential hardware products. Their composition produces the
analog waveform consumed by digitization.

Readout transforms consume a `ReadoutCollection` and return a new coherent
snapshot by adding or replacing one recognized field. They retain unaffected
fields by structural sharing and remove all transitive descendants of the new
or replaced dependency. Projection is different: it may retain any nonempty
canonical subset without invalidating a retained descendant because no
retained value changed. Centralize these rules; do not edit field mappings ad
hoc or leave stale derived fields beside an updated dependency.

The accepted invalidation graph is:

```text
readout.photoelectrons -> readout.charge -> readout.waveform.pure
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog -> readout.waveform.digitized
```

Replacing a field removes every reachable descendant. Replacing photoelectrons
does not remove noise when layout and sample-grid semantics are unchanged.

Readout execution has three layers:

1. Atomic free transforms remain independently callable. With `out=None`, a
   transform follows the functional allocating path, leaves inputs unchanged,
   may explicitly normalize arbitrary semantic order/strides with allocation,
   and preserves autograd for accepted differentiable deterministic behavior.
   With `out=`, it writes the complete target field of the exact supplied valid
   result collection and returns that exact collection. A workspace is valid
   only with `out`; supplying `out` selects a non-autograd simulation path.
   Without a workspace, ordinary scratch or explicitly accepted normalization
   may allocate and carries no allocation-free claim. Every writable target
   must be internally nonoverlapping even when the read-only source is not.
2. `ReadoutWorkspace` is a public caller-owned, runtime-only, scratch-only
   object. A compatible workspace lets atomic transforms reuse prepared private
   scratch. It is never a collection field, sidecar, scientific config, ID,
   product, cache record, or returned result.
3. `build_readout_collection(...)` is the higher-level local domain builder. It
   owns the fixed configured full transform chain and automatic scratch
   schedule while keeping the low-level atomic transforms available. It does
   not load sources, perform IO, move or cast tensors, or own Projects/dag
   scheduling. It preflights the complete configured chain, destination, and
   workspace before its first atomic RNG draw or write.

Every atomic call performs all destination, workspace, placement, dtype,
algorithm, and lease preflight before consuming RNG state or launching writes.
`out` must have the exact expected result field set and canonical order,
sidecars, layout, device, role dtypes, and `torch.strided` tensors. Its
unaffected fields are the exact retained source `TensorField` records; its
writable target does not alias a source tensor or workspace scratch. The target
is zero-initialized at public construction and is overwritten completely.
Preflight failure leaves RNG and tensors untouched. After a backend operation
has launched, failures carry no transactional rollback guarantee.

A `ReadoutWorkspace` is exact for one fixed ordered sequence of layout axis IDs
and sizes, device, role dtypes, algorithm scratch signature, destination
schema, and CPU/default or specific CUDA stream. The warmed profile additionally
requires sample-last order and contiguous participating tensors. Required-axis
positions are derived from the ordered IDs by axis-ID equality/index; the
signature stores no semantic-role sidecar or arbitrary stride tuple. It never
caches, resizes, moves, casts, or recycles source tensors. One nonreentrant
exclusive same-stream lease is allowed at a time. Scratch may use uninitialized
storage only when every consumer proves write-before-read; logical private
charge roles include source, total, and contribution storage, so the design
must not claim that two physical buffers always suffice. Returned collections
never reference workspace storage.

The full `build_readout_collection(...)` result contains photoelectrons,
charge, pure, noise, analog, and optional digitized fields. When timing jitter
is disabled it may structurally share the source photoelectron field. With both
`out=None` and no workspace, the builder is a functional allocating path. The
warmed steady-state hot path requires an exact caller-prepared `out` and
workspace. “Allocation-free” means no TensorDSLab-managed tensor-storage
allocation in that warmed path; it does not promise zero Python object,
PyTorch-internal, or vendor-library allocation.

Private count-domain ping-pong swaps references between contiguous buffers of
one exact shape/order/dtype/device class; it never permutes data. Integer
photoelectron/avalanche counts, floating charge/waveforms, and integer ADC
codes use separate storage classes. Public output preparation allocates fresh
targets contiguously in the source collection's existing semantic axis order;
it does not reorder retained fields. A separate one-time preparation step
outside the repeated loop is required when a source is not sample-last and
contiguous.

Caller-owned `out` storage remains stable until the caller submits that same
destination again. Reuse authorizes complete overwrite, so overlapping
consumers require caller-managed output banks. Mutation, allocation, workspace,
and stream policy are runtime control, not TensorCore identity; do not encode
them as a TensorCore `Id`, field ID, axis ID, coordinate, product label, or
scientific config.

TensorCore field selection, batching, movement, and like-allocation return
base `TensorCollection` records. TensorDSLab owns validated reconstruction and
canonical projection helpers. For TensorML, an explicit
`TensorFieldSelection` and its requested order are the positional model schema;
do not pass a growing full readout snapshot as an implicit model ABI or assume
that collection subclass identity survives the stock selection loop.
Keep noncanonical model selections as base `TensorCollection` values; do not
reorder them to recover subclass identity. A stock-loop model accepts that base
type unless a focused adapter reconstructs a canonical subset before
`forward_pass`.

Generic base results do not retain subclass sidecar attributes. Reconstruction
helpers explicitly carry `SampleGrid` and the conditional
`DigitizedWaveformSpec` around the TensorCore operation, update or prune them
for the result, and pass them to the semantic constructor. Required axis
meaning is recovered from the exact exported axis IDs, not a sidecar or
free-form metadata.

Projection and placement are explicit memory policy. Project the needed fields
before accelerator movement, move only that base collection through
TensorCore, then reconstruct `ReadoutCollection` if the target boundary needs
semantic collection identity. Structural sharing avoids copies but does not
free tensors while old snapshots remain referenced. Transforms must not
silently evict unrelated fields or move a full snapshot. A returned collection
must not reference `ReadoutWorkspace` scratch. Caller-managed output banks, not
hidden source recycling, provide stable lifetimes for overlapping consumers.

Reconstruct after sample-axis selection only for a contiguous increasing
unit-stride range, advancing both sample-grid origin and containing-grid
offset. Arbitrary, reordered, or strided sample selections remain base
`TensorCollection` values; do not attach a false regular `SampleGrid`.

Do not impose the readout sample-last execution profile on every TensorDSLab
domain. A future Readout-to-Reconstruction bridge selects fields, validates
the exact `ChannelId` set, reorders by stable axis IDs/coordinates, explicitly
materializes reconstruction-preferred contiguous storage once, and constructs
a reconstruction-owned semantic value before entering its own execution
profile. Cross-channel reconstruction may prefer channel-last even though
temporal readout prefers sample-last.

Projects/dag owns campaign fanout and fanin, scheduling, retry, repair,
compiled DAG objects, scheduler-visible grouping, status, and cross-shard
orchestration. TensorDSLab may later expose DAG-compatible executables,
operation specs, and recipe fragments only after local product and cache
contracts are accepted. Local fixed-chain composition inside
`build_readout_collection(...)` is a TensorDSLab domain-builder responsibility,
not campaign orchestration.

For future caches, TensorDSLab owns deterministic storage-level compaction over
caller-supplied complete compatible products. Projects/dag owns scheduling,
fan-in, retries, repair, and campaign or cross-shard compaction orchestration.

## Validation Boundaries

TensorDSLab should move toward boundary-first validation:

```text
external/source/config/artifact boundary
  -> validate/coerce into strong typed objects
  -> construct TensorDSLab domain records and TensorCore layout records
  -> hot path trusts those records
```

Validate strongly when data crosses into TensorDSLab or TensorCore native
records and when constructing new typed product, cache, tensor, layout, or
selection objects. Do not repeatedly revalidate already constructed object
graphs inside hot loops unless a work order accepts that specific check.

Workspace preparation validates its exact layout, axis sizes/order, device,
role dtypes, algorithm scratch signature, and execution stream. Each lease then
performs narrow exact-match and exclusivity preflight before RNG consumption or
writes; mismatch fails rather than resizing or adapting the workspace.

Use constrained scalar wrappers for meaningful numeric config/source/artifact
values where constraints matter. Tensor-local positive counts should use
TensorCore-owned `PositiveInteger`. Numeric wrappers should reject bool. Do not
add generic bool wrappers by default.

Implementation should not move validation out of hot-path functions casually.
Boundary-first validation is an accepted direction, but each migration should
name the new construction or boundary checks that make downstream trust safe.

## Thread Roles

### Design

Design owns the target behavior.

Design should:

- define feature scope, invariants, accepted contracts, and non-goals;
- update design, decision, and architecture documentation when contracts
  change;
- produce implementation work orders;
- ratify, condition, revise, reject, or defer cross-package proposals;
- decide package governance adoption, conformance findings, deviations, and
  routing state;
- resolve package routing, conformance, architecture, and scope disputes;
- say what would require coming back to Design.

Design should not implement production code for the feature branch unless the
user explicitly delegates that exception. Design may directly edit
documentation during a documentation-only Design stage; those edits do not
require an Implementation handoff unless the user asks for that review loop.

### Implementation

Implementation owns the feature branch and is the default code-writing role.

Implementation should:

- make production code, test, and docs-sync changes required by the work order;
- keep the diff scoped to the work order;
- apply fixes requested by Validation and Review;
- keep the branch coherent and committed when asked;
- report commands run, known risks, and unresolved questions;
- stop and return to Design when a requested change would alter accepted
  architecture, ownership, scope, or non-goals.

Other threads should not modify production code or tests unless the user
explicitly delegates that exception. By default, they send findings or
suggested patches back to Implementation.

### Validation

Validation owns behavioral confidence.

Validation should:

- derive a test strategy from the design contract;
- identify missing edge cases and invariants;
- check whether tests prove behavior rather than mirror implementation details;
- exercise external integration paths when those paths are part of the behavior
  contract;
- interpret test failures;
- send concrete test gaps or suggested test cases to Implementation;
- dispatch the fixed branch and commit to Review when Validation clears.

Validation should not broaden scope or reopen architecture. By default,
Validation does not edit the feature branch; Implementation applies any test
changes. If a Validation finding would require changing the accepted stage
architecture or scope, Validation should send the issue back to Design rather
than asking Implementation to widen the branch.

### Review

Review owns final independent critique.

Review should:

- review the final or near-final diff for correctness, maintainability, typing,
  API fit, and scope control;
- verify external compatibility gates named by the work order;
- report findings first, ordered by severity;
- cite exact file and line references where possible;
- distinguish blockers from follow-up polish;
- send findings back to Implementation for fixes unless resolving them would
  require changing the accepted architecture or stage scope;
- issue explicit clearance on the fixed commit or identify the remaining
  blockers.

Review is read-only by default. It should not rewrite the branch unless the
user explicitly asks it to. If a Review finding requires an architecture or
scope change, Review should route it to Design instead of asking Implementation
to patch around the work order.

## Production Work Order Handoff

This section applies when Design dispatches implementation or another
state-changing stage. A documentation-only Design pass may remain in the Design
thread while it is being discussed.

Design should dispatch production work only after the source-of-truth work
order is committed and the base branch is clean, unless the user explicitly
accepts an exploratory exception. A dispatch must satisfy the complete
work-order checklist below and use Active, verified execution routes.

A Design work order should include:

- a stable package-owned work-order key and task; by default, the committed
  `docs/implementation/stage_<number>_<slug>.md` path is the key;
- exact Design and document baseline;
- base branch or commit and target branch;
- target files, packages, and public surfaces;
- source-of-truth docs to keep synchronized;
- invariants and validation rules;
- donor reference, comparison boundary, parity classification, and intentional
  divergences when donor behavior is in scope;
- scope and non-goals;
- minimum tests and verification commands;
- verified persistent Implementation, Validation, and Review routes;
- a finite Implementation/Validation loop budget;
- package-owned work-order state vocabulary and its source;
- known risks or open questions;
- stale-routing, architecture, scope, and other escalation or stop conditions;
- Review and clean-closeout expectations;
- what requires coming back to Design.

The strongest work orders include concrete code or test sketches when code or
tests are in scope. Avoid vague requests such as "add coverage"; say which
module, test name, public imports, helper boundaries, assertions, and forbidden
shortcuts matter. If a stage is docs-only, audit-only, or test-only, say that
clearly and repeat that production behavior changes require Design escalation.

When a work order or architecture doc names a public surface, it should include
a concrete sketch of that surface unless the surface is explicitly deferred.
Sketch dataclasses, functions, modules, validation helpers, and expected tests
with enough detail that Implementation can execute without inventing the
contract and Review can compare the diff against a specific target.

## Production Implementation And Validation Loop

After Design sends a production-code work order, Implementation and Validation
may iterate until the branch is stable:

```text
Implementation builds -> Validation tests/critiques -> Implementation fixes
```

Implementation and Validation may message each other automatically only when
the work order explicitly authorizes the loop, provides Active and verified
logical routes, and defines the finite budget. Raw platform route identifiers
remain private routing attributes and are not work-order identity. This loop is
bounded:

- maximum three Implementation-to-Validation dispatches;
- maximum three Validation-to-Implementation dispatches;
- each message must be specific and actionable;
- no architecture changes or scope expansion;
- no branch ownership changes;
- stop early when Validation reports no blocking findings;
- stop if a required route becomes stale, Deferred, missing, or discrepant;
- stop and ask the user or Design if the same issue repeats twice, the loop
  budget is exhausted, or a Design decision is needed.

Expected message shapes:

```text
Implementation -> Validation:
branch/commit, scope, files changed, docs updated/checked, commands run,
invariants to attack

Validation -> Implementation:
severity-ordered findings, missing tests, edge cases, suggested test cases

Implementation -> Validation:
fixes made, new commit, verification run, deferred items

Validation -> Review:
fixed branch/commit, validation scope, commands run, residual risks,
cleared for Review

Validation -> Implementation:
remaining blockers, or architecture/scope issue routed to Design
```

Ready for Review means:

- scoped behavior is implemented;
- Validation findings are resolved or explicitly deferred;
- tests requested by Validation pass;
- applicable design, decision, implementation, validation, review, and
  architecture docs are synchronized with any changed public contracts;
- generated files, caches, and unrelated outputs are not staged;
- the branch is committed when the work order asks for a commit;
- the handoff names a fixed branch and commit rather than a moving target,
  unless the user accepted an exploratory exception and the handoff reports
  that risk;
- the handoff lists commands run and remaining risks.

When Validation clears a fixed branch, Validation should dispatch that branch
and commit to Review. If Validation does not clear, it should send actionable
findings back to Implementation. If those findings would require changing the
accepted architecture, non-goals, or stage scope, Validation should stop and
ask Design instead.

## Production Review Gate

Send a production branch to Review only after the implementation/validation
loop is quiet. Review should not be asked to review a moving target unless the
request is explicitly an early design or architecture review.
Documentation-only Design changes do not require this gate unless the user
asks for an independent review.

While reviewing, Review remains read-only and should send findings back to
Implementation rather than rewriting the branch. If a Review finding would
require changing accepted architecture, non-goals, or stage scope, Review
should route it to Design instead of Implementation. After Implementation
fixes Review findings, Review should recheck the fixed branch and commit.

After Review reports no remaining findings on a fixed branch, Review owns the
closeout merge:

- fast-forward merge the cleared branch into the work order's target base
  branch, normally `main`;
- run the post-merge verification commands named by the work order;
- report the resulting base branch HEAD, commands run, and any residual risk;
- ask Design to open discussion of the next stage or direction.

If the merge is not a clean fast-forward, verification fails, the worktree is
dirty, or the target branch is ambiguous, Review should stop and report the
blocker instead of resolving conflicts, rewriting history, force-pushing, or
changing implementation code.

## Documentation Synchronization Gate

Implementation, Validation, and Review should treat documentation drift as a
real review item. Every stage should leave the source-of-truth docs aligned
with the implemented contract.

Before Review, check whether the change requires updates to:

- `docs/implementation/stage_*.md` when a stage work order, scope, handoff, or
  accepted implementation surface changed;
- `docs/architecture/common.md`, when present, for shared helpers, cache
  schema, durable representation shape, manifest behavior, validation
  behavior, or cross-domain contracts;
- `docs/architecture/<domain>.md` when a domain public contract, product shape,
  builder/loader contract, validation rule, or representation bridge changed;
- `docs/architecture/tensors.md` when TensorCore integration, semantic product,
  layout, field, placement, output, workspace, allocation, or lifetime
  contracts changed;
- `docs/parity.md` when donor references, comparison boundaries, fixtures,
  tolerances, RNG comparisons, or intentional-divergence claims changed;
- `docs/design.md` when end-to-end domain flow or ownership boundaries changed;
- `docs/decisions.md` when a semantic choice was accepted, renamed,
  superseded, or explicitly deferred;
- `docs/validation.md` for expected behavior, validation cases, fixtures,
  failure modes, or numeric tolerances changed;
- `README.md`, `AGENTS.md`, or `CONTRIBUTING.md` when workflow, onboarding, or
  repository-wide expectations changed.

Before Review, the Implementation handoff must identify documentation updated,
documentation checked but unchanged with a reason, verification commands run,
residual risks, and intentionally deferred items.

Update `docs/governance/` when package adoption state, conformance evidence,
semantic rule mappings, deviations, routing posture, Coordination status, or
the adopted Governance Core version changes. Governance records must
distinguish a proposed decision from an issued package decision.

Implementation handoffs should explicitly say which docs were updated, or why
no docs update was needed. Validation and Review should run targeted stale-name
searches when a public term is renamed. Keep legitimate historical mentions
only when they are clearly framed as historical, deferred, or superseded.

## Verification Baseline

For documentation-only Design changes, run at minimum:

```bash
git diff --check
```

Also run targeted link, heading, and stale-term checks appropriate to the
change.

Before production Review, run the smallest relevant verification set for the
change. At minimum, run:

```bash
git diff --check
```

For Python changes after the package exists, also run the relevant test suite:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

For future DAG-compatible changes, also validate the DAG repo-facing operation
specs with the repository's accepted command.

If a repository requires a specific environment, use that environment and
report the exact command.
