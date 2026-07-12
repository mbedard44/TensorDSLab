# Contributing

TensorDSLab should be developed like professional scientific software: clear
ownership boundaries, typed public APIs, deterministic behavior, focused tests,
small coherent changes, and documentation that explains contracts rather than
narrating code.

## Governance And Delivery Maturity

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`, bound to
exact candidate `d634401a853915edeb4f83df4a4943b3553deced` and recorded in the
[package governance records](docs/governance/index.md). Conformance remains
`Not evaluated`, Coordination remains `Deferred`, and Profile B remains
`Disabled`.

The current identity and maturity are:

```text
Project/display name: TensorDSLab
Python import: tensor_dslab (accepted on main through Stage 2)
Distribution name: tensor-dslab (accepted metadata; not published or released)
Delivery maturity: active development / pre-deployment
Package maturity: Stage 2 structural foundation Merged / Closed
```

Stage 1 is Design-complete, and Stage 2 is Merged / Closed on `main` at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. Main now contains accepted package
metadata, the structural production package, and its focused test suite. The
post-merge evidence is limited to the exact repository/dependency commits,
Python and PyTorch versions, CPU execution mode, and conditional CUDA skips
recorded in the Stage 2 work order. No wheel or published artifact was built,
and TensorDSLab makes no deployability, release-readiness,
backward-compatibility, or broad cross-package compatibility claim.

Maintenance 1 was separately dispatched to correct only readout public-name
and module ownership. Before Review's clean fast-forward, its feature-branch
form is candidate evidence; if the updated surface is read on `main`, that
merge gate has completed. It changes no collection behavior, TensorCore pin,
scientific contract, deployment state, or compatibility finding.

The `tensor-dslab` distribution spelling is accepted package metadata, not an
installed, published, or released distribution claim. GPU residency
and no-silent-host-materialization requirements are TensorDSLab Design
constraints for future boundaries, not evidence that any TensorG4DS,
TensorCore, or TensorML baseline is compatible. A breaking change affecting
multiple repositories requires every affected package Design authority and a
synchronized migration plan. Compatibility shims, aliases, or deprecation
windows require demonstrated value and explicit Design acceptance.

## Repository Identity

TensorDSLab is a clean-slate, tensor-native detector data-lab package in this
intended data flow:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared tensor substrate rather than a pipeline stage. The
diagram is not an import graph and does not claim that the still-deferred
TensorG4DS-to-TensorDSLab or TensorDSLab-to-TensorML boundaries already exist.

TensorDSLab owns post-TensorG4DS detector/readout semantics and future cache
contracts:

- mapping an accepted future TensorG4DS public product into TensorDSLab
  provenance, coordinates, detector-window/readout-grid semantics, and binned
  photon-origin photoelectrons when a focused integration stage accepts that
  bridge;
- building the typed `ReadoutCollection`, its recognized product fields, and
  optional thin readout provenance/context records;
- defining future reconstruction examples and reconstruction products;
- defining TensorCore-backed semantic collections, fields, and domain
  transforms where a stage accepts a tensor-native contract;
- writing, validating, loading, and performing deterministic storage-level
  compaction of strict durable caches only after in-memory product contracts
  are stable;
- exposing DAG-compatible executables, operation specs, or recipe fragments
  only after local product and cache contracts are stable.

TensorDSLab does not own native G4DS parsing or simulation execution,
TensorG4DS low-level products or algorithms such as deposit clustering,
generic TensorCore primitives, downstream source adaptation for model
training, model assembly, training loops, evaluation loops, metrics,
checkpoints, campaign orchestration, scheduler behavior, repair, or retries.

Projects/dag owns scheduling and fan-in for future cache compaction, including
campaign/cross-shard grouping, retries, repair, and execution policy.
TensorDSLab owns only the deterministic storage primitive over caller-supplied
complete compatible products.

The production integration target keeps tensor payloads resident on one
explicit accelerator device across TensorG4DS, TensorDSLab, and TensorML.
Boundary code must not silently call `.cpu()`, `.numpy()`, serialize/reload, or
otherwise materialize payload data on the host as a package handoff. A
TensorG4DS-to-TensorDSLab bridge is a semantic transformation and may create
new tensors on that device; same-device residency is not a claim that the
input and output layouts are interchangeable or that every transform is
zero-copy. TensorCore IDs, layouts, and small semantic records may remain
host-side metadata. Device movement is always explicit.

The first discrete TensorG4DS bridge carries no end-to-end autograd promise.
It must not detach silently and should reject gradient-sensitive inputs unless
a focused differentiable detector contract is accepted. Existing functional
autograd guarantees for accepted deterministic waveform transforms remain
unchanged.

Historical predecessor code, if consulted outside this repository, is
parts-bin material only. Promote scientific facts, product semantics, cache
guarantees, algorithms, fixtures, and tests deliberately into TensorDSLab docs
and tests. Do not preserve old package layouts, helper frameworks,
compatibility shims, DAG wiring, or representation shortcuts by default.
Use `docs/parity.md` to classify every donor comparison and intentional
divergence; promoting a donor idea does not itself establish parity.

## Build Philosophy

Define the MVP early, but build toward it from the inside out. The first
accepted MVP direction is the post-binned tensor-native readout path:
already-binned photon-origin primary photoelectrons, aggregate SiPM charge
simulation, waveform products, analog waveform composition, and optional
digitization.

Native G4DS parsing belongs upstream of TensorDSLab. The typed TensorG4DS
handoff, detector-window construction, photoelectron binning, IO boundaries,
durable cache formats, table/array codecs, manifest rules,
compaction, package-local CLIs, DAG-compatible operation specs, recipes,
executable doors, and downstream adapter contracts should not shape the first
post-binned readout module boundaries.

Early implementation stages should be judged by whether the local field
dependency graph and collection contract are typed, deterministic, testable,
and easy to reason about. Compatibility with external orchestration or
downstream training packages is deferred until the local TensorDSLab contracts
are stable.

Scientific configs should describe physics and readout behavior. Device, dtype,
movement, RNG stream selection, output destination, `ReadoutWorkspace`,
accelerator stream, and execution/chunking policy are runtime controls unless a
focused Design stage accepts a different boundary. Do not hide runtime
placement, allocation, workspace, or mutation policy inside scientific config
records.

## Sibling Repository Shape

TensorDSLab should feel like a sibling of TensorML and TensorCore in engineering
style: explicit boundaries, staged implementation, typed records, small APIs,
clear docs, and disciplined review gates. Use the shared style, but do not copy
another repository's domain boundaries blindly.

The tree below is a menu of accepted surfaces, not a requirement to create
empty files:

```text
TensorDSLab/
  AGENTS.md
  CONTRIBUTING.md
  README.md
  docs/
    overview.md
    design.md
    decisions.md
    parity.md
    validation.md
    architecture/
      common.md
      detector.md
      readout.md
      reconstruction.md
      caches.md
      tensors.md
    implementation/
      index.md
      stage_<n>_<name>.md
  pyproject.toml              # when package metadata is accepted
  tensor_dslab/               # when production package code is accepted
    common/                   # shared IDs, quantities, validation only when real
    detector/                 # optional post-TensorG4DS semantics, only when real
    readout/                  # collection, atomic transforms, workspace, builder
    reconstruction/           # future reconstruction products
    caches/                   # durable/cache/load/write bridge, if not local
    executables/              # future DAG/task adapters, only when accepted
    operations/               # future DAG operation specs, only when accepted
    recipes/                  # future reusable composition fragments, only when accepted
  tests/
```

The project/display folder is `TensorDSLab`; the Python import package is
`tensor_dslab`. Do not create a flat TitleCase Python package that imports
as `TensorDSLab`; keep semantic subpackages directly below the import root.

Do not create placeholder modules to reserve architecture. Add a module only
when there is a real concept, behavior, or contract to house.

## TensorCore Backbone

The current TensorCore Design snapshot is `0.6.0`, as recorded in
[TensorCore Integration Architecture](docs/architecture/tensors.md#design-baseline).
Stage 2 declared and verified exact TensorCore commit
`dc554994061183776f23f65860a0594516074f2e` with Python `3.13.11`, PyTorch
`2.12.1`, and CPU execution. The three conditional CUDA tests were skipped
because CUDA was unavailable. Neither the accepted dependency pin nor this
passing exact tuple is broad cross-package compatibility evidence.

TensorDSLab should use TensorCore for generic tensor mechanics, not fork or
mirror them. Code that needs generic tensor identity, layout, field,
collection, selection, batching, movement, validation, or pure operation
helpers should import those surfaces from `tensor_core`.

TensorCore owns:

- `Id`, `TensorAxisId`, and `TensorFieldId`;
- `IdSequence` for ordered same-type ID values;
- generic constrained scalar wrappers such as `PositiveInteger`,
  `NonnegativeInteger`, `FiniteFloat`, `PositiveFloat`, `NonnegativeFloat`, and
  `Probability`;
- `TensorAxis`, `TensorAxes`, `TensorLayout`, `TensorField`, and
  `TensorCollection`;
- tensor selection records such as `TensorFieldSelection` and
  `TensorAxisSelection`;
- generic tensor builders, validators, immutable mapping helpers such as
  `freeze_mapping` and `freeze_metadata`, batching helpers, movement helpers,
  and pure tensor operations.

Import TensorCore public names from the root `tensor_core` package. Do not
import TensorCore implementation modules and do not re-export generic
TensorCore helpers through `tensor_dslab.common`.

`Id` and `TensorCollection` are TensorCore's intentional downstream extension
points. TensorDSLab may subclass `Id` for coordinates and defines the primary
readout tensor record as one direct `ReadoutCollection(TensorCollection)`
subclass. Do not subclass sealed TensorCore primitives:
axis/field IDs, constrained scalars, `IdSequence`, axes, layouts,
`TensorField`, selections, or `BatchConfig`.

TensorDSLab owns domain IDs and accepted records such as row identity,
post-TensorG4DS provenance, any future focused detector-response grouping,
readout examples, reconstruction examples, product labels, domain configs,
semantic tensor products, and domain-specific transform rules.
Future cache manifests belong here only after the in-memory product model is
accepted. TensorDSLab domain IDs may appear as TensorCore coordinates, but they
should not become TensorCore-owned primitives.

TensorCore is the dense tensor spine. TensorDSLab should give TensorCore
records detector/readout product meaning instead of competing with the generic
tensor substrate. Scripts and runtime builders may choose the concrete
TensorCore layout shape and dimension order, but TensorDSLab must make product
roles, field roles, required readout-axis identities, sample-grid facts, and
stochastic coordinate inputs explicit.

The first post-binned products are recognized fields inside one
`ReadoutCollection`, with a typed `SampleGrid`, a conditional typed
`DigitizedWaveformSpec`, and free transform functions. A future
`ReadoutExample` may wrap the collection with provenance or context, but it is
not the primary tensor handoff. Do not add caller-defined semantic axis-role
mappings, one single-field collection subclass per product, a generic
`Product` base, semantic `TensorField` subclasses, or a ToyProduct-like wrapper
hierarchy.
`ReadoutCollection` and the other stable readout value types live in
`tensor_dslab.readout.types`; `tensor_dslab.readout.tensors` retains only the
readout-semantic reconstruction, projection, selection, and movement helpers.

`tensor_dslab.common` owns and exports the shared stable coordinate types
`ExampleId` and `ChannelId`. `tensor_dslab.readout` owns and exports the
readout-specific axis and field constants, including these exact required axis
identities:

```python
EXAMPLE_AXIS_ID = TensorAxisId("example")
CHANNEL_AXIS_ID = TensorAxisId("channel")
SAMPLE_AXIS_ID = TensorAxisId("sample")
REQUIRED_AXIS_IDS = IdSequence(
    (EXAMPLE_AXIS_ID, CHANNEL_AXIS_ID, SAMPLE_AXIS_ID)
)
```

A `ReadoutCollection` is a nonempty, structurally immutable, partially
materialized snapshot. Transforms treat retained tensor payloads as read-only.
Callers must not mutate materialized field tensors in place; doing so bypasses
descendant invalidation. Only the fresh target of an atomic `out=` call, or the
prepared generated-field set held exclusively by
`build_readout_collection(...)`, may be written. Mutable private scratch
belongs to a caller-owned runtime `ReadoutWorkspace`, never to a collection
field or sidecar. Present fields must be an ordered subsequence of this
canonical schema:

```text
readout.photoelectrons
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.analog
readout.waveform.digitized
```

Every present field uses the exact same ordered layout and device, and every
tensor uses `torch.strided` layout; noncontiguous strided tensors remain valid
collection structure. Example, channel, and sample are
shared axes and may occur in any layout order. Locate them by
`TensorAxisId` value equality and `TensorAxes.index(...)`, never by fixed
dimension or object identity. Example is ID-backed by exact `ExampleId`
coordinates, channel by exact `ChannelId` coordinates, and sample is
count-only. Photoelectrons use exactly `torch.int64`; present charge, pure,
noise, and analog fields use one exact common `torch.float32` or
`torch.float64` dtype; and digitized ADC counts use exactly `torch.int32`. Any
accepted additional axes occur in every field and are
declared shared as well. `shared_axes` lists all common layout axis IDs in
layout order. TensorDSLab constructors enforce these stronger rules around the
generic TensorCore record. `SampleGrid` remains the typed source of regular
sample-grid facts; do not add `ReadoutAxisRoles`.

Collections define scientific meaning; execution profiles define memory
arrangement. General `ReadoutCollection` construction continues to accept
arbitrary axis order and noncontiguous `torch.strided` read-only values. It
does not reject expanded or internally overlapping source storage solely on
that basis. TensorLayout remains the only semantic axis-order authority; do
not add an execution-ready subclass, axis-order/stride sidecar, or runtime
storage policy to collection identity.

The warmed `out + workspace` MVP readout profile is deliberately narrower:
the sample axis is last, every participating source/generated output/scratch
tensor is contiguous, writable storage is internally nonoverlapping and
disjoint, and the complete ordered shape/device/dtype/algorithm/destination/
stream/lease signature matches. A different leading-axis order uses a
different workspace. Preflight rejects instead of permuting, making
contiguous, cloning, casting, moving, reshape-copying, or allocating a fallback
inside the warmed call. Contiguous strides derive from ordered shape, so the
MVP workspace signature stores no arbitrary stride tuple.

Require a typed `DigitizedWaveformSpec` exactly when the digitized field is
present. It retains bit depth from 1 through 16, voltage transfer, inclusive
`[0, 40]` dB analog gain, and `AdcQuantization.TRUNCATE`; digitized-field
projection retains it, and invalidation/removal drops it. Do not leave ADC
interpretation only in an ephemeral transform config.

### Coordinates, Indices, And Layouts

TensorCore terminology is strict:

- a coordinate is a stable `Id` value associated with an ID-backed axis;
- an index is a zero-based integer tensor position along an axis;
- a layout is ordered axes plus coordinate-to-index maps for ID-backed axes.

Coordinates and indices are not interchangeable. Do not use IDs as array or
tensor positions, and do not persist transient tensor, table, or array indices
as durable identity. Diagnostics, artifacts, and reports should keep reporting
semantic IDs when an axis is ID-backed.

Axis records describe the dimensions of a compiled layout. A `*Axis` describes
one dimension; a `*Axes` record describes the ordered collection of axes that
define the full tensor/layout shape. All tensor dimensions should be explicit
in an axes object, including numeric/bin dimensions such as time samples.

ID-backed axes use ordered ID sequences. IDs identify points in the abstract
space defined by their axis; IDs should not encode ranges, bin edges, units, or
physical interpretation. Normalize external quantities in TensorDSLab boundary
configs or builders before constructing IDs.

`IdSequence` preserves caller order and rejects empty sequences, duplicate IDs,
base `Id` values, and mixed concrete ID classes. TensorDSLab builders must not
sort or infer ID order. A domain axis builder should validate the exact
TensorDSLab coordinate class expected for that axis.

`TensorAxes` is the ordering source of truth. `TensorLayout` owns those ordered
axes and carries coordinate maps only for ID-backed axes. Count-only axes do
not need layout map entries because integer positions are already the native
position. A count-only axis is strictly zero-based and continuous.

For later coordinate-addressed stochastic transforms, logical coordinate
identity is independent of tensor dimension order and uses this sequence:

1. required example-axis ID and `ExampleId` coordinate;
2. required channel-axis ID and `ChannelId` coordinate;
3. every other ID-backed shared axis, ordered lexically by `axis_id.value`,
   paired with its coordinate;
4. `SampleGrid.sample_offset + local_sample_index` for the sample axis;
5. an operation-local draw/counter coordinate when needed.

The seed, caller namespace, and operation role precede that coordinate payload.
Adding or removing an ID-backed extra axis deliberately changes stochastic
identity. Extra count-only axes are structurally valid, but a later stochastic
transform must reject one without an accepted stable global-offset rule.

Use TensorCore record vocabulary consistently where applicable:

- `id` for one stable identity;
- `ids` for an ordered identity collection;
- `axis_id` for a TensorCore axis identity;
- `indices` only for zero-based tensor positions;
- `tensor`, `layout`, and `metadata` for their TensorCore meanings.

Avoid duplicate generic names such as `tensor_id`, `tensor_map`, or `payload`
when the TensorCore record already has a precise name. This rule does not
collapse TensorDSLab durable product labels into field IDs.

## Product Semantics

TensorDSLab should preserve the data-flow and ownership chain unless Design
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

This sequence is a product-flow and ownership rule, not a Python import graph
or scheduling policy. Projects/dag may
fan out, retry, cache, stream, compact, or parallelize work, but local readout
builders should not hide the upstream semantic bridge or downstream model
adaptation. As package dependencies, core `tensor_dslab.common` and
`tensor_dslab.readout` remain TensorCore-only. A future downstream-owned bridge
may import an exact accepted public TensorG4DS type; TensorG4DS must not import
TensorDSLab.

The primary readout tensor boundary is `ReadoutCollection`, not a loose product
tuple or required dataclass adapter. `tensor_dslab.common` owns and exports
`ExampleId` and `ChannelId` as shared opaque stable coordinate types. Source
event IDs, including TensorG4DS `EventId` and native G4DS event values, are
upstream provenance; a later bridge stage must define explicit
provenance-to-example and channel-coordinate mapping rather than equating
identities or using transient indices implicitly.

Producer product labels are durable TensorDSLab product labels. Runtime fields
use exact TensorCore `TensorFieldId` values. Corresponding labels and fields may
use the same string, but keep their namespaces and types explicit:

- `readout.photoelectrons`;
- `readout.charge`;
- `readout.waveform.pure`;
- `readout.waveform.noise`;
- `readout.waveform.analog`;
- `readout.waveform.digitized`;
- future reconstruction labels.

Consumer-facing adapters are deferred. TensorDSLab should first make the local
typed product graph coherent enough that future consumers can depend on it
without parsing raw `.fil`, table, array, manifest, or private representation
details.

## Domain Organization

Domain code should communicate through typed in-memory objects. Persistence,
caches, artifacts, tables, arrays, tensors, executables, operations, and
recipes are bridges around the domain model, not replacements for it.

Use these module names when they fit real behavior:

- `types.py` owns stable public records, type aliases, and domain value
  objects.
- `configs.py` owns public configuration records when configuration is
  nontrivial enough to deserve a boundary.
- `builders.py` owns in-memory construction when construction is meaningfully
  separate from loading, writing, or representation conversion, including the
  local fixed-chain `build_readout_collection(...)` domain builder.
- `validation.py` owns domain invariants, validation reports, issue codes, and
  validation errors.
- `artifacts.py` owns the durable/cache/load/write bridge for the domain.
- `tables.py` owns table schemas, row conversion, table conversion, and
  table-level parsing helpers when table representation exists.
- `arrays.py` owns array conversion, axes, shapes, and payload rendering when
  array representation exists.
- `tensors.py` owns focused TensorCore construction/reconstruction helpers when
  those mechanics are substantial enough to separate from semantic types and
  transforms.
- A focused execution module may own `ReadoutWorkspace`, workspace preparation,
  and exact-match lease validation when those runtime mechanics become real;
  workspace state does not belong in `types.py` product records or configs.
- `exports.py` may be added when a project has a real external export/catalog
  surface, such as DAG operation discovery, cache publication, or stable
  adapter metadata.

Do not use `exports.py` as a dumping ground for ordinary package re-exports.
Public imports still belong in deliberate package `__init__.py` surfaces.

`artifacts.py` should be readable as the domain bridge contract. If it becomes
mostly schema literals, row codecs, array plumbing, tensor plumbing, or backend
mechanics, split those mechanics into `tables.py`, `arrays.py`, or
`tensors.py` as appropriate.

Do not create placeholder `configs.py`, `builders.py`, `artifacts.py`,
`tables.py`, `arrays.py`, `tensors.py`, or `exports.py` modules. Split or add a
module only when the behavior is real enough to make the boundary useful.

## Domain Transform Surfaces

The first TensorDSLab readout transforms are free functions. Post-binned
operations such as timing jitter, aggregate charge simulation, waveform
rendering, analog waveform composition, and digitization are product
semantics, not generic TensorCore operations.

`readout.photoelectrons` represents binned photon-origin primary PE seeds.
Only timing jitter replaces it. One public `simulate_charge` transform consumes
that field; internally it adds dark-count avalanches, builds crosstalk and
afterpulse contributions from one frozen source snapshot, and applies aggregate
charge smearing. Intermediate avalanche-count tensors and low-level
contribution samplers stay private unless Design accepts a public product
contract.

The resulting `readout.charge` field is a finite floating aggregate
PE-equivalent response per readout channel and sample. It is not an SI-coulomb
quantity and does not claim an explicit individual-SPAD output. Pure and noise
waveforms are signal-only and noise-only components at a shared analog
reference plane, not sequential hardware products. Their composition produces
`readout.waveform.analog`; digitization converts that analog representation to
ADC counts.

Readout execution has three layers: atomic free transforms, caller-reusable
runtime-only `ReadoutWorkspace` scratch, and the higher-level local
`build_readout_collection(...)` domain builder. Low-level transforms remain
public and independently callable. The builder owns the fixed configured full
chain and its automatic logical scratch schedule; it does not load sources,
perform IO, move/cast inputs, or own Projects/dag scheduling.

Domain transforms consume and return coherent `ReadoutCollection` snapshots:

```python
updated = apply_timing_jitter(readout, jitter, rng=rng)
updated = simulate_charge(updated, charge, rng=rng)
updated = render_pure_waveform(updated, pulse)

# Stage 2 owns the exact buffer-factory spelling and keyword-only role inputs.
destination = build_readout_result_buffer(
    updated,
    target_field_id=READOUT_NOISE_WAVEFORM_FIELD_ID,
    target_dtype=updated.tensor(READOUT_CHARGE_FIELD_ID).dtype,
)
result = render_noise_waveform(updated, noise, rng=rng, out=destination)
```

With `out=None`, an atomic transform is functional and allocating, does not
mutate inputs, may explicitly normalize arbitrary semantic order/strides, and
preserves autograd for accepted differentiable deterministic behavior. With
`out=`, it writes the complete target field of the exact supplied valid
destination and returns that destination. A workspace is valid only with
`out`; supplying `out` selects a non-autograd simulation path. Without a
workspace, accepted normalization and scratch may allocate and carry no
allocation-free claim. All destination, workspace, lease, device, dtype,
algorithm, and stream preflight completes
before RNG is consumed or writes launch. Preflight failure leaves RNG and
tensors untouched; there is no transactional rollback guarantee after a
backend operation launches.

Each transform adds or replaces one recognized field, removes transitive
descendants that could disagree with the new dependency value, retains
unaffected fields by structural sharing, and returns the resulting canonical
collection. Projection or removal is not replacement: a projected collection
may retain any selected descendant without also retaining its dependencies.
These dependency and canonical-order rules belong in shared TensorDSLab
helpers, not ad hoc field-map edits.

Use this dependency graph for invalidation:

```text
readout.photoelectrons -> readout.charge -> readout.waveform.pure
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog -> readout.waveform.digitized
```

Replacing photoelectrons does not invalidate noise when the common layout and
sample-grid contract is unchanged. Replacing collection-level layout or grid
semantics requires rebuilding every retained field rather than claiming the
old snapshot remains coherent.

`out` must have the exact expected result field set and order, collection
sidecars, layout, device, per-field dtypes, and `torch.strided` layout.
Its unaffected fields are the exact retained source `TensorField` records. Its
target tensor is internally nonoverlapping and does not alias any source,
other live output, or workspace scratch. A field-scoped
result-buffer builder zero-initializes the target so every public collection is
valid before execution and allocates every new target contiguously in the
collection's existing semantic axis order without normalizing retained fields.
Private workspace scratch alone may use uninitialized
storage, and then only under a complete write-before-read proof. The transform
overwrites its target completely.

`ReadoutWorkspace` is caller-owned scratch-only runtime state. It is never a
field, sidecar, scientific config, ID, product, cache entry, or returned-storage
owner. It matches exactly one fixed ordered sequence of layout axis IDs and
sizes, device, role dtypes, algorithm scratch signature, destination schema,
and CPU/default or CUDA stream. Warmed use additionally requires sample-last
order and contiguous participating tensors. Required-axis positions are
derived from the ordered IDs by equality/index; no semantic-role sidecar or
arbitrary stride tuple belongs in the signature. It does not
cache, resize, move, cast, or recycle source tensors. One nonreentrant exclusive
same-stream lease is allowed at a time. Returned collections never reference
workspace storage. Logical private charge roles include source, total, and
contribution storage; do not assert that two physical buffers always suffice
merely because some lifetimes may eventually be coalesced.

`build_readout_collection(...)` produces photoelectrons, charge, pure, noise,
analog, and optional digitized fields. If timing jitter is disabled, the source
photoelectron field may be structurally shared. With `out=None` and no
workspace, the builder uses the functional allocating path. A workspace
without `out` is invalid; supplying `out` selects non-autograd simulation
execution. The warmed steady-state hot path requires exact caller-prepared
`out` plus workspace and performs no
TensorDSLab-managed tensor-storage allocation. This does not promise zero
Python-object, PyTorch-internal, or vendor-library allocation.

Count-domain ping-pong uses contiguous A/B buffers from one exact compatible
storage class and swaps references without permuting or copying. Count,
floating, and ADC domains require separate storage. If a source is not already
sample-last and contiguous, explicitly prepare/materialize it once outside the
repeated loop; do not hide that transition in warmed execution.

The builder preflights its entire configured chain, output, workspace, lease,
device, dtypes, algorithm signatures, and stream before the first atomic RNG
draw or write. It must not partially execute earlier stages and discover a
later-stage compatibility error afterward.

Caller-owned output storage remains stable until the caller submits that same
destination again. Resubmission authorizes complete overwrite, so overlapping
consumers require caller-managed output banks. Read-only structural sharing of
retained fields is allowed; a writable target never aliases a source or any
workspace scratch. Any storage sharing among private scratch roles requires an
explicit nonoverlapping-lifetime proof.

TensorCore selection, batching, movement, and like-allocation return base
`TensorCollection` records. TensorDSLab should reconstruct and validate a
`ReadoutCollection` at its domain boundary instead of copying generic
mechanics. For TensorML, explicit `TensorFieldSelection` order is the
positional model schema; collection class alone does not distinguish partial
field subsets and is not preserved by the stock selection path.
Noncanonical model selections remain base `TensorCollection` values; do not
reorder positional arguments merely to reconstruct `ReadoutCollection`.

Because base results do not retain subclass sidecar attributes, a TensorDSLab
helper must explicitly carry, update, or prune `SampleGrid` and the conditional
`DigitizedWaveformSpec` around the generic operation. Required axis meaning is
recovered from the exact exported axis IDs, not a role sidecar or free-form
metadata.

Project the required field subset before accelerator movement. Move only the
projected base collection through TensorCore, then reconstruct a semantic
collection when the target boundary needs one. Structural sharing avoids
payload copies but does not release memory while an older snapshot remains
referenced. Field eviction and placement are explicit runtime policy, never a
scientific config or hidden transform behavior.

Only contiguous increasing unit-stride sample selection can reconstruct a
`ReadoutCollection` under the current regular `SampleGrid`; advance origin and
sample offset together. Leave arbitrary, reordered, or strided selections as
base collections rather than assigning them false regular-grid semantics.

Do not make sample-last a universal TensorDSLab order. A future
Readout-to-Reconstruction bridge selects by stable field/axis IDs, validates
the complete `ChannelId` set, explicitly reorders/materializes once, constructs
a reconstruction-owned value, and then enters that domain's execution profile.
Cross-channel reconstruction may prefer channel-last.

Do not use a persistent ambient mutation mode or hidden workspace cache for
core transforms. `ReadoutWorkspace` is explicit caller-owned runtime state, not
ambient state. Do not encode mutation, allocation, workspace, lease, or stream
policy as a TensorCore `Id`, `TensorFieldId`, `TensorAxisId`, coordinate, product
label, or scientific config.

## Common Code

`common/` should stay dependency-light and semantic. Good candidates include
shared IDs, small value objects, shared exceptions, and validation primitives
that are used by multiple real domains.

Do not put representation dependencies in top-level `common/` merely because
multiple domains use tables, arrays, tensors, Parquet, NPZ, JSON, or another
format. Prefer domain-local `tables.py`, `arrays.py`, or `tensors.py`, or a
cache/artifact-local helper, until a shared semantic abstraction proves itself.

Avoid expanding `common/` because two modules happen to look similar. Wait
until the concept is actually shared.

Use small semantic quantity wrappers for stable public records when a scalar
field needs finite signed, strictly positive, or nonnegative numeric
semantics. Place wrappers domain-local when only one domain needs them; promote
them to `common/` only when multiple real domains share the same quantity
vocabulary or a common public record intentionally uses them.

## Public Surface Discipline

Package `__init__.py` files should re-export documented public surfaces
deliberately. They should not expose private representation, persistence,
normalization, or validation-helper functions by accident.

Public names should be stable and intention-revealing. When public names move
or change:

- update the relevant docs in the same stage;
- run targeted stale-name searches;
- keep historical mentions only when clearly framed as historical,
  superseded, or deferred;
- do not add compatibility wrappers or aliases unless Design accepts a
  compatibility window.

Downstream code should consume typed upstream objects. It should not parse
another domain's raw persisted files directly. If a cache or artifact is the
bridge, the owning domain must provide the loader that reconstructs typed
objects.

### Public Verb Vocabulary

Use consistent verbs for public and semipublic module-level APIs:

- `build_*` constructs an in-memory domain object or representation from
  already-available inputs. It does not perform filesystem IO or durable side
  effects.
- `read_*` parses or decodes a durable or boundary representation into a typed
  representation record or bridge record.
- `write_*` persists a representation to a durable output boundary.
- `load_*` crosses from durable or boundary storage into the typed domain
  object that downstream code should consume.
- `validate_*` reports contract violations without repairing inputs.
- `compact_*` is reserved for strict storage-level reduction over complete
  compatible durable products.
- `render_*` or `build_*_tensor*` may be used for explicit TensorCore-backed
  tensor rendering when a stage accepts that boundary.
- `assemble_*` packages already-built typed products into one coherent
  in-memory example or container. It does not load, write, or invoke DAG
  behavior.
- `compute_*` derives a result from already-available inputs without durable
  side effects.

Prefer the verb based on the return value and boundary crossed:

- If a function reads storage and returns the domain product, use `load_*`.
- If a function reads storage or in-memory boundary data and returns a
  representation or bridge record, use `read_*`.
- If a function creates an in-memory JSON, table, array, tensor, or other
  representation-shaped value, use `build_*`, not `write_*`.
- If a function writes files, cache entries, artifact files, or other durable
  outputs, use `write_*`.

`from_*` and `to_*` are acceptable for methods or very local/private
conversions. Public module-level bridge functions should prefer
`build/read/write/load`.

Reserve `parse_*` for textual grammars or user input. Avoid `serialize_*` and
`deserialize_*` unless the project is explicitly implementing a serialization
layer. Do not use `get_*` for functions that hide IO, construction,
validation, or expensive computation.

`validate_*` functions must not repair, fill, cast, write, or conceal missing
upstream work. `assemble_*` functions must not call loaders, writers, cache
APIs, or DAG APIs.

`build_readout_collection(...)` is a domain builder: it composes an already
validated in-memory source through the fixed configured atomic readout chain.
It may schedule caller-owned workspace scratch, but it must not load data,
perform durable IO, move or cast tensors, cache or resize workspaces, or invoke
DAG APIs.

## Deferred Integration Surfaces

Projects/dag owns campaign orchestration, sharding, scheduling, concrete DAG
construction, execution policy, repair, retries, status, and fanout/fanin.
TensorDSLab may later expose stable public surfaces for operation specs,
executable adapters, artifact/cache requirements, output validation, and
recipe fragments.

Local atomic-transform composition and scratch scheduling inside
`build_readout_collection(...)` are TensorDSLab domain construction, not
Projects/dag campaign orchestration.

For compaction, TensorDSLab owns a strict deterministic storage-level operation
over complete compatible caller-supplied products. Projects/dag owns discovering
or scheduling shards, campaign fan-in, retries, repair, and cross-shard
execution policy.

Use these optional package directories only when the project needs them:

- `operations/` for DAG-compatible operation specs;
- `recipes/` for reusable composition fragments;
- `executables/` for CLI, DAG, or task adapters.

Do not add DAG-compatible modules, downstream adapters, package-local workflow
CLIs, or cache-driven integration surfaces before local TensorDSLab contracts
are accepted. Keep atomic transforms dependency-light and campaign-
orchestration-free; local composition belongs only in the explicit domain
builder.

## Parts-Bin Rule

Historical predecessor code is donor material only. Reuse scientific facts,
small algorithms, naming lessons, fixtures, tests, and accepted cache semantics
after review. Do not preserve old package layouts, helper frameworks, local DAG
mechanics, compatibility wrappers, or representation shortcuts by default.

When promoting donor code or behavior:

- write down the accepted reason in the relevant implementation or decision
  doc;
- name the donor snapshot and source symbol, comparison boundary, parity
  classification, assumptions, observables, acceptance criteria, exclusions,
  and intentional divergences in `docs/parity.md`;
- adapt names to TensorDSLab's tensor-native design;
- remove compatibility baggage unless Design accepts a compatibility window;
- add tests around the promoted contract.

## Engineering Standard

Prefer boring, explicit, maintainable code over cleverness. The design should
be easy for a future contributor to reconstruct from module names, type
signatures, tests, and architecture documents.

Good changes should:

- preserve documented ownership boundaries;
- keep public APIs small, typed, and intention-revealing;
- use concrete typed records instead of unstructured dictionaries where a
  stable concept exists;
- keep implementation details private until they are real extension points;
- make behavior deterministic unless nondeterminism is explicit;
- include tests that protect behavior and invariants;
- update source-of-truth documentation when public contracts or accepted
  semantics change.

## Boundary-First Validation

TensorDSLab should move toward validated-once, trusted-downstream records.

Validate strongly when data enters or re-enters the TensorDSLab/TensorCore
typed path:

- the future TensorG4DS-to-TensorDSLab semantic bridge and its provenance,
  coordinate, unit, dtype, layout, and device contract;
- user configs;
- construction of public ID objects;
- construction of constrained scalar wrappers for meaningful numeric config or
  artifact values;
- construction of detector, readout, reconstruction, cache, table, array, and
  tensor records;
- construction of TensorCore axes, layouts, fields, collections, and
  selections.

Once an object has crossed into a valid native record, hot-path functions
should avoid repeatedly revalidating full object graphs. Product builders and
tensor renderers may still perform narrow function-specific checks, but they
should not rediscover layout validity, identity validity, or mapping
immutability every inner-loop call.

Use scalar wrappers at config, source, and artifact boundaries where
constraints are meaningful. Numeric wrappers should reject bool. Plain `bool`
is appropriate for boolean fields.

Prefer these migration directions as real code is introduced:

- runtime-significant `NewType` aliases become frozen runtime wrappers;
- repeated ad hoc numerical checks become constrained scalar records;
- repeated broad validation in hot paths moves to constructor invariants and
  boundary validators;
- recursive inner-loop object checks become trusted typed inputs plus narrow
  operation preconditions;
- scalar wrappers remain values and do not become tensor coordinates unless
  Design explicitly accepts that model.

## Scope Discipline

Implement only the accepted documentation stage or production work order. Do
not broaden package ownership, public APIs, cache shape, TensorCore contracts,
DAG semantics, or downstream integration implicitly.

If work reveals a contradiction in the accepted design, stop and route it to
Design. Do not patch around it by inventing architecture inside implementation.

## Code Expectations

- Use a short module context docstring when ownership or boundary is not
  obvious from the module path and public types. Do not add filler docstrings
  to tiny cohesive modules.
- Type public functions, methods, dataclass fields, and module constants.
- Avoid `Any`, unbounded `dict`, and stringly typed public interfaces unless
  the boundary is intentionally JSON-like or there is a documented reason.
- Prefer dataclasses for stable records.
- Use `value` for primitive payloads on ID and scalar-wrapper records.
- Prefer frozen runtime wrapper classes for IDs used on tensor-facing hot
  paths.
- `Enum`, `Literal`, `Protocol`, and generics remain useful for non-hot-path
  contracts when they make a public boundary clearer.
- Keep modules cohesive; split a module when it owns more than one meaningful
  boundary.
- Keep comments sparse and useful. Explain non-obvious intent, not mechanics.
- Do not hide cross-domain, TensorCore, cache, or adapter behavior behind broad
  helper modules.

## Test Expectations

Tests should prove intended behavior, not mirror implementation structure.

Good tests should:

- exercise success paths and meaningful failure modes;
- protect invariants and boundary conditions;
- cover serialization and round-trip behavior where persistence is involved;
- prove deterministic ordering where order is part of the contract;
- prove validation rejects malformed artifacts, caches, or domain objects;
- assert public exports and retired-name absence when a stage performs a clean
  public API transplant;
- include import-isolation or dependency-scan smoke tests when a stage extracts
  or layers packages;
- prove TensorCore extension-point rules: TensorDSLab IDs may subclass `Id`,
  `ReadoutCollection` may subclass `TensorCollection`, and sealed primitives
  are not subclassed;
- prove recognized field subsets use canonical order and exact common layouts;
  required axes use the exact exported IDs in arbitrary layout order, example
  and channel coordinates have their exact domain types, sample is count-only,
  and every layout axis is shared;
- prove transform-driven addition and replacement invalidate every reachable
  stale descendant while projection does not;
- prove differentiable functional rendering remains in the graph and
  `out`/workspace simulation paths reject gradient-sensitive use;
- prove all preflight precedes RNG consumption and writes, targets are fully
  overwritten, and launched backend failures make no rollback promise;
- prove workspace exact-match, same-stream exclusive lease, write-before-read,
  no hidden resize/cache/move/cast/source recycling, and no returned-workspace
  aliasing;
- prove the warmed exact-`out` plus workspace path performs no
  TensorDSLab-managed tensor-storage allocation, without overclaiming about
  Python, PyTorch, or vendor allocations;
- prove `build_readout_collection(...)` matches explicit atomic composition and
  performs no source IO or campaign orchestration;
- use small fixtures that make behavior visible;
- avoid depending on private implementation details unless testing a private
  helper is the only focused way to cover an edge case.

When a test would be expensive, slow, or fragile, prefer a smaller invariant
test plus one representative integration test.

## Documentation Expectations

Documentation should state contracts, boundaries, non-goals, and examples that
guide implementation. It should not become a second copy of the source code.
Use the `docs/` spine for design, validation, decisions, architecture
contracts, and staged work orders.

Update docs when a change affects:

- public APIs;
- TensorCore-backed tensor axes, layouts, fields, selections, or shapes;
- product ownership;
- cache files, manifest shape, durable guarantees, or compaction rules;
- validation rules;
- donor comparison boundaries, parity classifications, assumptions,
  tolerances, or intentional divergences;
- operation, recipe, or executable surfaces;
- implementation stages or accepted decisions.

Keep the relevant source of truth synchronized:

- `docs/implementation/...` for stage work orders, scope, public surfaces,
  invariants, and non-goals;
- `docs/architecture/<domain>.md` for public domain contracts, cache shapes,
  builders, validation boundaries, and representation bridges;
- `docs/architecture/tensors.md` for the TensorCore consumer contract,
  semantic tensor products, axis/field rules, placement, outputs, workspaces,
  lifetimes, and cross-repository coordination items;
- `docs/parity.md` for donor evidence, comparison classes, assumptions,
  tolerances, fixture provenance, and intentional divergences;
- `docs/architecture/common.md` for shared primitives and cross-domain rules;
- `docs/design.md` for end-to-end domain flow and ownership boundaries;
- `docs/decisions.md` for accepted, renamed, superseded, or explicitly
  deferred semantic choices;
- `docs/validation.md` for validation cases, fixtures, failure modes, and
  tolerances;
- `README.md`, `AGENTS.md`, or `CONTRIBUTING.md` for workflow, onboarding, or
  repository-wide engineering expectations.

## Documentation-Only Design Checks

Documentation-only Design changes remain in the Design thread unless the user
requests independent Validation or Review. At minimum, run:

```bash
git diff --check
```

Also run targeted link, heading, and stale-term searches for the changed
contracts. Do not create placeholder code or tests merely to exercise a docs
stage.

When package-governance records change, also run the state,
manifest, rule-coverage, source-anchor, dormant-trigger, deviation, raw-ID,
changed-file-allowlist, and forbidden-claim checks defined in
[Validation](docs/validation.md#governance-adoption-checks). Runtime, import,
dependency, export, environment, and post-merge commands are active for the
accepted Stage 2 foundation. Integration and later scientific-runtime commands
remain dormant until their corresponding implemented surfaces exist.

## Before Production Review

Before asking for Review, provide:

- branch and commit, when the repository is initialized as git;
- scope implemented;
- files changed;
- docs updated or checked;
- tests added or changed;
- commands run;
- known risks or deferred follow-ups.

At minimum, run:

```bash
git diff --check
```

For Python changes after the package exists, also run the relevant test suite:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

For future DAG-compatible changes, also run the accepted operation-spec or
domain-module validation command.

Never stage generated caches, local outputs, or unrelated files.
