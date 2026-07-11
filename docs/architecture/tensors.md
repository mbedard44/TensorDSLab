# TensorCore Integration Architecture

Status: active Design contract for the post-binned readout architecture. The
Stage 2 package and exact TensorCore dependency are accepted on `main` through
Merged / Closed commit `e8c62caf001ee7f58f766d7234747ed1d9a21e35`.

## Purpose

This page defines how TensorDSLab uses TensorCore. It records TensorDSLab's
consumer contract, not a second copy of TensorCore's API documentation.

TensorCore is the generic dense tensor substrate. TensorDSLab owns the detector
and readout meaning layered over that substrate:

```text
TensorDSLab domain values and semantic IDs
  -> TensorCore axes and layouts
  -> TensorCore fields and collections
  -> TensorDSLab ReadoutCollection snapshots
  -> TensorDSLab transforms and future examples/caches
```

The dependency direction is one-way:

```text
tensor_dslab -> tensor_core
```

TensorCore must not import TensorDSLab, and TensorDSLab must not fork or shadow
TensorCore primitives.

## Cross-Package Tensor Spine

The long-term data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared substrate rather than a data-flow stage. The current
post-binned foundation still depends only on TensorCore. A future
TensorDSLab-owned integration adapter may import one exact, accepted public
TensorG4DS product; TensorG4DS must never import TensorDSLab, and no loose
protocol over generic `TensorCollection` fields substitutes for the upstream
nominal contract.

The production boundary target keeps tensor payloads resident on one exact GPU
across TensorG4DS, TensorDSLab, and TensorML. It permits no implicit `.cpu()`,
NumPy/list conversion, serialization/reload, movement, cast, or detach.
TensorCore IDs, layouts, and small sidecar records may remain host-side
metadata. The TensorG4DS-to-TensorDSLab bridge is nevertheless a semantic
transform: event/deposit/cluster values become newly constructed
example/channel/sample values on the same device. Direct handoff therefore
means no host staging, not identical layouts, subclass casting, or zero-copy
detector computation.

TensorG4DS has not yet frozen the exact public product, GPU placement contract,
dtypes, or layouts needed by that bridge. Those questions remain outside this
post-binned page and must be accepted in a later cross-repository integration
stage. TensorG4DS `EventId` remains source provenance; it is never reused as
TensorDSLab `ExampleId`, and TensorDSLab `ChannelId` is never inferred from an
upstream index.

## Design Baseline

The current design baseline is TensorCore `0.6.0`. The Stage 2 Implementation
line pins exact commit `dc554994061183776f23f65860a0594516074f2e`, which is the
accepted Stage 2 dependency on `main`.

TensorDSLab may rely on these public TensorCore concepts:

- `Id`, `TensorAxisId`, `TensorFieldId`, and `IdSequence`;
- `PositiveInteger`, `NonnegativeInteger`, `FiniteFloat`, `PositiveFloat`,
  `NonnegativeFloat`, and `Probability`;
- `TensorAxis`, `TensorAxes`, `TensorLayout`, `TensorField`, and
  `TensorCollection`;
- `TensorFieldSelection`, `TensorAxisSelection`, and `BatchConfig`;
- public layout builders, selection and batching helpers, movement, reduction,
  detachment, field addition, output-buffer allocation, immutable mapping
  helpers, and generic validators.

Production code should import these names from the public `tensor_core` package
root. Do not import TensorCore implementation modules or re-export generic
TensorCore helpers through `tensor_dslab.common`.

## Extension Points And Sealed Primitives

TensorCore has two intentional downstream extension points:

- subclass `Id` for stable TensorDSLab coordinates;
- subclass or compose `TensorCollection` for semantic dense products.

TensorDSLab must not subclass sealed TensorCore primitives, including:

- `TensorAxisId` or `TensorFieldId`;
- constrained scalar wrappers;
- `IdSequence`;
- axes or layouts;
- `TensorField`;
- selections or `BatchConfig`.

TensorDSLab product meaning belongs around `TensorField` values, never in a
semantic `TensorField` subclass.

## Semantic Collection Shape

The first post-binned readout surface uses one concrete semantic subclass:

```python
class ReadoutCollection(TensorCollection):
    ...
```

`ReadoutCollection` is a structurally immutable, partially materialized
snapshot of recognized readout fields. Any nonempty recognized field subset is
valid when the present fields satisfy the collection contract. Charge and
waveform names are field roles inside this collection, not separate
`TensorCollection` subclasses. TensorDSLab should not introduce a generic
`Product` base or per-field semantic collection subclasses.

The accepted layering is:

```text
TensorCollection
  -> ReadoutCollection with a typed SampleGrid and conditional field sidecars
  -> free TensorDSLab transforms that add or replace recognized fields

ReadoutExample
  -> optional thin provenance/context wrapper around a ReadoutCollection
```

`ReadoutExample` is not the tensor handoff and does not duplicate the field
payloads. The collection itself remains directly consumable through the
TensorCore contract.

TensorCore selection, batching, movement, and like-buffer helpers return base
`TensorCollection` records. TensorDSLab must reconstruct and revalidate a
`ReadoutCollection` after generic operations rather than assuming subclass
preservation.

## IDs, Coordinates, Indices, And Ordering

TensorCore terminology is binding:

- a coordinate is a stable `Id` value on an ID-backed axis;
- an index is a zero-based tensor position;
- a layout is ordered axes plus coordinate-to-index maps for ID-backed axes.

TensorDSLab IDs identify stable domain points, not tensor positions, ranges,
units, or bin edges. Domain IDs such as channel, example, window, or future
detector-element IDs remain TensorDSLab-owned `Id` subclasses.

`tensor_dslab.common` owns and exports the shared coordinate types `ExampleId`
and `ChannelId`. `ChannelId` is deliberately not readout-specific: later
reconstruction products reuse the same detector-channel identity. The
readout-specific `TensorAxisId` and `TensorFieldId` constants remain owned and
exported by `tensor_dslab.readout`, with their definitions housed in
`tensor_dslab.readout.ids`.

`common`, `readout`, and future scientific packages are direct subpackages of
the package root; no intermediate domain namespace is part of the architecture.

`IdSequence` order is caller order and therefore tensor order. TensorDSLab must
not sort, infer, hash-order, or otherwise replace that order. ID sequences must
be non-empty, unique, and contain one exact concrete ID class.

Domain axis builders should validate the expected TensorDSLab coordinate class.
The canonical readout example axis contains exact `ExampleId` coordinates, and
the canonical readout channel axis contains exact `ChannelId` coordinates
rather than arbitrary `Id` values.

If an axis is ID-backed, public diagnostics and reproducibility keys should use
its coordinates. Integer indices may be used only after resolution through the
layout. Count-only axes use native zero-based positions and have no durable
coordinate identity.

## Axes And Layouts

TensorDSLab does not impose one global rank or axis order. Every field in one
`ReadoutCollection` has the exact same ordered `TensorLayout`, and operations
locate required dimensions by looking up exact public axis IDs in that ordered
layout.

The required public axis IDs are fixed TensorCore values owned by
`tensor_dslab.readout`, not caller-selected roles and not TensorDSLab
subclasses:

```python
READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")
READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")
READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")
```

These constants define semantic identity by `TensorAxisId` value equality.
Object identity is irrelevant: a freshly constructed equal `TensorAxisId`
denotes the same axis. Their positions remain runtime layout choices. An
operation resolves each position through
`layout.axes.index(READOUT_*_AXIS_ID)` and never assumes a conventional tuple
position.

The post-binned readout contract requires shared:

- `READOUT_EXAMPLE_AXIS_ID`;
- `READOUT_CHANNEL_AXIS_ID`;
- `READOUT_SAMPLE_AXIS_ID`;
- zero or more additional axes accepted by a focused contract.

The example axis is ID-backed by exact `ExampleId` coordinates, and the channel
axis is ID-backed by exact `ChannelId` coordinates, so identity survives
reordering, selection, batching, and diagnostics. The sample axis is
count-only; its physical period, origin, and containing-grid offset live in a
typed TensorDSLab `SampleGrid`, not in IDs. The three canonical axes are
declared collection-shared and agree exactly across every present field.

Additional axes are allowed only when every present field carries the same
ordered layout and the operation can preserve them. Every accepted extra axis
occurs in every field and is declared collection-shared. Every ID-backed shared
axis participates in stochastic identity; this is derived from the layout and
is not configurable through a role record. The coordinate-key sequence is:

1. `READOUT_EXAMPLE_AXIS_ID` paired with its `ExampleId` coordinate;
2. `READOUT_CHANNEL_AXIS_ID` paired with its `ChannelId` coordinate;
3. each additional ID-backed shared `axis_id` paired with its coordinate, with
   those pairs ordered lexicographically by `axis_id.value` rather than tensor
   position;
4. `READOUT_SAMPLE_AXIS_ID` paired with the global sample ordinal,
   `SampleGrid.sample_offset + local_sample_index`, on
   that count-only axis.

Lexical ordering applies only to the extra-axis components of the RNG key. It
does not reorder layout axes or any axis's caller-ordered coordinate sequence.

Extra count-only axes are allowed structurally for operations that preserve
them. Stochastic transforms reject a collection containing any count-only
extra axis until Design accepts an offset or coordinate contract that gives
each of its positions stable random-field identity. They must not fall back to
transient local indices.

`ReadoutCollection.shared_axes` is the ordered `IdSequence` of every axis ID in
the common layout, in layout order. It is not a smaller advisory subset.

Axis order is never inferred from conventional array positions. Tests should
exercise at least two valid axis orders for operations that claim order
independence.

## Semantic Layout And Warmed Execution Profile

Collections define meaning. Execution profiles define memory arrangement.
Domain handoffs are explicit materialization boundaries.

The semantic collection contract accepts arbitrary axis order and
noncontiguous `torch.strided` read-only payloads. TensorLayout remains the only
semantic axis-order source of truth. The general constructor does not reject
expanded or internally overlapping read-only source views merely because of
their storage arrangement. There is no `ExecutionReadyReadoutCollection`,
`ReadoutAxisOrder`, stride sidecar, or runtime storage flag in collection
identity.

The warmed `out + workspace` readout profile is stricter:

- `READOUT_SAMPLE_AXIS_ID` is the last tensor dimension;
- each participating source, generated public target, and private scratch
  tensor is contiguous;
- every writable tensor is internally nonoverlapping and storage-disjoint from
  sources, other live public outputs, and incompatible scratch roles;
- exact ordered axis IDs/sizes, shape, device, role dtypes, enabled algorithms,
  destination schema, stream, and exclusive workspace lease match; and
- all preflight completes before RNG consumption or writes.

After preflight, kernels may trust `sample_dimension == tensor.ndim - 1`, unit
sample stride, fixed shapes/dtypes/device, prepared scratch, and exclusive
lease. They may flatten leading axes with `view(-1, sample_count)` only because
the strict contiguous sample-last profile has already been proved. The warmed
path must not permute, call `.contiguous()`, clone, cast, move, use a reshape
that copies, or fall back to allocating storage.

Leading-axis order remains semantically arbitrary. For example,
`(example, channel, sample)` and `(channel, example, sample)` are two valid but
different warmed signatures; a sample-first layout is a valid collection but
not a valid warmed readout input. Contiguous strides derive from ordered shape,
so the MVP workspace signature does not store arbitrary stride tuples. A
future allocation-free noncontiguous profile requires a focused Design change,
an exact supported-kernel/stride matrix, and memory instrumentation.

Functional `out=None` execution may explicitly allocate to normalize arbitrary
semantic order or strides while preserving accepted autograd. Ordinary
`out=destination` without a workspace may use documented allocating scratch or
normalization and makes no allocation-free claim; its writable target must
still be internally nonoverlapping and storage-disjoint. A source that is not
warmed-ready is explicitly reordered/materialized once outside the repeated
loop. If that operation is only a coherent semantic axis permutation, it may
construct a new validated `ReadoutCollection`; once a bridge changes
representation, collapses axes, or introduces another domain's fields, the
result belongs to that domain instead.

This pattern is domain-specific. A future Readout-to-Reconstruction bridge
selects required fields, validates exact `ChannelId` coordinates/completeness,
reorders by stable axes rather than guessed indices, explicitly materializes a
contiguous reconstruction-preferred layout once, and constructs a
reconstruction-owned value before its own workspace/profile. Temporal readout
prefers sample-last; an all-channel reconstruction kernel may prefer
channel-last. TensorML likewise owns an explicit model-facing selection/layout.

## Fields, Field Roles, And Product Labels

TensorCore field IDs and TensorDSLab product labels are separate namespaces.
The exact readout field-ID constants and their canonical registry are owned by
`tensor_dslab.readout` and defined in `tensor_dslab.readout.ids`; they do not
belong in `tensor_dslab.common` merely because the generic ID machinery comes
from TensorCore.

The recognized first-MVP field IDs, in canonical topological order, are:

```text
TensorFieldId("readout.photoelectrons")
TensorFieldId("readout.charge")
TensorFieldId("readout.waveform.pure")
TensorFieldId("readout.waveform.noise")
TensorFieldId("readout.waveform.analog")
TensorFieldId("readout.waveform.digitized")
```

Every `ReadoutCollection` stores the canonical order filtered to the fields
that are present. Callers do not define a second insertion order for readout
snapshots. Any nonempty subset is valid; field presence says which products
are materialized, not that every upstream or downstream field must also be
present.

Durable producer labels remain TensorDSLab concepts:

```text
readout.photoelectrons
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.analog
readout.waveform.digitized
```

`readout.photoelectrons` is the in-memory input to the post-binned MVP. Its
producer implementation and durable representation remain deferred with
source photoelectron binning, but its semantic product label is accepted now.

The photoelectron field contains binned, photon-origin primary PE seed counts.
It does not contain dark-count, crosstalk, or afterpulse avalanches. Those
effects are private scratch states inside `simulate_charge(...)`, whose
materialized target is the aggregate floating PE-equivalent
`readout.charge` response. Intermediate avalanche-count grids are not
recognized fields or products.

Matching strings do not make a product label a `TensorFieldId`. Future cache
code should persist explicit product labels and rebuild a validated
`ReadoutCollection`; it should not persist Python subclass identity as the
durable contract. Even where a durable product label and field ID have the
same string payload, they remain values of different types with different
ownership.

Field mapping order is public TensorCore schema. Constructors and tests must
enforce the filtered canonical order so construction path does not change
field enumeration.

## Typed Collection Sidecars

Essential semantics should not live only in unstructured TensorCore metadata.
The `ReadoutCollection` constructor receives only these typed TensorDSLab
sidecars:

- `SampleGrid`, containing sample period, physical origin, and stable
  containing-grid offset;
- `DigitizedWaveformSpec` when and only when the digitized field is present,
  including bit depth, voltage range/offset, analog gain, and typed
  quantization policy needed to derive ADC bounds and interpret counts.

Required axis meaning and stochastic coordinate order are package-level
contracts derived from the canonical axis constants and the validated layout.
No configurable required-axis mapping or stochastic-axis record is carried
through collection operations.

TensorCore `metadata` remains appropriate for small descriptive provenance. It
must not carry hot tensors, hidden execution policy, mutable state, or the only
copy of an invariant required by a transform.

## Partial Snapshots, Projection, And Replacement

A `ReadoutCollection` is structurally immutable. Its records and mappings are
immutable, while tensor payloads retain ordinary PyTorch mutability.
TensorDSLab transforms treat source and retained tensor payloads as read-only.
A newly returned collection may structurally share retained `TensorField`
records with its source.

Callers must also treat every materialized field tensor as read-only. The only
public write is to the fresh target tensor of an explicitly prepared `out=`
result collection. Direct in-place edits to an existing field bypass
descendant invalidation and place the snapshot outside the TensorDSLab
contract; use a transform or validated replacement instead.

Projection and replacement are distinct operations:

- projection removes fields without changing the values or validity of fields
  that remain;
- transform-driven addition inserts an absent target field, structurally shares
  unaffected fields, and removes every materialized transitive descendant that
  could disagree with the new dependency value;
- replacement applies the same invalidation rule while substituting a present
  target field.

The first dependency graph is:

```text
readout.photoelectrons -> readout.charge -> readout.waveform.pure
common layout/sample grid -> readout.waveform.noise
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog
  -> readout.waveform.digitized
```

Replacing photoelectrons invalidates charge, pure, analog, and digitized.
Timing jitter is the only accepted public transform that replaces
`readout.photoelectrons`. `simulate_charge(...)` consumes that field without
replacing it and adds or replaces `readout.charge`; its dark-count, crosstalk,
afterpulse, and smearing states remain internal. Replacing charge invalidates
pure, analog, and digitized. Replacing pure or noise invalidates analog and
digitized. Replacing analog invalidates digitized. Noise is
layout/sample-grid dependent but does not depend on charge values, so
replacing charge does not by itself invalidate noise.

One central TensorDSLab result builder owns this invalidation table, canonical
field ordering, structural sharing, and target insertion. It removes reachable
descendants whether the transform target was absent or present. Individual
transforms must not hand-code different descendant policies. Pure projection
does not invoke invalidation because no retained value changed. Field
coexistence alone does not prove numeric derivation provenance; transforms and
future assembly boundaries establish the relationships they produce.

Field-specific sidecars are updated atomically with their fields. Removing or
invalidating `readout.waveform.digitized` removes its
`DigitizedWaveformSpec`; projecting or retaining that field preserves the spec;
adding or replacing it installs the validated expected spec.

## Generic Operations And Reconstruction

TensorCore generic operations may accept `TensorCollection` subclasses but
return base `TensorCollection` values. TensorDSLab semantic methods or free
helpers should follow this pattern:

```text
ReadoutCollection
  + explicitly retained SampleGrid and conditional DigitizedWaveformSpec
  -> generic TensorCore operation on the collection spine
  -> base TensorCollection
  + preserved/updated SampleGrid and preserved/pruned DigitizedWaveformSpec
  -> TensorDSLab collection constructor
  -> revalidated ReadoutCollection
```

Do not add local copies of TensorCore selection, batching, movement, mapping,
or layout mechanics merely to preserve subclass identity.

A base result does not contain subclass attributes and is not sufficient by
itself to reconstruct semantic meaning. The TensorDSLab helper that invokes a
generic operation explicitly carries `SampleGrid` and the conditional
`DigitizedWaveformSpec` from the source. It updates the sample grid for an
accepted sample selection, prunes the digitized spec when the field is absent,
and passes the resulting typed records to the constructor. Canonical axis
identity is recovered directly from the result layout and revalidated; no axis
mapping is reconstructed. The helper must not infer missing sidecars from
free-form TensorCore metadata.

TensorDSLab domain projection returns a canonical-order `ReadoutCollection`.
TensorML input/target projection is a different boundary: an explicit
`TensorFieldSelection` defines the positional model schema and may return a
base `TensorCollection` in selection order. Current TensorML batching and
selection erase downstream subclass identity, so no TensorML API change or
subclass-preserving assumption is accepted here.

A model-requested selection whose order is not the canonical filtered readout
order remains a base `TensorCollection`; reconstructing it as
`ReadoutCollection` would silently change the positional ABI. In current stock
TensorML training/evaluation loops, a model receiving selected fields should
therefore accept `input_type = TensorCollection`. It may require
`ReadoutCollection` only when a focused adapter reconstructs a canonical
selection before `forward_pass`. Likewise, a model output that claims
`ReadoutCollection` must be built and validated through TensorDSLab; TensorML's
class check alone does not prove its field IDs or schema.

Axis selection requires the same explicit reconstruction discipline.
Selections on `READOUT_EXAMPLE_AXIS_ID`, `READOUT_CHANNEL_AXIS_ID`, and accepted
extra axes may reconstruct a collection when the transformed common layout and
coordinate classes remain valid. The current
`SampleGrid(period, origin, offset)` can represent only a contiguous,
increasing, unit-stride range selected on `READOUT_SAMPLE_AXIS_ID`.
Reconstructing after such a selection updates:

```text
origin_ns = origin_ns + first_local_index * sample_period_ns
sample_offset = sample_offset + first_local_index
```

An arbitrary, reordered, duplicated, or strided sample selection must not be
relabeled as a valid `ReadoutCollection`; it remains a base
`TensorCollection` or returns to Design for a richer sampled-grid contract.
Every axis selection intended for semantic reconstruction applies coherently
to all present fields.

Readout field composition remains TensorDSLab-owned. For example, composing
pure and noise fields is a domain operation even though both already live in
one collection and share a layout.

## Memory And Projection Policy

A complete six-field snapshot owns references to the sum of its materialized
tensor payloads. Structural sharing prevents transforms from copying retained
payloads, but it does not make those payloads disappear while any live
snapshot still references them.

Field retention and placement are therefore explicit runtime policy:

```text
ReadoutCollection
  -> project the required canonical field subset without copying tensors
  -> move only that projected base TensorCollection through TensorCore
  -> reconstruct and validate ReadoutCollection on the target device
```

Callers should project before accelerator movement when they do not need the
full snapshot, and should release obsolete snapshots when their retained
fields are no longer needed. Transforms remove scientifically stale
descendants, but they must not evict unrelated valid fields, move tensors, or
hide a memory-retention policy. Scientific configs, TensorCore metadata, IDs,
and product labels must not encode projection, eviction, or placement policy.

TensorML projection may instead preserve the explicit model-requested field
order in a base `TensorCollection`; it need not reconstruct
`ReadoutCollection` merely to call a model. In both cases, the selected field
set is explicit and adding a field to a source snapshot does not silently
increase model inputs or accelerator memory.

## Output Buffers And Autograd

TensorDSLab atomic transforms have three explicit execution modes:

```text
out=None
  -> build the expected result ReadoutCollection
  -> structurally share retained source fields
  -> construct a fresh target field from computed values
  -> do not mutate the source
  -> explicit allocating order/stride normalization is permitted
  -> preserve autograd for differentiable deterministic transforms

out=collection
  -> validate the exact expected result collection
  -> require retained fields to be the exact shared source records
  -> write only the fresh, nonaliasing target field buffer
  -> return that exact collection
  -> ordinary internal scratch or documented normalization allocation remains allowed
  -> buffer-reuse simulation path, not an autograd guarantee

out=collection, workspace=compatible exclusive lease
  -> preserve the exact same one-target destination rules
  -> use prepared workspace storage for every named scratch tensor
  -> warmed TensorDSLab-managed tensor-storage-allocation-free path
  -> buffer-reuse simulation path, not an autograd guarantee
```

Supplied outputs must be exact `ReadoutCollection` instances matching the
expected post-invalidation field set, canonical order, `SampleGrid`,
conditional `DigitizedWaveformSpec`, layouts, shapes, common device, per-field
dtypes, and `torch.strided` layout. Every retained field must be the same
`TensorField` object structurally shared from the source. The target field
record and tensor must be internally nonoverlapping and must not alias the
source target being replaced, any retained field, or any other live output.

Output allocation is field-scoped. Public
`build_readout_result_buffer(...)` allocates only a zero-initialized contiguous
target field in the collection's existing semantic axis order and reuses
retained field records; it never reorders or materializes the retained source.
The transform then overwrites the target completely. It must not expose a
`ReadoutCollection` containing uninitialized `torch.empty_like` values because
public construction validates finite and role-specific nonnegative value
domains. Transforms must not call whole-collection `empty_like()` or
`zeros_like()` merely to produce one field.

Transforms must not silently move, cast, detach, or replace an output buffer.
Read-only structural sharing of retained source fields is required; aliasing
the writable target with any source tensor is forbidden. Internal scratch may
allocate in ordinary functional and destination-reuse execution. The accepted
warmed workspace mode below is the only mode that removes TensorDSLab-managed
scratch-storage allocation from the steady-state contract.

Differentiable pure-waveform rendering and analog-waveform composition must
remain in the graph when called without `out`. They must not run under
`torch.no_grad()` or detach inputs. Stochastic sampling and digitization are
not declared differentiable. An `out=` path should reject gradient-sensitive
use rather than silently changing gradient behavior.

TensorCore `require_compatible_collection(...)` currently requires exact base
`TensorCollection` objects. Until TensorCore accepts subclass-aware structural
compatibility, TensorDSLab must use an explicitly documented base-collection
projection or another Design-approved bridge. TensorDSLab should not duplicate
and slowly fork the compatibility algorithm.

## Full-Chain Builder And Readout Workspace

TensorDSLab provides one local full-chain convenience builder over the free
readout transforms. It is a domain execution surface, not a source loader,
cache operation, DAG executable, or replacement for the individual transform
APIs. Non-final public spelling is:

```python
def build_readout_collection(
    source: ReadoutCollection,
    config,
    *,
    rng,
    out: ReadoutCollection | None = None,
    workspace: ReadoutWorkspace | None = None,
) -> ReadoutCollection: ...
```

The source requires `readout.photoelectrons`. The builder executes the complete
configured local chain in dependency order:

```text
optional timing jitter
  -> charge simulation
  -> pure and noise rendering
  -> analog composition
  -> optional digitization
```

Its result contains photoelectrons, charge, pure, noise, and analog, plus
digitized exactly when configured. It recomputes the configured chain rather
than opportunistically retaining an existing descendant. When timing jitter is
disabled, the photoelectron field is the exact retained source record; when it
is enabled, photoelectrons are a fresh nonaliasing target. General partial-
output execution plans are deferred.

Public `build_readout_output_buffer(...)` prepares the exact final field schema
before execution. It structurally shares only fields that the full chain
retains and zero-initializes every generated target in contiguous storage with
its final semantic layout order, dtype, device, `SampleGrid`, and conditional
`DigitizedWaveformSpec`. It does not reorder a retained source field. A full
output is warmed-ready only when the source already has sample last and every
participating retained tensor also satisfies the strict profile.
`build_readout_collection(...)` validates the source, complete destination,
config, RNG, and optional workspace before the first write, then holds the
destination exclusively until every target has been overwritten. It does not
expose intermediate stage snapshots. Internally it may form exact
stage-specific collection views over the prepared field records so every free
transform still writes only its one current target.

The builder accepts exactly three mutually exclusive execution modes:

1. Functional: `out=None, workspace=None`. Allocate owned result fields and
   ordinary scratch or explicit normalization, leave the source unchanged,
   and preserve the accepted autograd behavior of differentiable deterministic
   transforms.
2. Destination reuse: `out=destination, workspace=None`. Return the exact
   caller-owned destination after complete target-only writes; documented
   scratch or normalization may allocate. This is a simulation path and
   rejects gradient-sensitive use.
3. Warmed tensor-storage-allocation-free: supply both `out=destination` and a
   compatible already allocated and warmed workspace. Use the same destination
   contract plus prepared scratch. This mode also rejects gradient-sensitive
   use.

Supplying a workspace without `out` is invalid. These modes do not create an
ambient mutation setting and do not change any field, product, or TensorCore
identity.

`ReadoutWorkspace` is a mutable runtime resource owned and supplied explicitly
by the caller. TensorDSLab borrows it only for one builder call. It owns private
scratch storage and no public field payload, source collection, coordinate
map, result collection, or mutable RNG stream. No returned
`ReadoutCollection` tensor may alias workspace scratch. TensorDSLab maintains
no hidden global, thread-local, device-local, or LRU workspace cache.

The MVP workspace has one fixed execution signature derived from source
geometry and the full-chain config:

- ordered axis IDs and exact axis sizes, without caching the current coordinate
  values; canonical example, channel, and sample positions are derived from
  that order by axis-ID lookup rather than stored as independent semantic
  inputs;
- exact device, including accelerator index;
- exact count, common floating, digitized, and derived complex scratch dtypes
  when those domains are used;
- enabled algorithm families and every parameter that changes scratch shape,
  such as sample count, FFT shape, pulse support, or delayed-response bucket
  geometry;
- exact configured destination schema; and
- one synchronous CPU execution domain or exact CUDA stream.

Warmed compatibility additionally requires sample-last layout and contiguous
participating source, generated output, and scratch tensors. Writable storage
is internally nonoverlapping and disjoint. The signature stores no arbitrary
stride tuple because contiguous strides derive from its ordered sizes.

A compatible same-shaped batch may carry different coordinates on its
ID-backed shared axes because stochastic keys come from the current source
collection, not the workspace. A different ordered axis-ID sequence, axis
size, sample-last position, device, dtype, algorithm family, destination
schema, stream, or scratch-shape requirement is incompatible. The builder
raises before writing; it never resizes, reallocates,
moves, or casts a supplied workspace. The MVP workspace is non-resizable.
Allocate a new one for a new signature, and close an old workspace only while
it is idle.

One workspace is non-reentrant, not thread-safe, and bound to one execution
stream. The builder acquires one exclusive execution lease for the full call.
CPU execution uses its synchronous execution domain; accelerator reuse is
accepted only on the same bound stream, where stream order protects scratch
reuse. Use one workspace per concurrent worker or stream. Cross-stream event
fencing, leased workspace pools, and synchronization fallbacks are deferred;
the builder must not hide a device-wide synchronization.

Workspace ownership does not shorten public output lifetime. A functional
result remains valid until its ordinary references are released. A supplied
destination remains a read-only snapshot after return until its caller
explicitly reuses that destination, at which point previous consumers must be
finished. Overlapped execution uses caller-owned output ping-pong: build into
destination A, consume A, build into B, and reuse A only after its consumer is
complete. Same-stream order is sufficient; cross-stream producer/consumer
events remain outside the MVP. Public fields that coexist in one result use
distinct storage and are never ping-ponged through one shared payload.

Private count-domain ping-pong swaps references among exact contiguous buffers
with the same ordered shape, dtype, and device; it never permutes or copies
data. A different axis order uses a different workspace. Integer count,
floating charge/waveform, and integer ADC values occupy separate storage
classes.

The warmed mode promises only *steady-state TensorDSLab-managed tensor-storage
allocation-free* execution. After output and workspace preparation plus
backend warm-up, no TensorDSLab target or named scratch tensor allocates new
storage for a compatible call. The phrase does not claim the absence of small
Python records, tensor views, allocator bookkeeping, backend plan/cache
initialization, or opaque PyTorch/CUDA-library scratch. `out=None`, a missing
workspace, first use, a changed signature, or workspace replacement is not
allocation-free. A stronger backend-wide zero-allocation claim requires
memory-instrumented evidence for every supported kernel.

No hidden permutation, `.contiguous()`, clone, cast, movement, copying reshape,
or fallback allocation is permitted after warmed preflight. Preparing an
incompatible semantic source is a visible one-time operation outside the
repeated loop.

The logical charge schedule freezes a post-dark-count source, accumulates the
total response separately, and reuses a contribution role for crosstalk and
afterpulses before smearing into the public charge target. The exact physical
scratch inventory and any safe fusion remain implementation decisions. A
compiled immutable execution plan, partial-output plans, cross-stream/event
pools, workspace resizing, CUDA Graph capture, and backend-wide zero
allocation also remain open.

## Device And Dtype Policy

TensorDSLab is GPU-oriented from the first production path:

- tensor transforms operate on the input device;
- no production hot path converts tensors to Python lists or NumPy arrays;
- no implicit CPU round trip is allowed;
- output device or dtype mismatch is a hard error;
- configs contain scientific values, not hidden placement policy;
- explicit collection movement uses TensorCore movement followed by semantic
  reconstruction.

Every field in one collection uses the same exact device and `torch.strided`
layout, but field dtypes may be mixed across semantic domains. Stage 2 fixes
the collection dtypes:

- photoelectrons use exactly `torch.int64`;
- charge, pure, noise, and analog use `torch.float32` or `torch.float64`, with
  one exact common floating dtype across every such field present;
- digitized ADC counts use exactly `torch.int32`.

A noncontiguous strided tensor remains valid collection structure; sparse and
other non-strided tensors do not. The collection foundation accepts fields
already resident on any PyTorch device when their devices match. CPU tests are
mandatory and structural CUDA tests run when available. Later transform work
orders must name their actual execution-backend matrix; structural acceptance
does not promise that every scientific kernel supports every PyTorch backend
or storage profile. General semantic construction does not make an
internally-overlapping read-only tensor writable; writable destination and
scratch boundaries impose their own nonoverlap rules.

## Boundary Validation

The `ReadoutCollection` constructor should validate once:

- exact TensorCore primitive types;
- a nonempty subset of recognized field IDs in filtered canonical order;
- the exact same ordered layout for every present field;
- `shared_axes` equal to all common-layout axis IDs in layout order, including
  `READOUT_EXAMPLE_AXIS_ID`, `READOUT_CHANNEL_AXIS_ID`, and
  `READOUT_SAMPLE_AXIS_ID`;
- exact `ExampleId` coordinates on the example axis, exact `ChannelId`
  coordinates on the channel axis, and count-only sample
  semantics;
- tensor/layout rank and shape agreement across fields;
- one common device and `torch.strided` layout, without requiring contiguous
  strides;
- exact `torch.int64` photoelectrons, exact `torch.int32` digitized counts, and
  one common `torch.float32` or `torch.float64` dtype across all present
  charge, pure, noise, and analog fields;
- typed sample-grid values, including a nonnegative containing-grid offset;
- conditional `DigitizedWaveformSpec` presence and derived ADC bounds;
- field-role-specific dtype, unit, and value-domain constraints;
- an immutable `SampleGrid` and conditional immutable
  `DigitizedWaveformSpec`;
- allowed extra-axis policy.

Hot transforms may trust those constructed records and perform only narrow
operation-specific checks. They should not rebuild full layout maps or recurse
through the entire product graph on each call. A stochastic-transform preflight
additionally rejects any count-only extra axis and derives the fixed RNG
coordinate sequence from the validated ID-backed axes.

Warmed preflight is a separate lightweight structural gate over already-valid
records. It additionally proves sample-last position, contiguity of every
participating tensor, internal nonoverlap and storage disjointness for writable
buffers, the exact workspace/output signature, stream, and exclusive lease.
Failure occurs before RNG consumption or writes and leaves source, output,
workspace generations, and RNG state unchanged.

## TensorCore Coordination Items

These items are observations for possible TensorCore Design work. They are not
accepted TensorDSLab dependencies:

1. `TensorAxisSelection` should materialize iterable indices before checking
   non-emptiness; an empty generator currently bypasses the intended check.
2. `require_compatible_collection(...)` should be reviewed for
   `TensorCollection` subclass inputs while continuing to compare only generic
   collection structure. TensorDSLab still needs field-scoped result
   validation because whole-collection compatibility cannot express retained
   structural sharing plus one replaced target field.
3. Bulk coordinate-to-index resolution may be useful if multiple consumers
   repeat the same exact-ID validation and lookup loops.
4. A generic tensor-geometry compatibility helper may be useful if
   cross-product operations repeatedly need layout/shape/device/dtype checks
   without matching field identity.
5. Dtype-aware or non-blocking collection movement should be considered only
   after downstream use demonstrates a shared need.
6. Auto-generated dataclass equality for `TensorField` and
   `TensorCollection` reaches elementwise PyTorch tensor equality and may raise
   for multi-element tensors. TensorCore should consider an explicit equality
   policy or `eq=False`; TensorDSLab meanwhile compares schema components,
   record identity, and tensor values explicitly rather than overriding the
   generic primitive locally.

TensorDSLab should proceed against the existing TensorCore baseline and raise a
focused TensorCore work order only when one of these items blocks or duplicates
real implementation.

## Non-Goals

- No TensorCore API copy inside TensorDSLab.
- No local compatibility aliases for retired TensorCore names.
- No semantic `TensorField` subclasses.
- No per-product `TensorCollection` subclasses for readout fields.
- No generic `Product` base.
- No materialized intermediate dark-count, crosstalk, afterpulse, or aggregate
  avalanche-count product between photoelectrons and charge.
- No fixed global tensor rank or axis order.
- No units, ranges, or bin edges encoded in IDs.
- No durable product labels collapsed into field IDs, even where their string
  payloads coincide.
- No hidden device movement, dtype casting, detachment, or mutation policy.
- No hidden workspace cache, implicit workspace resize, cross-stream workspace
  reuse, or scratch-backed public result.
- No hidden normalization or fallback allocation inside warmed execution.
- No universal TensorDSLab axis order; sample-last belongs only to the MVP
  temporal readout execution profile.
- No sparse or other non-`torch.strided` readout tensor layout in the MVP.
- No concrete cache, DAG, TensorML API, or upstream-adapter API in this page.

## Return To Design Before

- replacing the one-`ReadoutCollection` model with per-product collection
  subclasses or another wrapper hierarchy;
- changing recognized field IDs, canonical field order, or descendant
  invalidation rules;
- changing any canonical readout axis ID, its coordinate class or count-only
  semantics, or the fixed stochastic coordinate ordering;
- representing strided or irregular sample selections as `ReadoutCollection`;
- requiring a new TensorCore public API;
- allowing stochastic identity to depend on transient tensor order;
- weakening coordinate/index separation;
- allowing implicit movement or casting;
- accepting a non-`torch.strided` readout tensor layout;
- requiring contiguous tensors in the general semantic collection contract;
- accepting a warmed noncontiguous/stride-aware profile without a focused
  measured contract;
- changing functional versus output-buffer gradient behavior;
- allowing a returned field to alias `ReadoutWorkspace` scratch;
- adding implicit workspace caching, resizing, movement, cross-stream reuse,
  or a stronger allocation-free claim;
- changing the warmed sample-last/contiguous profile; or
- fixing one global TensorDSLab rank or axis order.
