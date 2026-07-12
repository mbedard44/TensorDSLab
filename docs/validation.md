# Validation

Validation should prove scientific behavior, public contracts, and ownership
boundaries. Tests should not mirror private implementation structure.

## Current Package Baseline

Stage 2 is Merged / Closed on `main` at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. The production package metadata,
structural package, focused test suite, and exact TensorCore runtime dependency
are accepted there. Stage 2 accepts no scientific transform, workspace, cache
schema, or later integration surface.

Maintenance 1 changes only readout public-name and module ownership. Its
feature-branch form is candidate evidence before fixed-commit Validation,
independent Review, and Review's clean fast-forward; if the updated surface is
read on `main`, those gates have completed. This status makes no scientific,
compatibility, deployment, conformance, or GPU claim.

Documentation-only Design changes remain in the Design thread unless the user
requests independent Validation or Review. Run:

```bash
git diff --check
```

Also run targeted checks for:

- local Markdown links;
- duplicate or malformed headings;
- stale product names and retired architecture alternatives;
- accidental package, test, cache, DAG, or generated-file additions;
- consistency among architecture, design, decisions, parity, validation, and
  stage documents.

## Governance Adoption Checks

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`.
Conformance remains unevaluated. Validate the
[governance index](governance/index.md),
[adoption record](governance/adoption_0_1_0.md),
[overlay](governance/overlay.md),
[semantic rule map](governance/rule_map_0_1_0.md), and `docs/decisions.md`
against all of these checks:

- verify the Governance Core manifest-file SHA-256
  `45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8`
  and all eight entries;
- verify the Council Charter manifest-file SHA-256
  `343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916`
  and all three entries;
- verify governed Design base
  `151b61fdc36475498219ee5fe7b045a3a72c2d09`, exact accepted candidate
  `d634401a853915edeb4f83df4a4943b3553deced`, its exact parent, and its
  authorized nine-path scope;
- confirm that `TDSLAB-GOV-D001` is issued once and records its issuance-time
  state accurately: package adoption `Adopted`, conformance `Not evaluated`,
  Coordination `Deferred`, Profile B `Disabled`, and Stage 2 Design-complete
  and undispatched at that historical gate;
- map `OP-01` through `OP-13` and `ENG-01` through `ENG-12` exactly once, with
  every required field, a resolvable package source and section, baseline,
  rationale, and evidence;
- retain six `Adopted` and 19 `Stronger local rule` dispositions, no
  whole-rule Not-applicable disposition, and no accepted deviation;
- require exact absence evidence and a focused activation trigger for every
  dormant package surface;
- limit Phase 3 closeout to `docs/decisions.md` and existing governance,
  workflow, onboarding, overview, and validation records that must state the
  adopted package state; do not change architecture, parity,
  implementation-stage records, production code, metadata, tests, routes,
  registries, or caches;
- confirm that durable files contain no raw task identifier and that no
  `.agents`, route, registry, or cache state was created;
- reject unsupported claims of conformance, Active routing, enabled Profile B,
  deployability, release readiness, backward compatibility, broad
  compatibility, or implemented integration;
- treat compatibility evidence as exact-baseline evidence only, and do not
  treat same-GPU residency or no-silent-host-materialization Design constraints
  as cross-package compatibility evidence;
- qualify the external working dossier's `Not adopted` line as a pre-decision
  snapshot without editing the sibling repository; and
- record the self-referential Phase 3 closeout commit in the external
  completion report.

At the documentation-only governance-adoption gate, runtime, import,
environment, integration, and post-merge test commands were dormant because
the corresponding package surfaces did not exist. The Stage 2 work order
activates package, import, environment, and required post-merge verification
for its exact foundation. Integration and later runtime or deployment checks
remain dormant until their focused Design-owned work orders. Passing the
governance documentary checks alone implies no installability, deployability,
compatibility, or production conformance.

## Validation Layers

Future validation should follow the boundary-first model:

```text
external/config or future TensorG4DS boundary values
  -> constrained values and TensorDSLab IDs
  -> TensorCore axes/layouts/fields/collections
  -> TensorDSLab ReadoutCollection snapshots
  -> focused transforms
  -> future examples and durable boundaries
```

Constructors and public boundary builders validate the full invariant-bearing
record. Hot transforms trust those records and perform narrow
operation-specific checks.

## Package And Dependency Checks

Stage 2 established these package and dependency checks; retain them as
regression requirements:

- project folder `TensorDSLab` and import package `tensor_dslab`;
- concrete packages such as `tensor_dslab.common` and
  `tensor_dslab.readout` live directly under the import root, with no
  intermediate `tensor_dslab.domain` namespace;
- deliberate root package exports;
- absolute `tensor_dslab.*` production imports;
- TensorCore imports only from public root `tensor_core`;
- no TensorCore implementation-module imports;
- no local re-export or copy of generic TensorCore helpers;
- no production imports from TensorML, DSLab, IV-DSLab, Projects/dag, or
  G4DS/g4ds11; a future focused TensorDSLab integration adapter may import only
  the exact accepted public TensorG4DS type, while common/readout remain
  TensorCore-only;
- no placeholder package modules;
- import isolation from deferred cache, DAG, and downstream dependencies.

## TensorCore Contract Checks

Tests should prove:

- TensorDSLab coordinate IDs subclass open TensorCore `Id`;
- the primary `ReadoutCollection` subclasses open `TensorCollection`;
- no sealed TensorCore primitive is subclassed;
- `IdSequence` rejects empty, duplicate, base-`Id`, and mixed-class values and
  preserves caller order;
- coordinates resolve through `TensorLayout` before becoming indices;
- count-only positions are not reported or persisted as semantic IDs;
- field mapping order is explicit and preserved;
- collection compatibility and value assertions compare explicit schema
  components and use `torch.equal`/`torch.testing` for payloads rather than
  relying on inherited dataclass `==`;
- generic TensorCore operations return base collections and TensorDSLab
  reconstruction restores `ReadoutCollection` only at an accepted semantic
  boundary;
- reconstruction helpers explicitly carry, update, or prune `SampleGrid` and
  conditional `DigitizedWaveformSpec` values from the semantic source;
  canonical axes are recovered from their exact public IDs in the transformed
  layout, and a base collection or free-form metadata alone is insufficient to
  recover subclass sidecars;
- contiguous increasing unit-stride sample selections advance sample-grid
  origin and offset during reconstruction, while arbitrary/reordered/strided
  selections cannot be relabeled as `ReadoutCollection`;
- public operations do not depend on unaccepted TensorCore subclass
  preservation;
- `ReadoutCollection` does not encode units, ranges, mutation, RNG, or durable
  product labels in TensorCore IDs.

## Readout Collection Construction Matrix

`ReadoutCollection` recognizes exactly six semantic fields:

| Exact field ID | Required meaning |
| --- | --- |
| `readout.photoelectrons` | finite nonnegative integer counts of binned photon-origin primary PE seeds |
| `readout.charge` | finite nonnegative floating aggregate PE-equivalent response amplitudes, not SI charge |
| `readout.waveform.pure` | finite floating signal-only mV values |
| `readout.waveform.noise` | finite floating noise/baseline mV values |
| `readout.waveform.analog` | finite floating composed pre-digitization mV values |
| `readout.waveform.digitized` | bounded nonnegative integer ADC counts |

Construction tests should prove:

- any nonempty subset of the six recognized fields is accepted;
- an empty collection or any unrecognized field ID is rejected;
- direct construction rejects any field mapping that is not already in the
  canonical topological order filtered to the fields present;
- boundary builders and transform-result helpers emit canonical order rather
  than preserving arbitrary external mapping order;
- every field has the same exact `TensorLayout` and device and uses dense
  `torch.strided` layout;
- noncontiguous `torch.strided` tensors are accepted;
- exact `EXAMPLE_AXIS_ID`, `CHANNEL_AXIS_ID`, and
  `SAMPLE_AXIS_ID` axes are required and shared in arbitrary layout
  order;
- canonical axis constants have exact `example`, `channel`, and `sample`
  values, use value equality, and resolve freshly constructed equal IDs;
- `REQUIRED_AXIS_IDS` is an exact `IdSequence` containing those three exported
  constants in example/channel/sample order;
- `ReadoutCollection.__module__` is exactly
  `tensor_dslab.readout.types`, while `tensor_dslab.readout.tensors` retains
  only the semantic reconstruction helpers;
- the four retired prefixed axis names are absent from package exports and
  `__all__`, and `types`, `validation`, `tensors`, `builders`, and the readout
  package root import independently in fresh processes;
- every accepted extra axis is common and shared by every present field;
- `shared_axes` contains every common-layout axis ID in exact layout order;
- example and channel axes are ID-backed by exact `ExampleId` and shared
  `ChannelId` coordinates imported from `tensor_dslab.common`;
- readout and future reconstruction use the same exact `ChannelId` class rather
  than domain-specific aliases or conversion IDs;
- sample is count-only and follows the accepted typed sample-grid contract;
- positive sample period, finite origin, and nonnegative sample offset are
  validated by typed sidecars;
- photoelectrons require exact `torch.int64`; all present floating fields share
  exact `torch.float32` or exact `torch.float64`; digitized ADC counts require
  exact `torch.int32`;
- layout/tensor shape agreement and each field's value domain are validated;
- `DigitizedWaveformSpec` is required exactly with the digitized field, derives
  its ADC maximum, accepts bit depth 1 through 16, inclusive gain 0 through 40
  dB, and `AdcQuantization.TRUNCATE`, survives digitized-only projection, and
  is absent whenever that field is absent;
- typed sidecars are structurally immutable and essential semantics are not
  available only through free-form metadata;
- construction is placement-neutral across PyTorch devices, requires one exact
  common device, has mandatory CPU coverage, and exercises CUDA conditionally
  when available without promising later-kernel CUDA support;
- general construction accepts noncontiguous and expanded/internally
  overlapping `torch.strided` read-only values without treating storage layout
  as scientific identity;
- there is one `ReadoutCollection` class and no semantic collection subclass
  per field.

## Semantic Shape And Execution-Profile Checks

General collection behavior and functional operations that claim semantic
layout-order independence should be tested with at least two valid axis orders,
for example:

```text
(example, channel, sample)
(sample, example, channel)
```

Tests should prove that:

- semantic results agree after resolving the exact canonical axis IDs;
- extra axes remain present and ordered;
- channel coordinates still identify the same detector channels;
- no operation assumes a singleton leading batch dimension;
- no operation quietly broadcasts across a missing or incompatible axis.

Semantic construction should also accept representative noncontiguous and
expanded/internally overlapping read-only `torch.strided` source views while
rejecting sparse/non-strided layouts. That acceptance never makes such storage
writable.

Warmed `out + workspace` tests use the separate strict readout profile:

- sample is last and every participating source, generated public target, and
  scratch tensor is contiguous;
- writable tensors are internally nonoverlapping and pairwise
  storage-disjoint;
- `(example, channel, sample)` and `(channel, example, sample)` both work with
  separate exact workspaces and agree semantically after axis-ID resolution;
- sample-first, noncontiguous, expanded writable, wrong-order-workspace, and
  incompatible destination cases fail before RNG, writes, or workspace reset;
- preflight never silently permutes, calls `.contiguous()`, clones, casts,
  moves, reshape-copies, or allocates fallback storage; and
- kernels flatten leading axes only after the strict profile has been proved.

Ordinary `out` without a workspace carries no allocation-free claim. Each
transform work order must name whether it supports a noncontiguous writable
target or explicitly allocates normalization; every writable target is
nonoverlapping regardless.

## Future TensorML Projection Checks

The collection is the primary tensor handoff, but its Python class is not a
complete model schema. A future adapter stage should prove:

- the explicit `TensorFieldSelection` order is the positional model-argument
  ABI;
- model-facing projection selects the exact intended subset in that order;
- a noncanonical model selection remains a base `TensorCollection` and is not
  reordered merely to reconstruct `ReadoutCollection`;
- adding a field to a source `ReadoutCollection` does not silently add a model
  argument when an explicit projection is used;
- generic TensorCore selection and batching return base `TensorCollection`
  records rather than preserving `ReadoutCollection` identity;
- projection before movement transfers only the requested field tensors, and
  semantic reconstruction occurs after that movement only when required;
- any semantic reconstruction happens in TensorDSLab or a focused adapter, not
  through an assumed TensorCore subclass-preservation contract;
- `ReadoutCollection` class compatibility is never treated as proof of exact
  model field IDs or positional order;
- stock-loop models receiving selected base collections use
  `input_type = TensorCollection` unless an accepted adapter reconstructs a
  canonical semantic collection before `forward_pass`;
- model outputs claiming `ReadoutCollection` pass TensorDSLab field/schema
  construction checks rather than relying only on TensorML's class check.

No current TensorML `input_fields` or `output_fields` API change is required.
Revisit that possibility only in a focused downstream-integration design if
explicit projection proves insufficient.

## Functional And Output-Buffer Checks

Every transform should test both call modes.

For `out=None`:

- a fresh `ReadoutCollection` snapshot is returned;
- exactly one semantic target field is added or replaced;
- all materialized descendants reachable from a transform-added or replaced
  target field are transitively absent;
- independent fields such as `readout.waveform.noise` remain valid when only
  an upstream charge value changes without changing layout or sample grid;
- every retained field is the same `TensorField` record and tensor object as in
  the source collection;
- transforms do not mutate retained tensor payloads, although PyTorch tensors
  are not intrinsically immutable;
- public collection and transform APIs expose no general in-place field update;
- callers are required to treat materialized field tensors as read-only, and a
  manual in-place PyTorch edit is documented as outside the value-object
  contract because it bypasses descendant invalidation;
- source structure, field values, sidecars, and layouts are unchanged;
- result sidecars, canonical filtered field order, and exact common layout are
  correct;
- differentiable deterministic transforms remain in the autograd graph.

Dependency-invalidation tests should cover at least:

```text
transform-add or replace readout.photoelectrons
  -> remove readout.charge, pure, analog, and digitized
  -> retain independent noise
transform-add or replace readout.charge
  -> remove pure, analog, and digitized
  -> retain independent noise
transform-add or replace pure or noise -> remove analog and digitized
transform-add or replace analog -> remove digitized
```

Projection and explicit field removal should be tested separately. They may
retain descendants because removing an ancestor without changing retained
values does not make those descendants stale.

For supplied `out`:

- the exact supplied destination is returned;
- every collection, config, workspace, lease, device, stream, gradient, and
  alias check completes before any RNG work or tensor write;
- every preflight failure leaves the destination and all source tensors bitwise
  unchanged, consumes no semantically relevant RNG work, and releases any
  workspace reservation;
- a field-scoped allocator zero-initializes the target in its valid value
  domain before collection construction;
- a public uninitialized `torch.empty_like` target cannot create an invalid
  pre-fill `ReadoutCollection` state;
- its field set is exactly the valid retained source fields plus the target
  after transitive invalidation;
- its field order is the canonical order filtered to that exact field set;
- its field-specific sidecars exactly match the expected result, including
  addition, retention, or removal of `DigitizedWaveformSpec` with its field;
- non-target fields are the same structurally shared records expected from the
  functional result;
- only the target tensor is written, and every target element is overwritten
  completely rather than depending on its prior valid value;
- wrong collection class, field set, sidecars, canonical axes, sample grid,
  layout, shape, device, target dtype, or non-`torch.strided` layout fails;
- the transform never replaces, moves, casts, or detaches the destination
  target;
- the writable target and every live workspace scratch tensor are internally
  nonoverlapping, mutually disjoint, and disjoint from every source tensor,
  retained field tensor, and non-target output tensor;
- transforms do not mutate retained tensors through shared storage;
- no non-target field is writable through the public transform API;
- gradient-sensitive use of a non-autograd-safe buffer path fails clearly.

Full-overwrite tests should fill the otherwise valid target with a
role-compatible sentinel chosen not to equal the expected output, run the
transform, compare every element with a functional or independent reference,
and prove no unwritten sentinel remains. Preflight-atomic tests should snapshot
that sentinel target, induce every accepted failure class, and prove the target
is untouched. A failed preflight followed by a valid stochastic call must equal
the corresponding clean call, proving that failure consumed no semantic RNG
work.

`out=` alone does not promise zero internal scratch allocation. The narrower
warmed steady-state allocation claim below requires both an exact supplied
destination and an exact reusable `ReadoutWorkspace`.

Memory-policy tests should prove that domain projection structurally shares
only requested fields, moving the projection does not move omitted fields, and
transforms never evict an unrelated valid field merely to reduce memory.

## Result Builders And `ReadoutWorkspace` Checks

Public `build_readout_result_buffer(...)`, public
`build_readout_output_buffer(...)`, and the later
`build_readout_collection(...)` scheduler must use one semantic field and
sidecar registry. For every public transform, representative partial source
collection, and target replacement/addition, tests should prove:

- the atomic result buffer derives exactly the transform's complete
  post-invalidation field set and required source fields;
- the full output buffer derives exactly the configured full-chain field set
  and photoelectron-retention/replacement policy;
- both use canonical order, correct retained-field identities, field-specific
  sidecars, shared layout, canonical axes, sample grid, device, tensor layout,
  and field dtype, and neither returns a target-only collection;
- every newly allocated target is contiguous in the existing semantic axis
  order, while retained noncontiguous fields preserve exact record/storage
  identity and are never normalized by the factory;
- caller mapping order cannot alter the canonical result order;
- workspace-backed execution obtains private scratch from the workspace while
  public target storage still comes only from the exact caller-owned
  destination; it does not change product meaning or descendant invalidation;
- an instrumented preparation trace orders schema derivation and all validation
  before workspace reset, RNG work, scratch reads, or target writes;
- builder failure leaves source, destination, workspace generations, and RNG
  behavior unchanged.

Full-chain builder tests should additionally prove:

- `readout.photoelectrons` is required and the fixed configured operation order
  matches explicit atomic composition;
- the result contains photoelectrons, charge, pure, noise, and analog, plus
  digitized exactly when configured, in canonical order;
- disabled timing retains the exact source photoelectron record, while enabled
  timing uses a fresh nonaliasing public target;
- pre-existing derived source fields are recomputed rather than trusted;
- exact stage-specific views preserve each atomic transform's one-target
  contract, and final assembly clones no field tensor;
- `out=None, workspace=None`, `out=destination, workspace=None`, and
  `out=destination, workspace=compatible` agree under the accepted exact,
  numerical, or stochastic reproducibility contract;
- workspace-without-`out` fails during preflight;
- a destination with the wrong full field set, coordinate maps, sidecars,
  device, or dtypes is rejected rather than rebuilt or resized.
- warmed execution rejects sample-not-last or noncontiguous participating
  source/destination tensors rather than normalizing them.

`ReadoutWorkspace` is explicit reusable runtime state, not a collection field,
sidecar, ID, product label, or scientific config. Workspace tests should prove:

- compatibility is exact for the accepted operation, sample-last common layout
  and shape, contiguous participating storage, device,
  floating/count/ADC dtypes, destination schema, stream, scratch roles, and
  configured capacity; no arbitrary stride tuple is needed;
- compatible same-shaped sources with different coordinate values may reuse
  scratch only when ordered axis IDs, sizes, and algorithm signature match;
  canonical positions are derived from that axis order, and the current source
  still supplies every RNG coordinate;
  public destinations remain exact to their own full coordinate maps;
- a mismatch fails during preflight; the hot path never resizes, replaces, or
  silently reallocates a workspace buffer;
- after construction/reservation and warmup, every target and scratch storage
  pointer plus its allocation-generation identifier remains stable across
  compatible calls and resets;
- reset clears logical readiness without reallocating storage: a scratch slot
  cannot be read until the current operation has completely written it and
  advanced its write generation;
- poisoning scratch with distinct values before reset, then running zero,
  sparse, and dense cases in different orders, produces the same result as a
  fresh functional call and detects any read-before-write or stale accumulation;
- target, source, retained, and simultaneously live scratch storages are pairwise
  disjoint; storage-sharing views are rejected even when they are different
  tensor objects;
- only one exclusive, nonreentrant lease exists per workspace, and it is bound
  to the acquiring CUDA stream when the device is CUDA;
- a second host call, nested call, or different-stream call using that workspace
  fails before RNG work or mutation; preflight failure releases a provisional
  lease cleanly;
- independent workspaces with disjoint destinations may execute concurrently,
  including on separate CUDA streams, and agree with serial reference results;
- private ping-pong slots are reused only after their last reader completes;
  reuse swaps references without copying/permuting and never crosses ordered
  axis/shape, device, dtype, contiguous-layout, or capacity classes;
  `simulate_charge` keeps its frozen post-dark source live until both crosstalk
  and afterpulse contributions have consumed the same generation;
- workspace reset and private scratch reuse never change a previously returned
  public output, because reusable scratch is disjoint from caller-owned output
  storage;
- a caller-owned `out` remains valid and unchanged across unrelated workspace
  reuse and expires as a stable prior result only when the caller explicitly
  resubmits that same destination for another write;
- no workspace automatically recycles storage backing an ordinary returned
  `ReadoutCollection` merely because another call begins.

On CUDA, successful completion means completion according to normal stream
ordering. Same-stream consumers are ordered after the producer; a caller moving
consumption to another stream must use the accepted explicit synchronization
boundary. Concurrent observation while a target is being written is outside the
contract. The MVP guarantees that preflight failures do not launch work or
modify `out`, and that a successfully completed operation fully defines its
target. Transactional rollback after an asynchronous kernel, device, library,
or allocator failure is deferred; an affected destination/workspace generation
must be discarded or explicitly reset according to the future implementation
work order.

Autograd tests must keep the modes separate. `out=None` preserves the accepted
gradient graph for deterministic differentiable transforms and is checked with
value and gradient references. Any gradient-sensitive supplied-`out` or
workspace-buffer call fails during preflight, before lease acquisition becomes
observable, RNG work begins, or tensors change. A reusable workspace must not
justify an implicit `no_grad`, detach, copy, or weakened functional-path
gradient contract.

### Workspace Allocation Instrumentation

The hard allocation claim is deliberately narrow:

> After explicit construction/reservation and warmup, repeated compatible
> transforms supplied with both the exact caller-owned `out` and exact
> `ReadoutWorkspace` perform no TensorDSLab-owned target or scratch
> tensor-storage allocation.

Tests for this claim should:

- instrument every TensorDSLab target/scratch allocation door with an allocation
  counter and generation identifier;
- record all target and scratch storage pointers after warmup;
- execute repeated zero, sparse, dense, and alternating-transform workloads;
- assert no allocation-counter increment, generation replacement, pointer
  change, implicit capacity growth, or steady-state live-memory growth after
  synchronization;
- instrument/forbid permutation materialization, `.contiguous()`, clone,
  copying reshape, cast, movement, and fallback target/scratch allocation after
  warmed preflight;
- verify that builder/validation Python records, backend kernels, and temporary
  library state are not mislabeled as TensorDSLab-owned tensor storage.

CUDA/PyTorch allocator statistics and memory-profiler traces should be retained
as diagnostic regression evidence. They are not a portable hard promise of zero
backend allocation requests. PyTorch caching-allocator behavior, CUDA runtime
state, FFT/convolution/library plans and private scratch, kernel compilation,
exact reserved-memory values, and asynchronous-failure rollback are deferred
backend/library concerns.

## Device Checks

Collection construction is placement-neutral. Stage 2 required CPU behavior
and conditional CUDA collection checks when CUDA was available; those checks
do not promise that every later scientific kernel supports CUDA. Its post-merge
verification passed the CPU suite, while three conditional CUDA tests were
skipped because CUDA was unavailable, so no GPU behavior is claimed. Each
transform work order must name and test its own execution-device matrix.
Applicable tests should prove:

- outputs stay on the input or supplied destination device;
- device mismatch fails instead of moving silently;
- all fields in one collection share device and use `torch.strided` layout,
  while noncontiguous strided tensors remain valid semantic structure and the
  warmed profile separately requires contiguous sample-last storage;
- photoelectrons use exact `torch.int64`; `readout.charge`, pure, noise, and
  analog fields share exact `torch.float32` or exact `torch.float64`; digitized
  waveforms use exact `torch.int32`;
- a target field's dtype mismatch fails instead of casting silently;
- production hot paths do not call `.cpu()`, `.numpy()`, `.tolist()`, or import
  NumPy for hidden reference execution;
- available accelerator results satisfy the declared exact, numerical,
  distributional, or statistical acceptance criteria;
- optional accelerator tests skip with a clear reason when hardware is absent.

Explicit one-time preparation of a non-ready collection must preserve semantic
axis IDs/coordinates and remain on the selected device. It must occur outside
the warmed repeated-call instrumentation window. No test should infer one
universal axis order for readout, reconstruction, and TensorML.

## Deterministic RNG Checks

Stochastic tests should separate probability-kernel correctness from sampled
fixture behavior.

Required invariants:

- the same source field snapshot, physics config, RNG seed, namespace,
  operation role, and semantic coordinates repeat exactly on the same backend;
- changing an accepted semantic coordinate or namespace changes the random
  field;
- the coordinate payload order is example, channel, every additional
  ID-backed shared axis in lexical `axis_id.value` order, then
  `sample_offset + local_sample_index`, independent of tensor layout order;
- lexical RNG-key ordering does not reorder tensor axes or coordinate
  `IdSequence` values;
- channel reordering does not change values associated with a channel ID;
- accepted ID-backed batching and chunking do not change per-coordinate
  results;
- unrelated batch members do not perturb existing values;
- sample identity uses `sample_offset + local_sample_index` because the sample
  axis is count-only;
- sample slicing advances physical origin and sample offset consistently;
- only contiguous increasing unit-stride sample selections reconstruct a
  `ReadoutCollection`; irregular selections remain base collections;
- channel identity uses channel coordinates, not channel indices;
- a stochastic transform rejects a collection with an additional count-only
  axis until that axis has an accepted typed containing-grid offset contract;
- collection construction exposes no configurable stochastic-axis membership
  list;
- no global RNG state or call-order dependency is observable;
- zero-effect configs consume no semantically relevant random draws;
- zero-effect configs still use the standard full-result field schema and
  descendant invalidation;
- no durable per-quantum identity is introduced merely to drive sampling.

CPU/GPU bitwise identity is not required until an RNG work order accepts one
algorithm that can provide it. Cross-backend distributional agreement with
accepted probability kernels is required, with finite-sample statistical
validation as evidence. Donor comparisons use the classification rules in
[IV-DSLab Parity](parity.md).

## Photoelectron And Charge-Simulation Checks

The public field boundaries are intentionally narrower than the internal
scientific submodels:

```text
apply_timing_jitter: readout.photoelectrons -> readout.photoelectrons
simulate_charge:     readout.photoelectrons -> readout.charge
```

`readout.photoelectrons` contains binned photon-origin primary PE seeds. Dark
counts, crosstalk avalanches, afterpulses, and their aggregate count grids exist
only inside `simulate_charge`. Focused tests may observe those private
submodels through accepted reference helpers or diagnostic hooks, but they must
not turn an intermediate avalanche/count grid into a seventh collection field,
public product, durable label, or required sidecar.

### Photoelectrons

Construction and boundary tests should prove:

- the field is exactly `readout.photoelectrons`;
- values are finite nonnegative integers;
- each input quantum represents a binned photon-origin primary PE seed;
- dark counts, crosstalk, afterpulses, charge spread, and analog response are
  absent from this field's meaning;
- the field is not described as electrical charge or an SI-charge measurement;
- native G4DS parsing and TensorG4DS low-level analysis remain permanently
  outside TensorDSLab; the typed TensorG4DS handoff and photoelectron binning
  remain deferred from the post-binned MVP.

### Timing Jitter

Tests should prove:

- the required input and replaced target are both
  `readout.photoelectrons`;
- zero sigma is exact target-field identity;
- latent uniform sub-bin phase is reflected in probability buckets;
- aggregate target plus drop probabilities sum to one within named tolerance;
- out-of-range migrated counts are dropped;
- output remains finite nonnegative integers;
- counts are conserved when the dropped bucket is included;
- no stale nearest-bin policy affects behavior.

### End-To-End `simulate_charge`

Tests should prove:

- the only required collection field is `readout.photoelectrons` and the only
  semantic target is `readout.charge`;
- the input photoelectron field is retained unchanged while stale charge
  descendants are invalidated in the result snapshot;
- private execution order is dark-count addition, one frozen post-dark source,
  parallel crosstalk and afterpulse contributions from that source, addition of
  each contribution exactly once, and final aggregate charge smearing;
- neither generated contribution feeds the other or recursively becomes a new
  source in the MVP;
- intermediate count tensors remain private, ephemeral, device-resident scratch
  and do not escape through the returned collection or metadata;
- no-effect settings reduce to an exact integer-to-floating conversion from
  photoelectron counts to PE-equivalent response amplitudes and consume no
  semantically relevant random draws;
- output values are finite nonnegative floating aggregate PE-equivalent
  responses and are never documented as Coulombs or another SI-charge unit;
- same-backend reproducibility, coordinate/chunk invariance, field-scoped
  `out=` behavior, descendant invalidation, and source immutability satisfy the
  repository-wide transform contract;
- end-to-end ensembles satisfy the named charge mean, variance, zero-cell
  probability, occupancy, tail, and edge-loss tolerances accepted by the work
  order without implying equality of the complete IV-DSLab stochastic law.

Focused private-submodel checks below complement this public end-to-end test;
they do not define additional collection operations.

### Dark Counts

Tests should prove:

- `lambda = rate_hz * sample_period_ns * 1e-9`;
- zero rate is exact identity of the private source-count grid;
- counts are nonnegative integers;
- cells use independent semantic random fields;
- channel/sample shape and all extra axes are preserved;
- the addition remains an internal `simulate_charge` observable rather than a
  `ReadoutCollection` target.

### Crosstalk

Tests should prove:

- the first model is same-channel and same-sample;
- the mean coefficient is interpreted as a Poisson mean multiplier;
- zero mean is exact zero contribution;
- generated contributions do not recursively feed crosstalk;
- crosstalk and afterpulses read the same frozen source snapshot;
- `simulate_charge` adds the frozen source and each enabled private contribution
  exactly once.

### Afterpulses

Do not dispatch production afterpulse tests until Design resolves unit-count
versus recovery-weighted amplitude. Probability-kernel fixtures may still prove:

- exponential delayed-target buckets;
- no-fire probability;
- dropped-out-of-range delayed probability;
- same-channel, first-generation behavior;
- no target sample precedes the source sample.

The accepted implementation fixture must name the recovery policy explicitly
and classify the ordinary exponential delay as an intentional divergence from
the literal IV reciprocal-exponential expression. It must also prove that the
afterpulse count/amplitude representation remains internal to
`simulate_charge`.

### Charge Smearing

Tests should prove:

- one aggregate `Normal(n, sqrt(n) * sigma)` draw per populated cell;
- zero count remains zero;
- zero sigma converts counts exactly to floating amplitudes;
- negative draws clip to zero;
- smearing happens after integer existence effects;
- the private final count grid is consumed exactly once to produce the public
  `readout.charge` target;
- `readout.photoelectrons` remains the unchanged primary-seed source field and
  is not overwritten with internally generated avalanches.

## Waveform Checks

### Pure Waveform

Tests should cover:

- target field `readout.waveform.pure`;
- signal-only values at the same pre-digitization analog reference plane and in
  the same mV units used by noise and analog composition;
- sampled pulse-template values and normalization;
- finite positive rise/fall/sample-period/support parameters;
- causal convolution truncated to the input sample count;
- configured gain and sign order, with no baseline contribution;
- no implicit eventwise or phase-marginalized sub-bin correction;
- peak and area bias remain within the parity tolerance accepted by the
  production work order;
- functional-path gradients through the input charge tensor;
- functional layout/order independence and device residency, plus strict
  sample-last/contiguous acceptance and rejection coverage for warmed mode.

### Noise Waveform

Tests should cover:

- target field `readout.waveform.noise`;
- noise/baseline-only values at the same pre-digitization analog reference
  plane and in the same mV units used by pure and analog composition;
- exact constant baseline;
- white-noise RMS or explicit PE-amplitude/SNR derivation;
- containing-grid sample-ordinal chunking invariance for white noise;
- exact FFT bin count and frequency grid;
- DC and Nyquist endpoint policies;
- deterministic interior phases by semantic coordinates;
- inverse real FFT, normalization, and scaling order;
- rejection of quiet crop, pad, truncate, or resample behavior;
- zero-noise configs.

### Analog Waveform

Tests should prove:

- target field `readout.waveform.analog`;
- exact `pure + noise` composition;
- optional analog clipping occurs after the sum;
- pure and noise are same-plane simulation components rather than separate
  Tile, PDU, or DAQ hardware-boundary products;
- the result is the composed voltage at the fixed TensorDSLab digitization-input
  reference plane, before the gain recorded by `DigitizedWaveformSpec`;
- inputs require compatible canonical axes, grids, layouts, shapes, devices,
  dtypes, and mV semantics;
- functional-path gradients reach both inputs;
- no implicit broadcasting is accepted in the first implementation.

### Digitization

Tests should prove:

- dB gain factor `10 ** (gain_db / 20)`;
- voltage range derived from peak-to-peak and offset;
- gain, clamp, map, and quantize order;
- exact `AdcQuantization.TRUNCATE` behavior at boundary and half-step cases;
- ADC bounds, exact `torch.int32` output, bit depths 1 through 16, and inclusive
  gain boundaries 0 through 40 dB, with just-outside values rejected as the
  intentional donor validation correction;
- `DigitizedWaveformSpec` records the validated gain, voltage transfer, bit
  depth, and quantization policy and reconstructs with a digitized-only field;
- shape/layout/canonical-axis preservation;
- the target is the distinct `readout.waveform.digitized` field;
- the required analog input is `readout.waveform.analog`;
- pure and noise values are never digitized separately before composition;
- non-finite analog inputs fail.

## Parity And Donor-Fixture Rules

Golden fixtures should remain small, reviewable, and owned by TensorDSLab.
Tests must not import or execute DSLab or IV-DSLab at runtime.

[IV-DSLab Parity](parity.md) defines the comparison taxonomy, audited donor
baseline, accepted divergences, and operation-level claims. A golden fixture
does not by itself prove distributional parity.

Every promoted donor fixture should state:

- donor source path/symbol and version, commit, or audited snapshot identity;
- comparison boundary and input/config domain;
- parity classification, assumptions, exclusions, and intentional
  divergences;
- declared observables and exact/numerical/distributional/statistical
  acceptance criteria;
- units, axes, layout, dtype, and operation order;
- RNG algorithm or probability contract;
- seed, namespace, and semantic coordinates when sampled;
- edge and clipping policy;
- quantization rule;
- sample size and confidence/error threshold for statistical evidence;
- whether the fixture provides exact/numerical evidence or finite-sample
  support for a distributional/statistical claim.

Do not preserve apparent donor bugs, global state, unsigned wraparound,
condition-DB loading, remote downloads, fixed singleton batch shape, or
CPU-list conversions merely for literal parity.

## Future TensorG4DS Handoff Checks

The future cross-repository integration stage must test a nominally typed
public TensorG4DS boundary rather than a loose protocol over generic field
names. Validation should prove:

- the adapter accepts only the exact TensorG4DS product/version and validates
  its event context, source relationships, units, layout, dtype, and device;
- the accepted upstream position/time/energy units are explicit—currently
  centimetres, nanoseconds, and keV in TensorG4DS Design—and any conversion is
  documented and performed on-device;
- TensorG4DS `EventId` remains upstream provenance and maps explicitly to zero,
  one, or multiple TensorDSLab `ExampleId` values rather than being cast or
  reused;
- empty upstream events and zero-output mappings follow the accepted
  cross-repository policy without sentinel deposits, channels, or semantic
  IDs;
- `ChannelId` is assigned from an accepted TensorDSLab detector/channel map,
  never inferred from a TensorG4DS index or matching string;
- deposit/cluster axes are transformed into new example/channel/sample
  semantics; they are not relabeled as a `ReadoutCollection` layout;
- input payloads are already on the accepted GPU, remain read-only, and every
  newly computed TensorDSLab payload is created on that same device;
- the bridge performs no implicit `.cpu()`, `.numpy()`, list conversion,
  serialization/reload, movement, cast, detach, or hidden synchronization;
- any accepted unit or dtype conversion is an explicit documented on-device
  detector transform;
- initial discrete detector integration rejects gradient-sensitive inputs
  unless a focused differentiable contract is accepted, while never detaching
  silently; and
- TensorDSLab common/readout import isolation remains intact and TensorG4DS
  never imports TensorDSLab.

These are integration gates, not claims about TensorG4DS's current
documentation-only CPU reference stage. Stage 2 must not add, mock, or
structurally approximate a TensorG4DS dependency.

## Ownership And Scope Checks

Validation and Review should reject accidental introduction of:

- native G4DS parsing, TensorG4DS clustering, a TensorG4DS adapter, or
  photoelectron binning in a post-binned stage;
- durable cache or manifest contracts before their work order;
- scheduler, retry, repair, fan-in, or campaign logic;
- TensorML model/training/evaluation concepts;
- per-field semantic collection subclasses or a second loose readout product
  graph;
- claims that `ReadoutCollection` subclass identity alone defines a TensorML
  model schema;
- TensorML `input_fields` or `output_fields` changes outside a focused future
  integration design;
- trigger, ZLE, hit-finder, reconstruction, or analysis-preprocessing products
  inside readout;
- local TensorCore forks or compatibility shims;
- placeholder modules;
- generated caches, outputs, or unrelated files.

## Future Command Baseline

After the package exists, local checks should run from the project root:

```bash
git diff --check
PYTHONPATH=. python -m unittest discover -s tests
```

Each production work order should add focused import, dependency-scan, device,
and public-surface commands appropriate to its scope.
