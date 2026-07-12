# Decisions

This document records accepted and open design decisions. Keep historical
entries when they explain why a path was superseded or deferred.

## Accepted

### TensorDSLab Is Tensor-Native From The Start

TensorDSLab is a clean-slate detector data-lab package. It should define its
product and cache semantics directly while using TensorCore as the generic
tensor-native backbone.

### TensorCore Owns Generic Tensor Contracts

TensorCore owns generic tensor identity, axes, layouts, fields, collections,
selections, batching, movement, validation, and pure tensor operations.
TensorDSLab should import those surfaces from `tensor_core` and should not
mirror, fork, or broaden them locally.

### TensorDSLab Owns Post-TensorG4DS Detector Data-Lab Products

TensorDSLab owns the accepted mapping from TensorG4DS provenance into its own
examples and coordinates, detector-response/channelization/window/binning
semantics, readout and future reconstruction products, product labels, domain
builders, validators, and domain-specific semantic tensor products. Cache
records and durable cache IO are future TensorDSLab surfaces after the
in-memory product model is accepted. Native G4DS ingestion and TensorG4DS
deposit/clustering products are not TensorDSLab surfaces.

### TensorG4DS Is The Upstream Tensor Boundary

The intended ecosystem data flow is
`G4DS -> TensorG4DS -> TensorDSLab -> TensorML`. TensorCore is the shared
substrate, so this is not an import graph. TensorDSLab never parses native G4DS
files or reproduces TensorG4DS low-level analysis. Core common/readout modules
remain TensorCore-only; a future TensorDSLab-owned leaf adapter may import an
exact accepted public TensorG4DS type, and TensorG4DS must never import
TensorDSLab.

The production integration target keeps payload tensors on one exact GPU
without an implicit CPU, NumPy/list, serialization, movement, cast, or detach
boundary. The bridge may produce new tensors on that device because upstream
deposit/cluster layouts and downstream example/channel/sample layouts have
different meaning. TensorG4DS `EventId` is typed provenance and is never cast
to TensorDSLab `ExampleId`; a future bridge defines an explicit, potentially
one-to-many mapping. `ChannelId` remains TensorDSLab-owned.

TensorG4DS has not yet frozen the exact public type, GPU device guarantee,
dtypes, or layouts required by this bridge. The discrete bridge carries no
end-to-end autograd promise and may not detach silently; exact gradient-input
rejection mechanics remain a future cross-repository integration contract.
Stage 2 acquires no TensorG4DS dependency.

### Project Naming Follows The Tensor Ecosystem

The project/display folder is `TensorDSLab`; the accepted Python import package
is `tensor_dslab`, and the accepted distribution metadata name is
`tensor-dslab`.

### Domain Packages Are Flat Under `tensor_dslab`

Concrete domain packages live directly under the import root, such as
`tensor_dslab.common`, `tensor_dslab.readout`, and future
`tensor_dslab.reconstruction`. TensorDSLab does not add an intermediate
`tensor_dslab.domain` namespace. This keeps imports aligned with TensorML's
package shape without weakening domain ownership terminology.

`ExampleId` and `ChannelId` are shared coordinate identities owned by
`tensor_dslab.common`. The readout channel axis and future reconstruction
products use the same exact `ChannelId` class; a readout-specific channel-ID
class would create an artificial identity conversion at that boundary.

### Documentation Comes Before Production Code

The initial stages began documentation-only. Design owns architecture,
decisions, validation expectations, and future work orders directly for such
work. Stage 2 activated and completed the first full
Implementation/Validation/Review loop; later production work requires its own
focused dispatch unless the user requests an earlier independent review.

### Stage 2 Structural Foundation Is Accepted

Stage 2 is Merged / Closed on `main` at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. It accepts the `tensor-dslab`
metadata, `tensor_dslab.common` and `tensor_dslab.readout` structural package,
typed IDs and sidecars, `ReadoutCollection`, semantic reconstruction,
descendant invalidation, output preparation, and focused tests against exact
TensorCore pin `dc554994061183776f23f65860a0594516074f2e`.

This decision records implementation closeout only. It accepts no scientific
transform, RNG, workspace, cache, integration surface, GPU behavior,
deployability, broad compatibility, or conformance finding. No later
production stage is dispatched.

### Build In-Memory Products Before IO

TensorDSLab should define its MVP early, but the first production priority is
the post-binned TensorCore-backed `ReadoutCollection` field graph. The opaque
`ExampleId` coordinate class is part of that foundation. The TensorG4DS
handoff, explicit provenance-to-example mapping, detector-response/binning
products, readout example provenance, and future reconstruction products
follow once the local readout collection and transforms are stable and an
accepted upstream type exists. In-memory contracts should be coherent before
durable IO, cache compatibility, compaction, DAG integration, or downstream
adapter contracts are introduced.

### DAG Compatibility Is Deliberate And Deferred

TensorDSLab should not add DAG-compatible specs or recipes until local
tensor-native product and cache contracts are accepted. Projects/dag remains
the owner of concrete orchestration.

### Downstream Integration Is Deferred

TensorDSLab should be designed in isolation until its local product contracts
are stable. Downstream model, training, evaluation, batching, checkpoint,
metric, and adapter requirements should not drive the first in-memory module
boundaries.

### First MVP Starts At Post-Binned Readout

The first production MVP should focus on the post-binned tensor-native readout
path: already-binned photon-origin primary photoelectrons, aggregate SiPM
charge simulation, waveform rendering, analog waveform composition, and
optional digitization. Native G4DS parsing remains upstream of TensorDSLab.
The typed TensorG4DS handoff, detector-window construction, photoelectron
binning, durable IO, cache compatibility, DAG integration, and TensorML
adapters are deferred until the post-binned contract is stable.

### Readout Names Follow The Simulated Sensor-To-DAQ Boundary

The first integer readout product is `readout.photoelectrons`, not the donor's
representation-oriented “binned charge” name. It contains binned photon-origin
primary PE seeds. The name denotes a simulation input boundary: photon
absorption initiates a SPAD avalanche, so it must not be described as a
physically stored population of free electrons waiting for a later avalanche.

Dark counts, crosstalk, and afterpulses add sensor-origin avalanches rather than
photoelectrons. They therefore remain private intermediate tensors inside one
public `simulate_charge` transform and are not written back into
`readout.photoelectrons`. The materialized `readout.charge` result is a
floating aggregate PE-equivalent response per readout channel and sample, not
SI charge or an individual-SPAD-resolved output.

The composed pre-digitization voltage field is
`readout.waveform.analog`, replacing the ambiguous donor-facing “physical
waveform” name. `readout.waveform.digitized` and `DigitizedWaveformSpec` are
retained rather than renamed to “digital”: *digitized* identifies the direct
ADC-code result, while *digital* could also include later filtering,
segmentation, compression, triggering, or firmware products. Pure and noise
waveforms remain signal-only and noise-only components at one analog reference
plane; they are not claimed as separate sequential Tile, PDU, and DAQ hardware
outputs.

### TensorDSLab Gives TensorCore Records Domain Meaning

TensorCore remains the dense tensor spine. TensorDSLab should give
`TensorCollection` and related TensorCore records detector/readout meaning
through one concrete `ReadoutCollection(TensorCollection)` subclass. The
collection uses a typed `SampleGrid`, a conditional
`DigitizedWaveformSpec`, and free transform functions. A future
`ReadoutExample` is an optional thin provenance/context wrapper around the
collection, not the tensor handoff. Per-product collection subclasses, a
generic `Product` base, semantic `TensorField` subclasses, and a ToyProduct-like
wrapper hierarchy are rejected.

TensorDSLab defers concrete rank and axis order to runtime TensorCore layouts
while making product roles, field roles, exact readout axis IDs, sample
metadata, and stochastic coordinate ordering explicit.

### TensorCore 0.6 Is The Design Baseline

TensorCore `0.6.0` is sufficient for the first post-binned architecture.
Production imports should come from the public `tensor_core` root. TensorDSLab
may extend the open `Id` and `TensorCollection` classes but must not subclass
sealed TensorCore primitives or fork generic helpers locally.

Focused TensorCore coordination items should not block TensorDSLab unless real
implementation demonstrates that they are required.

### Post-Binned Fields Form One Partial Readout Snapshot

The first in-memory readout type is `ReadoutCollection`. It recognizes these
field IDs in canonical topological order:

```text
readout.photoelectrons
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.analog
readout.waveform.digitized
```

Any nonempty subset is a valid partially materialized snapshot, stored in the
canonical order filtered to present fields. Photoelectrons preserve
nonnegative integer binned primary-PE count semantics; charge is the finite
nonnegative floating aggregate PE-equivalent response after the internal SiPM
effects and smearing; analog waveform fields remain distinct from digitized ADC
counts. Charge is not an SI-coulomb measurement or an explicit
individual-SPAD output. Durable product labels and TensorCore field IDs remain
different types even where their string payloads coincide, and
`readout.photoelectrons` is accepted in both namespaces.

Every present field has the exact same ordered layout. The required public
axis identities are
`EXAMPLE_AXIS_ID = TensorAxisId("example")`,
`CHANNEL_AXIS_ID = TensorAxisId("channel")`, and
`SAMPLE_AXIS_ID = TensorAxisId("sample")` in the `tensor_dslab.readout`
namespace; their tensor dimension order
is arbitrary, and the IDs compare by value rather than object identity.
Example and channel are ID-backed by exact `ExampleId` and shared `ChannelId`
coordinates, while sample is count-only. `SampleGrid` is the typed collection
sidecar for sample period, origin, and containing-grid offset. Every accepted
extra axis also occurs in every field and is declared shared.

Fields share a device and use `torch.strided` layout; contiguity is not
required. Photoelectrons use `torch.int64`. Present charge, pure, noise, and
analog fields share either `torch.float32` or `torch.float64`. Digitized
waveforms use `torch.int32`. Collection construction is placement-neutral and
accepts any
PyTorch device when all fields share it exactly. CPU behavior is mandatory,
CUDA construction checks are conditional on available hardware, and accepting
a coherent collection on a device does not promise support from every later
transform kernel.

`DigitizedWaveformSpec` is a typed field-specific sidecar required exactly when
the digitized field is present. It retains bit depth, voltage range and offset,
analog gain, and quantization policy so ADC bounds and interpretation survive a
digitized-only projection. Removing or invalidating the field removes the spec;
digitization adds or replaces both atomically. Accepted bit depth is 1 through
16, accepted `analog_gain_db` is 0 through 40 dB inclusive, and the first
policy is `AdcQuantization.TRUNCATE`. The gain check intentionally corrects the
donor's impossible `gain > 40 and gain < 0` condition.

### Readout Semantic Types Live In `types.py`

`ReadoutCollection` is a stable public domain value object and therefore lives
with `SampleGrid`, `DigitizedWaveformSpec`, and `AdcQuantization` in
`tensor_dslab.readout.types`. `tensor_dslab.readout.tensors` retains only the
readout-semantic reconstruction, projection, selection, and movement helpers
until Design resolves their longer-term home.

The readout package namespace already supplies the domain qualification, so
the public axis Python symbols omit the redundant `READOUT_` prefix:
`EXAMPLE_AXIS_ID`, `CHANNEL_AXIS_ID`, `SAMPLE_AXIS_ID`, and
`REQUIRED_AXIS_IDS`. Their exact TensorCore values and string payloads do not
change. No compatibility aliases are retained in this pre-deployment package.

The collection is a structurally immutable snapshot: records and mappings are
immutable, while TensorDSLab transforms treat ordinary PyTorch-mutable tensor
payloads as read-only. Callers must do the same; in-place mutation of an
existing field bypasses invalidation and is outside the public contract.
Projection only removes fields and does not invalidate retained descendants.
Transform-driven addition or replacement structurally shares unaffected
source records and centrally invalidates every materialized transitive
descendant that could disagree with the new dependency value. The dependency
and invalidation graph is:

```text
readout.photoelectrons -> readout.charge -> readout.waveform.pure
common layout/sample grid -> readout.waveform.noise
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog
  -> readout.waveform.digitized
```

Individual transforms must not invent different invalidation behavior or
mutate the source snapshot.

TensorCore generic selection, batching, movement, and like-buffer operations
return base `TensorCollection` records. TensorDSLab reconstructs and validates
`ReadoutCollection` after those operations. At a future TensorML handoff,
explicit field selection defines positional model schema; current TensorML
batching and selection erase the subclass, so this stage requires no TensorML
API change and assumes no subclass-preserving behavior.

The invoking TensorDSLab helper carries `SampleGrid` and any conditional
`DigitizedWaveformSpec` explicitly around the generic operation, then
preserves, updates, or prunes them before reconstruction. Canonical axis
meaning is recovered from the exact public axis IDs in the transformed layout,
not from a role sidecar or free-form base collection metadata.

Noncanonical model field order remains a base `TensorCollection`; it must not
be reconstructed as `ReadoutCollection` by reordering arguments. Stock-loop
models receiving selected fields accept the base type unless a focused adapter
reconstructs a canonical subset before `forward_pass`. TensorML class checks
alone do not validate readout field IDs on inputs or outputs.

GPU field retention is explicit runtime policy. Callers project the required
field subset before movement, use TensorCore to move only that subset, and
reconstruct `ReadoutCollection` only when a semantic collection is needed on
the target device. Structural sharing avoids copies but does not evict live
retained tensors. Transforms invalidate stale descendants; they do not
implicitly drop unrelated valid fields or move a full snapshot.

Semantic reconstruction after sample-axis selection is limited to contiguous,
increasing, unit-stride ranges. It advances both `SampleGrid.origin_ns` and
`sample_offset` by the first selected local index. Arbitrary, reordered, or
strided sample selection remains a base `TensorCollection` until Design
accepts a richer sample-grid representation.

### Semantic Collections And Execution Profiles Are Separate Contracts

Collections define meaning; execution profiles define memory arrangement;
domain handoffs are explicit materialization boundaries.

`ReadoutCollection` keeps arbitrary semantic axis order and accepts
noncontiguous `torch.strided` read-only values. TensorLayout remains the only
axis-order authority. The general constructor does not reject expanded or
internally overlapping source views solely for their storage arrangement, and
TensorDSLab adds no execution-ready subclass, stride sidecar, or runtime flag
to collection identity.

The warmed `out + workspace` readout profile instead requires the sample axis
last, every participating source/generated target/scratch tensor contiguous,
and every writable tensor internally nonoverlapping and storage-disjoint. Its
ordered axes/sizes, shape, device, role dtypes, algorithm choices, destination
schema, stream, and exclusive lease match exactly. Preflight rejects before
RNG use or writes; it never permutes, calls `.contiguous()`, clones, casts,
moves, uses a copying reshape, or falls back to allocation. Different
leading-axis orders require different workspaces. Contiguous strides derive
from shape, so the MVP signature records no arbitrary stride tuple.

Functional execution may explicitly allocate to normalize arbitrary semantic
order/strides while preserving accepted autograd. Ordinary `out` execution
without a workspace may use documented allocating scratch or normalization
and has no allocation-free claim. Public destination factories create new
targets contiguously in the existing semantic order without reordering or
materializing retained fields. A non-ready source is prepared explicitly once
outside the repeated warmed loop.

Count-domain ping-pong swaps references among identical contiguous buffers; it
does not permute data. Integer count, floating response, and integer ADC domains
remain separate storage classes. A future allocation-free noncontiguous
profile requires a focused measured contract.

Sample-last is specific to temporal readout, not a global TensorDSLab order. A
future Readout-to-Reconstruction bridge may explicitly select by stable IDs,
validate channel completeness, reorder/materialize once into a
reconstruction-preferred layout such as channel-last, and then enter a
reconstruction-owned execution profile.

### Fixed-Grid Readout Order Is The First MVP Contract

`readout.photoelectrons` is the recognized binned photon-origin primary-PE seed
field. Only timing jitter replaces it. One public `simulate_charge` transform
consumes that field and adds or replaces `readout.charge`. Internally its order
is dark counts, parallel crosstalk and afterpulse contributions from one frozen
post-dark-count snapshot, then charge smearing. Intermediate avalanche counts
are private ephemeral tensors, not recognized fields or durable products.
Generated crosstalk or afterpulse counts do not recursively feed either effect
in the first fixed-grid model.

Timing and delayed afterpulse counts outside the sample window are dropped.
Crosstalk is bounded, first-generation, same-channel, and same-sample. Charge
smearing uses one aggregate `Normal(n, sqrt(n) * sigma)` draw per populated cell
and clips negative draws to zero.

Pure and noise waveforms are signal-only and noise-only components at one
shared analog reference plane, not sequential hardware products. Their sum and
optional analog clipping produce `readout.waveform.analog`, which digitization
converts to `readout.waveform.digitized` ADC counts. `DigitizedWaveformSpec` and
the digitized field name remain unchanged.

### Scientific Config And Runtime Control Stay Separate

Scientific configs describe physical/model behavior. RNG stream selection,
output destination, placement, dtype movement, scratch workspace, and
execution/chunking policy are explicit runtime controls. They are not
TensorCore identity or hidden mutable config state.

Stochastic transforms are coordinate-addressed and must not use global RNG
state, channel indices when channel coordinates exist, or sequential streams
whose results change under accepted batching or chunking.

The canonical RNG coordinate payload is independent of tensor layout order:
example coordinate first, channel coordinate second, every other ID-backed
shared axis paired with its coordinate in lexical `axis_id.value` order, then
the count-only sample ordinal. The sample ordinal is the typed containing-grid
offset plus local sample index; that offset is grid metadata, not a TensorCore
coordinate or durable row identity. Sample slicing advances both physical
origin and offset consistently. There is no configurable stochastic-axis
membership list. Additional count-only axes may remain
structural dimensions, but stochastic transforms reject them until Design
accepts an offset contract that makes their identity chunk-invariant.

### Readout Execution Has Atomic, Workspace, And Builder Layers

Atomic scientific transforms remain free functions. `out=None` is their
allocating functional path and preserves accepted deterministic autograd
behavior. `out=destination` is the exact-target simulation path; internal
scratch or explicit normalization may still allocate unless a compatible
workspace lease is supplied.
Every failure detectable before kernel launch must occur before RNG consumption
or writes, and every successful transform completely overwrites its one target.
No transactional rollback is promised after an asynchronous backend failure.

Every atomic destination remains an already-valid `ReadoutCollection` with the
exact post-invalidation schema. Its factory zero-initializes new targets in
contiguous storage using the existing semantic order; reuse
may contain the prior valid value because successful execution fully overwrites
the target. Retained fields
are exact source records. The writable target is internally nonoverlapping and
aliases no source/retained storage, other live output, or live workspace slot.
Digitized-field sidecars follow the same
atomic result rules. A workspace is valid only together with `out`; supplying
`out` selects a non-autograd simulation path and must reject
gradient-sensitive use before mutation.

`ReadoutWorkspace` is a caller-owned, scratch-only runtime resource. It has one
fixed ordered-axis-ID/size, shape, device, dtype, algorithm, destination
schema, and stream signature. Warmed use requires sample-last order and
contiguous participating storage. Canonical-axis positions are derived from
the ordered IDs rather than stored
as a second role map; no arbitrary stride tuple is needed. The workspace has
one CPU execution domain or CUDA stream and one exclusive non-reentrant lease.
It is never persisted or
encoded as a collection field, sidecar, config, ID, coordinate, or product
label. A returned collection never references it. The MVP performs no hidden
workspace caching, resizing, movement, casting, source recycling, cross-stream
handoff, or concurrent sharing. Private scratch alone may use uninitialized
storage, subject to complete write-before-read.

`build_readout_collection(...)` is the local domain builder over the atomic
transforms. It recomputes the configured photoelectron-to-analog chain and
optional digitization, owns operation order and scratch liveness, writes final
values directly into prepared public destinations, and assembles the validated
collection without tensor clones. It does not load sources, move devices,
perform cache IO, or own Projects/dag orchestration. The first builder returns
the complete configured product set rather than a general partial-output plan.

Public outputs remain separate from workspace scratch. `out=None` returns
ordinarily owned functional results. A caller-supplied full destination remains
stable until the caller submits it as writable output again; doing so
authorizes overwrite and ends the earlier snapshot's stable lifetime. Output
banking for overlapping consumers is caller-managed. Reusing raw storage across
different semantic layouts or coordinates requires a future leased output-pool
design and is not smuggled into `ReadoutWorkspace`.

The accepted allocation claim is narrowly *steady-state TensorDSLab-managed
tensor-storage allocation-free*: after warm-up, an exact supplied destination
plus compatible workspace causes no TensorDSLab-owned target or named-scratch
tensor-storage allocation for that execution signature. It is not a promise of
zero Python/view allocation or zero opaque PyTorch/CUDA library planning and
scratch. Stronger backend-wide guarantees require focused instrumentation.

Mutation, workspace, output-lifetime, and allocation policy are runtime
control, not TensorCore identity. Do not encode them as a TensorCore `Id`, field
ID, axis ID, or coordinate, and do not introduce a persistent ambient mutation
mode.

### Cache Compaction Ownership Is Split By Level

TensorDSLab owns a future deterministic storage-level compaction primitive over
caller-supplied complete compatible products. Projects/dag owns campaign and
cross-shard discovery, scheduling, retries, repair, fan-in, and execution
policy.

### Donor Parity Is Scoped And Classified

TensorDSLab parity claims must name a donor reference, comparison boundary,
input/config domain, assumptions, declared observables, acceptance criteria,
exclusions, and intentional divergences. Use the exact, numerical,
distributional, statistical, intentional-divergence, deferred, and
not-applicable classifications defined in `docs/parity.md`.

Post-binned statistical or distributional parity is acceptable without
seedwise or bitwise IV-DSLab output identity. A tensor-native MVP simplification
may be accepted when its changed observables, risk, validation criteria, and
revisit trigger are explicit. Difficulty alone does not justify an unmeasured
scientific bias.

The ordinary exponential afterpulse delay is an accepted scientific correction
and intentional divergence from the literal reciprocal-exponential expression
in the audited IV source. Recovery-amplitude behavior remains open.

### Governance Core 0.1.0 Is Adopted (`TDSLAB-GOV-D001`)

Decision ID: `TDSLAB-GOV-D001`
Decision status: Issued / Adopted
Decision date: 2026-07-10
Governed Design base: `151b61fdc36475498219ee5fe7b045a3a72c2d09`
Accepted candidate: `d634401a853915edeb4f83df4a4943b3553deced`
Governance manifest-file SHA-256:
`45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8`
Council context manifest-file SHA-256:
`343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916`

TensorDSLab Design accepts the exact Governance Core `0.1.0` package-adoption
candidate above without conditions. The accepted package records are
`docs/governance/index.md`, `docs/governance/adoption_0_1_0.md`,
`docs/governance/overlay.md`, and `docs/governance/rule_map_0_1_0.md`. The
candidate maps every `OP-*` and `ENG-*` rule exactly once, accepts six rules
directly, records 19 stronger local rules, uses no whole-rule Not-applicable
disposition, and accepts no deviation.

This decision changed only the TensorDSLab package-adoption state to
`Adopted`. At issuance, conformance remained `Not evaluated`, Coordination
remained `Deferred`, Profile B remained `Disabled`, and Stage 2 remained
Design-complete and undispatched. It did not accept or implement a scientific contract,
dependency, device or data-flow boundary, compatibility or migration claim,
API, production surface, deployment state, backward-compatibility guarantee,
route, registry, cache, or council decision.

The central Design-qualified working dossier remains accurate on mission,
ownership, documentation-only maturity, designed-versus-implemented
boundaries, and Deferred Coordination. Its `Not adopted` status and remaining
adoption-prerequisite wording are a pre-decision snapshot superseded for
package state by this decision; conformance and Coordination qualifications
remain current. No sibling-repository edit is part of this decision.

The record-only closeout commit that publishes this decision is named in the
Phase 3 completion report rather than embedded self-referentially here.

## Superseded

### One Collection Subclass Per Readout Product

An earlier Stage 1 draft proposed separate single-field `TensorCollection`
subclasses for the then-named binned charge, charge, pure waveform, noise
waveform, physical waveform, and digitized waveform roles. No production code
used that historical design.

It was superseded because the classes would each contain only the value named
by the class, while TensorCore already provides an ordered multi-field
collection with shared-axis validation and mixed dtypes. The one-class design
also avoids a second assembly adapter before future multi-field consumers.
Scientific distinctions were not removed: exact field IDs, value domains,
dependency rules, and durable producer labels preserve them explicitly.

The replacement does not assume that one growing collection can be passed
positionally to every TensorML model. Canonical readout field order is the
domain snapshot order; explicit TensorML field selection remains the model
argument schema. It also does not permit mutable accumulation: transforms
return structurally new snapshots, invalidate descendants, and use a
field-scoped buffer contract.

## Open

### Later Production Dispatch Handoffs

Stage 2 was dispatched on 2026-07-11 from exact clean package baseline
`d097cb3cdde185c6814116e886e7844ea3f55178` through the verified logical
Implementation, Validation, and Review routes and the repository's three-round
maximum I/V loop. Review fast-forward merged the cleared foundation at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. Its work-order path remains the
stable package-owned key. Maintenance 1 was separately dispatched from
committed Design authority `d09cbad4a1538349e289523a9898f4e6dfd20a57` to
correct only readout public-name and module ownership. A feature-branch copy is
candidate evidence before fixed-commit Validation and independent Review; if
the updated surface is read on `main`, Review's clean fast-forward gate has
completed. Later production stages still require their own focused committed
work order, exact clean base, verified routes, bounded loop, fixed-commit
Review gate, and clean closeout expectation; Stage 2 completion does not
pre-authorize them.

### Workspace Inventory And Stronger Allocation Guarantees

The scratch-only workspace ownership, exact-signature rejection policy,
single-stream exclusive reuse, builder order, and caller-owned output lifetime
are accepted. The contiguous sample-last warmed profile and no-hidden-
normalization rule are also accepted. Implementation must still determine the
exact physical scratch inventory and fusion strategy for timing, charge
fan-out, convolution, FFT, and RNG kernels. A separate immutable compiled plan,
general partial-output execution, growable workspaces, event-aware cross-stream
pools, CUDA Graph capture, and leased public output pools remain deferred. No
work order may
upgrade the narrow TensorDSLab-managed tensor-storage claim to backend-wide
zero allocation without backend-specific memory instrumentation. A
noncontiguous/stride-aware allocation-free profile likewise requires a new
focused measured Design contract.

### One-Time Readout Execution Preparation Surface

The semantic-to-warmed preparation boundary is accepted, but its exact public
API spelling remains open for the Stage 3 work order. It must locate axes by
stable IDs, explicitly reorder and materialize contiguous sample-last tensors
on the already-selected device, preserve coordinates and semantic sidecars,
leave the source immutable, and return a newly validated semantic value. A
pure coherent axis permutation may remain a `ReadoutCollection`; a bridge that
changes representation or domain meaning may not. Stage 2 adds no placeholder
helper, and warmed execution must never invoke this preparation implicitly.

### TensorCore Semantic Reconstruction Boundary

The current `tensor_dslab.readout.tensors` functions delegate generic field
selection, axis selection, and device movement to TensorCore, then restore
readout-only sidecars and invariants. `SampleGrid` changes and conditional
digitized-sidecar pruning cannot be inferred generically by TensorCore, but the
repeated base-collection reconstruction may justify a future opt-in TensorCore
extension hook.

Design must compare that option with explicit `ReadoutCollection` methods or a
more precise TensorDSLab function home. Do not change TensorCore's accepted
exact-base return contracts, override inherited operations with stronger
preconditions, or move the helpers merely to hide this boundary.

### Afterpulse Recovery-Amplitude Policy

IV-DSLab weights afterpulses by
`1 - exp(-delay / recovery_tau)`. The later fixed-grid DSLab path emits a unit
count and omits recovery time. TensorDSLab must explicitly choose unit-count
private afterpulse contributions or define a typed recovery-amplitude policy
inside `simulate_charge`, including its order relative to charge smearing,
before afterpulse implementation is dispatched. The standard exponential delay
itself is already accepted as an intentional correction of the literal IV delay
expression.

### Exact Tensor RNG And Cross-Device Agreement

Coordinate payload and batching/chunking invariance are accepted. The exact
stateless/counter RNG algorithm, supported accelerator implementations, and
whether CPU/GPU bitwise identity is required remain open for the stochastic
implementation work order. Same-backend exact repeatability and cross-backend
distributional agreement for accepted probability kernels are required;
finite-sample statistical validation supplies evidence for the latter.

### Cache Compatibility Target

TensorDSLab has not yet decided whether the first durable cache stage should
target a compatibility-oriented format, a loader-compatible transition format,
or a new TensorCore-backed cache format.

### Donor-Code Promotion Policy Per Stage

Historical predecessor code is parts-bin material only. Each production stage
still needs to name which donor semantics, algorithms, fixtures, or tests are
being promoted and which old structures are intentionally left behind. Each
claim must use the scoped taxonomy and evidence requirements in
`docs/parity.md`.
