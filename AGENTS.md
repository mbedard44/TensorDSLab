# Agent Workflow

This repository uses role-separated Codex ownership. Design owns architecture,
decisions, validation expectations, and future work orders directly. Stage 2
completed the first production Implementation/Validation/Review loop on
2026-07-11. Documentation-only Design work outside a dispatched production
work order may remain in Design unless the user requests independent Validation
or Review.

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
stage touches TensorCore axes/fields/collections, TensorDSLab product semantics,
in-memory product relationships, durable cache shape, validation boundaries,
public typing, coordinates versus indices, artifacts, result storage, or future
integration boundaries.
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

Design may operate alone for documentation-only work. Design, Implementation,
Validation, and Review are persistent logical roles per workspace after
activation. Production dispatch requires every execution role
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
readout and future reconstruction products, while using TensorCore's semantic
axis, field, collection, constrained-scalar, validation, and relationship
roots directly.

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

Stage 2 is Merged / Closed at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`, and Maintenance 1 is Merged /
Closed at `3af8ab4acf834b07e3d027fb530e5f12934999a5`. They remain historical
TensorCore `0.6` evidence.

Stage 3 is Merged / Closed through exact implementation candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde` and Review's clean fast-forward
closeout `97e17c3177ac217aeb42a077db78f4bd223d51fa`; Design's accepted final
closeout is clean `main` at
`5ff13eb3c0735abfda454a334be59faac35259c2`. It implements the TensorCore
`0.7` product/config/collection foundation described by
`docs/architecture/rebuild.md`. Fixed-commit Validation, independent Review,
and Design's post-merge audit found no unresolved issue. The evidence is
CPU-only because CUDA was unavailable, and no wheel or editable-install claim
was made because the required build tooling was absent.

Stage 4 is Merged / Closed through exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da` and Review's clean fast-forward
closeout `b3ebfcd9473537dd385195afea374bd2f426c6c0`. It implements exactly the
private pure, analog, and digitized waveform producers under the
functionality-first contract in
`docs/implementation/stage_4_deterministic_waveform_products.md`. Fixed-commit
Validation, independent Review, and Design's post-merge audit found no
unresolved issue. The evidence is CPU-only because CUDA was unavailable, and
it makes no GPU-performance, fusion, editable-install, or wheel-build claim.
The complete noise producer remains a candidate Stage 5 slice; no Stage 5 or
later integration work order is dispatched.

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

The checkout root is the project folder. The `tensor_dslab/` directory is the
Python import package. Do not create a
flat TitleCase Python package that imports as `TensorDSLab`.

The rebuild uses a product-centered readout tree:

```text
tensor_dslab/
  common/
    axes.py
    sampling.py
  readout/
    types.py
    simulation.py
    _requirements.py
    _random.py
    photoelectrons/
      types.py
    charge/
      types.py
      _product.py
    pure_waveform/
      types.py
      _product.py
    noise_waveform/
      types.py
      _product.py
    analog_waveform/
      types.py
      _product.py
    digitized_waveform/
      types.py
      _product.py
```

This is an ownership target, not permission to create placeholders. Materialize
only modules with real behavior accepted by the active work order. Each product
owns its final `TensorField` leaf, public configs, product validation, and
eventual private `_product_*` builder. Private `_simulate_*` functions implement
scientific submodels. `readout.types` contains only `ReadoutConfig` and
`ReadoutCollection`; `readout.simulation` owns the one public
`simulate_readout(...)` orchestration function. Shared axes and sampling belong
in `common`. Readout-specific requirements and random mechanics remain private.

Keep import direction acyclic: TensorCore, common, private shared requirements,
product types, product producers plus explicit prerequisite product types,
readout composition types, readout simulation, then deliberate package-root
exports. Product packages must not import `ReadoutConfig`, `ReadoutCollection`,
or `simulate_readout(...)`. Do not promote `_random.py` to `common` until a
second TensorDSLab domain needs the exact same accepted mechanics.

`Photoelectrons` is an already-produced dense truth input. It has no
`PhotoelectronsConfig`, no TensorDSLab readout producer, and no `_product.py`.
Source construction and PE binning remain part of the future TensorG4DS bridge.

Runtime commands launched from the project root should use the project root on
`PYTHONPATH` so absolute `tensor_dslab.*` imports resolve:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

Stage 3 production imports should stay absolute, such as:

```python
from tensor_core import TensorAxis, TensorField
from tensor_dslab import Photoelectrons, ReadoutCollection, SamplingConfig
from tensor_dslab.common import ChannelAxis, ExampleAxis, SampleAxis
```

Do not rewrite imports to relative forms to satisfy editor-only diagnostics.
Editor analysis tools should mirror the runtime path by including the project
root on their analysis path.

Do not create placeholder modules to reserve architecture. Add a module only
when there is a real TensorDSLab concept, behavior, or contract to house.

## TensorCore Boundary

The rebuild targets exact TensorCore `0.7` semantic roots:

```text
TensorAxis(coordinates: tuple[str, ...])
TensorField(tensor: torch.Tensor, axes: tuple[TensorAxis, ...])
TensorCollection(fields: Iterable[TensorField])
```

Production imports come from the public `tensor_core` package root. TensorCore
owns universal representation validation, constrained scalars, exact-type
lookup, and generic relationship helpers. It has no ID/layout/metadata model,
generic selection or movement API, output-buffer/workspace API, persistence
API, or lifecycle service. TensorDSLab must not recreate retired TensorCore
`0.6` IDs, layouts, constants, sidecars, compatibility shims, or generic
operations.

`ExampleAxis`, `ChannelAxis`, and `SampleAxis` are direct final fieldless
`TensorAxis` leaves. The six product types are direct final fieldless
`TensorField` leaves. `ReadoutCollection` is a direct final fieldless
`TensorCollection` leaf. Each leaf has exactly that matching root in
`__bases__`, with no mixin or other base. Every semantic leaf uses inherited root
construction, `@final`, `__slots__ = ()`, no added stored fields, and one
TensorDSLab `_require()` narrowing hook. Do not reapply `@dataclass`, introduce
an intermediate semantic base, or override generic root behavior.

These are ordinary Python ABC extension points. TensorDSLab verifies its own
leaf declarations through static analysis, focused tests, and Review. Runtime
code validates documented public inputs and cheap correctness-critical
relationships; it does not police callers who subclass final leaves, mutate
classes, bypass constructors, call private functions, mutate exposed tensors,
or install custom Torch dispatch behavior. Such behavior is unsupported and
has no promised error category.

Coordinates are exact unique nonempty strings scoped by exact axis type. Axis
tuple order is tensor-dimension order. Code locates dimensions by exact axis
class, not loose names or constants. `SampleAxis` contains canonical increasing
uniform left-edge timestamps such as `"0ps"` and `"2000ps"`; numeric kernels
use indices and `SamplingConfig`, never parse semantic labels on the hot path.
There is no `ExampleId`, `ChannelId`, `TensorAxisId`, `TensorFieldId`,
`IdSequence`, `TensorLayout`, `SampleGrid`, or `DigitizedWaveformSpec` in the
rebuild.

Every field contains exactly one example, channel, and sample axis in any
order, uses `torch.strided`, and reuses the exact source axis tuple for
dimension-preserving results. `Photoelectrons` is `torch.int64`; `Charge`,
`PureWaveform`, `NoiseWaveform`, and `AnalogWaveform` use one common
`torch.float32` or `torch.float64`; `DigitizedWaveform` is `torch.int32`.
`ReadoutCollection` accepts any nonempty unordered subset of those exact six
product types, with equal ordered axes, one device, and one common floating
dtype. It is a completed requested result, not a partial pipeline snapshot; it
has no add, replace, descendant-invalidation, or reconstruction lifecycle.

TensorCore establishes neither universal freshness nor universal storage
sharing. Every TensorDSLab field-returning operation classifies each successful
path as exact return, guaranteed storage-sharing, sharing permitted but
unspecified, or guaranteed fresh storage independent of named inputs. The MVP
classifies requested source `Photoelectrons` as an exact return and every
generated product as guaranteed fresh and pairwise storage-independent. Every
operation also owns dtype, device, axes, layout/strides, autograd,
synchronization, failure effects, and output-to-output relationships.

No write may begin through an alias after a semantic field has been constructed
and exposed. Producers initiate or enqueue all writes before constructing the
result field, and never later write through an alias to its storage. The public
MVP has no `out=`, destination collection, workspace, allocator, or stream
lease. Any later reusable destination remains raw, exclusive, and unexposed
until writes have been enqueued and the semantic field is constructed exactly
once. TensorCore contract changes still require TensorCore Design acceptance;
TensorDSLab does not fork it.

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
  -> dense TensorDSLab Photoelectrons truth field
  -> simulate_readout(...)
  -> request-selected ReadoutCollection
  -> deferred reconstruction and TensorML boundaries
```

The external chain is data flow, not package dependency flow. Core readout and
common modules depend only on TensorCore. A future downstream-owned bridge may
import an exact accepted public TensorG4DS type; TensorG4DS must never import
TensorDSLab to construct downstream identities. The bridge is a semantic
transformation, not a subclass cast or an assumption that TensorG4DS and
TensorDSLab axes are interchangeable.

The production integration target keeps tensor payloads resident on one
explicit accelerator device across TensorG4DS, TensorDSLab, and TensorML.
Boundary code must not silently call `.cpu()`, `.numpy()`, serialize/reload, or
otherwise use host materialization as the package handoff. New computations
may allocate new tensors on that same device, and TensorCore axes and other
small semantic records may remain ordinary host-side objects. Device movement
is always explicit. Because TensorG4DS has not yet frozen a public GPU output
contract, the exact accepted input type, dtype/axis matrix, and
device-preservation tests belong to the future integration work order; they do
not add a TensorG4DS dependency to the local readout foundation.

The discrete TensorG4DS bridge carries no end-to-end autograd promise and must
not detach silently. Its first work order should reject gradient-sensitive
inputs unless Design accepts a separate differentiable detector surface. This
does not weaken functional autograd for accepted deterministic waveform
transforms later in TensorDSLab.

The primary readout tensor handoff is `ReadoutCollection`, not a loose product
tuple or a required dataclass adapter. Runtime product and axis identity is the
exact concrete Python class. Example, channel, and sample coordinates are
ordered strings scoped by `ExampleAxis`, `ChannelAxis`, and `SampleAxis`.
TensorG4DS event values remain upstream provenance; the future bridge owns
their explicit mapping to ordered example-coordinate strings. Durable labels
and serialization remain deferred and must not be inferred from Python class
names without a focused artifact contract.

Consumer-facing adapters are deferred. TensorDSLab should first make the local
typed product graph coherent enough that future consumers can depend on it
without parsing raw `.fil`, table, array, manifest, or private representation
details.

The following is the accepted completed-rebuild simulation contract, not the
Stage 3 implemented surface. The later public readout operation consumes an
already-produced dense truth field:

```text
Photoelectrons
  -> simulate_readout(products=..., config=..., seed=...)
  -> ReadoutCollection containing exactly the requested products
```

`Photoelectrons` contains binned photon-origin primary PE truth. It never
contains dark counts, timing jitter, correlated avalanches, or charge
smearing, and the builder never mutates or replaces it. Charge production uses
private working values in physical order: truth, optional dark counts,
optional timing jitter, optional correlated-avalanche simulation, and optional
smearing. Intermediate count, charge-ledger, and diagnostic values are private
implementation state rather than fields or durable products.

`Charge` is the finite floating aggregate PE-equivalent response per
readout channel and sample. It is not an SI-coulomb measurement and does not
claim an explicit individual-SPAD output. Pure and noise waveforms are
signal-only and noise-only components at one shared analog reference plane;
they are not sequential hardware products. Their composition produces the
analog waveform consumed by digitization.

`simulate_readout(...)` requires an explicit nonempty iterable of exact product
classes. It consumes the iterable once, rejects duplicates and unknown classes,
computes the transitive prerequisite closure, validates every required config
and source relationship before RNG use, executes each producer at most once,
and retains exactly the requested fields. Request order has no collection
semantics. Unrequested prerequisites remain private local values.

The computational graph is:

```text
Photoelectrons -> Charge -> PureWaveform
Photoelectrons axes/device/shape + SamplingConfig -> NoiseWaveform
PureWaveform + NoiseWaveform -> AnalogWaveform -> DigitizedWaveform
```

`ReadoutConfig` composes `SamplingConfig` with optional product configs. Config
absence is structural. A requested product requires the configs in its
transitive closure; an unrequested branch does not. Product configs describe
science, not persistence, device movement, allocation, mutation, streams, or
campaign policy. `products` controls only final in-memory retention. IO is
deferred.

When implemented, the initial builder is functional. It borrows
`Photoelectrons` read-only,
returns that exact field when requested, and creates guaranteed-fresh generated
products. Generated products retained together are storage-independent. The
builder does not mutate sources, silently move/cast/detach/host-materialize
inputs, expose private scratch, or write through any alias after exposing a
semantic result. Preflight failure occurs before RNG consumption or producer
writes; failures after backend launch carry no rollback guarantee.

Every generated dimension-preserving field reuses the source's exact immutable
axis tuple and axis instances. Axis order may vary semantically; upstream
construction should ordinarily use example/channel/sample order so samples are
last for temporal kernels. Positional RNG addresses use tensor indices in the
actual dimension order, not coordinate strings, and therefore do not promise
permutation or arbitrary chunking invariance.

The completed MVP public surface has no atomic public transforms, mutation
lifecycle, generic projection/reconstruction helpers, `out=`, workspace,
allocation-free claim, or public stream policy. A later optimization stage
starts from measured GPU evidence and must preserve request, freshness,
exposure, synchronization, and lifetime contracts rather than reviving Stage
2's preconstructed writable collection model.

Projects/dag owns campaign fanout and fanin, scheduling, retry, repair,
compiled DAG objects, scheduler-visible grouping, status, and cross-shard
orchestration. TensorDSLab may later expose DAG-compatible executables,
operation specs, and recipe fragments only after local product and cache
contracts are accepted. Local dependency planning inside
`simulate_readout(...)` is TensorDSLab scientific orchestration, not campaign
orchestration.

For future caches, TensorDSLab owns deterministic storage-level compaction over
caller-supplied complete compatible products. Projects/dag owns scheduling,
fan-in, retries, repair, and campaign or cross-shard compaction orchestration.

## Validation Boundaries

TensorDSLab should move toward boundary-first validation:

```text
external/source/config/artifact boundary
  -> validate/coerce into strong typed objects
  -> construct TensorDSLab semantic leaves and collections
  -> hot path trusts those records
```

Validate strongly when data crosses into TensorDSLab or TensorCore native
records and when constructing new typed axes, products, collections, configs,
or artifacts. Leaf construction checks cheap intrinsic structure. Explicit
deep validation owns device-wide scientific scans at untrusted ingress and
producer postconditions. Do not repeatedly revalidate already constructed
graphs or parse semantic coordinate strings inside hot loops.

Use constrained scalar wrappers for meaningful numeric config/source/artifact
values where constraints matter. Tensor-local positive counts should use
TensorCore-owned `PositiveInteger`. Numeric wrappers should reject bool. Do not
add generic bool wrappers by default.

Implementation should validate supported public use, not adversarial attempts
to escape it. Unsupported subclassing, constructor bypass, class mutation,
private-function calls, exposed-tensor mutation, and custom dispatch require no
exhaustive detection or stable error behavior.

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
- `docs/architecture/tensors.md` when TensorCore integration, semantic axes or
  fields, result sharing/freshness, placement, synchronization, exposure, or
  lifetime contracts changed;
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
