# Agent Workflow

This repository uses a four-thread Codex workflow:

```text
Design -> Implementation + Validation -> Review -> Implementation fixes -> Review recheck
```

For each active TensorDSLab workspace, use one persistent thread per role:

- Design
- Implementation
- Validation
- Review

Tasks and stages are passed through handoffs, not represented as new permanent
threads. The goal is to keep architecture, implementation, behavioral
validation, and independent critique separate enough that each thread can do
its job without blurring ownership.

When a stage spans multiple repositories, such as TensorDSLab, TensorCore,
TensorML, g4ds11, or Projects/dag, keep each workspace's role
threads explicit in the handoff. Do not reuse a TensorDSLab role thread as the
owner of another repository's implementation or validation unless the user
explicitly accepts that exception.

Agents should also follow `CONTRIBUTING.md`, which defines repository-wide
engineering standards. Start with `docs/overview.md` for the documentation map.
Design work orders should cite the relevant `CONTRIBUTING.md` standards when a
stage touches TensorCore layout, TensorDSLab product semantics, in-memory
product relationships, durable cache shape, validation boundaries, public
typing, IDs versus indices, artifacts, or future integration boundaries.
Validation and Review should treat violations of accepted `CONTRIBUTING.md`
standards as real findings, not style-only comments.

## Project Mode

TensorDSLab is a clean-slate, tensor-native detector data-lab package. It
should turn g4ds11 detector-simulation output into typed detector, readout,
and future reconstruction products, while using TensorCore as the generic
tensor identity, layout, field, collection, selection, batching, movement,
validation, and pure operation backbone.

The intended chain is:

```text
g4ds11 -> TensorDSLab -> future consumers
```

TensorDSLab owns detector data-lab products and future durable cache
contracts. It should not own generic TensorCore primitives, downstream
model/training/evaluation surfaces, checkpoint policy, metric reporting, or
campaign orchestration.

The first accepted MVP direction is the post-binned tensor-native readout
path: already-binned charge, stochastic charge transforms, waveform products,
physical waveform composition, and optional digitization. Defer source PE-hit
parsing, detector-window construction, charge binning, IO, cache
compatibility, DAG compatibility, and downstream integration until the
post-binned contract is stable.

Historical predecessor code, if consulted outside this repository, is parts-bin
material only. It may provide scientific facts, algorithms, fixtures, tests,
and cautionary examples, but it does not define current architecture by
default. Do not copy old package layouts, helper framework shape,
compatibility baggage, or DAG-facing mechanics into TensorDSLab by default.
Promote only reviewed behavior that fits the tensor-native design and is
recorded in TensorDSLab docs.

TensorML is a style and workflow reference, not a detector data-lab domain
template. Replace TensorML process semantics with TensorDSLab product and cache
semantics when adapting docs or patterns. TensorCore is the source of truth for
generic tensor vocabulary and contracts.

Current maturity mode is initial Design documentation. Until a focused work
order accepts production code, do not create package modules, cache schemas,
tests, DAG surfaces, downstream integration surfaces, or copied donor code.

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

Runtime commands launched from the project root should use the project root on
`PYTHONPATH` so absolute `tensor_dslab.*` imports resolve:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

Production imports should stay absolute, such as:

```python
from tensor_core import TensorAxisId, TensorLayout
from tensor_dslab.domain.detector import DetectorExample
```

Do not rewrite imports to relative forms to satisfy editor-only diagnostics.
Editor analysis tools should mirror the runtime path by including the project
root on their analysis path.

Do not create placeholder modules to reserve architecture. Add a module only
when there is a real TensorDSLab concept, behavior, or contract to house.

## TensorCore Boundary

TensorDSLab should use TensorCore directly for generic tensor contracts:

- `Id`, `TensorAxisId`, and `TensorFieldId`;
- `IdSequence` and `PositiveInteger`;
- `TensorAxis`, `TensorAxes`, `TensorLayout`, `TensorField`, and
  `TensorCollection`;
- tensor selections such as `TensorFieldSelection` and
  `TensorAxisSelection`;
- generic builders, validators, mapping helpers, batching helpers, movement,
  reduction, selection, detachment, addition, and other pure tensor operations.

TensorDSLab owns domain IDs, product records, builders, validators, and
domain-specific tensor renderings. Future cache records and loaders belong in
TensorDSLab only after in-memory product contracts are accepted. Domain IDs
may appear as TensorCore coordinates when they subclass TensorCore `Id`, but
they should not become TensorCore-owned primitives.

TensorCore is the dense tensor spine. TensorDSLab gives TensorCore records
detector/readout product meaning instead of recreating generic tensor
mechanics. Runtime scripts and builders may choose concrete TensorCore layout
shape, but TensorDSLab should make product roles, field roles, semantic axis
roles, sample metadata, and stochastic coordinate inputs explicit. Concrete
product wrappers are optional and require a focused TensorDSLab design reason;
do not copy a generic `Product` base or ToyProduct-like pattern by default.

TensorCore terminology is strict:

- a coordinate is a stable `Id` value associated with an ID-backed axis;
- an index is a zero-based integer tensor position along an axis;
- a layout is ordered axes plus coordinate-to-index maps for ID-backed axes.

Coordinates and indices are never interchangeable. Do not persist transient
tensor, table, or array indices as durable identity. Diagnostics, caches, and
reports should prefer semantic IDs when an axis is ID-backed.

TensorCore contract changes require explicit Design acceptance in the
TensorCore workspace. TensorDSLab should not fork TensorCore, keep local
compatibility shims for retired TensorCore names, or broaden TensorCore public
surfaces from a TensorDSLab implementation stage.

## Product Relationships And Boundaries

TensorDSLab should preserve this product dependency rule unless Design
accepts a focused change:

```text
g4ds11 native output
  -> DetectorExample
  -> ReadoutExample
  -> ReconstructionExample
  -> future consumer-facing tensor/product views
```

The domain-to-domain boundary is the typed example object, not a loose product
tuple. Source event IDs are provenance. Stable TensorDSLab row identity should
be explicit and should not be guessed from g4ds11 native indices.

Producer product labels such as `detector.pe_hits`, `readout.charge`,
`readout.waveform.pure`, `readout.waveform.noise`,
`readout.waveform.physical`, and future reconstruction labels are durable
TensorDSLab product labels. TensorCore `TensorFieldId` values are tensor-local
field identities. Do not casually collapse the two namespaces.

Consumer-facing adapters are deferred. TensorDSLab should first make the local
typed product graph coherent enough that future consumers can depend on it
without parsing raw `.fil`, table, array, manifest, or private representation
details.

The first readout operations should work from an already-binned charge product:

```text
binned charge
  -> timing jitter
  -> dark counts
  -> crosstalk and afterpulses
  -> charge smearing
  -> pure waveform
  -> noise waveform
  -> physical waveform
  -> optional digitization
```

Post-binned readout transforms should use explicit output buffers. If `out` is
omitted, the method allocates and returns a new product. If `out` is supplied,
the method writes into `out` and returns `out`. `out` must be the correct
TensorDSLab product type with compatible TensorCore layout, device, dtype,
semantic axis roles, and product meaning. Mutation policy is runtime control,
not TensorCore identity; do not encode it as a TensorCore `Id`, field ID, axis
ID, coordinate, or product label.

Projects/dag owns campaign fanout and fanin, scheduling, retry, repair,
compiled DAG objects, scheduler-visible grouping, status, and cross-shard
orchestration. TensorDSLab may later expose DAG-compatible executables,
operation specs, and recipe fragments only after local product and cache
contracts are accepted.

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
- say what would require coming back to Design.

Design should not implement production code for the feature branch unless the
user explicitly delegates that exception.

### Implementation

Implementation owns the feature branch and is the default code-writing role.

Implementation should:

- make production code, test, and docs-sync changes required by the work order;
- keep the diff scoped to the work order;
- apply fixes requested by Validation and Review;
- keep the branch coherent and committed when asked;
- report commands run, known risks, and unresolved questions.

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
  require changing the accepted architecture or stage scope.

Review is read-only by default. It should not rewrite the branch unless the
user explicitly asks it to. If a Review finding requires an architecture or
scope change, Review should route it to Design instead of asking Implementation
to patch around the work order.

## Work Order Handoff

Design should dispatch only after the source-of-truth work order is committed
and the base branch is clean, unless the user explicitly accepts an exploratory
exception. A dispatch should name an exact base commit, target branch,
source-of-truth stage doc, already-changed docs, required scope, non-goals,
expected commands, Validation thread, loop budget, and escalation conditions.

A Design work order should include:

- task;
- base branch or commit;
- target branch;
- target files or packages;
- source-of-truth docs to keep synchronized;
- public surfaces to add or change;
- invariants and validation rules;
- non-goals;
- minimum tests or doc checks;
- commands expected before Review;
- known risks or open questions;
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

## Implementation And Validation Loop

After Design sends a work order, Implementation and Validation may iterate
until the branch is stable:

```text
Implementation builds -> Validation tests/critiques -> Implementation fixes
```

Implementation and Validation may message each other automatically when the
work order provides the needed thread identifiers and explicitly authorizes the
loop. This automatic loop is bounded:

- maximum three Implementation-to-Validation dispatches;
- maximum three Validation-to-Implementation dispatches;
- each message must be specific and actionable;
- no architecture changes or scope expansion;
- no branch ownership changes;
- stop early when Validation reports no blocking findings;
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

## Review Gate

Send a branch to Review only after the implementation/validation loop is quiet.
Review should not be asked to review a moving target unless the request is
explicitly an early design or architecture review.

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
- `docs/design.md` when end-to-end domain flow or ownership boundaries changed;
- `docs/decisions.md` when a semantic choice was accepted, renamed,
  superseded, or explicitly deferred;
- `docs/validation.md` for expected behavior, validation cases, fixtures,
  failure modes, or numeric tolerances changed;
- `README.md`, `AGENTS.md`, or `CONTRIBUTING.md` when workflow, onboarding, or
  repository-wide expectations changed.

Implementation handoffs should explicitly say which docs were updated, or why
no docs update was needed. Validation and Review should run targeted stale-name
searches when a public term is renamed. Keep legitimate historical mentions
only when they are clearly framed as historical, deferred, or superseded.

## Verification Baseline

Before Review, run the smallest relevant verification set for the change. At
minimum, run:

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
