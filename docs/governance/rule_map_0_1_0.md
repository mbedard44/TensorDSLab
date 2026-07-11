# Governance Core 0.1.0 Semantic Rule Map

Document status: Phase 2 candidate support; not adopted
Accepted deviations: none
Whole-rule Not-applicable dispositions: none

Each record maps one common rule to authoritative TensorDSLab sources and
evidence. The candidate repository baseline is the Git tree containing this
file, whose parent Design baseline is
`151b61fdc36475498219ee5fe7b045a3a72c2d09`; the Phase 2 completion report and
Phase 3 handoff record the exact containing commit without making this file
self-referential.

Dormant implementation surfaces remain qualifications inside applicable
records. Each qualification names its absence evidence and activation trigger.
No dormant surface supports a Production conformance, deployability, or broad
compatibility claim.

## Operational Rules

```text
governance_core_version: 0.1.0
rule_id: OP-01
disposition: Adopted
package_source_and_section: AGENTS.md — Governance Candidate Authority And State; Agent Workflow; Thread Roles / Design
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab assigns package architecture, contracts, ownership, dependencies, documentation, work orders, governance adoption, conformance, routing, and deviations to package Design; it accepts immutable affected-Design ratification, package-source precedence, and stop-and-return handling for contradictions.
evidence: The Design baseline assigns architecture, target behavior, decisions, and work orders to Design and requires implementation contradictions to return to Design. Candidate synchronization adds explicit immutable-proposal, source-precedence, and contradiction-resolution language without changing package architecture.
```

```text
governance_core_version: 0.1.0
rule_id: OP-02
disposition: Stronger local rule
package_source_and_section: AGENTS.md — Agent Workflow; Governance Candidate Authority And State
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab permits Design-only documentation maturity, makes D/I/V/R persistent logical roles after activation, and requires every production execution route to be Active and verified; Coordination remains separately Deferred with Design as fallback.
evidence: docs/overview.md — Current Local Focus and the Stage 2 status establish documentation-only maturity and an undispatched production work order. The exact baseline has no production package, tests, or metadata. Activation trigger: a Design-approved production work order plus Active, verified Implementation, Validation, and Review routes; independent documentation Validation or Review activates only when separately requested.
```

```text
governance_core_version: 0.1.0
rule_id: OP-03
disposition: Adopted
package_source_and_section: AGENTS.md — Governance Candidate Authority And State; Thread Roles / Design
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: Design owns target behavior, architecture, documentation, work orders, cross-package proposal treatment, governance adoption, conformance, deviations, and routing disputes.
evidence: The current documentation-only phase and this candidate are directly Design-owned. Candidate changes are documentation-only, do not dispatch Stage 2, and add explicit governance and routing duties to the existing role.
```

```text
governance_core_version: 0.1.0
rule_id: OP-04
disposition: Adopted
package_source_and_section: AGENTS.md — Thread Roles / Implementation; Production Implementation And Validation Loop; Documentation Synchronization Gate
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: Implementation owns only a dispatched branch, applies required code, test, and documentation changes, handles actionable findings, reports evidence and risks, and stops rather than altering accepted architecture, ownership, scope, or non-goals.
evidence: Implementation is dormant because no production work has been dispatched. Activation trigger: Design dispatches a committed production work order after required execution routes are Active and verified.
```

```text
governance_core_version: 0.1.0
rule_id: OP-05
disposition: Adopted
package_source_and_section: AGENTS.md — Thread Roles / Validation; Production Implementation And Validation Loop
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: Validation is read-only by default, derives behavioral strategy from accepted contracts, attacks boundaries and failure modes, returns concrete findings to Implementation, escalates architecture or scope issues to Design, and dispatches a fixed cleared commit to Review.
evidence: Validation is dormant because Stage 2 is undispatched and no production surface exists. Activation trigger: a Design-approved production work order with an Active, verified Validation route, or a separately authorized independent documentation Validation request.
```

```text
governance_core_version: 0.1.0
rule_id: OP-06
disposition: Stronger local rule
package_source_and_section: AGENTS.md — Thread Roles / Review; Production Review Gate
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: Review performs independent read-only fixed-target critique and explicit clearance. TensorDSLab additionally assigns Review clean fast-forward merge and post-merge verification only after clearance; ambiguity, dirty state, non-fast-forward history, or failed verification stops closeout.
evidence: Review is dormant because Stage 2 is undispatched. Activation trigger: Validation clears a fixed production commit to an Active, verified Review route, or the user separately requests independent documentation Review.
```

```text
governance_core_version: 0.1.0
rule_id: OP-07
disposition: Adopted
package_source_and_section: AGENTS.md — Governance Candidate Authority And State; docs/governance/overlay.md — Routing And Coordination
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: Coordination is non-authoritative package representation and remains Deferred; Design fallback is procedural only. The Moderator is neutral and cannot represent TensorDSLab, ratify architecture, command Design, dispatch execution, modify this repository, or infer assent from silence.
evidence: Coordination was not contacted, activated, closed, or used; the candidate records Design as fallback and creates no route artifact. Activation trigger: a concrete recurring need, accepted Coordination charter and Design-return path, adopted routing/privacy procedure, verified route and fallback, no discrepancy, and explicit Design and user authorization.
```

```text
governance_core_version: 0.1.0
rule_id: OP-08
disposition: Stronger local rule
package_source_and_section: AGENTS.md — Agent Workflow; Project Mode
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: Every repository retains its own role ownership, and TensorDSLab additionally requires any cross-repository role exception to be explicit and accepted by the user and every affected package Design authority.
evidence: AGENTS.md requires explicit per-repository role threads and package boundaries. No cross-repository role exception is active, and the documented ecosystem chain remains data flow rather than an import or ownership graph.
```

```text
governance_core_version: 0.1.0
rule_id: OP-09
disposition: Stronger local rule
package_source_and_section: AGENTS.md — Production Work Order Handoff
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab adopts the complete work-order gate and retains stronger requirements for concrete public/test sketches, parity classification, an exact clean baseline, focused stop conditions, and documentation synchronization.
evidence: The committed Stage 2 work order is Design-complete but explicitly undispatched and lacks a selected exact production base and verified execution routes. Activation trigger: Design refreshes it with a stable key, exact baselines, complete checklist, package-owned state vocabulary/source, Active verified routes, and Review/closeout expectations.
```

```text
governance_core_version: 0.1.0
rule_id: OP-10
disposition: Stronger local rule
package_source_and_section: AGENTS.md — Production Implementation And Validation Loop
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab retains finite-loop safeguards and strengthens them to exactly three dispatches in each direction, with an earlier stop when the same issue repeats twice; verified routes, fixed branch ownership, no scope expansion, and stale-route stopping are mandatory.
evidence: No I/V loop is authorized because Stage 2 remains undispatched. Activation trigger: a production work order explicitly authorizes the finite budget and Active verified routes; the loop stops on clearance, exhausted budget, a second repeated issue, stale routing, or need for Design action.
```

```text
governance_core_version: 0.1.0
rule_id: OP-11
disposition: Stronger local rule
package_source_and_section: AGENTS.md — Documentation Synchronization Gate; Verification Baseline
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab treats documentation drift as substantive and retains a stronger source-document matrix, stale-term checks, documentary verification, maturity-conditioned runtime commands, and Review-owned post-merge verification.
evidence: Phase 2 requires diff, link, heading, fence, stale-term, completeness, privacy, and scope checks. Production, import, integration, environment, and runtime post-merge checks are dormant; activation trigger: the corresponding package, test, dependency, integration, DAG, or runtime surface is accepted and implemented.
```

```text
governance_core_version: 0.1.0
rule_id: OP-12
disposition: Stronger local rule
package_source_and_section: AGENTS.md — Governance Candidate Authority And State; docs/governance/overlay.md — Routing And Coordination
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab adopts logical-key, privacy, package-source precedence, status, and discrepancy safeguards while imposing the stronger current posture that Profile B is disabled and no routing artifact may exist without a later focused Design decision.
evidence: The exact Design baseline has no .agents path, registry, route table, cache, ignore rule, or committed raw platform identifier; the candidate creates none. Activation trigger: Design accepts the exact private path, ignore policy, permissions/operators, sharing, replacement/history/deletion, verification, and discrepancy procedure before storage is instantiated.
```

```text
governance_core_version: 0.1.0
rule_id: OP-13
disposition: Adopted
package_source_and_section: docs/governance/adoption_0_1_0.md — Candidate Declaration; State Separation; docs/governance/overlay.md — Deviations And State; docs/governance/rule_map_0_1_0.md — Operational Rules; Engineering Rules
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab keeps common ratification, package adoption, conformance, routing storage, Coordination, and work-order state separate and binds no package state until the exact candidate receives a later Design decision.
evidence: The candidate remains Not adopted; conformance is Not evaluated; Profile B is Disabled; Coordination is Deferred; deviations are none; and Stage 2 is undispatched. Activation trigger: Phase 3 reviews the exact candidate and Design issues the proposed decision; every other state remains separately decided.
```

## Engineering Rules

```text
governance_core_version: 0.1.0
rule_id: ENG-01
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Build Philosophy; Sibling Repository Shape; Engineering Standard; Scope Discipline
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab retains professional scientific-software safeguards and strengthens them with exact post-binned ownership, staged work-order boundaries, no placeholder package tree, and mandatory return to Design when implementation would change architecture, ownership, scope, or non-goals.
evidence: The exact source baseline contains only the accepted Markdown sources and no production package, metadata, tests, compatibility layer, or placeholder module. Production activates only through a separately accepted focused work order and verified routes; this candidate supplies no production, deployability, or compatibility evidence.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-02
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Repository Identity; TensorCore Backbone; Product Semantics; Deferred Integration Surfaces
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens ownership and dependency safeguards with explicit TensorDSLab, TensorCore, TensorG4DS, TensorML, G4DS, and Projects/dag boundaries; TensorCore-only planned core modules; downstream-owned future adapters; and no silent host materialization. Scientific data flow remains distinct from imports and scheduling.
evidence: The exact baseline and candidate contain no package metadata, Python import package, production import, or tested dependency set. Direct dependency/runtime evidence activates when package metadata and implementation are accepted and must name exact commits, Python/dependency versions, backend, and execution mode. Bridges, adapters, caches, and orchestration activate only through focused stages. GPU-residency constraints are Design targets, not sibling compatibility evidence.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-03
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Governance Candidate And Delivery Maturity; Sibling Repository Shape; Public Surface Discipline
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab records display name TensorDSLab, future Python import tensor_dslab, and no current distribution identity until metadata is accepted. It additionally requires flat semantic packages, deliberate exports, absolute imports, and no unaccepted alias, re-export, shim, or placeholder import tree.
evidence: The exact baseline and candidate contain no pyproject.toml, tensor_dslab directory, Python file, or installable artifact. Distribution identity/installability activate when Design accepts and implements package metadata; the tensor-dslab spelling in Stage 2 remains a future candidate. Compatibility aliases activate only through explicit Design acceptance with scope, evidence, and removal trigger.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-04
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — TensorCore Backbone; docs/architecture/tensors.md — Design Baseline; Extension Points And Sealed Primitives; docs/implementation/stage_2_package_and_readout_collection_foundation.md — TensorCore Dependency Baseline
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab requires public-root-only TensorCore use, exact open/sealed extension contracts, and no fork, mirror, generic re-export, representation trick, or concealed shim; missing generic contracts return to TensorCore Design.
evidence: TensorCore 0.6.0 is the Design snapshot, while the undispatched Stage 2 work order names commit dc554994061183776f23f65860a0594516074f2e as a future dependency candidate. No package metadata, import, installed dependency, or consumer test exists. Implemented consumer evidence activates only after Stage 2 is dispatched and implemented in an exact named environment; neither Design record is a compatibility finding.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-05
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Domain Organization; Common Code; Public Surface Discipline; Public Verb Vocabulary; Boundary-First Validation; Code Expectations
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens typed, cohesive APIs with exact semantic module roles, frozen IDs/configs/value records, constrained scalars, exact runtime identity where required, numeric rejection of bool, narrow Any boundaries, and an explicit exception only for caller-owned mutable runtime workspace state.
evidence: Package sources specify typed and frozen semantic surfaces, deliberate public verbs and exports, constrained values, cohesive modules, and immutable semantics separate from mutable scratch. Production typing/export evidence is dormant because no Python API exists; activation trigger: the first accepted production public surface supplies implementation, typing, import, and export checks at an exact baseline.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-06
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — TensorCore Backbone; Coordinates, Indices, And Layouts; Product Semantics; docs/architecture/tensors.md — IDs, Coordinates, Indices, And Ordering; Axes And Layouts
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens the TensorCore vocabulary with exact ExampleId and ChannelId classes, canonical readout axes, caller-order preservation, axis lookup by ID equality, no durable transient indices, and fixed semantic-coordinate ordering for future stochastic transforms.
evidence: Sources define stable IDs versus indices, ID-backed versus count-only axes, exact coordinate classes, complete layouts, product-label versus field-ID separation, and tensor-order-independent stochastic identity. Runtime evidence activates with Stage 2 implementation; durable cache/report identity enforcement activates only with an accepted durable surface. Each requires exact-baseline tests and implies no sibling compatibility.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-07
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Boundary-First Validation; Public Verb Vocabulary; docs/architecture/readout.md — Public Transform Shape; Construction Invariants; Atomic Output Buffer And Aliasing Contract
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab distinguishes construction invariants from narrow hot-path preconditions, prohibits repair by validate operations, and requires complete destination, workspace, stream, dtype, device, alias, lease, and algorithm preflight before RNG consumption or writes.
evidence: Sources define validated-once trusted records, non-repairing validation, exact construction, and preflight failure that leaves RNG and tensors untouched. Constructors, adapters, loaders, caches, bridges, and transforms are unimplemented; each validation obligation activates with its corresponding accepted implementation and requires focused failure/no-mutation evidence at that exact baseline.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-08
disposition: Stronger local rule
package_source_and_section: docs/architecture/readout.md — Scientific Config Versus Runtime Control; Deterministic Random Fields; docs/validation.md — Deterministic RNG Checks; docs/parity.md — RNG Donor Parity And Backend Agreement; docs/decisions.md — Exact Tensor RNG And Cross-Device Agreement
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens deterministic-by-default behavior with caller-owned RngSpec, explicit seed/namespace/operation/semantic coordinates, no ambient global RNG, preflight before consumption, same-backend repeatability, batching/chunking/order invariance, and cross-backend distributional agreement instead of an unsupported bitwise claim.
evidence: Sources define owner, inputs, coordinate order, consumption boundary, reproducibility target, device-resident direction, and cross-backend evidence. No stochastic code or tests exist, and exact algorithm/backend support remains open. Activation trigger: a focused work order accepts the algorithm/backends and implementation exists, then records exact environment/device evidence. No CPU/GPU bitwise, deployability, or compatibility claim is made.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-09
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Repository Identity; Parts-Bin Rule; Documentation Expectations; docs/parity.md — Authority And Interpretation; Audited Donor Baseline; Required Shape Of A Parity Claim; Validation And Fixture Rules
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens donor-as-evidence with transform-scoped source snapshots/symbols, comparison boundaries, classifications, assumptions, units, observables, criteria, exclusions, divergences, fixture provenance, and revisit triggers while excluding historical framework and compatibility baggage.
evidence: docs/parity.md identifies exact donor sources and explicitly classifies its records as Design targets rather than demonstrated production parity. No donor-derived production code or tests exist. Activation trigger: each production donor promotion uses a focused work order and TensorDSLab-owned tests. Scientific divergences are not governance deviations; accepted governance deviations remain none.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-10
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Test Expectations; Documentation-Only Design Checks; docs/validation.md — Current Documentation Baseline; Governance Candidate Checks; Validation Layers; docs/implementation/stage_2_package_and_readout_collection_foundation.md — Minimum Test Design
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens behavior-first testing with exact constructor, coordinate, extension-point, order, invalidation, projection, autograd, preflight, alias, workspace, allocation, builder, export, isolation, device, RNG, and parity expectations when their surfaces exist.
evidence: No tests, Python package, serialization, cache, export, dependency, bridge, or integration implementation exists, and documentation stages prohibit placeholder tests. Each test category activates with its corresponding accepted implementation and requires focused behavior evidence plus an applicable representative integration path. Absence cannot support Production conformance.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-11
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Public Surface Discipline; Parts-Bin Rule; Documentation Expectations; AGENTS.md — Documentation Synchronization Gate
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens documentation synchronization with a source-of-truth matrix spanning work orders, architecture, TensorCore integration, parity, design, decisions, validation, onboarding, targeted stale-name checks, and substantive Review treatment of drift.
evidence: Phase 1 committed the coherent Design baseline. Phase 2 changes only four governance artifacts and the authorized five package references; architecture, parity, decisions, and implementation-stage records remain check-only, with docs/decisions.md unchanged before Phase 3. Future semantic/API/dependency/durable/parity/maturity changes activate same-stage updates to their named sources.
```

```text
governance_core_version: 0.1.0
rule_id: ENG-12
disposition: Stronger local rule
package_source_and_section: CONTRIBUTING.md — Documentation-Only Design Checks; Before Production Review; AGENTS.md — Verification Baseline; docs/validation.md — Governance Candidate Checks; Future Command Baseline
design_baseline: TensorDSLab Design-complete documentation baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: Candidate Git tree containing this row, derived from exact parent Design baseline 151b61fdc36475498219ee5fe7b045a3a72c2d09; exact containing commit recorded by the Phase 2 report and Phase 3 handoff
rationale: TensorDSLab strengthens repository-root verification with mandatory diff, deterministic link/heading/fence/whitespace/stale-term/raw-ID/coverage/allowlist/cleanliness checks, fixed-baseline reporting, and maturity-conditioned runtime commands.
evidence: No code, tests, import root, metadata, environment contract, export, integration, DAG adapter, or runtime post-merge command exists. Those checks activate with their corresponding accepted surface and must name exact commits, Python/dependency versions, backend, and execution mode. Phase 2 records actual documentation commands/outcomes, fixed commit, changed and checked-unchanged files, risks, and clean status; documentary checks imply no deployability or compatibility guarantee.
```
