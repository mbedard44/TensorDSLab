# TensorDSLab Governance Overlay

Document status: Adopted through `TDSLAB-GOV-D001`
Governance Core version: `0.1.0`
Adopted candidate: `d634401a853915edeb4f83df4a4943b3553deced`
Package Design baseline: `151b61fdc36475498219ee5fe7b045a3a72c2d09`

This overlay records TensorDSLab-specific strengthening and maturity context
for the common governance rules. It does not replace `AGENTS.md`,
`CONTRIBUTING.md`, or the accepted architecture documents. It has no adoption
effect beyond the exact process rules accepted by `TDSLAB-GOV-D001` and does
not create scientific or package architecture.

## Authority And Precedence

TensorDSLab Design owns package architecture, public contracts, scope,
dependencies, work orders, governance adoption, deviations, conformance
findings, and routing disputes. Package sources govern package-local contracts.
Ratified council decisions govern only cross-package relationships accepted by
every required affected Design authority.

If records conflict, stop the affected work, identify exact sources and
baselines, return the contradiction to each affected Design authority, and
resume only from an explicit resolution. Tests and work orders operationalize
accepted contracts; they do not create architecture accidentally.

## Package Identity And Maturity

```text
project/display name: TensorDSLab
Python import package: tensor_dslab (accepted Stage 2 import root; not yet on main)
distribution name: tensor-dslab (accepted Stage 2 spelling; not yet on main)
delivery maturity: active development / pre-deployment
package maturity: Stage 2 production loop active; accepted main is documentation-only
Stage 1: Design-complete
Stage 2: Dispatched / production candidate not yet accepted
```

There is no accepted `pyproject.toml`, `tensor_dslab/` package, test suite, or
installed runtime dependency on `main`. Stage 2's feature branch and proposed
metadata are not evidence that the package is installable, published, or
deployed before fixed-commit Validation, independent Review, and clean merge.

## Pre-Deployment Compatibility Posture

TensorDSLab makes no deployability, release-readiness, certification,
backward-compatibility, or broad compatibility guarantee. Existing APIs and
designs remain changeable through their owning Design authorities.

Later compatibility evidence must name exact package commits, Python and
dependency versions, device/backend, and relevant execution mode. A passing
tuple says nothing about historical, future, or untested tuples. Cross-package
breaking changes require affected Design ratification and a synchronized
migration plan before implementation. Compatibility aliases, shims, and
deprecation windows require a focused accepted need; this adoption creates
none.

The current same-device residency and no-silent-host-materialization rules are
TensorDSLab Design constraints. They do not prove that TensorG4DS, TensorDSLab,
and TensorML interoperate on any device or package tuple.

## Role Lifecycle

- Documentation-only Design work is owned directly by persistent TensorDSLab
  Design unless the user requests independent documentation Validation or
  Review.
- Implementation, Validation, and Review activation is lazy until a
  Design-approved production work order requires them.
- Once activated for a workspace, D/I/V/R are persistent logical roles; work
  orders are assignments beneath those roles rather than replacement offices.
- Production dispatch requires every execution role to be Active, privately
  verified for the exact workspace and baseline, and named in the handoff.
- Coordination is optional, non-authoritative, and Deferred.

## Execution-Role Boundaries

Design defines scope, invariants, contracts, non-goals, work orders, adoption,
deviations, and escalation points. It does not implement dispatched production
features unless the user explicitly delegates that exception.

Implementation owns the dispatched feature branch and applies in-scope code,
test, and synchronized documentation changes. It stops when requested work
would alter architecture, ownership, scope, or non-goals.

Validation owns behavioral confidence and is read-only by default. It derives
tests from accepted contracts, sends actionable findings to Implementation,
and routes architecture or scope contradictions to Design.

Review owns independent fixed-target critique and is read-only while reviewing
and rechecking fixes. After explicit clearance only, Review owns TensorDSLab's
clean fast-forward closeout merge and named post-merge checks. Ambiguous target
branches, dirty state, non-fast-forward history, or failed verification stop
closeout.

## Work Orders And Bounded Iteration

Production work orders require a stable package-owned key, exact Design and
repository baseline, target branch, complete scope and non-goals, concrete
public/test sketches, invariants, source documents, commands, verified I/V/R
routes, work-order state vocabulary, a finite loop budget, risks, escalation
conditions, and Review/closeout expectations.

TensorDSLab's I/V loop permits at most three dispatches in each direction. It
stops earlier when Validation clears, the same issue repeats twice, routing
becomes stale, the budget is exhausted, or a Design decision is needed. Review
receives one fixed branch and commit after the loop is quiet.

## Cross-Repository Ownership

Each repository retains its own Design and execution roles. A TensorDSLab role
must not own sibling implementation, validation, review, or merge work. Any
exception requires an explicit handoff accepted by the user and every affected
package Design authority.

## TensorCore And Tensor Semantics

TensorDSLab consumes the accepted public `tensor_core` package root at a named
baseline. `Id` and `TensorCollection` are the accepted downstream extension
points; sealed primitives remain sealed. TensorDSLab does not fork, mirror,
re-export, or conceal missing generic behavior behind local shims.

The current Design snapshot is TensorCore `0.6.0`. Stage 2 proposes an exact
dependency candidate but has not implemented or tested it. A future work order
must record the exact constraint and tested commit actually used.

TensorDSLab's exact coordinate/index/layout, readout field, dtype, snapshot,
output, workspace, device, and lifetime rules remain package architecture in
`docs/architecture/`. This governance overlay neither changes nor ratifies
those scientific contracts.

## Determinism, Stochastic Behavior, And Parity

Behavior is deterministic unless an accepted contract says otherwise.
Coordinate-addressed stochastic Design uses caller-owned `RngSpec`, explicit
seed and namespace, operation role, semantic coordinates, preflight before
consumption, same-backend exact repeatability, and cross-backend
distributional agreement. The exact RNG algorithm and supported backend set
remain open and gate stochastic implementation.

Donor behavior is parts-bin evidence. Every promoted behavior or parity claim
names its source, comparison boundary, classification, assumptions,
observables, acceptance criteria, exclusions, intentional divergences, and
revisit triggers in `docs/parity.md`. No production parity is claimed while
the package has no implementation.

## Boundary Validation And Public API Discipline

Validate strongly when external, configuration, durable, cross-package, ID,
scalar, or TensorCore values enter a typed path. Accepted immutable records may
then be trusted downstream with narrow operation-specific preconditions.
`validate_*` reports violations and never repairs, fills, casts, writes, or
conceals missing work.

Public functions, records, fields, and constants are typed. Public exports are
deliberate. Historical aliases and compatibility shims are absent unless a
future Design decision accepts a bounded, evidenced window. Placeholder
modules and APIs are prohibited.

## Dormant-Surface Evidence And Activation Triggers

No common rule is wholly Not applicable. The following implementation
surfaces are dormant qualifications inside applicable rules:

| Dormant surface | Exact current evidence | Activation trigger |
| --- | --- | --- |
| Distribution/installability | Baseline and adopted candidate contain no `pyproject.toml` or package directory; Stage 2 metadata remains provisional | Design dispatches and implements Stage 2 package metadata |
| Runtime dependencies and TensorCore consumption | No production imports or dependency declaration exist; `docs/architecture/tensors.md` is Design-only | Stage 2 records and tests the exact dependency constraint and commit |
| Production tests and public exports | No `tests/` or importable package exists | First production package stage creates the corresponding surfaces |
| Deterministic waveform kernels and execution-workspace substrate | Stage 3 is a planning label only; no transform, output, preflight, or workspace code exists | Design accepts and dispatches a focused Stage 3 work order |
| RNG, charge, and stochastic-noise kernels | Exact algorithm/backend and some scientific choices remain open; no code exists | Focused Stage 4 and Stage 5 decisions and work orders are accepted |
| TensorG4DS bridge | Upstream public GPU product/device/layout contract is not frozen | Affected Designs accept the Stage 7 handoff contract |
| Durable caches and round trips | No durable format, loader, writer, or compatibility target exists | Design accepts a focused Stage 8 cache contract |
| TensorML/DAG/integration surfaces | Adapters, operation specs, executables, and recipes are deferred | Affected Designs accept a focused Stage 9 integration contract |
| Runtime/import/dependency/environment/post-merge commands | Only documentary checks are executable today | Each implemented surface supplies exact commands in its work order |

Dormancy supplies no Production conformance evidence and does not imply a
whole-rule exclusion.

## Documentation And Verification

Documentation drift is substantive. Work orders and handoffs identify every
source updated or checked unchanged. TensorDSLab's documentation matrix in
`AGENTS.md` and `CONTRIBUTING.md` routes changes to implementation, domain
architecture, TensorCore integration, parity, design, decisions, validation,
and onboarding sources.

The current executable minimum is `git diff --check` plus local-link, heading,
code-fence, stale-term, raw-ID, state, changed-file, and generated-output
checks. Python tests, import checks, dependency isolation, environment checks,
integration checks, and runtime post-merge commands activate only with their
implemented surfaces.

## Routing And Coordination

Profile B is Disabled. Adoption creates no `.agents`, ignore rule, registry,
route table, private live-route store, Moderator cache, or Active route. Raw
task identifiers remain private.

Coordination is Deferred and unused. TensorDSLab Design is the direct verified
procedural fallback. A future activation requires a concrete recurring topic,
accepted charter and Design-return path, adopted privacy/storage/replacement
procedures, verified fallback and route, no discrepancy, and explicit Design
and user authorization.

## Deviations And State

Accepted deviations: none.

```text
package adoption: Adopted
conformance: Not evaluated
prospective conformance profile: Documentation
Coordination: Deferred
Profile B: Disabled
decision: TDSLAB-GOV-D001 (Issued)
```

`TDSLAB-GOV-D001` changes only package adoption state. A later conformance
evaluation, routing bootstrap, Coordination activation, and Stage 2 dispatch
remain separate actions.
