# Package Governance

Package adoption: `Adopted`
Design decision: `TDSLAB-GOV-D001` — Issued
Conformance finding: `Not evaluated`
Coordination: `Deferred`
Profile B: `Disabled`

TensorDSLab adopts Tensor Ecosystem Governance Core `0.1.0` through
`TDSLAB-GOV-D001`. The package records here bind the exact accepted candidate
while leaving conformance, routing, Coordination, and production authority as
separate states.

## Exact Inputs And Package Baseline

- TensorDS20k Phase 2 authorization baseline:
  `17663127d2a35bdfa141b8c18029d117a5a6b014`.
- Governance Core decision: `GOV-D001`.
- Governance Core manifest-file SHA-256:
  `45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8`.
- Council Charter decision: `COUNCIL-D001`.
- Council Charter manifest-file SHA-256:
  `343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916`.
- TensorDSLab Design and repository source baseline:
  `151b61fdc36475498219ee5fe7b045a3a72c2d09`.
- Adopted TensorDSLab candidate:
  `d634401a853915edeb4f83df4a4943b3553deced`.
- TensorDSLab Design decision: `TDSLAB-GOV-D001` — Issued.

The Phase 3 completion report records the later state-record closeout commit.
That external record avoids placing a self-referential commit hash inside the
commit it identifies.

## Adopted Package Records

- [Adoption Record](adoption_0_1_0.md): declaration, immutable inputs, exact
  accepted candidate, Design finding, decision, states, and non-effects.
- [Package Overlay](overlay.md): TensorDSLab-specific workflow, maturity,
  engineering, routing, and verification rules.
- [Semantic Rule Map](rule_map_0_1_0.md): exactly one mapping for every common
  operational and engineering rule.

Accepted deviations: none. No deviations file is created while that set is
empty.

## Authority And Source Precedence

TensorDSLab Design owns package architecture, contracts, scope, work orders,
governance adoption, deviations, and any later conformance finding. The
ratified common records now govern adopted ecosystem process. They do not
create TensorDSLab scientific or package architecture.

When records conflict, stop the affected work, identify the exact package and
cross-package sources, and return the contradiction to every affected Design
authority. Work orders and tests execute accepted architecture; they do not
create or override it.

## Active-Development Posture

TensorDSLab and the surrounding ecosystem are in active development and
pre-deployment:

```text
ecosystem_delivery_maturity: Active development / pre-deployment
deployability_claim: None
backward_compatibility_guarantee: None
compatibility_evidence: Exact named repository, environment, device/backend,
                        and execution-mode baselines only
```

Existing APIs and designs remain changeable through their owning Design
authorities. A successful check at one exact package, Python, dependency,
device/backend, and execution-mode tuple does not imply compatibility with
historical, future, or untested tuples. TensorDSLab's same-device residency and
no-silent-host-materialization constraints are Design targets, not evidence
that a cross-package handoff is implemented or compatible.

## Current Routing Posture

Profile B is disabled and not instantiated. There is no package registry,
`.agents` path, committed route table, private live-route store, Moderator
cache, or Active Coordination route. Raw task identifiers are private and must
not appear in durable package records.

Coordination remains Deferred. Procedural requests use the directly verified
TensorDSLab Design fallback without converting that fallback into registry
state or letting silence stand for package assent.

## Adoption Decision And Closeout

TensorDSLab Design reviewed the unchanged fixed candidate
`d634401a853915edeb4f83df4a4943b3553deced`, accepted it without conditions,
and issued `TDSLAB-GOV-D001`. No Implementation, Validation, Review,
independent documentation Review, or Coordination role was activated for the
documentation-only gate.

Conformance evaluation, Profile B, Coordination activation, production
dispatch, cross-package architecture, and dossier maintenance remain separate
decisions or procedural records. The external working dossier remains
scientifically accurate, but its `Not adopted` line is a pre-decision snapshot
superseded for package state by `TDSLAB-GOV-D001`.
