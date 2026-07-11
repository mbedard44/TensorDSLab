# Package Governance

Candidate status: Phase 2 candidate prepared for later review; not adopted
Package adoption: `Not adopted`
Conformance finding: `Not evaluated`
Coordination: `Deferred`
Profile B: `Disabled`

This directory contains TensorDSLab's package-local candidate for adopting
Tensor Ecosystem Governance Core `0.1.0`. It is evidence for a future Phase 3
review and Design decision. It does not itself adopt the core, establish
conformance, activate a route, or authorize production work.

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

The fixed commit containing this candidate is recorded by the Phase 2
completion report and supplied unchanged to the future Phase 3 handoff. That
external commit record avoids placing a self-referential commit hash inside
the commit it identifies.

## Candidate Records

- [Adoption Candidate](adoption_0_1_0.md): declaration, immutable inputs,
  current states, non-effects, and the proposed but unissued Design gate.
- [Package Overlay](overlay.md): TensorDSLab-specific workflow, maturity,
  engineering, routing, and verification rules.
- [Semantic Rule Map](rule_map_0_1_0.md): exactly one mapping for every common
  operational and engineering rule.

Accepted deviations: none. No deviations file is created while that set is
empty.

## Authority And Source Precedence

TensorDSLab Design owns package architecture, contracts, scope, work orders,
governance adoption, deviations, and any later conformance finding. The
ratified common records govern ecosystem process only after package-local
adoption. They do not create TensorDSLab architecture.

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

## Phase 3 Boundary

Phase 2 ends with a fixed clean candidate commit. A later separately
authorized Phase 3 must receive that exact commit for direct TensorDSLab Design
review. If a finding requires any byte change, the current candidate is
rejected or superseded and the replacement fixed clean commit must receive the
complete Phase 3 review. The proposed decision `TDSLAB-GOV-D001` remains
unissued until Design explicitly accepts the exact reviewed commit. Only that
later decision may change package adoption state.

Phase 3 is not authorized by these files. Conformance evaluation, Profile B,
Coordination activation, production dispatch, and cross-package architecture
remain separate decisions. Phase 3 does not imply Implementation, Validation,
Review, independent documentation Review, or Coordination activation; any such
role use requires separate authorization.
