# Governance Core 0.1.0 Adoption Candidate

Document status: Phase 2 candidate
Package adoption state: `Not adopted`
Proposed Design decision: `TDSLAB-GOV-D001` — Proposed / Unissued
Prospective conformance profile: `Documentation`
Conformance finding: `Not evaluated`

## Candidate Declaration

```text
governance_core_version: 0.1.0
package_adoption_state: Not adopted
conformance_profile: Documentation
conformance_finding: Not evaluated
package_design_baseline: 151b61fdc36475498219ee5fe7b045a3a72c2d09
repository_or_document_baseline: fixed Phase 2 candidate commit named by the completion report and Phase 3 handoff
design_ratification_record: TDSLAB-GOV-D001 (Proposed / Unissued)
package_overlay: docs/governance/overlay.md
semantic_rule_map: docs/governance/rule_map_0_1_0.md
accepted_deviations: none
registry_storage_profile: Disabled
coordination_status: Deferred
last_verified: 2026-07-10
verified_by: TensorDSLab/default/Design
```

This declaration is deliberately still a candidate. The fixed candidate
commit is recorded outside itself in the Phase 2 completion report, then
passed unchanged to Phase 3. The exact source baseline above remains the
package Design baseline from which this governance-only candidate was derived.

## Immutable Common Inputs

Phase 2 central authorization:

```text
TensorDS20k baseline: 17663127d2a35bdfa141b8c18029d117a5a6b014
```

Governance Core `0.1.0`:

```text
decision: GOV-D001
ratified proposal: 0.1.0-proposal.2
manifest-file SHA-256: 45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8
manifest verification: all eight entries verified
```

Council Charter `0.1.0`:

```text
decision: COUNCIL-D001
ratified proposal: 0.1.0-proposal.1
manifest-file SHA-256: 343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916
manifest verification: all three entries verified
```

The Council Charter record supplies procedural context. It does not make the
council or Moderator a TensorDSLab authority.

## Candidate Scope

This candidate consists only of:

- this declaration;
- the package [overlay](overlay.md);
- the complete [semantic rule map](rule_map_0_1_0.md);
- the package-governance [index](index.md); and
- narrow cross-references and candidate-state synchronization in
  `AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `docs/overview.md`, and
  `docs/validation.md`.

`docs/decisions.md` is unchanged. Stage 2 remains Design-complete and
undispatched. No package code, metadata, test, dependency, cache, adapter,
registry, route, or production behavior is part of the candidate.

## Package Overlay And Rule Map

The package overlay specializes the common process for TensorDSLab without
copying common text into every package source. It preserves the existing
documentation-only Design lifecycle, exact role boundaries, Review-owned
clean fast-forward closeout, three-dispatches-each-way I/V limit, TensorCore
consumer boundary, readout semantics, parity rules, and documentary checks.

The semantic map contains one complete entry for each common rule. No
whole-rule Not-applicable disposition is used. Dormant implementation surfaces
are qualifications inside applicable rows and name exact absence evidence plus
an activation trigger.

## Deviations

Accepted deviations: none.

A future weakening of a common safeguard requires a focused TensorDSLab Design
decision. If it affects another package, every affected Design authority must
ratify the same immutable deviation proposal. Phase 2 authorizes no such
change.

## Pre-Deployment And Compatibility Non-Claims

TensorDSLab is documentation-only and not deployed. The candidate makes no
claim of:

- installability or deployability;
- release readiness or collaboration certification;
- backward compatibility or API stability;
- compatibility with historical, future, or untested package versions;
- implemented TensorCore consumption;
- a working TensorG4DS or TensorML handoff;
- accelerator correctness, performance, or allocation behavior; or
- a functioning vertical slice.

Compatibility evidence, when a later implemented surface exists, is limited
to exact named repository commits, Python and dependency versions,
device/backend, and execution-mode assumptions. The designed same-GPU and
no-silent-host-materialization constraints are not cross-package compatibility
evidence.

The exact package posture is:

```text
ecosystem_delivery_maturity: Active development / pre-deployment
deployability_claim: None
backward_compatibility_guarantee: None
compatibility_evidence: Exact named repository, environment, device/backend,
                        and execution-mode baselines only
```

## State Separation

These namespaces remain distinct:

```text
Governance Core decision: Ratified
TensorDSLab package adoption: Not adopted
prospective conformance profile: Documentation
TensorDSLab conformance finding: Not evaluated
Coordination: Deferred
Profile B: Disabled
Phase 2 evidence result: candidate preparation only
Stage 2 production work order: Design-complete / undispatched
```

No state implies another.

## Routing And Coordination

Profile B remains disabled and no registry or cache exists. The package stores
no raw task identifier in committed records. TensorDSLab Design is the direct
procedural fallback while Coordination remains Deferred. This candidate does
not contact, verify, activate, close, or route work through Coordination.

## Dormant Surfaces

Package metadata, runtime imports, production tests, external bridges,
stochastic kernels, caches, and integration commands do not exist at the
source baseline or in this candidate. The overlay and rule map identify the
authoritative absence evidence and the focused stage that activates each
obligation. Dormancy is not a whole-rule exclusion and cannot support a
Production conformance claim.

## Author-Side Verification

TensorDSLab Design performed the documentary candidate checks on 2026-07-10:

- `shasum -a 256 manifest.sha256` returned the exact Governance Core
  manifest-file SHA-256
  `45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8`
  and Council Charter manifest-file SHA-256
  `343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916`;
- `shasum -a 256 -c manifest.sha256` verified all eight Governance Core
  entries and all three Council Charter entries at their immutable proposal
  directories;
- the semantic map contains exactly 13 unique `OP-*` rows and 12 unique
  `ENG-*` rows, with 25 occurrences of every required schema field, six
  `Adopted` dispositions, 19 `Stronger local rule` dispositions, no
  whole-rule Not-applicable disposition, and no deviation row;
- every mapped package source and named section resolves in the candidate or
  in a check-only authoritative source;
- `git diff --cached --check` passed on the complete nine-path candidate, and
  a repository-wide Markdown audit covered 18 files and 65 local links with no
  missing target, missing anchor, heading jump, or unbalanced fence;
- the changed-file set is exactly the four candidate records plus the five
  authorized synchronized references; architecture, parity, decisions, and
  implementation-stage records are unchanged; and
- scans found no raw task identifier and no filesystem `.agents` path,
  registry/cache artifact, package metadata, production module, test suite, or
  premature package-state claim.

The Phase 2 completion report records the exact containing commit and final
clean status after commit creation. No runtime, import, dependency, device, or
integration test applies because the corresponding package surfaces do not
exist; this documentary evidence provides no production or compatibility
finding.

## Proposed Phase 3 Gate

A later authorized Phase 3 must:

1. receive the exact clean candidate commit named in the Phase 2 report;
2. have TensorDSLab Design validate and review that fixed commit without
   modifying architecture or current states;
3. reject or supersede that candidate if a finding requires any byte change,
   then subject the replacement fixed clean commit to the complete Phase 3
   review before a decision;
4. confirm that the candidate presented for decision is exactly the fixed
   commit that completed Phase 3 review;
5. ask TensorDSLab Design to issue, revise, defer, or reject the proposed
   decision; and
6. if and only if Design issues `TDSLAB-GOV-D001`, perform the separately
   authorized controlled state-record closeout.

The final Design decision and closeout are not part of Phase 2. Phase 3 does
not imply use of Implementation, Validation, Review, independent documentation
Review, or Coordination; any such role activation requires separate
authorization.

## Non-Effects

This candidate does not:

- adopt Governance Core `0.1.0`;
- establish Documentation or Production conformance;
- activate Profile B, a registry, a cache, a route, or Coordination;
- accept a dependency, data-flow, device, compatibility, migration, adapter,
  buffer, stream, or vertical-slice contract;
- create production code, package metadata, tests, or public API;
- dispatch Stage 2; or
- authorize sibling-repository work.
