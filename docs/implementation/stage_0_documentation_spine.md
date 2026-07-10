# Stage 0 Documentation Spine

Status: completed historical documentation stage. Stage 1 supersedes its open
architecture questions; keep this page as the record of the initial spine and
its then-current scope.

The later accepted ecosystem boundary also supersedes Stage 0's direct
`g4ds11 -> TensorDSLab` description. The current target is
`G4DS -> TensorG4DS -> TensorDSLab -> TensorML`: native G4DS parsing and
low-level analysis stay upstream, while a future typed same-GPU bridge maps an
accepted TensorG4DS product into TensorDSLab semantics. The old bullet below is
retained only as historical Stage 0 scope.

The superseding Stage 1 decision uses one primary
`ReadoutCollection(TensorCollection)` rather than one semantic subclass per
charge or waveform field. It is a structurally immutable partial snapshot over
six recognized `readout.*` field IDs, with canonical required axis IDs, a
typed `SampleGrid`, a conditional `DigitizedWaveformSpec`, and field
transforms.
A later accepted package decision places concrete domain packages directly
under `tensor_dslab`, such as `tensor_dslab.common` and
`tensor_dslab.readout`, without an intermediate `tensor_dslab.domain`
namespace. `ChannelId` is a common coordinate because readout and future
reconstruction reuse the same identity.
A future `ReadoutExample` is at most a thin provenance/context wrapper around
that collection. Accordingly, Stage 0's statements that exact product and
tensor-rendering APIs were undecided are historical facts, not current open
questions.

## Task

Create the initial process, design, validation, and implementation-stage
documentation spine for TensorDSLab.

This is a documentation-only stage. Its job is to make the new repository safe
for future staged work, not to create package code or commit to a cache schema.

## Scope

Implementation may create or update only:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/overview.md
docs/design.md
docs/decisions.md
docs/validation.md
docs/implementation/index.md
docs/implementation/stage_0_documentation_spine.md
```

## Design Contract

The docs should establish these accepted directions:

- TensorDSLab is a clean-slate, tensor-native detector data-lab package.
- At the time, TensorDSLab was described as sitting directly between g4ds11
  detector simulation and future consumers; the supersession note above is the
  current contract.
- At the time, TensorDSLab was also described as owning detector, readout, and
  future reconstruction examples. Current docs narrow this to post-TensorG4DS
  semantics and do not accept the old wrapper names by default.
- TensorDSLab owns future cache contracts only after in-memory product
  contracts are accepted.
- TensorDSLab should define the MVP early while prioritizing in-memory product
  modules and relationships before IO, cache, DAG, or downstream integration
  boundaries.
- TensorCore owns generic tensor identity, layout, field, collection,
  selection, batching, movement, validation, and pure tensor operation
  primitives.
- Historical predecessor code is parts-bin material only, not an architecture
  template to copy blindly.
- TensorML is a workflow and documentation style reference, not a detector
  data-lab domain template.
- Projects/dag owns campaign orchestration; TensorDSLab DAG-compatible and
  downstream integration surfaces are deferred until local contracts are
  accepted.
- The project/display folder is `TensorDSLab`; the future Python import package
  should be `tensor_dslab`.

## Non-Goals

Do not create:

- production package files;
- `tensor_dslab/` package skeletons;
- placeholder module trees;
- tests;
- package metadata;
- cache schemas;
- DAG-facing `executables/`, `operations/`, or `recipes/`;
- downstream adapter surfaces;
- local DAG factories;
- copied donor code from historical predecessor repositories.

Do not decide:

- the first production implementation slice;
- exact detector, readout, reconstruction, cache, or tensor-rendering APIs;
- exact durable cache compatibility policy;
- downstream adapter boundaries;
- final environment or dependency policy;
- DAG operation specs or recipe shapes.

## Implementation Notes

Use TensorML's `AGENTS.md` and `CONTRIBUTING.md` as the workflow base, then
replace TensorML process semantics with TensorDSLab product and cache
semantics. Cross-reference TensorCore for strict tensor terminology and
package-shape rules.

Keep docs short enough to stay maintainable. Prefer clear non-goals and staged
open questions over premature detail.

## Minimum Checks

Before Review, run:

```bash
git diff --check
```

If the repository is not yet initialized as git, report that and run an
equivalent whitespace sanity check.

For this docs-only stage, the full test suite is not applicable.

Report:

- files created or updated;
- confirmation that no production code, tests, package skeleton, cache schema,
  or DAG-facing surface was added;
- any wording that still needs user review.

## Escalate To Design If

- docs need to define production APIs;
- docs need to choose cache compatibility policy;
- docs need to promote donor code;
- docs need to create package modules to make examples concrete;
- docs need to change TensorCore public contracts;
- docs need to accept DAG-facing operation specs or recipe shapes.
