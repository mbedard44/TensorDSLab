# Stage 0 Documentation Spine

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
- TensorDSLab sits between g4ds11 detector simulation and future consumers.
- TensorDSLab owns detector, readout, and future reconstruction examples,
  products, product labels, validators, and domain-specific tensor renderings.
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
