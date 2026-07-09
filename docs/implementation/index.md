# Implementation Stages

This directory holds TensorDSLab implementation work orders. Stage documents
are handoff-ready instructions for the persistent Implementation, Validation,
and Review threads described in [AGENTS](../../AGENTS.md).

Keep stages small enough that Implementation can finish, Validation can
critique, and Review can reason about the diff.

## Stages

- [Stage 0: Documentation Spine](stage_0_documentation_spine.md)

## Candidate Future Stages

These are planning labels, not accepted work orders. Design must write a
focused stage document before Implementation starts any of them.

### Stage 1: Post-Binned Readout MVP Architecture

Possible goal: define the first post-binned readout MVP and its TensorCore
domain-surface architecture. The stage should specify binned charge,
stochastic charge transforms, waveform products, physical waveform
composition, optional digitization, semantic axis roles, field roles, config
records, explicit `out=` transform behavior, validation boundaries, and
promoted reference semantics.

This stage should explicitly decide whether TensorDSLab needs concrete product
wrapper classes, direct product functions, small role records, or another
surface shape. Do not copy a generic `Product` base or ToyProduct-like pattern
without a TensorDSLab-specific reason.

This stage should defer source PE-hit parsing, detector-window construction,
charge binning, durable IO, package metadata, DAG surfaces, cache
compatibility, and downstream adapters.

### Stage 2: Common Identity And Boundary Records

Possible goal: define the smallest shared TensorDSLab IDs, row identity,
source provenance, scalar wrappers, and validation primitives needed by the
first in-memory product stage.

### Stage 3: Source And Detector In-Memory Contract

Possible goal: implement typed source records, `ExampleContext`, and
`DetectorExample` construction in memory. Avoid durable IO and compatibility
surfaces unless Design explicitly narrows an input-boundary exception.

### Stage 4: Readout In-Memory Contract

Possible goal: implement `ReadoutExample` construction and readout product
records in memory once the post-binned TensorCore-backed readout product
surfaces are stable. Keep durable caches, DAG compatibility, and downstream
handoff deferred unless a focused work order accepts a narrow bridge.

### Stage 5: TensorCore-Backed Product Rendering

Possible goal: define the first explicit bridge from TensorDSLab domain
products into TensorCore axes, layouts, fields, and collections.

### Stage 6: Package And Import Foundation

Possible goal: create minimal package metadata, the `tensor_dslab` package
root, import smoke tests, and editor/runtime path configuration once there is a
clear in-memory contract to package.

### Stage 7: Durable Cache Contract

Possible goal: decide whether the first cache target is compatibility-oriented,
loader-compatible, or a new tensor-native format, then implement only the
accepted cache boundary.

### Stage 8: Integration Surfaces

Possible goal: add DAG-compatible operation specs, executable doors, recipe
fragments, or downstream adapters only after in-memory products, TensorCore
renderings, and durable cache contracts are stable.

## Expected Stage Discipline

Each stage should stay scoped to its work order. If implementation reveals a
real contradiction in TensorDSLab product ownership, TensorCore layout
semantics, in-memory relationships, cache shape, or future integration
compatibility, stop and send the issue back to Design rather than widening the
branch.

Stage work should preserve the shared `AGENTS.md` and `CONTRIBUTING.md`
standard unless the stage explicitly changes repository-wide workflow or
engineering expectations. Meaningful deviations from the sibling-repository
shape should be documented in `docs/decisions.md`, the relevant architecture
doc, or the stage work order.

Before Review, Implementation should run the commands named in the stage work
order and report known risks or deferred items.
