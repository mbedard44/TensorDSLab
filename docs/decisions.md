# Decisions

This document records accepted and open design decisions. Keep historical
entries when they explain why a path was superseded or deferred.

## Accepted

### TensorDSLab Is Tensor-Native From The Start

TensorDSLab is a clean-slate detector data-lab package. It should define its
product and cache semantics directly while using TensorCore as the generic
tensor-native backbone.

### TensorCore Owns Generic Tensor Contracts

TensorCore owns generic tensor identity, axes, layouts, fields, collections,
selections, batching, movement, validation, and pure tensor operations.
TensorDSLab should import those surfaces from `tensor_core` and should not
mirror, fork, or broaden them locally.

### TensorDSLab Owns Detector Data-Lab Products

TensorDSLab owns detector, readout, and future reconstruction examples,
products, product labels, domain builders, validators, and domain-specific
tensor renderings. Cache records and durable cache IO are future TensorDSLab
surfaces after the in-memory product model is accepted.

### Project Naming Follows The Tensor Ecosystem

The project/display folder is `TensorDSLab`; the Python import package should
be `tensor_dslab` when production package code is accepted.

### Documentation Comes Before Production Code

The initial stage is documentation-only. It establishes workflow, repository
identity, ownership boundaries, validation expectations, and staged work-order
discipline before package code begins.

### Build In-Memory Products Before IO

TensorDSLab should define its MVP early, but the first production priority is
the in-memory product graph and its relationships. Source records, example
identity, detector products, readout products, future reconstruction products,
and TensorCore-backed renderings should be coherent before durable IO, cache
compatibility, compaction, DAG integration, or downstream adapter contracts are
introduced.

### DAG Compatibility Is Deliberate And Deferred

TensorDSLab should not add DAG-compatible specs or recipes until local
tensor-native product and cache contracts are accepted. Projects/dag remains
the owner of concrete orchestration.

### Downstream Integration Is Deferred

TensorDSLab should be designed in isolation until its local product contracts
are stable. Downstream model, training, evaluation, batching, checkpoint,
metric, and adapter requirements should not drive the first in-memory module
boundaries.

### First MVP Starts At Post-Binned Readout

The first production MVP should focus on the post-binned tensor-native readout
path: already-binned charge, stochastic charge transforms, waveform rendering,
physical waveform composition, and optional digitization. Source PE-hit
parsing, detector-window construction, charge binning, durable IO, cache
compatibility, DAG integration, and downstream adapters are deferred until the
post-binned contract is stable.

### TensorDSLab Gives TensorCore Records Domain Meaning

TensorCore remains the dense tensor spine. TensorDSLab should give
`TensorCollection` and related TensorCore records detector/readout product
meaning through accepted domain surfaces. Concrete wrapper classes, a generic
`Product` base, and ToyProduct-like examples are not accepted by default.
TensorDSLab should defer concrete tensor shape to runtime TensorCore layouts
while making product roles, field roles, semantic axis roles, sample metadata,
and stochastic coordinate inputs explicit.

### TensorDSLab Transforms Use Explicit Output Buffers

TensorDSLab domain transforms should prefer an `out=` convention. If `out` is
omitted, the method allocates and returns a new product. If `out` is supplied,
the method writes into `out` and returns `out`. `out` must be the correct
TensorDSLab product type with compatible TensorCore layout, device, dtype, and
semantic axis roles.

Mutation and allocation policy is runtime control, not TensorCore identity.
Do not encode mutation policy as a TensorCore `Id`, field ID, axis ID, or
coordinate, and do not introduce a persistent ambient mutation mode for core
readout transforms.

## Open

### First Production Work Order Details

The first MVP boundary is post-binned readout, but the exact first production
work order still needs to define target files, public surface names, field role
records, semantic axis role records, config records, promoted behavior, minimum
tests, and validation commands.

### Cache Compatibility Target

TensorDSLab has not yet decided whether the first durable cache stage should
target a compatibility-oriented format, a loader-compatible transition format,
or a new TensorCore-backed cache format.

### Donor-Code Promotion Policy Per Stage

Historical predecessor code is parts-bin material only. Each production stage
still needs to name which donor semantics, algorithms, fixtures, or tests are
being promoted and which old structures are intentionally left behind.

### Tensor Product Surface Names

The exact names for TensorCore-backed detector, readout, and reconstruction
tensor renderings are deferred. Product labels and TensorCore field IDs should
remain distinct unless a future design stage accepts a specific bridge.
