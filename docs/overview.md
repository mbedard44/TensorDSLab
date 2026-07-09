# Overview

This is the quick architecture map for TensorDSLab design, implementation,
validation, and review threads.

## Project Identity

TensorDSLab is a clean-slate, tensor-native detector data-lab package. It
should convert accepted g4ds11 detector-simulation output into typed detector,
readout, and future reconstruction products, with durable cache contracts
deferred until a stage accepts durable outputs.

The intended chain is:

```text
g4ds11 -> TensorDSLab -> future consumers
```

TensorCore provides the generic tensor substrate. TensorDSLab owns domain
examples, product labels, cache semantics, validators, and domain-specific
tensor renderings. Downstream ML packages own source adaptation for model
training, batching policy, models, objectives, metrics, checkpoints, training,
and evaluation, but TensorDSLab should not design around those downstream
integration contracts until its local product model is stable.

## Current Local Focus

The repository is in initial Design documentation mode. The first accepted
local spine is documentation-only:

```text
AGENTS.md / CONTRIBUTING.md
  -> overview, design, decisions, validation
  -> implementation-stage work orders
  -> future architecture docs and package code
```

No production package, tests, cache schema, DAG surface, or copied donor code
has been accepted yet.

The first accepted MVP direction is the post-binned tensor-native readout path:
start from already-binned charge, then build stochastic charge transforms,
waveform products, physical waveform composition, and optional digitization
around TensorCore records. Source PE-hit parsing, detector-window
construction, charge binning, IO, cache, DAG, and downstream adapter boundaries
come later.

TensorCore is the dense tensor spine. TensorDSLab gives TensorCore records
detector/readout meaning through accepted domain surfaces and should defer
concrete tensor shape to runtime layouts while keeping product roles, semantic
axis roles, and transform contracts explicit. Concrete wrapper classes remain
a design option, not an accepted default.

## Intended Product Flow

TensorDSLab should preserve this product dependency rule unless a future
Design decision changes it:

```text
g4ds11 native output
  -> DetectorExample
  -> ReadoutExample
  -> ReconstructionExample
  -> future consumer-facing tensor/product views
```

This is a dependency rule, not a scheduling policy. Orchestration, durable
cache compatibility, compaction, and downstream handoff are later concerns.

## Package Shape

Use the ecosystem naming convention:

```text
TensorDSLab/
  tensor_dslab/
```

The outer project/display folder is `TensorDSLab`; the Python import package
should be `tensor_dslab` when package code is accepted. Do not create a flat
TitleCase import package.

## Documentation Map

- [Design](design.md): architecture thesis, ownership boundaries, and
  non-goals.
- [Decisions](decisions.md): accepted and open design decisions.
- [Validation](validation.md): validation philosophy and early expectations.
- [Implementation Stages](implementation/index.md): staged work orders and
  scope limits.

Future architecture docs should be added only when they carry real contracts,
such as detector products, readout products, reconstruction products, caches,
or TensorCore-backed tensor renderings.

Implementation threads should treat these docs, `CONTRIBUTING.md`, and
`AGENTS.md` as the source of truth. If implementation uncovers a contradiction,
route it back to Design before changing architecture.
