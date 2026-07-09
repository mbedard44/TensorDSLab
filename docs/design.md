# Design

## Core Thesis

TensorDSLab should define detector data-lab products around a tensor-native
backbone without turning generic tensor mechanics into local copies.

Shared TensorDSLab modules should use domain vocabulary for products and cache
boundaries:

- source records;
- example context and row identity;
- detector examples and products;
- readout examples and products;
- future reconstruction examples and products;
- product labels;
- cache manifests and cache rows;
- TensorCore-backed tensor renderings.

Generic tensor vocabulary belongs to TensorCore. Model, objective, training,
evaluation, metric, and checkpoint vocabulary belongs downstream unless a
future Design decision accepts a TensorDSLab-specific boundary.

## Build Philosophy

Define the MVP early, but build the system from the in-memory domain model
outward. The first production priority is not cache IO, DAG compatibility, or
downstream handoff; it is a coherent local product graph:

```text
source records
  -> ExampleContext / row identity / provenance
  -> DetectorExample / detector products
  -> ReadoutExample / readout products
  -> future ReconstructionExample / reconstruction products
  -> TensorCore-backed product renderings
```

IO boundaries should follow the product model, not lead it. Durable cache
schemas, table/array codecs, manifests, compaction, executable doors,
operation specs, recipe fragments, and downstream adapters are deferred until
the in-memory contracts are stable enough to deserve persistence or external
integration.

## Post-Binned Readout MVP

The first production MVP should focus on the post-binned tensor-native readout
path. TensorDSLab should start from an already-binned PE charge product and
defer source PE-hit parsing, event placement, detector-window construction,
charge binning, durable IO, cache compatibility, DAG integration, and
downstream adapters.

The initial readout chain should be:

```text
binned charge
  -> timing jitter
  -> dark counts
  -> crosstalk and afterpulses
  -> charge smearing
  -> pure waveform
  -> noise waveform
  -> physical waveform
  -> optional digitization
```

The old fixed-grid behavior is useful as a semantic reference, especially the
operation order and the rule that crosstalk and afterpulses use the same
post-jitter/post-dark-count source snapshot. TensorDSLab should not inherit a
fixed tensor rank, singleton batch convention, or package layout from that
reference.

## TensorCore Spine And Domain Semantics

TensorCore is the dense tensor spine. TensorDSLab gives detector and readout
meaning to that spine through accepted domain surfaces.

TensorCore owns what the dense tensor record is:

```text
TensorAxis / TensorAxes / TensorLayout
TensorField / TensorCollection
TensorAxisSelection / TensorFieldSelection
```

TensorDSLab owns what the record means in the detector data-lab process:

```text
binned charge role
readout charge role
pure waveform role
noise waveform role
physical waveform role
future detector / reconstruction roles
```

TensorDSLab should defer concrete tensor shape to scripts, runtime builders,
and TensorCore layouts. It should not defer tensor semantics. Domain surfaces
and configs should identify the required product role, field role, semantic
axis roles, sample-period metadata, and stochastic coordinate inputs needed by
the operation, while leaving axis order and optional extra axes to the
TensorCore layout provided at runtime.

A future stage may choose concrete product wrapper classes when they clarify a
real TensorDSLab role, but no generic `Product` base, `ToyProduct` pattern, or
specific wrapper hierarchy is accepted by default. The exact public surface
should be designed from TensorDSLab's readout needs rather than copied from a
reference repository.

## Explicit Output Transform Policy

TensorDSLab transforms should use explicit output buffers.

If `out` is omitted, the transform allocates and returns a new product:

```python
new_charge = time_jitter(charge, jitter)
pure = render_pure_waveform(charge, pulse)
```

If `out` is supplied, the transform writes into `out` and returns `out`:

```python
time_jitter(charge, jitter, out=charge)
scratch = empty_like(charge)
afterpulses(charge, afterpulse, out=scratch)
dark_counts(scratch, dark, out=charge)
render_pure_waveform(charge, pulse, out=pure)
```

`out` is runtime output policy, not TensorCore identity. Do not model mutation
policy as a TensorCore `Id`, coordinate, field ID, or axis ID. Do not use a
persistent ambient mutation mode for core readout transforms.

When supplied, `out` must be the correct TensorDSLab product type with a
compatible TensorCore layout, device, dtype, semantic axis roles, and product
meaning. This convention applies to product-preserving transforms and to
product-changing transforms whenever an output buffer is meaningful.

The accepted rule is explicit output behavior, not a decision that transforms
must be instance methods. Stage 1 should decide whether the clearest public
shape is method-oriented, function-oriented, or a small product-surface record
with methods.

## Ownership Boundaries

### g4ds11 Boundary

g4ds11 owns detector simulation execution and native simulation output. For the
initial rebuild, TensorDSLab should treat g4ds11 output as an external source
boundary to validate and load into typed records. TensorDSLab should not own
simulation execution policy.

### TensorDSLab Boundary

TensorDSLab owns the middle link:

- typed source records for accepted g4ds11 outputs;
- detector example construction and detector products;
- readout example construction and readout products;
- future reconstruction example construction and products;
- domain-specific tensor rendering over TensorCore primitives;
- future durable cache writing, loading, validation, and compaction after
  in-memory contracts are accepted;
- future integration surfaces after local contracts are accepted.

The domain-to-domain boundary is the typed example object, not a loose product
tuple:

```text
ExampleContext -> DetectorExample -> ReadoutExample -> ReconstructionExample
```

Source event IDs are provenance. They do not automatically define row identity.

### TensorCore Boundary

TensorCore owns generic tensor identity, axes, layouts, fields, collections,
selections, batching, movement, validation, and pure tensor operations.
TensorDSLab should import those surfaces from `tensor_core` instead of copying
or mirroring them.

TensorDSLab domain IDs may appear as TensorCore coordinates when they subclass
TensorCore `Id`. TensorCore should not import TensorDSLab or own domain
concepts.

### Downstream ML Boundary

Downstream ML packages own source adaptation for model training, split
planning, batching policy, models, objectives, metrics, training, evaluation,
checkpoints, and ML-specific artifacts.

TensorDSLab should not design its first stages around downstream package
requirements. Future consumers should be able to depend on stable typed
products or accepted adapters, but those adapters are deferred until the local
product graph is stable.

### Projects/dag Boundary

Projects/dag owns campaign fanout and fanin, scheduling, dispatch, retries,
repair, cancellation, status, concrete DAG objects, scheduler-visible grouping,
and cross-shard orchestration.

TensorDSLab may later expose DAG-compatible operation specs and recipe
fragments, but local product and cache contracts should be accepted first.

## Product Labels And Tensor Fields

TensorDSLab product labels are producer-owned durable labels, for example:

```text
detector.pe_hits
readout.charge
readout.waveform.pure
readout.waveform.noise
readout.waveform.physical
```

TensorCore `TensorFieldId` values identify fields inside a compiled tensor
layout. They may correspond to product labels through an explicit rendering or
adapter contract, but the namespaces should stay distinct.

The first tensor-native readout work should make axes, layouts, coordinates,
and indices explicit. It should not smuggle product labels, channel identity,
sample positions, or row identity through implicit array positions.

## Initial Package Direction

The likely future package tree is a roadmap, not permission to create
placeholder modules:

```text
tensor_dslab/
  domain/
    common/          # shared IDs, quantities, validation, only when real
    g4ds11/          # source boundary, only when real
    detector/        # detector examples and products
    readout/         # readout examples and products
    reconstruction/  # future reconstruction products
    tensors/         # product-to-TensorCore renderings, only when real
    caches/          # future durable cache bridge, after in-memory contracts
  executables/       # future task doors, only when accepted
  operations/        # future operation specs, only when accepted
  recipes/           # future recipe fragments, only when accepted
```

Required domains are those accepted by a concrete implementation stage. A
missing folder is better than a decorative folder.

## Parts-Bin Policy

Historical predecessor code is parts-bin material only. It can inform
scientific behavior, cache semantics, algorithms, tests, and naming lessons,
but it is not binding architecture.

Reference locations:

```text
/Users/mbedard/Projects/TensorCore
/Users/mbedard/Projects/TensorML
```

Reference precedence:

1. TensorCore is authoritative for generic tensor contracts.
2. TensorDSLab docs are authoritative for accepted product and cache behavior.
3. TensorML is a workflow, documentation, and tensor-spine style reference,
   not a detector data-lab domain template.

Promote only the reviewed concept that fits the tensor-native design. Do not
inherit donor architecture automatically.

## Current Non-Goals

- No production package code.
- No package metadata or dependency lock-in.
- No copied donor code.
- No tests until a production or validation contract exists.
- No cache schema commitment.
- No durable IO, manifest, compaction, or cache compatibility requirements
  before the in-memory product model is accepted.
- No DAG operation specs, recipe fragments, executable doors, local DAG
  factories, or downstream adapter surfaces.
- No downstream model, training, evaluation, objective, metric, checkpoint, or
  artifact surfaces.
- No local fork of TensorCore concepts.
