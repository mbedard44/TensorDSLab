# Design

## Core Thesis

TensorDSLab is a tensor-native detector Product library. It owns reusable
scientific transformations over exact semantic tensors. It does not own an
application's workflow graph.

```text
application sources
  -> selected TensorDSLab Product classmethods
  -> application retention / IO / integration
```

TensorCore is the generic semantic tensor and addressed-distribution
substrate. TensorDSLab adds detector units, roles, exact Product and
coefficient Specs, typed Kernel collections, Config punchcards, scientific
equations, private stochastic identities, and publication validation.

## Product Composition

Applications may compose `Photoelectrons -> Charge`, combine multiple
avalanche-compatible semantic sources into Charge, produce one waveform
without requesting the others, or assemble a collaboration-specific graph
outside this package. TensorDSLab requires the complete semantic/unit/device
relationship at each Product boundary but does not encode a universal chain.

The five generated Product families prepare all policy before execution.
Prepared Configs retain exact source-Spec provenance and aligned Kernel
geometry. Source order is significant; structurally equal replacement Specs
are reusable; changed structure is rejected before tensor arithmetic,
allocation, or RNG.

## Deferred Application Boundary

DS20k/Silex profiles, detector-window construction, photoelectron binning,
product orchestration, request retention, TensorG4DS/TensorML adapters, cache,
IO, and campaign execution belong to focused application or later package
authorities. Maintenance 15 provides no application demo and no compatibility
facade for the retired embedded readout workflow.

See the [architecture record](implementation/maintenance_15_spec_composed_products_and_application_boundary.md)
and [work order](implementation/maintenance_15_execution_work_order.md) for the
exact API, science, tests, and evidence boundary.
