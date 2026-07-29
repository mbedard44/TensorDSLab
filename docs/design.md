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

The six generated Product families prepare all policy before execution.
Prepared Configs retain exact source-Spec provenance and aligned Kernel
geometry. Source order is significant; structurally equal replacement Specs
are reusable; changed structure is rejected before tensor arithmetic,
allocation, or RNG.

Reusable semantic admission is composed from narrowly named private
requirements. Specs admit metadata and geometry; Fields and Kernels admit
their exact Specs and represented values. Short scientific laws unique to one
semantic leaf stay visible on that leaf. Product preparation retains
relationships that require aligned objects. This is direct composition, not a
validator framework or a public requirements API.

Each typed `*Kernels` collection is owned by its Product's singular
`kernel.py`, next to the exact Kernel vocabulary it admits. Config modules own
only same-type Config punchcards and compose those collections.

`EncodedWaveform` is the terminal reusable DAQ/readout Product, not a workflow
owner. It selects dense retained support from one exact DigitizedWaveform with
literal int64 policy Kernels and preserves every retained code. Applications
own whether and how that Product is serialized or passed to later
reconstruction.

## Deferred Application Boundary

DS20k/Silex profiles, detector-window construction, photoelectron binning,
product orchestration, request retention, TensorG4DS/TensorML adapters, cache,
IO, and campaign execution belong to focused application or later package
authorities. This application ownership does not prohibit a narrowly
authorized package-neutral demonstration that composes public Product calls as
ordinary user code. Such a notebook does not define a universal chain,
profile, calibration, retention policy, or compatibility facade for the
retired embedded readout workflow.

See the [Maintenance 15 architecture record](implementation/maintenance_15_spec_composed_products_and_application_boundary.md),
its [execution work order](implementation/maintenance_15_execution_work_order.md),
the [Maintenance 16 requirements-ownership work order](implementation/maintenance_16_declarative_requirements_and_kernel_ownership.md),
the [Maintenance 17 quickstart work order](implementation/maintenance_17_application_neutral_readout_quickstart.md),
and the [Maintenance 18 raw-ZLE work order](implementation/maintenance_18_encoded_waveform_raw_zle.md)
for the exact API, science, tests, and evidence boundary.
