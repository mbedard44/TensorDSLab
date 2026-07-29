# Decisions

## Accepted Current Direction

### Maintenance 15 Selects Spec-Composed Products

TensorDSLab selects exact TensorCore `0.22.0` and a reusable parts-bin
architecture. Generic Coordinates compose semantic axes; quantity axes add
physical scale and Unit without changing the integer coordinate lattice.
Quantity-aware FieldSpec and KernelSpec intermediates compose exact Product
and coefficient Specs.

Products are independent transformations rather than fixed nodes in a
package-owned readout workflow. Each generated Product owns exact
`create/prepare/produce/validate` classmethods and one Product-specific Config
punchcard. Preparation returns a fresh Config of the same exact type with
aligned Kernels and immutable source-Spec provenance. There are no Runtime,
Plan, generic Product, generic Config, or generic readout types.

The package no longer owns `tensor_dslab.readout`, `ReadoutConfig`,
`ReadoutCollection`, `simulate_readout`, `ds20k_veto`, or the readout demos.
Applications own orchestration, source assembly, retention, and persistence.
No compatibility shim is selected.

### Scientific And RNG Ownership

The selected Charge, pulse, noise, analog, and digitization equations remain
TensorDSLab-owned. The literal Maintenance 12 timing/branching/pulse laws and
the eight compact stochastic roles are preserved. Exact completed stochastic
bytes are not promoted across changed orchestration or environments; current
evidence proves same-stack replay, invariants, and independent laws.

TensorCore owns generic addressed RNG and Distribution mechanics. TensorDSLab
owns role keys, address schemas, scientific parameters, count ceilings,
generation semantics, and completed Product validation.

### Delivery State

TensorDSLab remains pre-deployment. Maintenance 15 authorizes local
Implementation, fixed-candidate Validation, independent Review, and a clean
local-main fast-forward only. It does not authorize CUDA, push, publication,
release, deployment, calibration, application-repository work, or broad
compatibility claims.

Historical decisions remain available in immutable stage and maintenance work
orders under `docs/implementation/`.
