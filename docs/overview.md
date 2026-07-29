# Overview

TensorDSLab is a pre-deployment tensor-native detector Product library:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is data flow, not an import graph. Native G4DS parsing and TensorG4DS
clustering remain upstream. TensorML training remains downstream. Applications
own workflow composition, IO, and persistence.

## Current Candidate

Maintenance 16 retains TensorCore `0.22.0` and the direct reusable Products
introduced by Maintenance 15:

```text
Coordinates -> semantic Axis -> exact Spec -> Tensor Product/Kernel
```

The public package provides four shared semantic axes, six Product/Spec pairs,
fifteen coefficient/Spec pairs, five typed Kernel collections, five Config
punchcards, one scalar `quantity` helper, and one package Pint registry.
Its public facades are unchanged.

Reusable package validation is organized in one private, export-empty
`common/requirements/` package with exact owners for axes, capacity,
collections, Configs, Fields, Kernels, tensors, and Units. Semantic class hooks
call these requirements directly. Specs own metadata and operation geometry;
Fields and Kernels own represented-value laws.

Generated Products expose `prepare`, `produce`, `validate`, and `create`.
Preparation returns a fresh same-type Config with immutable source-Spec,
alignment, dtype, temporal, and Kernel-dimension facts. No Runtime object is
created. Production consumes only prepared facts and exact supplied sources.

## Package Shape

```text
tensor_dslab/
  common/
    requirements/
  photoelectrons/
  charge/
  pure_waveform/
  noise_waveform/
  analog_waveform/
  digitized_waveform/
```

Each generated Product package owns `config.py`, `field.py`, singular
`kernel.py` where applicable, and a non-exported `runtime/` action package.
The five typed `*Kernels` collections live in their Product `kernel.py`;
Config modules own Config records only. The generic `readout/` namespace is
absent. Runtime modules are private by export and contain actions, not Runtime
records.

## Documentation Map

- [Public API](api.md)
- [Product architecture](architecture/readout.md)
- [TensorCore integration](architecture/tensors.md)
- [Scientific parity](parity.md)
- [Validation](validation.md)
- [Maintenance 15 architecture](implementation/maintenance_15_spec_composed_products_and_application_boundary.md)
- [Maintenance 15 work order](implementation/maintenance_15_execution_work_order.md)
- [Maintenance 16 requirements ownership](implementation/maintenance_16_declarative_requirements_and_kernel_ownership.md)

Earlier work orders remain immutable historical evidence. When an older living
claim conflicts with the Maintenance 15 architecture or Maintenance 16
requirements ownership, those exact current records govern.
