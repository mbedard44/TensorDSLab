# Overview

TensorDSLab is a pre-deployment tensor-native detector Product library:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is data flow, not an import graph. Native G4DS parsing and TensorG4DS
clustering remain upstream. TensorML training remains downstream. Applications
own workflow composition, IO, and persistence.

## Current Candidate

Maintenance 15 targets TensorCore `0.22.0` and replaces the embedded readout
workflow with direct reusable Products:

```text
Coordinates -> semantic Axis -> exact Spec -> Tensor Product/Kernel
```

The public package provides four shared semantic axes, six Product/Spec pairs,
fifteen coefficient/Spec pairs, five typed Kernel collections, five Config
punchcards, one scalar `quantity` helper, and one package Pint registry.

Generated Products expose `prepare`, `produce`, `validate`, and `create`.
Preparation returns a fresh same-type Config with immutable source-Spec,
alignment, dtype, temporal, and Kernel-dimension facts. No Runtime object is
created. Production consumes only prepared facts and exact supplied sources.

## Package Shape

```text
tensor_dslab/
  common/
  photoelectrons/
  charge/
  pure_waveform/
  noise_waveform/
  analog_waveform/
  digitized_waveform/
```

Each generated Product package owns `config.py`, `field.py`, `kernel.py` where
applicable, and a non-exported `runtime/` action package. The generic
`readout/` namespace is absent. Runtime modules are private by export and
contain actions, not Runtime records.

## Documentation Map

- [Public API](api.md)
- [Product architecture](architecture/readout.md)
- [TensorCore integration](architecture/tensors.md)
- [Scientific parity](parity.md)
- [Validation](validation.md)
- [Maintenance 15 architecture](implementation/maintenance_15_spec_composed_products_and_application_boundary.md)
- [Maintenance 15 work order](implementation/maintenance_15_execution_work_order.md)

Earlier work orders remain immutable historical evidence. When an older living
claim conflicts with Maintenance 15, the exact current architecture and work
order govern.
