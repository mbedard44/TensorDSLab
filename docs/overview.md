# Overview

TensorDSLab is a pre-deployment tensor-native detector Product library:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is data flow, not an import graph. Native G4DS parsing and TensorG4DS
clustering remain upstream. TensorML training remains downstream. Applications
own workflow composition, IO, and persistence.

## Current Package

Maintenance 22 changes only the committed presentation state of the newcomer
notebook. The production package remains the Maintenance 18 boundary, retains
TensorCore `0.22.0`, and provides the direct reusable Products introduced by
Maintenance 15:

```text
Coordinates -> semantic Axis -> exact Spec -> Tensor Product/Kernel
```

The public package provides four shared semantic axes, seven Product/Spec pairs,
twenty coefficient/Spec pairs, six typed Kernel collections, six Config
punchcards, one scalar `quantity` helper, and one package Pint registry.
Maintenance 18 deliberately expands only the root facade and adds the exact
encoded-waveform subpackage facade; every earlier facade remains unchanged.

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
  encoded_waveform/
```

Each generated Product package owns `config.py`, `field.py`, singular
`kernel.py` where applicable, and a non-exported `runtime/` action package.
The six typed `*Kernels` collections live in their Product `kernel.py`;
Config modules own Config records only. The generic `readout/` namespace is
absent. Runtime modules are private by export and contain actions, not Runtime
records.

## Newcomer Quickstart

The [application-neutral readout notebook](../demos/readout.ipynb) constructs
two independent examples over three sensors directly from seven public Product
calls. Every Product has its own `3 x 2` grid, and the normalized figures are
committed so the story is visible before execution. Rerunning the notebook on
the exact recorded eager-CPU stack deterministically refreshes that snapshot.
It is a teaching composition, not canonical data, durable output, a
package-owned workflow, profile, orchestration surface, calibration, or a
cross-platform image-byte promise. Applications remain responsible for
choosing and retaining Products in their real workflows. The seventh Product
demonstrates the deterministic EncodedWaveform boundary with suppressed
samples displayed as gaps.

## Documentation Map

- [Public API](api.md)
- [Product architecture](architecture/readout.md)
- [TensorCore integration](architecture/tensors.md)
- [Scientific parity](parity.md)
- [Validation](validation.md)
- [Maintenance 15 architecture](implementation/maintenance_15_spec_composed_products_and_application_boundary.md)
- [Maintenance 15 work order](implementation/maintenance_15_execution_work_order.md)
- [Maintenance 16 requirements ownership](implementation/maintenance_16_declarative_requirements_and_kernel_ownership.md)
- [Maintenance 17 readout quickstart](implementation/maintenance_17_application_neutral_readout_quickstart.md)
- [Maintenance 18 EncodedWaveform raw ZLE](implementation/maintenance_18_encoded_waveform_raw_zle.md)

Earlier work orders remain immutable historical evidence. When an older living
claim conflicts with the Maintenance 15 architecture, Maintenance 16
requirements ownership, the narrow Maintenance 17 demonstration boundary, or
the Maintenance 18 encoded-waveform boundary,
those exact current records govern.
