# Overview

This is the quick architecture map for TensorDSLab Design, Implementation,
Validation, and Review.

## Project Identity

TensorDSLab is a clean-slate, tensor-native detector data-lab package. Its
intended data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is not an import graph or a claim that every boundary exists. Native G4DS
parsing and low-level analysis such as deposit clustering remain upstream.
TensorCore is the shared generic tensor substrate. TensorDSLab owns its
post-TensorG4DS readout and future reconstruction semantics. TensorML owns
model, training, evaluation, and model-facing adaptation. Durable IO, caches,
DAG orchestration, and the exact cross-package adapters remain deferred.

## Current State

Stage 2 and Maintenance 1 are Merged / Closed historical TensorCore `0.6`
baselines. Stage 3 is Merged / Closed through exact implementation candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde` and Review's clean fast-forward
closeout `97e17c3177ac217aeb42a077db78f4bd223d51fa`; Design's accepted final
Stage 3 closeout is clean `main` at
`5ff13eb3c0735abfda454a334be59faac35259c2`. It deliberately replaced the
pre-deployment `0.6` representation without a compatibility layer.

[Rebuild Architecture](architecture/rebuild.md) remains the accepted complete
architecture. Stage 3 selected exact TensorCore `0.7.0` commit
`b454d738f6385ce6489d85492a618a3dab139bb6` and implements its ordinary
`TensorAxis`, `TensorField`, and `TensorCollection` ABC roots with direct final
TensorDSLab semantic leaves. Further scientific producers and public readout
orchestration remain Design targets only until their own focused work orders
are written, dispatched, validated, reviewed, and merged.

[Stage 4](implementation/stage_4_deterministic_waveform_products.md) is Merged /
Closed through exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da` and Review's clean fast-forward
closeout `b3ebfcd9473537dd385195afea374bd2f426c6c0`. It implements the private
pure, analog, and digitized waveform producers under a functionality-first
contract. Fixed-commit Validation, independent Review, and Design's post-merge
audit found no unresolved issue. The complete noise producer remains a
focused [Stage 5](implementation/stage_5_readout_rng_and_stochastic_noise.md)
slice whose work order is Design-complete / Undispatched. Measured GPU fusion
remains a later optimization stage.

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`, bound to
accepted candidate `d634401a853915edeb4f83df4a4943b3553deced`. Conformance is
`Not evaluated`, Coordination is `Deferred`, and Profile B is `Disabled`.
TensorDSLab remains in active development and pre-deployment; it makes no broad
deployability, backward-compatibility, release, or cross-package compatibility
claim.

## Selected Rebuild Model

The rebuild starts from an already-produced dense `Photoelectrons` truth
field. Native input loading, detector-window construction, and PE binning stay
in the future TensorG4DS bridge. `Photoelectrons` therefore has no config and
no local producer.

Exact classes carry semantic meaning:

- `ExampleAxis`, `ChannelAxis`, and `SampleAxis` directly subclass
  `TensorAxis`;
- `Photoelectrons`, `Charge`, `PureWaveform`, `NoiseWaveform`,
  `AnalogWaveform`, and `DigitizedWaveform` directly subclass `TensorField`;
- `ReadoutCollection` directly subclasses `TensorCollection`; and
- exact frozen config classes express scientific choices.

There are no loose axis or field constants, product-name registries,
`TensorLayout`, `SampleGrid`, or collection sidecars in the rebuild. Exact
field and axis classes carry in-process identity.

`SamplingConfig` owns positive integer `sample_period_ps` and `sample_count`.
It realizes a regular `SampleAxis` whose canonical string coordinates are the
left edges `"0ps"`, `"2000ps"`, and so on. Kernels use numeric sampling values
and integer indices, not timestamp strings, on the hot path.

Every readout field contains exactly one `ExampleAxis`, one `ChannelAxis`, and
one `SampleAxis`, in any tensor-dimension order. Fields use `torch.strided`
storage and every field in one returned collection shares the same exact axis
objects and device. `Photoelectrons` is `torch.int64`; floating products use
one common `torch.float32` or `torch.float64` dtype; and
`DigitizedWaveform` is `torch.int32`.

The selected product graph is:

```text
Photoelectrons
  -> Charge
       -> PureWaveform

Photoelectrons axes/device/shape + SamplingConfig -> NoiseWaveform
PureWaveform + NoiseWaveform
  -> AnalogWaveform
       -> DigitizedWaveform
```

The future public call is intentionally small:

```python
readout = simulate_readout(
    photoelectrons,
    products=[AnalogWaveform, DigitizedWaveform],
    config=config,
    seed=1234,
)

analog = readout.field(AnalogWaveform)
```

`simulate_readout(...)` consumes a nonempty unordered product request,
computes every transitive prerequisite once, and returns exactly the requested
products in an immutable completed `ReadoutCollection`. Unrequested
prerequisites remain private temporaries. `Photoelectrons` remains immutable
truth: dark roots and timing redistribution occur privately inside charge
production. The fixed-`K` model in `architecture/rebuild.md` is the sole active
correlated-avalanche baseline.

The rebuild begins with a functional simulation path. It does not carry the
old public `out`, preconstructed destination, `ReadoutWorkspace`, lease,
allocation-free, partial-snapshot, or descendant-invalidation architecture
forward. Later execution optimization follows measurement and TensorCore's
operation-owned freshness/sharing taxonomy. A generated tensor is not exposed
as a valid field until its writes have been enqueued, and supported operations
never write to a tensor after exposing it through a field.

## Intended Product Flow

```text
G4DS native products
  -> TensorG4DS typed tensor-native products
  -> deferred TensorG4DS-to-TensorDSLab bridge
       -> explicit provenance and coordinate mapping
       -> detector-window and SampleAxis construction
       -> photon-origin PE binning
  -> dense TensorDSLab Photoelectrons
  -> simulate_readout(..., products=...)
  -> request-selected ReadoutCollection
       -> TensorDSLab readout and future reconstruction product views
  -> deferred explicit TensorML product-selection/model boundary
```

This is a product-flow and ownership rule, not a campaign schedule. The future
bridge constructs new TensorDSLab semantics; it does not relabel an upstream
field or infer durable identity from transient tensor indices. The production
cross-package target keeps payloads on one explicit GPU and forbids silent host
staging, NumPy conversion, or serialization as a package handoff, but that
target is not yet an implemented compatibility claim.

## Selected Package Shape

```text
tensor_dslab/
  common/
    axes.py
    sampling.py
  readout/
    types.py                  # ReadoutConfig and ReadoutCollection only
    simulation.py             # later public orchestration
    _requirements.py
    _random.py                # Stage 5 private RNG behavior when dispatched
    photoelectrons/types.py
    charge/{types.py,_product.py}
    pure_waveform/{types.py,_product.py}
    noise_waveform/{types.py,_product.py}
    analog_waveform/{types.py,_product.py}
    digitized_waveform/{types.py,_product.py}
```

Each product package owns its field, configs, validation, and eventual private
`_product_*` producer. `readout/types.py` owns only the two cross-product
composition types. `readout/simulation.py` owns the one public orchestration
function. `_requirements.py` and `_random.py` are private. No behavior module
is created as an empty placeholder; the complete ownership and import rules
are in `architecture/rebuild.md`.

## Documentation Map

- [Design](design.md): architecture thesis, ownership boundaries, and
  non-goals.
- [Decisions](decisions.md): accepted, historical, and open decisions.
- [Rebuild Architecture](architecture/rebuild.md): complete selected rebuild
  contract, config sketches, scientific algorithms, RNG design, package
  ownership, closed decisions, and remaining gates.
- [TensorCore Integration](architecture/tensors.md): the accepted TensorCore
  `0.7` extension and result contracts for TensorDSLab.
- [Post-Binned Readout](architecture/readout.md): readout product semantics,
  product graph, config ownership, and simulation boundary.
- [IV-DSLab Parity](parity.md): donor evidence, comparison classes,
  assumptions, fixtures, and intentional divergences.
- [Validation](validation.md): validation philosophy and stage expectations.
- [Implementation Stages](implementation/index.md): staged work orders and
  dispatch state.
- [Stage 3 Work Order](implementation/stage_3_tensorcore_0_7_product_foundation.md):
  Merged / Closed TensorCore `0.7` semantic product/config foundation and
  clean replacement scope.
- [Stage 4 Work Order](implementation/stage_4_deterministic_waveform_products.md):
  Merged / Closed functionality-first pure, analog, and digitized waveform
  producer slice.
- [Stage 5 Work Order](implementation/stage_5_readout_rng_and_stochastic_noise.md):
  Design-complete / Undispatched private positional RNG and complete
  zero/white/PSD noise producer slice.
- [Package Governance](governance/index.md): adoption decision and declaration,
  TensorDSLab overlay, semantic rule map, state boundaries, and closeout.
- [Stage 2 Work Order](implementation/stage_2_package_and_readout_collection_foundation.md):
  historical first package foundation, Merged / Closed.
- [Maintenance 1 Work Order](implementation/maintenance_1_readout_surface_ownership.md):
  historical readout name/module correction, Merged / Closed.

Completed work orders and governance records remain historical records. If an
implementation uncovers a contradiction in the selected architecture, return
it to Design before changing the contract.
