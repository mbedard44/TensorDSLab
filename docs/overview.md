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
TensorDSLab semantic leaves. Private scientific producers are implemented
through Stage 6, and Maintenance 2 is Merged / Closed through exact candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. Public readout orchestration is
Merged / Closed through exact Review-cleared Stage 7 candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177`. Maintenance 3 is Merged / Closed
through exact Review-cleared candidate
`dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0` and its five-document Design
closeout. It corrects only the numerical-stack applicability of Maintenance
2's completed stochastic literals and changes no production, dependency,
RNG, or scientific contract. The first Stage 8 executable correctly stopped
before accepted measurement on that test-contract conflict; it cannot resume
without a new authority from the closed Maintenance 3 baseline. Later GPU
characterization and integration remain Design targets.

[Stage 4](implementation/stage_4_deterministic_waveform_products.md) is Merged /
Closed through exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da` and Review's clean fast-forward
closeout `b3ebfcd9473537dd385195afea374bd2f426c6c0`. It implements the private
pure, analog, and digitized waveform producers under a functionality-first
contract.

[Stage 5](implementation/stage_5_readout_rng_and_stochastic_noise.md) is
Merged / Closed through exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd` and Review closeout
`c6a506d3658b24197806b9e230480211a254a35a`. It implements the private
positional Threefry reference and complete exact-zero, IID-white, and
caller-supplied PSD noise producer. Fixed-commit Validation, independent
Review, and Design's post-merge audit found no unresolved issue. CUDA was
unavailable, so the evidence is eager CPU-only; measured GPU fusion remains a
later optimization stage.

[Stage 6](implementation/stage_6_charge_simulation.md) is Merged / Closed
through exact implementation candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58` and Review closeout
`ea979862b05f4ef543f6971c86641df317232479`. It implements the private
aggregate samplers, dark counts, analytic timing jitter, fixed-generation
DiCT/DeCT/AP cascade, S1/S2 ledgers, right-overflow diagnostics, charge
smearing, and `_produce_charge(...)`. It also retired `NormalDelayConfig` and
behavior-neutrally renamed the four waveform producer families. Fixed-commit
Validation, independent Review, and Design's post-merge audit found no
unresolved issue. CUDA was unavailable, so the evidence is eager CPU-only.
Stage 7 is Merged / Closed through exact Review-cleared candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177`. It implements public
`simulate_readout(...)`, complete closure preparation, execute-once product
planning, exact requested retention, and generated-product postconditions.
Its 12 conditional CUDA tests skipped, so it adds no GPU execution or
performance claim.

Stage 7 uses the generic TensorCore counter/distribution surface installed by
Maintenance 2—`uniform`, `gaussian`, `poisson`, and `binomial`—plus
`require_same_dtype`. TensorCore fulfilled the historical
[consumer proposal](implementation/proposed_tensorcore_counter_rng_and_distributions.md)
in published version `0.9.0` at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. TensorDSLab's
[Maintenance 2](implementation/maintenance_2_rng_and_product_module_ownership_migration.md)
implementation pins that exact dependency. Its stochastic producers require
`CounterRng`; exact stochastic leaf configs own default `RngKey` role
identities. Maintenance 2 is Merged / Closed at the exact candidate and Design
closeout above. Its eager-CPU evidence ran 157 tests: 148 passed and 9
conditional CUDA tests skipped; Pyright reported no diagnostics against either
exact dependency form.

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

The implemented public call is intentionally small:

```python
readout = simulate_readout(
    photoelectrons,
    products=[AnalogWaveform, DigitizedWaveform],
    config=config,
    rng=Threefry4x32(seed=1234),
)

analog = readout.field(AnalogWaveform)
```

`simulate_readout(...)` consumes a nonempty unordered product request,
computes every transitive prerequisite once, and returns exactly the requested
products in an immutable completed `ReadoutCollection`. Unrequested
prerequisites remain private temporaries. `Photoelectrons` remains immutable
truth: dark roots and timing redistribution occur privately inside charge
production. Timing jitter analytically prepares its binned Gaussian transition
law and samples aggregate counts rather than per-PE normal values. The fixed-
`K` model in `architecture/rebuild.md` is the sole active correlated-avalanche
baseline.

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
    config.py                 # implemented Maintenance 2 ownership
    collection.py             # implemented Maintenance 2 ownership
    simulation.py             # implemented Stage 7 public orchestration
    _requirements.py
    photoelectrons/field.py
    charge/
      config.py
      field.py
      _produce.py
      effects/
        _counts.py
        _delays.py
        _dark_counts.py
        _timing_jitter.py
        _correlated_avalanches.py
        _smearing.py
    pure_waveform/{config.py,field.py,_produce.py}
    noise_waveform/{config.py,field.py,_produce.py}
    analog_waveform/{config.py,field.py,_produce.py}
    digitized_waveform/{config.py,field.py,_produce.py}
```

Every generated product package owns its field, configs, validation, and
implemented private `_produce_*` producer. `Photoelectrons` remains the
producer-less truth input. Closed Stage 7 `readout/simulation.py` owns the one
public orchestration function. `_requirements.py` and
Charge effect modules are private. Generic RNG and distribution mechanics
belong to the selected TensorCore `0.9.0` Maintenance 2 dependency;
config-owned `RngKey` values select stochastic roles.
No behavior module is created as an empty placeholder; the complete ownership
and import rules are in `architecture/rebuild.md`.

The product/module ownership portion of the tree above is realized by
Maintenance 2; Stage 7 completes `readout/simulation.py` and the public
orchestration surface.
TensorCore RNG/distribution/same-dtype acceptance, publication, and the exact
TensorDSLab dependency pin are complete at `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The implementation preserves
default-key continuity and removes `types.py`, `_RngStream`, and
`readout/_random.py` without shims; their Stage 5/6 bytes remain closed
historical evidence. Maintenance 2 and Stage 7 are Merged / Closed.

Stage 6 behavior-neutrally renamed all four transitional Stage 4/5 waveform
modules, callables, imports, and tests from `_product.py` / `_product_*` to
`_produce.py` / `_produce_*`. That producer naming convention is implemented;
no retired producer name or compatibility shim remains.

## Documentation Map

- [Design](design.md): architecture thesis, ownership boundaries, and
  non-goals.
- [Decisions](decisions.md): accepted, historical, and open decisions.
- [Rebuild Architecture](architecture/rebuild.md): complete selected rebuild
  contract, config sketches, scientific algorithms, RNG design, package
  ownership, closed decisions, and remaining gates.
- [Correlated-Avalanche Physics](physics/correlated_avalanches.md):
  newcomer-facing explanation of the physical assumptions, aggregate
  statistics, fixed-generation algorithm, and visual tensor example.
- [TensorCore Integration](architecture/tensors.md): the semantic-root
  extension introduced at TensorCore `0.7`, the current `0.9.0` RNG boundary,
  and TensorDSLab result contracts.
- [Post-Binned Readout](architecture/readout.md): readout product semantics,
  product graph, config ownership, and simulation boundary.
- [IV-DSLab Parity](parity.md): donor evidence, comparison classes,
  assumptions, fixtures, and intentional divergences.
- [Validation](validation.md): validation philosophy and stage expectations.
- [Implementation Stages](implementation/index.md): staged work orders and
  dispatch state.
- [Historical TensorCore RNG Consumer Proposal](implementation/proposed_tensorcore_counter_rng_and_distributions.md):
  TensorDSLab consumer requirements fulfilled by TensorCore Stage 15 at exact
  published `0.9.0` commit `4708bf2...`; never TensorCore authority.
- [Maintenance 2 Work Order](implementation/maintenance_2_rng_and_product_module_ownership_migration.md):
  Merged / Closed TensorDSLab ownership migration against the selected exact
  TensorCore dependency.
- [Maintenance 3 Work Order](implementation/maintenance_3_environment_qualified_stochastic_continuity.md):
  Merged / Closed environment qualification for completed stochastic
  continuity fixtures, with no production or scientific change.
- [Stage 7 Work Order](implementation/stage_7_public_readout_orchestration.md):
  Merged / Closed public request planning, whole-closure
  preparation, execute-once orchestration, and exact-retention contract.
- [Stage 3 Work Order](implementation/stage_3_tensorcore_0_7_product_foundation.md):
  Merged / Closed TensorCore `0.7` semantic product/config foundation and
  clean replacement scope.
- [Stage 4 Work Order](implementation/stage_4_deterministic_waveform_products.md):
  Merged / Closed functionality-first pure, analog, and digitized waveform
  producer slice.
- [Stage 5 Work Order](implementation/stage_5_readout_rng_and_stochastic_noise.md):
  Merged / Closed private positional RNG and complete zero/white/PSD noise
  producer slice.
- [Stage 6 Work Order](implementation/stage_6_charge_simulation.md):
  Merged / Closed private Charge producer, aggregate sampler, timing,
  fixed-generation cascade, ledger, and smearing slice.
- [Package Governance](governance/index.md): adoption decision and declaration,
  TensorDSLab overlay, semantic rule map, state boundaries, and closeout.
- [Stage 2 Work Order](implementation/stage_2_package_and_readout_collection_foundation.md):
  historical first package foundation, Merged / Closed.
- [Maintenance 1 Work Order](implementation/maintenance_1_readout_surface_ownership.md):
  historical readout name/module correction, Merged / Closed.

Completed work orders and governance records remain historical records. If an
implementation uncovers a contradiction in the selected architecture, return
it to Design before changing the contract.
