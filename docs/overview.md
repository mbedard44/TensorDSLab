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

Maintenance 12 is **Merged / Closed** through exact Review-cleared and
fast-forwarded Candidate 2
`ba4f3408bf6b5cbd34d6736741026297b3e05c19`, tree
`4e3b34be19841de016f7c99a668999d2d8dadcc9`. It adopts exact published
TensorCore `0.21.0`, replaces the scalar effect-specific Charge/Pulse Config
hierarchy with literal physical quantity kernels and compiled Runtime state,
and rebaselines timing, branching, afterpulse, pulse rendering, and private RNG
addresses exactly as recorded in the
[work order](implementation/maintenance_12_tensorcore_0_21_kernel_geometry_quantity_refactor.md).
Immutable Candidate 1 remains
`da33a7e7f12e07341c06d66a96cbfdfccae4ebd1`; Candidate 2 changes only the
seven public leaf slot declarations and their immutability proof. The accepted
evidence is CPU-only; no current accelerator claim follows.

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
without a new authority after Maintenance 6. Later GPU
characterization and integration remain Design targets.

[Maintenance 4](implementation/maintenance_4_runtime_action_ownership.md) is
**Merged / Closed** through exact Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda`. It implements a
behavior-preserving internal action split—complete request preparation,
product Runtime construction, tensor production, and immediate product
validation—without changing `simulate_readout(...)`, scientific behavior,
RNG addressing, TensorCore `0.9.0`, or the supported device boundary. Exact
local source/archive evidence and separate fresh Validation and Review
full-A100 source/archive allocations cleared the final bytes.

[Maintenance 5](implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md)
is **Merged / Closed** through exact Review-cleared supplemental candidate
`81ad2f52fe4a1966e5b3a0ceb5063138e42e731f` and Design closeout
`021694b9479d02546405f6a815aedf21c9c831a4`. It atomically adopts published
TensorCore `0.13.0` exact
commit `202d8b1bc6259b8453d3d377570417f2480d782b`, compact
count/label/regular axis roots, and source-derived sampling. It removes
`SamplingConfig` and `ReadoutConfig.sampling` without changing readout science,
RNG addressing, product execution, or `simulate_readout(...)`.

[Maintenance 6](implementation/maintenance_6_pint_physical_configuration_boundary.md)
is **Merged / Closed** through exact Review-cleared target
`0257fb477ee04556ebbe26351123ae610b5d7925`, tree
`b4f5703ca5b756dc27d876c1dd17ee56cb43b4e8`. Its immutable production
Candidate 1 is `240e1492c466097b3059dfe9911ab338a4dd38a1`. It selects Pint `0.25.3`,
migrates public physical Config fields to copied canonical scalar quantities,
and extracts plain execution values once during preparation. It deliberately
keeps Runtime records, producers, validators, tensors, RNG mechanics, and
scientific equations unit-free. Its bounded action cleanup removes
annotation-only Config membership checks and uses TensorCore `0.13.0` where
the generic contract matches. Pint canonicalization, genuine Config
relationships, and package-owned scientific, axes-identity, storage, and
generated-result checks remain.

The user-selected evidence amendment closed Maintenance 6 through complete
local fixed-commit Validation and independent Review, with `13` conditional
CUDA skips disclosed and no accelerator claim. Local `main` remains unpushed.
Fresh cluster matrices are deferred until a separate TensorDSLab work order
closes adoption of published
TensorCore `0.15.0` exact commit
`0f974e9e7f52125bbe829e124beb24e69de811d3`; they are not a Maintenance 6
closure condition.

[Maintenance 7](implementation/maintenance_7_tensorcore_0_15_adoption.md) is
**Merged / Closed** through exact Review-cleared and fast-forwarded target
`205182f0c7a4359cece79211ad22b47b522c34e3`, tree
`4c9f0ed2700b5683debb6e658ff2ec832e3d6acf`. Immutable production Candidate 1
remains `68c2f62c2ce354dd6c92fde28b020c0ce71881d6`, tree
`a33750e4b4c094178ba4e65ffaaed530beb377d6`. The maintenance adopts published
TensorCore `0.15.0`, replaces `logical_positions(...)` with `RngPositions`,
migrates matching generic validation helpers to TensorCore, and centralizes the
unchanged readout RNG namespace and ten role keys in one private runtime
table. Public Configs expose no key fields and request preparation performs no
caller-key collision admission. It also makes pulse Config values positive
amplitude magnitudes and applies fixed DS20k negative polarity once in
preparation, with exact calibrated result continuity. Public field names, Pint
ownership, role streams, and scientific addresses remain unchanged. Complete
local Review evidence passed with `13` conditional CUDA skips; no accelerator
claim follows and local `main` remains unpushed.

[Maintenance 8](implementation/maintenance_8_python314_tensorcore_0_16_modernization.md)
is **Merged / Closed** through exact Review-cleared target
`e5cc70adddaed357298e3e3bc4d95df78d3a55b7`. The fixed work order selects
exact published TensorCore `0.16.0` containing commit
`e05324699892a8bcea024375720bfae1ed9569cc`, Python `3.14.6`, and
PyTorch `2.13.0`; migrates requirement imports to domain-owned validation
modules; and performs bounded syntax, typing, metadata, test, and docstring
modernization. It preserves all scientific, Pint, RNG, product, and public
TensorDSLab contracts. CUDA evidence and the first push remain separate.

[Maintenance 9](implementation/maintenance_9_ds20k_veto_profile_and_public_readout_demos.md)
is **Merged / Closed** through exact Review-cleared target
`2a04942229ab06d2cfc17ab7a5fd09afaf4e3c58`. It adds the provisional
`ds20k_veto()` factory, an executable CPU script and notebook, waveform-chain
plots, and a non-editable `tensor_dslab` Conda setup path while preserving the
existing simulation and scientific contracts. Full source/archive evidence
passed at `229/216/13`; the unavailable-CUDA skips authorize no accelerator
claim or push.

[Maintenance 11](implementation/maintenance_11_tensorcore_0_19_addressed_distributions.md)
is **Merged / Closed** through exact Review-cleared and fast-forwarded
Candidate 2 `a527042701ac56f368f26248381244fdfcfb7fd3`, tree
`5c76122b25d17b9fe0b796618613d7bff0b102c1`. It introduces one private
address owner, migrates stochastic effects to public Distribution and
ProbabilityKernel objects, retains the afterpulse stream with separate
occurrence/delay quanta, and rebaselines crosstalk through exact collapsed-rate
Poisson superposition. The CPU-only `demos/random.ipynb` makes addresses, raw
words, repeatability, chunk invariance, and global-RNG isolation inspectable
without creating a public TensorDSLab RNG surface. No integrated CUDA,
performance, compatibility, publication, release, deployment, calibration, or
production-readiness claim follows.

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

Exact classes carry semantic meaning. Under the implemented Maintenance 5
baseline:

- `ExampleAxis` directly subclasses `CountAxis` and represents nonempty
  zero-based local ordinals;
- `ChannelAxis` directly subclasses `LabelAxis` and retains nonempty unique
  string detector labels;
- `SampleAxis` directly subclasses `RegularAxis` and stores nonnegative
  integer-picosecond start, positive step, count at least two, and a bounded
  exclusive stop;
- `Photoelectrons`, `Charge`, `PureWaveform`, `NoiseWaveform`,
  `AnalogWaveform`, and `DigitizedWaveform` directly subclass `TensorField`;
- `ReadoutCollection` directly subclasses `TensorCollection`; and
- exact frozen config classes express scientific choices.

There are no loose axis or field constants, product-name registries,
`TensorLayout`, `SampleGrid`, or collection sidecars in the rebuild. Exact
field and axis classes carry in-process identity.

The source `SampleAxis(start, step, count)` is the sole sampling authority.
Count and regular coordinates are nonmaterializing `range` values; channel
coordinates remain explicit strings. Private `prepare_sampling(photoelectrons)`
derives count, integer-picosecond period, and tensor dimension exactly once.
The complete readout boundary requires `start == 0`; a semantic `SampleAxis`
may otherwise describe a valid nonzero-start subgrid. Kernels use indices and
plain prepared integers, never semantic coordinates, on the hot path.

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

Photoelectrons axes/device/shape -> source-derived SamplingRuntime
Photoelectrons + SamplingRuntime -> NoiseWaveform
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

The selected internal lifecycle is:

```text
Config + complete-request preflight
  -> ProductRuntime
  -> produce_<product>(...)
  -> Product
  -> validate_<product>(...)
  -> next dependent product
```

Every Runtime required by the transitive closure is prepared before the first
RNG request, product-production call, or semantic-output write. After that
boundary, each generated product is produced and validated immediately before
any descendant may consume it. One private `SamplingRuntime` binds the source
sample count, period, and dimension once and is shared by temporal product
Runtimes. Runtime records are final, frozen, and slotted and contain prepared
execution operands, never configs
or prerequisite semantic products.

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
  readout/
    config.py
    collection.py
    simulation.py             # sole public orchestration function
    runtime/
      keys.py                 # fixed non-exported stochastic role addresses
      prepare.py              # ReadoutRuntime and complete preflight
      requirements.py         # sole shared readout-domain relationship
      sampling.py             # SamplingRuntime and one-time axis binding
    photoelectrons/
      field.py
      runtime/validate.py
    charge/
      config.py
      field.py
      runtime/
        prepare.py
        produce.py
        validate.py
        effects/
          counts.py
          delays.py
          dark_counts.py
          timing_jitter.py
          correlated_avalanches.py
          smearing.py
    pure_waveform/{config.py,field.py,runtime/{prepare.py,produce.py,validate.py}}
    noise_waveform/{config.py,field.py,runtime/{prepare.py,produce.py,validate.py}}
    analog_waveform/{config.py,field.py,runtime/{prepare.py,produce.py,validate.py}}
    digitized_waveform/{config.py,field.py,runtime/{prepare.py,produce.py,validate.py}}
```

Every generated product package owns its public config and field plus
non-exported `prepare_<product>`, `produce_<product>`, and
`validate_<product>` actions. `Photoelectrons` remains the producer-less truth
input and owns only its field and ingress validator. `readout.simulation`
keeps the one public orchestration function; `readout.runtime.prepare`
composes the complete private Runtime closure. Generic RNG and distribution
mechanics entered through the selected TensorCore `0.9.0` dependency and
Maintenance 7 adopts exact `0.15.0` plus `RngPositions`. One non-exported
runtime table fixes the unchanged namespace and ten role keys; public Configs
contain no role-key fields or override surface.

Privacy is export-driven. Runtime paths remain importable Python
implementation details, but public facades do not export them and they carry
no compatibility promise. Runtime and effect `__init__.py` files are empty;
internal callers import exact defining modules. No behavior module is created
as a placeholder, no generic Runtime or Action framework is introduced, and
physical file visibility does not expand the collaborator-facing API.

Maintenance 2 realized the product-centered public ownership and Stage 7
completed `readout/simulation.py` and the public orchestration surface.
Maintenance 4 is the merged internal ownership refactor that realizes the
Runtime/action tree above.
TensorCore RNG/distribution/same-dtype acceptance, publication, and the exact
TensorDSLab dependency pin are complete at `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The implementation preserves
default-key continuity and removes `types.py`, `_RngStream`, and
`readout/_random.py` without shims; their Stage 5/6 bytes remain closed
historical evidence. Maintenance 2 and Stage 7 are Merged / Closed.
Maintenance 5 replaces that installed pin with exact published TensorCore
`0.13.0` while preserving the same public RNG/distribution behavior; the
`0.9.0` statements remain historical ownership evidence rather than the
post-Maintenance-5 dependency target.

Historically, Stage 6 behavior-neutrally renamed the four Stage 4/5 waveform
families from `_product.py` / `_product_*` to `_produce.py` / `_produce_*`.
Maintenance 4 deliberately supersedes those former private bundles with
product-owned `runtime/prepare.py`, `runtime/produce.py`, and
`runtime/validate.py`, again without compatibility shims. Closed work orders
retain their original private paths as historical evidence.

## Documentation Map

- [Public API](api.md): supported facade boundaries, precise-module public
  profiles, and collaborator-facing usage.
- [Design](design.md): architecture thesis, ownership boundaries, and
  non-goals.
- [Decisions](decisions.md): accepted, historical, and open decisions.
- [Rebuild Architecture](architecture/rebuild.md): complete selected rebuild
  contract, config sketches, scientific algorithms, RNG design, package
  ownership, closed decisions, and remaining gates.
- [Correlated-Avalanche Physics](physics/correlated_avalanches.md):
  newcomer-facing explanation of the physical assumptions, aggregate
  statistics, fixed-generation algorithm, and visual tensor example.
- [TensorCore Integration](architecture/tensors.md): historical semantic-root
  adoption, the accepted published TensorCore `0.13.0` compact-axis and
  golden-path boundary, and TensorDSLab result contracts.
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
- [Maintenance 4 Work Order](implementation/maintenance_4_runtime_action_ownership.md):
  Merged / Closed internal Runtime/action ownership refactor through exact
  Review-cleared supplemental candidate
  `b3c7c907004741ba67b8b92a54bbdc8c85216dda`; public API, science,
  dependency, and RNG behavior remain unchanged.
- [Maintenance 5 Work Order](implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md):
  Merged / Closed TensorCore `0.13.0`, compact-axis, and source-derived
  sampling migration.
- [Maintenance 6 Work Order](implementation/maintenance_6_pint_physical_configuration_boundary.md):
  Merged / Closed Pint physical-configuration boundary and TensorCore-aware
  preparation cleanup through exact Review-cleared target
  `0257fb477ee04556ebbe26351123ae610b5d7925`.
- [Maintenance 7 Work Order](implementation/maintenance_7_tensorcore_0_15_adoption.md):
  Merged / Closed exact TensorCore `0.15.0`, validation-ownership,
  RngPositions, and fixed readout role-key adoption through target
  `205182f0c7a4359cece79211ad22b47b522c34e3`.
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
