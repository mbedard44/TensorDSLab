# TensorDSLab Rebuild Architecture

Status: accepted Design architecture for the TensorCore-based rebuild.
Stages 3 through 6 are Merged / Closed. Stage 6's exact implementation
candidate is `fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58`, and Review's
evidence-only closeout is
`ea979862b05f4ef543f6971c86641df317232479`. Maintenance 2 is Merged / Closed
through exact implementation candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. It installs the selected
TensorCore `0.9.0` dependency and module/RNG ownership boundary. Stage 7 public
`simulate_readout(...)` orchestration is Merged / Closed through exact
Review-cleared candidate `6dd55024685013fb9412a7247d3ddde7be1a3177` under
the [Stage 7 work order](../implementation/stage_7_public_readout_orchestration.md).
This architecture page does not dispatch later implementation or make a
compatibility claim.

Maintenance 4 Runtime Action Ownership is **Merged / Closed** through exact
Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda` with tree
`2d35a0e926b912f3fa846da97726e4e2490c4cc3`, under the
[Maintenance 4 work order](../implementation/maintenance_4_runtime_action_ownership.md).
It is a behavior-preserving internal refactor that replaces the former
product-local `_produce.py` bundles and `*Plan` records with explicit,
non-exported preparation, production, and validation actions under product
`runtime/` packages. Fixed-commit Validation and independent Review cleared the
same final bytes locally and in separate fresh full-A100 source/archive runs.
It changes no public, scientific, RNG, dependency, supported-device,
performance, or Stage 8 contract. The first Stage 8 attempt remains stopped
evidence; any rerun requires a new Design authority after Maintenance 6.

Maintenance 5 Compact Axes And Sampling is **Merged / Closed** through exact
Review-cleared supplemental candidate
`81ad2f52fe4a1966e5b3a0ceb5063138e42e731f` and Design closeout
`021694b9479d02546405f6a815aedf21c9c831a4` under the focused
[work order](../implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md).
It adopts published TensorCore `0.13.0` exact commit
`202d8b1bc6259b8453d3d377570417f2480d782b` and fixes this accepted
replacement:

```text
ExampleAxis(CountAxis)
  -> nonempty zero-based local ordinal range

ChannelAxis(LabelAxis)
  -> nonempty unique string detector labels

SampleAxis(RegularAxis)
  -> nonnegative integer-picosecond start
  -> positive step
  -> count >= 2
  -> exclusive stop <= 2**63 - 1

source SampleAxis
  -> prepare_sampling(photoelectrons)
  -> SamplingRuntime(sample_count, sample_period_ps, sample_dimension)
```

The complete readout boundary requires `SampleAxis.start == 0`; the semantic
axis itself permits valid nonzero-start regular subgrids. `SamplingConfig`,
`ReadoutConfig.sampling`, `common/sampling.py`, timestamp-string sample
coordinates, and the duplicate config/source agreement check are retired
without shims. `ReadoutConfig()` becomes the truth-only configuration.
`simulate_readout(...)`, every scientific equation, RNG key/address/call,
ProductRuntime other than its unchanged sampling input, product result, and
supported device boundary remain unchanged.

This Maintenance 5 block and its focused work order supersede every later
unqualified `SamplingConfig`, direct string-valued `TensorAxis`, string
`SampleAxis`, or source/config-agreement sketch in this long-form document.
Those sketches remain here only as historical Stage 3 through Maintenance 4
design evidence until a later editorial compaction; they are not
implementation authority for Maintenance 5. Explicitly historical `0.7.0`
and `0.9.0` dependency statements remain exact for their closed stages.
TensorCore `0.13.0` also exposes `Scalar`, table roots, and `TensorArtifact`,
but Maintenance 5 introduces no Pint, table, artifact, persistence, or IO
surface.

Maintenance 6 Pint Physical Configuration Boundary is
**Merged / Closed** through exact Review-cleared target
`0257fb477ee04556ebbe26351123ae610b5d7925` under its focused
[work order](../implementation/maintenance_6_pint_physical_configuration_boundary.md).
It retains the exact Maintenance 5 axis/dependency baseline and selects Pint
`0.25.3` for public physical Config values only. The accepted flow is:

```text
caller Quantity
  -> Config canonical copy + one TensorCore Scalar.require normalization
  -> prepare_<product> extracts one plain canonical magnitude
  -> unit-free Runtime
  -> unit-free produce_<product>
  -> unit-free validate_<product>
```

Public physical field names become unit-neutral; private Runtime facts remain
unit-suffixed. `SampleAxis` stays compact and integer-backed while gaining one
Pint-aware construction convenience and four fresh quantity accessors.
Maintenance 6 also removes annotation-only Config membership checks and
duplicate private admission guards already owned by static typing, Review,
Config construction, or whole-request preparation. Config construction keeps
Pint canonicalization, primitive value domains, and genuine local
relationships. The maintenance does not remove exact model dispatch,
scientific laws, representability, tensor relationships, axes identity,
storage freshness, dtype/device, allocation/address/envelope, or
generated-product postcondition checks.

This Maintenance 6 block and its work order supersede every later unqualified
raw physical Config field, unit-suffixed public physical field name, Config-
bearing numerical helper, annotation-only Config membership check, or
statement that Pint remains merely deferred.
Those sketches remain historical Stage 3 through Maintenance 5 design evidence
until later editorial compaction; they are not Maintenance 6 implementation
authority. The stage changes no scientific equation, RNG address, product
meaning, IO/artifact boundary, or Stage 8 contract.

Maintenance 7 TensorCore 0.15 Adoption is **Merged / Closed** through exact
Review-cleared target `205182f0c7a4359cece79211ad22b47b522c34e3`,
tree `4c9f0ed2700b5683debb6e658ff2ec832e3d6acf`, under
[`maintenance_7_tensorcore_0_15_adoption.md`](../implementation/maintenance_7_tensorcore_0_15_adoption.md).
Immutable production Candidate 1 remains
`68c2f62c2ce354dd6c92fde28b020c0ce71881d6`. The maintenance pins exact
published TensorCore `0.15.0`, cleanly replaces
`logical_positions(...)` with `RngPositions`, moves only matching generic
validation mechanics into TensorCore ownership, and gives the unchanged
`0x54445331` readout namespace one non-exported source. It also ratifies the
DS20k pulse convention that public pulse Configs store positive
peak-voltage-per-photoelectron magnitudes while preparation applies the fixed
negative polarity exactly once. Calibrated rendered results remain exact. This
positive-magnitude block supersedes later unqualified signed-Config sketches
in this long-form document. Maintenance 7 also removes all public per-role
`RngKey` Config fields, fixes the same ten streams in
`readout/runtime/keys.py`, removes closure-wide caller-key collision admission,
and moves the sole shared readout-domain relationship from
`readout/requirements.py` to `readout/runtime/requirements.py`. Every later
unqualified reference in this long-form historical architecture to
config-owned keys, key overrides, `rng_roles`, request-time key uniqueness, or
the former requirements path is superseded by this paragraph. Every role
stream, raw position value/order, word schedule, other scientific limit,
product action, and facade remains unchanged.

Maintenance 8 is **Merged / Closed** through exact Review-cleared target
`e5cc70adddaed357298e3e3bc4d95df78d3a55b7` under its exact
fixed work order. It adopts published TensorCore `0.16.0` containing commit
`e05324699892a8bcea024375720bfae1ed9569cc` and Python `3.14.6` /
PyTorch `2.13.0` through dependency-import, typing-syntax, metadata, test, and
docstring changes only. It does not revise any accepted equation, bound,
sampling law, Pint rule, RNG address, product relationship, or parity
classification in this long-form architecture.

Maintenance 9 is **Merged / Closed** through exact Review-cleared target
`2a04942229ab06d2cfc17ab7a5fd09afaf4e3c58`. The new provisional profile and
public demonstrations instantiate this architecture without revising it. The
demo's source, axes, products, dtype, seed, device, and plots remain explicit
caller choices; its profile values create no run-calibration or new parity
authority.

Maintenance 11 is **Merged / Closed** through exact Review-cleared and
fast-forwarded Candidate 2 `a527042701ac56f368f26248381244fdfcfb7fd3`,
tree `5c76122b25d17b9fe0b796618613d7bff0b102c1`. It supersedes later
unqualified descriptions of raw `RngPositions`, manual offsets, `CounterRng`
law methods, private category draws, crosstalk retained/overflow draws, or
separately returned finite-window tail state. Exact TensorCore `0.19.0` owns
`RngElements`, `RngAddress`, Distribution, TensorKernel, and ProbabilityKernel
mechanics. TensorDSLab owns the fixed namespace and active keys, semantic
lattices and address schemas, scientific probability/rate construction,
displacement and boundary meaning, checked accumulation, ledgers, and
postconditions.

The active mapping is Gaussian for white/PSD noise and charge smearing;
Poisson for dark counts and collapsed direct/delayed-crosstalk destination
means; prepared ProbabilityKernel plus Multinomial for timing jitter and
afterpulse delay; and Binomial for afterpulse occurrence. Crosstalk collapse
is exact under Poisson splitting and superposition; it is not a total-first
Poisson-plus-Multinomial substitute. Afterpulse retains stream
`0x0000_0009`, occurrence quantum `0`, and delay quantum `1`. Obsolete
crosstalk overflow streams and outputs are retired without reservation or a
compatibility layer.

TensorDSLab Design historically selected and implemented the following
Maintenance 2 RNG and module-ownership foundation:
one caller-constructed TensorCore `CounterRng` per simulation invocation,
config-owned `RngKey` values for stochastic roles, public parameterized
Gaussian, Poisson, and binomial distributions on that RNG, product
`config.py` and `field.py` modules, a readout `config.py` and
`collection.py`, and focused Charge effect modules. This is a TensorDSLab
consumer decision. TensorCore first supplied the adopted generic surface in
published version `0.9.0` at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, which TensorDSLab Design selected
for Maintenance 2 after exact consumer probes. The implementation pins that
commit, uses the public RNG/distribution surface, and realizes the accepted
module ownership without compatibility shims. Published TensorCore `0.13.0`
preserves the same public RNG/distribution contract for Maintenance 5. The
Stage 5/6 private implementation remains closed historical evidence. Stage 7
and Maintenance 4 are separate Merged / Closed production slices. Maintenance
7 supersedes only the config-owned role-key and root-requirements ownership
parts of this historical record.

Within this architecture, the fixed-`K` algorithm under
[Fixed-Generation Correlated-Avalanche Baseline](#fixed-generation-correlated-avalanche-baseline)
is the sole active correlated-avalanche implementation baseline. The separate
avalanche-algorithm architecture pages have been removed. A work order may
implement only this baseline unless a new explicit user and TensorDSLab Design
decision first changes this page.

The architecture pass was started from clean TensorDSLab `main` at
`3af8ab4acf834b07e3d027fb530e5f12934999a5`. The TensorCore reference examined
for this design is clean TensorCore `0.7.0` `main` at
`b454d738f6385ce6489d85492a618a3dab139bb6`. That exact commit contains the
operative ordinary-ABC semantic roots and the Stage 13 operation-owned
aliasing/freshness documentation contract. Stage 3 selected that exact package
pin and passed TensorDSLab-owned consumer probes; Stages 4 through 6 retained
and reverified it. Later dependency changes remain explicit implementation-
work-order gates. This exact-baseline evidence is not a broad compatibility
claim.

The installed Maintenance 2 dependency adds public `RngKey`, `CounterRng`,
`Threefry4x32`, `logical_positions`, `uniform`, `gaussian`, `poisson`, and
`binomial` surfaces at exact TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The closed Stage 6 TensorCore
commit `b454d738...` does not provide them. TensorCore independently owns
their exact implementation and
generic validation contract; TensorDSLab owns only the required consumer
behavior recorded below. Maintenance 2 installed and verified that selected
surface at the exact closed commits above.

Stage 2 and Maintenance 1 remain valid historical evidence for the current
package. This is a clean pre-deployment redesign, not a compatibility
layer. Governance Core remains Adopted; conformance remains Not evaluated;
Coordination remains Deferred; and Profile B remains Disabled.

## Design Thesis

TensorDSLab should present a small class-and-function API to collaborators who
do not need to understand TensorCore internals, field registries, dependency
maps, partial pipeline state, or buffer scheduling.

The ordinary workflow is:

```python
readout = simulate_readout(
    photoelectrons,
    products=[
        AnalogWaveform,
        DigitizedWaveform,
    ],
    config=config,
    rng=Threefry4x32(seed=1234),
)

analog = readout.field(AnalogWaveform)
digitized = readout.field(DigitizedWaveform)
```

The returned collection contains exactly the products requested by the caller:

```python
readout.field_types == frozenset(
    {
        AnalogWaveform,
        DigitizedWaveform,
    }
)
```

The builder computes prerequisites privately and at most once. A prerequisite
does not become a collection member unless it was also requested.

```text
requested DigitizedWaveform
  -> requires AnalogWaveform
       -> requires PureWaveform and NoiseWaveform
            -> PureWaveform requires Charge
                 -> Charge requires truth Photoelectrons
```

The architecture is:

```text
dense truth Photoelectrons
  -> complete request-aware private preparation
  -> typed product production and immediate validation
  -> one public simulate_readout(...)
  -> one immutable completed ReadoutCollection
       containing exactly the requested products
```

Exact Python classes carry in-process axis, product, and collection meaning.
There is no parallel namespace of axis IDs, field IDs, semantic constants,
product-name strings, canonical sequences, or dependency registries.

The public API separates three concerns:

```text
config    -> scientific model choices
products  -> final in-memory retention policy
rng       -> algorithm plus invocation seed
builder   -> dependency planning and execution
```

Durable persistence and IO are deferred entirely from this rebuild.

## Goals

- Make the normal API understandable from product names and function
  signatures alone.
- Use TensorCore's `TensorAxis`, `TensorField`, and `TensorCollection` directly.
- Replace loose semantic constants with exact final TensorDSLab types.
- Keep the readout input and all generated payloads dense, tensor-native, and
  resident on the caller-selected device.
- Preserve `Photoelectrons` as truth rather than replacing it with an
  electronics-smeared value.
- Express product dependencies through typed producer calls rather than a
  public workflow graph.
- Let callers retain only the products they consume.
- Keep scientific configuration exact, immutable, and compositional.
- Preserve accepted detector/readout behavior and parity classifications where
  their comparison boundaries still apply.
- Leave explicit later boundaries for TensorG4DS, TensorML, reconstruction,
  artifacts, and measured execution optimization, including product-local
  waveform-tail fusion after the functional producers are established.

## Non-Goals

- Backward compatibility with the historical pre-deployment TensorCore `0.6`
  representation.
- Compatibility aliases for retired IDs, constants, sidecars, or helper
  modules.
- Passing CPU-resident jagged G4DS tables into the readout builder.
- Native G4DS file parsing or TensorG4DS deposit clustering.
- Persistence, cache formats, artifact stores, or write policy.
- TensorML model, training, metric, or checkpoint ownership.
- Projects/dag scheduling, retries, fan-out, or campaign policy.
- A generic `Config(ABC)` without a real polymorphic consumer.
- A public workspace, allocator, stream lease, or `out=` surface before
  profiling demonstrates the need.
- An exact-until-extinction or recovery-marked correlated-avalanche cascade.
  The selected charge path uses one caller-bounded fixed-generation process;
  `maximum_generations=1` is its first-generation case rather than a separate
  algorithm.
- Bitwise parity merely because an older implementation produced a particular
  RNG stream.

## Ecosystem And Input Boundary

The intended chain remains:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared substrate, not another data-flow stage.

- G4DS owns native simulation output.
- TensorG4DS owns native ingestion, CPU-resident jagged source tables, and
  low-level tensor processing such as deposit clustering.
- TensorDSLab owns the dense downstream detector/readout products defined here
  and future reconstruction products.
- TensorML owns model-facing schemas, models, training, and evaluation.

`simulate_readout(...)` accepts a dense TensorDSLab
`Photoelectrons` field. It does not accept a native G4DS table, an Awkward
Array, a jagged PE table, or an untyped mapping of columns. Jagged tables remain
on the CPU and outside the GPU readout hot path.

A separate future TensorDSLab-owned bridge will consume one exact accepted
TensorG4DS product and construct the dense truth `Photoelectrons` field. That
bridge owns event/channel mapping and constructs the caller-selected compact
`SampleAxis(start=0, step=..., count=...)` while binning truth PEs. It must not
infer the window from observed hits or apply readout timing jitter:
jitter is an electronics response effect inside charge simulation, not truth
construction.

The production handoff target keeps dense payloads on one explicit accelerator
device and does not silently call `.cpu()`, `.numpy()`, convert through Python
lists, serialize/reload, cast, move, or detach.

## Historical TensorCore `0.7` Consumer Contract

This section records the contract implemented by Stage 3 and consumed through
Maintenance 4. Maintenance 5 replaces only its axis/dependency/sampling parts
with the compact TensorCore `0.13.0` target fixed near the top of this
document. Direct string-axis constructors below are historical, not the
Maintenance 5 API.

The rebuild targets TensorCore's three semantic roots:

```text
TensorAxis
  coordinates: tuple[str, ...]

TensorField
  tensor: torch.Tensor
  axes: tuple[TensorAxis, ...]

TensorCollection
  immutable fields keyed by exact TensorField subtype
```

Every TensorDSLab semantic leaf:

- has `__bases__ == (matching_tensor_core_root,)`, with no mixin or other base;
- is a public `@final` class;
- declares `__slots__ = ()`;
- adds no stored fields;
- implements `_require()` for TensorDSLab semantic narrowing; and
- inherits TensorCore construction, validation, immutability, and lookup.

Leaves do not reapply `@dataclass`. They use the ordinary inherited root
constructor. Direct inheritance, `@final`, empty slots, fieldlessness, and
inherited root behavior are TensorDSLab static-analysis, test, and Review
obligations rather than runtime lineage enforcement supplied by TensorCore.

TensorCore owns universal representation validation. TensorDSLab owns axis,
product, collection, dtype, device, scientific, and operation relationships.
Exact concrete class identity replaces runtime axis and field IDs.

TensorCore `0.7` has no layout object, metadata mapping, generic selection,
generic movement, output buffer, workspace, persistence, or lifecycle API.
Exact Python type identity is an in-process contract, not a durable artifact
identifier.

Stage 3 selected and tested exact TensorCore `0.7.0` dependency
`b454d738f6385ce6489d85492a618a3dab139bb6`, including runtime construction,
package-root imports, static constructor typing, exact-leaf validation, public
relationship helpers, ordinary-ABC inherited constructor signatures, and
concrete result inference. Later stages have retained and reverified the same
pin. Any newly discovered generic gap still returns to TensorCore Design with
a minimal reproducer rather than being patched through a downstream fork.

TensorCore establishes neither universal freshness nor universal storage
sharing. Every TensorDSLab operation returning one or more fields must classify
each successful path using TensorCore's exact vocabulary:

- exact return;
- guaranteed storage-sharing result;
- sharing permitted but unspecified; or
- guaranteed fresh storage independent of named inputs.

The owning operation separately specifies subtype, dtype, device, layout and
strides, axes, autograd, synchronization, failure effects, and any promised
output-to-output storage relationship. Constructing or returning a field is
not a device synchronization point. TensorCore provides no runtime overlap
scanner, copy-on-write layer, lease, workspace, or stream-ordering service.

TensorDSLab validates documented public inputs, scientific configuration, and
cheap correctness-critical operation relationships. It does not harden the
package against callers who leave the public contract by subclassing final
semantic leaves, modifying classes, bypassing inherited construction, invoking
private functions directly, mutating exposed tensors, or installing custom
dispatch behavior. Such use is unsupported and may fail naturally or produce
invalid results; it does not require a stable error category, an eager guard,
or adversarial test coverage.

## Selected Rebuild Package Shape

Status: implemented and Design-accepted through Maintenance 4. The displayed
tree is the current merged production structure. A future work order may
materialize only files needed by its own implementation slice; this is not
authorization to create placeholders.

```text
tensor_dslab/
  __init__.py

  common/
    __init__.py
    axes.py                  # ExampleAxis, ChannelAxis, SampleAxis
    sampling.py              # SamplingConfig and canonical sample-grid facts

  readout/
    __init__.py
    config.py                # ReadoutConfig only
    collection.py            # ReadoutCollection only
    requirements.py          # shared non-exported readout requirements
    simulation.py            # thin public orchestration
    runtime/
      __init__.py            # empty; no internal facade
      sampling.py            # SamplingRuntime and prepare_sampling()
      prepare.py             # ReadoutRuntime and prepare_readout()

    photoelectrons/
      __init__.py
      field.py               # Photoelectrons; cheap intrinsic narrowing only
      runtime/
        __init__.py          # empty; no internal facade
        validate.py          # validate_photoelectrons()

    charge/
      __init__.py
      config.py              # Charge-related configs
      field.py               # Charge
      runtime/
        __init__.py          # empty; no internal facade
        prepare.py           # ChargeRuntime and prepare_charge()
        produce.py           # produce_charge()
        validate.py          # validate_charge()
        effects/
          __init__.py        # empty; no internal facade
          counts.py          # Charge multinomial/count-domain orchestration
          delays.py          # prepared delay/recovery laws
          dark_counts.py
          timing_jitter.py
          correlated_avalanches.py
          smearing.py

    pure_waveform/
      __init__.py
      config.py              # wrapper and TPC/Veto model configs
      field.py               # PureWaveform
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py

    noise_waveform/
      __init__.py
      config.py              # zero/white/PSD configs
      field.py               # NoiseWaveform
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py

    analog_waveform/
      __init__.py
      config.py              # analog and saturation configs
      field.py               # AnalogWaveform
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py

    digitized_waveform/
      __init__.py
      config.py              # DigitizedWaveformConfig
      field.py               # DigitizedWaveform
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py
```

The tree is organized around semantic products rather than implementation
layers. A product's `field.py` owns its exact `TensorField` leaf, and its
`config.py` owns that product's public configuration records. Every generated
product owns explicit `prepare`, `produce`, and `validate` actions below its
non-exported `runtime/` package. Charge's scientific submodels are large enough
to earn focused modules below `charge/runtime/effects/`. Every runtime
`__init__.py` stays empty so internal callers import the exact defining module
rather than a second private facade.

`readout/config.py` contains only `ReadoutConfig`, while
`readout/collection.py` contains only `ReadoutCollection`. Product packages
never import either cross-product composition module. `requirements.py`
contains only relationships genuinely shared across products and is not
exported. There is no
`readout/_random.py` or `readout/_rng.py`: generic counter generation,
validated `RngPositions`, uniforms, parameterized Gaussian draws, Poisson
inversion/PTRS, and binomial inversion/BTRS come from the accepted TensorCore
dependency. Charge-owned multinomial/category orchestration, checked
accumulation, and ledger bookkeeping remain in
`charge/runtime/effects/counts.py`.

`ExampleAxis`, `ChannelAxis`, and `SampleAxis` belong in
`tensor_dslab.common.axes`. `SamplingConfig` belongs in
`tensor_dslab.common.sampling` because the future TensorG4DS bridge, readout,
and future Reconstruction may share the same sample-grid contract. This does
not add source binning to the current readout package.

`Photoelectrons` is an already-produced dense truth input. Its package has no
`PhotoelectronsConfig`, preparer, producer, or Runtime record; source
construction and PE binning remain deferred to the future TensorG4DS bridge.
It owns `validate_photoelectrons(...)` because it is the untrusted public
ingress. Accepted Stage 7
`simulate_readout(...)` borrows the supplied field and validates its
realized `SampleAxis` against the caller's `SamplingConfig`.

Runtime modules are private because no supported facade exports them. Privacy
is an API and compatibility boundary, not a runtime access-control mechanism,
and clean internal names do not make deep modules public. The public RNG
surface is a TensorCore `CounterRng`; stochastic role identity lives in the
fixed non-exported `readout/runtime/keys.py` table. TensorDSLab does not wrap
those values in an all-encompassing RNG config, re-export raw-word generation,
or move scientific category planning and bookkeeping to TensorCore.

The dependency direction is acyclic:

```text
tensor_core
  -> tensor_dslab.common
  -> readout.runtime.requirements and readout.runtime.keys
  -> product config and field modules
  -> readout.runtime.sampling
  -> product runtime.prepare modules and Charge effect preparation
  -> product runtime.produce / runtime.validate modules
  -> readout.config and readout.collection
  -> readout.runtime.prepare
  -> readout.simulation
  -> deliberate package-root exports
```

Product packages do not import `ReadoutConfig`, `ReadoutCollection`, or
`simulate_readout(...)`. Product preparers may import their own configs,
fields, exact prerequisite types, `SamplingRuntime`, focused requirements, and
Charge effect preparation. Producers receive no Config and never import a
validator. `readout.runtime.prepare` composes the complete request Runtime;
`readout.simulation` alone owns the topological `produce -> validate ->
descendant` sequence and final retention.

The physical module path does not define public visibility. Package
`__init__.py` files and `__all__` deliberately re-export the implemented
collaborator-facing classes and configs; closed Stage 7 deliberately exports
`simulate_readout(...)`.
Collaborators need not import from
nested product modules. `simulation.py`, rather than a generic `api.py`, names
the accepted behavior it owns. The singular product-local `config.py` and
`field.py` names state their concrete ownership; they are not global dumping
grounds. Do not add global `configs/`, `fields.py`, `builders.py`, or
`validation.py` modules. Product-local `runtime/validate.py` modules are
intentional because validation is a first-class product action.

Create another future module only when an accepted implementation stage gives
it real behavior. Do not create empty files to reserve this tree.

### Implemented Runtime-Action Symbol Inventory

This is the merged Maintenance 4 live inventory. Maintenance 2 and Stage 7
remain closed predecessor evidence. Maintenance 4 moved private symbols
without aliases. Future slices may add private details needed to express
accepted contracts, but must not
introduce a second registry or create a later behavior module as a placeholder.

| Module | Public symbols in the active MVP | Shared private symbols |
| --- | --- | --- |
| `common/axes.py` | `ExampleAxis`, `ChannelAxis`, `SampleAxis` | none |
| `readout/runtime/keys.py` | none | one fixed readout namespace and ten exact role keys |
| `readout/runtime/requirements.py` | none | `require_readout_structure` |
| `readout/runtime/sampling.py` | none | `SamplingRuntime`, `prepare_sampling` |
| `readout/runtime/prepare.py` | none | `ReadoutRuntime`, `prepare_readout`, request/closure/RNG-capability preparation |
| `readout/photoelectrons/field.py` | `Photoelectrons` | none beyond inherited `_require()` narrowing |
| `readout/photoelectrons/runtime/validate.py` | none | `validate_photoelectrons` |
| `readout/charge/config.py` | `TimingJitterConfig`, `DarkCountConfig`, `FixedDelayConfig`, `ExponentialDelayConfig`, `DirectCrosstalkConfig`, `DelayedCrosstalkConfig`, `AfterpulseRecoveryConfig`, `AfterpulseConfig`, `CorrelatedAvalancheConfig`, `ChargeSmearingConfig`, `ChargeConfig` | none |
| `readout/charge/field.py` | `Charge` | none beyond inherited `_require()` narrowing |
| `readout/charge/runtime/prepare.py` | none | `ChargeRuntime`, `prepare_charge` |
| `readout/charge/runtime/produce.py` | none | `produce_charge` |
| `readout/charge/runtime/validate.py` | none | `validate_charge` |
| `readout/charge/runtime/effects/*.py` | none | Charge-owned prepared records, count/delay mechanics, and clean `prepare_*` / `simulate_*` actions |
| `readout/pure_waveform/config.py` | `TpcFebSnrPulseConfig`, `VetoPduPulseConfig`, `PureWaveformConfig` | none |
| `readout/pure_waveform/field.py` | `PureWaveform` | none beyond inherited `_require()` narrowing |
| `readout/pure_waveform/runtime/*.py` | none | `PureWaveformRuntime`, `prepare_pure_waveform`, `produce_pure_waveform`, `validate_pure_waveform` |
| `readout/noise_waveform/config.py` | `ZeroNoiseConfig`, `WhiteNoiseConfig`, `PsdNoiseConfig`, `NoiseWaveformConfig` | none |
| `readout/noise_waveform/field.py` | `NoiseWaveform` | none beyond inherited `_require()` narrowing |
| `readout/noise_waveform/runtime/*.py` | none | `NoiseWaveformRuntime`, `prepare_noise_waveform`, `produce_noise_waveform`, `validate_noise_waveform` |
| `readout/analog_waveform/config.py` | `AnalogSaturationConfig`, `AnalogWaveformConfig` | none |
| `readout/analog_waveform/field.py` | `AnalogWaveform` | none beyond inherited `_require()` narrowing |
| `readout/analog_waveform/runtime/*.py` | none | `AnalogWaveformRuntime`, `prepare_analog_waveform`, `produce_analog_waveform`, `validate_analog_waveform` |
| `readout/digitized_waveform/config.py` | `DigitizedWaveformConfig` | none |
| `readout/digitized_waveform/field.py` | `DigitizedWaveform` | none beyond inherited `_require()` narrowing |
| `readout/digitized_waveform/runtime/*.py` | none | `DigitizedWaveformRuntime`, `prepare_digitized_waveform`, `produce_digitized_waveform`, `validate_digitized_waveform` |
| `readout/config.py` | `ReadoutConfig` | none |
| `readout/collection.py` | `ReadoutCollection` | none |
| `readout/simulation.py` | `simulate_readout` | only the topological action sequence and exact retention |

Shared downstream requirement functions exist only where two or more product
actions need the exact same relationship and TensorCore does not already own
the generic mechanic. Product-specific validators remain explicit and
product-named even when they delegate generic primitives directly to
TensorCore; this preserves ownership and error context without duplicating
tensor scans.

Stage 3 historically implemented and exported `NormalDelayConfig`. Stage 6
removed the class, both crosstalk-union memberships, all three export layers,
and its tests without a compatibility shim. The closed Stage 3 work order
remains unchanged as historical evidence. TensorDSLab is pre-deployment and
makes no backward-compatibility claim.

Every product subpackage root re-exports only its public row above. The
`common` and `readout` roots compose those deliberate exports, and the package
root re-exports the collaborator-facing axes, sampling/config types, product
field types, `ReadoutConfig`, `ReadoutCollection`, and `simulate_readout`.
Generic TensorCore names are never re-exported. Importing the public package
deliberately loads the
Stage 7 orchestration module but must not transitively load private TensorCore
RNG mechanics, a compiler, TensorG4DS, TensorML, or an IO dependency.

Stages 4 through 6 subsequently implemented every generated product's private
`_produce_*` builder, Charge's private `_simulate_*` submodels, and the private
RNG mechanics consumed by noise and Charge. Closed Stage 7 implements
`readout.simulation.simulate_readout`. The staged separation kept the Stage 3
foundation testable without creating empty architectural scaffolding.

The rebuild retires:

- `common/ids.py`, `ExampleId`, and `ChannelId`;
- `readout/ids.py`;
- `TensorAxisId`, `TensorFieldId`, and `IdSequence` values;
- module-level axis/field constants and product registries;
- `TensorLayout`, `shared_axes`, and layout reconstruction;
- count-only sample semantics and `SampleGrid`;
- `DigitizedWaveformSpec` as a collection sidecar;
- ordered partial pipeline snapshots and descendant invalidation;
- `readout/tensors.py` reconstruction/projection helpers; and
- public mutation of a collection through atomic add-or-replace transforms.

The package root and subpackage roots export only deliberate public classes and
functions. Historical work orders remain unchanged as records of what they
implemented.

## Semantic Axes

```python
from typing import final

from tensor_core import TensorAxis


@final
class ExampleAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
            raise ValueError("ExampleAxis must be nonempty")


@final
class ChannelAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
            raise ValueError("ChannelAxis must be nonempty")


@final
class SampleAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if len(self.coordinates) < 2:
            raise ValueError("SampleAxis requires at least two timestamps")

        times_ps: list[int] = []
        for coordinate in self.coordinates:
            if not coordinate.endswith("ps"):
                raise ValueError("SampleAxis timestamps must end in 'ps'")
            magnitude = coordinate[:-2]
            if not (
                magnitude == "0"
                or (
                    magnitude
                    and magnitude[0] != "0"
                    and magnitude.isascii()
                    and magnitude.isdigit()
                )
            ):
                raise ValueError("noncanonical SampleAxis timestamp")
            time_ps = int(magnitude)
            if time_ps > (1 << 63) - 1:
                raise ValueError("SampleAxis timestamp exceeds int64")
            times_ps.append(time_ps)

        period_ps = times_ps[1] - times_ps[0]
        if period_ps <= 0:
            raise ValueError("SampleAxis timestamps must increase")
        if any(
            right - left != period_ps
            for left, right in zip(times_ps, times_ps[1:])
        ):
            raise ValueError("SampleAxis timestamps must be uniformly spaced")
        if times_ps[-1] + period_ps > (1 << 63) - 1:
            raise ValueError("SampleAxis exclusive stop exceeds int64")

    @property
    def start_ps(self) -> int:
        return int(self.coordinates[0][:-2])

    @property
    def sample_period_ps(self) -> int:
        return int(self.coordinates[1][:-2]) - self.start_ps

    @property
    def stop_ps(self) -> int:
        return int(self.coordinates[-1][:-2]) + self.sample_period_ps
```

There are no corresponding axis-ID constants. Code locates dimensions by exact
type:

```python
sample_dimension = field.dimension_of(SampleAxis)
sample_axis = field.axis(SampleAxis)
```

### Coordinate Contract

Every coordinate is an exact, unique, nonempty string. Tuple order is tensor
index order. Positional RNG uses that index and does not use the coordinate
string as random identity.

- `ExampleAxis` contains stable TensorDSLab example keys.
- `ChannelAxis` contains stable detector/readout channel keys.
- `SampleAxis` contains canonical time-ordered truth-bin timestamp strings
  constructed from the accepted sampling policy.

`SamplingConfig` owns the numeric policy from which a regular timestamp axis
is realized. The ordinary construction path has this shape:

```python
sampling = SamplingConfig(
    sample_period_ps=PositiveInteger(2_000),
    sample_count=PositiveInteger(8_192),
)
samples = sampling.build_axis()
```

TensorDSLab uses one bin convention everywhere: every stored bin coordinate is
the inclusive left edge, every bin is left-closed and right-open, and the final
exclusive stop is carried or derived separately. Public bin arrays therefore
never mix left edges with centers or terminal right edges. This applies to
sample timestamps, upstream numeric PE bins, PSD frequency bins, and later
histogram-like scientific inputs unless a focused Design change says otherwise.

The config first defines numeric left edges plus the exclusive window stop for
upstream PE binning and then generates canonical semantic left-edge coordinates
such as `"0ps"`, `"2000ps"`, and `"4000ps"`. A full window starts at
example-local zero. The exact timestamp grammar is ASCII
`^(0|[1-9][0-9]*)ps$`: lowercase `ps`, no sign, whitespace, decimal point,
exponent, alternate unit, or leading zero. Direct `SampleAxis(...)`
construction remains available for semantic reconstruction, but `_require()`
requires at least two coordinates, nonnegative signed-int64 values, strict
increase, one positive integer-picosecond spacing, and a derived exclusive
stop no greater than `2**63 - 1`.

This is the low-level construction surface for fixtures, custom sources, and
the future TensorG4DS bridge. It is not a second readout builder.
`simulate_readout(...)` receives `Photoelectrons` with a complete
`SampleAxis`, requires its already-validated start, period, and size to match
`config.sampling`, reuses that exact axis object for every generated field, and
never creates, rebases, or replaces it. Because construction has already
validated the complete regular tuple, agreement is an O(1) check of size,
`start_ps == 0`, and `sample_period_ps`; it neither rebuilds nor reparses the
full coordinate tuple in the repeated readout path.

The earlier count-only representation was never used and is not carried into
the rebuild. `SampleGrid` is retired. Timestamp strings describe dense readout
bins, not individual G4 hits.

TensorCore does not parse or chronologically validate strings. TensorDSLab's
accepted `SampleAxis._require()` contract owns the grammar, signed-int64
domain, chronological order, positive uniform spacing, and derivable period
and stop. Charge simulation, pulse convolution, and
power-spectral-density-shaped noise synthesis consume numeric `SamplingConfig`
values and tensor indices rather than parsing coordinate strings. Operation
preflight may impose additional algorithm-specific limits on an already-valid
period.

A shared `SampleAxis` means every example in one dense tensor uses the same
relative readout-bin coordinates. A complete bridge-produced example window
starts at zero. With left-edge coordinates and period `dt`, its timestamps are
`0, dt, ..., (sample_count - 1) * dt`, bin `i` represents
`[i * dt, (i + 1) * dt)`, and the exclusive window stop is
`sample_count * dt`. Per-example absolute G4 origins and trigger position, if
needed, belong in explicit bridge provenance rather than ambiguous sample
coordinates.

After the future bridge has normalized an upstream hit to an exact accepted
example-local integer-picosecond value, bin assignment is:

```text
if 0 <= time_ps < sampling.window_stop_ps:
    sample_index = time_ps // sampling.sample_period_ps.value
else:
    drop and account for the hit
```

This freezes boundary ownership but not the upstream conversion of floating G4
time into exact numeric picoseconds. That rounding/normalization policy remains
part of the focused TensorG4DS bridge contract.

A later contiguous selection of at least two samples preserves the selected
timestamp strings instead of rebasing them, so a valid subaxis may start above
zero while remaining relative to the original example origin. A singleton
selection cannot reconstruct a period-bearing `SampleAxis` in the MVP and
remains a non-readout semantic result until a separate timing-association
contract exists. The zero-start rule is therefore a full-window bridge
postcondition, not a universal `SampleAxis` invariant. The MVP public builder
accepts only a full source axis matching its zero-start `SamplingConfig`; making
a selected subwindow a new simulation input requires a later boundary/halo
policy rather than silently treating it as a complete window.

### Axis Order

Every readout field contains exactly one `ExampleAxis`, one `ChannelAxis`, and
one `SampleAxis`. Semantic construction accepts those axes in any order; the
tuple order remains tensor dimension order. Builders reuse the exact source
axis instances and locate dimensions by exact type. A different valid axis
tuple order is nevertheless a different positional RNG schema; the builder
does not attempt to reproduce the same draws across a tensor permutation.

The local relationship check is order-independent and does not accidentally
call TensorCore's ordered `require_axis_signature(...)`:

```python
def require_readout_structure(field: TensorField) -> None:
    axis_types = tuple(type(axis) for axis in field.axes)
    accepted = frozenset({ExampleAxis, ChannelAxis, SampleAxis})
    if len(axis_types) != 3 or frozenset(axis_types) != accepted:
        raise ValueError(
            "readout fields require exactly example, channel, and sample axes"
        )
    if field.tensor.layout is not torch.strided:
        raise ValueError("readout fields require dense strided tensors")
```

The shared `require_floating_dtype(...)` relationship accepts exactly
`torch.float32` or `torch.float64`. No product leaf expands that set to
`torch.float16` or `torch.bfloat16`.

The supported MVP contract and its storage, aliasing, compiled-kernel, and
fresh-result evidence cover ordinary `torch.Tensor` behavior. Custom tensor
subclasses and dispatch modes are outside that contract. TensorDSLab does not
need a defensive runtime guard that recognizes or rejects every unsupported
Torch extension.

The upstream bridge should ordinarily construct
`(ExampleAxis, ChannelAxis, SampleAxis)` so samples are last for temporal GPU
kernels. A future warmed execution profile may require sample-last contiguous
storage without changing semantic identity.

## Product Fields

TensorDSLab defines six direct final `TensorField` leaves:

| Product | Intrinsic leaf contract | Runtime-validation postcondition | Meaning |
| --- | --- | --- | --- |
| `Photoelectrons` | `torch.int64`, exact readout axes | nonnegative | dense binned photon-origin truth PE counts |
| `Charge` | `torch.float32` or `torch.float64` | finite and nonnegative | aggregate PE-equivalent SiPM response |
| `PureWaveform` | `torch.float32` or `torch.float64` | finite | signal-only waveform in mV |
| `NoiseWaveform` | `torch.float32` or `torch.float64` | finite | noise-only voltage excursion about zero in mV |
| `AnalogWaveform` | `torch.float32` or `torch.float64` | finite | zero-referenced composed analog waveform in mV |
| `DigitizedWaveform` | `torch.int32` | nonnegative and bounded by its prepared maximum code | immediate ADC-code output |

`DigitizedWaveform`, not `DigitalWaveform`, remains the accepted name.
`DigitalWaveform` is reserved for a possible later firmware/filter/trigger/
compression product.

Every field uses `torch.strided` tensor layout. Operations preserve existing
input fields and perform no implicit movement or in-place cast. Newly generated
products intentionally use their declared output dtype: truth `torch.int64`
becomes floating `Charge`, and floating `AnalogWaveform` becomes
`torch.int32` ADC codes. Charge, pure, noise, and analog use one
builder-selected `torch.float32` or `torch.float64` dtype.

Illustrative leaves are:

```python
@final
class Photoelectrons(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_dtype(self, torch.int64)


@final
class Charge(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_floating_dtype(self)


@final
class PureWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_floating_dtype(self)


@final
class NoiseWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_floating_dtype(self)


@final
class AnalogWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_floating_dtype(self)


@final
class DigitizedWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        require_readout_structure(self)
        require_dtype(self, torch.int32)
```

Repeated field relationships live in private functions. There is no artificial
`ReadoutField` base and no loose dtype/dependency/name mapping.

### Truth Meaning Of `Photoelectrons`

`Photoelectrons` is the dense, binned photon-origin truth input to readout
simulation. It never includes timing jitter, dark counts, crosstalk,
afterpulses, or charge smearing.

When configured effectively, timing jitter is a private step of charge
production after any private dark-count avalanches have been added. It
redistributes the then-current working counts but does not replace or mutate
the source field. Therefore:

- requesting `Photoelectrons` retains the exact input field;
- requesting `Charge` uses a private working-count representation that is
  jittered only when that stage executes;
- requesting both returns unjittered truth beside the derived charge; and
- the original input remains unchanged in every case.

### Structural And Deep Validation

TensorCore always validates tensor/axis shape and semantic lineage.
TensorDSLab leaf construction validates cheap intrinsic facts such as exact
axes, dtype, and `torch.layout`. It does not hide a full-device
`isfinite().all()` or nonnegativity scan inside every construction.

The exact class identifies declared semantic role and representation; it is
not proof that arbitrary caller-supplied values satisfy every scientific
postcondition. Public builders guarantee their output domains. Untrusted
ingress—including the source `Photoelectrons` supplied to the public builder
and any future artifact load—runs the explicit product-specific deep validator
before scientific execution.

This avoids repeated accelerator synchronization without weakening the named
trust boundary. Documentation must not claim that a bare constructor proves a
config-dependent ADC maximum. Builder postconditions and deep validators are
tested separately.

Stage 7 closed that builder result boundary inside each producer. The merged
Maintenance 4 implementation preserves the boundary while making validation a
separate product-owned action:
`simulate_readout(...)` calls the exact `validate_<product>(...)` once with the
new local result and its named prerequisite relationships immediately after
production and before any descendant. A successful prerequisite is deeply
valid before downstream use. A failed postcondition may follow payload work
and local field construction, but the field does not escape and no partial
collection is returned. These explicit scans may synchronize CUDA.

## `ReadoutCollection`

`ReadoutCollection` is an immutable completed result for one explicit product
request. It accepts any nonempty subset of the six recognized product types.
It is not a partially executed pipeline and exposes no add, replace, or
invalidation lifecycle.

Here *completed* means that collection membership is final and contains no
workflow state. It does not mean every variable calibration fact is stored in
the sidecar-free collection record. In particular, a digitized-only result is a
valid completed in-process result only while its caller separately retains the
exact `DigitizedWaveformConfig` needed to interpret it. Durable or
independently transported digitized values remain blocked on the explicit
association in Design gate 4.

Membership is semantically unordered. A collection requires:

- at least one recognized exact product type;
- no unrecognized type;
- at most one field of each exact type, already enforced generically by
  TensorCore;
- equal ordered axes on every present field;
- the same exact device on every present field;
- one common dtype among all present floating readout fields; and
- every product's intrinsic leaf contract.

The accepted schema is declared once on the owning collection class:

```python
@final
class ReadoutCollection(TensorCollection):
    __slots__ = ()

    @classmethod
    def accepted_field_types(
        cls,
    ) -> frozenset[type[TensorField]]:
        return frozenset(
            {
                Photoelectrons,
                Charge,
                PureWaveform,
                NoiseWaveform,
                AnalogWaveform,
                DigitizedWaveform,
            }
        )

    def _require(self) -> None:
        if not self.field_types:
            raise ValueError("ReadoutCollection must be nonempty")
        require_field_types(
            self,
            required=frozenset(),
            optional=self.accepted_field_types(),
        )

        fields = tuple(self.fields.values())
        require_same_axes(*fields)
        require_same_device(*fields)

        floating_fields = tuple(
            field
            for field in fields
            if field.tensor.is_floating_point()
        )
        require_same_dtype(*floating_fields)
```

This one class-owned method is the unavoidable accepted-schema declaration. It
replaces module constants, field IDs, canonical-order registries, floating-role
registries, and descendant maps. Its returned set has no order semantics.

Ordinary access is type-directed:

```python
analog = readout.field(AnalogWaveform)
analog_tensor = readout.tensor(AnalogWaveform)
```

A missing unrequested product raises `KeyError`:

```python
readout.field(Charge)  # computed as a prerequisite but not retained
```

Typed convenience properties may be added later only if they materially help
collaborators and remain thin exact-type lookups.

## Product Requests

`products` is a required keyword-only iterable of exact product classes. A
caller may pass a list, tuple, generator, or another iterable; its iteration
order has no semantic meaning.

The builder:

1. consumes the iterable exactly once;
2. rejects an empty request;
3. requires exact classes accepted by `ReadoutCollection`;
4. rejects duplicates before converting membership to a set;
5. computes the transitive prerequisite closure;
6. preflights every required config and runtime relationship;
7. executes each required product producer at most once; and
8. retains only requested fields.

Unknown, duplicate, empty, or unsatisfied requests fail before an RNG request,
product-producer invocation, or semantic-output write.

The planner is ordinary typed code, not a public dependency registry. A
conceptual implementation can derive booleans from the requested set:

```python
need_digitized = DigitizedWaveform in requested
need_analog = AnalogWaveform in requested or need_digitized
need_pure = PureWaveform in requested or need_analog
need_noise = NoiseWaveform in requested or need_analog
need_charge = Charge in requested or need_pure
```

`Photoelectrons` is always available as the source, but it is retained only
when explicitly requested. Each requested combination remains a completed
result because no collection member represents workflow state.

Changing retention alone must not change the value of a product common to two
requests. RNG design and operation scheduling must therefore isolate product
random fields from unrelated requested branches.

## Scientific Configuration

Except where explicitly historical, the product configuration contracts below
remain accepted. Maintenance 5 removes only `SamplingConfig` and
`ReadoutConfig.sampling`; every later sketch showing those values is historical
and must be read as source-derived `SamplingRuntime` under the supersession
block above.

`ReadoutConfig` composes one required shared sampling policy with optional
exact product configs. Every product preparer with scientific choices accepts
its exact config type. A time-dependent preparer also receives the shared exact
`SamplingConfig`; its producer consumes the resulting typed plan. No
subfunction receives the whole `ReadoutConfig` as a service locator.

Concrete configs are normal domain value classes:

```text
@final
@dataclass(frozen=True, slots=True, kw_only=True)
```

They validate exact component types, compose other configs, and may use
TensorCore constrained scalars. `None` disables an optional submodel.
Alternative algorithms use closed unions of exact config classes rather than
string switches.

The hierarchy is:

```text
ReadoutConfig
├── SamplingConfig
├── ChargeConfig | None
│   ├── DarkCountConfig | None
│   ├── TimingJitterConfig | None
│   ├── CorrelatedAvalancheConfig | None
│   │   ├── maximum_generations
│   │   ├── DirectCrosstalkConfig | None
│   │   ├── DelayedCrosstalkConfig | None
│   │   └── AfterpulseConfig | None
│   │       └── AfterpulseRecoveryConfig | None
│   └── ChargeSmearingConfig | None
├── PureWaveformConfig | None
│   └── model: TpcFebSnrPulseConfig | VetoPduPulseConfig
├── NoiseWaveformConfig | None
│   └── model: ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig
├── AnalogWaveformConfig | None
│   └── AnalogSaturationConfig | None
└── DigitizedWaveformConfig | None
```

This is the sole charge hierarchy selected by the rebuild. It does not reserve
a second first-generation crosstalk/afterpulse surface. A caller who wants one
offspring generation chooses `maximum_generations=1`; increasing `K` extends
the same coupled algorithm. No second recursive surface or competing
avalanche-algorithm document is part of the rebuild contract.

The two product wrappers deliberately use the same vocabulary:

```python
PureWaveformConfig(model=TpcFebSnrPulseConfig(...))
NoiseWaveformConfig(model=PsdNoiseConfig(...))
```

The exact model class selects the accepted algorithm without a string switch,
registry, loose type alias, or marker ABC. TPC and Veto pulse response have
different equations and parameter schemas, so they earn separate exact model
classes. Noise generation, analog composition, saturation, and digitization
remain shared algorithms whose calibrated values may differ by detector
subsystem.

### Scalar MVP Calibration

Every scientific calibration value in an MVP config is scalar and applies
uniformly to the complete channel axis for one `simulate_readout(...)` call.
The same configured pulse parameters, noise model and power, analog limits,
ADC transfer, and eventual charge-response parameters apply to every channel
and example in that invocation. This is parameter homogeneity, not output
equality: source values and position-addressed stochastic realizations may
differ independently at every tensor position.

One `PureWaveformConfig.model` likewise applies to the complete channel axis.
The MVP therefore has a caller precondition that each invocation is
homogeneous with respect to both its TPC or Veto electronics-response family
and its calibration values. TPC and Veto, or two differently calibrated
channel groups, are simulated in separate invocations. Generic `ChannelAxis`
strings carry no trusted family or calibration provenance, so preflight does
not infer parameters from coordinate text and performs no per-channel lookup
or implicit parameter broadcasting.

Future channel-varying calibration should use an explicit, strongly typed,
device-resident tensor representation whose channel axis is validated against
the simulated data. It should be passed as a deliberate scientific input and
prepared before the hot path. It should not be smuggled into these frozen
scalar configs as a mutable `torch.Tensor`, a channel-keyed dictionary, or a
large tuple whose ordering must be interpreted privately. The exact future
type, supported parameter axes, movement/lifetime rules, and composition with
scalar configs require a focused Design stage; the MVP does not reserve a
placeholder class or pretend that arbitrary broadcasting is supported.

There is no `PhotoelectronsConfig`: the cross-product `SamplingConfig` defines
the shared numeric readout window used to create dense truth and every later
time-dependent product. `Photoelectrons` remains an already-constructed truth
input to `simulate_readout(...)`; the future TensorG4DS bridge receives
the same `SamplingConfig` when it bins the upstream jagged PE table. Timing
jitter belongs to `ChargeConfig` because it is a private electronics-response
step used only to produce charge.

`SamplingConfig` is always required because it defines the realized source
grid. Top-level product configs are optional so a caller can configure only
the requested computation. Missing required product config is a
request-specific preflight error. A `None` nested inside an existing product
config disables that submodel; a `None` top-level product config means that
product cannot be built by this invocation.

Illustrative definitions are shown together below even though their production
owners are split across `tensor_dslab.common.sampling`, `readout.config`, and
the corresponding product package's `config.py`. TPC and Veto pulse-model
configs initially live with `PureWaveform` in
`readout.pure_waveform.config`; split them further inside that product package
only when real implementation size or behavior justifies it:

```python
def _require_exact(value: object, expected: type[object], field: str) -> None:
    if type(value) is not expected:
        raise TypeError(f"{field} must be exactly {expected.__name__}")


def _require_optional_exact(
    value: object | None,
    expected: type[object],
    field: str,
) -> None:
    if value is not None:
        _require_exact(value, expected, field)


def _require_one_of_exact(
    value: object,
    expected: tuple[type[object], ...],
    field: str,
) -> None:
    if type(value) not in expected:
        names = ", ".join(item.__name__ for item in expected)
        raise TypeError(f"{field} must be exactly one of: {names}")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingConfig:
    sample_period_ps: PositiveInteger
    sample_count: PositiveInteger

    def __post_init__(self) -> None:
        _require_exact(
            self.sample_period_ps,
            PositiveInteger,
            "SamplingConfig.sample_period_ps",
        )
        _require_exact(
            self.sample_count,
            PositiveInteger,
            "SamplingConfig.sample_count",
        )
        if self.sample_count.value < 2:
            raise ValueError("SamplingConfig.sample_count must be at least 2")
        if (
            self.sample_period_ps.value * self.sample_count.value
            > (1 << 63) - 1
        ):
            raise ValueError("SamplingConfig.window_stop_ps exceeds int64")

    @property
    def window_stop_ps(self) -> int:
        return self.sample_period_ps.value * self.sample_count.value

    def build_axis(self) -> SampleAxis:
        period = self.sample_period_ps.value
        return SampleAxis(
            coordinates=tuple(
                f"{index * period}ps"
                for index in range(self.sample_count.value)
            )
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TimingJitterConfig:
    sigma_ns: NonnegativeFloat
    rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0008,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.sigma_ns,
            NonnegativeFloat,
            "TimingJitterConfig.sigma_ns",
        )
        _require_exact(
            self.rng_key,
            RngKey,
            "TimingJitterConfig.rng_key",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DarkCountConfig:
    rate_hz: NonnegativeFloat
    rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0003,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.rate_hz,
            NonnegativeFloat,
            "DarkCountConfig.rate_hz",
        )
        _require_exact(
            self.rng_key,
            RngKey,
            "DarkCountConfig.rng_key",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FixedDelayConfig:
    delay_ns: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.delay_ns,
            NonnegativeFloat,
            "FixedDelayConfig.delay_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExponentialDelayConfig:
    mean_delay_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_delay_ns,
            PositiveFloat,
            "ExponentialDelayConfig.mean_delay_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig
    retained_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0004,
    )
    overflow_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0005,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DirectCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig),
            "DirectCrosstalkConfig.delay",
        )
        _require_exact(
            self.retained_rng_key,
            RngKey,
            "DirectCrosstalkConfig.retained_rng_key",
        )
        _require_exact(
            self.overflow_rng_key,
            RngKey,
            "DirectCrosstalkConfig.overflow_rng_key",
        )
        if self.retained_rng_key == self.overflow_rng_key:
            raise ValueError("direct crosstalk RNG keys must differ")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DelayedCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig
    retained_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0006,
    )
    overflow_rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0007,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DelayedCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig),
            "DelayedCrosstalkConfig.delay",
        )
        _require_exact(
            self.retained_rng_key,
            RngKey,
            "DelayedCrosstalkConfig.retained_rng_key",
        )
        _require_exact(
            self.overflow_rng_key,
            RngKey,
            "DelayedCrosstalkConfig.overflow_rng_key",
        )
        if self.retained_rng_key == self.overflow_rng_key:
            raise ValueError("delayed crosstalk RNG keys must differ")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseRecoveryConfig:
    time_constant_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.time_constant_ns,
            PositiveFloat,
            "AfterpulseRecoveryConfig.time_constant_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseConfig:
    probability: Probability
    mean_delay_ns: PositiveFloat
    recovery: AfterpulseRecoveryConfig | None = None
    rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0009,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.probability,
            Probability,
            "AfterpulseConfig.probability",
        )
        _require_exact(
            self.mean_delay_ns,
            PositiveFloat,
            "AfterpulseConfig.mean_delay_ns",
        )
        _require_optional_exact(
            self.recovery,
            AfterpulseRecoveryConfig,
            "AfterpulseConfig.recovery",
        )
        _require_exact(
            self.rng_key,
            RngKey,
            "AfterpulseConfig.rng_key",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CorrelatedAvalancheConfig:
    maximum_generations: NonnegativeInteger
    direct_crosstalk: DirectCrosstalkConfig | None = None
    delayed_crosstalk: DelayedCrosstalkConfig | None = None
    afterpulse: AfterpulseConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(
            self.maximum_generations,
            NonnegativeInteger,
            "CorrelatedAvalancheConfig.maximum_generations",
        )
        _require_optional_exact(
            self.direct_crosstalk,
            DirectCrosstalkConfig,
            "CorrelatedAvalancheConfig.direct_crosstalk",
        )
        _require_optional_exact(
            self.delayed_crosstalk,
            DelayedCrosstalkConfig,
            "CorrelatedAvalancheConfig.delayed_crosstalk",
        )
        _require_optional_exact(
            self.afterpulse,
            AfterpulseConfig,
            "CorrelatedAvalancheConfig.afterpulse",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeSmearingConfig:
    relative_sigma: NonnegativeFloat
    rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_000A,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.relative_sigma,
            NonnegativeFloat,
            "ChargeSmearingConfig.relative_sigma",
        )
        _require_exact(
            self.rng_key,
            RngKey,
            "ChargeSmearingConfig.rng_key",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeConfig:
    dark_count: DarkCountConfig | None = None
    timing_jitter: TimingJitterConfig | None = None
    correlated_avalanches: CorrelatedAvalancheConfig | None = None
    smearing: ChargeSmearingConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_exact(
            self.dark_count,
            DarkCountConfig,
            "ChargeConfig.dark_count",
        )
        _require_optional_exact(
            self.timing_jitter,
            TimingJitterConfig,
            "ChargeConfig.timing_jitter",
        )
        _require_optional_exact(
            self.correlated_avalanches,
            CorrelatedAvalancheConfig,
            "ChargeConfig.correlated_avalanches",
        )
        _require_optional_exact(
            self.smearing,
            ChargeSmearingConfig,
            "ChargeConfig.smearing",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TpcFebSnrPulseConfig:
    fast_time_constant_ns: PositiveFloat
    slow_time_constant_ns: PositiveFloat
    support_time_ns: PositiveFloat
    peak_voltage_mv_per_pe: FiniteFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.fast_time_constant_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.fast_time_constant_ns",
        )
        _require_exact(
            self.slow_time_constant_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.slow_time_constant_ns",
        )
        _require_exact(
            self.support_time_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.support_time_ns",
        )
        _require_exact(
            self.peak_voltage_mv_per_pe,
            FiniteFloat,
            "TpcFebSnrPulseConfig.peak_voltage_mv_per_pe",
        )
        if (
            self.slow_time_constant_ns.value
            <= self.fast_time_constant_ns.value
        ):
            raise ValueError(
                "slow time constant must exceed fast time constant"
            )
        if self.peak_voltage_mv_per_pe.value == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class VetoPduPulseConfig:
    gaussian_center_ns: FiniteFloat
    gaussian_width_ns: PositiveFloat
    edge_offset_1_ns: FiniteFloat
    edge_width_1_ns: PositiveFloat
    edge_offset_2_ns: FiniteFloat
    edge_width_2_ns: PositiveFloat
    support_time_ns: PositiveFloat
    peak_voltage_mv_per_pe: FiniteFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.gaussian_center_ns,
            FiniteFloat,
            "VetoPduPulseConfig.gaussian_center_ns",
        )
        _require_exact(
            self.gaussian_width_ns,
            PositiveFloat,
            "VetoPduPulseConfig.gaussian_width_ns",
        )
        _require_exact(
            self.edge_offset_1_ns,
            FiniteFloat,
            "VetoPduPulseConfig.edge_offset_1_ns",
        )
        _require_exact(
            self.edge_width_1_ns,
            PositiveFloat,
            "VetoPduPulseConfig.edge_width_1_ns",
        )
        _require_exact(
            self.edge_offset_2_ns,
            FiniteFloat,
            "VetoPduPulseConfig.edge_offset_2_ns",
        )
        _require_exact(
            self.edge_width_2_ns,
            PositiveFloat,
            "VetoPduPulseConfig.edge_width_2_ns",
        )
        _require_exact(
            self.support_time_ns,
            PositiveFloat,
            "VetoPduPulseConfig.support_time_ns",
        )
        _require_exact(
            self.peak_voltage_mv_per_pe,
            FiniteFloat,
            "VetoPduPulseConfig.peak_voltage_mv_per_pe",
        )
        if self.peak_voltage_mv_per_pe.value == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PureWaveformConfig:
    model: TpcFebSnrPulseConfig | VetoPduPulseConfig

    def __post_init__(self) -> None:
        _require_one_of_exact(
            self.model,
            (TpcFebSnrPulseConfig, VetoPduPulseConfig),
            "PureWaveformConfig.model",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroNoiseConfig:
    """Select the exact all-zero noise algorithm."""


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class WhiteNoiseConfig:
    rms_mv: PositiveFloat
    rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0001,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.rms_mv,
            PositiveFloat,
            "WhiteNoiseConfig.rms_mv",
        )
        _require_exact(
            self.rng_key,
            RngKey,
            "WhiteNoiseConfig.rng_key",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PsdNoiseConfig:
    frequency_left_edges_hz: tuple[NonnegativeFloat, ...]
    frequency_stop_hz: PositiveFloat
    power_density_mv2_per_hz: tuple[NonnegativeFloat, ...]
    rng_key: RngKey = RngKey(
        namespace=0x54445331,
        stream=0x0000_0002,
    )

    def __post_init__(self) -> None:
        _require_exact(
            self.rng_key,
            RngKey,
            "PsdNoiseConfig.rng_key",
        )
        if type(self.frequency_left_edges_hz) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.frequency_left_edges_hz must be a tuple"
            )
        if type(self.power_density_mv2_per_hz) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.power_density_mv2_per_hz must be a tuple"
            )
        if not self.frequency_left_edges_hz:
            raise ValueError("a PSD requires at least one frequency bin")
        if len(self.frequency_left_edges_hz) != len(
            self.power_density_mv2_per_hz
        ):
            raise ValueError("PSD left-edge and density counts must match")
        for edge in self.frequency_left_edges_hz:
            _require_exact(
                edge,
                NonnegativeFloat,
                "PsdNoiseConfig.frequency_left_edges_hz",
            )
        _require_exact(
            self.frequency_stop_hz,
            PositiveFloat,
            "PsdNoiseConfig.frequency_stop_hz",
        )
        if self.frequency_left_edges_hz[0].value != 0.0:
            raise ValueError("PSD frequency coverage must start at zero")
        if any(
            right.value <= left.value
            for left, right in zip(
                self.frequency_left_edges_hz,
                self.frequency_left_edges_hz[1:],
            )
        ):
            raise ValueError(
                "PSD frequency left edges must be strictly increasing"
            )
        if (
            self.frequency_left_edges_hz[-1].value
            >= self.frequency_stop_hz.value
        ):
            raise ValueError("PSD frequency stop must exceed its final left edge")
        for density in self.power_density_mv2_per_hz:
            _require_exact(
                density,
                NonnegativeFloat,
                "PsdNoiseConfig.power_density_mv2_per_hz",
            )
        if not any(
            density.value > 0.0
            for density in self.power_density_mv2_per_hz
        ):
            raise ValueError("use ZeroNoiseConfig for an all-zero PSD")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NoiseWaveformConfig:
    model: ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig

    def __post_init__(self) -> None:
        _require_one_of_exact(
            self.model,
            (
                ZeroNoiseConfig,
                WhiteNoiseConfig,
                PsdNoiseConfig,
            ),
            "NoiseWaveformConfig.model",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogSaturationConfig:
    minimum_mv: FiniteFloat | None = None
    maximum_mv: FiniteFloat | None = None

    def __post_init__(self) -> None:
        _require_optional_exact(
            self.minimum_mv,
            FiniteFloat,
            "AnalogSaturationConfig.minimum_mv",
        )
        _require_optional_exact(
            self.maximum_mv,
            FiniteFloat,
            "AnalogSaturationConfig.maximum_mv",
        )
        if self.minimum_mv is None and self.maximum_mv is None:
            raise ValueError("analog saturation requires at least one bound")
        if (
            self.minimum_mv is not None
            and self.maximum_mv is not None
            and self.minimum_mv.value >= self.maximum_mv.value
        ):
            raise ValueError("analog saturation minimum must be below maximum")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogWaveformConfig:
    saturation: AnalogSaturationConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_exact(
            self.saturation,
            AnalogSaturationConfig,
            "AnalogWaveformConfig.saturation",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DigitizedWaveformConfig:
    bit_depth: PositiveInteger
    input_min_mv: FiniteFloat
    input_max_mv: FiniteFloat
    analog_gain_db: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.bit_depth,
            PositiveInteger,
            "DigitizedWaveformConfig.bit_depth",
        )
        _require_exact(
            self.input_min_mv,
            FiniteFloat,
            "DigitizedWaveformConfig.input_min_mv",
        )
        _require_exact(
            self.input_max_mv,
            FiniteFloat,
            "DigitizedWaveformConfig.input_max_mv",
        )
        _require_exact(
            self.analog_gain_db,
            NonnegativeFloat,
            "DigitizedWaveformConfig.analog_gain_db",
        )
        if self.bit_depth.value > 16:
            raise ValueError("bit_depth must be between 1 and 16")
        if self.input_min_mv.value >= self.input_max_mv.value:
            raise ValueError("ADC input minimum must be below maximum")
        if self.analog_gain_db.value > 40.0:
            raise ValueError("analog_gain_db must be between 0 and 40")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutConfig:
    sampling: SamplingConfig
    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(
            self.sampling,
            SamplingConfig,
            "ReadoutConfig.sampling",
        )
        _require_optional_exact(
            self.charge,
            ChargeConfig,
            "ReadoutConfig.charge",
        )
        _require_optional_exact(
            self.pure_waveform,
            PureWaveformConfig,
            "ReadoutConfig.pure_waveform",
        )
        _require_optional_exact(
            self.noise_waveform,
            NoiseWaveformConfig,
            "ReadoutConfig.noise_waveform",
        )
        _require_optional_exact(
            self.analog_waveform,
            AnalogWaveformConfig,
            "ReadoutConfig.analog_waveform",
        )
        _require_optional_exact(
            self.digitized_waveform,
            DigitizedWaveformConfig,
            "ReadoutConfig.digitized_waveform",
        )
```

These names establish ownership, not calibration defaults. Exact calibrated
values are accepted by focused scientific work orders. A sampling policy is
never inferred from hit extrema or tensor shape: doing so would discard empty
tail bins and make dense batch shape data-dependent. `ZeroNoiseConfig` selects
a real deterministic algorithm; `AnalogWaveformConfig(saturation=None)`
selects the exact linear `pure + noise` transfer. Neither exists merely to
reserve a future hierarchy. The sketch elides repetitive exact-component-type
checks; production `__post_init__` methods must reject wrong config/scalar
classes before reading their values.

`DirectCrosstalkConfig` and `DelayedCrosstalkConfig` are distinct even when
their configured delay families happen to match. Each owns its own
nonnegative Poisson mean and exact causal delay model; neither name is merely a
synonym for offset zero or a later bin. A zero mean is a draw-free identity.
An exact fixed delay of zero produces an in-bin edge after uniform
phase marginalization, while nonzero fixed and exponential delay models may
place children in later bins. The MVP deliberately omits a normal-family delay:
the earlier zero-clipped proposal introduced a calibration-sensitive prompt
atom and disproportionate numerical/validation complexity. A later calibrated
truncated-normal, clipped-normal, lognormal, tabulated, or other delay family
requires a focused scientific and API decision rather than revival of the
retired class. `afterpulse=None` disables AP. A present
`AfterpulseConfig(recovery=None)` retains AP with unit deposited charge, and a
present recovery record selects the exact exponential recovery response
documented below. None of these records carries a persistence or execution
policy.

Every physical-delay model in the readout simulation is causal and must define
a nonnegative realized delay. That is a shared scientific and preparation
invariant, not a universal silent clamp: fixed inputs are validated and
exponential laws have nonnegative support by definition. Common kernel
preflight must reject any prepared negative-offset mass or underflow category
rather than repairing an invalid model. Timing jitter is a signed displacement
and is not a physical-delay model, so this invariant does not apply to it.

The caller-facing spectral contract is only `PsdNoiseConfig`. Callers provide
one left edge per PSD bin, one separate exclusive frequency stop, and one-sided
absolute power density in `mV^2/Hz`; they do not provide FFT frequencies, FFT
amplitudes, complex coefficients, FFT length, or an implementation scale. For
left edges `f[i]`, density `S[i]` applies on `[f[i], f[i + 1])`, and the final
density applies on `[f[-1], frequency_stop_hz)`. The PSD grid is independent of
the requested record length. Request preflight uses `SamplingConfig` to
construct left-closed/right-open target integration intervals, requires source
coverage from zero through Nyquist, and integrates the piecewise-constant
supplied PSD over those intervals. TensorDSLab may use an inverse real FFT
privately to synthesize the fixed-length noise tensor, but that mechanism is
absent from the public config and does not change the input's PSD semantics.
The integrated PSD outside the deliberately suppressed target DC cell
determines the noise variance; there is no second RMS or SNR scale in
`PsdNoiseConfig` and discarded DC power is not redistributed.
Density values are interval densities, not point samples evaluated at their
left-edge coordinates.
The supplied PSD describes the effective noise at the common analog reference
plane after any intended front-end/anti-alias response. Source coverage above
Nyquist is not folded or silently aliased into band; it is outside the target
discrete process. Modeling that analog transfer explicitly would be a separate
accepted submodel.

The MVP sampling contract fixes example-local start at zero, left-closed and
right-open bins, and dropping hits outside `[0, window_stop_ps)`. Callers
therefore choose only period and count. The future bridge computes numeric bin
indices from those values before it constructs the semantic timestamp axis;
validation must report `underflow_hit_count` for normalized times below zero
and `overflow_hit_count` for times at or beyond the exclusive stop rather than
hiding either loss. These names are distinct from arithmetic overflow. Absolute
G4 origin and trigger alignment remain explicit bridge provenance, not
additional meanings hidden in `SamplingConfig`.

Picoseconds are the single numeric *time* execution unit in this architecture.
Preflight normalizes nanosecond-valued jitter, pulse, and afterpulse config
values to floating picoseconds once before the affected submodel requests RNG
values or writes its result payload.
Waveform voltage is expressed in mV, PSD frequency left edges and exclusive
stop in Hz, and absolute one-sided PSD values in `mV^2/Hz`; the PSD rebinning
boundary performs the explicit frequency/time conversion implied by
`SamplingConfig`. Kernels never combine values expressed in different units
and never parse unit-bearing coordinate strings.

### No Generic `Config(ABC)`

The rebuild does not introduce a universal `Config(ABC)` or marker base. These
records share conventions, not one useful substitutable behavior. A generic
base would encourage APIs typed as `Config`, weaken exact product/config
pairing, and add inheritance/dataclass complexity without guaranteeing correct
validation.

If a real polymorphic consumer appears later, add the narrowest protocol or
abstract type at that boundary. Serialization alone does not justify a base;
artifact codecs can operate on exact config types.

### Fixed Package-Owned RNG Keys

`RngKey` is an exact TensorCore value type containing non-boolean unsigned
32-bit `namespace` and `stream` values. It identifies one stochastic role; it
is not a seed, mutable state, algorithm selector, or counter. Maintenance 7
stores no `RngKey` in a public Config. The private
`readout/runtime/keys.py` table owns the complete fixed mapping, and
preparation places the matching key in each stochastic Runtime.

The TensorDSLab namespace is `0x54445331` (`TDS1`). Stream assignments are
append-only:

| Role | Stream |
| --- | ---: |
| white noise | `0x0000_0001` |
| PSD noise | `0x0000_0002` |
| dark count | `0x0000_0003` |
| direct crosstalk retained | `0x0000_0004` |
| direct crosstalk overflow | `0x0000_0005` |
| delayed crosstalk retained | `0x0000_0006` |
| delayed crosstalk overflow | `0x0000_0007` |
| timing jitter | `0x0000_0008` |
| afterpulse | `0x0000_0009` |
| charge smearing | `0x0000_000A` |

Afterpulse intentionally uses one key for its coupled categorical outcome and
derived count/charge ledgers. Direct and delayed crosstalk each use separate
retained and overflow keys because those are independent Poisson fields. The
fixed table is proved exact and unique. Public construction exposes no key
override, Config equality and `repr` contain no role address, and request
preparation performs no closure-wide key collision admission. The caller
selects the realization through the required `CounterRng.seed`; TensorDSLab
never re-keys a role dynamically.

### Runtime Inputs Are Not Scientific Config

The following are explicit builder/runtime inputs or derived facts:

- requested retained products;
- one immutable `CounterRng` carrying the invocation seed;
- floating dtype;
- source axes, timestamps, shape, and device;
- destination storage, future workspace, and stream;
- chunking and scheduling; and
- future persistence and IO policy.

Future typed calibration tensors are scientific inputs rather than execution
controls, but they also do not belong inside the MVP's immutable scalar config
records. This distinction leaves room for channel-aligned GPU parameters
without making ordinary scalar configuration difficult for collaborators.

Subsystem-specific named scientific presets may be classmethods such as
`ReadoutConfig.darkside20k_tpc_nominal()` and
`ReadoutConfig.darkside20k_veto_nominal()` only after each exact calibration is
reviewed. A single nominal preset must not blur the distinct response families.
Do not add loose default constants or a scientifically unqualified `default()`.

## Product Runtime Actions

The merged Maintenance 4 implementation gives every generated product three
explicit, independently tested actions under its non-exported `runtime/`
package:

```text
Config plus preflight facts
  -> prepare_<product>(...)
  -> <Product>Runtime
  -> produce_<product>(...)
  -> <Product>
  -> validate_<product>(...)
  -> next descendant or final retention
```

`Photoelectrons` is already-produced truth, so it owns only
`validate_photoelectrons(...)`. No placeholder Config, Runtime, preparer, or
producer is created for it.

Each `<Product>Runtime` is a final, frozen, slotted, unexported data carrier.
It has no common base, methods, Config, semantic product, collection, mutable
cache, or hidden movement. It stores only prepared tensors and static Python
facts needed by one execution path, request-wide RNG-key validation, or the
product's immediate postcondition. The exact ProductRuntime types are not
interchangeable.

Product preparers own config interpretation, unit conversion, probability and
response-law construction, representability proofs, device scalar/kernel
materialization, and other predictable scientific/contextual work. A producer
receives only its exact prerequisite product or products, its exact Runtime,
and `CounterRng` when stochastic-capable. It never receives a Config, repeats
sampling/axis discovery, imports its validator, or performs the completed
product's deep publication scan.

Production owns the actual tensor/RNG computation, genuine draw-dependent
control flow, unavoidable dynamic count/ledger/overflow guards, operation-
mechanical alias checks, and final semantic-field construction. Validators
then own the completed product's value domain and its cheap publication
relationships to the named prerequisite products: exact axes, shape, dtype,
device, freshness, and any product-specific range such as the prepared ADC
maximum code. Validators are read-only and never repair, cast, normalize,
move, reconstruct, or write.

The clean cross-module names are intentional: `prepare_charge`,
`produce_charge`, `validate_charge`, and `simulate_dark_counts`, for example.
Privacy follows the supported facade exports rather than a leading underscore.
Runtime package roots remain empty and do not re-export these names; deep
imports are possible Python implementation access, not supported API. Tiny
genuinely module-local mathematical helpers may retain a leading underscore.

The closed Stage 6/7 `_produce.py`, `_produce_*`, `_prepare_*`, `_simulate_*`,
`*Plan`, and `_requirements.py` paths remain historical facts only. The merged
Maintenance 4 implementation removed the former private paths and cross-module
names without aliases or compatibility shims.

Only stochastic-capable product producers receive `rng`. The deterministic
pure, analog, and digitized producers do not accept it. A stochastic-capable
producer may execute a draw-free exact-zero or disabled path without requesting
any RNG value; the parameter expresses capability, not guaranteed use.
Deterministic preparation helpers likewise receive no RNG.

The two pointwise waveform-tail producers own their product arithmetic directly.
Do not add `_apply_analog_saturation(...)`, `_digitize(...)`, or another
one-line Python wrapper merely to rename either expression. The initial
implementation uses ordinary eager Torch expressions. A later measured
optimization may fuse either expression without adding another Python API
layer.

After complete preparation, `produce_analog_waveform(...)` evaluates
one elementwise product expression:

```text
analog[i] = clamp(
    pure[i] + noise[i],
    analog_minimum,
    analog_maximum,
)
```

Either bound may be absent. When both are absent, the expression is simply
addition. The producer returns one new `AnalogWaveform` with guaranteed fresh
storage independent of `pure.tensor` and `noise.tensor`. The functional stage
makes no claim about eager or backend-created target-sized intermediates.

`prepare_digitized_waveform(...)` computes its scalar transfer constants once
during preflight, and `produce_digitized_waveform(...)` consumes those prepared
operands:

```text
maximum_code = 2**bit_depth - 1
gain = 10**(analog_gain_db / 20)
span = input_max_mv - input_min_mv
slope = gain * maximum_code / span
intercept = -input_min_mv * maximum_code / span
lower_input_mv = input_min_mv / gain
upper_input_mv = input_max_mv / gain
```

It then evaluates one endpoint-guarded affine product expression:

```text
interior[i] = clamp(
    analog[i] * slope + intercept,
    0,
    maximum_code,
)

code_float[i] =
    0,                         if analog[i] <= lower_input_mv
    maximum_code,              if analog[i] >= upper_input_mv
    interior[i],               otherwise

digitized[i] = int32(code_float[i])
```

The guards compare directly against dtype-rounded thresholds in the pre-gain
analog domain and make the inclusive ADC endpoints exact even when affine
rounding would place the upper endpoint just below `maximum_code`. The clamp
and endpoint selection occur in floating point before conversion. Because
every selected value is nonnegative, float-to-`torch.int32` conversion
implements the accepted open-interior truncation rule without an explicit
`torch.trunc(...)` step. The producer returns one new `DigitizedWaveform` with
guaranteed fresh storage independent of `analog.tensor`. The functional stage
does not classify temporary storage created by eager Torch or its backend.

The normal materialized waveform tail therefore has exactly two semantic
product steps:

```text
PureWaveform + NoiseWaveform
  -> produce_analog_waveform(...)
  -> AnalogWaveform
  -> produce_digitized_waveform(...)
  -> DigitizedWaveform
```

This remains true when only `DigitizedWaveform` is retained: the analog product
is computed once as a private prerequisite and is then allowed to become
unreachable. Do not fuse across the `AnalogWaveform` product boundary in the
MVP. Such fusion would require a separate proof that requesting or retaining
`AnalogWaveform` cannot change digitized values, product execution, autograd,
or lifetime behavior.

`produce_noise_waveform(...)` uses `Photoelectrons` only as the authoritative
axes/device/shape reference; it does not read PE counts as a noise input.

Private Charge effects use the clean `simulate_dark_counts(...)`,
`simulate_timing_jitter(...)`, `simulate_correlated_avalanches(...)`, and
`simulate_charge_smearing(...)` names inside the non-exported runtime package.
They consume exact effect-specific facts already contained by `ChargeRuntime`;
they must not repeat dark-mean, timing-kernel,
correlated-plan, ledger-envelope, or smearing-envelope preparation after
product execution begins.

Only operations whose numerical behavior depends on sample timing receive
prepared sampling facts. Operations that shift values along the sample axis
also receive the already-resolved numeric `sample_dimension`; hot-path code
does not look up timestamp strings. The coupled cascade receives the selected
floating dtype because it constructs the S1/S2 ledgers. No subfunction receives
`ReadoutConfig` merely to reach one nested value. Every stochastic leaf receives
the same immutable `CounterRng` invocation and uses the exact `RngKey` owned by
its leaf config. Draw-free identities make no RNG call. Product producers never
share a mutable sequential stream between leaves. `produce_charge(...)`
remains the sole private typed Charge constructor; Stage 7 added the public
`simulate_readout(..., products=[Charge], ...)` request path.

## Public Builder

Maintenance 5 keeps the public `simulate_readout(...)` signature but removes
the sampling member from `ReadoutConfig`. Sampling preflight below is
source-derived; any sketch passing or comparing a `SamplingConfig` is
historical.

The target signature is:

```python
def simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype = torch.float32,
) -> ReadoutCollection:
    ...
```

The public surface requires one TensorCore `CounterRng` instance satisfying
the accepted public contract, even when the effective request is
deterministic. There is no simultaneous `seed=`
parameter: callers choose an accepted algorithm and invocation seed when they
construct the RNG. A deterministic closure requests no values from it. The
MVP exposes no `torch.Generator`, mutable generator state, ambient global RNG,
or TensorDSLab-specific RNG wrapper.

`readout.runtime.prepare` owns one private final frozen `ReadoutRuntime` with
one optional exact ProductRuntime per generated product. Optional Runtime
presence is the execution closure; duplicated `need_*` booleans are not kept.
Requested retention remains a separate typed set returned with the Runtime.
Neither value is a public graph, registry, config, TensorCore object,
collection member, workspace, or durable artifact. `prepare_readout(...)`
consumes the request iterable once, derives the fixed typed closure, prepares
every required product Runtime, and admits the required `CounterRng`. It does
not validate a caller-key relationship because role keys are fixed package
policy.

Conceptual orchestration:

```python
requested, runtime = prepare_readout(
    photoelectrons,
    products=products,  # consumed exactly once here
    config=config,
    rng=rng,
    floating_dtype=floating_dtype,
)

charge = None
if runtime.charge is not None:
    charge = produce_charge(
        photoelectrons,
        runtime=runtime.charge,
        rng=rng,
    )
    validate_charge(
        charge,
        source=photoelectrons,
        runtime=runtime.charge,
    )

pure = None
if runtime.pure_waveform is not None:
    pure = produce_pure_waveform(
        require_product(charge, Charge),
        runtime=runtime.pure_waveform,
    )
    validate_pure_waveform(pure, source=require_product(charge, Charge))

noise = None
if runtime.noise_waveform is not None:
    noise = produce_noise_waveform(
        photoelectrons,
        runtime=runtime.noise_waveform,
        rng=rng,
    )
    validate_noise_waveform(
        noise,
        source=photoelectrons,
        runtime=runtime.noise_waveform,
    )

analog = None
if runtime.analog_waveform is not None:
    analog = produce_analog_waveform(
        require_product(pure, PureWaveform),
        require_product(noise, NoiseWaveform),
        runtime=runtime.analog_waveform,
    )
    validate_analog_waveform(
        analog,
        pure=require_product(pure, PureWaveform),
        noise=require_product(noise, NoiseWaveform),
    )

digitized = None
if runtime.digitized_waveform is not None:
    digitized = produce_digitized_waveform(
        require_product(analog, AnalogWaveform),
        runtime=runtime.digitized_waveform,
    )
    validate_digitized_waveform(
        digitized,
        source=require_product(analog, AnalogWaveform),
        maximum_code=runtime.digitized_waveform.maximum_code,
    )

retained = tuple(
    value
    for value in (
        photoelectrons,
        charge,
        pure,
        noise,
        analog,
        digitized,
    )
    if value is not None and type(value) in requested
)
return ReadoutCollection(fields=retained)
```

The exact implementation may use local assertions or a narrow internal helper
for static type narrowing. The invariant is that each required producer
receives its exact prepared Runtime, executes at most once, is validated
immediately, and is never invoked before the complete `ReadoutRuntime` exists.

The fixed local tuple gives equivalent request sets the same mechanical
construction order while remaining nonsemantic to `ReadoutCollection`.
Caller iterable order never affects field values, `field_types`, mechanical
mapping iteration, or the documented collection contract. The tuple is private
assembly code, not a public canonical field sequence or registry.

Whole-request preparation completes before the first RNG raw-word request,
product-producer invocation, or semantic-product/output write. It validates:

- source deep-value validity;
- product request type, uniqueness, and nonemptiness;
- exact `SamplingConfig` validity and agreement between its period/count policy
  and the validated source `SampleAxis`, without regenerating all coordinates;
- every config required by the transitive closure;
- timestamp grammar and timing suitability for enabled operations;
- a CPU or CUDA source device and the supported tensor layout;
- `floating_dtype` is exactly `torch.float32` or `torch.float64` when the
  closure generates a floating product;
- analog saturation bounds and digitizer `maximum_code`, `gain`, `span`,
  `slope`, `intercept`, and pre-gain endpoint thresholds are valid,
  representable, and noncollapsed in the selected execution dtype before
  either waveform-tail producer launches;
- every statically known discrete-control probability and Poisson mean is
  valid in binary64, every enabled Poisson address/cap is representable, and
  each dynamically realized crosstalk rate field will be checked before its
  corresponding sampler requests words or writes its result;
- nominal `CounterRng` membership on every request; and
- exact `RngKey` fields and no duplicate key assigned to distinct stochastic
  roles in the requested transitive closure.

Preparation may create ephemeral unexposed scalar/control tensors and perform
read-only reductions required by accepted deep validation. Those reductions
may synchronize a CUDA current stream through scalar extraction. They do not
mutate the source, expose a semantic result, or begin the scientific product
chain. TensorCore exposes no non-consuming concrete-RNG capability query, so
Stage 7 neither inspects protected hooks nor issues a dummy draw. A real custom
RNG/backend incompatibility is detected at the first genuine distribution call
and belongs to the dynamic execution-failure boundary.

`ReadoutConfig(sampling=sampling)` is a valid uniform config argument for a
truth-only request. That deterministic closure checks nominal `CounterRng`
membership but neither queries the RNG nor validates `floating_dtype`, because
it generates no floating product. Irrelevant product configs and runtime
controls are neither consumed nor allowed to perturb common product values.

A request such as this fails before any RNG request, product-producer
invocation, or semantic-output write:

```python
simulate_readout(
    photoelectrons,
    products=[DigitizedWaveform],
    config=config_without_digitizer,
    rng=Threefry4x32(seed=1234),
)
# ValueError: DigitizedWaveform requires digitization configuration
```

A closure with two different stochastic roles assigned the same key also fails
before any RNG request, product-producer invocation, or semantic-output write:

```python
simulate_readout(
    photoelectrons,
    products=[Charge, NoiseWaveform],
    config=config_with_duplicate_role_keys,
    rng=Threefry4x32(seed=1234),
)
# ValueError: stochastic roles require distinct RngKey values
```

The builder performs no IO, loading, persistence, DAG scheduling, or implicit
movement/cast of an existing input field. Generated products use their declared
output dtypes. Allocator failure, TensorCore sampler exhaustion, dynamically
realized count/rate/ledger overflow, invalid generated-payload postconditions,
and internal producer invariant failure remain dynamic. No partial
`ReadoutCollection` or semantic field escapes a failed call, but private
allocations and already-computed local prerequisite fields have no rollback
promise.

## Scientific Chain

The selected rebuild computation is:

```text
`Photoelectrons` payload
  -> optional dark-count avalanche seeds
  -> optional private timing redistribution
  -> optional fixed-K coupled correlated-avalanche simulation
       -> integer count frontier for branching
       -> floating S1 deposited-charge ledger
       -> floating S2 charge-square-sum ledger
  -> optional terminal charge smearing from S1 and S2
  -> completed `Charge` product
  -> PureWaveform

truth field axes/device/shape + NoiseWaveformConfig
  -> NoiseWaveform

PureWaveform + NoiseWaveform
  -> AnalogWaveform
  -> optional DigitizedWaveform
```

### Timing Jitter Inside Charge Simulation

When its block executes, timing jitter redistributes the then-current private
primary-avalanche working counts after any effective dark-count block. It
therefore affects photon-origin truth seeds and any dark-count seeds that are
present without mutating or relabeling the public truth `Photoelectrons` field.
For source bin `s`, target bin `t`, shift
`k = t - s`, sample period `T`, latent source-bin phase
`U ~ Uniform([0, T))`, and jitter `J ~ Normal(0, sigma)`, the latent source
time is `s * T + U`; `s * T` is the source bin's left edge. Preflight first
normalizes `T` and `sigma` to floating picoseconds:

```text
target bin t = [t*T, (t + 1)*T)
p(target=t) = P(t*T <= s*T + U + J < (t + 1)*T)
            = P(k*T <= U + J < (k + 1)*T)
```

The latent phase is required because dense truth counts no longer retain each
PE's sub-bin time. The latent phase and ideal Gaussian are integrated
analytically during preflight; runtime draws neither value per PE and does not
use the Stage 5 Box-Muller primitive for jitter. Define the standard-normal CDF
and PDF as `Phi` and `phi`, and:

```text
X = U + J
H(z) = z * Phi(z) + phi(z)

F_X(x)
    = P(X < x)
    = (sigma / T)
      * (H(x / sigma) - H((x - T) / sigma))

q[k]
    = P(k*T <= X < (k + 1)*T)
    = F_X((k + 1)*T) - F_X(k*T)
```

The scientific closure assigns every avalanche one independent pair `(U, J)`;
the two values in a pair are mutually independent, and pairs are identically
distributed and independent across avalanches. That IID closure is what makes
one source cell's aggregate destination counts multinomial with probabilities
derived from `q`.

This formula is the scientific target for `sigma > 0`; it also gives the
symmetry `q[-k] = q[k]`. The first implementation supports the explicit
numerical domain:

```text
r = sigma / T
2**-52 <= r <= 64
2 <= S <= 8192
S * N <= 2**63
```

Here `S` is the sample count and `N` is the complete count-grid `numel`.
`sigma == 0` remains the separate exact identity. A positive ratio outside the
supported interval, an oversized timing window, or nonfinite preparation fails
before the timing-jitter submodel requests RNG values or writes its result
payload; it is not rounded into the identity or another supported law.
Preflight first forms `T_ps = float(sample_period_ps)`, checks
the unconverted config value against
`(T_ps * 2**-52) * 1e-3 <= sigma_ns <= (T_ps * 64) * 1e-3`, and only then
forms `sigma_ps = sigma_ns * 1e3`, divides `r = sigma_ps / T_ps`, and rechecks
the represented ratio. The accepted upper comparison proves that the later
unit conversion cannot overflow. Extending either bound requires a new
numerical sweep and a focused Design change.

Direct second differences of `H` are not the accepted evaluator: they suffer
catastrophic cancellation and can produce negative represented tail cells.
Preflight instead prepares the one-sided tail lattice

```text
G(z) = phi(z) - z * (1 - Phi(z)) = H(-z)

L[m]
    = P(X < -m*T)
    = P(offset >= m + 1)
    = r * (G(m / r) - G((m + 1) / r))
```

for every `m = 0, ..., S - 1`. It evaluates `log(G)` as follows:

```text
log_phi(z) = -0.5*z*z - 0.5*log(2*pi)

z == 0:
    log_G(z) = log_phi(0)

0 < z < 8:
    log_G(z) = log(
        phi(z) - 0.5*z*erfc(z / sqrt(2))
    )

z >= 8:
    A(z) = z**-2 - 3*z**-4 + 15*z**-6 - 105*z**-8 + ...
    log_G(z) = log_phi(z) + log(A(z))
```

The asymptotic branch starts with `term = total = z**-2`. For candidate terms
`n = 2, ..., 100`, it forms
`next = -(2*n - 1) * term / z**2`, stops *before* the first candidate whose
absolute value does not decrease, and also stops when adding a candidate does
not change the binary64 total. Otherwise it adds the candidate and advances
`term = next`. The final total must be finite and strictly positive. These
rules freeze evaluation machinery, not a probability cutoff.

With `ell[m] = log(L[m])`, each tail difference uses the stable identity

```text
ell[m]
    = log(r) + log_G(m / r)
      + log(-expm1(log_G((m + 1) / r) - log_G(m / r)))

q[0]
    = erf(1 / (sqrt(2)*r))
      + r*sqrt(2/pi)*expm1(-1 / (2*r*r))

q[k]
    = exp(
        ell[k - 1]
        + log(-expm1(ell[k] - ell[k - 1]))
      ),  k >= 1
```

Negative offsets reuse `q[abs(k)]`; they are never independently evaluated.
This gives exact represented offset symmetry. Natural binary64 underflow may
make a far evaluated tail or category exact zero, but no destination is omitted
from preparation because of a chosen tail radius.

Preflight requires `L` to be finite, in `[0, 0.5]`, and nonincreasing and every
`q` to be finite and in `[0, 1]` exactly. It never admits a small negative
value through a tolerance. Preflight uses the fixed absolute tolerance
`1e-12` for the central identity `q[0] + 2*L[0] = 1` and its analytic tail/
telescoping diagnostics; independent high-precision category-oracle
comparisons use the same bound in validation. A diagnostic retained-plus-drop
residual, formed with `math.fsum` over the retained categories plus the two
analytic tails, may have either sign within that tolerance; it is never used to
clip, normalize, or move probability mass. Validation additionally
requires the complete represented source law to differ from its high-precision
ideal law by at most `1e-11` in L1 distance. These are private algorithm
contracts, not caller-configurable accuracy knobs. Arbitrary precision is
validation-only fixture/oracle tooling; the production evaluator has no SciPy,
`mpmath`, or runtime arbitrary-precision dependency.

For one source bin `s`, retained target categories are the destination bins
`t = 0, ..., S - 1` in increasing order, with ideal probabilities
`p[t] = q[t - s]`. This is equivalently increasing signed-offset order for
that source. The analytic combined out-of-window mass is:

```text
p_drop[s]
    = P(X < -s*T) + P(X >= (S - s)*T)
    = L[s] + L[S - s - 1]
```

The runtime never constructs this mass as `1 - sum(p)`. Nor does it repeatedly
subtract represented category probabilities from a running value initially
equal to one. Both forms corrupt very small tails. Instead each conditional
binomial receives a prepared success mass `A` and the mass `B` of every later
category, including the final drop. For `k = t - s`:

```text
k = -m < 0:  A = q[m]   B = 1 - L[m - 1] + L[s]
k = 0:       A = q[0]   B = L[s] + L[0]
k > 0:       A = q[k]   B = L[s] + L[k]
```

The displayed grouping is normative. For a nondegenerate step, preflight
requires finite `A` and `B` in `[0, 1]`, forms finite positive
`total = A + B`, then forms `p_star = min(A, B) / total` and requires it in
`[0, 0.5]`. An explicit complement flag is true exactly when `B < A`. The
binomial core samples the smaller side and applies the complement only after
acceptance. It never recovers that side by forming `1 - p` from a rounded near-
one conditional probability.
`A == 0` assigns zero and draws nothing; `B == 0` assigns the complete remaining
integer count and draws nothing. If both are represented zero, the remaining
integer count must already be zero. A violation fails rather than inventing
probability mass.

For validation, the represented conditional law is reconstructed in high
precision from those exact binary64 `A`, `B`, complement, and division results.
Its categories, final drop, moments, and covariance are compared with the
ideal law. The analytic two-tail value above remains a diagnostic; malformed
or insufficiently accurate preparation fails rather than clipping or silently
renormalizing probabilities. Every in-window destination is evaluated even if
its represented probability later rounds to exact zero.

Runtime samples this `S + 1` category law through the accepted aggregate
conditional-binomial factorization. Category `c = t` uses
`logical_position = t * N + source_flat_position`, where `N` is the complete
count-grid `numel`. The combined drop category is last, receives the exact
remaining count, and consumes no draw. Preflight requires the complete
retained-category lattice to satisfy the domain above. The exact append-only
package-owned timing-jitter key is
`RngKey(namespace=0x54445331, stream=8)`;
every timing
conditional is an aggregate cell draw with `source_quantum = 0`. A represented
zero-probability destination may skip physical work without shifting any later
address. The implementation therefore redistributes dense integer counts
directly and never materializes a per-PE table or per-PE normal field.

Accepted policies:

- `sigma_ns == 0` is an exact logical identity and consumes no jitter draws;
- out-of-window shifted truth and dark-count seeds are dropped;
- retained destination counts plus the explicit dropped count equal every
  source count exactly;
- every possibly in-window destination is evaluated as its own category, with
  no arbitrary tail cutoff;
- the truth `Photoelectrons` tensor is never mutated or replaced.

Let `R = N / S` be the number of example-channel waveform rows. The
correctness-first reference may require `S` conditional-binomial category
steps for each of `N = R * S` source cells: `O(R * S**2)` work. That is
accepted for the first implementation. A later measured
optimization may exploit symmetry, sparsity, or a separately approved error
budget, but it must not silently truncate Gaussian mass or change the prepared
law.

Moving jitter into charge simulation changes the old public timing-transform
boundary. The synchronized parity document compares truth `Photoelectrons` to
the private jitter diagnostic or to requested `Charge`; it does not claim that
jitter produces a new public `Photoelectrons` value.

### Charge Response

The private order is:

```python
charge = photoelectrons.tensor
charge_square_sum: torch.Tensor | None = None

if (
    config.dark_count is not None
    and config.dark_count.rate_hz.value != 0.0
):
    charge = simulate_dark_counts(charge)

if (
    config.timing_jitter is not None
    and config.timing_jitter.sigma_ns.value != 0.0
):
    charge = simulate_timing_jitter(charge)

if config.correlated_avalanches is not None:
    correlated = simulate_correlated_avalanches(charge)
    charge = correlated.S1
    charge_square_sum = correlated.S2

if (
    config.smearing is not None
    and config.smearing.relative_sigma.value != 0.0
):
    charge = charge.to(dtype=floating_dtype)
    charge = simulate_charge_smearing(
        charge,
        charge if charge_square_sum is None else charge_square_sum,
    )

return Charge(
    tensor=charge.to(dtype=floating_dtype),
    axes=photoelectrons.axes,
)
```

Lowercase `charge` remains a `torch.Tensor` throughout `produce_charge(...)`.
It is the single evolving payload, not a `Charge` instance and not a durable
product identity. Each entered optional block completely replaces that tensor
with its stage result. When an enclosing condition is false, the block performs
no call or assignment and `charge` remains the exact preceding tensor. The
correlated block alone needs a private `_CorrelatedAvalancheResult`; it replaces
`charge` with the result's floating `S1` ledger and retains `S2` only as
`charge_square_sum` for a possible later smearing block. The function returns
the completed uppercase `Charge` product directly, so one local name never
changes between tensor and product types.

Valid chains therefore include, among others:

```text
truth -> dark counts -> timing jitter -> correlated avalanches -> smearing -> Charge
truth -> smearing -> Charge
truth -> timing jitter -> correlated avalanches -> Charge
truth -> dark counts -> smearing -> Charge
truth -> Charge
```

If smearing runs without a preceding correlated stage, every root in `charge`
has unit response, so one floating conversion supplies both `S1` and `S2`
without constructing a correlated result. If smearing is absent, the terminal
product construction performs the required floating conversion directly. A
configured smearing model draws only after every selected correlated generation
has updated both ledgers.

Dark counts use independent per-cell Poisson counts:

```text
lambda_per_cell = rate_hz * sampling.sample_period_ps.value * 1e-12
dark_count ~ Poisson(lambda_per_cell)
```

A zero rate is an exact zero contribution: the dark block is skipped, no
`dark` variable is bound, and no dark-count draw is consumed.
Every positive cell calls TensorCore's public Poisson sampler selected in
[Poisson Count Sampling](#poisson-count-sampling). The dark-count lattice is
noniterative: `logical_position = source_flat_position`,
`source_quantum = 0`, and the operation uses its dedicated dark-count stream.
The scalar mean is prepared in binary64. A negative, nonfinite, or greater-than-
`1e8` mean fails before this sampler requests a word or starts its output write.

For an unbounded homogeneous Poisson dark process, independent timing
displacement preserves the homogeneous law. The finite MVP window is more
subtle: it generates seeds only inside the window and drops values jittered
outside, so there is no compensating influx from dark events just beyond an
edge. The MVP accepts and measures this boundary truncation rather than
claiming exact edge invariance. A later stationary-boundary treatment may use
a configured halo and crop, but it is not silently inferred here.

This order improves the isolated dark-count-plus-jitter comparison with the IV
donor when gate, latent phase, binning, and drop policies agree. It does not
make the full chain equivalent. IV-DSLab creates recursive correlated
avalanches before its later independent PE timing operation, whereas the
rebuild jitters truth and dark seeds first and applies its prepared causal edge
kernels during the coupled cascade. Parent-child timing covariance therefore
remains an intentional parity difference.

### Fixed-Generation Correlated-Avalanche Baseline

Status: sole active correlated-avalanche algorithmic baseline for the rebuild.
It remains Design-only and does not dispatch implementation or add another
public API. It is the only avalanche algorithm available to a rebuild work
order and cannot be replaced without a new user-authorized TensorDSLab Design
decision.

The selected model is a synchronous, fixed-generation, unit-count
branching process with separate floating deposited-charge response over the
complete dense post-binned grid. Generation zero is the private integer seed
grid after dark-count addition and timing jitter. A caller-selected nonnegative
`maximum_generations = K` gives the number of offspring generations evaluated
after those roots:

```text
K = 0  -> roots only
K = 1  -> roots plus their direct children
K = 2  -> roots, children, and grandchildren
```

`K` is scientific configuration because changing it changes the returned
distribution. The algorithm does not infer it from an extinction
test, tolerance, finite-window bound, or hardware execution plan. A smarter
selection or exact-termination policy is deferred. Implementations may avoid
physical work for a provably zero frontier, but doing so must not change the
fixed-`K` semantics.

Each generation is organized around two collaborator-facing questions:

1. **How many avalanches does the current frontier produce?**
2. **Which sample bins receive those avalanches?**

The separate DiCT, DeCT, and AP draws answer both questions. Their retained
children are then summed into the next frozen frontier; their right-overflow
children are recorded but do not continue.

`K=0` performs no correlated-mechanism draw even when mechanism configs are
present. Such configs still validate structurally, but they do not make an
otherwise deterministic request query its required `CounterRng`. Likewise, a
zero DiCT/DeCT mean or zero AP probability is a draw-free identity for that
mechanism.
Contextual delay-kernel preparation follows the same rule: `K=0` prepares no
CT or AP delay/recovery kernel; a zero CT mean prepares no kernel for that CT
mode; and zero AP probability prepares neither the AP delay kernel nor its
optional recovery response. Only the already-constructed config records are
structurally valid on those unused paths. Sampling-dependent ratio,
sample-count, tail, and recovery numerical gates apply only when the
corresponding mechanism can execute. This skip is resolved during complete
public preparation before any RNG request, product-producer invocation, or
semantic-output write.

The exact `CorrelatedAvalancheConfig` and mechanism records are defined in the
configuration section above. Structural `None` disables a mechanism. With all
three absent, the operation is the exact identity for every `K` and needs no
generation storage or stochastic draw. Its logical final frontier is the root
grid when `K = 0` and zero when `K >= 1`, even when an optimized identity path
materializes neither value.

The recursive state distinguishes avalanche multiplicity from deposited
charge:

```text
F[g, t]       : int64 avalanche count in generation g and sample t
S1[t]         : floating accumulated pre-smearing PE-equivalent charge
S2[t]         : floating accumulated sum of squared response weights
```

Every successful primary, dark-count, DiCT, DeCT, or AP avalanche contributes
exactly one integer count to its generation frontier. `F[g]`, never `S1` or
`S2`, determines every enabled offspring law.

`S1`, `S2`, AP charge diagnostics, and the terminal product use the exact
`floating_dtype` selected for `Charge`; no physical charge accumulator silently
widens or narrows. Discrete stochastic control is deliberately separate. Every
prepared categorical probability, Poisson mean/rate field, and binomial or
Poisson control calculation uses binary64, represented by `torch.float64` on
the execution device. Config-derived CDF/PMF preparation likewise uses host
binary64 before one checked device materialization. This makes the integer
avalanche realization independent of whether the caller requests float32 or
float64 `Charge` on the same unchanged numerical execution stack. It does not
make the floating ledgers or final products bitwise equal across those dtypes.

Preflight and each dynamically constructed generation-rate boundary prove the
relevant probabilities and rates finite, nonnegative, and inside their selected
sampler domains before that draw begins. A nonfinite ledger, unsupported rate,
sampler exhaustion, or checked `int64` count overflow is a hard algorithm
failure and never a valid partial `Charge` result.

The response model makes this explicit microcell assumption:

- every DiCT or DeCT child triggers a microcell distinct from its parent that
  has not previously fired in the modeled cascade and is fully recovered, so
  the child deposits exactly one PE-equivalent charge; and
- every AP child retriggers its parent microcell, so it still contributes one
  avalanche count but deposits delay-dependent recovery-weighted charge.

The CT rule is a low-occupancy, effectively unlimited-fresh-cell
approximation. Crosstalk collisions, finite microcell exhaustion, or a CT seed
landing on a previously fired or recovering destination require microcell
identity and recovery state, which the dense post-binned frontier does not
carry and which remain outside this initial model. Recovery-dependent source
emission is also excluded: even an AP-born avalanche has the same future
offspring law as every other unit-count avalanche.

This is **unmarked recursion with recovery-weighted AP deposited charge**, not
recovery-dependent marked recursion. The recovery weight is consumed into the
floating charge accumulator and is not carried in `F[g + 1]` or used to scale
that avalanche's later DiCT, DeCT, or AP production. Every root, dark count,
DiCT, DeCT, and AP parent therefore uses the same recovery-independent DiCT
mean, DeCT mean, and AP fire probability. If recovery classes are ever used to
refine within-offset AP charge heterogeneity, they are transient sampling
categories that collapse immediately into the ordinary unmarked `F[g + 1]`;
they are not recursive state. The unit-count rule is what permits the three
integer contribution tensors to merge into one next-generation parent
frontier.

`F[g]` is frozen for the complete generation update. No child produced by one
mechanism may feed DiCT, DeCT, or AP again until `F[g + 1]` is complete. This
also applies to a zero-offset child that occupies its parent's sample bin:
sample offset zero does not mean genealogical generation zero.

For that current frontier, the three mechanisms are conditionally independent
and are drawn separately. DiCT and DeCT are two physical or calibrated modes of
correlated crosstalk; they are not synonyms for same-bin and later-bin
children. Each mode `m` in `{direct, delayed}` has its own fixed mean offspring
count `lambda_m` and its own exact `FixedDelayConfig` or
`ExponentialDelayConfig`. These are
sampling-independent causal physical-delay models. Preflight combines each
one with the exact `SamplingConfig` to prepare an integer-offset PMF
`q_m[d; sampling]`.

```text
FixedDelayConfig:       Delta_m = delay_ns
ExponentialDelayConfig: Delta_m ~ Exponential(mean_delay_ns)
```

`NormalDelayConfig` is not an accepted MVP family. The earlier zero-clipped
proposal is retired rather than left as an executable-looking dormant option.
Its negative latent tail would have become a calibration-sensitive prompt atom,
while a truncated, folded, or tabulated alternative would encode a different
law. Any later distributed family beyond the ordinary exponential requires a
new calibrated scientific decision and a new explicit config type.

The binned phase policy is independent per parent-child edge. Conceptually,
every realized CT child edge or fired AP edge receives a fresh
`U_edge ~ Uniform([0, T))`. It is independent of every sibling edge, mechanism,
and generation, is integrated into the prepared category law, and is never
stored or inherited. For sample period `T` and physical edge delay `Delta_m`:

```text
q_m[d; sampling]
    = P(d * T <= U_edge + Delta_m < (d + 1) * T),  d in Z_{≥0}
```

#### Fixed-Delay Preparation

For the represented fixed physical delay `D`, write:

```text
D / T = n + f
n = floor(D / T)
0 <= f < 1

q_fixed[n]     = 1 - f
q_fixed[n + 1] = f
q_fixed[d]     = 0 otherwise
```

When `f == 0`, only offset `n` has unit mass. Immediately below an exact
multiple `m*T`, the two offsets are `m - 1` and `m`; immediately above it they
are `m` and `m + 1`. Preparation never epsilon-snaps a represented delay across
that boundary.

`FixedDelayConfig.delay_ns` remains any finite nonnegative value. Preflight
first compares its exact binary64 integer ratio against the exact rational
window stop in nanoseconds: for
`delay_num, delay_den = delay_ns.as_integer_ratio()`, the all-overflow test is
`delay_num * 1000 >= delay_den * window_stop_ps`. A delay satisfying that test
becomes an analytic all-overflow plan before unit multiplication or index
formation. Otherwise it keeps the picosecond conversion exact in integer-ratio
arithmetic and performs:

```text
n, remainder = divmod(
    delay_num * 1000,
    delay_den * sample_period_ps,
)
f = float(Fraction(remainder, delay_den * sample_period_ps))
```

The exact integer `divmod` prevents either a rounded unit multiplication or a
rounded floating quotient from crossing an offset boundary. The only floating
rounding is the final fractional mass. For a nonzero exact remainder, both
represented masses must remain strictly between zero and one; if that
conversion makes either side deterministic, preflight rejects it instead of
snapping. The two masses must satisfy exact represented
`math.fsum((1.0 - f, f)) == 1.0`. This exact two-point construction uses no PMF
tolerance. There is no arbitrary maximum fixed delay: every finite nonnegative
value is either representable inside the window or analytically all-overflow.

For source bin `t`, let `R = S - t` be the number of bins from that source
through the right edge. Its exact source-relative overflow probability is:

```text
n >= R        -> 1
n + 1 == R    -> f
otherwise     -> 0
```

Code compares `n` with `R`; it never forms `t + n` in signed tensor arithmetic.
Fixed delay owns no RNG stream and draws no latent phase. Its prepared masses
only thin the existing retained and overflow CT Poisson rate fields. An exact
single-bin retained plan still uses the retained CT stream for a positive
rate, an all-overflow plan uses only the overflow stream, and an exact zero
rate requests no words.

#### Exponential-Delay Preparation

The same prepared exponential kernel is used by
`ExponentialDelayConfig.mean_delay_ns` for either CT mode and by
`AfterpulseConfig.mean_delay_ns`. Let:

```text
r = mean_delay / T
x = 1 / r
a = 1 - exp(-x) = -expm1(-x)
C = a / x
```

The first implementation supports the explicit domain:

```text
2**-52 <= r <= 2**52
2 <= S <= 8192
```

A mean outside this ratio interval or an active exponential preparation with
more than 8192 samples fails before the affected delay submodel requests RNG
values or writes its result payload rather than being rounded into a prompt or
infinite-delay law. An unused kernel is skipped under the
identity rules above and receives no contextual numerical gate. The
sample-count bound is an evidenced MVP preparation limit, not a property of
the exponential distribution; extending it requires a focused numerical
sweep. Active preflight uses the same overflow-safe unit pattern as timing
jitter: it bounds the unconverted nanosecond value against the exact
picosecond period first, converts to floating picoseconds only after the
accepted bounds prove the multiplication finite, forms `r`, and rechecks the
represented ratio. Concretely, for
`mean_num, mean_den = mean_delay_ns.as_integer_ratio()`, the exact host checks
are
`mean_num * 1000 * 2**52 >= mean_den * sample_period_ps` and
`mean_num * 1000 <= mean_den * sample_period_ps * 2**52` before forming
`mean_delay_ps = mean_delay_ns * 1000.0`.

For integer offset `d`, define the analytic right tail
`R[d] = P(offset >= d)`. The complete phase-marginalized law is:

```text
R[0] = 1
R[d] = C * exp(-(d - 1)*x),                 d >= 1

q_exp[0] = 1 - C
q_exp[d] = R[d] * a,                        d >= 1
```

Runtime preparation does not obtain positive-offset masses by subtracting two
rounded tails. It computes `a` with `expm1`, then uses
`log_C = log(C)`, `log_a = log(a)`,
`log_R[d] = log_C - (d - 1)*x`, and
`log_q[d] = log_R[d] + log_a`; `exp` rounds each final requested value once.
Natural IEEE binary64 underflow may make an evaluated far tail or category
exact zero. There is no explicit delay cutoff, clipping, residual assignment,
or renormalization.

For the central probability, direct `1 - C` is accepted when `x > 0.5`. At
`x <= 0.5`, preflight avoids cancellation with the convergent series:

```text
q_exp[0] = x/2 - x**2/6 + x**3/24 - x**4/120 + ...
```

The frozen evaluation is the degree-20 Horner polynomial
`q_exp[0] = x * Horner(c[1], ..., c[20])`, where
`c[n] = (-1)**(n + 1) / (n + 1)!`. It starts with `h = c[20]`, updates
`h = c[n] + x*h` for `n = 19, ..., 1`, and returns `x*h`. This fixed degree and
operation order are part of the binary64 mapping. Each coefficient is rounded
once with `float(Fraction((-1)**(n + 1), factorial(n + 1)))`; each multiply and
add is a separate host binary64 operation with no fused contraction.

Preflight requires finite `a`, `C`, tails, and categories in `[0, 1]`, a
nonincreasing tail, and no negative value under any tolerance. The fixed
absolute bound for `q_exp[0] + R[1] = 1`, telescoping/tail diagnostics, and
independent high-precision category/tail fixtures is `1e-12`. The complete
represented finite-window law
`q_exp[0:S]` plus `R[S]` must differ from its ideal high-precision law by at
most `1e-11` in L1 distance. These are private algorithm constants, not public
config knobs. Arbitrary precision is validation-only and adds no production
dependency.

For CT from source bin `t`, right overflow is the tail `R[S - t]`; retained
rate fields use the corresponding `q_exp[d]`. Neither value is recovered from
one minus a finite sum. Exponential delay owns no separate RNG stream: these
masses thin the existing retained and overflow CT Poisson roles.

For AP with represented fire probability `p`, retained offsets are ordered
increasingly, followed by overflow and then the final no-AP remainder. For a
retained offset `d` and source-relative first outside offset `L = S - t`, the
stable multinomial masses are:

```text
retained d:  A = p * q_exp[d]
             B = (1 - p) + p * R[d + 1]

overflow:    A = p * R[L]
             B = 1 - p

stop:        final integer remainder, no draw
```

Those `A`/`B` pairs feed the already-selected reduced conditional-binomial
core. The AP builder never constructs overflow as
`1 - sum(q_exp[0:L])` and never forms a tiny conditional side as `1-p_step`.
AP sampling uses the package-owned key at stream `0x0000_0009` and the
generation/category/source address lattice below. Delay-kernel preparation
itself consumes no random word.

#### Exponential Afterpulse-Recovery Preparation

The AP delay kernel answers **which time bin receives a fired afterpulse**.
When recovery is configured, a second preparation step answers **how much
charge that binned afterpulse deposits**. It does not change the sampled AP
count, destination, or later branching.

Let `tau` be the exponential AP-delay mean, `tau_recovery` the recovery time
constant, and:

```text
x = T / tau
y = T / tau_recovery
tau_effective = tau * tau_recovery / (tau + tau_recovery)
c = tau_effective / tau = x / (x + y)
```

Write `q_x[d]` and `R_x[L]` for the exponential category and right-tail
functions above when their dimensionless inverse mean is `x`. Multiplying the
exponential delay density by the unrecovered fraction
`exp(-Delta / tau_recovery)` gives another normalized exponential density with
mean `tau_effective`, scaled by `c`. Therefore the exact integrated recovery
mass is:

```text
h[d] = q_x[d] - c * q_(x + y)[d]
h_ap_tail[L] = R_x[L] - c * R_(x + y)[L]

rho_bar[d] = h[d] / q_x[d]                 when q_x[d] > 0
rho_bar_tail[L] = h_ap_tail[L] / R_x[L]
                                              when R_x[L] > 0
```

These identities are the production definition. They integrate both the
physical delay and the same latent uniform phase used for bin placement;
`rho_bar[d]` is not a bin-edge or bin-center evaluation. Preparation does not
evaluate either displayed subtraction directly, because the two terms can be
nearly equal when recovery is slow.

The recovery ratio has the same bounded binary64 domain as the delay ratio:

```text
2**-52 <= tau_recovery / T <= 2**52
```

When the nonzero AP mechanism and its recovery response are active, the exact
pre-conversion checks for
`recovery_num, recovery_den = tau_recovery_ns.as_integer_ratio()` are
`recovery_num * 1000 * 2**52 >= recovery_den * sample_period_ps` and
`recovery_num * 1000 <= recovery_den * sample_period_ps * 2**52`. Only after
those checks may preflight form the binary64 picosecond value and `y`.
Although the two configured ratios each retain that bound, recovery also
evaluates the exponential helper at the effective inverse ratio `x + y`.
Its frozen auxiliary domain is therefore:

```text
2**-51 <= x + y <= 2**53
```

This extension belongs only to prepared effective-mean evaluation; it does not
widen either public configured-ratio domain. The private helper accepts this
auxiliary interval without reapplying the public delay-ratio gate.

For stable evaluation, define:

```text
f(z) = log((-expm1(-z)) / z)
g(z) = log(q_exp_zero(z) / z)
```

where `q_exp_zero(z)` is the already-frozen `q_exp[0]` evaluator. For every
represented nonzero category or tail, preflight computes the log ratio of the
unrecovered mass to the complete category mass:

```text
ell[0] = g(x + y) - g(x)
ell[d] = 2 * (f(x + y) - f(x)) - (d - 1) * y,       d >= 1
ell_overflow[L]
    = -log1p(y / x) + f(x + y) - f(x) - (L - 1) * y

rho_bar = -expm1(ell)
h = q * rho_bar
```

The same `ell_overflow` mapping is used with the analytic right tail; overflow
recovery is never reconstructed by summing finite categories. The frozen
difference evaluator for either `f(x + y) - f(x)` or
`g(x + y) - g(x)` is:

1. If `x + y <= 0.5`, evaluate the appropriate degree-14 Taylor-polynomial
   difference with the frozen divided-power recurrence below. Production does
   not evaluate two nearby polynomials and subtract them.
2. Otherwise, if `y <= 2**-16 * max(1, x)`, use the frozen midpoint mapping
   `y * f'(x + y/2)` or `y * g'(x + y/2)`, with:

   ```text
   f'(z) = exp(-z) / (-expm1(-z)) - 1/z

   a(z) = -expm1(-z)
   q0(z) = 1 - a(z)/z
   q0'(z) = (a(z) - z*exp(-z)) / z**2
   g'(z) = q0'(z)/q0(z) - 1/z
   ```

3. Otherwise, use:

   ```text
   f(x + y) - f(x)
       = log1p(exp(-x) * (-expm1(-y)) / (-expm1(-x)))
         - log1p(y/x)

   g(x + y) - g(x)
       = log(q_exp_zero(x + y) / q_exp_zero(x))
         - log1p(y/x)
   ```

The degree-14 Taylor coefficients are fixed. Unlisted powers have coefficient
zero, and the constant in `g` cancels from the difference:

```text
f:
  1: -1/2
  2:  1/24
  4: -1/2880
  6:  1/181440
  8: -1/9676800
 10:  1/479001600
 12: -691/15692092416000
 14:  1/1046139494400

g (constant -log(2) omitted):
  1: -1/3
  2:  1/36
  3: -1/810
  4: -1/12960
  5:  1/68040
  6: -1/12247200
  7: -1/6123600
  8:  13/1175731200
  9:  307/218245104000
 10: -479/2036954304000
 11: -167/39720608928000
 12:  100921/28598838428160000
 13: -109/649973600640000
 14: -3391/85796515284480000
```

Each displayed rational coefficient is converted once with
`float(Fraction(numerator, denominator))` into host binary64 before the loop.
Let `z = x + y`. The exact reference loop for either coefficient table is:

```python
divided_power_difference = 1.0  # (z**1 - x**1) / y
x_power = 1.0                   # x**0
terms = []

for degree in range(1, 15):
    if degree in coefficients:
        terms.append(coefficients[degree] * divided_power_difference)
    if degree < 14:
        x_power = x_power * x
        divided_power_difference = (
            z * divided_power_difference + x_power
        )

difference = y * math.fsum(terms)
```

All multiply and add operations above are separate host binary64 operations;
the reference does not contract them into fused multiply-add. The increasing
degree order is fixed. A faster implementation may replace this loop only
after proving the accepted probability tolerances against this mapping.

Preflight requires every evaluated `ell` to be finite and nonpositive, every
`rho_bar` to be finite in `[0, 1]`, and every recovery mass to lie in `[0, q]`.
It clips neither a log ratio nor a response into range. Placement absence and
no-division behavior are controlled only by `q_x[d] == 0` or `R_x[L] == 0`.
If `h` or `h_ap_tail` rounds or underflows to zero while its placement mass
remains positive, the realized AP count and destination remain present and its
prepared deposited response is exact zero.

The same `1e-12` local absolute tolerance and `1e-11` complete-law L1 tolerance
used by the exponential delay kernel apply to independent high-precision
recovery fixtures. The prepared identity `h + c*q_(x+y) = q_x` is checked
within the local tolerance. `afterpulse_charge_square_sum` uses the square of
the category's prepared `rho_bar` for each realized AP, not a separately
prepared conditional second moment and not the square of aggregate AP charge.

Both CT modes are causal: `Delta_m >= 0`, so their prepared kernels have no
negative-offset or underflow category. A fixed zero-delay DiCT model therefore
lands in offset zero with probability one; a cross-bin DiCT model requires an
explicitly nonzero causal delay. Preflight derives each accepted kernel's
complete offset PMF and analytic right tail; the aggregate simulation draws no
per-edge delay tensor on the hot path. Every prepared delay kernel must have
nonnegative support and satisfy the accepted PMF-plus-right-tail numerical
contract, or preparation fails before any RNG request, product-producer
invocation, or semantic-output write. The selected TensorDSLab timing policy
does not fold a later independent child-jitter draw into `Delta_m`. IV's later
independent jitter of parent and child rows can produce a signed post-binned
relative displacement, but that is a donor timing divergence rather than an
alternate selected CT kernel. The independent-edge closure
preserves each one-edge offset PMF but intentionally omits covariance from a
shared hidden parent phase among siblings, mechanisms, or successive
generations. Sharing one phase per parent would require marked per-parent state
and would make the aggregate independent destination-Poisson equations below
inexact; it is not part of this dense unmarked model.

Per-mode Poisson thinning and within-mode source superposition give:

```text
R_m[g + 1, u]
    = lambda_m * sum_t F[g, t] * q_m[u - t; sampling]

A_direct_crosstalk[g + 1, u]
    ~ Poisson(R_direct[g + 1, u])

A_delayed_crosstalk[g + 1, u]
    ~ Poisson(R_delayed[g + 1, u])
```

Both retained fields call TensorCore's public Poisson sampler. Direct and
delayed crosstalk own distinct streams, and each actual destination-cell rate—not only
the configured scalar `lambda_m`—must lie in `[0, 1e8]`. For generation `g`, a
retained mode uses `logical_position = g * N + destination_flat_position`,
where `N` is the complete destination grid size, and `source_quantum = 0`.

The selected algorithm keeps these as two explicit Poisson draws. It does not
sample `Poisson(R_direct + R_delayed)` and does not recover the modes with a
conditional Binomial split, even though that alternative has the same
conditional joint law. Separate draws keep the scientific bookkeeping and RNG
roles direct. A disabled mode or zero rate produces exact zeros without a draw.

No Gamma latent intensity, Gamma-Poisson mixture, or negative-binomial
offspring law surrounds either supplied mean. Adding one would be a different
scientific model requiring a new Design decision and parity classification.

The audited IV source uses configured `direct_ct = 0.3` as
`lambda_direct = 0.3`: a fixed mean number of DiCT offspring in the following
genealogical generation per unit parent. TensorDSLab does not sample a Gamma
latent intensity around `0.3`, and does not reinterpret `0.3` as the probability
of at least one child. “Fixed” means that the configured scalar is not itself a
per-cell random variable; the IV comparison preset is exactly `0.3`, while later
calibrated presets may use a different reviewed scalar.

For a DiCT-only process with one root on an unbounded retained domain, the
expected total population in generation `g`, summed over destination bins, is
`lambda_direct**g`; its expected position profile is
`lambda_direct**g * convolution_power(q_direct, g)`. With
`lambda_direct = 0.3`, `K = 1` adds mean `0.3`, while the untruncated expected
additional progeny is `0.3 / (1 - 0.3) = 0.428571...`. Finite-window retained
means are smaller and position-dependent because of edge losses. The generation
loop obtains the larger cascade mean only by recursively processing realized
children; it never replaces the fixed direct-offspring mean with `0.428571...`.

DeCT is an optional TensorDSLab model rather than an IV-parity claim. For either
causal crosstalk mode, exact thinning places children beyond the right window
edge in a separate mode-specific overflow bucket. For source bin `t`:

```text
q_m_overflow[t]  = sum(q_m[d; sampling], d >= S - t)

A_m_overflow[g + 1, t]
    ~ Poisson(lambda_m * F[g, t] * q_m_overflow[t])
```

Each mode retains its own overflow draw and diagnostic; modes are not
superimposed there either. `direct_crosstalk=None` or
`delayed_crosstalk=None` disables the corresponding mode structurally and
requires no mode draw or physical zero contribution buffer.
Direct and delayed overflow each own a stream distinct from both one another
and their retained mode. Their generation address is
`logical_position = g * N + source_flat_position`, with `source_quantum = 0`.
An exact zero tail or rate requests no word. Every positive overflow mean uses
the same `[0, 1e8]` TensorCore Poisson domain.

The initial finite-window algorithm uses an absorbing right boundary. Every CT
or AP overflow child is counted and removed before the next generation; its
descendants are not simulated. Because every selected displacement is causal,
an overflow child cannot later return to the retained window. Given the same
in-window roots, this absorbing rule therefore agrees with simulating the
unbounded causal cascade and cropping its retained prefix.

The mathematical CT and AP offset PMFs are defined over nonnegative integer
offsets. A tensor-prepared mode plan must provide `q[0]` through `q[S - 1]`
plus its exact right tail, or obtain the same values from exact CDF differences.
No prepared kernel may collapse possibly retainable offsets into a tail,
truncate an infinite-support law, or silently renormalize it.

AP is also optional. Every parent produces at most one direct AP child with
probability `p_ap`. Conditional on firing, its nonnegative delay offset follows
`q_ap`, prepared from the ordinary
`Delta_ap ~ Exponential(AfterpulseConfig.mean_delay_ns)` law and the same
independent-edge phase closure. The exponential law satisfies the same shared
causal-delay invariant by construction, and its prepared kernel must likewise
have no underflow category and must preserve its complete right tail. For source
bin `t` containing `Q` parents, define
`q_ap_overflow[t] = R_ap[S - t]` from the analytic right tail. It is never
reconstructed as one minus retained mass. The exact direct outcome law is:

```text
(A[t, 0], ..., A[t, S - 1 - t], A[t, overflow], A[t, stop])
    ~ Multinomial(
        Q,
        p_ap * q_ap[0],
        ...,
        p_ap * q_ap[S - 1 - t],
        p_ap * R_ap[S - t],
        1 - p_ap,
    )
```

Retained AP categories are shifted into their destination bins to form
`A[g + 1]`; `A[overflow]` enters the AP overflow bucket. Every retained AP child
is one unit-count avalanche and receives the full common offspring law in the
next generation. The overflow probability includes both explicit kernel-tail
mass and offsets that cross the right edge from the current source bin. A no-AP
outcome and a fired-but-overflowed AP remain different diagnostics.

AP deposited charge is derived from the same realized multinomial categories,
not from an independent mean field. `afterpulse=None` disables AP completely.
`afterpulse.recovery=None` keeps AP count, timing, and future branching enabled
but gives every retained AP unit deposited charge. A present exact
`AfterpulseRecoveryConfig(time_constant_ns=tau_recovery)` selects:

```text
rho(Delta) = 1 - exp(-Delta / tau_recovery)
```

The recovery response changes deposited charge only. Let `Delta` be the
physical AP delay, `U_edge` the same independent edge-phase marginalization
used to assign this AP outcome, and
`J = floor((U_edge + Delta) / T)` the offset before window clipping. Preflight
prepares:

```text
q_ap[d]
    = P(J = d | AP fired)

h_ap[d]
    = E[rho(Delta) * 1{J = d} | AP fired]

rho_bar_ap[d]
    = h_ap[d] / q_ap[d]  when q_ap[d] > 0
```

`rho_bar_ap[d]` is the conditional mean recovery for that offset category. It
is not `rho(d * T)`, a bin-center evaluation, or a function of an unrelated
timing-jitter displacement. A zero-probability category is never sampled and
needs no recovery division. The response treats `Delta` as the delay
from the AP's immediate parent avalanche and assumes that parent reset the
relevant microcell. It does not reconstruct full same-cell firing history from
channel-level bins; that requires microcell-resolved state.

If `A[g + 1, t, d]` denotes the logical category count drawn from source bin
`t` before shifting, the selected binned response is:

```text
A_ap[g + 1, u]
    = sum(A[g + 1, t, d], t + d = u)

C_ap[g + 1, u]
    = sum(A[g + 1, t, d] * rho_bar_ap[d], t + d = u)
```

The full source-by-offset tensor need not be materialized: one exact
multinomial category scan or fused sampler may accumulate the integer
destination count and floating destination charge together. Applying the
recovery weight before source/offset categories collapse is essential because
different delays can reach the same destination bin.

This is the conditional expected physical charge given the realized binned AP
categories. It preserves the exact mean, AP fire/delay fluctuations,
multinomial cross-bin covariance, and covariance between the AP count frontier,
AP charge, and later count-driven generations. It omits only recovery-amplitude
variation within one offset category, in addition to the timing-latent
covariances already excluded by the unit-count binned model.

When `recovery is None`, preflight uses `rho_bar_ap[d] = 1` for every nonzero
category. Mathematically, `afterpulse_charge` and
`afterpulse_charge_square_sum` both reduce to retained `afterpulse_count` in the
charge dtype, while overflow remains excluded from the retained ledgers. The
implementation's generation-wise floating accumulations use the dtype-aware
ledger tolerances below rather than promising bitwise equality with one final
cast of the cumulative integer count.

For comparison, if `Lambda[t] = p_ap * F[g, t]`, then

```text
E[C_ap[g + 1, u] | F[g]]
    = sum_t Lambda[t] * h_ap[u - t]
```

is a useful analytic validation oracle. Setting the simulated `C_ap` equal to
that convolution would be a different mean-field approximation: it could
deposit positive expected charge when the sampled AP count is zero and would
remove the AP count/charge covariance still available after binning. It is not
the selected event-level algorithm. Likewise, a Poisson AP count is not
interchangeable with the selected bounded multinomial law because it permits
more than one direct AP child per parent.

For overflow-charge diagnostics, preflight also prepares the analytic recovery
tail from the frozen exponential-recovery mapping:

```text
h_ap_overflow[t] = h_ap_tail[S - t]

rho_bar_ap_overflow[t]
    = h_ap_tail[S - t] / R_ap[S - t]
      when R_ap[S - t] > 0
```

It applies that source-position-dependent conditional mean recovery to the same
sampled overflow count. A zero-probability tail is exact zero and needs no
division. `afterpulse=None` disables AP and every AP count, charge, squared
charge sum, and overflow diagnostic structurally.

The generation update keeps every birth mechanism explicit. DiCT and DeCT use
their two separate Poisson draws and accumulators; AP retains its separate
bounded categorical law. Cross-mode rate superposition and a conditional mode
split are not part of the selected algorithm.

The symmetric `draw_direct_crosstalk`, `draw_delayed_crosstalk`, and
`draw_afterpulses` names below label sampler roles inside
`simulate_correlated_avalanches(...)`. They are not public transforms or
additional product producers, and an implementation may inline or fuse them
while preserving their separate laws, streams, and diagnostics.
The two crosstalk roles construct their scientific rate fields and call
`rng.poisson(mean=..., key=..., positions=..., quantum=0)`; their exact
`DirectCrosstalkConfig` or `DelayedCrosstalkConfig` supplies distinct retained
and overflow keys. They do not own competing Poisson algorithms.

```python
plan = prepare_correlated_avalanche_plan(
    sample_dimension=sample_dimension,
    sampling=sampling,
    floating_dtype=floating_dtype,
    config=config,
)

if plan.all_mechanisms_disabled:
    return identity_avalanche_result(
        seed_avalanches,
        maximum_generations=config.maximum_generations,
        seed_S1=to_charge_dtype(seed_avalanches, dtype=floating_dtype),
        seed_S2=to_charge_dtype(seed_avalanches, dtype=floating_dtype),
    )

frontier_count = seed_avalanches
total_count = copy_to_fresh_total(seed_avalanches)
S1 = to_charge_dtype(seed_avalanches, dtype=floating_dtype)
S2 = to_charge_dtype(seed_avalanches, dtype=floating_dtype)

direct_crosstalk_count = (
    zeros_like(seed_avalanches)
    if plan.direct_crosstalk is not None
    else None
)
direct_crosstalk_overflow_count = (
    zeros_like(seed_avalanches)
    if plan.direct_crosstalk is not None
    else None
)
delayed_crosstalk_count = (
    zeros_like(seed_avalanches)
    if plan.delayed_crosstalk is not None
    else None
)
delayed_crosstalk_overflow_count = (
    zeros_like(seed_avalanches)
    if plan.delayed_crosstalk is not None
    else None
)
afterpulse_count = (
    zeros_like(seed_avalanches)
    if plan.afterpulse is not None
    else None
)
afterpulse_overflow_count = (
    zeros_like(seed_avalanches)
    if plan.afterpulse is not None
    else None
)
afterpulse_charge = (
    zeros_like_charge(seed_avalanches, dtype=floating_dtype)
    if plan.afterpulse is not None
    else None
)
afterpulse_overflow_charge = (
    zeros_like_charge(seed_avalanches, dtype=floating_dtype)
    if plan.afterpulse is not None
    else None
)
afterpulse_charge_square_sum = (
    zeros_like_charge(seed_avalanches, dtype=floating_dtype)
    if plan.afterpulse is not None
    else None
)

for generation_index in range(config.maximum_generations.value):
    children_count = logical_zero

    if plan.direct_crosstalk is not None:
        (
            new_direct_crosstalk_count,
            new_direct_crosstalk_overflow_count,
        ) = draw_direct_crosstalk(
            frontier_count,
            plan.direct_crosstalk,
        )
        direct_crosstalk_count += new_direct_crosstalk_count
        direct_crosstalk_overflow_count += (
            new_direct_crosstalk_overflow_count
        )
        direct_charge = to_charge_dtype(
            new_direct_crosstalk_count,
            dtype=floating_dtype,
        )
        S1 += direct_charge
        S2 += direct_charge
        children_count += new_direct_crosstalk_count

    if plan.delayed_crosstalk is not None:
        (
            new_delayed_crosstalk_count,
            new_delayed_crosstalk_overflow_count,
        ) = draw_delayed_crosstalk(
            frontier_count,
            plan.delayed_crosstalk,
        )
        delayed_crosstalk_count += new_delayed_crosstalk_count
        delayed_crosstalk_overflow_count += (
            new_delayed_crosstalk_overflow_count
        )
        delayed_charge = to_charge_dtype(
            new_delayed_crosstalk_count,
            dtype=floating_dtype,
        )
        S1 += delayed_charge
        S2 += delayed_charge
        children_count += new_delayed_crosstalk_count

    if plan.afterpulse is not None:
        (
            new_afterpulse_count,
            new_afterpulse_overflow_count,
            new_afterpulse_charge,
            new_afterpulse_overflow_charge,
            new_afterpulse_charge_square_sum,
        ) = draw_afterpulses(
            frontier_count,
            plan.afterpulse,
        )
        afterpulse_count += new_afterpulse_count
        afterpulse_overflow_count += new_afterpulse_overflow_count
        afterpulse_charge += new_afterpulse_charge
        afterpulse_overflow_charge += new_afterpulse_overflow_charge
        afterpulse_charge_square_sum += new_afterpulse_charge_square_sum
        S1 += new_afterpulse_charge
        S2 += new_afterpulse_charge_square_sum
        children_count += new_afterpulse_count

    frontier_count = materialize_or_reuse(children_count)
    total_count += frontier_count
```

The unsuffixed mechanism names above are accumulated over all evaluated
offspring generations. Retained count and charge diagnostics use the readout
grid and are indexed by destination bin. Each `*_overflow_count` and
`afterpulse_overflow_charge` uses the same logical grid but is indexed by the
source bin whose child crossed the right boundary. There is no underflow
diagnostic because every selected edge law is causal. Overflow never enters
`frontier_count`, `total_count`, `S1`, `S2`, or a waveform. DiCT and DeCT need no
separate persistent charge diagnostic: their deposited charge is exactly the
floating conversion of their retained count.

The all-disabled identity result reports the root grid as its logical final
frontier for `K=0` and an exact zero frontier for `K>=1`; it may represent the
latter structurally without allocating a public tensor. All diagnostics for an
absent mechanism are structurally absent rather than materialized zeros.

`logical_zero` above is an absent mathematical contribution, not an allocated
zero tensor, and `materialize_or_reuse` stands only for the later execution
plan's choice of an already required generation buffer. Thus a structurally
disabled mechanism has no sampler call, count/charge contribution buffer, or
overflow buffer. With all mechanisms disabled, the implementation takes the
identity path before this loop. The names show scientific roles, not a required
one-buffer-per-name execution plan; later workspace design may safely reuse
storage only after preserving every simultaneously live count and charge role.

The pseudocode names mathematical integer samplers. TensorCore owns the
selected Poisson algorithm, raw-word schedule, supported generic per-cell mean,
and sampler failure policy. TensorDSLab owns the scientific streams, addresses,
rates, and ledger interpretation fixed below. Fusion and scratch scheduling remain
later measured implementation choices. Avalanche counts remain nonnegative
`int64` throughout the branching simulation; binary64 rates/probabilities and
requested-dtype deposited charge remain separate floating computations. No
floating charge value is converted back into a parent count or sampler
parameter. The eventual implementation must detect or preclude integer
overflow rather than permit wrapping.

For homogeneous parameters, the unwindowed mean reproduction per unit parent
is `lambda_direct + lambda_delayed + p_ap`. Fixed `K` makes the algorithm finite
even when that value is at least one, but does not make explosive count growth
safe or well calibrated. The Poisson sampler accepts an actual cell mean no
larger than `1e8`; the exact per-cell count, relational `K`, address,
accumulator-depth, and allocation envelope is frozen below rather than hidden
inside an implicit change to `K`.

The frontier and two ledgers above are semantic roles, not a literal
three-buffer ceiling. Whenever the correlated stage executes, its algorithm
always accumulates the private numerical `S2` ledger; that ledger is not
conditioned on whether smearing is enabled. If the complete correlated stage is
skipped, no correlated result or `S2` tensor is constructed. A later effective
smearing stage instead derives the unit-response identity `S1 == S2` from the
then-current integer `charge` tensor. No gain-smearing draw occurs inside the
generation loop. After generation `K`, the terminal charge rule consumes the
completed ledgers.

Recovery-weighted AP deposits make the old unit-count smearing scale
`sqrt(total_count) * relative_sigma` insufficient. The selected independent
multiplicative per-avalanche Gaussian gain model with category response weights
`w_i` uses:

```text
S1[t] = sum_i w_i                  = charge_pe[t]
S2[t] = sum_i w_i**2               = charge_square_sum[t]

draw[t] | {w_i}
    ~ Normal(S1[t], relative_sigma * sqrt(S2[t]))

Charge[t] = max(draw[t], 0)
```

Roots, DiCT, and DeCT add one to both `S1` and `S2`; a realized AP category adds
`rho_bar_ap[d]` to `S1` and `rho_bar_ap[d]**2` to `S2`. Under the ideal
mathematical Gaussian model, those two ledgers are sufficient for the aggregate
gain distribution and reduce to the existing `sqrt(n)` rule for unit weights.
The selected finite-lattice implementation draws one aggregate digital normal;
it does not claim bitwise or distributional identity with summing separately
rounded per-avalanche digital normal draws. The model also does not restore the
intentionally omitted within-category recovery variance. `S2` is private
numerical scratch, not a physical response, branching tensor, product,
collection sidecar, or input to offspring sampling. A later Design may decide
not to expose it, but that does not remove it from this algorithm. Using only
`sqrt(total_count)` or `sqrt(charge_pe)` after fractional AP deposits would
instead select a different smearing law.

If `ChargeConfig.smearing is None`, or if its configured `relative_sigma` is
zero, the smearing block is skipped and consumes no smearing draw. The terminal
`Charge` is constructed directly from the then-current tensor, with the one
explicit floating conversion required when the last executed stage still holds
integer root counts. Otherwise `simulate_charge_smearing(...)` evaluates the
normal law above in the selected floating dtype and applies the documented
nonnegative clipping. No count, rate, or later offspring law reads the smeared
result.

Enabled smearing uses the package-owned key at stream `0x0000_000A`,
`source_quantum = 0`, and the row-major product-grid flat position directly as
`logical_position`. The eager reference
visits every one of the `N` grid positions, including a cell whose represented
`S2` is zero. Such a cell still owns and consumes its addressed scalar normal;
its zero scale makes the draw observationally inert without introducing a
value-dependent compacted schedule or a device-wide active-cell decision. A
scalar consumer always uses `z0` from the position's ordinary dtype-matched
Box-Muller pair and discards `z1`; it never lends that spare result to another
position. Absent smearing or exact zero `relative_sigma` skips the whole stream.

Within `draw_afterpulses(...)`, the corresponding private output is
`afterpulse_charge_square_sum`. It accumulates each realized category as
`category_count * rho_bar_ap[d]**2` before category collapse. Squaring the
already aggregated `afterpulse_charge` would add cross terms and is not this
quantity. If the response config selects unit AP charge instead,
`afterpulse_charge` and `afterpulse_charge_square_sum` both reduce
mathematically to `afterpulse_count` in the charge dtype; generation-wise
floating accumulation still follows the dtype-aware ledger rule. The sampler
surface does not change.

Contribution identity is the mechanism on the incoming birth edge, not the
root ancestry. If an AP-born avalanche later produces a DiCT child, the parent
is counted in `afterpulse_count` and the new child in
`direct_crosstalk_count`. This preserves mechanism-resolved multiplicity
without carrying a growing lineage state. Charge uses that same incoming-edge
identity: the AP parent's recovery-weighted deposit remains in
`afterpulse_charge`, while its unit-charge DiCT child contributes through the
floating conversion of `direct_crosstalk_count`.

Subject to absent mechanisms being mathematical zero, the count invariant for
the selected absorbing retained-window process is:

```text
total_count
    == seed_avalanches
     + direct_crosstalk_count
     + delayed_crosstalk_count
     + afterpulse_count
```

Before smearing, the corresponding mathematical deposited-charge identity is:

```text
S1
    == to_charge_dtype(seed_avalanches)
     + to_charge_dtype(direct_crosstalk_count)
     + to_charge_dtype(delayed_crosstalk_count)
     + afterpulse_charge
```

The unconditional mathematical charge-square-sum identity is:

```text
S2
    == to_charge_dtype(seed_avalanches)
     + to_charge_dtype(direct_crosstalk_count)
     + to_charge_dtype(delayed_crosstalk_count)
     + afterpulse_charge_square_sum
```

The integer count identity is checked exactly. The S1 and S2 identities define
the scientific ledgers in real arithmetic, but the implementation accumulates
them in the selected `float32` or `float64` dtype. Validation therefore uses a
frozen accumulation order plus dtype- and bound-aware tolerances; it must not
assert exact recomputed floating equality unless an implementation stage proves
that stronger property for its exact reduction plan.

Here `afterpulse_charge_square_sum` accumulates
`A[g + 1, t, d] * rho_bar_ap[d]**2` before delay categories collapse. It cannot
be reconstructed afterward from `afterpulse_count` and
`afterpulse_charge`, and it is not `afterpulse_charge**2`.

When the correlated stage executes, its private algorithm result retains the
three mechanism-specific count totals, the final integer frontier after
generation `K`, separate DiCT, DeCT, and AP overflow counts, the retained AP
charge and charge-square-sum, and the AP overflow-charge diagnostic needed for
validation and truncation. The common `S1` and `S2` ledgers are always present
in that result. These are private diagnostic/state values, not
`ReadoutCollection` fields, durable products, or independently ordered public
transforms. Only the terminally finalized `Charge` is a recognized field. The
public charge path invokes `simulate_correlated_avalanches(...)` at most once
inside `produce_charge(...)`, and only when that stage's enclosing condition
is true. The final frontier is the included generation `K` whose children were
not evaluated; it is a truncation indicator, not an estimate of the complete
omitted population.

The original IV-DSLab donor implements a real recursive avalanche tree in
`Projects/iv-dslab-main_db_PB/src/dselec/sipm.py`. Its default database path
uses the `_db` variants and includes DiCT plus AP; the non-database path also
contains the disabled-by-default prompt PHCT mechanism:

- `_poissonian_loop(...)` recursively samples Poisson offspring;
- `_add_phct(...)` and `_add_dict(...)` collapse same-type prompt recursion;
- `_add_ap(...)` creates delayed descendants and leaves
  afterpulse-of-afterpulse recursion enabled; and
- `_add_corr_noise(...)` walks a growing PE queue, so crosstalk and
  afterpulse descendants can seed further effects. Dark counts enter that
  queue before the correlated effects.

The audited `PEType` set has no DeCT member. `PHCT` is a `TODO`/`FIXME`, is
disabled by default, creates same-raw-time unit-charge rows, and is absent from
the database path; it is not evidence of an implemented delayed-crosstalk
model. IV's AP rows are delayed and recovery weighted, but that does not make
them DeCT. TensorDSLab therefore treats DeCT as absent from the donor rather
than calibrating it from IV's AP path. Any physical DeCT model remains a new,
separately calibrated inter-microcell mechanism with no IV-parity claim.

IV copies a DiCT descendant's raw continuous time from its source row, but it
later applies an independent Gaussian timing jitter to every parent and
generated row before binning. DiCT is therefore same-time only at IV's
pre-jitter table boundary. At a post-jitter binned comparison boundary, the
relative displacement contains `J_child - J_parent` and may produce a signed,
sampling-period-dependent offset kernel. Because an internally unrolled IV
subtree also shares one pre-jitter time, independently marking each binned edge
with the marginal kernel does not preserve every sibling or multigeneration
timing covariance.

Literal donor parity needs care. IV gives an AP row recovery-weighted charge and
then multiplies both that row's later DiCT coefficient and AP fire probability
by the same fractional value. Its prompt DiCT helper also freezes a source
charge-weighted offspring coefficient throughout an internally unrolled tree,
even though the emitted DiCT descendants have unit charge. Charge smearing is
applied only after the correlated-noise queue, so the smeared value does not
feed branching.

The first source-side scaling is not inherently unphysical. A partially
recovered source avalanche has lower multiplication charge and may emit fewer
secondary photons or carriers, reducing the number of DiCT or DeCT seeds. The
inter-microcell nature of crosstalk instead means that a seed which triggers a
different, recovered destination microcell normally produces a full-charge
child. IV does assign unit charge to those emitted DiCT children. Its specific
oddity is carrying the original fractional source coefficient through the
helper's whole hidden DiCT subtree rather than letting each unit-charge child
own its direct-offspring law.

This architecture deliberately chooses cleaner unit-count, recovery-independent
branching while retaining recovery only as AP deposited-charge response. That
is an intentional divergence from IV's charge-dependent future branching. The
donor's source-charge-frozen DiCT unrolling is classified as a historical
artifact, not a target for TensorDSLab. The selected model must not recover IV
behavior by feeding an aggregate charge bin, a conditional recovery weight,
or a transient recovery category back into any offspring sampler.

The previous tensor-path DSLab made the opposite, deliberate simplification.
`Projects/dslab/dslab/domain/readout/kernels/charge/__init__.py` freezes the
count grid after its enabled timing-jitter and dark-count steps, while
`crosstalk.py` and `afterpulses.py` independently produce first-generation
contributions. Its charge config fixes generation depth to one, and
`Projects/dslab/tests/test_readout_iv_grid_effects.py` tests frozen-snapshot
additivity. That is donor/comparator history, not the rebuild implementation
surface. The rebuild obtains a first-generation approximation only by choosing
`K=1` on the same coupled `simulate_correlated_avalanches(...)` path. It does
not preserve separate `_contribute_crosstalk(...)` and
`_contribute_afterpulses(...)` functions or let their results bypass the shared
integer frontier and S1/S2 ledgers.

There are analytic and tensor-native routes worth evaluating. Ignoring spatial
or temporal placement, a DiCT-only Galton-Watson tree with Poisson mean
offspring `lambda` has the Borel total-progeny distribution:

```text
P(T=n) = exp(-lambda*n) * (lambda*n) ** (n - 1) / n!
E[T]   = 1 / (1 - lambda)
Var[T] = lambda / (1 - lambda) ** 3
```

Multiple fixed seeds give the Borel-Tanner distribution. A direct total-progeny
sampler could therefore avoid explicit count recursion provided `lambda < 1`
and numerical-tail policy is explicit, but it does not by itself assign the
sampling-dependent DiCT offsets or reproduce their joint timing covariance. A
same-bin scalar sampler is only the delta-at-zero kernel special case.

For delayed afterpulsing with mean offspring-time kernel `H`, the expected
descendant response is the renewal series:

```text
R = H + H*H + H*H*H + ...
R_hat = (I - H_hat) ** -1 * H_hat
```

For `H(t) = eta * beta * exp(-beta*t)`, this mean response has the closed form
`R(t) = eta * beta * exp(-beta * (1 - eta) * t)`. A coupled channel/time model
has the corresponding Neumann series `I + H + H**2 + ...`; convergence
requires scalar mean reproduction below one or, for a multitype model, an
integrated offspring matrix with spectral radius below one.

These closed forms describe means and selected moments. They do not by
themselves reproduce the random cascade's variance, tails, joint timing, or
channel correlations. A fixed channel topology can be represented by a
multitype kernel. Finite microcell count, saturation, collisions,
recovery-dependent offspring laws, and occupancy- or state-dependent topology
break the simple count-branching closure. Recovery-weighted AP deposited
charge alone does not: it is a separate linear response kernel over the same
unmarked avalanche cascade.

The fixed-`K` generation loop intentionally truncates descendant chains by
genealogical depth regardless of their realized offset. A future optimization
or scientific alternative must begin as a new Design proposal rather than
entering implementation as an undocumented substitution.

The scientific transition law, config ownership, recovery response, causal
window policy, diagnostic roles, aggregate multinomial factorization, AP and
smearing streams, supported count/address/accumulation envelope, and
model-conformance tolerances are closed above and below. The promoted
TensorCore contract owns the stabilized inversion/BTRS mapping, hybrid Poisson
sampler, generic numerical domains, and distribution failure policy. The exact fixed-delay
mapping and the exponential delay/recovery evaluators, domains, and tolerances
are likewise closed. Fusion and scratch scheduling remain measured
implementation decisions; smarter automatic `K` selection would be a later
scientific Design change. Because descendants from every mechanism feed every
enabled mechanism in the following generation, the private boundary remains
one coupled `simulate_correlated_avalanches(...)` operation rather than three
sequential public effects.

Algorithm-level validation should cover `K = 0` and `K = 1` off-by-one cases;
the DiCT-only per-generation Poisson law, `lambda_direct**g` integrated mean,
and convolution-power position profile; sampling-period and phase-policy PMF
fixtures for both CT modes; nonnegative offset support, PMF-plus-right-tail
normalization, exact absence of underflow, and separate per-mode overflow
accounting; distinct DiCT and DeCT Poisson draws and accounting; sibling and
multigeneration timing covariance as named approximation diagnostics; AP's
one-child bound, destination multinomial covariance, and separate stop/overflow
outcomes; `q_ap`, `h_ap`, and `rho_bar_ap` fixtures from the joint latent phase
and physical-delay law; AP charge and `afterpulse_charge_square_sum` from the
same realized category counts; conditional charge mean and count/charge
covariance; the named missing
within-category recovery variance; exact unit deposited charge for fresh-cell
DiCT/DeCT children; parent-cell recovery weighting for AP; recovery-independent
cross-feeding in the following generation; all eight mechanism enablement
combinations; the exact integer count invariant; dtype-aware validation of the
mathematical S1/S2 component identities and their unit-weight reduction;
final-frontier semantics; and checked `int64` overflow behavior. These tests
need no final TensorDSLab stream assignment or downstream smearing decision to
establish the scientific transition law.

Research references for that stage are:

- M. Dwass, [*The total progeny in a branching process and a related random
  walk*](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/total-progeny-in-a-branching-process-and-a-related-random-walk/B9505DF87F411C873D3AA511F21EBF8C),
  for total-progeny validation evidence for Poisson branching limits.
- S. Vinogradov, [*Analytical models of probability distribution and excess
  noise factor of Solid State Photomultiplier signals with
  crosstalk*](https://arxiv.org/abs/1109.2014), including the Poisson branching
  and Borel model; journal DOI
  [10.1016/j.nima.2011.11.086](https://doi.org/10.1016/j.nima.2011.11.086).
- A. G. Hawkes and D. Oakes, [*A cluster process representation of a
  self-exciting process*](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/cluster-process-representation-of-a-selfexciting-process/E836A3D07D808068E2F9F3E7E366B081),
  the immigration-birth/Poisson-cluster foundation for the renewal view.
- A. Para, [*Afterpulsing in Silicon Photomultipliers: Impact on the
  Photodetectors Characterization*](https://arxiv.org/abs/1503.01525).
- J. Rosado et al., [*Modeling crosstalk and afterpulsing in silicon
  photomultipliers*](https://arxiv.org/abs/1409.4564), relevant to spatial
  crosstalk and recovery-aware delayed effects.
- J. Rosado and S. Hidalgo, [*Characterization and modeling of crosstalk and
  afterpulsing in Hamamatsu silicon
  photomultipliers*](https://arxiv.org/abs/1509.02286), which distinguishes
  low-amplitude same-pixel afterpulses during recovery from full-amplitude
  delayed crosstalk in other pixels.
- V. Moya and J. Rosado, [*Understanding the Nonlinear Response of
  SiPMs*](https://arxiv.org/abs/2401.06581), whose recovery-aware Monte Carlo
  scales correlated-noise seed production from a recovering source avalanche
  while separately modeling whether a seed triggers its destination pixel.
- Y. Guan et al., [*Study of Silicon Photomultiplier External
  Cross-Talk*](https://arxiv.org/abs/2312.12901), additional evidence for
  Borel-family fired-pixel models and for treating inter-device topology as a
  deliberate model choice.
- Y. Liu, X. Liu, and B. Xu, [*Closed-Form Analytical Charge Response Model
  for Silicon Photomultipliers with Recursive Correlated
  Avalanches*](https://arxiv.org/abs/2605.27340). This May 2026 preprint is a
  recent research lead and is treated here as unreviewed evidence, not an
  accepted TensorDSLab authority.

### Pure Waveform

`PureWaveformConfig.model` selects one electronics response for the complete
channel axis, using one scalar parameter set for the entire invocation.

For the MVP, TensorDSLab provisionally adopts the two mathematical pulse-shape
families implemented by IV-DSLab. This is an implementation/parity decision,
not a claim that collaborators have finished validating either model as the
best detector description. A later focused scientific decision may revise a
model or its calibration without reopening the typed product/config
architecture. The donor audit found no equation-level defect analogous to the
ADC cast-before-clip bug.

TensorDSLab expresses the IV TPC FEB-SNR response using its actual fast and
slow exponential constants:

```text
h_tpc(t) = exp(-t / tau_slow) - exp(-t / tau_fast)
tau_fast = IV sipm.feb_snr.tau_r
tau_slow = IV sipm.feb_snr.tau_r + IV sipm.feb_snr.tau_l
```

This avoids calling IV's `tau_l` the slow or fall time constant: the donor
places `tau_l + tau_r`, not `tau_l`, in the slow exponential denominator.
`TpcFebSnrPulseConfig` therefore stores `fast_time_constant_ns` and
`slow_time_constant_ns` directly and requires `slow > fast`.

The provisionally adopted IV Veto PDU response is:

```text
x = t - gaussian_center
h_veto(t) = exp(-x**2 / (2 * gaussian_width**2))
            / sqrt(2 * pi * gaussian_width**2)
            * (1 + erf(
                (x - edge_offset_1) / (sqrt(2) * edge_width_1)
            ))
            * (1 + erf(
                (x - edge_offset_2) / (sqrt(2) * edge_width_2)
            ))
```

The exact donor-to-config mapping is:

| TensorDSLab config field | IV-DSLab parameter | Mapping |
| --- | --- | --- |
| `TpcFebSnrPulseConfig.fast_time_constant_ns` | `sipm.feb_snr.tau_r` | direct |
| `TpcFebSnrPulseConfig.slow_time_constant_ns` | `sipm.feb_snr.tau_r`, `sipm.feb_snr.tau_l` | sum |
| `VetoPduPulseConfig.gaussian_center_ns` | `veto.offset1` | direct |
| `VetoPduPulseConfig.gaussian_width_ns` | `veto.fall_time` | renamed to its actual equation role |
| `VetoPduPulseConfig.edge_offset_1_ns` | `veto.offset2` | direct |
| `VetoPduPulseConfig.edge_width_1_ns` | `veto.rise_time1` | renamed to its actual equation role |
| `VetoPduPulseConfig.edge_offset_2_ns` | `veto.offset3` | direct |
| `VetoPduPulseConfig.edge_width_2_ns` | `veto.rise_time2` | renamed to its actual equation role |

IV's calibrated numbers remain provisional parity-fixture evidence, not public
TensorDSLab defaults. In particular, the donor's scalar Veto amplitude and its
condition-database path do not override the MVP decision that one explicit
scalar config applies to every channel in one invocation.

For an 8 ns parity fixture, the audited donor values translate to:

| Model | TensorDSLab fixture values |
| --- | --- |
| TPC FEB-SNR | `fast_time_constant_ns=83`, `slow_time_constant_ns=383`, `support_time_ns=3000`, `peak_voltage_mv_per_pe=7` |
| Veto PDU | `gaussian_center_ns=232.89`, `gaussian_width_ns=507.72`, `edge_offset_1_ns=-81.92`, `edge_width_1_ns=147.28`, `edge_offset_2_ns=-176.50`, `edge_width_2_ns=45.69`, `support_time_ns=2020.27`, `peak_voltage_mv_per_pe=14.5912372` |

The Veto support reproduces the donor's retained samples from 0 through
2016 ns at 8 ns spacing; it does not promote the donor's support heuristic as a
general rule. These values belong in parity fixtures until collaborators
approve named calibration presets.

For sample period `T` and template index `j`, each model is point-sampled as
`h[j] = h(j * T)` at left-edge times satisfying
`0 <= j * T < support_time`; `support_time` is the separate exclusive stop.
Preflight rejects a model/sampling combination whose exclusive support produces
no samples or whose sampled extremum is nonfinite or zero; normalization must
never divide by an unresolved template. The sampled template is normalized by
the magnitude of its sampled extremum, scaled once by the negative of its
strictly positive `peak_voltage_mv_per_pe` amplitude magnitude, convolved
causally with PE-equivalent charge, and truncated to the input sample count.
Negative-going detector polarity is fixed by preparation rather than encoded
as a caller-selected sign; there is no second gain or inversion switch. Output
axes and shape match charge. Baseline is not part of the signal-only
`PureWaveform`.

`prepare_pure_waveform(...)` prepares the config-derived sample times, pulse
values, sampled-extremum normalization, positive amplitude extraction, and one
fixed negative sign in Python binary64, and validates that finite template
before any product producer runs. The producer materializes those prepared
coefficients once in the exact `Charge` dtype and device. This is host-side
preparation of small configuration-derived coefficients, not host
materialization of the input payload. Causal convolution and every
payload-sized operation execute in the field dtype on the field device with no
hidden widening.

TensorDSLab intentionally standardizes the discretization around those donor
equations:

- IV analytically normalizes the continuous TPC curve but normalizes the
  sampled Veto curve. TensorDSLab normalizes both by the magnitude of the
  sampled extremum so `peak_voltage_mv_per_pe` means the realized discrete
  peak. At IV's 8 ns TPC fixture this changes the peak by about 66 parts per
  million.
- IV derives TPC support as `10 * max(tau_l, tau_r)` even though its actual slow
  constant is `tau_l + tau_r`, and derives Veto support through a heuristic
  strict crop. TensorDSLab instead requires one explicit exclusive
  `support_time_ns` and applies the repository-wide left-edge convention.
- IV scales the positive template and inverts after convolution. TensorDSLab
  applies the fixed negative polarity once to the configured positive amplitude
  magnitude, producing the same negative-going parity waveform without a
  second inversion switch.
- The first MVP applies none of IV's eventwise fractional-bin exponential
  amplitude correction. That correction reuses parameters inconsistently and
  is not an exact fractional delay of either adopted pulse equation.

These are documented tensor-path discretization corrections, not alternative
pulse-shape equations. Literal donor support values may be used by parity
fixtures, but they are not package defaults. Gate off-by-one behavior,
pre-window-tail loss, and donor noise/amplitude coupling remain outside the
pure-waveform equation contract and must not be copied accidentally.

### Noise Waveform

Recognized models are:

1. exact zero noise;
2. position-addressed IID Gaussian white noise with ensemble mean zero and
   explicit RMS; and
3. Gaussian noise shaped by a caller-supplied one-sided PSD, with an exactly
   zero DC coefficient and a record mean that is zero up to inverse-transform
   roundoff.

`PsdNoiseConfig` supplies arbitrary strictly increasing frequency left edges,
one exclusive `frequency_stop_hz`, and piecewise-constant absolute density in
`mV^2/Hz`. Left-edge and density counts are equal. It must start at zero, its
stop must exceed the final left edge, and it must cover the Nyquist frequency
implied by `SamplingConfig`; it need not already match the fixed-length
synthesis grid. Preflight integrates source density over left-closed/right-open
target intervals into the pre-suppression `Q` cells, conserving represented
source power before the accepted DC-cell discard constructs `P`.
The PSD is the effective post-front-end, post-anti-alias noise at the shared
analog reference plane. Any supplied coverage above Nyquist is ignored
deliberately rather than folded into band.
Raw FFT amplitudes and complex coefficients are not accepted inputs. No
persistent baseline bank, random crop, spectrum download, SNR-to-amplitude
coupling, or per-call spectral file loading belongs to the noise producer.

For sample rate `fs`, record length `N`, spacing `df = fs / N`, and
`K = floor(N / 2)`, the private Fourier basis frequencies are `c[k] = k * df`
for `k = 0, ..., K`. They are spectral-line/basis frequencies, not bin edges
or necessarily geometric centers of their endpoint cells. The separate
power-integration cells are represented by this left-edge array and exclusive
stop:

```text
target_left_edge[0] = 0
target_left_edge[k] = (k - 1/2) * df,  k = 1, ..., K
target_stop = fs / 2
```

Pre-suppression target cell `Q[k]` receives the PSD integral over
`[target_left_edge[k], target_left_edge[k + 1])`, with the final cell ending at
`target_stop`. The pre-suppression DC cell therefore owns `[0, df / 2)`, which
the accepted policy later discards. For even `N`, the pre-suppression real
Nyquist cell owns a half-width final interval. For odd `N`, there is no
Nyquist coefficient; the highest complex coefficient receives a full-width
final cell that ends at `fs / 2`. Source coverage means exactly
`frequency_stop_hz >= fs / 2`; equality is sufficient because the exclusive
endpoint has zero measure. Preflight rejects a supplied PSD whose retained
in-band power after the zero-DC policy is zero, even if it contains positive
density only above Nyquist or only inside the discarded DC cell.
Rebinning is interval-overlap integration, not interpolation:

```text
Q[k] = sum_i S[i] * max(
    0,
    min(source_right[i], target_right[k])
    - max(source_left[i], target_left_edge[k]),
)

P[0] = 0
P[k] = Q[k],  k = 1, ..., K
```

Stage 5 prepares these cells on the host with Python binary64 arithmetic.
Every target-cell overlap contribution is accumulated with `math.fsum`; DC is
discarded in binary64; and each retained `P[k]` is rounded exactly once into
the requested `torch.float32` or `torch.float64` execution dtype. Those
represented dtype-rounded powers, rather than an unattainable real-number
ideal, define the ideal-standard-normal target coefficient moments and
numerical validation oracle. The finite fixed-point Box-Muller lattice is not
silently renormalized to exact unit variance; its documented bounded-lattice
error is included in statistical acceptance. White-noise RMS follows the same
rule: prepare in binary64 and round once into the output dtype. Preflight
rejects a white RMS that is nonfinite or smaller than the dtype's least positive
normal value, and a PSD whose retained power is zero or nonfinite after this
rounding. The normal-range lower bound excludes a materially quantized
subnormal white-noise law from Stage 5. Payload generation, Box-Muller
arithmetic, coefficient construction, and `irfft` execute in the selected
floating/complex dtype with ambient autocast disabled; no widened payload path
is hidden behind a `float32` request.

The frequency conversion is explicit binary64 arithmetic:

```text
fs_hz = 1e12 / sampling.sample_period_ps.value
df_hz = fs_hz / N
nyquist_hz = fs_hz / 2
```

All three values must be finite and positive. PSD source coverage through
exactly `nyquist_hz` is sufficient; the exclusive endpoint has zero measure.

Preflight also closes the finite-output numerical domain analytically rather
than relying on the later required device-wide producer postcondition. Let
`normal_guard` be `8.0` for `float32` and `16.0` for `float64`, conservative
bounds above the accepted maximum Box-Muller radii.
The represented white RMS must satisfy both
`torch.finfo(dtype).tiny <= rms_mv` and
`normal_guard * rms_mv <= torch.finfo(dtype).max`. Represented PSD powers must
satisfy:

```text
N * normal_guard * math.fsum(sqrt(P[k]) for k = 1, ..., K)
    <= torch.finfo(dtype).max
```

The PSD bound conservatively covers coefficient construction and a complete
inverse-transform accumulation. Nonfinite host evaluation or a failed bound is
rejected before random-word generation or target-sized allocation.

Within the accepted numerical tolerance, `sum_k(Q[k])` equals the supplied PSD
integral over `[0, fs / 2)`. Synthesis then deliberately sets the DC-cell power
to zero. It therefore retains `sum_{k=1}^K(P[k])`, discards exactly the
integrated power over `[0, df / 2)`, and performs no redistribution or global
renormalization. This is a finite record-length DC notch, not removal of only
the measure-zero point `f = 0`.

The private real DC coefficient is exactly zero, so every synthesized PSD
record has zero sample mean up to inverse-transform roundoff. It cannot add a
record-wide random voltage offset when `NoiseWaveform` is composed with the
zero-baseline `PureWaveform`; the digitizer transfer remains the sole owner of
the quiescent ADC-code placement. This exact-record rule applies specifically
to `PsdNoiseConfig`. An IID `WhiteNoiseConfig` realization may have an ordinary
finite-sample mean fluctuation and is not silently demeaned, because demeaning
would change its accepted IID covariance.

PSD-shaped noise uses one fixed finite-lattice Box-Muller one-sided coefficient
law targeting the ideal Gaussian equations below. For each leading-index
waveform row, let all `u[k]`, `v[k]`, and `z` values below be mutually
independent finite-lattice Box-Muller components targeting standard normals in
the selected output floating dtype. Define the interior index set as
`I = {1, ..., floor((N - 1) / 2)}`. The private one-sided coefficients are:

```text
X[0] = 0 + 0j

X[k] = (N / 2) * sqrt(P[k]) * (u[k] + i * v[k]),  k in I

if N is even:
    X[N / 2] = N * sqrt(P[N / 2]) * z + 0j
```

The odd-`N` terminal coefficient belongs to `I` and remains complex. The
even-`N` Nyquist coefficient is real. Its imaginary component and the DC
imaginary component are exactly zero rather than values left for `irfft` to
ignore. The implementation constructs the two parts of every interior
coefficient from two explicit real standard-normal
draws; it must not silently substitute a native complex-normal draw whose real
and imaginary components each have variance `1 / 2`.

The output is normatively:

```python
noise = torch.fft.irfft(X, n=N, dim=-1, norm="backward")
```

`N` is always explicit because odd record length cannot be recovered uniquely
from one-sided coefficient count. For floating `torch.float32` output, `X` is
`torch.complex64`; for `torch.float64`, it is `torch.complex128`.
`norm="backward"` is the Fourier normalization convention, not an autograd
instruction: the paired forward transform is unscaled and the inverse applies
the factor `1 / N`. With `K = floor(N / 2)`, the exact odd/even inverse
equations are:

```text
N = 2*K:
    x[n] = (1 / N) * (
        2 * Re(sum_{k=1}^{K-1}(X[k] * exp(i * 2*pi*k*n/N)))
        + X[K] * (-1)**n
    )

N = 2*K + 1:
    x[n] = (2 / N) * Re(
        sum_{k=1}^{K}(X[k] * exp(i * 2*pi*k*n/N))
    )
```

`X[0]` is absent from both right-hand sides because it is exactly zero. The
factor of two comes from the omitted negative-frequency conjugate of each
interior coefficient. DC and even-length Nyquist are self-conjugate and do not
receive that factor. This explains the `N / 2` interior scale and `N` Nyquist
scale above.

The frozen ideal-standard-normal statistical oracle is:

```text
E[x[n]] = 0
mean_n(x[n]) = 0                         # up to transform roundoff
Var[x[n]] = sum_{k=1}^K(P[k])

N = 2*K:
    Cov(x[n], x[n + lag]) =
        sum_{k=1}^{K-1}(P[k] * cos(2*pi*k*lag/N))
        + P[K] * (-1)**lag

N = 2*K + 1:
    Cov(x[n], x[n + lag]) =
        sum_{k=1}^{K}(P[k] * cos(2*pi*k*lag/N))
```

For every paired coefficient,
`Var(Re(X[k])) = Var(Im(X[k])) = N**2 * P[k] / 4` and
`E[abs(X[k])**2] = N**2 * P[k] / 2`. The even-length real Nyquist coefficient
has variance `N**2 * P[N / 2]`. Parseval's identity consequently gives
`E[mean_n(x[n]**2)] = sum_{k=1}^K(P[k])`.

The executed finite-lattice Box-Muller law targets these equations but does not
make them exact digital moments: discretized open uniforms and target-dtype
transcendentals introduce a small accepted deviation. Validation includes the
frozen numerical/lattice allowance and must not post-normalize generated values
to force the ideal equations exactly.

The covariance is circular in the sample index. Different waveform rows use
independent coefficient variates in the MVP, so their cross-covariance is zero
in expectation. Realized finite-record power fluctuates stochastically; only
its expectation equals retained integrated PSD power. The builder performs no
post-`irfft` demeaning, unit-standard-deviation normalization, power
normalization, discarded-DC redistribution, or independent `scale_mv`. Tiny
nonzero numerical sample means caused by transform roundoff are not corrected.

The DC notch has an observable finite-record consequence. For an accepted PSD
record (`N >= 2`) with constant input density `S`, the retained variance is
`S * fs * (N - 1) / (2 * N)`, every nonzero circular lag has covariance
`-S * df / 2`, and the corresponding correlation is `-1 / (N - 1)`. A
fixed-length flat-PSD realization is therefore exactly zero-sum and weakly
anticorrelated, not IID white. Callers who want IID samples select
`WhiteNoiseConfig`; the two accepted models are deliberately distinct.

The private coefficient frequencies `k / (N * dt)` are therefore never exposed
as semantic bin coordinates or caller-supplied edges.

The MVP synthesizes exactly the configured `N = sampling.sample_count` samples.
An exact-length inverse transform therefore realizes a periodic finite record
with circular/circulant covariance across the two window edges. It performs no
hidden longer-record generation, padding, overlap, crop, or baseline-bank
selection. This boundary is validated and recorded as an intentional
tensor-path divergence from IV's long-baseline generation and random crop. A
later padded/cropped model would be a different accepted noise algorithm, not
a quiet implementation substitution.

White-noise positions are the final waveform's logical tensor positions.
PSD-synthesis random positions instead belong to a defined private one-sided
coefficient shape: every source dimension except the sample dimension,
preserved in its current order, followed by the target frequency dimension.
That intermediate shape/order is part of the PSD-noise synthesis algorithm
version; its flat positions are not interchangeable with flat positions in the
time-domain output waveform. Consistent with the scalar-calibration rule, the
MVP uses the same white-noise RMS or PSD for every channel while drawing the
channels independently. Per-channel spectra and cross-channel spectral
correlation require a later typed tensor-calibration/input contract.

The exact Stage 5 noise lattices use `q = 0` throughout:

```text
white:
    key = WHITE_NOISE_RNG_KEY
    p = 0, ..., output.numel() - 1

PSD, F = floor(N / 2) + 1:
    key = PSD_NOISE_RNG_KEY
    p = row * F + k,  k = 1, ..., floor(N / 2)
```

PSD row order is the private coefficient order defined above. `k = 0` has no
address and requests no draw. Every positive-frequency position keeps its
fixed address even when its represented power is zero; an eager implementation
may evaluate that draw and then force the coefficient to exact zero. Interior
positions use ordered `(z0, z1)` for real and imaginary components. An even
Nyquist position uses `z0` and discards `z1`.

The initial noise producer supports vectorized eager CPU execution and eager
CUDA execution only when that CUDA path has passed its required evidence.
MPS, Meta, compiled, Triton, and custom-kernel execution are outside Stage 5.
An unsupported device is rejected before output allocation or random-word
generation; the producer never moves an input or generated payload through the
host. Identical accepted inputs reproduce exactly on the same unchanged
numerical execution stack. Raw Threefry words and fixed-point uniform
conversions must agree exactly between accepted CPU and CUDA implementations,
while completed Box-Muller and PSD values require cross-backend statistical
agreement rather than bitwise identity because transcendental and FFT
implementations may differ.

`prepare_noise_waveform(...)` owns the contextual preparation needed for its
algorithm. It requires exact sampling/source agreement, an exact
`torch.float32` or `torch.float64` output dtype, and an accepted CPU/CUDA
device. `produce_noise_waveform(...)` receives the accepted Runtime and an
accepted `CounterRng`. `ZeroNoiseConfig` produces fresh exact zeros without
invoking the RNG; preparation gives white and PSD models their exact fixed
package-owned `RngKey`. Intrinsic config validity remains owned by the frozen config
constructors, and the private producer does not defend against fabricated
private objects or constructor bypass. All contextual and numeric preparation
completes before an RNG request, product-producer invocation, or semantic-output
write.

### Analog Waveform

```text
analog[i] = clamp(
    pure[i] + noise[i],
    optional_minimum_mv,
    optional_maximum_mv,
)
```

Pure and noise must have equal axes, device, dtype, shape, and mV reference
plane. One optional pair of scalar saturation bounds applies to every channel
and example. No implicit broadcast or coordinate-dependent limit lookup is
accepted. This clamp models physical analog/front-end saturation. It belongs
inside `produce_analog_waveform(...)` and is distinct from the finite ADC code
range. An absent lower or upper bound leaves that side unbounded; with no
bounds the equation reduces to `pure + noise`.

The producer evaluates the eager equation in the common input dtype and device.
Config-derived bounds are converted through that dtype and checked for finite
representability before payload calculation; two present bounds must remain
strictly ordered after conversion. The exact rounded values are materialized
as zero-dimensional tensors on the input device and used by the clamp. It
adopts the exact `pure.axes` tuple after requiring
`noise.axes == pure.axes`, equal device, and equal dtype. Autograd is preserved
through addition and saturation according to ordinary Torch clamp semantics;
no derivative is promised exactly at a saturation boundary.

The MVP introduces no deterministic analog baseline or pedestal.
`PureWaveform` is a signal excursion from 0 mV, `NoiseWaveform` is a stochastic
voltage fluctuation about 0 mV, and `AnalogWaveform` is their zero-referenced
composition. PSD-shaped noise has an exactly zero synthesized DC coefficient;
IID white noise remains ensemble-centered and may have an ordinary
finite-record mean fluctuation. Neither case carries a configured pedestal.

If a later detector model needs a physical front-end bias, a channel-dependent
quiescent voltage, or a separately retained baseline waveform, that effect
belongs explicitly in the analog stage before saturation. It may become an
`AnalogWaveformConfig` component or a distinct typed input/product after a
focused Design decision; it must not be hidden in `NoiseWaveform`. Time-varying
baseline wander may remain a named stochastic noise submodel when that is its
actual physical meaning.

### Digitization

The `AnalogWaveform` is expressed at the pre-digitizer-gain mV reference plane.
`input_min_mv` and `input_max_mv` describe the digitizer's post-gain analog
input range. Preflight computes:

```text
maximum_code = 2**bit_depth - 1
gain = 10**(analog_gain_db / 20)
span = input_max_mv - input_min_mv
slope = gain * maximum_code / span
intercept = -input_min_mv * maximum_code / span
lower_input_mv = input_min_mv / gain
upper_input_mv = input_max_mv / gain
```

These scalar constants are derived once in Python binary64, required to be
finite and representable in the analog input dtype, and then used by payload
arithmetic in that dtype and device. The rounded thresholds must remain
strictly ordered. Preflight preserves the exact rounded thresholds,
`maximum_code`, slope, and intercept and materializes them as zero-dimensional
tensors on the input device, so payload arithmetic uses the values that
validation accepted. This scalar preparation does not move or materialize the
analog payload on the host.

The product producer then evaluates:

```text
interior[i] = clamp(
    analog[i] * slope + intercept,
    0,
    maximum_code,
)

code_float[i] =
    0,                         if analog[i] <= lower_input_mv
    maximum_code,              if analog[i] >= upper_input_mv
    interior[i],               otherwise

digitized[i] = int32(code_float[i])
```

Bit depth is in `[1, 16]`, analog gain is in `[0, 40]` dB, and output is
nonnegative `torch.int32`. The endpoint comparisons occur directly in the
pre-gain analog domain. They are inclusive at the exact field-dtype thresholds
and avoid a one-code endpoint loss caused by floating affine rounding.
Clipping and endpoint selection precede conversion; unsigned wraparound is
forbidden. One scalar gain and voltage-transfer range applies to every channel
and example. Conversion of a nonnegative open-interior value truncates toward
zero, which is the accepted ADC quantization rule. No separate pedestal is
needed: an asymmetric input range determines the code corresponding to 0 mV.
That nonzero zero-voltage code is an ADC transfer property, not a baseline
voltage added to `AnalogWaveform`. Digitization is not declared
differentiable.

For scientific readability and validation, the algebraically equivalent
unfused reference transfer remains:

```text
gained_mv = analog_mv * gain
clipped_mv = clamp(gained_mv, input_min_mv, input_max_mv)
scaled_code = (
    (clipped_mv - input_min_mv)
    / span
    * maximum_code
)
interior_reference_code = int32(clamp(scaled_code, 0, maximum_code))

reference_code =
    0,                         if analog_mv <= lower_input_mv
    maximum_code,              if analog_mv >= upper_input_mv
    interior_reference_code,   otherwise
```

The gained/clipped and affine interiors are algebraically equivalent in real
arithmetic but need not be floating-point identical near code transitions.
The production expression using precomputed `slope` and `intercept` plus the
same endpoint guards is the normative execution form. Validation covers exact
dtype-rounded endpoints, code-transition neighborhoods, and the accepted
dtype/backend arithmetic rather than assuming cross-backend bitwise identity
under floating multiply-add reassociation. Its pre-conversion clamp and guards
intentionally fix IV-DSLab's cast-before-clip wraparound and endpoint-rounding
defects.

Because TensorCore leaves are fieldless, a bare `DigitizedWaveform` does not
carry variable bit depth, gain, or voltage transfer. The builder guarantees its
codes against the supplied `DigitizedWaveformConfig`, and the caller retains
that config for interpretation. Before durable IO or an independent
cross-process digitized handoff, a focused design must bind the config or an
equivalent versioned calibration record to the artifact.

## RNG And Positional Repeatability

The accepted model separates three facts:

```text
CounterRng instance = RNG algorithm + invocation seed
RngKey              = one independent stochastic role
position/quantum    = TensorDSLab scientific address coordinates
ordinal schedule    = direct-distribution call coordinates or
                      TensorCore-owned count-distribution internals
```

`CounterRng` is an immutable stateless TensorCore abstraction. Reusing one
instance intentionally reproduces the same positional realization; it never
advances a hidden counter and contains no mutable cache, tensor, workspace,
device stream, or scratch. Conforming implementations may therefore be reused
concurrently. A different accepted RNG algorithm intentionally need not
produce the same realization.

`RngKey` values live in exact TensorDSLab leaf configs because role identity is
part of the scientific stochastic specification. The invocation seed and
algorithm do not live in config. Existing default key assignments never depend
on requested products or execution order and must not be renumbered. Do not
derive keys with declaration order, Python `hash()`, request order, or
branch-dependent sequential consumption.

The random engine is general over arbitrary tensor rank and shape. Its logical
address is:

```text
CounterRng seed and algorithm
RngKey namespace and stream
logical flat tensor position
local source-quantum ordinal, when required
direct floating-distribution value ordinal, or a TensorCore-owned
count-distribution internal ordinal
```

For shape `(n0, n1, ..., nk)`, logical flat position is the ordinary row-major
flattening of the tensor's current dimension order, conceptually
`(((i0 * n1 + i1) * n2 + i2) * ... + ik)`. It is not a physical storage offset
and does not depend on strides. No axis class, coordinate string, timestamp,
product label, or physical time enters the random address. An operation may
still locate the sample dimension during preflight and use numeric sample
period or index for its physics; that is separate from random identity.

A rank-zero scalar has one logical position, `0`. A shape with any zero-sized
dimension has no logical positions and consumes no draws. Valid readout fields
have nonempty axes, so scalar and empty cases belong to private generic RNG
primitive tests rather than the public readout collection contract.

Conceptually, a one-raw-word-per-position random field is:

```python
positions = RngPositions.from_shape(tensor.shape, device=tensor.device)
random_field = rng.uniform(
    key=ROLE_RNG_KEY,
    positions=positions,
    dtype=tensor.dtype,
    ordinal=0,
    count=1,
)
```

TensorCore's `logical_positions(...)` produces row-major linear `torch.int64`
positions reshaped to the requested shape on the requested device. It supports
scalar and zero-sized shapes and depends on dimension order and shape, not
coordinates, strides, contiguity, or storage addresses. A functionality-first
implementation may materialize this tensor. A later fused implementation may
derive the same positions from thread/index arithmetic without changing their
identity.

An iterative stochastic role may extend this same positional rule with a
virtual leading iteration dimension when it must distinguish repeated draws at
the same tensor position. For a fixed per-iteration private lattice of size
`N`, zero-based global iteration `j`, and row-major local position `u`, use:

```text
p = j * N + u
```

The virtual dimension is conceptual and is never materialized. Its meaning,
the role's exact private shape and dimension order, and `N` are frozen with the
stochastic algorithm. Preflight uses checked host arithmetic and requires the
maximum processed iteration count `G` to satisfy `G * N <= 2**63`. It never
uses a block-local iteration, active-only compaction, semantic coordinate,
timestamp, label, or execution order. Ordinary noniterative tensor roles are
the special case `G = 1` and `p = u`.

The fixed-`K` correlated-avalanche simulation uses this rule directly. For the
draws that produce offspring generation `g + 1`, `j = g` with
`0 <= g < K`. Each stochastic role has its own fixed package-owned key and
fixed
per-generation lattice. Retained DiCT/DeCT Poisson fields use destination-grid
positions; their overflow fields use source-grid positions. In either case,
`p = g * N + u` and `q = 0`; TensorCore's Poisson mapping owns the attempt-to-
word schedule. A delay-category dimension, when required by a non-Poisson
role, is part of that role's frozen row-major lattice. Terminal smearing owns a
separate key and
noniterative product-grid lattice. A zero frontier requests no words
and may skip physical work, but it never derives later addresses from active-
only compaction. Preflight applies the role-specific relational address bounds
below: `K*N` for each effective CT role and `K*(S+1)*N` for effective AP.

### Required TensorCore RNG Surface And Threefry Address Schema

TensorDSLab requires TensorCore to expose the following public concepts before
Maintenance 2:

```text
RngKey(namespace, stream)
CounterRng(seed)
Threefry4x32(seed)
logical_positions(shape, device=...)
```

`CounterRng` must provide public `uniform(...)`, parameterized
`gaussian(...)`, `poisson(...)`, and `binomial(...)` methods. Uniform and
Gaussian requests expose explicit `RngKey`, positions, quantum, ordinal, and
count coordinates. `count == 1` returns `positions.shape`; larger counts append
one final count dimension. Poisson and binomial expose no public ordinal or
count: TensorCore owns their complete internal inversion/rejection word
schedules.

Public floating dtypes are `torch.float32` and `torch.float64`. Closed-open
uniform includes exact zero and excludes one; open-open uniform excludes both
endpoints. Gaussian accepts a mean and standard deviation rather than exposing
a public standard-normal primitive. TensorCore owns the protected raw-word
generation mechanism, validation of generic address components, logical
positions, fixed-point conversions, Box-Muller mapping, Gaussian affine
mapping, Poisson inversion/PTRS, binomial inversion/BTRS, generic sampler
domains, and deterministic exhaustion. TensorDSLab uses only the public
TensorCore surface.

The required public shape is:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RngKey:
    namespace: int
    stream: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterRng(ABC):
    seed: int

    @final
    def uniform(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        dtype: torch.dtype,
        quantum: int = 0,
        ordinal: int = 0,
        count: int = 1,
        include_zero: bool = True,
    ) -> torch.Tensor:
        ...

    @final
    def gaussian(
        self,
        *,
        mean: float | torch.Tensor,
        standard_deviation: float | torch.Tensor,
        key: RngKey,
        positions: torch.Tensor,
        dtype: torch.dtype,
        quantum: int = 0,
        ordinal: int = 0,
        count: int = 1,
    ) -> torch.Tensor:
        ...

    @final
    def poisson(
        self,
        *,
        mean: float | torch.Tensor,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int = 0,
    ) -> torch.Tensor:
        ...

    @final
    def binomial(
        self,
        *,
        counts: torch.Tensor,
        success_mass: float | torch.Tensor,
        failure_mass: float | torch.Tensor | None = None,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int = 0,
    ) -> torch.Tensor:
        ...


@final
class Threefry4x32(CounterRng):
    __slots__ = ()


def logical_positions(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    ...
```

Exact TensorCore constructor mechanics, protected-method spelling, exception
types, accepted device matrix, and cross-backend evidence remain TensorCore
Design decisions. TensorDSLab requires exact non-boolean uint32 key
components, an exact non-boolean uint64 seed, positive floating-distribution
`count`, nonnegative address components within the accepted packing,
same-device results, and the public value/shape behavior above.

For both `uniform(...)` and `gaussian(...)`, checked slice bounds require
`ordinal + count <= 2**34` for `torch.float32` and
`ordinal + count <= 2**33` for `torch.float64`. TensorCore also checks the
appended result shape and signed Torch `numel` range before word generation or
allocation.

For `gaussian(...)`, scalar parameters are finite exact Python `float` values
—not bool, int, NumPy scalar, or another coercible object—rounded once to the
requested dtype. Tensor parameters have exactly
`positions.shape`, the requested dtype, and the positions device. The only
implicit expansion is across the final variate dimension when `count > 1`;
arbitrary broadcasting, movement, or casting is not accepted. Standard
deviation is finite and nonnegative after representation. A positive Python
value that rounds to represented zero follows the zero-standard-deviation
branch. Every requested position/variate owns its ordinary Gaussian address
even when its represented standard deviation is zero; draw-free model branches
skip the entire call before reaching TensorCore. This preserves the existing
full-grid charge-smearing and PSD address contracts.

Let `Z` be the internal standard-normal value. The frozen affine mapping is:

```text
standard_deviation == 0                -> mean
standard_deviation == 1 and mean == 0  -> Z
mean == 0                              -> standard_deviation * Z
otherwise                              -> mean + standard_deviation * Z
```

These branches apply elementwise after dtype representation and final-count
expansion. The zero-standard-deviation branch returns the represented mean
exactly, but invoking `gaussian(...)` still owns and requests the complete
ordinary Gaussian address lattice; callers skip the whole method only when a
model branch is structurally or numerically disabled. The general path
multiplies before it adds. These exact zero-scale, identity, and zero-mean
branches preserve the Stage 5/6 white-noise, PSD-coefficient, and charge-
smearing executable mappings without making standard normal a separate public
method.

The initial public contract rejects law tensors with `requires_grad=True` and
returns `requires_grad=False`; pathwise stochastic autograd requires a later
focused TensorCore design rather than value-dependent shortcuts that silently
drop gradients. Before word generation, TensorCore also checks the represented
law against the maximum Box-Muller radius of the selected fixed-point lattice:
every element must satisfy a conservative finite-output envelope such as
`abs(mean) + standard_deviation * maximum_radius <= finfo(dtype).max`.
Accepted Gaussian output is therefore finite. TensorDSLab retains its stricter
model-specific ledger check and additionally preflights the prepared
target-dtype Charge-smearing scale against this public TensorCore envelope.
The supported contextual domain is the intersection of those two contracts.

For `poisson(...)`, `positions` fixes shape and device. A scalar mean fills
that exact shape; a tensor mean is exact-shaped/device-matched
`torch.float64`. Scalar means and masses are exact Python `float` values—not
bool, int, NumPy scalar, or another coercible object—and therefore enter the
count laws as binary64. Poisson means are finite in `[0, 1e8]`. For
`binomial(...)`, `counts` and `positions` are shape/device-matched
`torch.int64`, every count lies in `[0, 2**53 - 1]`, and scalar masses or
exact-shaped/device-matched `torch.float64` masses are accepted independently
without arbitrary broadcasting. When `failure_mass is None`, `success_mass`
is a finite normalized probability in `[0, 1]` and TensorCore derives the
represented `1 - success_mass`. When both masses are supplied, each is finite
and in `[0, 1]`, TensorCore uses them directly as relative success and failure
mass, and they need not sum to one. Their total must be positive wherever the
remaining count is positive; both may be zero only where the count is zero.
Exact `count == 0`, represented probability zero, and represented probability
one paths are word-free. TensorDSLab always supplies its independently prepared
later-category mass as `failure_mass`.

All four distribution methods return fresh, non-aliasing, contiguous
`torch.strided` tensors on the positions device, including word-free
deterministic paths. Floating results use the requested dtype; Poisson and
binomial results use `torch.int64`. They never mutate positions or law tensors,
silently move or cast an input, or expose a partially written result.
Validation completes before word generation and distribution-result writes.
Backend execution remains subject to the ordinary selected-device
synchronization semantics.

Distribution ordinals count returned variates, not raw words. The frozen
mapping is:

```text
float32 uniform v -> raw word v
float64 uniform v -> raw words 2v, 2v + 1

internal standard-normal variate v:
  pair      = v // 2
  component = v % 2

float32 normal pair j -> raw words 2j, 2j + 1
float64 normal pair j -> raw words 4j, 4j + 1, 4j + 2, 4j + 3
```

Consecutive internal-normal order is `z0(pair 0)`, `z1(pair 0)`,
`z0(pair 1)`, `z1(pair 1)`. Any Gaussian `(ordinal, count)` slice must equal
stacking the corresponding scalar requests, including odd starts. Current
consumers map as:

```text
white noise       -> gaussian(mean=0.0, standard_deviation=rms, ordinal=0, count=1)
charge smearing   -> gaussian(mean=S1, standard_deviation=sigma*sqrt(S2), ordinal=0, count=1)
PSD coefficient   -> gaussian(mean=0.0, standard_deviation=1.0, ordinal=0, count=2)
```

For each returned variate `j`, the law is fixed as:

```text
result[..., j] uses
    mean[position]
    standard_deviation[position]
    Z(key, position, quantum, ordinal + j)
```

Tensor law parameters never include the appended count dimension. Scalar law
parameters apply to every position and returned variate. When `count == 1`,
the conceptual `j = 0` dimension is elided and the returned shape is exactly
`positions.shape`.

Scalar consumers discard their unused `z1`; it is never reassigned to another
role. Poisson and binomial own their internal inversion/PTRS/BTRS schedules:
attempt `a` maps to the same two open-open binary64 uniforms and raw words as
Stage 6, but callers do not supply that ordinal. `quantum` remains the
operation's source-quantum coordinate and is never reused as a rejection-
attempt index.

For every distribution, the `RngKey` is the caller-owned domain separator.
Reusing one exact `(key, position, quantum)` address for different methods or
for repeated Poisson/binomial calls intentionally reuses or overlaps raw words;
unused lanes and rejection-attempt blocks are never available to a second
scientific role. TensorDSLab's closure-wide distinct-role key check prevents
that overlap in the public readout graph.

The required `Threefry4x32` implementation is standard Random123
`Threefry4x32` with exactly 20 rounds. The normative external definition is
Random123 `1.14.0` at commit
[`726a093`](https://github.com/DEShawResearch/random123/commit/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13),
specifically its
[`threefry.h`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/include/Random123/threefry.h)
`Threefry4x32_R<20>` algorithm and
[`kat_vectors`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/tests/kat_vectors)
word order. This freezes the standard rotation constants, key-injection
schedule, output tuple order, and `0x1bd1_1bda` parity constant. A reduced-round
variant, PyTorch-internal approximation, or different output-word order is not
the same algorithm.

The selection rationale is explicit. The
[Random123 paper](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
establishes the counter-based parallel-RNG model, while the
[JAX PRNG design](https://docs.jax.dev/en/latest/jep/263-prng.html) is
supporting evidence that a functional, array-oriented Threefry counter model
removes global-state sequencing and vectorizes over logical counters. Neither
source defines TensorDSLab's address schema. `Threefry4x32` was selected for v1
because its 128-bit key and 128-bit counter represent the accepted seed,
stream, domain, logical-position, quantum, and raw-word-block coordinates
without a derived-key collision argument, and its add/rotate/XOR core has a
clear masked-`int64` Torch reference. Philox remains a possible later
throughput challenger, but changing the algorithm or packing would require an
explicitly versioned RNG contract rather than silently changing v1 streams.

The concrete public class name `Threefry4x32` permanently means the accepted
20-round v1 behavior. An incompatible word algorithm or address mapping
requires a new type/version rather than silently changing existing
realizations. The historical Stage 5 identifier is:

```text
tensordslab.threefry4x32-20/v1
```

Maintenance 2 preserves that exact default-key word mapping through the
TensorCore public class. Callers construct the RNG object but never construct
raw counters or request protected raw words.

For invocation seed `s`, package-owned
`RngKey(namespace=d, stream=g)`, logical flat position `p`,
source-quantum ordinal `q`, and zero-based raw-word ordinal `r`, define:

```text
low32(x)  = x & 0xffff_ffff
high32(x) = (x >> 32) & 0xffff_ffff

block = r // 4
lane  = r % 4
```

Schema v1 packs words numerically as:

```text
key[0] = low32(s)
key[1] = high32(s)
key[2] = g
key[3] = d

counter[0] = low32(p)
counter[1] = high32(p)
counter[2] = q
counter[3] = block

words = Threefry4x32_20(counter, key)
raw_word = words[lane]
```

The low/high split is numerical and independent of host byte order. `lane`
selects Random123 output tuple member `v[0]` through `v[3]` in its declared
order. The exact accepted bounds are:

```text
0 <= seed < 2**64
0 <= namespace < 2**32
0 <= stream < 2**32
0 <= logical_flat_position < 2**63
0 <= source_quantum_ordinal < 2**32
0 <= raw_word_ordinal < 2**34
```

The counter encoding itself has two position words, but the accepted execution
contract uses the stricter signed Torch indexing/`numel` range. A per-quantum
algorithm therefore accepts at most `2**32` source quanta in one cell, with
ordinals `0` through `2**32 - 1`. Each block supplies four lanes, so the
raw-word bound makes `block` an exact unsigned 32-bit value. Input counts remain
`torch.int64`; the smaller per-cell quantum population limit is checked
explicitly rather than reached through a narrowing cast.

This encoding is injective over the accepted address domain: equal encoded
key, counter, and lane recover the same seed, namespace, stream, position,
quantum, and raw-word ordinal. That is an address-collision statement, not an
output-value uniqueness claim. Distinct addresses may naturally return the
same 32-bit raw word.

Every independently specified random field or stochastic substep owns one
exact `RngKey`. Cell-level operations use `q = 0`; per-source-quantum
operations use the ordinal within that source cell. A role must not mix those
two meanings. The ten accepted fixed role keys are listed under
[Fixed Package-Owned RNG Keys](#fixed-package-owned-rng-keys). They use namespace `TDS1`,
retain the Stage 5/6 stream values `1` through `10`, and never derive from
declaration order, requested-product order, execution order, `Enum.auto()`, or
Python `hash()`.

Closed Stage 6 production represented those values through the private
`readout._random._RngStream` enum. The closed Maintenance 2 implementation
removed that enum and module without a compatibility shim after selecting the
required TensorCore commit. Its then-default config keys reproduce the
existing Stage 5/6 raw-word addresses exactly; Maintenance 7 fixes the same
keys in one private table. Numeric stream order records append-only
identity, not physical execution order: timing jitter uses stream `8` while
still executing before the correlated-avalanche roles. One AP key owns its
complete coupled categorical realization; separate AP keys would incorrectly
break that coupling.

For seed zero, namespace `TDS1`, logical position zero, source quantum zero,
and block zero, the independent scalar Threefry oracle fixes these two
default-key blocks:

```text
afterpulse role key:      1f53a380 e9f15c80 6113c5f0 dd68b867
charge-smearing role key: 5f643fe4 c4c88a72 a83fd264 a1443af3
```

Threefry operates on mathematical unsigned 32-bit words. TensorCore owns the
reference and any optimized implementation, including carrier dtype, masking,
rotation, and backend details. TensorDSLab relies only on the public
distribution results and the frozen address mapping; it does not import or
test TensorCore's protected raw-word implementation.

The integer core has a strong boundary: every accepted TensorCore
implementation must return identical four-word output for an identical key
and counter. Stage 5
accepts one vectorized eager CPU implementation and conditionally accepts its
vectorized eager CUDA path when CUDA evidence is available; an independent
scalar implementation is a validation oracle rather than a production
execution mode. Compiled, Triton, MPS, and custom-kernel paths are not accepted
Stage 5 execution modes. TensorCore owns the authoritative known-answer and
cross-implementation tests retained by the closed Maintenance 2 implementation.
The initial fixed Random123
oracles include:

```text
counter: 00000000 00000000 00000000 00000000
key:     00000000 00000000 00000000 00000000
output:  9c6ca96a e17eae66 fc10ecd4 5256a7d8

counter: ffffffff ffffffff ffffffff ffffffff
key:     ffffffff ffffffff ffffffff ffffffff
output:  2a881696 57012287 f6c7446e a16a6732

counter: 243f6a88 85a308d3 13198a2e 03707344
key:     a4093822 299f31d0 082efa98 ec4e6c89
output:  59cd1dbb b8879579 86b5d00c ac8b6d84
```

Raw-word identity does not automatically make floating distribution transforms
bitwise identical across backends. Stage 5 requires exact accepted-CPU/CUDA
agreement for fixed-point uniform conversion, exact same-stack repeatability
for Box-Muller and completed noise products, and cross-backend statistical
agreement for completed Gaussian and PSD values. Bernoulli, exponential,
Poisson, categorical, and rejection behavior are not Stage 5 implementation
claims. Stage 6 subsequently implemented and validated the aggregate-binomial,
multinomial, and hybrid Poisson contracts on eager CPU; CUDA was unavailable,
so their cross-backend evidence remains unestablished.

For completed operations involving transcendentals, “same backend” is
shorthand for one unchanged numerical execution stack: OS and architecture,
Python and PyTorch build, backend/device implementation, eager execution mode,
dtype, and relevant math settings. Two systems both described as CPU are not
therefore bitwise equivalent. Exact literal fixtures are owned only by their
recorded stack; another accepted stack proves exact replay within itself plus
the applicable invariants and statistical laws. This qualification does not
weaken the separately documented exact raw-word or fixed-point-uniform scope.

Neither package may read or mutate PyTorch's global RNG state, create a
`torch.Generator`, use `torch.poisson` as the normative sampler, or depend on
private PyTorch RNG operations. TensorCore owns generic counter and
distribution mechanics. TensorDSLab owns scientific position/category
lattices, config-key assignment, complete multinomial orchestration, checked
count accumulation, mechanism bookkeeping, and physical ledgers.

### Selected RNG Distribution Contracts

Stage 5 implements the precision-matched uniform conversions and Box-Muller
mapping used by white and PSD noise. Stage 6 implements the Poisson and
aggregate-binomial contracts below inside TensorDSLab. Those closed stages
remain exact historical production evidence. The Maintenance 2 implementation
uses the same generic Gaussian, Poisson, and binomial mappings promoted to
TensorCore without changing their default-key results; TensorDSLab retains
only their scientific use and complete multinomial orchestration. The
standalone Bernoulli threshold and continuous exponential inversion remain
recorded generic candidates but have no accepted MVP consumer: AP uses
aggregate conditional binomials, and physical delay laws are integrated into
prepared categories.
Noise and continuous product-dtype transforms use the conventional precision-
matched Random123 fixed-point conversions rather than a widened `float64` path
for `float32` products. Discrete count probabilities, Poisson means, and Poisson
sampler control intentionally use binary64 so the integer avalanche history
does not depend on the requested Charge dtype. The normative
uniform-conversion reference is Random123 `1.14.0`
[`u01fixedpt.h`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/include/Random123/u01fixedpt.h).
This keeps the GPU path simple and makes the finite precision and tail limits
explicit.

For one raw word `w0`, the `float32` conversions are:

```text
m24 = w0 >> 8
m23 = w0 >> 9

U32[0, 1) = float32(m24) * 2**-24
U32(0, 1) = (float32(0.5) + float32(m23)) * float32(2**-23)
```

`U32[0, 1)` ranges from zero through `1 - 2**-24`; `U32(0, 1)` is the
Random123 midpoint lattice from `2**-24` through `1 - 2**-24`. The closed-open
conversion discards the lower eight raw bits; the open-open conversion discards
the lower nine. Discarded bits are never reused.

A `float64` uniform consumes two consecutive raw words. The earlier word is
the numerical high word, independent of host byte order:

```text
m53 = w0 * 2**21 + (w1 >> 11)
m52 = w0 * 2**20 + (w1 >> 12)

U64[0, 1) = float64(m53) * 2**-53
U64(0, 1) = (float64(0.5) + float64(m52)) * float64(2**-52)
```

This arithmetic assembly stays within signed `torch.int64`; it does not first
construct an overflowing unsigned 64-bit carrier. `U64[0, 1)` ranges from zero
through `1 - 2**-53`; `U64(0, 1)` is the Random123 midpoint lattice from
`2**-53` through `1 - 2**-53`. The closed-open conversion discards the lower
eleven bits of `w1`; the open-open conversion discards the lower twelve.
Discarded bits are never reused.

### Aggregate Multinomial Sampling

Timing redistribution and AP placement sample aggregate cell counts rather
than expanding individual avalanches. TensorDSLab owns one ordered multinomial
orchestration realized through sequential calls to TensorCore's public
`CounterRng.binomial(...)`. Each scientific law prepares two stable binary64
masses for every category except the final remainder: `A[c]` is that category's
mass and `B[c]` is the combined mass of every later category, including the
remainder. For total count `n` and remaining count `n_r`:

```text
n_r = n

for category c = 0 .. C - 2:
    x[c] = rng.binomial(
        counts=n_r,
        success_mass=A[c],
        failure_mass=B[c],
        key=role_key,
        positions=category_positions[c],
        quantum=0,
    )
    n_r -= x[c]

x[C - 1] = n_r
```

The final category is an exact remainder and consumes no draw. Preflight
constructs and validates every mass in binary64, uses stable analytic
cumulative tails or equivalent prepared remaining masses, and rejects a law
outside its explicit numerical contract. It never obtains `B` by repeatedly
subtracting rounded categories from one, silently clips, or renormalizes a
malformed law. Exact `n = 0`, `A = 0`, and `B = 0` are draw-free paths. If
`A == B`, the step is not complemented. If `A == B == 0`, `n_r` must already
be zero.

For a nontrivial conditional draw, TensorCore derives the reduced `p_star` in
`(0, 0.5]` and strict complement decision from the two supplied masses. This
is the stable realization of `min(p, 1 - p)`; it does not form `1 - p` from a
rounded near-one value. For
`n * p_star < 10`, it uses one `U64[0, 1)` and binary64 forward-CDF inversion.
For `n * p_star >= 10`, it uses Hoermann BTRS. The exact `n = 0`, `A = 0`, and
`B = 0` paths are handled before this core and request no raw word. Reflection
remains strict: only `B < A` returns `n - k` after a candidate has been
accepted; equal masses are not reflected. These are aggregate cell draws with
`source_quantum = 0`.

The small-mean inversion path uses raw-word block zero, lanes zero and one, to
construct its one `U64[0, 1)` value. With `q = 1 - p_star`, all arithmetic
below is binary64:

```text
f = exp(float64(n) * log1p(-p_star))  # P(X = 0)
cumulative = f
last_k = min(n, 63)

for k = 0 .. last_k:
    if U < cumulative:
        accept k
    if k == last_k:
        fail by deterministic inversion exhaustion
    f *= (float64(n - k) / float64(k + 1)) * (p_star / q)
    cumulative += f
```

The strict comparison makes the returned value the first `k` whose represented
CDF is greater than `U`. The guard covers at most the 64 probability terms
`k = 0 .. min(n, 63)` and never evaluates beyond the binomial support. A
rounded top-lattice CDF may accept the maximum closed-open uniform; if no
represented term accepts it, exhaustion remains the specified hard failure
rather than a clamp to `63` or `n`.

Across the complete supported inversion domain, every represented probability
term and cumulative CDF value must agree with an independent at-least-
80-decimal-digit Binomial oracle for the same represented `n` and `p_star`
within `1e-12` absolute error. This is a local executable-mapping gate, not a
Monte Carlo tolerance.

TensorCore's large-mean path uses the short transformed-rejection BTRS
algorithm from
[Hoermann's binomial paper](https://doi.org/10.1080/00949659308811496), with a
cancellation-resistant algebraic regrouping of the corrected log-domain
acceptance form used by
[PyTorch 2.12.1](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/native/Distributions.h).
The earlier tentative BTRD choice is retired. BTRD's decomposition primarily
reduces the number of uniform variates; the accepted address mapping already
reserves one whole Threefry block, containing exactly two binary64 uniforms,
for every addressed attempt. BTRS therefore uses the same block schedule and
central fast-accept probability without BTRD's variable-consumption and
near-/far-mode branches.

For `p_star` in `(0, 0.5]`, BTRS precomputes in binary64:

```text
s     = sqrt(float64(n) * p_star * (1 - p_star))
b     = 1.15 + 2.53 * s
a     = -0.0873 + 0.0248 * b + 0.01 * p_star
c     = float64(n) * p_star + 0.5
v_r   = 0.92 - 4.2 / b
r     = p_star / (1 - p_star)
alpha = (2.83 + 5.1 / b) * s
m     = int64(floor(float64(n + 1) * p_star))
```

Attempt `j = 0 .. 63` owns raw-word block `j`. Lanes zero and one construct
the first `U64(0, 1)` value `u_0`; lanes two and three construct the second
value `v`. The proposal and first acceptance region are:

```text
u = u_0 - 0.5
u_s = 0.5 - abs(u)
k_f = floor((2 * a / u_s + b) * u + c)

if k_f is nonfinite or k_f < 0 or k_f > n:
    reject this attempt
else:
    k = int64(k_f)

if u_s >= 0.07 and v <= v_r:
    accept k
```

The support check precedes both integer conversion and quick acceptance. For a
candidate outside the quick-accept region, define the displacement from the
center and three cancellation-resistant logarithms:

```text
ell = log(v * alpha / (a / (u_s * u_s) + b))

d = k - m
log_left = log1p(d / (n - k + 1))
log_right = log1p(-d / (k + 1))
log_ratio = log(r * (n - k + 1) / (k + 1))

main = (
    (n - m + 0.5) * log_left
    + (m + 0.5) * log_right
) + d * log_ratio
correction = ((fc(m) + fc(n - m)) - fc(k)) - fc(n - k)
upper_bound = main + correction

accept k if ell <= upper_bound
```

This main term is algebraically identical in real arithmetic to the earlier
three-log Hoermann/PyTorch expression, but avoids subtracting large nearly
equal terms when `n` is large and `k` is near `m`. For every supported
`0 <= k <= n`, both `log1p` arguments are strictly greater than `-1`. The
displayed parentheses and operation order are normative binary64 reference
behavior.

`fc` is the binary64 Stirling-tail correction. Its first ten frozen decimal
literals, each rounded to binary64 by ordinary Torch construction, are:

```text
0.0810614667953272,  0.0413406959554092,
0.0276779256849983,  0.02079067210376509,
0.0166446911898211,  0.0138761288230707,
0.0118967099458917,  0.0104112652619720,
0.00925546218271273, 0.00833056343336287
```

For integer `j >= 10`:

```text
x = float64(j + 1)
x2 = x * x
inner = (1/360) - ((1/1260) / x2)
fc(j) = ((1/12) - (inner / x2)) / x
```

After attempts `0 .. 63` reject, BTRS fails by deterministic exhaustion. Both
inversion and BTRS forbid reseeding, approximation, clipping, dependency RNG,
or an alternate fallback. The supported domain below ensures that every
accepted `n`, `n + 1`, proposal, and accepted candidate is exactly and safely
representable under this binary64/int64 mapping.

The assignments, displayed parentheses, and left-to-right sum grouping above
are normative for the eager reference. A later compiled or fused mode may not
reassociate them while claiming the same executable stream. This freezes a
finite binary64 BTRS mapping; it does **not** claim mathematically exact
sampling from the ideal real-arithmetic Binomial law for every supported
parameter. The published Stirling-tail evaluation is approximate, and finite
rounding can change a decision arbitrarily close to an acceptance boundary.
Independent at-least-80-decimal-digit Design sweeps through
`n = 2**53 - 1`, from the `n*p_star = 10` crossover through `p_star = 0.5`,
kept central candidates through 25 standard deviations within `1e-6` absolute
local log-bound error. Complete-support validation uses the same scale-aware
rule as PTRS. For high-precision reference side `x`, define:

```text
allowance(x) = 1e-6 + 64*eps(float64)*max(1, abs(x))
```

Both represented acceptance sides must lie within their allowances. When the
high-precision sides are separated by more than their summed allowances, the
represented decision must agree; fixed-word fixtures own decisions inside the
finite uncertainty band. This retains a strict `1e-6` central gate without
making an impossible absolute-error demand of overwhelmingly rejected support-
edge log magnitudes. The replaced cancellation-prone grouping reached
order-one central error near the top of the count domain and is not an accepted
implementation alternative. As with the delay evaluators, this is a frozen
finite-domain test matrix rather than a proof over every real-valued
probability; a violation inside the declared domain returns to Design instead
of silently narrowing or approximating the law.

Category order is part of each stochastic role. For each timing-jitter source
cell, retained destination bins `t = 0 .. S - 1` are scanned increasingly;
because `k = t - s`, this is also increasing signed-offset order. Conditional
category `c = t` uses
`logical_position = t * N + source_flat_position`. The combined out-of-window
drop category is the final exact count remainder and consumes no draw. AP scans
retained causal offsets in increasing order, then overflow, with stop as the
final exact remainder. Let `g` be the zero-based parent-generation index in
`0 <= g < K`, `d` a causal retained offset, `u` the source cell's row-major
flat position, and `N` the complete grid size. Its fixed category index and
logical position are:

```text
c = d,  0 <= d < S       retained offset slots
c = S                    right-overflow slot

p = ((g * (S + 1) + c) * N) + u
q = 0
```

For a source in sample bin `t`, only `0 <= d < S - t` is scientifically valid;
larger retained-offset slots stay reserved and unused rather than being
compacted. Overflow always uses `c = S`, and stop has no address or draw.
Preflight requires `K * (S + 1) * N <= 2**63` whenever nonzero AP can execute.
This is generation-major, offset/category-major, then source-position-major.
Zero-probability categories may skip physical work without changing later
positions. AP uses the package-owned key at stream `9`; it never shares a key
or derives identity from active-only compaction. Inversion uses
raw ordinals zero and one at one AP address; BTRS attempt `j` uses ordinals
`4*j` through `4*j + 3`, so attempt 63 ends at ordinal 255.

### Poisson Count Sampling

Dark counts and the retained and overflow draws for both crosstalk modes call
TensorCore's public `CounterRng.poisson(...)`. TensorDSLab prepares each
physical binary64 mean, selects the exact fixed package-owned key, and constructs the
owning operation's positional lattice. TensorCore validates and samples the
generic numeric law. A mean is either scalar or exactly the positions shape;
arbitrary implicit broadcasting is not part of the contract. The result is a
fresh nonnegative `torch.int64` tensor. TensorCore does not know about sampling
periods, delay PMFs, generations, crosstalk semantics, or overflow meaning.

Conceptually:

```python
sampled = rng.poisson(
    mean=mean,
    key=role_key,
    positions=positions,
    quantum=0,
)
```

The owning effect selects the semantic key from its config. TensorCore consumes
only the supplied key, positions, quantum, and numeric law; it never discovers
roles from config types.

The selected rate-by-rate algorithm is:

```text
lambda == 0             -> exact zero; no raw word
0 < lambda < 10         -> one-uniform forward-CDF inversion
10 <= lambda <= 1e8     -> Hoermann PTRS transformed rejection
otherwise               -> hard unsupported-rate failure
```

The crossover uses each actual aggregate cell mean. A configured crosstalk
offspring mean below ten does not force the small branch when source
superposition makes a destination rate ten or larger. Every mean and sampler
control value is binary64 regardless of the requested `Charge` dtype. Negative,
nonfinite, or greater-than-`1e8` rates fail before that sampler call requests a
word or writes its result. Exactly `1e8` is accepted. The fixed ceiling is a
conservative v1 numerical domain, not a clipping rule or a statement that such
a detector rate is physically reasonable.

For `0 < lambda < 10`, raw-word ordinals zero and one form one
`U64[0, 1)`. The remaining two lanes in block zero are unused and cannot be
reassigned. Binary64 inversion starts from:

```text
probability_0 = exp(-lambda)
probability_k = probability_(k - 1) * lambda / k

sample = the smallest k for which U < sum(probability_i, i=0..k)
```

The reference checks at most the 64 terms `k = 0` through `63`. The mathematical
tail beyond that bound is negligible for `lambda < 10`, but sequential
binary64 recurrence and cumulative addition can still leave a top-lattice
uniform unresolved through rounding. That case is the documented deterministic
exhaustion failure; it does not consume another uniform, restart, or fall back
to a different method. This single-uniform inversion is selected over Knuth's
product loop because it avoids an expected `lambda + 1` expensive Threefry
evaluations while retaining the same intended Poisson target law on successful
draws. Every represented term and cumulative CDF value must agree with an
independent at-least-80-decimal-digit Poisson oracle for the same represented
rate within `1e-12` absolute error.

For `10 <= lambda <= 1e8`, TensorCore uses Hoermann's transformed rejection
with squeeze (PTRS). Preflight computes in binary64:

```text
sqrt_lambda = sqrt(lambda)
log_lambda = log(lambda)
b = 0.931 + 2.53 * sqrt_lambda
a = -0.059 + 0.02483 * b
inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
v_rectangle = 0.9277 - 3.6224 / (b - 2)
```

Attempt `r` consumes exactly Threefry block `r`: raw-word ordinals
`4*r, 4*r + 1` form open-open `U`, and `4*r + 2, 4*r + 3` form open-open `V`.
The open endpoints keep division and logarithms finite. Each unresolved cell
then evaluates:

```text
u = U - 0.5
u_s = 0.5 - abs(u)
k = floor((2 * a / u_s + b) * u + lambda + 0.43)

quick accept: k >= 0 and u_s >= 0.07 and V <= v_rectangle
quick reject: k < 0 or (u_s < 0.013 and V > u_s)

full accept when:
log(V) + log(inverse_alpha) - log(a / u_s**2 + b)
    <= -lambda + k * log_lambda - lgamma(k + 1)
```

For every supported finite proposal, compare the complete represented
full-accept left and right sides with an independent at-least-80-decimal-digit
evaluation of the same represented inputs and equations. For high-precision
reference side `x`, the local log-space allowance is:

```text
1e-6 + 64*eps(float64)*max(1, abs(x))
```

When the high-precision sides are separated by more than their summed
allowances, the represented decision must agree. Fixed-word fixtures own
decisions inside that finite uncertainty band; ensemble tests separately own
agreement with the target Poisson law. This mixed absolute/relative guard
keeps the cancellation-sensitive central region strict without pretending an
absolute `1e-6` representation of obviously rejected, extremely large
log-probability magnitudes.

Candidates are checked for finiteness and `int64` representability before
conversion. The reference permits exactly 64 attempts, numbered zero through
63. Any unresolved cell after attempt 63 causes deterministic hard failure.
There is no reseed, wrap, clamp, Gaussian approximation, alternate algorithm,
or biased fallback. A finite rejection cap adds a failure outcome; it does not
change values successfully accepted before that cap.

An eager implementation may update unresolved masks or select unresolved cells
for physical work, but every cell retains its original full-grid logical
position. Active-only ordering never becomes random identity. Direct and
delayed rates are never superimposed, retained and overflow draws never share a
stream, and aggregate destination Poisson draws are never replaced by
per-parent expansion. `source_quantum` is zero for every MVP Poisson role.
`torch.poisson`, PyTorch global RNG, and `torch.Generator` are not normative or
fallback implementations.

Threefry words and uniform conversion remain exact across accepted CPU/CUDA
implementations. Completed Poisson samples require exact repeatability only for
the same unchanged numerical execution stack. Inversion uses `exp`; PTRS
additionally uses square root, logarithm, and `lgamma`, so completed CPU/CUDA
fields compare statistically rather than bitwise. On one unchanged numerical
execution stack, the same input, config, seed, and positional lattice must
produce the same integer history for float32 and float64 `Charge` requests.

The primary PTRS reference is W. Hoermann,
[*The transformed rejection method for generating Poisson random
variables*](https://research.wu.ac.at/files/18953249/document.pdf). NumPy's
`2.4.3` production implementation at exact release commit
`8bcb2e72e67c343e55165e6064fe6a9dc011e954` is corroborating evidence for the
crossover, constants, and acceptance equations; see
[`random_poisson_ptrs`](https://github.com/numpy/numpy/blob/8bcb2e72e67c343e55165e6064fe6a9dc011e954/numpy/random/src/distributions/distributions.c#L550-L613).

Bounded phase and interpolation, including the Box-Muller angle, use `[0, 1)`.
Any logarithm uses `(0, 1)`, so neither logarithmic infinity nor an artificial
exact-zero inverse-transform result is possible. The exact Random123 midpoint
evaluation order and target dtype are part of the contract; do not replace it
with a widened calculation or a different endpoint-safe mapping.

A standalone Bernoulli primitive, if a future accepted consumer needs it, may
bypass floating uniforms. From a finite binary64 configuration probability
`p`, its selected candidate mapping is:

```text
T = round_ties_to_even(p * 2**32)
fires = raw_word < T
```

`T` is an integer in `[0, 2**32]`, and the realized probability is exactly
`T / 2**32`, within `2**-33` of the represented configuration probability.
Threshold zero returns false without requesting a word; threshold `2**32`
returns true without requesting a word. Every interior threshold consumes one
assigned raw word. This avoids the systematic downward bias of a floored
threshold and gives both `float32` and `float64` consumers the same represented
Bernoulli law. Stage 6 did not implement this unused standalone path; its
aggregate binomial primitive owns AP outcomes.

An ordinary exponential variate of configured mean `tau` is:

```text
delay = -tau * log(U(0, 1))
```

The operation evaluates in its accepted execution dtype; a `float32` operation
does not silently widen to `float64`. The finite uniform lattice bounds a
`float32` delay below about `16.64 * tau` and a `float64` delay below about
`36.74 * tau`. An operation whose scientifically relevant window reaches that
limit must classify and validate the tail truncation rather than call the
sampled law an exact continuous exponential.

This inverse-transform primitive applies only to an operation that explicitly
samples one continuous exponential variate. The fixed-`K` correlated-avalanche
path does not use it for CT or AP edge placement: preflight integrates the
physical delay law and latent phase into complete categorical offset
probabilities plus an analytic right tail, and the runtime samples those
prepared categories. Its exponential-law qualification is therefore numerical
CDF/PMF preparation and categorical sampling, not the finite inverse-transform
tail above.

White noise, PSD coefficients, and terminal charge smearing call TensorCore's
parameterized `gaussian(...)` method. TensorCore internally obtains the
standard-normal value through ordinary Box-Muller in the selected execution
dtype, then applies the frozen affine mapping. Timing jitter is not a Gaussian-
variate consumer: it analytically integrates its ideal Gaussian into
categorical probabilities as specified above.

```text
radius = sqrt(-2 * log(U(0, 1)))
angle = tau * U[0, 1)
z0 = radius * cos(angle)
z1 = radius * sin(angle)
```

Stage 5 prepares `tau` from Python binary64 `math.tau` and rounds it exactly
once into the selected execution dtype/device before payload arithmetic. The
angle is one multiplication of that scalar by the closed-open uniform; it is
not a reassociable two-multiply `2 * pi * U` expression. The exact `-2` factor
is applied to `log(U)` before square root, with ambient autocast disabled.

One exact `(stream, logical_position, source_quantum, base_raw_word)` address
owns the ordered pair `(z0, z1)`. A `float32` pair consumes two consecutive raw
words; a `float64` pair consumes four. A scalar Gaussian consumer always uses
`z0` and discards `z1`. A natural two-component consumer at the same address,
such as the real and imaginary standard-normal components of one PSD
coefficient, uses `z0` and then `z1`. The spare result is never cached or
reassigned to another tensor position, source quantum, branch, or stochastic
substep. This intentionally favors a simple stable positional contract over
packing adjacent output positions into one pair.

The maximum Box-Muller radius is about `5.77` for `float32` and `8.57` for
`float64`. The `float32` cutoff is an accepted bounded-MVP approximation, not a
claim of an unbounded mathematical Gaussian. It must be included in tail-aware
validation and in the synchronized `docs/parity.md` comparison; evidence that
rare threshold observables are sensitive to it is the trigger for a separately
versioned widened or tail-complete normal algorithm.

Stage 5 specifies bit-for-bit target-dtype uniform agreement across accepted
CPU/CUDA implementations. Box-Muller outputs retain the same-stack
repeatability boundary because `log`, square root, sine, and cosine may differ
across backend/device implementations and numerical stacks. Stage 6 activated
and proved analytic timing preparation, aggregate-binomial and Poisson
behavior, and smearing's Box-Muller use on eager CPU. CUDA was unavailable, so
cross-backend Charge evidence remains unestablished. Standalone Bernoulli or
continuous-exponential behavior
activates only if a later accepted consumer actually uses it. None of these
selections creates a CPU/CUDA bitwise guarantee for completed stochastic
products.

An operation that genuinely expands source quanta would assign each one the
deterministic address `(source_flat_position, quantum_ordinal,
raw_word_ordinal)` before redistribution. `quantum_ordinal` would be the
zero-based ordinal within its source cell and could never depend on parallel
execution order. The MVP Charge stochastic roles do not use that expansion:
Poisson and multinomial roles operate on aggregate cell counts with
`source_quantum = 0`.
The generic address component remains available without imposing a `2**32`
population limit on aggregate Stage 6 inputs. A distribution-level draw may
consume more than one raw word, so `raw_word_ordinal` must not be renamed or
treated as a one-word draw number.

Stage 7 noise preparation is deliberately contextual: it checks the exact
floating dtype and accepted device, sampling/source shape agreement, exact
config key, checked logical-position arithmetic, accepted public Gaussian
ordinal/count bounds, and model-specific RMS/PSD representation before any
product producer or RNG request. The public builder separately checks nominal
`CounterRng` membership. A `ZeroNoiseConfig` path is draw-free. Preparation
does not adversarially police privately constructed position tensors. The
corresponding Charge population, address, accumulation, and failure envelope
is frozen below.
TensorCore fixes Poisson PTRS and binomial BTRS rejection at 64 addressed
attempts and fixes each inversion guard at 64 terms. Exhaustion fails
deterministically rather than wrapping, reseeding, reusing an address, changing
algorithms, clamping the sample, or emitting a biased fallback.

Required behavior is:

- exact repeatability for the same immutable RNG algorithm/seed, input values,
  tensor shape, dimension order, coordinate order, fixed role keys, dtype,
  algorithm/version, and supported backend;
- unchanged common product values when unrelated products are added to or
  removed from `products`, because their package-owned role keys are fixed;
- zero-effect configs consume no relevant draws;
- no hidden global RNG; and
- cross-backend statistical agreement for completed floating stochastic
  products without an assumed bitwise guarantee.

Coordinate values are semantic metadata but are not random identities.
Relabeling or reordering coordinate metadata without moving payload values
leaves the positional random bits unchanged but changes their semantic
interpretation. Explicitly reindexing a payload to preserve coordinate
identity generally moves an item to a different position and therefore gives
it a different draw. Tensor-dimension permutation likewise changes the
position-to-semantic mapping. TensorDSLab promises no coordinate-identity or
permutation invariance; a meaningful semantic comparison first reindexes both
tensors explicitly.

Selection and arbitrary chunking are also not invariant. Positional addresses
restart at zero in every builder invocation. Calls of any shapes using the same
RNG seed/algorithm and role key therefore reuse the same underlying random
prefix over their overlapping logical flat-position range; equal `numel`
reuses the complete flat sequence before reshaping. MVP callers must supply
distinct RNG seeds or deliberately distinct keys for statistically independent
realizations. A future
chunk-stable execution surface would require explicit global positional
offsets and a focused Design contract; TensorDSLab does not infer those
offsets from semantic labels.

The positional engine is deliberately rank- and domain-agnostic and therefore
belongs in TensorCore under the accepted dependency gate. TensorDSLab retains
only scientific role placement, schedules, and samplers.

### Stage 6 Count, Address, And Numeric Envelope

The active Charge path uses one contextual per-cell count ceiling:

```text
C_max = 2**53 - 1
```

`C_max` is the largest nonnegative integer for which both `n` and `n + 1` are
exact binary64 integers. It therefore preserves every integer operand used by
the stabilized BTRS mapping while leaving the Poisson mean ceiling as the
independent `1e8` bound above. This is a Charge-producer contract, not a new
generic `Photoelectrons` construction invariant, public configuration value,
or package-level identity constant.

Before Charge execution, every source cell must lie in `[0, C_max]`. The same
interval then applies to every post-dark and post-jitter cell, current and next
frontier cell, newly drawn mechanism cell, retained or overflow diagnostic
cell, cumulative per-cell count, conditional-binomial count/remainder, and
accepted Poisson sample. There is deliberately no whole-grid, row, batch, or
example population ceiling: many cells may each equal `C_max`.

All integer additions use checked nonnegative arithmetic before the addition:

```text
rhs <= C_max - lhs
```

This rule applies to destination/category superposition, combination of the
three child mechanisms, cumulative mechanism and overflow diagnostics, and
`total_count`. Conditional-binomial remainders subtract only after proving
`category_count <= remaining_count`. An accepted Poisson proposal greater than
`C_max` is a hard count-domain failure; it is never converted, retried as a
rejection, clamped, or allowed to wrap. A wide sparse row is not rejected in
advance merely because a hypothetical concentration could exceed `C_max`; a
realized destination addition is checked when it occurs.

Dark-count preparation compares its represented configuration exactly before
forming the binary64 rate. With
`rate_num, rate_den = rate_hz.as_integer_ratio()`, the accepted relation is:

```text
rate_num * sample_period_ps <= rate_den * 10**20
```

This is exactly `rate_hz * sample_period_ps * 1e-12 <= 1e8` without a
boundary-crossing floating multiplication. For either CT mode, preparation
first forms the nonnegative binary64 thinning/convolution basis from exact
binary64 conversions of counts no greater than `C_max`. For positive scalar
offspring mean `a`, a branched/scaled comparison proves
`basis <= 1e8 / a` before multiplication. If the represented division rounds
to positive infinity, that branch is accepted only after proving the
mathematical threshold exceeds the greatest finite binary64 basis, so every
finite basis is safe. The represented product is then rechecked finite in
`[0, 1e8]`. A zero basis or rate is draw-free. No independent scalar cap is
placed on `a`; the actual aggregate cell rate owns the sampler limit.

Let `S` be the sample count, `N` the complete source-grid element count, and
`K` the configured maximum generation count. Checked Python-integer arithmetic
requires `0 < N <= 2**63 - 1`, and each planned tensor satisfies
`N * element_size <= 2**63 - 1`. Logical address products obey:

```text
noniterative dark/smearing roles: N positions
timing jitter:                      S * N <= 2**63
each effective CT role:             K * N <= 2**63
effective AP:                       K * (S + 1) * N <= 2**63
```

The inclusive product bounds are correct because the greatest addressed
position is one less than the product. `S <= 8192` remains the separate
contextual limit for active jitter or exponential-kernel preparation. There is
no arbitrary fixed upper bound on `K`: ineffective generation mechanisms have
no execution/address gate, while effective mechanisms derive their supported
`K` from these address relations and the accumulator-depth relation below.
Implementations never materialize the `S*N` or `K*(S+1)*N` address lattices.

For one cell, let `T` be the exact cumulative retained avalanche count. The
count checks give `T <= C_max`; every response weight lies in `[0, 1]`, so the
scientific real-arithmetic ledgers obey:

```text
0 <= S2 = sum(w_i**2) <= S1 = sum(w_i) <= T <= C_max
```

The functionality-first eager reference also freezes logical accumulation
order. Generations increase from zero; within each generation the mechanism
order is direct crosstalk, delayed crosstalk, then afterpulsing. For each CT
destination, causal source sample bins contribute in increasing source-bin
order; CT overflow is source-indexed in increasing source-bin order. AP scans
source bins increasingly and, within one source, scans retained offsets
increasingly before overflow and the final no-draw stop remainder. Count,
charge, and squared-charge contributions enter their destination and cumulative
diagnostics in that traversal, followed by the mechanism's `S1`/`S2` update.
Independent nonsample index tuples and independent destinations may execute in
parallel, but the eager reference does not use a repeated-index scatter or
atomic reduction with unspecified order. A later optimized execution mode may
replace this traversal only with its own proved sampler and rounding mapping.

Let `p_d` be `24` for `float32` or `53` for `float64`, let
`u = 2**(-p_d)`, and let `E` be the number from zero through three of effective
retained mechanism contributions added per generation. For the eager
functionality-first plan, the conservative maximum rounding depth is:

```text
without a retained recovery-weighted AP contribution: L = E*K + 1
with a retained recovery-weighted AP contribution:    L = E*K + S + 3
```

The second expression covers response-weight formation, count conversion and
multiplication, at most `S - 1` source-category additions, and the top-level
ledger additions. A unit-response path without a correlated stage uses
`L = 1`. Preflight requires the exact integer relation `L < 2**p_d`; this is
the floating-accuracy generation gate rather than a magic `K` constant. Define

```text
gamma_L = L / (2**p_d - L)
```

and let `eta_d` be the smallest positive subnormal in the requested dtype. For
the exact realized counts and represented recovery weights, the per-cell
high-precision-oracle bound is:

```text
abs(represented_ledger - real_sum)
    <= gamma_L * T + L * eta_d
```

A conservative real-arithmetic finite magnitude bound for either represented
ledger is
`B_real = C_max*(1 + gamma_L) + L*eta_d`. Because a realized ledger is itself
a value in the requested floating dtype, same-device compatibility preflight
and any defensive represented-ledger comparison derive the greatest finite
target-dtype value `B_d` such that `B_d <= B_real`. This downward
representation remains a bound on every represented ledger: no target-dtype
value exists strictly between `B_d` and `B_real`. It also prevents an ordinary
nearest representation of the binary64 expression from inventing an
out-of-bound ledger above `B_real`. Exact zero remains exact. Any later
compiled, fused, widened, or reassociated accumulator must prove and document
its own rounding depth and result mapping; it cannot silently inherit this
eager bound.

An exact-zero smearing width skips the stage. Otherwise `relative_sigma` is
rounded exactly once to the requested Charge dtype and must remain finite and
strictly positive. Let:

```text
Z_float32 = sqrt(-2 * log(2**-24))
Z_float64 = sqrt(-2 * log(2**-53))
```

with each bound rounded upward. These are the maximum absolute `z0` magnitudes
on the accepted dtype-specific Box-Muller lattices. With target maximum finite
value `F_d`, the Stage 6 model check uses upward-rounded arithmetic to prove:

```text
B_real + Z_d * relative_sigma_d * sqrt(B_real) <= F_d
```

That upward-rounded analytic check over `B_real` is unchanged. Maintenance 2
adds a separate same-device compatibility check at `B_d`: it evaluates the
actual target-dtype `sqrt(B_d)` followed by multiplication by
`relative_sigma_d`, then applies TensorCore's documented represented-law radius
to that prepared scale. Any defensive represented-ledger check also compares
to `B_d`, so representing the Python `B_real` value to nearest cannot admit a
target-dtype value above the proved real bound. Both checks complete before
any Charge effect can request words or write. The runtime still calls public
`rng.gaussian(...)` directly with exact `mean=S1`,
`standard_deviation=relative_sigma_d*sqrt(S2)`, `ordinal=0`, and `count=1`;
there is no local standard-normal affine path, fallback, or clipping change.

This intersection preserves the frozen `K=0`, `L=1` boundaries: the maximum
valid represented `float32` ledger is `0x1.0000000000000p+53`, relative sigma
`0x1.f61fea0000000p+98` is accepted, and its immediate neighbor
`0x1.f61fec0000000p+98` is rejected. The `float64` accepted/rejected pair
remains `0x1.51e4a059b7cf4p+994` /
`0x1.51e4a059b7cf5p+994`. A contextual extreme can be narrower when the public
Gaussian prepared-scale envelope dominates; the verified `L=24` `float32`
pair is accepted `0x1.f61fd20000000p+98` and rejected
`0x1.f61fd40000000p+98`. These endpoints are a derived representation domain,
not a physics calibration cap. Ordinary scientific configurations, the
Gaussian law and clipping policy, RNG keys and addresses, equations, and
accumulation and operation order are unchanged. Enabled smearing still visits
`S2 == 0` cells under its frozen full-grid schedule; the represented zero scale
makes their draw inert.

Complete public request and product preparation finishes before any RNG
request, product-producer invocation, or semantic-output write. After that
boundary, the functionality-first implementation may allocate TensorDSLab-
managed output, private scratch, and ordinary backend intermediates as its
operations require. It makes no allocation-free or no-library-temporary claim.
Every package-planned shape uses the checked byte arithmetic above, while
actual allocator failure remains the resource gate rather than an arbitrary
device-memory ceiling. Managed writable storage remains raw and unexposed until
its writes are complete and the semantic field is constructed. Dynamically
realized rates, counts, additions, and ledgers are checked before their next
dependent draw or arithmetic operation.

Malformed supported public inputs retain their documented `TypeError` or
`ValueError` boundary. Sampler exhaustion or a dynamically realized
rate/count/ledger-domain violation raises `RuntimeError`. A preparation failure
requests no RNG words and invokes no producer. A failure after backend work
begins has no rollback promise for private scratch, allocations, or completed
local prerequisite fields, but source/config objects remain unchanged,
stateless RNG state cannot advance, and no partial collection, failed field, or
diagnostic escapes through the public API. Resource failures carry no retry-
outcome guarantee.

## Functional, Memory, And Lifetime Contract

The initial rebuild adopts TensorCore's operation-owned result taxonomy. Its
public and private field-returning paths are deliberately stricter than the
generic root:

- the source `Photoelectrons` field is borrowed read-only;
- requesting `Photoelectrons` classifies that member as an **exact return** of
  the named source field;
- every product producer classifies its new result as **guaranteed fresh
  storage independent of named inputs**;
- generated fields retained together in one result are also guaranteed
  storage-independent from one another;
- the initial `simulate_readout(...)` surface has no guaranteed-storage-sharing
  result and no sharing-permitted-but-unspecified result path;
- every dimension-preserving product reuses the exact source `axes` tuple and
  exact immutable axis instances rather than reconstructing merely equal axes;
- private mutable scratch and assembly state never enter the collection;
- no unrequested prerequisite receives a collection-owned reference, so it is
  reclaimable after the builder returns when ordinary Python/Torch reachability
  permits;
- autograd may retain intermediates when a differentiable result requires
  them;
- no operation silently detaches, moves, casts, or host-materializes an
  existing input field;
- deterministic differentiable waveform operations preserve autograd where
  accepted; and
- stochastic count simulation and digitization make no blanket autograd claim.

`produce_noise_waveform(...)` uses `Photoelectrons` only for its exact axes,
shape, and device; it never reads the integer payload. Every zero, white, or
PSD result is a guaranteed-fresh `NoiseWaveform` with the requested floating
dtype and `requires_grad=False`. It has no differentiable tensor input or
lineage to the truth payload. A later `AnalogWaveform` may still preserve the
independent `PureWaveform` autograd path when adding this nondifferentiable
noise value.

TensorCore frozen records do not make Torch storage physically immutable.
Callers must not mutate tensors held by fields or collections through any
alias while they remain observable. Private functions that return raw tensors
rather than fields are outside TensorCore's field-result taxonomy, but their
scratch remains exclusively owned and cannot be exposed as a semantic value
that TensorDSLab later overwrites.

TensorDSLab initiates or enqueues every producer write before constructing and
exposing the corresponding result field, and it initiates no later write
through any alias to that storage. Accelerator work uses the current PyTorch
stream for the input device. Deep ingress validation and producer postconditions
may synchronize CUDA through scalar extraction as an accepted correctness-first
cost; field construction itself adds no synchronization. Outside those checks,
same-stream consumers inherit ordinary stream ordering, and cross-stream
consumers must establish their own event/stream dependency before reading.
Strong references preserve lifetime, not write safety or stream ordering.

Request selection reduces the long-lived result footprint; it does not by
itself promise a lower peak during construction. The simple functional planner
may keep local prerequisite references until final assembly, and autograd may
retain them longer through the result graph.

The first waveform implementation is functionality-first. The direct eager
Torch equations owned by `produce_analog_waveform(...)` and
`produce_digitized_waveform(...)` are the correctness references and may
materialize ordinary backend intermediates. Their functional work order proves
scientific values, dtype/device/axes, autograd where applicable, source
immutability, and guaranteed-fresh outputs; it makes no kernel-count,
target-sized-temporary, throughput, or compiler claim.

One-kernel/no-target-sized-temporary behavior is a later evidence-backed
optimization claim, not something inferred from compact Python syntax. A
focused optimization stage may inspect compiled graphs and use accelerator
profiling and memory instrumentation on representative shapes, then retain the
direct eager equations as its correctness reference. If compiler fusion is
insufficient, that later stage may evaluate a purpose-built Triton or CUDA
kernel without changing the public API or adding a decorative Python wrapper.
Product outputs remain guaranteed-fresh fields relative to their named inputs;
neither the functional nor a later fused implementation creates an
allocation-free chain claim.

Fusion across product boundaries, earlier release of arbitrary prerequisites,
and scratch reuse remain later measured execution optimizations with their own
retention, value-invariance, autograd, and lifetime tests.

The initial public API has no destination collection, public `out=`, or
`ReadoutWorkspace`. After real GPU profiling, a focused design may add reusable
scratch, output banks, stream binding, contiguity profiles, and allocation
instrumentation. It may not revive the old contract that overwrote a target
inside an already exposed valid semantic field or collection: any reusable
writable destination must remain raw, exclusive, and unexposed until all
producer writes have been initiated or enqueued, after which TensorDSLab may
construct the completed field exactly once. Advanced execution must preserve
request semantics and may not leak subsequently reusable scratch into retained
fields.

## Persistence And IO Are Deferred

This architecture defines no writer, persistence request, path, artifact format,
cache, schema, overwrite behavior, or default retained-for-disk product.
`products` means only fields retained in the returned in-memory collection.

Scientific configs contain no `persist` flags. A future artifact design will
let users choose which present product fields to write, and it must record the
scientific config/calibration needed to interpret them. Exact Python types will
map to separately versioned durable labels at that boundary.

## TensorG4DS Bridge

The future TensorDSLab-owned bridge must freeze:

- the exact accepted TensorG4DS input type and commit;
- event/provenance-to-`ExampleAxis` mapping;
- detector-channel-to-`ChannelAxis` mapping;
- dense photon-origin PE binning from the numeric left edges and exclusive stop
  implied by the exact caller-supplied `SamplingConfig`;
- normalization of each example to the accepted provenance origin followed by
  `sampling.build_axis()` with local start zero;
- `underflow_hit_count` and `overflow_hit_count` reporting for normalized hits
  outside the accepted half-open window, separately from arithmetic-overflow
  rejection;
- input/output dtype and device matrices;
- gradient rejection or preservation; and
- same-device tests proving no silent host staging.

The bridge returns a deeply validated dense truth `Photoelectrons` field. It
does not return a `ReadoutCollection`, apply timing jitter, parse native G4DS
files, expose CPU jagged storage to the readout builder, or relabel a
TensorG4DS value by subclass cast.

Conceptually, the future bridge is called with the same shared policy later
contained by `ReadoutConfig`:

```python
photoelectrons = build_photoelectrons(
    pe_hits,
    sampling=config.sampling,
)
```

For local hit time `t_ps`, the MVP numeric bin index is
`floor(t_ps / sampling.sample_period_ps.value)` and is retained only when it
lies in `[0, sampling.sample_count.value)`. The bridge does not infer the window
extent from the largest observed hit because empty tail bins are part of the
configured dense shape. The focused bridge design must freeze how the upstream
G4 floating representation is normalized to canonical integer picoseconds at
exact bin boundaries.

## TensorML And Reconstruction Boundaries

`ReadoutCollection` membership is unordered. A model-facing schema is an
explicit ordered tuple of product types:

```python
model_inputs = (AnalogWaveform,)
```

TensorML or a focused adapter resolves each type explicitly and owns positional
model order. It does not infer an ABI from request or collection insertion
order, move a full collection when one field is needed, or assume a field not
requested is available.

Future reconstruction may reuse `ExampleAxis` and `ChannelAxis` but owns its
field classes, collection classes, geometry, and preferred storage. A bridge
constructs new semantic leaves; it does not mutate or relabel readout fields.

## Parity Contract

`docs/parity.md` remains the authority for donor comparisons. This Design pass
synchronizes its rebuild-facing boundaries by replacing retired field IDs and
public atomic transforms with exact product requests and private diagnostic
seams.

Examples:

- public charge comparison becomes
  `simulate_readout(..., products=[Charge]) -> Charge`;
- isolated timing redistribution is a private diagnostic inside charge
  simulation and never a replacement truth field;
- pure/noise/analog/digitized comparisons request the corresponding type; and
- retention, prerequisite lifetime, and collection membership do not alter a
  scientific classification.

The scientific targets retained by this architecture are:

- conditional statistical timing-jitter parity under the binned latent-phase
  assumption, using the analytically prepared ideal-Gaussian offset law and
  aggregate conditional-binomial runtime, including redistribution means,
  variances, edge loss, and named tails rather than equality of the donor's
  per-PE finite digital normal or random stream;
- conditional distributional homogeneous dark-count parity, including the
  finite-gate loss when private dark-count avalanches are jittered out of the
  configured window;
- intentional full-chain divergence where IV jitters recursive generated
  avalanches but the rebuild jitters truth and dark roots before its causal
  fixed-`K` edge-placement loop;
- the ideal unit-parent Poisson generation law as an analytic DiCT multiplicity
  oracle using the fixed IV mean, without Gamma-intensity or negative-binomial
  offspring; the implemented donor comparison remains statistical because the
  complete recursive and digital laws differ;
- post-binned DiCT placement only as a marginal/statistical comparison
  under an explicit `SamplingConfig`-to-`q_direct` mapping, aligned edge policy,
  and explicit divergence from IV's later independent parent/child timing
  jitter; the selected causal edge-placement law does not claim IV's signed
  post-binned displacement, hidden sibling timing, or multigeneration timing
  covariance;
- no IV-parity claim for optional DeCT, which is a TensorDSLab model with its
  own calibration and validation boundary;
- a full-cascade statistical comparison, rather than an accepted
  eventwise parity claim, reported as a function of `K`, with unit-count
  branching, recovery-weighted AP deposited charge, and finite-depth
  truncation classified explicitly;
- ordinary exponential afterpulse delay as an intentional donor correction,
  integrated during preflight into complete offset categories and an analytic
  right tail; the aggregate AP path does not use the finite inverse-transform
  primitive;
- fresh uniform within-bin afterpulse phase after aggregate jitter as an
  intentional binned approximation, without eventwise parent-child timing
  parity;
- AP charge from conditional mean recovery applied to
  the same realized delay-category counts, preserving binned count/charge
  covariance while intentionally omitting within-category recovery variance;
- intentional divergence from IV's recovery-weighted future branching: the
  rebuild uses only integer avalanche multiplicity for reproduction and
  never feeds deposited charge back into an offspring law;
- exact unit AP charge when `AfterpulseConfig.recovery is None`, and the
  composed exponential recovery response when it is present;
- statistical charge parity for named observables;
- equation-level TPC FEB-SNR pulse parity over aligned sample times using the
  explicit IV `tau_r -> tau_fast`, `tau_r + tau_l -> tau_slow` mapping, with
  sampled-peak normalization, explicit support, half-open windowing, and
  omission of IV's fractional-bin amplitude correction classified separately;
- equation-level Veto PDU Gaussian/two-erf pulse parity over aligned sample
  times using the explicit renamed parameter mapping, likewise separating the
  accepted sampled-support/sign convention from IV's heuristic crop,
  fractional-bin correction, and gate-boundary defects;
- exact zero-noise behavior plus statistical white-noise and PSD-shaped-noise
  parity after the documented PSD-to-target-grid integration; the finite
  precision-matched uniform lattice and Box-Muller radius cutoff preclude a
  literal unbounded-Gaussian or donor-digital-law distributional claim unless
  the donor comparison uses the exact same conversion and transform;
  IV raw spectral amplitudes are comparison evidence only after an explicit
  offline amplitude-squared-to-PSD-shape mapping plus an independently accepted
  absolute-power calibration; without that calibration, the absolute-PSD model
  is an intentional donor divergence rather than a fabricated parity claim;
  IV frequency coordinates label spectral lines and must not be copied into
  `frequency_left_edges_hz`; offline conversion must construct explicit
  interval left edges plus an exclusive stop, or fit an interval PSD, before
  comparison;
- intentional exact-length circular-covariance divergence from IV's long
  synthesized baseline and random-crop boundary;
- forced-zero synthesized DC in agreement with IV's broad zero-DC intent,
  while explicitly classifying the discarded fixed-record `[0, df / 2)` PSD
  power and making no absolute-power parity claim for that cell;
- exact analog composition and optional physical saturation at the
  `AnalogWaveform` product boundary; and
- exact in-range ADC codes under the frozen endpoint-guarded affine execution
  form, with inclusive field-dtype endpoints and intentional pre-conversion
  clipping divergence from IV's out-of-range integer wraparound.

Post-binned statistical parity remains acceptable without eventwise or bitwise
identity when tensor rebasing or RNG streams differ. Every production claim
still names observables, assumptions, units, tolerances, and fixtures. No donor
runtime becomes a production dependency.

## Validation Strategy

Maintenance 5 replaces string-axis/config-agreement validation with compact
axis constructor/narrowing evidence, exact source-derived sampling, and the
complete-input `SampleAxis.start == 0` preflight. It also retires—without a
replacement exception promise—the off-path
`collection.field(TensorField)` `TypeError` assertion under TensorCore
`0.13.0`'s golden-path boundary.

### Stage 6 Statistical Acceptance Policy

Stage 6 validates conformance to the selected TensorDSLab probability model;
it does not manufacture a universal IV-DSLab equivalence margin for mechanisms
whose laws intentionally differ. Statistical fixtures reuse the four frozen
Stage 5 seeds without tuning them after observing a candidate:

```text
0
1
0x0123_4567_89ab_cdef
0xffff_ffff_ffff_ffff
```

The fixed primary ensemble sizes are:

- scalar and one-parent laws: `M = 2**18`, with exactly `2**16` independent
  examples per seed;
- aggregate `Q = 32` multinomial or one-generation laws: `M = 2**16`, with
  exactly `2**14` independent examples per seed; and
- small-grid `K <= 3` cascade and completed-`Charge` model fixtures:
  `M = 2**16`, again with exactly `2**14` independent examples per seed.

One example is one independent replicate. Correlated channel/sample cells
within an example never inflate `M`. For a predeclared statistic `f(X)` with
analytic target `theta` and target-law standard error `SE`, the frozen gate is:

```text
abs(observed - theta) <= 8*SE + delta(dtype, scale, length)

delta(dtype, scale, length)
    = 64 * eps(dtype)
         * max(1, ceil(log2(length)))
         * abs(scale)
```

The target law, not the observed sample variance, supplies `SE`. A separately
bounded finite-lattice or sampler-mapping bias is added explicitly rather than
hidden inside Monte Carlo tolerance. A Bernoulli, PMF, CDF, or exceedance
frequency is asserted only when both expected hits and expected misses are at
least 256; rarer cases use high-precision probability fixtures and fixed-word
sampler tests. Tail checks use predeclared CDF thresholds, not unstable
empirical quantiles. Checks and sample sizes are frozen before candidate
execution and are not added, removed, or enlarged after seeing results.

Exact conservation, bypass, identity, source-immutability, integer-history,
address, and fixed-delay claims remain exact and receive no statistical
allowance. The existing `1e-12` local and `1e-11` complete-law numerical gates
for jitter and exponential delay/recovery also remain separate. Binomial and
Poisson executable-mapping bias is bounded against independent high-precision
oracles over the accepted count/rate domain before ensemble evidence is used.

Recovery-weighted `S1` and `S2` compare with a higher-precision reference by the
numeric-envelope bound above. With exact retained count `T`, conservative
rounding depth `L`, dtype precision `p_d`, and smallest positive subnormal
`eta_d`:

```text
gamma_L = L / (2**p_d - L)
abs(ledger - reference) <= gamma_L*T + L*eta_d
```

The work order must make its actual rounding path no longer than the accepted
`L`; it may not substitute a fixed percentage. Unit-weight cases whose integer
totals remain exactly representable in the requested dtype require exact
ledger equality.

Operation-level analytic oracles cover dark-count Poisson moments and zero
probability; jitter multinomial means/covariance and edge loss; separate
DiCT/DeCT Poisson fields and finite-`K` branching moments; AP retained,
overflow, stop, count/charge, and recovery moments; and rectified-normal
smearing. For `X ~ Poisson(lambda)` over `M` examples, the primary dark/CT
standard errors are:

```text
SE(mean(X)) = sqrt(lambda/M)
SE(mean((X-lambda)**2))
    = sqrt((lambda + 2*lambda**2)/M)
SE(fraction(X == 0))
    = sqrt(exp(-lambda)*(1-exp(-lambda))/M)
SE(independent centered cross-product) = lambda/sqrt(M)
```

For a one-bin DiCT-only Galton-Watson fixture with `Q` roots and Poisson mean
`lambda`, the generation oracle includes:

```text
E[Z_g] = Q*lambda**g
Var(Z_g) = Q*lambda**g*(lambda**g - 1)/(lambda - 1)
Cov(Z_g, Z_h) = lambda**(h-g)*Var(Z_g),  h >= g
```

The continuous limit at `lambda == 1` is used rather than dividing by zero.
Finite-window position moments come from an independent matrix oracle. For one
AP parent with retained category masses `pi[d]` and recovery weights `rho[d]`,
the joint count/charge oracle includes:

```text
E[W] = sum_d pi[d]*rho[d]
E[W**2] = sum_d pi[d]*rho[d]**2
Cov(retained_count, W)
    = E[W] - P(retained)*E[W]
```

Independent-parent moments add for `Q=32`; the omitted within-category
recovery variance is reported separately, never absorbed into tolerance.

For fixed smearing ledgers `mu = S1` and
`s = relative_sigma*sqrt(S2)`, validation uses:

```text
a = mu / s
E[max(N(mu, s**2), 0)]
    = mu*Phi(a) + s*phi(a)
E[max(N(mu, s**2), 0)**2]
    = (mu**2 + s**2)*Phi(a) + mu*s*phi(a)
P(output == 0) = Phi(-a)
```

Degenerate `s == 0` uses its exact deterministic identity. A small independent
scalar moment/PGF oracle supplies completed-model means, variances,
covariances, occupancy, edge loss, and predeclared CDF thresholds.

No detector-level IV equivalence margin is guessed. A later donor claim must
receive an observable-specific scientific margin `Delta` from calibration or
collaborator review and pass:

```text
abs(theta_TensorDSLab - theta_IV)
    + 8*sqrt(SE_TensorDSLab**2 + SE_IV**2)
    <= Delta
```

For paired examples, the paired-difference standard error replaces the
independent sum. The acceptance sample size is chosen beforehand so the
eight-SE half-width is no greater than `Delta/2`. Until such margins exist,
finite-`K` versus IV recursion, DeCT, corrected AP delay/recovery, clipped
smearing, and detector-level requested Charge remain explicitly unestablished
IV statistical targets rather than Stage 6 implementation gates.

The rebuild validation matrix includes:

- exact selection of TensorCore `0.9.0` commit
  `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59` for Maintenance 2,
  package-root-only imports of `RngKey`, `CounterRng`, `Threefry4x32`, and
  `logical_positions`, and no TensorDSLab import of protected TensorCore RNG
  mechanics;
- exact `RngKey` fields, namespace, ten append-only default streams,
  equality/`repr` participation, explicit exact-key overrides, crosstalk
  retained/overflow inequality, and absence of keys from deterministic,
  delay, recovery, and composite configs;
- complete removal in the closed Maintenance 2 implementation of `_RngStream`,
  `readout/_random.py`, and any replacement `readout/_rng.py`, while Charge
  multinomial/category orchestration, checked count helpers, and bookkeeping
  remain in `charge/runtime/effects/counts.py`;
- stochastic-capable private producers/effects accepting `CounterRng`,
  deterministic producers/helpers omitting it, and exact-zero/disabled paths
  making no public RNG request;
- default-key continuity through public TensorCore `uniform`, `gaussian`,
  `poisson`, and `binomial`, plus TensorDSLab multinomial orchestration,
  `NoiseWaveform`, and `Charge`, without importing or duplicating TensorCore
  raw-word or distribution implementation; and
- Stage 7 rejection of duplicate keys assigned to distinct roles in the
  requested transitive closure before any RNG request, producer invocation, or
  semantic-output write,
  including structurally present numeric no-op configs and excluding absent,
  zero-noise, and unrequested branches;
- exact TensorCore dependency, package-root imports, and ordinary-ABC static
  construction and exact lookup inference;
- static and Review evidence that every public semantic leaf has exactly one
  matching root in `__bases__`, with no mixin or other base, is `@final`,
  declares `__slots__ = ()`, adds no stored
  annotation or field, does not reapply `@dataclass` or override root behavior,
  and implements `_require(self) -> None`;
- public-boundary rejection of malformed documented inputs and unsatisfied
  supported relationships, without adversarial runtime policing of unsupported
  subclassing, class mutation, constructor bypass, or direct private calls;
- nonempty unique example/channel coordinates and at least two unique sample
  timestamps;
- exact `SamplingConfig` component types, positive period, count of at least
  two, and `window_stop_ps <= 2**63 - 1`;
- canonical ASCII `^(0|[1-9][0-9]*)ps$` acceptance plus rejection of signs,
  whitespace, decimals, exponents, alternate units, leading zeros, and values
  or derived stops outside signed int64;
- strict timestamp order, positive uniform integer-picosecond spacing, derived
  start/period/stop, a regular nonzero-start semantic subaxis, and O(1)
  full-source agreement on size, zero start, and configured period;
- exact sample-boundary fixtures proving `N` stored left-edge timestamps and no
  stop coordinate, assignment of times `0` and `i * T` to their corresponding
  bins, exclusion of negative times and the terminal time `N * T`, and separate
  `underflow_hit_count`/`overflow_hit_count` accounting;
- exact three-axis membership in any accepted order;
- tensor shape/axis agreement and `torch.strided` layout;
- intrinsic dtype checks and explicit deep-value validation;
- collection rejection of empty or unknown membership;
- collection same-axis, same-device, and common-floating-dtype coherence;
- one-pass product iterable consumption;
- empty, duplicate, non-class, base-class, and foreign-class request rejection;
- request order having no result semantics;
- exact transitive product preparation before any RNG request, producer
  invocation, or semantic-output write;
- merged Maintenance 4 implemented-tree and import proof for empty
  non-exporting runtime package roots, exact product-owned `prepare` /
  `produce` / `validate` actions, final frozen slotted Runtime records, no
  Config or semantic product
  retained in a Runtime, no Config/validator import in a producer, and no old
  private-path shim;
- one `SamplingRuntime` prepared once per public request and shared by exact
  identity among temporal ProductRuntime values, without repeated sample-axis
  lookup during production;
- exact `produce -> validate -> descendant` order, including validator receipt
  of the exact generated object and named prerequisite relationship, failure
  before every descendant and final collection construction, and one Charge
  terminal deep scan;
- proof that every MVP calibration value is scalar and applies uniformly to
  all example/channel positions in one invocation, without channel-coordinate
  lookup, implicit parameter broadcasting, or tensors hidden inside configs;
- `produce_analog_waveform(...)` reference checks for no saturation, each
  one-sided bound, and two-sided bounds, including exact bound values and proof
  that the input fields remain unchanged;
- analog scalar preflight rejects nonfinite field-dtype conversions and bounds
  that collapse or reverse after conversion before evaluating `pure + noise`;
- zero-input waveform fixtures proving that neither `PureWaveform`,
  `NoiseWaveform`, nor `AnalogWaveform` receives a deterministic pedestal, and
  that the digitizer's zero-voltage code follows only from its accepted
  endpoint-guarded affine transfer;
- digitizer preflight checks for finite representable `maximum_code`, `gain`,
  `span`, `slope`, `intercept`, and strictly ordered dtype-rounded pre-gain
  thresholds, followed by endpoint, asymmetric-zero-code, code-transition-
  neighborhood, truncation, `torch.int32`, and pre-cast clamping fixtures with
  no unsigned wraparound;
- equivalence of each direct waveform-tail production expression to its
  unfused reference equation under the accepted dtype/backend numerical
  contract;
- Stage 4 static and runtime proof that the waveform tail contains only the
  two owning product producers and does not introduce decorative pointwise
  Python wrappers or cross-product fusion;
- closed Stage 7 public-orchestration proof that requesting only
  `DigitizedWaveform` still computes `AnalogWaveform` exactly once as an
  unretained prerequisite;
- a later conditional accelerator optimization stage may add compiler-graph,
  profiler, and memory evidence for one fused backend kernel and no
  target-sized temporary; Stage 4 requires no such evidence and makes no
  fusion claim;
- exact `PureWaveformConfig.model` rejection of foreign/base values,
  exact dispatch to both accepted pulse models, and proof that one selected
  model and scalar parameter set are applied uniformly without inferring
  physical family or calibration from channel strings;
- exact TPC FEB-SNR and Veto PDU donor-equation oracles, including
  `tau_fast = tau_r`, `tau_slow = tau_r + tau_l`, `tau_slow > tau_fast`, and
  every renamed Gaussian/two-erf parameter mapping;
- sampled-template normalization/support oracles, positive amplitude plus one
  fixed negative-polarity scaling step, causal convolution alignment,
  same-length truncation, and no hidden gain/inversion/baseline;
- parity fixtures proving the separately classified differences from IV's TPC
  continuous normalization, heuristic TPC/Veto support, post-convolution
  inversion, fractional-bin amplitude correction, and gate-edge behavior;
- pulse-support fixtures proving inclusion exactly when `j * T < support_time`,
  exclusion at the support stop, and preflight rejection of an empty,
  nonfinite, or zero-extremum sampled template;
- exact `PsdNoiseConfig` left-edge/density count agreement, zero-start, strict
  left-edge order, final-left-edge/stop ordering, nonnegative finite absolute
  density, constructor-level nonzero supplied power, and request-preflight
  nonzero retained power after DC suppression;
- PSD coverage through the `SamplingConfig` Nyquist frequency, deterministic
  overlap integration onto the target one-sided frequency intervals,
  pre-suppression power conservation, and explicit DC-cell discard without
  redistribution or raw FFT amplitudes;
- PSD boundary fixtures with coincident source/target left edges, the final
  exclusive stop, exact odd- and even-`N` target-cell left-edge arrays, no
  double ownership at shared boundaries, and proof that Fourier basis
  frequencies never populate the public left-edge tuple;
- exact small odd- and even-`N` coefficient-to-time-domain oracles for
  `torch.fft.irfft(..., n=N, dim=-1, norm="backward")`, including cosine-only
  and sine-only interior bases, the `N / 2` interior scale, the `N` real
  Nyquist scale and alternating output, explicit real/imaginary
  standard-normal component convention, complex dtype, and absence of a DC draw;
- endpoint and degenerate-record fixtures: `N=1` is outside the accepted
  `SamplingConfig` domain and would retain no PSD power after DC suppression,
  `N=2` is a real-Nyquist-only process, odd-`N` terminal imaginary components
  affect the output, even-`N` Nyquist imaginary components are exactly zero,
  and zero-power cells produce exact zero coefficients;
- proof that no native complex-normal variance convention, implicit FFT
  normalization, post-transform demeaning, standard-deviation normalization,
  power normalization, or DC-power redistribution changes the frozen law;
- PSD-shaped-noise ensemble checks for an exactly zero real DC coefficient and
  a per-record sample mean bounded by deterministic inverse-FFT roundoff,
  retained expected variance, the accepted odd/even
  circular covariance equation, target spectral shape outside the discarded
  DC cell, real Nyquist behavior, fluctuating finite-record power, independent
  channel rows, zero cross-row covariance in expectation, Parseval mean-square
  power, coefficient moments, and the flat-density `-1 / (N - 1)` nonzero-lag
  correlation caused by the documented DC notch; ensemble variance oracles use
  population mean-square/`correction=0`, not an unrelated sample correction;
- a required accepted `CounterRng` instance on every public simulation request,
  including
  deterministic closures, no simultaneous `seed=`, and immutable reuse
  replaying the same realization without advancing state;
- TensorCore-owned exact Random123 `Threefry4x32_R<20>` known-answer,
  logical-position, fixed-point-uniform, internal Box-Muller, public Gaussian,
  Poisson, binomial, and backend evidence; TensorDSLab consumes only public
  distribution methods and preserves the Stage 5/6 default-key results;
- TensorCore prerequisite evidence for exact schema-v1 key/counter/lane
  packing for zero and maximum seeds, differing
  seed halves, every accepted namespace/stream key, positions around the `2**32`
  split and at the accepted maximum, zero and maximum quantum ordinals, and
  raw-word ordinals `0`, `1`, `2`, `3`, `4`, and the accepted maximum; the
  nonzero/max-quantum cases validate generic schema packing only and do not
  create a Stage 5 source-quantum consumer;
- explicit lane-three-to-next-block-lane-zero rollover, numerical low/high word
  order, the `0x54445331` default namespace, and deterministic rejection of
  seed, namespace, stream, position, quantum, and raw-word schema-bound
  violations without narrowing casts; Stage 6 separately completed
  source-population validation;
- TensorCore prerequisite evidence for exact `float32` and `float64`
  closed-open and open-open conversion oracles for
  zero, maximum, and representative raw words, including endpoint exclusion,
  numerical two-word order, discarded-bit behavior, adjacent midpoint cells
  around `0.5`, and no reuse of discarded bits;
- if a later work order accepts a standalone Bernoulli consumer, ties-to-even
  threshold construction, exact threshold-boundary word comparisons, quantized
  probability error no greater than `2**-33`, and draw-free threshold-zero and
  threshold-`2**32` results; Stage 6 did not implement this unused primitive;
- TensorCore prerequisite evidence for the Box-Muller raw-word schedule,
  ordered cosine/sine components at one exact positional address,
  scalar-consumer spare-result discard, native-dtype execution, same-stack
  repeatability, component moments and covariance, and explicit
  `float32`/`float64` radial cutoffs; TensorDSLab separately proves
  two-component PSD use and completed-product continuity;
- TensorCore public-Gaussian fixtures for exact Python-float versus
  exact-shaped tensor law parameters, no broadcasting/casting/movement,
  elementwise zero-standard-deviation exact-mean results, identity and
  zero-mean branches, multiply-before-add behavior, count-axis expansion,
  odd-ordinal slicing, `float32`/`float64` ordinal-plus-count bounds, checked
  result `numel`, fresh contiguous non-aliasing storage on every path,
  rejection of gradient-bearing law tensors, and conservative finite-output
  envelope rejection before word generation;
- if a later accepted consumer explicitly samples a continuous exponential
  variate, endpoint, mean, finite-tail, and native-dtype fixtures for that
  separately dispatched operation; Stage 6 CT/AP delay placement instead uses
  binary64 prepared categorical laws and activates no such variate consumer;
- globally unique fixed package-owned role-key assignments that do not change
  with the requested subset, enabled branches, or later appended operations,
  plus absence of public Config key fields and request-time collision
  admission;
- exact default streams `0x0000_0003` through `0x0000_000A` for dark counts,
  retained DiCT, DiCT overflow, retained DeCT, DeCT overflow, timing jitter,
  AP, and charge smearing respectively, including noncollision with the two
  noise defaults and exact absence of a value request for disabled or
  whole-stage zero-effect roles;
- independent high-precision timing-jitter oracles for the analytic
  latent-uniform plus ideal-Gaussian `q[k]` law, including ideal symmetry
  `q[-k] = q[k]` and represented agreement within the accepted numerical
  tolerance, representative and extreme supported `sigma / T` ratios, central
  cells, named tails, and the farthest destinations that can remain inside both
  window edges;
- timing-preflight proof over `2**-52 <= sigma / T <= 64` and
  `2 <= S <= 8192` that every possibly in-window destination is evaluated
  without an arbitrary tail cutoff; the exact `z = 8` evaluator boundary and
  asymptotic-series stopping paths; finite monotone nonnegative `L`, finite
  nonnegative `q`, exact represented offset symmetry, stable `A`/`B` category
  masses, and the `1e-12` category/tail/identity tolerance; and rejection of an
  invalid law without clipping, residual assignment, or renormalization;
- timing-runtime fixtures for increasing destination/address order, exact
  `S * N <= 2**63` address-bound enforcement, retained-plus-drop count
  conservation, the combined drop category as final no-draw remainder, source
  immutability, and `sigma == 0` whole-stage bypass;
  plus high-precision category fixtures and a complete represented-source-law
  L1 error no greater than `1e-11`; ensemble agreement with analytic
  multinomial mean, variance, covariance, displacement, and edge-loss
  observables; and proof that production jitter neither expands individual PEs
  nor calls Box-Muller;
- TensorCore prerequisite evidence for exact binomial zero/one/no-count
  branches, exact-shaped/device-matched `int64` counts in
  `[0, 2**53 - 1]`, direct normalized probability mode, independently supplied
  success/failure masses, both-zero/count-zero handling, no arbitrary
  broadcasting, fresh `int64` results, the small-mean inversion recurrence and strict CDF comparison,
  stabilized large-mean BTRS proposal/quick-accept/log-bound paths,
  real-algebra identity with the retired grouping, at-least-80-decimal-digit
  fixtures through `n = 2**53 - 1`, the central `1e-6` absolute local gate,
  complete-support mixed per-side allowances and decision separation,
  fixed-word uncertainty-band ownership, a large-count cancellation
  regression for the retired grouping, exact raw-block schedules,
  reflection/complement behavior, and 64-term/attempt exhaustion injection;
- TensorDSLab aggregate-multinomial fixtures for stable binary64
  current/later-category masses, exact calls to public `binomial(...)`,
  conditional conservation, fixed category and address order, final no-draw
  remainder, and covariance against the analytic multinomial law without
  per-avalanche expansion;
- AP address fixtures for generation-major, fixed offset-category-major, then
  source-position-major order; overflow fixed at category `S`; reserved
  invalid-from-source offsets left unused rather than compacted; stop as the
  no-draw remainder; exact `K * (S + 1) * N <= 2**63`; block zero, one, and 63
  schedules through raw ordinal 255; and identical category draws with and
  without recovery enabled;
- charge-smearing fixtures for exact stream `0x0000_000A`, every row-major
  full-grid position including zero-S2 cells, `source_quantum = 0`, scalar
  `z0` use with `z1` discarded in both dtypes, whole-stage zero-sigma bypass,
  and no value-dependent position compaction;
- active-Charge count-envelope fixtures for exact per-cell
  `C_max = 2**53 - 1` acceptance and `C_max + 1` rejection across sources,
  working grids, frontiers,
  mechanism/overflow diagnostics, cumulative counts, binomial counts, and
  accepted Poisson samples; checked-add success at the exact boundary and
  failure one above without wrap; and multiple `C_max` cells proving there is no
  whole-grid, row, batch, or example population ceiling;
- exact address-product boundaries and immediate rejection above them for
  `S*N`, `K*N`, and `K*(S+1)*N`, no arbitrary `K` gate for ineffective
  mechanisms, checked shape-byte products, and proof that no complete address
  lattice is materialized;
- accumulator-depth fixtures at `L = 2**p_d - 1` and `L = 2**p_d`, plus
  float32/float64 unit and recovered-AP ledgers against
  `gamma_L*T + L*eta_d`, scientific `S2 <= S1 <= T`, exact-zero behavior, and
  rejection of an unproved reassociated mapping;
- smearing-envelope fixtures proving the unchanged upward-rounded analytic
  check over `B_real`, the downward target-dtype ledger bound `B_d`, the
  preserved `K=0` float32/float64 adjacent sigma pairs, the contextual `L=24`
  float32 accepted/rejected pair, positive sigma that rounds to zero or
  infinity, maximum-radius raw words through public TensorCore `gaussian(...)`,
  and zero-S2 cells; rejection completes before any earlier enabled Charge
  effect requests words, and every accepted scale, excursion, pre-clipped
  draw, and returned Charge remains finite;
- TensorCore prerequisite Poisson scalar-oracle fixtures for `lambda = 0`,
  representative small rates,
  values immediately below and at the exact crossover `10`, representative
  PTRS rates, and the accepted endpoint `1e8`, plus rejection of negative,
  nonfinite, and greater-than-`1e8` means before that sampler requests words or
  writes its result;
- TensorCore at-least-80-decimal-digit Poisson executable-mapping fixtures requiring
  `1e-12` absolute agreement for every inversion term/CDF value and the frozen
  mixed absolute/relative local allowance for each PTRS full-accept side;
  high-precision/represented decision agreement outside the summed uncertainty
  band and fixed-word decisions inside it;
- TensorCore fixtures for scalar and exact-output-shaped `torch.float64`
  means, rejection of arbitrary
  broadcasting and wrong dtypes, exact-shape fresh nonnegative `torch.int64`
  results, and a mixed zero/inversion/PTRS tensor proving masks preserve every
  cell's original positional identity;
- TensorCore fixed-word Poisson inversion/CDF boundaries, PTRS quick
  acceptance, guarded
  rejection, full log acceptance, one-block-per-attempt addressing, and
  deterministic 64-attempt exhaustion through a reviewed synthetic sampler
  oracle rather than a searched production seed, plus the maximum closed-open
  uniform at a binary64 rate immediately below `10` to fix the inversion
  recurrence's exact success-or-exhaustion result;
- TensorCore Poisson mean, variance, zero probability, selected PMF/tail,
  no-`torch.poisson`, no-fallback, same-stack repeatability, and
  conditional CUDA statistical evidence; TensorDSLab separately proves
  physical aggregate superposition, exact public-call keys/positions, and
  same-stack integer-history equality across float32/float64 Charge requests
  rather than completed-field bitwise identity;
- arbitrary-rank and arbitrary-shape positional RNG oracles, including scalar
  and empty results where the selected backend accepts them;
- row-major logical flat positions derived from current dimension order rather
  than physical storage offsets, including noncontiguous-view fixtures;
- for every accepted iterative Charge role, exact virtual-leading-generation
  addressing, global rather than block-local generation identity, the fixed
  CT `K*N` and AP `K*(S+1)*N` bounds, and no activity-compacted positions;
- deterministic source-quantum and raw-word ordinals in generic schema tests,
  collision-free address encoding across every accepted root-seed/stream/
  position/quantum/raw-word tuple, the distinction between address uniqueness
  and ordinary repeated 32-bit output values, and exact `source_quantum = 0`
  for every aggregate MVP Charge sampler;
- exact raw-word agreement between the scalar oracle and every Stage 5
  accepted vectorized eager implementation, plus proof that TensorDSLab neither
  reads nor mutates PyTorch global RNG state and does not construct a
  `torch.Generator`;
- proof that axis classes, coordinate strings, and timestamps do not enter the
  random address or hot-path RNG inputs;
- exact same-stack repeatability only for an unchanged OS/architecture,
  Python/PyTorch build, backend/device implementation, execution mode, math
  settings, positional schema, values, config, dtype, algorithm/version, and
  seed;
- proof that coordinate relabeling alone preserves positional bits while
  changing semantic association, with no coordinate-identity or
  tensor-permutation invariance claim;
- selection/chunk noninvariance and same-seed separate invocations of different
  shapes explicitly shown to reuse the random prefix over overlapping flat
  positions;
- every prerequisite executed at most once;
- exactly requested final membership;
- unrequested prerequisite absence;
- source `Photoelectrons` identity and immutability when retained;
- all 16 structural presence/absence combinations of the dark, jitter,
  correlated, and smearing stages, proving that a skipped block constructs no
  identity result or replacement tensor, smearing without correlation uses the
  unit-response `S1 == S2` identity, lowercase `charge` remains a tensor, and
  its final value alone becomes the uppercase `Charge` payload;
- exact private orchestration with `simulate_dark_counts(...)` before
  `simulate_timing_jitter(...)` whenever both blocks execute;
- timing jitter affecting the then-current private working counts, including
  dark roots when present, but never the public truth field;
- `K=0` roots-only and `K=1` direct-child off-by-one fixtures for the one
  coupled `simulate_correlated_avalanches(...)` path;
- contextual preflight fixtures proving `K=0` skips every delay/recovery
  numerical gate, a zero CT mean skips that mode's kernel, and zero AP
  probability skips both AP delay and recovery preparation, even when the
  unused config/sampling pair would fall outside an active kernel's numerical
  domain; all remain draw-free and retain structural config validation;
- all eight DiCT/DeCT/AP enablement combinations, including exact
  all-disabled identity and no-draw behavior;
- one frozen unmarked integer frontier per generation, with every retained
  child entering the next frontier exactly once and no charge value entering
  an offspring law;
- exact eager traversal in increasing generation order; direct CT, delayed CT,
  then AP mechanism order; increasing CT source bins within each destination;
  increasing AP source bins and retained offsets before overflow/stop; and no
  repeated-index scatter or atomic reduction with unspecified accumulation
  order in the reference path;
- separate DiCT and DeCT rate fields, ordinary Poisson draws, streams,
  accumulators, and right-overflow diagnostics, with no rate superposition,
  conditional mode split, Gamma latent, or negative-binomial substitution,
  plus distinct retained-destination and overflow-source positional fixtures;
- analytic fixed- and exponential-delay PMFs under independent per-edge
  uniform phase marginalization, plus their frozen binary64 preparation
  contracts, zero-delay DiCT staying in-bin, no shared or inherited phase, and
  no correlated-stage underflow;
- fixed-delay fixtures for exact zero, `0.25*T`, `2.25*T`, exact `m*T`, and
  `math.nextafter` immediately below and above a representable boundary;
  exact two-point mass conservation; rejection when a nonzero exact remainder
  collapses to a represented deterministic law; source-relative overflow
  values zero, `f`, and one; window-stop and huge finite all-overflow plans;
  exact integer-ratio unit conversion without a boundary-crossing
  multiplication; no signed `source + n` formation; and exact absence of a
  delay-specific RNG request;
- exponential-delay fixtures across
  `mean_delay/T in {2**-52, 2**-40, 1e-6, 0.1, 0.5, 1, 2, 16,
  2**40, 2**52}`, both sides of the `x=0.5` central-mass branch, sample counts
  `{2, 3, 8, 64, 512, 8192}`, and immediate rejection outside every ratio or
  sample-count bound; independent 100-decimal-digit checks of every retained
  category and analytic right tail; nonincreasing tails; natural far-tail
  underflow; `1e-12` local identities; `1e-11` complete-law L1 error; and exact
  absence of cutoff, clipping, residual assignment, renormalization, or a
  delay-specific RNG request;
- stable AP `A`/`B` conditional masses and analytic overflow tails from the
  exponential law, plus exponential-recovery fixtures spanning both accepted
  delay/recovery ratio domains, the auxiliary
  `2**-51 <= x + y <= 2**53` domain including both endpoints, and all three
  frozen log-difference branches; high-precision agreement for `rho_bar[d]`,
  `h[d]`, and `h_ap_tail[L]`; exact
  `recovery=None` unit response; finite `0 <= h <= q` and
  `0 <= rho_bar <= 1` without clipping; the identities
  `h[d] + c*q_(x+y)[d] = q_x[d]` and
  `h_ap_tail[L] + c*R_(x+y)[L] = R_x[L]`; analytic zero-tail handling without
  division; and recovery changing neither realized AP count/destination nor
  descendant branching;
- exact absence of `NormalDelayConfig` from the active class, union, export,
  and package-contract surfaces after completed Stage 6, while the closed
  Stage 3 work order remains unchanged as historical evidence;
- AP's one-child multinomial law, shared realized categories for integer count
  and deposited charge, and separate stop, retained, and right-overflow
  accounting;
- `recovery=None` reducing AP `count`, `charge`, and `charge_square_sum` to the
  unit-weight law, plus configured recovery changing deposited charge without
  changing descendant probabilities;
- the exact integer mechanism-count invariant and dtype-aware validation of the
  mathematical S1/S2 ledger identities, including
  `afterpulse_charge_square_sum` as the sum of category weights squared rather
  than `afterpulse_charge**2`;
- all overflow excluded from the retained frontier, total count, `S1`, `S2`,
  terminal `Charge`, and waveforms;
- final-frontier truncation semantics after generation `K`, checked count
  overflow, and no partial valid result after an algorithm failure;
- independent-edge prepared PMFs compared with an explicit edge-level Monte
  Carlo reference, with the intentionally omitted shared-phase and
  within-category recovery variances measured as named approximation
  boundaries;
- zero dark rate and zero jitter as exact identities without unnecessary RNG
  consumption;
- jitter conservation through explicit retained and dropped buckets;
- dark-only finite-window ensembles matching
  `mean_t = lambda * sum_s(P[s, t])`, Poisson variance and zero probability,
  expected edge depletion, and expected total drop;
- conditional isolated IV dark-count-plus-jitter comparison under matching
  gate and drop conventions, while explicitly excluding full recursive-chain
  timing equivalence;
- common output invariance when unrelated requested products change;
- private product-builder and submodel fixtures;
- public single-product and multi-product composition fixtures;
- an exact returned source `Photoelectrons` field when requested, guaranteed-
  fresh generated fields independent of every named input, and pairwise
  storage-independent generated result fields;
- no new write initiated or enqueued after a field becomes observable, plus
  same-stream and explicit cross-stream ordering behavior;
- ordinary-`torch.Tensor` execution evidence, with custom tensor subclasses and
  dispatch modes explicitly unsupported rather than exhaustively detected;
- no CPU, NumPy, Python-list, device-movement, in-place/source-replacement,
  input-normalization, or detach path for an existing input payload; declared
  fresh generated-product dtype conversion remains required, including
  `Photoelectrons[torch.int64]` to floating Charge, and small config-derived
  scalar/template preparation remains allowed as explicitly documented by the
  owning producer;
- transform-specific scientific/parity tests;
- CPU tests and conditional CUDA tests with accurate qualifications;
- stale `0.6` names and compatibility aliases absent; and
- import isolation from TensorG4DS, TensorML, Projects/dag, and IO backends.

## Rebuild Migration

This Design pass accepted the architecture, synchronized the live Design
documents, and wrote the focused Stage 3 structural-foundation work order.
Stage 3 subsequently selected exact TensorCore `0.7.0` commit
`b454d738f6385ce6489d85492a618a3dab139bb6`, passed fixed-pin consumer probes,
and merged the product/config/collection foundation. Historical work orders
and governance records remain unchanged.

The completed production steps are:

- commit synchronized Design authority, verify the persistent role routes,
  replace the `0.6` package with the typed axes, sampling, product/config, and
  collection foundation, then clear fixed-commit Validation, independent
  Review, fast-forward merge, and Design closeout; and
- implement the deterministic pure, analog, and digitized product producers
  under the focused Stage 4 work order, then clear the same fixed-commit
  Validation, independent Review, merge, and Design closeout gates; and
- implement the private positional RNG behavior consumed by exact-zero,
  IID-white, and caller-supplied PSD noise under the focused Stage 5 work
  order, then clear its fixed-commit Validation, independent Review, merge,
  and Design closeout gates; and
- implement the complete private Charge producer under the focused Stage 6
  work order, including aggregate multinomial and hybrid Poisson samplers,
  dark counts, analytic timing jitter, the fixed-`K` DiCT/DeCT/AP cascade,
  S1/S2 ledgers, overflow diagnostics, smearing, and all eight Charge streams,
  then clear fixed-commit Validation, independent Review, merge, and Design
  closeout gates.

The completed prerequisites and remaining production sequence are:

1. Maintenance 2 is Merged / Closed through exact implementation candidate
   `89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
   `9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`, against selected TensorCore
   `0.9.0` commit `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`.
   It split config/field/collection ownership, created the focused Charge
   effects package, migrated generic RNG and count-distribution use to
   TensorCore, retained multinomial/category orchestration and checked count
   bookkeeping in `_counts.py`, added config-owned keys, removed `_RngStream`
   and `readout/_random.py`, consolidated the private scalar-to-dtype
   requirement, and preserved default-key continuity.
2. Stage 7 is Merged / Closed through exact Review-cleared candidate
   `6dd55024685013fb9412a7247d3ddde7be1a3177`. It implements request-aware
   `simulate_readout(...)`, whole-request preparation, product-owned plans,
   required `rng: CounterRng`, closure-wide duplicate-key validation,
   execute-once planning, exact requested retention, and generated-product
   postconditions.
3. Maintenance 4 Runtime Action Ownership is Merged / Closed through exact
   Review-cleared candidate `b3c7c907004741ba67b8b92a54bbdc8c85216dda`.
   It realizes explicit ProductRuntime, prepare, produce, and validate
   ownership while preserving Stage 7 behavior.
4. Maintenance 5 Compact Axes And Sampling is Merged / Closed.
5. Maintenance 6 Pint Physical Configuration Boundary is Merged / Closed
   through exact Review-cleared target
   `0257fb477ee04556ebbe26351123ae610b5d7925`.
6. Maintenance 7's adoption of published TensorCore `0.15.0` and matching
   TensorDSLab validation/RngPositions cleanup is Merged / Closed through exact
   target `205182f0c7a4359cece79211ad22b47b522c34e3`.
7. The first Stage 8 real-CUDA attempt remains stopped evidence. Any rerun
   requires a new Design authority after Maintenance 6.
   Profile real GPU memory and execution before designing workspace/output
   reuse.
8. Design the exact TensorG4DS-to-truth-Photoelectrons bridge.
9. Design explicit TensorML/reconstruction adapters.
10. Design durable artifacts only after in-memory contracts stabilize.

Each production slice uses the repository Implementation/Validation/Review
loop and fixed-commit evidence. No compatibility alias preserves `0.6`.

## Supersession Ledger

The accepted rebuild architecture and completed Stage 3 foundation replace
the historical `0.6` contracts in the first table.

| Historical `0.6` contract | Implemented Stage 3 or accepted rebuild target |
| --- | --- |
| TensorCore `0.6` ID/layout records | TensorCore `0.7` semantic roots |
| `TensorAxisId`, `TensorFieldId`, `IdSequence` | exact final leaf classes |
| `ExampleId`, `ChannelId` objects | strings scoped by exact axis type |
| `TensorLayout` plus `shared_axes` | ordered axes directly on each field |
| three required axes plus optional extra shared axes | exactly three readout axis types |
| sealed generic `TensorField` | six direct TensorDSLab product leaves |
| loose axis/field constants and registries | class-owned schema and typed calls |
| count-only sample plus collection-sidecar `SampleGrid` | compact integer-picosecond `SampleAxis(RegularAxis)` and source-derived `SamplingRuntime` |
| `DigitizedWaveformSpec` sidecar | builder config held externally; artifact binding deferred |
| partial ordered pipeline snapshots | request-selected completed unordered results |
| descendant invalidation | immutable one-shot construction |
| public timing transform replacing photoelectrons | truth photoelectrons; private charge-only jitter |
| public atomic collection transforms | private typed product producers plus one public request API |
| generic selection/movement plus reconstruction | explicit downstream operations when needed |
| `readout/tensors.py` | retired |
| global config/field/builder modules | product-owned public `config.py` / `field.py`, non-exported runtime actions, and thin `readout.simulation` orchestration |
| immediate public workspace architecture | functional first; optimize after measurement |
| field-ID model selection | explicit ordered product-type selection |
| field-ID parity boundaries | product-request/builder parity boundaries |
| semantic-coordinate RNG identity | fixed package-owned `RngKey` plus validated flat `RngPositions` |

The closed Maintenance 2 implementation supersedes these Stage 5/6 choices:

| Closed Stage 5/6 implementation | Maintenance 2 implementation |
| --- | --- |
| bare producer/invocation `seed=` plus central `_RngStream` | required invocation `CounterRng` plus leaf-config-owned `RngKey` |
| TensorDSLab-owned generic RNG engine in `readout/_random.py` | TensorCore generic RNG/distribution surface plus TensorDSLab-owned scientific keys, lattices, multinomial orchestration, and bookkeeping |

This synchronization pass updated:

- `AGENTS.md`;
- `CONTRIBUTING.md`;
- `docs/overview.md`;
- `docs/design.md`;
- `docs/decisions.md`;
- `docs/architecture/tensors.md`;
- `docs/architecture/readout.md`;
- `docs/parity.md`;
- `docs/validation.md`; and
- `docs/implementation/index.md`.

It also created the Stage 3 work order, which was later implemented, validated,
independently reviewed, fast-forwarded, and accepted as Merged / Closed.
Governance records and earlier completed work orders remain historical records.

The subsequent Stage 4 Design pass created the focused
`stage_4_deterministic_waveform_products.md` work order and synchronized the
functionality-first execution decision. Stage 4 is now Merged / Closed and
implements exactly the private pure, analog, and digitized producers. It added
no public API. The subsequent Stage 5 private-RNG and complete-noise work order
is also Merged / Closed through exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd` and Review closeout
`c6a506d3658b24197806b9e230480211a254a35a`. It likewise added no public API.
Stage 6 is Merged / Closed through exact implementation candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58` and Review's evidence-only
closeout `ea979862b05f4ef543f6971c86641df317232479`. It completed the private
Charge producer and likewise added no public API. Its evidence is eager
CPU-only because CUDA was unavailable. Measured GPU fusion and cross-backend
Charge evidence remain later gates.

## Closed Decisions And Remaining Design Gates

The merged Maintenance 4 package tree and import ownership are implemented and
closed. Shared semantic axes and sampling live in `tensor_dslab.common`;
each generated readout product owns its field, configs, ProductRuntime, and
explicit preparation, production, and validation actions; `Photoelectrons`
remains the producer-free truth input with an ingress validator;
`readout.config` contains only `ReadoutConfig`, `readout.collection` contains
only `ReadoutCollection`; and `readout.simulation` owns the one public
request-level orchestration function. The runtime packages, shared readout
requirements, and Charge effect modules are not public APIs.
`Photoelectrons` is an already-produced input with neither a config nor a
producer. The former Stage 6 `types.py` and `_random.py` layout remains closed
Stage 5/6 implementation evidence. Closed Maintenance 2 realizes the initial
product/module ownership portion without aliases, closed Stage 7 completes
`readout/simulation.py`, and closed Maintenance 4 realizes the internal
ownership replacement in present production. Reopening the implemented
structure requires a concrete import-cycle, cohesion, or implementation-size
finding rather than a preference for layer-oriented grouping.

Historical Stage 3 through Maintenance 4 sampling used one exact
`SamplingConfig` that owned positive integer `sample_period_ps`,
`sample_count >= 2`,
and a signed-int64 `window_stop_ps`. Its full axis begins at zero, contains
exactly the `N` canonical ASCII left-edge timestamps `i * T` in lowercase
picoseconds, and omits the exclusive `N * T` stop. Every `SampleAxis` is a
regular, increasing, period-bearing signed-int64 time axis; only complete
simulation inputs additionally require zero start and exact config agreement.
Kernels use numeric config values and indices, while upstream floating-time
normalization remains a TensorG4DS-bridge decision. Maintenance 5 supersedes
that representation with compact `SampleAxis(start, step, count)`,
source-derived `SamplingRuntime`, and no `SamplingConfig`, as fixed at the top
of this document.

The two pulse-shape equations and their donor-to-config parameter mappings are
closed provisionally for the MVP by the IV adoption above. Later collaborator
review may motivate a new scientific model or calibration, but it does not
block implementation of these explicitly provisional equations and must not
silently change them inside an implementation stage.

The complete PSD-shaped-noise mathematical contract is also closed for the
MVP: absolute one-sided interval density, overlap integration onto fixed-length
odd/even target cells, explicit `[0, df / 2)` DC-cell discard without
redistribution, Gaussian one-sided coefficients, two independent real
standard-normal components per interior coefficient,
`torch.fft.irfft(..., n=N, dim=-1, norm="backward")`, the documented endpoint
scales, retained expected variance, circular covariance, independent rows, and
no post-transform normalization or hidden longer-record crop. Stage 5
historically assigned streams `1` and `2`; Maintenance 2 preserves them as
then-config-owned white/PSD keys, and Maintenance 7 fixes the same values in
the private key table. The accepted
precision-matched uniforms and Box-Muller pair define how the corresponding
standard-normal values are generated. Neither choice reopens this PSD law.

Waveform baseline ownership is closed as well. The MVP has no deterministic
analog pedestal: pure and noise are zero-referenced voltage components, analog
is their optionally saturated sum, and the digitizer's endpoint-guarded affine
transfer owns the nonzero ADC code corresponding to 0 mV.

The Threefry word and positional address mapping are closed as a TensorDSLab
consumer requirement. RNG schema `tensordslab.threefry4x32-20/v1` uses the
exact standard Random123 `Threefry4x32_R<20>` word algorithm,
seed/stream/namespace key packing, logical-position/quantum/raw-word-block
counter packing, lane selection, and accepted bounds specified above. Stage 5
closed current raw-bit generation for its two noise roles; Stage 6 implemented
the eight Charge roles. Maintenance 2 transfers generic mechanics to
TensorCore without changing default-key addresses.

The noise-required distribution layer is closed for Stage 5:
precision-matched Random123-style `float32` and `float64`
closed-open/open-open conversions and address-local ordered Box-Muller pairs.
Ties-to-even 32-bit Bernoulli thresholds and native-dtype exponential inversion
remain unused generic candidate mechanics rather than Stage 5 or Stage 6
implementation scope.
The documented finite Box-Muller tail is an accepted bounded-MVP approximation
for actual variate consumers such as noise and charge smearing, not timing
jitter and not a hidden claim of unbounded continuous support.

Stage 3 completed the TensorCore selection, inherited-constructor typing,
public-import, and fixed consumer-probe gate at exact commit
`b454d738f6385ce6489d85492a618a3dab139bb6`.

Closed prerequisites and remaining Design gates are:

1. Maintenance 2 is Merged / Closed through exact implementation candidate
   `89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
   `9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. It preserved Stage 5/6 output
   continuity, kept Charge multinomial/category orchestration and checked count
   bookkeeping local, consolidated `_require_representable_float`, and removed
   `_RngStream`, `readout/_random.py`, and any replacement `readout/_rng.py`
   without shims.
2. The focused Stage 7 work order is Merged / Closed through exact candidate
   `6dd55024685013fb9412a7247d3ddde7be1a3177`. It implements
   request/config/RNG/key-collision preparation, product-owned typed plans,
   prerequisite execution at most once, requested-only retention, and the
   public `simulate_readout(...)` surface. Stage 6 already closed the private
   Charge scientific transition, delay/recovery preparation, stabilized
   aggregate samplers, all ten default role addresses, positional schedules,
   per-cell count ceiling, relational generation/address bounds,
   ledger/smearing envelope, failure effects, and TensorDSLab-model statistical
   policy on eager CPU. Standalone Bernoulli and sampled continuous-exponential
   equations remain recorded, but their implementation and evidence activate
   only if a later accepted consumer actually uses them.
3. Maintenance 4 Runtime Action Ownership is Merged / Closed through exact
   Review-cleared candidate `b3c7c907004741ba67b8b92a54bbdc8c85216dda`.
   It preserved the exact public, scientific, RNG, numerical, storage, and
   autograd boundary while realizing the product runtime action tree, shared
   `SamplingRuntime`, complete closure preparation, and immediate validation
   before descendant use.
4. Maintenance 5 is Merged / Closed at exact TensorCore `0.13.0`; compact
   axes and source-derived sampling are the implemented baseline.
5. Maintenance 6 is Merged / Closed through exact Review-cleared target
   `0257fb477ee04556ebbe26351123ae610b5d7925`: it adopts exact Pint `0.25.3`
   at the public physical Config boundary, extracts unit-free Runtime operands
   once, and completes the bounded private-admission cleanup without changing
   readout science.
6. Any Stage 8 rerun requires a new Design authority after Maintenance 6.
   Waveform-tail optimization evidence after the
   functional producers are accepted: compiler/execution mode, equivalence to
   the frozen eager reference, one-kernel/no-target-sized-temporary
   instrumentation, and the fallback gate for a purpose-built kernel.
   Cross-product analog/digitized
   fusion remains excluded.
7. Maintenance 7 TensorCore `0.15.0` adoption and matching TensorDSLab cleanup
   are Merged / Closed through exact target
   `205182f0c7a4359cece79211ad22b47b522c34e3`. Local `main` remains unpushed;
   any integrated CUDA gate remains separately authorized.
8. Digitization-config association for independent/durable consumers.
9. Exact TensorG4DS source and dense truth-binning bridge, including provenance
   origin, left-edge construction, exact boundary assignment at `0`, `i * T`,
   and exclusive `N * T`, plus `underflow_hit_count` and
   `overflow_hit_count` accounting.
10. Whether typed collection convenience properties materially improve the
    API.

The fixed-`K` correlated-avalanche model is implemented and closed on eager
CPU: exact config ownership, independent per-edge phase closure,
ordinary separate DiCT/DeCT Poisson laws, AP's bounded categorical law,
fixed/exponential CT delay families, optional composed exponential recovery
response, unmarked cross-feeding, S1/S2 ledgers, terminal
smearing rule, causal right-overflow policy, and private diagnostic vocabulary
are implemented under the selected baseline. CUDA execution and CPU/CUDA
agreement remain unestablished because CUDA was unavailable. No work order may
substitute a same-bin closure, generation-wave plan, marked recovery process,
Gamma-Poisson law, or separate public mechanism pipeline for this baseline
without a new Design decision.

`Config(ABC)`, product-level `persist` flags, jagged builder input, and public
truth-replacing timing jitter are deliberately omitted. Persistence remains a
future focused design.

## Collaborator Example

Illustrative pseudocode, using already-supplied validated `photoelectrons` and
`config` values:

```python
readout = simulate_readout(
    photoelectrons,
    products=[
        AnalogWaveform,
        DigitizedWaveform,
    ],
    config=config,
    rng=Threefry4x32(seed=1234),
)

analog: AnalogWaveform = readout.field(AnalogWaveform)
digitized: DigitizedWaveform = readout.field(DigitizedWaveform)

assert readout.field_types == frozenset(
    {
        AnalogWaveform,
        DigitizedWaveform,
    }
)
```

Requesting truth beside a derived product is unambiguous:

```python
readout = simulate_readout(
    photoelectrons,
    products=[Photoelectrons, Charge],
    config=config,
    rng=Threefry4x32(seed=1234),
)

assert readout.field(Photoelectrons) is photoelectrons
charge = readout.field(Charge)  # reflects every effective configured charge stage
```

The builder computes what is necessary, retains exactly what was requested,
and never changes the truth input.
