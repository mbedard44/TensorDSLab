# Design

## Core Thesis

TensorDSLab is a clean-slate, tensor-native detector data-lab package. It
accepts dense TensorDSLab truth photoelectrons, simulates the configured
readout response, and returns exact typed products without asking ordinary
collaborators to understand TensorCore internals.

The intended ecosystem flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is a data-flow statement, not an import graph. TensorCore is the shared
semantic tensor substrate beneath the three tensor packages. TensorDSLab owns
post-TensorG4DS detector/readout meaning and future reconstruction products; it
does not own native G4DS parsing, TensorG4DS clustering, TensorML training, or
campaign orchestration.

The accepted rebuild target is specified in
[`architecture/rebuild.md`](architecture/rebuild.md). Stage 3 replaced the
historical pre-deployment TensorCore `0.6` representation without a
compatibility layer. Stage 2 and Maintenance 1 remain historical evidence and
do not constrain the semantic-root architecture introduced against TensorCore
`0.7`. Maintenance 2's exact TensorCore `0.9.0` pin remains closed historical
RNG-ownership evidence. Closed Maintenance 5 adopts published
TensorCore `0.13.0` exact commit
`202d8b1bc6259b8453d3d377570417f2480d782b` for compact semantic axes and
the accepted golden-path structural boundary. TensorCore `0.13.0` also exposes
generic `Scalar`. Maintenance 6 is an Implementation candidate under a
Design-amended fixed-commit Validation gate. It selects exact Pint `0.25.3`
and uses `Scalar.require(...)` at the package-owned physical-configuration
boundary without putting Pint into TensorCore or tensor execution.

## Target Collaborator Surface

The normal workflow is one public function plus named product classes:

```python
readout = simulate_readout(
    photoelectrons,
    products=[AnalogWaveform, DigitizedWaveform],
    config=config,
    rng=Threefry4x32(seed=1234),
)

analog = readout.field(AnalogWaveform)
digitized = readout.field(DigitizedWaveform)
```

The returned `ReadoutCollection` contains exactly the requested product types.
The builder computes transitive prerequisites privately, computes each at most
once, and does not retain an intermediate unless the caller requested it.
Product-request iteration order has no semantic meaning.

Before executing any product, the builder composes private product-owned
Runtime records for the complete closure and checks closure-wide
stochastic-key uniqueness. Scientific preparation remains beside each product;
the public orchestration layer owns only request planning, the topological
`produce -> validate -> descendant` sequence, exact retention, and final
collection construction.

The public concepts are therefore:

```text
Photoelectrons   already-produced dense truth input
products         final in-memory retention request
ReadoutConfig    immutable scientific configuration
CounterRng       algorithm plus invocation seed
RngKey           config-owned stochastic role identity
simulate_readout dependency planning and execution
ReadoutCollection immutable completed requested result
```

Persistence and IO are not implied by `products` and are deferred.

## TensorCore Spine

Stage 3 introduced TensorCore's semantic tensor roots against exact `0.7.0`.
Maintenance 5 retains those concepts through the published `0.13.0` public
surface:

```text
TensorAxis[CoordinateT]
  CountAxis
  RegularAxis
  LabelAxis
TensorField        tensor payload plus ordered semantic axes
TensorCollection   immutable fields keyed by exact field type
```

Exact TensorCore `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b` is the accepted Maintenance 5
baseline. Exact consumer probes establish public imports, constructors,
typing, archive identity, and scientific continuity. Maintenance 6 retains
that exact pin. This is not a broad compatibility claim.

Every TensorDSLab semantic axis, field, or collection leaf has exactly one
appropriate TensorCore root in `__bases__`, with no mixin or other base, and is
public, `@final`, fieldless, and empty-slotted. TensorCore owns universal
representation validation; TensorDSLab
owns its exact axis, dtype, device, value-domain, product, and collection
relationships. Exact Python types replace axis IDs, field IDs, layout records,
semantic constants, and runtime product-name registries.

TensorCore `0.13.0` still provides no generic selection, movement,
reconstruction, output-buffer, workspace, or lifecycle service. It does
provide generic table roots and `TensorArtifact`; Maintenance 5 adopts neither
a TensorDSLab table nor an artifact/IO policy. TensorDSLab adds domain behavior
only where a real readout operation requires it.

The exact TensorCore `0.9.0` dependency selected historically for Maintenance 2 at
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59` adds public `RngKey`,
`CounterRng`, `Threefry4x32`, `logical_positions`, and
`require_same_dtype` surfaces. TensorCore owns generic counter generation and
exactly the public distribution methods `uniform(...)`, `gaussian(...)`,
`poisson(...)`, and `binomial(...)`.
`gaussian(...)` is parameterized by mean and standard deviation; there is no
public standard-normal method. TensorCore owns fixed-point conversion,
Box-Muller and affine mapping, Poisson inversion/PTRS, binomial inversion/BTRS,
sampler numerical domains and exhaustion, and the count distributions'
internal word schedules.
TensorCore's `require_same_dtype(...)` compares semantic field dtypes without
casting or adding a dtype allowlist. TensorDSLab uses it for Analog inputs and
only the floating subset of `ReadoutCollection`; raw tensor requirements and
one non-exported scalar representability helper remain TensorDSLab-owned.
TensorDSLab owns config placement of stochastic keys, scientific
position/category lattices, direct-uniform/Gaussian ordinals, draw-free
scientific policy, multinomial ordering and final remainders, count
accumulation, and ledgers. The Maintenance 2 implementation pins the
exact `0.9.0` dependency; closed Stage 3 through 6 evidence remains scoped to
the `0.7.0` pin. TensorCore `0.13.0` preserves those public RNG and
relationship surfaces; Maintenance 5 changes their dependency location, not
their TensorDSLab ownership or behavior.

## Semantic Axes And Sampling

Every readout product has exactly one `ExampleAxis`, one `ChannelAxis`, and one
`SampleAxis`. Their tuple order is tensor dimension order and may vary
semantically; code locates a dimension by exact axis type rather than a loose
ID or a fixed position. Builders reuse the exact source axes tuple and axis
instances for every dimension-preserving generated product.

Maintenance 5 gives each semantic role the generic representation it needs:

- `ExampleAxis(CountAxis)` stores only a nonzero count and exposes zero-based
  local integer ordinals;
- `ChannelAxis(LabelAxis)` stores nonempty unique string detector labels; and
- `SampleAxis(RegularAxis)` stores integer-picosecond `start`, positive
  `step`, count at least two, and a signed-int64-bounded exclusive stop.

Count and sample coordinates are nonmaterializing `range` values. The source
`SampleAxis` is the sole sampling policy. Private
`prepare_sampling(photoelectrons)` derives count, period, and dimension once.
The complete readout boundary requires example-local `start == 0`; a semantic
sample axis may represent a valid nonzero-start subgrid. Kernels and positional
RNG use tensor indices and plain prepared integers, not semantic coordinate
values.

`SamplingConfig`, timestamp strings, the earlier count-only sample proposal,
and `SampleGrid` are retired without shims.

## Product Semantics

TensorDSLab defines six exact final field types:

| Product | Payload | Meaning |
| --- | --- | --- |
| `Photoelectrons` | `torch.int64`, nonnegative | binned photon-origin truth PE counts |
| `Charge` | `torch.float32` or `torch.float64`, finite and nonnegative | aggregate PE-equivalent SiPM response |
| `PureWaveform` | `torch.float32` or `torch.float64`, finite | signal-only waveform in mV |
| `NoiseWaveform` | `torch.float32` or `torch.float64`, finite | zero-mean noise excursion in mV |
| `AnalogWaveform` | `torch.float32` or `torch.float64`, finite | composed zero-referenced analog waveform in mV |
| `DigitizedWaveform` | `torch.int32`, nonnegative and config-bounded | immediate ADC-code output |

`DigitizedWaveform` remains the accepted name; `DigitalWaveform` is reserved
for a possible later firmware or trigger product.

`Photoelectrons` is an already-produced dense truth input. It has no
`PhotoelectronsConfig` and no TensorDSLab readout producer. A future explicit
TensorG4DS bridge will construct it from an accepted upstream product using the
bridge-selected compact `SampleAxis`. Dark counts, timing jitter, crosstalk,
afterpulses, and
charge smearing never mutate or relabel this truth field.

`ReadoutCollection` is a nonempty immutable completed result containing any
requested subset of the six exact product types. Membership is unordered. All
present fields have equal ordered axes, one device, and one common dtype among
floating products. It is not a partial pipeline snapshot and has no public
add, replace, descendant-invalidation, or mutable-output lifecycle.

## Configuration And Scientific Chain

`ReadoutConfig` composes only optional exact generated-product configs. The
source axis owns sampling, so `ReadoutConfig()` is the valid truth-only
configuration. Each product config belongs with its product. Optional
scientific submodels use `None`; alternative algorithms use closed unions of
exact config types. There is no generic `Config` ABC, string algorithm switch,
product-level persistence flag, or scientific config containing runtime buffer
policy.

Maintenance 6 fixes the physical-value boundary as:

```text
caller Quantity
  -> frozen Config with a copied canonical scalar Quantity
  -> prepare_<product> extracts one plain canonical magnitude
  -> unit-free ProductRuntime
  -> unit-free producer and validator
```

TensorDSLab owns one private Pint registry and exports only `quantity(...)`
and `quantities(...)` construction helpers. Public physical field names are
unit-neutral because the value carries its unit; private Runtime values retain
unit suffixes. Each canonicalization uses exactly one TensorCore
`Scalar.require(...)` normalization and stores no Scalar wrapper. All public
Configs are explicitly unhashable. Configs containing only dimensionless
control/composition state remain ordinary Python/TensorCore records but follow
the same unhashable contract.

Config `__post_init__` is reserved for real construction behavior: Pint
canonicalization, unwrapped primitive-domain validation, or genuine local
relationships such as ordering, matching lengths, and distinct keys. It does
not repeat annotations by checking `Scalar`, `RngKey`, nested Config,
optional, or closed-union membership. Static typing and Review own that
supported composition; malformed typed composition has no stable runtime
diagnostic promise. No generic Config ABC is introduced.

The selected high-level computation is:

```text
Photoelectrons truth
  -> optional dark-count seeds
  -> optional private timing jitter
  -> optional fixed-generation coupled correlated avalanches
  -> optional terminal charge smearing
  -> Charge
  -> PureWaveform

Photoelectrons axes/device/shape + noise config -> NoiseWaveform
PureWaveform + NoiseWaveform -> AnalogWaveform -> DigitizedWaveform
```

Private timing jitter uses an analytically prepared latent-uniform plus
ideal-Gaussian offset law. Runtime samples aggregate destination counts with
conditional binomials and one final drop remainder; it never expands PEs or
draws one Gaussian value per PE. The first implementation preserves every
possibly in-window destination without an arbitrary tail cutoff, even though
that correctness-first rule can make work quadratic in sample count.
The accepted binary64 preparation uses a log-domain one-sided tail over
`2**-52 <= sigma / T <= 64` and `2 <= sample_count <= 8192`, with stable
success/later-category conditional masses, `1e-12` local probability
tolerance, `1e-11` source-law L1 tolerance, and the dedicated append-only
`TimingJitterConfig.rng_key` whose default stream is `8`. Unsupported values fail before
RNG use rather than being clipped or normalized.

The fixed-generation charge algorithm, pulse equations, PSD construction,
analog saturation, ADC transfer, RNG schema, and deliberate donor divergences
are specified in [`architecture/rebuild.md`](architecture/rebuild.md) and
[`parity.md`](parity.md). Those documents are the scientific source; this page
does not duplicate their equations.

## Product-Centered Package Ownership

The accepted rebuild tree is organized around semantic products:

```text
tensor_dslab/
  __init__.py

  common/
    __init__.py
    axes.py                  # ExampleAxis, ChannelAxis, SampleAxis

  readout/
    __init__.py
    config.py                # ReadoutConfig
    collection.py            # ReadoutCollection
    requirements.py          # shared non-exported readout relationships
    simulation.py            # sole public simulation API
    runtime/
      __init__.py            # empty; no internal facade
      prepare.py             # ReadoutRuntime and complete request preflight
      sampling.py            # shared SamplingRuntime and one-time binding

    photoelectrons/
      __init__.py
      field.py               # Photoelectrons; no config or producer
      runtime/
        __init__.py
        validate.py          # untrusted-ingress deep validation
    charge/
      __init__.py
      config.py              # charge configs and default RngKeys
      field.py               # Charge
      runtime/
        __init__.py
        prepare.py           # ChargeRuntime and prepare_charge()
        produce.py           # produce_charge()
        validate.py          # validate_charge()
        effects/             # prepared and executing scientific submodels
          __init__.py
          counts.py
          delays.py
          dark_counts.py
          timing_jitter.py
          correlated_avalanches.py
          smearing.py
    pure_waveform/
      __init__.py
      config.py
      field.py
      runtime/{__init__.py,prepare.py,produce.py,validate.py}
    noise_waveform/
      __init__.py
      config.py
      field.py
      runtime/{__init__.py,prepare.py,produce.py,validate.py}
    analog_waveform/
      __init__.py
      config.py
      field.py
      runtime/{__init__.py,prepare.py,produce.py,validate.py}
    digitized_waveform/
      __init__.py
      config.py
      field.py
      runtime/{__init__.py,prepare.py,produce.py,validate.py}
```

Maintenance 5 deletes `common/sampling.py`; `readout/runtime/sampling.py`
remains the private owner of source-derived `SamplingRuntime` and
`prepare_sampling(...)`.

This product-centered tree combines the Maintenance 2 public ownership target,
the Stage 7 public orchestration module, and the merged Maintenance 4 internal
Runtime/action split. Maintenance 2
realized the public product/module ownership migration; Stage 7 completed
`readout/simulation.py`. Closed Stage 5/6 production used `types.py`,
`_RngStream`, and `readout/_random.py`. TensorCore has
published its package-authoritative generic RNG plus independently testable
same-dtype sub-slice as version `0.9.0` at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. TensorDSLab Maintenance 2 is
Merged / Closed through exact candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. It pins that dependency, splits
the owned modules, and removes the retired local RNG surfaces without shims.
Stage 7 is Merged / Closed through exact Review-cleared candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177`.

Files are created only when an accepted implementation slice gives them real
behavior. Product Runtime modules do not import `ReadoutConfig`,
`ReadoutCollection`, or `simulate_readout`. `readout.runtime.prepare` may
compose product preparers; product Runtime modules must not import back into
that composition layer. `readout.simulation` remains the thin public layer
that executes the accepted topology and constructs the final collection.

Each generated product has three first-class non-exported actions:

```text
prepare_<product>   Config + execution facts -> ProductRuntime
produce_<product>   prerequisites + ProductRuntime (+ RNG) -> Product
validate_<product>  Product + minimal prepared facts -> None
```

Every ProductRuntime is one concrete final frozen slotted dataclass containing only
prepared tensors and static execution values. It contains no Config,
prerequisite semantic product, collection, mutable cache, or execution method.
There is no Runtime ABC, protocol, registry, generic action framework, or
string dispatch. One `SamplingRuntime` is constructed after source
`SampleAxis` validation and the exact same object is referenced by temporal
product Runtimes, so sample count, period, and dimension are bound once per
public request.

All required ProductRuntime values are prepared before the first producer,
RNG request, or semantic-output write. Production then performs tensor/RNG
execution and semantic-field construction without Config interpretation,
scientific-unit conversion, prepared-equation construction, or deep
publication scans. `simulate_readout` invokes each product validator exactly
once immediately after production and before any descendant. Product
`field.py` files retain only semantic identity and cheap intrinsic `_require()`
checks.

The accepted TensorCore `0.13.0` golden path is applied deliberately rather
than mechanically. `prepare_readout(...)` retains public ingress, closure,
dtype/device, RNG capability, and key-admission checks. Private child
preparers trust that admission and own contextual extraction, exact model
dispatch, scientific/representability checks, and Runtime construction.
Private Charge executors may trust exact Runtime and primitive types supplied
only through the typed path, but retain tensor relationships, count and
envelope limits, address/allocation bounds, and scientific laws. Generic
TensorCore relationship helpers are used only where their contract matches;
arbitrary axis order, exact source-axis tuple identity, absolute dtype domains,
storage freshness, and generated-product postconditions remain TensorDSLab
responsibilities.

The three action families are role-symmetric, not signature-identical.
Genuinely identical sampling, representability, finite-value, and relationship
logic is extracted into the narrowest non-exported owner, while explicit
product-named validators retain scientific ownership and product-specific
errors. Symmetry must not introduce unused parameters, untyped mappings, or a
broad `utils.py`/`helpers.py` dumping ground.

Privacy is export-driven. Runtime modules and clean action names remain
ordinary importable Python implementation details, but no Runtime, action,
effect, or shared requirement is exported from a public facade and none has a
compatibility promise. Runtime and effect `__init__.py` files are empty;
internal callers import exact defining modules. Historical Stage 6/7
`_produce.py`, `_produce_*`, and `*Plan` names remain true of those closed
candidates but are superseded as living architecture without shims.

## Functional, Storage, And Exposure Contract

The initial rebuild is functional and allocating. It has no public `out=`,
destination collection, workspace, allocator, lease, or warmed allocation-free
claim.

Each field-returning operation classifies its result using TensorCore's
operation-owned vocabulary. Initially:

- a requested `Photoelectrons` member is the exact source field;
- every generated product has guaranteed-fresh storage independent of named
  inputs;
- generated products retained together are storage-independent from one
  another; and
- dimension-preserving products reuse the exact source axes tuple and axes.

TensorDSLab initiates or enqueues all writes before constructing and exposing a
semantic field, and never later writes through an alias to that storage.
Private mutable scratch never enters a returned collection. A future measured
reusable-destination design must keep writable storage raw, exclusive, and
unexposed until producer writes have been enqueued; it may not revive the old
practice of overwriting a target already exposed as a valid field.

Operations do not silently move, cast, detach, or host-materialize an existing
input. This does not prohibit a producer's declared fresh generated-product
dtype conversion, including `Photoelectrons[torch.int64]` to floating Charge.
Accepted deep-value validation and producer postconditions use scalar
reductions that may synchronize CUDA as a functionality-first correctness
cost. Field construction itself adds no synchronization; outside those checks,
ordinary same-stream PyTorch ordering applies and cross-stream consumers
establish their own dependency.

## Validation Philosophy

Public boundaries validate legitimate public inputs, including exact product
requests, config relationships, axes, shape, dtype, device, sampling
agreement, value domains at untrusted ingress, an accepted `CounterRng`
instance, exact config-owned `RngKey` values, closure-wide key uniqueness, and
representable numerical bounds. Supported statically preparable request
failures occur before RNG requests, producer invocation, or semantic-output
writes. TensorCore exposes no non-consuming RNG capability query, so a real
custom-algorithm backend failure may occur only at its first genuine
distribution request and is an execution failure rather than preflight.

Cheap intrinsic leaf checks belong in `_require()`. Full-device value scans
belong at explicit trust boundaries and builder postconditions rather than in
every semantic constructor.

The selected Runtime/action architecture keeps deep validation at the public
simulation trust boundary but separates it from production. Every generated
product is constructed locally, passed to its product-owned
`validate_<product>` action exactly once with its named direct prerequisite
relationship, and only then made available to a descendant or the returned
collection. Invalid results do not cross that
boundary; the scan may synchronize CUDA. `validate_photoelectrons` remains an
untrusted-ingress preflight action. Complete closure preparation still occurs
before any RNG request, production call, or semantic-output write.

TensorDSLab does not harden itself against callers who deliberately leave the
public contract by subclassing final leaves, modifying classes, bypassing root
construction, calling private functions directly, mutating exposed tensors,
or installing custom Torch dispatch behavior. Such use is unsupported and has
no promised error category or adversarial-test obligation.

## Ownership Boundaries And Deferrals

The future TensorG4DS bridge is an explicit semantic conversion, not a subclass
cast. It will own provenance-to-example mapping, detector channel mapping,
dense PE binning on numeric left edges, boundary diagnostics, and exact-device
evidence. `simulate_readout` itself performs no IO, source loading, PE binning,
source movement, input-normalization cast, or provenance inference. This does
not prohibit a producer's declared fresh generated-product dtype conversion.

TensorML receives deliberately selected product fields through a future
model-facing boundary; a growing collection is not an implicit positional
model ABI. Reconstruction owns its own products and execution arrangements.
Projects/dag may orchestrate future public operations but does not own
scientific algorithms or in-memory product semantics.

Deferred work includes persistence and cache contracts, TensorG4DS and
TensorML adapters, reconstruction, public execution workspaces, advanced
buffer reuse, broad device/backend compatibility claims, and release or
backward-compatibility policy. A public `PureWaveformRenderer` is also deferred:
Maintenance 4 prepares a reusable internal preparation/production seam but
does not add, export, validate, or dispatch the renderer or any TensorML model
adapter.

## Parts-Bin Policy

Historical DSLab and IV-DSLab code may supply scientific facts, equations,
fixtures, and warning examples. It does not define current architecture.
Promoted behavior must fit the tensor-native design and name its comparison
boundary and parity class in [`parity.md`](parity.md). Intentional corrections
and statistical approximations are allowed when documented and validated at
the accepted boundary.

## Production Gate

Architecture documentation does not dispatch implementation. Each production
slice requires a focused work order, exact dependency and package baselines,
verified Implementation/Validation/Review routes, fixed-commit validation,
independent review, a clean Review merge gate, and Design closeout. The
TensorCore `0.7` structural foundation is Merged / Closed through Stage 3.
[Stage 4](implementation/stage_4_deterministic_waveform_products.md) is Merged /
Closed and implements the deterministic waveform producers. The private-RNG
and complete-noise
[Stage 5 work order](implementation/stage_5_readout_rng_and_stochastic_noise.md)
is also Merged / Closed through exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd` and Review closeout
`c6a506d3658b24197806b9e230480211a254a35a`. The complete private Charge
[Stage 6 work order](implementation/stage_6_charge_simulation.md) is also
Merged / Closed through exact candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58` and Review closeout
`ea979862b05f4ef543f6971c86641df317232479`. Fixed-commit Validation,
independent Review, and Design's post-merge audit found no unresolved issue;
CUDA was unavailable, so its evidence is eager CPU-only. Measured GPU
optimization remains later work. TensorCore RNG/same-dtype acceptance and
exact pin selection are complete at `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`; TensorDSLab Maintenance 2 uses
that exact dependency and is Merged / Closed. Stage 7 public orchestration is
also Merged / Closed through exact Review-cleared candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177` under its focused work order. Its
accepted evidence is eager CPU-only because CUDA was unavailable; measured GPU
characterization remains later work. Maintenance 4 Runtime Action Ownership is
**Merged / Closed** through exact Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda` under
[`implementation/maintenance_4_runtime_action_ownership.md`](implementation/maintenance_4_runtime_action_ownership.md).
It implements only the behavior-preserving internal action/tree refactor
described above and does not authorize the deferred renderer.

[Maintenance 5](implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md)
is **Merged / Closed** through exact Review-cleared supplemental candidate
`81ad2f52fe4a1966e5b3a0ceb5063138e42e731f` and Design closeout
`021694b9479d02546405f6a815aedf21c9c831a4`.
[Maintenance 6](implementation/maintenance_6_pint_physical_configuration_boundary.md)
is **Implementation candidate / Design-amended fixed-commit Validation
pending** at exact Candidate 1
`240e1492c466097b3059dfe9911ab338a4dd38a1`. Its exact work order authorizes
only the bounded Pint/config/runtime implementation and local evidence loop;
no Stage 8 rerun, IO/artifact surface, integration, or push follows.
