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
do not constrain the current TensorCore `0.7` package to the retired
representation.

## Collaborator Surface

The normal workflow is one public function plus named product classes:

```python
readout = simulate_readout(
    photoelectrons,
    products=[AnalogWaveform, DigitizedWaveform],
    config=config,
    seed=1234,
)

analog = readout.field(AnalogWaveform)
digitized = readout.field(DigitizedWaveform)
```

The returned `ReadoutCollection` contains exactly the requested product types.
The builder computes transitive prerequisites privately, computes each at most
once, and does not retain an intermediate unless the caller requested it.
Product-request iteration order has no semantic meaning.

The public concepts are therefore:

```text
Photoelectrons   already-produced dense truth input
products         final in-memory retention request
ReadoutConfig    immutable scientific configuration
seed             root for positional stochastic fields
simulate_readout dependency planning and execution
ReadoutCollection immutable completed requested result
```

Persistence and IO are not implied by `products` and are deferred.

## TensorCore `0.7` Spine

The rebuild uses the public TensorCore `0.7` semantic roots directly:

```text
TensorAxis       ordered string coordinates
TensorField      tensor payload plus ordered axes
TensorCollection immutable fields keyed by exact field type
```

The Design reference is exact clean TensorCore commit
`b454d738f6385ce6489d85492a618a3dab139bb6`. A production work order must
select the exact dependency pin and prove its public imports, runtime
constructors, inherited-constructor static typing, and operation-owned result
contracts. This reference is not a broad compatibility claim.

Every TensorDSLab semantic axis, field, or collection leaf has exactly one
appropriate TensorCore root in `__bases__`, with no mixin or other base, and is
public, `@final`, fieldless, and empty-slotted. TensorCore owns universal
representation validation; TensorDSLab
owns its exact axis, dtype, device, value-domain, product, and collection
relationships. Exact Python types replace axis IDs, field IDs, layout records,
semantic constants, and runtime product-name registries.

TensorCore does not provide generic selection, movement, reconstruction,
output buffers, workspaces, persistence, or lifecycle management in this
version. TensorDSLab adds domain behavior only where a real readout operation
requires it.

## Semantic Axes And Sampling

Every readout product has exactly one `ExampleAxis`, one `ChannelAxis`, and one
`SampleAxis`. Their tuple order is tensor dimension order and may vary
semantically; code locates a dimension by exact axis type rather than a loose
ID or a fixed position. Builders reuse the exact source axes tuple and axis
instances for every dimension-preserving generated product.

`SamplingConfig` defines the numeric regular grid:

- positive integer `sample_period_ps`;
- `sample_count >= 2`;
- example-local start at zero; and
- a signed-int64-representable exclusive stop.

It builds a `SampleAxis` whose ordered coordinates are canonical left-edge
timestamp strings such as `"0ps"`, `"2000ps"`, and `"4000ps"`. Kernels use
numeric config values and tensor indices; they do not parse coordinate strings
on the hot path. Positional RNG likewise uses indices rather than semantic
labels.

The earlier count-only sample axis and `SampleGrid` sidecar are retired.

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
same sampling policy. Dark counts, timing jitter, crosstalk, afterpulses, and
charge smearing never mutate or relabel this truth field.

`ReadoutCollection` is a nonempty immutable completed result containing any
requested subset of the six exact product types. Membership is unordered. All
present fields have equal ordered axes, one device, and one common dtype among
floating products. It is not a partial pipeline snapshot and has no public
add, replace, descendant-invalidation, or mutable-output lifecycle.

## Configuration And Scientific Chain

`ReadoutConfig` composes one required `SamplingConfig` with optional exact
product configs. Each product config belongs with its product. Optional
scientific submodels use `None`; alternative algorithms use closed unions of
exact config types. There is no generic `Config` ABC, string algorithm switch,
product-level persistence flag, or scientific config containing runtime buffer
policy.

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
stream `CHARGE_TIMING_JITTER = 0x0000_0008`. Unsupported values fail before
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
    sampling.py              # SamplingConfig

  readout/
    __init__.py
    types.py                 # ReadoutConfig and ReadoutCollection only
    simulation.py            # future Stage 7 public simulate_readout()
    _requirements.py         # shared private readout relationships
    _random.py               # private Stage 5/6 RNG and count samplers

    photoelectrons/
      __init__.py
      types.py               # Photoelectrons; no config or producer
    charge/
      __init__.py
      types.py               # Charge and charge configs
      _produce.py            # _produce_charge() and _simulate_* submodels
    pure_waveform/
      __init__.py
      types.py               # field and TPC/Veto pulse configs
      _produce.py
    noise_waveform/
      __init__.py
      types.py               # field and zero/white/PSD configs
      _produce.py
    analog_waveform/
      __init__.py
      types.py               # field and saturation config
      _produce.py
    digitized_waveform/
      __init__.py
      types.py               # field and digitization config
      _produce.py
```

Files are created only when an accepted implementation slice gives them real
behavior. Product packages do not import `ReadoutConfig`,
`ReadoutCollection`, or `simulate_readout`. `readout.simulation` is the sole
layer allowed to import and orchestrate the complete product graph. Package
roots deliberately re-export the collaborator-facing API; physical file paths
do not define public visibility.

Private `_produce_*` functions construct semantic products. Private
`_simulate_*` functions implement scientific submodels inside a product
producer. `_requirements.py` and `_random.py` remain unsupported private
implementation modules. There are no global `configs`, `fields`, `builders`,
or `validation` dumping grounds.

The producer module name is `_produce.py`, matching its `_produce_*` entry
point. Stage 6 behavior-neutrally renamed all four transitional Stage 4/5
waveform modules, callables, imports, and tests. `_produce.py` / `_produce_*`
is now the implemented convention; retired `_product.py` / `_product_*` names
must not return.

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

Operations do not silently move, cast, detach, host-materialize, or synchronize
an existing input. Ordinary same-stream PyTorch ordering applies;
cross-stream consumers establish their own dependency.

## Validation Philosophy

Public boundaries validate legitimate public inputs, including exact product
requests, config relationships, axes, shape, dtype, device, sampling
agreement, value domains at untrusted ingress, seed requirements, and
representable numerical bounds. Request failures occur before stochastic draws
or tensor writes.

Cheap intrinsic leaf checks belong in `_require()`. Full-device value scans
belong at explicit trust boundaries and builder postconditions rather than in
every semantic constructor.

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
movement, cast, or provenance inference.

TensorML receives deliberately selected product fields through a future
model-facing boundary; a growing collection is not an implicit positional
model ABI. Reconstruction owns its own products and execution arrangements.
Projects/dag may orchestrate future public operations but does not own
scientific algorithms or in-memory product semantics.

Deferred work includes persistence and cache contracts, TensorG4DS and
TensorML adapters, reconstruction, public execution workspaces, advanced
buffer reuse, broad device/backend compatibility claims, and release or
backward-compatibility policy.

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
optimization remains later work. Stage 7 public orchestration is undispatched
and has no accepted focused production work order.
