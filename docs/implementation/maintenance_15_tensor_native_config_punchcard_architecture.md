# Maintenance 15 Tensor-Native Config Punchcard Architecture

Status: **Architecture selected; TensorCore consultation complete;
TensorDSLab-local `TensorConfig` selected; Implementation undispatched**.

Stable key:
`TensorDSLab/maintenance-15-tensor-native-config-punchcard-architecture`

## Purpose

Replace TensorDSLab's parallel Config-to-Runtime representation with one
tensor-native Config model.

A Config is an immutable execution punchcard. It contains the complete axes,
target device, output representation, physical kernels, and constrained policy
needed to create one product. Preparation returns a fresh Config of the exact
same concrete type whose kernels have been aligned, rescaled, represented, and
materialized for execution:

```text
user-oriented Config
    -> Product.prepare(...)
fresh aligned Config of the exact same type
    -> Product.produce(...)
validated Product
```

This architecture deliberately retires product Runtime values rather than
renaming them or preserving them beside Config. It also moves public execution
ownership to the semantic result classes:

```text
Product.create(...)
Product.prepare(...)
Product.produce(...)
Product.validate(...)
```

The complete public readout golden path becomes:

```python
readout = Readout.create(
    source=photoelectrons,
    config=config,
    rng=rng,
)
```

`simulate_readout(...)` is redundant under that contract and is selected for
removal without an alias.

This document is the detailed TensorDSLab architecture selection. It is not
production dispatch. TensorCore Design independently reviewed the generic
`TensorConfig` ownership question against the exact initial architecture
commit and declined current TensorCore ownership. TensorDSLab therefore owns
`TensorConfig` locally as the structural root of its punchcard model. No
TensorCore change or new dependency publication is required.

## Governing Standards

This work follows:

- [CONTRIBUTING](../../CONTRIBUTING.md) for exact ownership, semantic tensor
  representation, unit admission, preparation, validation, public typing,
  artifacts, evidence, and no-silent-materialization requirements;
- [Overview](../overview.md) for the current package boundary and intended
  TensorG4DS-to-TensorDSLab-to-TensorML data flow;
- [Tensor Architecture](../architecture/tensors.md) for exact axis, field,
  collection, device, dtype, storage, and relationship contracts;
- [Readout Architecture](../architecture/readout.md) for the operative
  Maintenance 14 product graph that this future architecture will replace;
- [Parity](../parity.md) for every future scientific or stochastic rebaseline;
- [Maintenance 12](maintenance_12_tensorcore_0_21_kernel_geometry_quantity_refactor.md)
  for the currently implemented literal kernel, addressed Distribution, and
  physical-law baseline;
- [Maintenance 13](maintenance_13_runtime_hygiene_and_environment_reproducibility.md)
  for the currently implemented common kernel-alignment action; and
- [Maintenance 14](maintenance_14_test_suite_curation.md) for the exact closed
  documentation and test baseline.

The design is a clean pre-deployment replacement. It introduces no
backward-compatibility, deprecation, deployment, calibration, broad
compatibility, or release-readiness promise.

## Exact Design Baseline

This architecture document starts from exact locally closed Maintenance 14:

```text
TensorDSLab local main:
    856df702c124365c929bf993851a51fb8ff3c245
TensorDSLab tree:
    9e5ff69920699dc522980b164eaf1073116914c6
exact parent / immutable Maintenance 14 Candidate 1:
    60670e0bc6e54b87bd15177e36f46451abc64226
published origin/main at Design time:
    c8de1528d1ed57d3e86a9c37d1ad307127a23feb
origin/main tree:
    1d58e398428f35600e9bc582366c846c90d5f47c
package version:
    0.1.0
Python:
    >=3.14
Torch:
    >=2.13,<2.14
NumPy:
    2.5.1
Pint:
    0.25.3
TensorCore:
    exact published 0.21.0 commit
    78d0891bf6c0fefbcad4abe09980867c54202a9e
TensorCore tree:
    af5c4f6d693fa25cf767f3aaae31a47d86cf3a8d
```

Maintenance 14's complete fixed-commit source/archive evidence was
`305/302/3` in each dependency form. Its focused evidence was `79/78/1`;
Pyright reported zero diagnostics; the TensorCore dependency-negative fixture
retained exactly `82` intended diagnostics. The three TensorDSLab and two
TensorCore unavailable-CUDA skips remain explicit. No current integrated CUDA
or accelerator-support claim follows.

Those figures identify the starting evidence only. They are not frozen future
test-count targets.

## Selected Terminology

The selected names are:

```text
ReadoutCollection  -> Readout
Pulse              -> PulseResponse
simulate_readout() -> removed
```

There are no compatibility aliases, forwarding functions, deprecated names,
or duplicate public paths.

`Readout` is the completed typed collection and the owner of whole-readout
creation. `PulseResponse` is a physical response kernel. The name does not
imply normalization, dimensionlessness, or a particular voltage convention.

`PowerSpectralDensity` is the selected semantic `QuantityKernel` containing
the already prepared PSD representation consumed by noise execution. This
maintenance does not design the upstream operation that constructs it.

## Core Representation

The selected hierarchy is:

```text
TensorField
    QuantityField
        Photoelectrons
        Charge
        PureWaveform
        NoiseWaveform
        AnalogWaveform
        DigitizedWaveform

TensorKernel
    QuantityKernel
        DarkCountRate
        TimingJitter
        DirectCrosstalk
        DelayedCrosstalk
        Afterpulse
        SmearingWidth
        PulseResponse
        PowerSpectralDensity
        other accepted product-specific physical kernels

TensorConfig
    QuantityConfig
        ChargeConfig
        PureWaveformConfig
        NoiseWaveformConfig
        AnalogWaveformConfig
        DigitizedWaveformConfig

    ReadoutConfig

TensorCollection
    Readout
```

There is initially no:

- `QuantityFieldCollection`;
- `TensorKernelCollection`;
- `QuantityKernelCollection`;
- `ReadoutKernels`;
- `ChargeKernels`;
- `Runtime` ABC;
- product `*Runtime`;
- `ReadoutRuntime`;
- `QuantityFieldSpec`;
- prepared Config wrapper or subclass;
- preparation token or flag; or
- generic execution graph, registry, callback, or reflection layer.

`Readout` derives directly from `TensorCollection`. A quantity-specific
collection intermediate would have no selected reusable behavior and is not
created merely for hierarchy symmetry.

## Config Is The Punchcard

Config has two valid representations of the same scientific intent:

```text
unaligned representation:
    admitted public quantities and kernels;
    complete declared axes;
    target device, product unit, and product dtype;
    conditioning coordinates may require reordering;
    kernels may require unit rescaling or device materialization.

aligned representation:
    exact same concrete Config type;
    exact same scientific meaning;
    exact same declared axes and target representation;
    fresh aligned and independently owned kernels;
    exact law-required kernel dtypes;
    exact target device;
    ready for fail-closed production.
```

Preparation is analogous to returning a tensor in another valid
representation. It does not create a second semantic kind of object.

The rule is:

> Every user-configurable physical numerical parameter needed during
> execution is represented tensor-natively, normally as a `QuantityKernel`
> composed into Config.

This rule does not move the following into Config kernels:

- caller-supplied RNG engines;
- package-owned RNG keys, roles, domains, and quanta;
- fixed law tolerances;
- count and allocation ceilings;
- exact branching generation count;
- ADC bit depth;
- model selection;
- booleans and optional mechanism selection; or
- package-owned algorithm constants.

Those remain explicit execution dependencies, constrained nonphysical Config
values, or package implementation constants according to their meaning.

## TensorCore Ownership

### Existing accepted TensorCore ownership

TensorCore continues to own the generic, unit-independent substrate:

```text
TensorAxis
CountAxis
RegularAxis
LabelAxis
OffsetAxis
TensorField
TensorKernel
TensorCollection
require_kernel_dimensions(...)
RngKey
RngElements
RngAddress
CounterRng
Distribution
UniformDistribution
GaussianDistribution
PoissonDistribution
BinomialDistribution
MultinomialDistribution
generic tensor validation and relationship requirements
```

TensorDSLab must not fork, duplicate, wrap for compatibility, or reinterpret
those generic contracts.

### TensorCore disposition on `TensorConfig`

TensorCore Design reviewed exact initial TensorDSLab architecture commit
`cb24c9b6ed187a2b77ec4648c3e554d50a54027f`, tree
`6d2abe61cfbe4d9c2af5707f4ed463269fd82320`, and **declined current
TensorCore ownership**.

The package-owned reasons are:

- TensorCore's living requirements explicitly exclude a generic
  configuration root;
- axes plus a target device describe future execution intent rather than a
  realized generic tensor representation;
- TensorCore has no package operation consuming that state;
- neither TensorML nor TensorG4DS currently demonstrates the same exact state,
  invariants, lookup needs, and package-neutral operation; and
- a generic superclass permitting arbitrary downstream Config fields would
  enforce little beyond shared dataclass mechanics while blurring
  TensorCore's semantic representation boundary.

This is an ownership decision, not a contradiction in the punchcard
architecture. TensorCore does not reserve or export `TensorConfig`. No
TensorCore `0.22` stage or publication is required.

TensorCore may reconsider only after a genuine second package independently
demonstrates the same exact state and invariants plus a package-neutral
operation consuming them. Such a future review must first determine whether
the concept is more precisely a tensor domain or tensor target rather than a
general Config. TensorDSLab does not anticipate or depend on that possibility.

### TensorDSLab-local `TensorConfig`

TensorDSLab owns the structural representation root in
`tensor_dslab.common.config`:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    kw_only=True,
)
class TensorConfig[
    AxesT: tuple[TensorAxis[Any], ...],
](ABC):
    device: torch.device
    axes: AxesT

    __hash__ = None

    @abstractmethod
    def _require(self) -> None:
        """Enforce the concrete semantic Config contract."""
```

The exact TensorDSLab-local root owns only:

- admission and retention of one exact `torch.device`; device availability is
  not checked;
- one exact ordered tuple of semantic `TensorAxis` values;
- exact constructed-axis admission;
- exact-axis-type uniqueness within one completed product domain;
- preservation of the exact supplied axis objects and tuple;
- frozen, slotted, identity-equal, explicitly unhashable value semantics;
- zero-rank and zero-extent domain admission;
- derived `rank`, `shape`, and `element_count`;
- strict `axis_at(...)`, `dimension_of(...)`, and exact-type `axis(...)`
  lookup;
- Python-integer element-count multiplication, where a scalar domain has one
  element and any zero extent makes the count zero;
- final universal root validation followed by one protected semantic
  `_require()` hook; and
- no tensor allocation, transfer, host extraction, or device-availability
  effect.

`TensorConfig` owns no:

- Pint, physical units, or quantity conversion;
- output dtype;
- concrete Product, kernel, Config-leaf, or profile meaning;
- Config preparation or materialization;
- `create()`, `prepare()`, `produce()`, or `validate()`;
- Config iteration, reflection, registries, or factories;
- readout product selection or closure;
- scientific validation;
- detector axes, geometry, boundary policy, or RNG roles;
- Runtime or a Runtime replacement; or
- artifact or IO policy.

The local name is appropriate because it has one concrete meaning inside
TensorDSLab: immutable intent for a TensorDSLab Product punchcard. It is not a
compatibility wrapper, provisional TensorCore import, or claim of ecosystem
generality.

No TensorDSLab production work may import or probe for a TensorCore
`TensorConfig`.

## TensorDSLab Ownership

TensorDSLab owns:

- `TensorConfig`, `QuantityField`, `QuantityKernel`, and `QuantityConfig`;
- the private Pint registry and public quantity/unit construction boundary;
- instance unit admission and normalization;
- every concrete readout Product, Config, and physical kernel;
- product output dtype policy;
- same-type Config preparation;
- conditioning-coordinate correspondence and reordering;
- kernel rescaling and execution representation;
- device materialization of TensorDSLab Config state;
- Config request closure and product selection;
- product-owned scientific execution;
- detector/readout unit equations;
- sample, channel, example, and future detector-axis meaning;
- kernel-index-to-destination mapping;
- finite-window behavior;
- count ceilings and checked accumulation;
- readout RNG roles and address schemas;
- intrinsic and transform relationship validation;
- `Readout` membership and orchestration;
- profiles such as `ds20k_veto(...)`;
- future artifact, `load()`, and `write()` contracts; and
- every scientific rebaseline and parity classification.

TensorDSLab must not move those responsibilities into TensorCore merely
because the values are tensor-native.

## Quantity Roots

### `QuantityField`

`QuantityField` is a TensorDSLab-owned abstract representation root:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    init=False,
)
class QuantityField(TensorField, ABC):
    _unit: pint.Unit

    def __init__(
        self,
        *,
        tensor: torch.Tensor,
        axes: tuple[TensorAxis[Any], ...],
        unit: pint.Unit,
    ) -> None:
        ...

    @property
    def unit(self) -> pint.Unit:
        ...
```

It owns:

- one registry-normalized immutable Pint Unit;
- the existing TensorCore tensor snapshot and axes relationship;
- ordinary derived `device` and `dtype` access where needed for the common
  source contract; and
- no fixed dimensional family.

Concrete product leaves remain final and fieldless relative to
`QuantityField`. Their semantic meaning is the product role, not one fixed
physical dimension.

`QuantityField` does not expose a Pint Quantity over a CUDA tensor. Tensor
payloads remain Torch-native and unit metadata remains a small host-side
semantic value.

### `QuantityKernel`

`QuantityKernel` remains a TensorDSLab-owned abstract `TensorKernel` root, but
unit is instance state:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    init=False,
    repr=False,
    kw_only=True,
)
class QuantityKernel[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](
    TensorKernel[ConditioningAxesT, OperationAxesT],
    ABC,
):
    _unit: pint.Unit

    @property
    def unit(self) -> pint.Unit:
        ...
```

There is no class-level `canonical_unit`. One kernel instance owns:

- one exact tensor magnitude snapshot;
- literal conditioning and operation axes;
- one immutable Unit; and
- exact semantic leaf identity.

Preparation may return a fresh kernel of the same leaf type with reordered
conditioning coordinates, another compatible unit scale, another required
dtype, and another device. It never mutates or aliases the Config-owned input
kernel.

### `QuantityConfig`

`QuantityConfig` describes one future quantity-bearing Product:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    kw_only=True,
)
class QuantityConfig[
    AxesT: tuple[TensorAxis[Any], ...],
](TensorConfig[AxesT], ABC):
    unit: pint.Unit
    dtype: torch.dtype
```

The exact field name is `unit`, not `output_unit` or `canonical_unit`.

The four common source-contract facts are:

```text
axes
device
unit
dtype
```

A `QuantityField` supplies those facts for an already realized Product. A
prepared `QuantityConfig` supplies them for a Product that will be realized
later. No separate `QuantityFieldSpec` value is needed.

`QuantityConfig.dtype` is the requested result dtype. Internal law kernels may
use a different exact dtype where the scientific or numerical contract
requires it.

## Unit Boundary

TensorDSLab's private Pint registry remains the sole package unit authority.
The future work order must freeze:

- public `quantity(...)`, `quantities(...)`, and `unit(...)` constructors;
- exact recognition of foreign-registry Pint values, if admitted;
- exact normalization into the package registry;
- defensive ownership of numerical magnitudes;
- Unit equality and conversion behavior;
- rejection of inappropriate affine or logarithmic units;
- multiplicative conversion-factor representation;
- zero, finite, normal/subnormal, and dtype-envelope admission; and
- diagnostics for incompatible dimensional equations.

The three quantity-bearing roots use symmetric metadata:

```text
QuantityField.unit:
    unit of the realized Product tensor

QuantityKernel.unit:
    unit of the physical coefficient tensor

QuantityConfig.unit:
    requested unit of the Product created by the Config
```

Products do not impose one fixed dimensional family merely because of their
semantic class. Transforms own dimensional equations.

For example:

```text
photoelectron * mV / photoelectron -> mV
avalanche     * V / avalanche      -> V
coulomb       * V / coulomb        -> V
photon        * A / photon         -> A
```

An operation may still narrow its admitted unit algebra. Multiplication must
not silently apply affine or logarithmic conversion. Addition requires
compatible units. Thresholds must be compatible with the quantity they bound.
Probability inputs must satisfy the exact dimensionless law. Count
Distributions require an explicit law-specific semantic-unit audit.

## Product Configs

A Product Config owns:

- target axes;
- target device;
- requested Product unit;
- requested Product dtype;
- every user-configurable physical numerical parameter, normally as a
  `QuantityKernel`;
- constrained nonphysical policy values; and
- exact product-specific relationships.

It owns no:

- source Product;
- output Product;
- mutable cache;
- callback or Distribution factory;
- RNG engine;
- package-owned RNG role key;
- generic execution method; or
- provenance record.

### `ChargeConfig`

The selected conceptual shape is:

```python
@final
@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ChargeConfig(QuantityConfig[ReadoutAxesT]):
    correlated_avalanche_generations: NonnegativeInteger

    timing_jitter: TimingJitter | None = None
    direct_crosstalk: DirectCrosstalk | None = None
    delayed_crosstalk: DelayedCrosstalk | None = None
    afterpulse: Afterpulse | None = None
    dark_counts: DarkCountRate | None = None
    smearing_width: SmearingWidth | None = None
```

All six physical mechanisms remain immutable kernels. `None` disables one
mechanism. Branching generation depth remains a constrained scalar rather than
a kernel.

Package-owned RNG roles, count ceilings, complete-law tolerances, address
schemas, and Distribution algorithms remain outside Config.

### `PureWaveformConfig`

The selected shape is:

```python
@final
@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class PureWaveformConfig(QuantityConfig[ReadoutAxesT]):
    pulse_response: PulseResponse
```

Its dimensional equation is:

```text
source.unit * pulse_response.unit -> config.unit
```

`PulseResponse` may be global or conditioned on any Config-admitted axes. Its
operation geometry remains literal, normally including an
`OffsetAxis(relative_to=SampleAxis, ...)`.

### `NoiseWaveformConfig`

Noise remains one Product Config with exact model selection. Its physical
state is tensor-native.

The PSD branch contains:

```python
class PowerSpectralDensity(QuantityKernel):
    ...
```

This architecture assumes that `PowerSpectralDensity` has already been
prepared for the intended sampling geometry before it enters the readout
Config. The future implementation must validate its exact compatibility, but
this maintenance does not design:

- analytic or tabulated density integration;
- frequency-grid construction;
- extrapolation or tail policy;
- normalization helpers;
- calibration; or
- a user-facing PSD preparation API.

White-noise scale and other user-configurable physical scalars are represented
as rank-zero or conditioned QuantityKernels rather than permanently scalar
Pint values.

### `AnalogWaveformConfig`

Analog Config owns tensor-native saturation or transfer bounds when present.
Preparation validates the addition equation for Pure and Noise sources and
represents every bound in the requested output convention and dtype.

Analog production performs no Pint conversion. It consumes prepared Product
tensors and prepared Config kernels on one exact device.

### `DigitizedWaveformConfig`

Digitized Config owns:

- exact ADC bit depth as a constrained integer;
- tensor-native physical input bounds;
- tensor-native gain or exact selected transfer representation;
- exact output Unit;
- exact integer result dtype; and
- all local relationships needed to define the ADC mapping.

The detailed work order must state whether public construction stores
user-facing bounds/gain and prepares derived transfer coefficients, or whether
one custom constructor stores the selected execution representation directly.
Either choice remains one Config type and creates no Runtime.

## Scalar Physical Parameters

A global physical scalar is a genuine rank-zero `QuantityKernel`:

```text
conditioning_axes = ()
operation_axes = ()
tensor.shape = ()
```

The same semantic kernel may later admit conditioning axes for per-example,
per-channel, per-sample, per-microcell, or other accepted variation.

This applies where scientifically meaningful to:

- white-noise RMS;
- saturation bounds;
- digitizer input bounds;
- response widths;
- global rates; and
- other execution-time physical coefficients.

No anonymous singleton dimensions are introduced to imitate scalar
broadcasting.

## Same-Type Preparation

Every generated Product owns its own exact `prepare(...)` method. There is no
common execution ABC or inherited implementation.

For direct use:

```python
prepared = PureWaveform.prepare(
    source=charge,
    config=pure_config,
)

assert type(prepared) is PureWaveformConfig
```

For whole-readout preflight, a prepared upstream Config describes the future
source Product:

```python
prepared_charge = Charge.prepare(
    source=photoelectrons,
    config=config.charge,
)

prepared_pure = PureWaveform.prepare(
    source=prepared_charge,
    config=config.pure_waveform,
)
```

The public typing contract must explicitly admit the exact source forms
required by each Product:

```text
realized source:
    QuantityField

prospective source during complete preflight:
    prepared QuantityConfig with the source Product's output contract
```

This may use precise overloads or a narrow structural typing protocol. It does
not introduce a stored specification object, generic graph node, or Runtime
replacement.

Preparation owns:

- exact Config and source-contract admission;
- source/Config axes and device relationships;
- dimensional-equation validation;
- conditioning-coordinate correspondence and reordering;
- conditioning-dimension permutation;
- operation-geometry admission;
- kernel unit rescaling;
- law-required internal kernel dtype;
- requested Product dtype representation;
- tensor contiguity where required;
- materialization on `config.device`;
- shape/allocation preflight that is independent of source tensor values;
- stable derived execution state selected for storage in Config; and
- every error that must occur before scientific production begins.

Preparation does not:

- mutate or alias the input Config;
- mutate or alias an input kernel;
- retain a source Product;
- retain source tensor values;
- inspect source values merely to specialize a reusable punchcard;
- return a different Config type;
- add a preparation flag or token;
- create Runtime;
- create `QuantityFieldSpec`;
- execute a Distribution;
- request random words; or
- create a Product.

The prepared Config is reusable with any realized source satisfying the exact
prepared source contract and the Product's intrinsic scientific domain.

## Structural Readiness

Prepared and unprepared Configs have the same concrete type. Readiness is
proved by existing state:

- exact Config axes and target device;
- exact source-contract relationship;
- exact result unit and dtype;
- kernel tensors on the Config device;
- exact law-required kernel dtypes;
- exact conditioning coordinate order;
- exact operation geometry;
- exact unit representation;
- execution-compatible shapes;
- required contiguous tensor storage; and
- completed immutable derived state.

Do not add:

- `prepared: bool`;
- `is_prepared`;
- a trusted token;
- a private constructor token used as readiness proof;
- `PreparedConfig`;
- aligned subclasses;
- a hidden mutable cache; or
- a registry of prepared objects.

`produce(...)` checks structural readiness and fails before allocation, random
words, or scientific tensor effects when the Config is incompatible. It does
not repair an unprepared Config.

Preparing an already prepared Config is scientifically idempotent. It must not
apply unit scaling, polarity, coordinate permutation, or another scientific
mapping twice. Whether it returns the same immutable object or a fresh exact
representation is an implementation decision to freeze before dispatch;
observable scientific and execution state must remain exact.

## Derived Execution Facts

Removing Runtime does not authorize recomputation of raw user intent on the
hot path.

Every fact previously stored in a Runtime record must receive one exact
disposition:

```text
prepared semantic QuantityKernel in Config;
private immutable derived tensor/scalar state in the same Config;
constrained nonphysical public Config value;
cheap structural derivation from exact axes/kernel geometry;
fixed package implementation constant; or
the actual product-owned scientific tensor operation.
```

This inventory includes at least:

- sample count, sample period, and sample dimension;
- conditioning dimension mappings;
- operation target dimensions and offset products;
- represented noise scale and numerical envelope;
- prepared PSD power state;
- dark-count mean conversion;
- ADC maximum code, gain, slope, intercept, and thresholds;
- branching destination geometry;
- allocation ceilings;
- output unit and dtype; and
- role/address selection.

No fact requiring Pint interpretation, host extraction of tensor payload,
device transfer, dtype conversion, PSD integration, coordinate permutation, or
Config model dispatch may be rediscovered in `produce(...)`.

Private derived Config fields are immutable representation state, not a
preparation token. They must have a stable scientific or execution meaning and
must be independently reconstructed when preparation returns a fresh Config.

## Production

Every generated Product owns an exact product-specific `produce(...)` method.
There is no inherited signature.

For PureWaveform:

```python
pure = PureWaveform.produce(
    source=charge,
    config=prepared_pure_config,
)
```

Production:

1. validates the exact realized source against the prepared Config contract;
2. validates structural readiness without converting or repairing;
3. performs the product-owned scientific tensor operation;
4. constructs the semantic Product with `config.axes`, `config.unit`, and the
   fresh result tensor;
5. validates intrinsic Product state and source/Config/result relationships;
6. returns only the validated Product.

Production performs no:

- Config preparation;
- Pint Quantity interpretation;
- unit conversion;
- coordinate permutation;
- implicit device movement;
- dtype conversion;
- PSD discretization;
- mutable caching;
- source-type-dependent callback dispatch; or
- user-selected Distribution construction.

Public `produce(...)` never returns an unvalidated Product. A caller is not
required to remember a separate validation call for correctness.

## Validation

Each concrete Product owns a directly callable, read-only, idempotent
validation boundary.

Validation distinguishes:

```text
intrinsic Product validity:
    tensor structure, axes, unit, dtype, device, values, and semantic domain

transform relationship validity:
    source/Config/result axes, unit equation, dtype, device, shape, storage
    freshness, and product-specific postconditions
```

`produce(...)` performs both before returning.

A future `load(...)` operation can perform intrinsic validation without an
original source. It must not invent source relationships or provenance that
are absent from the artifact.

The exact method signatures may use one product-specific `validate(...)`
method with precise optional source/Config pairs or one public intrinsic method
plus a private transform relationship action. They must not silently weaken
either obligation.

## Product-Owned Scientific Execution

Physical kernels are immutable transform parameters. They are not Products,
effects, callbacks, Distribution factories, or mutable execution objects.

The selected mapping is:

```text
TimingJitter         -> MultinomialDistribution
DarkCountRate        -> PoissonDistribution
DirectCrosstalk      -> PoissonDistribution
DelayedCrosstalk     -> PoissonDistribution
Afterpulse           -> PoissonDistribution
SmearingWidth        -> GaussianDistribution
PulseResponse        -> deterministic convolution
PowerSpectralDensity -> addressed Gaussian spectral synthesis
```

TensorCore owns generic Distribution and random execution. TensorDSLab owns
which law implements each physical concept, the physical coefficients,
scientific keys, address domains, boundary mapping, and result validation.

No arbitrary Distribution class, callback, sampler registry, or factory enters
public Config state.

## Charge Production

The accepted high-level algorithm remains:

```python
jittered = produce_timing_jitter(
    source=source,
    config=config,
    rng=rng,
)

dark_counts = produce_dark_counts(
    config=config,
    rng=rng,
)

frontier = checked_add(
    jittered,
    dark_counts,
)

total = frontier

for _ in range(config.correlated_avalanche_generations.value):
    direct = produce_direct_crosstalk(
        frontier=frontier,
        config=config,
        rng=rng,
    )

    delayed = produce_delayed_crosstalk(
        frontier=frontier,
        config=config,
        rng=rng,
    )

    afterpulse = produce_afterpulse(
        frontier=frontier,
        config=config,
        rng=rng,
    )

    children = checked_add(
        direct,
        delayed,
        afterpulse,
    )

    total = checked_add(
        total,
        children,
    )

    frontier = children

charge_tensor = produce_smeared_charge(
    counts=total,
    config=config,
    rng=rng,
)
```

Disabled optional mechanisms are skipped explicitly. A helper that requires a
kernel receives an actual kernel rather than `None`.

All branching mechanisms in one generation observe the same frontier. Only
their pooled children advance the next generation. Direct and delayed
crosstalk retain collapsed destination-rate Poisson execution. Afterpulse
retains the accepted simplified Poisson law without recovery weighting.

The exact unit equations for count-valued stochastic laws require a dedicated
per-product audit before implementation dispatch. No generic Pint manipulation
may change TensorCore Distribution parameters or silently reinterpret semantic
count units.

## `PowerSpectralDensity`

`PowerSpectralDensity` is a TensorDSLab `QuantityKernel`.

For this architecture stage, it is treated as already scientifically prepared
for the exact sampling geometry declared by the Config. The accepted contract
requires only that future preparation and production can verify:

- exact relationship to the Config's `SampleAxis`;
- exact tensor geometry;
- exact conditioning axes;
- exact device and law-required dtype;
- exact compatible unit; and
- exact finite, nonnegative scientific domain.

This stage does not freeze how a user derives the kernel from an analytic,
tabulated, historical, or measured PSD. It does not choose density integration,
FFT normalization, interpolation, extrapolation, tail, or calibration policy.
Those are explicit future design questions and are not readout hot-path work.

## `ReadoutConfig`

`ReadoutConfig` directly derives from `TensorConfig`, not `QuantityConfig`,
because a complete Readout contains multiple product units and dtypes.

The selected request shape is:

```python
@final
@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ReadoutConfig(TensorConfig[ReadoutAxesT]):
    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None
```

Config presence selects the requested generated Products. There is no separate
`products=` request.

`ReadoutConfig` requires:

- at least one generated Product Config;
- every present child has `child.axes is config.axes`;
- every present child has `child.device == config.device`;
- PureWaveform implies Charge;
- AnalogWaveform implies PureWaveform and NoiseWaveform;
- DigitizedWaveform implies AnalogWaveform and therefore its prerequisites;
- every child Config relationship is locally valid; and
- no unrequested placeholder Config is created.

One shared immutable axes tuple is constructed once and reused by identity:

```python
axes = (
    example_axis,
    channel_axis,
    sample_axis,
)
```

Equivalent `torch.device` values compare by equality rather than Python object
identity.

## Whole-Readout Preparation

`Readout.prepare(...)` returns an aligned `ReadoutConfig`:

```python
prepared = Readout.prepare(
    source=photoelectrons,
    config=config,
)

assert type(prepared) is ReadoutConfig
```

It recursively prepares the exact requested closure before any Product is
generated:

```python
prepared_charge = (
    None
    if config.charge is None
    else Charge.prepare(
        source=photoelectrons,
        config=config.charge,
    )
)

prepared_pure = (
    None
    if config.pure_waveform is None
    else PureWaveform.prepare(
        source=prepared_charge,
        config=config.pure_waveform,
    )
)

prepared_noise = (
    None
    if config.noise_waveform is None
    else NoiseWaveform.prepare(
        source=photoelectrons,
        config=config.noise_waveform,
    )
)

prepared_analog = (
    None
    if config.analog_waveform is None
    else AnalogWaveform.prepare(
        pure=prepared_pure,
        noise=prepared_noise,
        config=config.analog_waveform,
    )
)

prepared_digitized = (
    None
    if config.digitized_waveform is None
    else DigitizedWaveform.prepare(
        source=prepared_analog,
        config=config.digitized_waveform,
    )
)
```

Exact static typing and closure admission must make each non-`None`
prerequisite evident without unchecked casts or placeholder values.

The returned Config:

- retains the exact `config.axes` object;
- retains an equal target device;
- contains fresh prepared child Configs;
- shares no mutable or tensor storage with the input Config's physical kernels;
- has the exact same requested Product presence; and
- is ready for `Readout.produce(...)`.

## Whole-Readout Production

The public one-shot path is:

```python
readout = Readout.create(
    source=photoelectrons,
    config=config,
    rng=rng,
)
```

Its exact conceptual implementation is:

```python
@classmethod
def create(
    cls,
    *,
    source: Photoelectrons,
    config: ReadoutConfig,
    rng: CounterRng,
) -> Self:
    prepared = cls.prepare(
        source=source,
        config=config,
    )

    return cls.produce(
        source=source,
        config=prepared,
        rng=rng,
    )
```

Advanced staged use is:

```python
prepared = Readout.prepare(
    source=photoelectrons,
    config=config,
)

readout = Readout.produce(
    source=photoelectrons,
    config=prepared,
    rng=rng,
)
```

`Readout.produce(...)` executes present Products in topological order:

```text
Photoelectrons source
    -> Charge, when configured
    -> PureWaveform, when configured

Photoelectrons source
    -> NoiseWaveform, when configured

PureWaveform + NoiseWaveform
    -> AnalogWaveform, when configured

AnalogWaveform
    -> DigitizedWaveform, when configured

source plus exactly configured Products
    -> Readout
```

Every generated Product is validated by its own `produce(...)` before becoming
an input to another Product.

`Readout` contains the source `Photoelectrons` plus exactly the configured
generated Products. Retaining the immutable source field does not copy its
tensor storage.

`Readout.validate(...)` owns only:

- exact recognized membership;
- exact match to the configured Product set;
- one field of each exact semantic type;
- shared axes identity/relationship;
- shared device;
- child Product presence dependencies; and
- whole-readout relationships not already owned by one Product.

It does not rerun each Product's scientific validator redundantly.

The collection does not impose one common floating dtype merely by ownership.
Each transform preparation owns the exact input/output dtype relationships
required by its law.

## Public API

The selected public concepts include:

```text
QuantityField
QuantityKernel
QuantityConfig
Readout
ReadoutConfig
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
ChargeConfig
PureWaveformConfig
NoiseWaveformConfig
AnalogWaveformConfig
DigitizedWaveformConfig
DarkCountRate
TimingJitter
DirectCrosstalk
DelayedCrosstalk
Afterpulse
SmearingWidth
PulseResponse
PowerSpectralDensity
quantity
quantities
unit
ds20k_veto
```

The exact export census remains future work-order evidence. The following
supported names are selected for removal:

```text
ReadoutCollection
Pulse
simulate_readout
```

No alias, forwarding function, alternate spelling, or compatibility module is
added.

Product lifecycle methods are supported on the exact Product class. Config
preparation types are ordinary public Config types, so no public Runtime path
is required.

## Module Organization

The product-centered tree remains:

```text
tensor_dslab/
  common/
    axis.py
    config.py
    field.py
    kernel.py
    units.py

  readout/
    collection.py            # Readout
    config.py                # ReadoutConfig
    profiles.py

    charge/
      config.py
      field.py
      kernel.py
      runtime/
        prepare.py
        produce.py
        validate.py
        branching.py
        counts.py

    pure_waveform/
      config.py
      field.py
      kernel.py
      runtime/
        prepare.py
        produce.py
        validate.py

    noise_waveform/
      config.py
      field.py
      kernel.py
      runtime/
        prepare.py
        produce.py
        validate.py

    analog_waveform/
      config.py
      field.py
      kernel.py
      runtime/
        prepare.py
        produce.py
        validate.py

    digitized_waveform/
      config.py
      field.py
      kernel.py
      runtime/
        prepare.py
        produce.py
        validate.py

    runtime/
      addresses.py
      kernel.py
      keys.py
      requirements.py
```

The `runtime/` name remains an implementation-organization boundary for
preparation, production, validation, addressed stochastic execution, and
shared narrow actions. It contains no Runtime value.

Delete or retire:

```text
tensor_dslab/readout/simulation.py
tensor_dslab/readout/runtime/prepare.py        # ReadoutRuntime owner
tensor_dslab/readout/runtime/sampling.py       # SamplingRuntime owner
every product or mechanism *Runtime class
every Runtime-only runtime.py module, if introduced
```

Sampling facts are derived from exact Config axes or stored as immutable
same-Config representation state according to the derived-fact inventory.

Module-level actions retain simple names:

```text
prepare_charge(...)
produce_charge(...)
validate_charge(...)
prepare_pure_waveform(...)
produce_pure_waveform(...)
validate_pure_waveform(...)
```

A module-level producer may return a tensor. The public Product class owns
semantic construction and validated return.

Runtime modules must not import the concrete Product class when that creates a
cycle:

```text
field.py -> runtime/produce.py -> field.py
```

Execution helpers operate on exact generic inputs and return execution tensors
or validated facts. Public Product classes own final semantic construction.

## Profile Boundary

Profiles construct ordinary unprepared Configs over caller-supplied axes and
target device:

```python
config = ds20k_veto(
    sample_axis=sample_axis,
    channel_axis=channel_axis,
    example_axis=example_axis,
    device=torch.device("cpu"),
)
```

The exact signature remains a future work-order decision, but:

- `SampleAxis` is required;
- optional conditioning axes may be supplied;
- a Config cannot condition on an axis absent from its complete declared axes;
- global rank-zero kernels remain valid when Example or Channel axes exist;
- Config axes use the exact caller-supplied immutable objects;
- the profile returns user-oriented tensor-native Configs;
- `Readout.prepare(...)` owns alignment and device materialization; and
- the profile makes no calibration claim.

The profile may accept target output dtypes or choose explicit provisional
defaults. No global floating dtype argument is inferred later by production.

## Direct Product Use

Every Product Config remains independently usable outside `Readout`:

```python
prepared = PureWaveform.prepare(
    source=charge,
    config=pure_config,
)

pure = PureWaveform.produce(
    source=charge,
    config=prepared,
)
```

or:

```python
pure = PureWaveform.create(
    source=charge,
    config=pure_config,
)
```

This direct path and the equivalent whole-Readout path must use the exact same
module-level preparation, production, and validation actions. They must
produce identical results and, for stochastic Products, identical address and
word traces from identical complete inputs.

## IO Boundary

Future IO may use:

```python
product = Product.load(...)
product.write(...)
```

No artifact format, manifest, cache, provenance, serialization, `load()`, or
`write()` implementation is selected here.

A future `load()` reconstructs the semantic Product and performs intrinsic
validation. It does not infer absent source/transform relationships.

IO remains a separate focused stage after the in-memory representation and
factory lifecycle are stable.

## Explicit Retirements

The future implementation retires without aliases:

- `ReadoutRuntime`;
- every product `*Runtime`;
- every mechanism Runtime;
- `SamplingRuntime`;
- `QuantityFieldSpec`;
- Runtime preparation records;
- prepared Config wrappers;
- prepared Config subclasses;
- preparation flags and tokens;
- class-level `canonical_unit`;
- `ReadoutCollection`;
- `Pulse`;
- `simulate_readout`;
- `tensor_dslab/readout/simulation.py`;
- generic Config execution/reflection frameworks;
- user-supplied Distribution factories; and
- Pint Quantity magnitudes in prepared execution state.

The future implementation must prove absence rather than retaining dormant
compatibility paths.

## Frozen Non-Goals

This architecture does not:

- implement or dispatch production;
- edit TensorCore;
- add, request, probe for, or infer a TensorCore `TensorConfig`;
- change the exact TensorCore dependency pin;
- design PSD discretization;
- add a graph planner, transform registry, effect framework, or callbacks;
- add artifact IO, cache shape, provenance, `load()`, or `write()`;
- add native G4DS ingestion or TensorG4DS adaptation;
- add TensorML integration;
- add arbitrary user-selected Distributions;
- add calibration or detector-conformance claims;
- add compatibility aliases or deprecation windows;
- claim CUDA, accelerator performance, deployment, compatibility, release, or
  production readiness; or
- change the paired exact-1.0 integrated-CUDA schedule.

## Completed TensorCore Consultation

TensorCore Design completed the required exact read-only consultation against
the initial architecture bytes. Its disposition is frozen here:

```text
TensorCore ownership:
    declined for the current package

TensorDSLab ownership:
    accepted package-local structural fallback

TensorCore publication:
    none required

TensorDSLab dependency baseline:
    published TensorCore 0.21.0 remains exact
```

The disposition changes no TensorCore or TensorDSLab production byte and makes
no adoption, compatibility, CUDA, merge, push, or publication claim.

The future implementation work order must freeze the local root's exact
constructor, validation order, diagnostics, properties, lookup signatures,
typing, module export, and subclass contract. It must not reopen TensorCore
ownership merely to avoid local boilerplate.

## Required Per-Product Inventory

Before production dispatch, TensorDSLab Design must freeze an exact inventory
for every Config:

- public fields and constructor signatures;
- physical kernel leaf names and precise modules;
- optional mechanism semantics;
- axes and conditioning admission;
- output unit and dtype;
- internal law-required kernel dtypes;
- unit equations;
- affine/logarithmic restrictions;
- Config construction versus preparation validation;
- same-type reconstruction mechanics;
- derived execution state;
- structural readiness checks;
- source-contract typing;
- device behavior;
- product construction;
- intrinsic validation;
- transform postconditions;
- stochastic keys, domains, quanta, and word schedules;
- scientific parity/rebaseline classification; and
- exact errors and effects.

That inventory must explicitly disposition every current Runtime field before
deleting Runtime.

## Recommended Implementation Sequence

The architecture is one end-state but should not be implemented as an
unbounded rewrite.

### Phase 1: representation foundation

- TensorDSLab-local `TensorConfig`;
- `QuantityField`;
- instance-unit `QuantityKernel`;
- `QuantityConfig`;
- public `unit(...)`;
- exact unit normalization;
- same-type Config reconstruction;
- structural readiness helpers; and
- typing/API evidence.

### Phase 2: deterministic pilot

- `Pulse` to `PulseResponse`;
- explicit PureWaveform unit/dtype Config;
- `PureWaveform.create/prepare/produce/validate`;
- direct and prospective source contracts;
- exact convolution;
- no Runtime; and
- direct/staged equivalence.

### Phase 3: remaining Products

- Charge Config and stochastic laws;
- Noise Config and assumed prepared `PowerSpectralDensity`;
- Analog Config and unit-compatible addition;
- Digitized Config and ADC mapping;
- complete Runtime-fact disposition; and
- exact science/RNG evidence.

### Phase 4: whole Readout

- `ReadoutCollection` to `Readout`;
- same-type `Readout.prepare`;
- `Readout.create/produce/validate`;
- Config-selected product closure;
- `simulate_readout` and `simulation.py` retirement;
- profile and demo migration; and
- exact public facade.

Each phase requires its own bounded work order or one later explicitly
accepted combined work order with a tractable exact allowlist and evidence
route. No phase may infer authority for the next.

## Evidence Required Before Final Adoption

The eventual package-owned implementation must prove at least:

### Representation

- all six Products are exact final `QuantityField` leaves;
- every Product owns one immutable registry-normalized Unit;
- all physical kernels own instance units;
- `canonical_unit` is absent;
- every Product Config owns exact axes, device, unit, and dtype where
  applicable;
- Configs and tensor-bearing roots are frozen, slotted, identity-equal, and
  explicitly unhashable;
- tensor snapshots are fresh, contiguous where required, and logically
  read-only; and
- public signatures and exports match the exact target.

### Preparation

- exact same-type return for every Product Config and ReadoutConfig;
- fresh Config and kernel identity;
- no input mutation or tensor-storage aliasing;
- conditioning-coordinate reorder and dimension permutation;
- exact unit rescaling;
- exact law dtype;
- exact target device;
- structural readiness without flags/tokens;
- scientific idempotence;
- unprepared Config rejection before allocation/words;
- prepared upstream Config use during complete downstream preflight; and
- no source Product retention.

### Production and validation

- direct `create(...)` equals explicit `prepare(...)` plus `produce(...)`;
- direct Product and whole-Readout paths are identical;
- Product construction uses exact Config axes, unit, dtype, and device;
- public `produce(...)` returns only validated Products;
- source/Config/result relationships are exact;
- no Pint, transfer, dtype conversion, coordinate permutation, or Config model
  interpretation occurs in production;
- stochastic address/word schedules match the accepted rebaseline;
- checked count conservation and accumulation remain;
- finite-window policy remains exact; and
- every high-risk scientific mutant is killed by committed independent proof.

### Readout

- Config presence exactly selects generated Products;
- dependency closure is fail-closed;
- `Readout` contains the source and exactly configured outputs;
- exact-type lookup and one-per-type membership remain;
- `Readout.prepare(...)` returns exact `ReadoutConfig`;
- all child Configs are prepared before any Product generation;
- `Readout.create(...)` equals staged preparation/production;
- `simulate_readout`, `ReadoutCollection`, and their aliases are absent; and
- profiles, examples, demos, and notebooks use the new golden path.

### Static and artifact evidence

- strict Pyright is clean;
- dependency-negative fixtures match the accepted TensorCore version;
- package topology and import direction are acyclic;
- Runtime value definitions and retired paths are absent;
- source and extracted-archive suites pass;
- wheel/sdist payloads match source;
- isolated-wheel imports resolve only from the artifact;
- direct and whole-readout examples execute;
- Markdown fences and links pass;
- privacy and generated-artifact scans pass; and
- the repository is clean.

CUDA remains deferred to the exact mutually adopted TensorCore/TensorDSLab
1.0.0 release-candidate pairing. Every interim disposition must explicitly
state that no current integrated CUDA or accelerator-support claim is made.

## Stop Conditions

Stop and return to TensorDSLab Design if implementation would require:

- importing, probing for, or adding a TensorCore `TensorConfig`;
- retaining Runtime under another public or private name;
- adding a readiness token or prepared wrapper;
- making `produce(...)` repair or convert Config;
- inspecting source values during preparation merely to specialize a reusable
  Config;
- moving TensorDSLab unit or scientific policy into TensorCore;
- treating `PowerSpectralDensity` preparation as hot-path work;
- restoring `simulate_readout`, `ReadoutCollection`, or `Pulse` through an
  alias;
- widening product science without a parity decision;
- adding a graph, registry, callback, or arbitrary Distribution factory;
- changing RNG identities or completed stochastic values without an exact
  rebaseline;
- performing implicit device or host materialization;
- adding IO or compatibility work; or
- claiming CUDA, deployment, compatibility, release, or production readiness.

## Authority And Next Action

This document records TensorDSLab Design's selected punchcard architecture.
Implementation remains undispatched.

The TensorCore ownership consultation is complete and the
TensorDSLab-local root is selected. TensorDSLab Design may next:

1. complete the per-product Config and derived-state inventory;
2. freeze exact local `TensorConfig` diagnostics, typing, and exports;
3. divide the end state into bounded implementation work orders;
4. define exact dependency, scientific, typing, artifact, and mutation
   evidence for the first bounded candidate; and
5. verify persistent Implementation, Validation, and Review routes before any
   production dispatch.

No repository action in another package, dependency change, production edit,
test edit, merge, push, CUDA action, compatibility claim, or publication is
authorized by this architecture record.
