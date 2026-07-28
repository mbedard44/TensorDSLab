# Maintenance 15 Tensor-Native Config Punchcard Architecture

Status: **Architecture selected; exact TensorCore Stage 30 Design
consumer-confirmed; TensorCore implementation/publication and TensorDSLab
adoption pending; TensorDSLab Implementation undispatched**.

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

The complete public readout golden path becomes collaboration-specific:

```python
readout = DS20kVeto.create(
    sources=(photoelectrons,),
    config=config,
    rng=rng,
)
```

`simulate_readout(...)` is redundant under that contract and is selected for
removal without an alias.

This document is the detailed TensorDSLab architecture selection. It is not
production dispatch. The user subsequently superseded the initial local-root
direction and accepted TensorCore ownership of a deliberately narrow generic
`TensorConfig`. TensorCore Design froze exact Stage 30 Design commit
`79bb5ae00c3dbf6a49131001030ea56175e8461e`, tree
`44eaf757720c9dab41d5932814d70daba0e74721`, for provisional `0.22.0`.
TensorDSLab independently reviewed those exact bytes with zero findings. No
TensorCore implementation or publication and no TensorDSLab dependency
adoption has occurred.

TensorDSLab must not implement a temporary local root while that selected
sequence is active. It remains on exact published TensorCore `0.21.0` until
TensorCore implements, validates, reviews, closes, and publishes the exact
independently confirmed `TensorConfig` bytes.

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
        Axioelectrons
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
        DS20kVetoConfig
        SilexConfig

TensorCollection
    Readout
        DS20kVeto
        Silex
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

### Renewed `TensorConfig` ownership direction

TensorCore Design initially declined ownership after reviewing exact
TensorDSLab architecture commit
`cb24c9b6ed187a2b77ec4648c3e554d50a54027f`, tree
`6d2abe61cfbe4d9c2af5707f4ed463269fd82320`. The user subsequently reconsidered
that disposition and accepted TensorCore ownership of one deliberately narrow
unrealized tensor-domain placement root.

TensorCore Design froze the following exact Stage 30 / provisional `0.22.0`
Design surface:

```text
exact Design commit:
    79bb5ae00c3dbf6a49131001030ea56175e8461e
exact Design tree:
    44eaf757720c9dab41d5932814d70daba0e74721
exact parent / published 0.21.0:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
source of truth:
    docs/implementation/stage_30_tensor_config_structural_vocabulary.md
TensorDSLab consumer disposition:
    exact-byte CONFIRMED; zero findings
```

The conceptual public root is:

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
    axes: AxesT
    device: torch.device

    __hash__: ClassVar[None] = None

    @final
    def __post_init__(self) -> None:
        self._validate()
        self._require()

    @abstractmethod
    def _require(self) -> None:
        """Enforce the concrete semantic Config contract."""
```

The proposed TensorCore root owns only:

- one exact ordered tuple of semantic `TensorAxis` values;
- exact constructed-axis admission;
- exact-axis-type uniqueness within one completed product domain;
- preservation of the exact supplied axis objects and tuple;
- admission and retention of one exact `torch.device`; device availability is
  not checked;
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
- tensor payload, layout, or storage;
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

TensorCore explicitly permits downstream abstract intermediate Config roots
and final concrete leaves to add frozen, slotted stored fields. It does not
impose TensorField/TensorKernel's fieldless-leaf convention or runtime
subclass-shape policing on Config.

The one most-derived `_require()` implementation is called only after every
inherited and added dataclass field has been initialized and after universal
root validation. A downstream intermediate such as `QuantityConfig` must make
its own validation non-bypassable, for example by making its `_require()`
final and delegating concrete narrowing to another protected hook. TensorCore
does not invent a generic hook chain.

The same Stage 30 direction adds storage-free common vocabulary:

```python
TensorField.device
TensorField.dtype
TensorKernel.device
TensorKernel.dtype
```

Each property returns the corresponding state from the owned tensor. No
duplicate device or dtype field is stored. `TensorConfig` deliberately owns no
dtype because a complete Readout Config may govern Products with heterogeneous
output dtypes.

### Publication-bound adoption

TensorDSLab confirms the exact Design candidate but must bind only exact
published package bytes. The required sequence is:

1. **Complete:** TensorCore froze exact Stage 30 Design authority.
2. **Complete:** TensorDSLab reviewed and confirmed that exact documentation
   candidate.
3. **Pending:** TensorCore implements, validates, independently reviews,
   closes, and publishes the exact selected `0.22.0` package.
4. **Pending:** TensorDSLab synchronizes this architecture and its exact
   dependency pin to the published containing commit.
5. **Pending:** TensorDSLab freezes and dispatches its own bounded
   implementation work order.

Before step 4, TensorDSLab must neither import a provisional `TensorConfig` nor
implement a temporary package-local duplicate.

## TensorDSLab Ownership

TensorDSLab owns:

- `QuantityField`, `QuantityKernel`, and `QuantityConfig`;
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

## Working Dtype

The requested Product dtype and the hot-path working dtype are related but not
identical:

```text
config.dtype:
    dtype of the completed semantic Product

source.dtype:
    actual dtype of each realized source tensor

kernel.dtype:
    actual dtype of each prepared physical-kernel tensor

working dtype:
    promoted execution dtype after the Product's numerical floor
```

`TensorField` and `TensorKernel` expose storage-free `.dtype` properties
derived from their tensors. `QuantityConfig` stores the requested output
dtype. Generic `TensorConfig` stores no dtype.

There is no literal `max(torch.dtype, ...)` operation because Torch dtypes do
not form one simple total ordering across boolean, integer, real, and complex
categories. The conceptual rule is:

```python
candidate = promote_dtypes(
    config.dtype,
    *(source.dtype for source in sources),
    *(kernel.dtype for kernel in participating_kernels),
)

working_dtype = require_product_working_dtype(candidate)
```

The exact implementation may use `torch.promote_types(...)` plus explicit
category admission. Promotion policy is Product-owned TensorDSLab behavior,
not a TensorCore Config contract.

The rule is:

> A Product's requested output dtype establishes the user-controlled precision
> floor. A higher-precision source or participating kernel widens intermediate
> execution. The Product may impose a stronger numerical or algorithmic floor.

Preparation owns:

- the exact set of source and kernel dtypes participating in promotion;
- output-dtype admission;
- promotion and category relationships;
- the Product's numerical floor;
- law-specific fixed dtypes;
- kernel materialization in the selected representation;
- final-output representability preflight; and
- stable storage or structural recovery of the resulting source contract.

The exact per-Product work order must decide whether working dtype is retained
as meaningful private immutable Config state or is recovered cheaply from the
prepared source contract and prepared kernel dtypes. It must not be represented
by a readiness-only token.

Production may perform two explicit dtype effects:

1. convert each realized source tensor to the already selected working dtype
   when its dtype differs; and
2. cast or quantize the completed execution tensor exactly once to
   `config.dtype` at the semantic Product boundary.

Those are planned scientific tensor operations. Production still may not
discover promotion policy, reinterpret Config models, convert units, align
coordinates, move devices implicitly, or repair unprepared kernels.

The generic rule does not erase law-specific domains:

- Charge count frontiers and checked accumulation remain exact integer state;
- probability, intensity, and Distribution preparation may retain exact
  binary64 requirements;
- Gaussian and PSD execution retain their accepted law dtypes and numerical
  envelopes;
- convolution and floating reductions may impose a floating accumulation
  floor;
- Analog addition promotes all participating sources and bounds;
- DigitizedWaveform performs transfer calculations in a selected floating
  dtype even though its completed Product dtype is integer; and
- no output dtype silently narrows an intermediate that a stronger source,
  kernel, accumulator, or stochastic law requires.

This contract gives the user a simple precision control without pretending
that final output dtype is the only numerical dtype used by an algorithm.

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

## Ordered Product Sources

Every generated Product lifecycle receives one exact ordered tuple of
`QuantityField` sources:

```python
@classmethod
def prepare(
    cls,
    *,
    sources: tuple[QuantityField, ...],
    config: ProductConfig,
) -> ProductConfig:
    ...

@classmethod
def produce(
    cls,
    *,
    sources: tuple[QuantityField, ...],
    config: ProductConfig,
    rng: CounterRng,
) -> Self:
    ...
```

The tuple is a neutral transport contract. It does not claim that every
QuantityField is scientifically interchangeable, and the shared architecture
does not maintain a semantic source-type allowlist merely to prevent
unproductive caller choices.

Each concrete Product owns:

- whether zero, one, or multiple sources are admitted;
- whether source order has meaning;
- required axes, device, unit, dtype, storage, and value relationships;
- whether compatible sources are summed, convolved, transformed
  independently, paired by role, or used another way;
- exact failure before effects when the operation's mathematical domain is not
  satisfied; and
- exact relationship validation against the completed Product.

Static typing remains exact at the representation and result boundaries:
every input is a `QuantityField`, every Config is the precise Product Config,
and every lifecycle returns the precise Config or Product type. A concrete
Product may use overloads or a narrower public type alias when that improves
call-site inference, but the architecture does not require an ecosystem-wide
closed union of acceptable source leaf classes.

The caller-supplied tuple order is part of the complete input contract.
Identical ordered sources must replay exactly. No permutation-invariance
promise is inferred for floating accumulation or an order-sensitive transform.
A Product whose law is provably order-independent may freeze a stronger
contract separately.

Source addition syntax is not selected:

```python
axioelectrons + photoelectrons
```

Such an overload would need to define a new result Product type, unit, dtype,
axes, device, overflow behavior, provenance, and validation boundary before
the target Product sees the value. The explicit tuple already expresses the
desired input without requiring that additional abstraction. A later semantic
aggregate Product may introduce addition only if it has independent value.

### Charge source aggregation

`Axioelectrons` is a selected future final `QuantityField` leaf. Its exact
class name preserves avalanche origin independently from `Photoelectrons`.

Charge accepts one nonempty ordered tuple of QuantityFields satisfying its
count-source law. For the selected Silex path:

```python
sources = (
    axioelectrons,
    photoelectrons,
)

charge = Charge.create(
    sources=sources,
    config=charge_config,
    rng=rng,
)
```

Charge:

1. validates every source's complete axes, device, unit, dtype, storage, and
   nonnegative count domain;
2. promotes or retains the exact count representation required by checked
   accumulation;
3. sums sources in the frozen tuple order with overflow checks;
4. applies timing jitter and dark counts to the aggregate source frontier;
5. applies the configured branching generations;
6. applies charge smearing in the selected floating working dtype; and
7. returns one validated Charge Product in `config.dtype`.

The exact work order must freeze the common count-unit convention. If
Photoelectrons and Axioelectrons use one shared avalanche-count unit, their
semantic classes alone preserve origin. If they use genuinely distinct units,
Charge must own explicit conversion laws and may not merely add their
magnitudes.

All configured branching mechanisms in one generation continue to observe the
same aggregate frontier. Combining source origins before branching is an
explicit scientific and RNG-schedule decision that requires a deterministic
rebaseline; it is not inferred as byte continuity from a single-source path.

## Same-Type Preparation

Every generated Product owns its own exact `prepare(...)` method. There is no
common execution ABC or inherited implementation.

For direct use:

```python
prepared = PureWaveform.prepare(
    sources=(charge,),
    config=pure_config,
)

assert type(prepared) is PureWaveformConfig
```

For complete collaboration preflight, prepared upstream Configs describe
future source Products:

```python
prepared_charge = Charge.prepare(
    sources=(photoelectrons,),
    config=config.charge,
)

prepared_pure = PureWaveform.prepare(
    sources=(prepared_charge,),
    config=config.pure_waveform,
)
```

The public typing contract must explicitly admit the exact source forms
required by each Product:

```text
realized sources:
    exact tuple of QuantityField values

prospective sources during complete preflight:
    exact tuple of prepared QuantityConfig values carrying the future
    Products' output contracts
```

This may use precise overloads or a narrow structural typing protocol. It does
not introduce a stored specification object, generic graph node, or Runtime
replacement.

Preparation owns:

- exact Config and source-contract admission;
- every source/Config axes and device relationship;
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
- retain any source Product;
- retain source tensor values;
- inspect source values merely to specialize a reusable punchcard;
- return a different Config type;
- add a preparation flag or token;
- create Runtime;
- create `QuantityFieldSpec`;
- execute a Distribution;
- request random words; or
- create a Product.

The prepared Config is reusable with any ordered realized source tuple
satisfying the exact prepared source contracts and the Product's intrinsic
scientific domain.

## Structural Readiness

Prepared and unprepared Configs have the same concrete type. Readiness is
proved by existing state:

- exact Config axes and target device;
- exact ordered source-contract relationships;
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
device-placement policy, working-dtype policy, PSD integration, coordinate
permutation, or Config model dispatch may be rediscovered in `produce(...)`.
The already selected source-to-working and working-to-output dtype conversions
remain explicit production tensor effects.

Private derived Config fields are immutable representation state, not a
preparation token. They must have a stable scientific or execution meaning and
must be independently reconstructed when preparation returns a fresh Config.

## Production

Every generated Product owns an exact product-specific `produce(...)` method.
There is no inherited signature.

For PureWaveform:

```python
pure = PureWaveform.produce(
    sources=(charge,),
    config=prepared_pure_config,
)
```

Production:

1. validates the exact ordered realized sources against the prepared Config
   contracts;
2. validates structural readiness without preparing or repairing;
3. converts realized source tensors to the already selected working dtype when
   required;
4. performs the product-owned scientific tensor operation;
5. casts or quantizes once to `config.dtype` when required;
6. constructs the semantic Product with `config.axes`, `config.unit`, and the
   fresh result tensor;
7. validates intrinsic Product state and source/Config/result relationships;
8. returns only the validated Product.

Production performs no:

- Config preparation;
- Pint Quantity interpretation;
- unit conversion;
- coordinate permutation;
- implicit device movement;
- unplanned dtype promotion or conversion;
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
    ordered sources/Config/result axes, unit equations, working/output dtypes,
    device, shape, storage freshness, and product-specific postconditions
```

`produce(...)` performs both before returning.

A future `load(...)` operation can perform intrinsic validation without an
original source. It must not invent source relationships or provenance that
are absent from the artifact.

The exact method signatures may use one product-specific `validate(...)`
method with precise optional source-tuples/Config relationships or one public
intrinsic method plus a private transform relationship action. They must not
silently weaken either obligation.

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
source_counts = checked_sum_charge_sources(
    sources,
    config=config,
)

jittered = produce_timing_jitter(
    source=source_counts,
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

## Collaboration-Specific Readout Architecture

One universal Product topology is not selected. `Readout` is an abstract
semantic collection root; concrete collaboration Readouts own their own Config
shape, source interpretation, Product graph, lifecycle methods, and
relationship validation:

```text
TensorCollection
    Readout
        DS20kVeto
        Silex

TensorConfig
    ReadoutConfig
        DS20kVetoConfig
        SilexConfig
```

`Readout` owns only common collection structure:

- a nonempty exact-type TensorField mapping inherited from TensorCollection;
- QuantityField membership;
- common axes identity and device relationships;
- no common unit or dtype;
- no universal accepted Product census;
- no universal source census;
- no universal dependency graph; and
- a protected collaboration-specific narrowing hook.

`Readout` does not implement generic `create()`, `prepare()`, `produce()`, or
`validate()` methods whose broad signatures concrete subclasses must narrow.
Each concrete final Readout leaf defines those class methods directly with its
exact Config return type and workflow. This avoids unsafe override narrowing
and keeps static typing precise.

`ReadoutConfig` directly derives from `TensorConfig`, not `QuantityConfig`,
because one Readout may govern Products with heterogeneous units and output
dtypes. It is an abstract structural root and does not contain the union of
every collaboration's possible Product Config.

Each concrete Readout Config:

- owns only the Product Config fields available to that collaboration;
- uses Config presence when optional Products select a requested closure;
- requires each present child to retain the exact complete axes tuple by
  identity;
- requires each child device to equal the Readout Config device;
- validates its own Product dependency graph;
- contains no placeholder Config for an unavailable Product; and
- returns the exact same concrete Config type after preparation.

Equivalent `torch.device` values compare by equality rather than Python object
identity.

## `DS20kVeto`

The DS20k Veto path owns the existing complete waveform topology:

```text
ordered input QuantityFields
    -> Charge, when configured
    -> PureWaveform, when configured

no source values
    -> NoiseWaveform, when configured

PureWaveform + NoiseWaveform
    -> AnalogWaveform, when configured

AnalogWaveform
    -> DigitizedWaveform, when configured
```

Its conceptual Config is:

```python
@final
@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class DS20kVetoConfig(ReadoutConfig[DS20kVetoAxesT]):
    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None
```

It requires:

- at least one generated Product Config;
- PureWaveform implies Charge;
- AnalogWaveform implies PureWaveform and NoiseWaveform;
- DigitizedWaveform implies AnalogWaveform and therefore its prerequisites;
- exact child axes/device relationships; and
- no unrequested placeholder Product or Config.

One shared axes tuple is constructed once:

```python
axes = (
    example_axis,
    channel_axis,
    sample_axis,
)
```

The profile remains a convenient Config factory:

```python
config = ds20k_veto(
    example_axis=example_axis,
    channel_axis=channel_axis,
    sample_axis=sample_axis,
    device=torch.device("cpu"),
)
```

It returns `DS20kVetoConfig`, not a Readout instance.

## `Silex`

The selected initial Silex topology is deliberately smaller:

```text
Axioelectrons + Photoelectrons
    -> Charge

sources + Charge
    -> Silex
```

Its domain may use:

```python
axes = (
    example_axis,
    microcell_x_axis,
    microcell_y_axis,
    sample_axis,
)
```

The exact optionality of ExampleAxis remains a future Config-signature
decision. Microcell X, Microcell Y, and Sample roles are required by the
selected pixelated-SiPM operation geometry.

The conceptual Config initially contains only Charge:

```python
@final
@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class SilexConfig(ReadoutConfig[SilexAxesT]):
    charge: ChargeConfig
```

Silex does not expose PureWaveform, NoiseWaveform, AnalogWaveform, or
DigitizedWaveform Config fields merely because TensorDSLab supports those
Products for another collaboration. Their absence from Silex is structural,
not a disabled placeholder.

A future profile may construct the Config:

```python
config = silex(
    example_axis=example_axis,
    microcell_x_axis=microcell_x_axis,
    microcell_y_axis=microcell_y_axis,
    sample_axis=sample_axis,
    device=torch.device("cpu"),
)
```

This document selects the architectural surface but does not claim a calibrated
Silex detector profile.

## Collaboration-Specific Preparation And Production

The DS20k Veto one-shot path is:

```python
readout = DS20kVeto.create(
    sources=(photoelectrons,),
    config=config,
    rng=rng,
)
```

The Silex path is:

```python
readout = Silex.create(
    sources=(
        axioelectrons,
        photoelectrons,
    ),
    config=config,
    rng=rng,
)
```

Each concrete implementation composes its own exact same-type preparation:

```python
@classmethod
def create(
    cls,
    *,
    sources: tuple[QuantityField, ...],
    config: ConcreteReadoutConfig,
    rng: CounterRng,
) -> Self:
    prepared = cls.prepare(
        sources=sources,
        config=config,
    )

    return cls.produce(
        sources=sources,
        config=prepared,
        rng=rng,
    )
```

Advanced staged use remains:

```python
prepared = Silex.prepare(
    sources=(
        axioelectrons,
        photoelectrons,
    ),
    config=config,
)

readout = Silex.produce(
    sources=(
        axioelectrons,
        photoelectrons,
    ),
    config=prepared,
    rng=rng,
)
```

Complete preparation resolves the concrete graph before any Product is
generated. DS20k Veto prospectively prepares Charge from its input Config
contracts, PureWaveform from prepared Charge, source-free NoiseWaveform,
AnalogWaveform from prepared Pure and Noise, and DigitizedWaveform from
prepared Analog. Silex prospectively prepares one Charge Config from every
ordered input contract.

Exact static typing and closure admission must make every prerequisite evident
without unchecked casts or placeholder values.

The returned Config:

- has the exact same concrete collaboration Config type;
- retains the exact axes tuple object;
- retains an equal target device;
- contains fresh prepared child Configs;
- shares no mutable or tensor storage with input physical kernels;
- retains the exact collaboration Product topology; and
- is structurally ready for the same collaboration's `produce(...)`.

Every generated Product is validated by its own `produce(...)` before becoming
another Product's input.

Each concrete Readout retains all exact caller-supplied source fields plus
exactly the generated Products selected by its Config. Retaining immutable
source fields does not copy their tensor storage.

Concrete Readout validation owns:

- exact recognized membership for that collaboration;
- exact match to caller sources and configured outputs;
- one field of each exact semantic type under TensorCollection's invariant;
- shared axes identity and device;
- collaboration-specific Product dependencies; and
- whole-readout relationships not already owned by one Product.

It does not rerun every Product's scientific validator redundantly.

The collection does not impose one common unit or dtype. Each transform owns
its exact source/kernel/working/output dtype and unit relationships.

## Public API

The selected public concepts include:

```text
QuantityField
QuantityKernel
QuantityConfig
Readout
ReadoutConfig
DS20kVeto
DS20kVetoConfig
Silex
SilexConfig
Photoelectrons
Axioelectrons
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
silex
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
    config.py                # QuantityConfig
    field.py
    kernel.py
    units.py

  readout/
    collection.py            # Readout
    config.py                # ReadoutConfig

    ds20k_veto/
      config.py              # DS20kVetoConfig
      readout.py             # DS20kVeto
      profile.py             # ds20k_veto

    silex/
      config.py              # SilexConfig
      readout.py             # Silex
      profile.py             # silex

    axioelectrons/
      field.py
      runtime/
        validate.py

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

## Profile Boundaries

Profiles construct ordinary unprepared concrete collaboration Configs over
caller-supplied axes and target device:

```python
config = ds20k_veto(
    sample_axis=sample_axis,
    channel_axis=channel_axis,
    example_axis=example_axis,
    device=torch.device("cpu"),
)

silex_config = silex(
    sample_axis=sample_axis,
    microcell_x_axis=microcell_x_axis,
    microcell_y_axis=microcell_y_axis,
    example_axis=example_axis,
    device=torch.device("cpu"),
)
```

The exact signatures remain future work-order decisions, but:

- each profile returns its exact concrete Readout Config;
- `SampleAxis` is required by both selected collaborations;
- DS20k Veto admits its exact Example/Channel/Sample domain;
- Silex admits its exact Example/Microcell-X/Microcell-Y/Sample domain;
- optional conditioning axes may be supplied;
- a Config cannot condition on an axis absent from its complete declared axes;
- global rank-zero kernels remain valid when additional domain axes exist;
- Config axes use the exact caller-supplied immutable objects;
- the profile returns user-oriented tensor-native Configs;
- `DS20kVeto.prepare(...)` or `Silex.prepare(...)` owns alignment and device
  materialization; and
- the profile makes no calibration claim.

The profile may accept target output dtypes or choose explicit provisional
defaults. No global floating dtype argument is inferred later by production.

## Direct Product Use

Every Product Config remains independently usable outside `Readout`:

```python
prepared = PureWaveform.prepare(
    sources=(charge,),
    config=pure_config,
)

pure = PureWaveform.produce(
    sources=(charge,),
    config=prepared,
)
```

or:

```python
pure = PureWaveform.create(
    sources=(charge,),
    config=pure_config,
)
```

This direct path and the equivalent collaboration-specific Readout path must
use the exact same module-level preparation, production, and validation
actions. They must produce identical results and, for stochastic Products,
identical address and word traces from identical complete ordered inputs.

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
- adopt or import an unpublished TensorCore `TensorConfig`;
- change the exact TensorCore dependency pin before exact Stage 30 publication
  and package-owned adoption authority;
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

## Renewed TensorCore Consultation

TensorCore Design completed two read-only consultations. The initial decline
was superseded in principle after the user accepted narrow TensorCore
ownership. The current selected direction is:

```text
TensorCore ownership:
    accepted in principle for one narrow generic TensorConfig

TensorDSLab ownership:
    QuantityConfig, Product Configs, Readout Configs, preparation, dtype policy,
    units, science, and workflows

TensorCore publication:
    exact Stage 30 Design consumer-confirmed;
    implementation, closeout, and 0.22.0 publication pending

TensorDSLab dependency baseline:
    published TensorCore 0.21.0 remains exact until a later package-owned
    adoption commit
```

TensorDSLab confirmed the proposed boundary in principle and requested derived
`.device` and `.dtype` properties on TensorField and TensorKernel plus exact
subclass/hook evidence. Exact Stage 30 commit
`79bb5ae00c3dbf6a49131001030ea56175e8461e`, tree
`44eaf757720c9dab41d5932814d70daba0e74721`, includes those requirements and
received exact-byte TensorDSLab confirmation with zero findings. No additional
generic operation is required.

The consultation changes no TensorCore or TensorDSLab production byte and
makes no adoption, compatibility, CUDA, merge, push, or publication claim.
Only exact published TensorCore `0.22.0` package bytes may become TensorDSLab
dependency-adoption authority.

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
- ordered source-contract typing, arity, and interpretation;
- device behavior;
- relevant dtype set, promotion rule, and numerical floor;
- source-to-working and working-to-output conversion effects;
- product construction;
- intrinsic validation;
- transform postconditions;
- stochastic keys, domains, quanta, and word schedules;
- scientific parity/rebaseline classification; and
- exact errors and effects.

That inventory must explicitly disposition every current Runtime field before
deleting Runtime. The Readout inventory must separately freeze DS20k Veto and
Silex Config fields, source domains, Product graphs, retained fields, concrete
lifecycle signatures, and exact validation.

## Recommended Implementation Sequence

The architecture is one end-state but should not be implemented as an
unbounded rewrite.

### Phase 1: representation foundation

- exact published TensorCore Stage 30 / `0.22.0` adoption;
- TensorCore `TensorConfig`;
- derived TensorField/TensorKernel device and dtype vocabulary;
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
- abstract `ReadoutConfig` and `Readout`;
- concrete `DS20kVetoConfig` / `DS20kVeto`;
- concrete `SilexConfig` / `Silex`;
- `Axioelectrons`;
- ordered multi-source Product lifecycle;
- same-type collaboration Config preparation;
- collaboration-owned `create/prepare/produce/validate`;
- collaboration-specific Product closure;
- `simulate_readout` and `simulation.py` retirement;
- profile and demo migration; and
- exact public facade.

Each phase requires its own bounded work order or one later explicitly
accepted combined work order with a tractable exact allowlist and evidence
route. No phase may infer authority for the next.

## Evidence Required Before Final Adoption

The eventual package-owned implementation must prove at least:

### Representation

- Photoelectrons, Axioelectrons, Charge, PureWaveform, NoiseWaveform,
  AnalogWaveform, and DigitizedWaveform are exact final `QuantityField` leaves;
- Readout is abstract and DS20kVeto/Silex are exact concrete collection leaves;
- ReadoutConfig is abstract and both collaboration Configs are exact concrete
  leaves;
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

- exact same-type return for every Product Config and concrete Readout Config;
- fresh Config and kernel identity;
- no input mutation or tensor-storage aliasing;
- conditioning-coordinate reorder and dimension permutation;
- exact unit rescaling;
- exact law dtype;
- exact working-dtype promotion and Product numerical floors;
- exact target device;
- structural readiness without flags/tokens;
- scientific idempotence;
- unprepared Config rejection before allocation/words;
- prepared upstream Config tuples during complete downstream preflight; and
- no source Product retention.

### Production and validation

- direct `create(...)` equals explicit `prepare(...)` plus `produce(...)`;
- direct Product and whole-Readout paths are identical;
- Product construction uses exact Config axes, unit, dtype, and device;
- public `produce(...)` returns only validated Products;
- ordered sources/Config/result relationships are exact;
- only the prepared source-to-working and working-to-output dtype conversions
  occur in production;
- no Pint, implicit device transfer, coordinate permutation, dtype-policy
  discovery, or Config model interpretation occurs in production;
- stochastic address/word schedules match the accepted rebaseline;
- checked count conservation and accumulation remain;
- finite-window policy remains exact; and
- every high-risk scientific mutant is killed by committed independent proof.

### Readout

- each collaboration Config exposes only its accepted Product topology;
- Config presence exactly selects optional generated Products;
- each collaboration dependency closure is fail-closed;
- each concrete Readout retains every exact input source and configured output;
- exact-type lookup and one-per-type membership remain;
- concrete `prepare(...)` returns the exact concrete Config type;
- all child Configs are prepared before any Product generation;
- concrete `create(...)` equals staged preparation/production;
- Silex proves ordered Axioelectrons plus Photoelectrons Charge aggregation;
- DS20k Veto contains no Silex-only Product or Config placeholder;
- Silex contains no waveform Product or Config placeholder;
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

- importing or adopting TensorCore `TensorConfig` before exact Stage 30
  publication and package-owned dependency authority;
- implementing a temporary TensorDSLab-local TensorConfig while the selected
  Stage 30 sequence is active;
- retaining Runtime under another public or private name;
- adding a readiness token or prepared wrapper;
- making `produce(...)` repair or convert Config;
- treating an unplanned dtype cast as ordinary promotion;
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
exact Stage 30 Design candidate is consumer-confirmed. TensorDSLab Design may
next:

1. wait for exact TensorCore implementation, closeout, and publication;
2. synchronize the exact published dependency target;
3. complete each Product and collaboration Config/derived-state inventory;
4. freeze working-dtype, ordered-source, DS20k Veto, and Silex contracts;
5. divide the end state into bounded implementation work orders;
6. define exact dependency, scientific, typing, artifact, and mutation
   evidence for the first bounded candidate; and
7. verify persistent Implementation, Validation, and Review routes before any
   production dispatch.

No repository action in another package, dependency change, production edit,
test edit, merge, push, CUDA action, compatibility claim, or publication is
authorized by this architecture record.
