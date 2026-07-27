# Proposed Kernel Geometry And Quantity Architecture

Status: **Architecture selected / TensorCore 0.21.0 Stage 29 published /
Maintenance 12 fixed-commit candidate lifecycle active**

Stable planning key:
`TensorDSLab/kernel-geometry-and-quantity-architecture`

## Purpose

Define the future TensorCore and TensorDSLab ownership boundary for literal
N-dimensional physical kernels, Pint-aware tensor coefficients, profile-bound
geometry, Runtime alignment, and a staged Charge/PureWaveform migration.

This record is deliberately detailed enough to review class shape and
scientific meaning before production work begins. It is not an implementation
work order, dependency adoption, public compatibility promise, or permission
to edit TensorCore. Exact production allowlists, dependency commits, API
censuses, diagnostics, and candidate routes must be frozen by later
package-owned work orders.

The selected direction follows:

- `CONTRIBUTING.md` for TensorCore ownership, semantic axes, public typing,
  Config/Runtime separation, validation, scientific evidence, and scope;
- `docs/architecture/tensors.md` for TensorCore-backed axis, tensor, placement,
  relationship, and synchronization boundaries;
- `docs/architecture/rebuild.md` and `docs/architecture/readout.md` for the
  current product graph and Config-to-Runtime execution boundary;
- `docs/parity.md` for intentional scientific divergence and RNG/output
  rebaselining; and
- the user-accepted kernel architecture handoff that supersedes the earlier
  behavioral-effect hierarchy proposal.

## Planning Baseline And Selected Publication

The current TensorDSLab baseline remains exact:

```text
TensorDSLab main/origin-main:
    8517f09d6ecdf72434626bce0524f9f032998fd8
TensorDSLab tree:
    3506fbf92d79473a3431e390ba3518ad5f166414
package version:
    0.1.0
TensorCore dependency:
    0.19.0 at ed17f4b637258f0a7f4544f235648b747f17fa44
```

The selected TensorCore publication is:

```text
TensorCore main/origin-main:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
TensorCore tree:
    af5c4f6d693fa25cf767f3aaae31a47d86cf3a8d
package version:
    0.21.0
package implementation anchor:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
```

TensorCore `0.21.0` supplies the final foundation described here. TensorDSLab
production remains on its exact `0.19.0` pin until Maintenance 12 atomically
closes its package-owned adoption and scientific rebaseline; publication of a
dependency does not alter the current resolved package by itself.

The package-owned Stage 29 lineage is:

```text
stable key:
    TensorCore/stage-29-literal-kernel-offset-axis-multinomial
substantive Design commit:
    397807ce634c29e6f3909acab7006cf2b8d5267d
Design tree:
    e7a853d173c71f53e58787f1678126ca88e8bb61
final Design authority:
    828017780321269fbace28e481aadf2d9e39adde
final implementation/publication:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
exact parent / published 0.20.0:
    e20b1e1594be894f210bafee2f55e7c46d6caf9c
```

TensorDSLab Design independently confirmed the substantive Design contract and
the exact published package bytes with zero findings. The publication does not
itself adopt a dependency or make a compatibility claim. The linked
publication-bound Maintenance 12 work order separately owns TensorDSLab
dispatch.

The governed `main` production baseline remains authoritative while the
Maintenance 12 bytes are absent from `main`:

- `ProbabilityKernel` remains a current dependency surface;
- afterpulse occurrence remains at-most-one per parent per generation;
- afterpulse delay remains conditionally multinomial;
- optional recovery weighting remains implemented;
- the current parametric pulse Configs remain public;
- `ds20k_veto()` remains the current no-argument provisional profile; and
- `common/axes.py` remains the current module path.

If the Maintenance 12 bytes appear unchanged on `main`, Review's fast-forward
has completed and the selected target is present, but final Design acceptance
remains pending until the work order and implementation index record
**Merged / Closed**.

The target below must not be described as already implemented.

## Architectural Outcome

TensorCore owns unit-independent tensor structure:

```text
TensorAxis
CountAxis
OffsetAxis
TensorKernel
generic kernel-role-to-field-dimension resolution
MultinomialDistribution-owned probability preparation
```

TensorDSLab owns physical quantities and detector/readout meaning:

```text
QuantityKernel
ExampleAxis
ChannelAxis
SampleAxis
profile construction
physical kernel leaves
scientific normalization and intensity laws
Runtime alignment and broadcasting policy
kernel-index-to-destination mapping
finite-window behavior
RNG roles and address schemas
products and orchestration
```

Product-owned semantic kernels are:

```text
Charge:
    DarkCountRate
    TimingJitter
    DirectCrosstalk
    DelayedCrosstalk
    Afterpulse
    SmearingWidth

PureWaveform:
    Pulse
```

`Recovery` is intentionally absent from the first target. Recovery-weighted
afterpulse charge is deferred until the structural and Poisson-branching
refactor is settled.

The dependency direction is strictly:

```text
TensorCore structural roots
    ↓
TensorDSLab QuantityKernel
    ↓
TensorDSLab product-owned physical kernels
    ↓
TensorDSLab Config-to-Runtime preparation
    ↓
TensorCore Distributions and RNG execution
```

TensorCore must not import Pint, TensorDSLab axes, detector profiles, products,
physical units, readout keys, or scientific policy. TensorDSLab must not fork
or duplicate generic TensorCore axes, tensor-kernel mechanics, probability
preparation, Distributions, or RNG engines.

## Kernel Vocabulary

A kernel contains immutable physical coefficients and the axes that give every
tensor dimension meaning.

It does not:

- execute an effect;
- own mutable RNG state;
- expose `apply()`, `draw()`, `sample()`, or `produce()`;
- own a simulation frontier;
- decide finite-window boundaries;
- choose execution dtype or device;
- own a mutable prepared cache; or
- replace Config-to-Runtime validation.

Every kernel has two ordered axis groups:

```text
conditioning axes
    choose which coefficient law applies

operation axes
    describe the literal geometry over which the law operates
```

The public tensor layout is exact:

```text
kernel.tensor.shape
    ==
tuple(axis.size for axis in (
    *kernel.conditioning_axes,
    *kernel.operation_axes,
))
```

There are no anonymous public dimensions. A global scalar kernel is a genuine
rank-zero tensor:

```text
conditioning_axes = ()
operation_axes = ()
tensor.shape = ()
```

It is not a tensor with undocumented singleton dimensions.

## Conditioning And Operation Axes

A conditioning axis selects which coefficient applies. Examples include:

- one value per example;
- one value per channel;
- one value per example and channel;
- one source-dependent law per sample; or
- future detector-region or operating-state axes.

An operation axis describes the coefficient geometry. Examples include:

- relative sample displacement;
- relative microcell-x displacement;
- relative microcell-y displacement;
- frequency;
- category; or
- another exact tensor-domain coordinate.

For one kernel:

```text
tensor dimensions =
(
    *conditioning_axes,
    *operation_axes,
)
```

Conditioning dimensions are leading dimensions. Operation dimensions are
trailing dimensions. Runtime may prepare another internal layout, but it must
not mutate the public kernel or reinterpret dimension meaning from coincident
lengths.

Exact conditioning-axis types are unique within the conditioning tuple.
TensorCore separately requires every operation target role to be unique: an
`OffsetAxis` contributes its `relative_to` role and another operation axis
contributes its exact concrete type. Multiple operation axes may therefore
have exact type `OffsetAxis` so long as their `relative_to` roles differ.

A conditioning role may intentionally equal an operation target role. For
example, a source-dependent law may condition on `SampleAxis` while operating
over `OffsetAxis(relative_to=SampleAxis, ...)`. The two tuple positions have
different structural functions even though both refer to the same target
field role.

## TensorCore-Owned `OffsetAxis`

`OffsetAxis` is a generic relative-index representation, not a detector or
Pint concept. It belongs in TensorCore.

It must derive directly from `TensorAxis[int]`, not from `CountAxis`.
`CountAxis` owns identity-free local ordinals and its equality is based on
count. An offset axis instead owns signed coordinates and a target semantic
axis role.

The provisional public sketch is:

```python
@final
@dataclass(frozen=True, slots=True, eq=False, init=False)
class OffsetAxis(TensorAxis[int]):
    relative_to: type[TensorAxis[Any]]
    offsets: tuple[int, ...]

    @final
    def __init__(
        self,
        *,
        relative_to: type[TensorAxis[Any]],
        offsets: tuple[int, ...],
    ) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def coordinates(self) -> tuple[int, ...]: ...

    def coordinate_at(self, index: int) -> int: ...

    def index_of(self, coordinate: int) -> int: ...

    @final
    @override
    def _require(self) -> None:
        return
```

`relative_to` is the package-authoritative recommended public name. The stored
value is always an exact semantic axis class, never one execution field's axis
instance:

```python
OffsetAxis(
    relative_to=SampleAxis,
    offsets=(-2, -1, 0, 1, 2),
)
```

not:

```python
OffsetAxis(
    relative_to=sample_axis,  # invalid
    offsets=(-2, -1, 0, 1, 2),
)
```

TensorCore should own:

- admission of `relative_to` as an exact `TensorAxis` class role;
- exact non-boolean built-in integer admission;
- tuple admission;
- preserved ordering;
- duplicate-offset rejection;
- positive, zero, and negative offsets;
- coordinate/index conversion;
- value equality and hashing over the exact final `OffsetAxis` class, exact
  `relative_to` class identity, and the complete ordered offsets; and
- type resolution under Python 3.14.

TensorCore should not own:

- physical distance or time units;
- mapping an index displacement into a destination;
- causality;
- anchor selection;
- boundary treatment;
- convolution;
- scatter;
- detector geometry; or
- whether a particular physical kernel may contain negative offsets.

The concrete generic class admits empty support. A TensorDSLab physical kernel
may require nonempty support and may narrow `relative_to` to one exact
package-owned role. For example, `TimingJitter`, `Afterpulse`, and `Pulse`
require their sole operation axis to have `relative_to is SampleAxis`.
TensorDSLab does not create `SampleOffsetAxis`, microcell-offset subclasses, or
another leaf-per-target hierarchy.

## TensorCore-Owned `TensorKernel`

The target `TensorKernel` owns literal axis instances rather than only axis
class roles.

The provisional shape is:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class TensorKernel[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](ABC):
    tensor: torch.Tensor
    conditioning_axes: ConditioningAxesT
    operation_axes: OperationAxesT

    @property
    def axes(self) -> tuple[TensorAxis[Any], ...]:
        return (*self.conditioning_axes, *self.operation_axes)

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def conditioning_shape(self) -> tuple[int, ...]: ...

    @property
    def operation_shape(self) -> tuple[int, ...]: ...

    @property
    def conditioning_element_count(self) -> int: ...

    @property
    def operation_element_count(self) -> int: ...

    @property
    def element_count(self) -> int: ...

    def index_at(self, flat_index: int) -> tuple[int, ...]: ...

    def flat_index_of(self, index: tuple[int, ...]) -> int: ...

    @abstractmethod
    def _require(self) -> None: ...
```

Generic `index_at()` and `flat_index_of()` continue to cover the complete
literal kernel shape. TensorDSLab first selects one conditioning cell or
aligned conditioning slab, then traverses `operation_shape` locally. It must
not treat complete conditioning-plus-operation flat identity as multinomial
category identity.

The structural constructor must require:

- one ordinary strided no-grad Torch tensor;
- exact tuple axis groups containing only `TensorAxis` instances;
- unique exact conditioning-axis types within the conditioning tuple;
- unique operation target roles;
- tensor rank equal to total axis count;
- tensor shape equal to exact axis sizes;
- one same-device contiguous defensive snapshot;
- identity equality;
- explicit unhashability; and
- one semantic `_require()` hook for final downstream leaves.

The current generic `_prepare()` hook leaves `TensorKernel` when
`ProbabilityKernel` is retired. Universal validation/snapshotting is final;
the concrete semantic leaf owns `_require()`.

Generic `TensorKernel` must not:

- flatten conditioning and operation dimensions into one law;
- assume every operation axis is an `OffsetAxis`;
- perform physical value validation;
- choose units;
- align against a field; or
- introduce a public prepared-kernel framework.

## Generic Role Resolution And Downstream Coordinate Alignment

TensorCore retains and revises the existing generic relationship:

```python
def require_kernel_dimensions(
    field: TensorField,
    kernel: TensorKernel[Any, Any],
) -> tuple[int, ...]:
    """Resolve complete ordered kernel roles to field dimensions."""
```

The exact rule is:

- each conditioning axis resolves by exact `type(axis)`;
- each `OffsetAxis` operation axis resolves by exact `axis.relative_to`;
- another operation axis resolves by exact `type(axis)`;
- returned dimensions follow complete kernel-axis order;
- `kernel.conditioning_rank` gives the conditioning/operation split; and
- absence of any required role is a generic `ValueError`.

TensorCore does not initially construct a richer alignment-plan value,
permute tensors, select conditioning coordinates, insert broadcasting
dimensions, or decide scientific legality. These actions carry effects and
downstream policy.

TensorDSLab owns coordinate correspondence and reordering after generic role
resolution. Its private Runtime preparation requires equal cardinality and a
one-to-one coordinate correspondence for coordinate-bearing conditioning
axes; rejects missing, extra, or duplicate coordinates; preserves exact target
order; and requires exact compatibility for identity-free positional axes.

TensorDSLab also decides:

- which conditioning axes may align for one physical operation;
- whether omission means global broadcast;
- what target field supplies context;
- how the tensor dimension is permuted;
- whether a view or materialized value is needed; and
- every error label exposed at its public Config boundary.

## Retirement Of Public `ProbabilityKernel`

The final architecture has no public `ProbabilityKernel`.

Probability is not the universal meaning of a physical kernel:

```text
TimingJitter:
    probability mass; sum over operation axes == 1

DirectCrosstalk:
    expected offspring intensity; no unity ceiling

DelayedCrosstalk:
    expected offspring intensity; no unity ceiling

Afterpulse:
    expected offspring intensity; no unity ceiling

Recovery:
    deterministic fraction in [0, 1] when later restored

Pulse:
    signed voltage response; no probability meaning
```

The current `ProbabilityKernel` also flattens its complete tensor for one
represented total and reverse suffix. For a `(channel, offset)` tensor that
would incorrectly create one global channel-offset law rather than one offset
law per channel.

TensorCore does not replace `ProbabilityKernel` with another public parameter
class in the first target. TensorDSLab selects one operation slab from its
conditioned physical kernel and constructs one complete law directly:

```python
MultinomialDistribution(
    counts=counts,
    probabilities=probabilities,
    completion_probability=completion,
)
```

`MultinomialDistribution` privately owns:

- exact float64 represented-probability admission;
- one defensive contiguous same-device probability snapshot;
- one ordered probability-value host extraction for finite elementwise
  `[0, 1]` admission and backend-independent `math.fsum` total;
- one bounded combined law-status/partition summary after that total is
  available;
- one device-side reverse represented-suffix preparation reused by that
  immutable Distribution instance;
- dynamic count and completion-probability validation;
- completed-law and allocation admission; and
- ordered conditional-Binomial execution.

`address.shape` equals `probabilities.shape`. Draw retains row-major
represented allocation, the unaddressed/word-free completion outcome, exact
count conservation, and TensorCore's deterministic address and word schedule.
Each conditioned slab constructs an independent complete Distribution, so
TensorCore never flattens distinct conditioning laws together.

Repeated probability preparation across distinct Distribution instances is
accepted in the first target. No current TensorDSLab evidence establishes
enough repeated rebinding or synchronization cost to justify a supported
`MultinomialParameters` value, argument, export, alias, typing surface, or
second constructor path. A later focused TensorCore stage may extract
evidence-backed reusable parameters if a real consumer demonstrates that
private constructor preparation is materially repeated.

The exact TensorCore Stage 29 work order must still freeze direct-probability
diagnostics, constructor validation order, probability and result shapes,
snapshot identity, export census, synchronization boundary, exact empty
operation-shape/zero-category behavior, and repeated-draw behavior. Draw may
not perform probability host materialization. Stage 29 must not introduce a
hidden reusable cache, trust token, factory, compatibility spelling, or
prepared-kernel framework in place of the removed public surface.

## TensorDSLab-Owned `QuantityKernel`

`QuantityKernel` belongs in `tensor_dslab/common/kernel.py`.

It introduces Pint and TensorDSLab's physical-unit policy without moving Pint
into TensorCore or Runtime execution.

The selected constructor shape is:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    init=False,
)
class QuantityKernel[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](
    TensorKernel[ConditioningAxesT, OperationAxesT],
    ABC,
):
    canonical_unit: ClassVar[str]
    _unit: pint.Unit

    def __init__(
        self,
        *,
        quantity: Quantity,
        conditioning_axes: ConditioningAxesT,
        operation_axes: OperationAxesT,
    ) -> None:
        checked = canonical_tensor_quantity(
            quantity,
            unit=type(self).canonical_unit,
            field=f"{type(self).__name__}.quantity",
        )
        object.__setattr__(self, "_unit", checked.units)
        super().__init__(
            tensor=checked.magnitude,
            conditioning_axes=conditioning_axes,
            operation_axes=operation_axes,
        )

    @property
    def quantity(self) -> Quantity:
        return cast(Quantity, _REGISTRY.Quantity(self.tensor, self._unit))
```

The helper name is provisional and remains unsupported. The important
constructor rule is exact: TensorCore owns the one stored tensor snapshot;
TensorDSLab stores its canonical Pint unit and exposes one Quantity view over
that exact owned magnitude. It does not retain the caller's Quantity or a
second competing tensor payload.

Construction owns only representation invariants:

- the input is a Pint `Quantity`;
- it is rebuilt into TensorDSLab's package-owned registry;
- it is converted once into the semantic leaf's canonical unit;
- an exact built-in scalar, NumPy array, or CPU Torch tensor magnitude is
  admitted and normalized through one package-owned path;
- the magnitude is exactly a CPU `torch.float64` tensor;
- the magnitude is strided and gradient-free;
- one contiguous defensive CPU tensor snapshot is owned;
- rank and shape match all axes;
- the unit is convertible to the leaf's canonical unit;
- the exposed Quantity and inherited tensor field reference the same owned
  magnitude; and
- caller mutation cannot change stored state.

The class is identity-equal and unhashable. Default dataclass tensor equality,
hashing, and representation are prohibited.

Construction does not:

- align with an execution field;
- insert broadcasting dimensions;
- choose an execution device;
- build a Distribution;
- prepare RNG addresses;
- enforce generation or accumulation ceilings; or
- mutate the caller's quantity.

Config-to-Runtime preparation remains the sole contextual validation boundary.
It explicitly materializes required coefficient tensors on the product device.
Stochastic probability, intensity, rate, and width operands remain
`torch.float64`; deterministic `Pulse` coefficients are transferred and cast
once to the requested waveform dtype. Public Config and profile construction
therefore remain host-side and device-independent.

The existing public `quantity(...)` scalar helper remains supported.
`quantities(...)` retains its current tuple-vector input and additionally
accepts an arbitrary-rank CPU Torch tensor, so users can hand-build
multidimensional coefficient quantities without accessing the private Pint
registry. Both helpers return fresh package-registry quantities; the kernel
constructor still owns its independent canonical tensor snapshot.

## TensorDSLab Module Shape

The selected module direction is:

```text
tensor_dslab/
  common/
    axis.py
      # ExampleAxis, ChannelAxis, SampleAxis
    kernel.py
    units.py
  readout/
    profiles.py
    charge/
      config.py
      field.py
      kernel.py
      runtime/
        branching.py
        counts.py
        prepare.py
        produce.py
        validate.py
    pure_waveform/
      config.py
      field.py
      kernel.py
      runtime/
        prepare.py
        produce.py
        validate.py
```

`common/axes.py` is renamed to singular `common/axis.py` in the eventual
breaking migration. The old plural module is deleted without a forwarding
module, alias, or compatibility import.

`common/kernel.py` owns only `QuantityKernel`. Product-specific physical
kernels remain in the owning product's `kernel.py`.

`common/axis.py` owns the current readout semantic axes. TensorCore's concrete
`OffsetAxis` composes with those roles directly; TensorDSLab adds no offset
subclasses. Future microcell semantic axes belong to the package that owns
those actual product dimensions and are not added as placeholders.

Charge Runtime becomes flat. The current `runtime/effects/` hierarchy is
retired after its real algorithms move to cohesive owners:

```text
branching.py
    fixed-generation direct/delayed/afterpulse branching

counts.py
    shared checked count arithmetic and ceilings

prepare.py
    Config-to-Runtime alignment, units, and prepared laws

produce.py
    Charge orchestration

validate.py
    immediate Charge/product relationships
```

No `effect.py`, effect ABC, registry, callback graph, `frontier.py`,
`ledger.py`, `count_domain.py`, or TensorDSLab multinomial implementation is
created merely for structural symmetry.

The `readout/` product boundary remains intact. A flat Charge Runtime does not
authorize merging substantial Charge algorithms into
`readout/runtime/prepare.py` or `readout/simulation.py`.

## Runtime Record Granularity

Runtime records are compiled product execution state, not validated
field-for-field reflections of public Configs or kernels.

The product-level records remain:

```text
ChargeRuntime
PureWaveformRuntime
NoiseWaveformRuntime
AnalogWaveformRuntime
DigitizedWaveformRuntime
SamplingRuntime
```

A product Runtime may store a simple mechanism directly as one execution
scalar or tensor. For example, an already aligned dark-count mean or smearing
width does not require `DarkCountRuntime` or `SmearingWidthRuntime` merely to
preserve a one-to-one class pairing.

A nested mechanism Runtime is justified only when several coherent derived
facts must travel together and their relationship owns a useful invariant,
such as an aligned probability tensor plus operation offsets, resolved
dimensions, and RNG/address facts. The exact work order must justify every
nested Runtime by its fields and consumers; naming symmetry is not evidence.

Conceptually:

```python
@final
@dataclass(frozen=True, slots=True)
class ChargeRuntime:
    correlated_avalanche_generations: int
    dark_count_mean: torch.Tensor | None
    smearing_width: torch.Tensor | None
    timing_jitter: TimingJitterRuntime | None
    # Additional nested mechanism records exist only where their
    # multi-field invariant is demonstrated by the exact work order.


@final
@dataclass(frozen=True, slots=True)
class PureWaveformRuntime:
    pulse: torch.Tensor
    sample_offsets: tuple[int, ...]
```

These sketches establish granularity, not an exact future field census.
Runtime stores no public Config, `QuantityKernel`, Pint `Quantity`, semantic
product, mutable cache, or execution method. Preparation extracts, aligns, and
materializes invariant execution operands. Producers combine those Runtime
facts with dynamic product tensors, construct any complete dynamic
Distribution, draw it, and publish the result.

## Semantic Physical Kernels

### `DarkCountRate`

```python
@final
class DarkCountRate[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
](QuantityKernel[ConditioningAxesT, tuple[()]]):
    """Represent independent avalanche rate."""
```

Contract:

- canonical unit: `Hz`;
- finite and nonnegative;
- no operation axes;
- arbitrary accepted conditioning axes;
- rank zero means one global rate; and
- Runtime forms dimensionless Poisson means as rate times exposure.

Examples:

```text
global:
    conditioning_axes = ()
    tensor.shape = ()

per channel:
    conditioning_axes = (channel_axis,)
    tensor.shape = (channel_count,)

per example/channel:
    conditioning_axes = (example_axis, channel_axis)
    tensor.shape = (example_count, channel_count)
```

### `TimingJitter`

```python
@final
class TimingJitter[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
](
    QuantityKernel[
        ConditioningAxesT,
        tuple[OffsetAxis],
    ],
):
    """Represent a conservative relative-sample allocation law."""
```

Contract:

- canonical unit: dimensionless;
- finite and nonnegative;
- exactly one sample-offset operation axis in the first implementation;
- that axis has `relative_to is SampleAxis`;
- operation-axis mass equals unity independently for every conditioning
  coordinate within the accepted numerical tolerance;
- no implicit completion or drop category in the abstract law; and
- preparation records the aligned selected probability slab while the
  producer binds frontier counts, constructs TensorCore's complete
  `MultinomialDistribution`, and draws it.

Unity means the abstract translation law conserves input count. It does not
guarantee finite-field conservation: allocations whose translated
destinations lie outside the finite `SampleAxis` are discarded.

The public kernel is one complete finite discrete PMF. It is not an
unbounded analytic Gaussian plus a hidden tail. The first work order accepts
the finite PMF through direct `TimingJitter` construction and does not specify
a Gaussian factory or another analytic-to-discrete construction API. Any later
such API must explicitly select a finite discretization policy and its parity
classification; it must not silently truncate and renormalize the current
analytic Gaussian law while claiming scientific continuity.

### `DirectCrosstalk`

```python
@final
class DirectCrosstalk[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[OffsetAxis, ...],
](QuantityKernel[ConditioningAxesT, OperationAxesT]):
    """Represent expected direct offspring intensity per operation cell."""
```

Contract:

- canonical unit: dimensionless;
- finite and nonnegative;
- every operation axis is an exact `OffsetAxis`, and the ordered
  `relative_to` roles are the literal destination geometry;
- exactly one operation axis targets `SampleAxis`;
- its sample offsets are nonnegative for direct crosstalk;
- no operation axis targets `ExampleAxis`, because independent examples never
  exchange avalanches;
- sum over operation axes is expected direct offspring multiplicity per
  parent;
- the sum may be greater than one; and
- the producer constructs retained destination rates from Runtime facts and
  performs one tensor-valued Poisson draw.

### `DelayedCrosstalk`

```python
@final
class DelayedCrosstalk[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[OffsetAxis, ...],
](
    QuantityKernel[ConditioningAxesT, OperationAxesT],
):
    """Represent expected delayed offspring intensity per operation cell."""
```

It has the same generic intensity law as `DirectCrosstalk`, while remaining a
distinct public semantic type for configuration, diagnostics, RNG role
identity, causality, and future physical divergence. It likewise requires
exactly one sample-targeting operation axis, rejects `ExampleAxis` as an
operation target, and requires strictly positive sample offsets.

### `Afterpulse`

```python
@final
class Afterpulse[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
](
    QuantityKernel[
        ConditioningAxesT,
        tuple[OffsetAxis],
    ],
):
    """Represent expected afterpulse offspring intensity by sample offset."""
```

The selected first refactor deliberately changes the current model:

- each parent may produce zero, one, or multiple afterpulses in one
  generation;
- each operation cell stores expected afterpulse offspring per parent;
- values are finite and nonnegative;
- the sole operation axis has `relative_to is SampleAxis`;
- there is no probability-sum ceiling;
- Runtime uses the same deterministic retained-rate construction followed by
  tensor-valued `PoissonDistribution` used for crosstalk;
- positive-delay/causality requirements remain TensorDSLab-owned;
- out-of-window destinations are discarded before the retained-rate draw; and
- every retained afterpulse contributes one full PE-equivalent charge.

The first implementation has no recovery-weighted charge, occurrence
`BinomialDistribution`, conditional delay `MultinomialDistribution`, or
no-afterpulse remainder.

### `SmearingWidth`

```python
@final
class SmearingWidth[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
](QuantityKernel[ConditioningAxesT, tuple[()]]):
    """Represent dimensionless relative Gaussian charge width."""
```

Contract:

- canonical unit: dimensionless;
- finite and nonnegative;
- no operation axes;
- conditions only on axes on which the response width truly varies; and
- Runtime converts it into the tensor-valued Gaussian scale.

It stores standard deviation, not variance.

### `Pulse`

```python
@final
class Pulse[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
](
    QuantityKernel[
        ConditioningAxesT,
        tuple[OffsetAxis],
    ],
):
    """Represent one complete sampled voltage response per input charge."""
```

Contract:

- canonical unit: `mV`, interpreted per one unit PE-equivalent input charge;
- finite signed values;
- exactly one sample-offset operation axis in the current readout product;
- that axis has `relative_to is SampleAxis`;
- no probability normalization;
- deterministic convolution; and
- explicit discrete-convolution convention with no hidden sample-width factor.

The first work order requires only direct construction from the complete
sampled response. It adds no analytic pulse class, builder, calibration
loader, or alternate constructor. The DS20k negative polarity is applied
exactly once in the directly supplied sampled `Pulse`; Runtime consumes the
signed coefficients and does not flip them again.

## Kernel Construction Boundary

The only required public creation path in the first TensorDSLab target is each
kernel's ordinary keyword-only constructor:

```python
timing_jitter = TimingJitter(
    quantity=probability_quantity,
    conditioning_axes=(),
    operation_axes=(
        OffsetAxis(
            relative_to=SampleAxis,
            offsets=(-2, -1, 0, 1, 2),
        ),
    ),
)
```

The stage does not design or authorize public builders, analytic-law
factories, calibration loaders, parsers, `from_*` constructors, registries, or
serialization-backed reconstruction. A package-authored profile may call the
same direct constructors internally; that does not create a second public
kernel-construction abstraction.

## Profile Geometry Contract

The target profile signature is:

```python
def ds20k_veto(
    *,
    sample_axis: SampleAxis,
    channel_axis: ChannelAxis | None = None,
    example_axis: ExampleAxis | None = None,
) -> ReadoutConfig:
    """Return a fresh provisional profile for one supplied target geometry."""
```

`sample_axis` is required because pulse sampling, delay integration, and
sampling-dependent noise preparation require an exact grid.

`channel_axis` and `example_axis` are optional availability inputs. Supplying
them does not require every kernel to vary over them.

The exact invariant is:

```text
conditioning axes of every returned kernel
    ⊆
axes supplied to ds20k_veto()
```

Operation geometry obeys the parallel target-role invariant:

```text
relative_to roles of every returned OffsetAxis
    ⊆
axis types supplied to the profile
```

Therefore:

- a global `DarkCountRate` remains rank zero even when channel/example axes are
  supplied;
- a per-channel kernel requires `channel_axis`;
- a per-example kernel requires `example_axis`;
- a per-example/per-channel kernel requires both;
- omitting an axis prohibits every returned kernel from conditioning on it;
- omitting an axis also prohibits an operation offset relative to that role;
- the profile must not invent a hidden semantic axis;
- the profile must not silently average or collapse a conditioned kernel; and
- supplied axes define target availability, not stored tensor density.

The profile should use the exact supplied axis objects when it constructs a
conditioned kernel. Global kernels omit them. Runtime later requires the
actual simulation field to contain compatible target axes.

The profile remains provisional, uses the same direct kernel constructors as
hand-written configuration, and makes no calibration claim.

## Runtime Alignment And Broadcasting

Public kernels preserve their native conditioning-axis order and coordinates.
Direct kernel construction does not require borrowing the execution field's
exact axis instances.

For every present conditioning axis, Runtime:

1. locates the target field axis by exact semantic type;
2. establishes compatible semantics;
3. requires a coordinate bijection;
4. obtains the generic TensorCore permutation;
5. reindexes the corresponding tensor dimension;
6. permutes conditioning dimensions into the operation's expected order; and
7. binds the prepared representation to the actual target axes.

Missing, extra, or duplicate coordinates fail. Equal length alone is
insufficient.

If a target field axis is absent from the kernel's conditioning axes, the
kernel is global over that axis. Runtime may insert singleton dimensions and
use Torch broadcasting. It must not store a physically expanded public
kernel, and it must not materialize the broadcast unless an operation
specifically requires owned storage.

If a kernel includes a conditioning axis, it must cover that complete target
axis. Partial conditioning is not treated as global.

Operation axes remain in declared order. Runtime maps operation index tuples
to destination displacements. It never infers operation meaning from tensor
length.

## Charge Execution Order And Generations

The first target preserves one explicit product sequence:

```text
Photoelectrons
    -> add dark counts
    -> apply timing jitter
    -> run correlated-avalanche generations
    -> apply charge smearing
    -> Charge
```

Dark-count avalanches therefore pass through timing jitter and may seed
correlated avalanches. Smearing applies only after every accepted branching
generation has completed.

`ChargeConfig.correlated_avalanche_generations` is the exact number of
descendant-producing rounds:

- `0` performs no correlated-avalanche draw;
- `1` lets the post-dark-count/post-jitter seed frontier produce one child
  generation; and
- `N` performs exactly `N` rounds, with round indices `0` through `N - 1`.

Within one round, direct crosstalk, delayed crosstalk, and afterpulse
independently consume the same current frontier. Their three child tensors are
checked and summed once to form the next frontier. Same-round children do not
feed another mechanism until the following round. The initial frontier and
every retained child contribute one full PE-equivalent count, charge, and
charge-square term before optional final smearing.

Config validation is fail-closed:

- when all three branching kernels are `None`,
  `correlated_avalanche_generations` must be zero; and
- when any branching kernel is present,
  `correlated_avalanche_generations` must be positive.

This leaves exactly one way to disable each mechanism (`None`) and prevents
configured kernels from being silently ignored by a zero generation count.

## Poisson Branching Rebaseline

The first branching target makes direct crosstalk, delayed crosstalk, and
afterpulse share one mathematical execution pattern:

```text
frontier counts
    ×
conditioned operation-cell offspring intensities
    ↓ deterministic finite-window destination-rate construction
retained destination rates
    ↓ one tensor-valued Poisson draw
retained child counts
```

For destination `d`:

```text
lambda[d]
    =
sum over source s and represented operation cell k mapping s -> d (
    frontier[s] * intensity[k]
)
```

Then:

```text
children[d] ~ Poisson(lambda[d])
```

Poisson thinning and superposition make this exactly equivalent to independent
per-parent Poisson offspring assigned through the represented operation
intensity. No source-by-kernel allocation tensor is required.

The Runtime:

- aligns and broadcasts conditioning axes;
- maps each operation index tuple to a destination displacement;
- accumulates only represented in-window rates;
- discards out-of-window intensity contributions before drawing;
- enforces TensorCore's Poisson mean domain per retained destination;
- performs checked count accumulation;
- uses fixed package-owned mechanism/generation addresses; and
- retains the fixed-generation frontier dependency.

The three mechanisms remain distinct semantic types and RNG roles even though
they share implementation mechanics.

## Clean RNG Rebaseline

The first target deliberately rebases private TensorDSLab role and address
identity. Namespace `0x54445331` remains package-owned, while active stream
values become one compact execution-oriented table:

```text
0x0000_0001    white noise
0x0000_0002    PSD noise
0x0000_0003    dark counts
0x0000_0004    timing jitter
0x0000_0005    direct crosstalk
0x0000_0006    delayed crosstalk
0x0000_0007    afterpulse
0x0000_0008    charge smearing
```

Retired numeric gaps and old afterpulse/smearing values carry no reservation,
alias, or compatibility meaning. No public Config exposes keys, streams,
addresses, domains, or quanta.

The address schema is exact:

- white noise, PSD noise, dark counts, and charge smearing use one complete
  atomic address with `shape=()` and `quantum=0`;
- timing jitter uses one root whose domain shape is the selected operation
  probability shape, with `quantum=0`;
- direct crosstalk, delayed crosstalk, and afterpulse each use their own key
  and one root domain of shape
  `(correlated_avalanche_generations,)`, select the exact zero-based
  generation, and retain `quantum=0`; and
- every tensor-valued draw uses the non-renumbered complete product
  `RngElements` lattice appropriate to its output destinations.

Destination-based Poisson addresses are independent of source traversal and
kernel accumulation order. Timing-jitter category addresses retain declared
row-major operation order. TensorDSLab uses no manual ordinal offset,
user-supplied key, mutable stream, global Torch RNG, or retired occurrence/
delay address split.

This is a predeployment completed-output rebaseline. No eventwise output or
old-address continuity is claimed. TensorCore must still preserve its generic
raw-word law for an identical new complete `RngAddress` and ordinal; the
TensorDSLab candidate freezes all newly selected keys, shapes, generations,
category order, and representative output fixtures.

## Afterpulse Intentional Divergence

The afterpulse change is not behavior-preserving.

| Boundary | Current package | Selected first refactor |
| --- | --- | --- |
| multiplicity per parent/generation | Bernoulli; at most one | Poisson; unbounded nonnegative count |
| public parameter | occurrence probability plus mean delay | expected offspring intensity by offset |
| delay execution | conditional Multinomial | collapsed destination Poisson rate |
| no-event outcome | explicit completion | inherent Poisson zero count |
| charge | one or recovery-weighted | exactly one PE-equivalent per retained child |
| charge-square contribution | recovery squared where active | exactly one per retained child |
| RNG | occurrence and delay quanta | one addressed destination Poisson law |
| tail/boundary | current conditional-law remainder and finite discard | finite represented in-window rate only |

This is an intentional scientific divergence from:

- the current TensorDSLab at-most-one afterpulse law;
- the current recovery-weighted Charge ledgers; and
- donor behavior wherever the donor models one exclusive afterpulse or
  recovery suppression.

The reason is to establish one clear, vectorized intensity-kernel branching
architecture before adding coupled cell-state/recovery physics.

Required evidence includes:

- analytic destination means from an independently evaluated intensity
  convolution;
- Poisson variance/mean and zero-count statistics;
- fixed-generation recursion expectations;
- finite-window discard;
- exact full-charge ledger identities;
- mechanism isolation;
- same-address replay;
- chunk/traversal invariance where promised by TensorCore;
- mutants that halve intensity, shift one offset, reuse one key, double-apply
  charge, or reintroduce recovery weighting; and
- an explicit `docs/parity.md` intentional-divergence entry in the eventual
  production candidate.

No distributional or eventwise parity with the retired at-most-one model is
claimed.

## Deferred Recovery

No `Recovery` class or `AfterpulseRecoveryConfig` replacement is implemented
in the first kernel migration.

Every retained afterpulse initially deposits one PE-equivalent charge:

```text
afterpulse count contribution
    ==
afterpulse charge contribution
    ==
afterpulse charge-square contribution
```

A later focused stage may add a deterministic recovery coefficient, microcell
state, or another physical model only after deciding:

- whether recovery depends on delay alone or prior cell history;
- whether the same cell can fire more than once in one generation;
- how conditioning axes represent bias/temperature/cell identity;
- whether recovery affects only charge or later offspring laws;
- how the new state interacts with finite boundaries and recursion; and
- what donor/parity boundary applies.

The later stage must not be inferred from the provisional class sketch in the
superseded handoff.

## Timing Jitter And Finite Boundaries

Timing jitter remains a conservative abstract translation law:

```text
sum of represented operation mass per conditioning coordinate == 1
```

Runtime allocation conserves each source count across the complete offset law.
After allocation, translated destinations outside the finite `SampleAxis` are
discarded. Therefore:

```text
abstract law:
    count conserving

finite product window:
    may lose translated charge at a boundary
```

The implementation must not weaken the unity law merely because finite-window
execution can discard results.

## Future Literal Spatial Geometry

The architecture must support future pixelated SiPM products without teaching
TensorCore detector dimensions.

A translation-invariant spatial/temporal crosstalk kernel uses one operation
axis per destination displacement:

```python
operation_axes = (
    OffsetAxis(
        relative_to=MicrocellXAxis,
        offsets=(-1, 0, 1),
    ),
    OffsetAxis(
        relative_to=MicrocellYAxis,
        offsets=(-1, 0, 1),
    ),
    OffsetAxis(
        relative_to=SampleAxis,
        offsets=(0, 1, 2),
    ),
)
```

The concrete three-dimensional operation tuple has static type
`tuple[OffsetAxis, OffsetAxis, OffsetAxis]`. Its values, rather than three
target-specific subclasses, distinguish the microcell-x, microcell-y, and
sample roles.

The tensor shape is literally:

```text
(
    microcell_x_offset,
    microcell_y_offset,
    sample_offset,
)
```

If the law also varies by source region, bias state, or channel, those axes
precede the operation axes as conditioning dimensions.

Kernel geometry does not itself generalize today's product fields. The current
readout fields remain exactly `(ExampleAxis, ChannelAxis, SampleAxis)` until a
separate product-schema stage introduces microcell dimensions and a matching
TensorG4DS/Silex boundary.

## Config Direction

The final conceptual Config shape is:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeConfig:
    correlated_avalanche_generations: NonnegativeInteger
    timing_jitter: TimingJitter | None = None
    direct_crosstalk: DirectCrosstalk | None = None
    delayed_crosstalk: DelayedCrosstalk | None = None
    afterpulse: Afterpulse | None = None
    dark_counts: DarkCountRate | None = None
    smearing_width: SmearingWidth | None = None


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PureWaveformConfig:
    pulse: Pulse
```

Every optional `ChargeConfig` field is an immutable physical kernel. It is not
an executable effect object, Distribution subclass, callback, or Distribution
factory. `correlated_avalanche_generations` is a direct orchestration control,
not a reason to restore a wrapper around the three branching kernels. Its
generation-index meaning and fail-closed zero relationship are frozen above.

Use `None` when absence skips an algorithmic operation. Use a rank-zero kernel
for a global physical value. Do not use undocumented singleton dimensions to
mean global.

The execution mapping is fixed:

```text
TimingJitter       -> MultinomialDistribution
DarkCountRate      -> PoissonDistribution
DirectCrosstalk    -> PoissonDistribution
DelayedCrosstalk   -> PoissonDistribution
Afterpulse         -> PoissonDistribution
SmearingWidth      -> GaussianDistribution
Pulse              -> deterministic convolution
```

Users configure physical coefficients, geometry, conditioning, and the direct
generation ceiling. TensorDSLab Runtime preparation owns alignment and
invariant execution operands; producers bind dynamic operands, construct and
draw complete Distributions, and map results into TensorDSLab products.
TensorCore owns the generic stochastic algorithms. `PureWaveformConfig` holds
`Pulse` directly, while noise and digitizer Configs retain their
product-specific structures.

The following current Config families become retirement candidates in the
exact migration that replaces their behavior:

```text
TimingJitterConfig
DarkCountConfig
FixedDelayConfig
ExponentialDelayConfig
DirectCrosstalkConfig
DelayedCrosstalkConfig
AfterpulseConfig
AfterpulseRecoveryConfig
CorrelatedAvalancheConfig
ChargeSmearingConfig
TpcFebSnrPulseConfig
VetoPduPulseConfig
```

There are no aliases or dual old/new Config unions in the final state. Whether
one temporary local implementation branch uses adapters is an implementation
detail; no intermediate compatibility surface may merge or publish.

## Public API Direction

The intended supported paths are:

```text
tensor_dslab.common
    ExampleAxis
    ChannelAxis
    SampleAxis
    QuantityKernel

tensor_dslab.readout.charge
    ChargeConfig
    Charge
    DarkCountRate
    TimingJitter
    DirectCrosstalk
    DelayedCrosstalk
    Afterpulse
    SmearingWidth

tensor_dslab.readout.pure_waveform
    PureWaveformConfig
    PureWaveform
    Pulse

tensor_dslab.readout.profiles
    ds20k_veto
```

Exact package-root re-exports remain a future work-order decision. Runtime
records, preparation actions, branching mechanics, alignment helpers, and RNG
address constructors remain unsupported implementation details.

## Easy-To-Hard Migration Sequence

The architecture should be implemented through bounded package-owned stages,
not one uncontrolled rewrite.

### Foundation A: TensorCore Stage 29 / `0.21.0` — complete

TensorCore completed one focused package-owned Stage 29 rather than a
transitional public topology. It independently designed, implemented,
validated, reviewed, closed, and published:

- `OffsetAxis`;
- literal-axis `TensorKernel`;
- revised `require_kernel_dimensions(...)`;
- the direct-`probabilities` `MultinomialDistribution` constructor with
  private stable-total and reverse-suffix preparation;
- no public reusable multinomial-parameter surface;
- retirement of public `ProbabilityKernel` without an alias;
- preservation of deterministic `RngKey`, `RngElements`, `RngAddress`,
  Threefry words, row-major category addresses, and equivalent Multinomial
  results for the same numerical law/address; and
- exact typing, validation, snapshot, synchronization, and export contracts.

The exact published version is TensorCore `0.21.0`. TensorDSLab did not author
or implement that stage.

### Foundation B: TensorDSLab representation

TensorDSLab adopts the exact published TensorCore dependency and adds:

- `common/axis.py`;
- `common/kernel.py`;
- `QuantityKernel`;
- structural and typing tests;
- Runtime conditioning-axis alignment/broadcasting; and
- the new profile axis-availability contract.

Because TensorCore Stage 29 removes `ProbabilityKernel` in the same release,
TensorDSLab's exact `0.21.0` dependency adoption and every current
`ProbabilityKernel` consumer must close atomically. Implementation may work
through the easy-to-hard internal sequence, but no intermediate pin-only or
dual-surface state may merge or publish.

### Simple scalar/deterministic kernels

Migrate:

- `DarkCountRate`;
- `SmearingWidth`;
- `Pulse`; and
- `ds20k_veto(...)`.

These prove rank-zero/global state, conditioned scalar state, sampled operation
geometry, unit conversion, and broadcasting before branching changes.

### Poisson branching kernels

Migrate:

- `DirectCrosstalk`;
- `DelayedCrosstalk`;
- Poisson `Afterpulse`;
- flat `branching.py`; and
- full-charge count/ledger identities.

Retire the current at-most-one and recovery-weighted afterpulse behavior in
this same exact scientific rebaseline.

### Conservative allocation

Migrate `TimingJitter` through the new Distribution-owned multinomial
preparation. Prove unity per conditioning coordinate and separate finite-window
discard.

### Later physical complexity

Only after the above is stable, consider:

- recovery weighting or cell state;
- source-dependent kernels;
- measured calibration loaders;
- microcell product axes;
- Silex profiles;
- Axioelectrons source composition; and
- the exact mutually adopted `1.0.0` CUDA release-candidate matrix.

TensorCore's package stage stays free of `QuantityKernel`, detector profiles,
physical leaves, recovery policy, branching, finite-window execution, or
TensorDSLab scientific changes.

## Required Evidence

### TensorCore consumer evidence

TensorDSLab must independently verify:

- exact future dependency commit/tree/version;
- `OffsetAxis` runtime and typing behavior;
- literal `TensorKernel` axis instances and shape;
- conditioning versus operation row-major identity;
- exact TensorCore role-resolution output and TensorDSLab-owned coordinate
  permutation;
- defensive snapshotting;
- retired `ProbabilityKernel` absence from every supported surface/path;
- exact direct multinomial probability preparation;
- absence of a public reusable multinomial-parameter surface;
- permitted repeated preparation across distinct Distribution instances;
- import isolation;
- package/export topology; and
- no changed RngAddress/CounterRng word identity outside declared rebaseline.

### Quantity representation

Tests must prove:

- rank-zero and arbitrary-rank Torch magnitudes;
- dimensionless and unit-bearing quantities;
- direct construction from mixed convertible units;
- incompatible-unit rejection;
- exact canonical CPU `torch.float64` snapshotting;
- explicit Runtime materialization on the product device;
- no caller-storage alias;
- no gradients;
- exact axis/rank/shape relationships;
- identity equality and unhashability; and
- Pint absence from Runtime records and execution tensors.

### Axis alignment

Tests must prove:

- same coordinates/same order;
- same coordinates/different order;
- different coordinates with equal length;
- missing, extra, and duplicate coordinates;
- exact positional-axis handling;
- global broadcasting;
- per-channel conditioning;
- per-example conditioning;
- per-example/per-channel conditioning;
- profile availability-subset admission;
- omitted-axis rejection for conditioned kernels;
- no hidden profile axes;
- no public stored expansion; and
- operation dimensions preserved in order.

### Scientific kernels

Tests must prove:

- global and conditioned dark-count means;
- timing-jitter unity per conditioning coordinate;
- finite-window timing-jitter loss without weakening abstract unity;
- direct/delayed/afterpulse intensity meanings;
- Poisson destination mean/variance and recursion;
- fixed-generation termination;
- exact one-PE afterpulse charge and charge-square contribution;
- recovery behavior and API absence in the first target;
- signed sampled pulse polarity applied once;
- smearing-width standard-deviation meaning;
- count and floating-ledger ceilings; and
- no duplicate or silently normalized physical coefficients.

### RNG and execution

Tests must freeze:

- separate mechanism keys;
- explicit generation identity;
- destination-based Poisson addresses;
- no occurrence/delay Multinomial afterpulse addresses in the new model;
- no role collision;
- no global Torch RNG effect;
- exact replay;
- accepted chunk/traversal invariance;
- predeployment completed-output rebaseline; and
- exact compact role table with no retired numeric reservation.

### Parity and documentation

The final production candidate must update:

- `docs/architecture/rebuild.md`;
- `docs/architecture/readout.md`;
- `docs/architecture/tensors.md`;
- `docs/physics/correlated_avalanches.md`;
- `docs/parity.md`;
- `docs/api.md`;
- `docs/validation.md`;
- `README.md`;
- `CONTRIBUTING.md`;
- `AGENTS.md`; and
- the exact implementation work order.

The current afterpulse law must remain described as current until the changing
candidate is approved and merged. The future work order needs lifecycle-neutral
or self-effecting wording so fixed evidence does not become stale on
fast-forward.

## Non-Goals

This architecture does not authorize:

- native G4DS ingestion;
- TensorG4DS binning;
- a public calibration IO format;
- a generic effect framework;
- mutable kernel application objects;
- TensorCore Pint support;
- detector axes inside TensorCore;
- automatic product-rank generalization;
- hidden axis construction;
- partial-coordinate filling;
- probability normalization or intensity clipping;
- per-avalanche expansion;
- a TensorDSLab Distribution implementation;
- restored overflow products;
- recovery weighting in the first migration;
- CUDA work before the exact mutually adopted `1.0.0` release candidates;
- performance claims;
- deployment or calibration claims; or
- a production Implementation dispatch.

## Cross-Package Stop Conditions

The publication-bound Maintenance 12 work order records these prerequisites as
complete:

1. TensorCore Design must freeze exact Stage 29 `OffsetAxis`, `TensorKernel`,
   `require_kernel_dimensions`, direct-probability
   `MultinomialDistribution`, no-parameters, and `ProbabilityKernel`
   retirement contracts. This is satisfied by exact confirmed Design commit
   `397807ce634c29e6f3909acab7006cf2b8d5267d` and its contract-identical
   evidence-cadence authority `828017780321269fbace28e481aadf2d9e39adde`.
2. TensorCore completed package-owned implementation, validation, review,
   local closeout, and GitHub publication at exact `78d0891`.
3. TensorDSLab reviewed that exact published containing commit.
4. TensorDSLab Design selected that exact commit in the linked Maintenance 12
   work order.
5. Exact changed paths, exports, diagnostics, tests, scientific rebaseline,
   and old-surface retirements are frozen there.
6. Implementation, Validation, and Review routes are verified at dispatch.

Any mismatch between this planning record, current package sources, later
TensorCore contracts, or a future work order returns to TensorDSLab Design.
No role may silently choose the closest available API.

## Design Disposition

The architecture is selected in principle:

> TensorCore owns literal tensor geometry and generic Distribution execution.
> TensorDSLab owns physical quantities, profile geometry, semantic kernel laws,
> contextual alignment, finite boundaries, branching, and products.

The Poisson/full-charge afterpulse simplification is the selected first
scientific target because it makes direct crosstalk, delayed crosstalk, and
afterpulse share one exact intensity-to-Poisson branching pattern. Recovery is
deferred deliberately.

This planning record remains non-dispatchable. Its focused, publication-bound
TensorDSLab work order is
[Maintenance 12](maintenance_12_tensorcore_0_21_kernel_geometry_quantity_refactor.md),
which is the sole execution authority for the adopted package and scientific
scope.
