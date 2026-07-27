# Maintenance 12 TensorCore 0.21 Kernel Geometry And Quantity Refactor

Status: **Design-complete / exact TensorCore 0.21.0 publication confirmed /
Implementation authorized under the exact containing work-order bytes**

Stable key:
`TensorDSLab/maintenance-12-tensorcore-0-21-kernel-geometry-quantity-refactor`

## Purpose

Adopt exact published TensorCore `0.21.0` Stage 29 and replace
TensorDSLab's current effect-specific scalar Config hierarchy with literal
physical quantity kernels, compiled product Runtime state, and direct
TensorCore Distribution execution.

This is one atomic predeployment architecture and scientific rebaseline. It:

- adopts TensorCore's concrete compositional `OffsetAxis`;
- adopts literal-axis `TensorKernel`;
- retires the consumed `ProbabilityKernel` surface;
- uses direct `MultinomialDistribution(..., probabilities=...)`;
- adds TensorDSLab's Pint-aware `QuantityKernel`;
- adds the physical Charge and Pulse kernels;
- makes detector profiles bind explicit available geometry;
- simplifies afterpulse to full-charge Poisson offspring;
- flattens Charge execution ownership without creating an effect framework;
- rebases the package-owned stochastic role/address table compactly; and
- updates the public Config/API boundary without compatibility aliases.

The accepted architecture is specified in
[Proposed Kernel Geometry And Quantity Architecture](proposed_kernel_geometry_and_quantity_architecture.md).
This work order extracts that architecture into an executable package boundary.
This publication-bound work order supersedes provisional dependency and
dispatch wording in the planning record. The physical-kernel architecture is
unchanged.

The stage follows:

- `CONTRIBUTING.md` for dependency ownership, semantic tensor roots,
  Config/Runtime separation, public typing, validation, scope, and evidence;
- `docs/architecture/tensors.md` for axes, fields, relationships, placement,
  snapshots, and synchronization;
- `docs/architecture/readout.md` and `docs/architecture/rebuild.md` for the
  product graph and readout execution boundary;
- `docs/physics/correlated_avalanches.md` for the current scientific baseline;
  and
- `docs/parity.md` for the deliberate afterpulse, timing-jitter, pulse,
  stochastic-address, and completed-output rebaselines.

## Dispatch State And Hard Stop

Every cross-package prerequisite is complete:

```text
TensorCore Stage 29 substantive Design:
    397807ce634c29e6f3909acab7006cf2b8d5267d
tree:
    e7a853d173c71f53e58787f1678126ca88e8bb61
TensorCore Stage 29 final Design authority:
    828017780321269fbace28e481aadf2d9e39adde
tree:
    5aaf779caec72cec25a9f37ea5c3cd69c66071f0
parent / published 0.20.0:
    e20b1e1594be894f210bafee2f55e7c46d6caf9c
TensorDSLab consumer disposition:
    exact substantive contract CONFIRMED / zero findings
TensorCore final implementation and publication:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
tree:
    af5c4f6d693fa25cf767f3aaae31a47d86cf3a8d
TensorDSLab published-byte disposition:
    exact package and publication contract CONFIRMED / zero findings
```

TensorDSLab Design independently verified the live GitHub main ref, local
TensorCore main/tracking identity, exact package source, direct constructor
signatures, retired surfaces, supported exports, and tracked topology. The
user explicitly authorized Maintenance 12 to begin against these bytes.

The exact containing commit of this work order is the TensorDSLab Design
authority. Persistent Implementation, Validation, and Review routes must
verify their role and clean exact starting state in the dispatch handoff.
Implementation must branch from the exact Design authority as:

```text
codex/maintenance-12-tensorcore-0-21-kernel-geometry-quantity-refactor
```

It may use only the allowlist below. Validation and Review inspect immutable
fixed commits in their role-private checkouts rather than a moving
Implementation branch.

No role may replace exact TensorCore `78d0891` with a branch head, nearby
release, floating Git dependency, superseded implementation candidate
`c7065074e798e4f61bb555779a6ee675023ad492`, or inferred API. Any contract
difference stops the affected work and returns to both package Design
authorities.

## Exact Current Baseline

The operative TensorDSLab baseline is:

```text
branch:
    main
commit:
    8517f09d6ecdf72434626bce0524f9f032998fd8
tree:
    3506fbf92d79473a3431e390ba3518ad5f166414
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
    exact published 0.19.0 containing commit
    ed17f4b637258f0a7f4544f235648b747f17fa44
```

Maintenance 11 is Merged / Closed at this exact commit. It is the current
scientific and addressed-Distribution baseline.

The exact selected TensorCore dependency is:

```text
version:
    0.21.0
published containing commit:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
published containing tree:
    af5c4f6d693fa25cf767f3aaae31a47d86cf3a8d
package implementation anchor:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
exact parent / superseded initial implementation candidate:
    c7065074e798e4f61bb555779a6ee675023ad492
canonical commit-bound source archive:
    tensorcore-stage29-78d0891.tar
    1,003,520 bytes
    SHA-256 0f8ca6a5270845c272e941ef928a325f1a0e57aa7fe81c965d04086a5823363f
    git archive --format=tar
    --prefix=tensorcore-stage29-78d0891/
    78d0891bf6c0fefbcad4abe09980867c54202a9e
wheel:
    tensor_core-0.21.0-py3-none-any.whl
    51,644 bytes
    SHA-256 29ff9dc4f0fcead0120da2b3c1993dae2bc6c79106c757cc90fd2a446c4f8bc6
    SOURCE_DATE_EPOCH=0
supported export census:
    root/scalar/scalar.validation/tensor/tensor.validation/table/random/
    random.generator/random.distribution/random.validation
    30/7/3/9/16/3/11/2/6/1
tracked package-file census:
    41 files / 40 Python modules
```

`pyproject.toml` must select exactly:

```text
tensor-core @ git+https://github.com/mbedard44/TensorCore.git@78d0891bf6c0fefbcad4abe09980867c54202a9e
```

Python, Torch, NumPy, Pint, Hatchling, Pyright, and demo-tool versions remain
exactly those of the current TensorDSLab baseline.

TensorCore's final package evidence used CPython `3.14.6`, Torch `2.13.0` CPU,
Hatchling `1.31.0`, and Pyright `1.1.411`: `88` unique tests passed with
exactly two accepted unavailable-CUDA skips; positive typing had zero
diagnostics and the negative fixture had exactly `82` intentional errors.
Two independent checkout wheel builds and one extracted-archive build were
byte-identical. The canonical archive reproduced all `118` tracked blobs,
the same suite result, and the accepted wheel. No fresh CUDA or accelerator
claim follows.

## Atomicity

The following changes form one package-owned atomic target:

```text
exact TensorCore Stage 29 adoption
    +
common axis module rename
    +
QuantityKernel and physical kernels
    +
Config/API clean break
    +
Runtime alignment/materialization
    +
Charge/Pulse execution migration
    +
scientific and RNG rebaseline
    +
profile and demo migration
```

A pin-only intermediate cannot work because Stage 29 retires
`ProbabilityKernel` while TensorDSLab changes every affected consumer. A
new-kernel/old-Config dual surface is also prohibited because it would create
two conflicting public scientific models. No intermediate branch may merge or
publish with adapters, aliases, duplicated Configs, or forwarding modules.

Implementation may develop internally in the easy-to-hard order described
below, but Candidate 1 must be one coherent final package state.

## TensorCore-Owned Contract

The exact published TensorCore contract is linked and frozen above. Its
consumer boundary is:

### `OffsetAxis`

One concrete final non-generic semantic value:

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
```

It owns exact non-boolean integer coordinates, preserved order, uniqueness,
coordinate/index conversion, and value equality/hash over exact class,
`relative_to` class identity, and complete offsets. It owns no units,
displacement, anchor, causality, or finite-boundary policy.

There is no `SampleOffsetAxis`, microcell-specific offset class, target-role
type parameter, abstract leaf family, or compatibility alias.

### Literal `TensorKernel`

```python
class TensorKernel[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](ABC):
    tensor: torch.Tensor
    conditioning_axes: ConditioningAxesT
    operation_axes: OperationAxesT
```

The tensor dimensions are exactly:

```text
(*conditioning_axes, *operation_axes)
```

The generic root owns rank/shape, a same-device contiguous defensive snapshot,
identity equality, explicit unhashability, complete row-major structural
identity, and derived conditioning/operation shapes and counts.

The exact uniqueness law is:

- conditioning-axis exact concrete types are unique only within
  `conditioning_axes`;
- operation target roles are unique within `operation_axes`;
- `OffsetAxis` contributes `relative_to` as its operation target role;
- another operation axis contributes its exact concrete type;
- repeated exact `OffsetAxis` classes are therefore permitted when they target
  different roles;
- no combined exact-type uniqueness applies across both axis groups; and
- a conditioning role may equal one operation target role.

The generic root owns no physical value admission or Runtime effects.

### Role resolution

```python
def require_kernel_dimensions(
    field: TensorField,
    kernel: TensorKernel[Any, Any],
) -> tuple[int, ...]: ...
```

The function resolves conditioning axes by exact type, `OffsetAxis` operation
axes by `relative_to`, and other operation axes by exact type. The output
follows complete kernel-axis order. TensorCore does not permute, select,
broadcast, expand, or decide scientific legality.

### Direct multinomial law

The supported constructor is:

```python
MultinomialDistribution(
    counts=counts,
    probabilities=probabilities,
    completion_probability=completion,
)
```

The Distribution privately owns its defensive float64 probability snapshot,
finite elementwise `[0, 1]` admission, backend-independent stable total,
reverse suffix, completed-law validation, allocation preflight, and ordered
conditional-Binomial execution.

For nonempty accelerator-backed law state, construction permits exactly one
ordered probability-value extraction plus one bounded combined
law-status/partition summary after the stable total is available. Draw performs
no probability-value host observation.

There is no public or hidden `MultinomialParameters`,
`MultinomialPreparation`, probability cache, factory, trust token, alternate
constructor, or compatibility keyword. One probability scan per Distribution
construction is accepted. No draw-time host probability materialization is
accepted.

The exact published contract preserves:

- address shape equal to represented probability shape;
- category-major contiguous int64 results;
- frozen row-major category identity;
- the independently supplied completion probability;
- fixed completed-law tolerance;
- the internal completion outcome as unaddressed, word-free, and unreturned;
- exact input-count conservation including that internal outcome; and
- accepted Binomial branch/address/word behavior under the exact Stage 29
  package evidence.

### `ProbabilityKernel` retirement

`ProbabilityKernel` is absent from every supported TensorCore facade and
precise path without alias. TensorDSLab must not recreate it locally.

## TensorDSLab Ownership

TensorDSLab owns:

- Pint-aware physical coefficient representation;
- detector/readout semantic axes;
- operation-axis admissibility for each physical kernel;
- canonical units;
- probability-unity and offspring-intensity laws;
- profile axis availability;
- conditioning-coordinate correspondence;
- contextual dimension permutation and broadcasting;
- execution-device materialization;
- kernel-index-to-displacement mapping;
- anchors, causality, finite-window discard, convolution, and scatter;
- scientific keys and address schemas;
- count, rate, and ledger ceilings;
- products, Runtime preparation, producers, and validation; and
- every intentional completed-output rebaseline.

TensorDSLab does not own generic TensorCore tensor-kernel structure,
Distribution algorithms, counter words, or address encoding.

## Target Module Topology

The exact production target is 59 Python modules. This census was reconciled
against the published TensorCore API and the current 61-module TensorDSLab
baseline: eight deletion endpoints and six addition endpoints yield 59.

```text
tensor_dslab/
  __init__.py
  common/
    __init__.py
    axis.py
    kernel.py
    units.py
  readout/
    __init__.py
    collection.py
    config.py
    profiles.py
    simulation.py
    runtime/
      __init__.py
      addresses.py
      keys.py
      prepare.py
      requirements.py
      sampling.py
    photoelectrons/
      ...
    charge/
      __init__.py
      config.py
      field.py
      kernel.py
      runtime/
        __init__.py
        branching.py
        counts.py
        prepare.py
        produce.py
        validate.py
    pure_waveform/
      __init__.py
      config.py
      field.py
      kernel.py
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py
    noise_waveform/
      ... unchanged product shape ...
    analog_waveform/
      ... unchanged product shape ...
    digitized_waveform/
      ... unchanged product shape ...
```

Exact clean breaks:

- `tensor_dslab/common/axes.py` is renamed to
  `tensor_dslab/common/axis.py`;
- the old plural module is deleted without forwarding imports;
- `tensor_dslab/readout/charge/runtime/effects/` is deleted completely;
- cohesive shared count laws move to `charge/runtime/counts.py`;
- fixed-generation branching moves to `charge/runtime/branching.py`;
- product orchestration remains in `charge/runtime/produce.py`;
- no `effect.py`, `kernel_runtime.py`, registry, callback graph, generic
  frontier framework, or placeholder microcell module is added.

## Public `QuantityKernel`

`tensor_dslab/common/kernel.py` owns one abstract representation root:

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
    ) -> None: ...

    @property
    def quantity(self) -> Quantity: ...
```

Construction:

- recognizes a Pint `Quantity`;
- rebuilds it in TensorDSLab's private registry;
- converts to the exact semantic canonical unit;
- admits an exact built-in scalar, NumPy array, or CPU Torch tensor magnitude;
- stores one contiguous defensive CPU `torch.float64` magnitude snapshot;
- rejects gradients, non-strided layout, incompatible units, rank/shape
  mismatches, and leaf-specific value violations;
- exposes a Quantity view over the exact owned magnitude;
- retains no caller Quantity or duplicate tensor payload;
- is identity-equal, explicitly unhashable, and has no tensor-derived repr.

`QuantityKernel` never chooses an execution device or dtype. Public Config and
profile construction are host-side and device-independent.

Runtime preparation explicitly materializes stochastic probability, intensity,
rate, and width operands as float64 on the product device. It transfers and
casts `Pulse` once to the requested waveform floating dtype.

The existing public `quantity(...)` helper retains scalar construction.
`quantities(...)` retains its tuple-vector input and additionally accepts an
arbitrary-rank CPU Torch tensor. That is the first supported hand-construction
path for multidimensional kernel quantities; it does not create a kernel
factory or expose the private Pint registry.

## Physical Kernel Classes

All seven leaves are public, final, directly constructible, frozen, slotted,
identity-equal, unhashable, keyword-only values. No public builder, factory,
loader, parser, registry, `from_*`, callback, or serialization constructor is
added.

### `DarkCountRate`

```python
@final
class DarkCountRate[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
](QuantityKernel[ConditioningAxesT, tuple[()]]): ...
```

- canonical unit `Hz`;
- finite and nonnegative;
- no operation axes;
- arbitrary admitted conditioning axes;
- rank zero is one global rate.

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
): ...
```

- canonical unit dimensionless;
- finite and nonnegative;
- exactly one nonempty `OffsetAxis`;
- `relative_to is SampleAxis`;
- mass over the operation axis equals one per conditioning coordinate within
  the package's accepted `1e-11` complete-law tolerance;
- abstract translation conserves charge;
- finite-window destination discard may still lose output charge;
- no analytic Gaussian constructor, hidden tail, normalization, clipping, or
  residual assignment.

### `DirectCrosstalk`

```python
@final
class DirectCrosstalk[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[OffsetAxis, ...],
](QuantityKernel[ConditioningAxesT, OperationAxesT]): ...
```

- canonical unit dimensionless;
- finite and nonnegative;
- nonempty operation geometry;
- every operation axis is exact `OffsetAxis`;
- operation target roles are unique;
- exactly one operation axis targets `SampleAxis`;
- direct sample offsets are nonnegative;
- `ExampleAxis` is never an operation target;
- each element is expected direct offspring multiplicity landing at that
  operation cell;
- the sum is not required to equal or remain below one.

### `DelayedCrosstalk`

Same structural intensity law as `DirectCrosstalk`, but a distinct public
semantic class and RNG role. Its operation geometry must contain exactly one
sample-targeting axis, and that axis admits only positive sample offsets in the
first product. `ExampleAxis` is never an operation target.

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
): ...
```

- canonical unit dimensionless;
- finite and nonnegative;
- exactly one nonempty sample-targeting operation axis;
- all offsets are positive;
- each cell is expected full-charge afterpulse offspring multiplicity;
- no probability-sum ceiling;
- multiple afterpulses per parent per generation are admitted;
- no occurrence Binomial, conditional delay Multinomial, or recovery weight.

### `SmearingWidth`

- canonical unit dimensionless;
- finite and nonnegative;
- no operation axes;
- value is Gaussian standard deviation relative to one PE response, not
  variance.

### `Pulse`

- canonical unit `mV` per one PE-equivalent input charge;
- finite signed values;
- exactly one nonempty sample-targeting operation axis;
- offsets are nonnegative and unique;
- deterministic finite convolution;
- no hidden sample-width factor;
- supplied coefficients already contain final polarity;
- DS20k negative polarity is applied exactly once by the profile/direct caller,
  never again by Runtime.

## Config Clean Break

The exact new Config sketches are:

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

The fail-closed generation relationship is:

```text
all three branching kernels are None
    iff
correlated_avalanche_generations == 0
```

Equivalently:

- if all three branching kernels are `None`, generations must be zero;
- if any branching kernel is present, generations must be positive.

The old public Config classes are removed from modules, facades, root exports,
typing surfaces, tests, examples, and documentation without aliases:

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

No old/new union, deprecation wrapper, or compatibility property remains.

## Fixed Config-To-Execution Mapping

Public Config state contains physical values, never executable Distribution
types, factories, callbacks, or law selectors.

```text
TimingJitter       -> MultinomialDistribution
DarkCountRate      -> PoissonDistribution
DirectCrosstalk    -> PoissonDistribution
DelayedCrosstalk   -> PoissonDistribution
Afterpulse         -> PoissonDistribution
SmearingWidth      -> GaussianDistribution
Pulse              -> deterministic convolution
```

TensorDSLab owns this mapping. Users cannot substitute a Distribution class
into a Config.

## Profile Geometry

The exact profile signature becomes:

```python
def ds20k_veto(
    *,
    sample_axis: SampleAxis,
    channel_axis: ChannelAxis | None = None,
    example_axis: ExampleAxis | None = None,
) -> ReadoutConfig: ...
```

`sample_axis` is required. Channel/example axes are optional declarations of
available conditioning geometry, not requirements that every kernel vary over
those axes.

Every returned kernel must obey:

```text
set(conditioning-axis roles)
    <=
set(supplied profile-axis roles)
```

Therefore:

- a global `DarkCountRate` is valid even when channel/example axes are
  supplied;
- a channel-conditioned kernel is invalid when `channel_axis` is omitted;
- an example-conditioned kernel is invalid when `example_axis` is omitted;
- the profile cannot synthesize hidden conditioning axes;
- returned kernels retain the exact supplied axis values where conditioned;
- every call returns a fresh complete Config/kernel tree.

The provisional `ds20k_veto()` profile retains its accepted demonstration
numbers while expressing them through the new physical values:

- `ChargeConfig` has `correlated_avalanche_generations=0`, one global
  `DarkCountRate` of exactly `100 kHz`, and no timing-jitter, crosstalk,
  afterpulse, or smearing kernel;
- `Pulse` is the chronological sampled form of the current Veto PDU law using
  exact binary64 parameters `232.89 ns`, `507.72 ns`, `-81.92 ns`,
  `147.28 ns`, `-176.50 ns`, `45.69 ns`, support `2020.27 ns`, and a final
  negative peak of `-14.5912372 mV` per PE-equivalent charge;
- pulse sampling uses the supplied `SampleAxis` period, the first excluded
  left edge at the support time, the existing full-support maximum-absolute
  normalization, and then caps represented offsets to the supplied sample
  count;
- the pulse operation axis is an `OffsetAxis` relative to `SampleAxis` with
  `offsets=tuple(range(coefficient_count))`;
- the accepted nine-edge/eight-band IV-like PSD, `250 MHz` stop, empty
  `AnalogWaveformConfig`, and `16`-bit `[-3900, 100] mV` digitizer with
  `3.5218 dB` analog gain remain exact;
- supplied `channel_axis` and `example_axis` are availability declarations
  only in this profile version; none of its returned kernels condition on
  them; and
- two calls share no Config, kernel, Quantity, or axis-container node other
  than the caller-supplied immutable semantic axis values.

The profile may use a private module-local evaluator for its fixed Veto pulse
law. That is profile-owned construction, not a public analytic kernel builder
or a second pulse Config surface.

No profile loader, named calibration registry, file format, or Silex profile is
added. Silex remains a future demonstrated consumer.

## Runtime Compilation Boundary

Runtime is compiled product execution state, not a validated reflection of
Config. There is no one-to-one Config/Runtime class rule.

The product records remain:

```text
ChargeRuntime
PureWaveformRuntime
NoiseWaveformRuntime
AnalogWaveformRuntime
DigitizedWaveformRuntime
SamplingRuntime
```

A simple mechanism is represented directly by a scalar/tensor field in its
product Runtime. A nested mechanism Runtime is added only when several
coherent derived values have one useful invariant.

The intended Charge granularity is:

```python
@final
@dataclass(frozen=True, slots=True)
class TimingJitterRuntime:
    probabilities: torch.Tensor
    conditioning_dimensions: tuple[int, ...]
    sample_offsets: tuple[int, ...]


@final
@dataclass(frozen=True, slots=True)
class BranchingRuntime:
    intensities: torch.Tensor
    conditioning_dimensions: tuple[int, ...]
    target_dimensions: tuple[int, ...]
    offsets: tuple[tuple[int, ...], ...]


@final
@dataclass(frozen=True, slots=True)
class ChargeRuntime:
    sampling: SamplingRuntime
    floating_dtype: torch.dtype
    correlated_avalanche_generations: int
    dark_count_mean: torch.Tensor | None
    timing_jitter: TimingJitterRuntime | None
    direct_crosstalk: BranchingRuntime | None
    delayed_crosstalk: BranchingRuntime | None
    afterpulse: BranchingRuntime | None
    smearing_width: torch.Tensor | None
```

The private names and exact field order remain subject to final source-level
feasibility, but the ownership/granularity is frozen:

- dark-count mean and smearing width do not get wrapper Runtime classes;
- timing jitter has a distinct probability-law record;
- all three Poisson branching mechanisms may share one representation-only
  private Runtime class;
- that class owns no effect enum, callback, Distribution, RNG key, or mutable
  state;
- ChargeRuntime stores no Config, Quantity, QuantityKernel, product, execution
  method, or cache.

`PureWaveformRuntime` directly stores the materialized pulse coefficients,
sample offsets, conditioning alignment facts, and existing sampling facts; it
does not require a nested `PulseRuntime`.

Preparation:

- validates public Config ingress once through whole-request preparation;
- resolves kernel roles with TensorCore;
- enforces TensorDSLab coordinate correspondence and profile legality;
- records a stable target-order permutation;
- inserts broadcast views/singleton execution dimensions without expanding
  public kernel storage;
- materializes stochastic operands as float64 on the product device;
- materializes Pulse once on the product device and waveform dtype;
- checks allocation, count, Poisson mean, generation, and ledger ceilings;
- emits Pint-free immutable Runtime state.

Producers:

- receive typed products, Runtime, and `CounterRng`;
- bind dynamic counts/means;
- construct complete TensorCore Distribution values;
- draw through exact package-owned addresses;
- map offsets to finite destinations;
- discard out-of-window destinations where specified;
- publish fresh validated products.

## Exact Charge Execution Order

```text
Photoelectrons
    -> dark counts
    -> timing jitter
    -> N correlated-avalanche rounds
    -> charge smearing
    -> Charge
```

Consequences:

- dark counts are added before timing jitter;
- dark counts therefore receive timing jitter when enabled;
- dark counts can seed correlated avalanches;
- timing jitter runs exactly once over the combined primary/dark frontier;
- smearing runs after all retained branching;
- `N` is the exact number of descendant-producing rounds;
- `N == 0` performs no correlated draw;
- round zero consumes the post-jitter seed and produces generation one;
- within one round, direct, delayed, and afterpulse mechanisms independently
  consume the same immutable frontier;
- their retained children are pooled into the next frontier;
- children never cascade within the round that created them;
- the next round, if any, consumes only that pooled child frontier.

The aggregate Charge before smearing is the sum of the post-jitter seed and all
retained descendant frontiers. Every retained avalanche contributes exactly
one PE-equivalent to S1 and exactly one to the pre-smearing square ledger.

## Collapsed Poisson Branching

For direct crosstalk, delayed crosstalk, and afterpulse, TensorDSLab
deterministically maps the selected intensity kernel into one retained
destination mean tensor:

```text
lambda[destination]
    =
sum_source(
    frontier[source]
    * intensity[condition(source), destination - source]
)
```

Only in-window destinations contribute. The producer then constructs one
tensor-valued `PoissonDistribution(mean=lambda)` and performs one atomic draw
for the complete destination tensor.

This is the selected exact Poisson thinning/superposition law. It avoids
per-source total draws, Multinomial category materialization, scatter of
source-by-kernel allocations, and random draws for out-of-window destinations.

The stage freezes:

- float64 deterministic rate construction;
- checked additions/multiplications before draw;
- each final destination mean within TensorCore's exact `1e8` ceiling;
- output counts within the TensorDSLab `2**53 - 1` count ceiling;
- finite-window restriction by retained rates, not overflow draws;
- no returned tail/overflow count;
- no per-avalanche expansion;
- no normal approximation, `torch.poisson`, clipping, or normalization.

## Timing-Jitter Allocation

Timing jitter selects one complete finite probability slab for each applicable
conditioning coordinate and constructs:

```python
distribution = MultinomialDistribution(
    counts=frontier_counts,
    probabilities=probabilities,
    completion_probability=0.0,
)
```

The complete abstract law has unity represented mass. Allocations are drawn in
frozen operation-axis row-major order. TensorDSLab then maps each sample offset
to its destination and discards out-of-window allocations.

The producer must prove:

- exact count conservation inside the complete Multinomial law;
- unity per conditioning coordinate;
- no probability normalization or residual repair;
- separate finite-window loss;
- stable category/address identity;
- no source renumbering under accepted deterministic chunking.

The current analytic Gaussian jitter preparation and destination-by-destination
conditional-binomial implementation are intentionally retired. No completed
output continuity is claimed across this scientific representation change.

## Dark Counts And Smearing

Dark-count Runtime computes a dimensionless mean from the aligned rate and
sample exposure. The producer uses one tensor-valued `PoissonDistribution`.

Smearing constructs one tensor-valued `GaussianDistribution` with:

```text
mean = aggregate avalanche count
standard deviation = aligned relative width * sqrt(aggregate count)
```

because all retained avalanches carry unit pre-smearing charge in this stage.
Zero width retains TensorCore's documented deterministic word-free path.

No global Torch RNG state changes.

## Pulse Execution

`PureWaveformConfig` supplies one complete sampled signed `Pulse`.

Runtime:

- validates one sample-targeting offset axis;
- aligns any admitted conditioning axes;
- materializes the magnitude once on the product device;
- casts once to the requested waveform floating dtype;
- stores no Pint state.

The producer performs deterministic discrete convolution using the exact
offsets. Contributions outside the finite `SampleAxis` are discarded. No
analytic pulse evaluation, peak renormalization, polarity flip, or hidden
sampling-width factor occurs at execution.

## Conditioning, Broadcasting, And Alignment

For each kernel, preparation:

1. calls TensorCore's exact generic role resolver;
2. separates conditioning and operation dimensions;
3. validates every conditioning role is available in the target product;
4. requires equal coordinate cardinality;
5. requires exact one-to-one coordinate correspondence;
6. computes the target-order permutation;
7. rejects missing, extra, or duplicate coordinates;
8. permits omitted target axes as global broadcast;
9. inserts only private execution views/singleton dimensions;
10. preserves operation-axis order and offsets exactly.

Supplying `channel_axis` or `example_axis` to a profile makes that role
available; it does not force dependence. The converse is prohibited: a kernel
cannot condition on a role absent from the profile/target geometry.

No public kernel tensor is expanded to product shape merely to express
broadcasting.

## RNG Rebaseline

The namespace remains:

```text
0x54445331
```

Because TensorDSLab is predeployment and this is a deliberate complete
stochastic rebaseline, the target uses one compact table and does not preserve
numeric holes for retired streams:

```text
0x0000_0001  white noise
0x0000_0002  PSD noise
0x0000_0003  dark counts
0x0000_0004  timing jitter
0x0000_0005  direct crosstalk
0x0000_0006  delayed crosstalk
0x0000_0007  afterpulse
0x0000_0008  charge smearing
```

Address schemas:

```text
white noise:
    root shape=(), quantum=0

PSD noise:
    root shape=(), quantum=0

dark counts:
    root shape=(), quantum=0

timing jitter:
    root shape=operation probability shape, quantum=0

direct/delayed/afterpulse:
    root shape=(correlated_avalanche_generations,), quantum=0
    select(generation_index)

charge smearing:
    root shape=(), quantum=0
```

Each mechanism has its own key. Branching destination identity is the
unrenumbered destination `RngElements`; generation identity is the selected
address-domain coordinate. Timing category identity is the exact operation
row-major index.

The candidate freezes a new exact replay baseline. It does not claim completed
output continuity with Maintenance 11 and does not permanently reserve retired
occurrence/delay/overflow identifiers. It must still prove:

- collision freedom;
- exact replay;
- unchanged global Torch RNG state;
- no traversal/chunk renumbering;
- stable same-candidate words/results for every exact operation address;
- explicit relationship between Runtime chunking and retained root capacity.

## Public API Target

The exact target facade counts are:

```text
tensor_dslab root:
    31
tensor_dslab.common:
    6
tensor_dslab.readout:
    25
tensor_dslab.readout.charge:
    8
tensor_dslab.readout.pure_waveform:
    3
```

The package-root target, in order, is:

```python
__all__ = (
    "Afterpulse",
    "AnalogSaturationConfig",
    "AnalogWaveform",
    "AnalogWaveformConfig",
    "ChannelAxis",
    "Charge",
    "ChargeConfig",
    "DarkCountRate",
    "DelayedCrosstalk",
    "DigitizedWaveform",
    "DigitizedWaveformConfig",
    "DirectCrosstalk",
    "ExampleAxis",
    "NoiseWaveform",
    "NoiseWaveformConfig",
    "Photoelectrons",
    "PsdNoiseConfig",
    "Pulse",
    "PureWaveform",
    "PureWaveformConfig",
    "QuantityKernel",
    "ReadoutCollection",
    "ReadoutConfig",
    "SampleAxis",
    "SmearingWidth",
    "TimingJitter",
    "WhiteNoiseConfig",
    "ZeroNoiseConfig",
    "quantities",
    "quantity",
    "simulate_readout",
)
```

`tensor_dslab.common` exports exactly:

```python
(
    "ChannelAxis",
    "ExampleAxis",
    "QuantityKernel",
    "SampleAxis",
    "quantities",
    "quantity",
)
```

`tensor_dslab.readout` exports exactly:

```python
(
    "Afterpulse",
    "AnalogSaturationConfig",
    "AnalogWaveform",
    "AnalogWaveformConfig",
    "Charge",
    "ChargeConfig",
    "DarkCountRate",
    "DelayedCrosstalk",
    "DigitizedWaveform",
    "DigitizedWaveformConfig",
    "DirectCrosstalk",
    "NoiseWaveform",
    "NoiseWaveformConfig",
    "Photoelectrons",
    "PsdNoiseConfig",
    "Pulse",
    "PureWaveform",
    "PureWaveformConfig",
    "ReadoutCollection",
    "ReadoutConfig",
    "SmearingWidth",
    "TimingJitter",
    "WhiteNoiseConfig",
    "ZeroNoiseConfig",
    "simulate_readout",
)
```

`tensor_dslab.readout.charge` exports exactly:

```python
(
    "Afterpulse",
    "Charge",
    "ChargeConfig",
    "DarkCountRate",
    "DelayedCrosstalk",
    "DirectCrosstalk",
    "SmearingWidth",
    "TimingJitter",
)
```

`tensor_dslab.readout.pure_waveform` exports exactly:

```python
(
    "Pulse",
    "PureWaveform",
    "PureWaveformConfig",
)
```

These counts were reconciled against exact published TensorCore `0.21.0` and
the current TensorDSLab source. Any changed count or order requires an explicit
Design amendment, not Implementation discretion.

Runtime records, actions, requirements, address factories, and branching/count
mechanics remain unsupported precise implementation details. Physical kernel
classes and `QuantityKernel` receive their own truthful public docstrings.

## Demo Contract

Both installed-wheel demonstrations remain CPU-runnable and are migrated to
the new APIs:

- `demos/readout.py`;
- `demos/readout.ipynb`; and
- `demos/random.ipynb`.

The readout demo:

- constructs `SampleAxis` first;
- calls `ds20k_veto(sample_axis=...)`;
- demonstrates direct physical-kernel construction where useful;
- preserves the accepted 10,000 ns/four-deposit/IV-like digitizer display;
- stores only current Config/kernel public names;
- executes from the installed wheel without project-root shadowing.

The random demo continues to explain actual package-owned addresses and
Threefry through one delayed-crosstalk example, but updates its science to the
collapsed destination-rate Poisson law. It must not teach a total-first
Poisson-plus-Multinomial crosstalk path or retired stream identifiers.

Notebook stored outputs are refreshed only after real execution in the exact
final environment. They must contain no timestamp, private path, token,
environment-specific prefix, or error output.

## Exact Changed-Path Allowlist

The final allowlist contains exactly `81` unique endpoints: `35` dependency
and production paths, three demos, `27` tests/typing paths, and `16` current
documents. Rename/delete pairs count as separate endpoints. No path outside
this final allowlist is authorized.

### Dependency and production

```text
pyproject.toml
tensor_dslab/__init__.py
tensor_dslab/common/__init__.py
tensor_dslab/common/axes.py
tensor_dslab/common/axis.py
tensor_dslab/common/kernel.py
tensor_dslab/common/units.py
tensor_dslab/readout/__init__.py
tensor_dslab/readout/profiles.py
tensor_dslab/readout/runtime/addresses.py
tensor_dslab/readout/runtime/keys.py
tensor_dslab/readout/runtime/prepare.py
tensor_dslab/readout/runtime/requirements.py
tensor_dslab/readout/runtime/sampling.py
tensor_dslab/readout/charge/__init__.py
tensor_dslab/readout/charge/config.py
tensor_dslab/readout/charge/kernel.py
tensor_dslab/readout/charge/runtime/effects/__init__.py
tensor_dslab/readout/charge/runtime/effects/correlated_avalanches.py
tensor_dslab/readout/charge/runtime/effects/counts.py
tensor_dslab/readout/charge/runtime/effects/dark_counts.py
tensor_dslab/readout/charge/runtime/effects/delays.py
tensor_dslab/readout/charge/runtime/effects/smearing.py
tensor_dslab/readout/charge/runtime/effects/timing_jitter.py
tensor_dslab/readout/charge/runtime/branching.py
tensor_dslab/readout/charge/runtime/counts.py
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/charge/runtime/produce.py
tensor_dslab/readout/charge/runtime/validate.py
tensor_dslab/readout/pure_waveform/__init__.py
tensor_dslab/readout/pure_waveform/config.py
tensor_dslab/readout/pure_waveform/kernel.py
tensor_dslab/readout/pure_waveform/runtime/prepare.py
tensor_dslab/readout/pure_waveform/runtime/produce.py
tensor_dslab/readout/pure_waveform/runtime/validate.py
```

`common/axes.py` and every `charge/runtime/effects/*` path above are deletion
endpoints. `common/axis.py`, both product kernel modules, and flat
`branching.py`/`counts.py` are addition endpoints.

### Demos

```text
demos/readout.py
demos/readout.ipynb
demos/random.ipynb
```

`create_environment.sh` remains unchanged unless the exact TensorCore
publication changes a direct environment command rather than only
`pyproject.toml` resolution. Such a change requires Design amendment.

### Tests and typing

```text
tests/readout_fixtures.py
tests/test_charge_correlated_avalanches.py
tests/test_charge_delay_preparation.py
tests/test_charge_product.py
tests/test_charge_timing_jitter.py
tests/test_deterministic_waveform_products.py
tests/test_package_contracts.py
tests/test_pint_physical_configuration.py
tests/test_random_demo.py
tests/test_readout_axes_and_sampling.py
tests/test_readout_configs.py
tests/test_readout_profiles_and_demos.py
tests/test_readout_product_types.py
tests/test_readout_simulation.py
tests/test_rng_ownership_migration.py
tests/test_runtime_action_ownership.py
tests/test_tensorcore_0_19_adoption.py
tests/test_tensorcore_0_21_adoption.py
tests/test_kernel_geometry_and_quantity.py
tests/typing/maintenance_11_tensorcore_0_19_addressed_distributions.py
tests/typing/maintenance_12_tensorcore_0_21_kernel_geometry_quantity_refactor.py
tests/typing/maintenance_4_runtime_action_ownership.py
tests/typing/maintenance_6_pint_physical_configuration_boundary.py
tests/typing/maintenance_9_ds20k_veto_profile_and_public_readout_demos.py
tests/typing/stage_3_semantic_leaf_contracts.py
tests/typing/stage_4_deterministic_waveform_products.py
tests/typing/stage_7_public_readout_orchestration.py
```

`test_charge_delay_preparation.py`, the 0.19 adoption test, and the Maintenance
11 typing fixture are deletion/rename endpoints. The listed historical typing
fixtures are allowlisted because their live import and Config contracts change;
unlisted historical fixtures remain protected.

### Current documentation

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/api.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/design.md
docs/decisions.md
docs/overview.md
docs/parity.md
docs/physics/correlated_avalanches.md
docs/validation.md
docs/implementation/index.md
docs/implementation/proposed_kernel_geometry_and_quantity_architecture.md
docs/implementation/maintenance_12_tensorcore_0_21_kernel_geometry_quantity_refactor.md
```

Every lifecycle statement must be branch/main-neutral or self-effecting. No
candidate-specific “pending Validation/Review” sentence may become stale when
unchanged bytes advance.

### Protected paths

Unless this work order explicitly says otherwise, all other paths are
protected, including:

```text
LICENSE
.github/
create_environment.sh
tensor_dslab/py.typed
tensor_dslab/readout/config.py
tensor_dslab/readout/collection.py
tensor_dslab/readout/simulation.py
tensor_dslab/readout/photoelectrons/
tensor_dslab/readout/noise_waveform/
tensor_dslab/readout/analog_waveform/
tensor_dslab/readout/digitized_waveform/
docs/implementation/<all historical closed work orders>
```

If implementation requires a protected byte, it stops and returns the exact
contradiction to Design.

## Implementation Sequence

One Implementation role may use this internal order after exact dispatch:

1. verify authority/branch/dependency/disjointness gates;
2. adopt exact TensorCore publication and update dependency probes;
3. rename `common/axis.py`, add `QuantityKernel`, and implement physical
   kernel classes;
4. replace Configs/facades and update profile construction;
5. compile Runtime alignment and device materialization;
6. migrate DarkCountRate and SmearingWidth;
7. migrate Pulse and deterministic convolution;
8. migrate direct/delayed/afterpulse collapsed Poisson branching;
9. migrate TimingJitter direct Multinomial allocation;
10. rebase keys/addresses and exact stochastic fixtures;
11. update demos/notebooks;
12. synchronize current docs;
13. run focused mutation-resistant gates;
14. run final complete source/archive/static/artifact/demo evidence once on
    final candidate bytes;
15. commit one immutable candidate and dispatch Validation.

Intermediate incomplete commits may exist only on the private implementation
branch and are not candidate dispatches. Candidate 1 must be coherent.

## Evidence Strategy

This stage retains independent evidence while avoiding avoidable duplicated
release-certificate work on every narrow correction.

- Implementation runs fast focused tests throughout development.
- Mutation-resistant oracles must exist before candidate dispatch for every
  deliberate scientific law.
- The complete environment/dependency reconstruction, source/archive matrix,
  artifacts, isolated installs, and notebooks are final-candidate gates.
- A test-only correction after a proven byte-identical production candidate
  reruns affected focused/static evidence plus any gate whose committed bytes
  changed; it need not reconstruct unrelated dependencies unless the work
  order's risk boundary requires it.
- Validation independently reconstructs the exact dependency and complete
  final candidate.
- Review audits production and proof quality and runs proportionate independent
  gates, including mutants for the high-risk scientific changes.
- Any production/dependency/scientific correction invalidates carried
  executable evidence and requires the complete applicable matrix.

No role may use this proportionality rule to skip a gate affected by changed
bytes.

## Required Functional Evidence

### TensorCore dependency

- exact commit/tree/parent/version;
- exact package payload equality between source and canonical archive;
- exact supported facade/export census;
- exact package topology;
- TensorCore suite on source/archive;
- absence of retired `ProbabilityKernel`;
- absence of multinomial parameters/preparation aliases;
- direct `probabilities` constructor signature;
- `OffsetAxis`, literal `TensorKernel`, and role resolution;
- dependency import isolation.

### Quantity and public API

- every public production module has a module docstring;
- every supported public class/function has its own truthful nonempty docstring;
- exact constructor signatures and annotations resolve under Python 3.14;
- own-class docstring evidence uses `class.__dict__.get("__doc__")`;
- canonical CPU float64 snapshots;
- exact scalar, NumPy, tuple-vector, and arbitrary-rank CPU tensor quantity
  inputs;
- caller mutation isolation;
- unit conversion and incompatible-unit rejection;
- exact facade order/counts;
- exact `ds20k_veto(...)` signature, fixed value signature, availability
  ceiling, and complete fresh-node tree;
- hand-built/profile Config equivalence in the readout demonstration;
- retired Config/module imports fail;
- no alias or forwarding module.

### Alignment

- global scalar kernels;
- channel/example conditioned kernels;
- combined conditioning;
- coordinate permutation;
- equal length/different coordinates rejection;
- missing/extra/duplicate coordinate rejection;
- omitted profile-axis rejection;
- supplied-but-unused axis admission;
- operation-target uniqueness with repeated concrete `OffsetAxis`;
- no public expanded coefficient storage;
- exact device/dtype materialization.

### Scientific laws

- independent analytic/direct oracles for each destination mean;
- Poisson mean/variance for dark/direct/delayed/afterpulse;
- direct/delayed/afterpulse same-frontier independence;
- generation count and no same-round cascade;
- full-charge afterpulse S1/S2;
- no recovery contribution or API;
- timing unity and finite-window discard;
- Gaussian smearing mean/variance/zero-width behavior;
- Pulse discrete convolution/polarity/boundary behavior;
- count/ledger/Poisson ceilings and pre-draw failure ordering.

At least one plausible production mutant per high-risk law must be killed,
including:

- afterpulse Poisson mean multiplied by `0.5`;
- branching child frontier fed back within the same round;
- timing probabilities normalized after construction;
- out-of-window branching included in destination rates;
- pulse polarity flipped twice;
- conditioning coordinate permutation omitted.

### RNG

- exact compact key table;
- exact address schemas;
- stable generation identity;
- mechanism disjointness;
- no global RNG effect;
- same-candidate replay;
- accepted deterministic chunk/traversal invariance;
- no retired afterpulse occurrence/delay address use;
- no overflow draw/product;
- explicit expected divergence from Maintenance 11 output fixtures.

### Products and orchestration

- source product/axes identity where required;
- fresh Charge/PureWaveform products;
- requested retention and collection behavior;
- complete `simulate_readout(...)` chain;
- no Config/Pint/kernel state in Runtime;
- no Runtime/Distribution state in public Config;
- unchanged downstream analog/digitized contracts except changed scientific
  inputs.

## Static And Artifact Evidence

The final candidate must pass:

- complete source-form test discovery;
- extracted canonical TensorCore archive form;
- Pyright `1.1.411` with Python `3.14`;
- exact negative typing fixture diagnostics with no incidental findings;
- build with Hatchling `1.31.0`;
- wheel and sdist payload equality to candidate source;
- core-only isolated wheel import without demo dependencies;
- demos-extra isolated wheel execution;
- CPU script and both notebooks from installed artifacts;
- exact notebook cell/output/hash/privacy inventory;
- relative Markdown link and balanced-fence checks;
- API/README executable code fences;
- protected-byte and allowlist checks;
- no bytecode, cache, build, dist, or egg-info residue;
- clean final candidate checkout.

No fresh integrated CUDA evidence is required. Every disposition must state
that the package is CPU-qualified only and makes no current accelerator claim.
The complete coordinated but package-owned CUDA matrix remains release-blocking
at the exact mutually adopted TensorCore/TensorDSLab `1.0.0`
release-candidate pairing.

## Loop And Candidate Policy

The exact active route uses these finite candidate/return budgets:

```text
Implementation -> Validation:
    at most three immutable candidate submissions

Validation -> Implementation:
    at most three production/test correction returns

Validation clear -> independent Review:
    exact unchanged candidate

Review finding:
    return through Design unless the frozen work order explicitly authorizes
    a bounded correction route
```

Design-owned documentation contradictions do not authorize Implementation to
edit protected Design records. Candidate exhaustion returns to Design; it does
not infer a fourth slot.

Review receives the exact Validation-cleared commit/tree and must not merge
until final same-byte Design approval. Merge is `git merge --ff-only` from the
governed clean main. No push follows from local closeout.

## Publication And Compatibility

This stage is a predeployment clean break. It makes no backward-compatibility
promise for:

- retired Config classes;
- `common.axes`;
- private Runtime/effects paths;
- RNG stream numbers;
- address schemas;
- completed stochastic outputs;
- current afterpulse occurrence/recovery law;
- current analytic pulse models;
- raw reflection, pickle, or qualified-name provenance.

It must preserve unchanged product field/collection contracts and explicitly
rebaseline every changed scientific/RNG fixture.

Local merge/closeout does not authorize push. Any later ordinary GitHub
publication requires its own narrow Design-owned lifecycle authority.

## Non-Goals

Maintenance 12 does not add:

- a Silex profile;
- microcell product axes;
- TensorG4DS integration;
- native G4DS ingestion;
- calibration files or loaders;
- kernel builders/factories;
- analytic delay/pulse constructors;
- recovery weighting;
- per-cell recovery state;
- a generic effect/graph/frontier framework;
- a TensorDSLab Distribution;
- public Runtime or address helpers;
- mutable RNG state;
- overflow/tail products;
- IO/cache schemas;
- TensorML integration;
- CUDA or performance evidence;
- deployment, calibration, broad compatibility, or 1.0 readiness claims.

## Design Authority

TensorDSLab Design owns this package work order. TensorCore Design owns Stage
29 and its published dependency bytes. Agreement between the two Design
threads is required for the cross-package boundary, but neither package role
may edit, dispatch, validate, review, merge, or publish the other package.

The accepted architecture and exact TensorCore dependency are frozen for
execution. The user authorizes TensorDSLab Implementation under the exact
containing work-order bytes. Implementation may edit only allowlisted
production, dependency, test, demo, and synchronized current-document paths
and may dispatch one coherent committed candidate to persistent Validation.
Validation may return findings or dispatch the exact unchanged clear candidate
to independent Review. Review remains read-only until Validation dispatch and
must obtain final same-byte Design approval before any local fast-forward.

No merge, push, CUDA action, compatibility claim, release, deployment, or
publication is authorized by this work order.
