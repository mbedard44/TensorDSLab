# Maintenance 15 Spec-Composed Products And Application Boundary

Status: **Architecture selected and synchronized with the
[executable work order](maintenance_15_execution_work_order.md); exact
published TensorCore `0.22.0` commit
`19bfae35fbc773b55cac7bcd659dda57c4dee6d6`, tree
`53aa10520a50c0714e79c685d814cbae1b6f7740`, accepted as the Maintenance 15
dependency target; execution state follows the work order's self-effecting
exact-byte route**.

Stable key:
`TensorDSLab/maintenance-15-spec-composed-products-and-application-boundary`

## Purpose

Replace TensorDSLab's current product graph, Runtime records, axis inheritance
model, and collaboration-specific readout surface with a smaller
parts-bin architecture built from:

- generic coordinate representations;
- semantic tensor axes composed with those representations;
- immutable field and kernel specifications;
- tensor values that carry one exact specification;
- TensorDSLab quantity-aware specifications consumed directly by Product
  Fields and physical Kernels;
- independent Product transforms with Product-specific Config punchcards; and
- collaboration-owned applications that assemble Products into workflows.

The selected architecture treats each Product as a complete scientific
transformation in its own right:

```text
sources + Product Config -> one Product
```

It does not treat a Product as a permanently assigned stage in a package-owned
pipeline. `Charge` means the result of the Charge transformation configured by
one `ChargeConfig`; it does not mean "the object that always comes after
Photoelectrons and before PureWaveform." An application may choose:

```text
Photoelectrons -> Charge
```

or:

```text
Axioelectrons + Photoelectrons -> Charge
```

or another scientifically valid source assembly without requiring
TensorDSLab's reusable Product package to own that application graph.

The same rule applies to every Product. A Product owns:

- the exact source relationship it accepts;
- its Config contract;
- its preparation;
- its numerical transformation;
- its result validation; and
- its semantic result type.

An application owns:

- which Products are used;
- their order and dependency graph;
- values of TensorDSLab's shared semantic axes and genuinely
  collaboration-specific semantic axis classes;
- profiles and defaults;
- whole-workflow collections;
- retained intermediate products;
- user-facing commands and demonstrations; and
- application-level IO and persistence policy.

This record is intentionally detailed because it defines a breaking
pre-deployment architecture boundary shared with TensorCore. It selects the
architecture and cross-package ownership needed for future bounded work
orders. It does not edit production, adopt an unpublished dependency, dispatch
Implementation, authorize a collaboration application repository, run CUDA,
or claim compatibility, publication, deployment, calibration, or production
readiness.

## Governing Standards

This architecture follows:

- [CONTRIBUTING](../../CONTRIBUTING.md) for semantic representation,
  coordinates versus indices, public typing, unit ownership, dtype and device
  policy, preparation, validation, relationships, artifacts, tests, and
  cross-package changes;
- [Overview](../overview.md) for the currently operative package boundary and
  intended ecosystem data flow;
- [Tensor Architecture](../architecture/tensors.md) for the current
  TensorCore-backed field and axis baseline that the future implementation
  will replace;
- [Readout Architecture](../architecture/readout.md) for the currently
  operative Maintenance 12 product graph and Maintenance 13 preparation
  mechanics;
- [Validation](../validation.md) for deterministic, statistical, typing,
  artifact, device, and scientific evidence;
- [Parity](../parity.md) for every preserved law, deliberate stochastic
  rebaseline, or accepted divergence;
- [Maintenance 12](maintenance_12_tensorcore_0_21_kernel_geometry_quantity_refactor.md)
  for the current physical-kernel and stochastic-law baseline;
- [Maintenance 13](maintenance_13_runtime_hygiene_and_environment_reproducibility.md)
  for the current shared kernel-alignment boundary; and
- [Maintenance 14](maintenance_14_test_suite_curation.md) for the current test
  organization and evidence baseline.

The living architecture pages describe the current package until future
implementation work is separately cleared and merged. This Design record
describes the selected future boundary. A future implementation must update
the living pages atomically rather than treating this record as a substitute
for current-reference documentation.

## Exact Design Baseline

The selected architecture starts from exact locally closed Maintenance 14:

```text
TensorDSLab local main:
    856df702c124365c929bf993851a51fb8ff3c245
TensorDSLab tree:
    9e5ff69920699dc522980b164eaf1073116914c6
exact parent / immutable Maintenance 14 Candidate 1:
    60670e0bc6e54b87bd15177e36f46451abc64226
published origin/main at Design start:
    c8de1528d1ed57d3e86a9c37d1ad307127a23feb
origin/main tree:
    1d58e398428f35600e9bc582366c846c90d5f47c
TensorDSLab package version:
    0.1.0
Python:
    >=3.14
Torch:
    >=2.13,<2.14
NumPy:
    2.5.1
Pint:
    0.25.3
TensorCore published dependency:
    0.21.0
TensorCore published commit:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
TensorCore published tree:
    af5c4f6d693fa25cf767f3aaae31a47d86cf3a8d
```

Maintenance 14 retained the complete Maintenance 13 production behavior.
Complete source and extracted-archive evidence passed:

```text
305 tests run
302 passed
3 conditional unavailable-CUDA skips
```

Pyright reported zero diagnostics, and the exact TensorCore dependency
negative fixture retained `82` intended diagnostics. Those totals identify the
starting evidence only. They are not future module-count, file-count, or
test-count contracts.

### Historical unpublished TensorCore state at Design selection

TensorCore Stage 30 had already completed its local package loop before this
replacement architecture was selected. Its exact unpublished state is:

```text
TensorCore local main / HEAD:
    de235057ee7c0bf702c40e8f331fc4e89a67b7c7
TensorCore local tree:
    c31f007e38ebfa068233419703a061306a9678e4
TensorCore local state:
    clean, three linear commits ahead of origin/main
TensorCore live origin/main:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
TensorCore live origin/main tree:
    af5c4f6d693fa25cf767f3aaae31a47d86cf3a8d
publication state:
    Stage 30 not pushed or published
```

The unpublished Stage 30 bytes add a narrow `TensorConfig` and derived
Field/Kernel device and dtype vocabulary. TensorDSLab had truthfully confirmed
that superseded contract before selecting this broader compositional
architecture. The replacement is therefore an architecture change, not a
finding against those exact bytes.

TensorCore must preserve its accepted local history. A future TensorCore
replacement stage must advance from the exact local state by ordinary forward
history. It must not reset, amend, rewrite, hide, or pretend the locally closed
Stage 30 package loop never happened.

TensorDSLab did not adopt the unpublished Stage 30 bytes. Its dependency was
required to remain exact published TensorCore `0.21.0` until TensorCore:

1. accepts a synchronized replacement Design;
2. implements and independently clears the complete compositional contract;
3. closes the replacement locally;
4. publishes one exact containing commit; and
5. supplies exact source, artifact, typing, and consumer evidence.

TensorCore has now satisfied all five conditions through exact published
`0.22.0` commit `19bfae35fbc773b55cac7bcd659dda57c4dee6d6`.
That publication makes the exact commit eligible for a future package-owned
TensorDSLab adoption candidate; it does not itself change TensorDSLab's
dependency bytes.

### Exact Stage 31 Design authority

TensorCore has since frozen the replacement Design at:

```text
stable key:
    TensorCore/stage-31-compositional-tensor-spec-substrate
exact Design authority:
    25f48e3398c68217b060d94743f8abd810e7f7e8
exact tree:
    4bd15c7db276acc6d23848bf301e493dee3d2278
exact parent / first Design candidate:
    98ab3ee3b88ae903e1535dbe2c0df5ff9a673c02
forward baseline:
    de235057ee7c0bf702c40e8f331fc4e89a67b7c7
```

TensorDSLab Design reviewed and confirmed those exact replacement bytes.
Stage 31 additionally admits gradient-bearing generic Fields, preserves
ordinary Torch autograd connectivity across differentiable explicit
movement/casting, and supplies representation-preserving contiguous
Coordinates/Axis windows plus exact FieldSpec axis replacement. These are
generic capabilities rather than a TensorDSLab workflow or batching policy.

TensorCore Design also confirmed that the Stage 31 fieldful-Spec contract
already supports the selected representation-owned unit model:

- `QuantityAxis` owns physical coordinate scale and coordinate Unit;
- `QuantityFieldSpec` and `QuantityKernelSpec` add the sole represented-tensor
  Unit state;
- exact semantic Product Specs specialize `QuantityFieldSpec`, and Product
  leaves directly specialize TensorField with those exact Specs;
- exact semantic physical-coefficient Specs specialize
  `QuantityKernelSpec`, and coefficient leaves directly specialize
  TensorKernel with those exact Specs;
- no `QuantityField` or `QuantityKernel` intermediate root is required; and
- exact-subtype Spec and payload transformations preserve unit state and rerun
  the existing most-derived validation exactly once.

That synchronization authorized TensorCore to resume its independently owned
Stage 31 implementation under exact authority `25f48e3`. It did not adopt an
unpublished dependency or dispatch TensorDSLab Implementation.

### Exact published Stage 31 package

TensorCore completed its complete package loop and published:

```text
repository:
    https://github.com/mbedard44/TensorCore.git
published local/live origin main:
    19bfae35fbc773b55cac7bcd659dda57c4dee6d6
exact tree:
    53aa10520a50c0714e79c685d814cbae1b6f7740
version:
    0.22.0
prior published main:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
wheel:
    54,052 bytes
wheel SHA-256:
    6ac2f29c562504d7e87e1caf404b10019b08d60252fc496ad55b090e6b8b154f
commit-bound source archive:
    1,095,680 bytes
archive SHA-256:
    deb09f72595a44f3b8551f01971986aa265a28a3f4475ee2afe59fb2b63f0c84
```

Independent TensorDSLab read-only verification found local main, tracking
`origin/main`, and live GitHub `refs/heads/main` exact at `19bfae3`; tree
`53aa105`; metadata version `0.22.0`; dependencies Python `>=3.14` and Torch
`>=2.13,<2.14`; exact export counts `34 / 13 / 15`; `41` package files / `40`
Python modules; a six-commit linear fast-forward from `78d0891`; zero merges;
and a clean synchronized TensorCore checkout.

Accepted TensorCore evidence is `85` tests with exactly two unavailable-CUDA
skips, Pyright zero positive diagnostics and exactly `97` intentional negative
diagnostics, deterministic artifacts at the hashes above, and complete
package/typing/documentation/privacy gates. No tag, GitHub Release,
package-index publication, fresh CUDA evidence, or compatibility claim
follows.

TensorDSLab Design accepts exact `19bfae3` as Maintenance 15's future
dependency target. Actual adoption remains a TensorDSLab production change
that requires the separately frozen executable work order and package loop.

## Architecture Principles

### Specifications describe representations

A specification describes what a tensor representation is:

```text
semantic axes
device
dtype
```

A quantity specification additionally describes:

```text
physical unit
```

Specifications contain no tensor payload and execute no scientific operation.
They are immutable, structurally comparable values that can be created,
checked, transformed explicitly, and shared before any payload is allocated.

### Fields and kernels carry exact specifications

Every TensorField and TensorKernel carries one exact Spec object. A consumer
never has to infer the representation from a tensor, a semantic class name, a
parallel Config field, or a Runtime record.

```text
TensorField  = tensor payload + TensorFieldSpec
TensorKernel = tensor payload + TensorKernelSpec
```

The tensor must match its Spec exactly. Construction does not silently repair
shape, device, dtype, axes, coordinates, or units.

### Configs describe transformations

A Product-specific Config describes one transformation. It contains:

- the exact output Spec;
- one exact typed collection of computational kernels;
- the exact bounded scientific policy; and
- meaningful prepared facts when the Config has been aligned.

A Config is not a generic tensor representation, so there is no generic
`TensorConfig`, `QuantityConfig`, or universal Config root in the selected end
state.

### Every axis-varying coefficient is a kernel

The selected Config boundary is fail-closed:

> If a configurable value may vary over example, channel, microcell, sample,
> or another semantic axis while the Product still executes the same
> algorithm, that value is represented by its own semantic TensorKernel leaf.

This rule applies to:

- physical rates, widths, probabilities, intensities, and responses;
- numerical gains, thresholds, lower and upper bounds, and digitizer depth;
- integer, floating, complex, Boolean-mask, dimensional, and dimensionless
  coefficient representations; and
- future configurable values that participate elementwise in one unchanged
  Product equation.

A caller-supplied global coefficient is not a scalar Config shortcut. It is a
rank-zero kernel with empty conditioning and operation axes. A coefficient
that varies by channel has `ChannelAxis` as a conditioning axis. A coefficient
that varies by example and channel has both roles in its exact ordered
conditioning geometry. Translation support or another literal destination
geometry uses operation axes instead.

This is a rule for configurable coefficient state, not a claim that every
source of output variation is a kernel. Source tensors, deterministic
coordinate values, and random draws naturally vary Products without becoming
Config kernels.

The following remain structural Config or Product state rather than kernels:

- the output Spec and its axes, device, dtype, and unit;
- the exact typed kernel collection and optional-member presence;
- Product source-count and source-relationship policy;
- algorithm or stochastic-law selection;
- bounded iteration and recursion depth;
- semantic relationships derived from the exact Product and Kernel Specs;
- boundary policy;
- RNG role and address identity; and
- application workflow and retention policy.

The dividing line is whether a value changes coefficients inside one fixed
algorithm or changes the algorithm/domain itself. `BitDepth` may vary by
channel while the digitizer equation remains unchanged, so it is a kernel.
`correlated_avalanche_generations` changes the recursion topology, so it
remains structural ChargeConfig state. Optional kernel presence may select
one already frozen Product branch, but the numerical member itself remains a
kernel.

Product preparation must reject a kernel whose conditioning role is absent
from the Product domain. Supplying `ChannelAxis` to an application permits a
channel-conditioned coefficient; omitting that role means no Product Config
may smuggle in channel variation. Preparation aligns every admitted
conditioning coordinate exactly and never infers a missing role.

### Products are parts, not pipeline stages

Every Product is independently usable. TensorDSLab does not encode one
universal graph, registry, pipeline, reflection mechanism, or Product
dependency hierarchy.

The application chooses the graph:

```text
application inputs
    -> Product.create(...)
    -> optional retained Product
    -> another Product.create(...)
    -> application result
```

### Preparation pays policy cost before production

Preparation performs:

- relationship and semantic-axis admission;
- coordinate correspondence;
- conditioning-axis reordering;
- kernel alignment;
- unit conversion;
- dtype planning;
- device materialization;
- allocation and count preflight; and
- preparation of meaningful immutable execution facts.

Production consumes the aligned Config and sources. It performs numerical
tensor work but no Pint interpretation, coordinate search, semantic-axis
permutation discovery, device movement, or dtype-policy selection.

### Explicit movement is allowed; silent movement is not

Specs, Fields, Kernels, and Collections may provide explicit `.to(...)`
operations with exact contracts. Product preparation may call those operations
deliberately. Product production must not silently move caller values.

### Generic mechanics stay generic

TensorCore owns package-neutral coordinate, axis, Spec, Field, Kernel, and
Collection mechanics. TensorDSLab owns quantities, physical transformations,
scientific laws, Configs, preparation, product validation, and stochastic role
identity. Applications own collaboration semantics and workflows.

## Selected Vocabulary

The selected generic names are:

```text
Coordinates
CountCoordinates
RegularCoordinates
LabelCoordinates
OffsetCoordinates
TensorAxis
OffsetAxis
TensorFieldSpec
TensorKernelSpec
TensorField
TensorKernel
TensorCollection
TensorArtifact
```

`TensorFieldSpec` lives with `TensorField` in
`tensor_core/tensor/field.py`. `TensorKernelSpec` lives with `TensorKernel` in
`tensor_core/tensor/kernel.py`. Separate `spec.py` modules are not selected.

The selected TensorDSLab quantity representation names are:

```text
ExampleAxis
ChannelAxis
QuantityAxis
TimeAxis
FrequencyAxis
QuantityFieldSpec
QuantityKernelSpec
```

There is no initial `QuantityField` or `QuantityKernel` root. Unit is
representation metadata beside axes, device, and dtype, so it lives only in
the exact quantity Spec. Semantic identity belongs to an Axis or Spec class;
representation belongs to its composed Coordinates or generic TensorCore
state.

The selected Product Spec names are:

```text
PhotoelectronsSpec
ChargeSpec
PureWaveformSpec
NoiseWaveformSpec
AnalogWaveformSpec
DigitizedWaveformSpec
```

The selected computational-kernel Spec names are:

```text
TimingJitterSpec
DirectCrosstalkSpec
DelayedCrosstalkSpec
AfterpulseSpec
DarkCountRateSpec
SmearingWidthSpec
PulseResponseSpec
WhiteNoiseRmsSpec
PowerSpectralDensitySpec
AnalogMinimumSpec
AnalogMaximumSpec
BitDepthSpec
InputMinimumSpec
InputMaximumSpec
AnalogGainSpec
```

Product leaves specialize `TensorField` through their exact Product Spec.
Physical coefficient leaves specialize `TensorKernel` through their exact
physical-kernel Spec. Quantity-disabled Fields and Kernels use ordinary
TensorCore Specs without creating a parallel tensor-value hierarchy.

The selected Product and computational-kernel names include:

```text
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform

ChargeConfig
ChargeKernels
PureWaveformConfig
PureWaveformKernels
NoiseWaveformConfig
NoiseWaveformKernels
AnalogWaveformConfig
AnalogWaveformKernels
DigitizedWaveformConfig
DigitizedWaveformKernels

TimingJitter
DirectCrosstalk
DelayedCrosstalk
Afterpulse
DarkCountRate
SmearingWidth
PulseResponse
WhiteNoiseRms
PowerSpectralDensity
AnalogMinimum
AnalogMaximum
BitDepth
InputMinimum
InputMaximum
AnalogGain
```

`PulseResponse` replaces `Pulse`. Every Product Config is the complete
punchcard; its corresponding `*Kernels` value is only the typed computational
kernel collection. No caller-supplied configurable algorithm coefficient is
stored directly as a Python scalar, Pint Quantity, or raw tensor in a Config.
Immutable execution facts derived by Product preparation are a separate,
Product-owned category.

## TensorCore Ownership

TensorCore owns the following generic representation substrate. This section
is the TensorDSLab consumer contract synchronized with exact TensorCore Stage
31 Design authority `25f48e3`; it is not authority to edit TensorCore.

## Coordinates

### Root purpose

`Coordinates` represents one complete ordered coordinate set. The plural name
is deliberate: an instance owns the entire coordinate representation for an
axis, not one coordinate value.

A conceptual root is:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Coordinates[CoordinateT](ABC):
    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def coordinate_at(self, index: int) -> CoordinateT:
        ...

    @abstractmethod
    def index_of(self, coordinate: CoordinateT) -> int:
        ...

    @final
    def window(
        self,
        *,
        start_index: int,
        count: int,
    ) -> Self:
        ...
```

The exact TensorCore implementation may choose a different internal method
layout, but the public semantics are frozen:

- immutable, slotted, tensor-free state;
- structural equality and hashing over exact semantic class and complete
  representation state;
- deterministic ordered coordinate identity;
- strict index admission;
- strict coordinate admission;
- strict contained contiguous-window admission;
- exact representation-subtype preservation across a changed window;
- exact extent admission in `[0, 2**63 - 1]`, matching realizable Torch
  dimension bounds;
- no units;
- no device or dtype;
- no tensor materialization;
- no scientific axis role;
- no movement;
- no interpolation;
- no tolerance-based lookup; and
- no normalization or repair.

Coordinate representation classes may use compact state. They need not
materialize a tuple containing every coordinate.

`window()` requires exact non-boolean built-in integer arguments and:

```text
0 <= start_index <= size
0 <= count <= size - start_index
```

A complete no-op returns the same object. A changed window reconstructs the
same exact Coordinates subtype in the same represented order and reruns its
ordinary validation. A zero-length window is valid at any admitted start,
including the exclusive end.

### `CountCoordinates`

`CountCoordinates` compactly represents:

```text
start, start + 1, ..., start + count - 1
```

Its selected state is:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CountCoordinates(Coordinates[int]):
    count: int
    start: int = 0
```

Its contract is:

- `count` and `start` are exact non-boolean built-in integers;
- `0 <= count <= 2**63 - 1`;
- `start` uses exact unbounded Python-integer coordinate arithmetic;
- `size == count`;
- coordinates are exact built-in integers;
- `coordinate_at(i) == start + i`;
- `index_of(c) == c - start` when
  `start <= c < start + count`;
- `CountCoordinates(count=n)` retains the shorthand `range(n)`;
- a changed window adjusts `start` and `count` without dense materialization
  or coordinate renumbering;
- rank-zero consumers remain possible because a coordinate representation is
  not itself a tensor rank;
- zero extent is valid; and
- no semantic meaning such as "example" or "event" enters TensorCore.

### `RegularCoordinates`

`RegularCoordinates` represents:

```text
start + index * step
```

Its selected state is:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RegularCoordinates(Coordinates[int]):
    start: int
    step: int
    count: int
```

Its generic contract is:

- `start`, `step`, and `count` are exact non-boolean built-in integers;
- `0 <= count <= 2**63 - 1`;
- `step != 0`;
- positive and negative steps are both generic representation values;
- `size == count`;
- coordinate calculation uses exact unbounded Python-integer arithmetic, so
  represented coordinate magnitudes are not narrowed merely because the
  realizable extent is signed-int64 bounded;
- `index_of()` admits only exact represented coordinates;
- lookup performs no floating comparison or tolerance;
- no physical period, sampling frequency, origin convention, or time unit
  enters TensorCore; and
- downstream semantic leaves may narrow the generic contract, such as
  requiring a positive step.

The representation remains compact for large coordinate counts. Its changed
window adjusts `start` by exact arithmetic, retains `step`, and changes only
`count`.

### `LabelCoordinates`

Its selected state is:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class LabelCoordinates(Coordinates[str]):
    labels: tuple[str, ...]
```

Its contract is:

- `labels` is exactly a tuple;
- every label is an exact built-in `str`;
- every label is nonempty;
- labels are ordered and unique;
- supplied label identity and spelling are preserved;
- `size == len(labels)`;
- `coordinate_at()` returns the exact stored label;
- `index_of()` uses exact string equality;
- a changed window retains the exact ordered label tuple slice and string
  identities;
- duplicates are rejected rather than silently disambiguated; and
- TensorCore assigns no channel, sensor, detector, or collaboration meaning.

### `OffsetCoordinates`

Its selected state is:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OffsetCoordinates(Coordinates[int]):
    offsets: tuple[int, ...]
```

Its contract is:

- `offsets` is exactly a tuple;
- each offset is an exact non-boolean built-in integer;
- offsets are ordered and unique;
- negative, zero, and positive values are all admitted;
- empty support is generically valid;
- a changed window retains the exact ordered offset tuple slice;
- no sorting, symmetry, regularity, contiguity, causality, unit, anchor,
  padding, convolution, or boundary policy is implied;
- `coordinate_at()` and `index_of()` use exact ordered values; and
- structural equality includes the complete ordered offset tuple.

## TensorAxis

### Composition replaces representation inheritance

`TensorAxis` composes a Coordinates value:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class TensorAxis[CoordinateT](ABC):
    coordinates: Coordinates[CoordinateT]
```

The root intentionally has one type parameter. Python and Pyright cannot
express a second TypeVar whose bound is parameterized by the first TypeVar.
Downstream semantic classes narrow the stored `coordinates` annotation
instead.

The previous representation subclasses:

```text
CountAxis
RegularAxis
LabelAxis
```

are retired without aliases, wrappers, forwarding imports, or parallel
vocabulary.

Semantic axes specialize `TensorAxis`. A semantic role may deliberately admit
more than one exact Coordinates representation:

```python
@final
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class ExampleAxis[
    CoordinateT: (int, str),
](TensorAxis[CoordinateT]):
    coordinates: Coordinates[CoordinateT]


@final
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class ChannelAxis[
    CoordinateT: (int, str),
](TensorAxis[CoordinateT]):
    coordinates: Coordinates[CoordinateT]
```

TensorDSLab owns these shared semantic roles; TensorCore does not. The first
runtime contract admits CountCoordinates, LabelCoordinates,
RegularCoordinates, and OffsetCoordinates. Static typing rejects a
Coordinates value whose coordinate type is neither exact integer nor exact
string, while runtime semantic validation rejects every unsupported
Coordinates class.

### Generic axis contract

TensorCore owns:

- exact Coordinates instance admission;
- preservation of the supplied Coordinates object;
- `size` forwarding;
- `coordinate_at()` forwarding;
- `index_of()` forwarding;
- representation-preserving `window(start_index=..., count=...) -> Self`;
- exact semantic axis class identity;
- structural equality and hashing over exact semantic class, exact coordinate
  representation, and complete downstream immutable representation fields;
- frozen and slotted value semantics;
- downstream abstract intermediate roots that may add immutable, tensor-free,
  structurally comparable fields; and
- zero-size axes.

An Axis no-op window returns the same object. A changed window delegates to
its Coordinates exactly once, reconstructs the exact semantic Axis subtype,
retains every downstream immutable field, and reruns universal plus
most-derived Axis validation exactly once. It performs no tensor slicing or
batch-selection policy.

TensorCore does not own:

- Pint;
- physical dimensions;
- collaboration role names;
- interpretation of coordinate magnitudes;
- a requirement that every semantic leaf be fieldless;
- device or dtype;
- axis alignment policy;
- target-product legality;
- coordinate conversion between different representations; or
- semantic equivalence between different exact axis classes.

An exact semantic axis class remains the role identity used by Fields,
Kernels, Specs, and Products.

## OffsetAxis

`OffsetAxis` is the one generic semantic axis with additional relationship
state:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class OffsetAxis(TensorAxis[int]):
    coordinates: OffsetCoordinates
    relative_to: type[TensorAxis[Any]]
```

The selected contract is:

- `OffsetAxis` is concrete, final, and non-generic with respect to its target
  role;
- `coordinates` is exact `OffsetCoordinates`;
- `relative_to` is one exact TensorAxis class role, not an axis instance;
- the target class is preserved by identity;
- equality and hashing include exact `relative_to` class identity;
- multiple OffsetAxis values may coexist in one kernel when they target
  different semantic roles;
- no `SampleOffsetAxis`, `MicrocellXOffsetAxis`, or similar subclass family is
  introduced;
- no PEP 695 target-role parameter is introduced;
- TensorCore owns no signed-displacement application, anchor, causality,
  geometry, finite-boundary behavior, or units; and
- downstream kernel validation may require `relative_to` to be a particular
  semantic role.

For example, a future pixelated detector may use:

```python
operation_axes = (
    OffsetAxis(
        relative_to=MicrocellXAxis,
        coordinates=OffsetCoordinates(offsets=(-1, 0, 1)),
    ),
    OffsetAxis(
        relative_to=MicrocellYAxis,
        coordinates=OffsetCoordinates(offsets=(-1, 0, 1)),
    ),
    OffsetAxis(
        relative_to=TimeAxis,
        coordinates=OffsetCoordinates(offsets=(0, 1, 2, 3)),
    ),
)
```

The ordinary row-major tensor index of each operation cell remains its generic
cell identity. Signed displacement meaning is downstream-owned.

## TensorFieldSpec

### TensorFieldSpec location and representation

`TensorFieldSpec` belongs in `tensor_core/tensor/field.py` beside
`TensorField`.

A conceptual root is:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class TensorFieldSpec[
    AxesT: tuple[TensorAxis[Any], ...],
]:
    axes: AxesT
    device: torch.device
    dtype: torch.dtype
```

It is a tensor-free structural value. It describes the complete expected
representation of one TensorField.

### Exact contract

TensorCore owns:

- `axes` admitted as exactly one tuple of constructed TensorAxis values;
- exact axis types unique within the completed field domain;
- exact supplied axis objects and tuple preserved;
- `device` admitted as exact `torch.device`;
- exact supplied device preserved without availability checks or
  normalization;
- `dtype` admitted as exact `torch.dtype`;
- no supported/unsupported dtype policy beyond structural admission at the
  generic root;
- rank-zero domains;
- zero-extent axes;
- derived `rank`;
- derived exact `shape`;
- Python-integer `element_count`, equal to `1` at rank zero and `0` when any
  extent is zero;
- strict `axis_at(index)`;
- exact-type `dimension_of(axis_type)`;
- typed exact-type `axis(axis_type)` lookup;
- structural equality and hashing over exact Spec class and complete immutable
  state;
- downstream Spec subclasses that may add immutable, tensor-free, hashable
  dataclass fields; and
- an explicit `.to(device=..., dtype=...)` representation transformation.

The root owns no:

- tensor;
- allocation;
- movement;
- units;
- layout promise;
- gradient policy beyond what the eventual Field contract requires;
- product;
- scientific law;
- workflow;
- source relationship; or
- preparation state.

### TensorFieldSpec axis replacement

Stage 31 supplies:

```python
target_spec = source_spec.with_axis(
    dimension=dimension,
    axis=window_axis,
)
```

The replacement Axis must have the same exact semantic type as the current
Axis at that dimension. A no-op returns `self`; a change retains the supplied
Axis by identity, reconstructs the exact concrete FieldSpec subtype, preserves
all other axes/device/dtype/downstream state, and reruns universal plus
most-derived validation exactly once. TensorCore performs no payload slicing,
batch-axis selection, coordinate-alignment policy, or workflow scheduling.
TensorDSLab or an application may pair the transformed Spec with its own
tensor view when a Product contract admits that window.

### TensorFieldSpec transformation

The selected generic transformation is:

```python
target_spec = source_spec.to(
    device=torch.device("cuda:0"),
    dtype=torch.float32,
)
```

Its contract is:

- either keyword may be omitted;
- omitted values remain exact;
- no tensor is allocated or moved;
- exact axis objects and tuple are retained;
- downstream immutable fields are retained;
- the returned object has the same exact concrete Spec type;
- an exact no-op target returns `self`;
- every changed reconstruction reruns universal validation and the existing
  most-derived semantic requirement exactly once after all fields exist;
- availability is not checked;
- no dtype promotion is inferred;
- no units are converted; and
- Spec subclasses must remain reconstructible under this operation.

TensorCore must freeze one exact safe reconstruction mechanism. It must not use
an unchecked public constructor, lose downstream fields, or return the base
Spec type.

## TensorKernelSpec

### TensorKernelSpec location and representation

`TensorKernelSpec` belongs in `tensor_core/tensor/kernel.py` beside
`TensorKernel`.

A conceptual root is:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class TensorKernelSpec[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
]:
    conditioning_axes: ConditioningAxesT
    operation_axes: OperationAxesT
    device: torch.device
    dtype: torch.dtype
```

The complete literal tensor axis order is:

```text
(*conditioning_axes, *operation_axes)
```

### Conditioning-axis invariants

- `conditioning_axes` is exactly a tuple of constructed TensorAxis values;
- exact conditioning-axis classes are unique within that tuple;
- supplied axis objects and tuple are preserved;
- no generic conditioning subset, broadcasting, selection, or source policy is
  implied; and
- zero conditioning rank and zero extents are admitted.

### Operation-axis invariants

- `operation_axes` is exactly a tuple of constructed TensorAxis values;
- operation target roles are unique;
- an OffsetAxis contributes `axis.relative_to` as its target role;
- any other operation axis contributes `type(axis)` as its target role;
- repeated exact concrete OffsetAxis types are therefore valid when their
  `relative_to` roles differ;
- there is no blanket exact-concrete-type uniqueness rule across the complete
  axis tuple;
- a conditioning role may equal an operation target role;
- zero operation rank and empty operation support are generically admitted;
  and
- downstream physical leaves may require nonempty operation support.

### Derived facts

TensorCore owns:

```text
axes
conditioning_rank
operation_rank
rank
conditioning_shape
operation_shape
shape
conditioning_element_count
operation_element_count
element_count
axis_at
operation_target_roles
```

All element counts use exact Python-integer multiplication. Complete row-major
index conversion remains structural and deterministic.

`axis_at(dimension)` is the only generic complete-axis lookup in the first
TensorKernelSpec surface. An unqualified global `dimension_of(axis_type)` or
`axis(axis_type)` is deliberately absent:

- one operation geometry may contain several exact `OffsetAxis` values whose
  distinct `relative_to` roles make them valid;
- one semantic role may occur once as a conditioning axis and once as an
  operation target; and
- selecting by exact concrete axis type or by target role would therefore be
  ambiguous without an explicit conditioning-versus-operation scope.

Role-set-specific lookup may be added only after a demonstrated consumer
requires it and freezes both the role scope and whether returned dimensions
are local to that role set or refer to the complete kernel tuple.

### TensorKernelSpec transformation

`TensorKernelSpec.to(device=..., dtype=...)` follows the same exact
same-subclass, no-allocation, no-unit-conversion contract as
`TensorFieldSpec.to(...)`.

## Generic role resolution

TensorCore owns a focused generic relationship operation equivalent to:

```python
dimensions = require_kernel_dimensions(field_spec, kernel_spec)
```

The exact accepted resolution rule is:

- each conditioning axis resolves by exact `type(axis)`;
- each OffsetAxis operation axis resolves by exact `axis.relative_to`;
- each non-offset operation axis resolves by exact `type(axis)`;
- returned dimensions follow complete kernel-axis order;
- absence of any required semantic role is a generic error;
- ambiguous exact-type resolution is impossible because of the relevant Spec
  construction invariants; and
- resolution performs no permutation, expansion, broadcast, movement, unit
  conversion, coordinate comparison, or tensor allocation.

TensorDSLab owns the next layer:

- whether a required role is scientifically allowed;
- exact coordinate correspondence;
- conditioning-coordinate reordering;
- tensor permutation;
- insertion of broadcast dimensions;
- storage expansion, if ever explicitly selected;
- condition selection;
- signed displacement application;
- finite-window behavior; and
- Product-specific diagnostics.

## Construction And Validation Hooks

TensorCore Design owns the exact decorators and protected hook spelling, but
the replacement must satisfy these consumer requirements:

- generic construction validates root representation state before
  downstream scientific narrowing;
- every downstream stored dataclass field is initialized before validation;
- the most-derived semantic requirement runs exactly once;
- direct generic Spec values remain constructible;
- TensorAxis supports fieldful immutable downstream semantic roots;
- TensorFieldSpec and TensorKernelSpec support fieldful immutable downstream
  Spec subclasses;
- TensorField, TensorKernel, and TensorCollection semantic leaves remain
  fieldless;
- a downstream package can validate its added Spec fields without replacing or
  bypassing universal TensorCore validation;
- same-subtype Spec, Field, Kernel, and Collection reconstruction reruns the
  existing most-derived semantic requirement exactly once after all fields
  exist;
- cooperative intermediate validation, if required, is explicit and
  statically testable;
- no public subclass token, reflection registry, runtime finality scan, or
  constructor bypass is introduced; and
- diagnostics identify the exact semantic class and field relationship.

One acceptable shape is a final root `__post_init__` that performs universal
validation and then calls one protected most-derived requirement hook.
TensorCore may select another equally strict mechanism in its exact work order.
What is not acceptable is requiring TensorDSLab to duplicate generic
shape/device/dtype/axis validation or to override the complete constructor.

## TensorField

### TensorField representation

The selected generic shape is:

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class TensorField[
    SpecT: TensorFieldSpec[Any],
](ABC):
    tensor: torch.Tensor
    spec: SpecT

    __hash__ = None
```

TensorField owns one payload and one exact Spec.

### TensorField construction

Generic construction requires:

- ordinary exact `torch.Tensor` admission;
- strided tensor layout;
- exact `tensor.shape == spec.shape`;
- exact `tensor.device == spec.device`;
- exact `tensor.dtype is spec.dtype`;
- no silent movement;
- no silent cast;
- no reshape;
- no axis inference;
- no unit inference;
- no tensor-value equality;
- identity equality;
- explicit unhashability; and
- logical read-only public tensor state under the existing TensorCore
  mutability contract.

TensorField does not duplicate axes, device, dtype, or unit fields. Its
generic structural properties forward from `spec`; a quantity-bearing
downstream value accesses unit explicitly through its statically narrowed
`spec.unit`.

Generic TensorField admits both gradient-bearing and non-gradient tensors,
preserves the exact tensor reference, does not detach it, and does not suppress
ordinary Torch autograd recording. Autograd admission remains
operation-owned downstream policy. A TensorDSLab Product may reject
gradient-sensitive sources or results when its exact stochastic or
nondifferentiable contract requires that restriction; it must not claim that
TensorCore imposed the Product policy.

### TensorField movement and casting

`TensorField.to(device=..., dtype=...)`:

- returns the same exact semantic Field subtype;
- constructs a matching exact concrete Spec subtype;
- delegates explicit tensor movement/casting to Torch;
- retains exact axes and downstream Spec state;
- returns `self` for an exact no-op target;
- performs no unit conversion;
- performs no dtype promotion policy;
- adds no new TensorCore scientific policy, but normal exact-subtype
  reconstruction reruns the subtype's already-defined semantic validation
  exactly once because dtype conversion can change represented values; and
- does not change semantic class identity;
- does not detach the source or alter ambient grad mode; and
- preserves ordinary Torch autograd connectivity whenever the explicit
  conversion is differentiable under Torch's own contract.

Generic exact-subtype reconstruction requires supported TensorField semantic
leaves to add no stored dataclass fields beyond `tensor` and `spec`.
TensorCore must freeze this fieldless-leaf contract for Fields even though
Spec subclasses may add state.

## TensorKernel

### TensorKernel representation

The selected generic shape is:

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class TensorKernel[
    SpecT: TensorKernelSpec[Any, Any],
](ABC):
    tensor: torch.Tensor
    spec: SpecT

    __hash__ = None
```

### TensorKernel construction

TensorKernel requires:

- exact `torch.Tensor`;
- exact `TensorKernelSpec`;
- strided, gradient-free tensor structure;
- exact shape/device/dtype agreement;
- one fresh same-device contiguous defensive snapshot;
- no axis, device, or dtype inference;
- no units;
- no physical-law validation;
- identity equality; and
- explicit unhashability.

TensorKernel semantic leaves add no stored dataclass fields. Their semantic
state belongs in the exact concrete KernelSpec subtype.

### TensorKernel movement and casting

`TensorKernel.to(device=..., dtype=...)` follows the same semantic-subtype and
Spec-preservation rules as TensorField, while retaining the defensive
ownership contract. Its newly allocated result reruns universal
tensor/Spec agreement and the existing most-derived semantic requirement
exactly once.

TensorCore must implement movement without exposing:

- an unchecked public constructor;
- a public trust token;
- an unsafe alias of caller-owned tensor state;
- a base-class result;
- duplicate defensive snapshots beyond the exact accepted contract; or
- bypassed semantic validation; or
- lost downstream Spec fields.

The exact private reconstruction mechanism is TensorCore-owned and must be
frozen in its work order. A package-private trusted adoption path for the
freshly allocated movement result is acceptable only when it avoids a
duplicate snapshot while still running universal and most-derived validation.

## TensorCollection

### TensorCollection purpose

`TensorCollection` is a typed immutable parts collection, not a workflow, a
common-domain assertion, or a product graph.

It may contain:

```text
TensorField values
TensorKernel values
or an explicitly selected mixture
```

A conceptual root is:

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class TensorCollection[
    MemberT: TensorField[Any] | TensorKernel[Any],
](ABC):
    members: tuple[MemberT, ...]

    __hash__ = None
```

The exact representation may use a private immutable mapping plus a stable
tuple view. The public semantics are:

- members are keyed by exact semantic member type;
- at most one member of each exact type;
- stable insertion order;
- exact supplied member objects preserved;
- empty collection admitted;
- immutable membership;
- identity equality and explicit unhashability;
- typed exact-type lookup;
- no string or class-name lookup;
- no subclass matching;
- no common axes requirement;
- no common device requirement;
- no common dtype requirement;
- no common unit requirement;
- no shape requirement;
- no workflow edge;
- no production order; and
- no reflection-driven execution.

Selected accessors are conceptually:

```python
collection.members
collection.member_types
collection.member(ProductType)
```

The replacement is deliberately complete. The published field-named
collection vocabulary is retired without aliases:

```text
fields
field_types
field(...)
tensor(...)
require_field_types(...)
```

`member(...)` returns the exact semantic member object; callers access that
Field or Kernel's `.tensor` directly. The first replacement stage does not add
`require_member_types(...)` merely to preserve a renamed generic validator.
Concrete collection leaves enforce their exact allowed member types through
their semantic requirement. A future generic member validator requires a
demonstrated second consumer and a separately frozen contract.

### Explicit collection movement

The generic collection operation is device-only:

```python
moved = collection.to(device=torch.device("cuda:0"))
```

It:

- delegates to each member's exact `.to(device=...)`;
- preserves exact collection subtype;
- preserves stable member order;
- returns `self` when every member already targets that exact device;
- reconstructs a changed exact collection subtype and reruns its existing
  most-derived semantic requirement exactly once;
- performs no generic dtype cast because a heterogeneous collection may
  deliberately contain float, integer, and complex representations;
- performs no unit conversion;
- does not require a common device before movement; and
- does not infer a workflow.

Supported semantic collection leaves initially add no stored dataclass fields
beyond members. If a second consumer later demonstrates fieldful collection
state, TensorCore must design an exact reconstruction contract rather than
smuggling policy into the base.

## TensorArtifact

TensorArtifact generalization is explicitly deferred.

The current field-oriented artifact behavior remains unchanged, but the
generic Collection replacement requires one narrow static correction:

```text
TensorArtifact.materialize(...)
    -> TensorCollection[TensorField[Any]]
```

The exact existing artifact method spelling and parameters remain
TensorCore-owned. Its return annotation and evidence must prove that artifact
materialization remains field-only rather than accidentally admitting Kernels
or mixed members.

This record does not:

- generalize artifacts to kernels;
- persist TensorKernelSpec;
- persist physical kernels;
- add a generic collection artifact;
- choose a durable schema for the new Specs;
- select cache compatibility;
- select lazy loading;
- define cross-version migration; or
- authorize IO implementation.

Any future artifact stage must start only after the in-memory compositional
contracts are implemented and stable.

## TensorCore Retirements

The synchronized future TensorCore stage retires, without alias:

```text
TensorConfig
CountAxis
RegularAxis
LabelAxis
TensorCollection.fields
TensorCollection.field_types
TensorCollection.field(...)
TensorCollection.tensor(...)
require_field_types(...)
```

`TensorConfig` is unpublished local Stage 30 state. It must be removed by
forward history in the replacement stage before any containing publication.

The old Axis classes are published pre-1.0 surfaces. Their removal is a
deliberate breaking pre-deployment change. TensorCore must not retain:

- forwarding modules;
- compatibility subclasses;
- aliases;
- dual constructor forms;
- automatic conversion into Coordinates;
- parallel old/new documentation; or
- hidden acceptance of old axes.

TensorCore keeps `OffsetAxis` but revises it to compose
`OffsetCoordinates`.

The selected Collection member vocabulary is the sole replacement:

```text
members
member_types
member(...)
```

No field-named alias, forwarding requirement, or parallel validation facade
survives.

## TensorDSLab Ownership

TensorDSLab consumes the generic TensorCore substrate and owns quantities,
physical kernels, Config punchcards, Product transformations, stochastic roles
and addresses, and Product validation.

## QuantityAxis

### QuantityAxis representation

`QuantityAxis` is a TensorDSLab abstract intermediate semantic axis:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class QuantityAxis[
    CoordinatesT: Coordinates[int],
](TensorAxis[int], ABC):
    coordinates: CoordinatesT
    coordinate_scale: float = 1.0
    unit: pint.Unit
```

It owns:

- exact Pint Unit recognition through the package registry;
- registry normalization under TensorDSLab's accepted unit policy;
- immutable unit state;
- one exact non-Boolean built-in binary64 `coordinate_scale`;
- finite strictly positive coordinate-scale admission;
- integer lattice coordinates supplied by the composed Coordinates value;
- quantity-returning conveniences built from lattice magnitude, coordinate
  scale, and unit; and
- downstream physical-dimension narrowing.

It does not store Pint Quantity tensors.

The physical coordinate represented at index `i` is:

```text
coordinates.coordinate_at(i) * coordinate_scale * unit
```

`coordinate_scale` is ordinary binary64 representation state. Structural
Axis equality and hashing compare the stored float exactly. Physical
relationship checks use their explicitly frozen numerical tolerances.
Application-owned cold-path factories must compute shared scales through one
canonical operation whenever exact structural equality is required.

### Magnitude and quantity access

The raw TensorAxis operations stay representation-oriented:

```text
coordinate_at(index) -> exact integer magnitude
index_of(magnitude)  -> exact integer index
```

QuantityAxis additionally exposes deliberately named quantity accessors, such
as:

```python
axis.quantity_at(index)
axis.quantity_of(magnitude)
```

The executable work order freezes the exact names as `quantity_at()` and
`quantity_of()`. The ownership boundary is:

- TensorCore coordinate lookup remains Pint-free;
- TensorDSLab creates scalar Pint quantities only at an explicit public or
  preparation boundary;
- production never iterates over Pint quantities; and
- unit compatibility is not inferred from the semantic axis class name.

### Cooperative validation

TensorCore calls one inherited `_require()` after every Axis field exists.
`QuantityAxis` owns the cooperative physical-representation chain:

```python
@override
def _require(self) -> None:
    require_supported_integer_coordinates(self.coordinates)
    require_coordinate_scale(self.coordinate_scale)
    require_package_unit(self.unit)
    self._require_quantity_axis()

@abstractmethod
def _require_quantity_axis(self) -> None:
    ...
```

Concrete quantity-axis leaves implement only `_require_quantity_axis()`.
They cannot bypass the common coordinate, scale, or Unit admission. The exact
implementation may use an equivalent protected hook spelling, but it must run
the universal QuantityAxis checks once followed by the semantic leaf checks
once.

## Semantic Axis Roles

TensorDSLab owns reusable detector/readout semantic roles. Coordinate
representation never creates a parallel semantic role:

```text
Axis class             = semantic role
Coordinates instance   = coordinate representation
```

### `ExampleAxis`

`ExampleAxis` is one final semantic role admitting integer or string
coordinates:

```python
@final
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class ExampleAxis[
    CoordinateT: (int, str),
](TensorAxis[CoordinateT]):
    coordinates: Coordinates[CoordinateT]
```

The first supported representations are `CountCoordinates`,
`LabelCoordinates`, `RegularCoordinates`, and `OffsetCoordinates`.
`OffsetCoordinates` composed with `ExampleAxis` is an explicit ordered set of
integer example identifiers. It has no displacement meaning; displacement
semantics arise only from TensorCore `OffsetAxis`, which additionally owns
`relative_to`.

These values have one exact semantic role despite different representations:

```python
ExampleAxis(coordinates=CountCoordinates(count=100))
ExampleAxis(
    coordinates=LabelCoordinates(labels=("background", "signal"))
)
```

Consequently one Field Spec may contain at most one `ExampleAxis`, while two
different Specs may use different ExampleAxis coordinate representations.

### `ChannelAxis`

`ChannelAxis` has the same representation-polymorphic integer-or-string
contract as `ExampleAxis`. Counted channels and labeled channels are both
exact `ChannelAxis` values. TensorDSLab does not require every application to
invent a new semantic class merely to choose numeric identifiers or names.

### `TimeAxis`

`TimeAxis[CoordinatesT: Coordinates[int]]` is one final QuantityAxis semantic
role. It requires a time-compatible package Unit. Its general representation
may use supported integer Coordinates, while each consuming Product Spec or
Config narrows the representation required by its equation.

For a regular physical grid, the canonical representation is:

```python
TimeAxis(
    coordinates=RegularCoordinates(start=0, step=1, count=5000),
    coordinate_scale=2.0,
    unit=unit("ns"),
)
```

Regular waveform laws require `RegularCoordinates.step == 1`; physical
spacing lives in `coordinate_scale * unit`. A temporal window may retain a
nonzero or negative integer `start`.

### `FrequencyAxis`

`FrequencyAxis[CoordinatesT: Coordinates[int]]` is the corresponding final
frequency semantic role and requires a frequency-compatible package Unit.
The PSD law narrows it to `RegularCoordinates` with `start == 0` and
`step == 1`.

For example:

```python
FrequencyAxis(
    coordinates=RegularCoordinates(start=0, step=1, count=2),
    coordinate_scale=1.0 / 3.0,
    unit=unit("GHz"),
)
```

represents the binary64 physical grid `0 GHz` and `(1.0 / 3.0) GHz` while
retaining exact integer coordinate identity.

### Application-specific roles

Applications instantiate the shared `ExampleAxis`, `ChannelAxis`, `TimeAxis`,
and `FrequencyAxis` roles. They define a new Axis class only for genuinely new
semantics, such as `MicrocellXAxis` and `MicrocellYAxis`. Such leaves may
specialize `QuantityAxis` and participate normally in TensorCore Specs without
TensorDSLab knowing them in advance.

## QuantityFieldSpec

`QuantityFieldSpec` is the abstract TensorDSLab quantity specialization of
`TensorFieldSpec`:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class QuantityFieldSpec[
    AxesT: tuple[TensorAxis[Any], ...],
](TensorFieldSpec[AxesT], ABC):
    unit: pint.Unit
```

It owns:

- axes;
- target device;
- target dtype;
- normalized Pint Unit;
- complete structural equality and hashing including unit;
- same-exact-subtype `.with_axis(...)` and
  `.to(device=..., dtype=...)`, each preserving the exact unit; and
- no tensor payload.

It uses the same cooperative validation pattern as QuantityAxis: common Unit
admission runs exactly once before one semantic Product-Spec hook.

TensorDSLab defines these final, directly constructible semantic Product
Specs:

```text
PhotoelectronsSpec
ChargeSpec
PureWaveformSpec
NoiseWaveformSpec
AnalogWaveformSpec
DigitizedWaveformSpec
```

Each adds no duplicate axes, device, dtype, or unit field. Its exact class is
the static and runtime representation contract for the corresponding Product.
Product-local representation invariants belong in that Spec. Cross-object
relationships that depend on sources or Config kernels remain Config/Product
preparation policy.

## QuantityKernelSpec

`QuantityKernelSpec` is the abstract kernel counterpart:

```python
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class QuantityKernelSpec[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](TensorKernelSpec[ConditioningAxesT, OperationAxesT], ABC):
    unit: pint.Unit
```

It owns normalized Pint Unit alongside exact literal geometry, device, and
dtype.

It owns the common Unit admission and one cooperative physical-kernel-Spec
hook. TensorDSLab defines one final, directly constructible semantic Spec for
each physical coefficient:

```text
TimingJitterSpec
DirectCrosstalkSpec
DelayedCrosstalkSpec
AfterpulseSpec
DarkCountRateSpec
SmearingWidthSpec
PulseResponseSpec
WhiteNoiseRmsSpec
PowerSpectralDensitySpec
AnalogMinimumSpec
AnalogMaximumSpec
InputMinimumSpec
InputMaximumSpec
AnalogGainSpec
```

`BitDepthSpec` directly specializes ordinary TensorKernelSpec because
BitDepth has no Pint meaning.

`PowerSpectralDensitySpec` strongly narrows its complete operation geometry:

```python
class PowerSpectralDensitySpec[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
](
    QuantityKernelSpec[
        ConditioningAxesT,
        tuple[FrequencyAxis[RegularCoordinates]],
    ],
):
    ...
```

It requires exactly one regular FrequencyAxis operation axis with
`start == 0` and `step == 1`. Runtime validation remains mandatory; static
typing is not a deserialization or dynamic-Python security boundary.

## Direct quantity-Spec specialization

There is no selected `QuantityField` or `QuantityKernel` class. The tensor
value roots remain exactly TensorCore's `TensorField` and `TensorKernel`.
Quantity-aware versus quantity-free representation is determined by the exact
Spec subtype:

```python
@final
class Charge(TensorField[ChargeSpec[Any]]):
    __slots__ = ()


@final
class DarkCountRate(TensorKernel[DarkCountRateSpec[Any, Any]]):
    __slots__ = ()
```

The exact implementation follows TensorCore's supported fieldless semantic
leaf convention rather than duplicating dataclass fields in these sketches.
The important contract is:

- the tensor stores magnitudes in `spec.unit`;
- unit exists only on the exact semantic quantity Spec;
- a Product or physical-kernel leaf adds no duplicate unit,
  `canonical_unit`, or implied-unit state;
- `.to(...)` reconstructs the exact quantity Spec and payload subtype and
  preserves the exact unit;
- physical dimension, sign, range, normalization, geometry, and Product
  equations remain leaf/Product validation;
- a physical kernel is not an executable effect, Distribution factory,
  callback, or workflow node; and
- a quantity-free Field or Kernel uses an ordinary TensorCore Spec without
  participating in a parallel quantity-value class hierarchy.

This retires the current `canonical_unit` pattern. Unit conversion belongs to
Product preparation, where source, kernel, and output equations are known.

Not every computational kernel has quantity meaning. A coefficient with
physical or dimensionless quantity meaning directly specializes TensorKernel
with its semantic QuantityKernelSpec subtype carrying the literal unit. A
discrete coefficient that participates elementwise in an otherwise unchanged
numerical equation directly specializes TensorKernel with its semantic
ordinary TensorKernelSpec subtype. The first selected example is integer
`BitDepth` with `BitDepthSpec`.

TensorDSLab does not add a generic `ParameterKernel`, `CoefficientKernel`, or
Config-kernel framework. Exact semantic leaves and Product-specific typed
collections make the supported meaning visible.

## Unit Policy

### Units are explicit representation state

Every quantity-bearing Field and Kernel carries a quantity Spec. The Spec's
unit is the literal unit in which tensor magnitudes are represented.

No Product class implies one mandatory unit convention. A user may choose any
unit compatible with the Product's equation:

```text
source unit
kernel unit
output unit
```

The Product validates dimensional compatibility and prepares exact conversion
scales. It does not require one permanent dimensional family merely because a
class historically used one convention.

### Equations, not class names, determine compatibility

For convolution:

```text
source.spec.unit * pulse_response.spec.unit -> config.spec.unit
```

For direct summation:

```text
each source.spec.unit -> config.spec.unit
```

For dark-count generation:

```text
dark_count_rate.spec.unit * temporal_coordinate_step.unit -> expected count
```

For charge smearing:

```text
smearing_width.spec.unit -> dimensionless relative one-avalanche width
standard_deviation magnitude
    = smearing_width magnitude * sqrt(nonnegative Charge-count magnitude)
```

The executable work order freezes the precise equations and coefficient units
for every Product. Unit admission is fail-closed and occurs before stochastic
words or large allocations.

### Multi-source unit compatibility

Every Product that accepts more than one source must validate the complete
source-unit relationship before combining any tensor values.

For an additive Product, preparation requires:

1. every source has a QuantityFieldSpec;
2. every source unit is dimensionally convertible to the Product's selected
   common compute unit;
3. the common compute unit is dimensionally convertible to the configured
   output unit;
4. an exact conversion scale is prepared independently for every source;
5. no source tensor is cast, moved, converted, or summed until every source
   passes; and
6. failure identifies the Product, source tuple index, source unit, and
   required unit relationship.

Compatibility is dimensional, not exact-spelling equality. These may be valid
when the Product equation admits them:

```text
V + mV -> mV
avalanche + photoelectron-equivalent -> avalanche
```

These must fail before tensor work:

```text
avalanche + mV
voltage + time
count + spectral-density
```

For example:

```python
sources = (
    photoelectrons,
    axioelectrons,
)
```

is a valid Charge candidate only when both represented units are convertible
to the Charge source/accumulation dimension selected by `ChargeConfig`.
Replacing either source with a voltage waveform must fail during
`Charge.prepare(...)`.

A Product whose source operation is not addition must freeze its own complete
unit equation with the same fail-before-effects rule. The generic source-tuple
surface never means that arbitrary quantity-Spec TensorFields can be
numerically combined.

### Source device identity

Every nonempty Product source tuple has one exact execution device:

```text
source.spec.device is equal to config.spec.device
```

for every source. Therefore:

- all sources are on the same exact device as one another;
- every source device equals the configured output device;
- accelerator-backed Product Specs use an indexed concrete device such as
  `cuda:0`, not an unresolved current-device spelling;
- `prepare()` validates the complete source tuple before moving or converting
  any Config-owned kernel;
- `create()` does not move sources as a convenience;
- `produce()` does not move sources;
- source dtype casts and unit-scale multiplication occur only on the already
  selected device;
- Config-owned kernels may be explicitly converted and materialized onto that
  device during same-type Config preparation; and
- stochastic address elements and distribution tensors must satisfy the
  Product's exact same-device relationship before words.

If one source is on `cpu` and another is on `cuda:0`, or if every source is on
`cpu` while the output Spec requests `cuda:0`, preparation fails before:

```text
unit conversion
dtype casting
kernel movement
summation
allocation
RNG word requests
```

The error identifies the Product, source tuple index, supplied device, and
required output device.

Callers choose movement explicitly:

```python
photoelectrons_gpu = photoelectrons.to(
    device=torch.device("cuda:0"),
)

axioelectrons_gpu = axioelectrons.to(
    device=torch.device("cuda:0"),
)

charge = Charge.create(
    sources=(
        photoelectrons_gpu,
        axioelectrons_gpu,
    ),
    config=charge_config,
    rng=rng,
)
```

Zero-source Products, such as configured NoiseWaveform generation, have no
source-device relationship. Their output Spec still selects the exact device,
and every participating kernel/RNG representation must satisfy that Product's
explicit device contract.

### No Pint on the hot path

Preparation computes immutable scalar conversion facts. Production uses:

- plain Python scalar magnitudes where safe;
- scalar tensors on the execution device where required;
- tensor magnitudes;
- exact prepared dtype; and
- no Pint Quantity operations.

Pint remains a public configuration and preparation boundary.

## Dtype Policy

### Every realized tensor representation declares dtype

`TensorFieldSpec.dtype` and `TensorKernelSpec.dtype` make dtype explicit for
sources, kernels, and outputs.

`ProductConfig.spec.dtype` declares the required output dtype. A Product may
not inspect unrelated global state or silently choose a result dtype.

### Product-owned working dtype

Each Product distinguishes two dtype concepts:

- **representation dtype** is the exact dtype stored by one source, output
  Spec, or kernel Spec and admitted by that semantic value; and
- **working dtype** is a Product-owned arithmetic dtype used for one prepared
  equation.

Preparation never treats the working dtype as a command to reconstruct every
source and kernel at one common dtype. TensorCollection remains
heterogeneous, and every prepared kernel retains a representation dtype that
its exact semantic contract admits.

Each Product deterministically derives its working dtype during preparation
from the exact numerical inputs that participate in that arithmetic:

```text
output Spec dtype
source Spec dtypes
participating arithmetic-kernel Spec dtypes
Product numerical floor
```

The derivation is an ordered fold using exact `torch.promote_types`, not
Python numeric `max`, dtype enumeration order, backend defaults, or implicit
Torch expression promotion.

Conceptually:

```python
working_dtype = product_floor

for dtype in (
    config.spec.dtype,
    *(source_spec.dtype for source_spec in source_specs),
    *(kernel.spec.dtype for kernel in arithmetic_kernels),
):
    working_dtype = torch.promote_types(working_dtype, dtype)
```

`arithmetic_kernels` is selected by the Product's frozen equation. It contains
only members whose semantic representation contract admits participation in
that arithmetic promotion. Exact discrete or structural-valued kernels are
not cast merely because their values affect the equation. The Product first
uses such a kernel in its own exact domain and then prepares an explicitly
checked derived value for arithmetic where needed.

The exact Product floor is scientific/numerical policy:

- discrete probability, expected-count, and stochastic parameter preparation
  may require `torch.float64`;
- waveform convolution may accept the promoted floating dtype selected by the
  Product;
- ADC code output is integer, but analog gain and scaling use a prepared
  floating working dtype;
- integer `BitDepth` retains its exact integer representation dtype and does
  not promote or get reconstructed at that floating dtype; and
- unsupported dtype families fail during preparation.

This realizes the user's precision control: increasing an output Spec or
kernel/source dtype can raise working precision. A Product may still require a
higher floor to preserve its accepted law.

### Explicit casts

Preparation freezes:

- each source-to-working cast;
- each arithmetic-admissible kernel representation-to-working cast;
- each representation-preserved kernel use and checked derived conversion;
- any scalar representation;
- the final working-to-output cast; and
- the exact device on which each cast occurs.

Production performs those planned casts as numerical tensor operations. It
does not rediscover promotion policy. Moving a heterogeneous kernel collection
changes device only; it never silently homogenizes member dtypes.

Rounding is explicit and unavoidable when the output dtype is narrower than
the working dtype. Validation must prove the selected output-domain error and
overflow contracts.

## Product Model

### Product identity

A Product is a final semantic class that directly specializes
`TensorField[ItsExactProductSpec[...]]` and owns these class methods:

```text
create
prepare
produce
validate
```

There is no separate universal Product ABC or intermediate `QuantityField`
root beyond TensorCore's generic `TensorField`.

The common vocabulary improves navigability, but signatures remain
Product-specific. TensorDSLab does not add a registry, reflection loop,
callback framework, or universal source law merely to invoke those methods
uniformly.

### Config identity

Each Product has one exact Product-specific Config:

```text
ChargeConfig
PureWaveformConfig
NoiseWaveformConfig
AnalogWaveformConfig
DigitizedWaveformConfig
```

Every Config is:

- frozen;
- slotted;
- keyword-only;
- identity-equal;
- explicitly unhashable;
- directly typed;
- free of execution methods;
- free of TensorCore inheritance; and
- a complete punchcard for one Product transformation.

Every Config stores:

- one exact output Spec;
- one exact Product-specific typed `*Kernels` collection; and
- only the additional structural policy genuinely required by that Product.

No public caller-facing Config constructor accepts a caller-supplied
configurable algorithm coefficient as a primitive scalar, Pint Quantity, or
raw tensor. A caller-supplied global value is represented by a rank-zero
semantic kernel. This makes global and conditioned configuration use the same
public type and the same validation/alignment path.

This prohibition does not apply to immutable execution facts derived by
Product preparation. A prepared same-type Config may retain directly typed
scalar conversion scales, resolved dimensions, selected dtypes, checked
ceilings, and comparable non-configurable facts. Any derived-only stored field
must be excluded from ordinary caller construction through the exact
Product-owned construction contract. If a derived tensor must persist, the
Product must either retain it through an exact semantic Kernel/Spec value or
freeze a separate logically read-only, defensively owned private tensor
contract in the executable Product work order. A mutable raw caller tensor is
never an accepted coefficient shortcut.

There is no:

```text
TensorConfig
QuantityConfig
ProductConfig
Config registry
Config reflection protocol
```

as a shared runtime root.

### Same-type preparation

Preparation returns a fresh value of the same exact Config type:

```python
prepared = PureWaveform.prepare(
    source_specs=tuple(source.spec for source in sources),
    config=config,
)

assert type(prepared) is PureWaveformConfig
```

The prepared Config may replace caller-oriented kernel representations with
aligned, converted, and materialized representations and may retain meaningful
derived execution facts. It does not become a separate `Runtime`,
`PreparedConfig`, `Plan`, token, cache, or opaque compiled wrapper.

Structural readiness is visible in ordinary Config state and enforced by
Product-specific validation.

### Ordered source tuples

Product entry points accept:

```python
sources: tuple[TensorField[Any], ...]
```

The broader value annotation is deliberate. TensorCore TensorField is
invariant in its exact Spec parameter, so a direct semantic Product carrying
`PhotoelectronsSpec` is not statically a
`TensorField[QuantityFieldSpec[Any]]`. Every Product source-taking boundary
therefore first requires the source's exact `.spec` to be a
QuantityFieldSpec, then applies the Product-specific semantic, unit, device,
dtype, and numerical source law. A nonquantity TensorField may type-check at
this generic tuple boundary but must fail runtime admission before effects.

The tuple is ordered and exact. TensorDSLab does not globally require every
Product to interpret sources the same way.

Each Product owns:

- accepted source count;
- whether zero sources are meaningful;
- whether source semantic classes matter;
- axis and coordinate relationships;
- the complete source-unit equation;
- per-source conversion into one declared compute representation;
- exact source-to-output device identity;
- combination order;
- count and allocation ceilings;
- dtype promotion;
- deterministic summation order; and
- diagnostics.

This deliberately does not try to prevent every scientifically foolish user
combination at a generic framework layer. It ensures that each Product's own
mathematical contract is validated and statically visible. In particular, it
does not permit a Product to add dimensionally incompatible sources merely
because both are TensorField values with QuantityFieldSpecs.

### Lifecycle

A representative deterministic Product shape is:

```python
@final
class PureWaveform(TensorField[PureWaveformSpec[Any]]):
    __slots__ = ()

    @classmethod
    def create(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: PureWaveformConfig,
    ) -> Self:
        prepared = cls.prepare(
            source_specs=tuple(source.spec for source in sources),
            config=config,
        )
        product = cls.produce(
            sources=sources,
            config=prepared,
        )
        cls.validate(
            product=product,
            sources=sources,
            config=prepared,
        )
        return product

    @classmethod
    def prepare(
        cls,
        *,
        source_specs: tuple[QuantityFieldSpec[Any], ...],
        config: PureWaveformConfig,
    ) -> PureWaveformConfig:
        ...

    @classmethod
    def produce(
        cls,
        *,
        sources: tuple[TensorField[Any], ...],
        config: PureWaveformConfig,
    ) -> Self:
        ...

    @classmethod
    def validate(
        cls,
        *,
        product: Self,
        sources: tuple[TensorField[Any], ...],
        config: PureWaveformConfig,
    ) -> None:
        ...
```

A stochastic Product adds the exact RNG argument it needs:

```python
charge = Charge.create(
    sources=(photoelectrons, axioelectrons),
    config=charge_config,
    rng=rng,
)
```

There is no fake RNG parameter on deterministic Products and no generic
`**kwargs` escape hatch.

### Direct and staged use

The one-shot path is:

```python
product = Product.create(
    sources=sources,
    config=config,
)
```

An application that wants to prepare a complete graph before execution may
use:

```python
prepared_config = Product.prepare(
    source_specs=source_specs,
    config=config,
)

product = Product.produce(
    sources=sources,
    config=prepared_config,
)

Product.validate(
    product=product,
    sources=sources,
    config=prepared_config,
)
```

Both paths call the same Product-owned preparation, production, and validation
actions. Neither path moves a source. An application that chooses another
device explicitly moves its source Products and constructs or transforms the
output Spec before preparation.

### Private action modules

The Product class owns the public API, while focused private runtime modules
retain readable numerical functions:

```text
product/runtime/prepare.py
product/runtime/produce.py
product/runtime/validate.py
```

For example:

```python
@classmethod
def produce(
    cls,
    *,
    sources: tuple[TensorField[Any], ...],
    config: PureWaveformConfig,
) -> Self:
    tensor = produce_pure_waveform(
        sources=sources,
        config=config,
    )
    return cls(
        tensor=tensor,
        spec=config.spec,
    )
```

`produce_pure_waveform()` may return the raw tensor because the public Product
class owns semantic result construction. The simple Product-specific function
name is retained for navigability.

The runtime package contains actions, not Runtime records. There is no
`_produce_prepared()` duplicate API. `Product.produce()` is already the
prepared execution boundary.

### Output Spec identity

A produced Product retains the exact configured output Spec:

```python
product.spec is config.spec
```

Production must not silently reconstruct, normalize, replace, or widen the
output Spec. If preparation needs a different output representation, it
returns a new same-type Config containing the new exact Spec before
production.

## Preparation Contract

Every Product-specific preparation follows this generic order where
applicable:

1. admit the exact Config type;
2. admit the exact source-Spec tuple;
3. validate Product source count and relationship;
4. validate output Spec semantic requirements;
5. validate every source device equals the exact output Spec device;
6. resolve required semantic roles;
7. validate exact source/output coordinate relationships;
8. admit the exact Product-specific kernel collection, required/optional
   member set, and every semantic kernel value;
9. validate every kernel conditioning role is available in the Product
   domain;
10. determine coordinate reordering and dimension permutation;
11. validate Product unit equations;
12. select the deterministic Product working dtype and the per-member
    representation/conversion plan;
13. convert quantity-kernel units;
14. align every kernel's conditioning coordinates and dimensions;
15. materialize every aligned kernel on the output device and its exact
    Product-selected prepared representation dtype;
16. prepare immutable conversion and execution facts;
17. preflight element, byte, count, and address ceilings;
18. return a fresh Config of the same exact type; and
19. perform no random draw and consume no RNG word.

The exact order matters. Invalid public meaning fails before expensive
materialization, allocation, or stochastic execution.

### Kernel alignment

TensorCore role resolution supplies structural dimensions. TensorDSLab
alignment additionally proves:

- every required conditioning role is present;
- exact coordinate values correspond one-to-one;
- caller coordinate order may differ;
- the stable coordinate permutation is exact;
- conditioning dimensions are permuted into source/output order;
- operation axes remain ordered and untouched;
- broadcast insertion is explicit;
- no storage expansion occurs unless the Product work order explicitly
  selects it;
- quantity-kernel units are converted once;
- each final tensor is contiguous on the target device and its exact
  Product-selected prepared representation dtype;
- a representation-preserved member is never cast merely to homogenize its
  containing collection; and
- the returned Kernel preserves its exact semantic type and has the exact
  aligned TensorKernelSpec or QuantityKernelSpec subtype.

The Maintenance 13 `align_quantity_kernel()` behavior is parts-bin evidence
for this future preparation action. Its current signature and current
Runtime-oriented ownership are not frozen future API.

### Prepared-state visibility

Meaningful derived Config facts may include:

```text
working dtype
source conversion scales
output conversion scale
resolved temporal dimension
aligned computational kernels
checked scalar values derived from coefficients
preflight ceilings
```

They must be:

- immutable;
- directly typed;
- scientifically named;
- derivable from public Config and source-Spec state;
- validated by the Product;
- free of mutable caches;
- free of callable execution;
- free of RNG state; and
- absent when the Product does not need them.

The list does not authorize an unexplained raw tensor field. A persistent
derived tensor must use an exact semantic Kernel/Spec value or an explicit
Product-private logical-read-only ownership contract frozen by the executable
work order. These derived facts are not caller-configurable coefficients, and
their storage does not create scalar, Pint, or raw-tensor shortcuts in the
public Config constructor.

The executable work order freezes the exact common and Product-specific
private prepared Config fields. It creates no parallel generic prepared
framework.

## Production Contract

Product production:

- accepts exact typed sources;
- accepts the exact Product Config;
- verifies fail-closed structural readiness before numerical work;
- trusts no caller-supplied opaque token;
- performs planned source conversion and dtype casts;
- consumes already aligned kernel tensors;
- performs Product mathematics;
- constructs one fresh result tensor;
- constructs the exact semantic Product using `config.spec`;
- does not call Pint;
- does not search coordinates;
- does not permute conditioning coordinates;
- does not move sources or kernels;
- does not select dtype policy;
- does not create or mutate Config state;
- does not mutate sources;
- does not mutate kernels;
- does not use global RNG;
- requests words only after complete validation and preflight; and
- preserves exact package-owned address identity.

Product production may use private tensor workspaces. Those are ordinary local
execution values, not public Runtime objects or Config fields.

## Validation Contract

Product validation owns:

- exact semantic Product type;
- exact output Spec object identity;
- exact tensor shape/device/dtype;
- finite/value-domain requirements;
- source/result relationship;
- product-specific count, code, or saturation limits;
- required freshness and no-alias relationships;
- stochastic conservation or statistical law;
- exact boundary behavior;
- and any accepted scientific postcondition.

Generic TensorCore construction does not replace Product validation.

Validation is not a silent repair path. It does not:

- cast;
- move;
- clip;
- normalize;
- renormalize;
- reshape;
- relabel axes;
- replace nonfinite values;
- or mutate Config or Product state.

## Computational Kernel Contracts

Every configurable coefficient is a final, directly constructible, fieldless
TensorKernel semantic leaf. Dimensional and dimensionless physical
coefficients directly specialize TensorKernel with their exact semantic
QuantityKernelSpec subtype; discrete coefficient values such as BitDepth
directly specialize TensorKernel with their exact semantic ordinary
TensorKernelSpec subtype. Generic TensorKernel construction proves exact
tensor/Spec agreement and defensive ownership. The semantic Spec, leaf, and
consuming Product prove the coefficient's value, dtype, unit, conditioning,
and operation-geometry contract.

All computational kernels require:

- the exact semantic TensorKernelSpec subtype selected by the leaf;
- exact literal conditioning and operation geometry;
- no arbitrary stored fields outside `tensor` and `spec`;
- no `__dict__`;
- no callable, Distribution, Config, Runtime, RNG, or mutable cache;
- no inferred canonical unit; and
- exact semantic type preservation under `.to(...)`.

Every floating or complex coefficient additionally requires finite represented
values. Integer and Boolean leaves freeze their exact dtype and value domains
instead of passing through a floating-point admission path.

The universal coefficient-geometry rule is:

- empty conditioning axes mean one global rank-zero coefficient when operation
  axes are also empty;
- nonempty conditioning axes identify exactly where the coefficient may vary;
- every conditioning role must exist in the Product domain;
- conditioning coordinates align exactly, including stable reordering;
- no operation axes are admitted for an ordinary pointwise coefficient;
- operation axes are present only when the coefficient literally carries
  support or destination geometry; and
- adding a conditioning axis changes represented coefficient state but never
  selects a different algorithm.

The selected first computational contracts are:

| Kernel | Value and unit law | Geometry law |
|---|---|---|
| `TimingJitter` | finite nonnegative dimensionless represented probabilities; complete operation-cell sum equals one within the frozen binary64 tolerance for every conditioning point | exactly one nonempty OffsetAxis operation axis targeting TimeAxis |
| `DirectCrosstalk` | finite nonnegative dimensionless unconditional expected-offspring intensity; represented total no greater than one per conditioning point in the first accepted law | nonempty OffsetAxis operation geometry; temporal displacement, when present, is nonnegative |
| `DelayedCrosstalk` | finite nonnegative dimensionless unconditional expected-offspring intensity; represented total no greater than one per conditioning point in the first accepted law | nonempty OffsetAxis geometry with exactly one positive temporal target |
| `Afterpulse` | finite nonnegative dimensionless unconditional expected-offspring intensity; represented total no greater than one per conditioning point in the first accepted law | one nonempty positive temporal OffsetAxis in the current law |
| `DarkCountRate` | finite nonnegative rate compatible with inverse time | no operation axes |
| `SmearingWidth` | finite nonnegative dimensionless relative Gaussian width under the preserved Charge law | no operation axes |
| `PulseResponse` | finite signed response coefficients; unit participates literally in the convolution equation | nonempty OffsetAxis geometry accepted by PureWaveform; current use has one nonnegative temporal target |
| `WhiteNoiseRms` | finite nonnegative magnitude compatible with NoiseWaveform output unit | no operation axes |
| `PowerSpectralDensity` | finite nonnegative prepared per-bin output powers with the exact squared output unit required by NoiseWaveform | `PowerSpectralDensitySpec` owns exactly one regular FrequencyAxis operation representation; preparation verifies its count and spacing against the unique TimeAxis selected by the output law |
| `AnalogMinimum` | finite magnitude compatible with AnalogWaveform output unit | no operation axes; conditioning allowed |
| `AnalogMaximum` | finite magnitude compatible with AnalogWaveform output unit | no operation axes; conditioning allowed |
| `BitDepth` | exact integer values in the retained `[1, 16]` domain; ordinary TensorKernelSpec with an exact integer dtype and no Pint unit | no operation axes; conditioning allowed |
| `InputMinimum` | finite quantity magnitude compatible with the DigitizedWaveform source unit | no operation axes; conditioning allowed |
| `InputMaximum` | finite quantity magnitude compatible with the DigitizedWaveform source unit | no operation axes; conditioning allowed |
| `AnalogGain` | finite strictly positive linear dimensionless multiplier | no operation axes; conditioning allowed |

`InputMinimum < InputMaximum` is a Product-owned pointwise relationship after
alignment; neither leaf can validate it in isolation. `BitDepth` remains a
kernel because the same digitizer equation may use a different code depth by
example, channel, microcell, sample, or another admitted semantic role.
`AnalogGain` stores the literal linear multiplier used by the equation. A
collaboration profile that starts from a voltage gain in decibels converts it
explicitly while constructing the kernel:

```text
linear_gain = 10 ** (gain_db / 20)
```

TensorDSLab does not retain a scalar `analog_gain_db` Config field or perform
that application-facing convention conversion on the Product hot path.

The explicit represented-total ceiling on the three first-generation branching
kernels encodes the currently selected physical meaning: one tensor answers
both "how strongly does this mechanism occur?" and "where does its offspring
land?" If a future detector requires an expected multiplicity greater than one
per parent, that is a scientific contract change requiring a focused
TensorDSLab Design decision. It must not be obtained by quietly removing the
sub-unity check.

TimingJitter differs deliberately: it redistributes an existing count and
therefore represents a complete unity law before finite-window application.

The executable work order freezes exact reduction dimensions, stable host
summation, tolerances, validation order, empty behavior, and device
synchronization. The physical distinction is not an implementation shortcut.

## Product Kernel Collections

Every Product with configurable coefficients owns one exact final typed
TensorCollection leaf:

```text
ChargeKernels
PureWaveformKernels
NoiseWaveformKernels
AnalogWaveformKernels
DigitizedWaveformKernels
```

Each collection:

- admits only its exact supported semantic kernel types;
- rejects duplicate exact member types;
- freezes which members are required and which are optional;
- preserves supplied member identity and order;
- exposes typed Product-meaningful properties;
- performs no reflection-driven execution;
- owns no output Spec, source, RNG, or workflow;
- may contain TensorKernel members with QuantityKernelSpec and ordinary
  TensorKernelSpec representations where the Product requires both; and
- may be explicitly moved as one value during same-type Config preparation.

The common suffix is vocabulary, not a generic TensorDSLab collection base.
Each Product collection owns its exact member contract directly.

## Charge

### `ChargeKernels`

`ChargeKernels` is a final typed TensorCollection leaf containing zero or one
of each exact supported physical kernel:

```python
@final
class ChargeKernels(TensorCollection[TensorKernel[Any]]):
    @property
    def timing_jitter(self) -> TimingJitter | None:
        ...

    @property
    def direct_crosstalk(self) -> DirectCrosstalk | None:
        ...

    @property
    def delayed_crosstalk(self) -> DelayedCrosstalk | None:
        ...

    @property
    def afterpulse(self) -> Afterpulse | None:
        ...

    @property
    def dark_count_rate(self) -> DarkCountRate | None:
        ...

    @property
    def smearing_width(self) -> SmearingWidth | None:
        ...
```

It:

- admits only the six exact kernel classes;
- admits an empty collection;
- rejects duplicates;
- preserves exact member order;
- provides typed properties;
- owns no generation count;
- owns no output Spec;
- owns no temporal-axis policy;
- owns no RNG keys;
- owns no execution;
- owns no scientific combination law; and
- may be explicitly moved as a collection during preparation.

### `ChargeConfig`

The complete Config is conceptually:

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class ChargeConfig:
    spec: ChargeSpec[Any]
    kernels: ChargeKernels
    correlated_avalanche_generations: NonnegativeInteger

    __hash__ = None
```

The exact constrained-scalar annotation follows TensorCore's accepted public
scalar vocabulary.

`ChargeConfig` gains only the narrowly named immutable private derived facts
frozen in the executable work order. It does not gain a generic Config base,
arbitrary Distribution class, callback, factory, role key, or Runtime.

### Source relationship

`Charge` accepts one nonempty ordered tuple of TensorField sources whose exact
Specs are QuantityFieldSpecs.

Preparation requires:

- every source is a TensorField with an exact QuantityFieldSpec;
- every source has the same complete set of exact semantic axis roles as the
  output;
- source axis tuple order may differ;
- each corresponding semantic axis has exactly equivalent coordinate
  representation and downstream axis state;
- every source unit is compatible with the configured Charge output unit;
- every source unit is dimensionally compatible with every other source under
  the exact Charge accumulation equation;
- each source-to-Charge conversion scale is prepared independently;
- a source such as an AnalogWaveform represented in millivolts is rejected
  when the other source and Charge output represent avalanche counts;
- every source device exactly equals `ChargeConfig.spec.device`;
- no source is moved by Charge preparation or production;
- source tensors may be explicitly cast to the working dtype only on that
  already selected device;
- source element counts and checked sum stay within accepted count and Charge
  ceilings; and
- source tuple order is retained for deterministic accumulation.

The Product does not require exact source semantic Product classes. An
application may supply:

```python
sources=(photoelectrons, axioelectrons)
```

provided both satisfy the Charge source law.

Charge combines compatible sources by exact prepared conversion and
deterministic ordered summation before applying Charge mechanisms.

### Temporal-axis relationship

`ChargeSpec` carries the complete output domain. Preparation requires one
exact `TimeAxis` when an enabled mechanism needs time:

- TimingJitter;
- DelayedCrosstalk;
- Afterpulse; or
- DarkCountRate.

DirectCrosstalk may be purely spatial and time-independent, or may include an
exact `OffsetAxis(relative_to=TimeAxis, ...)`. Only offsets targeting
`TimeAxis` receive temporal nonnegativity policy.

When required:

- `TimeAxis` exists exactly once in `config.spec.axes`;
- the axis uses CountCoordinates or RegularCoordinates as admitted by the
  selected Charge law;
- the unit is time-compatible;
- a regular representation uses `step == 1`;
- `coordinate_scale` is finite and strictly positive;
- the Product derives the exact temporal dimension;
- dark-count expected counts use the exact prepared physical spacing; and
- operation OffsetAxis values targeting that role are interpreted in
  coordinate-index displacement units.

No Config field duplicates the semantic role. Time meaning comes from the
exact `TimeAxis` class rather than unit dimensionality or tensor position.

### Current scientific law

Unless a later explicit parity decision changes it, Charge preserves the
Maintenance 12 laws:

```text
TimingJitter
    -> MultinomialDistribution

DarkCountRate
    -> PoissonDistribution

DirectCrosstalk
    -> deterministic kernel-to-destination-rate construction
    -> one tensor-valued PoissonDistribution

DelayedCrosstalk
    -> deterministic kernel-to-destination-rate construction
    -> one tensor-valued PoissonDistribution

Afterpulse
    -> deterministic kernel-to-destination-rate construction
    -> one tensor-valued PoissonDistribution

SmearingWidth
    -> tensor-valued GaussianDistribution
```

Charge retains:

- timing-jitter complete-law conservation before finite-window application;
- deterministic row-major operation-cell identity;
- direct Multinomial probabilities without a prepared wrapper;
- collapsed destination-rate Poisson branching;
- same-round direct, delayed, and afterpulse mechanisms all reading the same
  immutable frontier;
- pooled children forming only the next generation;
- finite-window discard;
- full-unit afterpulse charge;
- no recovery weighting;
- exact configured generation depth;
- checked count accumulation;
- package-owned role keys;
- package-owned RngAddress schemas; and
- exact Product postconditions.

### Crosstalk and afterpulse rate construction

For each Poisson branching mechanism, TensorDSLab deterministically constructs:

```text
lambda_destination
    = sum_source(
        parent_count[source]
        * kernel_intensity[source, destination - source]
      )
```

and then draws the complete destination tensor once.

The physical kernel tensor is the unconditional expected-offspring intensity
for each represented destination cell. It already combines the mechanism's
occurrence strength with its destination distribution. Charge does not require
a second `mean_offspring` scalar or a conditional-probability kernel.

This uses exact Poisson splitting and superposition. It does not:

- draw a total per source;
- allocate through Multinomial;
- materialize source-shape × kernel-shape categories;
- draw out-of-window categories;
- return an overflow count;
- restore retired overflow roles;
- narrow the domain by an unused total-source Poisson ceiling; or
- feed same-generation children back into the current round.

### Timing-jitter law

TimingJitter is a literal physical TensorKernel whose QuantityKernelSpec
carries its unit. It does not contain a Distribution.

After preparation selects the exact applicable probability slab, Charge
constructs:

```python
distribution = MultinomialDistribution(
    counts=counts,
    probabilities=probabilities,
    completion_probability=completion_probability,
)

allocations = distribution.draw(
    rng=rng,
    address=address,
)
```

TensorCore owns generic Multinomial validation and execution. TensorDSLab owns:

- interpreting the quantity-Spec TensorKernel as timing probabilities;
- requiring the abstract complete timing law selected by Config;
- mapping operation cells to destinations;
- finite-window discard;
- scientific role and address construction; and
- final Charge conservation checks.

A unity abstract timing kernel conserves total allocation across the complete
translation law. Finite-window execution may still discard charge whose mapped
destination lies outside the configured temporal domain. These are distinct
contracts.

### Physical-kernel geometry

Concrete physical kernels literally carry their operation geometry.

A sample-only timing law may use:

```text
operation_axes:
    (OffsetAxis(relative_to=TimeAxis, ...),)
```

A future direct-crosstalk kernel over a pixelated detector may use:

```text
operation_axes:
    (
        OffsetAxis(relative_to=<application MicrocellXAxis>, ...),
        OffsetAxis(relative_to=<application MicrocellYAxis>, ...),
        OffsetAxis(relative_to=TimeAxis, ...),
    )
```

Each tensor element is the physical coefficient for one literal row-major
operation cell. TensorDSLab applies the Product-specific displacement and
boundary law.

### Global and conditioned kernels

An application may provide ExampleAxis or ChannelAxis to a Product Config
without requiring every kernel to condition on those roles.

For example, DarkCountRate may be global:

```text
conditioning_axes = ()
```

even when the Charge output has:

```text
(ExampleAxis, ChannelAxis, TimeAxis)
```

A kernel may condition on any validated subset of the available source/output
roles. It cannot condition on a role absent from the Product domain.

Preparation owns subset admission, coordinate alignment, and broadcast
placement. Production does not expand application policy implicitly.

## PureWaveform

### `PulseResponse`

`PulseResponse` is a TensorKernel with QuantityKernelSpec containing the
literal deterministic response coefficients and operation geometry.

It:

- is not required to be a probability;
- is not required to normalize;
- carries exact unit through its QuantityKernelSpec;
- may be global or conditioned;
- uses one or more operation axes as selected by the Product;
- owns no convolution method;
- owns no output unit convention;
- owns no source class requirement; and
- contains no Config or Runtime state.

### `PureWaveformKernels`

```python
@final
class PureWaveformKernels(TensorCollection[TensorKernel[Any]]):
    @property
    def pulse_response(self) -> PulseResponse:
        ...
```

The collection requires exactly one PulseResponse. It admits no empty or
additional member set.

### `PureWaveformConfig`

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class PureWaveformConfig:
    spec: PureWaveformSpec[Any]
    kernels: PureWaveformKernels

    __hash__ = None
```

### Source and output law

PureWaveform accepts one nonempty ordered tuple of compatible TensorField
sources whose exact Specs are QuantityFieldSpecs.

Preparation:

- validates exact source/output semantic domain correspondence;
- aligns source dimensions and coordinates;
- validates source unit compatibility for deterministic summation;
- validates PulseResponse conditioning availability;
- resolves each operation target;
- validates the convolution/displacement support;
- validates:

  ```text
  combined_source_unit * pulse_response.spec.unit -> config.spec.unit
  ```

- selects working dtype;
- converts and aligns the PulseResponse;
- freezes source and output conversion scales; and
- preflights convolution allocation.

Production:

1. converts each source to the working representation;
2. sums sources in exact tuple order;
3. applies the literal PulseResponse convolution;
4. applies the exact operation-axis displacement and finite-window policy;
5. converts to `config.spec.dtype`;
6. returns `PureWaveform(tensor=..., spec=config.spec)`; and
7. performs no polarity convention beyond the literal configured
   PulseResponse values.

The current DS20k negative pulse sign becomes application Profile data in the
PulseResponse tensor. TensorDSLab does not hard-code one detector polarity.

## NoiseWaveform

### Physical kernels

Selected public physical representations are:

```text
WhiteNoiseRms
PowerSpectralDensity
```

Both are TensorKernel leaves whose exact semantic QuantityKernelSpec subtypes
carry their units.

`PowerSpectralDensity` contains an already prepared PSD tensor compatible with
the intended output sampling representation. This maintenance deliberately
does not design the upstream PSD preparation operation.

### `NoiseWaveformKernels`

```python
@final
class NoiseWaveformKernels(TensorCollection[TensorKernel[Any]]):
    @property
    def white_noise_rms(self) -> WhiteNoiseRms | None:
        ...

    @property
    def power_spectral_density(self) -> PowerSpectralDensity | None:
        ...
```

The collection admits an empty set or exactly one of the two current noise
kernels. The simultaneous two-member set is rejected until a later scientific
contract explicitly defines additive independent branches.

### `NoiseWaveformConfig`

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class NoiseWaveformConfig:
    spec: NoiseWaveformSpec[Any]
    kernels: NoiseWaveformKernels

    __hash__ = None
```

The exact selected branch law is:

- empty kernel collection: exact-zero NoiseWaveform;
- WhiteNoiseRms only: IID Gaussian white noise; and
- PowerSpectralDensity only: PSD-shaped noise.

NoiseWaveform accepts an empty source tuple. It does not pretend noise is a
transformation of an unrelated Product merely to satisfy a generic pipeline
shape.

Preparation owns:

- output floating dtype admission;
- kernel conditioning availability;
- kernel/output unit relationship;
- PSD sampling compatibility;
- exact-zero branch;
- dtype floor;
- device materialization;
- allocation preflight; and
- package-owned RNG address facts.

Production preserves the current accepted white and PSD laws unless a future
parity record explicitly rebaselines them.

PowerSpectralDensity is already prepared per-bin output power.
`PowerSpectralDensitySpec` owns exactly one
`FrequencyAxis[RegularCoordinates]` operation axis with canonical
`start == 0` and `step == 1`. This is spectral operation geometry, not a
conditioning axis and not output displacement support.

The PSD branch conditionally requires `NoiseWaveformSpec` to contain exactly
one `TimeAxis`, independent of tensor dimension order. NoiseWaveform owns this
focused exception to generic role resolution: preparation obtains the two
semantic axes, verifies

```text
frequency_count == time_count // 2 + 1
frequency_spacing == 1 / (time_count * time_spacing)
```

under the exact frozen binary64 tolerance, and aligns only the PSD
conditioning axes against the output domain. The Spec cannot conditionally
know which Config kernel branch was selected, so this cross-object
relationship belongs to Config/Product preparation rather than
`NoiseWaveformSpec._require()` alone.

Applications construct both axes and prepare the per-bin powers before
Product execution. Preparation does not construct, interpolate, integrate,
resample, normalize, or repair PSD state. Production consumes the prepared
bin order and primitive execution facts directly and performs no Pint or Axis
coordinate arithmetic.

## AnalogWaveform

### Saturation kernels

Selected optional TensorKernel leaves with QuantityKernelSpecs are:

```text
AnalogMinimum
AnalogMaximum
```

They may be global or condition on an accepted subset of output roles. Their
Specs carry literal axes, device, dtype, and unit.

### `AnalogWaveformKernels`

```python
@final
class AnalogWaveformKernels(TensorCollection[TensorKernel[Any]]):
    @property
    def minimum(self) -> AnalogMinimum | None:
        ...

    @property
    def maximum(self) -> AnalogMaximum | None:
        ...
```

The collection admits empty, lower-only, upper-only, or lower-plus-upper
membership.

### `AnalogWaveformConfig`

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class AnalogWaveformConfig:
    spec: AnalogWaveformSpec[Any]
    kernels: AnalogWaveformKernels

    __hash__ = None
```

AnalogWaveform accepts one nonempty ordered tuple of compatible TensorField
sources whose exact Specs are QuantityFieldSpecs. An application may pass
PureWaveform and NoiseWaveform, but the Product does not import or require
those exact semantic classes.

Preparation:

- aligns all sources to the output semantic domain;
- validates source units are convertible to the output unit;
- selects working dtype;
- aligns optional saturation kernels;
- requires minimum not greater than maximum where both exist;
- freezes exact source conversions; and
- preflights output allocation.

Production deterministically:

1. converts and sums sources in tuple order;
2. applies lower and upper saturation where configured;
3. casts to output dtype; and
4. returns the exact AnalogWaveform.

## DigitizedWaveform

### Computational kernels

DigitizedWaveform owns four required coefficient leaves:

```text
BitDepth
InputMinimum
InputMaximum
AnalogGain
```

BitDepth is an ordinary integer TensorKernel leaf with TensorKernelSpec. The
other three are TensorKernel leaves whose exact QuantityKernelSpecs carry
their units; AnalogGain uses a literal dimensionless linear multiplier. Every
leaf has empty operation geometry and may condition on an accepted subset of
the output semantic roles.

### `DigitizedWaveformKernels`

The heterogeneous typed collection is:

```python
@final
class DigitizedWaveformKernels(TensorCollection[TensorKernel[Any]]):
    @property
    def bit_depth(self) -> BitDepth:
        ...

    @property
    def input_minimum(self) -> InputMinimum:
        ...

    @property
    def input_maximum(self) -> InputMaximum:
        ...

    @property
    def analog_gain(self) -> AnalogGain:
        ...
```

It requires exactly one of every named member and rejects every additional
member.

### `DigitizedWaveformConfig`

A conceptual Config is:

```python
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class DigitizedWaveformConfig:
    spec: DigitizedWaveformSpec[Any]
    kernels: DigitizedWaveformKernels

    __hash__ = None
```

DigitizedWaveform:

- requires exactly one compatible TensorField source with QuantityFieldSpec;
- requires an integer output dtype;
- requires a dimensionless code-like output unit;
- aligns all four coefficient kernels to the output domain;
- validates `InputMinimum < InputMaximum` pointwise after alignment;
- validates the positive linear AnalogGain;
- keeps the prepared `BitDepth` member at its exact integer representation
  dtype while the heterogeneous `DigitizedWaveformKernels` collection remains
  heterogeneous;
- selects a floating working dtype for the three quantity coefficients, source
  gain, and scaling;
- computes each pointwise `maximum_code = 2**bit_depth - 1` through checked
  exact integer arithmetic before any floating conversion;
- proves the exact integer maximum code fits the configured output dtype and
  the selected floating arithmetic representation before converting that
  derived value for scaling;
- preflights every code bound against the output dtype and every intermediate
  exactness/overflow requirement;
- performs no stochastic draw;
- applies the same frozen digitizer equation at every output element using the
  aligned coefficient values;
- returns exact output Spec identity; and
- does not own a detector-specific digitizer Profile.

The aligned elementwise equation is conceptually:

```text
amplified = source * analog_gain
maximum_code = 2**bit_depth - 1
code_float =
    (amplified - input_minimum)
    * maximum_code
    / (input_maximum - input_minimum)
code = integer_cast(clamp(code_float, 0, maximum_code))
```

The executable Product work order must preserve the exact accepted threshold
branches and integer-cast behavior, not substitute a new rounding rule.

The current IV-DSLab-like values remain application Profile data rather than
generic TensorDSLab constants. A profile may create rank-zero kernels for the
current globally uniform law or conditioned kernels for detector-dependent
coefficients.

## Photoelectrons And Other Sources

`Photoelectrons` remains a reusable TensorField semantic Product/value whose
exact `PhotoelectronsSpec` carries its unit and which may enter another
Product as a source. It need not own a Config or producer in TensorDSLab if it
is constructed upstream.

`Axioelectrons` is not added to TensorDSLab core through this maintenance. A
Silex application may own:

```python
@final
class Axioelectrons(TensorField[AxioelectronsSpec[Any]]):
    __slots__ = ()
```

and pass it with Photoelectrons to Charge.

TensorDSLab Products accept their documented structural and physical source
relationships without importing every collaboration source class.

## Product Independence

The package must not encode any of the following as universal truths:

```text
Photoelectrons always precedes Charge
Charge always precedes PureWaveform
PureWaveform and NoiseWaveform always precede AnalogWaveform
AnalogWaveform always precedes DigitizedWaveform
every application produces a Readout
every application retains every intermediate Product
```

Those are possible application graphs, not reusable Product invariants.

The reusable Product surface supports direct use:

```python
charge = Charge.create(
    sources=(photoelectrons,),
    config=charge_config,
    rng=rng,
)
```

and:

```python
pure = PureWaveform.create(
    sources=(charge,),
    config=pure_waveform_config,
)
```

without requiring construction of a whole-readout object.

## Application Ownership

### TensorDSLab core boundary

The selected reusable TensorDSLab package owns:

- reusable ExampleAxis, ChannelAxis, TimeAxis, and FrequencyAxis roles;
- quantity Axis and semantic Spec representations;
- reusable Product semantic classes;
- Product-specific Configs;
- physical TensorKernel leaves with QuantityKernelSpecs;
- preparation, production, and validation actions;
- generic TensorDSLab kernel alignment;
- Product scientific equations;
- Product stochastic role keys and address schemas;
- Product boundaries and postconditions; and
- no collaboration workflow.

It does not own:

- MicrocellXAxis;
- MicrocellYAxis;
- collaboration-specific coordinate values;
- `ds20k_veto()`;
- `silex()`;
- DS20k detector defaults;
- Silex detector defaults;
- whole-readout orchestration;
- a universal `Readout`;
- `ReadoutConfig`;
- `ReadoutCollection`;
- `simulate_readout()`;
- application CLI;
- application demos;
- collaboration IO;
- workflow persistence; or
- application result retention policy.

### DS20k Veto application

A separate collaboration-owned application may define:

```text
DS20kVetoSettings
Readout
ds20k_veto()
```

It instantiates TensorDSLab `ExampleAxis`, `ChannelAxis`, `TimeAxis`, and
`FrequencyAxis` values with its selected Coordinates, scales, and Units.

and assemble:

```text
Photoelectrons
    -> Charge
    -> PureWaveform

NoiseWaveform

PureWaveform + NoiseWaveform
    -> AnalogWaveform
    -> optional DigitizedWaveform
```

The application decides:

- exact axis instances and coordinates;
- any additional application-specific axis classes;
- output Specs;
- units and dtypes;
- kernel tensors;
- requested products;
- RNG root and application domain;
- preparation order;
- production order;
- retained intermediates;
- whole-result collection;
- demonstrations;
- CLI;
- and application IO.

### Silex application

A separate Silex application may define:

```text
MicrocellXAxis
MicrocellYAxis
Axioelectrons
SilexSettings
Readout
Reconstruction
silex()
```

It reuses TensorDSLab `ExampleAxis`, `TimeAxis`, and `FrequencyAxis` and adds
only its genuinely application-specific microcell roles.

Its initial graph may be only:

```text
Axioelectrons + Photoelectrons
    -> Charge
```

It is not required to create placeholder:

```text
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
```

Future pixelated crosstalk kernels can literally use operation geometry:

```text
(MicrocellXAxis offset, MicrocellYAxis offset, TimeAxis offset)
```

without TensorCore or TensorDSLab core importing Silex semantic classes.

### Application factories

An application may expose a semantic factory class such as `DS20kVeto` or
`Silex`. It need not subclass a TensorDSLab `Readout` root because no universal
readout topology or result family has been demonstrated.

Inside an application namespace, the natural result name is unqualified:

```python
@final
class Readout(TensorCollection[TensorField[Any]]):
    pass
```

Package qualification distinguishes `ds20k.veto.Readout` from
`silex.Readout`. Artificial core-level names such as `DS20kReadout` and
`SilexReadout` are not required.

If two applications later prove a meaningful common application protocol,
that abstraction belongs to a separately reviewed application-layer design.
It must not be inferred from similar method names.

## Ownership Matrix

The package boundary is fail-closed:

| Concern | TensorCore | TensorDSLab | Collaboration application |
|---|---|---|---|
| Coordinate representation | Owns generic `Coordinates` values | Uses them | Chooses concrete coordinate values |
| Semantic Axis mechanics | Owns `TensorAxis` and exact class identity | Owns reusable `ExampleAxis`, `ChannelAxis`, `QuantityAxis`, `TimeAxis`, and `FrequencyAxis` roles | Instantiates shared roles and owns additional detector-specific Axis leaves |
| Offset identity | Owns `OffsetAxis.relative_to` and ordered offsets | Interprets displacement per Product | Supplies target semantic Axis classes |
| Field representation | Owns `TensorFieldSpec` and `TensorField` | Owns `QuantityFieldSpec`, semantic Product Specs, and direct Product leaves | Instantiates and consumes Products |
| Kernel representation | Owns `TensorKernelSpec` and `TensorKernel` | Owns `QuantityKernelSpec`, semantic coefficient Specs, and direct computational-coefficient leaves | Instantiates global or conditioned kernel values |
| Device/dtype | Owns explicit generic representation and `.to` mechanics | Owns Product promotion, floors, and readiness | Chooses requested placement and precision |
| Units | Excludes Pint | Owns Pint registry, quantity Specs, equations, and conversions | Chooses physically valid units and values |
| Kernel alignment | Owns exact generic role resolution | Owns coordinate reorder, permutation, broadcast placement, and materialization | Supplies domains and conditioning values |
| Collections | Owns exact-type immutable mechanics | Owns one exact typed `*Kernels` collection per configurable Product | Owns Readout/Reconstruction result collections |
| Configs | Owns no generic Config | Owns exact Product Config punchcards | Owns application Settings and profile composition |
| Product laws | Excludes detector science | Owns reusable Product transformations | Chooses and orders Products |
| RNG engine and laws | Owns words, addresses, and generic Distributions | Owns Product roles, schemas, traversal, and scientific mappings | Owns root domain and workflow invocation |
| Boundary mapping | Excludes detector geometry | Owns Product-specific operation-cell mapping and discard | Supplies semantic target geometry |
| Profiles | Excludes | Excludes collaboration profiles | Owns `ds20k_veto()`, `silex()`, and defaults |
| Readout/Reconstruction | Owns only generic Collection mechanics | Owns no universal workflow result | Owns collaboration result classes |
| Demos and CLI | Excludes | May document direct reusable Product use only | Owns end-to-end application demos and commands |
| IO and artifacts | Owns current generic field artifact only | Defers quantity/Product durable format | Owns application persistence until a focused shared stage |
| CUDA evidence | Owns package-generic evidence | Owns exact Product package evidence | Owns end-to-end workflow evidence |

No column may silently implement another column's policy merely because a
generic mechanism could technically express it.

## Application Package Placement

The preferred target is a separate installable collaboration package or
repository:

```text
ds20k_veto -> tensor_dslab -> tensor_core
silex      -> tensor_dslab -> tensor_core
```

TensorDSLab must not import a collaboration application.

If a temporary in-repository application project is ever authorized, it must
have:

- a separate distribution name;
- separate import package;
- separate metadata;
- separate tests;
- explicit dependency on tensor-dslab;
- wheel payload isolation;
- no import from TensorDSLab private modules; and
- a focused extraction plan.

This Maintenance does not authorize that temporary topology. A separate
cross-repository work order must choose the actual application package.

## Selected TensorDSLab Package Shape

The future reusable core target is product-centered and no longer nested under
one `readout` workflow:

```text
tensor_dslab/
  __init__.py
  common/
    __init__.py
    axis.py
    field.py
    kernel.py
    units.py
    alignment.py
  photoelectrons/
    __init__.py
    field.py
    runtime/
      __init__.py
      validate.py
  charge/
    __init__.py
    config.py
    field.py
    kernel.py
    runtime/
      __init__.py
      prepare.py
      produce.py
      validate.py
      branching.py
      counts.py
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
    __init__.py
    config.py
    field.py
    kernel.py
    runtime/
      __init__.py
      prepare.py
      produce.py
      validate.py
  analog_waveform/
    __init__.py
    config.py
    field.py
    kernel.py
    runtime/
      __init__.py
      prepare.py
      produce.py
      validate.py
  digitized_waveform/
    __init__.py
    config.py
    field.py
    kernel.py
    runtime/
      __init__.py
      prepare.py
      produce.py
      validate.py
```

This is an architecture target, not permission to create placeholder modules.
The exact implementation filetree is frozen by the executable work order
after TensorCore publication and the complete Product inventory.

### Common owners

Selected responsibilities are:

```text
common/axis.py
    ExampleAxis
    ChannelAxis
    QuantityAxis
    TimeAxis
    FrequencyAxis

common/field.py
    QuantityFieldSpec

common/kernel.py
    QuantityKernelSpec

common/units.py
    one package Pint registry and scalar quantity admission/conversion

common/alignment.py
    narrow package-owned kernel/source coordinate alignment mechanics
```

No `utils.py`, `helpers.py`, generic effects package, Config framework, Product
registry, workflow graph, or Runtime base is selected.

### Product owners

Each Product package owns:

- public semantic Product;
- public semantic Product Spec;
- public exact Config;
- public exact typed kernel collection;
- public semantic coefficient Specs;
- public computational kernels when applicable;
- private preparation action;
- private production action;
- private validation action; and
- narrower private numerical owners demonstrated by real behavior.

Runtime packages export nothing. Action names remain ordinary readable
implementation names such as:

```text
prepare_charge
produce_charge
validate_charge
```

They are private by facade ownership, not by awkward spelling.

## Public TensorDSLab Surface

The selected supported surface includes:

```text
QuantityAxis
ExampleAxis
ChannelAxis
TimeAxis
FrequencyAxis
QuantityFieldSpec
QuantityKernelSpec

PhotoelectronsSpec
ChargeSpec
PureWaveformSpec
NoiseWaveformSpec
AnalogWaveformSpec
DigitizedWaveformSpec

Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform

ChargeConfig
ChargeKernels
PureWaveformConfig
PureWaveformKernels
NoiseWaveformConfig
NoiseWaveformKernels
AnalogWaveformConfig
AnalogWaveformKernels
DigitizedWaveformConfig
DigitizedWaveformKernels

TimingJitter
DirectCrosstalk
DelayedCrosstalk
Afterpulse
DarkCountRate
SmearingWidth
PulseResponse
WhiteNoiseRms
PowerSpectralDensity
AnalogMinimum
AnalogMaximum
BitDepth
InputMinimum
InputMaximum
AnalogGain

TimingJitterSpec
DirectCrosstalkSpec
DelayedCrosstalkSpec
AfterpulseSpec
DarkCountRateSpec
SmearingWidthSpec
PulseResponseSpec
WhiteNoiseRmsSpec
PowerSpectralDensitySpec
AnalogMinimumSpec
AnalogMaximumSpec
BitDepthSpec
InputMinimumSpec
InputMaximumSpec
AnalogGainSpec

quantity
unit_registry
```

The executable work order freezes the exact root order at `61` names and every
subpackage facade tuple. Tests assert the exact named tuples, not only counts.

## Explicit TensorDSLab Retirements

The future atomic migration retires without alias:

```text
ReadoutConfig
ReadoutRuntime
ReadoutCollection
simulate_readout
prepare_readout
SamplingRuntime
all Product Runtime value classes
ds20k_veto from tensor_dslab
SampleAxis and collaboration-profile axis factories from tensor_dslab
the tensor_dslab.readout workflow package
Pulse
canonical_unit fields/properties
parallel scalar numerical Config values superseded by semantic TensorKernel
leaves with ordinary or quantity Specs
Config-to-Runtime reflection
whole-request prerequisite planning
requested-product workflow closure
```

It must not leave:

- compatibility imports;
- deprecated aliases;
- forwarding modules;
- duplicate old/new facades;
- wrapper factories;
- a hidden universal pipeline;
- a provisional TensorConfig;
- or an application package embedded in the TensorDSLab wheel.

The package is pre-deployment. The clean removal is deliberate.

## RNG Ownership

TensorCore continues to own:

- `RngKey`;
- `RngElements`;
- `RngAddress`;
- `CounterRng`;
- Threefry word generation;
- generic Distribution validation and execution;
- Uniform, Gaussian, Poisson, Binomial, and Multinomial laws; and
- exact low-level address/word invariants.

TensorDSLab Product packages own:

- scientific role names;
- exact role keys;
- application-independent Product address schemas;
- operation/category mapping;
- generation dependency order;
- deterministic source traversal;
- finite-window mapping;
- count accumulation;
- stochastic Product postconditions; and
- any deliberately rebaselined Product result fixtures.

Applications own:

- the root application random domain;
- mapping application sources into Product source elements;
- Product invocation order;
- independent application-level domains; and
- retained result assembly.

No Config stores a mutable RNG, cursor, Distribution factory, or user-selected
role key.

### Rebaseline boundary

The representation and application extraction may change Product traversal or
address construction. Future work orders must distinguish:

```text
preserved:
    TensorCore word identity for exact complete address and ordinal
    Product scientific equations unless explicitly changed
    deterministic same-input replay
    fixed generation dependency order

potentially rebaselined:
    Product address schemas when source/application domain changes
    completed stochastic output bytes
    application retained-result ordering
```

No address or output rebaseline is implied by this Design record. Each exact
change requires:

- old and new schema;
- reason;
- collision proof;
- deterministic fixtures;
- statistical/scientific proof;
- parity classification; and
- retired role disposition.

Because the package is pre-deployment, retired identifiers need not remain
permanently reserved unless an exact future work order selects continuity.
They must never be accidentally reused within one accepted schema.

## Scientific Preservation And Selected Changes

### Preserved laws

Unless a future bounded Product work order says otherwise, preserve:

- white-noise Gaussian law;
- PSD synthesis law;
- dark-count Poisson law;
- charge-smearing Gaussian law;
- timing-jitter Multinomial law;
- collapsed Poisson direct crosstalk;
- collapsed Poisson delayed crosstalk;
- collapsed Poisson full-charge afterpulse;
- same-round frontier semantics;
- fixed-generation recursion;
- finite-window discard;
- deterministic pulse convolution;
- analog summation and saturation;
- digitizer equation and exact code bounds; and
- checked count and allocation ceilings.

### Selected architectural changes

The selected changes are:

- compositional Coordinates instead of representation Axis inheritance;
- Specs as explicit representation values;
- quantity units in Specs;
- direct Product APIs;
- Config punchcards instead of Runtime records;
- semantic kernels for every configurable axis-varying coefficient;
- rank-zero kernels instead of scalar Config coefficient shortcuts;
- one exact typed kernel collection per configurable Product;
- generic ordered source tuples interpreted per Product;
- no package-owned Product graph;
- no generic Readout;
- collaboration-specific axes and profiles outside TensorDSLab core;
- explicit movement and dtype planning;
- no canonical unit encoded by physical kernel class; and
- reusable core package independent of DS20k and Silex workflows.

These changes do not, by themselves, authorize a new scientific result.

The current digitizer gain migration is a representation-only intentional
divergence at the Config boundary. The DS20k application profile must evaluate
the same accepted binary64 mapping
`10.0 ** (3.5218 / 20.0)` exactly once while constructing a rank-zero
AnalogGain kernel. The Product then consumes that literal linear multiplier.
Evidence must prove output equivalence to the current `analog_gain_db=3.5218`
path; moving the conversion out of Product preparation does not authorize a
different gain or digitizer result. The same scalar-to-rank-zero rule preserves
the current uniform bit depth and input bounds exactly while admitting future
conditioned values.

## Documentation Boundary

Future implementation must rewrite living pages to describe:

- compositional Coordinates;
- Specs and exact tensor relationships;
- quantity ownership;
- independent Product use;
- Product-specific Config preparation;
- application ownership;
- exact stochastic responsibilities;
- explicit movement;
- dtype planning; and
- current no-CUDA qualification.

Historical work orders remain historical evidence. Living pages must not keep
the retired `readout/`, Runtime, old Axis, ReadoutCollection, or
simulate_readout architecture as current package guidance.

The current DS20k demo cannot remain a TensorDSLab core demo after application
extraction. A collaboration application may adopt and rewrite it under its own
package authority.

## Cross-Package Sequencing

### Phase 0: TensorDSLab architecture selection

TensorDSLab:

1. freezes this detailed package position;
2. commits only this replacement architecture record and synchronized index;
3. sends the exact immutable document to TensorCore Design; and
4. makes no production edit.

This phase is complete through the synchronized amendments recorded above.

### Phase 1: TensorCore replacement Design

TensorCore Design:

1. started from exact preserved local main `de235057...`;
2. created ordinary forward Stage 31 Design authority
   `25f48e3398c68217b060d94743f8abd810e7f7e8`;
3. froze exact Coordinates, Axis, Spec, Field, Kernel, Collection, movement,
   typing, diagnostics, topology, exports, tests, and artifact contracts;
4. retired unpublished TensorConfig, published old Axis representations, and
   the field-named Collection/validation vocabulary without aliases;
5. consulted exact TensorDSLab and every other affected consumer;
6. resolved every exact consumer finding;
7. left TensorDSLab physics and application policy downstream; and
8. prohibited publication until the complete same-byte package loop cleared.

This phase is complete. TensorCore subsequently selected and published exact
`0.22.0` only after its package gates cleared. Exact consumer confirmation did
not replace those package gates.

### Phase 2: TensorCore implementation and publication

TensorCore independently:

1. implements the exact accepted replacement;
2. validates runtime and strict typing;
3. validates source and canonical archive;
4. builds deterministic artifacts;
5. obtains independent Review;
6. closes locally by exact fast-forward;
7. creates a narrow publication authority;
8. performs an ordinary non-force push; and
9. supplies exact containing commit, tree, version, wheel, archive, export,
   suite, typing, and qualification evidence.

No TensorDSLab adoption occurs merely because TensorCore publishes.

This phase is complete at exact published containing commit
`19bfae35fbc773b55cac7bcd659dda57c4dee6d6`, tree
`53aa10520a50c0714e79c685d814cbae1b6f7740`. TensorDSLab independently
verified and accepted that immutable publication as its future Maintenance 15
dependency target.

### Phase 3: TensorDSLab representation adoption

TensorDSLab Design then freezes one bounded work order for:

- exact dependency pin;
- compositional Coordinates consumption;
- representation-polymorphic ExampleAxis and ChannelAxis;
- QuantityAxis with physical coordinate scale;
- TimeAxis and FrequencyAxis;
- abstract quantity Specs and exact semantic Product/kernel Specs;
- direct Product/physical-leaf specialization of TensorField/TensorKernel with
  their exact semantic Specs;
- explicit movement;
- common alignment;
- facade and typing migration;
- the generic-representation slice of the test-obligation ledger, including
  replacement of the TensorCore 0.21 adoption/typing fixtures;
- protected scientific bytes where possible; and
- removal of obsolete generic representation use.

This may be combined with Product migration only if exact scope and evidence
remain tractable. A representation-only first candidate is preferred if the
Product rewrite would obscure dependency correctness.

### Phase 4: independent Product migrations

TensorDSLab may migrate Products in increasing risk order:

1. PureWaveform as the deterministic convolution pilot;
2. AnalogWaveform;
3. DigitizedWaveform;
4. NoiseWaveform;
5. Charge;
6. Photoelectrons source boundary.

Each stage must leave the package coherent. Temporary compatibility surfaces
are prohibited, so exact grouping may change if an intermediate state could
not be importable or truthful. Each Product stage carries its exact slice of
the test-obligation ledger: the replacement proof lands in the same candidate
that retires the old Product/Runtime/Config surface, and stale tests are not
deferred to a later cleanup stage.

### Phase 5: application retirement and future extraction

Maintenance 15 explicitly retires the currently embedded DS20k profile,
generic readout workflow, result collection, and demos from TensorDSLab core
without inventing a replacement application package. This pre-deployment
retirement is selected in the
[executable work order](maintenance_15_execution_work_order.md). Reusable
Product scientific, stochastic, unit, dtype, and numerical obligations move
to direct Product tests before the old workflow tests are deleted.

A later collaboration package Design authority may:

- select package/repository identity;
- define semantic axes;
- define profiles;
- define workflows;
- adopt reusable TensorDSLab Products;
- create a new DS20k or Silex demo;
- define an application result collection;
- validate wheel isolation; and
- publish, if desired, under separate authority.

The future application is new package-owned work. TensorDSLab neither blocks
its reusable Product migration on a currently nonexistent application nor
silently embeds a permanent application substitute.

## TensorCore Evidence Requirements

The future TensorCore replacement must prove at least:

### Coordinates and axes

- exact public class/decorator/signature contracts;
- one-parameter `TensorAxis[CoordinateT]` typing with downstream exact
  `coordinates`-field narrowing;
- static rejection of the wrong Coordinates representation for a semantic
  Axis leaf;
- structural equality/hash;
- Count, Regular, Label, and Offset admission;
- exact signed-int64 extent-boundary admission and rejection immediately above
  `2**63 - 1`;
- unbounded exact Python-integer Regular coordinate arithmetic within a valid
  extent;
- nonempty unique LabelCoordinates labels;
- zero extents and empty supports;
- negative Regular step generic behavior;
- strict coordinate and index lookup;
- exact contained contiguous windows for every Coordinates representation;
- nonzero Count windows preserving compact start state without renumbering;
- exact Coordinates subtype, semantic Axis subtype, and downstream-field
  preservation through windows;
- exact semantic Axis identity;
- downstream fieldful immutable Axis subclass behavior;
- OffsetAxis `relative_to` class identity;
- multiple OffsetAxis operation dimensions with different target roles;
- retired old Axis surfaces absent; and
- no Pint import.

### Specs

- exact axes/device/dtype identity;
- rank-zero and zero-extent behavior;
- exact shape and Python-integer element count;
- axis lookup;
- complete-axis `axis_at(dimension)` with no ambiguous unqualified KernelSpec
  `dimension_of(...)` or `axis(...)` lookup;
- conditioning and operation uniqueness rules;
- structural equality/hash;
- fieldful downstream Spec subclass support;
- exact FieldSpec `.with_axis(...)` subtype/downstream-field reconstruction and
  supplied Axis identity retention;
- exact same-subclass `.to`;
- no-op identity;
- changed `.to` reconstruction rerunning the existing most-derived semantic
  validation exactly once;
- a dtype/device change that violates a downstream Spec requirement failing
  during reconstruction;
- no allocation or availability check;
- no lost downstream fields; and
- no TensorConfig.

### Fields and kernels

- exact tensor/Spec agreement;
- fail-closed mismatch diagnostics;
- semantic subtype preservation;
- gradient-bearing Field reference admission without detachment;
- ordinary autograd connectivity across differentiable changed Field `.to`;
- consumer-owned zero-copy tensor views paired with exact window Specs;
- Field no-op `.to` identity;
- changed Field `.to` rerunning existing most-derived semantic validation
  exactly once, including a narrowing cast that creates a nonfinite semantic
  value;
- Kernel defensive ownership;
- Kernel no-op `.to` identity;
- changed Kernel `.to` rerunning existing most-derived semantic validation
  exactly once without a duplicate defensive snapshot;
- safe exact-subtype movement without public trust surface;
- unhashability/identity equality;
- no unit or Product policy; and
- no silent repair.

### Collections

- empty, field-only, kernel-only, and explicit mixed collections;
- exact-type uniqueness and lookup;
- insertion order;
- heterogeneous axes/device/dtype/unit;
- device-only `.to`;
- no-op identity;
- exact subtype preservation;
- changed Collection `.to` rerunning its existing most-derived semantic
  validation exactly once;
- complete absence of `fields`, `field_types`, `field`, `tensor`, and
  `require_field_types` supported surfaces;
- sole `members`, `member_types`, and `member` vocabulary;
- no reflection;
- no artifact generalization; and
- exact field-only `TensorArtifact.materialize` return typing, with strict
  static rejection of artifact implementations that return Kernel or mixed
  collections.

### Static and artifact

- exact supported exports and modules;
- strict positive typing;
- negative fixtures for wrong Coordinates, axes, Specs, members, movement,
  subclass state, and lookup;
- source/archive equality;
- deterministic wheel payload;
- isolated install;
- no retired path;
- no compatibility alias;
- docs and examples compile;
- privacy and clean-tree gates; and
- explicit unavailable-CUDA qualification.

## TensorDSLab Evidence Requirements

Future TensorDSLab implementation must prove:

### Computational coefficient kernels

- every configurable numerical coefficient is represented by one exact
  semantic TensorKernel leaf;
- no public Config constructor accepts a caller-supplied configurable
  coefficient as a primitive scalar, Pint Quantity, or raw tensor;
- prepared Config scalar conversion scales, resolved dimensions, dtypes, and
  checked ceilings are demonstrably derived immutable facts rather than
  caller-configurable shortcuts;
- any persistent derived tensor is either an exact semantic Kernel/Spec value
  or follows an explicitly tested Product-private logical-read-only ownership
  contract;
- one rank-zero global kernel and one or more conditioned representations
  produce the expected broadcast-equivalent or deliberately varying result;
- example-, channel-, microcell-, sample-, and combined-conditioning fixtures
  for every Product that admits those roles;
- exact conditioning-coordinate permutation and absent-role rejection;
- ordinary pointwise coefficients reject operation axes;
- support-bearing kernels retain exact operation geometry;
- quantity coefficients use TensorKernel with QuantityKernelSpec while exact
  discrete coefficients such as BitDepth use TensorKernel with ordinary
  TensorKernelSpec;
- the five Product-specific `*Kernels` collections enforce exact required,
  optional, duplicate, and alien-member contracts;
- collection movement preserves exact semantic member types and Config
  preparation returns aligned same-type heterogeneous collections without
  silently homogenizing member dtypes;
- arithmetic-admissible coefficient kernels follow the Product's explicit
  representation-to-working conversion plan while discrete kernels retain
  their exact semantic representation dtype;
- structural Config state such as recursion depth remains outside kernels;
  semantic time/frequency relationships are derived from exact Specs; and
- no generic ParameterKernel, coefficient registry, reflection dispatch, or
  scalar compatibility spelling exists.

### Quantity representation

- exact Pint registry ownership;
- QuantityAxis integer-lattice, coordinate-scale, Unit, and quantity boundary;
- exact built-in binary64, finite, strictly positive coordinate-scale
  admission and hash/equality behavior;
- physical coordinate evaluation as integer lattice magnitude multiplied by
  coordinate scale and Unit;
- canonical regular TimeAxis/FrequencyAxis `step == 1` representation;
- representation-polymorphic ExampleAxis and ChannelAxis identity;
- Count/Label ExampleAxis and ChannelAxis values retaining one exact semantic
  role;
- OffsetCoordinates used as explicit identifiers without displacement meaning;
- exact TimeAxis and FrequencyAxis semantics;
- QuantityAxis rejection of LabelCoordinates at runtime and through strict
  typing;
- semantic Product and physical-kernel Spec structural equality/hash;
- common quantity-Spec validation followed by exactly one semantic-Spec hook;
- exact Config rejection of a generic or wrong semantic Spec subtype;
- direct Product specialization of `TensorField[ItsProductSpec[...]]`;
- direct physical-leaf specialization of
  `TensorKernel[ItsKernelSpec[...]]`;
- complete absence of `QuantityField` and `QuantityKernel` classes, exports,
  aliases, and compatibility imports;
- exact Field/Kernel Spec typing and unit access through `value.spec.unit`;
- semantic QuantityFieldSpec and QuantityKernelSpec subtypes as the sole
  stored Product/kernel unit owners;
- no duplicate unit state;
- no canonical_unit;
- unit-preserving `.to`;
- no implicit unit conversion;
- and no Pint tensor state.

### Preparation

- same-exact-Config return type;
- fresh Config when representation changes;
- complete all-source dimensional compatibility before any source conversion,
  cast, movement, summation, allocation, or RNG request;
- accept differently scaled but dimensionally compatible source units and
  freeze one exact conversion per source;
- reject a multi-source `[avalanche] + [mV]` mutant at the intended source
  index before effects;
- require every source device to equal every other source device and the exact
  output Spec device;
- reject mixed `cpu` / `cuda:0` sources and same-device sources targeting a
  different output device before unit conversion, dtype casts, kernel
  movement, summation, allocation, or RNG words;
- prove that `create()` and `produce()` never call source `.to(...)`;
- exact no-op behavior if selected;
- source-Spec relationship;
- conditional exact TimeAxis admission for enabled Charge temporal laws;
- PSD selection requiring one unique TimeAxis independent of dimension order;
- PowerSpectralDensitySpec requiring exactly one regular FrequencyAxis
  operation axis;
- exact RFFT count and reciprocal time/frequency spacing validation;
- application-owned pre-hot-path time/frequency grid and PSD preparation;
- no Config temporal-axis role field;
- coordinate correspondence;
- conditioning permutation;
- operation geometry preservation;
- exact typed kernel-collection admission;
- rank-zero and conditioned coefficient alignment through one path;
- unit conversion;
- dtype fold and numerical floor;
- target device materialization;
- exact preflight order;
- no RNG words;
- no PSD interpolation, integration, resampling, grid construction, or Pint
  arithmetic in production;
- no Runtime/Plan/token;
- and mutation-resistant failures for omitted coordinate reorder, omitted
  dimension permutation, wrong unit scale, wrong dtype floor, and silent
  movement.

### Products

- one-shot and staged path equivalence;
- exact output Spec identity;
- direct standalone use;
- Product-specific source tuple law;
- deterministic source-order accumulation;
- no pipeline dependency;
- no private Runtime values;
- exact public signatures;
- exact scientific laws;
- exact stochastic replay;
- Product result freshness;
- Product validation;
- and no production-time Pint, alignment discovery, movement, or dtype-policy
  selection.

### Charge evidence

- Photoelectrons-only source;
- Axioelectrons-like second semantic source fixture without production import;
- source axis-order permutation;
- source unit conversion;
- source accumulation ceilings;
- global and conditioned kernels;
- exact ChargeKernels member-set admission;
- absent-role rejection;
- exact temporal-axis admission;
- timing conservation before finite-window discard;
- direct/delayed/afterpulse collapsed means;
- no same-round feedback;
- finite-window exclusion;
- generation depth;
- smearing;
- stochastic address identity;
- and high-strength independent analytic/statistical oracles.

### Waveforms and digitizer

- exact PureWaveformKernels, NoiseWaveformKernels, AnalogWaveformKernels, and
  heterogeneous DigitizedWaveformKernels contracts;
- PulseResponse literal polarity and unit equation;
- convolution geometry;
- multiple compatible PureWaveform sources;
- exact-zero noise;
- white and PSD branches;
- Analog source summation and saturation;
- Digitized exact-one-source law;
- rank-zero and conditioned BitDepth/InputMinimum/InputMaximum/AnalogGain
  mapping;
- exact linear-gain equivalence for the retained profile value converted from
  decibels at profile construction;
- pointwise input-range rejection;
- exact-integer pointwise maximum-code derivation before a checked floating
  conversion for scaling;
- BitDepth integer representation preservation through preparation and
  production;
- pointwise maximum-code, floating-exactness, and output-dtype ceiling
  preflight;
- output integer dtype;
- and configured unit freedom.

### Package and application boundary

- TensorDSLab core imports no collaboration package;
- core exports no collaboration-specific axes, profile, or Readout;
- exact reusable Product facades;
- exact wheel isolation;
- application imports only supported TensorDSLab surface;
- application-specific workflow evidence remains in application package;
- no placeholder Silex waveform chain;
- no embedded hidden application;
- and no retired aliases.

## TensorDSLab Test-Suite Reconciliation

Maintenance 15 changes enough public and internal architecture that its tests
must be deliberately reconciled with the replacement. The current
Maintenance 14 suite is valid evidence for the current production baseline;
it is not a requirement to retain tests whose subject no longer exists.

The accepted Maintenance 14 source/archive total was:

```text
305 tests run
302 passed
3 conditional unavailable-CUDA skips
```

Those totals are evidence, not a target count. No replacement test may assert
a repository-wide test-method, file, line, package-module, or import-edge
count. Validation reports actual discovered totals and proves obligations.

### Obligation ledger before deletion

The executable work order inventories every current test module and maps each
substantive obligation to exactly one disposition:

```text
retain unchanged
rewrite against the replacement public contract
move to an application-owned package with synchronized evidence
merge into one stronger test
retire because the tested surface is deliberately absent
```

No test disappears merely because its import path or fixture is inconvenient.
A deletion is accepted only when the ledger identifies the replaced
obligation, the exact new proof, or the explicit architectural retirement.
Scientific and stochastic proofs survive independently of the class/module
layout that currently hosts them.

The ledger must distinguish:

- public contract tests;
- generic representation and quantity tests;
- Product preparation/production/validation tests;
- independent scientific oracles and statistical tests;
- deterministic RNG address/word/replay tests;
- strict positive and negative typing fixtures;
- package topology, facade, isolation, and retired-path tests;
- artifact/source/archive/environment evidence; and
- collaboration-specific profile, workflow, demo, and notebook tests that
  require an application owner.

### Required replacements

The following current test subjects are architectural migration targets, not
permanent compatibility surfaces:

- TensorCore 0.21 Axis/Field/Kernel/Collection adoption becomes exact
  published Stage 31/0.22 adoption evidence;
- current physical-kernel geometry tests become composed
  Coordinates/Axis/Spec/Kernel and coefficient-leaf tests, including absence
  of `QuantityKernel`;
- Runtime preparation/alignment/action-ownership tests become same-type
  Config preparation and Product classmethod tests;
- `ReadoutConfig`, `ReadoutCollection`, `simulate_readout()`, and DS20k
  profile/demo tests move to the accepted application owner or retire only
  after synchronized replacement evidence;
- current `Pulse` assertions become literal `PulseResponse` tests;
- current scalar Config/digitizer tests become rank-zero and conditioned
  Product-specific kernel tests;
- current package-contract tests replace 0.21 exports and paths with the exact
  new facade, Spec, Product, and retired-surface inventory; and
- historical typing fixtures that import retired public names are replaced by
  focused current positive/negative fixtures rather than kept through aliases.

The executable work order freezes the exact file deletions, renames, merges,
and new paths. The package must not temporarily weaken coverage while waiting
for a later cleanup stage.

### Selected replacement organization

The future suite should mirror supported concepts rather than reproduce the
old production filetree mechanically. Its first-order ownership is:

```text
generic TensorCore adoption and composed representations
TensorDSLab quantity Specs and direct Product/kernel specialization
Product-specific Config and typed kernel collections
Product prepare / produce / validate / create equivalence
independent Product scientific laws
stochastic identity and deterministic replay
application/core isolation
public typing, facades, artifacts, and environment
```

Product modules may split deterministic contract tests from expensive
statistical/scientific proofs when that improves navigability. Shared private
test support may own immutable fixtures and independent oracle calculations,
but it must not own discovered tests, import target test modules, depend on
production preparation helpers for expected values, or become a generic
testing framework.

Use explicit test methods and clear table-driven `subTest` cases where several
semantic leaves share one genuine contract. Do not dynamically attach dozens
of near-identical methods, retain no-op assertions, or clone cases that change
only a magnitude/index without increasing mutation resistance.

### Required preserved proof strength

Reconciliation must preserve or strengthen:

- every accepted scientific equation and parity boundary;
- exact source order, unit conversion, device admission, coordinate
  permutation, and dtype-plan behavior;
- each configured global and conditioned coefficient law;
- stochastic role/address identity and same-input replay;
- independent analytic/statistical Charge and noise oracles;
- exact Product one-shot/staged equivalence and output Spec identity;
- strict rejection of retired imports and compatibility aliases;
- public facade and subclass typing;
- source/archive/package-byte equality and isolated-wheel behavior; and
- explicit unavailable-CUDA qualifications.

Tests should assert stable observable behavior and exact supported type
surfaces, not incidental private call graphs, helper names, line numbers, or
module counts. A private implementation detail receives a direct test only
when it is itself the narrowest owner of a mandatory effect such as defensive
ownership, host observation, or RNG-word use.

### Mutation and evidence cadence

The executable work order names an exact bounded mutation set for the
highest-risk replacement boundaries. It kills:

- skipping source-unit compatibility or same-device admission;
- resolving roles by name rather than exact semantic type;
- omitting conditioning-coordinate reordering or dimension permutation;
- accepting a caller scalar/Pint/raw-tensor coefficient shortcut;
- homogenizing a heterogeneous kernel collection;
- casting integer BitDepth to the floating working dtype;
- bypassing most-derived validation during Spec/Field/Kernel/Collection
  movement;
- duplicating the defensive Kernel snapshot;
- restoring Runtime or generic Readout orchestration;
- weakening timing conservation or collapsed branching means; and
- importing a collaboration profile into TensorDSLab core.

Implementation runs focused changed-area tests and reports the obligation
ledger. Validation owns one complete final source and canonical-archive gate,
strict typing and negative fixtures, artifact/isolation checks, scientific
oracles, and the selected mutants. Risk-based Review audits the ledger,
high-risk public/scientific boundaries, and proof strength without rerunning
every complete gate. Documentation-only corrections carry executable evidence
forward only when every executable/test/dependency byte is identical.

## Risk-Based Review

Independent Review must focus on the highest-risk boundaries rather than
mechanically duplicate every complete Validation gate:

- generic versus package-owned ownership;
- Coordinates/Axis semantic identity;
- Spec structural state and same-subclass reconstruction;
- Field/Kernel exact Spec relationship;
- kernel defensive ownership;
- collection heterogeneity;
- unit equations;
- dtype planning;
- Product source relationships;
- Config readiness;
- no hot-path policy discovery;
- Charge stochastic laws and address identity;
- application/core isolation;
- public typing;
- retired-surface absence; and
- truthfulness of current living documentation.

Review should use targeted mutants for plausible incorrect implementations.
Examples include:

- treating coordinate order as irrelevant;
- accepting CountCoordinates where a semantic leaf narrows its representation
  to LabelCoordinates;
- resolving semantic roles by class name rather than exact class;
- losing downstream Spec fields in `.to`;
- reconstructing a base Field or Kernel;
- bypassing subtype validation after a narrowing dtype cast;
- duplicating the Kernel defensive snapshot during movement;
- accepting Kernel or mixed members from TensorArtifact materialization;
- retaining a field-named Collection accessor as an alias;
- retaining a scalar or Pint numerical coefficient directly in a Product
  Config;
- accepting a scalar/raw-tensor shortcut for a caller-configurable coefficient
  merely because prepared Configs may retain derived execution facts;
- casting integer BitDepth to the Product floating working dtype or
  homogenizing the heterogeneous digitizer collection;
- deriving maximum code through floating exponentiation rather than checked
  exact integer arithmetic;
- treating a conditioned coefficient as one global scalar;
- accepting a coefficient conditioned on an absent semantic role;
- admitting operation geometry on an ordinary pointwise coefficient;
- replacing integer BitDepth with a TensorKernel whose QuantityKernelSpec
  supplies a floating representation;
- interpreting linear AnalogGain values as decibels on the Product hot path;
- silently moving a source during production;
- inferring a canonical unit from kernel class;
- using output dtype below the Product floor;
- normalizing timing probabilities;
- using same-generation children as a frontier;
- drawing out-of-window crosstalk categories;
- retaining an embedded DS20k import;
- and accepting a retired Axis alias.

## CUDA And Accelerator Boundary

No fresh CUDA action is authorized by this Design.

During pre-1.0 development:

- complete CPU behavior;
- strict typing;
- exact address/word evidence;
- device-contract tests;
- source/archive/artifact evidence;
- and explicit unavailable-CUDA qualifications

remain mandatory.

The complete integrated CUDA matrix remains deferred to one exact mutually
adopted TensorCore/TensorDSLab `1.0.0` release-candidate pairing. Any
collaboration application seeking its own accelerator claim must independently
validate its exact package bytes and workflow. The final coordinated CUDA gate
is functional-correctness evidence, not performance, deployment, calibration,
or broad hardware conformance.

## Frozen Non-Goals

This architecture does not select:

- TensorArtifact generalization;
- durable cache schema;
- IO;
- lazy loading;
- TensorML integration;
- TensorG4DS adapter implementation;
- native G4DS parsing;
- detector calibration;
- arbitrary workflow graphs in TensorDSLab;
- a universal Product base;
- a generic Config base;
- a generic ParameterKernel or CoefficientKernel base;
- a generic Readout base;
- a Product registry;
- reflection-driven execution;
- Distribution factories in Config;
- callbacks;
- arbitrary user RNG keys;
- mutable preparation caches;
- per-product Runtime classes;
- PSD construction;
- recovery-weighted afterpulse;
- dynamic-shape kernel geometry;
- sparse kernel representation;
- per-element algorithm or recursion-topology selection through a coefficient
  kernel;
- implicit unit conversion;
- implicit device movement;
- automatic mixed precision;
- performance optimization;
- CUDA implementation;
- compatibility aliases;
- a package release;
- or a deployment claim.

## Stop Conditions

Future work stops and returns to the relevant Design authority if:

1. TensorCore declines or substantively changes the compositional contract.
2. TensorCore cannot safely preserve exact Spec/Field/Kernel/Collection
   subclasses and rerun their existing semantic validation exactly once in
   `.to(...)` without a public unchecked mechanism or duplicate Kernel
   snapshot.
3. TensorCore requires TensorDSLab units, Products, Configs, or application
   policy upstream.
4. The one-parameter TensorAxis plus downstream narrowed `coordinates` field
   cannot statically reject the wrong Coordinates representation, or a second
   exact axis role requires restoring parallel Axis inheritance vocabulary.
5. Fieldful downstream Specs cannot coexist with fieldless semantic
   Field/Kernel leaves.
6. TensorCollection cannot preserve exact semantic subtype/validation during
   movement, or TensorArtifact cannot retain a statically field-only
   materialization boundary.
7. TensorDSLab production would need Pint, coordinate search, silent movement,
   or dtype-policy discovery.
8. A Product cannot express its source law without importing an application
   semantic Product class.
9. The selected Charge source relationship changes its scientific law.
10. A physical kernel requires a probability-only generic hierarchy.
11. A configurable numerical coefficient cannot be represented by one exact
    semantic kernel without changing the Product algorithm, or a future
    implementation retains a caller-supplied scalar/Pint/raw-tensor
    coefficient Config shortcut.
12. Application extraction lacks an accepted package owner or creates a
    dependency cycle.
13. A future work order attempts to publish the unpublished Stage 30
    TensorConfig contract unchanged.
14. A stochastic address or result changes without an explicit parity and
    collision disposition.
15. Protected scientific behavior changes inside a structural migration
    without package-owned authority.
16. A compatibility alias or hidden old path is introduced without explicit
    user and Design acceptance.
17. Production scope begins before the exact TensorCore containing commit is
    published and independently accepted.
18. An execution role named by a future production work order is stale,
    missing, or discrepant.
19. Any package source conflicts with a cross-package handoff.
20. The implementation cannot keep application bytes out of the reusable
    TensorDSLab wheel.
21. The exact paired pre-1.0 CUDA deferral is misrepresented as accelerator
    support.

No implementation role may silently narrow, widen, or reinterpret the
architecture to bypass a stop.

## Executable Work-Order Inventory

The exact executable inventory is frozen in
[Maintenance 15 Executable Work Order](maintenance_15_execution_work_order.md).
It includes:

- exact published TensorCore commit/tree/version/artifacts;
- exact TensorDSLab parent commit/tree;
- exact package filetree;
- exact public facade tuples;
- exact Config fields and constructor signatures;
- exact five Product-specific kernel-collection decorators, member contracts,
  typed properties, and constructor signatures;
- exact computational-kernel decorators, Specs, dtype/unit/value domains, and
  conditioning/operation geometry;
- exact QuantityAxis, ExampleAxis, ChannelAxis, TimeAxis, and FrequencyAxis
  decorators, generics, representation admission, coordinate-scale contract,
  and signatures;
- exact abstract QuantityFieldSpec/QuantityKernelSpec and all concrete
  Product/coefficient Spec decorators, generics, hooks, and signatures;
- exact direct Product/physical-kernel TensorField/TensorKernel bases and
  absence of QuantityField/QuantityKernel roots;
- exact Product classmethod signatures;
- exact preparation fields and readiness diagnostics;
- exact per-Product working-dtype floors;
- exact per-member representation preservation and arithmetic-conversion
  plans, including every representation-preserved discrete kernel;
- exact ownership and construction exclusion for every stored derived Config
  fact;
- exact unit equations;
- exact source tuple relationships;
- exact semantic TimeAxis/FrequencyAxis and Charge/PSD relationship rules;
- exact kernel value and geometry constraints;
- exact source and kernel movement policy;
- exact current-test obligation ledger and disposition for every test module;
- exact replacement test filetree, focused groups, and shared-support owners;
- exact retained scientific/statistical/RNG obligations and their new proof
  locations;
- exact positive/negative typing fixture replacement;
- exact test deletions/renames/splits and the no-alias retirement proofs;
- exact bounded mutation set and evidence cadence;
- exact stochastic roles, addresses, and any rebaseline;
- exact count/allocation ceilings;
- exact current scientific fixtures;
- exact retirements;
- exact application extraction boundary;
- exact changed-path allowlist and protected bytes;
- focused and complete test commands;
- strict typing and negative fixtures;
- source/archive/artifact evidence;
- isolated wheel evidence;
- documentation updates;
- privacy and hygiene gates;
- risk-based Review duties;
- finite candidate/return route;
- final same-byte approval and fast-forward authority; and
- explicit no-push/no-CUDA effects unless separately authorized.

## Authority And Next Action

This document selects TensorDSLab's architecture and exact consumer position.
The separate
[Maintenance 15 Executable Work Order](maintenance_15_execution_work_order.md)
owns production scope, evidence, routing, and dispatch. This architecture
record authorizes only:

1. committing this Design record and synchronized implementation index;
2. sending the exact immutable record to TensorCore Design;
3. receiving and reviewing a TensorCore replacement Design candidate;
4. refining this record if cross-package contracts require a substantive
   synchronized change; and
5. recording exact dependency publication;
6. preparing the bounded executable Maintenance 15 work order; and
7. synchronizing this architecture record with that exact work order.

It does not authorize:

- TensorCore edits;
- TensorDSLab production or test edits;
- dependency adoption;
- application repository creation;
- Implementation dispatch;
- Validation or Review dispatch;
- merge to local main;
- push;
- publication;
- CUDA or cluster work;
- compatibility claims;
- release claims;
- or deployment.

TensorCore's required Stage 31 sequence is complete. TensorDSLab's executable
work order is now frozen against exact published `19bfae3`. Design must commit
those exact bytes, verify every required Implementation/Validation/Review
route, and dispatch the committed authority. This architecture record alone
does not dispatch any execution role.

TensorDSLab remains on exact published TensorCore `0.21.0` and the current
Maintenance 14 production package until an exact future Maintenance 15
candidate adopts and independently clears `0.22.0`.
