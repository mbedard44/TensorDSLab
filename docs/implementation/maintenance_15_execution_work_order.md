# Maintenance 15 Executable Work Order

Status: **Self-effecting under the frozen exact-byte route**.

Before the complete same-byte package gate and clean local-`main`
fast-forward, the applicable state is the latest completed exact-byte handoff.
After the exact candidate appears unchanged on local `main` through Review's
clean fast-forward, Maintenance 15 is **Merged / Closed**.

Stable key:
`TensorDSLab/maintenance-15-spec-composed-products-and-application-boundary`

This is the executable authority for
[Maintenance 15](maintenance_15_spec_composed_products_and_application_boundary.md).
The architecture record remains the detailed semantic source of truth. This
work order freezes the exact dependency, target package, public API, execution
scope, test reconciliation, evidence cadence, candidate route, and effects
required before production work may begin.

## Purpose

Maintenance 15 atomically converts TensorDSLab from one embedded readout
workflow into a reusable parts bin of independently callable tensor Products.
It:

- adopts exact published TensorCore `0.22.0`;
- composes quantity meaning through `QuantityFieldSpec` and
  `QuantityKernelSpec`;
- makes Product and physical-coefficient leaves directly specialize
  TensorCore `TensorField` and `TensorKernel`;
- represents every caller-configurable axis-varying numerical coefficient as
  one semantic Kernel;
- replaces scalar/quantity Config coefficients and compiled Runtime records
  with same-type Config punchcards;
- gives Products exact `create`, `prepare`, `produce`, and `validate`
  classmethod boundaries;
- moves reusable Product packages out of the application-shaped `readout/`
  namespace;
- retires the embedded DS20k profile, generic readout orchestration, result
  collection, and demos without compatibility aliases;
- preserves the selected Product scientific laws and RNG role identities;
- deliberately reconciles the tests around supported concepts instead of
  retaining tests for retired architecture; and
- leaves application workflow, collaboration axes, profiles, end-to-end
  demos, IO, and CUDA integration to later package-owned work.

This is a pre-deployment clean replacement. No deprecation window, alias,
forwarding module, wrapper factory, or parallel old/new API is authorized.

## Governing Sources

Implementation, Validation, and Review must read:

- [AGENTS](../../AGENTS.md);
- [CONTRIBUTING](../../CONTRIBUTING.md), especially TensorCore ownership,
  coordinates versus indices, Product semantics, public-surface discipline,
  boundary-first validation, scope, tests, and documentation;
- [Architecture record](maintenance_15_spec_composed_products_and_application_boundary.md);
- [Tensor architecture](../architecture/tensors.md);
- [Readout architecture](../architecture/readout.md);
- [Design](../design.md);
- [Validation](../validation.md);
- [Parity](../parity.md);
- [Maintenance 12](maintenance_12_tensorcore_0_21_kernel_geometry_quantity_refactor.md)
  for the current literal-kernel scientific baseline;
- [Maintenance 13](maintenance_13_runtime_hygiene_and_environment_reproducibility.md)
  for current alignment and environment evidence; and
- [Maintenance 14](maintenance_14_test_suite_curation.md) for the current
  curated-suite baseline and evidence-economy rules.

Package sources and this exact work order take precedence over informal
handoffs. A conflict returns to Design.

## Exact Design Authority And Baseline

This work order is committed as an ordinary descendant of:

```text
TensorDSLab architecture publication-binding commit:
    703f95c067fc1413f155594e71f4d4a4f1f9e142
tree:
    2ae66ae48db12742689c57ace23cadb6fa0cc0f6
architecture synchronization parent:
    bb5bed89c0b40b4bcde786e5c28afc610071304f
local governed main:
    856df702c124365c929bf993851a51fb8ff3c245
local-main tree:
    9e5ff69920699dc522980b164eaf1073116914c6
live origin/main at Design time:
    c8de1528d1ed57d3e86a9c37d1ad307127a23feb
package version before Maintenance 15:
    0.1.0
```

The architecture branch is an ordinary linear descendant of governed local
`main`. Implementation must use the committed form of this work order as its
exact parent. It must not reset, rebase, squash, amend, or rewrite the accepted
Design history.

The current accepted Maintenance 14 evidence is:

```text
source and canonical archive:
    305 tests run
    302 passed
    3 conditional unavailable-CUDA skips
Pyright:
    0 errors / 0 warnings / 0 informations
```

These totals are baseline evidence, not future count requirements.

## Exact TensorCore Dependency

Maintenance 15 replaces the current TensorCore pin with:

```text
repository:
    https://github.com/mbedard44/TensorCore.git
version:
    0.22.0
containing and implementation commit:
    19bfae35fbc773b55cac7bcd659dda57c4dee6d6
tree:
    53aa10520a50c0714e79c685d814cbae1b6f7740
prior published 0.21.0 commit:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
```

Accepted dependency artifacts are:

```text
wheel size:
    54,052 bytes
wheel SHA-256:
    6ac2f29c562504d7e87e1caf404b10019b08d60252fc496ad55b090e6b8b154f
commit-bound source archive size:
    1,095,680 bytes
archive SHA-256:
    deb09f72595a44f3b8551f01971986aa265a28a3f4475ee2afe59fb2b63f0c84
```

The exact dependency contract is:

```text
34 package-root exports
13 tensor exports
15 tensor.validation exports
41 package files
40 Python modules
85 source tests OK
2 accepted unavailable-CUDA skips
0 positive Pyright diagnostics
97 intentional dependency-negative diagnostics
```

Validation must independently reconstruct the exact commit-bound archive and
wheel identity. A GitHub-generated archive is not a substitute for the
commit-bound archive.

## Target Package Version And Dependencies

The candidate package version is `0.2.0`.

The exact dependency lines remain:

```text
Python >=3.14
NumPy ==2.5.1
Pint ==0.25.3
Torch >=2.13,<2.14
Hatchling ==1.31.0
Pyright ==1.1.411 for evidence
TensorCore exact commit 19bfae35fbc773b55cac7bcd659dda57c4dee6d6
```

The `demos` optional dependency group is retired because TensorDSLab core no
longer owns an application demo. No replacement dependency is added.

The distribution is not published, tagged, released, or uploaded by this work
order.

## Exact Target Package Tree

The exact tracked `tensor_dslab/` target is `56` package files and `55` Python
modules:

```text
tensor_dslab/
  __init__.py
  py.typed
  common/
    __init__.py
    alignment.py
    axis.py
    field.py
    kernel.py
    units.py
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
      branching.py
      counts.py
      prepare.py
      produce.py
      random.py
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
    __init__.py
    config.py
    field.py
    kernel.py
    runtime/
      __init__.py
      prepare.py
      produce.py
      random.py
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

No `tensor_dslab/readout/` path survives. No placeholder application,
reconstruction, artifact, IO, workflow, registry, generic Product, generic
Config, Runtime base, effect package, or utility module is added.

`charge/runtime/random.py` and `noise_waveform/runtime/random.py` own only
their Product's fixed role keys and exact address factories. They are not RNG
wrappers and do not duplicate TensorCore word generation or distributions.

Runtime `__init__.py` files import and export nothing.

## Exact Public Facades

The package root exports exactly this ordered `36`-name tuple:

```python
(
    "Afterpulse",
    "AnalogGain",
    "AnalogMaximum",
    "AnalogMinimum",
    "AnalogWaveform",
    "AnalogWaveformConfig",
    "AnalogWaveformKernels",
    "BitDepth",
    "Charge",
    "ChargeConfig",
    "ChargeKernels",
    "DarkCountRate",
    "DelayedCrosstalk",
    "DigitizedWaveform",
    "DigitizedWaveformConfig",
    "DigitizedWaveformKernels",
    "DirectCrosstalk",
    "InputMaximum",
    "InputMinimum",
    "NoiseWaveform",
    "NoiseWaveformConfig",
    "NoiseWaveformKernels",
    "Photoelectrons",
    "PowerSpectralDensity",
    "PulseResponse",
    "PureWaveform",
    "PureWaveformConfig",
    "PureWaveformKernels",
    "QuantityAxis",
    "QuantityFieldSpec",
    "QuantityKernelSpec",
    "SmearingWidth",
    "TimingJitter",
    "WhiteNoiseRms",
    "quantity",
    "unit_registry",
)
```

Exact subpackage facade tuples are:

```text
tensor_dslab.common:
    QuantityAxis
    QuantityFieldSpec
    QuantityKernelSpec
    quantity
    unit_registry

tensor_dslab.photoelectrons:
    Photoelectrons

tensor_dslab.charge:
    Afterpulse
    Charge
    ChargeConfig
    ChargeKernels
    DarkCountRate
    DelayedCrosstalk
    DirectCrosstalk
    SmearingWidth
    TimingJitter

tensor_dslab.pure_waveform:
    PulseResponse
    PureWaveform
    PureWaveformConfig
    PureWaveformKernels

tensor_dslab.noise_waveform:
    NoiseWaveform
    NoiseWaveformConfig
    NoiseWaveformKernels
    PowerSpectralDensity
    WhiteNoiseRms

tensor_dslab.analog_waveform:
    AnalogMaximum
    AnalogMinimum
    AnalogWaveform
    AnalogWaveformConfig
    AnalogWaveformKernels

tensor_dslab.digitized_waveform:
    AnalogGain
    BitDepth
    DigitizedWaveform
    DigitizedWaveformConfig
    DigitizedWaveformKernels
    InputMaximum
    InputMinimum
```

Root imports are the supported golden path. Subpackage facades are deliberate
supported precision paths. Runtime actions, RNG roles, alignment functions,
private prepared facts, and validators are not exported.

## Quantity Representations

### `QuantityAxis`

`QuantityAxis[CoordinatesT: Coordinates[int]]` is an abstract frozen, slotted,
keyword-only, fieldful `TensorAxis[int]` specialization with:

```text
coordinates: CoordinatesT
unit: pint.Unit
```

It normalizes the exact package-registry Unit, retains TensorCore coordinate
behavior, and provides:

```text
quantity_at(index: int) -> pint.Quantity
quantity_of(magnitude: int) -> pint.Quantity
```

It owns no application semantic leaf. Example, channel, sample, and microcell
axes are absent from TensorDSLab.

### `QuantityFieldSpec`

`QuantityFieldSpec` is final, directly constructible, frozen, slotted,
keyword-only, structurally equal, and hashable. Its exact public constructor
state is:

```text
axes: tuple[TensorAxis[Any], ...]
device: torch.device
dtype: torch.dtype
unit: pint.Unit
```

It directly specializes TensorCore `TensorFieldSpec`, normalizes one package
Unit, and retains exact subtype and unit across `with_axis(...)` and
`to(...)`.

### `QuantityKernelSpec`

`QuantityKernelSpec` is the final kernel counterpart with exact public state:

```text
conditioning_axes: tuple[TensorAxis[Any], ...]
operation_axes: tuple[TensorAxis[Any], ...]
device: torch.device
dtype: torch.dtype
unit: pint.Unit
```

It directly specializes `TensorKernelSpec`, retains TensorCore structural
contracts, and preserves exact subtype and unit across `to(...)`.

There is no `QuantityField`, `QuantityKernel`, `TensorConfig`,
`ParameterKernel`, or `CoefficientKernel`.

`unit_registry` is the one TensorDSLab Pint registry. `quantity(...)` remains
the scalar construction convenience. The tensor-valued `quantities(...)`
shortcut is retired: tensor magnitudes and unit state enter through exact
TensorCore tensor-plus-Spec construction.

## Product And Kernel Value Contracts

The six Product leaves are final, fieldless, frozen/slotted through their
TensorCore root, identity-equal, unhashable values with `__slots__ = ()`:

```text
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
```

Each directly specializes `TensorField[QuantityFieldSpec[Any]]`. Unit is read
only through `product.spec.unit`.

The fifteen coefficient leaves are final, fieldless, direct TensorKernel
specializations with `__slots__ = ()`:

```text
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

All except `BitDepth` specialize
`TensorKernel[QuantityKernelSpec[Any, Any]]`. `BitDepth` specializes
`TensorKernel[TensorKernelSpec[Any, Any]]` and admits only exact integer
representation dtypes selected by its semantic contract.

Product and coefficient leaves add no fields, unit property, canonical-unit
class state, constructor, factory, distribution, callback, or execution
method beyond the Product classmethods explicitly selected below.

## Exact Kernel Collections

Every `*Kernels` value is final, fieldless, slotted, identity-equal,
unhashable, and inherits the exact TensorCore constructor:

```text
CollectionType(*, members: Iterable[TensorKernel[Any]])
```

Exact membership is:

```text
ChargeKernels:
    zero or one each of
    TimingJitter
    DirectCrosstalk
    DelayedCrosstalk
    Afterpulse
    DarkCountRate
    SmearingWidth

PureWaveformKernels:
    exactly one PulseResponse

NoiseWaveformKernels:
    empty, WhiteNoiseRms only, or PowerSpectralDensity only

AnalogWaveformKernels:
    empty, AnalogMinimum only, AnalogMaximum only, or both

DigitizedWaveformKernels:
    exactly one each of
    BitDepth
    InputMinimum
    InputMaximum
    AnalogGain
```

Each collection exposes only the typed properties frozen in the architecture
record. Exact member insertion order is preserved. Duplicate exact types,
alien members, and invalid required/optional combinations fail during
construction.

Collection movement is device-only. It preserves every member's exact type,
Spec subtype, unit, representation dtype, and heterogeneous collection
semantics.

## Exact Config Public State

Every Config is:

```text
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
identity-equal
explicitly unhashable
```

Exact public constructor fields are:

```text
class ChargeConfig:
    spec: QuantityFieldSpec[Any]
    kernels: ChargeKernels
    correlated_avalanche_generations: NonnegativeInteger
    temporal_axis: type[TensorAxis[Any]] | None = None

class PureWaveformConfig:
    spec: QuantityFieldSpec[Any]
    kernels: PureWaveformKernels

class NoiseWaveformConfig:
    spec: QuantityFieldSpec[Any]
    kernels: NoiseWaveformKernels
    temporal_axis: type[TensorAxis[Any]] | None = None

class AnalogWaveformConfig:
    spec: QuantityFieldSpec[Any]
    kernels: AnalogWaveformKernels

class DigitizedWaveformConfig:
    spec: QuantityFieldSpec[Any]
    kernels: DigitizedWaveformKernels
```

Public constructors accept no prepared facts and no scalar, Pint Quantity, or
raw-tensor shortcut for a configurable coefficient. `spec`, `kernels`, and
`correlated_avalanche_generations` where present are required keyword-only
arguments. The two exact `temporal_axis` fields default to `None`; no other
public Config field has a default.

Each Config additionally owns these private `init=False`, non-repr prepared
facts:

```python
_is_prepared: bool
_source_dimensions: tuple[tuple[int, ...], ...]
_source_scales: tuple[float, ...]
_working_dtype: torch.dtype | None
_kernel_dimensions: tuple[tuple[int, ...] | None, ...]
```

`_kernel_dimensions` follows the exact Product collection property order.
Charge additionally owns:

```python
_temporal_dimension: int | None
_temporal_step_seconds: float | None
```

NoiseWaveform additionally owns the same two private temporal facts.

For Charge, `temporal_axis` is required when timing jitter, dark counts,
delayed crosstalk, or afterpulse is present. It is optional for a purely
spatial DirectCrosstalk law and absent when no selected mechanism needs time.
At least one of DirectCrosstalk, DelayedCrosstalk, or Afterpulse is present
exactly when `correlated_avalanche_generations.value > 0`; all three are absent
when it is zero.
For NoiseWaveform, `temporal_axis` is present exactly when
PowerSpectralDensity is present; exact-zero and white-noise Configs require
`None`.

No other persistent raw tensor is stored as derived Config state. Prepared
kernel tensors remain represented by the exact semantic Kernel members.
Static shape/count/byte ceilings are checked during preparation but not stored
unless a later correction demonstrates that one named immutable scalar is
required. Such a correction returns to Design.

An ordinary public constructor produces `_is_prepared is False`, empty source
facts, absent working dtype, absent temporal facts, and an exact
property-ordered kernel-dimension tuple:

```text
ChargeConfig:
    (None, None, None, None, None, None)
PureWaveformConfig:
    (None,)
NoiseWaveformConfig:
    (None, None)
AnalogWaveformConfig:
    (None, None)
DigitizedWaveformConfig:
    (None, None, None, None)
```

The positions follow the typed property order stated in the architecture
record. A missing optional member remains `None` after preparation; every
present member has one complete dimension tuple. Product-owned preparation
returns a fresh same-exact-type Config with `_is_prepared is True` and complete
facts. A private module-level same-type reconstruction helper may populate
`init=False` state; no public trusted constructor, token, alternate Config
class, Runtime, Plan, or cache is added.

## Exact Product Classmethods

All generated Products expose:

```python
@classmethod
def create(
    cls,
    *,
    sources: tuple[TensorField[QuantityFieldSpec[Any]], ...],
    config: ProductConfig,
    # rng: CounterRng only for Charge and NoiseWaveform
) -> Self: ...

@classmethod
def prepare(
    cls,
    *,
    source_specs: tuple[QuantityFieldSpec[Any], ...],
    config: ProductConfig,
) -> ProductConfig: ...

@classmethod
def produce(
    cls,
    *,
    sources: tuple[TensorField[QuantityFieldSpec[Any]], ...],
    config: ProductConfig,
    # rng: CounterRng only for Charge and NoiseWaveform
) -> Self: ...

@classmethod
def validate(
    cls,
    *,
    product: Self,
    sources: tuple[TensorField[QuantityFieldSpec[Any]], ...],
    config: ProductConfig,
) -> None: ...
```

The exact concrete Config annotation replaces `ProductConfig` in every class.
`Charge` and `NoiseWaveform` require keyword-only `rng: CounterRng` on
`create` and `produce`. Deterministic Products expose no RNG argument.

`Photoelectrons` has no Config, preparation, production, or creation method.
It exposes only:

```python
@classmethod
def validate(cls, *, product: Self) -> None: ...
```

Public Product methods delegate to private product-named actions. Private
`produce_*` functions return raw tensors; the Product class constructs the
semantic result with exact `config.spec` identity.

`create()` is exactly `prepare -> produce -> validate`. Staged calls use the
same methods and must be result-equivalent to `create()` for the same immutable
inputs and RNG address state.

## Exact Coefficient Unit And Geometry Laws

The package registry defines the exact base unit:

```text
avalanche = [avalanche]
```

Applications may use different semantic Product classes to distinguish
physical origins while using avalanche-compatible source units when the
Charge equation combines them. The registry recognizes no detector-specific
channel, example, sample, microcell, or ADC profile unit.

QuantityAxis and both Quantity Specs accept only exact `pint.Unit` values
belonging to `unit_registry`. A Unit from another registry is rejected rather
than copied implicitly. `quantity(...)` returns a scalar Quantity from the
same exact registry.

Exact first-law coefficient contracts are:

| Kernel | Unit | Representation and geometry |
|---|---|---|
| `TimingJitter` | dimensionless | binary64, finite and nonnegative; exactly one nonempty OffsetAxis operation axis targeting `ChargeConfig.temporal_axis`; complete represented sum equals one within absolute tolerance `1.0e-11` per conditioning point |
| `DirectCrosstalk` | dimensionless | binary64, finite and nonnegative; one or more nonempty OffsetAxis operation axes with unique target roles; represented sum no greater than one per conditioning point; offsets targeting the configured temporal role are nonnegative |
| `DelayedCrosstalk` | dimensionless | binary64, finite and nonnegative; one or more nonempty OffsetAxis operation axes including exactly one axis targeting the configured temporal role with strictly positive offsets; represented sum no greater than one |
| `Afterpulse` | dimensionless | binary64, finite and nonnegative; exactly one nonempty OffsetAxis operation axis targeting the configured temporal role with strictly positive offsets; represented sum no greater than one |
| `DarkCountRate` | avalanche / time | binary64, finite and nonnegative; no operation axes |
| `SmearingWidth` | dimensionless | binary64, finite and nonnegative; no operation axes |
| `PulseResponse` | output unit / source unit | floating, finite signed coefficients; one or more nonempty OffsetAxis operation axes |
| `WhiteNoiseRms` | NoiseWaveform output unit | floating, finite and strictly positive; no operation axes |
| `PowerSpectralDensity` | squared NoiseWaveform output unit | floating, finite and nonnegative prepared per-bin powers; exactly one frequency QuantityAxis operation axis |
| `AnalogMinimum` | AnalogWaveform output unit | floating and finite; no operation axes |
| `AnalogMaximum` | AnalogWaveform output unit | floating and finite; no operation axes |
| `BitDepth` | no quantity unit | exact integer dtype and values in `[1, 16]`; no operation axes |
| `InputMinimum` | DigitizedWaveform source unit | floating and finite; no operation axes |
| `InputMaximum` | DigitizedWaveform source unit | floating and finite; no operation axes |
| `AnalogGain` | dimensionless | floating, finite, and strictly positive linear multiplier; no operation axes |

Every no-operation coefficient may use empty conditioning axes or an accepted
subset of output roles. An operation-bearing coefficient may also condition on
an accepted subset. Exact role uniqueness and tensor/Spec agreement remain
TensorCore-owned.

Each probability/intensity leaf performs one construction-time ordered host
snapshot of its already TensorCore-owned tensor and validates each
conditioning row over exactly the trailing operation dimensions. Finite and
elementwise sign/range checks precede the row total. Row totals use
backend-independent `math.fsum` in frozen row-major operation order.
TimingJitter requires
`abs(total - 1.0) <= 1.0e-11`; DirectCrosstalk,
DelayedCrosstalk, and Afterpulse require
`total <= 1.0 + 1.0e-11`. Inputs are never normalized, clamped, repaired, or
replaced. Empty required operation support fails before the host snapshot.
This constructor boundary is the one documented accelerator synchronization
for those semantic leaves; Product production does not rescan their values.

PowerSpectralDensity requires exact zero DC power and at least one strictly
positive represented non-DC bin. The exact-zero branch is represented by an
empty NoiseWaveformKernels collection rather than an all-zero PSD.

The PowerSpectralDensity operation axis is not interpreted as an output
displacement and is the one Product-specific exception to generic
field-dimension role resolution. It must be a QuantityAxis composed with
RegularCoordinates, use a frequency-compatible unit, begin at zero, have a
strictly positive step, and contain exactly
`sample_count // 2 + 1` coordinates. During preparation, its converted
frequency step must equal:

```text
1 / (sample_count * temporal_coordinate_step)
```

within fixed binary64 relative tolerance `1.0e-12` and absolute tolerance
zero. The PSD tensor already contains integrated per-bin output-power values;
Maintenance 15 does not integrate, interpolate, or otherwise prepare a
caller-supplied density curve.

The temporal output axis must be a QuantityAxis composed with
RegularCoordinates, use a time-compatible unit, and have a strictly positive
step. NoiseWaveform uses the prepared PSD operation order directly and removes
that one spectral operation dimension from the generated output.

## Preparation And Production State

Preparation follows the exact order in the architecture record. It must:

1. validate exact Config and source-Spec tuples;
2. validate source count, units, complete semantic roles, coordinate
   equivalence, and same exact device;
3. validate output Spec and Product kernel collection;
4. validate conditioning-role availability and operation geometry;
5. resolve source dimension order and TensorCore kernel dimensions;
6. resolve stable conditioning-coordinate reorder and dimension permutation;
7. select working dtype and per-member representation conversion;
8. convert quantity-kernel units;
9. move only Config-owned kernels to the output device;
10. preserve exact integer BitDepth representation;
11. preflight shape, allocation, count, dtype, and address ceilings;
12. construct one fresh same-type prepared Config; and
13. consume no RNG words.

Every source device must equal every other source device and exact
`config.spec.device`. No Product method moves a source implicitly.

`_source_dimensions[i]` records, in output-Spec order, the source dimensions
needed to permute source `i`. Source axes must have equivalent complete
semantic state after role matching; source coordinate reordering is not
performed. Kernel conditioning coordinates may be reordered during
preparation because the owned kernel is reconstructed explicitly.

`_source_scales[i]` converts source `i` magnitudes into the Product's selected
working unit equation. `_kernel_dimensions` records each aligned member's
conditioning dimensions followed by operation-target dimensions in Product
output order. Production uses these facts directly and performs no Pint,
coordinate search, alignment discovery, device movement, or dtype-policy
selection.

Product production may create local tensor workspaces. It retains no mutable
workspace, semantic output, or RNG state in Config.

## Product Source Laws

Exact source counts are:

```text
Charge:
    one or more QuantityFieldSpec sources

PureWaveform:
    one or more QuantityFieldSpec sources

NoiseWaveform:
    exactly zero sources

AnalogWaveform:
    one or more QuantityFieldSpec sources

DigitizedWaveform:
    exactly one QuantityFieldSpec source
```

Charge accepts any semantic TensorField source satisfying its avalanche-count
unit and domain equation; it does not require `Photoelectrons` by exact type.
This permits application-owned values such as `Axioelectrons`.

Source tuple order is deterministic accumulation order. All sources must have
the exact output semantic role set and equivalent axis state, though dimension
order may differ. Unit compatibility is proved before conversion, casting,
summation, allocation, or RNG use. Incompatible `[avalanche] + [mV]` sources
fail at the exact offending source index.

Charge source tensors are exact `torch.int64`, nonnegative, count-valued
magnitudes no greater than `2**53 - 1`, with avalanche-compatible Specs.
Accepting an arbitrary TensorField semantic class does not weaken that
numerical source law.

Exact completed Product domains are:

```text
Photoelectrons:
    torch.int64, nonnegative, <= 2**53 - 1, avalanche-compatible unit
Charge:
    torch.float32 or torch.float64, finite, nonnegative,
    avalanche-compatible unit
PureWaveform:
    torch.float32 or torch.float64, finite
NoiseWaveform:
    torch.float32 or torch.float64, finite
AnalogWaveform:
    torch.float32 or torch.float64, finite
DigitizedWaveform:
    exact torch.int32, values in each aligned [0, maximum_code]
```

PureWaveform, NoiseWaveform, and AnalogWaveform units follow their exact
configured equations rather than one package-wide voltage convention.

PureWaveform and AnalogWaveform admit gradient-bearing floating sources and
preserve ordinary Torch autograd connectivity through differentiable casts,
ordered sums, convolution, and saturation. They do not detach sources, enter
`torch.no_grad()`, or replace ambient grad mode. Their result
`requires_grad` follows ordinary Torch semantics.

Photoelectrons and Charge sources are exact integer counts. Charge and
NoiseWaveform stochastic results and DigitizedWaveform integer results require
`requires_grad is False`. Maintenance 15 makes no differentiable stochastic
or digitization claim.

Every generated Product owns one fresh strided contiguous tensor whose storage
does not alias any source, Config-owned Kernel, or other generated Product.
Production completes or enqueues every TensorDSLab write before constructing
the semantic Product and performs no later write through an alias.
Photoelectrons remains a caller-constructed source value and is never copied,
mutated, or relabeled merely by validation.

## Dtype And Unit Plans

Every Product's working dtype is the deterministic promotion of:

- `config.spec.dtype`;
- every source Spec dtype;
- every arithmetic-admissible participating kernel dtype; and
- the Product's numerical floor.

Exact floors are:

```text
Charge:
    torch.float64 for probabilities, rates, branching, and smearing

PureWaveform:
    torch.float32

NoiseWaveform:
    torch.float32, with output restricted to torch.float32 or torch.float64;
    PSD complex workspace is complex64 or complex128 according to working dtype

AnalogWaveform:
    torch.float32

DigitizedWaveform:
    torch.float32 arithmetic plus exact integer BitDepth representation
```

The final Product tensor uses exact `config.spec.dtype`. Charge,
PureWaveform, NoiseWaveform, and AnalogWaveform output dtypes are floating.
DigitizedWaveform output dtype is one accepted signed integer dtype and must
fit every represented maximum code.

The floor is only the minimum. The exact ordered
`torch.promote_types` fold over output, source, and arithmetic-kernel
representation dtypes raises PureWaveform, NoiseWaveform, AnalogWaveform, or
DigitizedWaveform arithmetic to float64 whenever any participating
representation requires it. This preserves caller precision control without
using dtype ordering or backend defaults.

## Exact Count, Allocation, And Numerical Ceilings

The retained exact ceilings are:

```text
Charge count cell:
    integer in [0, 2**53 - 1]
TensorCore Poisson mean element:
    finite in [0.0, 1.0e8]
Tensor allocation element span:
    strictly less than 2**63
Tensor allocation byte span:
    strictly less than 2**63
RNG address capacity:
    exact TensorCore RngAddress/RngElements capacity contract
Multinomial category allocation:
    exact TensorCore constructor preflight before words
BitDepth:
    integer in [1, 16]
```

All source count conversion and ordered addition is checked before the
corresponding stochastic mechanism consumes words. Dark-count and collapsed
branching destination means must satisfy TensorCore's exact Poisson domain
after deterministic retained-window rate construction. No mean is clipped,
chunked into a changed law, or approximated to bypass the ceiling.

Every output, aligned kernel, convolution workspace, PSD coefficient
workspace, Multinomial category tensor, and address domain is preflighted for
both exact element count and byte span before allocation or words. Python
integer arithmetic owns the preflight. Integer overflow, floating
nonrepresentability, and output-domain overflow fail closed.

BitDepth is never cast into a floating Kernel representation. Production
derives `2**bit_depth - 1` through checked exact integer tensor arithmetic and
only then converts the derived value to the working floating dtype after exact
fit and representability preflight.

Pint operations occur only in public unit construction and Product
preparation. Production and validation use prepared magnitude tensors and
scales.

## Scientific And RNG Contracts

Maintenance 15 preserves the current accepted Product laws:

```text
TimingJitter       -> direct MultinomialDistribution
DarkCountRate      -> PoissonDistribution
DirectCrosstalk    -> collapsed destination-rate PoissonDistribution
DelayedCrosstalk   -> collapsed destination-rate PoissonDistribution
Afterpulse         -> collapsed destination-rate PoissonDistribution
SmearingWidth      -> GaussianDistribution
PulseResponse      -> deterministic convolution
WhiteNoiseRms      -> GaussianDistribution
PowerSpectralDensity -> prepared PSD synthesis
Analog bounds      -> deterministic pointwise saturation
Digitizer kernels  -> retained linear digitizer equation and integer cast
```

Charge retains timing unity before finite-window discard, same-frontier
branching, no same-round feedback, fixed generation depth, full-unit
afterpulse charge, no recovery weighting, checked accumulation, and
finite-window exclusion.

The exact private RNG namespace remains:

```text
0x54445331
```

Exact role streams remain:

```text
white noise        0x0000_0001
PSD noise          0x0000_0002
dark count         0x0000_0003
timing jitter      0x0000_0004
direct crosstalk   0x0000_0005
delayed crosstalk  0x0000_0006
afterpulse         0x0000_0007
charge smearing    0x0000_0008
```

No role is renumbered or reused. Each Product owns only its own roles and
address factories. RngElements are derived from the exact output Spec shape
and Product traversal. TensorCore owns words and Distribution algorithms.

The architectural rewrite does not promise old completed-value identity when
source combination, axis order, preparation, or Product orchestration changes.
For identical complete RngAddress, law inputs, dependency order, and execution
path, exact TensorCore word/result continuity applies. Every deliberate
completed-result rebaseline must be recorded in `docs/parity.md` with exact
same-candidate replay and statistical/analytic evidence.

## Application Retirement Decision

The user and TensorDSLab Design select a clean pre-deployment retirement of the
currently embedded application layer. Maintenance 15 deletes:

```text
tensor_dslab.readout
ReadoutConfig
ReadoutCollection
simulate_readout
prepare_readout
SamplingRuntime
ExampleAxis
ChannelAxis
SampleAxis
ds20k_veto
demos/readout.py
demos/readout.ipynb
the demos optional dependency group
```

No replacement application package is created in this work order. This is an
explicit retirement, not an inferred migration or compatibility claim.
Reusable Product scientific, RNG, unit, dtype, and numerical obligations move
to the new Product tests before old workflow tests are deleted.

A future collaboration-owned DS20k or Silex application may define axes,
profiles, workflows, result collections, demos, and IO in a separately
reviewed package. The future readout demo discussion starts from that
application boundary rather than restoring a hidden core workflow.

`create_environment.sh` remains and is changed to install the core wheel
without a demos extra. Its isolated smoke constructs a rank-zero
`QuantityFieldSpec`, a `Charge`, and validates exact TensorDSLab/TensorCore
versions, the PEP 610 TensorCore commit, site-packages resolution, public
facades, and absence of project-root shadowing.

## Exact Test Target

The exact tracked Python test/support target is `22` files:

```text
tests/__init__.py
tests/_product_support.py
tests/test_analog_waveform.py
tests/test_charge_correlated_avalanches.py
tests/test_charge_product.py
tests/test_charge_timing_jitter.py
tests/test_digitized_waveform.py
tests/test_environment_script.py
tests/test_kernel_contracts.py
tests/test_noise_waveform_branches.py
tests/test_noise_waveform_psd.py
tests/test_noise_waveform_statistics.py
tests/test_package_contracts.py
tests/test_photoelectrons.py
tests/test_product_configs.py
tests/test_product_types.py
tests/test_pure_waveform.py
tests/test_quantity_representations.py
tests/test_rng_identity.py
tests/test_tensorcore_0_22_adoption.py
tests/typing/maintenance_15_spec_composed_products.py
tests/typing/negative/maintenance_15_spec_composed_products.py
```

`tests/_product_support.py` owns immutable fixtures and independent numerical
oracles only. It defines no `TestCase`, discovered `test*` callable, production
preparation call, target-test import, or dynamic method attachment.

The committed negative typing fixture is excluded from the zero-diagnostic
positive command and run separately. It must yield exactly `12` intentional
diagnostics and no incidental diagnostic, covering:

- wrong Quantity Spec subtype;
- wrong Config kernel collection;
- wrong Product source tuple;
- missing required RNG;
- RNG on a deterministic Product;
- wrong coefficient Spec subtype;
- wrong collection member;
- wrong BitDepth Spec;
- retired import;
- wrong Product Config;
- incompatible exact Product return typing; and
- misuse of TensorCollection movement dtype.

The suite must contain no more than `150` discovered test methods and no more
than `6,000` tracked Python test/support lines. These are curation ceilings,
not production topology claims. No test may assert repository-wide test,
module, line, or import-edge counts.

## Exact Current-Test Obligation Ledger

Every current subject maps as follows:

| Current path | Maintenance 15 disposition |
|---|---|
| `tests/__init__.py` | retain |
| `tests/_noise_waveform_support.py` | merge independent oracles into `_product_support.py` |
| `tests/readout_fixtures.py` | replace with Spec/Product fixtures in `_product_support.py` |
| `tests/test_charge_correlated_avalanches.py` | rewrite in place against Charge classmethods and retain independent analytic/statistical proof |
| `tests/test_charge_product.py` | rewrite in place |
| `tests/test_charge_timing_jitter.py` | rewrite in place |
| `tests/test_deterministic_waveform_products.py` | split into `test_pure_waveform.py`, `test_analog_waveform.py`, and `test_digitized_waveform.py` |
| `tests/test_environment_script.py` | rewrite exact core-only smoke |
| `tests/test_kernel_geometry_and_quantity.py` | replace with `test_kernel_contracts.py` and `test_quantity_representations.py` |
| `tests/test_noise_waveform_branches.py` | rewrite in place |
| `tests/test_noise_waveform_psd_preparation.py` | merge into `test_noise_waveform_psd.py` |
| `tests/test_noise_waveform_psd_synthesis.py` | merge into `test_noise_waveform_psd.py` |
| `tests/test_noise_waveform_statistics.py` | rewrite in place |
| `tests/test_package_contracts.py` | rewrite exact target tree/facades/retirements |
| `tests/test_pint_physical_configuration.py` | replace with quantity-representation and Product-preparation tests |
| `tests/test_readout_axes_and_sampling.py` | retire application axes; preserve temporal QuantityAxis fixtures locally in Product tests |
| `tests/test_readout_collection.py` | retire generic workflow collection; replace kernel-collection contracts |
| `tests/test_readout_configs.py` | replace with `test_product_configs.py` |
| `tests/test_readout_product_types.py` | replace with `test_product_types.py` and `test_photoelectrons.py` |
| `tests/test_readout_profiles_and_demos.py` | explicitly retire embedded application subject |
| `tests/test_readout_simulation.py` | explicitly retire embedded workflow subject; preserve each Product law in direct tests |
| `tests/test_rng_ownership_migration.py` | replace with `test_rng_identity.py` |
| `tests/test_runtime_action_ownership.py` | replace with classmethod lifecycle and no-Runtime proofs |
| `tests/test_runtime_kernel_alignment.py` | replace with kernel-contract and Product-preparation alignment proofs |
| `tests/test_tensorcore_0_21_adoption.py` | replace with `test_tensorcore_0_22_adoption.py` |
| all seven current positive typing fixtures | replace with one exact current positive fixture |

No scientific or stochastic obligation is retired merely because its old
workflow host disappears.

## Required Focused Tests

Implementation must run, at minimum:

```bash
PYTHONPATH=. python -m unittest \
  tests.test_tensorcore_0_22_adoption \
  tests.test_quantity_representations \
  tests.test_kernel_contracts \
  tests.test_product_types \
  tests.test_product_configs \
  tests.test_photoelectrons \
  tests.test_pure_waveform \
  tests.test_analog_waveform \
  tests.test_digitized_waveform
```

and:

```bash
PYTHONPATH=. python -m unittest \
  tests.test_charge_product \
  tests.test_charge_timing_jitter \
  tests.test_charge_correlated_avalanches \
  tests.test_noise_waveform_branches \
  tests.test_noise_waveform_psd \
  tests.test_noise_waveform_statistics \
  tests.test_rng_identity
```

Implementation must also run:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
pyright --pythonversion 3.14 tensor_dslab \
  tests/typing/maintenance_15_spec_composed_products.py
bash -n create_environment.sh
git diff --check
```

The negative fixture runs separately and must produce exactly the frozen
intentional diagnostics.

## Required Mutation Evidence

Validation must independently kill at least these `19` process-local or
checkout-local mutants:

1. skip source-unit compatibility;
2. skip same-device source admission;
3. move a source silently;
4. resolve an axis by spelling rather than exact semantic type;
5. omit conditioning-coordinate reorder;
6. omit conditioning-dimension permutation;
7. accept a caller scalar/Pint/raw-tensor coefficient shortcut;
8. homogenize a heterogeneous kernel collection;
9. cast BitDepth to the floating working dtype;
10. bypass most-derived validation during Spec movement;
11. bypass most-derived validation during Field/Kernel/Collection movement;
12. detach a deterministic Product source or sever its autograd graph;
13. duplicate the TensorKernel defensive snapshot;
14. multiply afterpulse mean by `0.5`;
15. feed branching children back within the same round;
16. normalize timing probabilities silently;
17. include out-of-window branching destinations;
18. flip PulseResponse polarity a second time; and
19. restore a collaboration profile or generic readout orchestration import.

Each mutant must fail at one named committed proof for the intended reason.
Mutation evidence may be implemented through temporary process-local source
replacement; mutated bytes must never be committed.

## Complete Validation Gate

Validation owns one complete immutable-candidate gate:

1. verify exact commit/tree/parent/branch and direct ancestry;
2. verify the exact changed-path allowlist and protected bytes;
3. reconstruct exact TensorCore source/archive/wheel identities;
4. run TensorCore's complete source and extracted-archive suites;
5. run TensorDSLab focused source and extracted-archive groups;
6. run complete TensorDSLab source and extracted-archive discovery;
7. run Pyright positive checks in source and archive dependency forms;
8. run the exact TensorCore `97`-diagnostic negative fixture;
9. run the TensorDSLab exact `12`-diagnostic negative fixture;
10. execute all `19` required mutants;
11. build two deterministic TensorDSLab wheels with
    `SOURCE_DATE_EPOCH=0` and compare them byte-for-byte;
12. build one sdist and compare extracted package/test bytes to the candidate;
13. verify exact package tree, module tree, facades, signatures, and retired
    import absence;
14. create an isolated environment from the wheel and exact dependency;
15. run the installed core smoke outside the checkout;
16. run `create_environment.sh` in a fresh real temporary environment when
    Conda is available;
17. verify PEP 610 exact TensorCore commit identity;
18. run Markdown links, anchors, fences, Python-fence parsing, and stale-term
    checks;
19. inspect privacy and absence of raw route IDs;
20. verify no cache, bytecode, build, dist, egg-info, or generated residue;
21. verify a clean candidate checkout; and
22. report unavailable CUDA explicitly without making an accelerator claim.

If a candidate correction changes any production, test, dependency,
metadata, environment, or executable documentation byte, complete Validation
reruns the affected focused evidence and the complete source/archive gate. A
truth-only documentation child may carry executable evidence only after exact
byte-scope proof.

## Artifact Contract

The wheel contains exactly the `56` package files in the target tree and no
tests, demos, application profile, notebook, route identifier, cache, or build
residue.

The sdist contains the tracked candidate source under normal Hatchling
behavior. Validation records exact wheel/sdist filenames, byte sizes, and
SHA-256 values. No Design-time artifact hash is predicted.

The extracted artifact must import from installed site-packages outside the
repository and must not resolve TensorDSLab or TensorCore from a source
checkout.

## Documentation Scope

Implementation synchronizes current-contract wording in:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/api.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_15_execution_work_order.md
docs/implementation/maintenance_15_spec_composed_products_and_application_boundary.md
docs/overview.md
docs/parity.md
docs/physics/correlated_avalanches.md
docs/validation.md
```

Historical work orders remain byte-identical. Living docs must describe
TensorCore `0.22.0`, quantity Specs, direct Products, kernel Config punchcards,
no Runtime records, no generic readout, and the application boundary as the
current candidate contract without narrating obsolete stages as operative.

`docs/parity.md` preserves every selected Product scientific law, records the
representation/workflow rebaseline, and avoids claiming old completed-value
continuity where exact orchestration changed.

## Exact Changed-Path Allowlist

Implementation may change only:

```text
AGENTS.md
CONTRIBUTING.md
README.md
create_environment.sh
pyproject.toml
tensor_dslab/**
tests/**
docs/api.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_15_execution_work_order.md
docs/implementation/maintenance_15_spec_composed_products_and_application_boundary.md
docs/overview.md
docs/parity.md
docs/physics/correlated_avalanches.md
docs/validation.md
demos/readout.py
demos/readout.ipynb
```

The `tensor_dslab/**` and `tests/**` directory allowances are exact only in
combination with the target package and test trees frozen above. No other path
under those directories may survive or be added.

The two demo paths are deletion-only. Historical work orders, governance,
LICENSE, CI, repository metadata, unrelated artifacts, and every other path
are protected.

## Implementation Responsibilities

Implementation must:

- work only from the exact committed authority;
- use `git mv` for coherent product moves where practical;
- preserve current scientific algorithms rather than rewrite them casually;
- keep generic TensorCore mechanics in TensorCore;
- keep Pint, Product policy, units, dtype planning, roles, addresses, and
  science in TensorDSLab;
- maintain the obligation ledger while tests move;
- avoid aliases and transitional surfaces;
- run focused and complete local evidence;
- produce one immutable direct-child candidate;
- leave the candidate clean; and
- hand off exact commit/tree/parent/scope/evidence to Validation.

Implementation must stop rather than infer a new architecture if the exact
target cannot remain coherent.

## Validation Responsibilities

Validation is independent and may not repair candidate bytes. It verifies the
complete gate, reports one consolidated finding set, and either:

- returns exact findings to Implementation through Design authority; or
- dispatches the exact clear candidate to independent Review.

Validation must not accept Implementation's environment or artifact identity
without reconstruction. It may reuse independently verified unchanged
evidence across a documentation-only correction.

## Risk-Based Independent Review

Review reads the complete authority and independently audits:

- TensorCore versus TensorDSLab ownership;
- exact Quantity Spec and direct semantic-leaf structure;
- no duplicate unit state or Quantity value roots;
- Config punchcard readiness and absence of Runtime/Plan/token state;
- Product public signatures and exact one-shot/staged equivalence;
- source unit/device/order contracts;
- heterogeneous coefficient representation and BitDepth integer handling;
- role/coordinate alignment and operation geometry;
- scientific and RNG preservation/rebaseline truth;
- current-test obligation ledger and mutation strength;
- exact facades, retirements, package isolation, and application removal;
- complete Validation evidence consistency;
- exact scope, privacy, artifacts, and cleanliness; and
- CPU-only qualification.

Review reruns focused high-risk evidence and selected mutants. It may rely on
complete Validation source/archive/artifact reconstruction absent a concrete
discrepancy.

Review returns findings to Design. It does not edit, merge, or push before
final same-byte Design approval.

## Candidate And Return Route

Persistent logical roles are:

```text
TensorDSLab/default/Design
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Raw platform route identifiers must not enter repository bytes.

The route is:

```text
exact committed Design authority
    -> focused Implementation
    -> one immutable candidate
    -> complete fixed-candidate Validation
    -> risk-based independent Review
    -> final same-byte Design approval
    -> Review-owned git merge --ff-only to local main
    -> exact identity/diff/cleanliness closeout
```

The finite budget is:

```text
at most 3 immutable Implementation candidates
at most 3 Validation returns to Implementation
at most 2 Review returns requiring an executable correction
```

Every correction is a direct child of the preceding immutable candidate.
Design owns architecture dispositions and may stop or replace the route if a
finding changes frozen contracts.

No evidence-only closeout commit is required merely to restate a successful
same-byte fast-forward. A later ordinary push may be separately authorized
without repeating package gates only if the pushed bytes are exact
already-cleared local main.

## Hard Stops

Stop and return to Design if:

- TensorCore exact commit/tree/artifacts cannot be reconstructed;
- a required TensorCore public contract differs from the frozen substrate;
- a Config requires a generic base, Runtime, Plan, token, mutable cache, or
  public trusted constructor;
- a Product needs hidden source movement or production-time Pint/alignment
  discovery;
- a coefficient cannot remain an exact semantic Kernel;
- BitDepth cannot remain integer while digitizer arithmetic remains correct;
- a scientific law, probability boundary, count ceiling, RNG role, address,
  category order, or finite-window policy must change;
- a current test obligation would disappear without an exact replacement or
  explicit retired subject;
- application behavior must be embedded to keep the core coherent;
- the exact package/test target or public facade must widen;
- a compatibility alias or forwarding module appears necessary;
- a non-allowlisted path must change;
- CUDA work appears necessary;
- an execution route is stale, missing, dirty, or discrepant;
- any package source conflicts with an external handoff; or
- the candidate cannot remain a clean linear descendant of the authority.

## Explicit Non-Goals And Effects

This work order does not authorize:

- TensorCore edits;
- a DS20k or Silex package;
- an end-to-end application demo;
- IO, cache, artifact schema, CLI, or DAG surfaces;
- a universal Product/Config/Readout abstraction;
- PSD preparation design;
- arbitrary user Distribution injection;
- recovery-weighted afterpulse;
- new scientific laws;
- an RNG engine or Distribution fork;
- backward compatibility;
- package-index upload;
- tag or GitHub Release;
- push;
- deployment;
- CUDA or cluster execution;
- integrated TensorCore/TensorDSLab compatibility;
- release readiness;
- governance change; or
- production-readiness claims.

CPU qualification remains exact to the final candidate environment. Integrated
CUDA remains deferred to exact mutually adopted TensorCore/TensorDSLab `1.0.0`
release candidates.

## Completion Criteria

Maintenance 15 is complete only when:

- exact TensorCore `0.22.0` is pinned and independently reconstructed;
- the exact `56`-file package tree exists with no retired path;
- exact root and subpackage facades match;
- quantity units live only in Specs;
- all Product and coefficient leaves have exact direct TensorCore roots;
- all configurable coefficients are exact Kernels;
- all five typed kernel collections enforce exact membership;
- all five Config punchcards prepare into fresh same-type ready values;
- every Product public lifecycle and source law passes;
- unit/device/dtype/alignment preflights occur before effects;
- selected scientific and stochastic laws pass independent proof;
- the embedded application layer and demos are absent;
- the exact `22`-file test tree and obligation ledger reconcile;
- the `19` mutants are killed;
- source/archive/typing/artifact/environment/documentation/privacy/hygiene
  gates pass;
- unavailable CUDA is explicit and unclaimed;
- independent Review returns no unresolved finding;
- Design approves the exact same bytes;
- Review fast-forwards the exact candidate unchanged to local main; and
- post-merge commit/tree/parent/diff/cleanliness identity passes.

Only then does the self-effecting status resolve to **Merged / Closed** on
local main.
