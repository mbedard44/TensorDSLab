# Stage 2 Package And Readout Collection Foundation Work Order

Status: Stage 2 closeout record. The user authorized production execution on
2026-07-11 from the exact clean Design baseline recorded below.
Implementation prepared the scoped package, tests, and status synchronization;
fixed-commit Validation cleared a candidate and independent Review returned
narrow closeout findings. Every resulting fix commit returns through Validation
before Review recheck. A feature-branch checkout is not accepted package state;
if this record is read on `main`, Review's clean fast-forward gate has completed
and Stage 2 is accepted there.

## Task

Create the first installable `tensor_dslab` package and the complete structural
foundation for post-binned readout values:

- TensorDSLab-owned example and shared channel coordinate IDs;
- exact readout axis and field IDs;
- `SampleGrid`, `AdcQuantization`, and `DigitizedWaveformSpec`;
- one partial `ReadoutCollection(TensorCollection)` semantic collection;
- focused projection, axis-selection, and device-movement reconstruction
  helpers around TensorCore's base-collection-returning operations;
- field-scoped atomic-transform output preparation;
- exact full-chain collection-output preparation;
- focused public exports and construction tests.

This stage creates no scientific transform, RNG, workspace, full-chain
execution builder, cache, source adapter, or downstream integration surface.

## Purpose

Stage 1 accepted the readout architecture. Stage 2 turns only its stable value
and destination contracts into production code so later stages do not have to
invent collection schema, semantic axes, reconstruction, dtype rules,
descendant invalidation, or public-output ownership while implementing
physics.

The foundation must be useful on its own. It is not a placeholder package
tree: every module below owns a concrete Stage 2 contract.

## Dispatch State, Base, And Branch

This work order was designed in a documentation pass originally based on
repository commit:

```text
29c5589358b4ad38afe68596a4f77efc52464ee6
```

That commit is historical provenance, not an implementation base. The exact
clean package baseline accepted for Stage 2 production dispatch is:

```text
d097cb3cdde185c6814116e886e7844ea3f55178
```

The governed Design base remains:

```text
151b61fdc36475498219ee5fe7b045a3a72c2d09
```

Implementation branch:

```text
codex/stage-2-readout-collection-foundation
```

The package-owned logical execution routes are:

```text
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Their raw platform route identifiers are private and do not belong in this
record. Each route was created for the TensorDSLab workspace and independently
bootstrapped against the exact clean dispatch baseline before production work.
Coordination remains Deferred and is not part of this route.

The accepted Implementation/Validation loop budget is at most three
Implementation-to-Validation and three Validation-to-Implementation
dispatches, stopping earlier on clearance or any `AGENTS.md` stop condition.
Review receives only a Validation-cleared fixed commit. Review owns the clean
fast-forward merge and post-merge verification after explicit clearance.

The work-order key is this committed path. Its execution states are:

```text
Design-complete -> Dispatched -> Implementation candidate
  -> Validation-cleared -> Review-cleared -> Merged / Closed
```

`Returned to Design` and `Blocked` are terminal states for the current
execution attempt; neither authorizes scope expansion. The documentation work
that defined this order remained in Design, while production code, tests, and
required implementation-status synchronization now belong to Implementation.

## Source Of Truth

Implementation must follow:

- [TensorCore Integration Architecture](../architecture/tensors.md);
- [Post-Binned Readout Architecture](../architecture/readout.md);
- [Design](../design.md);
- [Decisions](../decisions.md);
- [Validation](../validation.md);
- [Parity](../parity.md), for naming and donor-boundary context;
- repository-wide [Contributing](../../CONTRIBUTING.md) and
  [Agent Workflow](../../AGENTS.md).

This work order is authoritative for Stage 2 file placement, public spelling,
exact dtypes, constructor shape, and minimum tests. The architecture pages
remain authoritative for cross-stage field meaning and ownership.

## TensorCore Dependency Baseline

Use TensorCore `0.6.0` at exact commit:

```text
dc554994061183776f23f65860a0594516074f2e
```

Design reconfirmed this dependency point at dispatch. It remains the published
`origin/main` point available to the direct reference, and the later local
TensorCore main has no change under `tensor_core/` or `tests/` relative to this
commit. The dispatch environment uses Python `3.13.11` and PyTorch `2.12.1`;
that exact tuple is evidence for this stage, not a broad compatibility claim.

The dispatch checkpoint exercised public `TensorCollection.empty_like()`,
`TensorCollection.zeros_like()`, and `require_compatible_collection(...)`.
They work for generic whole-collection structure as documented. They return
base collections, like-allocation preserves source memory format, and generic
compatibility intentionally excludes TensorDSLab sidecars, retained-record
identity, target-only replacement, and alias analysis. Stage 2 therefore uses
TensorCore operations where their generic contract fits and retains the
already-specified TensorDSLab semantic reconstruction and field-scoped public
destination builders. No TensorCore change or compatibility shim is required.

`pyproject.toml` should follow the ecosystem package pattern:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tensor-dslab"
version = "0.1.0"
description = "Tensor-native detector data-lab products and transforms."
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [
    { name = "Michael Bedard, Princeton University" },
]
dependencies = [
    "torch",
    "tensor-core @ git+https://github.com/mbedard44/TensorCore.git@dc554994061183776f23f65860a0594516074f2e",
]

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["tensor_dslab"]
```

Production imports use only public root `tensor_core` exports. Do not import a
TensorCore implementation module, copy a TensorCore helper, re-export generic
TensorCore names, or change TensorCore in this stage.

## Target Files

Implementation should add exactly these production surfaces unless a narrow
same-purpose split is agreed with Design:

```text
LICENSE
pyproject.toml
pyrightconfig.json
tensor_dslab/
  __init__.py
  py.typed
  common/
    __init__.py
    ids.py
  readout/
    __init__.py
    ids.py
    types.py
    tensors.py
    builders.py
    validation.py
tests/
  __init__.py
  readout_fixtures.py
  test_package_contracts.py
  test_readout_collection.py
  test_readout_tensor_operations.py
  test_readout_output_preparation.py
```

`tests/readout_fixtures.py` is test support, not a production abstraction.
Do not add `configs.py`, transform modules, kernels, execution/workspace
modules, caches, artifacts, source packages, detector/reconstruction packages,
executables, operations, or recipes.

The scientific domains are direct package subpackages. Do not insert a
decorative `tensor_dslab.domain` namespace between the package root and
`common`, `readout`, or later `detector`/`reconstruction` packages.

## Canonical Axis Contract

`ReadoutAxisRoles` is rejected from the Stage 2 surface. TensorDSLab deliberately
owns three exact required axis IDs instead:

```python
READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")
READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")
READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")

READOUT_REQUIRED_AXIS_IDS = IdSequence(
    (
        READOUT_EXAMPLE_AXIS_ID,
        READOUT_CHANNEL_AXIS_ID,
        READOUT_SAMPLE_AXIS_ID,
    )
)
```

These are value constants. Code must compare `TensorAxisId` values with `==`,
not object identity with `is`. A freshly constructed `TensorAxisId("sample")`
must resolve the same semantic axis.

`READOUT_REQUIRED_AXIS_IDS` is the canonical semantic enumeration of the three
roles; it is not automatically a collection's `shared_axes`. Each collection's
`shared_axes` follows its actual complete layout order and includes all extra
axes.

The constants fix semantic identity, not tensor position. A valid layout may
place the required axes in any order. Operations locate them through, for
example:

```python
sample_dimension = collection.layout.axes.index(READOUT_SAMPLE_AXIS_ID)
```

The required backing modes are exact:

- example is ID-backed by `IdSequence[ExampleId]`;
- channel is ID-backed by `IdSequence[ChannelId]`;
- sample is count-only and has no coordinates;
- all three occur exactly once because `TensorAxes` already rejects duplicate
  axis IDs;
- every additional layout axis occurs in every field and is also declared
  shared.

The public coordinate types are opaque stable string IDs owned together by
`tensor_dslab.common`:

```python
@dataclass(frozen=True, slots=True)
class ExampleId(Id):
    pass


@dataclass(frozen=True, slots=True)
class ChannelId(Id):
    pass
```

Stage 2 does not decide which future TensorG4DS public product TensorDSLab
consumes, how TensorG4DS `EventId` provenance maps—potentially one-to-many—to
`ExampleId`, or how a hardware channel map becomes `ChannelId`. The same
`ChannelId` coordinates are intended for readout and later reconstruction
products; they are not owned by the readout package. This stage only prevents
either downstream identity from being guessed from upstream or transient
tensor indices. It adds no TensorG4DS dependency.

### Canonical Stochastic Identity Consequence

There is no configurable per-collection stochastic-axis list. Later stochastic
stages use this canonical, tensor-order-independent sequence:

1. example axis ID and coordinate;
2. channel axis ID and coordinate;
3. every other ID-backed shared axis, ordered lexically by
   `TensorAxisId.value`, paired with its coordinate;
4. `SampleGrid.sample_offset + local_sample_index` for the sample axis;
5. an operation-local draw/counter coordinate when needed.

The seed, caller namespace, and operation role precede that coordinate
payload. Lexical ordering applies only to labeled extra-axis entries in the
RNG key; it never sorts tensor axes or coordinate `IdSequence` values. Adding
or removing an ID-backed axis changes logical stochastic
identity deliberately. A count-only extra axis is valid collection structure,
but a later stochastic transform must reject it unless Design first accepts a
stable global-offset rule for that axis. Stage 2 implements no RNG behavior.

## Canonical Field Contract

`tensor_dslab.readout.ids` owns and exports these exact constants:

```python
READOUT_PHOTOELECTRONS_FIELD_ID = TensorFieldId("readout.photoelectrons")
READOUT_CHARGE_FIELD_ID = TensorFieldId("readout.charge")
READOUT_PURE_WAVEFORM_FIELD_ID = TensorFieldId("readout.waveform.pure")
READOUT_NOISE_WAVEFORM_FIELD_ID = TensorFieldId("readout.waveform.noise")
READOUT_ANALOG_WAVEFORM_FIELD_ID = TensorFieldId("readout.waveform.analog")
READOUT_DIGITIZED_WAVEFORM_FIELD_ID = TensorFieldId(
    "readout.waveform.digitized"
)

READOUT_FIELD_IDS = IdSequence(
    (
        READOUT_PHOTOELECTRONS_FIELD_ID,
        READOUT_CHARGE_FIELD_ID,
        READOUT_PURE_WAVEFORM_FIELD_ID,
        READOUT_NOISE_WAVEFORM_FIELD_ID,
        READOUT_ANALOG_WAVEFORM_FIELD_ID,
        READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    )
)
```

Every valid collection holds a nonempty subset in that canonical order
filtered to the fields present. Direct construction rejects a differently
ordered mapping; domain builders construct the correct order rather than
sorting arbitrary input after the fact.

## Exact Numeric And Device Contract

Stage 2 fixes these tensor dtypes:

| Field role | Accepted dtype |
| --- | --- |
| photoelectrons | exactly `torch.int64` |
| charge, pure, noise, analog | `torch.float32` or `torch.float64` |
| digitized | exactly `torch.int32` |

Every floating field present in one collection has the same exact dtype.
`torch.bool`, unsigned integer, other signed integer, half, bfloat16, complex,
and other floating dtypes are invalid for their respective roles.

Stage 2 collection construction is placement-neutral: it accepts tensors
already resident on any PyTorch device when all fields share the exact device
and satisfy the structural/value contract. CPU construction is mandatory
coverage. Repeat structural tests on CUDA when available. This does not promise
that later stochastic or waveform kernels support every PyTorch backend.

No Stage 2 helper implicitly casts, detaches, or moves. The explicit movement
helper accepts an exact `torch.device` and delegates movement to TensorCore.

Every tensor must use `torch.strided` layout. Stage 2 does not add a separate
contiguity restriction: a noncontiguous but valid strided tensor is accepted.
Sparse and other non-strided layouts are rejected. General semantic
construction also does not reject an expanded or internally overlapping
read-only source solely because of storage arrangement. This flexibility does
not weaken the already-accepted later warmed execution profile: Stage 3/4
`out + workspace` preflight requires sample-last, contiguous participating
tensors and internally nonoverlapping writable storage. Stage 2 adds no
execution-ready subclass, sidecar, workspace, or preflight API.

## Public Value Types

`tensor_dslab.readout.types` owns this concrete surface:

```python
class AdcQuantization(StrEnum):
    TRUNCATE = "truncate"


@dataclass(frozen=True, slots=True)
class SampleGrid:
    sample_period_ns: PositiveFloat
    origin_ns: FiniteFloat
    sample_offset: NonnegativeInteger


@dataclass(frozen=True, slots=True)
class DigitizedWaveformSpec:
    bit_depth: PositiveInteger
    voltage_pp_mv: PositiveFloat
    voltage_offset_mv: FiniteFloat
    analog_gain_db: FiniteFloat
    quantization: AdcQuantization

    @property
    def adc_min(self) -> int:
        return 0

    @property
    def adc_max(self) -> int:
        return (1 << self.bit_depth.value) - 1
```

`AdcQuantization.TRUNCATE` is the only Stage 2/MVP value. `bit_depth.value`
must be in `[1, 16]`; signed `torch.int32` preserves that complete ADC domain
without relying on PyTorch's limited unsigned-16 operation support. A future
durable cache may compact validated values to an accepted unsigned
representation. Truncation retains the donor's intended in-range conversion
after clipping for nonnegative ADC values; adding rounding is a future public
policy change, not an undocumented kernel choice.

`analog_gain_db.value` must be in the inclusive range `[0.0, 40.0]`. This is
the intended donor validation range and corrects the audited IV condition that
could never reject an out-of-range value because it tested mutually exclusive
inequalities. `0.0` and `40.0` are valid; values immediately outside either
bound are invalid.

These dataclasses require the exact TensorCore wrapper types shown. They do not
coerce raw numbers. The wrappers already reject bool, non-finite values, and
invalid signs. The digitized spec is required exactly when the digitized field
is present.

## `ReadoutCollection` Public Surface

The semantic collection is a keyword-constructed, frozen, slotted dataclass:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutCollection(TensorCollection):
    sample_grid: SampleGrid
    digitized_waveform_spec: DigitizedWaveformSpec | None = None

    def __post_init__(self) -> None:
        TensorCollection.__post_init__(self)
        require_valid_readout_collection(self)

    @property
    def layout(self) -> TensorLayout:
        ...

    @property
    def device(self) -> torch.device:
        ...

    @property
    def example_dimension(self) -> int:
        ...

    @property
    def channel_dimension(self) -> int:
        ...

    @property
    def sample_dimension(self) -> int:
        ...
```

The explicit `TensorCollection.__post_init__(self)` call is required. A
zero-argument `super().__post_init__()` is unsafe here because
`dataclass(slots=True)` replaces the class object and can make zero-argument
`super()` fail. `kw_only=True` is required because inherited `metadata` already
has a default while `sample_grid` is required.

`@final` communicates the semantic boundary to type checkers, and constructor
validation requires exact `ReadoutCollection` at runtime. Do not create a
second hierarchy below this partial-snapshot type.

Call all constructor arguments by keyword even though the inherited generated
signature technically leaves base fields positional:

```python
readout = ReadoutCollection(
    fields=fields,
    shared_axes=shared_axes,
    metadata={},
    sample_grid=sample_grid,
    digitized_waveform_spec=digitized_waveform_spec,
)
```

Do not add `ReadoutAxisRoles`, arbitrary caller-defined required axis IDs,
per-field semantic subclasses, a generic `Product`, mutable field insertion,
or transform methods.

Stage 2 defines no semantic collection `==` contract. TensorCore's inherited
dataclass equality reaches PyTorch payload equality and can raise the
multi-element “Boolean value of Tensor is ambiguous” error. Production code
and tests compare explicit IDs, layouts, sidecars, record identity, and tensor
values with `torch.equal` or `torch.testing.assert_close` as appropriate. Do
not override equality locally merely to hide this TensorCore behavior.

## Collection Validation Boundary

`require_valid_readout_collection(...)` is a public domain validator used by
the constructor. Callers who receive an already constructed value may use it
as an explicit assertion, but it must not duplicate construction or return a
report object in Stage 2.

```python
def require_valid_readout_collection(
    collection: ReadoutCollection,
) -> None: ...
```

Use standard error categories consistently:

- `TypeError` for wrong Python/TensorCore/domain record types;
- `ValueError` for invalid semantic schema, order, layout, backing mode,
  dtype, device, sidecar state, or tensor values.

After TensorCore's base post-init validation, require:

- a recognized, nonempty, canonically ordered field subset;
- one structurally equal full `TensorLayout` across all fields;
- `shared_axes` exactly equal to all common-layout axis IDs in layout order;
- exact required axis IDs, with example/channel ID-backed by the exact
  coordinate classes and sample count-only;
- one exact common tensor device and `torch.strided` layout;
- exact role dtype rules and one common floating dtype;
- nonnegative photoelectron values;
- finite, nonnegative charge values;
- finite pure, noise, and analog values;
- digitized values in `[0, digitized_waveform_spec.adc_max]`;
- exact typed `SampleGrid`;
- `DigitizedWaveformSpec` present if and only if digitized is present.

“Same exact layout” means structural `TensorLayout` equality, not Python object
identity. TensorCore index selection may rebuild equal layout records for
different fields.

The Stage 2 constructor performs full value-domain checks, including on device
tensors. If later profiling demonstrates that repeated trusted reconstruction
creates an unacceptable synchronization cost, return to Design for an explicit
trusted-construction boundary; do not silently skip public validation in an
implementation thread.

## Semantic Reconstruction Helpers

TensorCore generic operations accept collection subclasses but return base
`TensorCollection` records. `tensor_dslab.readout.tensors` owns only the
accepted semantic reconstruction needed around those generic mechanics:

```python
def _reconstruct_readout_collection(
    collection: TensorCollection,
    *,
    sample_grid: SampleGrid,
    digitized_waveform_spec: DigitizedWaveformSpec | None = None,
) -> ReadoutCollection: ...


def project_readout_fields(
    collection: ReadoutCollection,
    selection: TensorFieldSelection,
) -> ReadoutCollection: ...


def select_readout_indices(
    collection: ReadoutCollection,
    selection: TensorAxisSelection,
) -> ReadoutCollection: ...


def move_readout_collection(
    collection: ReadoutCollection,
    *,
    device: torch.device,
) -> ReadoutCollection: ...
```

Required behavior:

- the private reconstruction boundary reuses the base collection's exact
  `TensorField` records, `shared_axes`, and metadata, then runs the full
  semantic constructor;
- projection delegates field selection to TensorCore, requires a nonempty
  canonical-order selection, structurally shares selected fields, reuses the
  exact `SampleGrid`, and retains the exact digitized spec only when digitized
  remains;
- an explicitly noncanonical model selection should use TensorCore directly
  and remain a base collection;
- index selection delegates tensor/index/layout work to TensorCore;
- example, channel, and extra-axis selections preserve the exact `SampleGrid`;
- sample selection is accepted only for a contiguous increasing unit-stride
  tuple, and advances origin and offset by its first local index;
- arbitrary, reordered, duplicated, or strided sample selection is rejected
  before TensorCore allocates selected tensors;
- movement delegates to `TensorCollection.to(device=...)` and reconstructs
  with the exact unchanged sidecars; it accepts no dtype or non-blocking
  parameter;
- no helper assumes TensorCore preserves the subclass.

Do not export a generic reconstruction escape hatch. TensorCore operations do
not all preserve readout meaning, and a caller must not be able to attach an
arbitrary `SampleGrid` to an arbitrary base collection through a convenience
cast. Add a new semantic helper only when its operation-specific preservation
rules are accepted.

Do not call `TensorLayout(axes=...)` with ID-backed axes and an omitted index
map in tests or examples. Use TensorCore `build_id_axis(...)` and
`build_tensor_layout(...)` so coordinate maps are complete.

## Descendant Invalidation Registry

One private registry in the readout package owns the accepted graph:

```python
readout.photoelectrons -> readout.charge -> readout.waveform.pure
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog -> readout.waveform.digitized
```

The exact stale-descendant sets are:

| Target | Remove before target insertion |
| --- | --- |
| photoelectrons | charge, pure, analog, digitized |
| charge | pure, analog, digitized |
| pure | analog, digitized |
| noise | analog, digitized |
| analog | digitized |
| digitized | none |

Do not export this as a mutable public mapping. Both Stage 2 preparation
factories and later transforms must call one shared helper rather than repeat
the table.

## Public Output Preparation

`tensor_dslab.readout.builders` exports two factories:

```python
def build_readout_result_buffer(
    source: ReadoutCollection,
    *,
    target_field_id: TensorFieldId,
    target_dtype: torch.dtype,
    digitized_waveform_spec: DigitizedWaveformSpec | None = None,
) -> ReadoutCollection: ...


def build_readout_output_buffer(
    source: ReadoutCollection,
    *,
    floating_dtype: torch.dtype,
    replace_photoelectrons: bool,
    digitized_waveform_spec: DigitizedWaveformSpec | None = None,
) -> ReadoutCollection: ...
```

These factories prepare caller-owned `out=` simulation destinations. They are
not the construction path for a differentiable `out=None` transform, which
must build its target from the computed tensor so autograd remains connected.

### Atomic Transform Output

`build_readout_result_buffer(...)` computes the exact post-invalidation
schema, structurally shares every unaffected source `TensorField`, allocates
one new zero-initialized target tensor and field, updates the conditional
digitized spec, and returns a valid collection. A newly generated target field
uses empty TensorCore metadata; it does not inherit descriptive metadata from
an invalidated prior value. The result preserves the source collection
metadata and exact immutable `SampleGrid` object.

The new target tensor is contiguous in the source collection's existing
semantic axis order. Preparation does not permute, call `.contiguous()` on,
clone, or otherwise materialize any retained source field. This makes the
writable target internally nonoverlapping without imposing contiguity on the
general `ReadoutCollection` constructor.

Preparation enforces the atomic transform's materialized source requirements:

| Target | Required source fields |
| --- | --- |
| photoelectrons | photoelectrons |
| charge | photoelectrons |
| pure | charge |
| noise | none beyond a valid collection layout and `SampleGrid` |
| analog | pure and noise |
| digitized | analog |

A partial snapshot may validly contain a descendant without its dependencies,
but it cannot be used to prepare an operation whose required source field is
absent.

`target_dtype` is always explicit. It must be `torch.int64` for
photoelectrons, `torch.int32` for digitized, and an accepted common floating
dtype for a floating target. When any retained floating field exists, the
target must match it.

Only a digitized target accepts `digitized_waveform_spec`, and it requires one.
All other target roles require that argument to be `None`. Their invalidation
rules remove any old digitized field/spec.

### Full Collection Output

`build_readout_output_buffer(...)` requires a source photoelectron
field and creates exactly:

```text
photoelectrons, charge, pure, noise, analog
+ digitized when digitized_waveform_spec is supplied
```

If `replace_photoelectrons` is false, the output shares the exact source
photoelectron `TensorField`. If true, it allocates a distinct zero-initialized
contiguous `torch.int64` photoelectron target. Every floating output is a
distinct zero-initialized contiguous tensor using `floating_dtype`; optional
digitized output is a distinct zero-initialized contiguous `torch.int32`
tensor. All new tensors use the source's existing semantic axis order.

Every generated field uses empty TensorCore metadata. The structurally shared
photoelectron field retains its existing immutable metadata, and the output
preserves source collection metadata and exact `SampleGrid` object.

`replace_photoelectrons` must be an exact `bool`, and `floating_dtype` must be
exactly `torch.float32` or `torch.float64`.

The factory ignores old generated descendants because the future builder
recomputes the complete configured chain. It never clones a retained
photoelectron tensor and never aliases one generated public field with another.

Both factories use shape-based `torch.zeros(...)` or an equivalent explicitly
contiguous initialized allocation that satisfies the public value domain
immediately. Plain `torch.zeros_like(...)` with preserved noncontiguous memory
format is not sufficient. They must not use `torch.empty`/`empty_like`,
whole-collection like-allocation, reorder/materialize retained fields, or
private scratch. They allocate no workspace and execute no science.

These factories do not certify warmed readiness. A full output can enter the
later strict warmed profile only when the source layout already has sample
last, every participating retained source is contiguous, and Stage 3/4
preflight accepts the exact output/workspace signature.

## Stage 3 Destination-Validation Constraint

Stage 2 prepares owned valid destinations; it does not yet accept an arbitrary
caller-supplied candidate for execution. Do not add a public compatibility API
or package-private placeholder validator before the first `out=` transform.

The Stage 3 deterministic-transform work order must build candidate preflight
on the same field-set/dtype/spec planning boundary used here and require exact
result order and sidecars, structural layout/device/dtype compatibility,
identical retained `TensorField` objects, a distinct target field, and target
storage that is internally nonoverlapping and disjoint from every source and
other output tensor. It should use
conservative common-storage rejection unless that work order accepts a proven
view-overlap analysis.

Stage 3 must keep execution modes distinct. Functional transforms accept
arbitrary valid semantic order/strides and may explicitly allocate
normalization while preserving accepted autograd. Ordinary `out` execution may
allocate documented scratch/normalization and makes no allocation-free claim.
Warmed `out + workspace` preflight requires sample-last order, contiguous
participating source/output/scratch tensors, exact destination/workspace/
stream/lease signature, and rejection before RNG or writes. It must not
normalize or allocate fallback storage inside the warmed call.

TensorCore `require_compatible_collection(...)` requires exact base
`TensorCollection` inputs and cannot express retained-record identity,
sidecars, target-only replacement, or aliasing. Do not call it directly with a
`ReadoutCollection`, weaken its exact-type rule locally, or fork it. A future
TensorCore improvement may accept subclasses for generic structural checks,
but Stage 2 is not blocked by that coordination item.

## Public Imports

The package root remains intentionally small:

```python
# tensor_dslab/__init__.py
"""TensorDSLab package."""
```

Do not re-export the full domain surface from `tensor_dslab`. Public Stage 2
imports are:

```python
from tensor_dslab.common import ChannelId, ExampleId
from tensor_dslab.readout import (
    AdcQuantization,
    DigitizedWaveformSpec,
    READOUT_ANALOG_WAVEFORM_FIELD_ID,
    READOUT_CHANNEL_AXIS_ID,
    READOUT_CHARGE_FIELD_ID,
    READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    READOUT_EXAMPLE_AXIS_ID,
    READOUT_FIELD_IDS,
    READOUT_NOISE_WAVEFORM_FIELD_ID,
    READOUT_PHOTOELECTRONS_FIELD_ID,
    READOUT_PURE_WAVEFORM_FIELD_ID,
    READOUT_REQUIRED_AXIS_IDS,
    READOUT_SAMPLE_AXIS_ID,
    ReadoutCollection,
    SampleGrid,
    build_readout_output_buffer,
    build_readout_result_buffer,
    move_readout_collection,
    project_readout_fields,
    require_valid_readout_collection,
    select_readout_indices,
)
```

Both subpackage `__all__` tuples must match their actual deliberate exports.
Private invalidation, destination-validation, canonical-order, storage-alias,
and value-scan helpers remain unexported.

## Minimum Test Design

Tests use `unittest` and assert public behavior rather than private helper
implementation. The production sketches above are binding interfaces; the
following is a naming and assertion sketch, not required literal test code:

```python
class PackageContractTest(unittest.TestCase):
    def test_public_readout_imports_resolve(self): ...
    def test_tensorcore_symbols_are_not_reexported(self): ...
    def test_only_public_tensorcore_root_is_imported(self): ...
    def test_stage2_imports_no_tensor_g4ds_g4ds_or_tensor_ml_package(self): ...
    def test_coordinate_ids_extend_id_but_axis_and_field_ids_do_not(self): ...


class ReadoutCollectionTest(unittest.TestCase):
    def test_accepts_all_sixty_three_nonempty_canonical_field_subsets(self): ...
    def test_rejects_empty_unknown_and_noncanonical_field_sets(self): ...
    def test_rejects_runtime_readout_collection_subclasses(self): ...
    def test_axis_constants_have_exact_values_and_use_value_equality(self): ...
    def test_requires_example_channel_and_sample_axes_in_any_order(self): ...
    def test_requires_exact_example_and_channel_coordinate_classes(self): ...
    def test_requires_count_only_sample_axis(self): ...
    def test_requires_all_layout_axes_in_shared_axes_layout_order(self): ...
    def test_requires_structurally_equal_layouts_and_one_device(self): ...
    def test_accepts_noncontiguous_strided_and_rejects_sparse_layout(self): ...
    def test_accepts_expanded_read_only_semantic_source_storage(self): ...
    def test_enforces_exact_role_dtypes_and_one_float_dtype(self): ...
    def test_enforces_field_value_domains(self): ...
    def test_requires_digitized_spec_exactly_with_digitized_field(self): ...
    def test_digitized_spec_requires_one_to_sixteen_bits(self): ...
    def test_digitized_spec_derives_int32_safe_adc_bounds(self): ...
    def test_digitized_spec_requires_zero_to_forty_db_gain(self): ...
    def test_sidecars_and_tensorcore_mappings_are_frozen(self): ...
    def test_dimension_properties_follow_two_axis_orders(self): ...


class ReadoutTensorOperationTest(unittest.TestCase):
    def test_tensorcore_projection_returns_base_collection(self): ...
    def test_domain_projection_reconstructs_and_shares_field_records(self): ...
    def test_noncanonical_model_projection_remains_tensorcore_owned(self): ...
    def test_projection_retains_or_drops_digitized_spec_with_its_field(self): ...
    def test_example_channel_and_extra_selection_preserve_sample_grid(self): ...
    def test_contiguous_sample_selection_advances_origin_and_offset(self): ...
    def test_irregular_sample_selection_fails_before_tensor_allocation(self): ...
    def test_move_reconstructs_on_exact_device_without_dtype_cast(self): ...


class ReadoutOutputPreparationTest(unittest.TestCase):
    def test_atomic_output_requires_its_exact_source_fields(self): ...
    def test_each_target_uses_exact_descendant_invalidation_table(self): ...
    def test_atomic_output_shares_only_unaffected_field_records(self): ...
    def test_atomic_target_is_distinct_zero_initialized_and_role_typed(self): ...
    def test_generated_targets_are_contiguous_without_normalizing_retained_fields(self): ...
    def test_atomic_target_never_aliases_source_or_retained_storage(self): ...
    def test_atomic_digitized_target_installs_exact_spec(self): ...
    def test_full_output_has_exact_required_and_optional_schema(self): ...
    def test_full_output_shares_photoelectrons_only_when_not_replaced(self): ...
    def test_full_generated_fields_are_distinct_zero_initialized_storage(self): ...
    def test_generated_writable_storage_is_internally_nonoverlapping(self): ...
    def test_outputs_preserve_collection_metadata_and_use_empty_new_field_metadata(self): ...
    def test_preparation_never_mutates_source(self): ...
```

Construction fixtures must use TensorCore's real builders:

```python
example_axis = build_id_axis(
    READOUT_EXAMPLE_AXIS_ID,
    IdSequence((ExampleId("example-0"), ExampleId("example-1"))),
)
channel_axis = build_id_axis(
    READOUT_CHANNEL_AXIS_ID,
    IdSequence((ChannelId("channel-0"),)),
)
sample_axis = TensorAxis(id=READOUT_SAMPLE_AXIS_ID, size=PositiveInteger(4).value)
layout = build_tensor_layout(TensorAxes((example_axis, channel_axis, sample_axis)))
```

Tests comparing two axis orders must permute values to the corresponding
semantic axes and make tensors strided-valid; they must not compare raw
positions as if axis order were fixed.

CUDA tests are conditional on `torch.cuda.is_available()`. They repeat a small
construction, projection, movement, and output-preparation slice; they do not
claim Stage 3/4 kernel support.

## Documentation Duties

Implementation should update only status or concrete API details that differ
after accepted narrow corrections in:

- this work order;
- [Implementation Stages](index.md);
- [TensorCore Integration Architecture](../architecture/tensors.md);
- [Post-Binned Readout Architecture](../architecture/readout.md);
- [Validation](../validation.md);
- `README.md` if current-status or public-import instructions become stale.

Do not rewrite parity science in this structural stage. If implementation
requires changing a semantic field, axis, dtype, sidecar, invalidation, or
output-lifetime contract, stop and return to Design.

## Expected Verification

Implementation should run at least:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -c "import sys, tensor_dslab; print('tensor_g4ds' in sys.modules, 'tensor_ml' in sys.modules, 'dslab' in sys.modules, 'g4ds11' in sys.modules)"
```

If `pyright` is available in the accepted implementation environment, also
run:

```bash
pyright
```

Report the exact Python, PyTorch, and TensorCore versions and whether CUDA
coverage ran. Validation should independently exercise the tests from the
fixed implementation commit before dispatching to Review.

## Non-Goals

- No timing, charge, waveform, noise, composition, clipping, or digitization
  execution.
- No scientific config records beyond the durable digitized interpretation
  spec.
- No RNG type, seed, counter, or random-field implementation.
- No `ReadoutWorkspace`, scratch allocation, stream lease, or concurrency.
- No warmed-execution preflight, execution-ready collection subclass, storage
  sidecar, or source-normalization helper.
- No public `build_readout_collection(...)`; it becomes complete in Stage 4.
- No full-chain schedule or internal placeholder builder engine.
- No native G4DS parsing, TensorG4DS dependency/adapter, detector response, or
  photoelectron binning.
- No readout example, reconstruction-domain example/product/package surface,
  cache, artifact, DAG, or TensorML adapter.
- No `tensor_dslab.domain` namespace layer; scientific subpackages live
  directly under `tensor_dslab`.
- No per-product collection subclass, `TensorField` subclass, generic product
  wrapper, or mutable collection accumulator.
- No caller-configurable required readout axis IDs or axis-role sidecar.
- No hidden cast, movement, detachment, device fallback, or whole-collection
  output allocation.
- No TensorCore API or repository change.

## Return To Design Before

- changing any exact axis ID, field ID, coordinate class, canonical order, or
  product meaning;
- moving `ExampleId` or shared `ChannelId` out of `tensor_dslab.common`, or
  reintroducing a `tensor_dslab.domain` package layer;
- reintroducing `ReadoutAxisRoles` or configurable stochastic-axis membership;
- changing the all-ID-backed-extra-axis stochastic identity rule;
- accepting stochastic use of a count-only extra axis without a global-offset
  contract;
- changing exact dtypes, the common-floating rule, ADC bit-depth limit, or
  truncation policy;
- requiring contiguous tensors in the general semantic constructor or
  accepting non-`torch.strided` tensors;
- changing the accepted rule that newly allocated public targets are
  contiguous in the existing semantic order without normalizing retained
  fields;
- weakening value-domain validation or adding an internal trusted constructor;
- changing partial-snapshot, projection, reconstruction, or sample-selection
  semantics;
- changing descendant invalidation or conditional digitized-spec behavior;
- permitting writable target/source aliasing or scratch-backed public output;
- adding a scientific transform, workspace, full-chain execution builder,
  source/cache/integration surface, or TensorCore change;
- expanding the target file tree with placeholder architecture.

## Readiness Checklist

Design completed the dispatch-readiness gate on 2026-07-11:

1. every source-of-truth document remained synchronized and no later Design
   change altered the Stage 2 contract;
2. Markdown links, headings, code fences, stale-name searches, and
   `git diff --check` passed against the clean dispatch candidate;
3. the exact clean production base is recorded above;
4. the Design worktree and all three role bootstrap worktrees were clean; and
5. the persistent logical routes and bounded loop budget are recorded above.

Any later baseline drift, route discrepancy, architecture ambiguity, dirty
state, or scope expansion returns this work order to Design.
