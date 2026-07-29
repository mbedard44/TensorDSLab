# Maintenance 16 Declarative Requirements And Kernel Ownership

Status: **Self-effecting under the frozen exact-byte route**.

Before the complete same-byte package gate and clean local-`main`
fast-forward, the applicable state is the latest completed exact-byte handoff
under the route below. After the exact Review-cleared and Design-approved
candidate appears unchanged on local `main` through Review's clean
fast-forward, Maintenance 16 is **Merged / Closed**.

Stable key:
`TensorDSLab/maintenance-16-declarative-requirements-and-kernel-ownership`

## Purpose

Make TensorDSLab semantic representation classes readable as declarative
contracts without changing what valid TensorDSLab programs compute.

Maintenance 15 established the correct parts-bin architecture:

- semantic Axis values compose exact TensorCore Coordinates;
- quantity meaning lives in Axis and Field/Kernel Spec state;
- Product leaves directly specialize TensorCore `TensorField`;
- coefficient leaves directly specialize TensorCore `TensorKernel`;
- typed `*Kernels` collections compose configurable coefficient sets;
- Config punchcards retain prepared execution facts and source-Spec
  provenance; and
- Product classmethods own one-shot and staged preparation, production, and
  validation.

That architecture is sound, but its semantic admission mechanics remain
scattered:

- reusable coordinate requirements are loose functions in `common/axis.py`;
- allocation and RNG-address capacity requirements are mixed into
  `common/alignment.py`;
- Field and Kernel tensor-value checks are repeated across Product modules;
- Charge's combined row-mass, offset-geometry, and temporal-semantics helper
  conflates independent laws;
- Config provenance and generated-Field freshness requirements remain in
  alignment;
- `*Kernels` collection definitions live beside Config records rather than
  beside the kernel vocabulary they constrain; and
- several Kernel leaves validate dtype, Unit, and operation geometry even
  though those facts are owned structurally by their Specs.

This maintenance replaces that scattering with one private requirements
package and one exact rule:

> Semantic class contracts read as ordered compositions of narrowly named
> requirements.

This rule does **not** mean that all validation logic must leave semantic class
modules. A scientific law used by only one class may remain directly visible
inside that class when extraction would merely relocate it. Reusable
cross-module `require_*()` functions, however, have one predictable home under
`tensor_dslab.common.requirements`.

The maintenance also moves every public Product-specific `*Kernels`
collection into its Product's singular `kernel.py`, leaving `config.py`
focused on Config punchcards and their local relationships.

## Governing Sources

Implementation, Validation, and Review must read:

- [AGENTS](../../AGENTS.md);
- [CONTRIBUTING](../../CONTRIBUTING.md), especially TensorCore ownership,
  boundary-first admission, exact public typing, helpers in the narrowest
  meaningful owner, nonduplicative tests, and fail-closed validation;
- [Tensor architecture](../architecture/tensors.md);
- [Design](../design.md);
- [Overview](../overview.md);
- [Validation](../validation.md);
- [Parity](../parity.md);
- [Maintenance 14](maintenance_14_test_suite_curation.md) for test
  reconciliation and evidence economy; and
- [Maintenance 15 executable work order](maintenance_15_execution_work_order.md)
  plus its
  [architecture record](maintenance_15_spec_composed_products_and_application_boundary.md)
  for the exact current Product, Spec, Config, coefficient, scientific, RNG,
  and application-boundary contracts.

Package sources and this exact work order take precedence over informal
handoffs. A contradiction returns to Design.

No donor implementation or scientific approximation is selected.
`docs/parity.md` remains byte-identical.

## Exact Baseline

Maintenance 16 starts only from exact locally closed Maintenance 15:

```text
branch / local main:
    ee7a6a27bdb407c6dbb3d987f520a7aacd98fed0
tree:
    4edfe5b5b890fb239a5f8271d4bd28b390e3164b
exact parent / amended Maintenance 15 authority:
    698ba904cbae4a48e1d1e30c85612737b0f1b4dc
immutable ordinary Candidate 3:
    531ca3183abff689c5c7cb514d0763200a745d64
published origin/main at Design time:
    c8de1528d1ed57d3e86a9c37d1ad307127a23feb
package version:
    0.2.0
```

Maintenance 15's final supplemental candidate passed:

```text
TensorDSLab source:
    47 tests run / 47 passed / 0 skipped
TensorDSLab extracted archive:
    47 tests run / 47 passed / 0 skipped
Pyright positive source and archive:
    0 errors / 0 warnings / 0 informations
TensorDSLab negative typing source and archive:
    exactly 12 intended errors
mutation matrix:
    26 / 26 killed
package topology:
    56 package files / 55 Python modules
test topology:
    22 Python test/support files
    47 discovered methods
    18 TestCase classes
    2,359 Python lines
public root facade:
    61 names
```

Its exact artifacts were:

```text
wheel:
    tensor_dslab-0.2.0-py3-none-any.whl
wheel size:
    50,080 bytes
wheel SHA-256:
    ac3553c87b6bdd2f440ca7c26e4ebf8afa741a644bf5c497dd23ca7f9b907a0c
sdist:
    tensor_dslab-0.2.0.tar.gz
sdist size:
    527,354 bytes
sdist SHA-256:
    a8caca62273fe57b29a2b34a9877e673c21f6d82cb8542a1e1f573f117a96c3a
```

The accepted evidence is eager CPU. CUDA was unavailable and remains
unclaimed.

These figures are baseline evidence. Maintenance 16 deliberately adds private
package modules and may reconcile test methods, so candidate topology, method
totals, and artifact identities must be reported rather than assumed equal.

## Candidate 1 Stop And Replacement Amendment

Immutable Candidate 1 is:

```text
commit:
    a897fcdc0e5066113688f4b0fd0879833ab1f802
tree:
    5b2d9300ca2eb17b2ebb2bd95fa92b436fc51bd2
exact parent / original Design authority:
    35aba9b480cef04f3dec28dbc8eee504327496a1
```

Design stopped its complete fixed-commit Validation at the next safe boundary
after the user added one readability requirement: every module and every
module-level function, class, or Protocol introduced or moved by Maintenance
16 must carry an intentional docstring.

Validation also independently found six unintended positive Pyright
diagnostics:

- five diagnostics at `tests/test_product_configs.py` from erasing five
  Product-specific `prepare()` overload pairs into one heterogeneous loop; and
- one diagnostic at `tests/test_requirements.py` from erasing the generic
  coordinate representation into one heterogeneous union.

Candidate 1 remains immutable, is **not clear**, and is ineligible for Review
or merge. Its completed source/archive and topology evidence remains
read-only stopped evidence, not clearance.

This amended authority is the direct child of Candidate 1. It authorizes
exactly one ordinary Candidate 2 direct child changing only:

```text
tensor_dslab/common/requirements/tensor.py
tests/test_product_configs.py
tests/test_requirements.py
```

Candidate 2 must add the missing intentional Protocol docstrings, add the
committed definition-docstring proof, and express the two tests through
statically precise calls without `Any`, a broad cast, a new ignore, or weaker
runtime coverage. Every other Candidate 1 byte is protected and must remain
identical.

This consumes ordinary candidate slot `2 / 3` and Validation-return slot
`1 / 3`. Review-return consumption remains `0 / 2`.

## Exact Dependency Boundary

The dependency contract remains exact:

```text
Python:
    >=3.14
Torch:
    >=2.13,<2.14
NumPy:
    ==2.5.1
Pint:
    ==0.25.3
Hatchling:
    ==1.31.0
Pyright evidence:
    1.1.411
TensorCore:
    exact published 0.22.0 commit
    19bfae35fbc773b55cac7bcd659dda57c4dee6d6
TensorCore tree:
    53aa10520a50c0714e79c685d814cbae1b6f7740
```

Accepted TensorCore artifacts remain:

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

Maintenance 16 changes no dependency, version constraint, TensorCore
consumer contract, environment script, or package metadata. Validation
verifies the exact pin and import boundary but does not rebuild or rerun the
unchanged TensorCore package absent a concrete discrepancy.

No TensorCore coordination or publication is required.

## Selected Design Rule

The selected objective is:

> Semantic class contracts should read as ordered compositions of narrowly
> named requirements.

The rejected stronger rule is:

> All validation logic must live in `common/requirements`.

The distinction is normative:

- a reusable cross-module function named `require_*` belongs in the private
  requirements package;
- semantic `_require()` and `_require_*()` hooks remain on their classes;
- a short one-class scientific relationship may remain directly in its class;
- Product preparation retains relationships that exist only after multiple
  objects are aligned;
- runtime `validate_*()` actions remain Product-owned postcondition owners;
- normalization and conversion are not requirements because they return
  transformed values; and
- no registry, reflection system, validator object, callback list, annotation
  processor, or generic declarative framework is introduced.

Every requirement function:

- is private by facade ownership, even though cross-module functions use clean
  spelling without a leading underscore;
- returns `None`;
- fails closed with `TypeError` for representation/type-domain failures and
  `ValueError` for value/relationship-domain failures;
- does not mutate the admitted object;
- does not normalize, convert, align, move, allocate persistent execution
  state, return a transformed value, or produce;
- may use bounded ephemeral validation scratch or host materialization only
  where the exact contract below explicitly requires it;
- may perform only the explicit construction-time or preflight
  synchronization frozen below; and
- has one precise semantic name and owner.

## Exact Requirements Package

Add exactly:

```text
tensor_dslab/common/requirements/
  __init__.py
  axis.py
  capacity.py
  collection.py
  config.py
  field.py
  kernel.py
  tensor.py
  unit.py
```

`requirements/__init__.py` exports nothing and contains no convenience
facade. Callers import from the precise module:

```python
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_operation_row_total,
)
from tensor_dslab.common.requirements.tensor import (
    require_finite,
    require_nonnegative,
)
```

No name in this package is added to `tensor_dslab.common`, a Product facade,
or the package root. No public API or compatibility promise follows.

No `admission.py`, `common.py`, `quantity.py`, `device.py`, `preflight.py`,
`registry.py`, `base.py`, `utils.py`, or `helpers.py` is created.

The eight substantive modules must each own real accepted behavior; no empty
or placeholder module is permitted.

## Definition Docstrings

Readability is a frozen executable requirement:

- each of the exact nine added `common/requirements/` modules has one short
  module context docstring;
- every module-level function, class, and Protocol introduced by Maintenance
  16 has one short intentional docstring;
- each of the five moved `*Kernels` classes retains an intentional class
  docstring in its new defining module;
- private structural definitions are not exempt merely because their names
  begin with an underscore; and
- docstrings explain semantic purpose or ownership rather than restating the
  identifier.

Candidate 1 already satisfies this contract except that `_DtypeBearing` and
`_TensorBearing` in `common/requirements/tensor.py` have no class docstrings.
Candidate 2 adds meaningful docstrings to those two Protocols.

Imports, aliases, `__all__`, and module constants are not Python
module/class/function definitions and do not receive filler string literals.
This requirement does not retroactively add docstrings to unchanged
pre-Maintenance-16 implementation definitions.

`tests/test_requirements.py` must contain one static proof that parses the
exact nine requirement modules and rejects any module-level function/class
definition without a docstring. It also proves nonempty module docstrings and
the five moved `*Kernels` class docstrings. A temporary removal of either
Protocol docstring must fail that named proof. This static adversarial check is
separate from, and does not renumber, the thirty runtime/architecture mutants
below.

## Axis Requirements

`common/requirements/axis.py` owns:

```text
require_supported_coordinates(coordinates) -> None
require_supported_integer_coordinates(coordinates) -> None
require_coordinate_scale(coordinate_scale) -> None
require_regular_coordinates(
    coordinates,
    *,
    start: int,
    step: int,
) -> None
```

The first two functions move from `common/axis.py` without weakening exact
type admission:

```text
ExampleAxis / ChannelAxis:
    exact CountCoordinates
    exact LabelCoordinates
    exact RegularCoordinates
    exact OffsetCoordinates

QuantityAxis:
    exact CountCoordinates
    exact RegularCoordinates
    exact OffsetCoordinates
```

Subclasses, alien Coordinates implementations, spelling-equivalent classes,
and `LabelCoordinates` for `QuantityAxis` remain rejected.

`require_coordinate_scale()` requires an exact built-in `float` that is
finite and strictly positive.

`require_regular_coordinates()` requires exact `RegularCoordinates` and exact
integer `start` / `step` equality. It does not infer physical spacing,
construct a grid, inspect a Unit, or perform Product policy.

`QuantityAxis._require()` becomes:

```python
@final
@override
def _require(self) -> None:
    require_supported_integer_coordinates(self.coordinates)
    require_coordinate_scale(self.coordinate_scale)
    object.__setattr__(self, "unit", normalize_unit(self.unit))
    self._require_quantity_axis()
```

`ExampleAxis` and `ChannelAxis` each call only
`require_supported_coordinates()`.

`TimeAxis` and `FrequencyAxis` use the Unit requirement below for their exact
semantic dimensionality.

Coordinate representation, equality, hashing, windows, indices, physical
scale, and Unit meaning remain unchanged.

## Unit Requirements And Normalization

`common/requirements/unit.py` owns only:

```text
require_unit_compatible(
    unit,
    *,
    target,
    field: str,
) -> None
```

It accepts only an already-normalized exact package-registry Pint Unit and
proves conversion compatibility with `target`. It performs no normalization,
conversion of represented tensor values, magnitude extraction, or result
construction.

`common/units.py` continues to own:

```text
unit_registry
normalize_unit(unit) -> pint.Unit
```

The current cross-module `_normalize_unit()` is renamed to
`normalize_unit()` because it is an ordinary private cross-module operation
under the package's export-driven privacy rule. It remains absent from every
facade.

`normalize_unit()` preserves the exact Maintenance 15 contract:

- exact concrete package-registry Unit class only;
- exact package registry identity;
- no Pint Unit subclass;
- no foreign registry;
- no string, Quantity, or duck-typed unit shortcut; and
- the exact supplied admitted Unit is retained.

The abstract `QuantityAxis`, `QuantityFieldSpec`, and `QuantityKernelSpec`
normalize exactly once before their most-derived semantic hook.

`require_unit_compatible()` replaces repeated local try/convert blocks for:

```text
TimeAxis:
    time-compatible
FrequencyAxis:
    frequency-compatible
PhotoelectronsSpec:
    avalanche-compatible
ChargeSpec:
    avalanche-compatible
DigitizedWaveformSpec:
    dimensionless
TimingJitterSpec:
    dimensionless
DirectCrosstalkSpec:
    dimensionless
DelayedCrosstalkSpec:
    dimensionless
AfterpulseSpec:
    dimensionless
DarkCountRateSpec:
    avalanche / time-compatible
SmearingWidthSpec:
    dimensionless
AnalogGainSpec:
    dimensionless
```

Pulse-response, waveform, saturation-bound, input-bound, white-noise, and PSD
Units remain structurally valid package Units whose cross-object
compatibility is proven during Product preparation. This maintenance does not
invent a narrower standalone dimensional law for them.

## Tensor Requirements

`common/requirements/tensor.py` is the shared Field/Kernel extraction point.
It owns:

```text
require_exact_dtype(value, dtype) -> None
require_dtype_in(value, dtypes) -> None
require_floating_dtype(value) -> None
require_signed_integer_dtype(value) -> None
require_finite(value) -> None
require_nonnegative(value) -> None
require_positive(value) -> None
require_values_between(
    value,
    *,
    minimum,
    maximum,
) -> None
```

Private structural protocols may express exact `.dtype` and `.tensor`
requirements for strict typing. No new public base class, Protocol, mixin, or
export is introduced.

The dtype functions accept dtype-bearing Specs, Fields, and Kernels where
semantically applicable. The represented-value functions accept
tensor-bearing Fields and Kernels and inspect their existing tensor without
copying, detaching, casting, or moving it.

The functions preserve exact comparisons:

- `require_exact_dtype()` uses dtype identity;
- `require_dtype_in()` admits exactly the supplied tuple;
- `require_floating_dtype()` uses Torch floating-dtype semantics;
- `require_signed_integer_dtype()` admits exactly `torch.int8`,
  `torch.int16`, `torch.int32`, and `torch.int64`;
- `require_finite()` rejects every nonfinite represented value;
- `require_nonnegative()` rejects values below zero;
- `require_positive()` rejects zero and values below zero; and
- `require_values_between()` uses inclusive lower and upper bounds.

No clipping, replacement, normalization, promotion, tolerance, approximation,
or allocation occurs.

Construction-time device-to-host scalar observation caused by Torch boolean
reduction remains allowed exactly where current semantic constructors already
require represented-value admission. No new production-time host
materialization is authorized.

## Field Requirements

`common/requirements/field.py` owns:

```text
require_exact_field_spec(field, spec_type) -> None
require_fresh_product(
    product,
    *,
    sources,
    kernels,
) -> None
```

`require_exact_field_spec()` requires exact concrete Spec identity, not
`isinstance`, spelling, inheritance, or compatible structure.

`require_fresh_product()` moves unchanged in law from `common/alignment.py`.
It requires:

- contiguous generated Product storage;
- storage disjoint from every supplied source Field; and
- storage disjoint from every supplied Kernel.

It does not require different storage among sources, different storage among
Kernels, a common dtype, a common device, or a particular Product law.

The exact Product declarative contracts become:

```text
PhotoelectronsSpec:
    exact torch.int64
    avalanche-compatible Unit
Photoelectrons:
    exact PhotoelectronsSpec
    inclusive represented values [0, 2**53 - 1]

ChargeSpec:
    exact torch.float32 or torch.float64
    avalanche-compatible Unit
Charge:
    exact ChargeSpec
    finite
    nonnegative

PureWaveformSpec:
    exact torch.float32 or torch.float64
PureWaveform:
    exact PureWaveformSpec
    finite

NoiseWaveformSpec:
    exact torch.float32 or torch.float64
NoiseWaveform:
    exact NoiseWaveformSpec
    finite

AnalogWaveformSpec:
    exact torch.float32 or torch.float64
AnalogWaveform:
    exact AnalogWaveformSpec
    finite

DigitizedWaveformSpec:
    exact torch.int32
    dimensionless Unit
DigitizedWaveform:
    exact DigitizedWaveformSpec
    nonnegative
```

`Photoelectrons._require()` and public `Photoelectrons.validate()` share the
same represented-value requirements without duplicating the numerical law.
The public validation action remains Product-owned.

No Product classmethod, signature, source rule, unit conversion, storage law,
or completed value changes.

## Kernel Requirements

`common/requirements/kernel.py` owns:

```text
require_exact_kernel_spec(kernel, spec_type) -> None
require_no_operation_axes(spec) -> None
require_operation_axis_count(
    spec,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None
require_operation_axes_type(spec, axis_type) -> None
require_nonempty_operation_extents(spec) -> None
require_operation_target_count(
    spec,
    *,
    relative_to,
    count: int,
) -> None
require_offset_bounds(
    spec,
    *,
    relative_to,
    minimum: int,
    inclusive: bool,
) -> None
require_operation_row_total(
    kernel,
    *,
    exact: float | None = None,
    maximum: float | None = None,
    tolerance: float,
) -> None
```

Operation requirements inspect literal `TensorKernelSpec` geometry:

- no-operation means `operation_axes == ()`;
- operation-axis count uses inclusive minimum/maximum bounds and expresses an
  exact count by supplying equal bounds;
- operation-axis type means every axis has the exact supplied concrete type;
- nonempty operation extents reject any zero operation extent;
- target count compares exact `OffsetAxis.relative_to` class identity;
- offset bounds examine only OffsetAxes targeting the exact supplied role; and
- no helper sorts, normalizes, inserts, removes, maps, or broadcasts an axis.

`require_offset_bounds()` replaces the trivial `_offsets()` indirection and
uses `axis.coordinates.offsets` directly.

`require_operation_row_total()` is independent of temporal semantics, axis
type, target role, offsets, causality, and Product meaning. It:

- treats the complete ordered trailing operation geometry as one row;
- retains row-major operation identity;
- performs one explicit construction-time binary64 host materialization for
  backend-independent `math.fsum` totals;
- takes no additional defensive Kernel snapshot and does not clone an already
  owned Kernel merely to validate it;
- accepts exactly one of `exact` or `maximum`;
- uses the supplied absolute tolerance;
- performs no normalization, residual assignment, clipping, completion,
  transformation, or tensor mutation; and
- admits any operation geometry already accepted by the semantic Spec.

The current `_require_probability_kernel()` is deleted. It is not renamed or
retained as an alias because it conflates:

- finite/nonnegative represented values;
- complete or bounded row mass;
- OffsetAxis geometry;
- TimeAxis role count; and
- temporal offset sign.

The row-total requirement is intentionally generic enough for a future
non-temporal probability or bounded-mass Kernel without introducing a public
ProbabilityKernel root.

## Spec Versus Represented-Value Ownership

This maintenance deliberately reassigns existing checks to their truthful
construction boundary:

```text
Spec:
    axes
    operation geometry
    target roles
    offsets
    device
    dtype
    Unit

Field or Kernel:
    exact semantic Spec type
    represented tensor value domain
    represented row aggregate law
```

TensorCore continues to own universal Spec/tensor agreement, shape, device,
dtype, defensive Kernel ownership, Field reference ownership, and
most-derived hook timing. TensorDSLab narrows only downstream semantics.

This is an intentional invalid-input timing and diagnostic rebaseline:

- invalid dtype, Unit, or operation geometry fails while constructing the
  semantic Spec;
- invalid represented values fail while constructing the semantic Field or
  Kernel; and
- relationships requiring multiple aligned objects fail during Config
  preparation or Product validation.

Valid Specs, Fields, Kernels, Configs, and completed Product tensors remain
exact. No check is performed twice merely to preserve its former location.

## Exact Kernel Declarative Contracts

### Charge

```text
TimingJitterSpec:
    exact torch.float64
    dimensionless Unit
    exactly one nonempty exact OffsetAxis targeting TimeAxis
TimingJitter:
    exact TimingJitterSpec
    finite and nonnegative
    each operation row totals exactly 1.0 within 1.0e-11

DirectCrosstalkSpec:
    exact torch.float64
    dimensionless Unit
    at least one nonempty exact OffsetAxis
    every represented TimeAxis offset, if present, is >= 0
DirectCrosstalk:
    exact DirectCrosstalkSpec
    finite and nonnegative
    each operation row totals at most 1.0 within 1.0e-11

DelayedCrosstalkSpec:
    exact torch.float64
    dimensionless Unit
    at least one nonempty exact OffsetAxis
    exactly one TimeAxis target whose offsets are > 0
DelayedCrosstalk:
    exact DelayedCrosstalkSpec
    finite and nonnegative
    each operation row totals at most 1.0 within 1.0e-11

AfterpulseSpec:
    exact torch.float64
    dimensionless Unit
    exactly one nonempty exact OffsetAxis targeting TimeAxis
    every TimeAxis offset is > 0
Afterpulse:
    exact AfterpulseSpec
    finite and nonnegative
    each operation row totals at most 1.0 within 1.0e-11

DarkCountRateSpec:
    exact torch.float64
    avalanche / time-compatible Unit
    no operation axes
DarkCountRate:
    exact DarkCountRateSpec
    finite and nonnegative

SmearingWidthSpec:
    exact torch.float64
    dimensionless Unit
    no operation axes
SmearingWidth:
    exact SmearingWidthSpec
    finite and nonnegative
```

Direct crosstalk may remain purely spatial and therefore need not contain a
TimeAxis target. TensorCore's operation-target uniqueness remains authority
when a target is present.

The words "probability" and "time" do not enter the generic row-total
implementation. Charge's bounded Poisson offspring intensities remain
TensorDSLab scientific meaning rather than a generic probability hierarchy.

### Pure waveform

```text
PulseResponseSpec:
    floating dtype
    at least one nonempty exact OffsetAxis
PulseResponse:
    exact PulseResponseSpec
    finite
```

Pulse-response Unit compatibility with source and output Units remains a
Product preparation relationship.

### Noise waveform

```text
WhiteNoiseRmsSpec:
    floating dtype
    no operation axes
WhiteNoiseRms:
    exact WhiteNoiseRmsSpec
    finite and positive

PowerSpectralDensitySpec:
    floating dtype
    exactly one exact FrequencyAxis operation axis
    exact RegularCoordinates with start == 0 and step == 1
PowerSpectralDensity:
    exact PowerSpectralDensitySpec
    frequency extent >= 2
    finite and nonnegative
    exact zero DC power
    at least one positive non-DC represented power
```

PSD's zero-DC and positive-non-DC expressions may remain directly visible in
`PowerSpectralDensity._require()` because they are unique scientific laws.
The TimeAxis/FrequencyAxis reciprocal-grid relationship remains Config
preparation policy because it depends on output Spec and PSD Kernel together.

### Analog waveform

```text
AnalogMinimumSpec / AnalogMaximumSpec:
    floating dtype
    no operation axes
AnalogMinimum / AnalogMaximum:
    exact matching Spec
    finite
```

The pointwise minimum-below-maximum relationship remains in preparation after
both optional Kernels are aligned and broadcast.

### Digitized waveform

```text
InputMinimumSpec / InputMaximumSpec:
    floating dtype
    no operation axes
InputMinimum / InputMaximum:
    exact matching Spec
    finite

AnalogGainSpec:
    floating dtype
    dimensionless Unit
    no operation axes
AnalogGain:
    exact AnalogGainSpec
    finite and positive

BitDepthSpec:
    exact signed integer dtype
    no operation axes
BitDepth:
    exact BitDepthSpec
    inclusive represented values [1, 16]
```

The pointwise input minimum-below-maximum relationship remains in preparation
after alignment and broadcast. Integer BitDepth remains integer.

## Collection Requirements And Ownership

`common/requirements/collection.py` owns:

```text
require_admitted_member_types(collection, *, admitted) -> None
require_exact_member_types(collection, *, required) -> None
require_member_count(
    collection,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> None
```

Every member comparison uses exact concrete type. No helper uses inheritance,
spelling, registration, duck typing, or constructor order.

Exact collection laws remain:

```text
ChargeKernels:
    any subset of the six exact Charge Kernel leaves
PureWaveformKernels:
    exactly PulseResponse
NoiseWaveformKernels:
    zero or one exact WhiteNoiseRms / PowerSpectralDensity
AnalogWaveformKernels:
    any subset of exact AnalogMinimum / AnalogMaximum
DigitizedWaveformKernels:
    exactly BitDepth / InputMinimum / InputMaximum / AnalogGain
```

Move the five definitions:

```text
ChargeKernels:
    charge/config.py -> charge/kernel.py
PureWaveformKernels:
    pure_waveform/config.py -> pure_waveform/kernel.py
NoiseWaveformKernels:
    noise_waveform/config.py -> noise_waveform/kernel.py
AnalogWaveformKernels:
    analog_waveform/config.py -> analog_waveform/kernel.py
DigitizedWaveformKernels:
    digitized_waveform/config.py -> digitized_waveform/kernel.py
```

Each Product retains singular `kernel.py`. No `kernels.py`, `collection.py`,
compatibility module, forwarding class, alias class, registry, or coefficient
framework is added.

Each moved class has its new `kernel.py` as `__module__`. Root and Product
facades continue exporting the same public name in the same order.

Config modules then own Config records only. They compose their exact
`*Kernels` values but do not own the allowed kernel vocabulary.

## Config Requirements

`common/requirements/config.py` owns:

```text
require_config_components(
    *,
    spec,
    spec_type,
    kernels,
    kernels_type,
    field: str,
) -> None
require_prepared_config(
    *,
    is_prepared: bool,
    working_dtype,
    field: str,
) -> None
require_prepared_sources(sources, *, source_specs) -> None
```

`require_config_components()` uses exact concrete types and category-specific
diagnostics without exposing a package-wide generic `require_exact_type()`.

`require_prepared_config()` preserves the common production boundary: a
same-type Config is marked prepared and contains its resolved working dtype.

`require_prepared_sources()` moves from `common/alignment.py` unchanged in
law. Before arithmetic, allocation, or RNG it proves:

- exact tuple source container;
- exact prepared source count;
- every source is a TensorCore `TensorField`;
- every source Spec is a `QuantityFieldSpec`;
- positional exact structural equality between live and retained source
  Specs; and
- no recomputation of Unit conversion, coordinate alignment, dtype planning,
  or source order.

The source-Spec binding remains structural, so a distinct equal immutable Spec
remains valid.

Charge's relationship:

```text
branching Kernel present
    iff
correlated_avalanche_generations.value > 0
```

remains directly visible in `ChargeConfig.__post_init__`. It is a
Product-specific scientific relationship, not a generic Config primitive.

Config preparation continues owning:

- source/output role and coordinate alignment;
- exact source/output device equality;
- source Unit compatibility and conversion scales;
- working-dtype selection;
- Config-owned Kernel materialization;
- aligned Kernel dimension facts;
- pointwise minimum/maximum relationships;
- PSD time/frequency-grid reciprocity;
- count, distribution, allocation, and RNG-address preflight; and
- immutable prepared-fact construction.

No Config field, signature, equality, hashability, one-shot/staged
equivalence, or public lifecycle changes.

## Capacity Requirements

`common/requirements/capacity.py` owns:

```text
require_tensor_capacity(shape, *, dtype, field: str) -> None
require_address_capacity(
    element_shape,
    *,
    address_shape,
    field: str,
) -> None
```

These functions move from `common/alignment.py`.

They are capacity requirements, not device operations. They do not inspect
CPU versus CUDA, check device availability, query free memory, allocate, move
a tensor, select a backend, or claim that an admitted allocation will succeed.

`require_tensor_capacity()` proves the tensor shape and byte span fit the
current signed-int64-supported allocation domain using dtype item size.

`require_address_capacity()` proves the exact `RngElements` and `RngAddress`
shape spans fit their existing domains.

Current upper bounds, arithmetic order, diagnostics, and fail-before-effect
behavior remain exact.

No `common/device.py` is introduced. A future device module requires a
demonstrated shared device operation or policy; generic Field/Kernel movement
remains TensorCore-owned.

## Alignment After Extraction

After the move, `common/alignment.py` owns only:

```text
align_source(...)
prepare_sources(...)
kernel_dimensions(...)
prepare_kernel(...)
```

These functions perform or compile actual alignment, conversion, permutation,
or Kernel materialization. They are not requirement-only functions.

`prepare_sources()` retains source tuple/count admission,
`QuantityFieldSpec` admission, exact semantic roles and coordinates, exact
source/output device equality, Unit conversion-scale derivation, ordered
dimension plans, and working-dtype promotion.

`prepare_kernel()` retains exact conditioning-role lookup, exact coordinate
reordering, stable conditioning permutation, untouched operation geometry,
explicit target-device/Unit materialization, exactly one TensorCore defensive
Kernel snapshot, and the exact semantic Kernel subtype.

No circular dependency is permitted.

## Exact Target Package Delta

The Maintenance 15 package has `56` files / `55` Python modules.

Maintenance 16 adds exactly nine private Python modules and deletes no package
file:

```text
target:
    65 package files
    64 Python modules
```

The common subtree becomes:

```text
tensor_dslab/common/
  __init__.py
  alignment.py
  axis.py
  field.py
  kernel.py
  units.py
  requirements/
    __init__.py
    axis.py
    capacity.py
    collection.py
    config.py
    field.py
    kernel.py
    tensor.py
    unit.py
```

The Product subtrees retain their current filenames. Only the `*Kernels`
definition owner moves from `config.py` to `kernel.py`.

## Public And Typing Boundary

Exact public facades remain:

```text
package root:
    61 names
common:
    9 names
photoelectrons:
    2 names
charge:
    16 names
pure_waveform:
    6 names
noise_waveform:
    8 names
analog_waveform:
    8 names
digitized_waveform:
    12 names
```

Every facade sequence remains exact except for a private import-source change
needed to resolve a moved class. Exported names and order do not change.

No requirement, normalization, alignment, capacity, Runtime, or prepared-fact
name becomes public.

Strict positive typing continues proving exact semantic Axis/Coordinates,
Specs, Fields, Kernels, Collections, Configs, Product classmethods, and
TensorCore `0.22.0` use. The existing negative fixture retains its twelve
intended invalid programs.

## Scientific, Numerical, RNG, And Execution Preservation

Maintenance 16 changes no:

- Product equation or source combination order;
- Unit conversion result or dtype-promotion result;
- conditioning-coordinate permutation or operation-axis order;
- convolution, noise, Poisson, Gaussian, or Multinomial law;
- Charge branching, finite-window, timing-jitter, crosstalk, afterpulse,
  dark-count, or smearing law;
- saturation or digitizer equation;
- BitDepth integer treatment;
- count, allocation, or address ceiling;
- RNG namespace, stream, quantum, role, address, word schedule, or result;
- Config prepared source-Spec binding;
- Product storage freshness;
- one-shot/staged equivalence;
- supported device boundary; or
- CPU/CUDA qualification.

Valid construction and completed Product values remain exact to Maintenance 15
on the same accepted environment and address.

The only deliberate behavioral change is invalid-object failure timing:
metadata and geometry fail at Spec construction rather than later Kernel
construction. Exception categories remain fail-closed and diagnostics name
the truthful semantic owner.

## Test Reconciliation

Maintenance 16 follows Maintenance 14 evidence economy: reconcile the suite
around obligations rather than create a Cartesian product of helper functions
and semantic leaves.

Add one cohesive module:

```text
tests/test_requirements.py
```

The exact tracked test/support target is therefore `23` Python files.
No test may freeze repository-wide package-module, discovered-method, or
test-line totals as a permanent architecture contract.

The candidate test tree remains:

```text
<= 30 Python test/support files
<= 80 discovered test methods
<= 25 TestCase classes
<= 4,000 physical Python test lines
```

The current `47` substantive tests remain obligations, not necessarily
one-for-one method identities. Duplicated constructor cases may consolidate
only when the replacement proves the same law.

The requirements module directly proves:

- exact Coordinates admission and finite positive scale;
- exact regular-coordinate start/step;
- Unit compatibility without normalization;
- dtype identity/set/floating/signed-integer admission;
- finite, nonnegative, positive, and inclusive bounded values;
- exact Field and Kernel Spec identity;
- Field storage freshness;
- operation count/type/extent/target/offset laws;
- row totals in exact and maximum modes;
- row-total independence from TimeAxis semantics with a non-temporal
  operation geometry;
- Collection member sets and cardinality;
- Config components, prepared state, and source provenance;
- tensor and RNG-address capacity; and
- no mutation, normalization, allocation, movement, or clipping.

Semantic class tests prove wiring, validation timing, moved collection
ownership, unchanged facades, requirement privacy, and valid representative
Product continuity. Unique scientific proofs remain Product-focused.

The bounded replacement additionally proves:

- all newly added requirement modules and their module-level definitions carry
  intentional docstrings;
- all five moved `*Kernels` classes carry intentional docstrings;
- the positive Product-preparation proof retains all five exact
  Product/Config/Spec cases without collapsing their overloads into one union;
- the supported-coordinate proof retains Count, Label, Regular, and Offset
  coverage without collapsing the generic argument into one union; and
- Pyright reports zero positive diagnostics without a new ignore, `Any`, or
  broad cast.

## Required Mutation Matrix

Validation kills at least these exact high-risk mutants:

1. admit a Coordinates subclass by spelling or `isinstance`;
2. admit `LabelCoordinates` through `QuantityAxis`;
3. admit nonfinite or nonpositive coordinate scale;
4. normalize or accept a foreign/subclass Unit inside a Unit requirement;
5. resolve Field Spec identity through `isinstance`;
6. resolve Kernel Spec identity through spelling or inheritance;
7. admit integer dtype through the floating requirement;
8. omit finite represented-value admission;
9. swap nonnegative and positive admission;
10. make inclusive Photoelectrons or BitDepth bounds exclusive;
11. admit an operation axis of the wrong concrete type;
12. ignore one zero operation extent;
13. resolve an OffsetAxis target by spelling rather than exact identity;
14. weaken positive delayed offsets to nonnegative;
15. renormalize an operation row rather than reject it;
16. move temporal semantics into the generic row-total requirement;
17. weaken exact-one row mass to bounded mass;
18. remove the bounded-mass upper limit;
19. admit an alien `*Kernels` member by inheritance;
20. omit Noise branch cardinality;
21. omit one required DigitizedWaveform Kernel;
22. accept an unprepared Config or absent working dtype;
23. ignore changed positional source-Spec provenance;
24. omit generated Product storage disjointness;
25. weaken tensor-capacity arithmetic;
26. weaken RNG-address capacity arithmetic;
27. take a second defensive Kernel snapshot;
28. retain a duplicate reusable `require_*` outside the requirements package;
29. leave one `*Kernels` class defined in `config.py`; and
30. export one requirement through a supported facade.

Mutants are temporary evidence only. Each fails a named committed proof for
its intended reason. Validation reports the final exact matrix.

## Documentation

Update only living pages needed to describe:

- the private requirements package;
- declarative semantic contracts;
- Spec metadata/geometry versus Field/Kernel value ownership;
- `*Kernels` ownership in singular Product `kernel.py`;
- alignment after extraction;
- capacity terminology;
- unchanged public facades and Product workflow; and
- invalid-construction timing rebaseline.

Historical work orders, including both Maintenance 15 records, remain
byte-identical.

No page may claim a public requirement API, generic validation ownership
beyond TensorDSLab, broader CUDA support, backward compatibility, performance,
release readiness, or deployment.

## Exact Changed-Path Allowlist

Implementation may change only:

```text
CONTRIBUTING.md
tensor_dslab/common/alignment.py
tensor_dslab/common/axis.py
tensor_dslab/common/field.py
tensor_dslab/common/kernel.py
tensor_dslab/common/units.py
tensor_dslab/common/requirements/**
tensor_dslab/photoelectrons/field.py
tensor_dslab/photoelectrons/runtime/validate.py
tensor_dslab/charge/config.py
tensor_dslab/charge/field.py
tensor_dslab/charge/kernel.py
tensor_dslab/charge/__init__.py
tensor_dslab/charge/runtime/prepare.py
tensor_dslab/charge/runtime/produce.py
tensor_dslab/charge/runtime/validate.py
tensor_dslab/pure_waveform/config.py
tensor_dslab/pure_waveform/field.py
tensor_dslab/pure_waveform/kernel.py
tensor_dslab/pure_waveform/__init__.py
tensor_dslab/pure_waveform/runtime/prepare.py
tensor_dslab/pure_waveform/runtime/produce.py
tensor_dslab/pure_waveform/runtime/validate.py
tensor_dslab/noise_waveform/config.py
tensor_dslab/noise_waveform/field.py
tensor_dslab/noise_waveform/kernel.py
tensor_dslab/noise_waveform/__init__.py
tensor_dslab/noise_waveform/runtime/prepare.py
tensor_dslab/noise_waveform/runtime/produce.py
tensor_dslab/noise_waveform/runtime/validate.py
tensor_dslab/analog_waveform/config.py
tensor_dslab/analog_waveform/field.py
tensor_dslab/analog_waveform/kernel.py
tensor_dslab/analog_waveform/__init__.py
tensor_dslab/analog_waveform/runtime/prepare.py
tensor_dslab/analog_waveform/runtime/produce.py
tensor_dslab/analog_waveform/runtime/validate.py
tensor_dslab/digitized_waveform/config.py
tensor_dslab/digitized_waveform/field.py
tensor_dslab/digitized_waveform/kernel.py
tensor_dslab/digitized_waveform/__init__.py
tensor_dslab/digitized_waveform/runtime/prepare.py
tensor_dslab/digitized_waveform/runtime/produce.py
tensor_dslab/digitized_waveform/runtime/validate.py
tests/_product_support.py
tests/test_analog_waveform.py
tests/test_charge_correlated_avalanches.py
tests/test_charge_product.py
tests/test_charge_timing_jitter.py
tests/test_digitized_waveform.py
tests/test_kernel_contracts.py
tests/test_noise_waveform_branches.py
tests/test_noise_waveform_psd.py
tests/test_package_contracts.py
tests/test_photoelectrons.py
tests/test_product_configs.py
tests/test_product_types.py
tests/test_pure_waveform.py
tests/test_quantity_representations.py
tests/test_requirements.py
tests/typing/maintenance_15_spec_composed_products.py
tests/typing/negative/maintenance_15_spec_composed_products.py
docs/architecture/tensors.md
docs/design.md
docs/overview.md
docs/validation.md
```

The requirements wildcard is exact only with the frozen nine-file target. No
additional requirements module is authorized.

Every omitted Product test, Runtime module, RNG module, metadata file,
environment script, demo, facade, historical work order, governance record,
CI file, and repository file is protected. `tensor_dslab/**` and `tests/**`
are not blanket allowances.

## Implementation Responsibilities

Implementation must:

1. start from exact committed Design authority;
2. preserve exact Maintenance 15 dependency and supported facade names,
   ordering, signatures, and behavior;
3. add only the frozen requirements package;
4. move all five `*Kernels` definitions into singular `kernel.py`;
5. use direct ordered calls rather than a validator registry/list;
6. move metadata/geometry checks to Specs and value checks to Fields/Kernels;
7. delete duplicate loose reusable requirements;
8. preserve unique local scientific laws visibly;
9. keep alignment, normalization, preparation, and validation in truthful
   owners;
10. add intentional docstrings to every newly introduced module-level
    definition and prove that contract statically;
11. reconcile tests under the frozen ceilings;
12. run focused and complete local source evidence;
13. run strict positive and negative typing;
14. run diff, import, facade, privacy, docs, and hygiene checks;
15. produce one immutable direct-child candidate;
16. leave the candidate clean; and
17. return exact commit/tree/parent/scope/evidence to Design.

Implementation stops rather than widening policy, changing valid results or
RNG, adding a public export, changing a dependency, editing a non-allowlisted
path, or inventing compatibility baggage.

Implementation does not build final deterministic artifacts, create Conda
environments, execute CUDA, contact Review, merge, or push.

## Focused Implementation Evidence

Implementation reports:

- exact branch/commit/tree/parent and changed-path delta;
- exact package/test topology;
- focused requirements, representation, kernel, Config, and package tests;
- complete source CPU discovery;
- Pyright positive and exact negative-fixture results;
- diff/show checks;
- facade/import/privacy scans;
- changed-page links/fences;
- artifact/cache/build hygiene; and
- environment plus unavailable-CUDA qualification.

Implementation may run a selected mutant sample but does not claim the
complete independent matrix.

## Complete Fixed-Candidate Validation

Validation independently verifies:

- exact identity, topology, parentage, scope, and cleanliness;
- exact package and absence of speculative modules;
- complete module/function/class/Protocol docstring coverage for newly
  introduced definitions;
- pure fail-closed requirement effects;
- Spec-versus-value ownership;
- moved collection ownership and unchanged facades;
- complete source and extracted-archive TensorDSLab suites;
- strict positive typing and exact negative diagnostics in both forms;
- complete mutation matrix;
- valid-result scientific/RNG continuity;
- deterministic wheel and sdist reconstruction;
- source/artifact equality;
- isolated exact wheel installation with exact TensorCore;
- public import and downstream isolation;
- docs links/fences and living-contract consistency;
- privacy and retired-path scans;
- artifact/bytecode/cache/build hygiene; and
- final cleanliness.

Because dependencies, `create_environment.sh`, metadata, and supported Python
environment are byte-identical to Maintenance 15, Validation does not create a
fresh real Conda environment or rebuild unchanged TensorCore artifacts absent
a concrete discrepancy. It verifies the exact pin and installed identity.

Validation runs no CUDA and may not repair candidate bytes. It returns one
consolidated finding set to Design or clears the exact candidate for Review.

## Risk-Based Independent Review

Review independently audits:

- whether requirements simplify rather than obscure semantic classes;
- absence of a validator framework, registry, reflection, or callback list;
- requirement names, signatures, effects, diagnostics, and imports;
- intentional definition docstrings without filler or retroactive churn;
- Unit normalization versus Unit requirement separation;
- Spec metadata/geometry and Field/Kernel value ownership;
- generic row-total independence from time semantics;
- exact OffsetAxis role and bound handling;
- exact collection membership and class ownership;
- Config provenance and capacity arithmetic;
- no duplicate validation or Kernel snapshot;
- scientific/RNG continuity;
- typing and facade preservation;
- test reconciliation and mutant strength;
- complete Validation evidence consistency;
- documentation, privacy, artifacts, and cleanliness; and
- CPU-only qualification.

Review reruns focused high-risk tests, typing, and selected mutants. It may
rely on complete Validation artifact reconstruction absent a discrepancy.
Review returns findings to Design and does not edit, merge, or push before
final same-byte approval.

## Candidate Route And Evidence Cadence

Persistent logical roles are:

```text
TensorDSLab/default/Design
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Raw route identifiers remain private.

The route is:

```text
committed Design authority
    -> focused Implementation
    -> immutable candidate
    -> complete fixed-candidate Validation
    -> risk-based independent Review
    -> final same-byte Design approval
    -> Review-owned git merge --ff-only
    -> exact identity/diff/cleanliness verification
```

Evidence cadence:

```text
Implementation:
    focused plus complete source evidence
Validation:
    one complete source/archive/typing/mutation/artifact gate
Review:
    focused independent risk-based evidence
post-fast-forward:
    identity, lineage, diff, and cleanliness only
```

The finite ordinary budget is:

```text
at most 3 immutable Implementation candidates
at most 3 Validation returns requiring executable correction
at most 2 Review returns requiring executable correction
```

Every correction is one direct child of the prior immutable candidate.
Documentation-only lifecycle corrections may carry executable evidence only
when Design authorizes exact scope and Validation proves executable identity.
No recursive evidence-only closeout commit is required merely to restate a
successful same-byte fast-forward.

## Hard Stops

Stop and return to Design if:

- a requirement must mutate its input, normalize, convert, align, move,
  allocate persistent execution state, return a transformed value, or
  produce;
- a registry, callback list, reflection, annotation processor, or generic
  validation object appears necessary;
- extracting a one-class law makes its semantic class less clear;
- a newly introduced definition lacks an intentional docstring or requires
  filler prose to satisfy the static proof;
- ownership cannot follow the Spec/Field/Kernel split;
- moving a check changes a valid object or completed Product;
- row-total validation cannot remain independent of time;
- collection membership or supported facade name/signature identity changes;
- Config source-Spec provenance weakens;
- capacity arithmetic changes;
- a distribution, RNG role, address, word schedule, or result changes;
- TensorCore, dependencies, version, or requirements topology must change;
- a compatibility path or non-allowlisted edit appears necessary;
- source/archive/typing/artifact evidence diverges;
- positive typing requires `Any`, a broad cast, or a new ignore to pass;
- CUDA appears necessary;
- a role route is stale, dirty, missing, or discrepant; or
- the candidate cannot remain a clean linear descendant.

## Explicit Non-Goals

Maintenance 16 does not authorize:

- a public requirement API;
- a TensorCore primitive;
- a generic TensorDSLab Config/Product/Field/Kernel/Collection root;
- a declarative validator DSL or registry;
- `common/device.py`;
- automatic device movement;
- new Unit conversion or kernel geometry;
- a probability-kernel hierarchy;
- new coefficients or Products;
- application workflow, profiles, demos, or notebooks;
- IO, cache, artifact schema, CLI, or DAG work;
- TensorG4DS or TensorML adoption;
- scientific or RNG rebaseline;
- performance or compatibility claims;
- version change;
- package-index upload, tag, Release, push, CUDA, cluster work, deployment, or
  production-readiness claims.

## Completion Criteria

Maintenance 16 is complete only when:

- exact Maintenance 15 main is the authority ancestor;
- the exact nine-file requirements package exists;
- every Maintenance-16-added module and module-level function/class/Protocol
  has an intentional docstring;
- every reusable TensorDSLab cross-module `require_*()` lives there;
- no speculative or empty requirement module exists;
- `requirements/__init__.py` exports nothing;
- Unit normalization remains separate from Unit requirements;
- all five `*Kernels` classes live in singular Product `kernel.py`;
- every public facade name and order remains exact;
- Specs own metadata, dtype, Unit, axes, and operation geometry;
- Fields/Kernels own exact Spec identity and represented-value laws;
- collection, Config provenance, Field freshness, and capacity requirements
  have their frozen owners;
- alignment owns only alignment/materialization mechanics;
- `_offsets()` and `_require_probability_kernel()` are absent without aliases;
- generic row-total validation contains no temporal semantics;
- unique scientific laws remain visible and exact;
- valid Product results and RNG traces remain exact;
- positive Pyright is clean without suppressing either stopped Candidate 1
  diagnostic family;
- source/archive/typing/mutation/artifact/import/docs/privacy/hygiene pass;
- unavailable CUDA remains explicit and unclaimed;
- Review returns no unresolved finding;
- Design approves the exact same bytes;
- Review fast-forwards the candidate unchanged; and
- post-merge identity, lineage, diff, and cleanliness pass.

Only then does the self-effecting status resolve to **Merged / Closed** on
local main.
