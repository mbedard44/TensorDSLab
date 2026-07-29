# Maintenance 18 EncodedWaveform Raw ZLE

Status: **Design authority / Implementation not dispatched**.

Stable key:
`TensorDSLab/maintenance-18-encoded-waveform-raw-zle`

## Purpose

Add `EncodedWaveform` as the deterministic terminal DAQ/readout Product after
`DigitizedWaveform`:

```text
DigitizedWaveform
    -> raw ADC threshold and time-over selection
    -> EncodedWaveform
```

The initial transformation emulates the accepted raw-waveform zero-length
encoding selection law from IV-DSLab. It identifies qualifying negative-going
ADC excursions independently in every example and detector lane, extends the
selected support by exact pre-trigger and post-trigger sample counts, and
returns one dense tensor containing:

```text
configured negative suppression code
    sample not retained by ZLE

nonnegative ADC code
    exact retained source code
```

`EncodedWaveform` belongs to the readout/DAQ side of the scientific boundary,
not to offline reconstruction. It is the terminal Product that a DAQ-like
application may retain or serialize, and it is the exclusive waveform ingress
for later reconstruction Products:

```text
simulation:
    DigitizedWaveform -> EncodedWaveform -> future reconstruction

experimental data:
    application-owned DAQ loading and densification
        -> EncodedWaveform
        -> future reconstruction
```

The package nevertheless keeps the Maintenance 15 root-level Product layout.
The new public precision package is:

```text
tensor_dslab.encoded_waveform
```

There is no restored `tensor_dslab.readout` workflow package and no new
`tensor_dslab.reconstruction` package. Product ownership and workflow
ownership are separate concerns: TensorDSLab owns the reusable Product and its
transformation, while applications own the chain in which it is used.

Maintenance 18 also generalizes the supported signed integer representation of
`DigitizedWaveform`, updates the application-neutral readout quickstart to
create and plot `EncodedWaveform` as a seventh separate Product, and records
the exact raw-donor parity boundary.

## Governing Sources

Implementation, Validation, and Review must read:

- [AGENTS](../../AGENTS.md);
- [CONTRIBUTING](../../CONTRIBUTING.md);
- [Overview](../overview.md);
- [Design](../design.md);
- [Public API](../api.md);
- [Parity](../parity.md);
- [Validation](../validation.md);
- [Tensor architecture](../architecture/tensors.md);
- [Product-composition architecture](../architecture/readout.md);
- [Maintenance 15 architecture record](maintenance_15_spec_composed_products_and_application_boundary.md);
- [Maintenance 15 executable work order](maintenance_15_execution_work_order.md);
- [Maintenance 16](maintenance_16_declarative_requirements_and_kernel_ownership.md);
- [Maintenance 17](maintenance_17_application_neutral_readout_quickstart.md).

The following repository-wide standards are specifically active:

- coordinates and tensor indices are distinct;
- exact Axis class identifies semantic role;
- Product preparation owns alignment, conversion, relationship validation,
  capacity preflight, and immutable positional source-Spec provenance;
- Product production owns prepared tensor execution only;
- every generated Product owns exact `create`, `prepare`, `produce`, and
  `validate` classmethods plus one same-type Config punchcard;
- configurable values that may vary by admitted semantic axis are Kernels;
- algorithm/domain policy that does not represent a varying numerical field
  remains structural Config or Spec state;
- durable records and application IO require a separately accepted boundary;
- donor behavior is promoted only through an exact named comparison boundary
  in `docs/parity.md`; and
- tests prove semantic behavior and meaningful failures rather than mirroring
  implementation structure.

This work order is the executable TensorDSLab authority for Maintenance 18.
The conceptual handoff that preceded it is supporting Design evidence, not a
second implementation authority.

## Exact Baseline

Maintenance 18 starts from exact locally closed Maintenance 17:

```text
local main / Maintenance 17 Candidate 2:
    58d3030c250fca4af88696b00f0165faf7757b12
tree:
    ef61c6b3dfce071259032e74b8dcf409dadc1633
exact parent / replacement Maintenance 17 authority:
    4ae15f81adc9182931610df6bc8b04f1786e9af4
package version:
    0.2.0
```

Maintenance 17 accepted:

```text
TensorDSLab source:
    60 tests run / 60 passed / 0 skipped
TensorDSLab extracted archive:
    60 tests run / 60 passed / 0 skipped
Pyright positive source and archive:
    0 errors / 0 warnings / 0 informations
TensorDSLab negative typing source and archive:
    exactly 12 intended errors
notebook mutation matrix:
    24 / 24 killed
package topology:
    65 package files / 64 Python modules
test topology:
    24 Python test/support files
    60 discovered methods
    20 TestCase classes
    3,594 Python lines
public root facade:
    61 names
```

Its exact deterministic artifacts were:

```text
wheel:
    tensor_dslab-0.2.0-py3-none-any.whl
wheel size:
    55,011 bytes
wheel SHA-256:
    d02ae7fdc497d93e758ffe4286d334ff611725fe157d4dc6247da04f8f43490d
sdist:
    tensor_dslab-0.2.0.tar.gz
sdist size:
    578,921 bytes
sdist SHA-256:
    df9e4df0929a3b67f23083482d0ced97b8c663b15c2abc2c11092f78b705d0aa
```

The exact TensorCore dependency remains published TensorCore `0.22.0`:

```text
commit:
    19bfae35fbc773b55cac7bcd659dda57c4dee6d6
tree:
    53aa10520a50c0714e79c685d814cbae1b6f7740
parent:
    25f48e3398c68217b060d94743f8abd810e7f7e8
canonical prefixed source archive:
    1,095,680 bytes
canonical archive SHA-256:
    deb09f72595a44f3b8551f01971986aa265a28a3f4475ee2afe59fb2b63f0c84
wheel:
    54,052 bytes
wheel SHA-256:
    6ac2f29c562504d7e87e1caf404b10019b08d60252fc496ad55b090e6b8b154f
```

Maintenance 18 changes no TensorCore surface and requires no TensorCore
publication.

The accepted baseline evidence is eager CPU. CUDA was unavailable and remains
unclaimed. Baseline artifact sizes and test totals are evidence identities,
not target identities for the new candidate.

## Selected Ownership Boundary

### Terminal DAQ/readout Product

`EncodedWaveform` is a readout/DAQ Product because ZLE selection is part of
what the acquisition system retains. It is not an offline reconstruction
result.

The future conceptual boundary is:

```text
Photoelectrons
    -> Charge
    -> PureWaveform
    -> AnalogWaveform
    -> DigitizedWaveform
    -> EncodedWaveform

EncodedWaveform
    -> future SignalWaveform
    -> future Hits
    -> future PulseMembership
```

This drawing is illustrative workflow composition, not a package-owned graph.
Every Product remains independently constructible. Applications may omit
Products, provide other admitted sources, retain intermediate Products, or
load an `EncodedWaveform` directly.

Future reconstruction must consume only `EncodedWaveform`. It must not recover
or consult suppressed `DigitizedWaveform` samples merely because simulation
created them earlier. An application may retain both Products for comparison,
but that retention does not widen the reconstruction dependency.

### Simulation path

The simulation transformation accepts exactly one `DigitizedWaveform` source:

```python
encoded_waveform = EncodedWaveform.create(
    sources=(digitized_waveform,),
    config=encoded_waveform_config,
)
```

There is no RNG parameter. The transformation is deterministic.

### Experimental-data path

Applications own DAQ parsing, channel mapping, record validation, corrupt-data
policy, and densification. After those responsibilities are complete, an
application may construct:

```python
encoded_waveform = EncodedWaveform(
    tensor=dense_codes_and_sentinel,
    spec=encoded_waveform_spec,
)
```

The TensorDSLab Field constructor validates the intrinsic representation:

- exact Spec;
- exact signed integer dtype/device/shape agreement through TensorCore;
- every represented value is either the configured suppression code or a
  nonnegative ADC code.

The application remains responsible for proving that retained codes came from
valid DAQ records, that ordinary acquisition gaps were not confused with ZLE
suppression, and that channel/time provenance is correct. Maintenance 18 adds
no loader, parser, durable record schema, cache, artifact, missing-data policy,
or IO facade.

## Product Domain

Conceptually:

```text
DigitizedWaveform: (E, D..., T)
EncodedWaveform:   (E, D..., T)

E:
    ExampleAxis
D...:
    zero or more non-Time semantic detector/readout axes
T:
    TimeAxis
```

Actual dimension order is the exact ordered Axis tuple stored by the Spec.
The simulation path requires the output and source to have structurally equal
ordered Axis tuples. It does not reorder or reduce dimensions.

`EncodedWaveform` preserves:

- the complete ordered Axis tuple;
- complete shape;
- every detector/readout semantic role and coordinate;
- exact TimeAxis coordinates, scale, and Unit;
- source device;
- source signed integer dtype;
- dimensionless ADC-code Unit; and
- the exact tensor index of every retained source sample.

The transformation changes only whether one ADC code is present. It does not
resample, downsample, shift, interpolate, filter, baseline-subtract, normalize,
or aggregate.

### ExampleAxis independence

Every ExampleAxis coordinate is one independent waveform realization. It is
not a consecutive DAQ fragment.

For every example `e` and detector coordinate tuple `d...`:

```text
EncodedWaveform[e, d..., :]
    depends only on
DigitizedWaveform[e, d..., :]
and the prepared coefficient values for that same lane.
```

Raw-ZLE state starts fresh for every example. Pre-trigger and post-trigger
support is clipped to that example's TimeAxis. The implementation must never
carry:

- trigger qualification;
- release state;
- consecutive-sample count;
- retrigger state;
- pre/post support;
- holdoff;
- filtering state; or
- downsampling phase

between ExampleAxis entries.

Any future continuous-stream or FPGA-emulation context must be represented
inside one example or supplied through an explicit future halo/window
contract. Adjacent ExampleAxis coordinates must never be interpreted as
contiguous time.

### Detector-lane independence

The raw decision factorizes over all non-Time coordinates. Changing one
detector lane must not change retained support or values in another lane.

The initial law performs no:

- detector-wide sum;
- channel coincidence;
- cross-channel trigger;
- pulse finding;
- detector-neighbor lookup; or
- reduction over an admitted non-Time role.

Kernels may be global or conditioned on admitted output non-Time roles. The
resolved coefficient applicable to one lane is explicit. It is never inferred
from another lane.

### TimeAxis admission

The simulation path requires exactly one `TimeAxis` in the ordered source and
output Specs. Its coordinates must be exact `RegularCoordinates` with:

```text
step == 1
```

The integer `start` may be any admitted exact value. The coordinate count may
be zero. Consecutive tensor indices then correspond to consecutive source
sample bins, while physical period remains:

```text
TimeAxis.coordinate_scale * TimeAxis.unit
```

The raw algorithm is defined in sample indices. It neither assumes zero
physical time nor converts physical durations.

`prepare()` validates this relationship and compiles the exact Time dimension
as an immutable execution fact. `produce()` must not perform Pint arithmetic,
look up an Axis, inspect coordinate values, construct a time grid, or derive
physical sampling.

## Signed Integer Representation

### Supported dtypes

`DigitizedWaveformSpec` and `EncodedWaveformSpec` admit exactly:

```python
(
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
```

The exact dtype is a backend/application choice expressed in the Product Spec.
There is no hard-coded `torch.int32` requirement.

The simulation path requires:

```text
encoded output dtype is source DigitizedWaveform dtype
```

Production performs no source-code cast. This makes exact retained-value
preservation literal and avoids an implicit backend representation change.

### DigitizedWaveform capacity correction

Generalizing `DigitizedWaveformSpec` requires a pre-production relationship
between `BitDepth` and the selected output dtype.

For every aligned lane:

```text
maximum_code = 2**bit_depth - 1
maximum_code <= torch.iinfo(output_dtype).max
```

The maximum code is derived in exact checked integer arithmetic before any
floating conversion used by digitizer scaling. If any configured BitDepth
cannot be represented by the output dtype, `DigitizedWaveform.prepare()` fails
before tensor arithmetic, output allocation, or RNG use.

The existing `BitDepth` domain remains exactly `1..16`. Examples:

```text
8-bit codes in torch.int8:
    invalid because 255 > 127

7-bit codes in torch.int8:
    valid because 127 is representable

12-bit codes in torch.int16:
    valid

12-bit codes in torch.int32:
    valid and remains the quickstart choice
```

This correction changes no digitizer equation, rounding/truncation law,
clipping law, BitDepth Kernel dtype, or current valid `torch.int32` result.

### Suppression code

`EncodedWaveformSpec` owns one explicit structural field:

```python
suppression_code: int
```

It is not a Config field and not a Kernel because it defines how missing ADC
codes are represented in the Product itself. Every consumer of an
`EncodedWaveform`, including a future reconstruction Product or application
loader, can interpret the tensor from the Field and its Spec alone.

Admission is exact:

- `type(suppression_code) is int`; Boolean is rejected;
- `suppression_code < 0`;
- it is representable by the exact Spec dtype.

There is deliberately no implicit default. Backend/application code selects
the sentinel explicitly. The package quickstart uses `-1`, but `-1` is not a
universal storage contract.

A nonnegative sentinel is forbidden because zero and every other nonnegative
value may be a valid retained ADC code. The Field distinguishes:

```text
value == spec.suppression_code:
    no ADC code was retained

value >= 0:
    exact retained ADC code, including literal zero
```

Every other negative represented value is invalid.

Ordinary ZLE suppression does not mean corrupt input, malformed record,
unexplained acquisition gap, data outside the declared waveform, or loader
failure. Applications must reject or separately represent those conditions;
they must not silently map them to the suppression code.

## Public Spec And Product

### EncodedWaveformSpec

The exact conceptual shape is:

```python
@final
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class EncodedWaveformSpec[
    AxesT: tuple[TensorAxis[Any], ...],
](
    QuantityFieldSpec[AxesT],
):
    suppression_code: int

    @override
    def _require_quantity_field_spec(self) -> None:
        require_signed_integer_dtype(self)
        require_unit_compatible(
            self.unit,
            target="",
            field="EncodedWaveformSpec.unit",
        )
        require_negative_representable_suppression_code(self)
```

The implementation may use the narrowest existing requirement signatures, but
the public dataclass shape and semantic result are exact.

Unlike the current fieldless Product Specs, `EncodedWaveformSpec` has one
meaningful stored semantic field. TensorCore `TensorFieldSpec` explicitly
supports frozen/slotted downstream fields and includes them in exact-concrete-
type structural equality and hashing. `suppression_code` therefore
participates in:

- structural Spec equality;
- positional prepared source provenance where an EncodedWaveform is later a
  source;
- subtype-preserving `.to()` reconstruction; and
- subtype-preserving `.with_axis()` reconstruction.

### EncodedWaveform

The exact Product leaf is:

```text
@final
class EncodedWaveform(
    TensorField[EncodedWaveformSpec[Any]],
):
    __slots__ = ()

    @override
    def _require(self) -> None:
        require_exact_field_spec(self, EncodedWaveformSpec)
        require_encoded_values(self)

    @classmethod
    def prepare(...) -> EncodedWaveformConfig:
        ...

    @classmethod
    def produce(...) -> Self:
        ...

    @classmethod
    def validate(...) -> None:
        ...

    @classmethod
    def create(...) -> Self:
        ...
```

The leaf remains final, fieldless, frozen through TensorCore storage semantics,
identity-equal, and explicitly unhashable under the existing Product
convention.

Intrinsic construction accepts either:

- a simulation-produced dense tensor; or
- an application-densified experimental tensor.

The intrinsic represented-value law is:

```text
for every value:
    value == spec.suppression_code or value >= 0
```

Simulation-only source preservation belongs to `validate()`, not the
standalone Field constructor.

## Public ZLE Coefficient Kernels

Every caller-configurable numerical parameter that may vary by Example,
Channel, or another admitted non-Time output role is one semantic Kernel.
Maintenance 18 adds exactly five:

```text
TriggerThresholdCode
ReleaseThresholdCode
RequiredTimeOverSamples
PreTriggerSamples
PostTriggerSamples
```

Each has one exact semantic Spec:

```text
TriggerThresholdCodeSpec
ReleaseThresholdCodeSpec
RequiredTimeOverSamplesSpec
PreTriggerSamplesSpec
PostTriggerSamplesSpec
```

### Shared Spec contract

All five Specs directly subclass TensorCore `TensorKernelSpec`. They do not
subclass `QuantityKernelSpec` because literal ADC codes and source-bin counts
carry no Pint Unit.

Every Spec requires:

- exact `torch.int64` dtype;
- no operation axes;
- no `TimeAxis` conditioning axis.

The generic form remains:

```python
class TriggerThresholdCodeSpec[
    ConditioningAxesT: tuple[TensorAxis[Any], ...],
    OperationAxesT: tuple[TensorAxis[Any], ...],
](
    TensorKernelSpec[ConditioningAxesT, OperationAxesT],
):
    ...
```

`operation_axes == ()` is exact. A global coefficient is one rank-zero Kernel:

```text
conditioning_axes == ()
operation_axes == ()
tensor shape == ()
```

A varying coefficient may condition on any subset of non-Time output roles.
Preparation requires exact role identity and coordinate coverage, reorders
conditioning coordinates, permutes conditioning dimensions, moves to the
output device, and preserves exact `torch.int64`.

A coefficient conditioned on `TimeAxis` is rejected intrinsically. A
coefficient conditioned on an alien or absent semantic role is rejected during
Product preparation. Time-dependent ZLE thresholds or sample-count policies
are outside the initial raw law.

### Kernel value domains

The exact local laws are:

```text
TriggerThresholdCode:
    values >= 0

ReleaseThresholdCode:
    values >= 0

RequiredTimeOverSamples:
    values >= 1

PreTriggerSamples:
    values >= 0

PostTriggerSamples:
    values >= 0
```

Each Kernel requires its exact semantic Spec and represented-value law through
an ordered declarative `_require()` composition.

The relationship:

```text
release_threshold_code >= trigger_threshold_code
```

is checked pointwise after both Kernels are aligned to the output domain in
`EncodedWaveform.prepare()`. It does not belong to either leaf in isolation.

TensorDSLab deliberately does not:

- infer thresholds from observed waveform minima or maxima;
- normalize ADC codes;
- derive a baseline;
- convert a physical amplitude or PE fraction;
- rescale from BitDepth;
- clip an unreasonable threshold;
- require either threshold to be below the intended maximum ADC code;
- reject a time-over/pre/post count merely because it exceeds the TimeAxis
  count; or
- repair a scientifically poor parameter choice.

The application/backend owns selecting literal policies consistent with its
digitizer, calibration, and intended selection. Counts larger than the
available TimeAxis remain well-defined through clipping/no-qualification
semantics.

### EncodedWaveformKernels

The exact public collection is:

```python
@final
class EncodedWaveformKernels(
    TensorCollection[TensorKernel[Any]],
):
    """Hold the exact raw-ZLE coefficient set."""

    __slots__ = ()

    def _require(self) -> None:
        require_exact_member_types(
            self,
            required=(
                TriggerThresholdCode,
                ReleaseThresholdCode,
                RequiredTimeOverSamples,
                PreTriggerSamples,
                PostTriggerSamples,
            ),
        )
```

It exposes exact typed properties:

```text
trigger_threshold_code
release_threshold_code
required_time_over_samples
pre_trigger_samples
post_trigger_samples
```

The collection requires all five members exactly once. Empty, partial,
duplicate, inherited, or alien membership is invalid. Its movement remains
TensorCore device-only and preserves every member's exact integer dtype and
semantic subtype.

## EncodedWaveformConfig

The caller-facing exact shape is:

```python
@final
@dataclass(
    frozen=True,
    slots=True,
    eq=False,
    repr=False,
    kw_only=True,
)
class EncodedWaveformConfig:
    spec: EncodedWaveformSpec[Any]
    kernels: EncodedWaveformKernels
```

Like every generated Product Config, the concrete implementation also owns
private `init=False` prepared execution facts. The exact initial inventory is:

```python
_is_prepared: bool
_source_specs: tuple[QuantityFieldSpec[Any], ...]
_working_dtype: torch.dtype | None
_time_dimension: int | None
_kernel_dimensions: tuple[
    tuple[int, ...] | None,
    tuple[int, ...] | None,
    tuple[int, ...] | None,
    tuple[int, ...] | None,
    tuple[int, ...] | None,
]
```

Unprepared state is:

```text
_is_prepared == False
_source_specs == ()
_working_dtype is None
_time_dimension is None
every _kernel_dimensions entry is None
```

Prepared state retains:

- the exact admitted source-Spec tuple and immutable Spec objects;
- exact output/source signed dtype as `_working_dtype`;
- exact output Time dimension; and
- exact output dimensions corresponding to every aligned Kernel
  conditioning axis.

The Config stores no Runtime or Plan record, tensor-derived support, source
Field, output Field, callable, mutable cache, coordinate lookup table, physical
duration, RNG role, donor interval, or record metadata.

Preparation returns a fresh `EncodedWaveformConfig` of the same exact type. It
does not mutate the caller's Config or Kernels.

## Public Lifecycle

### Signatures

The exact public classmethod shape follows the Maintenance 15 invariant-safe
source annotation:

```python
@classmethod
def prepare(
    cls,
    *,
    source_specs: tuple[QuantityFieldSpec[Any], ...],
    config: EncodedWaveformConfig,
) -> EncodedWaveformConfig:
    ...

@classmethod
def produce(
    cls,
    *,
    sources: tuple[TensorField[Any], ...],
    config: EncodedWaveformConfig,
) -> Self:
    ...

@classmethod
def validate(
    cls,
    *,
    product: Self,
    sources: tuple[TensorField[Any], ...],
    config: EncodedWaveformConfig,
) -> None:
    ...

@classmethod
def create(
    cls,
    *,
    sources: tuple[TensorField[Any], ...],
    config: EncodedWaveformConfig,
) -> Self:
    ...
```

`create()` performs exactly:

```text
prepare(source Specs, Config)
produce(source Fields, prepared Config)
validate(Product, source Fields, prepared Config)
return Product
```

No method accepts a scalar threshold, raw tensor shortcut, physical duration,
RNG, device override, dtype override, sentinel override, or compatibility
alias.

### Preparation order

`prepare()` is the complete fail-closed cold-path boundary. It performs the
following ordered phases before production:

1. require exact `EncodedWaveformConfig`;
2. require exact tuple input with exactly one source Spec;
3. require the source Spec to be exact `DigitizedWaveformSpec`;
4. require exact Config component types;
5. require output and source Units to be exact package-registry values
   compatible with dimensionless ADC codes, and require the two Units to be
   structurally equal so production applies no scale conversion;
6. require exact ordered Axis-tuple structural equality;
7. require exact equal shape and device;
8. require exact equal supported signed dtype;
9. locate exactly one `TimeAxis`;
10. require exact `RegularCoordinates` with `step == 1`;
11. align/move every Kernel to output conditioning coordinates and device
    without changing its exact `torch.int64` representation;
12. require every Kernel to omit TimeAxis conditioning after alignment;
13. compare the aligned release and trigger thresholds pointwise and require
    `release >= trigger`;
14. compile the Time dimension and five Kernel-dimension tuples;
15. preflight output allocation and every required Boolean/int64 scratch
    representation, including the transition workspace whose Time extent is
    `T + 1`;
16. retain exact positional source-Spec provenance; and
17. return one fresh prepared same-type Config.

The source and output Axis tuples are structurally equal and in the same order,
so production performs no source permutation. The prepared source-Spec binding
nevertheless remains mandatory: a separately staged caller must not prepare
with one source and produce with another structurally different source.

Preparation may inspect Axis/Coordinate/Unit objects and may synchronize to
validate the small aligned Kernel relationships. It performs no Product
allocation, source-code transformation, retained-support computation, or RNG
request.

### Production entry

Before reading source tensor values or allocating scratch/output storage,
`produce()` requires:

- exact prepared Config;
- exact one source Field;
- exact `DigitizedWaveform` source;
- exact `DigitizedWaveformSpec`;
- positional exact structural equality between the live source Spec and the
  retained prepared source Spec; and
- the complete non-`None` private prepared fact inventory.

Structural Spec equality includes exact concrete Spec type, axes/coordinates,
device, dtype, Unit, and every downstream Spec field. A structurally equal
distinct source Spec object is admitted. A changed Axis, coordinate, order,
device, dtype, Unit, semantic Spec class, count, or tuple position fails before
tensor arithmetic, allocation, or RNG use.

Production then consumes only:

- the source tensor;
- prepared aligned Kernel tensors;
- prepared Kernel dimensions;
- prepared Time dimension;
- output dtype/device/shape; and
- `config.spec.suppression_code`.

It performs no Pint conversion, Axis lookup, coordinate search, device move,
dtype-policy selection, Kernel alignment, donor call, Python-side per-sample
state machine, or RNG operation.

### Validation

`validate()` requires:

- exact `EncodedWaveform` Product;
- exact prepared Config;
- exact Product Spec identity with the prepared output Spec;
- the same positional live-source provenance used by production;
- exact shape/device/dtype/Unit/Axis relationships;
- intrinsic sentinel-or-nonnegative represented values;
- exact source value equality at every retained position;
- exact suppression code at every non-retained position under an independent
  focused support oracle;
- source immutability;
- fresh contiguous output storage;
- storage disjoint from the source and every prepared Kernel; and
- no post-construction package write through an alias.

The production algorithm must not call the donor implementation. Validation
tests may use the exact donor parity harness described below.

## Literal Raw-ZLE Law

Consider one independent lane:

```text
x[0], x[1], ..., x[T - 1]
```

with aligned scalar lane values:

```text
q:
    trigger_threshold_code
p:
    release_threshold_code
r:
    required_time_over_samples
a:
    pre_trigger_samples
b:
    post_trigger_samples
```

The prepared relationships are:

```text
0 <= q <= p
r >= 1
a >= 0
b >= 0
```

### Trigger qualification

A run qualifies when at least `r` consecutive samples satisfy:

```text
x[t] <= q
```

Comparisons are inclusive. Qualification is recognized when the required
count is reached, but the trigger start is backdated to the first sample of
the qualifying run.

A run of `r - 1` or fewer trigger-threshold samples does not qualify and does
not create retained support.

### Release hysteresis

After qualification, the excursion remains active while:

```text
x[t] <= p
```

Because `p >= q`, a sample in:

```text
q < x[t] <= p
```

ends the consecutive trigger-threshold run but continues the already-qualified
active excursion. The excursion ends immediately before the first sample with:

```text
x[t] > p
```

When `p == q`, there is no separate hysteresis band.

### Equivalent episode definition

For every maximal interval `[u, v)` whose samples all satisfy `x[t] <= p`:

1. find the first run of at least `r` consecutive samples satisfying
   `x[t] <= q`;
2. if no qualifying run exists, the interval contributes no support;
3. if the first qualifying run begins at `s`, define the raw episode as
   `[s, v)`.

Samples in `[u, s)` are not inherently active. They are retained only if the
pre-trigger extension reaches them.

### Pre/post extension

Every raw episode `[s, v)` produces:

```text
[max(0, s - a), min(T, v + b))
```

Pre/post values are copied from the original source, including values above
both thresholds.

### Canonical support

The final lane support is the union of all padded qualifying episodes:

```text
support = union(padded episodes)
```

Touching or overlapping padded intervals are canonically one uninterrupted
dense retained region. No record boundary can be represented in the Field
without separate metadata, and Maintenance 18 adds no such metadata.

The output is:

```text
output[t] =
    x[t],                     when support[t]
    spec.suppression_code,    otherwise
```

The encoder never writes a threshold, filtered value, baseline-subtracted
value, normalized value, or Boolean indicator into a retained position.

### Holdoff and retrigger interpretation

The donor derives:

```text
holdoff = pre_trigger_samples + post_trigger_samples
```

It has no sixth caller-controlled holdoff parameter. Under the dense Product
contract, donor retrigger extension, overlapping padding, and touching records
are represented exactly by the union of padded qualifying episodes.

A too-short threshold excursion does not qualify merely because it lies in
another episode's holdoff. It may appear in output only when ordinary pre/post
support from a genuine qualifying episode covers it.

### Boundaries

For every independent example/lane:

- pre-trigger extension clips at index zero;
- post-trigger extension clips at exclusive index `T`;
- a qualifying excursion reaching the final sample is retained through the
  final sample;
- a run that first reaches `r` at the final sample is valid;
- a boundary run shorter than `r` is invalid;
- no sample is borrowed from another example or lane;
- no imaginary source value is synthesized; and
- zero-length TimeAxis input produces a fresh zero-length output without
  qualification.

Large `r`, `a`, or `b` values remain accepted. Production uses exact
saturating effective counts:

```text
effective required count:
    values greater than T can never qualify

effective pre/post extension:
    min(configured count, T)
```

The implementation must avoid signed-integer overflow while computing these
effective values.

## Tensor-Native Execution Design

The exact implementation strategy is representation-independent at the public
boundary but must be tensor-native. A Python loop over Time samples or detector
lanes is not accepted production.

The selected execution works with Time moved to the final scratch dimension
and all remaining axes flattened into independent lanes. This is a prepared
representation detail; output is reconstructed in the exact original ordered
Spec shape.

### Prepared lane values

Each aligned Kernel tensor is reshaped/broadcast through its prepared
conditioning-dimension tuple. It is constant along Time by construction.

Production may flatten those broadcast lane values to one value per lane. It
must not materialize a full Time-dependent threshold tensor merely to repeat
one lane value across all samples.

### Trigger-run detection

For lane tensor `x[L, T]`:

```text
trigger_mask = x <= q
release_mask = x <= p
```

An int64 prefix sum over `trigger_mask` plus a gathered, lane-specific window
start proves every qualifying end position:

```text
trigger_count(t, r)
    = prefix[t + 1] - prefix[t + 1 - r]

qualifying_end(t)
    = t + 1 >= r and trigger_count(t, r) == r
```

Configured `r > T` must be handled as never qualifying without negative gather
indices or overflow.

### Release components and first qualifying start

Maximal true components of `release_mask` receive lane-local component IDs
from transition detection and cumulative sum. Component IDs must be made
globally unique across flattened lanes before a reduction.

For every qualifying end:

```text
qualifying_start = qualifying_end_index - r + 1
```

An exact minimum reduction per global release-component ID selects the first
qualifying start. Every release sample at or after that start belongs to the
raw episode; earlier release-band samples do not.

The implementation may use a fresh private `scatter_reduce` workspace or an
equivalent exact public Torch operation. It must not call a Python donor state
machine, round-trip support through host lists, or synchronize per sample.

### Padding and union

Transitions of the raw episode mask identify exact start and exclusive-end
positions. Lane-specific effective pre/post counts shift those positions:

```text
padded_start = max(0, raw_start - effective_pre)
padded_end   = min(T, raw_end + effective_post)
```

A fresh int64 difference/event tensor of shape:

```text
[lane_count, T + 1]
```

receives `+1` at padded starts and `-1` at padded ends. An int64 cumulative sum
is positive exactly on the union of every padded episode.

The `T + 1` extent and complete event tensor receive explicit preparation-time
capacity preflight. Empty Time is handled without attempting an invalid
transition allocation.

### Output construction

One `torch.where` or equivalent exact Torch operation constructs fresh output:

```text
where(support, source, suppression_code)
```

The suppression scalar is created on the source device in the exact source
dtype from the already-validated Spec integer. The result is returned
contiguous in the exact original Spec dimension order.

Production must not mutate:

- source tensor;
- Kernel tensor;
- Config;
- Spec;
- support after Product construction; or
- output after the semantic Field is constructed.

### Autograd and randomness

Signed integer source/output tensors do not participate in autograd.
Maintenance 18 adds no differentiability claim.

The transformation consumes no RNG, creates no RNG role, address, word, or
distribution, and changes none of the existing eight stochastic streams.

## Failure Types And Effect Ordering

Use `TypeError` for wrong public representation/type admission:

- wrong Config/Spec/Kernels/Product class;
- wrong tuple/member class;
- non-TensorField source;
- source not exact `DigitizedWaveform`;
- wrong source Spec class;
- unsupported or mismatched dtype;
- non-int or Boolean suppression code;
- Kernel Spec not exact semantic type;
- Kernel dtype not exact `torch.int64`;
- operation axis present;
- TimeAxis used as Kernel conditioning role;
- TimeAxis coordinates not exact `RegularCoordinates`; or
- alien public component type.

Use `ValueError` for valid representations with invalid semantic values or
relationships:

- nonnegative suppression code;
- suppression code outside dtype range;
- invalid EncodedWaveform negative value;
- missing or extra source;
- changed prepared source provenance;
- source/output axes, coordinates, shape, device, Unit, or dtype mismatch;
- missing TimeAxis;
- TimeAxis regular step other than one;
- negative threshold/count value;
- zero required-time-over value;
- aligned release threshold below trigger threshold;
- BitDepth maximum code outside selected DigitizedWaveform dtype;
- allocation/address/scratch capacity failure;
- unprepared Config;
- retained output value differing from source;
- off-support value differing from the suppression code; or
- output freshness/contiguity/storage failure.

All public admission, source-provenance, relationship, and capacity failures
must occur before source tensor arithmetic, output allocation, or RNG use.
There is no RNG in this Product, but the ordering is retained as a package
invariant.

Torch backend failures after valid production begins are ordinary execution
failures. No rollback claim is made for launched backend work. No partially
constructed Product is returned.

## Donor Parity Boundary

### Exact donor

The initial behavior promotes only the raw interval state machine from the
inspected IV-DSLab checkout:

```text
repository checkout label:
    iv-dslab-main_db_PB

source:
    src/dselec/zle.py
SHA-256:
    c06b5e9cdf35ec41e487518e3b1b0baa0c957899645bbd9ac2479c902bb1b304
exact symbol:
    _find_zle_intervals

tests:
    tests/test_zle.py
SHA-256:
    ab85ec0f4deff32c1a3bdba81a6a7617c12f9aaf73d17b5e0dfa6eb6424ed187

configuration:
    data/config_files/dselec.ini
SHA-256:
    fd42244bb4405dc328496efb8043fff522584a1922b811246670ac0e940e1c64
```

The wrapper `find_zle_intervals` and the current config-selected
`zle.algo = "Downsampled"` are not the initial parity boundary.

The absolute local checkout route is an uncommitted Validation input and must
not enter repository records. File identities are bound by the repository
label, relative path, exact symbol, and SHA-256 above.

The parity harness must explicitly isolate the exact raw recurrence and inject
the same resolved literal ADC thresholds and sample counts as TensorDSLab. It
must not claim parity by running the donor's current default wrapper.

Donor files are read-only evidence. Implementation must not modify the donor
checkout, import it as a package dependency, or copy its package structure,
record classes, wrappers, configuration system, or downstream machinery into
TensorDSLab production.

### Exact observable

For each example, detector coordinate tuple, and source Time index, define:

```text
IVSupport[t] =
    true when one raw donor interval satisfies
    interval.sample <= t < interval.sample + interval.length

TensorDSLabSupport[t] =
    encoded.tensor[t] != encoded.spec.suppression_code
```

Required parity is:

```text
TensorDSLabSupport == IVSupport

encoded.tensor[IVSupport]
    == digitized.tensor[IVSupport]

encoded.tensor[~IVSupport]
    == encoded.spec.suppression_code
```

Comparison is exact, with no tolerance.

### Intentional representation divergences

Parity deliberately excludes:

- donor interval ordering;
- donor interval count when touching intervals collapse;
- donor record identity;
- ragged storage;
- channel-map record layout;
- record integrals;
- `nhits`;
- allocation strategy;
- internal state-machine representation;
- wrapper-selected downsampling;
- FIR/IIR filtering;
- fixed-point FPGA arithmetic; and
- durable DAQ record encoding.

Touching donor records have identical dense sample support and are
canonically indistinguishable in `EncodedWaveform`.

`docs/parity.md` must record the donor hashes, symbol, exact support oracle,
accepted parity classification, and intentional divergences.

### Worked acceptance example

The following exact fixture is mandatory:

```text
trigger_threshold_code:
    950
release_threshold_code:
    970
required_time_over_samples:
    3
pre_trigger_samples:
    2
post_trigger_samples:
    3

source:
    1000 1000 1000 1000 948 945 940 960 975 1000
    1000 949 947 944 965 980 1000 1000 949 948

first raw episode:
    [4, 8)
first padded episode:
    [2, 11)

second raw episode:
    [11, 15)
second padded episode:
    [9, 18)

union:
    [2, 18)

output with suppression_code = -1:
    -1 -1 1000 1000 948 945 940 960 975 1000
    1000 949 947 944 965 980 1000 1000 -1 -1
```

The two final low samples are only a two-sample run and do not qualify.

## Quickstart Update

Maintenance 18 must update the accepted application-neutral
`demos/readout.ipynb`. The notebook remains a newcomer demonstration, not a
developer document or an application profile.

### Product construction

After the existing `DigitizedWaveform` section, add one plain-language section
that explains:

- `EncodedWaveform` represents which ADC samples a DAQ-like ZLE stage keeps;
- retained values remain exact ADC codes;
- the configured negative value means no code was retained;
- threshold and sample-count Kernels are global in this small example but may
  be conditioned on non-Time roles; and
- the settings are illustrative rather than detector calibration.

Construct exact global `torch.int64` Kernels:

```text
TriggerThresholdCode:
    2500
ReleaseThresholdCode:
    2800
RequiredTimeOverSamples:
    3
PreTriggerSamples:
    25
PostTriggerSamples:
    50
```

Construct:

```text
EncodedWaveformSpec dtype:
    torch.int32
EncodedWaveformSpec unit:
    dimensionless
EncodedWaveformSpec suppression_code:
    -1
```

Then call exactly one public:

```python
encoded_waveform = EncodedWaveform.create(
    sources=(digitized_waveform,),
    config=encoded_waveform_config,
)
```

The existing `DigitizedWaveform` quickstart stays `torch.int32`, so the new
simulation path demonstrates exact dtype preservation.

On the accepted deterministic notebook inputs, these values produce visible
support around all four pulse responses without retaining the complete
waveform. The expected retained index intervals are evidence for the focused
demo test:

```text
sensor-0:
    [280, 362)
    [3721, 4261)

sensor-1:
    [1363, 1749)

sensor-2:
    [2541, 3085)
```

The exact intervals are bound to the accepted deterministic TensorCore
`0.22.0`, TensorDSLab quickstart values, eager CPU stack, and raw-ZLE law. They
are demo regression evidence, not calibration or cross-backend parity
goldens.

### Shared-shape proof

Rename the newcomer prose from six Products to seven and add exactly one light
assertion:

```python
assert encoded_waveform.tensor.shape == expected_shape
```

The notebook continues to use only the seven shared-shape assertions. It must
not add a large internal validation block.

### Product views

Expand the figure from six to seven aligned panels:

```text
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
EncodedWaveform
```

All three sensors retain their stable blue/orange/green colors in every panel.
The EncodedWaveform panel is a step panel.

At the explicit presentation boundary only, replace the configured suppression
code with `float("nan")` in the ordinary CPU plotting list. This creates visible
gaps instead of drawing misleading vertical/zero-like lines through suppressed
regions. The TensorDSLab Product itself remains an untouched integer tensor.

The final panel label must make the sentinel meaning clear without presenting
it as a physical ADC value. Suggested newcomer prose:

> The last panel leaves suppressed regions blank. Every visible segment is an
> unchanged ADC-code interval retained by the ZLE stage.

The figure remains:

- one display;
- one PNG;
- one non-overlapping title/legend;
- seven readable aligned panels;
- one shared Time axis;
- stable sensor colors;
- visible pulses and PSD noise;
- non-railed ADC codes; and
- visibly suppressed EncodedWaveform gaps with all four pulse regions present.

### Notebook structure

The source-only notebook becomes exactly:

```text
22 cells
11 Markdown cells
11 code cells
strict Markdown/code alternation
```

Every code cell remains immediately preceded by simple newcomer prose.
Execution counts and outputs remain absent from the committed notebook.

Imports remain public and limited to:

```text
Python standard library
torch
matplotlib.pyplot
tensor_core public facade
tensor_dslab public facade
```

No NumPy, private TensorDSLab module, application package, profile, Readout
collection, filesystem read, network access, or dynamic dependency install is
allowed.

## Package Topology And Public API

### New package

Add exactly:

```text
tensor_dslab/
  encoded_waveform/
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

`runtime/__init__.py` imports and exports nothing.

Ownership is:

```text
field.py:
    EncodedWaveformSpec
    EncodedWaveform

kernel.py:
    five coefficient Specs
    five coefficient Kernels
    EncodedWaveformKernels

config.py:
    EncodedWaveformConfig

runtime/prepare.py:
    prepare_encoded_waveform

runtime/produce.py:
    produce_encoded_waveform

runtime/validate.py:
    validate_encoded_waveform
```

No `_produce.py`, generic encoder, ZLE framework, state-machine object,
interval record class, Runtime record, Plan, factory registry, compatibility
module, or forwarding import is added.

### Root facade

Add exactly these 14 public names:

```text
EncodedWaveform
EncodedWaveformConfig
EncodedWaveformKernels
EncodedWaveformSpec
PostTriggerSamples
PostTriggerSamplesSpec
PreTriggerSamples
PreTriggerSamplesSpec
ReleaseThresholdCode
ReleaseThresholdCodeSpec
RequiredTimeOverSamples
RequiredTimeOverSamplesSpec
TriggerThresholdCode
TriggerThresholdCodeSpec
```

The root facade changes from exactly `61` names to exactly `75` names.

The new `tensor_dslab.encoded_waveform` facade exports exactly those same
14 names in the same relative root order.

Existing facade names and relative order remain unchanged. No name is removed,
renamed, aliased, or forwarded.

### Topology target

The exact package target is:

```text
73 package files
72 Python modules
```

The existing package has `65` files / `64` Python modules. The new Product
package adds eight Python files/modules and no data file.

The package version remains `0.2.0`. Dependency spelling remains unchanged.
The existing `[demos]` optional dependency group remains unchanged.

## Declarative Requirements

Maintenance 16 remains operative:

> Semantic class contracts should read as ordered compositions of narrowly
> named requirements.

Reusable mechanics belong in the narrowest existing private requirements
module:

```text
common/requirements/axis.py:
    regular TimeAxis coordinate law when genuinely reusable

common/requirements/field.py:
    exact Field Spec and freshness relationships

common/requirements/kernel.py:
    no operation axes and Time-conditioning exclusion

common/requirements/tensor.py:
    signed dtype, suppression/value domains, exact integer domains

common/requirements/config.py:
    exact Config components and prepared source provenance

common/requirements/capacity.py:
    output/scratch allocation preflight
```

The implementation must not create `encoded_waveform/requirements.py`, a
generic ZLE helper framework, or duplicate a reusable `require_*` outside
`common/requirements`.

One-off raw-ZLE scientific relationships may remain narrowly visible in the
Product runtime when moving them would create a misleading generic primitive.
No rule requires every validation expression to leave its semantic owner.

Every introduced module-level function, class, or Protocol and every new
module receives one intentional semantic docstring under the repository
readability policy.

## Exact Implementation Scope

Implementation may change only:

```text
AGENTS.md
CONTRIBUTING.md
README.md
demos/readout.ipynb
docs/api.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_18_encoded_waveform_raw_zle.md
docs/overview.md
docs/parity.md
docs/validation.md
tensor_dslab/__init__.py
tensor_dslab/common/requirements/axis.py
tensor_dslab/common/requirements/capacity.py
tensor_dslab/common/requirements/config.py
tensor_dslab/common/requirements/field.py
tensor_dslab/common/requirements/kernel.py
tensor_dslab/common/requirements/tensor.py
tensor_dslab/digitized_waveform/field.py
tensor_dslab/digitized_waveform/runtime/prepare.py
tensor_dslab/encoded_waveform/**
tests/test_digitized_waveform.py
tests/test_encoded_waveform.py
tests/test_kernel_contracts.py
tests/test_package_contracts.py
tests/test_product_configs.py
tests/test_product_types.py
tests/test_quantity_representations.py
tests/test_readout_demo.py
tests/test_requirements.py
tests/typing/maintenance_15_spec_composed_products.py
tests/typing/negative/maintenance_15_spec_composed_products.py
```

The wildcard is constrained to the exact eight-file package topology listed
above. No other path is authorized.

In particular, Implementation must not change:

```text
pyproject.toml
create_environment.sh
tensor_dslab/common/alignment.py
any existing Product other than the exact DigitizedWaveform dtype/preflight paths
any RNG/distribution implementation
any TensorCore source
any downstream repository
any donor source
any historical implementation record
```

If implementation discovers that one additional existing path is required,
it must stop and return the exact contradiction to Design. It must not widen
the allowlist silently.

## Test Reconciliation

### Curation target

Add exactly one new test module:

```text
tests/test_encoded_waveform.py
```

Existing test modules absorb facade, typing, Config, Kernel, requirement,
digitizer-dtype, and notebook changes. Do not add a separate test module for
each new semantic class.

The target is:

```text
25 tracked Python test/support files
70 discovered unittest methods
no more than 22 TestCase classes
no more than 5,000 physical Python test lines
```

The exact 10 new methods should be concentrated in
`tests/test_encoded_waveform.py`; existing methods are extended rather than
multiplied. Subtests and compact table-driven fixtures should cover the
behavioral matrix without repetitive constructor boilerplate.

No `assertEqual(index, index)`, dynamic test attachment, broad generated
near-duplicates, package-module-count-only brittleness without an API
relationship, or untyped fixture union is permitted.

### Required behavioral evidence

The committed suite must prove:

1. exact five Spec/Kernel leaves, dtype, geometry, Time-conditioning
   exclusion, value domains, collection membership, properties, and public
   exports;
2. exact configurable signed dtype and sentinel admission on
   `EncodedWaveformSpec`;
3. exact standalone Field sentinel-or-nonnegative value law, including valid
   retained zero;
4. exact `DigitizedWaveformSpec` signed dtype generalization and BitDepth
   capacity preflight;
5. exact source count/type/Spec/axes/order/coordinates/device/dtype/Unit and
   regular-Time preparation law;
6. exact aligned pointwise `release >= trigger` relationship under global and
   conditioned coefficients;
7. exact prepare-produce-validate and one-shot create paths;
8. exact staged source-Spec binding for changed axes/order/coordinates/device/
   dtype/Unit/semantic Spec/count;
9. no crossing and exactly `r - 1` versus exactly `r` trigger samples;
10. trigger-threshold and release-threshold equality;
11. hysteresis-band continuation and termination above release;
12. long qualifying runs;
13. too-short and qualifying retriggers;
14. overlapping, touching, and separated padded episodes;
15. pre/post clipping at both boundaries;
16. qualification exactly at the final sample and insufficient final run;
17. zero-length TimeAxis;
18. configured counts larger than TimeAxis without overflow;
19. exact retained source zero and maximum intended code;
20. configurable non-`-1` negative sentinel;
21. multiple examples with strict state independence;
22. multiple detector lanes with strict independence;
23. invariance of one lane when unrelated lanes change;
24. global, Example-conditioned, Channel-conditioned, and combined
   non-Time-conditioned policies with coordinate permutation;
25. deterministic replay;
26. exact source-code preservation on support;
27. exact sentinel off support;
28. source and Kernel immutability;
29. fresh contiguous disjoint output storage;
30. exact worked acceptance example;
31. exact donor support parity over named fixtures;
32. exhaustive short-state parity through sequence length nine over the three
   relation states `x <= q`, `q < x <= p`, and `x > p` for a bounded table of
   `r/a/b` values;
33. public import isolation and absence of readout/reconstruction workflow
   surfaces;
34. exact `75`-name root and `14`-name subpackage facades;
35. strict positive typing for exact semantic Spec and Product composition;
36. strict negative typing for wrong Coordinates, wrong Kernel Spec, wrong
   source type, wrong Config/Kernels type, and unsupported semantic
   composition; and
37. updated source-only newcomer notebook construction, execution, shape,
   interval, plot, and public-import contracts.

### Donor harness

The test suite may reproduce the narrow raw recurrence as an independent
test-only oracle with an explicit donor citation and hash. It must not import
the donor checkout at runtime as a dependency.

Validation separately reads and hashes the exact donor files, executes the
narrow donor recurrence or an isolated exact copy against the same fixtures,
and compares exact support. If the donor checkout is unavailable, Validation
must return a qualification/finding rather than silently replacing the exact
donor identity with a different snapshot.

The committed production algorithm and the independent parity oracle must not
call each other or share interval/support helpers.

## Required Mutation Matrix

Validation must inject and kill at least the following exact 30 high-risk
mutants on private temporary copies:

1. restore exact `torch.int32`-only DigitizedWaveformSpec;
2. admit unsigned DigitizedWaveform/EncodedWaveform dtype;
3. omit BitDepth-to-output-dtype capacity preflight;
4. admit Boolean suppression code;
5. admit nonnegative suppression code;
6. admit suppression code outside dtype range;
7. admit any negative EncodedWaveform value;
8. allow source/output dtype cast instead of exact equality;
9. admit non-DigitizedWaveform source;
10. ignore changed staged source-Spec provenance;
11. admit non-Regular TimeAxis coordinates;
12. admit `RegularCoordinates.step != 1`;
13. admit TimeAxis-conditioned policy Kernel;
14. admit an operation axis on a policy Kernel;
15. weaken exact `torch.int64` Kernel dtype;
16. weaken required-time-over positivity to nonnegative;
17. reverse or omit aligned `release >= trigger`;
18. change inclusive trigger comparison from `<=` to `<`;
19. change inclusive release comparison from `<=` to `<`;
20. qualify `r - 1` trigger samples;
21. start support at qualification end rather than backdated run start;
22. retain pre-qualification release-band samples without pre-trigger cover;
23. end an episode when leaving trigger rather than release threshold;
24. allow a too-short excursion to retrigger during holdoff;
25. omit pre-trigger clipping or borrow from another example;
26. omit post-trigger clipping or carry state to another example;
27. preserve only interval boundaries instead of dense union support;
28. replace retained source codes with threshold/Boolean values;
29. use hard-coded `-1` instead of the Spec suppression code;
30. restore a Python per-Time-sample production loop or host-list round trip.

The exact candidate mutation record must name:

- mutation;
- changed path/symbol;
- named committed proof that fails;
- failure reason; and
- confirmation that candidate bytes were restored unchanged.

The notebook's existing 24-mutant Maintenance 17 proof matrix remains
applicable where the underlying cells are unchanged. Maintenance 18 adds
focused notebook mutations for:

- omission of EncodedWaveform construction;
- wrong ZLE Kernel literal;
- missing seventh shape assertion;
- suppression plotted as an ordinary ADC code rather than a gap;
- one sensor interval omitted;
- seventh panel omitted;
- source notebook committed with output; and
- isolated installed execution shadowed by the checkout.

## Implementation Evidence

Implementation must freeze one immutable direct-child candidate only after:

```bash
git diff --check
git show --check
PYTHONPATH=. python -m unittest discover -s tests
pyright
```

It must also provide:

- exact candidate commit/tree/parent/ref;
- exact allowlist and protected-byte proof;
- exact package/test/facade topology;
- focused new Product, digitizer, requirement, package, typing, and notebook
  results;
- complete source-form suite;
- positive Pyright zero diagnostics;
- exact negative-fixture diagnostic count with no incidental diagnostic;
- donor file identities and focused parity result;
- source-only notebook inventory and two immediate deterministic CPU
  executions;
- one saved private render for visual inspection, not committed;
- link/anchor/fence/Python-fence checks for every changed living page;
- intentional docstring census for introduced module definitions;
- privacy/raw-route scan;
- artifact/bytecode/cache/build hygiene; and
- final clean worktree.

Implementation does not own the complete mutation matrix, deterministic
artifact equivalence, isolated installed execution, fresh environment, or
independent donor/visual assessment. Those belong to Validation.

Implementation must not contact Validation or Review directly. It returns the
fixed candidate to Design for routing.

## Validation Gate

Validation receives one exact immutable Design-dispatched candidate and
independently performs:

1. exact identity/tree/parent/ref/direct-child and branch cleanliness;
2. exact allowlist, wildcard expansion, protected-byte equality, diff/show
   checks, and topology;
3. exact TensorCore source/archive/version/commit/tree/wheel identity;
4. focused Product/digitizer/Kernel/Config/requirement/typing/notebook tests in
   source and canonical extracted-archive forms;
5. complete source and archive test discovery;
6. positive Pyright zero diagnostics in both dependency forms;
7. exact negative typing diagnostics with no incidental diagnostic;
8. all 30 production mutants plus the focused notebook mutants;
9. exact donor hashes, raw recurrence isolation, worked fixture, named donor
   fixtures, exhaustive short-state comparison, and exact support/value
   parity;
10. deterministic `SOURCE_DATE_EPOCH=0` wheel and sdist builds from two
    independent exact archive roots;
11. source/artifact byte reconciliation, exact package/test/notebook/doc
    payload, metadata, dependencies, version, and facades;
12. isolated exact TensorCore-wheel plus TensorDSLab-wheel installation outside
    every checkout;
13. installed-site-packages/no-shadowing public import and Product smoke;
14. source-only notebook execution from the exact sdist outside the checkout;
15. two immediate exact notebook executions, summary/interval/plot
    determinism, and source immutability;
16. independent PNG inspection of seven readable panels, stable three-sensor
    colors, visible pulses/noise, non-railed ADC, and clear EncodedWaveform
    gaps;
17. living documentation links, anchors, fences, code fences, public census,
    current architecture, and parity wording;
18. privacy/raw-route/import/downstream isolation and retired-surface scans;
19. bytecode/cache/build/dist/egg-info/coverage hygiene; and
20. final exact detached cleanliness.

The fresh real-Conda environment may carry from Maintenance 17 only if
`pyproject.toml`, `create_environment.sh`, TensorCore dependency spelling,
package version, and complete optional demo dependency inputs are byte-exact.
The isolated artifact execution remains mandatory because production package
and notebook bytes change.

Validation stops at the first candidate-ineligibility boundary after
consolidating already-observed findings. It edits no candidate byte and does
not contact Review.

## Review Gate

Review receives only an exact Validation-cleared immutable candidate from
Design. It independently reads this complete work order, exact candidate diff,
and relevant production/tests/docs.

Risk-based Review must focus on:

- DAQ/readout versus reconstruction/application ownership;
- fieldful `EncodedWaveformSpec` semantics;
- configurable signed dtype and explicit negative sentinel;
- DigitizedWaveform BitDepth capacity;
- exact policy-Kernel geometry and non-Time conditioning;
- preparation failure ordering and staged source binding;
- example/lane independence;
- trigger/time-over/release/padding/union/boundary correctness;
- tensor-native execution without host/per-sample fallback;
- source value preservation and fresh storage;
- exact donor boundary and exclusions;
- quickstart newcomer clarity and honest plotting; and
- public topology, typing, docs, and protected scope.

Review may accept unchanged complete Validation evidence under evidence
economy, but must independently run focused adversarial cases and inspect the
exact implementation of the highest-risk semantic paths.

Any finding returns exact immutable bytes to Design. Review edits nothing.
After zero findings, Review returns the exact candidate identity and evidence
to Design for final same-byte approval.

Review may perform `git merge --ff-only` onto governed local `main` only after
explicit final Design approval of that exact candidate. It then verifies exact
commit/tree identity, linear ancestry, byte equality, diff/show checks, and
cleanliness. Review does not push.

## Candidate Loop

The finite loop is:

```text
ordinary Implementation candidates:
    at most 3
Validation returns:
    at most 3
Review returns:
    at most 2
```

Every candidate is immutable. A correction is one direct child of the
Design-authorized parent or an exact documentation-only amended authority as
explicitly dispositioned by Design.

A concrete architecture contradiction, exhausted loop, unavailable exact
donor identity, need for an unallowlisted path, or inability to preserve exact
raw support stops the route and returns to Design. No role silently widens the
contract.

## Completion

Maintenance 18 is complete only when:

- one exact candidate implements the complete accepted contract;
- independent Validation clears that exact candidate;
- independent Review returns zero findings on the same bytes;
- Design gives final same-byte approval;
- Review fast-forwards governed local `main` with `git merge --ff-only`;
- post-merge exact identity/diff/cleanliness checks pass; and
- the self-effecting lifecycle records truthfully resolve to
  **Merged / Closed** without an evidence-only rewrite.

No push follows automatically. Publication remains a separate explicit user
decision.

The completed evidence must remain qualified to its exact:

- TensorDSLab commit/tree;
- TensorCore `0.22.0` commit/tree;
- donor file hashes;
- Python/Torch/NumPy/Pint/Pyright versions;
- source/archive/artifact form;
- device/backend/execution mode; and
- unavailable-CUDA qualification.

## Deferred FPGA ZLE Decision Front End

A later named Design stage may change how retained support is selected while
preserving the public `EncodedWaveform` meaning.

Deferred topics include:

- ADC downsampling;
- downsampling phase/alignment;
- matched FIR filtering;
- IIR filtering;
- filter coefficient calibration;
- unsigned/fixed-point intermediate arithmetic;
- accumulator widths;
- rounding and truncation;
- saturation and overflow;
- source-bin mapping from decision bins;
- window-boundary filter initialization;
- explicit temporal halo/context;
- authoritative firmware comparison; and
- complete detector-DAQ equivalence.

Downsampling or filtering belongs to the private decision front end. It does
not imply a downsampled public `EncodedWaveform`: the Product retains the
source TimeAxis and exact original retained ADC codes.

Future context must never flow across ExampleAxis entries.

## Explicit Non-Goals

Maintenance 18 does not add:

- full detector calibration;
- a DS20k, Silex, 3DPi, or other application profile;
- positive-going or configurable-polarity ZLE;
- time-conditioned thresholds;
- threshold inference or baseline estimation;
- physical-duration-to-sample conversion;
- FPGA/downsampled/FIR/IIR selection;
- record-boundary metadata;
- ragged donor record objects;
- DAQ file parsing;
- durable IO, cache, artifact, schema, or recovery policy;
- corrupt/missing-data representation;
- `SignalWaveform`, `Hits`, `PulseMembership`, or any reconstruction Product;
- a reconstruction package, result collection, or `reconstruct()` facade;
- `Readout`, `ReadoutCollection`, `ReadoutConfig`, `simulate_readout`, or a
  package-owned workflow;
- generic Product/Config/encoder/ZLE framework;
- Runtime or Plan records;
- compatibility aliases or forwarding modules;
- new RNG roles or stochastic behavior;
- TensorCore changes;
- dependency or package-version changes;
- CUDA qualification;
- performance/fusion claims;
- calibration or detector-equivalence claims;
- release, deployment, compatibility, or production-readiness claims;
- merge, push, publication, tag, release, or package-index action; or
- downstream repository edits.

The accepted outcome is deliberately narrow: one reusable deterministic
terminal readout Product, the minimal signed-dtype correction needed to express
backend representation honestly, exact raw-ZLE parity evidence, and one clear
seventh-Product quickstart demonstration.
