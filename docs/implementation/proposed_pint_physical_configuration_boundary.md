# Candidate Pint Physical Configuration Boundary Work Order

Status: **Design draft / Undispatched**.

Stable candidate work-order key:
`TensorDSLab/pint-physical-configuration-boundary`.

This is a nonoperative TensorDSLab Design candidate. It records the intended
Pint-based public configuration boundary after TensorCore published the
required compact axes and generic `Scalar` surface in `0.13.0`. It is not an
implementation authority, dependency selection, compatibility claim, or
reservation of a Maintenance number. Before dispatch, Design must reconcile it
against the exact closed TensorDSLab Maintenance 5 migration and an exact
selected Pint release.

## Objective

Make physical units explicit where collaborators configure the simulation,
without carrying unit objects into tensor execution:

```text
caller Pint Quantity
  -> public Config construction
       validate dimension and scalar domain
       copy into TensorDSLab's private registry
       canonicalize to the product's documented unit
  -> prepare_<product>(...)
       extract each active required canonical magnitude once
       combine it with sampling, dtype, device, and algorithm facts
       construct a plain-number <Product>Runtime
  -> produce_<product>(...)
       tensor math and RNG only
  -> validate_<product>(...)
       product and relationship checks only
```

The candidate must:

- let collaborators express physical values naturally, such as
  `quantity(2, "ns")`, `quantity(500, "MHz")`, and
  `quantity(3.5, "mV")`;
- remove unit suffixes from public physical config field names;
- preserve explicit suffixes on private plain-number runtime facts;
- keep units out of TensorCore, tensor payloads, RNG addressing, hot paths,
  product fields, and completed collections;
- keep `SampleAxis` compact and integer-backed while offering deliberate Pint
  conversion conveniences at the TensorDSLab boundary;
- preserve the current scientific equations and operation order when the
  canonical numeric operands are unchanged; and
- retain the explicit `prepare -> produce -> validate` ownership established
  by Maintenance 4.

This stage is an API and representation migration, not an attempt to make
Pint a tensor library or an execution substrate.

## Authority, Current Evidence, And Dispatch Gate

Package authority is `TensorDSLab/default/Design`.

The Design draft was first written from clean TensorDSLab `main` at:

```text
repository:       TensorDSLab
commit:           a46899c4e3bacd6deec23ea64da5e68b382816e9
package version:  0.1.0
Python:           >=3.11
TensorCore pin:   0.9.0 at 4708bf2ca063a1bcd37a30a342733b9e3dbe9f59
```

TensorCore subsequently published the exact prerequisite:

```text
repository:       TensorCore
reference:        origin/main
commit:           202d8b1bc6259b8453d3d377570417f2480d782b
tree:             48fa9a28db6d043abc07d9963b2015983ca436ea
package version:  0.13.0
```

That dependency supplies the compact axes and `Scalar.require()` contract, but
TensorDSLab has not adopted it at the draft's starting baseline. Maintenance 5
is the separately authorized package-local adoption and sampling migration.
The values above are Design evidence only, not this later work's executable
baseline. The current package states remain:

```text
package_adoption_state:    Adopted
conformance_finding:       Not evaluated
coordination_status:       Deferred
registry_storage_profile:  Disabled
maintenance_4:             Merged / Closed
maintenance_5:             separately authorized prerequisite
stage_8:                   new authority required
this_candidate:            Design draft / Undispatched
```

Dispatch is prohibited until all of the following are true:

1. TensorCore `0.13.0` exact commit
   `202d8b1bc6259b8453d3d377570417f2480d782b` remains the accepted published
   dependency supplying compact axes and generic scalars.
2. TensorDSLab Maintenance 5 compact-axis and sampling migration has
   been implemented, validated, independently reviewed, merged, and closed.
   It must establish `ExampleAxis` as a nonempty `CountAxis`, `ChannelAxis` as
   a nonempty string `LabelAxis`, and `SampleAxis` as a positive regular
   integer-picosecond `RegularAxis`.
3. That closed prerequisite has retired `SamplingConfig` and
   `ReadoutConfig.sampling`, and `prepare_sampling(photoelectrons)` derives
   sampling facts from the source `SampleAxis`.
4. TensorDSLab Design has selected one exact Pint release after source,
   archive/wheel, typing, supported-Python, macOS, and full-A100 probes.
5. The living architecture documents and this work order have been
   synchronized against one clean Design authority commit.
6. The persistent TensorDSLab Implementation, Validation, and Review routes
   are Active, current, return-capable, and explicitly bound to this work.

The compact-axis migration is a separately closed prerequisite. This Pint
candidate must not absorb it into a combined production dispatch.

## Applicable Contracts And Source Precedence

Before dispatch, the fixed work order must reconcile:

- `AGENTS.md` for role, routing, authority, source-precedence, and finite-loop
  requirements;
- `CONTRIBUTING.md`, especially TensorCore Backbone, Product Semantics,
  Boundary-First Validation, Public Surface Discipline, Public Typing,
  Coordinates Versus Indices, Test Expectations, and Scope Discipline;
- [Rebuild Architecture](../architecture/rebuild.md);
- [Readout Architecture](../architecture/readout.md);
- [TensorCore Integration](../architecture/tensors.md);
- [Validation](../validation.md);
- [Parity](../parity.md);
- the closed Maintenance 4 work order and implementation as the exact current
  `prepare -> produce -> validate` ownership evidence;
- TensorCore's exact published `0.13.0` compact-axis and generic-Scalar API
  and integration documentation;
  and
- the selected Pint release's public documentation and package artifacts.

This candidate controls only the later Pint migration slice. Closed work
orders remain immutable historical evidence. Living architecture controls
current package meaning. If the published TensorCore axis contract or selected
Pint behavior conflicts with this candidate, Design must revise and refreeze
the work order rather than asking Implementation to improvise.

`docs/parity.md` requires synchronization only if the final stage changes a
donor-comparison boundary, a scientific equation, or an accepted numerical
comparison class. Adding explicit units alone does not authorize any of those
changes.

## Design Finding

Public configuration is the correct home for physical quantities. It is where
a collaborator says what a value means. A producer is the wrong home: by that
point the operation should already have exact, device-aware, algorithm-ready
numeric operands.

The selected boundary is therefore:

```text
Config:       physical meaning, dimensions, canonical unit, scalar domain
Preparation: contextual conversion to plain execution operands
Runtime:      immutable plain numbers and tensors only
Production:   tensor computation and RNG requests only
Validation:   completed-product and relationship checks only
```

This is intentionally stricter than merely accepting arbitrary Pint objects
throughout the package. Pint quantities are valuable at human-facing
boundaries but would make execution records registry-sensitive, add Python
object work to repeated numerical paths, complicate equality and copying, and
blur the existing action ownership.

The public API uses unit-neutral names because the value carries its unit:

```python
TimingJitterConfig(sigma=quantity(2.0, "ns"))
WhiteNoiseConfig(rms=quantity(0.35, "mV"))
```

Private plain-number values remain explicit:

```text
sigma_ns
rms_mv
sample_period_ps
power_density_mv2_per_hz
```

This naming tells a maintainer exactly where unit interpretation has ended.

## Package-Local Pint Ownership

TensorDSLab owns exactly one private `pint.UnitRegistry`, created in:

```text
tensor_dslab/common/units.py
```

The package deliberately exports only construction helpers:

```python
def quantity(magnitude: int | float, unit: str) -> pint.Quantity:
    ...


def quantities(
    magnitudes: tuple[int | float, ...],
    unit: str,
) -> tuple[pint.Quantity, ...]:
    ...
```

The exact type annotation spelling may be adjusted for the selected Pint
release, but the runtime contract is fixed:

- `quantity` accepts only exact built-in `int` or `float` magnitudes;
- `bool`, `complex`, `Decimal`, `Fraction`, arrays, tensors, lists, tuples, and
  arbitrary numeric duck types are rejected as scalar magnitudes;
- a nonfinite magnitude is rejected;
- the unit argument is exactly a nonempty `str` accepted by the private
  registry;
- `quantities` accepts exactly a tuple of accepted scalar magnitudes and
  returns an equal-length tuple of independently constructed scalar
  quantities;
- `quantities((), unit)` is accepted after validating the unit expression and
  returns exact `()`; a PSD Config independently rejects an empty physical
  table;
- neither helper accepts or constructs a Pint-wrapped Torch or NumPy array;
  and
- returned quantities belong to TensorDSLab's private registry.

The helpers raise `TypeError` for the wrong argument types and `ValueError`
for a nonfinite magnitude or an empty, malformed, or undefined unit
expression. They validate a unit expression even when `quantities` receives an
empty magnitude tuple.

The package root and `tensor_dslab.common` deliberately re-export
`quantity` and `quantities`. They do not export the registry, a `Q_` alias, a
mutable units object, an application registry, or a package-defined Quantity
wrapper class. TensorDSLab never calls `pint.set_application_registry(...)`.
The registry remains discoverable through a returned Pint object's own
implementation-facing attributes because Pint quantities are registry-bound.
Direct discovery or mutation of it through Pint internals is unsupported and
is not a TensorDSLab API.

One unexported common function, conceptually:

```python
def _canonical_quantity(
    value: object,
    *,
    unit: str,
    field: str,
    constraint: type[Scalar[float]],
) -> pint.Quantity:
    ...
```

owns the exact recognition, conversion, copying, normalized-scalar constraint,
and exception translation shared by all physical Config fields. `Scalar` and
the accepted leaf constraints are imported from TensorCore's package root.
Config classes add only genuinely field-specific nonzero, ordering, tuple, and
cross-field rules. TensorDSLab must not implement 26 subtly different copies
of the registry boundary.

TensorCore remains Pint-free. Count, regular, and label-axis representation is
generic TensorCore behavior; physical time interpretation and unit conversion
are TensorDSLab behavior.

### External registries and defensive copies

Public physical config fields accept compatible scalar Pint quantities from a
caller's registry. Construction must not compare registry-owned Unit objects
or attach an external-registry object directly. For each physical field, the
common canonical-copy function performs these steps and returns the fresh
canonical quantity:

1. requires a supported real Pint Quantity and an exact built-in `int` or
   `float` source magnitude before invoking any conversion;
2. calls `.to(canonical_unit)` on the input quantity in its own registry;
3. requires the converted magnitude to remain an exact built-in `int` or
   `float`;
4. calls exactly once
   `normalized = constraint.require(converted_magnitude, field)` and
   translates an `OverflowError` into the candidate's public `ValueError`
   boundary while preserving `TypeError` and constraint `ValueError`;
5. constructs a fresh Quantity with the returned exact built-in `float`
   magnitude in TensorDSLab's private registry and canonical unit.

The owning Config then:

6. stores the returned object with `object.__setattr__` on the frozen config;
   and
7. applies only field-specific rules not already expressed by the selected
   scalar constraint.

The helper does not store a `Scalar` wrapper and does not use
`Scalar.accepts()`. `FiniteFloat`, `NonnegativeFloat`, or `PositiveFloat` is
passed as the constraint according to the ordinary finite/sign requirement.
The normalized built-in float becomes the magnitude of the fresh canonical
Pint quantity.

The final fixed work order must name one public predicate supported by the
selected Pint release that recognizes ordinary registry-created Quantity
instances, including those created by another ordinary `UnitRegistry`.
TensorDSLab does not accept an arbitrary object merely because it has `.to()`
and `.magnitude` attributes. Pint Measurement values, Pint-wrapped arrays, and
other duck-typed objects are outside the first accepted surface. A custom
Quantity subclass that overrides ordinary Pint conversion or magnitude
behavior is unsupported and receives no promised error category; TensorDSLab
does not add hostile-subclass policing around Pint's accepted public root.

External registries may use Pint's standard definitions and may add compatible
units. A registry that redefines the canonical standard units or their
relationships is unsupported; conversion in the source registry cannot prove
that its definitions are scientifically identical to TensorDSLab's registry.

The public error taxonomy is:

- `TypeError` for a non-Quantity physical field, a non-scalar magnitude, or an
  otherwise unsupported Quantity kind;
- `ValueError` for an unknown/invalid unit expression, incompatible physical
  dimension, nonfinite or unrepresentable float magnitude, or a violated
  sign/order/domain rule; and
- the field-specific exception is chained from the original Pint exception
  when Pint rejected parsing or conversion.

Native Pint parser or dimensionality exceptions do not escape as the primary
public error category. The selected dependency probe must demonstrate the
exact recognized Quantity hierarchy and caught Pint exception classes before
this work order is dispatched.

This happens even when the input already belongs to TensorDSLab's registry.
The input quantity and stored quantity must therefore never be the same object.
Two fields built from the same caller object also receive distinct stored
objects.

Frozen dataclasses provide assignment protection, not deep immutability of a
Pint object. Calling mutating Pint methods such as `.ito(...)` on a quantity
retrieved from a config is unsupported. Defensive construction prevents
caller-owned inputs from mutating a config after construction; it does not
promise to police deliberate mutation through the config's own exposed Pint
object.

### Hashing

Every public TensorDSLab Config in the migrated graph is explicitly
unhashable. This includes Configs containing only control fields and composite
Configs such as `ReadoutConfig`, not merely direct physical leaves.

Every Config declares `__hash__ = None` in its class body. Because static
typing models `object.__hash__` as callable, the implementation work order
must authorize one narrow assignment-line suppression for that exact
declaration (for Pyright, `# pyright: ignore[reportAssignmentType]`).
Focused tests require `type(config).__hash__ is None` and `hash(config)` to
raise `TypeError` across every valid composition. No broader file-level
suppression is accepted. Relying on a field's incidental hashability or only
setting `unsafe_hash=False` is insufficient.

Without that rule, frozen dataclass hashing would vary according to whether a
particular composition happened to contain a Pint quantity. Equality remains
ordinary config equality over canonical copied values and control fields;
hashability is not part of the configuration contract.

## Canonical Units

Canonical units preserve the current numerical conventions rather than
forcing every equation into SI base units:

| Domain | Canonical representation |
| --- | --- |
| compact sample-axis coordinates | exact integer picoseconds (`ps`) |
| pulse, delay, jitter, and recovery times | scalar nanoseconds (`ns`) |
| rates and frequency bin edges | scalar hertz (`Hz`) |
| voltage, noise RMS, saturation, and ADC input | scalar millivolts (`mV`) |
| one-sided noise PSD density | scalar `mV**2/Hz` |
| analog gain | existing numeric decibels (`dB` convention), not Pint |

Photoelectron count is dimensionless detector state, not a custom Pint unit.
`peak_voltage_per_photoelectron` therefore carries voltage dimension; the
field name supplies the per-photoelectron meaning.

Canonical configuration quantities store exact finite built-in `float`
magnitudes. `SampleAxis` construction is the exception: it converts compatible
time quantities to exact integral picoseconds because coordinates are compact
integers rather than floating execution parameters.

## Public Configuration Migration

Maintenance 5 removes `SamplingConfig` and the
`ReadoutConfig.sampling` field before this migration. The target configuration
graph therefore has 22 Config classes and 61 fields:

- 26 physical fields represented by Pint quantities; and
- 35 dimensionless, algorithmic, stochastic-address, or composition fields
  retaining their current TensorCore/Python types.

Raw numbers are rejected for every physical field, even when a caller intends
the canonical unit. Dimensionless fields reject Pint quantities. There is no
implicit convention such as "a float here means nanoseconds."

### Charge physical fields

| Config | Current field | Target field | Canonical unit | Local rule |
| --- | --- | --- | --- | --- |
| `TimingJitterConfig` | `sigma_ns` | `sigma` | ns | finite, >= 0 |
| `DarkCountConfig` | `rate_hz` | `rate` | Hz | finite, >= 0 |
| `FixedDelayConfig` | `delay_ns` | `delay` | ns | finite, >= 0 |
| `ExponentialDelayConfig` | `mean_delay_ns` | `mean_delay` | ns | finite, > 0 |
| `AfterpulseConfig` | `mean_delay_ns` | `mean_delay` | ns | finite, > 0 |
| `AfterpulseRecoveryConfig` | `time_constant_ns` | `time_constant` | ns | finite, > 0 |

The following Charge values remain dimensionless/control values and preserve
their current types and names:

- all `RngKey` fields;
- direct and delayed crosstalk `mean_offspring_per_parent`;
- afterpulse `probability`;
- correlated-avalanche `maximum_generations`;
- charge-smearing `relative_sigma`;
- delay/recovery/mechanism composition; and
- the four optional `ChargeConfig` effect fields.

### Pure-waveform physical fields

| Config | Current field | Target field | Canonical unit | Local rule |
| --- | --- | --- | --- | --- |
| `TpcFebSnrPulseConfig` | `fast_time_constant_ns` | `fast_time_constant` | ns | finite, > 0 |
| `TpcFebSnrPulseConfig` | `slow_time_constant_ns` | `slow_time_constant` | ns | finite, > fast |
| `TpcFebSnrPulseConfig` | `support_time_ns` | `support_time` | ns | finite, > 0 |
| `TpcFebSnrPulseConfig` | `peak_voltage_mv_per_pe` | `peak_voltage_per_photoelectron` | mV | finite, nonzero |
| `VetoPduPulseConfig` | `gaussian_center_ns` | `gaussian_center` | ns | signed finite |
| `VetoPduPulseConfig` | `gaussian_width_ns` | `gaussian_width` | ns | finite, > 0 |
| `VetoPduPulseConfig` | `edge_offset_1_ns` | `edge_offset_1` | ns | signed finite |
| `VetoPduPulseConfig` | `edge_width_1_ns` | `edge_width_1` | ns | finite, > 0 |
| `VetoPduPulseConfig` | `edge_offset_2_ns` | `edge_offset_2` | ns | signed finite |
| `VetoPduPulseConfig` | `edge_width_2_ns` | `edge_width_2` | ns | finite, > 0 |
| `VetoPduPulseConfig` | `support_time_ns` | `support_time` | ns | finite, > 0 |
| `VetoPduPulseConfig` | `peak_voltage_mv_per_pe` | `peak_voltage_per_photoelectron` | mV | finite, nonzero |

`PureWaveformConfig.model` remains a control/composition field.

### Noise physical fields

| Config | Current field | Target field | Canonical unit | Local rule |
| --- | --- | --- | --- | --- |
| `WhiteNoiseConfig` | `rms_mv` | `rms` | mV | finite, > 0 |
| `PsdNoiseConfig` | `frequency_left_edges_hz` | `frequency_left_edges` | tuple of Hz | nonempty, first 0, strictly increasing |
| `PsdNoiseConfig` | `frequency_stop_hz` | `frequency_stop` | Hz | finite, > final left edge |
| `PsdNoiseConfig` | `power_density_mv2_per_hz` | `power_density` | tuple of mV**2/Hz | finite, >= 0, same length, some > 0 |

The PSD fields are exact tuples of scalar quantities. TensorDSLab does not
accept one Pint-wrapped vector, a NumPy array, a Torch tensor, or a mutable
sequence at this public boundary. Left edges retain their established meaning;
the stop remains the right edge of the final PSD bin. `ZeroNoiseConfig`, both
noise RNG keys, and `NoiseWaveformConfig.model` remain unitless controls.

### Analog and digitized physical fields

| Config | Current field | Target field | Canonical unit | Local rule |
| --- | --- | --- | --- | --- |
| `AnalogSaturationConfig` | `minimum_mv` | `minimum` | optional mV | finite when present |
| `AnalogSaturationConfig` | `maximum_mv` | `maximum` | optional mV | finite when present |
| `DigitizedWaveformConfig` | `input_min_mv` | `input_minimum` | mV | finite |
| `DigitizedWaveformConfig` | `input_max_mv` | `input_maximum` | mV | finite, > minimum |

`AnalogSaturationConfig` still requires at least one bound and, when both are
present, `minimum < maximum`. `AnalogWaveformConfig.saturation` remains a
composition field. `DigitizedWaveformConfig.bit_depth` remains an integer
control. `analog_gain_db` remains `NonnegativeFloat` and keeps the accepted
amplitude law:

```python
gain = 10.0 ** (analog_gain_db / 20.0)
```

Pint logarithmic-unit handling is deliberately out of scope.

## Compact `SampleAxis` And Pint

TensorCore `0.13.0` `RegularAxis` owns compact integer coordinate mechanics.
TensorDSLab's final `SampleAxis` narrows those mechanics to a positive regular
picosecond grid. Its generic state remains:

```python
SampleAxis(start=0, step=2000, count=4096)
```

where `start`, `step`, and `coordinate_at(index)` are ordinary Python integers
in canonical picoseconds. `coordinates` is the nonmaterializing range-backed
sequence supplied by TensorCore. No per-instance unit string is stored.

TensorDSLab adds a public alternate constructor and physical accessors,
conceptually:

```python
axis = SampleAxis.from_period(
    period=quantity(2, "ns"),
    count=4096,
)

axis.start              # 0
axis.step               # 2000
axis.coordinate_at(3)   # 6000

axis.start_time         # fresh TensorDSLab-registry Quantity: 0 ps
axis.sample_period      # fresh TensorDSLab-registry Quantity: 2000 ps
axis.time_at(3)         # fresh TensorDSLab-registry Quantity: 6000 ps
axis.stop_time          # fresh TensorDSLab-registry Quantity: 8192000 ps
```

The physical accessors return Pint scalar quantities, not formatted strings.
`stop_time` is the exclusive stop. Each access returns a fresh quantity so a
caller's mutation cannot affect the axis or another returned value.

`from_period(...)` applies the same supported-Quantity and original scalar-
magnitude precheck as Config canonicalization, converts to `ps` in the source
registry, and accepts only an exact `int` or a finite `float` for which
`.is_integer()` is true. It rejects `bool`, another magnitude type, a
fractional/nonfinite value, and a nonpositive period, then converts to `int`
and calls the inherited constructor so count and exclusive-stop bounds are
rechecked. It fixes `start` at exact zero. Direct inherited integer
construction remains the public nonzero-start surface; this stage adds no
second quantity-based grid constructor. `count` remains an exact Python
integer subject to the TensorDSLab SampleAxis bounds.

`time_at(index)` first delegates to TensorCore's `coordinate_at(index)` so the
accepted exact-int, `TypeError`, and `IndexError` behavior is preserved. It
then constructs a fresh package-registry quantity from that canonical integer
picosecond coordinate. All other physical accessors likewise create a fresh
Quantity; they do not cache Quantity or Unit objects on the axis.

The semantic axis may generically represent a nonzero regular grid. The
complete TensorDSLab `Photoelectrons`/readout input boundary requires exactly
`start == 0`, because readout time is example-local. It also requires positive
`step`, `count >= 2`, and the accepted representability bounds. That boundary
is checked before RNG consumption or product writes.

Pint does not interact with `ExampleAxis` or `ChannelAxis`:

- `ExampleAxis` contains zero-based, identity-free local integer ordinals;
- `ChannelAxis` contains explicit unique string labels; and
- neither axis stores or returns physical quantities.

The package does not render sample coordinates as labels such as `"2000ps"`.
Indices are used in kernels; the compact integer coordinates and Pint
accessors are semantic/configuration conveniences outside the hot path.

## Preparation Contract

Public Config construction validates information knowable from one config:

- exact supported quantity shape;
- dimensional compatibility;
- conversion into the canonical unit and private registry;
- finite scalar magnitude;
- sign, nonzero, and local ordering rules; and
- exact types of dimensionless/control fields.

Preparation owns contextual facts that require product inputs, sampling,
dtype, device, another config, or an algorithm:

- complete source-axis requirements and example-local sample start;
- sample count/period/window and representability bounds;
- mapping time laws onto the discrete sample grid;
- PSD coverage of the derived Nyquist band;
- pulse-template support and normalization at the selected period;
- dtype-represented scalar constants and overflow envelopes;
- destination shapes/devices and output allocation facts;
- stochastic-role closure and address limits; and
- cross-product or cross-config relationships.

Each physical quantity consulted by an active preparation path is extracted at
most once, and every active required operand is extracted exactly once before
contextual arithmetic. Unrequested products and scientifically inactive
branches need not extract unused values. After extraction, the quantity object
is not passed into a helper, loop, tensor operation, RNG method, Runtime
record, producer, or validator.

The extraction boundary is conceptually:

```python
def prepare_timing_jitter(
    config: TimingJitterConfig,
    *,
    sampling: SamplingRuntime,
) -> TimingJitterRuntime | None:
    sigma_ns = float(config.sigma.magnitude)  # one extraction
    if sigma_ns == 0.0:
        return None
    # All remaining work uses sigma_ns and plain sampling facts.
    ...
```

The precise implementation may use a narrow private units helper to enforce
the already-established canonical scalar representation, but it must not call
`.to(...)` repeatedly or introduce a generic preparation framework.

### Sampling preparation

After Maintenance 5, sampling preparation is source-derived:

```python
def prepare_sampling(photoelectrons: Photoelectrons) -> SamplingRuntime:
    ...
```

It reads the exact `SampleAxis` object, validates the complete readout
boundary, and records the three plain execution facts retained by Maintenance
5:

```text
sample_dimension: int
sample_period_ps: int
sample_count: int
```

It does not use the Pint accessors: the axis already stores canonical integer
picoseconds. This avoids creating unit objects merely to recover the canonical
integers needed by every downstream operation.

### Charge preparation

Charge's effect-owned preparers perform the one-time extraction nearest the
scientific law they prepare:

- dark count extracts `rate_hz` once before computing the per-bin Poisson
  mean from integer `sample_period_ps`;
- timing jitter extracts `sigma_ns` once before its latent-uniform plus
  Gaussian redistribution law;
- fixed and exponential delay preparation extracts `delay_ns` or
  `mean_delay_ns` once before binary64 phase-marginalized bin probabilities;
- afterpulse preparation extracts `mean_delay_ns` and optional recovery
  `time_constant_ns` once before retained/overflow and charge-weight kernels;
  and
- smearing remains dimensionless and introduces no Pint operation.

Parent `prepare_charge(...)` coordinates the active effect closure but does
not read physical Quantity magnitudes itself. Each effect preparer owns its
exact-zero/no-op decision. An active afterpulse mean is extracted once and
passed as a plain scalar to both delay and recovery preparation rather than
being reread from the Config.

The conversion must preserve the accepted `as_integer_ratio`, ns-to-ps,
tail-evaluation, tolerance, category-order, and RNG-address operation order
when canonical operands match the closed implementation.

### Pure-waveform preparation

Pure-waveform preparation extracts all pulse-model quantities once before
template sampling. In particular, `_tpc_raw` and `_veto_raw` or their later
module-local equivalents receive plain numeric parameters; they must not
receive Configs or read `.magnitude` during every sample evaluation.

The preparer derives `sample_period_ns` once from integer
`SamplingRuntime.sample_period_ps`, then preserves the accepted TPC/Veto
equations, support count, normalization, dtype representation, and coefficient
ordering.

### Noise preparation

White-noise preparation extracts one `rms_mv`. PSD preparation extracts the
frequency-left-edge, frequency-stop, and power-density magnitudes into plain
tuples exactly once. All overlap integration, Nyquist coverage, DC removal,
coefficient scaling, and dtype preparation then use those tuples.

No quantity may appear in an FFT call, tensor coefficient vector, or noise
Runtime. The public PSD remains a physical density; preparation remains the
owner of mapping it onto the requested finite sample grid.

### Analog and digitized preparation

Analog preparation extracts target fields `minimum` and `maximum`, when
present, exactly once into unit-suffixed `minimum_mv` and `maximum_mv` locals
and prepares dtype-represented scalar bounds. Digitized preparation extracts
target fields `input_minimum` and `input_maximum` once into unit-suffixed
locals, combines them with the numeric decibel gain and bit depth, and records
plain values such as:

```text
maximum_code
slope_per_mv
intercept
lower_input_mv
upper_input_mv
```

The accepted analog clamp and ADC transfer equations, including clamp before
integer conversion, remain unchanged.

### Whole-request preflight

`simulate_readout(...)` retains its public shape apart from the prior removal
of `ReadoutConfig.sampling`. It receives already-canonical exact Configs and
prepares only the requested transitive closure. Whole-request preparation must
complete active magnitude extraction, sampling preparation, product
preparation, destination checks, and RNG-role checks before the first RNG word
request or semantic product write.

A failure in Pint dimension handling occurs during Config construction. A
failure involving the actual source grid or requested algorithm occurs during
whole-request preparation. Neither failure may partially execute the readout.

## Runtime, Production, And Validation Contract

Every `*Runtime` and `SamplingRuntime` remains a concrete final frozen slotted
dataclass. Recursively, a Runtime contains no:

- public Config;
- Pint Quantity, Unit, or UnitRegistry;
- unit string;
- package registry reference;
- semantic product or collection;
- mutable cache; or
- execution method.

Selected physical execution facts use unit-suffixed names whenever an
otherwise generic scalar or magnitude name would be ambiguous:

```text
sigma_ns
rate_hz
delay_ns
sample_period_ps
represented_rms_mv
represented_powers_mv2
minimum_mv
maximum_mv
lower_input_mv
upper_input_mv
slope_per_mv
```

This does not require every product-specific tensor role to encode its full
physical dimension in its attribute name. Established semantic names such as
`kernel` may remain when the Runtime type and preparation contract already fix
their interpretation. The fixed work order must list every actual Runtime
rename rather than infer a package-wide mechanical rewrite.

Producers accept typed prerequisites, one prepared Runtime, and the generic RNG
where needed. They do tensor math, request random values, and construct exactly
one fresh semantic product. They do not import Pint, Config classes, the units
module, or any unit helper.

Validators accept completed fields, typed prerequisites, and only the exact
prepared facts needed to validate their result. They validate dtype, device,
axes, shape, storage, value domain, and product relationships. They do not
interpret units or reconvert physical values.

Because Runtime classes may be defined beside preparers, importing a producer
can transitively load a module that itself imports a Quantity type for
annotations. The meaningful contract is therefore behavioral and structural:
no Quantity reaches production or validation and no unit operation occurs
there. The stage must not make the brittle claim that `pint` is absent from
the process's global `sys.modules` after a normal package import.

## Numerical And Stochastic Continuity

This migration changes representation at the public boundary, not the
accepted scientific model.

For each old physical scalar `x` expressed in its historical canonical unit,
the migrated configuration:

```python
quantity(x, canonical_unit)
```

must prepare the same binary64 operand and preserve the same subsequent
operation order wherever Pint conversion returns that same scalar. Under that
condition, deterministic products, stochastic addresses, word requests,
category order, exact-zero draw-free branches, and same-stack completed values
remain unchanged.

Tests must include physically equivalent exact-scale units, for example:

```text
2 ns == 2000 ps
1 MHz == 1_000_000 Hz
1 V == 1000 mV
```

when those values are exactly representable at the accepted boundary.

TensorDSLab does not promise bitwise identity for every arbitrary decimal unit
conversion. If two inputs canonicalize to different binary64 magnitudes, they
are different numerical inputs even when their displayed decimal values seem
close. The selected Pint release, canonicalization path, and accepted
comparison boundaries must be documented. No tolerance, rounding, clipping,
or quantization may be added merely to force equality.

The migration changes no RNG key, logical position, ordinal, distribution,
draw schedule, or no-draw policy.

## Exact Dependency Selection Gate

This draft intentionally does not name a Pint version. Before dispatch,
TensorDSLab Design must evaluate and freeze one exact release and artifact.
The evidence must establish at least:

- Python 3.11 support and the package's then-supported upper Python endpoint;
- import and scalar conversion without requiring NumPy in the frozen Della
  runtime;
- scalar external-registry `.to(...)` behavior;
- equality and copying behavior needed by frozen configs;
- type-checking behavior under the selected Pyright version;
- source versus wheel/archive identity used by package tests;
- no application-registry mutation;
- unit-parser behavior for every canonical public spelling; and
- license and direct/transitive dependency acceptability.

The exact Pint release is then added to `pyproject.toml` while retaining the
exact TensorCore `0.13.0` commit selected by closed Maintenance 5.
TensorDSLab must not rely
on an unbounded Pint range, an ambient application registry, or whichever
version happens to be installed. TensorCore must not acquire Pint because of
this work.

Primary upstream material for the dependency evaluation includes:

- [Using Pint in projects](https://pint.readthedocs.io/en/develop/getting/pint-in-your-projects.html);
- [Pint performance guidance](https://pint.readthedocs.io/en/develop/advanced/performance.html);
- [Pint serialization guidance](https://pint.readthedocs.io/en/stable/advanced/serialization.html);
- [Pint and NumPy](https://pint.readthedocs.io/en/stable/user/numpy.html); and
- [Pint logarithmic units](https://pint.readthedocs.io/en/stable/user/log_units.html).

These links are evidence to inspect, not a delegation of TensorDSLab's
contracts to Pint documentation. Before dispatch, moving `develop`/`stable`
links must be supplemented or replaced with selected-version documentation and
source anchors for every behavior on which the fixed work order relies.

## Expected Production Scope

The final work order is expected to authorize only the package-local files
needed for:

- the exact Pint dependency pin and packaging metadata;
- `tensor_dslab/common/units.py` and deliberate facade exports;
- TensorDSLab `SampleAxis` Pint constructor/accessors after closed
  Maintenance 5 compact-axis adoption;
- the 22 public Config classes and their exports;
- product and effect preparation modules;
- Runtime field renames necessary to make the selected unit-bearing execution
  operands explicit;
- narrow producer/validator adjustments required solely by Runtime field
  renames;
- public and private typing probes;
- focused unit/config/axis/preparation/continuity tests; and
- synchronized living architecture, API, validation, and implementation
  documentation.

The fixed work order must provide an exact path allowlist and protected-byte
set from its future authority baseline. This Design draft does not authorize
any edit to production or tests.

## Required Focused Evidence

The dispatched stage must include committed evidence for all of the following.

### Registry and helper behavior

- package-registry quantities from `quantity` and `quantities`;
- exact accepted and rejected magnitude/container types;
- helper rejection of empty, malformed, and undefined unit expressions;
- external-registry scalar acceptance;
- config rejection of dimensionally incompatible quantities;
- rejection of Pint-wrapped arrays and non-Pint duck objects;
- rejection of Pint Measurement values when the selected dependency exposes
  them without installing an otherwise-unneeded optional dependency solely
  for this negative test;
- fresh canonical copies for package- and external-registry inputs;
- no application-registry mutation; and
- absence of exported registry/Q_/units/Quantity-wrapper surfaces.

### Complete config matrix

- every one of the 26 physical fields accepts its canonical dimension;
- every required physical field, and every present optional saturation bound,
  rejects a raw number, wrong dimension, non-scalar quantity, nonfinite
  magnitude, and its exact sign/order violations;
- all config-local PSD tuple shape, matching-length, start-at-zero, monotonic-
  edge, stop-after-final-edge, and nonzero-power rules;
- preparation alone proves that the configured PSD covers the source-derived
  Nyquist band;
- dimensionless fields reject quantities and retain exact current types;
- dB remains numeric and uses the accepted amplitude conversion; and
- all 22 target Config classes are explicitly unhashable for every valid
  composition.

### Axis and sampling behavior

- compact, range-backed integer `SampleAxis` state;
- exact integral-picosecond `from_period(...)` conversion and rejection of a
  positive fractional-picosecond period;
- fresh `start_time`, `sample_period`, `time_at`, and exclusive `stop_time`
  quantities in canonical picoseconds;
- generic nonzero-start acceptance where the semantic axis permits it;
- complete readout rejection of nonzero start before RNG or writes; and
- `prepare_sampling` reads integer axis state directly and creates no Pint
  object.

### Preparation ownership

- each physical quantity consulted by an active path is extracted at most
  once, every active required operand exactly once, and an inactive/unrequested
  branch need not extract its unused quantities;
- pulse raw-equation helpers receive plain values rather than Configs;
- PSD integration receives plain tuples rather than quantities;
- Runtime records recursively contain no Pint or Config object;
- producer and validator source/import/call probes reject unit conversion or
  Config access;
- all invalid contextual combinations fail before RNG consumption or writes;
  and
- prepared plain operands match the closed implementation for exact canonical
  fixtures.

### Result and continuity behavior

- deterministic pure, analog, and digitized reference equations;
- white/PSD noise and Charge same-stack replay for matching canonical
  operands;
- exact-zero branches remain draw-free;
- exact equivalent-unit fixtures produce equal canonical configs, Runtime
  operands, RNG calls, and products;
- an incompatible physical dimension prevents Config construction before a
  simulation call can advance RNG or expose a result; tests must not bypass a
  Config constructor to manufacture this case;
- contextual sampling/algorithm preflight failures advance no RNG and expose
  no partial result;
- source immutability, generated-product freshness, storage independence,
  dtype/device/axes, and autograd contracts remain unchanged; and
- no new host materialization or device movement occurs in simulation.

### Static and packaging behavior

- positive and negative Pyright probes for every public migrated signature;
- exact facade identities and `__all__` contents;
- import-direction and private-runtime export scans;
- exact selected Pint and retained TensorCore `0.13.0` pins in source and
  built/archive forms;
- clean package-root import from an isolated artifact; and
- no retired unit-suffixed public config field or compatibility shim.

## Full Validation And Independent Review

The final stage must use the accepted finite Implementation/Validation loop
and one immutable fixed-commit independent Review. At minimum:

1. Implementation runs the focused matrix, full test discovery, Pyright,
   source/archive or wheel isolation, diff, import, privacy, and artifact
   gates on the exact candidate.
2. Validation independently reconstructs the exact TensorCore and Pint
   dependencies, repeats all local gates, runs mutation-style probes against
   the unit/preparation boundary, and executes a fresh full-A100 source and
   artifact matrix under the accepted runtime.
3. Review independently audits API clarity, registry isolation, every config
   migration, one-time preparation extraction, Runtime purity, numerical/RNG
   continuity, dependency identity, typing, and documentation. Review uses a
   separate fresh full-A100 allocation.
4. Only an unchanged cleared commit may be fast-forwarded. Design then performs
   a documentation/evidence closeout bounded by the fixed work order.

GPU evidence in this stage proves correctness and device preservation, not
fusion, allocation freedom, or performance. Stage 8 characterization remains
a separate future authority.

## Documentation And History

The eventual Design authority should synchronize, at minimum:

- `AGENTS.md` current dependency/API/axis/runtime state;
- `CONTRIBUTING.md` unit/config/preparation standards;
- `README.md` and public quickstart examples;
- `docs/overview.md`, `docs/design.md`, and `docs/decisions.md`;
- `docs/architecture/rebuild.md`, `readout.md`, and `tensors.md`;
- `docs/validation.md`;
- `docs/implementation/index.md`; and
- this work order.

`docs/parity.md` changes only if the final numerical comparison boundary or
donor interpretation changes. Closed work orders and historical blobs must not
be rewritten to pretend they used Pint or compact axes.

This draft remains nonoperative. The Maintenance 5 Design authority may update
its prerequisite/status wording and its implementation-index entry while
synchronizing the compact-axis migration, but that does not dispatch Pint
implementation or change the living physical-configuration contract.

## Non-Goals

This candidate does not authorize:

- TensorCore changes or a Pint dependency in TensorCore;
- execution before TensorDSLab Maintenance 5 is Merged / Closed;
- IO, `TensorArtifact`, config serialization, persistence, or schema versioning;
- a public UnitRegistry, application registry, unit-definition plugin, or
  custom detector units;
- Pint-wrapped Torch/NumPy arrays or quantities in fields/collections;
- Pint Quantity/Unit/UnitRegistry objects or stored unit strings in Runtime
  records, producers, validators, or kernels;
- a unit string stored on `SampleAxis`;
- parsing string timestamps such as `"2000ps"`;
- Pint logarithmic units for analog gain;
- normal/lognormal delay restoration or any scientific-law change;
- TensorG4DS, TensorML, sibling-package, or integration edits;
- public compatibility aliases for retired config field names;
- Stage 8 benchmarking, compiler fusion, workspaces, `out=`, or allocation
  claims;
- release, deployment, conformance, backward-compatibility, or broad
  cross-package compatibility claims; or
- a push without separate authorization.

## Stop Conditions

Stop the affected work and return exact evidence to TensorDSLab Design if:

- the closed Maintenance 5 compact-axis/sampling contract or exact TensorCore
  `0.13.0` pin differs from the accepted prerequisite assumed here;
- no exact Pint release satisfies the selected Python, Della, typing, and
  scalar-registry requirements;
- an external-registry scalar cannot be safely copied without relying on
  mutable global registry state;
- the migration changes a scientific equation, stochastic address/draw order,
  or accepted canonical operand unexpectedly;
- a Runtime, producer, or validator requires a Pint object to meet the design;
- unit conversion cannot complete before RNG consumption and writes;
- `SamplingConfig` or string-valued SampleAxis coordinates remain live at the
  proposed baseline;
- source/archive or CPU/CUDA behavior conflicts at the accepted comparison
  boundary;
- an edit is required outside the future exact allowlist;
- the repository, dependency, branch, or execution route is dirty, stale, or
  discrepant; or
- implementation would require IO, persistence, integration, performance, or
  another non-goal to proceed.

## Completion Boundary

This candidate becomes eligible for final acceptance only when one exact
future commit:

- pins one accepted Pint release while retaining the exact TensorCore `0.13.0`
  dependency adopted by Maintenance 5;
- implements the complete public physical-config migration without shims;
- provides compact integer SampleAxis state plus the selected Pint
  conveniences;
- proves Config canonicalization and one-time preparation extraction;
- keeps Runtime, production, validation, and tensor execution unit-free;
- preserves accepted scientific, numerical, RNG, storage, device, and
  autograd behavior at the declared boundaries;
- passes the complete local and fresh-A100 Validation/Review matrices; and
- is independently Review-cleared and cleanly fast-forwarded under a later
  explicit user-authorized work order.

Until those gates are fixed and dispatched, this document is architecture and
planning evidence only.
