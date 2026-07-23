# Maintenance 6 Pint Physical Configuration Boundary Work Order

Status: **Merged / Closed** through exact Review-cleared target
`0257fb477ee04556ebbe26351123ae610b5d7925`, tree
`b4f5703ca5b756dc27d876c1dd17ee56cb43b4e8`.

Stable work-order key:
`TensorDSLab/maintenance-6-pint-physical-configuration-boundary`.

## Final Evidence

Implementation completed Candidate 1 as one clean direct child of exact
corrected Design authority
`6f7eba9b1c4e680930007836433f15e517288a9a`:

```text
commit:  240e1492c466097b3059dfe9911ab338a4dd38a1
tree:    1e5cae8c0e905c9638eb40e7f9d24fac950fee59
```

Candidate 1 remains immutable. The later Design-owned evidence-scheduling and
closeout-scope amendments changed only documentation on top of that candidate;
they changed no production, test, dependency, scientific, RNG, or public API
byte. Validation cleared exact final target `0257fb477ee04556ebbe26351123ae610b5d7925`
unchanged, and Review fast-forwarded that exact commit to local `main`.

The local evidence environment was Python `3.13.11`, PyTorch `2.12.1`, eager
CPU, macOS `15.7.4` arm64, with CUDA unavailable. Implementation independently
reconstructed exact TensorCore `0.13.0` source at commit
`202d8b1bc6259b8453d3d377570417f2480d782b`, parent
`f62506b6d2f6926db446e2d163f26870575c9419`, and tree
`48fa9a28db6d043abc07d9963b2015983ca436ea`. Its fresh canonical no-prefix ZIP
was `373491` bytes with SHA-256
`ed804c71a617a79a63b53be86157e2045322d6c6868ca3766dc75d5526cb8b09`,
and source/archive package bytes compared equal.

The exact Pint `0.25.3` wheel was `307488` bytes with SHA-256
`27eb25143bd5de9fcc4d5a4b484f16faf6b4615aa93ece6b3373a8c1a3c1b97d`;
the exact sdist was `255106` bytes with SHA-256
`f8f5df6cf65314d74da1ade1bf96f8e3e4d0c41b51577ac53c49e7d44ca5acee`.
Role-owned environments imported the published wheel and an independently
sdist-built installation.

Across all four TensorCore source/archive by Pint wheel/sdist combinations,
the focused Maintenance 6 module passed `10/10`, full discovery ran `207`
tests with `194` passed and `13` expected unavailable-CUDA skips, and Pyright
`1.1.411` reported zero errors, warnings, or informations. The external
negative typing probe produced the same `38` required diagnostics in the
representative source/wheel and archive/sdist forms. Focused source and
process-local mutation probes rejected restoration of the three retired
annotation-only helper definitions or Config calls while retaining every
Pint, primitive-domain, PSD, ordering, and distinct-key relationship.

Dependency, facade/export, import-isolation, public-TensorCore import,
Runtime/producer/validator Pint-privacy, retired-surface, exact-scope,
protected-byte, diff, artifact, bytecode, and repository-cleanliness gates are
required to remain clear on the frozen candidate. This local evidence makes no
GPU, performance, release, deployment, conformance, or broad compatibility
claim.

## Post-Candidate Evidence-Scheduling Amendment

After Candidate 1 was frozen, the user explicitly directed TensorCore and
TensorDSLab to defer fresh cluster CUDA gates until their new dependency
surfaces are stable together. Maintenance 6 therefore closes through complete
local fixed-commit Validation and independent Review, with its `13`
conditional CUDA skips retained as an explicit qualification. Validation and
Review must not submit a Della or other cluster allocation for this
maintenance, and Maintenance 6 makes no new accelerator-correctness claim.

This amendment does not adopt TensorCore `0.15.0`. Maintenance 6 remains
exactly pinned to published TensorCore `0.13.0` at
`202d8b1bc6259b8453d3d377570417f2480d782b`. After Maintenance 6 closes, a
separate TensorDSLab work order may adopt published TensorCore `0.15.0` at
`0f974e9e7f52125bbe829e124beb24e69de811d3`. Only after that exact adoption
also closes may separately authorized, package-owned two-PyTorch-minor CUDA
matrices qualify the integrated TensorCore/TensorDSLab pairing. Those future
matrices are not a retroactive Maintenance 6 closure condition and do not
replace the separately authorized Stage 8 performance work.

This is the package-authoritative TensorDSLab Design work order for the next
breaking, pre-deployment configuration migration. It binds the already-closed
TensorCore `0.13.0` compact-axis and generic-`Scalar` baseline to exact Pint
`0.25.3`, moves every physical public Config value to a canonical copied Pint
quantity, and tightens the private preparation boundary without changing
readout science.

The user selected Pint as the next package direction, authorized the Design
work, and explicitly dispatched its production implementation. Execution is
limited to TensorDSLab's persistent Implementation, Validation, and Review
routes and the exact scope below.

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

Maintenance 6 must:

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

## Authority, Exact Baselines, And Dispatch Gate

Package authority is `TensorDSLab/default/Design`.

The exact clean package baseline is the Maintenance 5 Design closeout:

```text
repository:       TensorDSLab
reference:        main
commit:           021694b9479d02546405f6a815aedf21c9c831a4
tree:             698b34cdcc380c81d444d9e1e6c6108867bd05fa
package version:  0.1.0
Python:           >=3.11
TensorCore pin:   0.13.0 at 202d8b1bc6259b8453d3d377570417f2480d782b
```

Maintenance 5 is Merged / Closed through exact Review-cleared supplemental
candidate `81ad2f52fe4a1966e5b3a0ceb5063138e42e731f`, tree
`1c9ce87237544c32dee4b4f594e97ab929234475`. Its production, tests, and
metadata are contained unchanged in the baseline above. It already:

- adopts the exact TensorCore dependency below;
- defines `ExampleAxis`, `ChannelAxis`, and `SampleAxis` through the compact
  count, label, and regular representation roots;
- removes `SamplingConfig` and `ReadoutConfig.sampling`; and
- derives one private `SamplingRuntime` from the source `SampleAxis`.

The exact published TensorCore dependency is:

```text
repository:       TensorCore
repository URL:   https://github.com/mbedard44/TensorCore.git
reference:        origin/main
commit:           202d8b1bc6259b8453d3d377570417f2480d782b
direct parent:    f62506b6d2f6926db446e2d163f26870575c9419
tree:             48fa9a28db6d043abc07d9963b2015983ca436ea
package version:  0.13.0
Python:           >=3.11
Torch:            >=2.11,<2.13
root exports:     30
installed files:  14
```

The exact selected physical-units dependency is:

```text
distribution:     Pint
PyPI release:     0.25.3
release date:     2026-03-19
Python:           >=3.11
license:          BSD
wheel:            pint-0.25.3-py3-none-any.whl
wheel bytes:      307488
wheel SHA-256:    27eb25143bd5de9fcc4d5a4b484f16faf6b4615aa93ece6b3373a8c1a3c1b97d
source archive:   pint-0.25.3.tar.gz
source bytes:     255106
source SHA-256:   f8f5df6cf65314d74da1ade1bf96f8e3e4d0c41b51577ac53c49e7d44ca5acee
```

The dependency line added by this maintenance is exactly:

```toml
"pint==0.25.3",
```

No Pint extra is selected. The exact wheel metadata names the direct
dependencies `flexcache>=0.3`, `flexparser>=0.4`,
`platformdirs>=2.1.0`, and `typing-extensions>=4.0.0`; NumPy is optional and
is not required by the accepted scalar boundary. Role-owned environment
inventories bind the exact resolved transitive versions used for evidence.

The current package states are:

```text
package_adoption_state:    Adopted
conformance_finding:       Not evaluated
coordination_status:       Deferred
registry_storage_profile:  Disabled
maintenance_4:             Merged / Closed
maintenance_5:             Merged / Closed
maintenance_6:             Merged / Closed
stage_8:                   deferred until after Maintenance 6
```

Dispatch is prohibited until all of the following are true:

1. clean `main` remains exactly the baseline above, or Design explicitly
   reconciles a later nonoverlapping baseline;
2. the exact TensorCore `0.13.0` and Pint `0.25.3` source/artifact identities
   remain reconstructible;
3. the living architecture documents and this work order are committed
   together as one clean Design authority;
4. the persistent TensorDSLab Implementation, Validation, and Review routes
   are Active, current, return-capable, and explicitly bound to this work.

The implementation branch is
`codex/maintenance-6-pint-physical-configuration-boundary`. The candidate
must be a clean direct child chain of the exact Design authority named in the
dispatch. Maintenance 5 remains immutable closed evidence; this work consumes
its public state rather than reopening it.

## Applicable Contracts And Source Precedence

The dispatched roles must reconcile:

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
- the closed Maintenance 4 work order and implementation as the exact
  `prepare -> produce -> validate` ownership evidence;
- the closed Maintenance 5 work order and implementation as the exact compact-
  axis, source-derived-sampling, TensorCore `0.13.0`, and golden-path
  structural baseline;
- TensorCore's exact published `0.13.0` compact-axis and generic-Scalar API
  and integration documentation; and
- Pint `0.25.3`'s exact PyPI artifacts, tagged source, and versioned public
  documentation.

This work order controls only the Maintenance 6 Pint migration slice. Closed work
orders remain immutable historical evidence. Living architecture controls
current package meaning. If the published TensorCore contract or exact Pint
behavior conflicts with this work order, Design must revise and refreeze
the work order rather than asking Implementation to improvise.

`docs/parity.md` is synchronized because Pint conversion adds one explicit
physical-equivalence-versus-binary-equality boundary. That documentation
change does not alter a donor equation, comparison oracle, or accepted
scientific approximation.

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

TensorDSLab owns exactly one private `pint.UnitRegistry`, created with
`cache_folder=None` in:

```text
tensor_dslab/common/units.py
```

The package deliberately exports only construction helpers. Their exact public
typing uses Pint's unparameterized public `Quantity` class:

```python
from pint import Quantity


def quantity(magnitude: int | float, unit: str) -> Quantity:
    ...


def quantities(
    magnitudes: tuple[int | float, ...],
    unit: str,
) -> tuple[Quantity, ...]:
    ...
```

- `quantity` accepts only exact built-in `int` or `float` magnitudes;
- `bool`, `complex`, `Decimal`, `Fraction`, arrays, tensors, lists, tuples, and
  arbitrary numeric duck types are rejected as scalar magnitudes;
- each accepted magnitude is normalized through
  `FiniteFloat.require(magnitude, field)` to an exact built-in `float`;
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

The helpers raise `TypeError` for wrong argument types and `ValueError` for a
nonfinite or unrepresentable magnitude or an empty, malformed, scaled, or
undefined unit expression. An oversized exact integer may make
`FiniteFloat.require(...)` raise `OverflowError`; the helper translates that
exact failure to a chained `ValueError`, matching the public finite-scalar
boundary. `quantities(...)` fails the complete call if any element is
unrepresentable. Pint treats an empty unit string as dimensionless, so TensorDSLab
rejects `unit.strip() == ""` before calling Pint. Unit strings are unit
expressions, not quantity expressions: `"ns"` is accepted and `"2 ns"` is
rejected. The helper validates the unit expression even when `quantities`
receives an empty magnitude tuple.

Pint `0.25.3` exposes several parser failure families for malformed exact
strings. The parser-only translation boundary catches exactly:

```python
(
    pint.PintError,
    ValueError,
    TypeError,
    ArithmeticError,
    AssertionError,
    tokenize.TokenError,
)
```

and raises chained `ValueError`. This deliberately narrow tuple is used only
while parsing an exact built-in string; it is not a general catch around Pint
or package execution.

The package root and `tensor_dslab.common` deliberately re-export
`quantity` and `quantities`. They do not export the registry, a `Q_` alias, a
mutable units object, an application registry, or a package-defined Quantity
wrapper class. TensorDSLab never calls `pint.set_application_registry(...)`.
The registry remains discoverable through a returned Pint object's own
implementation-facing attributes because Pint quantities are registry-bound.
Direct discovery or mutation of it through Pint internals is unsupported and
is not a TensorDSLab API.

The common units module owns these unexported functions, conceptually:

```python
def _canonical_quantity(
    value: object,
    *,
    unit: str,
    field: str,
    constraint: type[Scalar[float]],
) -> Quantity:
    ...


def canonical_magnitude(value: Quantity) -> float:
    ...


def _integer_quantity(magnitude: int, *, unit: str) -> Quantity:
    ...
```

owns the exact recognition, conversion, copying, normalized-scalar constraint,
and exception translation shared by all physical Config fields. `Scalar` and
the accepted leaf constraints are imported from TensorCore's package root.
Config classes add only genuinely field-specific nonzero, ordering, tuple, and
cross-field rules. TensorDSLab must not implement 26 subtly different copies
of the registry boundary.

`canonical_magnitude(...)` is package-private by facade omission. It performs
no conversion and introduces no second validation policy; it centralizes the
one narrow static cast needed because Pyright `1.1.411` does not preserve the
scalar magnitude type through Pint's public generic annotations. Preparation
calls it exactly once per active physical operand. Narrow casts are accepted
only inside `common/units.py`; they must not spread through Config,
preparation, production, or validation modules.

`_integer_quantity(...)` is a separate unexported construction primitive only
for exact integer semantic-axis coordinates. It requires an exact built-in
`int` (never `bool`) and constructs a fresh private-registry Quantity without
normalizing through `FiniteFloat`; therefore every legal signed-int64
picosecond coordinate remains exact even above `2**53`. It is not used by
Configs or public `quantity(...)`/`quantities(...)`, and no public caller
selects its unit.

TensorCore remains Pint-free. Count, regular, and label-axis representation is
generic TensorCore behavior; physical time interpretation and unit conversion
are TensorDSLab behavior.

### External registries and defensive copies

Public physical config fields accept compatible scalar Pint quantities from a
caller's registry. Construction must not compare registry-owned Unit objects
or attach an external-registry object directly. For each physical field, the
common canonical-copy function performs these steps and returns the fresh
canonical quantity:

1. requires `isinstance(value, pint.Quantity)` and an exact built-in `int` or
   `float` source magnitude before invoking any conversion;
2. calls `.to(canonical_unit)` on the input quantity in its own registry;
3. requires the converted magnitude to remain an exact built-in `int` or
   `float`;
4. calls exactly once
   `normalized = constraint.require(converted_magnitude, field)` and
   translates an `OverflowError` into the maintenance's public `ValueError`
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

The exact `isinstance(value, pint.Quantity)` predicate recognizes ordinary
registry-created Quantity instances, including those created by another
ordinary `UnitRegistry`.
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

Around only `.to(canonical_unit)`, `_canonical_quantity` catches
`(pint.PintError, OverflowError)` and raises a chained `ValueError`.
`DimensionalityError`, `UndefinedUnitError`, and ordinary offset/logarithmic
conversion failures are Pint errors. Native Pint parser or dimensionality
exceptions do not escape as the primary public error category.

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

Closed Maintenance 5 removed `SamplingConfig` and
`ReadoutConfig.sampling`. The exact starting configuration graph has 22 Config
classes and 61 dataclass fields. Maintenance 6 migrates:

- 26 physical fields represented by Pint quantities; and
- 35 dimensionless, algorithmic, stochastic-address, or composition fields
  retaining their current TensorCore/Python types.

Raw numbers are rejected for every physical field, even when a caller intends
the canonical unit. Dimensionless fields reject Pint quantities. There is no
implicit convention such as "a float here means nanoseconds."

Every physical field uses the same small constructor shape. For example:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExponentialDelayConfig:
    mean_delay: Quantity

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean_delay",
            _canonical_quantity(
                self.mean_delay,
                unit="ns",
                field="ExponentialDelayConfig.mean_delay",
                constraint=PositiveFloat,
            ),
        )
```

There is no Config ABC, field registry, descriptor layer, or reflection-driven
normalizer. Product-specific Config classes remain ordinary explicit
dataclasses; the one common function owns only the genuinely identical Pint
boundary.

Config `__post_init__` methods remain only where construction performs real
behavior:

- canonicalizing and defensively copying a physical Pint quantity;
- validating an unwrapped primitive value domain that the field itself does
  not own; or
- enforcing a genuine local relationship such as ordering, matching lengths,
  nonempty scientific data, or distinct stochastic keys.

They do not recheck annotated `Scalar` wrappers, `RngKey` values, nested
Config records, optional Configs, or closed Config unions merely to prove
membership. Those relationships are owned by static typing and Review on the
supported public path. Deliberately malformed typed composition has no
promised construction-time failure or exception category.

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
Their magnitudes are the exact built-in integer picosecond values stored or
derived by the axis, including legal values above `2**53`; they do not route
through the public float-normalizing `quantity(...)` helper.

`from_period(...)` applies the same supported-Quantity and original scalar-
magnitude precheck as Config canonicalization and converts to `ps` in the
source registry. Pint `0.25.3` does not always return an exact integer for an
exact scale conversion; for example, `2 ns` becomes
`2000.0000000000002 ps`. Sample-axis integerization therefore has one explicit
exception to the general no-rounding rule:

1. an exact converted built-in `int` is accepted directly;
2. a converted built-in `float` must be finite and no larger than `2**53` in
   magnitude;
3. `nearest = round(converted_ps)` is accepted only when
   `abs(converted_ps - nearest) <= math.ulp(converted_ps)`; and
4. every other fractional value is rejected.

This is an axis-representability rule, not a general physical-quantity
tolerance. It accepts ordinary integral periods such as `2 ns` and `0.5 ns`
while rejecting a genuine fractional-picosecond grid. After integerization,
the inherited constructor enforces positive step, exact count, count minimum,
and exclusive-stop bounds. `from_period(...)` fixes `start` at exact zero.
Direct inherited integer construction remains the public nonzero-start
surface; this stage adds no second quantity-based grid constructor.

`time_at(index)` first delegates to TensorCore's `coordinate_at(index)` so the
accepted exact-int, `TypeError`, and `IndexError` behavior is preserved. It
then constructs a fresh package-registry quantity from that canonical integer
picosecond coordinate through the unexported exact-integer constructor. All
other physical accessors likewise create a fresh exact-integer Quantity; they
do not cache Quantity or Unit objects on the axis. Converting any returned
accessor value back to picoseconds retains its exact integer magnitude, and an
accepted `sample_period` can round-trip through `from_period(...)` without
precision loss.

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

The redundant integer aliases `start_ps`, `sample_period_ps`, and `stop_ps`
are retired without shims. Callers needing compact state use inherited
`start`, `step`, and `start + step * count`; callers needing physical values
use the four Pint accessors above. Private `prepare_sampling(...)` continues
to read `start`, `step`, and `count` directly.

## Preparation Contract

Public Config construction validates information knowable from one config:

- exact supported quantity shape;
- dimensional compatibility;
- conversion into the canonical unit and private registry;
- finite scalar magnitude;
- sign, nonzero, and local relationship rules; and
- primitive value domains not already owned by a typed wrapper.

Annotated wrapper, key, nested-Config, optional, and union membership is a
static typing and Review obligation rather than duplicated runtime admission.

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
    sigma_ns = canonical_magnitude(config.sigma)  # one extraction
    if sigma_ns == 0.0:
        return None
    # All remaining work uses sigma_ns and plain sampling facts.
    ...
```

`canonical_magnitude(...)` owns the narrow static cast for the already-
established canonical scalar representation. Preparation must not call
`.to(...)` again or introduce a generic preparation framework.

### Sampling preparation

Sampling preparation remains source-derived:

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
exact-zero/no-op decision. In particular,
`prepare_timing_jitter(...) -> TimingJitterRuntime | None` owns both sigma
extraction and the exact-zero identity decision; the parent does not inspect
the Quantity first. An active afterpulse mean is extracted once and passed as
a plain scalar to both delay and recovery preparation rather than being reread
from the Config.

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

## TensorCore `0.13.0` And Runtime-Action Cleanup

Maintenance 6 uses TensorCore where its generic contract is stronger and
simpler:

- `Scalar.require(...)` is the single generic scalar normalization engine at
  the Pint boundary;
- compact-axis `axis(...)`, `dimension_of(...)`, `start`, `step`, `count`, and
  `coordinate_at(...)` remain the only sampling-coordinate mechanics;
- `ReadoutCollection` continues to use `require_field_types`,
  `require_same_axes`, `require_same_device`, and `require_same_dtype`; and
- stochastic production continues to use only public `CounterRng`,
  `logical_positions`, `uniform`, `gaussian`, `poisson`, and `binomial`.

It does not substitute generic helpers mechanically:

- readout fields accept arbitrary semantic-axis order, so
  `require_readout_structure(...)` remains local rather than forcing one
  `require_axis_signature(...)`;
- generated results must reuse the exact source axis tuple object, while
  `require_same_axes(...)` establishes value equality only;
- TensorCore does not own TensorDSLab's absolute dtype allowlists, storage-
  freshness rules, Charge count/envelope laws, physical delay kernels, or
  product-specific value domains; and
- generated-product validators remain the package-owned immediate
  postcondition boundary.

The current action split is already correct. This maintenance performs only
the following bounded cleanup while touching the physical configuration
boundary:

1. public `prepare_readout(...)` keeps exact ingress, request closure, dtype,
   device, RNG capability, and stochastic-key checks;
2. Config constructors keep Pint canonicalization, primitive value-domain
   validation, and genuine scientific or cross-field relationships, but
   remove annotation-only membership checks;
3. `require_exact`, `require_optional_exact`, and `require_one_of_exact` are
   deleted from `readout/requirements.py` once their Config call sites are
   removed, without replacement or a new diagnostic promise;
4. private child preparers remove duplicate exact Config, floating-dtype, and
   device admission checks already established by `prepare_readout(...)`;
5. private Charge effect executors remove duplicate exact Runtime-type and
   primitive-type guards for values supplied only through the typed package
   path;
6. exact model-union dispatch, tensor relationship checks, scientific
   representability checks, allocation/address/envelope checks, and
   generated-product validation remain;
7. `_tpc_raw(...)` and `_veto_raw(...)` receive plain floats rather than whole
   Configs;
8. PSD integration receives already-extracted plain tuples rather than a
   `PsdNoiseConfig`;
9. timing-jitter preparation owns extraction and its exact-zero decision;
10. afterpulse delay and recovery share one extracted mean delay; and
11. ADC bounds and every other active physical value are extracted once before
    arithmetic.

This is the TensorCore `0.13.0` golden path: runtime checks protect documented
public data and scientific relationships, not deliberate calls into private
actions or Config constructors with malformed typed objects. It adds no
generic Runtime ABC, Config ABC, validator framework, graph, registry, or
broad `utils.py` module.

TensorCore has identified exact integer-range normalization, field
dtype/layout requirements, and eager floating representability as possible
future generic requirements. They are unpublished and are not dependency
authority here. Maintenance 6 retains matching generic behavior locally and
makes no future TensorCore API claim; a later focused dependency maintenance
may adopt only an exact published surface. The current future Design direction
uses one variadic field-dtype requirement for both exact and accepted-set
checks, but Maintenance 6 neither implements nor depends on that proposal.

The existing validators require no redesign. They remain Pint-free and retain
their current `ValueError`/`RuntimeError` categories, exact source-axis tuple
identity, absolute/result dtype checks, device relationships, value scans, and
storage-freshness checks. Normalizing their historical exception categories or
weakening exact identity is outside this maintenance.

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

The exact Runtime rename map is:

| Runtime | Current field | Target field |
| --- | --- | --- |
| `WhiteNoiseRuntime` | `represented_rms` | `represented_rms_mv` |
| `PsdNoiseRuntime` | `represented_powers` | `represented_powers_mv2` |
| `AnalogWaveformRuntime` | `minimum` | `minimum_mv` |
| `AnalogWaveformRuntime` | `maximum` | `maximum_mv` |
| `DigitizedWaveformRuntime` | `slope` | `slope_per_mv` |
| `DigitizedWaveformRuntime` | `lower_input` | `lower_input_mv` |
| `DigitizedWaveformRuntime` | `upper_input` | `upper_input_mv` |

This does not require every product-specific tensor role to encode its full
physical dimension in its attribute name. `kernel`, `maximum_code`, `zero`,
`maximum`, and `intercept` retain their current names because the Runtime type
and preparation contract already fix their interpretation. No other Runtime
field is renamed.

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

Tests must include exact-equivalence fixtures where Pint `0.25.3` returns the
same canonical binary64 magnitude:

```text
1 MHz == 1_000_000 Hz
1 V == 1000 mV
```

They must separately lock the observed nonexact scale conversion:

```text
2 ns     -> 2000.0000000000002 ps
2000 ps  -> 1.9999999999999996 ns
```

TensorDSLab does not promise bitwise identity for every arbitrary decimal unit
conversion. If two inputs canonicalize to different binary64 magnitudes, they
are different numerical inputs even when their displayed decimal values seem
close. No tolerance, rounding, clipping, or quantization may be added merely
to force Config, Runtime, RNG, or product equality. The one-ULP
`SampleAxis.from_period(...)` integerization rule is the single explicit
exception and exists only because an integer axis cannot store a fractional
picosecond.

The migration changes no RNG key, logical position, ordinal, distribution,
draw schedule, or no-draw policy.

## Exact Pint `0.25.3` Selection Evidence

Design selected the exact release and artifacts named in the baseline after
read-only source, metadata, typing, registry, parser, and no-NumPy probes.
The fixed evidence establishes:

- Python `>=3.11` metadata and classifiers for 3.11, 3.12, and 3.13;
- BSD licensing;
- ordinary scalar construction and conversion with NumPy absent;
- `UnitRegistry(cache_folder=None)` without an application-registry mutation;
- `isinstance(value, pint.Quantity)` across ordinary independent registries;
- fresh reconstruction into the private TensorDSLab registry;
- `Measurement` and non-scalar magnitudes outside the accepted Quantity
  surface;
- the parser exception families translated above;
- Pyright `1.1.411`'s need for unparameterized public `Quantity` annotations
  and one localized construction/magnitude cast boundary; and
- the exact binary64 conversion behavior recorded above.

Authoritative upstream references are:

- [Pint 0.25.3 on PyPI](https://pypi.org/project/Pint/0.25.3/);
- [the exact 0.25.3 source tag](https://github.com/hgrecco/pint/tree/0.25.3);
- [Using Pint in projects, 0.25.3](https://pint.readthedocs.io/en/0.25.3/getting/pint-in-your-projects.html);
- [Pint base API, 0.25.3](https://pint.readthedocs.io/en/0.25.3/api/base.html);
  and
- [Pint typing, 0.25.3](https://pint.readthedocs.io/en/0.25.3/advanced/typing.html).

These are dependency evidence, not a delegation of TensorDSLab's contracts to
Pint. TensorDSLab retains the exact bounded public behavior in this work order.
It does not rely on an ambient application registry or whichever Pint version
happens to be installed, and TensorCore remains Pint-free.

## Exact Candidate Scope

The future implementation candidate may change exactly these 26 production
and dependency paths:

```text
pyproject.toml
tensor_dslab/__init__.py
tensor_dslab/common/__init__.py
tensor_dslab/common/axes.py
tensor_dslab/common/units.py
tensor_dslab/readout/config.py
tensor_dslab/readout/requirements.py
tensor_dslab/readout/runtime/prepare.py
tensor_dslab/readout/analog_waveform/config.py
tensor_dslab/readout/analog_waveform/runtime/prepare.py
tensor_dslab/readout/analog_waveform/runtime/produce.py
tensor_dslab/readout/charge/config.py
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/charge/runtime/effects/correlated_avalanches.py
tensor_dslab/readout/charge/runtime/effects/dark_counts.py
tensor_dslab/readout/charge/runtime/effects/delays.py
tensor_dslab/readout/charge/runtime/effects/smearing.py
tensor_dslab/readout/charge/runtime/effects/timing_jitter.py
tensor_dslab/readout/digitized_waveform/config.py
tensor_dslab/readout/digitized_waveform/runtime/prepare.py
tensor_dslab/readout/digitized_waveform/runtime/produce.py
tensor_dslab/readout/noise_waveform/config.py
tensor_dslab/readout/noise_waveform/runtime/prepare.py
tensor_dslab/readout/noise_waveform/runtime/produce.py
tensor_dslab/readout/pure_waveform/config.py
tensor_dslab/readout/pure_waveform/runtime/prepare.py
```

It may change exactly these 19 test and typing paths:

```text
tests/test_pint_physical_configuration.py
tests/typing/maintenance_6_pint_physical_configuration_boundary.py
tests/test_charge_correlated_avalanches.py
tests/test_charge_delay_preparation.py
tests/test_charge_product.py
tests/test_charge_timing_jitter.py
tests/test_deterministic_waveform_products.py
tests/test_noise_waveform_product.py
tests/test_package_contracts.py
tests/test_readout_axes_and_sampling.py
tests/test_readout_configs.py
tests/test_readout_product_types.py
tests/test_readout_simulation.py
tests/test_rng_ownership_migration.py
tests/test_runtime_action_ownership.py
tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py
tests/typing/maintenance_4_runtime_action_ownership.py
tests/typing/stage_4_deterministic_waveform_products.py
tests/typing/stage_7_public_readout_orchestration.py
```

The first two are new paths. Every other path exists at the Design baseline.
`tests/test_readout_product_types.py` may change only to remove imports,
executions, and diagnostic assertions for the three retired structural
membership helpers while preserving its dtype, floating-dtype,
representability, field, Runtime, and validator evidence.
Implementation may update only these two lifecycle records:

```text
docs/implementation/index.md
docs/implementation/maintenance_6_pint_physical_configuration_boundary.md
```

All other paths are protected candidate inputs. In particular:

- this Design authority's other synchronized living docs are not
  Implementation-owned;
- every closed work order, `docs/governance/**`,
  `docs/physics/correlated_avalanches.md`, `LICENSE`, `pyrightconfig.json`, and
  `tensor_dslab/py.typed` are protected;
- `tensor_dslab/readout/__init__.py`, every product facade and `field.py`,
  `ReadoutCollection`, `simulation.py`, Photoelectrons, runtime package
  markers, `readout/runtime/sampling.py`, product validators, Charge
  `produce.py`/`validate.py`/`effects/counts.py`, and the pure producer are
  protected; and
- scientific equations, RNG keys/addresses/calls, category ordering, result
  laws, storage/autograd contracts, and dtype/device behavior are protected
  even inside an allowlisted path.

Allowlisted producers may change only exact Runtime field references. The
allowlisted effect executors may remove only the redundant private guards
fixed above. Any other need stops to Design.

The public export counts are frozen:

```text
tensor_dslab:          33 -> 35
tensor_dslab.common:    3 -> 5
tensor_dslab.readout:  30 -> 30
```

The only new exports are `quantity` and `quantities`, added to the package
root and `tensor_dslab.common`. No product facade changes.

## Required Focused Evidence

The dispatched stage must include committed evidence for all of the following.

### Registry and helper behavior

- package-registry quantities from `quantity` and `quantities`;
- exact accepted and rejected magnitude/container types;
- chained `ValueError` for an oversized exact integer that cannot normalize to
  a finite built-in float;
- helper rejection of empty, whitespace, scaled, malformed, and undefined
  unit expressions across every fixed parser-exception family;
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
- static typing fixes the exact current wrapper, key, nested-Config,
  optional, union, and other dimensionless field types;
- focused source and mutation checks prove that annotation-only
  `require_exact`, `require_optional_exact`, and `require_one_of_exact` calls
  and their helper definitions are absent;
- tests do not replace the retired membership checks with a promised runtime
  exception for malformed typed composition;
- dB remains numeric and uses the accepted amplitude conversion; and
- all 22 target Config classes are explicitly unhashable for every valid
  composition.

### Axis and sampling behavior

- compact, range-backed integer `SampleAxis` state;
- exact-int and one-ULP-nearest-integer `from_period(...)` conversion,
  including `2 ns` and `0.5 ns`, plus rejection of a genuine positive
  fractional-picosecond period and an ambiguous oversized float;
- fresh `start_time`, `sample_period`, `time_at`, and exclusive `stop_time`
  quantities with exact built-in-int magnitudes in canonical picoseconds,
  including a legal coordinate above `2**53` and exact accessor-to-
  `from_period(...)` period round-trip;
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
- exact mutation probes reject restored child-preparer admission policing,
  restored effect Runtime- or primitive-type admission policing, parent-side
  jitter magnitude access, repeated ADC/PSD/afterpulse extraction, and
  Config-bearing numerical helpers;
- exact mutation probes reject restoration of annotation-only Config
  membership checks, while preserving every genuine Config relationship;
- all invalid contextual combinations fail before RNG consumption or writes;
  and
- prepared plain operands match the closed implementation for exact canonical
  fixtures.

### Result and continuity behavior

- deterministic pure, analog, and digitized reference equations;
- white/PSD noise and Charge same-stack replay for matching canonical
  operands;
- exact-zero branches remain draw-free;
- exact equivalent-unit fixtures (`1 MHz`/`1_000_000 Hz` and
  `1 V`/`1000 mV`) produce equal canonical Configs, Runtime operands, RNG
  calls, and products;
- the `2 ns`/`2000 ps` fixture proves physical equivalence without falsely
  requiring equal canonical binary64 Config magnitudes;
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
- exact Pint `0.25.3` wheel/source hashes and retained TensorCore `0.13.0`
  commit pin in source and built/archive forms;
- clean package-root import from an isolated artifact; and
- no retired unit-suffixed public config field or compatibility shim.

## Full Validation And Independent Review

The stage uses at most three Implementation-to-Validation candidate
submissions and at most three Validation-to-Implementation returns. Each
Validation submission is one immutable clean commit; a rejected candidate
remains immutable history. Exhausting either budget returns to Design rather
than silently extending the loop.

1. Implementation runs the focused matrix, full test discovery, Pyright,
   exact TensorCore source/archive forms, exact Pint wheel and sdist-built
   forms, TensorDSLab source/artifact isolation, diff, import, privacy, and
   artifact gates on the exact candidate.
2. Validation independently reconstructs the exact TensorCore and Pint
   dependencies, repeats all local gates, and runs mutation-style probes
   against the unit/preparation boundary. It must exercise the exact Pint
   wheel and independently verify the sdist-built installation before
   dispatching one unchanged fixed commit. The `13` conditional CUDA skips are
   recorded qualifications rather than failures.
3. Review independently audits API clarity, registry isolation, every config
   migration, one-time preparation extraction, Runtime purity, numerical/RNG
   continuity, dependency identity, typing, documentation, and the complete
   local source/artifact evidence on one immutable Validation-cleared commit.
4. Review returns at most one complete finding packet. Any correction requires
   explicit bounded Design authority and a fresh Validation pass; no
   Review-to-Implementation loop is implicit.
5. Only an unchanged Review-cleared commit may be fast-forwarded with
   `git merge --ff-only`. Design then performs the bounded evidence-only
   closeout defined under Documentation And History.

No fresh GPU evidence is collected in this maintenance. Accelerator
qualification waits for the exact integrated TensorCore `0.15.0` adoption
baseline described above. Stage 8 characterization remains a separate future
authority.

## Documentation And History

This Design authority synchronizes exactly:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_6_pint_physical_configuration_boundary.md
docs/implementation/proposed_pint_physical_configuration_boundary.md
docs/overview.md
docs/parity.md
docs/validation.md
```

The `docs/parity.md` change records the exact physical-equivalence-versus-
binary-equality boundary; it changes no donor classification. Closed work
orders and historical blobs are not rewritten to pretend they used Pint.
Once this authority is committed, Implementation may edit only the two
lifecycle records listed in the candidate allowlist. That Implementation
allowance does not limit the later Design-owned final closeout.

After Review fast-forwards the exact cleared target, Design is explicitly
authorized to create one evidence-only direct-child closeout commit changing
exactly these eleven live-status records:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_6_pint_physical_configuration_boundary.md
docs/overview.md
docs/validation.md
```

That commit may only replace pending lifecycle wording with **Merged /
Closed**, bind the exact Review-cleared commit/tree and accepted evidence, and
record the unavailable-CUDA/no-accelerator qualification and the no-push
sequence. It must change no production, test, dependency, scientific, API,
RNG, parity, governance, or historical byte. TensorDSLab `main` remains local
and unpushed after Maintenance 6 closeout. The next package gate is the
separately designed TensorCore `0.15.0` adoption; only after that adoption and
the exact integrated CUDA gates close may TensorDSLab be pushed.

This work order is operative through the user's explicit Maintenance 6
dispatch and the exact Design authority named in the implementation handoff.
The post-candidate evidence-scheduling amendment synchronizes only
current-state documentation within the Design allowlist above. It preserves
Candidate 1 immutably and does not consume an Implementation correction slot.

## Non-Goals

This work order does not authorize:

- TensorCore changes or a Pint dependency in TensorCore;
- depending on, claiming, or locally recreating an unpublished future
  TensorCore requirements surface;
- adopting TensorCore `0.15.0` or performing its downstream migration inside
  Maintenance 6;
- reopening or rewriting TensorDSLab Maintenance 5;
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
- a cluster submission or fresh real-CUDA qualification for Maintenance 6;
- Stage 8 benchmarking, compiler fusion, workspaces, `out=`, or allocation
  claims;
- release, deployment, conformance, backward-compatibility, or broad
  cross-package compatibility claims; or
- any TensorDSLab push before the separate TensorCore `0.15.0` adoption and
  exact integrated CUDA gates close.

## Stop Conditions

Stop the affected work and return exact evidence to TensorDSLab Design if:

- the closed Maintenance 5 compact-axis/sampling contract or exact TensorCore
  `0.13.0` pin differs from the accepted prerequisite assumed here;
- exact Pint `0.25.3` artifacts or their required scalar behavior cannot be
  reconstructed;
- an external-registry scalar cannot be safely copied without relying on
  mutable global registry state;
- the migration changes a scientific equation, stochastic address/draw order,
  or accepted canonical operand unexpectedly;
- a Runtime, producer, or validator requires a Pint object to meet the design;
- unit conversion cannot complete before RNG consumption and writes;
- `SamplingConfig` or string-valued SampleAxis coordinates reappear at the
  execution baseline;
- source/archive, wheel/sdist, local execution, or typing behavior conflicts
  at the accepted comparison boundary;
- an edit is required outside the exact allowlist;
- the repository, dependency, branch, or execution route is dirty, stale, or
  discrepant; or
- implementation would require IO, persistence, integration, performance, or
  another non-goal to proceed.

## Completion Boundary

Maintenance 6 becomes eligible for final acceptance only when one exact
candidate:

- pins exact Pint `0.25.3` while retaining exact TensorCore `0.13.0` commit
  `202d8b1bc6259b8453d3d377570417f2480d782b`;
- implements the complete public physical-config migration without shims;
- exports exactly `quantity` and `quantities`, keeps readout exports unchanged,
  and retires the three redundant integer SampleAxis aliases;
- provides compact integer SampleAxis state plus the four selected Pint
  conveniences and the bounded one-ULP integerization rule;
- proves Config canonicalization and one-time preparation extraction;
- retires the three annotation-only structural membership helpers without
  weakening Pint, primitive-domain, or genuine relationship validation;
- completes exactly the bounded TensorCore/golden-path action cleanup;
- keeps Runtime, production, validation, and tensor execution unit-free;
- preserves accepted scientific, numerical, RNG, storage, device, and
  autograd behavior at the declared boundaries;
- passes the complete local source/artifact, typing, mutation,
  import/privacy, scope, protected-byte, and hygiene Validation/Review
  matrices with the CUDA skips disclosed; and
- is independently Review-cleared and cleanly fast-forwarded under this
  explicit user-authorized dispatch.

This document records the closed Maintenance 6 implementation and evidence
loop. Local `main` remains unpushed pending the separate TensorCore `0.15.0`
adoption and exact integrated CUDA gates.
