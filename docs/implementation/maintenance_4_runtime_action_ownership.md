# Maintenance 4 Runtime Action Ownership Work Order

Status: **Design-complete / User-authorized / Undispatched**.

Stable work-order key:
`TensorDSLab/maintenance-4-runtime-action-ownership`.

This is the package-authoritative TensorDSLab Design work order for a focused,
behavior-preserving internal readout refactor. It replaces the current
product-local `_produce.py` bundles with explicit preparation, production, and
validation actions under non-exported `runtime/` packages. It changes no
collaborator-facing API, scientific equation, stochastic address, product
meaning, result law, dependency, or supported device boundary.

The user authorized Design to finish and dispatch this work order. The exact
containing Design commit is named in the dispatch handoff because a commit
cannot contain its own hash. Production must not begin from an uncommitted or
later-modified version of this document.

## Objective

Reorganize the implemented readout pipeline around one explicit lifecycle:

```text
public Config plus preflight facts
  -> prepare_<product>(...)
  -> <Product>Runtime
  -> produce_<product>(...)
  -> <Product>
  -> validate_<product>(...)
  -> next dependent product
```

The maintenance must:

- keep `simulate_readout(...)` as the unchanged public request-level API;
- complete preparation for the entire requested closure before the first RNG
  request, product production call, or semantic-output write;
- replace ambiguous private `*Plan` records with exact product-owned
  `*Runtime` execution records;
- split every generated product into `runtime/prepare.py`,
  `runtime/produce.py`, and `runtime/validate.py`;
- move Charge scientific effects below `charge/runtime/effects/`;
- move explicit deep product validation out of `field.py` and producer
  functions into product-owned runtime validators;
- execute `produce -> validate -> descendant` for every generated product;
- keep production functions focused on tensor execution, RNG requests, and
  final semantic-field construction;
- consolidate genuinely identical preparation, production-support, and
  validation logic without introducing a framework; and
- preserve exact public behavior, same-stack results, RNG calls, source and
  result contracts, and accepted failure ordering.

The maintenance deliberately prepares the internal seam needed by a later
public `PureWaveformRenderer`, but it does not implement, export, or dispatch
that renderer.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`.

The exact clean starting package baseline is the Maintenance 3 Design
closeout:

```text
repository:              TensorDSLab
reference:               main
starting commit:         5fdd3fafe2c44357b09df2a04b88cb121f2d3638
Maintenance 3 candidate: dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0
package version:         0.1.0
Python requirement:      >=3.11
```

The selected dependency remains unchanged:

```text
repository:       TensorCore
reference:        origin/main
commit:           4708bf2ca063a1bcd37a30a342733b9e3dbe9f59
direct parent:    0e72f0e69cf9140b692d408e49a504cbdcb101b7
tree:             1012acf512933f4077fd63267d95fc97f9ee8842
package version:  0.9.0
archive SHA-256:  f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd
```

The Design branch is:

```text
codex/maintenance-4-runtime-action-ownership-design
```

The Implementation branch is fixed as:

```text
codex/maintenance-4-runtime-action-ownership
```

Implementation creates that branch from the exact committed Design authority.
Raw platform route identifiers remain private and must not enter committed
files.

Package state remains:

```text
package_adoption_state:    Adopted
conformance_finding:       Not evaluated
coordination_status:       Deferred
registry_storage_profile:  Disabled
maintenance_3:             Merged / Closed
maintenance_4:             Design-complete / User-authorized / Undispatched
stage_8:                   Stopped / superseding authority required
```

Coordination is not an execution route. This work authorizes no push.

## Applicable Contracts And Source Precedence

Implementation, Validation, and Review must read and reconcile:

- `AGENTS.md` for role, routing, authority, handoff, finite-loop, and merge
  rules;
- `CONTRIBUTING.md`, especially Domain Organization, Target Domain Simulation
  Surface, Boundary-First Validation, Public Surface Discipline, Code
  Expectations, Test Expectations, and Scope Discipline;
- [Rebuild Architecture](../architecture/rebuild.md), especially Runtime
  Inputs, Product Runtime Actions, Public Builder, Functional/Memory/Lifetime
  Contract, and Validation Strategy;
- [Readout Architecture](../architecture/readout.md);
- [TensorCore Integration](../architecture/tensors.md);
- [Validation](../validation.md);
- [Parity](../parity.md);
- the closed Stage 3 through Stage 7 and Maintenance 2/3 work orders as exact
  historical implementation and evidence records; and
- TensorCore's public API at exact `0.9.0` commit
  `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`.

This work order controls the Maintenance 4 production slice. Live architecture
documents control current package meaning. Closed work orders remain immutable
historical evidence; their old private paths and names are not current
architecture after Maintenance 4. If sources appear to conflict, stop the
affected work and return the exact contradiction to TensorDSLab Design.

`docs/parity.md` requires no semantic change because this maintenance promotes
no donor behavior, changes no comparison boundary, and accepts no new
approximation or intentional divergence. Validation and Review must confirm
that it remains unchanged.

## Design Finding

The current implementation correctly separates product identity and config
from readout orchestration, but each `_produce.py` still owns three different
responsibilities:

```text
scientific/contextual preparation
tensor execution and product construction
deep result validation
```

Stage 7 also uses `*Plan` for records that are not schedules or requests: they
are the already-prepared operands consumed by one execution path. The selected
architecture names these records `*Runtime` and gives each action one physical
module.

The split is meaningful rather than decorative:

- preparation may inspect configs, convert units, prove bounds, derive
  probabilities, round constants, and materialize device operands;
- production consumes typed prerequisites and prepared operands and performs
  the actual tensor/RNG computation; and
- validation inspects the completed semantic product before any descendant or
  collection publication may consume it.

Privacy is defined by deliberate facade exports and documentation. Runtime
paths remain importable Python implementation details, but are unsupported and
carry no compatibility promise. A leading underscore is therefore not used on
cross-module runtime actions or records. TensorCore's inherited `_require()`
semantic-leaf hook and genuinely module-local helpers remain exceptions
because their underscore has a different, established meaning.

## Exact Target Package Tree

The accepted production target is:

```text
tensor_dslab/
  common/
    axes.py
    sampling.py
  readout/
    __init__.py
    config.py
    collection.py
    requirements.py
    simulation.py
    runtime/
      __init__.py
      prepare.py
      sampling.py
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
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py
        effects/
          __init__.py
          counts.py
          delays.py
          dark_counts.py
          timing_jitter.py
          correlated_avalanches.py
          smearing.py
    pure_waveform/
      __init__.py
      config.py
      field.py
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py
    noise_waveform/
      __init__.py
      config.py
      field.py
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py
    analog_waveform/
      __init__.py
      config.py
      field.py
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py
    digitized_waveform/
      __init__.py
      config.py
      field.py
      runtime/
        __init__.py
        prepare.py
        produce.py
        validate.py
```

Every listed new module other than the required package-marker `__init__.py`
files owns real behavior in this maintenance. No placeholder behavior module
is authorized. `Photoelectrons` remains already-produced truth and
therefore owns no config, preparer, producer, or Runtime record; it owns only
its field and explicit deep validator.

Every `runtime/__init__.py` and `runtime/effects/__init__.py` is empty. It must
not re-export actions, records, effects, or helper names and must not create an
internal facade. Internal callers import the exact defining module.

The following old live paths are removed without shims:

```text
tensor_dslab/readout/_requirements.py
tensor_dslab/readout/*/_produce.py
tensor_dslab/readout/charge/effects/
```

Closed work orders may continue to name those paths as historical facts.

## Public And Private Contract

The exact public surface and object identities remain unchanged. In
particular, these continue to be deliberate facade exports:

```text
tensor_dslab
tensor_dslab.readout
tensor_dslab.readout.<product>
```

No `Runtime`, preparation, production, validation, effect, requirement, or
request-planning name is added to any existing `__all__` or imported by any
public facade. The following public files are protected from change except
where explicitly noted otherwise:

```text
tensor_dslab/__init__.py
tensor_dslab/readout/__init__.py
tensor_dslab/readout/analog_waveform/__init__.py
tensor_dslab/readout/charge/__init__.py
tensor_dslab/readout/digitized_waveform/__init__.py
tensor_dslab/readout/noise_waveform/__init__.py
tensor_dslab/readout/photoelectrons/__init__.py
tensor_dslab/readout/pure_waveform/__init__.py
```

Direct deep imports such as:

```python
from tensor_dslab.readout.charge.runtime.produce import produce_charge
```

remain possible because this is Python, but are unsupported. Maintenance 4
promises no stable signature, error category, import path, compatibility shim,
or deprecation window for such use. Tests must define privacy through facade
exports rather than through `hasattr(package, "runtime")`, because importing a
submodule may attach it as an ordinary package attribute.

## Runtime Records

Every product Runtime is a concrete frozen slotted dataclass defined in that
product's `runtime/prepare.py`:

```python
@final
@dataclass(frozen=True, slots=True)
class ChargeRuntime:
    ...
```

The exact top-level records are:

```text
SamplingRuntime
ReadoutRuntime
ChargeRuntime
PureWaveformRuntime
NoiseWaveformRuntime
AnalogWaveformRuntime
DigitizedWaveformRuntime
```

Focused product-owned variant or effect preparation may define additional
records when the execution genuinely has a closed typed variant. This includes
the current `ZeroNoiseRuntime | WhiteNoiseRuntime | PsdNoiseRuntime` closed
union and may include prepared Charge effects such as:

```text
DelayRuntime
DarkCountRuntime
TimingJitterRuntime
CorrelatedAvalancheRuntime
ChargeSmearingRuntime
```

A Runtime record:

- has no base other than `object`;
- is frozen and slotted;
- is not a TensorCore semantic value;
- is never exported;
- contains no `Config` object;
- contains no `Photoelectrons`, `Charge`, waveform field, collection, or other
  previous semantic product;
- contains no mutable Python collection/cache, lazy property, execution method,
  or hidden device movement;
- stores only prepared tensors and static Python values required by
  production, request-wide stochastic-role checks, or immediate result
  validation;
- treats every prepared tensor operand as read-only after construction; no
  runtime action mutates it; and
- is discarded after the one request.

Do not introduce a `Runtime` ABC, protocol, mixin, generic base, registry,
mapping, visitor, dependency graph object, action object, or framework.

`ReadoutRuntime` is one private composition of optional product runtimes:

```python
@final
@dataclass(frozen=True, slots=True)
class ReadoutRuntime:
    charge: ChargeRuntime | None
    pure_waveform: PureWaveformRuntime | None
    noise_waveform: NoiseWaveformRuntime | None
    analog_waveform: AnalogWaveformRuntime | None
    digitized_waveform: DigitizedWaveformRuntime | None
```

Requested retention is returned separately as the exact
`frozenset[type[TensorField]]`. It is not execution data and does not belong in
a product Runtime. Optional Runtime presence is the closure execution signal;
the old five duplicated `need_*` booleans are removed.

## Shared Sampling Runtime

`tensor_dslab.readout.runtime.sampling` is a small dependency-leaf module. It
exists separately because both request preparation and product preparers need
its record. Defining it in `readout.runtime.prepare` would create a cycle when
that module imports product preparers; placing it in `common.sampling` would
mix private readout execution state into the public cross-domain sampling
model.

It owns exactly one record and one preparer, conceptually:

```python
@dataclass(frozen=True, slots=True)
class SamplingRuntime:
    sample_count: int
    sample_period_ps: int
    sample_dimension: int


def prepare_sampling(
    photoelectrons: Photoelectrons,
    *,
    config: SamplingConfig,
) -> SamplingRuntime:
    ...
```

The values remain Python integers because they control dimensions, shapes,
loop bounds, FFT lengths, and prepared reference execution. They are not
payload tensors and must not require `.item()` on the production path.

`prepare_readout(...)` establishes this record exactly once after validating
the public sampling relationship. Every temporal product Runtime that retains
sampling facts must reference that exact object. No product independently
rediscovers the source sample dimension during one `simulate_readout(...)`
call.

This simulation-bound axis binding does not freeze the future renderer design.
A later reusable renderer may bind the supplied field's sample dimension once
per `forward()` while reusing an already-prepared scientific kernel.

## Readout Preparation

`tensor_dslab.readout.runtime.prepare` owns:

```text
ReadoutRuntime
prepare_readout(...)
request parsing and typed closure
required-config closure
closure-wide RNG-key uniqueness
composition of product Runtime values
```

`tensor_dslab.readout.simulation` becomes the thin public orchestration module.
It owns only the public signature, topological action sequence, exact
retention, and final `ReadoutCollection` construction.

Conceptually:

```python
def prepare_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> tuple[frozenset[type[TensorField]], ReadoutRuntime]:
    ...
```

It must preserve Stage 7's complete-preflight boundary:

```text
consume request exactly once
  -> reject empty, duplicate, or unknown product classes
  -> derive exact transitive closure
  -> validate exact source/config/RNG and required config presence
  -> validate source axes, sampling, device, dtype, and deep PE values
  -> prepare shared SamplingRuntime
  -> prepare every required product Runtime
  -> reject duplicate stochastic-role keys
  -> return requested membership plus ReadoutRuntime
```

Every supported request-level and statically preparable relationship failure
must still occur before the first RNG request, product production call, or
semantic-output write. Preparation may allocate unexposed scalar/control
tensors and perform accepted read-only validation reductions. It does not
promise allocation-free or synchronization-free preflight.

`prepare_readout` may import product `runtime.prepare` modules. Product runtime
modules must not import `ReadoutConfig`, `ReadoutRuntime`,
`ReadoutCollection`, `simulation`, or `simulate_readout`.

## Product Preparation

Each generated product `runtime/prepare.py` owns:

- its concrete ProductRuntime definition;
- config interpretation;
- unit conversion;
- scientific/contextual equations that derive execution operands;
- representability and numerical-bound proofs;
- materialization of device tensors required by production; and
- focused preparation helpers used only by that product.

The conceptual signatures are:

```python
def prepare_charge(
    config: ChargeConfig,
    *,
    photoelectrons: Photoelectrons,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
) -> ChargeRuntime:
    ...


def prepare_pure_waveform(
    config: PureWaveformConfig,
    *,
    sampling: SamplingRuntime,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> PureWaveformRuntime:
    ...


def prepare_noise_waveform(
    config: NoiseWaveformConfig,
    *,
    sampling: SamplingRuntime,
    shape: tuple[int, ...],
    floating_dtype: torch.dtype,
    device: torch.device,
) -> NoiseWaveformRuntime:
    ...


def prepare_analog_waveform(
    config: AnalogWaveformConfig,
    *,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> AnalogWaveformRuntime:
    ...


def prepare_digitized_waveform(
    config: DigitizedWaveformConfig,
    *,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> DigitizedWaveformRuntime:
    ...
```

The exact source/execution facts may be narrowed during implementation if the
record remains sufficient and no accepted validation or result changes. A
preparer may inspect a prerequisite/source product during preflight when its
values are scientifically required for a bound; it must never retain that
product in the Runtime.

Because the public `ReadoutConfig` constructor and `prepare_readout` have
already validated exact config membership, private product preparers need not
repeat the entire public type/config boundary. Public surfaces added later,
such as a renderer accepting a config directly, must validate their own
arguments before calling a private preparer.

### PureWaveform Preparation

Pure-waveform preparation becomes independent of a particular source field.
It consumes the shared sampling facts, product config, floating dtype, and
device. It retains the existing binary64 TPC/Veto equations, support,
normalization, coefficient rounding, and record-length cropping exactly.

The prepared coefficient tensor is flipped and shaped once during preparation:

```python
kernel = coefficients.flip(0).reshape(1, 1, coefficient_count)
```

`PureWaveformRuntime` stores the convolution-ready kernel and the shared
`SamplingRuntime`. Production must not repeat coefficient construction,
flipping, or reshaping. Exact same-stack output and autograd behavior must
remain unchanged.

### DigitizedWaveform Preparation

`DigitizedWaveformRuntime` must not retain the original
`DigitizedWaveformConfig`. It retains the already-prepared scalar/device
operands needed by production and the exact Python `maximum_code` needed by
validation. This prevents a config/runtime disagreement and keeps production
and validation independent of Config objects.

## Product Production

Each generated product `runtime/produce.py` owns exactly one first-class
cross-module action:

```text
produce_charge
produce_pure_waveform
produce_noise_waveform
produce_analog_waveform
produce_digitized_waveform
```

Conceptually:

```python
def produce_charge(
    photoelectrons: Photoelectrons,
    *,
    runtime: ChargeRuntime,
    rng: CounterRng,
) -> Charge:
    ...


def produce_pure_waveform(
    charge: Charge,
    *,
    runtime: PureWaveformRuntime,
) -> PureWaveform:
    ...


def produce_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    runtime: NoiseWaveformRuntime,
    rng: CounterRng,
) -> NoiseWaveform:
    ...


def produce_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    runtime: AnalogWaveformRuntime,
) -> AnalogWaveform:
    ...


def produce_digitized_waveform(
    analog: AnalogWaveform,
    *,
    runtime: DigitizedWaveformRuntime,
) -> DigitizedWaveform:
    ...
```

A producer receives only:

- its exact prerequisite semantic product or products;
- its exact prepared ProductRuntime; and
- `CounterRng` when the product is stochastic-capable.

No producer imports or receives a Config, ReadoutConfig, ReadoutRuntime,
ReadoutCollection, public simulation function, or product validator.

Production is tensor execution rather than scientific interpretation. It may:

- read prerequisite tensor payloads and metadata already bound by preflight;
- issue the accepted TensorCore RNG distribution requests;
- execute tensor views, indexing, arithmetic, FFT, convolution, redistribution,
  dtype conversion for a declared fresh result, and product-specific control
  flow over prepared values;
- dispatch by exact type over a closed prepared Runtime variant union, such as
  `ZeroNoiseRuntime | WhiteNoiseRuntime | PsdNoiseRuntime`;
- enforce narrow dynamic guards that can only be known after RNG/backend
  execution, including count, overflow, and ledger invariants;
- prevent any write through named input storage; and
- construct exactly one completed semantic field after all writes to its
  storage have been initiated or enqueued.

It must not:

- interpret Config values or rediscover/select a scientific model from Config;
- convert scientific units;
- derive probability tables, response coefficients, PSD powers, saturation
  bounds, ADC transfer constants, or count/ledger envelopes;
- choose dtype or device policy;
- rediscover the sample dimension for the simulation request;
- perform the product's deep publication scan;
- call `.cpu()`, `.numpy()`, `.tolist()`, detach, or move an existing input;
  or
- write through an alias after constructing the result field.

“Tensor execution” does not forbid correctness-critical dynamic checks in the
stochastic Charge reference. It forbids moving predictable scientific
preparation or completed-product validation back into the hot path merely to
avoid the new action boundary.

## Product Validation

Each product `runtime/validate.py` owns one first-class deep validator:

```text
validate_photoelectrons
validate_charge
validate_pure_waveform
validate_noise_waveform
validate_analog_waveform
validate_digitized_waveform
```

Product `field.py` contains only the final TensorField leaf and its cheap
intrinsic TensorCore `_require()` narrowing. It contains no deep tensor-value
scan or imported runtime action.

A runtime validator:

- validates the exact completed product supplied to it;
- receives the named direct prerequisite product or products needed to prove
  the result relationship, plus only the minimal prepared scalar facts required
  by that product;
- performs no repair, clipping, casting, movement, replacement construction,
  write, or config interpretation;
- preserves the product's accepted finite/range/value law and verifies its
  axes, shape, dtype, device, and fresh-storage relationship to those named
  prerequisites; and
- runs exactly once at the public simulation boundary before a descendant or
  returned collection may consume the result.

`validate_digitized_waveform` receives the prepared exact `maximum_code` or
the exact `DigitizedWaveformRuntime`; it does not receive a
`DigitizedWaveformConfig`.

The conceptual validation signatures are relationship-specific rather than
artificially identical:

```python
def validate_photoelectrons(photoelectrons: Photoelectrons) -> None: ...

def validate_charge(
    charge: Charge,
    *,
    source: Photoelectrons,
    runtime: ChargeRuntime,
) -> None: ...

def validate_pure_waveform(pure: PureWaveform, *, source: Charge) -> None: ...

def validate_noise_waveform(
    noise: NoiseWaveform,
    *,
    source: Photoelectrons,
    runtime: NoiseWaveformRuntime,
) -> None: ...

def validate_analog_waveform(
    analog: AnalogWaveform,
    *,
    pure: PureWaveform,
    noise: NoiseWaveform,
) -> None: ...

def validate_digitized_waveform(
    digitized: DigitizedWaveform,
    *,
    source: AnalogWaveform,
    maximum_code: int,
) -> None: ...
```

Implementation may pass an exact ProductRuntime instead of one extracted
prepared scalar when that is the narrower, clearer signature, but it must not
pass a Config or make the validator recompute preparation.

`validate_charge` becomes the sole deep terminal-value scan for a generated
Charge. The current producer performs one pre-construction finite/nonnegative
scan that raises `RuntimeError` and then repeats equivalent checks through the
field validator. Maintenance 4 removes that duplicate. The producer may
construct one local `Charge`, after which `validate_charge` performs the scan
and preserves the public `RuntimeError` behavior for an invalid generated
terminal value. The invalid local field is never returned, retained, or passed
to a descendant. `validate_charge` also owns the current metadata-only
fresh-storage postcondition and preserves its public `RuntimeError`; the
producer creates fresh values but does not duplicate that publication check.

`validate_photoelectrons` remains an untrusted-ingress validator and retains
its accepted `ValueError` semantics during whole-request preflight. The other
generated validators preserve their current public simulation error
categories and messages where those are asserted accepted behavior.

`validate_photoelectrons` must not absorb Charge's count-domain ceiling. A
truth-only request continues to accept any nonnegative `torch.int64`
Photoelectrons payload representable by its tensor. The per-cell `2**53 - 1`
and related count/allocation bounds remain Charge-specific preparation and run
only when the requested closure requires Charge.

## Execution And Publication Order

After successful whole-request preparation, `simulate_readout(...)` uses this
exact topological pattern:

```python
requested, runtime = prepare_readout(...)

if runtime.charge is not None:
    charge = produce_charge(photoelectrons, runtime=runtime.charge, rng=rng)
    validate_charge(charge, source=photoelectrons, runtime=runtime.charge)

if runtime.pure_waveform is not None:
    pure = produce_pure_waveform(charge, runtime=runtime.pure_waveform)
    validate_pure_waveform(pure, source=charge)

if runtime.noise_waveform is not None:
    noise = produce_noise_waveform(
        photoelectrons,
        runtime=runtime.noise_waveform,
        rng=rng,
    )
    validate_noise_waveform(
        noise,
        source=photoelectrons,
        runtime=runtime.noise_waveform,
    )

if runtime.analog_waveform is not None:
    analog = produce_analog_waveform(
        pure,
        noise,
        runtime=runtime.analog_waveform,
    )
    validate_analog_waveform(analog, pure=pure, noise=noise)

if runtime.digitized_waveform is not None:
    digitized = produce_digitized_waveform(
        analog,
        runtime=runtime.digitized_waveform,
    )
    validate_digitized_waveform(
        digitized,
        source=analog,
        maximum_code=runtime.digitized_waveform.maximum_code,
    )

return ReadoutCollection(fields=retain_exactly_requested(...))
```

Optional narrowing may use explicit internal assertions after complete
preflight. The implementation must not produce all fields and validate only at
the end. A failed Charge or PureWaveform validation must occur before the
NoiseWaveform RNG request; a failed NoiseWaveform validation must occur before
AnalogWaveform production; and so forth. No invalid intermediate may reach a
descendant.

The only collection construction remains last. No failed call returns a
partial collection or exposes a failed local product. Dynamic execution
failure carries no rollback guarantee for private allocations, RNG work
already requested, or valid local prerequisites already computed.

## Charge Runtime Effects

Charge effects move intact beneath:

```text
tensor_dslab.readout.charge.runtime.effects
```

The first-class cross-module action names are clean and direct:

```text
prepare_dark_counts / simulate_dark_counts
prepare_timing_jitter / simulate_timing_jitter
prepare_correlated_avalanches / simulate_correlated_avalanches
prepare_charge_smearing / simulate_charge_smearing
prepare_delay
```

`counts.py` retains Charge-owned aggregate multinomial/category orchestration,
count-domain requirements, and allocation/address helpers. `delays.py` retains
the exact fixed/exponential and recovery preparation. The effect package
contains no public facade.

Moving or renaming an effect must not change:

- an accepted probability, rate, delay, recovery, pulse, or smearing equation;
- binary64 evaluation and rounding order;
- tensor operation order where same-stack values are frozen;
- RNG key, position, quantum, ordinal, raw-word request, or final no-draw
  remainder;
- category or generation order;
- exact-zero/no-op draw behavior;
- count/ledger/overflow bounds;
- right-overflow accounting; or
- source and result storage behavior.

Records that are genuinely prepared execution operands use a `Runtime` suffix.
Ephemeral result carriers and genuinely module-local mathematical helpers need
not be renamed merely to satisfy a superficial pattern.

## Symmetry And Duplicate-Logic Reduction

The end-state must be reconstructible from every generated product package:

```text
config.py             public scientific intent
field.py              public semantic result identity
runtime/prepare.py    Config + execution facts -> ProductRuntime
runtime/produce.py    prerequisites + ProductRuntime (+ RNG) -> Product
runtime/validate.py   Product + minimal prepared facts -> None
```

`Photoelectrons` is the explicit ingress exception: `field.py` plus
`runtime/validate.py`, with no Config, Runtime, preparer, or producer.

Implementation must perform an explicit duplicate-logic audit before freezing
Candidate 1. At minimum, inspect:

- repeated exact-type, axis, shape, dtype, device, sampling, and scalar
  representability requirements;
- repeated finite and nonnegative tensor scans;
- repeated scalar-to-device materialization;
- repeated freshness or relationship checks;
- repeated config checks already guaranteed by public construction and whole-
  request preparation;
- the duplicate `_PRODUCT_TYPES` inventory in current orchestration, which
  must be replaced by `ReadoutCollection.accepted_field_types()` while keeping
  the dependency closure as explicit typed code rather than a registry; and
- duplicate preparation repeated in production, including the PureWaveform
  kernel flip/reshape.

Genuinely identical behavior should be extracted into the narrowest sensible
private owner, usually `readout/requirements.py`, one product preparer, or one
Charge effect helper. Product validators remain explicit product-named
wrappers even when they delegate to one shared finite-value primitive; this
keeps errors and scientific ownership legible.

Do not force symmetry through identical unused parameters, a generic Action or
Runtime base, untyped dictionaries, reflection, registries, product IDs,
string dispatch, decorators that conceal control flow, or broad `utils.py` /
`helpers.py` modules. Extraction is accepted only when semantics, dtype/device
behavior, error category, operation ordering, RNG behavior, and autograd remain
identical.

Cross-module internal action names and Runtime record names omit leading
underscores because package privacy is export-driven. Genuinely module-local
special-function evaluators and small implementation helpers may retain a
leading underscore; a blanket rename of unrelated local symbols is not part
of this maintenance.

## Shared Requirements

The current `readout/_requirements.py` moves to
`readout/requirements.py` without a compatibility shim. Cross-module helper
names drop leading underscores consistently where they are imported by
multiple runtime/config/field modules. TensorCore's semantic-leaf `_require()`
method name remains unchanged because it is the accepted TensorCore extension
hook rather than a TensorDSLab privacy marker. Genuinely module-local helpers
may retain a leading underscore; no cross-module `_prepare_*`, `_produce_*`,
`_simulate_*`, or shared `_require_*` alias remains.

`requirements.py` may own only genuinely shared readout relationships and
validation primitives. It must not become a product registry, config union,
dependency map, runtime framework, scientific-equation dumping ground, or
public facade.

## Behavior-Preservation Contract

Maintenance 4 changes private ownership and one redundant Charge validation
implementation detail. It preserves the complete public Stage 7 contract.

On an unchanged accepted numerical stack, the refactor must preserve:

- exact `simulate_readout(...)` signature and facade object identity;
- exact config, axis, product field, collection, and TensorCore class identity;
- every successful result tensor's dtype, device, shape, axes identity,
  values, strides/layout where currently guaranteed, and autograd behavior;
- exact source return and source immutability;
- generated-product freshness and pairwise retained-output independence;
- exact product-request parsing, closure, execute-once behavior, request-order
  invariance, and requested-only retention;
- complete-preflight-before-RNG/production/write ordering;
- exact RNG key, address, distribution call, call count, call ordering,
  no-draw branch, and same-stack completed stochastic values;
- every scientific equation, bound, tolerance, and accepted statistical law;
- immediate generated-product validation before descendant use;
- no partial collection or exposed failed field;
- no silent source movement, casting, detachment, host materialization, IO, or
  persistence; and
- the Maintenance 3 numerical-stack qualification of completed stochastic
  literals.

Private path, record, and function identities intentionally change and have no
compatibility promise. The narrow direct-call error behavior of removed private
guards is not a public contract. The public `simulate_readout(...)` behavior
and accepted failure ordering remain the comparison boundary.

## Exact Production Scope

Implementation may change only the following production scope:

```text
M/D/A tensor_dslab/readout/_requirements.py -> requirements.py
M     tensor_dslab/readout/config.py
M     tensor_dslab/readout/simulation.py
A     tensor_dslab/readout/runtime/__init__.py
A     tensor_dslab/readout/runtime/prepare.py
A     tensor_dslab/readout/runtime/sampling.py

M     tensor_dslab/readout/photoelectrons/field.py
A     tensor_dslab/readout/photoelectrons/runtime/__init__.py
A     tensor_dslab/readout/photoelectrons/runtime/validate.py

M     tensor_dslab/readout/charge/config.py
M     tensor_dslab/readout/charge/field.py
D     tensor_dslab/readout/charge/_produce.py
D     tensor_dslab/readout/charge/effects/__init__.py
D     tensor_dslab/readout/charge/effects/_counts.py
D     tensor_dslab/readout/charge/effects/_delays.py
D     tensor_dslab/readout/charge/effects/_dark_counts.py
D     tensor_dslab/readout/charge/effects/_timing_jitter.py
D     tensor_dslab/readout/charge/effects/_correlated_avalanches.py
D     tensor_dslab/readout/charge/effects/_smearing.py
A     tensor_dslab/readout/charge/runtime/__init__.py
A     tensor_dslab/readout/charge/runtime/prepare.py
A     tensor_dslab/readout/charge/runtime/produce.py
A     tensor_dslab/readout/charge/runtime/validate.py
A     tensor_dslab/readout/charge/runtime/effects/__init__.py
A     tensor_dslab/readout/charge/runtime/effects/counts.py
A     tensor_dslab/readout/charge/runtime/effects/delays.py
A     tensor_dslab/readout/charge/runtime/effects/dark_counts.py
A     tensor_dslab/readout/charge/runtime/effects/timing_jitter.py
A     tensor_dslab/readout/charge/runtime/effects/correlated_avalanches.py
A     tensor_dslab/readout/charge/runtime/effects/smearing.py

M     tensor_dslab/readout/pure_waveform/config.py
M     tensor_dslab/readout/pure_waveform/field.py
D     tensor_dslab/readout/pure_waveform/_produce.py
A     tensor_dslab/readout/pure_waveform/runtime/__init__.py
A     tensor_dslab/readout/pure_waveform/runtime/prepare.py
A     tensor_dslab/readout/pure_waveform/runtime/produce.py
A     tensor_dslab/readout/pure_waveform/runtime/validate.py

M     tensor_dslab/readout/noise_waveform/config.py
M     tensor_dslab/readout/noise_waveform/field.py
D     tensor_dslab/readout/noise_waveform/_produce.py
A     tensor_dslab/readout/noise_waveform/runtime/__init__.py
A     tensor_dslab/readout/noise_waveform/runtime/prepare.py
A     tensor_dslab/readout/noise_waveform/runtime/produce.py
A     tensor_dslab/readout/noise_waveform/runtime/validate.py

M     tensor_dslab/readout/analog_waveform/config.py
M     tensor_dslab/readout/analog_waveform/field.py
D     tensor_dslab/readout/analog_waveform/_produce.py
A     tensor_dslab/readout/analog_waveform/runtime/__init__.py
A     tensor_dslab/readout/analog_waveform/runtime/prepare.py
A     tensor_dslab/readout/analog_waveform/runtime/produce.py
A     tensor_dslab/readout/analog_waveform/runtime/validate.py

M     tensor_dslab/readout/digitized_waveform/config.py
M     tensor_dslab/readout/digitized_waveform/field.py
D     tensor_dslab/readout/digitized_waveform/_produce.py
A     tensor_dslab/readout/digitized_waveform/runtime/__init__.py
A     tensor_dslab/readout/digitized_waveform/runtime/prepare.py
A     tensor_dslab/readout/digitized_waveform/runtime/produce.py
A     tensor_dslab/readout/digitized_waveform/runtime/validate.py
```

Changes to config and field modules are limited to shared-requirement import
and call-site identifier rewiring plus removal of deep validators from field
modules. Public class identities, signatures, dataclass fields, defaults,
equality, repr, and constructor semantics are protected. Mechanical use of a
renamed shared requirement inside an existing `_require()` or `__post_init__`
body is explicitly authorized; changing that requirement's behavior is not.

The following production paths are protected unchanged:

```text
tensor_dslab/__init__.py
tensor_dslab/common/**
tensor_dslab/readout/__init__.py
tensor_dslab/readout/collection.py
tensor_dslab/readout/analog_waveform/__init__.py
tensor_dslab/readout/charge/__init__.py
tensor_dslab/readout/digitized_waveform/__init__.py
tensor_dslab/readout/noise_waveform/__init__.py
tensor_dslab/readout/photoelectrons/__init__.py
tensor_dslab/readout/pure_waveform/__init__.py
pyproject.toml
```

If implementation requires another production path, stop and return the exact
need to Design before editing it.

## Test And Typing Scope

Implementation may update the existing tests whose private imports or patch
targets move:

```text
tests/test_charge_correlated_avalanches.py
tests/test_charge_count_orchestration.py
tests/test_charge_delay_preparation.py
tests/test_charge_product.py
tests/test_charge_timing_jitter.py
tests/test_deterministic_waveform_products.py
tests/test_noise_waveform_product.py
tests/test_package_contracts.py
tests/test_readout_product_types.py
tests/test_readout_simulation.py
tests/test_rng_ownership_migration.py
tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py
tests/typing/stage_4_deterministic_waveform_products.py
```

It may add exactly:

```text
tests/test_runtime_action_ownership.py
tests/typing/maintenance_4_runtime_action_ownership.py
```

Existing tests must retain their scientific and public-contract strength.
Mechanical import/patch rewrites must target the binding actually consumed by
the tested path. Do not delete an assertion merely because a private symbol
moved. `tests/readout_fixtures.py` and other tests remain protected unless an
exact finding shows a necessary in-scope import-only change and Design
authorizes it.

The implementation candidate may update only these lifecycle/evidence docs:

```text
docs/implementation/maintenance_4_runtime_action_ownership.md
docs/implementation/index.md
```

All other live architecture docs are synchronized in the Design authority and
remain protected during Implementation.

## Required Focused Evidence

Tests must prove at least:

### Tree, Imports, And Privacy

- every target runtime/effect module exists and every retired live private path
  is absent without a shim;
- runtime and effect `__init__.py` modules export/import nothing;
- existing product, readout, and package facade exports are byte-for-byte or
  object-identity unchanged as applicable;
- no Runtime/action/helper appears in a public `__all__`;
- every cross-module runtime/effect/requirement name follows the clean naming
  rule, while `_require()` and genuinely module-local helpers remain exempt;
- direct imports use exact defining modules rather than runtime facades;
- product runtime modules never import simulation, ReadoutConfig,
  ReadoutCollection, TensorML, TensorG4DS, DAG, or private TensorCore modules;
  and
- production uses absolute package imports.

### Runtime Records

- every listed Runtime record, including ReadoutRuntime and effect Runtimes, is
  final, frozen, slotted, non-inheriting, unexported, and exactly typed;
- Runtime records contain no Config, semantic product, collection, stored
  callable, mutable Python collection, or hidden cache; prepared tensor
  operands are explicitly permitted and remain read-only after preparation;
- production and validation leave every prepared Runtime tensor's values and
  storage identity unchanged;
- the requested retention set remains outside every Runtime record, including
  ReadoutRuntime;
- optional Runtime presence exactly matches the required closure;
- recognized request membership is obtained from
  `ReadoutCollection.accepted_field_types()` with no second product inventory;
- all temporal ProductRuntime values retain the exact same SamplingRuntime
  object where sampling is stored; and
- the simulation source sample dimension is discovered once per public call.

### Action Ownership

- each generated product has one exact `prepare_*`, `produce_*`, and
  `validate_*` action in its owning module;
- Photoelectrons has only `validate_photoelectrons`;
- product prepare modules own scientific/config preparation;
- produce modules import no Config and perform no product deep validator call;
- validate modules own deep scans and perform no repair or construction;
- `simulation.py` owns no scientific equation and uses the exact action
  sequence; and
- no generic runtime/action/product framework, registry, reflection, or
  string dispatch exists.

### Preflight And Failure Ordering

- every product Runtime in a requested closure is prepared before the first
  producer or RNG request;
- all Stage 7 public preflight failures remain before RNG, producer calls, and
  semantic-output writes;
- each generated validator is called exactly once with the exact produced
  field and required prepared fact;
- a forced Charge/Pure validator failure prevents Noise RNG and every later
  producer;
- a forced Noise validator failure prevents Analog and Digitized production;
- a forced Analog validator failure prevents Digitized production;
- a forced Digitized validator failure prevents collection construction;
- no failed call returns a field or partial collection; and
- Photoelectrons deep validation remains in whole-request preflight, including
  a truth-only request;
- a truth-only request above Charge's per-cell count ceiling remains accepted,
  while the same source fails Charge preparation before RNG or production;

### Validator Relationship Ownership

- focused direct-validator and public-orchestration mutation cases prove that
  each validator enforces the relationships it owns rather than merely being
  called;
- independently valid prerequisite/result fields with unequal ordered axes or
  unequal shapes are rejected by the responsible validator;
- a supported but wrong floating dtype relative to the named prerequisite or
  prepared Runtime is rejected, including the Charge and Noise Runtime dtype
  relationships;
- same-device relationships are proved on CPU and through conditional CUDA
  negative cases; the fixed Della gate must exercise the accepted CUDA
  relationship path;
- same-dtype result/input storage aliasing is rejected where the relationship
  permits construction of such a field, including at least PureWaveform and
  AnalogWaveform, without weakening the other products' freshness proofs;
- deep finite, nonnegative, range, and digitizer-code checks fail when their
  owning validator is bypassed or made a no-op; and
- TensorCore-constructor-owned intrinsic failures are not misrepresented as
  evidence for a TensorDSLab relationship check.

### Behavior And Numerical Continuity

- all 63 nonempty product subsets retain Stage 7 closure and exact-retention
  behavior;
- representative requests in different caller orders produce identical common
  products;
- every producer executes at most once;
- exact-zero/draw-free branches request no RNG values;
- representative Charge, white-noise, and PSD closures issue the exact same
  public TensorCore distribution calls, keys, positions, quanta, ordinals,
  counts, and order as the starting baseline;
- same-stack product tensor values remain exact against the starting baseline;
- Maintenance 3's environment-qualified stochastic-literal branching remains
  unchanged;
- TPC/Veto PureWaveform, analog saturation, ADC transfer, zero/white/PSD
  noise, dark, jitter, DiCT/DeCT/AP, and smearing reference tests remain
  unchanged in meaning;
- PureWaveform kernel construction occurs in preparation exactly once and
  production performs no coefficient flip/reshape or coefficient/kernel
  materialization;
- DigitizedWaveformRuntime contains no config and validation uses its prepared
  maximum code;
- Charge performs one deep terminal scan, preserves invalid-generated-result
  `RuntimeError`, and never passes an invalid result downstream; and
- every extracted helper is covered by product-level/reference behavior rather
  than only implementation-shape assertions.

### Results, Storage, And Autograd

- requested Photoelectrons remains the exact source object and source storage
  remains unchanged;
- generated products remain guaranteed fresh and pairwise storage-independent;
- exact source axes tuple and axis-object identity are reused;
- dtype, device, shape, and accepted stride/layout behavior remain unchanged;
- deterministic PureWaveform and AnalogWaveform autograd behavior and float64
  gradient checks remain accepted;
- stochastic and digitized products retain their existing autograd contract;
- no producer or validator moves, casts, detaches, or mutates an existing
  input outside the accepted fresh result conversion, and none bulk-
  materializes an input payload through `.cpu()`, `.numpy()`, `.tolist()`, or
  equivalent transfer; accepted scalar extraction for deep validation and
  dynamic correctness guards remains permitted and may synchronize the current
  device stream; and
- no write occurs through an alias after field construction.

### Duplicate-Logic Audit

Implementation must include in its handoff:

- the duplicated clusters inspected;
- the exact extractions made;
- the clusters intentionally left product-local and why their semantics differ;
- confirmation that no `utils.py`, `helpers.py`, ABC, registry, decorator
  framework, or untyped action mapping was introduced; and
- focused tests or source scans proving the final action symmetry and import
  boundaries.

## Full Verification Matrix

Implementation and Validation must test against both:

1. a clean exact TensorCore source checkout at
   `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`; and
2. an independently recreated canonical no-prefix Git archive with SHA-256
   `f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd`.

At minimum, from the TensorDSLab root:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:<tensorcore-source> \
  python -B -m unittest tests.test_runtime_action_ownership -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:<tensorcore-source> \
  python -B -m unittest tests.test_readout_simulation -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:<tensorcore-source> \
  python -B -m unittest tests.test_deterministic_waveform_products -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:<tensorcore-source> \
  python -B -m unittest tests.test_noise_waveform_product -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:<tensorcore-source> \
  python -B -m unittest tests.test_charge_product -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:<tensorcore-source> \
  python -B -m unittest discover -s tests -v
```

Repeat the focused and full suites against the independently recreated exact
archive. Record exact run/pass/fail/skip totals rather than assuming the
current count.

Run Pyright in standard mode against both dependency forms with the repository
Python target and record zero errors, warnings, and informations. Prove source
and archive import isolation in fresh processes, exact dependency identity,
public export identity, retired-path absence, forbidden imports, clean diff,
and absence of bytecode/cache/build/dist/egg-info artifacts.

Conditional local CUDA tests remain required. Because this maintenance moves
the production execution boundary, fixed-commit Validation and independent
Review must additionally run the focused runtime/readout suites and the full
package suite in separate fresh Della allocations on the accepted full A100
environment used by Maintenance 3, with zero skips. Each allocation must run
both the exact TensorCore source-checkout form and the independently recreated
canonical-archive form. Before and after execution it must prove the exact
TensorCore commit, parent, tree, version, canonical archive hash, frozen
Maintenance 3 runtime stack, and frozen source/clone inventory identities.
These are correctness and same-stack replay gates only: run no Stage 8
benchmark, profiler, threshold, kernel-count, memory, or performance
measurement. Implementation may obtain a fresh Della run as candidate evidence
but may not substitute it for Validation or Review's independent allocations.

If the accepted Della environment is unavailable or differs materially, stop
and return the exact environment evidence to Design. Do not weaken the gate to
conditional skips or reuse Maintenance 3 output.

## Documentation And Historical Records

This Design authority synchronizes the living package sources:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/overview.md
docs/design.md
docs/decisions.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/validation.md
docs/implementation/index.md
docs/implementation/maintenance_4_runtime_action_ownership.md
```

Closed Stage 2 through Stage 7, Maintenance 1 through 3, governance records,
and `docs/parity.md` remain unchanged historical evidence. Their old private
paths must not be rewritten as though those past candidates used the new tree.

Implementation may record candidate and Validation lifecycle evidence only in
this work order and the implementation index. Validation and Review are
report-only. After Review merges an unchanged cleared candidate, final Design
may perform an evidence-only documentation closeout over the same living docs
without changing cleared production, tests, metadata, dependency, governance,
or scientific bytes.

## Non-Goals And Forbidden Scope

Maintenance 4 does not authorize:

- `PureWaveformRenderer`, another `nn.Module`, public atomic product function,
  public preparer/producer/validator, or new public export;
- a Config, axis, field, collection, product graph, dtype, sampling, scientific
  model, parameter, default, or semantic change;
- a probability, delay, recovery, pulse, PSD, analog, ADC, smearing, count,
  ledger, overflow, or tolerance change;
- an RNG algorithm, key, address, position, quantum, ordinal, distribution
  mapping, word schedule, draw count/order, or no-draw change;
- a TensorCore edit, dependency move, private TensorCore import, or local
  generic RNG/distribution implementation;
- a Runtime/Action/Config/Product ABC, protocol, registry, reflection system,
  generic graph, plugin, decorator framework, or untyped mapping;
- a compatibility alias, wrapper, shim, re-export, or deprecation window for
  old private paths or names;
- Photoelectrons production, source loading, PE binning, provenance mapping,
  TensorG4DS integration, TensorML integration, model/training code, or
  Reconstruction;
- IO, TensorArtifact/ReadoutArtifact, persistence, cache, serialization,
  migration, DAG operation, recipe, or executable;
- `out=`, destination collection, workspace, allocator, pool, lease, stream
  coordinator, scratch API, allocation-free claim, or storage reuse;
- compile, fusion, Triton, custom CUDA, optimization, benchmark, profiler,
  kernel-count, memory, throughput, or performance work;
- Stage 8 restart or use of its stopped authority/executable input;
- release, deployment, backward-compatibility, conformance, production
  readiness, or broad cross-package/device claim;
- Coordination, Profile B, routing registry/cache, council architecture, or
  another repository's work; or
- a push.

## Stop And Return-To-Design Conditions

Stop the affected work and return exact evidence if:

- the package baseline, Design authority, dependency, branch, route, allowlist,
  or worktree is stale, dirty, discrepant, or ambiguous;
- the complete public preflight boundary or immediate
  produce/validate/descendant order cannot be preserved;
- Runtime records must retain Configs or previous semantic products;
- a producer requires config interpretation, repeated scientific preparation,
  or a deep publication scan;
- one product cannot fit the explicit action shape without a generic framework
  or semantic duplication;
- duplicate extraction would change operation ordering, numerical results,
  error category, RNG behavior, or autograd;
- the PureWaveform kernel move changes accepted outputs or gradients;
- the Charge single-scan boundary cannot preserve public invalid-terminal
  behavior;
- a public export or compatibility shim appears necessary;
- a required change lies outside the production/test/doc allowlist;
- an accepted closed test fails for a substantive rather than stale-private-
  import reason;
- TensorCore's public `0.9.0` API is insufficient for an accepted consumer;
- local or Della evidence contradicts an accepted result contract;
- the finite correction budget is exhausted or a finding repeats without
  convergence; or
- completion requires any forbidden scientific, public API, dependency,
  integration, persistence, optimization, governance, or cross-package work.

Do not patch around a stop with weakened tests, changed goldens, tolerances,
skips, expected failures, aliases, dummy RNG requests, fallback algorithms,
silent casts/moves, or unrecorded scope expansion.

## Dispatch And Finite Role Loop

Design may dispatch only the exact clean committed authority after privately
reverifying the persistent package-owned Implementation, Validation, and
Review routes as Active, workspace-correct, role-correct, and return-capable.
The user has authorized this Maintenance 4 scope, but user authorization does
not substitute for a clean authority or verified routes.

The workflow is:

```text
Design authority
  -> Implementation Candidate 1
  -> fixed-commit Validation
  -> bounded Implementation correction when required
  -> fixed-commit Validation recheck
  -> independent fixed-commit Review
  -> Design-authorized bounded correction if Review returns a finding
  -> Validation recheck
  -> Review recheck
  -> Review clean fast-forward and post-merge verification
  -> final Design evidence closeout
```

The ordinary Implementation-to-Validation budget is three fixed candidates.
Validation and Review do not edit candidate bytes. Review must not contact
Implementation directly for a correction or merge an uncleared candidate.
The first independent Review may return all in-scope findings to Design once.
Design may then authorize at most one exact direct-child supplemental
Implementation candidate covering that fixed finding packet; the supplemental
candidate receives one fresh fixed-commit Validation recheck and one Review
recheck. Any residual, repeated, or newly material finding after that recheck,
or any need beyond the fixed packet, returns the maintenance to Design revision
and requires a new authority. No exceptional extra candidate is implicit.

Only Review may run `git merge --ff-only` after clearing the exact
Validation-cleared bytes and proving a clean, unambiguous main ancestry. No
role rebases, squashes, amends, force-updates, or pushes.

Lifecycle vocabulary is:

```text
Design-complete / User-authorized / Undispatched
Dispatched / Active
Implementation candidate / Validation pending
Returned / Implementation correction
Validation-cleared / Review pending
Review-cleared / Merge authorized
Merged / Design acceptance pending
Merged / Closed
Returned to Design
Blocked
```

## Merge And Final Closeout

Independent Review may authorize and perform a clean fast-forward only after:

- Validation clears one exact fixed candidate;
- Review independently finds no unresolved issue on those exact bytes;
- the candidate is a clean linear descendant of the Design authority;
- every changed path is allowlisted and every protected path is unchanged;
- public outputs and stochastic call records match the exact starting behavior
  at the accepted comparison boundaries;
- focused/full source/archive, typing, import/export, privacy, forbidden-scan,
  hygiene, and independent real-CUDA gates pass; and
- main is clean and an unambiguous ancestor.

Review repeats the required post-merge package checks on unchanged main and
reports the exact resulting commit. State then becomes
`Merged / Design acceptance pending`.

Final Design independently checks topology, exact dependency, candidate diff,
public-export immutability, target tree, Runtime/action ownership, preflight,
validation order, numerical/RNG continuity, storage/autograd results,
source/archive and real-CUDA evidence, typing, docs, and qualifications. If
accepted, Design records `Merged / Closed` through a documentation-only
direct-child closeout that changes no cleared production or test byte.

## Completion Boundary

Maintenance 4 is complete only when:

- the accepted runtime-action tree is implemented without old-path shims;
- public facades and `simulate_readout(...)` are unchanged;
- Config -> Runtime -> Product -> validation ownership is explicit for every
  generated product, while Photoelectrons remains field -> ingress validation;
- complete preparation remains before RNG/production/write;
- every generated result is validated immediately before descendant use;
- product producers contain no Config interpretation or deep publication scan;
- Runtime records contain no Configs, products, framework, or mutable cache;
- duplicate logic is reduced where semantics are genuinely identical and the
  explicit product action shape remains readable;
- exact same-stack behavior, RNG calls, result/storage/autograd contracts, and
  Maintenance 3 qualification are preserved;
- fixed-commit Validation and independent Review clear the same bytes,
  including separate real-CUDA evidence;
- Review fast-forwards and verifies unchanged main;
- final Design accepts and synchronizes the living docs; and
- the work order and implementation index record `Merged / Closed`.

Completion authorizes no renderer, Stage 8 restart, optimization, IO,
integration, compatibility, deployment, conformance, Coordination, Profile B,
or push action.
