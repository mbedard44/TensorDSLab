# Stage 7 Public Readout Orchestration Work Order

Status: **Implementation candidate / Validation pending** while these exact
implementation bytes are absent from `main`. If they are present unchanged on
`main`, Review's clean fast-forward has completed and the state is
**Merged / Design acceptance pending** until this work order and the
implementation index record **Merged / Closed**.

Stable work-order key:
`TensorDSLab/stage-7-public-readout-orchestration`.

This is the package-authoritative TensorDSLab Design work order for the first
public readout simulation operation. Its containing Design commit is exact
authority `254a624b39993c4dc0b9a2a832ebd07398ac5a24`. The user separately
authorized production, Design reverified the persistent Implementation,
Validation, and Review routes, and Implementation prepared Candidate 1 on
`codex/stage-7-public-readout-orchestration`. Validation returned Candidate 1
for four bounded test-proof corrections, and Implementation prepared the
direct-child Candidate 2. Candidate commits are named externally in their
fixed-commit handoffs because a commit cannot embed its own hash. No Validation
or Review clearance is claimed by these bytes.

## Objective

Implement the one ordinary collaborator-facing readout API over the already
accepted private product producers:

```python
def simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype = torch.float32,
) -> ReadoutCollection:
    ...
```

The stage must:

- consume and validate one explicit product request;
- compute its fixed typed prerequisite closure;
- prepare the complete closure before invoking a product producer;
- reject one `RngKey` assigned to distinct stochastic roles in that closure;
- execute every required product producer at most once;
- retain exactly the products requested by the caller;
- return one completed `ReadoutCollection`;
- return the exact source `Photoelectrons` object when requested; and
- publish `simulate_readout` deliberately from `tensor_dslab.readout` and the
  package root.

This stage composes accepted behavior. It changes no product meaning,
scientific equation, RNG address, configuration schema, axis contract, field
contract, collection contract, dependency version, or private result law.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`.

The exact clean package baseline is the Maintenance 2 Design closeout:

```text
repository:             TensorDSLab
reference:              main
commit:                 9cbf8af3692740cd8e0bfbd1734d7ea91d95806a
implementation parent:  89a188abe330c06aa0b54c27cd61ac32a4fe9f63
package version:        0.1.0
Python requirement:     >=3.11
```

Maintenance 2 is Merged / Closed. It installed exact TensorCore `0.9.0`, the
product-owned module split, config-owned stochastic keys, and the public
TensorCore RNG/distribution boundary. Its accepted eager-CPU evidence ran 157
tests: 148 passed and 9 conditional CUDA tests skipped; Pyright reported no
diagnostics against either exact dependency form. That evidence makes no GPU,
performance, release, deployment, conformance, or broad compatibility claim.

The selected dependency remains unchanged:

```text
repository:       https://github.com/mbedard44/TensorCore.git
reference:        origin/main
commit:           4708bf2ca063a1bcd37a30a342733b9e3dbe9f59
direct parent:    0e72f0e69cf9140b692d408e49a504cbdcb101b7
package version:  0.9.0
archive SHA-256:  f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd
```

The implementation branch selected by the later dispatch is:

```text
codex/stage-7-public-readout-orchestration
```

At dispatch, Design verified that the committed work-order authority was a
clean linear descendant of the package baseline; the dependency line still
pinned the exact commit above; the implementation branch was absent; and all
three execution routes were Active, current, return-capable, and bound to this
workspace and work-order key. Raw platform route identifiers remain private
and must not enter committed files.

Package governance remains:

```text
package_adoption_state: Adopted
conformance_finding: Not evaluated
coordination_status: Deferred
registry_storage_profile: Disabled
maintenance_2: Merged / Closed
stage_7: topology-dependent candidate state recorded above
```

Coordination is not an execution route. This work authorizes no push.

## Applicable Contracts And Source Precedence

Implementation, Validation, and Review must read and reconcile:

- `AGENTS.md` for role, routing, authority, and handoff rules;
- `CONTRIBUTING.md`, especially TensorCore Backbone, Product Semantics,
  Domain Organization, Target Domain Simulation Surface, Boundary-First
  Validation, Public Surface Discipline, Test Expectations, and Scope
  Discipline;
- [Rebuild Architecture](../architecture/rebuild.md), especially Product
  Requests, Scientific Configuration, Private Product Builders, Public
  Builder, RNG And Positional Repeatability, Functional/Memory/Lifetime
  Contract, Validation Strategy, and Remaining Design Gates;
- [Readout Architecture](../architecture/readout.md);
- [TensorCore Integration](../architecture/tensors.md);
- [Parity](../parity.md);
- [Validation](../validation.md);
- the closed Stage 3 through Stage 6 and Maintenance 2 work orders as exact
  historical implementation evidence; and
- TensorCore's public `0.9.0` API and RNG architecture at exact commit
  `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`.

This work order controls the Stage 7 production slice. The live architecture
documents control package meaning. Closed work orders control the exact bytes
and evidence they accepted, but do not expand Stage 7. If these sources appear
to conflict, stop the affected work and return the exact contradiction to
TensorDSLab Design rather than selecting one silently.

## Exact Public Surface

Stage 7 adds exactly one public callable:

```python
simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype = torch.float32,
) -> ReadoutCollection
```

It is defined in `tensor_dslab.readout.simulation` and re-exported with the
same object identity from:

```text
tensor_dslab.readout.simulate_readout
tensor_dslab.simulate_readout
```

Neither package root re-exports `TensorField`, `CounterRng`, `Threefry4x32`,
`RngKey`, private preparation records, request planners, producer functions,
or validation helpers. Callers import the generic RNG implementation from
`tensor_core` and TensorDSLab semantics from `tensor_dslab`.

There is no simultaneous `seed=` argument, overload, alias, compatibility
wrapper, public atomic transform, or second simulation function.

## Product Request And Typed Closure

`products` is a required keyword-only iterable. `simulate_readout` consumes it
exactly once into a tuple before performing request analysis. It then:

1. rejects an empty request;
2. requires every item to be an exact recognized product class;
3. rejects duplicates before converting membership to a set; and
4. ignores caller iteration order as a semantic input.

Accepted exact classes are the six types already owned by
`ReadoutCollection`:

```text
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
```

Instances, `TensorField`, another base class, foreign field classes, non-class
values, and duplicate recognized classes are rejected at the public boundary.
This is documented-input validation, not adversarial policing of unsupported
final-leaf subclassing, class mutation, constructor bypass, or private calls.

The dependency graph is fixed:

```text
Photoelectrons -> Charge -> PureWaveform
Photoelectrons axes/device/shape + SamplingConfig -> NoiseWaveform
PureWaveform + NoiseWaveform -> AnalogWaveform -> DigitizedWaveform
```

The planner uses ordinary typed local booleans, conceptually:

```python
need_digitized = DigitizedWaveform in requested
need_analog = AnalogWaveform in requested or need_digitized
need_pure = PureWaveform in requested or need_analog
need_noise = NoiseWaveform in requested or need_analog
need_charge = Charge in requested or need_pure
```

It must not add a public graph, registry, plugin system, field ID, string
product name, dependency constant, canonical public sequence, or generic
planner framework. `Photoelectrons` is always available as the source but is
retained only when explicitly requested.

## Required Configuration Closure

`config` must be exactly `ReadoutConfig`. Its required members follow only the
computed closure:

| Needed product | Required config |
| --- | --- |
| `Charge` | `config.charge` |
| `PureWaveform` | `config.pure_waveform` and the Charge prerequisite |
| `NoiseWaveform` | `config.noise_waveform` |
| `AnalogWaveform` | `config.analog_waveform`, Pure, Noise, and their prerequisites |
| `DigitizedWaveform` | `config.digitized_waveform`, Analog, and its prerequisites |

A truth-only request is valid with
`ReadoutConfig(sampling=accepted_sampling)`. An absent required config raises
`ValueError` before producer invocation. Configs outside the effective closure
are not scientifically prepared, do not contribute stochastic roles, and may
not alter a common product or cause a request failure through a contextual
relationship that the request does not use.

Constructors continue to own intrinsic config validity. Stage 7 validates
closure-specific relationships such as sampling agreement, dtype
representability, prepared probability laws, allocation/address bounds,
analog saturation, and ADC transfer constants.

## Whole-Request Preparation

The accepted public sequence is:

```text
consume and validate request
  -> derive typed closure
  -> prepare source and every required product
  -> validate closure-wide stochastic keys
  -> invoke required producers in fixed topological order
  -> retain exactly requested fields
  -> construct one ReadoutCollection
```

No required product producer may be invoked until every deterministic
request-, source-, configuration-, sampling-, dtype-, and product-preparation
check in the complete closure has succeeded. This prevents, for example, a
DigitizedWaveform request from consuming Charge RNG values before discovering
an invalid PSD, saturation, pulse, or digitizer relationship.

`tensor_dslab.readout.simulation` owns one private immutable `_ReadoutPlan` or
an equivalently precise private typed record. It contains only the requested
membership, closure booleans, and optional product-owned prepared plans needed
to execute that one call. It is runtime-only and must not become a public
class, TensorCore semantic record, scientific config, collection member,
durable artifact, dependency registry, workspace, or cache value.

Each product `_produce.py` owns its own private immutable preparation record
and `_prepare_*` helper where nontrivial contextual preparation is required.
The cross-product plan composes those values; it does not duplicate their
scientific equations in `simulation.py`.

The narrow target shape is:

```python
@dataclass(frozen=True, slots=True)
class _ReadoutPlan:
    requested: frozenset[type[TensorField]]
    need_charge: bool
    need_pure: bool
    need_noise: bool
    need_analog: bool
    need_digitized: bool
    charge: _ChargePlan | None
    pure: _PureWaveformPlan | None
    noise: _NoiseWaveformPlan | None
    analog: _AnalogWaveformPlan | None
    digitized: _DigitizedWaveformPlan | None
```

Field names may be adjusted narrowly for clarity, but Implementation must not
replace this with untyped dictionaries, `Any`, public plans, or a generic
framework. Exact private plan classes remain implementation details and are
not re-exported.

Product-owned preparation refactors the current private surface into typed
prepare/produce pairs, conceptually:

```python
_prepare_charge(..., config: ChargeConfig, ...) -> _ChargePlan
_produce_charge(photoelectrons, *, plan: _ChargePlan, rng: CounterRng) -> Charge

_prepare_pure_waveform(..., config: PureWaveformConfig, ...) -> _PureWaveformPlan
_produce_pure_waveform(charge, *, plan: _PureWaveformPlan) -> PureWaveform

_prepare_noise_waveform(..., config: NoiseWaveformConfig, ...) -> _NoiseWaveformPlan
_produce_noise_waveform(
    photoelectrons,
    *,
    plan: _NoiseWaveformPlan,
    rng: CounterRng,
) -> NoiseWaveform

_prepare_analog_waveform(..., config: AnalogWaveformConfig, ...) -> _AnalogWaveformPlan
_produce_analog_waveform(pure, noise, *, plan: _AnalogWaveformPlan) -> AnalogWaveform

_prepare_digitized_waveform(
    ...,
    config: DigitizedWaveformConfig,
) -> _DigitizedWaveformPlan
_produce_digitized_waveform(
    analog,
    *,
    plan: _DigitizedWaveformPlan,
) -> DigitizedWaveform
```

Exact arguments hidden by the ellipses are the already accepted source,
sampling, device, shape, and floating-dtype facts required by that product.
Every scientific config still enters through an exact typed parameter on its
owning product preparer. The prepared plan is the trusted execution input, so
the producer does not receive a second config that could disagree with it.
This is a private behavior-preserving signature change. It keeps `_produce_*`
as the sole product constructor and avoids re-evaluating or duplicating
scientific equations after execution begins.

Preparation may create ephemeral unexposed scalar/control tensors and may run
read-only scalar reductions required by accepted deep validation. Therefore,
the precise failure guarantee is:

> Every supported request-level or statically preparable relationship failure
> occurs before the first RNG raw-word request, product-producer invocation,
> or semantic-product/output write.

It is not a promise that preflight performs no allocation, creates no tiny
temporary tensor, or issues no read-only device operation.

## Product-Local Preparation Ownership

The product split must preserve these ownership rules:

- Charge preparation remains in `readout/charge/_produce.py` and existing
  focused effect modules. It owns source count/value checks, count/output
  allocation arithmetic, sample-dimension resolution, dark mean, optional
  timing plan, correlated-avalanche plan and ledger envelope, and optional
  smearing envelope. Charge effects consume the accepted prepared values
  rather than recomputing them after another product could already have run.
- Pure-waveform preparation remains in
  `readout/pure_waveform/_produce.py`. It owns binary64 sampling conversion,
  pulse template extent, exact model equation, normalization, and coefficients
  rounded for the selected floating dtype.
- Noise preparation remains in `readout/noise_waveform/_produce.py`. It owns
  shape/address arithmetic, white-RMS representation, PSD integration,
  coefficient geometry, and sample-dimension resolution.
- Analog preparation remains in `readout/analog_waveform/_produce.py`. It owns
  selected-dtype saturation bounds and their noncollapse/order checks.
- Digitization preparation remains in
  `readout/digitized_waveform/_produce.py`. It owns maximum code, gain, span,
  slope, intercept, pre-gain thresholds, selected-dtype representation, and
  endpoint/noncollapse checks. Its plan retains the exact immutable
  `DigitizedWaveformConfig` for the existing config-dependent deep
  postcondition.

`simulation.py` may validate cross-product facts derivable from the source and
closure, including exact source axes/device, common output floating dtype, and
which configs/plans are required. It must not copy pulse equations, PSD
integration, delay preparation, RNG distributions, analog transfer, ADC
transfer, count arithmetic, or charge-ledger rules.

Private producers and scientific helpers remain independently testable, but
their paths and signatures are not supported public API. No compatibility shim
for their pre-Stage-7 signatures is allowed.

## RNG Contract And Closure-Wide Key Uniqueness

`rng` is required for every request and must satisfy
`isinstance(rng, CounterRng)`. The public abstraction remains open to a
conforming TensorCore `CounterRng`; Stage 7 must not require exact
`Threefry4x32` merely because it is the ordinary supplied implementation.

TensorCore `0.9.0` exposes no non-consuming
`supports(device, dtype, distribution)` query. Stage 7 therefore must not:

- issue a dummy uniform, Gaussian, Poisson, binomial, or raw-word request;
- inspect protected algorithm hooks;
- infer support from the concrete RNG class name;
- catch a real failure and retry through another algorithm; or
- narrow the public signature silently to `Threefry4x32`.

Preparation validates TensorDSLab-owned device, dtype, shape, address, law,
and config relationships plus nominal `CounterRng` membership. A deterministic
closure never queries the RNG for a value. For a stochastic closure, the first
real distribution call is where TensorCore validates any concrete custom
algorithm/backend behavior not expressible through the public ABC. Such a
failure is an execution-time failure, not a preflight failure, and carries no
rollback guarantee. Unsupported or adversarial custom implementations outside
the TensorCore public contract have no stable TensorDSLab error semantics.

Before producer invocation, Stage 7 collects the exact `RngKey` values for
every distinct stochastic role structurally present in the effective closure.
The collision set includes:

- `WhiteNoiseConfig.rng_key` or `PsdNoiseConfig.rng_key` when that selected
  noise model is required;
- `DarkCountConfig.rng_key` when `dark_count` is present, even at zero rate;
- both `DirectCrosstalkConfig` keys when that config is present, even at zero
  offspring mean or `maximum_generations == 0`;
- both `DelayedCrosstalkConfig` keys under the same structural rule;
- `TimingJitterConfig.rng_key` when present, even at zero sigma;
- `AfterpulseConfig.rng_key` when present, even at zero probability or
  `maximum_generations == 0`; and
- `ChargeSmearingConfig.rng_key` when present, even at zero relative sigma.

It excludes absent configs, `ZeroNoiseConfig`, and every key-bearing config in
an unrequested branch. Two distinct roles with equal keys raise `ValueError`
before any producer invocation or RNG request. Stage 7 never silently rekeys,
derives another key, or treats numeric no-op values as permission to reuse a
role identity.

Role collection is ordinary private typed branching. Do not introduce a
public key registry, enum, string-driven reflection system, or module-level
dependency map.

## Execution, Assembly, And Retention

After the complete plan is accepted, producer execution uses this fixed
topological order:

```text
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
```

The ordering is private orchestration, not collection semantics or a public
canonical product sequence. Every needed producer is invoked exactly once;
every unneeded producer is not invoked. A required prerequisite may remain a
private local value even when it is not requested.

Final assembly examines a fixed local tuple:

```text
Photoelectrons, Charge, PureWaveform, NoiseWaveform,
AnalogWaveform, DigitizedWaveform
```

and retains exactly values whose exact types occur in the request. Equivalent
request sets therefore receive the same mechanical collection mapping order,
but callers must continue to use exact-type lookup rather than relying on that
order as dependency, provenance, persistence, or model schema.

The builder constructs `ReadoutCollection` exactly once, after every requested
product has succeeded. No partial collection is returned. Requesting
`Photoelectrons` returns the exact input object; Stage 7 never constructs a
replacement truth field.

Changing only request order or adding an unrelated retained product must not
change the value of a product common to two calls made with the same accepted
source, config, RNG, dtype, backend, and execution mode. Unrequested
prerequisites do not become collection members.

## Source, Sampling, Dtype, And Device Boundary

The public boundary requires:

- `photoelectrons` is exactly `Photoelectrons`;
- its payload passes the explicit nonnegative deep-value check;
- it has exactly one `ExampleAxis`, `ChannelAxis`, and `SampleAxis`, with shape
  agreement and `torch.strided` storage under its existing leaf contract;
- `config.sampling` is exactly `SamplingConfig` and agrees with the source
  sample-axis count, zero start, and period;
- the source device type is exactly `cpu` or `cuda`; and
- generated products remain on that exact source device.

The CPU/CUDA restriction applies even to a truth-only
`simulate_readout(...)` request so the public operation has one predictable
device boundary. Callers may still construct valid semantic records directly
on any device permitted by the lower-level representation, but Stage 7 makes
no simulation claim for MPS, XLA, meta, private-use, or another backend.

`floating_dtype` is validated as exactly `torch.float32` or `torch.float64`
only when the closure generates a floating product. A truth-only request does
not consume or validate it as an execution control. A digitized-only request
does validate it because Charge, Pure, Noise, and Analog remain required
floating prerequisites.

The builder never moves, detaches, calls `.cpu()` or `.numpy()` on, serializes,
reloads, reconstructs, or replaces the source payload. It does not normalize
an existing semantic input by silently moving or casting it. This restriction
does not prohibit a producer from allocating its declared generated-product
dtype: Charge production necessarily performs the fresh semantic conversion
from `Photoelectrons[torch.int64]` to the selected floating dtype. Generated
products reuse the exact source axes tuple and exact immutable axis instances.

## Result, Storage, Autograd, And Exposure Contract

Stage 7 preserves the operation-owned result taxonomy already accepted for
each producer:

- requested `Photoelectrons` is an exact return of the named source object;
- every generated field has guaranteed-fresh storage independent of every
  named input to its producer;
- generated fields retained together are pairwise storage-independent;
- no public path returns a guaranteed-sharing or sharing-unspecified result;
  and
- private scratch and unrequested intermediates never enter the collection.

Source payload and global PyTorch RNG state remain unchanged. TensorDSLab
initiates or enqueues every write before constructing and exposing its
semantic product and initiates no later write through an alias.

Pure and Analog preserve their accepted functional autograd paths. Noise has
no differentiable source lineage; Charge stochastic/count behavior and
DigitizedWaveform make only their existing nondifferentiable claims. Stage 7
must not detach an accepted deterministic path merely because it orchestrates
multiple products.

### Generated-Product Deep Postconditions

Stage 7 closes the result trust boundary deliberately deferred by Stage 4.
Every generated producer must complete its payload, construct its one local
semantic field, and call that product's existing private
`_require_valid_values(...)` exactly once before returning the field to
orchestration. The accepted checks are:

- `Charge`: finite and nonnegative;
- `PureWaveform`, `NoiseWaveform`, and `AnalogWaveform`: finite; and
- `DigitizedWaveform`: nonnegative and no greater than the maximum code from
  its exact `DigitizedWaveformConfig`.

`_DigitizedWaveformPlan` therefore retains the exact immutable digitizer config
needed by the existing config-dependent validator, in addition to its prepared
transfer constants. No field helper or constructor contract changes. Intrinsic
leaf construction remains cheap; this explicit producer-return boundary owns
the full-device scan.

A produced field is not available to a downstream producer, retention, or
collection assembly until its deep postcondition succeeds. Failure may occur
after payload work and local field construction, so it is a dynamic execution
failure: the invalid field does not escape, no downstream producer is invoked,
and no partial collection is returned. Private allocations have no rollback
promise. These scans use scalar extraction and may synchronize CUDA.

Deep `Photoelectrons` ingress validation and existing scientific
postconditions use scalar reductions such as `.item()`. On CUDA these may
synchronize the current device as an accepted functionality-first correctness
cost. The package still performs no silent payload transfer or host staging.
Ordinary producer kernels enqueue on the current PyTorch stream. After the
accepted validation/postcondition synchronizations, same-stream consumers use
ordinary ordering and cross-stream consumers establish their own dependency.
Stage 7 makes no asynchronous-return, zero-synchronization, stream-lease, or
performance promise.

## Failure Effects

Supported request and statically preparable relationship failures occur before
the first RNG raw-word request, product-producer invocation, or semantic-output
write. The source, config, RNG value, and global RNG state remain unchanged.

After producer execution begins, ordinary backend allocation failures,
kernel failures, dynamically realized count/rate/ledger failures, and a real
custom `CounterRng` backend failure may occur. There is no rollback guarantee
for private work already performed. The source remains logically read-only,
no partial `ReadoutCollection` is returned, and no failed private field or
scratch is exposed through the public API.

Error classification follows existing package boundaries:

- malformed documented input types use `TypeError`;
- well-formed but unsatisfied request/config/device/dtype/relationship inputs
  use `ValueError`; and
- dynamic invariant or execution failures use `RuntimeError` where the owning
  producer already defines that contract.

Exact message text is tested only where this work order names it as a
diagnostic requirement. Unsupported final-leaf subclassing, class mutation,
constructor bypass, direct private calls, exposed-tensor mutation, and custom
Torch dispatch behavior remain outside the stable error surface.

## Package Ownership And Import Direction

Stage 7 materializes the already accepted package layer:

```text
product configs/fields and private producers
  -> readout.config / readout.collection
  -> readout.simulation
  -> readout package exports
  -> tensor_dslab package exports
```

`simulation.py` may import the six exact product classes, five private
product preparation/producer modules, `ReadoutConfig`, `ReadoutCollection`,
`CounterRng`, and `TensorField`. Product packages must not import
`ReadoutConfig`, `ReadoutCollection`, or `simulate_readout` and must remain
acyclic.

No new `api.py`, planner package, global builder module, dependency registry,
`types.py`, `validation.py`, `random.py`, `_rng.py`, or placeholder module is
allowed. The public function lives in `simulation.py` because it performs the
one readout-domain simulation action; package exports make its physical path
unimportant to ordinary users.

## Exact Candidate Allowlist

Implementation may add or modify only the following production paths unless
Design explicitly revises this work order:

```text
A tensor_dslab/readout/simulation.py
M tensor_dslab/readout/__init__.py
M tensor_dslab/__init__.py
M tensor_dslab/readout/_requirements.py
M tensor_dslab/readout/charge/_produce.py
M tensor_dslab/readout/charge/effects/_dark_counts.py
M tensor_dslab/readout/charge/effects/_timing_jitter.py
M tensor_dslab/readout/charge/effects/_correlated_avalanches.py
M tensor_dslab/readout/charge/effects/_smearing.py
M tensor_dslab/readout/pure_waveform/_produce.py
M tensor_dslab/readout/noise_waveform/_produce.py
M tensor_dslab/readout/analog_waveform/_produce.py
M tensor_dslab/readout/digitized_waveform/_produce.py
```

An allowlisted file may change only for whole-request preparation,
prepared-value consumption, orchestration, exports, or an exact regression
required by those changes. Scientific laws, default keys, raw-word schedules,
product equations, public configs, fields, axes, collection behavior, and
TensorCore mechanics are protected even when housed in an allowlisted file.

The expected test scope is:

```text
A tests/test_readout_simulation.py
A tests/typing/stage_7_public_readout_orchestration.py
M tests/readout_fixtures.py
M tests/test_package_contracts.py
M tests/test_charge_product.py
M tests/test_charge_timing_jitter.py
M tests/test_charge_correlated_avalanches.py
M tests/test_noise_waveform_product.py
M tests/test_deterministic_waveform_products.py
M tests/test_rng_ownership_migration.py
M tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py
M tests/typing/stage_4_deterministic_waveform_products.py
```

Existing tests should change only when a private preparation/producer
signature moves under this work order. Do not mechanically rewrite unrelated
fixtures or weaken closed assertions.

The stage record and implementation index may receive lifecycle/evidence
updates through the authorized roles. README and the live architecture,
decision, validation, parity, workflow, and contribution documents remain
Design-owned: the final Design closeout synchronizes them after unchanged
Review-cleared production is merged. Closed work orders, governance records,
package metadata, configs, fields, collection code, axes, sampling, and the
TensorCore pin are protected.

## Required Behavioral Evidence

### Request And Closure Matrix

Tests must cover all 63 nonempty subsets of the six accepted products with a
small deterministic or recorded fixture. For every subset they prove:

- exact prerequisite booleans;
- every needed producer invoked once and every unneeded producer omitted;
- exact returned `field_types`;
- absence of unrequested prerequisites;
- fixed topological producer order; and
- fixed mechanical collection assembly independent of caller request order.

At least one one-shot iterable must raise if iterated twice. Separate fixtures
must reject the empty request, duplicate types, instances, `TensorField`, base
classes, foreign classes, and non-class values before any producer invocation
or RNG request. Tests must not create unsupported subclasses of final semantic
leaves merely to harden behavior outside the public contract.

Every transitive missing-config edge must be exercised. Representative
irrelevant product configs—including contextually invalid but intrinsically
constructible values—must be proven unconsumed when their branches are not in
the closure.

### Complete Preparation And Failure Ordering

Focused spies must prove that a late product-preparation failure invokes no
earlier product producer and requests no RNG word. Required cases include:

- invalid source deep values;
- sampling/source-axis disagreement;
- invalid floating dtype on a floating closure;
- Charge contextual preparation failure;
- pulse-template preparation failure;
- white-noise or PSD preparation failure;
- analog saturation collapse in the selected dtype; and
- digitizer transfer failure in the selected dtype.

Preparation helpers must be called only for required products. The tests must
show that no product equation or stochastic sampler is executed merely to
probe support. The public boundary may create ephemeral private validation
temporaries and perform read-only scalar reductions; it must not construct a
semantic output field before complete preparation succeeds.

Separate product-return tests must prove that each generated producer invokes
its exact deep validator once before returning, that representative successful
outputs satisfy the named domain, and that a forced Pure, Noise, Analog, or
Digitized postcondition failure invokes no downstream producer and exposes no
field or partial collection. Existing Charge postcondition coverage remains the
reference pattern. Removing or bypassing any one postcondition must make the
focused evidence fail.

### RNG And Key Evidence

Use focused recording/failing `CounterRng` implementations through the public
TensorCore ABC contract to prove:

- every request requires a `CounterRng` instance;
- truth-only and other deterministic closures request zero words;
- a conforming non-`Threefry4x32` implementation is not rejected by concrete
  class identity;
- Stage 7 performs no dummy/capability draw;
- the same key on distinct active roles fails before producers or words;
- structurally present numeric no-op dark, jitter, crosstalk, afterpulse, and
  smearing roles participate in collision checks;
- `maximum_generations == 0` does not remove structurally present correlated
  roles from the collision set;
- `ZeroNoiseConfig`, absent configs, and unrequested branches contribute no
  key; and
- distinct keys preserve the exact closed Maintenance 2 product outputs on the
  same eager CPU backend/mode.

Do not test unsupported `RngKey` subclassing or protected TensorCore mechanics.
Do not duplicate TensorCore random known-answer, distribution-algorithm, or
exhaustion suites except for narrow TensorDSLab consumer probes required by
this boundary.

### Composition And Retention Invariance

Representative real end-to-end calls must cover:

- each single-product request;
- `Photoelectrons` beside a derived product;
- the two independent branches `PureWaveform` and `NoiseWaveform`;
- `AnalogWaveform` with its private prerequisites;
- `DigitizedWaveform` with `AnalogWaveform` retained and unretained; and
- a complete six-product request.

For a fixed accepted source/config/RNG/dtype/backend/mode, common product
values must be exactly equal when only request order or unrelated retention
changes. Producer-call records prove each common prerequisite is computed once
per invocation rather than recomputed for retention.

### Field, Storage, Axis, And Source Evidence

Tests must prove:

- requested `Photoelectrons` is the exact input object;
- the source tensor and axes remain unchanged on success and every preflight
  failure;
- every generated field has fresh storage independent of its named inputs;
- all simultaneously retained generated fields are pairwise storage-
  independent;
- every generated field has passed its exact product-specific deep-value
  postcondition before downstream use or retention;
- every generated field reuses the exact source axes tuple and axis objects;
- arbitrary accepted semantic axis order is preserved;
- an accepted noncontiguous source is not silently normalized, moved, or
  rewritten;
- output device and exact dtype follow the accepted contracts;
- truth-only ignores `floating_dtype` while a digitized-only request validates
  it for floating prerequisites;
- source devices outside CPU/CUDA fail before producer invocation; and
- no write is initiated through an output alias after field construction.

Storage checks must reason about actual overlapping storage ranges, not merely
Python object identity or unequal `data_ptr()` values when views could overlap.

### Autograd, Global State, And Boundary Exclusions

Representative composition tests must preserve accepted Pure/Analog autograd
behavior and the existing nondifferentiable/stochastic boundaries. They must
also prove:

- PyTorch global RNG state is unchanged;
- no `torch.Generator` is constructed;
- no source device movement, in-place/source replacement, input normalization,
  `.cpu()`, `.numpy()`, or serialization/reload occurs; declared fresh
  generated-product dtype conversion remains required, including
  `Photoelectrons[torch.int64]` to floating Charge;
- no filesystem, cache, persistence, network, DAG, TensorG4DS, TensorML, or
  Reconstruction surface is invoked; and
- no partial `ReadoutCollection` escapes on execution failure.

The no-host-staging check does not prohibit explicit scalar `.item()`
reductions used by accepted deep validation and postconditions.

## Public Imports And Static Typing

Fresh-process and static import-direction checks must prove:

- `tensor_dslab.simulate_readout is
  tensor_dslab.readout.simulate_readout`;
- `tensor_dslab.readout.simulation` imports without loading TensorG4DS,
  TensorML, Projects/dag, IO, persistence, or retired RNG modules;
- static import-direction checks prove that no product module imports
  `readout.simulation`, `ReadoutConfig`, or `ReadoutCollection`; ordinary eager
  parent-package initialization is not misclassified as a product-to-simulation
  dependency;
- `__all__` includes the one new public name exactly once at both TensorDSLab
  export layers;
- private plans/preparers/producers are absent from public `__all__` values;
- TensorCore generic RNG/types remain absent from TensorDSLab root exports;
- retired `readout._random`, `_RngStream`, `_rng`, `_product.py`, and
  `_product_*` surfaces remain absent; and
- the package import graph remains acyclic.

The two deliberate package-root exports may use the repository's ordinary eager
import pattern, so importing a nested product through Python's package machinery
may initialize the parent root and load `readout.simulation`. Stage 7 requires
static inward-dependency isolation, not a lazy `__getattr__`, `TYPE_CHECKING`
facade, or impossible `sys.modules` absence after parent-package initialization.

The Stage 7 Pyright fixture must establish at least:

```python
readout: ReadoutCollection = simulate_readout(
    photoelectrons,
    products=[AnalogWaveform, DigitizedWaveform],
    config=config,
    rng=rng,
)
analog: AnalogWaveform = readout.field(AnalogWaveform)
digitized: DigitizedWaveform = readout.field(DigitizedWaveform)
```

It must also prove keyword-only `products`, `config`, `rng`, and
`floating_dtype`; no `seed=`; and an iterable-of-product-classes input. Static
typing need not encode every runtime-accepted exact product class through a
new public union alias or registry.

## Conditional CUDA Evidence

CPU is required. CUDA tests are conditional and must use the same public
contracts when a CUDA device is available. They should cover at least:

- one deterministic multi-product closure;
- one stochastic Charge or noise closure;
- one closure-wide key rejection before any RNG request, producer invocation,
  or semantic-output write;
- exact device and axes preservation;
- source identity/immutability and generated storage independence; and
- the accepted deep-validation synchronization boundary without asserting a
  zero-synchronization or performance result.

Skipped CUDA tests are qualifications, not GPU evidence. Stage 7 makes no
throughput, fusion, kernel-count, memory-peak, allocation-free, cross-backend
bitwise, or broad accelerator claim.

## Verification Environment And Commands

Implementation must record exact Python, PyTorch, OS/architecture, available
devices, CUDA runtime/device when present, Pyright, package metadata, and
TensorCore source/archive identities.

At minimum, run from the TensorDSLab project root against both a clean exact
TensorCore source checkout and an independently created exact archive:

```bash
git diff --check
PYTHONPATH=. python -m unittest tests.test_readout_simulation -v
PYTHONPATH=. python -m unittest discover -s tests -v
pyright --project pyrightconfig.json
```

Use the exact Python/Pyright commands available in the accepted environment
and record them verbatim. If Pyright, CUDA, archive tooling, build tooling, or
another named verifier is unavailable, report the absence as a qualification;
do not substitute a claim.

Additional required gates include:

- exact dependency-line and archive-hash verification;
- package-root export and fresh-process import probes;
- source/archive import isolation;
- allowlist and protected-path comparison from the Design authority;
- forbidden import/call and retired-surface searches;
- no raw route identifiers in committed files;
- no bytecode, cache, build, distribution, or egg-info artifacts; and
- final worktree cleanliness.

The existing full source/archive suite is the regression floor. Test totals may
increase, but no closed test may be removed, skipped unconditionally, or
weakened merely to make Stage 7 pass.

## Implementation Candidate 1 Evidence

Implementation completed the dispatched Stage 7 scope on the exact branch and
Design authority recorded above. Candidate 1 adds the one public
`simulate_readout(...)` orchestration surface, composes the five product-owned
typed preparation plans, preserves the accepted private scientific execution,
and adds the required request, failure-ordering, key-collision, continuity,
storage, autograd, import, typing, and conditional-CUDA evidence. Its exact
commit is identified by the fixed-commit Validation handoff; this record does
not claim Validation or Review clearance.

Implementation independently recreated TensorCore `0.9.0` as a clean detached
source checkout at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, with direct parent
`0e72f0e69cf9140b692d408e49a504cbdcb101b7`. A new exact Git ZIP reproduced
SHA-256
`f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd`.
The final Implementation evidence was:

```text
focused source suite:  28 run,  25 passed,  3 conditional CUDA skips
focused archive suite: 28 run,  25 passed,  3 conditional CUDA skips
full source suite:    186 run, 174 passed, 12 conditional CUDA skips
full archive suite:   186 run, 174 passed, 12 conditional CUDA skips
Pyright source:         0 errors, 0 warnings, 0 informations
Pyright archive:        0 errors, 0 warnings, 0 informations
import isolation:       False False False False in both dependency forms
```

Pyright was exact version `1.1.411` in standard mode. The environment was
Python `3.13.11`, PyTorch `2.12.1`, macOS `15.7.4` on arm64, and eager CPU.
`torch.version.cuda` was `None`, CUDA availability was `False`, and the CUDA
device count was zero. The three Stage 7 CUDA tests and nine retained
conditional CUDA tests therefore skipped; this is a CPU result and makes no
GPU execution or performance claim. Dependency identity, package metadata,
source/archive loading, public exports, fresh-process isolation, retired and
forbidden surfaces, import direction, exact changed-path allowlist, protected
bytes, diff hygiene, privacy, and artifact gates passed. No push occurred.

## Implementation Candidate 2 Evidence

Fixed-commit Validation returned Candidate 1 without a production,
architecture, TensorCore, dependency, documentation, or scope finding. It
identified four required proof gaps in `tests/test_readout_simulation.py`:
truth-only ingress isolation, `RngKey` value equality and absent-role
exclusion, exact no-probe stochastic call records, and exact generated-field
validator targets. Candidate 2 is Candidate 1's direct child and corrects
those proofs without changing production, public API, science, dependency, or
closed evidence. Its exact commit is identified by the fixed-commit Validation
handoff; these bytes remain Validation pending.

Candidate 2 independently reran the exact TensorCore source/archive gates. The
fixed evidence is:

```text
focused source suite:  30 run,  27 passed,  3 conditional CUDA skips
focused archive suite: 30 run,  27 passed,  3 conditional CUDA skips
full source suite:    188 run, 176 passed, 12 conditional CUDA skips
full archive suite:   188 run, 176 passed, 12 conditional CUDA skips
Pyright source:         0 errors, 0 warnings, 0 informations
Pyright archive:        0 errors, 0 warnings, 0 informations
import isolation:       False False False False in both dependency forms
```

The dependency commit, direct parent, archive SHA-256, environment, unavailable
CUDA/build-tool qualifications, no-effects, allowlist/protected bytes, import
direction, retired/forbidden surfaces, privacy, artifact, and no-push
dispositions remain exactly as recorded for Candidate 1.

## Stage Checkpoints

Implementation should deliver the stage in these reviewable checkpoints on one
linear feature branch:

1. private product-owned preparation records and behavior-preserving producer
   consumption;
2. complete `_ReadoutPlan`, request validation, closure and key preflight;
3. execute-once orchestration, exact retention, collection construction, and
   public exports;
4. complete behavioral, typing, import, source/archive, and conditional CUDA
   evidence; and
5. a fixed clean candidate handoff to Validation.

These are checkpoints, not separate public stages or permanent role threads.
Implementation may combine commits when the resulting candidate remains easy
to review. Every candidate must remain a linear descendant of the Design
authority.

## Dispatch And Finite Role Loop

The Design issuance of this document was not itself a dispatch. The later user
authorization and Design handoff named and reverified the exact committed
authority, dependency, branch, scope, routes, and work-order key before
Candidate 1 work began.

The accepted production workflow is:

```text
Design
  -> Implementation candidate
  -> fixed-commit Validation
  -> Implementation correction when required
  -> fixed-commit Validation recheck
  -> independent fixed-commit Review
  -> Implementation correction only with Design authorization
  -> Validation recheck
  -> Review recheck
  -> Review clean fast-forward and post-merge verification
  -> final Design evidence closeout
```

The ordinary Implementation-to-Validation correction budget is three fixed
candidates total. Validation reports findings to Design and Implementation but
does not edit production or tests. Review remains read-only and must not merge
an uncleared candidate. Review findings after the ordinary budget return to
Design; Design may authorize only a bounded correction that preserves this
work order, otherwise the stage returns to Design for revision.

Only Review may perform the clean `git merge --ff-only` after exact candidate
clearance and a clean unambiguous main baseline. No role rebases, squashes,
amends, force-updates, or pushes unless separately authorized.

The lifecycle vocabulary is:

```text
Design-complete / Undispatched
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

## Documentation Synchronization

At issuance, this Design authority synchronized the live package documents to
Maintenance 2 Merged / Closed and Stage 7 Design-complete / Undispatched.
Candidates 1 and 2 update only the lifecycle/evidence portions of this work
order and the implementation index; closed historical work orders and
governance records remain unchanged.

Implementation may record candidate and Validation evidence in this work order
and the implementation index when its dispatched handoff requires that
documentation. Validation and Review remain report-only. After Review merges
unchanged cleared production, final Design records Review/merge evidence and
synchronizes `README.md`, `AGENTS.md`,
`CONTRIBUTING.md`, and the live overview/design/decision/architecture/
validation/parity documents to the exact accepted Stage 7 state. That final
documentation-only closeout must not alter cleared production or tests.

Publishing the public requested-Charge boundary makes that comparison runnable;
it does not itself establish IV-DSLab statistical parity. `docs/parity.md`
must continue to distinguish TensorDSLab-model conformance from donor
equivalence and require named observable-specific margins.

## Non-Goals And Forbidden Scope

Stage 7 does not authorize:

- a scientific equation, probability law, delay/recovery model, pulse model,
  PSD law, analog equation, ADC transfer, RNG algorithm, key, address, word
  schedule, count bound, or ledger change;
- a TensorCore edit, dependency move, private TensorCore import, or local copy
  of generic RNG/distribution mechanics;
- a config, axis, product field, collection, constrained scalar, dtype, or
  semantic-coordinate change;
- Photoelectrons production, PE binning, source loading, provenance mapping,
  or a TensorG4DS dependency/adapter;
- IO, persistence, cache/artifact, digitizer-config association, migration,
  compatibility, serialization, or DAG behavior;
- TensorML, Reconstruction, training, model-schema, or downstream adapter
  work;
- public private-product transforms, a public plan/graph/registry, mutable
  collections, projection, reconstruction, invalidation, or partial results;
- `out=`, destination collections, workspace, allocator, pool, lease, stream
  coordinator, generation bank, scratch API, or allocation-free claim;
- compiler integration, fusion, custom Triton/CUDA kernels, kernel-count,
  memory, throughput, or performance claims;
- MPS/XLA/private-use support or broad CPU/CUDA compatibility claims;
- release, deployment, backward-compatibility, conformance, or production
  readiness claims;
- Coordination, Profile B, routing registry/cache, council architecture, or
  another package's work; or
- a push.

No compatibility aliases or deprecated entry points are required because the
package is pre-deployment and `simulate_readout` did not previously exist.

## Stop And Return-To-Design Conditions

Stop the affected work and return exact evidence to TensorDSLab Design if:

- the package baseline, dependency pin, work-order authority, branch, route,
  or allowlist is dirty, stale, discrepant, or ambiguous;
- complete preparation cannot precede every product-producer invocation
  without changing an accepted scientific or product-result contract;
- a required product-local plan would duplicate scientific equations in
  `simulation.py` or require a public planner/framework;
- TensorCore's public `CounterRng` contract is insufficient for a real accepted
  consumer rather than merely lacking a speculative capability query;
- common-product values depend on unrelated request order or retention;
- one required producer cannot execute at most once;
- source identity/immutability, exact axes reuse, fresh output storage,
  pairwise retained-output independence, or no-post-exposure-write cannot be
  maintained;
- a private signature change requires a compatibility shim or public exposure;
- an accepted closed test fails for a substantive reason;
- CUDA evidence contradicts rather than merely qualifies the accepted CPU
  contract;
- the correction budget is exhausted or the same finding repeats without
  convergence;
- implementation requires a TensorCore, TensorG4DS, TensorML, DAG,
  persistence, workspace, or scientific decision; or
- completing the work would require any forbidden scope above.

Do not patch around a stop condition with fallback algorithms, dummy RNG
draws, silent movement/casts, skipped tests, weakened invariants, aliases, or
unrecorded scope expansion.

## Merge And Final Closeout

Independent Review may authorize and perform a clean fast-forward only after:

- Validation clears one exact fixed commit;
- Review finds no unresolved issue on those same bytes;
- the candidate is a clean linear descendant of the Design authority;
- every changed path is allowlisted and every protected path is unchanged;
- source/archive tests, static typing, imports, exports, forbidden scans,
  hygiene, and applicable conditional CUDA gates are reconciled; and
- `main` is clean and an unambiguous ancestor of the candidate.

Review repeats the required post-merge checks on unchanged `main` and reports
the exact resulting commit. The state then becomes
`Merged / Design acceptance pending`.

Final TensorDSLab Design independently checks the merged topology, exact
dependency, candidate diff, public signature/exports, whole-request
preparation, request matrix, key collision boundary, execute-once/retention,
source and storage results, source/archive tests, typing, import isolation,
documentation, and all qualifications. If accepted, Design records
`Merged / Closed` through a documentation-only closeout whose parent is the
exact Review-cleared implementation and which changes no production or test
byte. No push occurs without separate authorization.

## Completion Boundary

Stage 7 is complete only when all of the following are true:

- the exact public `simulate_readout(...)` surface exists at both deliberate
  export layers;
- complete preparation precedes every product-producer invocation;
- request parsing, typed closure, config selection, key collision, execution,
  retention, and collection construction satisfy this work order;
- all required producers execute at most once and only requested fields are
  retained;
- source exact-return/immutability, generated freshness, pairwise storage
  independence, exact axes/device/dtype, autograd, synchronization, and
  generated-product deep postconditions and failure effects are proven;
- fixed-commit Validation and independent Review clear the same bytes;
- Review cleanly fast-forwards and verifies unchanged `main`;
- final Design accepts the merged evidence and synchronizes live docs; and
- the work order and implementation index record `Merged / Closed`.

Stage 7 has been dispatched and Candidate 2 advances only through the lifecycle
states above. While the candidate bytes are absent from `main`, fixed-commit
Validation and Review remain required. If the exact bytes are present unchanged
on `main`, Review's clean fast-forward has completed but final Design acceptance
remains pending until this work order and the implementation index record
**Merged / Closed**. Completion does not authorize any deferred integration,
optimization, IO, compatibility, deployment, conformance, Coordination,
Profile B, or push action.
