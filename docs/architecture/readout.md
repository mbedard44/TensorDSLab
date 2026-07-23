# Post-Binned Readout Architecture

## Purpose And Authority

This page summarizes the accepted post-binned readout architecture introduced
with TensorCore `0.7`. Closed Maintenance 5 installs published TensorCore
`0.13.0`; User-authorized / Dispatched Maintenance 6 fixes the next public
physical-configuration boundary. The complete scientific equations, config
sketches, RNG encoding, numerical gates, and source citations live in
[`rebuild.md`](rebuild.md). Donor comparison and intentional divergences live
in [`../parity.md`](../parity.md).

Architecture pages do not themselves dispatch production. Stages 3 through 6
are Merged / Closed. Stage 6 implemented the complete private Charge slice at
exact candidate `fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58`; Review's
merge/closeout is `ea979862b05f4ef543f6971c86641df317232479`. It retained
exact TensorCore `0.7.0` pin
`b454d738f6385ce6489d85492a618a3dab139bb6`. Its evidence is eager CPU-only
because all conditional CUDA tests were skipped. Public request-aware
`simulate_readout(...)` is Merged / Closed through exact Review-cleared Stage 7
candidate `6dd55024685013fb9412a7247d3ddde7be1a3177`. Maintenance 2 is Merged /
Closed through exact candidate `89a188abe330c06aa0b54c27cd61ac32a4fe9f63`
and
Design closeout `9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. Stage 2 and
Maintenance 1 remain historical evidence for the superseded TensorCore `0.6`
foundation. Maintenance 3 is Merged / Closed through exact Review-cleared
candidate `dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0` and its five-document
Design closeout. Maintenance 4 Runtime Action Ownership is
**Merged / Closed** through exact Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda`. It implements the accepted
behavior-preserving private ownership refactor with no public or scientific
contract change. Fixed-commit Validation and independent Review cleared the
exact source/archive forms locally and in separate fresh full-A100 allocations.
Maintenance 5 is **Merged / Closed** through exact Review-cleared supplemental
candidate `81ad2f52fe4a1966e5b3a0ceb5063138e42e731f` and Design closeout
`021694b9479d02546405f6a815aedf21c9c831a4`. Maintenance 6 is
**User-authorized / Dispatched**. It selects exact Pint `0.25.3` for public
physical Config values and the compact SampleAxis convenience boundary while
preserving the implemented axes, sampling authority, product graph, science,
RNG, and unit-free execution core.

## Scope

The first readout surface starts from an already-produced dense
`Photoelectrons` field and can produce:

```text
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
```

It does not parse G4DS files, accept jagged CPU PE tables, cluster deposits,
construct truth bins, perform IO, write caches, manage a DAG, move an input,
or define a TensorML model schema. Those are explicit later boundaries.

## Physical Interpretation

`Photoelectrons` is dense binned photon-origin truth. It contains no readout
timing jitter, dark counts, crosstalk, afterpulses, recovery weighting, or
charge smearing. The field is immutable input to readout simulation.

`Charge` is finite nonnegative aggregate PE-equivalent SiPM response per
example, channel, and sample. It is not an SI-coulomb measurement and does not
claim individual-microcell resolution.

`PureWaveform` and `NoiseWaveform` are signal-only and noise-only voltage
components at one zero-referenced analog plane. They are not sequential
hardware products. Their optional-saturation sum is `AnalogWaveform`, which is
the direct input to `DigitizedWaveform` ADC-code production.

The term *digitized* is deliberate. A future firmware/filter/trigger result
could be a distinct `DigitalWaveform` product.

## Public Surface

The implemented Stage 7 collaborator call is:

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

For example:

```python
readout = simulate_readout(
    photoelectrons,
    products=[AnalogWaveform, DigitizedWaveform],
    config=config,
    rng=Threefry4x32(seed=1234),
)

assert readout.field_types == frozenset(
    {AnalogWaveform, DigitizedWaveform}
)
analog = readout.field(AnalogWaveform)
```

`simulate_readout` is the one ordinary public simulation API. Product runtime
actions and scientific submodels remain independently testable internal units,
not alternate supported entry points. Runtime privacy is defined by deliberate
facade exports and documentation rather than a leading underscore: a direct
deep import remains possible Python behavior but is unsupported and carries no
compatibility promise.

## Axes And Sampling

Every product has exactly one `ExampleAxis`, one `ChannelAxis`, and one
`SampleAxis`. Their tuple order is tensor dimension order and may be arbitrary.
Producers locate dimensions by exact axis type and reuse the exact truth axes
tuple and exact axis instances for dimension-preserving results.

`ExampleAxis(CountAxis)` stores nonempty local ordinal count.
`ChannelAxis(LabelAxis)` stores nonempty unique detector labels.
`SampleAxis(RegularAxis)` stores integer-picosecond `start`, positive `step`,
count at least two, and a signed-int64-representable exclusive stop.

All readout bin conventions are left-closed and right-open. Stored values are
left edges; the final right edge is an exclusive stop and is not an extra axis
coordinate.

Count and regular coordinates are exact nonmaterializing `range` values.
Integer indices and one source-derived `SamplingRuntime` drive kernels.
Neither hot paths nor RNG use semantic coordinate values as random addresses.

The complete readout boundary requires `SampleAxis.start == 0`; the semantic
axis itself may describe a valid nonzero-start subgrid. `SamplingConfig`,
timestamp-string sample coordinates, and the collection-level `SampleGrid`
are retired. The future TensorG4DS bridge constructs the compact axis while it
bins PE truth.

Maintenance 6 keeps the stored axis exactly integer-backed and adds
`SampleAxis.from_period(...)` plus fresh Pint-valued `start_time`,
`sample_period`, `time_at(...)`, and exclusive `stop_time` conveniences.
Preparation continues to consume integer `start`, `step`, and `count`
directly; semantic time quantities never enter hot-path indexing or RNG
addresses.

## Product Fields And Collection

The field representation is:

| Product | Dtype | Deep value domain |
| --- | --- | --- |
| `Photoelectrons` | `torch.int64` | nonnegative |
| `Charge` | `torch.float32` or `torch.float64` | finite, nonnegative |
| `PureWaveform` | `torch.float32` or `torch.float64` | finite |
| `NoiseWaveform` | `torch.float32` or `torch.float64` | finite |
| `AnalogWaveform` | `torch.float32` or `torch.float64` | finite |
| `DigitizedWaveform` | `torch.int32` | nonnegative and config-bounded |

Every product uses ordinary dense `torch.strided` storage and exactly the three
readout axis types. Cheap structural and dtype facts are intrinsic leaf
requirements. Full-device value-domain checks occur in explicit product-owned
runtime validators at untrusted ingress and generated-product publication
boundaries rather than inside every constructor.

`ReadoutCollection` is a completed immutable result containing any nonempty
requested subset of those six exact types. It requires equal ordered axes, one
device, and one common dtype among present floating products. Membership is
semantically unordered and exposes no mutation, invalidation, or partial
pipeline state.

There is no digitization sidecar in the collection. In-process callers retain
the exact `DigitizedWaveformConfig`; durable association remains a later IO
design gate.

## Product Requests And Planning

`products` is a required keyword-only iterable of exact product classes. The
builder:

1. consumes it exactly once;
2. rejects an empty request;
3. rejects duplicate or unrecognized exact classes;
4. computes the transitive prerequisite closure;
5. preflights every required config and runtime relationship;
6. executes each prerequisite at most once; and
7. retains only requested fields.

The dependency graph is:

```text
Photoelectrons -> Charge -> PureWaveform
Photoelectrons axes/device/shape -> NoiseWaveform
PureWaveform + NoiseWaveform -> AnalogWaveform -> DigitizedWaveform
```

`Photoelectrons` is always available as input but is returned only when
requested. Requesting it returns the exact input field. An intermediate such
as `Charge` may be computed privately for a digitized request without becoming
a collection member.

Request order has no semantic meaning. Changing only the retained product set
must not change a product common to two requests. Config-owned stochastic-role
keys and deterministic planning enforce that independence.

Unknown products, duplicates, missing required config, invalid sampling,
an invalid `CounterRng`, duplicate role keys, or another request-level problem
fails before an RNG request, product-production invocation, or semantic-output
write. Product-owned private Runtime records complete the entire closure
before execution; the private whole-request preparer does not duplicate their
scientific equations.

The planner is ordinary typed code, not a public dependency registry or
workflow graph.

## Scientific Configuration

Every config is a final frozen slotted keyword-only dataclass. Exact component
types and TensorCore constrained scalars express the accepted domain. `None`
disables an optional submodel; a closed exact union selects an alternative
model.

```text
ReadoutConfig
├── ChargeConfig | None
│   ├── DarkCountConfig | None
│   ├── TimingJitterConfig | None
│   ├── CorrelatedAvalancheConfig | None
│   │   ├── maximum_generations
│   │   ├── DirectCrosstalkConfig | None
│   │   ├── DelayedCrosstalkConfig | None
│   │   └── AfterpulseConfig | None
│   │       └── AfterpulseRecoveryConfig | None
│   └── ChargeSmearingConfig | None
├── PureWaveformConfig | None
│   └── TpcFebSnrPulseConfig | VetoPduPulseConfig
├── NoiseWaveformConfig | None
│   └── ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig
├── AnalogWaveformConfig | None
│   └── AnalogSaturationConfig | None
└── DigitizedWaveformConfig | None
```

There is no `PhotoelectronsConfig`: the field already exists. There is no
generic `Config` ABC, string model selector, product-level persistence flag,
or runtime workspace policy in scientific config.
`ReadoutConfig()` is the valid truth-only configuration; source sampling is
not duplicated in a public config.

Maintenance 6 replaces each public physical scalar field with a scalar Pint
Quantity in the field's documented dimension and removes the unit suffix from
that public name. One private TensorDSLab registry and one common canonicalizer
copy every accepted caller Quantity into a canonical package-owned Quantity
and invoke the selected TensorCore `Scalar.require(...)` rule exactly once.
Dimensionless, algorithmic, key, and composition fields retain their current
types. All public Configs become explicitly unhashable.

The unit boundary ends during preparation:

```text
physical Config Quantity
  -> one canonical magnitude extraction
  -> unit-suffixed plain Runtime fact
  -> tensor/RNG production
```

Runtime records, products, collections, producers, validators, and RNG
addresses contain no Pint object or unit string. This target is
Design-complete but remains undispatched; the currently implemented Config
constructors remain the Maintenance 5 surface until implementation closes.

Exact stochastic leaf configs own defaulted TensorCore `RngKey` values:
white/PSD noise use streams `1`/`2`; dark count uses `3`; retained/overflow
direct crosstalk use `4`/`5`; retained/overflow delayed crosstalk use `6`/`7`;
timing jitter uses `8`; afterpulse uses `9`; and charge smearing uses `10`.
All use namespace `0x54445331` (`TDS1`). Keys identify stochastic roles and
participate in config equality and `repr`; they do not contain a seed,
algorithm, mutable state, device stream, or execution policy.

Each product preparer receives only its exact product config and shared
sampling/source facts when relevant. It returns one private immutable product
Runtime. Private submodels receive prepared values or their exact nested config
rather than the complete `ReadoutConfig`.

## Private Product Runtime Actions

Maintenance 4 implements one explicit private action triad for every generated
product:

```text
runtime/prepare.py   Config + execution facts -> ProductRuntime
runtime/produce.py   prerequisites + ProductRuntime (+ RNG) -> Product
runtime/validate.py  Product + minimal prepared facts -> None
```

The cross-module actions and records use clean names such as
`prepare_charge`, `produce_charge`, `validate_charge`, and `ChargeRuntime`.
They are private because no product, readout, or package facade exports them.
Every `runtime/__init__.py` and Charge `runtime/effects/__init__.py` remains
empty; internal callers import the exact defining module rather than an
internal facade. TensorCore's inherited `_require()` leaf hook and genuinely
module-local mathematical helpers may retain their established underscores.

Each ProductRuntime is a concrete final frozen slotted dataclass with no Runtime
base, Config, semantic product, mutable cache, method, or hidden device
movement. It contains only prepared tensors and static Python values required
by production, request-wide stochastic-role checks, or immediate result
validation. There is no Runtime ABC, registry, reflection layer, generic graph,
or action framework.

`readout.runtime.sampling` owns one private `SamplingRuntime` and
`prepare_sampling(...)`:

```python
@final
@dataclass(frozen=True, slots=True)
class SamplingRuntime:
    sample_count: int
    sample_period_ps: int
    sample_dimension: int
```

Request preparation constructs it once from the exact source `SampleAxis`
after enforcing complete-input start zero. Temporal ProductRuntime values
reference that exact object, so no product independently rediscovers the
simulation source sample dimension. The values remain Python integers because
they control dimensions, shapes, FFT lengths, and loop bounds; they are not
payload tensors.

`readout.runtime.prepare` owns `ReadoutRuntime`, request parsing, typed closure,
required-config checks, closure-wide key uniqueness, and composition of the
optional ProductRuntime values. Optional Runtime presence is the execution
closure signal; no duplicate `need_*` booleans or public dependency registry
is introduced. Complete closure preparation still precedes the first RNG
request, production call, or semantic-output write.

Product preparers own contextual config interpretation, canonical-magnitude
extraction, scientific equations, representability proofs, and
device-operand materialization. Public Config construction owns quantity
recognition, unit conversion, canonical copying, and scalar-domain
validation. Product producers
receive only explicit prerequisite products, the exact ProductRuntime, and
`CounterRng` when stochastic-capable. Production performs tensor/RNG execution,
narrow dynamic guards that cannot be proved beforehand, and final field
construction; it imports no Config and performs no deep publication scan.
Only Charge and NoiseWaveform production receive `CounterRng`. An exact-zero
or disabled stochastic path simply makes no RNG request.

The Maintenance 6 cleanup applies TensorCore's golden-path philosophy at the
existing boundary. Config construction retains physical canonicalization,
unwrapped primitive domains, and genuine local relationships, while retiring
annotation-only wrapper, key, nested-Config, optional, and union membership
checks. Whole-request preparation retains public ingress, request,
dtype/device, RNG capability, and key checks. Private child preparers stop
repeating those exact admission checks but retain exact model dispatch,
scientific laws, tensor relationships, representability, allocation/address,
and envelope limits. Private Charge effect executors may trust Runtime and
primitive types supplied only by the typed package path, while retaining every
scientific and tensor correctness guard. Validators remain unchanged and
package-owned because TensorCore does not establish exact source-axis tuple
identity, generated-product storage freshness, absolute product dtype domains,
or deep scientific value validity.

Product runtime validators are first-class, read-only actions. They neither
repair nor reconstruct values and run immediately after their product is
created, before any descendant can consume it. Each generated validator also
receives its named direct prerequisite product or products and proves the
completed result's axes, shape, dtype, device, and fresh-storage relationship
alongside its value domain. `DigitizedWaveformRuntime`
retains the prepared exact maximum code, so digitized validation receives no
Config. `validate_charge` owns the single finite/nonnegative terminal scan and
preserves the accepted `RuntimeError` boundary for an invalid generated
Charge. `validate_photoelectrons` remains the sole action in the truth
product's runtime package because Photoelectrons has no config, Runtime, or
producer. It checks the accepted truth value domain but does not impose
Charge's `2**53 - 1` per-cell count ceiling; that bound remains conditional
Charge preparation so a truth-only request is not narrowed by an unrequested
product.

The runtime layout is a deliberate internal seam for a possible later
`PureWaveformRenderer`, not its implementation. Maintenance 4 adds no renderer,
`nn.Module`, public atomic product action, or TensorML surface.

## Charge Production

The private charge producer evolves one local tensor through only the enabled
stages:

```python
charge = photoelectrons.tensor
charge_square_sum: torch.Tensor | None = None

if dark_effective:
    charge = simulate_dark_counts(charge)

if jitter_effective:
    charge = simulate_timing_jitter(charge)

if correlated_avalanches_enabled:
    correlated = simulate_correlated_avalanches(charge)
    charge = correlated.S1
    charge_square_sum = correlated.S2

if smearing_effective:
    charge = charge.to(dtype=floating_dtype)
    charge = simulate_charge_smearing(
        charge,
        charge if charge_square_sum is None else charge_square_sum,
    )

return Charge(
    tensor=charge.to(dtype=floating_dtype),
    axes=photoelectrons.axes,
)
```

An absent or exact logical-identity block disappears completely. Therefore
truth-to-charge, truth-to-jitter-to-charge, truth-to-dark-to-smearing-to-
charge, and every other supported stage-presence combination use the same
simple chain. Lowercase `charge` is a private tensor; uppercase `Charge` is the
final semantic product.

The physical order remains:

```text
truth + dark seeds -> timing jitter -> correlated avalanches -> smearing
```

Dark counts are exogenous primary avalanches. Jitter therefore redistributes
both truth and any present dark seeds while leaving the public truth field
unchanged.

### Timing Jitter

For each source bin, private timing jitter marginalizes a uniform latent
within-bin phase with a zero-mean ideal-normal displacement. The phase and
displacement are independent within each avalanche and IID across avalanches.
Preflight analytically integrates that law into binary64 probabilities for
every target bin that remains inside the finite window. Runtime then
redistributes aggregate integer counts through TensorDSLab's sequential
multinomial orchestration, whose nontrivial category steps call TensorCore's
public `rng.binomial(...)`; it does not draw a Gaussian per PE, invoke
Box-Muller for jitter, or materialize a jagged PE table.

For a source bin `s`, target bins are sampled in increasing `t` order, which is
also increasing signed offset `t - s`. The one combined out-of-window category
is the exact final count remainder and consumes no draw. No arbitrary Gaussian
tail cutoff may discard a destination that can still land in the window.
`sigma == 0` is draw-free identity. Retained counts plus explicit dropped
counts conserve the input exactly.

The first implementation prepares a log-domain one-sided cumulative tail for
`2**-52 <= sigma / T <= 64` and `2 <= sample_count <= 8192`, also requiring
`S * N <= 2**63`. It derives exact-symmetric offset masses and stable success/
later-category masses for each conditional binomial; it never repeatedly
subtracts categories from one, clips, or renormalizes. Category/tail/identity
error is bounded by `1e-12`, the complete represented source law by `1e-11`
L1, and `TimingJitterConfig.rng_key` owns the dedicated role whose default
stream is `8`. Full evaluator and validation details are normative in
`rebuild.md`.

This is a private Charge stage, not a transform that returns jittered
`Photoelectrons`.

### Fixed-Generation Correlated Avalanches

The only active cascade baseline uses a caller-configured
`maximum_generations = K` and one frozen unmarked integer count frontier per
generation. Every primary, dark, direct-crosstalk, delayed-crosstalk, or
afterpulse avalanche in one frontier receives the same recovery-independent
offspring laws for the next generation.

- Direct and delayed crosstalk use distinct Poisson means and distinct draws;
  their rates are never silently combined.
- Dark counts and retained/overflow crosstalk call TensorCore's public
  `rng.poisson(...)`: exact-zero no-draw, one-uniform CDF inversion below mean
  `10`, and Hoermann PTRS from `10` through the accepted per-cell Poisson mean
  ceiling `1e8`.
- Direct/delayed crosstalk children are fresh unit-charge avalanches.
- Afterpulse children are integer avalanches whose deposited charge may be
  weighted by the configured delay-dependent recovery response.
- Recovery affects deposited charge only; it never changes offspring
  probability or creates marked recursive state.
- Every child enters the same unmarked next frontier.
- All mechanisms are causal; only right overflow exists, and overflow is
  excluded from retained charge and later waveform products.

The simulator maintains separate physical ledgers:

```text
integer avalanche frontier
S1 = sum of deposited-charge weights
S2 = sum of squared individual deposited-charge weights
```

`S2` is terminal smearing scratch, not branching state. It is the sum of
individual squared weights, not the square of aggregate charge.
Prepared probabilities, Poisson rate fields, and discrete sampler control use
binary64 independently of the requested `Charge` dtype. `S1`, `S2`, AP charge
diagnostics, and the returned product remain in that requested dtype. Thus one
unchanged numerical execution stack produces the same integer avalanche
history for float32 and float64 requests even though their floating ledgers
need not be bitwise equal.

Crosstalk delay choices are exact fixed or exponential laws. A shared causal
guard rejects prepared negative-delay mass rather than silently clamping an
invalid model. The earlier `NormalDelayConfig` proposal is retired from the MVP
rather than left as a dormant public option. Stage 6 removed the class, both
union memberships, all export layers, and current tests without a compatibility
shim. Afterpulse delay remains exponential.

Physical delay plus independently marginalized uniform source-bin phase
determines the integer destination offset for each parent-child edge. Full
equations, overflow ledgers, and exact config ownership are normative in
[`rebuild.md`](rebuild.md).

`maximum_generations=1` expresses the first-generation case of the same
caller-configured finite-`K` MVP. There is no second public first-generation
API and no until-extinction, same-bin closure, generation-wave, or
recovery-marked implementation alternative.

### Charge Smearing

Terminal aggregate gain smearing consumes `S1` and `S2` after the complete
configured cascade. If correlation was disabled, every root has unit weight,
so the same converted charge tensor represents both `S1` and `S2`. Smearing
does not feed back into branching.

The Poisson crossover, equations, binary64 control, 64-attempt exhaustion,
no-fallback policy, and `1e8` mean ceiling are closed promoted TensorCore
requirements in `rebuild.md`; TensorDSLab owns the five exact scientific keys,
rate fields, and positional schedules. Aggregate multinomial factorization,
category order, prepared masses, and the final no-draw remainder remain
TensorDSLab contracts, while the exact binomial inversion/BTRS mappings, word
schedules, comparisons, budgets, and exhaustion behavior are promoted
TensorCore requirements. Timing jitter's log-tail
evaluator, numerical domain, tolerances, conditional masses, and exact stream
are also closed. Fixed/exponential phase-marginalized PMFs, analytic right
tails, and stable exponential AP-recovery preparation are closed in
`rebuild.md`: fixed delay has an exact two-point mapping with no PMF tolerance,
while exponential delay/recovery own bounded binary64 domains and
`1e-12`/`1e-11` tolerances. AP/smearing streams, the universal per-cell
`2**53 - 1` count ceiling, relational generation/address and accumulator
bounds, checked overflow/failure mechanics, smearing finiteness, and the frozen
TensorDSLab-model statistical policy are also closed in `rebuild.md`. Stage 6
implemented and validated these private Charge contracts at the exact reviewed
candidate above. Full source/archive runs each executed 174 tests: 164 passed
and 10 conditional CUDA tests skipped. The focused Stage 6 run executed 65
tests: 60 passed and 5 skipped. This is eager CPU evidence only and makes no
CUDA, GPU-performance, compile/fusion, or allocation-free claim. Stage 5
remains the historical implementation boundary for the two noise streams and
noise-required mechanics; Stage 6 owns the eight Charge streams and Charge
samplers.

## Waveform Products

### PureWaveform

`PureWaveformConfig.model` is exactly one of:

- `TpcFebSnrPulseConfig`; or
- `VetoPduPulseConfig`.

The MVP provisionally adopts the two audited IV-DSLab pulse equations while
giving their actual mathematical parameters explicit config names. A later
collaborator calibration review may change a model through Design; an
implementation stage may not silently reinterpret it.

### NoiseWaveform

`NoiseWaveformConfig.model` is exactly one of:

- `ZeroNoiseConfig`;
- `WhiteNoiseConfig`; or
- `PsdNoiseConfig`.

The caller supplies one-sided absolute PSD density in `mV^2/Hz`, left-edge
frequency bins, and a separate exclusive frequency stop. The caller does not
supply FFT coefficients or an FFT length. Preflight integrates PSD intervals
over the fixed record's frequency cells, discards the DC cell without
redistribution so no analog pedestal is added twice, generates the accepted
Gaussian spectrum, and uses:

```python
torch.fft.irfft(..., n=sample_count, dim=-1, norm="backward")
```

The exact odd/even endpoint scales, expected variance, covariance, and
precision policy are frozen in `rebuild.md`.

Stage 5 prepares white RMS and PSD overlaps in Python binary64, accumulates each
target cell with `math.fsum`, discards DC, and rounds each executed value once
into the requested floating dtype. White RMS must remain in that dtype's
positive normal range; represented subnormal RMS is outside the Stage 5 law.
Those represented values define the
ideal-standard-normal target statistics. The accepted finite Box-Muller lattice
is validated within its frozen allowance and is not post-normalized. The
private producer uses only the source axes, shape, and device; it does not read
the `Photoelectrons` payload and returns fresh `requires_grad=False` noise.

### AnalogWaveform

The product producer owns one pointwise expression:

```text
analog[i] = clamp(pure[i] + noise[i], minimum, maximum)
```

Either saturation bound may be absent; with both absent this is addition.
There is no deterministic analog pedestal. This clamp represents physical
front-end saturation. Present bounds must be finite in the waveform execution
dtype, and two bounds must remain strictly ordered after dtype conversion;
the producer uses those exact rounded device scalars.

### DigitizedWaveform

Preflight computes the ADC transfer constants once:

```text
maximum_code = 2**bit_depth - 1
gain = 10**(analog_gain_db / 20)
span = input_max_mv - input_min_mv
slope = gain * maximum_code / span
intercept = -input_min_mv * maximum_code / span
lower_input_mv = input_min_mv / gain
upper_input_mv = input_max_mv / gain
```

The producer evaluates:

```text
interior[i] = clamp(
    analog[i] * slope + intercept,
    0,
    maximum_code,
)

code_float[i] =
    0,                         if analog[i] <= lower_input_mv
    maximum_code,              if analog[i] >= upper_input_mv
    interior[i],               otherwise

digitized[i] = int32(code_float[i])
```

The thresholds, `maximum_code`, slope, and intercept are converted once to the
waveform dtype and materialized as exact zero-dimensional device scalars; the
thresholds must remain finite and strictly ordered. Direct pre-gain endpoint
comparisons make the accepted inclusive endpoints exact even when affine
rounding would otherwise lose the upper code. Floating clamp and endpoint
selection precede integer conversion, intentionally correcting the donor
cast-before-clip wraparound defect. Nonnegative open-interior float-to-int
conversion provides truncation. The endpoint-guarded affine digitizer
transfer, not `NoiseWaveform`, owns the ADC code corresponding to zero analog
voltage.

The initial analog and digitized product producers use their direct eager Torch
equations and make no kernel-count or temporary-allocation claim. A later
measured optimization stage may target one fused backend kernel without a
target-sized temporary. Cross-product fusion remains excluded unless a focused
Design change proves product and retention invariance.

## Random Fields

The public RNG input is one required immutable TensorCore `CounterRng`. A
caller ordinarily writes:

```python
rng = Threefry4x32(seed=1234)
```

The RNG contains the algorithm and invocation seed. Reusing it intentionally
replays the same positional realization; it never advances mutable state.
Deterministic requests still receive the RNG but request no values. There is
no simultaneous `seed=`, TensorDSLab RNG wrapper, `torch.Generator`, or global
RNG.

Each stochastic leaf config owns an exact `RngKey` identifying its role.
Default keys use namespace `TDS1` and streams `1` through `10` in the
historical Stage 5/6 order. Afterpulse uses one coupled key; direct and delayed
crosstalk each use distinct retained/overflow keys. The public builder rejects
one key assigned to different roles in the requested closure before any RNG
request, product-producer invocation, or semantic-output write.

TensorCore owns counter generation, logical positions, uniforms, parameterized
Gaussian draws, Poisson sampling, binomial sampling, and the two count
distributions' internal word schedules. TensorDSLab owns product-specific key
placement, scientific position/category lattices, direct-uniform/Gaussian
ordinals, draw-free scientific policy, multinomial ordering and final
remainders, count accumulation, and ledgers. Positions depend on actual
tensor-dimension indices, not semantic coordinate values, strides, or storage
addresses. A dimension or coordinate reordering is therefore a different
positional interpretation and carries no permutation-invariance promise.

Selection and arbitrary chunking are not automatically stable because each
builder invocation starts logical positions at zero. Callers use different RNG
seeds for independent invocations. A future chunk-stable API requires explicit
global offsets.

Closed Stage 5/6 production used a private `_RngStream` and
`readout/_random.py`. The Maintenance 2 implementation removes both
and historically pins TensorCore `0.9.0` exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, which supplies the required
public RNG API and focused `require_same_dtype` relationship. The historical
consumer proposal is fulfilled, and Maintenance 2 is Merged / Closed at the
exact candidate and Design closeout above. The closed Stage 5/6 and Maintenance
2 evidence are CPU-only because CUDA was unavailable.
Maintenance 5 replaces the installed pin with exact TensorCore `0.13.0`
without changing those RNG contracts.

TensorCore exposes no non-consuming concrete-algorithm capability query. The
public builder accepts nominal `CounterRng` membership, performs no dummy draw,
and never restricts the abstraction to exact `Threefry4x32`. A real custom RNG
backend failure occurs at its first genuine distribution request and belongs
to the execution-failure boundary.

## Functional, Memory, And Exposure Contract

The source `Photoelectrons` field is borrowed read-only. Requesting it is an
exact return. Every generated product has guaranteed-fresh storage independent
of named inputs, and simultaneously retained generated products are
storage-independent from one another. All dimension-preserving products reuse
the exact source axes.

No producer silently moves, casts, detaches, host-materializes, or mutates an
existing field. Generated fields use their declared result dtype.

Producer writes are initiated or enqueued before constructing and exposing the
semantic result. TensorDSLab never later writes through an alias to that
storage. Private scratch and unrequested intermediates never enter the result.

Stage 7 established the public result trust boundary deferred by Stage 4.
Maintenance 4 gives that boundary explicit product-owned runtime validators.
Each producer constructs one local result field after its payload writes and
returns it to `simulate_readout(...)`; orchestration immediately invokes the
matching `validate_<product>(...)` before any descendant. Charge must be finite
and nonnegative; Pure, Noise, and Analog must be finite; Digitized must be
nonnegative and bounded by its prepared exact maximum code. Charge performs
one terminal deep scan rather than its former duplicate checks. A failed
postcondition exposes no field, invokes no downstream producer, and returns no
partial collection. The scan may synchronize CUDA through scalar extraction.

The initial API has no `out=`, destination collection, public workspace,
stream lease, pool, or allocation-free claim. Future measured reuse keeps raw
writable storage private and exclusive until writes are enqueued; an exposed
valid field is never a mutable destination.

Constructing a field does not itself add synchronization. Deep source
validation and producer postconditions use scalar reductions that may
synchronize CUDA as an accepted correctness-first cost. Outside those checks,
same-stream consumers use normal PyTorch ordering and cross-stream consumers
establish an explicit dependency.

## Validation

Validation separates universal structure, intrinsic semantics, trust-boundary
value domains, and operation behavior.

The accepted Stage 7 public preparation contract covers at least:

- exact source type, CPU/CUDA device, and deep nonnegative truth domain;
- exactly three readout axes, source shape, dtype, Torch layout, and device;
- exact source `SampleAxis` narrowing and complete-input start zero;
- one nonempty unique recognized product request;
- exact required config closure and no irrelevant influence;
- selected floating dtype and representable scalar constants;
- a required `CounterRng` instance, exact config-owned `RngKey` values, and no
  duplicate key assigned to different roles in the requested closure;
- no dummy RNG capability probe; a deterministic closure does not query the
  RNG, while a real custom stochastic backend failure remains dynamic;
- scientific parameter and prepared-kernel normalization;
- causal delay and window/overflow policies; and
- every statically preparable failure before the first RNG request, producer
  invocation, or semantic-output write.

Maintenance 4 preserves that complete-preparation boundary while making the
execution order explicit:

```text
produce Charge -> validate Charge
  -> produce PureWaveform -> validate PureWaveform
  -> produce NoiseWaveform -> validate NoiseWaveform
  -> produce AnalogWaveform -> validate AnalogWaveform
  -> produce DigitizedWaveform -> validate DigitizedWaveform
  -> retain requested products -> construct ReadoutCollection
```

Only actions present in the requested transitive closure run. A Charge or Pure
validation failure occurs before the NoiseWaveform RNG request; every other
failure likewise prevents all descendants and final collection construction.

Cross-stage behavioral validation includes:

- all request subsets and prerequisite retention rules;
- common-product invariance under unrelated retention changes;
- exact source identity when requested and source immutability always;
- exact axes reuse, result dtype/device, fresh-storage, and pairwise output
  independence;
- deterministic and stochastic enabled/disabled identities;
- all 16 optional Charge stage-presence combinations;
- fixed-generation mechanism ledgers, S1/S2 conservation, and overflow;
- parity observables and tolerances from `parity.md`;
- waveform reference-equation agreement and gradient behavior where accepted;
- PSD variance/covariance/DC and odd/even endpoint cases;
- analog/ADC boundary and saturation behavior;
- CPU plus conditional CUDA evidence without inferring a GPU claim from
  skipped tests; and
- no writes after semantic exposure.

Package tests and Review enforce direct final fieldless TensorCore leaves and
public exports. They do not attempt adversarial subclassing, class mutation,
private-call misuse, direct tensor mutation, or exotic dispatch hardening.

## Product-Centered Module Ownership

Shared semantic axes live in `tensor_dslab.common`; there is no public
sampling config or `common.sampling` module. Private source-bound sampling execution facts live in
`readout/runtime/sampling.py`. Every generated product owns public `config.py`
and `field.py`, plus unexported
`runtime/{prepare,produce,validate}.py`. Photoelectrons owns only its
already-produced truth field and `runtime/validate.py`. Charge effects live
beneath `charge/runtime/effects/`.

`readout/config.py` contains only `ReadoutConfig`, and
`readout/collection.py` contains only `ReadoutCollection`.
`readout/requirements.py` owns only genuinely shared private relationships;
runtime packages import no cross-product orchestration layer. The former
`readout/_requirements.py`, product `_produce.py` bundles, and
`charge/effects/` paths are retired without shims by Maintenance 4. Closed
work orders continue to name their historical paths as exact evidence.
`readout/_random.py` and `_RngStream` remain absent under closed Maintenance 2.

The exact tree and import direction are normative in
[`rebuild.md`](rebuild.md). Do not add empty placeholders, a global validation
layer, or global `configs.py`, `fields.py`, `builders.py`, `utils.py`, or
`helpers.py` modules. The explicit product-local runtime validators are not a
generic validation dumping ground.

## Deferred Boundaries

Deferred work includes:

- the exact TensorG4DS-to-Photoelectrons bridge;
- persistence, durable labels, config association, and cache formats;
- TensorML product selection and model schema;
- a public reusable `PureWaveformRenderer` and its module-state boundary;
- Reconstruction products and execution arrangements;
- public workspace or destination reuse;
- chunk-stable RNG offsets;
- broad compatibility, deployment, and release policy; and
- convenience properties beyond exact-type lookup.

## Production Slices

Stage 3 is Merged / Closed and implements only the TensorCore `0.7` semantic
foundation: exact dependency pin, axes, sampling, product field/config types,
`ReadoutCollection`, exports, and focused structural/static tests. It created
no empty simulation, RNG, or product-builder module.

Stage 4 is Merged / Closed and implements exactly the private pure, analog, and
digitized waveform producers under the functionality-first contract. Stage 5
is Merged / Closed and implements the private positional RNG behavior consumed
by complete exact-zero, IID-white, and caller-supplied PSD noise. Stage 6 is
Merged / Closed and implements the complete private Charge producer, all eight
Charge RNG streams, aggregate samplers, and delay/jitter/cascade/ledger/smearing
mechanics. The Maintenance 2 implementation pins selected TensorCore
`0.9.0` commit `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, splits module
ownership, migrates to config-owned keys, and preserves default-key outputs;
it is Merged / Closed. Stage 7 is also Merged / Closed and implements complete
request-aware `simulate_readout(...)`; no partial public API implies
unsupported product closure. Maintenance 3 is Merged / Closed and qualifies
completed stochastic literals by their exact numerical stack without changing
production. Maintenance 4 is Merged / Closed through exact candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda`; it implements the
behavior-preserving runtime-action ownership split described above and
authorizes no renderer. Measured GPU characterization remains a separate
evidence stage; any justified fusion work remains a later optimization stage.

Maintenance 5 atomically adopts exact published TensorCore `0.13.0`, the
compact semantic axes, and source-derived sampling while preserving all product
execution. It is Merged / Closed. Maintenance 6 is User-authorized /
Dispatched; it owns the next physical Config representation and bounded
runtime-admission cleanup. Its work order and the implementation index are
the sole lifecycle records.

## Return To Design Before

Return to Design before:

- changing product names, meanings, dtypes, axes, or dependency graph;
- introducing another public simulation entry point;
- implementing or exporting a reusable renderer before its focused work order;
- changing Photoelectrons truth or moving its production into readout;
- changing sampling coordinates or bin conventions;
- changing the fixed-generation cascade or one of its mechanism laws;
- replacing the selected pulse, PSD, analog, ADC, or RNG contract;
- exposing protected/raw RNG mechanics, private producers, scratch,
  destinations, or workspace;
- adding persistence, TensorG4DS, TensorML, Reconstruction, or DAG scope;
- relaxing exact-request, fresh-result, no-post-exposure-write, or source-
  immutability guarantees; or
- accepting a compatibility, allocation-free, GPU, deployment, or scientific
  parity claim not backed by the named work order and evidence.
