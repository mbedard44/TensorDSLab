# Post-Binned Readout Architecture

## Purpose And Authority

This page summarizes the accepted post-binned readout architecture for the
TensorCore `0.7` rebuild. The complete scientific equations, config sketches,
RNG encoding, numerical gates, and source citations live in
[`rebuild.md`](rebuild.md). Donor comparison and intentional divergences live
in [`../parity.md`](../parity.md).

The architecture is a production target, not an implementation dispatch.
Stage 2 and Maintenance 1 remain historical evidence for the superseded
TensorCore `0.6` foundation.

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

The normal collaborator call is:

```python
def simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    seed: int | None = None,
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
    seed=1234,
)

assert readout.field_types == frozenset(
    {AnalogWaveform, DigitizedWaveform}
)
analog = readout.field(AnalogWaveform)
```

`simulate_readout` is the one ordinary public simulation API. Private product
producers and scientific submodels remain independently testable internal
units, not alternate supported entry points.

## Axes And Sampling

Every product has exactly one `ExampleAxis`, one `ChannelAxis`, and one
`SampleAxis`. Their tuple order is tensor dimension order and may be arbitrary.
Producers locate dimensions by exact axis type and reuse the exact truth axes
tuple and exact axis instances for dimension-preserving results.

`SamplingConfig` owns positive integer `sample_period_ps`, `sample_count >= 2`,
example-local start zero, and signed-int64-representable exclusive window stop.
It constructs canonical `SampleAxis` coordinates from left edges:

```text
"0ps", "2000ps", "4000ps", ...
```

All readout bin conventions are left-closed and right-open. Stored values are
left edges; the final right edge is an exclusive stop and is not an extra axis
coordinate.

Timestamp strings are semantic labels. Numeric config values and integer
indices drive kernels. Neither hot paths nor RNG use semantic labels as random
addresses.

The previous count-only sample axis and collection-level `SampleGrid` are
retired. The future TensorG4DS bridge uses the same `SamplingConfig` to bin PE
truth before calling readout.

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
requirements. Full-device value-domain checks occur at untrusted ingress and
producer postconditions rather than inside every constructor.

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
must not change a product common to two requests. Product-local random streams
and deterministic planning enforce that independence.

Unknown products, duplicates, missing required config, invalid sampling,
invalid seed, or another request-level problem fails before an RNG draw or
tensor write.

The planner is ordinary typed code, not a public dependency registry or
workflow graph.

## Scientific Configuration

Every config is a final frozen slotted keyword-only dataclass. Exact component
types and TensorCore constrained scalars express the accepted domain. `None`
disables an optional submodel; a closed exact union selects an alternative
model.

```text
ReadoutConfig
├── SamplingConfig
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

Each producer receives only its exact product config and shared sampling facts
when relevant. Private submodels receive their exact nested config rather than
the complete `ReadoutConfig`.

## Private Product Functions

Private functions use one naming distinction:

- `_product_*` constructs one semantic product; and
- `_simulate_*` implements a scientific submodel inside a product producer.

Conceptual product signatures are:

```python
def _product_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    seed: int | None,
    floating_dtype: torch.dtype,
) -> Charge:
    ...


def _product_pure_waveform(
    charge: Charge,
    *,
    sampling: SamplingConfig,
    config: PureWaveformConfig,
) -> PureWaveform:
    ...


def _product_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: NoiseWaveformConfig,
    seed: int | None,
    floating_dtype: torch.dtype,
) -> NoiseWaveform:
    ...


def _product_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    config: AnalogWaveformConfig,
) -> AnalogWaveform:
    ...


def _product_digitized_waveform(
    analog: AnalogWaveform,
    *,
    config: DigitizedWaveformConfig,
) -> DigitizedWaveform:
    ...
```

There is no Photoelectrons producer. Product packages receive explicit
prerequisites rather than a complete collection as a service locator.

Private functions may trust complete public preflight. They do not reproduce
the supported public boundary or defend direct unsupported calls.

## Charge Production

The private charge producer evolves one local tensor through only the enabled
stages:

```python
charge = photoelectrons.tensor
charge_square_sum: torch.Tensor | None = None

if dark_effective:
    charge = _simulate_dark_counts(charge)

if jitter_effective:
    charge = _simulate_timing_jitter(charge)

if correlated_avalanches_enabled:
    correlated = _simulate_correlated_avalanches(charge)
    charge = correlated.S1
    charge_square_sum = correlated.S2

if smearing_effective:
    charge = charge.to(dtype=floating_dtype)
    charge = _simulate_charge_smearing(
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
within-bin phase with a zero-mean normal displacement. It prepares aggregate
target/drop probabilities and redistributes counts without materializing a
jagged PE table. `sigma == 0` is draw-free identity. Shifted values outside the
window are dropped with explicit conservation diagnostics.

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

Crosstalk delay choices are exact fixed, exponential, or zero-clipped normal
laws. The normal law is `max(Normal(location, sigma), 0)` and therefore has an
atom at zero; it is not a truncated or folded normal. A shared causal guard
rejects prepared negative-delay mass rather than silently clamping unrelated
models. Afterpulse delay remains exponential.

Physical delay plus independently marginalized uniform source-bin phase
determines the integer destination offset for each parent-child edge. Full
equations, overflow ledgers, and exact config ownership are normative in
[`rebuild.md`](rebuild.md).

`maximum_generations=1` expresses the bounded first-generation MVP. There is
no second public first-generation API and no until-extinction, same-bin
closure, generation-wave, or recovery-marked implementation alternative.

### Charge Smearing

Terminal aggregate gain smearing consumes `S1` and `S2` after the complete
configured cascade. If correlation was disabled, every root has unit weight,
so the same converted charge tensor represents both `S1` and `S2`. Smearing
does not feed back into branching.

Exact Charge RNG streams, Poisson sampling, PMF numerical preparation,
supported count/rate/generation bounds, checked overflow, and Charge parity
tolerances remain gates before stochastic Charge implementation. Stage 5
separately freezes only the two noise streams and noise-required random
mechanics.

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

The public RNG input is one exact non-boolean Python `int` seed in
`[0, 2**64)`, or `None` when the entire effective request is deterministic.
Users do not pass a TensorDSLab RNG object or `torch.Generator`.

Private stochastic leaves use fixed numeric operation streams and logical flat
tensor positions. The address is rank-agnostic and depends on indices, not
`ExampleAxis`, `ChannelAxis`, or timestamp strings. A tensor-dimension or
coordinate reordering is a different positional interpretation and carries no
permutation-invariance promise.

Unrelated requested branches do not perturb a common product's random field.
Selection and arbitrary chunking are not automatically stable because each
builder invocation starts its logical positions at zero; callers use distinct
seeds for independent invocations. A future chunk-stable API requires explicit
global offsets.

The fixed Threefry word engine and noise-required distribution transforms are
selected in `rebuild.md`. The central private enum begins with exactly:

```python
NOISE_WHITE = 0x0000_0001
NOISE_PSD_COEFFICIENT = 0x0000_0002
```

Stream zero is unassigned; zero noise owns no stream. Stage 5 is vectorized
eager CPU plus conditional eager CUDA only. Raw words and fixed-point uniforms
must agree exactly between accepted CPU/CUDA paths; completed Box-Muller and
PSD values require exact same-backend repeatability and cross-backend
statistical agreement. Exact Charge stream assignment and Poisson details
remain later Charge gates.

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

The initial API has no `out=`, destination collection, public workspace,
stream lease, pool, or allocation-free claim. Future measured reuse keeps raw
writable storage private and exclusive until writes are enqueued; an exposed
valid field is never a mutable destination.

Constructing a field does not synchronize the device. Same-stream consumers
use normal PyTorch ordering; cross-stream consumers establish an explicit
dependency.

## Validation

Validation separates universal structure, intrinsic semantics, trust-boundary
value domains, and operation behavior.

Public preflight covers at least:

- exact source type and deep nonnegative truth domain;
- exactly three readout axes, source shape, dtype, Torch layout, and device;
- exact `SampleAxis` agreement with `SamplingConfig`;
- one nonempty unique recognized product request;
- exact required config closure and no irrelevant influence;
- selected floating dtype and representable scalar constants;
- seed presence/type/range for every effective stochastic branch;
- scientific parameter and prepared-kernel normalization;
- causal delay and window/overflow policies; and
- every failure before the first draw or write.

Behavioral validation includes:

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

Shared axes and sampling live in `tensor_dslab.common`. Each product subpackage
owns its field, configs, validation, and eventual `_product.py`. The
Photoelectrons package owns only its field type.

`readout/types.py` contains only `ReadoutConfig` and `ReadoutCollection`.
`readout/simulation.py` owns public orchestration. `_requirements.py` and
`_random.py` are private readout support and appear only when real behavior is
implemented. Product packages never import the cross-product orchestration
layer.

The exact tree and import direction are normative in
[`rebuild.md`](rebuild.md). Do not add empty placeholders or global
`configs.py`, `fields.py`, `builders.py`, or `validation.py` modules.

## Deferred Boundaries

Deferred work includes:

- the exact TensorG4DS-to-Photoelectrons bridge;
- persistence, durable labels, config association, and cache formats;
- TensorML product selection and model schema;
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
by complete exact-zero, IID-white, and caller-supplied PSD noise. Later focused
stages close stochastic Charge RNG and finally expose complete request-aware
`simulate_readout`. A partial public simulation API must not imply unsupported
product closures. Measured GPU fusion remains a separate optimization stage
after functional producers exist.

## Return To Design Before

Return to Design before:

- changing product names, meanings, dtypes, axes, or dependency graph;
- introducing another public simulation entry point;
- changing Photoelectrons truth or moving its production into readout;
- changing sampling coordinates or bin conventions;
- changing the fixed-generation cascade or one of its mechanism laws;
- replacing the selected pulse, PSD, analog, ADC, or RNG contract;
- exposing private RNG, producers, scratch, destinations, or workspace;
- adding persistence, TensorG4DS, TensorML, Reconstruction, or DAG scope;
- relaxing exact-request, fresh-result, no-post-exposure-write, or source-
  immutability guarantees; or
- accepting a compatibility, allocation-free, GPU, deployment, or scientific
  parity claim not backed by the named work order and evidence.
