# Decisions

This page records accepted, superseded, and open Design decisions. Detailed
scientific contracts live in [`architecture/rebuild.md`](architecture/rebuild.md)
and comparison policy in [`parity.md`](parity.md).

## Accepted

### TensorDSLab Is Tensor-Native From The Start

TensorDSLab is a clean-slate detector data-lab package. It defines its own
post-TensorG4DS readout and future reconstruction semantics while using
TensorCore as its generic tensor-native substrate.

The intended data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorDSLab does not parse native G4DS files, reproduce TensorG4DS deposit or
cluster algorithms, own TensorML training, or own Projects/dag orchestration.

### Project And Package Names Follow The Tensor Ecosystem

The project folder is `TensorDSLab`, the Python import package is
`tensor_dslab`, and the distribution name is `tensor-dslab`. Semantic domains
are flat beneath the import root, such as `tensor_dslab.common`,
`tensor_dslab.readout`, and future `tensor_dslab.reconstruction`; there is no
intermediate `domain` namespace.

### Documentation Precedes Production Dispatch

Design owns documentation-only architecture and work-order preparation.
Production requires a focused stage document plus the repository's persistent
Implementation, Validation, and Review routes. Documentation acceptance does
not itself dispatch code.

Stage 2 is Merged / Closed at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`, followed by Maintenance 1 at
`3af8ab4acf834b07e3d027fb530e5f12934999a5`. Those commits are accepted
historical evidence for the TensorCore `0.6` package foundation. They accept
no scientific simulation, GPU claim, workspace, cache, integration surface,
deployability, broad compatibility, or conformance finding. The rebuild is an
intentional pre-deployment replacement, not an amendment of those historical
work orders.

### TensorCore `0.7` Established The Rebuild Roots

The rebuild uses TensorCore's public `TensorAxis`, `TensorField`, and
`TensorCollection` ordinary-ABC roots. Exact final TensorDSLab leaf types carry
axis, product, and collection meaning. TensorCore owns universal
representation validation; TensorDSLab owns domain relationships and
scientific meaning.

The reviewed TensorCore Design reference is exact clean commit
`b454d738f6385ce6489d85492a618a3dab139bb6`. Stage 3 selected that exact
dependency and proved public imports, runtime construction,
inherited-constructor static typing, and the operation-owned result-storage
boundary against both the clean source checkout and an independently archived
pin. TensorDSLab will not fork TensorCore or reproduce its private mechanics.

Maintenance 2 later installed exact TensorCore `0.9.0` for public RNG and
same-dtype behavior. Closed Maintenance 5 supersedes that dependency with
published TensorCore `0.13.0` exact commit
`202d8b1bc6259b8453d3d377570417f2480d782b`. The later version retains the
accepted tensor roots and RNG surface while adding compact axis
representations, generic `Scalar`, table roots, `TensorArtifact`, and a
golden-path structural boundary. Maintenance 5 consumes the compact axes and
preserved generic mechanics only; it adds no TensorDSLab table, artifact/IO,
or Pint surface. Maintenance 6 retains the exact pin and consumes only
`Scalar.require(...)` at TensorDSLab's package-owned quantity boundary.

### Exact Types Replace Loose Semantic Namespaces

The rebuild has no runtime axis IDs, field IDs, layout records, semantic name
constants, canonical field sequence, floating-field registry, or descendant
map. It uses exact Python types and typed calls:

```python
sample_dimension = field.dimension_of(SampleAxis)
analog = readout.field(AnalogWaveform)
```

Every TensorDSLab semantic leaf directly subclasses one TensorCore root, is
public and `@final`, declares `__slots__ = ()`, adds no stored fields, and
inherits the root constructor. Static analysis, tests, and Review enforce that
package-authored shape. TensorDSLab does not add runtime lineage policing for
callers who deliberately violate finality or otherwise leave the public API.

### Three Typed Axes Define Readout Coordinates

Every readout field has exactly one `ExampleAxis`, one `ChannelAxis`, and one
`SampleAxis`. The ordered axis tuple is tensor dimension order and may vary;
code locates axes by exact type rather than fixed dimension. Dimension-
preserving products reuse the exact truth axes tuple and exact axis objects.

Maintenance 5 implements:

- `ExampleAxis(CountAxis)` with a nonempty zero-based local ordinal range;
- `ChannelAxis(LabelAxis)` with nonempty unique string labels; and
- `SampleAxis(RegularAxis)` with nonnegative integer-picosecond start,
  positive step, count at least two, and signed-int64-bounded exclusive stop.

Count and regular coordinates are compact `range` values. The source
`SampleAxis` is the sole sampling authority; private
`prepare_sampling(photoelectrons)` derives count, period, and dimension once.
The complete readout boundary requires example-local `start == 0`, while the
semantic sample axis may represent a valid nonzero-start subgrid.

`SamplingConfig`, timestamp-string sample coordinates, the earlier count-only
sample proposal, and `SampleGrid` are retired without shims. Hot paths and
positional RNG use tensor positions and plain prepared integers, not semantic
coordinate values.

Maintenance 6 leaves that storage and hot-path contract unchanged. It adds a
Pint-aware `SampleAxis.from_period(...)` conversion and fresh physical-time
accessors without storing a Quantity or unit on the axis.

### Six Typed Fields Define Readout Products

The accepted product types are `Photoelectrons`, `Charge`, `PureWaveform`,
`NoiseWaveform`, `AnalogWaveform`, and `DigitizedWaveform`.

- `Photoelectrons` is nonnegative `torch.int64` binned photon-origin truth.
- `Charge` is finite nonnegative floating aggregate PE-equivalent response.
- Pure, noise, and analog waveforms are finite floating mV values sharing one
  dtype when present together.
- `DigitizedWaveform` is nonnegative config-bounded `torch.int32` ADC code.

`DigitizedWaveform` is preferred over `DigitalWaveform`; the latter name is
reserved for a possible later firmware/filter/trigger product concept and is
not part of the rebuild surface.

### Photoelectrons Is An Already-Produced Truth Input

Readout simulation accepts one dense `Photoelectrons` field. It has no
`PhotoelectronsConfig`, Runtime, preparer, or producer. It owns its field and
its explicit untrusted-ingress deep validator. A future TensorDSLab-owned
bridge will construct it from an exact accepted TensorG4DS product using a
bridge-selected compact `SampleAxis`. That bridge, not `simulate_readout`, owns provenance
mapping, channel mapping, and PE binning.

Dark counts, timing jitter, crosstalk, afterpulses, and smearing are private
charge-production effects. They never mutate or replace the truth field.
Requesting `Photoelectrons` returns the exact supplied field.

### ReadoutCollection Is A Completed Request-Selected Result

`ReadoutCollection` is an immutable nonempty collection containing any exact
requested subset of the six accepted product types. Membership is unordered.
All present fields use equal ordered axes, one device, and one common floating
dtype where applicable.

The collection is not a partially executed pipeline. It has no public add,
replace, projection-reconstruction, descendant invalidation, or output-buffer
lifecycle. An unrequested prerequisite may be computed privately but is not a
member and may become unreachable after construction.

### One Public Function Owns Readout Orchestration

`simulate_readout(photoelectrons, *, products, config, rng,
floating_dtype)` is the ordinary collaborator-facing simulation API. It
consumes the product iterable once, rejects empty/duplicate/unrecognized
requests, computes the typed transitive prerequisite closure, preflights the
entire effective request, executes each producer at most once, and retains
exactly the requested fields.

Request order has no meaning. Changing retention alone must not change a
common product value. Missing config, invalid `CounterRng`, duplicate role
keys, or another statically preparable request-level error fails before any
RNG call, product-producer invocation, or semantic-output write. Product-owned
Runtime records keep scientific equations beside their products; the private
readout preparer composes the complete Runtime closure.

Every generated product owns three non-exported actions:

```text
prepare_<product>   Config + execution facts -> ProductRuntime
produce_<product>   prerequisites + ProductRuntime (+ RNG) -> Product
validate_<product>  Product + minimal prepared facts -> None
```

Every required Runtime is prepared before the first producer, RNG request, or
semantic-output write. Execution then follows `produce -> validate ->
descendant`; an invalid intermediate never reaches another product or the
returned collection. Scientific-effect `simulate_*` functions remain below
the owning product boundary, so there is no public sequential avalanche API.

These clean internal names are private by facade exports and documentation,
not by leading underscores. Runtime paths remain importable implementation
details with no signature, path, or compatibility promise. Historical Stage 6
and Stage 7 `_produce.py`, `_produce_*`, and `*Plan` names remain accurate for
their closed candidates but are superseded as living architecture without
shims.

### Scientific Configuration Is Immutable And Compositional

`ReadoutConfig` contains only optional exact generated-product configs; the
source `SampleAxis` owns sampling and `ReadoutConfig()` is valid for a
truth-only request. Each product owns its field and its config types. `None`
disables an optional submodel; closed unions of exact config types select real
alternative models. Each product preparer receives its exact config and
relevant shared sampling/source facts, then its producer receives the trusted
typed ProductRuntime and explicit prerequisites. Neither receives the whole
`ReadoutConfig` as a service locator.

There is no generic `Config` ABC without a polymorphic consumer, no string
algorithm selector, no product-level `persist` flag, and no mixing of
scientific choices with mutable RNG state, invocation seeds, runtime
allocation, or device-stream control. Exact stochastic leaf configs may own
immutable `RngKey` role identities.

### The Package Tree Is Product-Centered

Shared `ExampleAxis`, `ChannelAxis`, and `SampleAxis` live in
`common/axes.py`. Maintenance 5 deletes `common/sampling.py` without a
replacement public sampling module.

`readout/config.py` contains only `ReadoutConfig`;
`readout/collection.py` contains only `ReadoutCollection`; and
`readout/simulation.py` owns thin public orchestration. Non-exported shared
relationships live in `readout/requirements.py`; complete request preparation
and one-time sampling binding live in `readout/runtime/{prepare,sampling}.py`.

Each generated product subpackage owns `field.py`, `config.py`, and
`runtime/{prepare,produce,validate}.py`. `photoelectrons` owns only `field.py`
and `runtime/validate.py`. Charge submodels live in focused non-exported
modules under `charge/runtime/effects`, including Charge-owned
multinomial/category orchestration, checked count helpers, delay preparation,
and bookkeeping in `counts.py`. Generic RNG and distribution mechanics come
from TensorCore; the current implementation has no `readout/_random.py` or
replacement `_rng.py`. Product Runtime modules never import the cross-product
collection, config, or public simulation layer.

`SamplingRuntime` is a private dependency-leaf record containing Python
sample count, period, and dimension values. `prepare_readout` derives it once
from the source `SampleAxis` after enforcing complete-input start zero, and temporal product
Runtimes reference that exact object rather than rediscovering the sample
dimension. Every ProductRuntime is a concrete final frozen slotted dataclass with no
Config, prerequisite semantic product, collection, mutable cache, action
method, inheritance hierarchy, or hidden movement.

Runtime and effect `__init__.py` files are empty and internal imports target
exact defining modules. There are no global `configs`, `fields`, `builders`,
or `validation` dumping grounds, and no empty placeholder behavior modules.
Package roots deliberately export only the collaborator-facing classes and
function; privacy tests inspect those facades rather than treating every
importable deep module as public.

### Functional Allocation Comes Before Buffer Architecture

The initial rebuild has no public `out=`, destination collection,
`ReadoutWorkspace`, stream lease, allocator, or allocation-free promise. Each
generated product has guaranteed-fresh storage independent of named inputs and
of other generated products retained in the same result. A requested
`Photoelectrons` field is an exact return of the source.

Every operation documents result type, device, dtype, axes, autograd,
synchronization, failure effects, and storage relationship using TensorCore's
operation-owned result vocabulary. TensorDSLab enqueues every producer write
before exposing the corresponding semantic field and never subsequently
writes through an alias. Private scratch never enters a collection.

If measurement later justifies reusable destinations, writable storage must
remain raw, exclusive, and unexposed until writes are enqueued. The retired
model of overwriting an already valid public field or collection will not
return.

### Deterministic Waveforms Are Functionality-First

The Merged / Closed Stage 4 work order implements exactly the private pure,
analog, and digitized waveform producers. It freezes scalar precision and
requires scientific/reference equations, dtype/device/axes, accepted autograd
behavior, source immutability, and guaranteed-fresh outputs. It makes no
kernel-count, target-sized-temporary, throughput, or compiler claim.

The Merged / Closed Stage 5 work order implements the complete exact-zero,
IID-white, and caller-supplied PSD noise producer plus only the private RNG
behavior those models consume. It likewise makes no compiler, fusion,
target-sized-temporary, throughput, or accelerator-performance claim.

The pure-waveform preparer derives its config-defined pulse template and
sampled normalization in Python binary64; the producer then materializes that
prepared template once in the `Charge` dtype/device. Analog saturation bounds
and ADC transfer constants use the same binary64 preparation rule before
field-dtype representability checks. Digitization retains an affine open
interior and explicitly saturates at inclusive dtype-rounded pre-gain
thresholds so the upper endpoint cannot lose one code to rounding.
Payload-sized convolution, analog, and digitizer arithmetic remains in the
input field dtype and device;
existing input payloads are never host-materialized.

### Stage 5/6 Private RNG Is Historical Production Evidence

The Merged / Closed Stage 5 work order selects private
`tensordslab.threefry4x32-20/v1` and one central strongly typed `_RngStream`
enum. Its initial exact members are `NOISE_WHITE = 0x0000_0001` and
`NOISE_PSD_COEFFICIENT = 0x0000_0002`; stream zero is unassigned, zero noise
owns no stream, and Charge assignments were outside that historical work
order. Merged Stage 6 appends and implements all eight Charge members through
`CHARGE_SMEARING = 0x0000_000A` without changing either Stage 5 value. The root
API remains one ordinary non-boolean 64-bit seed. No public RNG object,
`torch.Generator`, semantic coordinate, timestamp, or loose stream constant is
introduced.

Stage 5 implements only the raw engine, fixed-point uniforms, and Box-Muller
behavior consumed by zero/white/PSD noise. At that historical boundary,
Bernoulli, exponential, Poisson, categorical, rejection, source-quantum, and
generation mechanics remained later Design scope. The decision below now
selects aggregate binomial and Poisson behavior; standalone Bernoulli and
continuous exponential inversion still have no accepted MVP consumer. White
RMS and PSD cells are prepared in Python binary64, PSD overlaps use `math.fsum`,
and executed values are rounded once into the requested dtype.
Those values define ideal-standard-normal targets; the finite Box-Muller
lattice is not renormalized. White RMS must remain in the selected dtype's
positive normal range. Conservative host bounds reject that unsupported
subnormal regime and white/PSD scales that could overflow the selected dtype.

The accepted reference is vectorized eager CPU with conditional eager CUDA.
Raw words and fixed-point uniforms are exact across accepted implementations;
completed normal and PSD products are exactly repeatable only on one unchanged
numerical execution stack and compare statistically across backends. Results
are fresh, source-payload-independent `NoiseWaveform` values with
`requires_grad=False`.
Compiled execution and performance optimization remain later measured work.

### Invocation `CounterRng` And Config-Owned `RngKey` Are The Accepted Target

The public Stage 7 call requires one immutable TensorCore `CounterRng`:

```python
rng = Threefry4x32(seed=1234)

readout = simulate_readout(
    photoelectrons,
    products=products,
    config=config,
    rng=rng,
)
```

There is no simultaneous `seed=` parameter. The RNG carries the algorithm and
invocation seed. Reusing it intentionally replays the same positional
realization; it does not advance mutable state. Deterministic requests still
require the argument but request no values. Deterministic private producers
receive no RNG; only stochastic-capable Charge and noise producers do.

The accepted TensorCore distribution surface is exactly
`uniform(...)`, `gaussian(...)`, `poisson(...)`, and `binomial(...)`.
`gaussian(...)` accepts explicit mean and standard deviation; there is no
public `standard_normal(...)` method. Poisson and binomial own their internal
word/attempt schedules and expose no distribution ordinal. Timing jitter keeps
its fixed zero-mean analytic law and calls `binomial(...)`; it does not gain a
mean config field or call `gaussian(...)`.

Every stochastic leaf config owns an exact defaulted TensorCore `RngKey`.
Namespace is `0x54445331` (`TDS1`), and the append-only streams are:

```text
WhiteNoiseConfig.rng_key                     1
PsdNoiseConfig.rng_key                       2
DarkCountConfig.rng_key                      3
DirectCrosstalkConfig.retained_rng_key       4
DirectCrosstalkConfig.overflow_rng_key       5
DelayedCrosstalkConfig.retained_rng_key      6
DelayedCrosstalkConfig.overflow_rng_key      7
TimingJitterConfig.rng_key                   8
AfterpulseConfig.rng_key                     9
ChargeSmearingConfig.rng_key                10
```

Keys are ordinary immutable config state, participate in equality and `repr`,
and may be overridden with another exact `RngKey`. Afterpulse uses one coupled
key; each crosstalk mechanism uses separate retained and overflow keys.
Deterministic, delay, recovery, and composite configs own no key.

TensorCore owns generic counter generation, logical positions, fixed-point
uniforms, parameterized Gaussian draws, Poisson sampling, binomial sampling,
their numerical domains, and their internal word schedules. TensorDSLab owns
semantic key placement, scientific position/category lattices, direct
uniform/Gaussian ordinals, draw-free scientific policy, complete multinomial
orchestration, checked count accumulation, and physical ledgers. The accepted
Threefry packing preserves the Stage 5/6 bytes:

```text
key = seed low32, seed high32, RngKey.stream, RngKey.namespace
counter = position low32, position high32, quantum, raw-word block
```

The selected TensorCore `0.9.0` dependency at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59` also provides the focused generic
`require_same_dtype(*fields)` relationship. Analog composition uses it for its
two semantic inputs, and `ReadoutCollection` uses it only for present floating
products. TensorDSLab retains raw-tensor requirements and one private
`_require_representable_float(...)` helper for repeated scalar conversion into
an accepted floating dtype.

The Maintenance 2 implementation installs the selected exact
TensorCore commit, splits module ownership, migrates stochastic functions,
preserves default-key output continuity, and removes `_RngStream` plus
`readout/_random.py` without a shim. It is Merged / Closed through exact
candidate `89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. Closed Stage 7
rejects duplicate keys assigned to distinct roles in the requested transitive
closure before any RNG request, producer invocation, or semantic-output write.

### Charge Uses Aggregate Multinomial And Hybrid Poisson Sampling

Timing jitter and AP placement use aggregate multinomial laws realized through
TensorDSLab-owned sequential orchestration calls to TensorCore's public
`rng.binomial(...)`, never per-avalanche categorical expansion. TensorCore's
binomial mapping uses exact zero/one/no-count branches, probability reflection,
one-uniform forward-CDF inversion when `n * p_star < 10`, and Hoermann BTRS
otherwise. Inversion checks at most the terms
`k = 0 .. min(n, 63)` using the frozen binary64 recurrence and strict
`U < cumulative` test. BTRS owns one
Threefry block per addressed attempt, maps its two word pairs to `u` then `v`,
checks support before conversion or quick acceptance, uses the frozen
log-domain bound, and permits attempts `0 .. 63`. Each multinomial law prepares
the current-category mass `A` and later-category mass `B`; reflection is strict
at `B < A`, and complementation occurs only after acceptance. Equal masses are
not reflected. This avoids repeated remaining-mass subtraction and avoids
forming a tiny side as `1-p`. Either exhaustion is a deterministic hard
failure. For each timing-jitter source cell, increasing
target-bin order is increasing signed-offset order; the combined drop bucket
is the final exact count remainder. AP orders retained causal offsets
increasingly, then overflow, with stop as the exact remainder.

BTRD is rejected for this v1 mapping. Its additional decomposition primarily
saves uniform variates, while the accepted positional RNG reserves the same
complete two-uniform Threefry block for each attempt. BTRS retains the same
transformed-rejection target and central fast-accept region with fewer
branches and one fixed tensor-friendly word schedule. “Exact BTRS mapping” in
this Design means the frozen binary64 equations, operation grouping, word
addresses, and comparisons; it is not a claim of exact ideal-binomial sampling
despite finite Stirling-tail approximation and rounding. The accepted
cancellation-resistant `log1p` grouping is algebraically identical to the
earlier three-log form and supports `n <= 2**53 - 1`. Central candidates
through 25 standard deviations own a `1e-6` absolute local log-bound gate;
complete support uses
`1e-6 + 64*eps(float64)*max(1,abs(reference_side))` per side, exact decision
agreement outside the summed uncertainty band, and fixed-word decisions inside
it. The frozen statistical-law gate provides the separate distribution
evidence.

Dark counts and retained/overflow DiCT and DeCT call TensorCore's public
`rng.poisson(...)`. Its exact-zero path requests no word; positive means below
`10` use one-uniform binary64 forward-CDF inversion; means from `10` through
`1e8` inclusive use Hoermann PTRS; and unsupported means fail. Inversion checks
64 probability terms. PTRS consumes two open-open float64 uniforms from one
Threefry block per attempt and permits 64 attempts. Exhaustion never reseeds,
clamps, approximates, changes algorithms, or returns a fallback.
Poisson inversion owns a `1e-12` absolute term/CDF oracle gate. PTRS owns the
mixed `1e-6 + 64*eps(float64)*max(1,abs(reference_side))` local log-side gate
and exact decision agreement outside the resulting uncertainty band; fixed
words define decisions within that band.

All discrete probabilities, Poisson rate fields, and sampler control use
binary64 independently of the requested `Charge` dtype. Avalanche counts are
`int64`; S1/S2, AP charge diagnostics, and the final product remain in the
requested floating dtype. On one unchanged numerical execution stack, integer
avalanche history must therefore be identical for float32 and float64 Charge
requests.
Completed CPU/CUDA Poisson fields compare statistically rather than bitwise
because the selected algorithms use transcendental functions.

The complete append-only MVP Charge stream assignments are:

```text
CHARGE_DARK_COUNTS                  = 0x0000_0003
CHARGE_DIRECT_CROSSTALK             = 0x0000_0004
CHARGE_DIRECT_CROSSTALK_OVERFLOW    = 0x0000_0005
CHARGE_DELAYED_CROSSTALK            = 0x0000_0006
CHARGE_DELAYED_CROSSTALK_OVERFLOW   = 0x0000_0007
CHARGE_TIMING_JITTER                = 0x0000_0008
CHARGE_AFTERPULSES                  = 0x0000_0009
CHARGE_SMEARING                     = 0x0000_000A
```

Dark counts use `p = source_flat_position`. Every crosstalk role uses
`p = generation * N + local_flat_position`, where retained roles are
destination-indexed and overflow roles are source-indexed. All are aggregate
cell draws with `source_quantum = 0`; an attempt is the raw-word block. Direct
and delayed rates are never superimposed, and retained/overflow roles never
share a stream. Timing jitter uses
`p = target_bin * N + source_flat_position`, `q = 0`, and leaves its combined
drop category as a no-draw integer remainder. `torch.poisson`, global RNG,
normal approximations, and per-parent expansion are rejected substitutions.

AP uses one stream for its coupled retained, overflow, charge, and S2 outcome:
`p = ((generation * (S + 1) + category) * N) + source_flat_position`, with
retained offset categories `0..S-1`, fixed overflow category `S`, stop as the
no-draw remainder, and `q = 0`. Enabled smearing uses one full-grid scalar
normal per row-major flat position with `q = 0`; zero-scale cells retain their
addresses, while absent or zero-sigma smearing skips the stream entirely.

### Charge Uses A Relational Numeric Envelope

Active Charge execution uses the universal per-cell count ceiling
`C_max = 2**53 - 1`, covering source/working/frontier cells, mechanism and overflow
diagnostics, cumulative counts, aggregate-binomial counts, and accepted Poisson
samples. Every nonnegative integer addition proves `rhs <= C_max - lhs` before it
executes. The Poisson mean ceiling remains the independent `1e8`; neither bound
is a whole-grid, row, batch, or example population limit.

There is no magic maximum-generation constant. Checked role addresses require
`S*N <= 2**63` for jitter, `K*N <= 2**63` for each effective CT role, and
`K*(S+1)*N <= 2**63` for effective AP. The eager ledger plan additionally
requires `L < 2**p_d`, where `p_d` is 24 or 53 and
`L = E*K + 1`, or `E*K + S + 3` when recovered AP charge is retained. Its
forward-error bound is `gamma_L*T + L*eta_d`, with
`gamma_L = L/(2**p_d-L)`. Because this real bound constrains represented
ledgers, Maintenance 2 floors it to the greatest target-dtype ledger not above
the real bound; rounding it upward would create a value the proof does not
cover. Smearing retains the Stage 6 worst-ledger check and intersects it with
TensorCore's public Gaussian prepared-scale finite-output envelope. The
`K=0` float32/float64 adjacent endpoints remain frozen; only a contextual
extreme may narrow, as proved by the `L=24` float32 pair. This is a
representation-domain correction, not a change to scientific equations, RNG
addresses, clipping, or accumulation/operation order. The eager reference
fixes generation, direct-CT/delayed-CT/AP mechanism, source-bin, and AP-offset
accumulation order and forbids an unspecified repeated-index atomic reduction.
Details and failure effects are normative in
[`architecture/rebuild.md`](architecture/rebuild.md#stage-6-count-address-and-numeric-envelope).

### Stage 6 Separates Model Conformance From Donor Equivalence

Stage 6 reuses the four frozen Stage 5 seeds and fixes `2**18` independent
examples for scalar/one-parent laws and `2**16` for aggregate `Q=32`,
small-grid `K<=3`, and completed-Charge fixtures. A predeclared statistic must
satisfy `abs(observed-target) <= 8*SE + delta`, with target-law moments defining
`SE` and the frozen dtype/reduction allowance defining `delta`. Exact identities
and the `1e-12`/`1e-11` probability-preparation gates remain separate.

This validates TensorDSLab's selected model. It does not invent an IV-DSLab
percentage for finite-`K` recursion, DeCT, corrected AP, clipped smearing, or
detector-level Charge. A later donor-equivalence claim requires an
observable-specific collaborator/calibration margin and the combined-SE rule
in [`parity.md`](parity.md).

### Timing Jitter Uses An Analytically Prepared Gaussian Kernel

The timing-jitter scientific law is `U + J`, where the lost source-bin phase
`U` is uniform on the bin and `J` is an ideal zero-mean Gaussian. Each
avalanche receives one `(U, J)` pair whose components are independent, and the
pairs are IID across avalanches. Preflight analytically integrates that law
into binary64 probabilities for every destination that can remain inside the
finite sample window. The runtime owns
no per-PE Gaussian or Box-Muller draw: TensorDSLab's aggregate multinomial
orchestration calls TensorCore `binomial(...)` for increasing destination bins
and leaves the one combined out-of-window category as the final no-draw count
remainder.

There is no arbitrary Gaussian tail cutoff. Every possibly in-window
destination is evaluated, retained destinations plus the drop category
conserve each source count, and `sigma == 0` skips the complete stage without
RNG.

The stable evaluator is now closed. With `r = sigma / T`, the first
implementation supports `2**-52 <= r <= 64` and `2 <= sample_count <= 8192`,
subject also to `S * N <= 2**63`. It prepares the one-sided cumulative tail
`L[m] = r * (G(m/r) - G((m+1)/r))`, where
`G(z) = phi(z) - z*(1-Phi(z))`, in the log domain. `log(G)` uses the direct
`erfc` form below `z = 8` and the frozen decreasing-term asymptotic series at
and above `8`; stable `expm1` log differences produce both tails and offset
masses. Negative offsets reuse the same positive-offset mass exactly.

Each timing conditional receives stable success/later-category masses `A` and
`B`; it samples `min(A,B)/(A+B)` with an explicit complement flag. It never
repeatedly subtracts represented categories from one or obtains a tiny failure
probability as `1-p`. The fixed absolute category/tail/identity tolerance is
`1e-12`, and the complete represented source law must be within `1e-11` L1 of
the high-precision ideal oracle. Neither value is public configuration or
permission to admit negative probabilities, clip, assign a residual, or
renormalize. The exact stream is
`CHARGE_TIMING_JITTER = 0x0000_0008`.

This selection follows a finite Design sweep through the supported ratio and
sample-count boundaries, central and far tails, and natural binary64
underflow. Direct `H` second differences and naive remaining-mass subtraction
produced negative or materially corrupted tail values and are rejected. The
study is finite-grid evidence, not a proof over real values outside the
accepted domain. Stage 6 implemented and validated the correctness-first
evaluator in eager CPU mode; CUDA was unavailable, so CPU/CUDA agreement
remains an unobserved contract. The reference may perform quadratic sample-
count work; a later optimization requires measured evidence and may not
silently change the law. Completed jitter requires exact repeatability only on
the same unchanged numerical execution stack, inputs, axis order, config keys,
RNG algorithm, and invocation seed; CPU/CUDA completed
values compare statistically only after both paths have evidence.

### Public Validation Does Not Mean Adversarial Hardening

TensorDSLab validates supported public input relationships: exact product
requests, axes, shape, dtype, device, sampling, configs, scientific value
domains at trust boundaries, an accepted `CounterRng` instance, role-key
uniqueness, and numerical bounds. Cheap intrinsic checks occur in semantic
leaves; full-device
scans occur at explicit ingress or product-owned publication validators.

TensorDSLab makes no promise for callers who subclass final leaves, modify
classes, bypass construction, call private functions directly, mutate exposed
tensors, or install custom Torch dispatch behavior. Unsupported use may fail
naturally or produce invalid results and requires no stable error type or
adversarial test suite.

### The Fixed-Generation Correlated-Avalanche Model Is Selected

The sole active correlated-avalanche baseline is the fixed caller-bounded
generation process in [`architecture/rebuild.md`](architecture/rebuild.md).
It uses one unmarked integer frontier; separate direct-crosstalk,
delayed-crosstalk, and afterpulse mechanisms; floating deposited-charge `S1`
and charge-square `S2` ledgers; causal right-overflow diagnostics; and terminal
charge smearing. `maximum_generations=1` is the first-generation case.

Earlier same-bin-closure, generation-wave, and recovery-marked algorithms are
not implementation authorities. No work order may substitute one without a
new explicit Design decision.

### In-Memory Products Precede IO And Integration

Persistence, durable product labels, cache formats, compaction, TensorG4DS
bridges, TensorML adapters, Reconstruction, and Projects/dag integration are
deferred until the local in-memory products and operations are stable.
`products` means returned in-memory membership, not persistence.

A public `PureWaveformRenderer` is also deferred. Maintenance 4 prepares the
internal Runtime/producer seam that a later renderer may reuse, but it does
not add, export, validate, or dispatch a renderer, a generic TensorRenderer,
or a TensorML adapter.

The production integration target avoids silent CPU, NumPy/list,
serialization, movement, cast, or detach boundaries. Exact upstream/downstream
types and compatibility evidence require later focused work orders.

### Donor Parity Is Scoped And Classified

Historical DSLab and IV-DSLab are evidence, not architecture authorities.
Every promoted behavior names its donor, comparison boundary, input/config
domain, observables, criteria, exclusions, and parity classification in
[`parity.md`](parity.md). Statistical post-binning parity may be accepted
without seedwise or bitwise identity. Intentional scientific corrections and
bounded approximations must be explicit and measured.

### Governance Core 0.1.0 Is Adopted (`TDSLAB-GOV-D001`)

Decision ID: `TDSLAB-GOV-D001`

Decision status: Issued / Adopted

Decision date: 2026-07-10

Governed Design base: `151b61fdc36475498219ee5fe7b045a3a72c2d09`

Accepted candidate: `d634401a853915edeb4f83df4a4943b3553deced`

Governance manifest-file SHA-256:
`45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8`

Council context manifest-file SHA-256:
`343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916`

TensorDSLab accepts the exact Governance Core `0.1.0` package-adoption
candidate without conditions. Package adoption is `Adopted`; conformance is
`Not evaluated`; Coordination is `Deferred`; and Profile B is `Disabled`.
This decision created no scientific, dependency, device, compatibility,
production, routing, registry, cache, or deployment authority.

### Stage 6 Private Charge Simulation Is Merged / Closed

Stage 6 is Merged / Closed through exact implementation candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58` and Review's evidence-only
closeout `ea979862b05f4ef543f6971c86641df317232479`. It implements the closed
Charge scientific, numerical, stream, address, count, accumulator,
failure-effect, and TensorDSLab-model statistical contracts. Fixed-commit
Validation, independent Review, and Design's post-merge audit found no
unresolved issue. CUDA was unavailable, so the accepted evidence is eager
CPU-only and establishes no CUDA, GPU-performance, fusion, allocation-free, or
cross-backend claim.

This closeout establishes conformance to the selected TensorDSLab private
Charge model, not an IV-DSLab equivalence margin. Intentional donor divergences
still require an observable-specific collaborator or calibration margin under
[`parity.md`](parity.md).

### Stage 7 Public Readout Orchestration Is Merged / Closed

The focused
[Stage 7 work order](implementation/stage_7_public_readout_orchestration.md)
is Merged / Closed through exact Review-cleared implementation candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177`. It implements one public
`simulate_readout(...)` function, complete product-owned preparation before
producer execution, one private typed readout plan, closure-wide structural
role-key uniqueness, fixed execute-once topology, and exact requested
retention. It also closes the result trust boundary deferred by Stage 4: every
generated producer applies its exact private deep-value postcondition before
the field can reach a downstream producer or the returned collection.

TensorCore exposes no non-consuming algorithm-capability query. Stage 7 accepts
the public `CounterRng` abstraction without a dummy draw or exact-Threefry
restriction; a real custom RNG backend failure occurs at its first genuine
distribution request and belongs to the execution-failure boundary. Deep
value validation and product postconditions may synchronize CUDA through
scalar reductions as an accepted functionality-first correctness cost.

Fixed-commit Validation and independent Review cleared the exact merged bytes,
and Review completed a clean fast-forward plus post-merge verification. The
accepted source/archive evidence ran 188 tests in each dependency form: 176
passed and 12 conditional CUDA tests skipped. Pyright `1.1.411` reported zero
diagnostics in both forms. CUDA was unavailable, so this decision establishes
no GPU execution, cross-backend, fusion, allocation, or performance claim.

### Maintenance 4 Runtime Action Ownership Is Merged / Closed

[Maintenance 4](implementation/maintenance_4_runtime_action_ownership.md) is
**Merged / Closed** through exact Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda`. It implements a focused,
behavior-preserving internal refactor from product-local `_produce.py` bundles
and `*Plan` records to product-owned Runtime actions:

```text
Config + execution facts
  -> prepare_<product>
  -> ProductRuntime
  -> produce_<product>
  -> Product
  -> validate_<product>
  -> descendant or final retention
```

Whole-request preparation remains complete before the first RNG request,
production call, or semantic-output write. `SamplingRuntime` binds the source
sample facts once and is shared by temporal ProductRuntime records. Validation
runs immediately after each produced field with its named direct prerequisite
relationship, not after the complete chain, and invalid local products are
neither exposed nor passed downstream.

Runtime records and clean action names are non-exported implementation
details. Public facades, `simulate_readout(...)`, product/config/collection
identities, scientific equations, TensorCore `0.9.0`, RNG addresses and calls,
storage/autograd contracts, and supported devices remain unchanged. Runtime
packages do not create an internal facade, and no Runtime/Action framework or
compatibility shim is accepted. Fixed-commit Validation and independent Review
cleared the exact source and archive forms locally and in separate fresh
full-A100 allocations. This decision does not authorize the deferred renderer.

### Maintenance 5 Compact Axes And Source-Derived Sampling Is Accepted

The user-authorized
[Maintenance 5 work order](implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md)
accepts one atomic dependency/API migration from exact TensorCore `0.9.0` to
published `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b`. The three semantic axes adopt
the generic count, label, and regular representations fixed above.
`SamplingConfig`, `ReadoutConfig.sampling`, materialized sample timestamp
strings, and the duplicate source/config agreement check are removed without
aliases. The source `SampleAxis` becomes the single sampling authority.

The migration preserves arbitrary axis order, complete-readout start zero,
the public `simulate_readout(...)` signature, private three-field
`SamplingRuntime`, every product equation, RNG address/call, result
relationship, and supported device boundary. It also accepts TensorCore
`0.13.0`'s golden-path rule: supported exact lookup remains, while malformed
class-object calls have no promised diagnostic. The old
`collection.field(TensorField)` `TypeError` assertion is deleted rather than
replaced.

The decision authorizes no Pint, units, table, artifact/IO, integration,
renderer, Stage 8 measurement, optimization, compatibility shim, release, or
push. This decision does not itself dispatch production; the exact execution
lifecycle is recorded only in the work order and implementation index.

### Maintenance 6 Makes Pint A Public Boundary, Not An Execution Substrate

The user-selected
[Maintenance 6 work order](implementation/maintenance_6_pint_physical_configuration_boundary.md)
accepts exact Pint `0.25.3` for public physical Config fields while retaining
exact TensorCore `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b`.

TensorDSLab owns one private registry and exports only `quantity(...)` and
`quantities(...)`. A physical Config accepts a compatible scalar Quantity,
copies it into the package registry in a documented canonical unit, and calls
one TensorCore `Scalar.require(...)` rule to normalize its scalar magnitude.
Public physical names become unit-neutral. Every Config is explicitly
unhashable.

Preparation extracts each active physical magnitude once into a plain,
unit-suffixed Runtime fact. Runtime records, producers, validators, tensors,
collections, and RNG addressing remain unit-free. `SampleAxis` remains a
compact integer-picosecond `RegularAxis`; it gains one bounded Pint-aware
constructor and four fresh quantity accessors without storing a unit or
Quantity.

The same maintenance accepts a narrow golden-path cleanup: public Config and
whole-request preparation retain admission responsibility, while private
child preparers and Charge executors stop duplicating exact Config/Runtime,
primitive dtype/device admission already guaranteed by the typed path.
Scientific laws, exact model dispatch, tensor relationships, axes identity,
storage freshness, absolute product dtype domains, allocation/address/envelope
limits, and generated-product validation remain.

The decision is Design-complete but does not dispatch production. It
authorizes no TensorCore edit, scientific equation or RNG change, IO/artifact
surface, integration, Stage 8 work, optimization, compatibility shim, release,
or push.

### Completed Stochastic Literals Are Numerical-Stack Qualified

Maintenance 2's completed Gaussian, Poisson, binomial, noise, and Charge
hexadecimal fixtures remain exact evidence for their explicitly recorded
macOS 15.7.4 arm64, Python 3.13.11, PyTorch 2.12.1 eager-CPU stack. They are not
a portable CPU literal contract. For operations involving transcendentals,
exact replay requires one unchanged numerical execution stack, including the
OS/architecture, Python/PyTorch build, backend/device implementation,
execution mode, dtype, and relevant math settings.

Threefry raw words and fixed-point uniforms retain their separately documented
exact scope. Another accepted stack proves exact replay within itself plus the
existing structural, invariant, analytic, and statistical contracts. It does
not acquire a post-observation ULP tolerance, alternate platform golden table,
skip, or expected failure.

This validation-boundary correction changes no RNG algorithm, address, key,
scientific equation, production code, TensorCore dependency, or historical
literal. The first Stage 8 executable correctly stopped before accepted
measurement when its work order over-applied the macOS literals on Della. That
authority and executable input remain immutable stopped evidence; a later
Stage 8 restart requires a new authority after Maintenance 6.

## Superseded

### One Collection Subclass Per Product

The Stage 1 concept of six single-field collection subclasses was superseded
first by a multi-field readout snapshot and then by the typed-field rebuild.
Exact product field classes inside one request-selected collection preserve
the useful semantic distinctions without artificial single-member wrappers.

### TensorCore `0.6` IDs, Layouts, And Sidecars

The Stage 2 representation used `Id` subclasses, `TensorAxisId`,
`TensorFieldId`, `TensorLayout`, `shared_axes`, field-ID constants,
`SampleGrid`, and conditional `DigitizedWaveformSpec`. It also needed
TensorDSLab reconstruction around generic TensorCore operations. Those remain
valid historical implementation facts but are superseded for the rebuild by
TensorCore `0.7` typed axes and fields.

The rebuild intentionally offers no compatibility aliases for the retired
types, constants, sidecars, or module paths.

### Partial Snapshots And Descendant Invalidation

The former collection represented partially materialized pipeline state.
Atomic transforms added or replaced fields, shared unaffected records, and
invalidated derived descendants. The rebuild instead constructs one immutable
completed collection for one explicit request. Private prerequisite planning
replaces public snapshot mutation and invalidation.

### Public Atomic `out=` And Workspace Architecture

The former three-layer plan exposed atomic collection transforms, prepared
valid destination collections, and a public `ReadoutWorkspace` with a strict
warmed profile. It is superseded by a simple functional API until profiling
provides evidence for a narrower optimization.

In particular, the old plan conflicts with TensorCore's current no-write-after-
semantic-exposure contract. Any future buffer reuse keeps writable storage
private and raw until writes are enqueued.

### Public Truth-Replacing Timing Jitter

Timing jitter no longer produces a replacement `Photoelectrons` field. It is a
private optional charge stage after dark-count seed construction. Truth remains
unchanged and may be returned exactly beside derived charge.

### Earlier Sampling Representations And Semantic-Coordinate RNG

The Stage 2 count-only sample axis plus collection-level `SampleGrid` sidecar
was superseded by Stage 3's timestamp-string `SampleAxis` plus
`SamplingConfig`. Maintenance 5 supersedes that second representation with a
compact regular integer `SampleAxis` and source-derived sampling preparation.
Coordinate-addressed random schemes remain retired: tensor-position indices,
not semantic coordinate values, drive RNG.

### Bare `seed=`, Central `_RngStream`, And TensorDSLab-Owned Generic RNG

Stage 5/6 correctly implemented the then-accepted private design and remains
closed implementation evidence. The Maintenance 2 implementation
supersedes its bare private producer/invocation seed, central stream enum, and
generic `readout/_random.py` ownership with required `CounterRng`, config-owned
`RngKey`, and TensorCore-owned generic counter/distribution mechanics. It
preserves default-key output continuity and removes the retired local surfaces
without aliases. Maintenance 2 is Merged / Closed at the exact candidate and
Design closeout recorded above.

### Separate Avalanche Architecture Attempts

Same-bin recursive closure, causal-scan, generation-wave, and recovery-marked
alternatives are superseded as implementation directions. The fixed-generation
model in `architecture/rebuild.md` is the only active baseline.

### Normal Delay Is Retired From The MVP

The active crosstalk config union is exactly
`FixedDelayConfig | ExponentialDelayConfig`. The earlier zero-clipped
`NormalDelayConfig` proposal is retired: its negative latent tail creates a
calibration-sensitive prompt atom, it has no IV-parity basis, and carrying its
Gaussian CDF/tail machinery is disproportionate for the first Charge
implementation. This does not select a truncated, folded, lognormal, or
tabulated replacement. A later calibrated distributed family requires a new
scientific decision and an explicit new config type.

Stage 3 historically implemented and exported `NormalDelayConfig`. Because
TensorDSLab is pre-deployment and makes no backward-compatibility claim, the
completed Stage 6 slice removed that class, its union memberships, all export
layers, and its tests completely, without a compatibility shim. The closed
Stage 3 work order remains unmodified historical evidence.

### Fixed And Exponential Delay Preparation Is Frozen

The independent-edge latent-uniform phase closure has exact prepared kernels
for the two accepted MVP delay families. Fixed delay uses its exact represented
rational position in the sample period to produce at most two adjacent offset
masses, with analytic source-relative overflow and no delay RNG. Exponential
delay uses analytic phase-marginalized categories and right tails, stable
binary64 central-mass evaluation, and no cutoff, clipping, residual assignment,
or renormalization.

The initial exponential domain is
`2**-52 <= mean_delay / sample_period <= 2**52` with
`2 <= sample_count <= 8192`. A configured AP recovery constant has the same
ratio domain. Exponential delay and integrated recovery use a `1e-12` local
absolute tolerance and `1e-11` complete-law L1 tolerance against independent
high-precision oracles. The exact formulas, operation branches, Taylor
coefficients, overflow construction, and failure rules in
[`architecture/rebuild.md`](architecture/rebuild.md) are part of the frozen
mapping, not implementation suggestions.

Afterpulse recovery remains charge-only. It uses the exact difference between
the ordinary exponential category and a scaled effective-mean category, but
preparation evaluates the corresponding log ratio stably and obtains the
conditional response with `-expm1`. It never clips a response into range,
changes the realized AP destination or count, or enters recursive state.

## Open

### IV-DSLab Charge Equivalence Margins

Stage 6 validates the selected TensorDSLab probability model but does not
establish IV-DSLab equivalence for intentionally divergent mechanisms. Stage 7
implements and validates the public requested-`Charge` composition contract,
but it adds no donor comparison or equivalence margin. Each donor claim still
requires a named observable and a collaborator- or calibration-owned
acceptance margin under [`parity.md`](parity.md).

### Waveform-Tail Optimization Evidence

A later measured optimization stage may instrument product-local fusion and
evaluate a purpose-built fallback kernel after the functional producers exist.
It must freeze the compiler/execution mode, representative shapes, profiler and
memory evidence, and equivalence to the accepted eager reference.
Cross-product fusion remains excluded without a focused Design change.

### Durable Digitization Association

An in-process caller retains the exact `DigitizedWaveformConfig` separately.
Before digitized values can be independently transported or persisted, Design
must define how the configuration/calibration required to interpret them is
durably associated.

### TensorG4DS Truth Bridge

The bridge still needs an exact upstream type and commit, provenance origin,
example/channel mapping, numeric half-open binning, exact edge behavior,
underflow/overflow diagnostics, dtype/device matrix, and gradient policy.

### Optional Collection Convenience Properties

Thin typed lookup properties may be added only if collaborator evidence shows
they materially improve use over `readout.field(ProductType)`.

### Persistence, Compatibility, And Deployment

Cache/artifact formats, compatibility targets, release readiness,
backward-compatibility policy, deployment, conformance, Coordination, and
Profile B remain separate future decisions.
