# IV-DSLab Parity And Intentional Divergences

Status: active Design contract for evaluating IV-DSLab and DSLab donor
behavior. No production parity implementation or fixture corpus has been
dispatched.

## Purpose

This page defines what TensorDSLab means by parity with IV-DSLab. It records
which legacy behaviors can be preserved exactly, numerically,
distributionally, or statistically; which behaviors intentionally change on
the post-binned tensor-native path; and which decisions remain deferred.

Parity is always scoped to a named comparison boundary and observable. It does
not mean that TensorDSLab should reproduce IV-DSLab package structure, global
state, RNG streams, sparse PE-row growth, condition-database loading, fixed
array rank, or incidental implementation defects.

The target behavior remains authoritative in
[Post-Binned Readout Architecture](architecture/readout.md). This page explains
how that target relates to donor behavior; it does not independently create or
override a TensorDSLab scientific contract.

## Authority And Interpretation

Use this precedence when sources disagree:

1. `AGENTS.md` and `CONTRIBUTING.md` govern repository workflow, ownership, and
   engineering constraints.
2. Accepted TensorDSLab decisions and active architecture pages govern
   TensorDSLab collection, field, and scientific behavior.
3. This page governs the classification and evidence required for donor-parity
   claims and intentional divergences.
4. Focused production work orders define stage scope and must name the parity
   target they implement; they may not silently override durable contracts.
5. `docs/validation.md`, TensorDSLab-owned fixtures, and tests operationalize
   accepted contracts. They do not create a new scientific model by accident.
6. TensorCore documentation is authoritative for generic tensor mechanics.
7. TensorML is an integration and workflow reference.
8. DSLab and IV-DSLab are donor evidence only. Their behavior is not
   authoritative merely because it existed first.

If TensorDSLab documents conflict, return the contradiction to Design. Do not
select whichever donor function or fixture is easiest to reproduce.

## Audited Donor Baseline

Audit date: 2026-07-09.

The current comparison was made against these local snapshots:

- IV-DSLab checkout: `Projects/iv-dslab-main_db_PB`. The available checkout
  has no Git metadata, so claims are pinned to the folder name plus exact source
  paths and symbols. A future donor replacement must be re-audited.
- DSLab checkout: `Projects/dslab` at commit
  `5c3b32608aa9c23301eff1a72f27d4f3e8890469`.

Key IV-DSLab files are pinned by SHA-256 because that checkout lacks a commit:

| Donor file | SHA-256 |
| --- | --- |
| `src/dselec/sipm.py` | `8b65d5b3a243b998390d2d26937043f27d0f1295439e39f6553d9ada03827af4` |
| `src/dselec/daq.py` | `212ef1ee55cde78250c91c2c7eb5040c8b20c5232ed75027281a10ff7aab8570` |
| `src/dselec/waveform.py` | `5eb5b29e6958184e520b2151877a678f6d98cdbe6e53cbf9d1b4c4e64e0f82b5` |
| `src/dsutils/general_tools.py` | `2724b10d61441d4cd0e3a6065d200b0333b294371cd34fd150894357208a65b0` |
| `exe/daq_slices.py` | `ab82d4c67d800a63c0387eff17dfe4a90bc372e905320a819ff825deb0800430` |
| `data/config_files/dselec.ini` | `fd42244bb4405dc328496efb8043fff522584a1922b811246670ac0e940e1c64` |

Primary IV-DSLab source locations are:

| Area | Donor source |
| --- | --- |
| effect order and orchestration | `exe/daq_slices.py:process_event`, `src/dselec/sipm.py:add_noise_pes` |
| timing jitter | `src/dselec/daq.py:add_daq_jitter` |
| dark counts | `src/dselec/sipm.py:_add_dcr`, `_add_dcr_db` |
| crosstalk and recursive correlated noise | `src/dselec/sipm.py:poissonian_loop`, `_add_phct`, `_add_dict`, `_add_corr_noise` and DB variants |
| afterpulses | `src/dselec/sipm.py:_add_ap`, `_add_ap_db` |
| charge smearing | `src/dselec/sipm.py:_add_charge_smearing_db` |
| pulse template, sub-bin correction, and waveform composition | `src/dselec/waveform.py:_feb_snr_template`, `spe_template`, `create_waveforms`, `_create_veto_waveforms*` |
| convolution utility | `src/dsutils/general_tools.py:convolve` |
| white and FFT baseline noise | `src/dselec/waveform.py:create_noisy_baselines*`, `_get_precomputed_baselines`, `_create_baselines_from_spectrum` |
| analog gain and ADC mapping | `src/dselec/daq.py:apply_analog_gain`, `voltage_to_adc` |

Primary DSLab fixed-grid donor locations at the pinned commit are:

| Area | Donor source |
| --- | --- |
| effect order | `dslab/domain/readout/kernels/charge/__init__.py:build_charge_array` |
| timing, dark counts, crosstalk, afterpulses, and smearing | `dslab/domain/readout/kernels/charge/timing.py`, `dark_counts.py`, `crosstalk.py`, `afterpulses.py`, `smearing.py` |
| pure, noise, and analog waveform composition | `dslab/domain/readout/kernels/waveforms/pure.py`, `noise.py`, and the historically named donor module `physical.py` |
| digitization | `dslab/domain/readout/kernels/digitization.py` |

DSLab is useful as a reviewed fixed-grid translation donor, especially its
`dslab/domain/readout/kernels/charge/` and
`dslab/domain/readout/kernels/waveforms/` modules. DSLab fixtures prove DSLab's
accepted fixed-grid behavior; they do not by themselves prove IV-DSLab parity.

The audited IV tests are primarily functionality, shape, dtype, and loose
amplitude smoke tests. They are supporting evidence, not a complete scientific
parity suite. Source behavior, analytic derivations, and new TensorDSLab-owned
fixtures must anchor production claims.

## Parity Vocabulary

These classifications apply only when discussing a donor comparison. Ordinary
contract phrases such as “exact field ID,” “exact target-field identity,” or
“exact output destination” are not parity claims.

| Classification | Meaning | Minimum evidence |
| --- | --- | --- |
| **Exact parity** | The declared deterministic observables are identical over the stated input/config domain. A claim must say explicitly whether exact means bitwise identity or equality after a representation mapping. | Reviewable fixtures covering boundaries, dtype/quantization rules, and exact assertions. |
| **Numerical parity** | The same deterministic mathematical model is implemented, but discretization, dtype, reduction order, or backend arithmetic permits bounded differences. | Named observables, units, absolute/relative tolerances, and the reason for each tolerance. |
| **Distributional parity** | The stochastic law or probability kernel is the same under explicit assumptions, although individual RNG draws and sampled outputs differ. | Analytic derivation or probability-kernel fixture plus finite-sample corroboration where useful. |
| **Statistical parity** | The complete stochastic laws may differ, but named scientifically relevant statistics agree within accepted bounds. | Named statistics, sample size, seeds/replicates, confidence or error thresholds, and tail/correlation checks appropriate to the model. |
| **Intentional divergence** | TensorDSLab deliberately chooses different observable behavior. | The donor behavior, replacement behavior, reason, scientific risk, validation target, and revisit trigger. |
| **Deferred** | TensorDSLab has not accepted a parity target or replacement behavior yet. | The unresolved decision and the stage that must resolve it. |
| **Not applicable** | The donor behavior cannot or should not be represented at the accepted TensorDSLab boundary. | The lost information or ownership reason that makes comparison inapplicable. |

Distributional parity is stronger than finite-sample statistical agreement.
A sampled golden output cannot prove distributional parity. Conversely, two
implementations may pass selected statistical checks without sharing the same
law; such a result must remain classified as statistical parity.

Because TensorDSLab has no scientific-transform implementation or parity
fixture corpus yet, classifications in this page are Design targets or
accepted divergence labels, not claims that a test suite has already
demonstrated parity. A future work order must supply the required evidence
before reporting a target as achieved.

## Required Shape Of A Parity Claim

Every production work order or fixture that claims parity must name:

```text
reference:
comparison boundary:
dependency field IDs:
result field ID:
ReadoutCollection field subset (context only):
private ephemeral observable (if no public field boundary):
input/config domain:
classification:
assumptions:
declared observables:
acceptance criteria:
accepted exclusions:
intentional divergences:
source/fixture provenance:
revisit triggers:
```

Claims are transform-specific and composition-specific. Distributional parity
for one isolated transform does not imply parity for the full chain when the
operation order or upstream population differs.

## ReadoutCollection Presence And Parity Claims

The primary in-memory container is one partially materialized
`ReadoutCollection`, but parity never attaches to the container merely because
a field is present. A public transform claim attaches to its exact recognized
dependency field ID or IDs and exact result field ID; the dependency-ID set is
empty for noise rendering, whose dependencies are the common layout and sample
grid. A private `simulate_charge` submodel claim instead names its ephemeral
working-grid boundary and observable and must not invent a field ID. Every claim
also names the relevant canonical-axis layout, `SampleGrid`, conditional
field-specific spec, configuration, comparison boundary, and observable.

A collection may contain any nonempty canonical subset of recognized fields.
Transform-generated snapshots retain unaffected source fields, replace or add
the target, and remove stale descendants according to the readout architecture.
Those retained fields are context, not additional parity results. Unless a
claim explicitly covers a composition, its acceptance criteria apply only to
the named result field.

Explicit projection or field removal does not invalidate a retained descendant.
A projected analog or digitized field therefore remains a valid collection
member without its dependencies, but its presence alone records neither
lineage nor parity evidence. A new analog-composition claim still requires both
`readout.waveform.pure` and `readout.waveform.noise`; a digitization claim
requires `readout.waveform.analog`.

## Workspace, Builder, And Scheduling Neutrality

Functional execution, caller-supplied `out`, and exact
`ReadoutWorkspace`-backed execution are runtime realizations of the same public
transform contract. Result-buffer builders, workspace storage, private
ping-pong scheduling, lease state, CUDA stream placement, allocation reuse, and
target publication mechanics are nonsemantic execution details. They do not
create a donor comparison boundary, strengthen or weaken a parity
classification, or justify different field values for the same accepted
input/config/RNG coordinates and backend contract.

For a public result, all accepted execution modes must agree at the exact,
numerical, distributional, or statistical level already required for that
TensorDSLab operation. Comparing functional and workspace-backed outputs is
TensorDSLab backend/execution agreement, not a new IV-DSLab parity claim. A
workspace schedule must preserve the documented scientific operation order,
coordinate-addressed random field, edge policies, frozen post-dark
crosstalk/afterpulse source, and complete target overwrite. If scheduling or
buffer reuse changes a named scientific observable, that is an implementation
defect or an architecture change requiring Design review, not an acceptable
allocator-dependent tolerance.

Private dark-count, crosstalk, afterpulse, or smearing observables may be backed
by reusable workspace tensors for focused validation. They remain ephemeral
comparison aids: workspace storage never promotes them to recognized field IDs,
public products, durable labels, sidecars, or lineage records. Reusing scratch
must not alter a previously returned public output; explicit caller reuse of an
`out` destination is an output-lifetime decision, not a parity operation.

Preflight atomicity, alias rejection, leases, stream safety, allocation
instrumentation, and failure recovery belong to contract validation rather than
donor parity. In particular, the warmed steady-state absence of
TensorDSLab-owned target/scratch storage allocation is a performance/resource
claim. Backend allocator requests, library-private scratch and plans, and
rollback after asynchronous failure are diagnostic or deferred behavior and do
not enter a scientific parity acceptance criterion.

## MVP Deviation Policy

It is acceptable for the MVP to choose a simpler tensor-native algorithm when
a more literal IV-DSLab translation would disproportionately impede progress.
That exception is valid only when:

- the replacement preserves the accepted TensorDSLab collection and field-role
  semantics and ownership boundaries;
- the unavailable or changed donor information is stated explicitly;
- critical means, variances, correlations, tails, rates, edge losses, waveform
  features, or transfer functions are either preserved or named as changed;
- validation criteria and tolerances are defined before production acceptance;
- the deviation does not masquerade as exact or distributional parity;
- the scientific risk and a concrete revisit trigger are recorded;
- Design accepts the deviation before it becomes a public or durable contract.

Implementation difficulty alone is not sufficient when a change would create
an unbounded or unmeasured scientific bias. Apparent donor bugs, unsafe casts,
global-state coupling, and representation accidents should not be preserved
merely to improve literal output agreement.

A divergence justified primarily by implementation cost is provisional unless
Design separately accepts it as the desired scientific model. Its revisit
trigger should include evidence that downstream conclusions are sensitive to
the omitted behavior or that a practical tensor-native formulation has become
available.

## Post-Binned Parity Envelope

IV-DSLab operates primarily on sparse PE rows. The first TensorDSLab MVP starts
from the `readout.photoelectrons` field in a `ReadoutCollection`. This field is
a dense grid of binned photon-origin primary PE seeds; it is not electrical
charge and does not already include dark counts, crosstalk avalanches, or
afterpulses. The representation has discarded:

- each PE's exact sub-bin time;
- individual seed identity and any source-row ordering;
- any source-side weight beyond the accepted unit primary-seed count.

The MVP also deliberately does not carry donor global RNG state/draw order or
implicit condition-database and channel-map state.

Consequently, TensorDSLab does not claim end-to-end eventwise parity with
IV-DSLab for the first MVP. Legitimate comparisons use one of three explicitly
different boundaries:

1. the public `readout.photoelectrons -> readout.photoelectrons` boundary for
   timing redistribution from a common binned primary-seed grid;
2. the public `readout.photoelectrons -> readout.charge` boundary for
   end-to-end `simulate_charge` statistics; or
3. a private ephemeral count-grid boundary used only to validate an internal
   dark-count, crosstalk, afterpulse, or smearing submodel.

The third boundary does not accept another collection field or product. Its
count grids are test observables or scratch representations only. Comparisons at
any boundary may use a marginalized ensemble when TensorDSLab explicitly models
lost variables such as sub-bin phase and states the assumed distribution.

The common default for unknown source-bin phase is:

```text
U ~ Uniform(0, sample_period)
```

That assumption can establish conditional distributional parity for a binned
transition kernel. It cannot reproduce eventwise IV behavior when the original
PE phases were known or were not conditionally uniform. Nor does an isolated
private-submodel result establish parity for the public `simulate_charge`
composition.

## Full-Chain Order

The audited IV-DSLab flow is approximately:

```text
sparse source PEs
  -> dark-count PE rows
  -> recursively expanding correlated-noise PE rows
  -> optional per-PE DB charge smearing
  -> Gaussian jitter of every resulting PE time
  -> binning with sub-bin amplitude correction
  -> convolution + baseline noise + saturation + ADC conversion
```

The accepted TensorDSLab MVP comparison begins after a future
TensorG4DS-to-TensorDSLab bridge has produced the binned input. Native G4DS and
TensorG4DS behavior are outside this readout parity boundary. The MVP flow is:

```text
readout.photoelectrons (binned photon-origin primary PE seeds)
  -> timing redistribution -> readout.photoelectrons
  -> simulate_charge
       -> private dark-count addition
       -> private frozen source snapshot
            -> private first-generation crosstalk contribution
            -> private first-generation afterpulse contribution
       -> private aggregate charge smearing
  -> readout.charge
  -> pure/noise/analog waveform fields
  -> optional safe digitization
```

Effect-order classification: **intentional divergence**.

The public `simulate_charge` result may target **statistical parity** for named
charge observables over an accepted input/config ensemble. TensorDSLab must not
claim distributional or eventwise parity for that composed stochastic flow.
Passing isolated internal-submodel checks is necessary evidence, but it does not
prove the end-to-end target. Any campaign-level comparison likewise remains
statistical validation of named observables, not evidence that the composed
stochastic laws are identical.

## Summary Matrix

| Surface | MVP target classification | Core qualification |
| --- | --- | --- |
| package, collection/field, rank, and TensorCore representation | not applicable | TensorDSLab is a clean-slate tensor-native architecture. |
| full stochastic effect order | intentional divergence | IV jitters after recursive noise and optional smearing; TensorDSLab jitters primary seeds first and uses bounded private contributions inside `simulate_charge`. |
| end-to-end `simulate_charge` | statistical | Compare named aggregate `readout.charge` observables from common `readout.photoelectrons` ensembles; internal divergences preclude equality of the complete law. |
| timing jitter in isolation | distributional, conditional | Requires iid uniform latent sub-bin phases, matching Gaussian width, sample grid, and drop policy. |
| dark counts in complete equal-width cells | distributional, conditional | Follows from Poisson splitting for homogeneous rates; DB variation and edge/order effects are excluded. |
| crosstalk | intentional divergence | IV uses recursive branching; TensorDSLab uses one bounded Poisson contribution. |
| correlated-noise recursion | intentional divergence | TensorDSLab forbids generated contributions from feeding other effects in the MVP. |
| afterpulse delay law | intentional divergence | TensorDSLab accepts an ordinary exponential mean-delay model instead of IV's literal reciprocal-exponential expression. |
| afterpulse recovery amplitude | deferred | Unit-count versus recovery-weighted amplitude and its ordering remain open. |
| charge smearing | statistical, with a conditional distributional subcase | Aggregate Gaussian equality holds only for equal independent unit charges before clipping. |
| FEB-SNR pulse template and convolution | numerical | Same functional family, but normalization and binned-photoelectron edge/sub-bin behavior differ. |
| eventwise IV sub-bin amplitude correction | not applicable | True PE phase is absent after binning. |
| omission of a phase-marginalized amplitude correction | intentional divergence | A latent-phase expectation is possible, but the first MVP does not apply one. |
| exact zero baseline | exact | Both can produce an all-zero noise waveform. |
| arbitrary constant baseline | not applicable | This is an explicit TensorDSLab model, not audited IV emulation. |
| white Gaussian noise | distributional | Same iid law when RMS matches; RNG streams intentionally differ. |
| FFT noise | statistical | Preserve RMS/spectral intent, not IV bank/crop traces or their exact distribution. |
| analog `pure + noise` composition | exact, conditional | Exact algebraically when inputs, units, sign, shape, and clipping policy match. |
| in-range ADC mapping | exact after representation mapping, conditional | When the mapped floating value and transfer parameters match, both truncate to the same integer code; TensorDSLab stores it in signed `torch.int32` rather than donor `uint16`. Floating-dtype/backend differences are numerical, not covered by this exact claim. |
| out-of-range ADC behavior | intentional divergence | TensorDSLab clips before integer conversion instead of preserving IV unsigned wraparound. |
| fixed-seed or bitwise IV RNG streams | not applicable | TensorDSLab uses coordinate-addressed tensor-native random fields. |
| per-channel condition DB variation | deferred | Requires a typed TensorDSLab channel-parameter boundary. |

## Timing Jitter

### IV Timing Behavior

`src/dselec/daq.py:add_daq_jitter` adds an independent Gaussian draw to every
sparse PE time. Because the true PE time is available, IV preserves the actual
sub-bin phase before jitter. The executable applies jitter after dark counts,
correlated-noise expansion, and optional DB smearing.

### TensorDSLab Timing Behavior

The public transform consumes and replaces `readout.photoelectrons`; it does not
create charge or any avalanche/count product. TensorDSLab receives only source
sample `s`. For sample period `T`, latent phase `U`, and jitter `J`:

```text
U ~ Uniform(0, T)
J ~ Normal(0, sigma)
target = s + floor((U + J) / T)
```

The target tensor-native implementation will sample aggregate target/drop
bucket counts rather than materializing one PE row per quantum.

### Timing Parity Claim

Classification: **conditional distributional parity in isolation**.

The binned target law is the same as individually jittering PEs only when:

- conditional source-bin phases are independent and uniform;
- jitter draws are independent with the same Gaussian width;
- the sample period, window, and boundary convention match;
- out-of-window targets are dropped in both comparisons;
- the comparison ignores the different position of jitter in the full chain.

If IV source phases are known and nonuniform, TensorDSLab's result is a
statistical approximation to the marginalized binned behavior, not
distributional parity for that event.

Validation should compare transition probabilities, the explicit drop bucket,
conditional mean/variance of bin displacement, and count conservation including
drops. It should not compare same-seed PE destinations.

## Dark Counts

### IV Dark-Count Behavior

The non-DB path samples a total Poisson count over gate duration and channel
count, then assigns uniform times and uniformly selected channels. The audited
checkout uses the veto channel map in this path despite commented TPC lines.
The DB path samples a separate Poisson total using each channel's configured
rate and assigns uniform times within the gate.

### TensorDSLab Dark-Count Behavior

Inside `simulate_charge`, the first TensorDSLab model adds an independent count
to each eligible channel/sample cell of a private working grid:

```text
count ~ Poisson(rate_hz * sample_period_ns * 1e-9)
```

The MVP uses one global rate. Typed per-channel rates are deferred. The
resulting grid is an ephemeral submodel observable, not a recognized
`ReadoutCollection` field.

### Dark-Count Parity Claim

Classification: **conditional distributional parity**.

Poisson splitting makes the IV total-count/uniform-placement model equivalent
to independent per-cell Poisson counts when rates are homogeneous and all
cells represent complete equal channel-time exposure. The claim excludes:

- per-channel DB variation;
- partial final samples or mismatched gate conventions;
- the different ordering relative to timing jitter;
- later recursive branching, smearing, and boundary filtering.

For a stationary homogeneous Poisson process, independent timing displacement
preserves the interior rate. Finite-window edge behavior can still differ
because IV jitters generated dark-count PEs and TensorDSLab generates dark
counts after source timing redistribution.

Validation should compare expected counts per channel-time exposure, Poisson
variance, zero-count probability, cell covariance, and edge-cell behavior.

## Crosstalk And Correlated-Noise Recursion

### IV Crosstalk And Recursion Behavior

IV direct and photon crosstalk use recursive `poissonian_loop(p * charge)`
branching. The returned count is total descendants, not a single first
generation. For a unit source and subcritical coefficient `p`, the recursive
mean is `p / (1 - p)`, not `p`.

IV then processes a growing PE queue. Photon crosstalk, direct crosstalk, and
afterpulse outputs can seed other effects; dark counts participate in that
queue. Same-type guards and internal loop unrolling limit some paths, but the
overall model still contains cross-effect recursive feeding.

### TensorDSLab Crosstalk Behavior

For source count `n`, the first TensorDSLab model samples one same-channel,
same-sample first-generation contribution:

```text
contribution ~ Poisson(mean_additional_counts_per_source * n)
```

Crosstalk and afterpulse contributions read one frozen post-dark snapshot and
are added once inside `simulate_charge`. Generated counts do not become new
sources. The snapshot and contributions are private scratch representations,
not collection products.

### Crosstalk Parity Claim

Classification: **intentional divergence**.

The TensorDSLab coefficient is a Poisson mean coefficient, not IV's recursive
branching parameter. Reusing the same numeric value does not establish mean,
variance, or tail parity. A future calibration intended to approximate IV must
explicitly define the parameter mapping and the moments/tails it preserves.

The MVP accepts bounded shape and nonrecursive execution to keep behavior
reviewable and tensor-native. Validation should prove the TensorDSLab law and
shared-snapshot rule at the private submodel boundary. Statistical comparison
with IV may report multiplicity mean, variance, zero probability, and
upper-tail quantiles, but it must remain labeled divergence unless Design
accepts a calibrated approximation target. Those diagnostics do not create a
public crosstalk field or transform.

## Afterpulses

### Literal IV-DSLab Behavior

For each evaluated PE, IV attempts one Bernoulli fire with probability
`ap * pe.charge`. As written, the delay expression is:

```text
delay = 1 / np.random.exponential(1 / ap_tau)
```

This is a reciprocal-exponential expression, not an ordinary
`Exponential(mean=ap_tau)` draw. Treating `ap_tau` as the mean of a standard
exponential is an inference about intended physics or a correction of an
apparent legacy defect, not literal source parity.

The audited code does not define a recovery policy when `ap * pe.charge`
exceeds one and relies on the underlying binomial call to reject an invalid
probability. TensorDSLab instead treats fire probability as an explicitly
validated scientific parameter applied to integer source quanta.

IV assigns recovery-weighted charge:

```text
charge = 1 - exp(-delay / recovery_tau)
```

Afterpulse rows also participate in the growing correlated-noise queue; the
self-recursion guard is commented out in the audited source.

### TensorDSLab Direction

The current architecture accepts bounded first-generation firing with an
ordinary exponential delay parameterized by `mean_delay_ns`, latent uniform
source-bin phase, an explicit out-of-window drop bucket, and no ragged PE rows.
The standard exponential law is a deliberate scientific interpretation and
correction, not literal reproduction of the audited IV expression. Its working
counts or amplitudes remain private to `simulate_charge`.

### Afterpulse Parity Claim

Delay-law classification: **intentional divergence**.

Recovery-amplitude classification: **deferred**.

TensorDSLab must resolve all of the following before implementation:

- unit-count afterpulses versus a typed recovery-amplitude stage;
- ordering relative to charge smearing;
- the accepted statistical comparison after recursive feeding is removed.

The ordinary exponential delay and removal of recursive feeding are already
recorded intentional divergences. If the MVP accepts unit counts, that choice
must be added as another divergence. If it accepts recovery-weighted amplitude,
recursion and effect-order differences will still prevent full distributional
parity.

## Charge Smearing

### IV Charge-Smearing Behavior

The audited DB path multiplies each PE, including generated and fractional
PEs, by a channel-dependent draw:

```text
q_i' = q_i * Normal(1, sigma_channel)
```

IV does not clip negative values at this stage. The non-DB path does not apply
the same explicit smearing implementation.

### TensorDSLab Charge-Smearing Behavior

As the terminal private submodel inside `simulate_charge`, aggregate integer
cell count `n` and one configured width produce the public `readout.charge`
value:

```text
draw = Normal(n, sqrt(n) * sigma)
amplitude = max(draw, 0)
```

### Charge-Smearing Parity Claim

Classification: **statistical parity**, with a **conditional distributional
subcase**.

Before clipping, the aggregate draw has exactly the same probability
distribution as summing `n` independent `Normal(1, sigma)` unit charges. That
identity does not hold for heterogeneous `q_i` or channel widths: the donor
variance is proportional to `sum(q_i**2)`, not simply `n`. Recovery-weighted
afterpulses are therefore outside the distributional subcase. TensorDSLab's
zero clipping also changes the negative tail and mean.

Validation should compare conditional mean, variance, zero-count behavior,
negative-tail probability before policy, post-clipping bias, and high/low count
regimes at the private-count-to-public-charge boundary. Per-channel variation
requires a later typed parameter contract. The private integer input does not
become a recognized collection field, and `readout.charge` denotes a floating
aggregate PE-equivalent response rather than SI charge.

## End-To-End `simulate_charge`

The public comparison boundary is:

```text
readout.photoelectrons -> simulate_charge -> readout.charge
```

Classification: **statistical parity for named observables**.

This claim compares a common ensemble of binned photon-origin primary PE seeds
to the aggregate PE-equivalent charge response. It does not expose or require
the donor's sparse generated-PE rows, and it does not assert that the complete
joint law matches. The different effect order, nonrecursive crosstalk,
first-generation afterpulsing, aggregate smearing, negative clipping, and RNG
construction preclude an end-to-end distributional claim for the MVP.

Validation must exercise the public composition as well as the private
submodels. At minimum it should compare per-channel and per-sample charge mean
and variance, zero-cell probability, total response, occupancy, edge loss,
selected tail quantiles, and any accepted time-profile statistic over named
input/config ensembles. Each internal dark-count, crosstalk, afterpulse, and
smearing check must retain its own classification; passing them separately does
not prove this end-to-end statistical target. Conversely, a passing
`simulate_charge` ensemble does not upgrade an intentionally divergent internal
submodel to distributional parity.

Fixtures may retain private intermediate grids as test-only diagnostics when
needed to localize a failure. They must not record those grids as recognized
field IDs, public products, durable producer labels, or required collection
sidecars.

## Pure Waveform Rendering

### IV Pure-Waveform Behavior

IV's FEB-SNR template uses:

```text
h(t) = exp(-t / (tau_fall + tau_rise)) - exp(-t / tau_rise)
```

It normalizes by the analytic continuous-time maximum, applies a sub-bin
amplitude correction `exp(-sample_skew / tau_b)` to each PE before binning,
uses a pulse-length prebuffer, performs causal convolution, inverts the signal,
and removes the prebuffer.

### TensorDSLab Pure-Waveform Behavior

TensorDSLab uses the same template family but normalizes by its sampled maximum,
performs causal full convolution truncated to the input sample count, and
applies explicit gain and sign. `readout.waveform.pure` remains signal-only;
baseline is owned by `readout.waveform.noise`.

### Pure-Waveform Parity Claim

Classification: **numerical parity** for the aligned in-window template and
convolution domain.

The sampled-maximum and analytic-normalization choices are close but not
identical; for the audited IV default parameters their peak normalization
differs by roughly 66 parts per million. The production work order must lock a
fixture and tolerance rather than rely on that audit estimate.

IV's eventwise sub-bin amplitude correction is **not applicable** at the
post-binned boundary because true PE phase has been discarded. Loss of the
eventwise phase does not make every statistical correction impossible: under
an accepted latent phase `U ~ Uniform(0, T)`, TensorDSLab could derive and apply
the expected correction factor for the IV response model. The first MVP omits
both eventwise and marginalized correction, which is an **intentional
divergence** rather than an impossibility claim.

IV's internal prebuffer/crop indexing is not adopted as a TensorDSLab field or
collection contract; aligned impulse-response fixtures must establish the
intended boundary behavior. TensorDSLab must not call its complete post-binned
pure waveform exact IV parity. If the omission creates an unacceptable peak,
area, or energy bias, return to Design for a phase-marginalized tensor
formulation.

Validation should compare sampled template values, peak, area, time-to-peak,
impulse response, same-length truncation, sign/gain, and boundary behavior.

## Noise Waveforms

### Zero And Constant Baseline

IV returns exact float32 zeros when noise is disabled. TensorDSLab can provide
**exact parity** for an all-zero `readout.waveform.noise` result field over the
declared shape after any explicit dtype mapping is accounted for.

An arbitrary nonzero constant baseline is **not applicable** as an IV-emulation
claim. It is a useful explicit TensorDSLab model.

### White Noise

IV non-DB white noise is iid Gaussian with RMS `pe_amplitude / snr`. The DB path
generates iid standard normal values and later scales by per-channel RMS.
TensorDSLab generates a coordinate-addressed per-sample Gaussian field.

Classification: **distributional parity** when mean, RMS, dtype domain, and
independence assumptions match. Global RNG state, sequential draw order, and
same-seed sample values are intentionally excluded. TensorDSLab additionally
requires reorder and chunk invariance.

Validation should compare mean, RMS, marginal normality, sample/channel
covariance, and coordinate-invariance properties. A fixed white-noise array is
a reproducibility fixture, not proof of distributional parity.

### FFT Noise

IV zeros DC, adds global random phases, synthesizes long inverse-FFT baselines,
normalizes each long baseline to unit standard deviation, persists or loads a
bank, and randomly crops per-channel segments.

TensorDSLab directly synthesizes the exact requested sample length from an
explicit one-sided spectrum using semantic phases and explicit normalization.
It does not reproduce the bank/crop process.

Classification: **statistical parity**.

Validation should compare RMS, integrated power, one-sided PSD shape,
autocorrelation, DC/Nyquist policy, marginal distribution, and any accepted
cross-channel independence. It should not compare trace identity or infer that
matching PSD alone proves equality of the full waveform distribution.

Remote downloads, implicit spectrum loading, persistent baseline banks, and
random crop mechanics are intentionally rejected architecture.

## Analog Waveform Composition

IV adds baseline noise to the inverted signal waveform and then applies its
analog saturation policy before ADC conversion. IV does not expose separate
pure, noise, and analog field roles. Historical DSLab code and fixtures use
“physical waveform” terminology, including the donor filename
`dslab/domain/readout/kernels/waveforms/physical.py`; that is donor provenance,
not the TensorDSLab target name.

TensorDSLab defines:

```text
analog = optional_clip(pure + noise)
```

Pure and noise are signal-only and noise-only components at the same analog
reference plane and in the same voltage units. `readout.waveform.analog` is
their composed pre-digitization voltage waveform; the component fields do not
claim to be separate Tile, PDU, or DAQ hardware-boundary outputs.

Classification: **conditional exact parity** for the algebraic composition
when pure/noise values, units, polarity, shape, and clipping threshold are
identical. Upstream differences in pulse rendering or noise generation remain
upstream differences; exact composition does not imply exact end-to-end
waveforms.

Separating the three semantic field roles inside `ReadoutCollection` is an
architectural difference for which donor parity is not applicable.

## Digitization

### IV Digitization Behavior

IV applies analog gain, maps the configured voltage interval to
`[0, 2**bits - 1]`, casts immediately to `uint16`, and only afterward clips in
waveform callers. Its gain-range condition uses an impossible conjunction and
does not validate the advertised range correctly.

Values already inside the voltage range use truncation toward zero after
mapping into a nonnegative interval. Values outside the interval can wrap
during unsigned conversion; later clipping cannot reconstruct the intended
saturation.

### TensorDSLab Digitization And Parity Claim

TensorDSLab digitizes `readout.waveform.analog` into the distinct
`readout.waveform.digitized` ADC-count field. “Digitized” is retained because
the field is specifically the ADC result, not an arbitrary later digital or
firmware-processed waveform.

For in-range values, the target is **exact parity after representation
mapping**. TensorDSLab uses `AdcQuantization.TRUNCATE`, accepts bit depths from
1 through 16, and stores the resulting nonnegative ADC code in signed
`torch.int32`. IV stores its code in `uint16`; dtype identity is not the parity
claim, but equal integer code values are required when the mapped floating
value, gain, voltage range, offset, and bit depth match. Differences introduced
earlier by accepted floating dtype or backend arithmetic use numerical parity
with Stage 3 tolerances and boundary fixtures.

For out-of-range values, classification is **intentional divergence**.
TensorDSLab gains and clamps analog voltage before quantization, validates gain,
range, bit depth, and dtype, and forbids unsigned-wrap accidents. It enforces
the donor's intended inclusive analog-gain range of `[0, 40]` dB, correcting
IV's impossible conjunction that failed to reject out-of-range gain.

Validation should cover exact lower/upper endpoints, half-step and near-step
values, negative and over-range inputs, gain boundaries, all accepted bit
depths, and ADC bounds.

## RNG Donor Parity And Backend Agreement

TensorDSLab does not target fixed-seed or bitwise RNG-stream parity with
IV-DSLab. The algorithms consume different representations, group draws
differently, use different operation order, and require coordinate-addressed
device-resident streams.

The accepted RNG targets are:

- exact repeatability for identical TensorDSLab input/config/coordinates on the
  same supported backend;
- cross-backend distributional agreement with the accepted probability kernel;
- finite-sample statistical validation as evidence for that cross-backend
  agreement contract;
- no CPU/GPU bitwise guarantee until a focused RNG work order accepts an
  algorithm capable of providing it.

The parity classifications above apply to the comparison with IV-DSLab. The
cross-backend requirement compares TensorDSLab implementations to one accepted
TensorDSLab probability contract and is therefore called agreement, not donor
parity.

Changing channel order, ID-backed batching, chunking, or unrelated batch
members must not change values associated with the same semantic coordinates.
This is a stronger reproducibility contract than the audited donor streams and
is an intentional TensorDSLab improvement.

## Condition-Database Variation

IV uses package-load global configuration, detector channel maps, and a
condition database for per-channel dark-count, crosstalk, afterpulse, charge
spread, amplitude, and RMS values.

The first TensorDSLab MVP uses explicit global scientific configs where
accepted. Per-channel variation is **deferred**, not rejected scientifically.
It requires a typed channel-ID-keyed parameter record with explicit defaults,
missing-channel behavior, units, provenance, device rendering, and cache
policy. TensorDSLab must not import the legacy DB or channel-map runtime to
claim parity.

## Accepted Intentional Divergences

The current MVP accepts these donor differences:

- start from the post-binned `readout.photoelectrons` primary-seed field rather
  than sparse PE rows;
- apply timing redistribution before dark counts and secondary effects;
- keep dark-count, frozen-snapshot, crosstalk, afterpulse, and pre-smearing count
  grids private inside `simulate_charge` rather than exposing avalanche products;
- use one frozen post-dark source for bounded first-generation crosstalk and
  afterpulse contributions;
- omit recursive generated-count feeding;
- use an ordinary exponential mean-delay model rather than IV's literal
  reciprocal-exponential afterpulse expression;
- use an aggregate charge-smearing boundary and explicit negative policy;
- omit eventwise sub-bin pulse-amplitude correction that cannot be reconstructed
  from binned input and defer an available phase-marginalized correction;
- generate direct exact-length FFT noise rather than bank/crop baselines;
- separate pure, noise, analog, and digitized field roles at one declared
  pre-digitization voltage reference plane;
- use coordinate-addressed RNG rather than donor global/sequential streams;
- clamp before integer ADC conversion and validate gain/range constraints.

These choices are acceptable for the MVP because they support a bounded,
reviewable tensor-native path. They are not evidence of end-to-end eventwise or
distributional IV parity; the narrower statistical target for
`simulate_charge` requires its own named ensemble observables and tolerances.
Afterpulse recovery amplitude is not included in this accepted list because
that scientific decision remains open.

## Validation And Fixture Rules

Parity evidence should prefer the strongest stable form:

1. analytic probability kernels and identities;
2. tiny deterministic fixtures;
3. numerical fixtures with named tolerances;
4. finite-sample statistical studies when no stronger evidence is available.

Every parity fixture must state:

- donor source path/symbol and snapshot identity;
- TensorDSLab comparison boundary;
- exact dependency and result field IDs plus collection field subset for a
  public boundary, or an explicit private-ephemeral designation for an internal
  submodel observable;
- field-specific interpretation sidecars, including
  `DigitizedWaveformSpec` for ADC comparisons;
- units, axes, layout, dtype, and configuration;
- parity classification;
- assumptions and accepted intentional divergences;
- RNG algorithm/seed/namespace/coordinates when sampled;
- execution mode and backend when needed for reproducibility, while treating
  `out`, workspace, scheduling, stream, and allocation choices as nonsemantic;
- observable, tolerance, sample size, and confidence/error criterion;
- whether it is exact/numerical evidence or distribution/statistical evidence.

Tests must not import or execute IV-DSLab or DSLab at runtime. Reference values
should be small, reviewed, and owned by TensorDSLab. A large binary donor output
or opaque Monte Carlo dump is not a substitute for an explained contract.

Suggested minimum parity observables are:

| Operation | Required observables |
| --- | --- |
| timing jitter | transition/drop probabilities, displacement mean/variance, conservation including drops |
| dark counts | exposure-normalized rate, variance, zero probability, cell covariance, edge cells |
| crosstalk | multiplicity mean/variance, zero probability, tail quantiles, generation policy |
| afterpulses | fire probability, delay law/CDF, drop fraction, recovery-amplitude relation, recursion policy |
| charge smearing | conditional mean/variance, negative tail, clipping bias, heterogeneous-charge exclusions |
| end-to-end `simulate_charge` | per-channel/sample charge mean and variance, zero-cell probability, total response, occupancy, edge loss, selected tails and time profile |
| pure waveform | template samples, peak, area, time-to-peak, impulse response, truncation/edge behavior |
| white noise | mean, RMS, marginal law, covariance, coordinate invariance |
| FFT noise | RMS, integrated power, PSD, autocorrelation, endpoint policy, marginal law |
| analog waveform | exact sum, clip order, polarity, units, common component reference plane |
| digitization | gain/map/clip/quantize order, transfer-curve boundaries, ADC bounds |

Production work orders must replace qualitative phrases such as “close enough”
with concrete tolerances and sample sizes before implementation is accepted.

## Deferred Decisions

- afterpulse recovery-amplitude model and ordering;
- exact RNG algorithm and CPU/GPU bitwise policy;
- per-channel parameter representation;
- whether any calibrated crosstalk approximation should match selected IV
  moments or remain a standalone TensorDSLab model;
- whether pure rendering needs a latent-phase-marginalized amplitude
  correction to meet peak/area tolerances;
- numeric and statistical tolerances for each production slice;
- whether a future source-aware path should preserve actual sub-bin phase,
  explicitly accepted window history, or per-PE charge weights;
- whether campaign-level IV comparisons are scientifically necessary after
  operation-level contracts are validated.

## Non-Goals

- No promise of bitwise IV stochastic output.
- No runtime dependency on IV-DSLab, DSLab, NumPy, Numba, pyFFTW, legacy
  condition DBs, remote spectrum downloads, or donor channel maps.
- No requirement to reproduce donor package layout, global config, cache, DAG,
  or fixed `(1, channel, sample)` shape.
- No parity claim for trigger, ZLE, hit finding, reconstruction, or
  baseline-subtracted analysis products in the post-binned readout MVP.
- No preservation of unsafe unsigned wrapping or apparent donor bugs.
- No use of this page to accept production code or widen a work order.

## Return To Design Before

- upgrading a statistical claim to distributional or exact parity;
- accepting an intentional divergence without named scientific observables and
  validation criteria;
- changing the post-binned comparison boundary or effect order;
- reintroducing recursive correlated-noise growth;
- changing the accepted ordinary exponential afterpulse delay or selecting
  recovery-amplitude behavior;
- depending on donor runtime state, large donor artifacts, or opaque fixtures;
- claiming campaign-level parity from isolated transform checks;
- changing a parity classification after a public field contract or durable
  cache has depended on it.

## Maintenance

Update this page in the same stage when:

- donor behavior is newly promoted, rejected, corrected, or deferred;
- a parity classification, assumption, observable, or tolerance changes;
- an RNG or backend comparison contract changes;
- a donor snapshot or source locator changes;
- a fixture begins or stops serving as parity evidence;
- implementation reveals that an accepted tensor algorithm changes a named
  scientific statistic.

Keep target behavior in architecture and decision pages, detailed validation
criteria in `docs/validation.md`, and stage-specific implementation scope in
the focused work order. This page remains the comparison and deviation ledger.
