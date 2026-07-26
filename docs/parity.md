# IV-DSLab Parity And Intentional Divergences

Status: active Design contract for evaluating IV-DSLab and DSLab donor
behavior. Merged Stage 4 tests provide bounded deterministic waveform
evidence, Stage 5 tests provide bounded TensorDSLab-model stochastic-noise
evidence, and Stage 6 tests provide bounded private-Charge model-conformance
evidence on eager CPU. Closed Stage 7 adds bounded public-orchestration and
requested-product composition/contract evidence. This does not establish a
comprehensive donor fixture corpus, public requested-`Charge` donor
comparison, accepted IV equivalence margin, CUDA evidence, or broad stochastic
donor parity.

The bounded Stage 6 evidence is exact implementation candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58` and Review closeout
`ea979862b05f4ef543f6971c86641df317232479`, against TensorCore `0.7.0`
`b454d738f6385ce6489d85492a618a3dab139bb6`. Its conditional CUDA cases were
skipped. The public Stage 7 comparison boundary now exists through exact
Review-cleared candidate `6dd55024685013fb9412a7247d3ddde7be1a3177`.
Maintenance 2 is Merged / Closed through exact implementation candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. It uses exact TensorCore commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59` (version `0.9.0`) and preserves
the accepted scientific/address mappings while changing their module and
generic-RNG owner.
CUDA was unavailable, so it adds no GPU or completed-value cross-backend
evidence. Stage 7 is Merged / Closed, but no observable-specific IV comparison
or acceptance margin follows merely from implementing the executable boundary.
Closed Maintenance 5 adopts exact published TensorCore `0.13.0`
commit `202d8b1bc6259b8453d3d377570417f2480d782b`, compact semantic axes,
and source-derived sampling. It changes representation and removes duplicate
sampling policy; it does not change a donor comparison boundary or parity
classification.

The exact Maintenance 5 representation-continuity fixture holds the tensor
payload, tensor axis order, channel-label order, product configs, dtype,
device, and `CounterRng` fixed while replacing:

```text
old:
  SamplingConfig(period=T, count=N)
  SampleAxis("0ps", "Tps", ..., "(N-1)Tps")

new:
  SampleAxis(start=0, step=T, count=N)
```

The accepted comparison requires identical private sampling runtime values,
RNG requests, and same-stack completed products. This is representation
continuity, not a new donor-parity classification.

The Maintenance 6 implementation candidate adds one new representation
comparison boundary for physical Config values:

- when Pint canonicalization returns the same binary64 magnitude as the closed
  scalar field, exact same-stack Runtime operands, RNG calls, and completed
  products remain required;
- physically equivalent unit expressions are required to represent the same
  physical quantity even when Pint's converted binary64 magnitudes differ by
  rounding, such as `2 ns` and `2000 ps`; and
- physical equivalence alone does not justify silently rounding ordinary
  Config magnitudes or claiming bitwise product parity when prepared binary64
  operands differ.

The only bounded integerization exception is
`SampleAxis.from_period(...)`, where a converted picosecond value within one
ULP of an integer is accepted as that exact compact coordinate. SampleAxis
physical accessors preserve exact integer magnitudes, including above
`2**53`. This is a representation rule, not a donor-science approximation.

## Purpose

This page defines what TensorDSLab means by parity with IV-DSLab. It records
which legacy behaviors can be preserved exactly, numerically,
distributionally, or statistically; which behaviors intentionally change on
the post-binned tensor-native path; and which decisions remain deferred.

Parity is always scoped to a named comparison boundary and observable. It does
not mean that TensorDSLab should reproduce IV-DSLab package structure, global
state, RNG streams, sparse PE-row growth, condition-database loading, fixed
array rank, or incidental implementation defects.

The semantic-root architecture was introduced against TensorCore `0.7` and is
installed in the current production baseline against exact TensorCore
`0.13.0` after Maintenance 5. Maintenance 6 retains those compact-axis roots
and changes only the physical Config representation and bounded private
admission ownership without changing scientific behavior. The architecture is governed by
[Rebuild Architecture](architecture/rebuild.md),
[Post-Binned Readout Architecture](architecture/readout.md), and
[TensorCore Integration](architecture/tensors.md). The fixed-`K` section of
the rebuild architecture is the sole active correlated-avalanche baseline.
TensorDSLab retains no competing avalanche-algorithm architecture page. This
page classifies donor relationships; it does not independently dispatch
implementation or override an architecture contract.

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
contract phrases such as “exact product type,” “exact target-product identity,” or
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

## Maintenance 9 Provisional DS20k Veto Profile

The precise-module public `ds20k_veto()` factory is documented in the
[Public API](api.md#provisional-ds20k-veto-profile). It identifies the
represented detector/readout family, but the returned Config is a
**provisional demonstration profile, not an approved run calibration**.
The factory contains a mixture of retained donor fixture values and
illustrative settings, so its name does not promote the complete Config to one
donor-parity or calibration claim.

The comparison boundary for the retained pulse values is the existing sampled
Veto pulse equation described under
[Pure Waveform Rendering](#pure-waveform-rendering). The values are exact
Config magnitudes retained from the audited donor fixture; rendered waveform
behavior remains classified as **numerical parity** over the aligned
in-window template and convolution domain. Exact retention of an input
magnitude is not evidence that the magnitude is an approved run calibration.

| Surface | Profile or demo value | Classification and comparison boundary |
| --- | --- | --- |
| Veto Gaussian center | `232.89 ns` | Retained donor fixture input to the existing Veto **numerical parity** boundary; no run-calibration claim. |
| Veto Gaussian width | `507.72 ns` | Retained donor fixture input to the existing Veto **numerical parity** boundary; no run-calibration claim. |
| Veto first edge offset | `-81.92 ns` | Retained donor fixture input to the existing Veto **numerical parity** boundary; no run-calibration claim. |
| Veto first edge width | `147.28 ns` | Retained donor fixture input to the existing Veto **numerical parity** boundary; no run-calibration claim. |
| Veto second edge offset | `-176.50 ns` | Retained donor fixture input to the existing Veto **numerical parity** boundary; no run-calibration claim. |
| Veto second edge width | `45.69 ns` | Retained donor fixture input to the existing Veto **numerical parity** boundary; no run-calibration claim. |
| Veto support | `2020.27 ns` | Retained 8 ns donor fixture input to the existing Veto **numerical parity** boundary; no run-calibration claim. |
| Veto peak magnitude | `14.5912372 mV` | Retained donor fixture magnitude. Maintenance 7 applies the fixed negative-going polarity once during preparation; the existing Veto rendering boundary remains **numerical parity**, not run calibration. |
| Dark-count rate | `100 kHz` | **Deferred** calibration choice used illustratively; no donor-equivalence claim. |
| Timing jitter | disabled | **Deferred** illustrative simplification; no claim that a detector run has zero jitter. |
| Correlated avalanches | disabled | **Deferred** illustrative simplification; no claim that a detector run has no correlated avalanches. |
| Charge smearing | disabled | **Deferred** illustrative simplification; no claim that a detector run has no charge spread. |
| PSD frequency coverage | `[0, 250] MHz` | **Deferred** illustrative choice matched to the demo's 2 ns grid; no measured-noise parity claim. |
| PSD nonzero support | below `62.5 MHz` | **Deferred** IV-DSLab-like demonstration shape; no distributional or statistical parity claim. |
| Integrated PSD RMS | approximately `0.5 mV` | **Deferred** illustrative noise scale; no measured-noise or run-calibration claim. |
| Analog saturation | disabled | **Deferred** illustrative linear-composition choice; no detector-saturation claim. |
| ADC bit depth | `16` | **Deferred** illustrative IV-DSLab-like digitizer choice; no hardware/run parity claim. |
| ADC input interval | `[-3900, 100] mV` | **Deferred** illustrative IV-DSLab-like digitizer choice; no hardware/run parity claim. |
| Analog gain | `3.5218 dB` | **Deferred** illustrative IV-DSLab-like digitizer choice; no hardware/run parity claim. |
| Sampling | start `0`, period `2 ns`, count `5000`, exclusive stop `10000 ns` | **Not applicable** to detector calibration or donor sampling parity: this demo grid is owned by the source `SampleAxis`, not by `ds20k_veto()`. |
| Example/channel axes | two examples and labels `veto-0` through `veto-3` | **Not applicable** to detector calibration: these are local demonstration identities, not physical channel provenance. |
| Toy source and RNG | explicit `1`, `2`, `3`, and `4` PE deposits at `200 ns`, `2600 ns`, `5000 ns`, and `7400 ns`; readout seed `17` | **Not applicable** to donor parity: deterministic tutorial inputs and repeatability, not an event or RNG-stream equivalence claim. The seeded dark-count model may add later Charge events. |
| Execution and plots | `float32`, full/selected product requests, one displayed example/channel, four panels | **Not applicable** to calibration: public-API and presentation choices only. |

The profile owns only its returned Config tree. Source construction, axes,
sampling, products, dtype, device, seed, retention, and visualization remain
explicit caller or demo choices. Later promotion or replacement of a
provisional value requires a focused scientific/API decision and a newly named
comparison boundary; it must not be inferred from the factory name or from a
successful demonstration.

Maintenance 9 is **Merged / Closed** through exact Review-cleared target
`2a04942229ab06d2cfc17ab7a5fd09afaf4e3c58`. Its complete profile, script,
notebook, plot, source/archive, and artifact evidence does not strengthen any
classification in the table above: retained pulse fixtures remain numerical
parity only over their named waveform boundary, and every illustrative
profile/demo choice remains Deferred or Not applicable as recorded.

Maintenance 10 refines only the provisional digitizer and caller-owned demo
source/grid/plot evidence recorded in the table. The `16`-bit
`[-3900, 100] mV`, `3.5218 dB` digitizer remains a **Deferred**
IV-DSLab-like illustrative choice rather than an approved calibration. The
explicit four-deposit source, `10000 ns` grid, fixed seed, and black Charge
overlays are **Not applicable** to detector calibration or donor-event parity.
Nothing in this refinement strengthens waveform numerical parity, establishes
digitizer calibration or IV eventwise parity, or changes a scientific
equation.

## Maintenance 11 Addressed-Randomness Rebaseline

Maintenance 11 is an intentional predeployment RNG comparison-boundary
change, not a new donor-equivalence claim. TensorCore `0.19.0` supplies public
addressed elements, addresses, distributions, and probability kernels.
TensorDSLab retains its model probabilities/rates, fixed active role keys,
semantic indices, physical displacement and boundary interpretation, checked
accumulation, and product validation.

| Surface | Classification and boundary |
| --- | --- |
| white/PSD noise, dark counts, smearing, timing jitter | **Numerical/statistical continuity at their existing named boundaries**; generic execution moves to TensorCore Distribution/ProbabilityKernel owners without a new donor claim. |
| direct and delayed crosstalk | **Intentional RNG rebaseline**; deterministic retained destination means are sampled once with a tensor Poisson distribution, exact in law by Poisson splitting/superposition. Eventwise words and completed seeded samples are not required to match the retired factorization. |
| direct/delayed finite-window overflow outputs | **Not applicable** to the retained product law; the scientifically unused outputs and their streams are retired without a reservation promise. |
| afterpulse | **Address rebaseline with retained active key**; stream `0x0000_0009` remains, occurrence uses quantum `0`, and conditional delay allocation uses quantum `1`. The physical occurrence, delay, recovery, boundary, and ledger laws remain unchanged. |
| `demos/random.ipynb` toy frontier | **Not applicable** to detector or donor parity; it is a CPU educational fixture for public address metadata, raw words, repeat equality, retained-capacity chunk equality, and global Torch RNG isolation. |

Exact word/address fixtures apply only to the named new address schemas and
execution stack. Same-stack repeatability, chunk/traversal invariance where
specified, analytic probability/rate checks, and product invariants remain
mandatory. No cross-version completed-value equality, integrated CUDA,
performance, calibration, deployment, or production-readiness claim follows.

Stage 4's reviewed tests establish the named TPC/Veto pulse checkpoints,
independent deterministic reference equations, conditional analog composition,
and representation-mapped ADC behavior in that work order's accepted CPU
evidence domain. Stage 5 and Stage 6 add bounded stochastic-noise and private-
Charge conformance evidence for the selected TensorDSLab models. Those stages
do not establish complete eventwise IV parity, a comprehensive donor fixture
corpus, public requested-`Charge` composition, accepted donor-equivalence
margins, or GPU evidence. Closed Stage 7 adds public composition/contract
evidence but no donor comparison or stronger parity classification. Every
stronger classification still requires its named focused
evidence.

## Required Shape Of A Parity Claim

Every production work order or fixture that claims parity must name:

```text
reference:
comparison boundary:
input product types:
result product type:
requested ReadoutCollection product subset (context only):
private ephemeral observable (if no public product boundary):
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

The rebuild's primary in-memory result is one request-selected
`ReadoutCollection`, but parity never attaches to the container merely because
a product is present. A public claim names the exact input and requested result
product types. A private charge-submodel claim names its ephemeral
frontier/ledger boundary and observable and must not invent another product
type. Every claim also names the exact compact axes, source-derived sampling
facts, product config, dtype, comparison boundary, and observable.

The Stage 7 builder returns exactly the nonempty product set requested
by the caller. Privately computed prerequisites and avalanche diagnostics are
not additional parity results. Unless a claim explicitly covers a composition,
its acceptance criteria apply only to the named product.

An analog-only or digitized-only requested result is therefore valid without
retaining its prerequisites, but its presence alone records neither lineage
nor parity evidence. A new analog-composition claim still requires both
`PureWaveform` and `NoiseWaveform`; a digitization claim requires
`AnalogWaveform`.

## Execution And Scheduling Neutrality

The selected rebuild implements one functional Stage 7 `simulate_readout(...)`
surface. It does not carry the historical Stage 2 architecture's public `out`,
`ReadoutWorkspace`, fixed-chain builder, lease, or allocation-free contracts
forward by default. Those existing contracts remain historical evidence, not
implicit rebuild API.

The broader parity rule is independent of either execution design. Product
planning, scratch storage, buffer reuse, fusion, stream placement, and target
publication do not create a donor comparison boundary, strengthen or weaken a
parity classification, or justify different scientific values for the same
accepted input/config/RNG-algorithm-and-seed/key/backend contract. A later
rebuild optimization must
preserve the documented scientific order, position-addressed random field,
causal edge policies, frozen per-generation frontier, and S1/S2 ledgers. If it
changes a named observable, that is an architecture change requiring Design
review, not an allocator- or scheduler-dependent tolerance.

Private dark-count, crosstalk, afterpulse, overflow, or smearing values remain
ephemeral validation observables regardless of their eventual storage plan.
Scratch storage never promotes them to recognized product types, public
products, durable labels, sidecars, or lineage records. Allocation, aliasing,
stream safety, failure behavior, and fusion are separate implementation and
resource claims; they do not enter a scientific parity acceptance criterion.

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

IV-DSLab operates primarily on sparse PE rows. The rebuild starts from a dense
`Photoelectrons` product containing binned photon-origin primary PE seeds. It
is not electrical charge and does not already include dark counts, crosstalk
avalanches, or afterpulses. The representation has discarded:

- each PE's exact sub-bin time;
- individual seed identity and any source-row ordering;
- any source-side weight beyond the accepted unit primary-seed count.

The rebuild also deliberately does not carry donor global RNG state/draw order or
implicit condition-database and channel-map state.

Consequently, TensorDSLab does not claim end-to-end eventwise parity with
IV-DSLab for the rebuild. Legitimate comparisons use one of three explicitly
different boundaries:

1. a private timing-redistribution diagnostic from a common binned
   `Photoelectrons` input, without treating jitter as a replacement truth
   product;
2. the implemented Stage 7 `simulate_readout(..., products=[Charge])` from
   common `Photoelectrons` for end-to-end charge statistics; or
3. a private ephemeral count-grid boundary used only to validate an internal
   dark-count, crosstalk, afterpulse, or smearing submodel.

The third boundary does not accept another collection field or product. Its
count grids are test observables or scratch representations only. Comparisons at
any boundary may use a marginalized ensemble when TensorDSLab explicitly models
lost variables such as sub-bin phase and states the assumed distribution.

The common default for unknown source-bin phase is:

```text
U ~ Uniform([0, sample_period))
```

That assumption can establish conditional distributional parity for an ideal
mathematical binned transition kernel. Binary64 analytic preparation and the
selected finite aggregate sampler still reduce the implemented timing-jitter
claim to statistical parity. The assumption cannot reproduce eventwise IV
behavior when the original PE phases were known or were not conditionally
uniform. Nor does Stage 6's isolated private-submodel result establish parity
for the implemented public requested-`Charge` composition; Stage 7's
composition evidence is not donor-parity evidence.

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

The selected TensorDSLab rebuild comparison begins after a future
TensorG4DS-to-TensorDSLab bridge has produced the binned input. Native G4DS and
TensorG4DS behavior are outside this readout parity boundary. The rebuild flow is:

```text
Photoelectrons (binned photon-origin primary PE seeds)
  -> optional private dark-count addition
  -> optional private timing redistribution of then-current seeds
  -> optional fixed-K coupled DiCT/DeCT/AP simulation
       -> integer avalanche frontier
       -> floating S1 deposited-charge ledger
       -> floating S2 charge-square-sum ledger
       -> separate right-overflow diagnostics
  -> optional terminal S1/S2 charge smearing
  -> Charge -> PureWaveform

Photoelectrons axes/device/shape + source SampleAxis
  -> NoiseWaveform

PureWaveform + NoiseWaveform
  -> AnalogWaveform -> DigitizedWaveform

retain exactly the caller-requested products
```

Effect-order classification: **intentional divergence**.

The implemented public requested `Charge` boundary may target **statistical
parity** for named charge observables over an accepted input/config ensemble.
TensorDSLab must not claim distributional or eventwise parity for that composed
stochastic flow.
Passing isolated internal-submodel checks is necessary evidence, but it does not
prove the end-to-end target. Any campaign-level comparison likewise remains
statistical validation of named observables, not evidence that the composed
stochastic laws are identical.

## Summary Matrix

| Surface | Target classification | Core qualification |
| --- | --- | --- |
| package, collection/field, rank, and TensorCore representation | not applicable | TensorDSLab is a clean-slate tensor-native architecture. |
| full stochastic effect order | intentional divergence | IV jitters after recursive noise and optional smearing; the rebuild adds dark roots, jitters truth plus dark roots, then runs causal fixed-`K` branching. |
| end-to-end requested `Charge` | statistical | Compare named aggregate `Charge` observables from common `Photoelectrons` ensembles; internal divergences preclude equality of the complete law. |
| timing jitter in isolation | statistical, with an ideal-kernel analytic oracle | The ideal iid-uniform-phase/Gaussian transition law is integrated analytically; binary64 preparation, aggregate conditional-binomial sampling, and different RNG streams preclude literal donor identity. |
| dark counts in complete equal-width cells | distributional, conditional | Follows from Poisson splitting for homogeneous rates; DB variation and edge/order effects are excluded. |
| crosstalk | statistical/intentional divergence | The rebuild uses fixed-`K` unmarked recursion with separate ordinary-Poisson DiCT and DeCT mechanisms; IV has no audited DeCT and uses charge-dependent recursive quirks. |
| correlated-noise recursion | statistical/intentional divergence | Generated retained children feed every enabled mechanism only in the next frozen generation, with caller-selected finite `K`. |
| fixed-`K` DiCT/DeCT edge placement | intentional divergence | The sole active rebuild baseline uses independently phase-marginalized causal nonnegative-offset kernels with right-overflow diagnostics only; it does not reproduce signed post-binned displacement from IV's later independent parent/child jitter. |
| afterpulse delay law | intentional divergence | TensorDSLab accepts an ordinary exponential mean-delay model instead of IV's literal reciprocal-exponential expression. |
| afterpulse recovery amplitude | intentional divergence / binned approximation | `recovery=None` gives unit AP charge instead of IV recovery weighting; a composed recovery config uses conditional category means without changing branching and omits within-category recovery variance. |
| charge smearing | statistical | The ideal aggregate-Gaussian identity is an analytic oracle; one finite-lattice aggregate draw is not the donor's per-avalanche digital draw sequence. |
| FEB-SNR pulse template and convolution | numerical | Same functional family and negative-going detector polarity, with a positive public amplitude magnitude and one preparation-owned negative sign; normalization and binned-photoelectron edge/sub-bin behavior differ. |
| eventwise IV sub-bin amplitude correction | not applicable | True PE phase is absent after binning. |
| omission of a phase-marginalized amplitude correction | intentional divergence | A latent-phase expectation is possible, but the first MVP does not apply one. |
| exact zero baseline | exact | Both can produce an all-zero noise waveform. |
| arbitrary constant baseline | not applicable to the MVP | No accepted config selects a deterministic nonzero analog baseline; the digitizer transfer owns the ADC code at 0 mV. |
| white Gaussian noise | statistical, with an ideal-normal analytic oracle | Mean, RMS, and independence target the same ideal law; finite digital samplers and RNG streams intentionally differ. |
| PSD-shaped noise | statistical | Preserve calibrated spectral intent, not IV bank/crop traces or their exact distribution. |
| analog `pure + noise` composition | exact, conditional | Exact algebraically when inputs, units, sign, shape, and clipping policy match. |
| open-interior ADC mapping | exact after representation mapping, conditional | When the mapped floating value and transfer parameters match away from a transition, both truncate to the same integer code; TensorDSLab stores it in signed `torch.int32` rather than donor `uint16`. Inclusive endpoint guards are an intentional numerical correction. |
| out-of-range ADC behavior | intentional divergence | TensorDSLab clips before integer conversion instead of preserving IV unsigned wraparound. |
| fixed-seed or bitwise IV RNG streams | not applicable | TensorDSLab uses position-addressed tensor-native random fields. |
| per-channel condition DB variation | deferred | Requires a typed TensorDSLab channel-parameter boundary. |

## Timing Jitter

### IV Timing Behavior

`src/dselec/daq.py:add_daq_jitter` adds an independent Gaussian draw to every
sparse PE time. Because the true PE time is available, IV preserves the actual
sub-bin phase before jitter. The executable applies jitter after dark counts,
correlated-noise expansion, and optional DB smearing. The audited donor config
uses `sigma = 5 ns` with `125 MHz` sampling, hence `T = 8 ns` and
`sigma / T = 0.625`; the executable is an IID PE-time displacement even though
the nearby configuration comment describes clock-period variation.

### TensorDSLab Timing Behavior

The rebuild keeps `Photoelectrons` as immutable truth. Inside
`_produce_charge(...)`, an effective `_simulate_dark_counts(...)` block precedes
an effective `_simulate_timing_jitter(...)` block. Jitter therefore
redistributes truth plus dark roots when both execute, truth alone when the dark
block is skipped, and performs no redistribution when jitter itself is skipped.
For source sample `s`, sample period `T`, latent phase `U`, and jitter `J`:

```text
U ~ Uniform([0, T))
J ~ Normal(0, sigma)
target = s + floor((U + J) / T)
```

Preflight analytically integrates the ideal Gaussian over the latent uniform
phase to prepare every in-window target probability. Runtime samples aggregate
target counts through sequential conditional binomials and assigns the one
combined out-of-window category the exact remaining integer count. The binary64
preparation uses the log-domain one-sided cumulative tail selected in
`architecture/rebuild.md`; it does not repeatedly subtract rounded categories
from one, clip a negative tail, or normalize a residual. It does not draw a
Gaussian per PE, use Box-Muller for timing jitter, or materialize one PE row per
quantum. Target bins are scanned increasingly, which is increasing signed
offset order for each fixed source bin. No arbitrary tail cutoff removes a
possibly in-window destination.

The first accepted numerical domain is
`2**-52 <= sigma / T <= 64`, `2 <= sample_count <= 8192`, and
`S * N <= 2**63`; exact `sigma == 0` is a separate no-draw identity. The
fixed category/tail/identity absolute tolerance is `1e-12`, and the complete
represented source law must be within `1e-11` L1 of the independent high-
precision ideal law. Unsupported positive ratios and window sizes fail rather
than silently changing the physics. The dedicated append-only stream is
`CHARGE_TIMING_JITTER = 0x0000_0008`.

### Timing Parity Claim

Classification: **conditional statistical parity in isolation**, with the
ideal latent-uniform-plus-Gaussian transition kernel retained as an analytic
oracle.

The ideal binned target law is the same as individually jittering PEs only
when:

- conditional source-bin phases are independent and uniform;
- ideal latent Gaussian displacements are independent with the same width;
- the sample period, window, and boundary convention match;
- out-of-window targets are dropped in both comparisons;
- the comparison ignores the different position of jitter in the full chain.

TensorDSLab's analytically prepared binary64 kernel and aggregate multinomial
orchestration do not reproduce IV's per-PE finite digital normal or draw
sequence exactly. Closed Stage 6 executed the frozen binomial mapping through
private `readout/_random.py`; the closed Maintenance 2 implementation calls the
same promoted mapping through TensorCore's positional `binomial(...)`.
Validation therefore measures the analytic transition/drop probabilities,
multinomial moments and covariance, and named tails rather than upgrading the
implemented comparison to distributional parity.

If IV source phases are known and nonuniform, TensorDSLab's result is a
statistical approximation to the marginalized binned behavior, not
distributional parity for that event.

Validation should compare transition probabilities through the farthest
in-window destinations, the explicit final drop bucket, the represented law
induced by the prepared conditional masses, conditional mean/variance and
multinomial covariance of bin displacement, and exact count conservation
including drops. It should include the evaluator boundary, natural underflow,
and rejection just outside the supported domain. It should not compare same-
seed PE destinations.

## Dark Counts

### IV Dark-Count Behavior

The non-DB path samples a total Poisson count over gate duration and channel
count, then assigns uniform times and uniformly selected channels. The audited
checkout uses the veto channel map in this path despite commented TPC lines.
The DB path samples a separate Poisson total using each channel's configured
rate and assigns uniform times within the gate.

### TensorDSLab Dark-Count Behavior

Inside the private `_produce_charge(...)` path, TensorDSLab adds an independent
count to each eligible channel/sample cell of a private working grid:

```text
count ~ Poisson(rate_hz * sample_period_ns * 1e-9)
```

The rebuild uses one global rate. Typed per-channel rates are deferred. The
resulting grid is an ephemeral submodel observable, not a recognized
`ReadoutCollection` field.
TensorDSLab prepares each cell's binary64 mean. Closed Stage 6 executed the
selected hybrid mapping through private `readout/_random.py`; the closed
Maintenance 2 implementation calls TensorCore's `poisson(...)`: one-uniform
inverse-CDF sampling below mean `10` and Hoermann PTRS from `10` through its
accepted execution ceiling `1e8`. This is an implementation of the named
Poisson target law, not donor RNG or same-seed parity.

### Dark-Count Parity Claim

Classification: **conditional distributional parity**.

Poisson splitting makes the IV total-count/uniform-placement model equivalent
to independent per-cell Poisson counts when rates are homogeneous and all
cells represent complete equal channel-time exposure. The claim excludes:

- per-channel DB variation;
- partial final samples or mismatched gate conventions;
- the different ordering relative to timing jitter;
- later recursive branching, smearing, and boundary filtering.

The mathematical-kernel equivalence does not upgrade executable samples to
bitwise or eventwise equality. TensorDSLab uses finite binary64 uniform
lattices—the 53-bit closed-open lattice for inversion and 52-bit midpoint
open-open lattices for PTRS—plus position-addressed Threefry streams, bounded
failure policies, and backend transcendental arithmetic. Donor comparisons
therefore validate the Poisson observables statistically even where the ideal
count law has conditional distributional parity.

For a stationary homogeneous Poisson process, independent timing displacement
preserves the interior rate. Finite-window edge behavior can still differ
because both systems jitter dark-count PEs but differ in their surrounding
recursive effect order and finite-window boundary construction.

Validation should compare expected counts per channel-time exposure, Poisson
variance, zero-count probability, cell covariance, and edge-cell behavior.

## Crosstalk And Correlated-Noise Recursion

### IV Crosstalk And Recursion Behavior

IV direct and photon crosstalk use recursive `poissonian_loop(p * charge)`
branching. The returned count is total descendants, not a single first
generation. For a unit source and subcritical coefficient `p`, the recursive
mean is `p / (1 - p)`, not `p`.

For a fractional-charge source, IV freezes `p * source_charge` as the offspring
coefficient throughout that helper's entire hidden recursive tree, then emits
all flattened crosstalk descendants with unit charge. This is not coherent
per-node marked branching, where each child's own charge would determine only
its direct offspring. TensorDSLab records the frozen-source behavior as a donor
artifact rather than a marked-recursion target.

IV then processes a growing PE queue. Photon crosstalk, direct crosstalk, and
afterpulse outputs can seed other effects; dark counts participate in that
queue. Same-type guards and internal loop unrolling limit some paths, but the
overall model still contains cross-effect recursive feeding.

### TensorDSLab Crosstalk Behavior

The rebuild uses one frozen integer frontier per generation. For each unit
parent, DiCT and DeCT own separate configured mean offspring counts, separate
causal physical-delay models, separate prepared offset PMFs, and separate
ordinary-Poisson draws:

```text
A_direct[g + 1, u]  ~ Poisson(R_direct[g + 1, u])
A_delayed[g + 1, u] ~ Poisson(R_delayed[g + 1, u])
```

The closed Maintenance 2 implementation gives the two retained modes and their
two overflow roles separate config-owned `RngKey` values and calls TensorCore's
public hybrid `poisson(...)`; closed Stage 6 used the corresponding private
streams and the same frozen sampler mapping. Every actual aggregate cell rate
must be binary64, finite, nonnegative, and no greater than `1e8`. The sampler
choice changes neither the declared Poisson law nor the intentional donor
divergences below; its raw-word sequence has no IV-parity claim.

Rates are never superimposed and no Gamma latent surrounds either supplied
mean. Every retained child contributes one avalanche and unit deposited charge
and enters the ordinary unmarked next frontier. A caller-selected `K` bounds
genealogical depth; `K=1` is the first-generation case. DeCT is a distinct
optional TensorDSLab model with no audited IV counterpart.

Each CT mode may independently select a fixed or exponential physical-delay
family. Both satisfy the shared nonnegative causal-delay contract; common
preparation rejects negative-offset support or an underflow category instead
of silently clipping an invalid model. The earlier zero-clipped normal proposal
is retired from the MVP: its negative latent tail would create a calibration-
sensitive prompt atom, it has no IV-parity basis, and a future truncated,
clipped, lognormal, or tabulated family requires a separately calibrated
scientific decision and a new explicit config type.

For the selected fixed and exponential families, phase-marginalized category
masses and analytic right tails use the frozen binary64 mappings and numerical
domains in `architecture/rebuild.md`. Exponential AP recovery uses the same
integrated category law and its stable effective-mean difference. These rules
define TensorDSLab execution; they do not strengthen either mechanism's donor
parity classification.

### Crosstalk Parity Claim

Classification: **intentional divergence**.

The DiCT coefficient remains an ordinary direct-offspring Poisson mean. Reusing
IV's numeric value can support generation-level statistical comparison but
does not reproduce IV's source-charge-frozen hidden subtree or signed
post-jitter edge displacement. Validation should report results as a function
of `K`, sampling, and boundary policy. Private mechanism diagnostics do not
create public crosstalk products or transforms. No normal-delay option remains
in the MVP. Any future delay family must be calibrated and must not be
described as better IV parity or greater scientific accuracy without evidence.

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
self-recursion guard is commented out in the audited source. Literal IV then
uses that row's fractional `charge` in both `direct_ct * pe.charge` and
`ap * pe.charge`, so recovery suppresses later DiCT intensity and AP fire
probability. Its final per-row charge smearing occurs only after correlated
branching and does not feed back.

For DiCT, that multiplication scales the seed yield from the source avalanche;
it does not give the crosstalk child fractional charge. Source-yield scaling can
be physically motivated because a partially recovered avalanche has lower
multiplication and may emit fewer secondary photons. The separate IV artifact
is that `poissonian_loop(direct_ct * source_charge)` freezes this reduced
coefficient through its hidden recursive subtree even though every emitted
DiCT descendant is assigned unit charge.

The audited `PEType` set contains no DeCT member. The non-database `PHCT` path
is disabled by default, marked `TODO`/`FIXME`, produces same-raw-time
unit-charge rows, and is absent from the database path. IV's delayed,
recovery-weighted AP rows therefore do not establish a delayed-crosstalk model.
TensorDSLab must treat DeCT as a new calibrated inter-microcell mechanism, not
infer it from IV's AP parameters or recovery law. This separation is consistent
with measured amplitude-versus-delay analyses that identify low-amplitude
same-pixel afterpulses during recovery and full-amplitude delayed crosstalk in
other pixels; see [Rosado and Hidalgo
(2015)](https://arxiv.org/abs/1509.02286).

### TensorDSLab Direction

The sole active rebuild baseline uses ordinary exponential AP delay
parameterized by `mean_delay_ns`, a fresh independent latent uniform phase for
each parent-child edge, an explicit right-overflow category, and no ragged PE
rows. The standard exponential law is a deliberate correction rather than
literal reproduction of IV's reciprocal-exponential expression.

Every successful AP contributes one integer avalanche to the shared unmarked
next-generation frontier. `afterpulse=None` disables the mechanism;
`afterpulse.recovery=None` retains AP with unit deposited charge. A present
`AfterpulseRecoveryConfig.time_constant_ns` selects
`rho(delay) = 1 - exp(-delay / time_constant)`. Before source-relative
delay categories collapse into destination bins, the same realized category
counts update both integer AP count and their conditional-mean recovery-weighted
charge. The corresponding private square-sum is the sum of category weights
squared, not the square of aggregate AP charge. Recovery never changes AP,
DiCT, or DeCT offspring probability and never becomes frontier state.

### Afterpulse Parity Claim

Delay-law classification: **intentional divergence**.

Unit-recovery mode classification: **intentional divergence from IV's
recovery weighting**.

Configured fixed-`K` recovery-response classification: **intentional binned
approximation**, eligible only for statistical comparison of named charge and
count/charge observables.

The rebuild preserves AP existence/delay fluctuations and their
covariance with deposited charge by weighting the same sampled categories. It
intentionally omits recovery variation within one delay category and diverges
from IV through recovery-independent future branching: every root, dark-count,
DiCT, DeCT, and AP parent uses the same DiCT/DeCT rates and AP fire probability,
with no recovery state carried forward. Its recovery clock is the immediate AP
birth edge rather than reconstructed microcell firing history. Recovery
classes, if later used to refine within-category charge integration, are
transient AP sampling categories and are summed immediately into the shared
unmarked frontier.

The ordinary exponential delay, independent per-edge phase closure,
recovery-independent future branching, fixed `K`, and conditional-mean binned
recovery weights remain recorded intentional divergences. They prevent a full
distributional or eventwise parity claim even when named aggregate observables
agree statistically.

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

As the terminal private submodel inside `_produce_charge(...)`, the completed
fixed-`K` ledgers are:

```text
S1 = sum_i w_i
S2 = sum_i w_i**2
```

Roots and CT children have `w_i=1`; recovered AP categories have their selected
conditional-mean weights. With no smearing config, public `Charge` is a new
field with guaranteed fresh storage independent of the source
`Photoelectrons`, over `S1`. With smearing enabled, the terminal rule is
`Normal(S1, relative_sigma * sqrt(S2))` followed by nonnegative clipping.
Avalanche count and `S1` alone do not determine that variance. `S2` is always
accumulated as private numerical scratch; it is not branching state or a
collection product.

Maintenance 2 does not change that scientific law. It retains the
upward-rounded Stage 6 analytic check over the real ledger bound, derives the
greatest target-dtype ledger that bound covers for represented-ledger checks,
and intersects the scientific check with TensorCore's public Gaussian
prepared-scale finite-output envelope. The frozen `K=0` float32 and float64
endpoints remain unchanged. Only contextual extreme float32 acceptance can
narrow (the verified `L=24` adjacent pair), while the ordinary relative-sigma
`0.1` fixture, RNG keys and addresses, equations, clipping, and operation and
accumulation order remain bitwise or semantically unchanged as applicable.
This numerical-domain intersection has no calibrated donor meaning and does
not alter the parity classification below.

### Charge-Smearing Parity Claim

Classification: **statistical parity**.

In the ideal real-valued Gaussian model and before clipping, one aggregate draw
has the same mathematical distribution as summing `n` independent
`Normal(1, sigma)` unit charges. That is an analytic oracle, not an implemented
distributional-parity claim: TensorDSLab draws one bounded finite-lattice
Box-Muller normal, while the donor draws and rounds per avalanche. The identity
also does not hold for heterogeneous `q_i` or channel widths: the donor variance
is proportional to `sum(q_i**2)`, not simply `n`. Recovery-weighted afterpulses
are outside the unit-count subcase. The fixed-`K` path's explicit `S2` preserves
the selected ideal-model variance for its conditional-mean binned weights; it
does not restore IV's continuous within-category recovery variation or literal
delay law. TensorDSLab's finite normal tail and zero clipping further change the
tail, bias, and mean.

Validation should compare conditional mean, variance, zero-count behavior,
negative-tail probability before policy, post-clipping bias, and high/low count
regimes at the private-count-to-public-charge boundary. Per-channel variation
requires a later typed parameter contract. The private integer input does not
become a recognized collection product, and `Charge` denotes a floating
aggregate PE-equivalent response rather than SI charge.

## End-To-End Requested `Charge`

The implemented Stage 7 public comparison boundary is:

```text
Photoelectrons -> simulate_readout(..., products=[Charge]) -> Charge
```

Target classification: **statistical parity for named observables**.

This claim compares a common ensemble of binned photon-origin primary PE seeds
to the aggregate PE-equivalent charge response. It does not expose or require
the donor's sparse generated-PE rows, and it does not assert that the complete
joint law matches. The different effect order, fixed-depth unmarked recursion,
causal binned edge placement, recovery approximation, aggregate smearing,
negative clipping, and RNG construction preclude an end-to-end distributional
claim.

Stage 6 validated `_produce_charge(...)` and its private submodels on eager
CPU. Stage 7 validated public request planning, collection
retention, and composition on eager CPU, but it does not establish IV
equivalence. A later parity stage must compare the implemented public
composition with donor evidence. At minimum it should compare per-channel and
per-sample charge mean and variance, zero-cell probability, total response,
occupancy, edge loss, selected tail quantiles, and any accepted time-profile
statistic over named
input/config ensembles. Each internal dark-count, crosstalk, afterpulse, and
smearing check must retain its own classification; passing them separately does
not prove this end-to-end statistical target. Conversely, a passing
requested-`Charge` ensemble does not upgrade an intentionally divergent internal
submodel to distributional parity.

Fixtures may retain private intermediate grids as test-only diagnostics when
needed to localize a failure. They must not record those grids as recognized
product types, public products, durable producer labels, or required collection
or config state.

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
accepts one strictly positive peak-voltage-per-photoelectron amplitude
magnitude. Preparation applies the fixed DS20k negative polarity exactly once;
the public Config does not expose a polarity switch or signed amplitude.
`PureWaveform` remains signal-only; baseline is owned by `NoiseWaveform`.

### Pure-Waveform Parity Claim

Classification: **numerical parity** for the aligned in-window template and
convolution domain.

The sampled-maximum and analytic-normalization choices are close but not
identical; for the audited IV default parameters their peak normalization
differs by roughly 66 parts per million. Stage 4 locks reviewed TPC and Veto
checkpoint values, independent binary64 reference equations, and explicit
`float32`/`float64` tolerances rather than relying on that audit estimate.
Maintenance 7 changes only the public calibration convention from a signed
negative quantity to an equal positive magnitude followed by one
preparation-owned negative sign. This is an intentional API narrowing faithful
to the negative-going DS20k DAQ response; calibrated rendered kernels and
waveforms remain exact.

IV's eventwise sub-bin amplitude correction is **not applicable** at the
post-binned boundary because true PE phase has been discarded. Loss of the
eventwise phase does not make every statistical correction impossible: under
an accepted latent phase `U ~ Uniform([0, T))`, TensorDSLab could derive and apply
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
impulse response, same-length truncation, positive Config magnitude,
preparation-owned negative polarity, and boundary behavior.

## Noise Waveforms

### Zero Baseline

IV returns exact float32 zeros when noise is disabled. TensorDSLab can provide
**exact parity** for an all-zero `NoiseWaveform` over the declared shape after
any explicit dtype mapping is accounted for.

An arbitrary nonzero constant baseline is not an accepted MVP
`NoiseWaveformConfig` model. The analog products remain zero-referenced, and
the digitizer transfer owns the nonzero ADC code corresponding to 0 mV. Adding
a deterministic analog pedestal later requires a focused Design decision and
new parity boundary rather than an implementation-local constant-noise mode.

### White Noise

IV non-DB white noise is iid Gaussian with RMS `pe_amplitude / snr`. The DB path
generates iid standard normal values and later scales by per-channel RMS.
TensorDSLab generates a position-addressed per-sample Gaussian field.

Classification: **statistical parity**, with the matching ideal iid-normal law
retained as an analytic oracle when mean, RMS, and independence assumptions
match. Global RNG state, sequential draw order, finite digital normal law, and
same-seed IV/TensorDSLab values are intentionally excluded. TensorDSLab's own
same-stack repeatability additionally requires an unchanged OS/architecture,
Python/PyTorch build, backend/device implementation, eager execution mode,
math settings, shape, dimension order, coordinate order, config, dtype, and
positional RNG version; it does not promise reorder or chunk invariance.

Stage 5 fixes `NOISE_WHITE = 0x0000_0001` under private
`tensordslab.threefry4x32-20/v1`. White RMS is prepared in binary64 and rounded
once into the selected output dtype; represented subnormal RMS values are
rejected, and that normal-range RMS plus the accepted finite Box-Muller lattice
define the implemented law. The finite normal cutoff
therefore remains part of the statistical-parity qualification rather than a
claim of an unbounded ideal Gaussian.

Stage 5 is Merged / Closed through exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd`. Its frozen eager-CPU evidence
clears the accepted same-stack repeatability and analytic statistical gates.
CUDA was unavailable, so this closeout adds no cross-backend execution or GPU
performance claim.

Validation should compare mean, RMS, relevant tail behavior, sample/channel
covariance, and same-shape same-stack repeatability using analytic estimator
uncertainty rather than an arbitrary normality-test p-value. Reordering and
chunking should verify the documented positional contract rather than assert
invariant values. A fixed white-noise array is a reproducibility fixture, not
proof of distributional parity.

### PSD-Shaped Noise

IV zeros DC, adds global random phases, synthesizes long inverse-FFT baselines,
normalizes each long baseline to unit standard deviation, persists or loads a
bank, and randomly crops per-channel segments.

TensorDSLab directly synthesizes the exact requested sample length from a
caller-supplied absolute one-sided PSD. It integrates source intervals onto the
target frequency cells, discards the DC-cell power without redistribution,
draws position-addressed Gaussian Fourier coefficients with fixed endpoint
scaling, and applies the documented inverse real transform without post-transform
RMS normalization. It does not reproduce the donor's bank/crop process.

Classification: **statistical parity**.

Stage 5 fixes `NOISE_PSD_COEFFICIENT = 0x0000_0002`. Overlap integration uses
Python binary64 and `math.fsum`; retained powers are rounded once into the
execution dtype and define the ideal-standard-normal target
variance/covariance oracle. The finite Box-Muller lattice is not renormalized
and remains inside the statistical allowance. The private DC coefficient is
exact zero. The synthesized sample mean is zero only within inverse-FFT
roundoff and is not post-corrected.

The Merged / Closed Stage 5 evidence covers odd/even endpoint scaling,
isolated cosine and sine bases, real-only Nyquist behavior, Parseval power,
lagged covariance, coefficient moments, and cross-row independence. Those
checks establish the documented TensorDSLab law, not trace identity with IV's
bank-and-crop process.

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
reference plane and in the same voltage units. `AnalogWaveform` is their
composed pre-digitization voltage waveform; the component fields do not claim
to be separate Tile, PDU, or DAQ hardware-boundary outputs.

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

TensorDSLab digitizes `AnalogWaveform` into the distinct `DigitizedWaveform`
ADC-count product. “Digitized” is retained because the product is specifically
the ADC result, not an arbitrary later digital or firmware-processed waveform.

For open-interior in-range values, the target is **exact parity after
representation mapping**. TensorDSLab derives the pre-gain lower and upper
input thresholds, rounds them once to the waveform execution dtype, and
assigns code zero or `maximum_code` inclusively when the analog value reaches
those exact representable thresholds. The open interval uses the accepted
affine mapping, floating clamp, and truncating conversion to signed
`torch.int32`; bit depth is 1 through 16. The endpoint guard prevents ordinary
affine rounding from placing a configured upper endpoint just below the
maximum code. IV stores its code in `uint16`; dtype identity is not the parity
claim, but equal interior integer code values are required when the mapped
floating value, gain, voltage range, offset, and bit depth match. Differences
introduced by accepted floating dtype or backend arithmetic use numerical
parity with focused-work-order tolerances and transition fixtures. Inclusive
endpoint saturation is an **intentional numerical correction** wherever
literal donor/affine operation order would lose one code to rounding; it
preserves the configured ADC range rather than the accidental finite-
arithmetic result.

For out-of-range values, classification is **intentional divergence**.
TensorDSLab gains and clamps analog voltage before quantization, validates gain,
range, bit depth, and dtype, and forbids unsigned-wrap accidents. It enforces
the donor's intended inclusive analog-gain range of `[0, 40]` dB, correcting
IV's impossible conjunction that failed to reject out-of-range gain.

Validation should cover exact lower/upper endpoints, half-step and near-step
values, negative and over-range inputs, gain boundaries, all accepted bit
depths, dtype-rounded threshold collapse/rejection, and ADC bounds. The exact
endpoint claim is scoped to the documented field-dtype thresholds, not to an
unrepresentable real-number voltage between adjacent floating values.

## RNG Donor Parity And Backend Agreement

TensorDSLab does not target fixed-seed or bitwise RNG-stream parity with
IV-DSLab. The algorithms consume different representations, group draws
differently, use different operation order, and require position-addressed
device-resident streams.

The selected aggregate multinomial implementation uses TensorDSLab-owned
sequential category planning with stable prepared current/later-category
masses. Closed Stage 6 executed the frozen forward-CDF/BTRS mapping through
private `readout/_random.py`; the closed Maintenance 2 implementation calls
that promoted mapping through TensorCore's `binomial(...)`. It does not
repeatedly subtract represented categories from one; that implementation detail was
rejected after the timing-jitter numerical study corrupted small residual
tails. TensorCore ownership is implemented at exact version `0.9.0`; this does
not imply that IV or its NumPy dependency uses the same binomial algorithm or
consumes uniforms in the same order. Parity is therefore assessed against the
named binomial/multinomial probability laws and their moments, covariance,
boundaries, and tails rather than returned samples.
The frozen BTRS acceptance calculation uses a finite binary64 Stirling-tail
approximation and therefore does not claim ideal bitwise Binomial sampling.
Its accepted cancellation-resistant grouping is algebraically identical to the
earlier three-log form and supports aggregate counts through `2**53 - 1`.
Central candidates own the frozen `1e-6` absolute local gate; complete support
uses the documented mixed absolute/relative side allowances, decision-
separation rule, and fixed-word uncertainty-band ownership. The Stage 6
statistical-law policy separately validates the resulting Binomial law.

The rebuild RNG targets are:

- exact repeatability for identical TensorDSLab input values, shape, dimension
  order, coordinate order, config-owned keys, dtype, RNG algorithm/version,
  and invocation seed on the same unchanged numerical execution stack;
- TensorCore-owned exact Threefry raw-word and fixed-point-uniform agreement
  between accepted implementations;
- exact same-stack integer avalanche history across float32 and float64
  Charge requests because discrete sampler control is always binary64;
- cross-backend statistical agreement for completed Box-Muller, PSD, Poisson,
  conditional-binomial, and aggregate-multinomial values with the accepted
  probability kernels;
- finite-sample statistical validation as evidence for that cross-backend
  agreement contract;
- no CPU/GPU bitwise guarantee for completed values involving transcendental
  functions or FFTs.

CUDA was unavailable for the merged Stage 5 and Stage 6 evidence. Their eager
CPU results therefore do not establish raw-word, completed-value, or Charge
cross-backend agreement; those bullets remain explicit later validation
targets.

The parity classifications above apply to the comparison with IV-DSLab. The
cross-backend requirement compares TensorDSLab implementations to one accepted
TensorDSLab probability contract and is therefore called agreement, not donor
parity.

Semantic coordinates are not RNG identities. Reordering axes or coordinates,
reindexing payloads, selecting, or invoking arbitrary chunks generally changes
logical flat positions and therefore sampled values. Positional addresses
restart for each builder invocation. Product-request changes preserve a common
product because package-owned role keys are fixed, but chunk stability would
require explicit global positional offsets and a later Design contract.

The Maintenance 2 implementation uses immutable TensorCore `CounterRng` instances
and config-owned `RngKey` values. Reusing one RNG intentionally replays the
same realization; changing algorithms intentionally need not. Stage 5/6's
central `_RngStream` and bare seed remain closed historical implementation
evidence. Its default keys preserve the same stream values and internal
continuity without claiming donor RNG parity. CUDA was unavailable, so this
migration adds no completed-value cross-backend claim.

## Condition-Database Variation

IV uses package-load global configuration, detector channel maps, and a
condition database for per-channel dark-count, crosstalk, afterpulse, charge
spread, amplitude, and RMS values.

The first TensorDSLab MVP uses scalar scientific configs uniformly across one
invocation. Per-channel variation is **deferred**, not rejected scientifically.
The rebuild anticipates a typed device-resident parameter tensor aligned to the
exact `ChannelAxis`, with explicit supported axes, units, provenance,
validation, movement, and lifetime rules. TensorDSLab must not import the
legacy DB or channel-map runtime to claim parity.

## Accepted Intentional Divergences

The rebuild accepts these donor differences:

- start from the post-binned `Photoelectrons` primary-seed field rather
  than sparse PE rows;
- add private dark roots before timing redistribution and then branch from the
  post-jitter truth-plus-dark frontier;
- keep dark-count, generation-frontier, mechanism, overflow, S1, and S2 values
  private inside `_produce_charge(...)` rather than exposing avalanche products;
- use fixed-`K` unmarked recursive feeding with independently phase-marginalized
  causal edges rather than IV's sparse queue and source-charge-dependent quirks;
- keep DiCT and DeCT as separate ordinary-Poisson mechanisms and make no
  IV-parity claim for DeCT;
- use an ordinary exponential mean-delay model rather than IV's literal
  reciprocal-exponential afterpulse expression;
- use unit AP charge when recovery is absent or conditional-mean binned
  recovery weights when it is configured, without recovery-dependent future
  branching;
- use an aggregate charge-smearing boundary and explicit negative policy;
- omit eventwise sub-bin pulse-amplitude correction that cannot be reconstructed
  from binned input and defer an available phase-marginalized correction;
- generate direct exact-length PSD-shaped noise rather than bank/crop baselines;
- separate pure, noise, analog, and digitized field roles at one declared
  pre-digitization voltage reference plane;
- use position-addressed counter RNG rather than donor global/sequential streams;
- clamp before integer ADC conversion and validate gain/range constraints.

These choices are acceptable for the rebuild because they support a bounded,
reviewable tensor-native path. They are not evidence of end-to-end eventwise or
distributional IV parity; the narrower statistical target for
requested `Charge` requires its own named ensemble observables and tolerances.

## Validation And Fixture Rules

Parity evidence should prefer the strongest stable form:

1. analytic probability kernels and identities;
2. tiny deterministic fixtures;
3. numerical fixtures with named tolerances;
4. finite-sample statistical studies when no stronger evidence is available.

Every parity fixture must state:

- donor source path/symbol and snapshot identity;
- TensorDSLab comparison boundary;
- exact input and result product types plus requested collection subset for a
  public boundary, or an explicit private-ephemeral designation for an internal
  submodel observable;
- the exact `DigitizedWaveformConfig` for ADC comparisons;
- units, exact axis types and coordinate order, tensor dimension order, dtype,
  and configuration;
- parity classification;
- assumptions and accepted intentional divergences;
- RNG algorithm/version, invocation seed, exact `RngKey`, logical position,
  source-quantum ordinal, distribution ordinal/count, and raw-word schedule
  when sampled;
- execution mode and backend when needed for reproducibility, while treating
  scheduling, scratch, device-stream placement, fusion, and allocation choices
  as nonsemantic;
- observable, tolerance, sample size, and confidence/error criterion;
- whether it is exact/numerical evidence or distribution/statistical evidence.

Tests must not import or execute IV-DSLab or DSLab at runtime. Reference values
should be small, reviewed, and owned by TensorDSLab. A large binary donor output
or opaque Monte Carlo dump is not a substitute for an explained contract.

Suggested minimum parity observables are:

| Operation | Required observables |
| --- | --- |
| timing jitter | analytic transition/drop probabilities, displacement mean/variance, multinomial covariance, conservation including drops, and no-cutoff edge behavior |
| dark counts | exposure-normalized rate, variance, zero probability, cell covariance, edge cells |
| crosstalk | multiplicity mean/variance, zero probability, tail quantiles, generation policy, fixed/exponential offset PMF/time profile, right-overflow fraction |
| afterpulses | fire probability, delay law/CDF, overflow fraction, recovery-amplitude relation, integer-count/charge covariance, omitted within-category recovery variance, recursion policy |
| charge smearing | conditional mean/variance, negative tail, clipping bias, heterogeneous-charge exclusions |
| end-to-end requested `Charge` | per-channel/sample charge mean and variance, zero-cell probability, total response, occupancy, edge loss, selected tails and time profile |
| pure waveform | template samples, peak, area, time-to-peak, impulse response, truncation/edge behavior |
| white noise | mean, RMS, marginal law, covariance, and positional repeatability under an unchanged invocation identity |
| PSD-shaped noise | RMS, integrated power, PSD, autocorrelation, endpoint policy, marginal law |
| analog waveform | exact sum, clip order, polarity, units, common component reference plane |
| digitization | gain/map/clip/quantize order, transfer-curve boundaries, ADC bounds |

Production work orders must replace qualitative phrases such as “close enough”
with concrete tolerances and sample sizes before implementation is accepted.

Stage 6 conformance to the selected TensorDSLab model used the frozen
statistical policy with seeds `0`, `1`, `0x0123_4567_89ab_cdef`, and
`0xffff_ffff_ffff_ffff`; `M=2**18` for scalar/one-parent laws and `M=2**16`
for aggregate `Q=32`, small-grid `K<=3`, and completed-`Charge` fixtures. One
example is one independent replicate. Each predeclared statistic must satisfy
`abs(observed-target) <= 8*SE + delta`, where target-law moments determine
`SE` and
`delta = 64*eps(dtype)*max(1, ceil(log2(length)))*abs(scale)`. Frequency
assertions require at least 256 expected hits and misses; rarer cases use
probability and fixed-word evidence. These model-conformance tolerances do not
upgrade an intentional donor divergence.

No universal IV percentage is accepted. A future observable-specific donor
margin `Delta` must come from calibration or collaborator review and requires
`abs(theta_TensorDSLab-theta_IV) + 8*combined_SE <= Delta`, with its sample
size frozen so the eight-SE half-width is at most `Delta/2`. Until then,
finite-`K` recursion, DeCT, corrected AP delay/recovery, clipped smearing, and
detector-level requested Charge remain unestablished IV statistical targets,
not Stage 6 implementation failures. This is an honest absence of a donor
equivalence claim rather than an open numerical implementation tolerance.

## Deferred Decisions

- CUDA Charge execution, CPU/CUDA agreement, and GPU performance/fusion;
  eager CPU implementation, sampler, and public Stage 7 orchestration evidence
  under the selected aggregate algorithms is closed;
- per-channel parameter representation;
- whether any calibrated crosstalk approximation should match selected IV
  moments or remain a standalone TensorDSLab model;
- whether pure rendering needs a latent-phase-marginalized amplitude
  correction to meet peak/area tolerances;
- observable-specific IV equivalence margins for intentionally divergent
  Charge mechanisms and later integration slices;
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
- changing the selected fixed-`K` correlated-avalanche law or substituting a
  different correlated-avalanche algorithm;
- changing the accepted ordinary exponential afterpulse delay, unit-recovery
  mode, or composed exponential recovery response;
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
