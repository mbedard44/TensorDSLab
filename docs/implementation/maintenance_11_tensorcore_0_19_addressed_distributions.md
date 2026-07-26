# Maintenance 11 TensorCore 0.19 Addressed Distributions

Status: **Design-complete / Implementation pending**

Stable key:
`TensorDSLab/maintenance-11-tensorcore-0-19-addressed-distributions`

## Purpose

Prepare one package-owned TensorDSLab migration from the published
TensorCore `0.16.0` random surface to published TensorCore `0.19.0`'s
class-first addressed Distribution and TensorKernel surfaces, and add one
focused educational notebook that makes the resulting counter-based RNG
architecture inspectable through a delayed-crosstalk example.

This is the package-owned executable production work order. It freezes the
accepted migration decisions, exact TensorCore publication evidence,
predeployment RNG rebaseline, changed-path ceiling, protected paths, and
required evidence.

The user authorized finalization and Implementation after accepting the
predeployment stream rebaseline. This authority permits only the
Implementation-to-Validation-to-Review candidate loop described below. It
does not authorize compatibility claims, CUDA work, merge, push, publication,
or release.

This maintenance follows:

- `CONTRIBUTING.md` for dependency ownership, private Runtime records,
  validation, typing, tests, documentation, and scope discipline;
- `docs/architecture/rebuild.md` for the active TensorDSLab scientific and
  product model;
- `docs/architecture/tensors.md` for TensorCore-backed semantic tensor
  ownership, placement, synchronization, and relationship boundaries; and
- `docs/parity.md` for intentional scientific divergence, RNG comparison,
  exact-versus-statistical evidence, and required rebaseline declarations.

## Current TensorDSLab Baseline

The exact current published TensorDSLab baseline is:

```text
TensorDSLab main/origin-main:
    3052f24f051203a914bc2204a4657e197498315f
TensorDSLab tree:
    97d913269653a3257efc49b89a29d22f6ef4209f
package version:
    0.1.0
Python:
    >=3.14
Torch:
    >=2.13,<2.14
TensorCore pin:
    e05324699892a8bcea024375720bfae1ed9569cc
TensorCore resolved version:
    0.16.0
```

Maintenance 10 is Merged / Closed at this baseline. TensorDSLab remains an
eager-CPU development preview. No current integrated CUDA, production
readiness, deployment, calibration, or broad compatibility claim exists.

The current stochastic implementation uses:

- `RngPositions`;
- public `CounterRng.gaussian(...)`, `poisson(...)`, and `binomial(...)`
  methods;
- manual positional `.offset(...)` arithmetic for category and generation
  identity;
- TensorDSLab-owned sequential `draw_ordered_categories(...)`;
- private crosstalk and afterpulse overflow draws and bookkeeping; and
- ten package-owned role keys in namespace `0x54445331`.

Those are operative current facts until this maintenance is separately
completed. This work order does not silently supersede the current
source, architecture, parity, or validation contract.

## Selected TensorCore Target

The exact published TensorCore dependency target is:

```text
repository:
    https://github.com/mbedard44/TensorCore.git
live refs/heads/main / containing commit:
    ed17f4b637258f0a7f4544f235648b747f17fa44
containing tree:
    ef8c706a83d03c3bd8b4094855af8064ab86743b
containing parent:
    e9ad9c09f108e9e7951bd5efcdafa43afad8bf7b
TensorCore Stage 26 package anchor:
    fdfc96d428d62847dddc1a52eb956dd598874ae1
TensorCore package tree:
    ff1fc63d6453f3a3e6aa684df7d6d2ad96b30e9c
package version:
    0.19.0
```

The containing commit is selected as the immutable future TensorDSLab
dependency pin. This Design selection does not edit `pyproject.toml`, adopt
the dependency in the current package, or claim compatibility.

TensorCore's publication handoff records:

```text
containing wheel:
    51208 bytes
    SHA-256 9666ff7811cc1bdfe289290f3ae18517d7f47c316ad1ff766477a7d08f075dda
containing source archive:
    492382 bytes
    SHA-256 a76983491a0b6dc019be725695010f38707a5a90a0cfd151da4596eab77fef07
```

The containing commit changes only approved publication-lifecycle
documentation after the package anchor. Its `tensor_core/`, `tests/`, and
`pyproject.toml` bytes are exact to `fdfc96d`. README-derived wheel METADATA
and RECORD are the only accepted wheel differences from the package-anchor
artifact.

The published Stage 26 package surface provides:

- `RngElements` and `RngAddress`;
- `CounterRng.words(...)` with `_generate_block(...)` retained as the
  protected subclass/test-double hook;
- `Distribution`;
- `UniformDistribution`;
- `GaussianDistribution`;
- `PoissonDistribution`;
- `BinomialDistribution`;
- `MultinomialDistribution`;
- `TensorKernel`;
- `ProbabilityKernel`; and
- `require_kernel_dimensions(...)`.

The exact published export census is:

```text
tensor_core:                         30
tensor_core.scalar:                   7
tensor_core.tensor:                   9
tensor_core.table:                    3
tensor_core.random:                  11
tensor_core.random.generator:         2
tensor_core.random.distribution:      6
tensor_core.scalar.validation:        3
tensor_core.tensor.validation:       16
tensor_core.random.validation:        1
```

Local read-only verification confirms local main, tracking `origin/main`, and
the containing tree at exact `ed17f4b`, and confirms protected package bytes
are identical to the Stage 26 anchor. Dispatch must reverify the live ref and
artifact identities. No alias for `RngPositions`, old CounterRng Distribution
methods, or retired import path is accepted.

## Selected Ownership Boundary

TensorCore owns:

- immutable generic `RngElements` identities and retained root capacity;
- collision-safe virtual `RngAddress` encoding;
- deterministic counter words;
- generic Distribution validation, preparation, and execution;
- generic TensorKernel and ProbabilityKernel representation;
- probability value, total, and reverse-suffix preparation;
- generic count-law and tensor validation; and
- the exact Stage 26 execution-invariance contract for identical complete
  addresses and law inputs.

TensorDSLab owns:

- the fixed readout RNG namespace and scientific role keys;
- the meaning and hierarchy of every address-domain dimension;
- sample, generation, mechanism, and application address schemas;
- kernel-index-to-sample-displacement mapping;
- anchors, causality, finite-window mapping, and boundary discard;
- timing-jitter, fixed-delay, exponential-delay, afterpulse-occurrence,
  recovery, and mean-offspring physical laws;
- deterministic frontier-to-rate construction;
- count accumulation and Charge ledger ceilings;
- product preparation, production, validation, and orchestration;
- the exact scientific and RNG rebaseline; and
- every donor-parity or intentional-divergence classification.

TensorCore must not import TensorDSLab axes, configs, products, effects,
detector geometry, role keys, or scientific policy. TensorDSLab must not add a
generic RNG facade, duplicate TensorCore Distribution classes, forward retired
TensorCore APIs, or expose private random mechanics through its public facade.

## Atomic Migration Requirement

TensorCore `0.19.0` removes the old `RngPositions` and CounterRng Distribution
methods without compatibility shims. TensorDSLab therefore adopts the exact
published dependency and migrates every stochastic production call in one
atomic package candidate.

Implementation may organize the work internally in coherent steps, but no
intermediate published or merged state may:

- pin TensorCore `0.19.0` while retaining old random APIs;
- add a TensorDSLab compatibility wrapper;
- carry `RngPositions | RngElements` unions;
- retain old and new address schemes in parallel; or
- claim compatibility before the complete TensorDSLab package gate closes.

## Address Model

### Package-Owned Role Table

`tensor_dslab/readout/runtime/keys.py` remains the sole owner of namespace
`0x54445331` and fixed role streams.

Existing active role identifiers are not reassigned. Stream identifiers
retired by this maintenance are removed from the active table. Because this is
a predeployment package-wide RNG rebaseline, their former numeric values carry
no permanent reservation or compatibility promise. Any later reuse still
requires an explicit future Design decision.

The selected role treatment is:

| Existing role | Future treatment |
| --- | --- |
| white noise | retained |
| PSD noise | retained |
| dark counts | retained |
| direct crosstalk retained | retained for collapsed destination Poisson |
| direct crosstalk overflow | retired with no surviving key |
| delayed crosstalk retained | retained for collapsed destination Poisson |
| delayed crosstalk overflow | retired with no surviving key |
| timing jitter | retained for kernel allocation |
| afterpulse | retained for occurrence and delay under distinct quanta |
| charge smearing | retained |

`AFTERPULSE_RNG_KEY` remains exact stream `0x0000_0009`. Afterpulse occurrence
uses quantum `0`; afterpulse delay allocation uses quantum `1`. This keeps one
scientific role key while giving the two laws collision-distinct counter
domains. No new stream is added, and the retired overflow streams are not
repurposed by this maintenance.

### Address Construction

Add one real private shared owner:

```text
tensor_dslab/readout/runtime/addresses.py
```

It contains only role-named address construction mechanics shared by multiple
products. It is not exported and is not a general schema, registry, mutable
plan, stream, cursor, or RNG wrapper.

The golden path is:

1. Create the canonical full-product `RngElements` lattice once when its
   shape/device is known.
2. Use only trusted `movedim`, `select`, and `slice` transformations.
3. Preserve original values and retained root capacity across source or chunk
   selection.
4. Encode application, generation, and kernel-cell identity in
   package-frozen `RngAddress` shapes and selections.
5. Never reproduce the removed `.offset(...)` arithmetic.
6. Never renumber selected sources or chunks.
7. Never derive semantic RNG identity from TensorAxis coordinates or labels.

Atomic laws receive an address whose remaining domain shape is `()`.
Kernel Multinomial laws receive an address whose remaining shape is exactly
the selected kernel shape.

For generation-indexed collapsed crosstalk, the selected conceptual address
is:

```python
root = RngAddress.root(
    key=mechanism_key,
    elements=destination_elements,
    shape=(maximum_generations,),
    quantum=0,
)
generation_address = root.select(generation_index)
```

`generation_address` is atomic and its element shape equals the complete
destination-rate tensor.

For generation-indexed afterpulse occurrence, the selected conceptual address
uses the complete frontier elements:

```python
occurrence_root = RngAddress.root(
    key=AFTERPULSE_RNG_KEY,
    elements=frontier_elements,
    shape=(maximum_generations,),
    quantum=0,
)
occurrence_address = occurrence_root.select(generation_index)
```

For delay allocation from one selected source sample, the source
`RngElements` retain their original identities and root capacity:

```python
delay_root = RngAddress.root(
    key=AFTERPULSE_RNG_KEY,
    elements=source_elements,
    shape=(maximum_generations, *delay_kernel.shape),
    quantum=1,
)
delay_address = delay_root.select(generation_index)
```

`occurrence_address` is atomic. `delay_address.shape` equals
`delay_kernel.shape`, as required by `MultinomialDistribution`. Source identity
comes from the non-renumbered selected `RngElements`; no extra source-domain
dimension or positional offset is added.

`produce_charge(...)` constructs one canonical full-product `RngElements`
lattice when at least one stochastic Charge effect is active and passes that
same exact object through the effect sequence. A fully deterministic Charge
request constructs no random lattice. `produce_noise_waveform(...)` constructs
one model-specific lattice: the complete output lattice for white noise, the
coefficient lattice for PSD noise, and no lattice for exact-zero noise.

Reusable timing-jitter, delay, and recovery probability kernels are prepared
once by the existing Charge preparation actions and stored by their existing
private Runtime records. Dynamic laws whose operands depend on the current
frontier remain execution-local. White-noise, PSD-noise, dark-count, and charge-
smearing Distribution objects are one-use execution values and do not add
Runtime fields merely to cache one draw. Per-source and bounded-chunk
selection remains downstream execution policy.

## Distribution Selection By Effect

| TensorDSLab effect | TensorCore `0.19.0` execution |
| --- | --- |
| White noise | scalar `GaussianDistribution` |
| PSD noise | scalar standard `GaussianDistribution(count=2)` |
| Dark counts | scalar `PoissonDistribution` |
| Charge smearing | tensor-valued `GaussianDistribution` |
| Timing jitter | `ProbabilityKernel` plus `MultinomialDistribution` |
| Afterpulse occurrence | `BinomialDistribution` |
| Afterpulse delay | conditional delay `ProbabilityKernel` plus `MultinomialDistribution` |
| Direct crosstalk | deterministic collapsed destination rates plus tensor-valued `PoissonDistribution` |
| Delayed crosstalk | deterministic collapsed destination rates plus tensor-valued `PoissonDistribution` |

`UniformDistribution` has no current production consumer and must not be added
merely to exercise the dependency surface.

Reusable immutable ProbabilityKernels are stored by the existing private
Charge Runtime records when preparation owns all their inputs. No scalar
Distribution is stored solely to reuse a one-draw value. Each Runtime record
remains a final frozen slotted dataclass with no base, execution method,
mutable cache, Config, product, or collection.

Dynamic laws whose tensor parameters depend on the current frontier, counts,
or Charge ledgers are constructed during product execution. Their generic
validation and snapshots remain TensorCore-owned; downstream preparation must
not duplicate generic scans merely to preserve the previous helper shape.

## White And PSD Noise

White noise uses:

```python
GaussianDistribution(
    mean=0.0,
    standard_deviation=represented_rms,
    dtype=floating_dtype,
    ordinal=0,
    count=1,
)
```

over one atomic address built from the full output `RngElements`.

PSD noise retains the current coefficient lattice and DC exclusion. It uses:

```python
GaussianDistribution(
    mean=0.0,
    standard_deviation=1.0,
    dtype=floating_dtype,
    ordinal=0,
    count=2,
)
```

over non-renumbered sliced frequency elements. TensorDSLab still owns PSD
integration, coefficient scaling, Hermitian endpoint policy, inverse FFT, and
finite-output validation.

No scientific noise law, Config, units, field, or public surface changes.
Exact completed values must nevertheless be rebaselined against the new
TensorCore address/Distribution execution rather than assumed byte-identical.

## Dark Counts

Dark counts use one scalar `PoissonDistribution(mean=prepared_mean)` over the
complete product lattice and an atomic dark-count address.

TensorCore owns the exact accepted Poisson mean domain and sampling mechanics.
TensorDSLab retains physical rate-to-bin-exposure preparation, count addition,
the Charge count ceiling, and immediate product relationships.

The exact-zero configured mean remains an accepted word-free no-effect path.

## Charge Smearing

Charge smearing uses:

```python
GaussianDistribution(
    mean=charge_pe,
    standard_deviation=scale,
    dtype=charge_pe.dtype,
    ordinal=0,
    count=1,
)
```

where TensorDSLab still computes:

```text
scale = represented_sigma * sqrt(charge_square_sum)
```

TensorCore owns tensor parameter structure, snapshots, finite envelope, mixed
zero-scale partitioning, address-preserving selection, and Gaussian
execution. TensorDSLab retains S1/S2 validation, relative-width science,
finite ledger envelope, nonnegative output policy, and result checks.

The configured zero-sigma path remains word-free. Mixed zero-scale cells also
become TensorCore-owned word-free cells. Their completed outputs and raw-word
trace require an explicit new baseline.

## Timing Jitter

Define one final fieldless downstream semantic
`TimingJitterProbabilityKernel` over the TensorCore `ProbabilityKernel`
representation. Its sole operation-axis role is `SampleAxis`.

For sample count `N`, the represented kernel has shape `(2 * N - 1,)` and
row-major cell order:

```text
-(N - 1), ..., -1, 0, 1, ..., N - 1
```

Kernel index `i` maps downstream to signed displacement:

```text
displacement = i - (N - 1)
```

If the existing prepared probabilities are:

```text
q[0], q[1], ..., q[N - 1]
```

the represented tensor is:

```text
q[N - 1], ..., q[1], q[0], q[1], ..., q[N - 1]
```

and the independently prepared non-cell completion probability is:

```text
2 * left_tail[N - 1]
```

TensorDSLab selects one source sample or one bounded deterministic source
chunk, constructs `MultinomialDistribution` with the source counts, kernel,
and completion probability, draws represented displacement allocations, maps
each kernel index to a destination sample, accumulates in-window counts, and
discards represented allocations mapped outside the finite SampleAxis.

The terminal completion count is not returned, addressed, or sampled.
TensorDSLab retains the analytic Gaussian integration, accepted ratio/sample
domains, local and complete-law tolerances, source/destination mapping, count
ceiling, and product validation.

This migration preserves the declared probability law, not the old exact
category traversal or returned samples. It must receive new exact address,
conservation, boundary, and statistical evidence.

## Afterpulse

Afterpulse execution is factored into two distinct laws:

1. `BinomialDistribution(counts=frontier, probability=fire_probability)`
   decides which parents produce an afterpulse.
2. A fixed conditional delay `ProbabilityKernel` and
   `MultinomialDistribution` allocate successful occurrences across
   represented nonnegative delay cells.

For sample count `N`, the conditional delay kernel has shape `(N,)`, cell
`i` maps to delay offset `i`, and the independently prepared completion
probability is `right_tail[N]`.

TensorDSLab maps represented offsets to `source + offset`, discards
out-of-window allocations, and applies the existing conditional-mean recovery
weight to the same represented category allocation. A downstream
`AfterpulseRecoveryKernel(TensorKernel)` may own the aligned represented
recovery weights when that removes duplicated shape/axis checks; it remains
TensorDSLab-owned and must not repeat ProbabilityKernel validation.

The following retire:

- the explicit outside-window afterpulse category;
- returned afterpulse overflow count;
- overflow recovery weights and charge;
- final no-event/remainder output; and
- manual category position offsets.

Occurrence and delay allocation require collision-distinct addresses. Their
exact selected split is the shared stream-`0x0000_0009` key with quantum `0`
for occurrence and quantum `1` for delay allocation.

This factorization preserves the intended occurrence and conditional delay
laws but deliberately rebaselines exact results and word schedules.

## Direct And Delayed Crosstalk

### Selected Collapsed Rate-First Formulation

Direct and delayed crosstalk do **not** draw total offspring followed by
Multinomial allocation.

TensorDSLab deterministically applies the validated delay
`ProbabilityKernel` to the current generation frontier and constructs the
complete retained destination rate:

```text
lambda[destination]
    = sum_source(
        frontier[source]
        * mean_offspring
        * probability[destination - source]
      )
```

Only causal represented offsets contribute. TensorDSLab performs the
accumulation in binary64, validates every destination rate against its
scientific and TensorCore Poisson domain, constructs one tensor-valued
`PoissonDistribution(mean=lambda)`, and draws the complete destination tensor
over the mechanism's atomic generation address.

This is the exact Poisson splitting and superposition formulation, not an
approximation:

- a Poisson offspring process thinned into delay categories gives independent
  category Poisson variables; and
- independent contributions arriving at one destination superpose into one
  Poisson variable whose rate is their sum.

This explicitly resolves TensorCore's `1e8` Poisson-mean adoption decision.
It is the algebraic collapse of category-wise addressed Poisson thinning, not
a total-offspring Poisson. Admission is therefore governed by each retained
destination mean; no scientifically unused total-source mean is constructed.

The selected formulation:

- closely matches current TensorDSLab destination-rate construction;
- avoids a source-total Poisson draw;
- avoids sequential conditional Binomial execution;
- avoids `source_shape + kernel_shape` allocation materialization;
- avoids downstream category scatter;
- avoids draws for represented cells outside the finite window; and
- preserves the useful accepted domain where every destination rate is at
  most `1e8` even if an unused total source mean would exceed `1e8`.

The finite-window tail requires no draw or returned count. Restricting the
Poisson process to represented retained destinations is exactly the thinned
retained-rate law.

TensorDSLab retains kernel displacement semantics, causality, finite-window
mapping, mean-offspring science, deterministic rate construction, generation
frontier order, count accumulation, and ledger validation. TensorCore owns
the generic kernel representation, tensor-valued Poisson validation and
execution, and addressed words.

### Retired Crosstalk State

The following retire without replacement:

- direct-crosstalk overflow Poisson draw;
- delayed-crosstalk overflow Poisson draw;
- overflow rate tensor;
- overflow result and cumulative fields;
- overflow role-key use; and
- any overflow recovery/bookkeeping path.

The two historical overflow key constants and semantic assignments are
removed. Their former numeric stream values receive no reservation or
compatibility status through this predeployment rebaseline.

The public `Charge` product already excludes these private diagnostic values.
Their removal changes private execution and exact RNG traces, not the public
field schema.

## Kernel Placement And File Ownership

Do not create a generic TensorDSLab kernel framework.

The expected narrow owners are:

- `readout/charge/runtime/effects/timing_jitter.py` for
  `TimingJitterProbabilityKernel` and jitter mapping;
- `readout/charge/runtime/effects/delays.py` for the shared fixed/exponential
  delay ProbabilityKernel and optional aligned recovery kernel;
- `readout/charge/runtime/effects/correlated_avalanches.py` for crosstalk
  rate construction, afterpulse factorization, finite-window mapping, and
  generation orchestration; and
- `readout/runtime/addresses.py` for role-named shared address construction.

Add a new module only if the final source inventory demonstrates real shared
behavior that cannot remain coherently in these owners. No placeholder
`kernels/`, `distribution/`, schema registry, callback framework, or generic
execution root is accepted.

## Low-Level Cleanup

After every consumer migrates, retire:

- `RngPositions`;
- `original_positions(...)`;
- `.offset(...)` address arithmetic;
- `draw_ordered_categories(...)`;
- its success/failure-mass tuple protocol;
- `checked_subtract(...)` if no scientific caller remains;
- duplicate generic count or Poisson-domain validation already owned by the
  exact TensorCore constructors; and
- private overflow fields, calculations, and tests made unreachable by the
  no-tail-output decision.

Retain:

- `checked_add(...)` and the TensorDSLab Charge accumulation ceiling;
- genuine TensorDSLab rate-product and ledger relationships;
- physical probability and recovery preparation;
- dtype-representability policy where it is scientific;
- product axes/device/dtype/freshness checks;
- generated-product validation; and
- the public `simulate_readout(...)` orchestration contract.

The final work order must inventory each helper and prove that every deletion
is unused. It must not move scientific policy into TensorCore merely because
that policy validates a tensor.

## Mechanical Migration Versus Rebaseline

### Mechanical Dependency Work

The mechanical portion includes:

- pinning the exact future TensorCore `0.19.0` GitHub containing commit;
- updating exact dependency source/archive identities and artifact hashes;
- changing `RngPositions` imports and annotations to `RngElements`;
- replacing CounterRng Distribution-method calls with immutable Distribution
  construction and `.draw(...)`;
- adding exact `RngAddress` construction;
- updating package/export/import-isolation probes;
- preserving all current test doubles through the protected
  `_generate_block(...)` hook;
- proving no test or downstream subclass overrides final
  `_checked_block(...)`; and
- updating static typing and negative dependency fixtures.

### Deliberate Scientific And RNG Rebaseline

The following are not mechanical:

- every role's complete address-domain shape and selection order, including
  the selected afterpulse quantum split;
- timing-jitter displacement kernel order;
- boundary-discard behavior under represented cells;
- collapsed crosstalk destination-rate execution;
- afterpulse occurrence/delay factorization;
- removal of overflow draws and outputs;
- mixed-zero-scale Gaussian word use;
- Stage 26 Poisson/Binomial execution;
- exact completed stochastic fixtures; and
- chunk/traversal invariance.

TensorDSLab must establish a new exact package baseline. It must not claim
completed-value, physical block-call, synchronization, or raw trace continuity
from TensorCore `0.16.0`.

Where the mathematical output law is unchanged—such as collapsed crosstalk
through Poisson splitting/superposition—the work order must state
distributional equivalence and separately acknowledge exact sample
rebaselining.

`docs/parity.md` must be synchronized in the production candidate to replace
obsolete current RNG/category/overflow wording and to name the new address,
chunk, traversal, completion, and boundary evidence. Until that candidate is
closed, the current parity document remains operative.

## Public And Product Boundary

The intended TensorDSLab public facade remains unchanged:

```text
tensor_dslab:          35
tensor_dslab.common:    5
tensor_dslab.readout:  30
```

No public Config, field, collection, profile, or
`simulate_readout(...)` signature changes.

The migration changes private Runtime fields and execution ownership only.
Pint remains confined to public Config construction and preparation.
Runtime records, Distribution inputs, kernels, addresses, tensor payloads,
producers, and validators remain quantity-free.

The `ds20k_veto()` profile and public demos remain provisional. Their exact
stored stochastic outputs may change under the accepted RNG rebaseline, but
their Config values, source, axes, CPU boundary, product relationships, and
plot contract do not change unless a separately authorized demonstration
amendment says otherwise.

## Addressed-Randomness Demonstration

Add one new CPU-only executed notebook:

```text
demos/random.ipynb
```

The notebook is an educational source artifact for the exact adopted
TensorCore version. It is not a second stochastic implementation, a supported
TensorDSLab RNG facade, or a user-configurable scientific-key surface.

The notebook must use the real public TensorCore classes:

- `RngKey`;
- `RngElements`;
- `RngAddress`;
- `Threefry4x32`;
- `CounterRng.words(...)`;
- `ProbabilityKernel`; and
- `PoissonDistribution`.

It may import the package-owned
`DELAYED_CROSSTALK_RETAINED_RNG_KEY` from the precise private
`tensor_dslab.readout.runtime.keys` module solely to show the actual role
identity used by TensorDSLab. The surrounding Markdown must state that this
module is an unsupported implementation detail, that users do not configure
production role keys, and that importing it in the notebook creates no public
compatibility promise. The notebook must not duplicate the namespace or
stream integer as a second literal.

### Narrative

The notebook should proceed in this order:

1. Explain that counter-based generation is a pure mapping from seed, role
   key, complete address, and raw-word ordinal to deterministic words. There
   is no mutable stream position and no use of Torch's global RNG state.
2. Construct one small CPU `torch.int64` tensor named `parent_counts`,
   representing the current delayed-crosstalk avalanche frontier over a
   `SampleAxis`.
3. State explicitly that this private execution frontier is not the public
   floating-point `Charge` TensorField, even though it contributes to the
   eventual Charge product.
4. Construct canonical row-major `RngElements` for the destination shape and
   show their public shape/device metadata. If explicit logical integers are
   displayed, retain the caller-owned source tensor used to construct the
   elements; do not expose or depend on TensorCore private element storage.
5. Construct a generation-domain `RngAddress.root(...)`, select one generation
   into an atomic address, and display the root/selected shape,
   `element_shape`, device, and completion state.
6. Request a small frozen tuple of raw ordinals with
   `CounterRng.words(...)`, display the resulting word table, and explain how
   Threefry maps address identity to words. Do not copy or reimplement
   Threefry rounds, constants, packing, or private permutation methods.
7. Construct a demo-local final fieldless delayed-crosstalk
   `ProbabilityKernel` leaf over `SampleAxis`, using a short causal
   nonnegative delay-probability tensor.
8. Deterministically form retained destination means:

   ```text
   lambda[destination]
       = sum_source(
           parent_counts[source]
           * mean_offspring
           * probability[destination - source]
         )
   ```

   with no contribution from negative or out-of-window displacements.
9. Construct one tensor-valued `PoissonDistribution(mean=lambda)` and draw
   the complete destination avalanche tensor over the atomic generation
   address. This is the selected one-step Poisson splitting/superposition
   formulation; the notebook must not draw a source-total Poisson or invoke
   Multinomial for crosstalk.
10. Repeat the identical draw and prove exact equality.
11. Slice the original `RngElements` into at least two deterministic
    destination chunks, rebuild the same generation-domain address over each
    retained-capacity slice, draw the corresponding mean slices, concatenate
    them, and prove exact equality with the full draw. Constructing fresh
    renumbered chunk elements is prohibited.
12. Snapshot Torch's global RNG state before the addressed operations and
    prove it remains byte-identical afterward.

The notebook should include compact plots of:

- the parent avalanche frontier;
- the represented delayed-crosstalk probability kernel;
- the deterministic destination-rate tensor; and
- the sampled destination avalanche counts.

It should also include a small table relating displayed destination element
identities, selected address-domain state, raw ordinals, and returned
Threefry words. The dependency and delayed-crosstalk address schema are now
selected; exact displayed values are frozen during implementation and then
locked by the committed notebook proof.

The existing root `create_environment.sh` and the existing `[demos]` optional
dependency group remain the sole environment path. The new notebook adds no
dependency and contains no environment creation or activation cell. Its first
Markdown cell directs users to create and activate `tensor_dslab` before
launching Jupyter, matching `demos/readout.ipynb`. The script's installed
dependency smoke assertion must advance from TensorCore `0.16.0` to exact
`0.19.0` in the same candidate.

The notebook must be committed with a clean deterministic execution:

- CPU device only;
- one supported installed-wheel kernel;
- no project-root import shadowing;
- no execution errors or timestamp/private-path/token output;
- frozen cell IDs and execution counts;
- only intentional compact tables and plots retained; and
- source execution plus copied-notebook installed-wheel execution in the
  focused proof.

Prefer a focused new `tests/test_random_demo.py` proof owner rather than
further expanding the already broad readout-profile/demo test module. The
new test must contain the real notebook contract and execution proof; it is
not a placeholder.

## Exact Changed-Path Ceiling

Implementation may change only the exact paths in this section. Rename pairs
count as both old and new endpoints. A required path outside this ceiling is a
Design contradiction and stops Implementation before that byte changes.

### Dependency, Environment, And Demos

```text
pyproject.toml
create_environment.sh
demos/readout.ipynb
demos/random.ipynb
```

`demos/random.ipynb` is new. The existing readout notebook may change only to
refresh exact executed stochastic outputs and synchronized dependency wording;
its source Config/profile/product/plot contract otherwise remains exact.

### Production

```text
tensor_dslab/readout/runtime/addresses.py
tensor_dslab/readout/runtime/keys.py
tensor_dslab/readout/noise_waveform/runtime/produce.py
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/charge/runtime/produce.py
tensor_dslab/readout/charge/runtime/effects/counts.py
tensor_dslab/readout/charge/runtime/effects/dark_counts.py
tensor_dslab/readout/charge/runtime/effects/delays.py
tensor_dslab/readout/charge/runtime/effects/smearing.py
tensor_dslab/readout/charge/runtime/effects/timing_jitter.py
tensor_dslab/readout/charge/runtime/effects/correlated_avalanches.py
```

`addresses.py` is the sole new production module. The exact production target
is therefore `61` `.py` modules. No package facade or `__init__.py` changes.

### Tests

```text
tests/test_charge_correlated_avalanches.py
tests/test_charge_count_orchestration.py
tests/test_charge_delay_preparation.py
tests/test_charge_product.py
tests/test_charge_timing_jitter.py
tests/test_noise_waveform_product.py
tests/test_package_contracts.py
tests/test_pint_physical_configuration.py
tests/test_random_demo.py
tests/test_readout_profiles_and_demos.py
tests/test_readout_simulation.py
tests/test_rng_ownership_migration.py
tests/test_runtime_action_ownership.py
tests/test_tensorcore_0_16_modernization.py
tests/test_tensorcore_0_19_adoption.py
tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py
tests/typing/maintenance_11_tensorcore_0_19_addressed_distributions.py
```

The exact planned topology changes are:

- delete `tests/test_charge_count_orchestration.py` after its
  TensorDSLab-owned generic category sampler is removed;
- add `tests/test_random_demo.py`;
- rename `tests/test_tensorcore_0_16_modernization.py` to
  `tests/test_tensorcore_0_19_adoption.py`; and
- rename the active Maintenance 2 typing fixture to the Maintenance 11 path
  while replacing `RngPositions` with the supported addressed surface.

Historical work-order documents are not renamed or rewritten. Tests retain
historical facts only where they are still operative package contracts.

### Current Package Documents

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/api.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/design.md
docs/decisions.md
docs/integration.md
docs/overview.md
docs/parity.md
docs/quickstart.md
docs/validation.md
docs/implementation/index.md
docs/implementation/maintenance_11_tensorcore_0_19_addressed_distributions.md
```

These are current-state synchronization paths, not permission to rewrite
unrelated architecture or historical evidence. Candidate lifecycle wording
must be branch/main-neutral and remain truthful before Validation, after a
clearance handoff, after an unchanged Review fast-forward, and until the
evidence-only closeout.

The complete changed-path ceiling is `48` rename-expanded endpoints:

- `4` dependency/environment/demo paths;
- `11` production paths;
- `17` test endpoints; and
- `16` current documentation paths.

Every other tracked or untracked repository path is protected. In particular,
`LICENSE`, `tensor_dslab/py.typed`, all package facades, public Config and
field modules, `docs/implementation/` historical work orders, the provisional
DS20k profile proposal, and all nonlisted tests remain byte-identical.

## Candidate Route

The ordinary finite route is:

```text
Design authority
    -> Implementation Candidate 1
    -> fixed-commit Validation
    -> independent read-only Review
    -> final same-byte Design approval
    -> Review-owned clean fast-forward
    -> Design evidence-only closeout
    -> Review closeout verification
```

Implementation-to-Validation submissions are limited to `3`. Validation
returns to Implementation are limited to `3`. A Design-owned documentation
finding returns to Design and does not authorize Implementation to alter
Design-owned bytes outside the committed ceiling. No fourth ordinary
candidate, exceptional correction, or widened scope is implicit.

Implementation works on:

```text
codex/maintenance-11-tensorcore-0-19-addressed-distributions
```

The candidate is one atomic dependency/scientific-address/demo rebaseline.
No partial migration commit may be merged or published.

## Implementation Sequence

The one atomic candidate should proceed internally in this order:

1. exact dependency pin, package probes, imports, and negative typing;
2. fixed key/quantum ledger and role-named address construction;
3. white noise and PSD Gaussian migration;
4. dark-count Poisson migration;
5. tensor-valued charge-smearing Gaussian migration;
6. timing-jitter ProbabilityKernel and Multinomial migration;
7. shared delay/recovery kernel representation;
8. afterpulse occurrence and delay factorization;
9. collapsed direct/delayed crosstalk Poisson migration;
10. retirement of old offsets, category orchestration, and overflow state;
11. the addressed-randomness notebook and its focused installed-wheel proof;
12. synchronized parity, architecture, validation, API, and lifecycle docs;
13. complete fixed-commit source/archive, typing, artifact, demo, and hygiene
    evidence.

This order is an implementation plan, not permission to commit partial
candidate states or widen the final allowlist.

## Required Evidence Before Review

The candidate handoff must report exact commands, environments, dependency
artifacts, file/export/test counts, candidate scope, and finite-loop
accounting. Artifact hashes generated from immutable candidate bytes are
handoff evidence; candidate-bearing documentation must not make a
self-referential hash claim.

At minimum, the exact candidate must prove:

### Dependency And API

- exact published TensorCore containing commit, tree, parent, version, and
  live GitHub ref;
- exact containing wheel `51208` bytes /
  `9666ff7811cc1bdfe289290f3ae18517d7f47c316ad1ff766477a7d08f075dda`;
- exact containing source archive `492382` bytes /
  `a76983491a0b6dc019be725695010f38707a5a90a0cfd151da4596eab77fef07`;
- source/archive package-byte identity;
- exact supported TensorCore export counts and package topology;
- retired `RngPositions`, central validation, and old CounterRng law methods
  remain absent;
- exact TensorDSLab pin in source, wheel, and sdist metadata;
- unchanged TensorDSLab version `0.1.0`, Python `>=3.14`, Torch
  `>=2.13,<2.14`, NumPy `2.5.1`, Pint `0.25.3`, Hatchling `1.31.0`, and
  Pyright `1.1.411`;
- exact TensorDSLab facades remain `35/5/30`; and
- import isolation and no downstream-package import.

### Address And Word Identity

- exact package-owned role keys, removed overflow keys, quantum values,
  address-domain shapes, and selection order;
- no encoded-position collision over every accepted shape/generation/kernel
  ceiling;
- no `.offset(...)`, manual mixed-radix arithmetic, or selected-element
  renumbering;
- repeated-run identity;
- full versus deterministic source/destination chunk identity;
- traversal-order identity where scientific dependencies permit reordering;
- product-request invariance;
- unchanged global Torch RNG state; and
- direct CounterRng raw-word fixtures for representative complete addresses.

### Scientific Laws

- white-noise mean, RMS, covariance, and repeatability;
- PSD coefficient/endpoints, RMS, integrated power, and autocorrelation;
- dark-count mean, variance, zero probability, and cell covariance;
- charge-smearing conditional mean/variance, zero-scale behavior, and
  nonnegative policy;
- timing-jitter represented/completion identities, destination mapping,
  conservation including discarded completion/boundary allocations,
  moments, covariance, and edge behavior;
- afterpulse occurrence probability, conditional delay law, recovery-weighted
  charge/S2, boundary discard, and fixed-generation recursion;
- direct/delayed crosstalk deterministic destination-rate fixtures;
- crosstalk Poisson splitting/superposition equivalence against a small
  independent reference;
- destination-rate acceptance up to the exact `1e8` boundary without a
  total-source-mean restriction;
- count and Charge ledger ceilings;
- exact-zero and deterministic word-free paths; and
- complete requested-`Charge` statistical evidence under newly frozen seeds,
  address fixtures, observables, tolerances, and sample sizes.

### Runtime Effects

- TensorCore-owned constructor synchronization boundaries are not duplicated
  by TensorDSLab generic scans;
- TensorDSLab's own scientific host observations are explicitly inventoried;
- kernel construction happens once per reusable prepared law;
- dynamic Distribution construction occurs only when frontier/count state
  requires it;
- Multinomial result allocation is preflighted and bounded by source/chunk
  policy;
- no public remainder, overflow, or tail result survives;
- all generated tensors remain same-device and no payload silently moves to
  host; and
- source, Config, Runtime, kernel, and address ownership/freshness promises are
  proved.

### Package And Documentation

- focused source/archive suites covering every changed effect, address owner,
  dependency contract, and both notebooks;
- complete source/archive discovery with no regression below the current
  `229` discovered / `216` passed / `13` unavailable-CUDA-skipped baseline
  and with every added test discovered;
- exact TensorCore source/archive suite with its two accepted
  unavailable-CUDA skips;
- strict Pyright with zero TensorDSLab diagnostics and the exact TensorCore
  `82`-diagnostic negative fixture in both dependency forms;
- fresh wheel and sdist with source-equal package bytes;
- isolated core-wheel and demos-wheel execution;
- CPU script and stored notebook replay after exact-output refresh if needed;
- exact source and installed-wheel execution of `demos/random.ipynb`;
- notebook proof of public `RngElements`/`RngAddress` metadata, actual private
  role-key use without a duplicated literal, repeated-draw equality,
  retained-capacity chunk equality, unchanged global Torch RNG state, and the
  one-step collapsed Poisson crosstalk path;
- static notebook proof that no private TensorCore storage/permutation method,
  source-total crosstalk Poisson, crosstalk Multinomial, environment mutation,
  or production-key customization is presented;
- `docs/parity.md`, architecture, decisions, validation, API, README,
  overview, CONTRIBUTING, AGENTS, and implementation index synchronized only
  where the final changed contract requires it;
- Markdown fences, links, anchors, stale terminology, privacy, raw-route,
  artifact, bytecode, build, and physical-cleanliness checks; and
- independent fixed-commit Validation and Review under the accepted finite
  route.

The ordinary local matrix uses exact CPython `3.14.6`, PyTorch `2.13.0`,
NumPy `2.5.1`, Pint `0.25.3`, Hatchling `1.31.0`, and Pyright `1.1.411`.
Implementation and independent roles may each create at most one temporary
role-named environment when needed to prove the root environment script; it
must be removed after the recorded check. Do not create repeated exploratory
Conda environments.

## CUDA And Release Qualification

No fresh CUDA or cluster work is authorized for this maintenance.

The current ecosystem schedule intentionally defers integrated accelerator
evidence until TensorCore and TensorDSLab freeze one exact mutually adopted
`1.0.0` release-candidate pairing. Every interim disposition must state that
it makes no current integrated CUDA, accelerator-support, performance,
deployment, calibration, or production-readiness claim.

CPU, strict typing, exact word/address fixtures, chunk/traversal invariance,
artifacts, documentation, and consumer evidence remain mandatory during this
pre-1.0 migration.

## Dispatch Readiness

Design has completed the dispatch gates:

1. TensorCore local main and tracking `origin/main` resolve exact published
   containing commit `ed17f4b` / tree `ef8c706`, with package bytes exact to
   Stage 26 anchor `fdfc96d`.
2. The publication handoff supplies the selected containing wheel/archive
   identities and exact package/export evidence.
3. The afterpulse occurrence/delay identity is frozen as one key with quantum
   `0`/`1`.
4. The exact `48`-endpoint changed-path ceiling and all-path protection rule
   are frozen above.
5. The required dependency, address, scientific, runtime, typing, artifact,
   demo, documentation, and hygiene evidence is frozen above.
6. Persistent TensorDSLab Implementation, Validation, and Review routes were
   independently located in their package-owned worktrees and are dormant at
   clean command boundaries.
7. The user explicitly accepted the predeployment RNG rebaseline and
   authorized proceeding toward Implementation.
8. Integrated CUDA remains prohibited and deferred to the mutually adopted
   `1.0.0` release-candidate pairing.

The exact committed bytes containing this section are the immutable Design
authority. Any TensorCore contract change, publication mismatch, required
path outside the ceiling, unresolved address collision, scientific-domain
narrowing, public-surface change, or package-source contradiction stops
Implementation and returns to Design.

## Non-Goals

This maintenance does not authorize or imply:

- a TensorCore repository edit or a dependency target other than exact
  published `0.19.0` containing commit `ed17f4b`;
- implementation or test changes outside the frozen allowlist;
- a compatibility shim, alias, or deprecation layer;
- a public TensorDSLab API change;
- a public TensorDSLab RNG tutorial API or supported import for private role
  keys;
- user-configurable RNG keys, addresses, streams, or domains;
- mutable RNG state or a cursor;
- per-avalanche expansion;
- a TensorDSLab Distribution or Kernel framework;
- total-first crosstalk Poisson plus Multinomial allocation;
- returned completion, tail, boundary, or overflow counts;
- new delay, recovery, crosstalk, afterpulse, noise, or smearing science;
- donor eventwise or RNG parity;
- CUDA, cluster, performance, compilation, fusion, or deployment work;
- TensorG4DS, TensorML, IO, cache, DAG, reconstruction, or trigger work;
- package-index publication, tag, release, or version-1.0 claim; or
- merge, push, release, compatibility, conformance, or governance effects.

## Authority

This document is Design-complete executable authority. Implementation is
authorized to create the exact feature branch, change only the frozen
allowlist, execute the required local evidence, freeze one coherent candidate,
and dispatch that immutable candidate to persistent Validation.

Validation may inspect only a fixed commit, return findings within the finite
route, or dispatch unchanged clear bytes to persistent independent Review.
Review remains read-only until a Validation-cleared fixed commit arrives.

No role may infer permission to change a protected path, run CUDA or cluster
work, merge, push, publish, claim compatibility, or broaden the scientific
boundary. Final same-byte Design approval remains required before Review may
fast-forward local main. A later Design-owned evidence-only closeout and
independent Review verification remain required before any separately
authorized push.
