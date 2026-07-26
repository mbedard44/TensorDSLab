# Maintenance 11 TensorCore 0.19 Addressed Distributions

Status: **Provisional Design / Non-dispatchable**

Stable key:
`TensorDSLab/maintenance-11-tensorcore-0-19-addressed-distributions`

## Purpose

Prepare one package-owned TensorDSLab migration from the published
TensorCore `0.16.0` random surface to TensorCore `0.19.0`'s class-first
addressed Distribution and TensorKernel surfaces.

This record preserves the accepted migration decisions while TensorCore
completes its separate `0.19.0` GitHub publication lifecycle. It is not yet a
production work order. It authorizes no dependency change, implementation,
test edit, compatibility claim, CUDA work, merge, or push.

Before dispatch, TensorDSLab Design must replace every provisional dependency
reference below with the exact published TensorCore containing commit, verify
the live GitHub ref, freeze the final changed-file and test allowlists, obtain
renewed exact consumer evidence, and record explicit user authorization.

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
completed. This provisional record does not silently supersede the current
source, architecture, parity, or validation contract.

## Provisional TensorCore Target

The audited local TensorCore Stage 26 package implementation anchor is:

```text
TensorCore Stage 26 package anchor:
    fdfc96d428d62847dddc1a52eb956dd598874ae1
TensorCore package tree:
    ff1fc63d6453f3a3e6aa684df7d6d2ad96b30e9c
package version:
    0.19.0
```

TensorCore's documentation-only publication lifecycle replacement
`d22788f0ee5e98d5d9babf14e22cb50d4b6553d9`, tree
`72b2248b47ee5372dc597b5aa6ad95e5862e4734`, has received exact-byte
TensorDSLab consumer confirmation. It is consultation evidence, not a
TensorDSLab dependency selection. The future exact GitHub containing commit
remains unresolved in this record and must be supplied by TensorCore after its
package-owned publication closeout.

The audited Stage 26 package surface provides:

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

The exact audited export census is:

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

Before dispatch, the exact published containing commit must reproduce these
package bytes and supported surfaces. No alias for `RngPositions`, old
CounterRng Distribution methods, or retired import path is accepted.

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
retired by this maintenance remain permanently reserved and must be recorded
as such rather than reused for a new law.

The selected role treatment is:

| Existing role | Future treatment |
| --- | --- |
| white noise | retained |
| PSD noise | retained |
| dark counts | retained |
| direct crosstalk retained | retained for collapsed destination Poisson |
| direct crosstalk overflow | retired and permanently reserved |
| delayed crosstalk retained | retained for collapsed destination Poisson |
| delayed crosstalk overflow | retired and permanently reserved |
| timing jitter | retained for kernel allocation |
| afterpulse | retained for occurrence Binomial |
| charge smearing | retained |

Afterpulse delay allocation requires identity distinct from afterpulse
occurrence. The final dispatch amendment must freeze either one new append-only
role stream or an equally collision-safe explicit quantum/domain separation.
No Implementation thread may choose that distinction implicitly.

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

Exact constructors and Runtime storage locations remain to be frozen in the
dispatch amendment after a focused source/test inventory. Reusable roots may
be prepared once; per-source and bounded-chunk selection remains downstream
execution policy.

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

Reusable immutable scalar laws and kernels may be stored by private product
Runtime records when preparation owns all their inputs. The Runtime record
itself remains a final frozen slotted dataclass with no base, execution method,
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
exact role-key/quantum choice is a mandatory pre-dispatch decision.

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

The two historical overflow stream identifiers remain permanently reserved.
They must not be reassigned to another role.

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

- every role's complete address-domain shape and selection order;
- afterpulse occurrence/allocation identity separation;
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

## Provisional Implementation Sequence

After the exact publication and dispatch gates close, one atomic candidate
should proceed internally in this order:

1. exact dependency pin, package probes, imports, and negative typing;
2. fixed key/reserved-stream ledger and role-named address construction;
3. white noise and PSD Gaussian migration;
4. dark-count Poisson migration;
5. tensor-valued charge-smearing Gaussian migration;
6. timing-jitter ProbabilityKernel and Multinomial migration;
7. shared delay/recovery kernel representation;
8. afterpulse occurrence and delay factorization;
9. collapsed direct/delayed crosstalk Poisson migration;
10. retirement of old offsets, category orchestration, and overflow state;
11. synchronized parity, architecture, validation, API, and lifecycle docs;
12. complete fixed-commit source/archive, typing, artifact, demo, and hygiene
    evidence.

This order is an implementation plan, not permission to commit partial
candidate states or widen the final allowlist.

## Required Evidence Before Review

The final dispatch amendment must freeze exact commands, environments,
dependency artifacts, file counts, export counts, test counts, candidate
allowlist, and finite candidate-loop accounting.

At minimum, the exact candidate must prove:

### Dependency And API

- exact published TensorCore containing commit, tree, parent, version, and
  live GitHub ref;
- source/archive package-byte identity;
- exact supported TensorCore export counts and package topology;
- retired `RngPositions`, central validation, and old CounterRng law methods
  remain absent;
- exact TensorDSLab pin in source, wheel, and sdist metadata;
- exact TensorDSLab facades remain `35/5/30`; and
- import isolation and no downstream-package import.

### Address And Word Identity

- exact package-owned role keys, reserved stream identifiers, quantum values,
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

- focused and complete source/archive suites;
- strict Pyright and exact negative fixture;
- fresh wheel and sdist with source-equal package bytes;
- isolated core-wheel and demos-wheel execution;
- CPU script and stored notebook replay after exact-output refresh if needed;
- `docs/parity.md`, architecture, decisions, validation, API, README,
  overview, CONTRIBUTING, AGENTS, and implementation index synchronized only
  where the final changed contract requires it;
- Markdown fences, links, anchors, stale terminology, privacy, raw-route,
  artifact, bytecode, build, and physical-cleanliness checks; and
- independent fixed-commit Validation and Review under the accepted finite
  route.

## CUDA And Release Qualification

No fresh CUDA or cluster work is authorized for this provisional stage.

The current ecosystem schedule intentionally defers integrated accelerator
evidence until TensorCore and TensorDSLab freeze one exact mutually adopted
`1.0.0` release-candidate pairing. Every interim disposition must state that
it makes no current integrated CUDA, accelerator-support, performance,
deployment, calibration, or production-readiness claim.

CPU, strict typing, exact word/address fixtures, chunk/traversal invariance,
artifacts, documentation, and consumer evidence remain mandatory during this
pre-1.0 migration.

## Pre-Dispatch Gates

TensorDSLab Design must complete all of the following before dispatch:

1. receive TensorCore's exact `0.19.0` GitHub publication handoff;
2. verify the live containing commit and package-anchor byte identity;
3. replace the provisional dependency evidence in this record;
4. freeze afterpulse occurrence/allocation identity separation;
5. perform the final exact source/test/document inventory;
6. freeze the complete changed-path and protected-path allowlists;
7. freeze exact tests, counts, artifacts, hashes, and environment inputs;
8. synchronize any package-authoritative architecture/parity decision bytes
   needed before production starts;
9. verify persistent TensorDSLab Implementation, Validation, and Review roles;
10. commit one immutable Design/work-order authority; and
11. obtain explicit user authorization to dispatch.

Any TensorCore contract change, publication mismatch, unresolved address
collision, scientific-domain narrowing, public-surface change, or
package-source contradiction returns to Design and requires renewed review.

## Non-Goals

This provisional maintenance does not authorize or imply:

- a TensorCore dependency edit;
- implementation or test changes;
- a compatibility shim, alias, or deprecation layer;
- a public TensorDSLab API change;
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

This document records accepted provisional TensorDSLab Design direction only.
Implementation remains undispatched.

TensorCore's local Stage 26 anchor and publication-lifecycle candidate provide
read-only consumer evidence but do not select a TensorDSLab dependency.
Only a later exact TensorDSLab Design amendment satisfying every pre-dispatch
gate can convert this record into executable authority.
