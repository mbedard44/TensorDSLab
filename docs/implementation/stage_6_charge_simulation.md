# Stage 6 Charge Simulation Work Order

Status: **Merged / Design acceptance pending**. The user authorized execution
on 2026-07-15 from committed Design/dispatch authority
`21de93a239302a8c31edf3f7fec120ecb1eeea57`. The exact linear candidate chain
was:

```text
21de93a239302a8c31edf3f7fec120ecb1eeea57
  -> 00488a3d975104ec0967c988c278540f57b71598
  -> 40c10f0f598ea3ac95a11426fc927978553215fb
  -> fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58
```

Candidate 1 implemented the accepted Stage 6 slice. Candidate 2 was the
tests-only Validation correction that completed the predeclared statistical
evidence. Fixed-commit Validation cleared Candidate 2. Independent Review then
returned one P1 numerical-contract finding: the exact dark-count ceiling guard
accepted a represented configuration, but the later grouped binary64
multiplication could round its mean above `1e8` and reject it. The exact
reproducer was:

```text
sample_period_ps = 2677300530967072003
sample_count     = 2
rate_hz          = 37.35105523020191

exact ceiling relation:       accepted
correctly rounded exact mean:  100000000.0
old grouped mean:              100000000.00000001
```

Candidate 3 changed only `readout/charge/_produce.py` and
`tests/test_charge_product.py`. It forms the binary64 mean once from the exact
represented rational product after the exact guard. Its regression proves the
accepted endpoint reaches the sampler as exactly `100000000.0`, while the
immediately greater binary64 rate is exactly outside the ceiling and fails
before RNG use or producer writes with source and global RNG unchanged.
Independent Review rechecked and cleared the correction with no remaining
finding, then cleanly fast-forwarded unchanged `main` from the Design authority
to exact Candidate 3. No merge commit, rebase, squash, amend, or push occurred.

The cumulative candidate changes exactly 24 allowlisted paths: four
behavior-neutral producer renames, the private Charge producer and RNG/sampler
work, required config/export retirement, README synchronization, and focused
runtime/static tests. The cumulative diff is 7038 insertions and 183 deletions.
Candidate 3's correction is exactly two paths, 92 insertions, and one deletion.

TensorCore remained clean at exact `0.7.0` pin
`b454d738f6385ce6489d85492a618a3dab139bb6`. Review recreated its exact ZIP;
the SHA-256 was
`649c4daac3b953397371cb64647dcaf9a7ca7a857b32fae58c4ec4a856c79796`.
Before and after the merge, the TensorCore source and ZIP suites each ran 174
tests: 164 passed and 10 conditional CUDA tests skipped, with no failure or
error. A newly extracted exact-pin tree produced the same fixed-candidate
result. The focused count/delay/jitter/cascade/product run executed 65 tests:
60 passed and 5 CUDA tests skipped.

Pyright `1.1.408` reported 0 errors, warnings, or informational findings
against both the exact TensorCore source and newly extracted archive.
Import-isolation output was `False False False False`. Exact topology,
allowlist, diff, public/private surface, source and global-RNG immutability,
forbidden-call/import, and generated-artifact gates passed. The evidence
environment was Python `3.13.11`, PyTorch `2.12.1`, macOS `15.7.4` on arm64,
and eager CPU execution. CUDA and MPS were unavailable. The `build` and
`hatchling` modules were unavailable. The closeout therefore makes no CUDA,
GPU-performance, compile/fusion, editable-install, or wheel-build claim.

The statistical evidence used frozen seeds `0`, `1`,
`0x0123456789abcdef`, and `0xffffffffffffffff`. Every row below is
`observed / target / standard error / deterministic accumulation delta /
accepted bound`. A zero delta means that test's frozen gate is exactly eight
standard errors. All tests used population moments and predeclared laws,
sample sizes, and bounds.

Poisson and aggregate-binomial sampler evidence was:

```text
Poisson, M=2**18:
  lambda=1 mean      .99852371215820312 / 1 / .001953125 / 2.5579538487363607e-13 / .015625000000255795
  lambda=1 variance  .99209758253709879 / 1 / .0033829117335329633 / 2.5579538487363607e-13 / .027063293868519502
  lambda=9.5 mean    9.4965553283691406 / 9.5 / .0060199355497743906 / 2.4300561562995426e-12 / .048159484400625181
  lambda=9.5 var     9.4569507196781792 / 9.5 / .026921970218926214 / 2.4300561562995426e-12 / .21537576175383977
  lambda=10 mean     10.000404357910156 / 10 / .0061763235550163663 / 2.5579538487363607e-12 / .049410588442688884
  lambda=10 var      9.9999388013384305 / 10 / .028303470207401246 / 2.5579538487363607e-12 / .22642776166176792
  lambda=100 mean    100.00160217285156 / 100 / .01953125 / 2.5579538487363607e-11 / .15625000002557954
  lambda=100 var     100.18950396371972 / 100 / .27690325935073878 / 2.5579538487363607e-11 / 2.2152260748314898

Binomial n=32, p=.25, M=2**16:
  mean      7.99517822265625 / 8 / .0095683193077467886 / 1.8189894035458565e-12 / .076546554463793298
  variance  6.0171581469476223 / 6 / .03297254495338698 / 1.3642420526593924e-12 / .26378035962846008

Multinomial n=32, p=(.10,.15,.20,.55), M=2**16:
  mean c0  3.196976 / 3.2 / .006629 / 7.28e-13 / .053033
  mean c1  4.794937 / 4.8 / .007890 / 1.09e-12 / .063122
  mean c2  6.407791 / 6.4 / .008839 / 1.46e-12 / .070711
  mean c3  17.600296 / 17.6 / .010993 / 4.00e-12 / .087945
  var c0   2.884839 / 2.88 / .016533 / 6.55e-13 / .132264
  var c1   4.100477 / 4.08 / .022861 / 9.28e-13 / .182890
  var c2   5.166699 / 5.12 / .028339 / 1.16e-12 / .226716
  var c3   7.986431 / 7.92 / .043077 / 1.80e-12 / .344618
  cov 01  -.465146 / -.48 / .013360 / 1.09e-13 / .106880
  cov 02  -.627071 / -.64 / .015039 / 1.46e-13 / .120312
  cov 03  -1.792622 / -1.76 / .019862 / 4.00e-13 / .158898
  cov 12  -.990574 / -.96 / .018049 / 2.18e-13 / .144395
  cov 13  -2.644756 / -2.64 / .024405 / 6.00e-13 / .195237
  cov 23  -3.549053 / -3.52 / .028270 / 8.00e-13 / .226164
```

The independent Poisson-4 and dark-operation evidence used `M=2**18`:

```text
Direct Poisson(4):
  mean      3.996342 / 4 / .003906 / 1.02e-12 / .031250
  variance  3.981907 / 4 / .011719 / 1.02e-12 / .093750
  P(0)      .018547 / .018316 / .000262 / 4.69e-15 / .002095
  P(4)      .195423 / .195367 / .000774 / 5.00e-14 / .006195
  P(X>=8)   .050095 / .051134 / .000430 / 1.31e-14 / .003442

Independent superposition Poisson(1.5) + Poisson(2.5):
  lambda=1.5 mean  1.497169 / 1.5 / .002392 / 3.84e-13 / .019137
  lambda=1.5 var   1.491486 / 1.5 / .004784 / 3.84e-13 / .038273
  lambda=2.5 mean  2.498020 / 2.5 / .003088 / 6.39e-13 / .024705
  lambda=2.5 var   2.505806 / 2.5 / .007564 / 6.39e-13 / .060515
  sum mean         3.995190 / 4 / .003906 / 1.02e-12 / .031250
  sum variance     4.002094 / 4 / .011719 / 1.02e-12 / .093750
  component cov    .002401 / 0 / .003782 / 0 / .030258
  joint P(0)       .018581 / .018316 / .000262 / 4.69e-15 / .002095

Dark-count operation, lambda=4:
  mean      3.995975 / 4 / .003906 / 1.02e-12 / .031250
  variance  3.996624 / 4 / .011719 / 1.02e-12 / .093750
  P(0)      .018436 / .018316 / .000262 / 4.69e-15 / .002095
  P(4)      .194878 / .195367 / .000774 / 5.00e-14 / .006195
  P(X>=8)   .050182 / .051134 / .000430 / 1.31e-14 / .003442
```

The three-generation DiCT and DeCT checks each used 32 roots, offspring mean
`.2`, and `M=2**16`. DeCT used an exact one-bin delay and diagnosed generation
three as right overflow:

```text
DiCT:
  G1 mean  6.382767 / 6.4 / .009882 / 1.46e-12 / .079057
  G1 var   6.426342 / 6.4 / .036710 / 1.46e-12 / .293684
  G2 mean  1.278366 / 1.28 / .004841 / 2.91e-13 / .038730
  G2 var   1.529128 / 1.536 / .011123 / 3.49e-13 / .088983
  G3 mean  .258209 / .256 / .002201 / 5.82e-14 / .017607
  G3 var   .319319 / .31744 / .003894 / 7.22e-14 / .031153
  cov 12   1.289277 / 1.28 / .014087 / 2.91e-13 / .112694
  cov 13   .251226 / .256 / .006070 / 5.82e-14 / .048559
  cov 23   .308460 / .3072 / .004166 / 6.98e-14 / .033327

DeCT:
  G1 mean      6.405350 / 6.4 / .009882 / 1.46e-12 / .079057
  G1 var       6.422100 / 6.4 / .036710 / 1.46e-12 / .293684
  G2 mean      1.276184 / 1.28 / .004841 / 2.91e-13 / .038730
  G2 var       1.535004 / 1.536 / .011123 / 3.49e-13 / .088983
  G3 overflow  .252579 / .256 / .002201 / 5.82e-14 / .017607
  G3 ovf var   .313245 / .31744 / .003894 / 7.22e-14 / .031153
  cov 12       1.284482 / 1.28 / .014087 / 2.91e-13 / .112694
  cov 13       .265824 / .256 / .006070 / 5.82e-14 / .048559
  cov 23       .308062 / .3072 / .004166 / 6.98e-14 / .033327
```

The AP check used 32 parents, `p=.25`, source bin 2, inverse delay ratio `.2`,
inverse recovery ratio `.1`, and `M=2**16`:

```text
  offset0 mean    .750305 / .749230 / .003341 / 1.70e-13 / .026731
  offset0 var     .728730 / .731688 / .005096 / 1.66e-13 / .040769
  offset1 mean    1.320496 / 1.314342 / .004385 / 2.99e-13 / .035083
  offset1 var     1.273114 / 1.260357 / .007948 / 2.87e-13 / .063581
  retained mean   2.070801 / 2.063572 / .005427 / 4.69e-13 / .043419
  retained var    1.945418 / 1.930499 / .011512 / 4.39e-13 / .092097
  overflow mean   5.932343 / 5.936428 / .008589 / 1.35e-12 / .068715
  overflow var    4.844987 / 4.835141 / .026839 / 1.10e-12 / .214715
  stop mean       23.996857 / 24 / .009568 / 5.46e-12 / .076547
  stop var        5.971619 / 6 / .032973 / 1.36e-12 / .263780
  charge mean     .144285 / .143689 / .000411 / 3.27e-14 / .003288
  charge var      .0111920 / .0110737 / 6.87e-5 / 2.52e-15 / 5.49e-4
  S2 mean         .0117714 / .0117189 / 3.67e-5 / 2.66e-15 / 2.93e-4
  S2 var          8.905e-5 / 8.812e-5 / 5.54e-7 / 2.00e-17 / 4.44e-6
  ovf charge mean 2.521241 / 2.522978 / .003651 / 5.74e-13 / .029204
  ovf charge var  .875123 / .873345 / .004848 / 1.99e-13 / .038783
  cov off0,off1  -.028213 / -.030773 / .003698 / 7.00e-15 / .029583
  cov ret,charge  .135810 / .134423 / .000845 / 3.06e-14 / .006762
  cov ret,ovf    -.409393 / -.382820 / .011888 / 8.70e-14 / .095104
  cov charge,ovf -.012379 / -.011329 / .000382 / 2.58e-15 / .003057
```

The AP prepared laws agreed with independent Decimal-80 references within
`1e-12` locally and `1e-11` over the complete law. Every represented-weight
ledger check passed its `gamma_L` envelope. The independently positive
within-category recovery variance remained explicitly outside the selected
conditional-mean response model.

Timing jitter used one parent in source bin 1, four samples, `sigma/T=.5`,
`M=2**18`, and only the independent integrated latent-uniform plus Gaussian
equation:

```text
  mean b0  .191673 / .190984 / .000768 / 4.89e-14 / .006142
  mean b1  .609116 / .609548 / .000953 / 1.56e-13 / .007623
  mean b2  .190777 / .190984 / .000768 / 4.89e-14 / .006142
  mean b3  .004341 / .004238 / .000127 / 1.08e-15 / .001015
  var b0   .154935 / .154509 / .000474 / 3.95e-14 / .003796
  var b1   .238094 / .237999 / .000209 / 6.09e-14 / .001670
  var b2   .154381 / .154509 / .000474 / 3.95e-14 / .003796
  var b3   .004322 / .004220 / .000126 / 1.08e-15 / .001006
  cov 01  -.116751 / -.116414 / .000386 / 2.98e-14 / .003085
  cov 02  -.036567 / -.036475 / .000181 / 9.33e-15 / .001450
  cov 03  -.000832 / -.000809 / 2.43e-5 / 2.07e-16 / .000195
  cov 12  -.116205 / -.116414 / .000386 / 2.98e-14 / .003085
  cov 13  -.002644 / -.002583 / 7.71e-5 / 6.61e-16 / .000617
  cov 23  -.000828 / -.000809 / 2.43e-5 / 2.07e-16 / .000195
  drop     .004093 / .004245 / .000127 / 1.09e-15 / .001016
  disp mean .007786 / .008476 / .001233 / 2.17e-15 / .009868
  disp var  .399754 / .398849 / .001051 / 1.02e-13 / .008410
```

Exact fixed-word inversion/PTRS/BTRS, 80-to-110-digit decision/law fixtures,
all bypasses, count/address/allocation/ledger endpoints, all 16 Charge stage
combinations, all eight mechanism combinations, conservation, overflow,
freshness, axis-order, noncontiguous-source, and stream-isolation checks also
passed. Conditional CUDA Charge/sampler/statistical tests skipped because no
CUDA device was available; they produced no GPU observation.

This Review-owned closeout changes only this work order and the implementation
index. Cleared production, tests, README, package metadata, architecture,
parity, governance, and dependency bytes remain exactly those merged at
Candidate 3. Final TensorDSLab Design acceptance remains required before the
stage becomes **Merged / Closed**. This merge and closeout do not dispatch
Stage 7, activate Coordination or Profile B, establish conformance or broad
compatibility, or authorize a push.

## Objective

Implement the complete functionality-first private Charge producer selected in
the rebuild architecture:

- retire the unsupported MVP `NormalDelayConfig` surface without a shim;
- finish the behavior-neutral `_product.py`/`_product_*` to
  `_produce.py`/`_produce_*` waveform-producer rename;
- append all eight fixed Charge roles to the private RNG stream registry;
- implement the exact aggregate conditional-binomial, multinomial, and hybrid
  Poisson machinery actually consumed by Charge;
- implement private dark-count, timing-jitter, fixed-generation correlated-
  avalanche, S1/S2-ledger, overflow-diagnostic, and charge-smearing behavior;
- expose exactly one new product-owned private entry point,
  `_produce_charge(...)`; and
- establish eager CPU plus conditional eager CUDA correctness evidence against
  independent mathematical and statistical oracles.

The stage ends at a completed private `Charge` product. It does not add
`simulate_readout(...)`, a public atomic transform, another semantic field,
public diagnostics, a mutable collection lifecycle, or any integration or IO
surface. Functionality and scientific correctness are the closure criteria.
Compiler fusion, kernel count, allocation elimination, throughput, and GPU
optimization remain later measured work.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`.

The exact clean TensorDSLab starting baseline is `main` at:

```text
bd5e8042a7aab54cb8c5ac15c1e79918b62e840d
```

That commit is the accepted Stage 5 Design closeout. The exact Stage 5
implementation candidate remains:

```text
538089910be0fcaceff363c43e41e92e87af2efd
```

The target production branch after explicit dispatch is:

```text
codex/stage-6-charge-simulation
```

This committed path is the stable package-owned work-order key. The state
snapshot at Design completion is:

```text
package_adoption_state: Adopted
conformance_finding: Not evaluated
coordination_status: Deferred
registry_storage_profile: Disabled
stage_5: Merged / Closed
stage_6: Design-complete / Undispatched
```

The only permitted Stage 6 execution states are:

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

`Returned to Design` and `Blocked` terminate the current execution attempt and
authorize no scope expansion. Design alone dispatches the stage and accepts
final closeout. Implementation, Validation, and Review report the intermediate
dispositions defined here.

Before dispatch, Design must commit this work order and every synchronized live
Design source, verify a clean tree, and name that exact commit. Implementation
branches from that committed authority, not from `bd5e804` or an uncommitted
documentation tree.

The exact TensorCore dependency remains clean TensorCore `0.7.0` at:

```text
b454d738f6385ce6489d85492a618a3dab139bb6
```

No dependency move or TensorCore edit is in scope. TensorDSLab remains in
active development and pre-deployment. This work order makes no release,
deployment, backward-compatibility, conformance, broad compatibility, GPU
execution, or GPU-performance claim beyond exact recorded evidence.

## Source Of Truth

Implementation, Validation, and Review must read and reconcile:

- [Agent Workflow](../../AGENTS.md);
- [Contributing](../../CONTRIBUTING.md), especially boundary-first validation,
  product ownership, private-surface discipline, stochastic-field rules,
  numeric-envelope rules, complete-write/failure effects, storage freshness,
  and testing standards;
- [Rebuild Architecture](../architecture/rebuild.md), especially Configuration
  Composition, Private Product Builders, Charge Response, Timing Jitter Inside
  Charge Simulation, the sole active Fixed-Generation Correlated-Avalanche
  Baseline, Delay Kernels, Aggregate Multinomial Sampling, Poisson Count
  Sampling, RNG And Positional Repeatability, Stage 6 Count/Address/Numeric
  Envelope, Functional/Memory/Lifetime, and Validation Strategy;
- [Post-Binned Readout Architecture](../architecture/readout.md);
- [Correlated-Avalanche Physics Guide](../physics/correlated_avalanches.md);
- [Validation](../validation.md);
- [IV-DSLab Parity](../parity.md), including every Stage 6 model-conformance,
  statistical-parity, deferred-equivalence, and intentional-divergence label;
- the closed [Stage 5 RNG And Stochastic Noise](stage_5_readout_rng_and_stochastic_noise.md)
  work order for inherited Threefry, uniform, Box-Muller, private-surface, and
  exact-pin evidence;
- TensorCore `0.7.0` `docs/api.md`, `docs/architecture/tensors.md`,
  `docs/integration.md`, and the public package implementation at the exact pin;
- W. Hoermann, *The Transformed Rejection Method for Generating Poisson Random
  Variables*, Insurance: Mathematics and Economics 12 (1993), for PTRS;
- W. Hoermann, *The Generation of Binomial Random Variates*, Journal of
  Statistical Computation and Simulation 46 (1993), for BTRS; and
- the audited IV-DSLab and DSLab donor baselines, paths, and checksums named in
  `docs/parity.md`.

The architecture equations and raw-word mappings are normative. The papers are
algorithm evidence and the donor repositories are comparison evidence; neither
may override TensorDSLab's exact streams, categories, bounds, accumulation
order, or parity classifications.

If a live source instead requires recursive same-bin closure, generation-wave
execution, recovery-marked branching state, per-avalanche expansion,
`NormalDelayConfig`, a mutable RNG, public diagnostics, or end-to-end public
orchestration in Stage 6, stop before implementation and return the contradiction
to Design.

## Dispatch And Finite Role Loop

Before dispatch, Design must privately verify these persistent logical routes:

```text
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Each must be Active, current for this workspace, and able to return to Design.
Coordination remains Deferred and is not used. Raw route identifiers must not
appear in committed files.

The authorized loop after explicit dispatch is:

```text
Design dispatch
  -> Implementation candidate
  -> fixed-commit Validation
  -> independent fixed-commit Review
  -> Implementation corrections, if any
  -> Validation/Review recheck
  -> Review clean fast-forward
  -> post-merge verification
  -> Design closeout
```

The loop allows at most three Implementation-to-Validation dispatches and at
most three Validation-to-Implementation returns. Review is read-only and
reviews a fixed Validation-cleared commit. A repeated finding, exhausted loop,
route discrepancy, architecture contradiction, dirty baseline, or required
scope expansion returns to Design.

## Stage Slices And Checkpoints

Implementation owns one coherent branch and one final candidate, but should
develop in these reviewable slices:

1. **Surface cleanup.** Retire `NormalDelayConfig`; rename the four existing
   waveform producer modules and callables without behavior drift; update only
   their imports, tests, and typing fixtures.
2. **Private random mechanics.** Append the eight streams and implement the
   exact Poisson, conditional-binomial, and aggregate-multinomial consumers with
   independent scalar/high-precision and fixed-word tests.
3. **Delay and timing preparation.** Implement fixed/exponential prepared
   delay laws, AP recovery weights, analytic timing-jitter categories, and
   their local/complete-law numerical oracles.
4. **Charge submodels.** Implement dark counts, timing redistribution,
   correlated-avalanche generations and diagnostics, then smearing.
5. **Product closure.** Implement `_produce_charge(...)`, all structural
   configuration combinations, source/freshness/typing/package gates, and the
   complete statistical suite.

These are checkpoints, not separate public stages and not permission to merge a
partial Charge implementation. Every slice remains private. Validation clears
only a fixed candidate containing the complete objective.

## Selected Stage Decisions

Stage 6 freezes these implementation decisions:

1. `docs/architecture/rebuild.md` is the sole active correlated-avalanche
   algorithm. Earlier recursive, same-bin-closure, generation-wave, or
   recovery-marked alternatives are not implementation options.
2. `Photoelectrons` is the already-produced dense `torch.int64` truth input.
   It is borrowed read-only and is never changed into a jittered truth product.
3. Charge executes the effective private stages in physical order: truth,
   optional dark counts, optional timing jitter, optional correlated
   avalanches, optional smearing, then the terminal `Charge` construction.
   Every disabled block is skipped as a block; the preceding working tensor
   simply remains current.
4. Correlated avalanches use a caller-configured finite `K`, one unmarked
   integer frontier per generation, distinct DiCT/DeCT/AP laws, and explicit
   S1/S2 ledgers. AP recovery weights deposited charge only and never offspring
   probability or recursive state.
5. CT retained and overflow counts use separate Poisson draws and streams for
   DiCT and DeCT. Their rates are never superimposed. AP uses aggregate
   multinomial factorization and at most one direct AP child per parent.
6. Causal fixed and exponential delay models plus latent independent
   uniform-within-bin phase marginalization are the complete MVP delay surface.
   `NormalDelayConfig` is removed without alias, warning, or compatibility shim.
7. Discrete probabilities, rate fields, sampler control, and prepared PMFs are
   binary64 independently of the requested Charge dtype. Integer avalanche
   history must therefore be exactly equal for float32 and float64 Charge on
   the same backend/mode with otherwise equal inputs.
8. The exact per-cell count ceiling is `2**53 - 1`; the independent Poisson
   mean ceiling is `1e8`. Every accepted addition is checked before it occurs.
   There is no whole-grid population ceiling and no arbitrary fixed limit on
   `K` beyond relational address, allocation, and accumulation proofs.
9. The eager reference uses the frozen logical accumulation order: increasing
   generation; direct CT, delayed CT, AP; increasing CT source bin within each
   destination; increasing AP source bin and retained offset before overflow
   and stop. Unspecified repeated-index atomic/scatter reduction is forbidden.
10. The stage may allocate functional outputs, private scratch, positional
    address blocks, prepared device values, and backend intermediates. It makes
    no allocation-free or target-temporary claim and materializes no complete
    address lattice where checked chunking is sufficient.
11. Every successful generated `Charge` is guaranteed-fresh and storage-
    independent of the source. Producers enqueue all writes before semantic
    construction and never write through an alias afterward.
12. Detailed private helper decomposition is Implementation-owned. The stage
    freezes product/submodel names, scientific equations, streams, addresses,
    numerical mappings, results, diagnostics, and evidence—not every local
    helper signature.

## Public Config Cleanup

Delete `NormalDelayConfig` from:

- `tensor_dslab/readout/charge/types.py`;
- `tensor_dslab/readout/charge/__init__.py`;
- `tensor_dslab/readout/__init__.py`;
- `tensor_dslab/__init__.py`;
- runtime package-contract/config tests; and
- static-typing fixtures if referenced.

The exact crosstalk delay annotations and validators become:

```python
FixedDelayConfig | ExponentialDelayConfig
```

There is no deprecated name, alias, compatibility import, warning path, or
string-based fallback. Closed historical work-order records are immutable and
retain the names that accurately describe their old bytes.

## Producer Rename

Rename these exact production modules and callables:

```text
pure_waveform/_product.py       -> pure_waveform/_produce.py
_product_pure_waveform          -> _produce_pure_waveform

noise_waveform/_product.py      -> noise_waveform/_produce.py
_product_noise_waveform         -> _produce_noise_waveform

analog_waveform/_product.py     -> analog_waveform/_produce.py
_product_analog_waveform        -> _produce_analog_waveform

digitized_waveform/_product.py  -> digitized_waveform/_produce.py
_product_digitized_waveform     -> _produce_digitized_waveform
```

The rename changes no equation, argument, return type, source/freshness rule,
RNG address, private visibility, or accepted test result. Do not retain shim
modules or alias callables. Closed Stage 4 and Stage 5 documents remain
historically exact.

## Exact New Private Product Surface

Create exactly one new product producer:

```python
def _produce_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    seed: int | None,
    floating_dtype: torch.dtype,
) -> Charge:
    ...
```

It lives only in:

```text
tensor_dslab/readout/charge/_produce.py
```

It is private and must not appear in any `__init__.py`, `__all__`, package-root
surface, method, convenience wrapper, or public documentation example. New
private scientific submodels use the accepted `_simulate_*` names. The product
module may contain private plan/result records and numerical helpers with real
behavior; do not create placeholder modules or decorative one-line wrappers.

The accepted submodel roles are:

```python
_simulate_dark_counts(...)
_simulate_timing_jitter(...)
_simulate_correlated_avalanches(...)
_simulate_charge_smearing(...)
```

Their exact collaborator-facing behavioral signatures are the sketches in
`rebuild.md`. The correlated result is a private immutable record containing
at least S1, S2, final frontier, total count, mechanism-resolved retained and
overflow counts, retained AP charge and charge-square-sum, and AP overflow
charge when the corresponding mechanism is present. Structurally absent
mechanisms remain `None`, not allocated zero diagnostics. No diagnostic becomes
a TensorField or collection member.

## Private RNG Registry

Extend the one existing private `Enum` in `readout/_random.py` to exactly:

```python
@unique
class _RngStream(Enum):
    NOISE_WHITE = 0x0000_0001
    NOISE_PSD_COEFFICIENT = 0x0000_0002
    CHARGE_DARK_COUNTS = 0x0000_0003
    CHARGE_DIRECT_CROSSTALK = 0x0000_0004
    CHARGE_DIRECT_CROSSTALK_OVERFLOW = 0x0000_0005
    CHARGE_DELAYED_CROSSTALK = 0x0000_0006
    CHARGE_DELAYED_CROSSTALK_OVERFLOW = 0x0000_0007
    CHARGE_TIMING_JITTER = 0x0000_0008
    CHARGE_AFTERPULSES = 0x0000_0009
    CHARGE_SMEARING = 0x0000_000A
```

Use `Enum`, not `IntEnum`; stream zero remains unassigned. Existing values may
not move. Disabled and exact-zero branches request no word from their role.
The module remains private and may not import TensorCore, TensorDSLab product
types/configs/axes, NumPy, SciPy, donor packages, or private Torch RNG APIs.

## Aggregate Distribution Mechanics

Implement only the distribution consumers required by the active Charge path.
The exact formulas, crossover rules, reflection, strict comparison, attempt and
term limits, raw-word mappings, and uncertainty-band ownership are the
normative contracts in `rebuild.md`.

### Poisson

One private `_sample_poisson(...)`-equivalent implementation accepts an exact
output shape plus only a scalar or exactly output-shaped `torch.float64` mean.
It returns a fresh nonnegative `torch.int64` tensor with exactly the explicit
output shape:

```text
lambda == 0             -> exact zero, no draw
0 < lambda < 10         -> one-uniform forward-CDF inversion
10 <= lambda <= 1e8     -> Hoermann PTRS
```

Inversion has its frozen 64-term limit. PTRS uses one raw block per attempt and
attempts `0..63`. Exhaustion is a hard failure; there is no reseed, fallback,
clipping, normal approximation, or `torch.poisson`. A mixed tensor preserves
each cell's original positional address across zero/inversion/PTRS masks.

### Conditional Binomial

Multinomial factorization uses one exact conditional-binomial primitive. It
reflects `p > 0.5`, handles exact boundaries without draws, and selects:

```text
n * p_star < 10   -> one-uniform forward-CDF inversion
n * p_star >= 10  -> stabilized Hoermann BTRS
```

The inversion recurrence has its frozen 64-term limit. BTRS attempt `j` uses
both binary64 uniforms formed from raw ordinals `4*j` through `4*j + 3` in its
one addressed block, through attempt 63. It uses the cancellation-resistant
`d = k - m`, `log1p` grouping and the exact central and complete-support
high-precision decision gates specified in `rebuild.md`.
Outside summed uncertainty the represented decision must agree with the
high-precision reference; inside it the fixed-word represented decision owns
the result.

### Aggregate Multinomial

Consume categories once in their declared order. Each scientific law prepares
stable binary64 `A[c]` as category `c`'s mass and `B[c]` as the combined mass
of every later category, including the final remainder. For remaining count
`n_r`, category `c` uses:

```text
if A[c] == 0:
    x[c] = 0
else if B[c] == 0:
    x[c] = n_r
else:
    total = A[c] + B[c]
    complement = B[c] < A[c]
    p_star = min(A[c], B[c]) / total
    y ~ Binomial(n_r, p_star)
    x[c] = n_r - y if complement else y
n_r <- n_r - x[c]
```

The final drop/stop category is the exact no-draw remainder. Counts are exactly
conserved, and no individual avalanche is expanded. Preflight prepares `B`
from stable analytic cumulative tails or equivalent remaining masses; runtime
never obtains it by repeatedly subtracting rounded categories from one, clips,
or renormalizes a malformed law. Category order and positional address are
scientific contracts, not optimization choices.

No standalone Bernoulli, continuous exponential variate, categorical alias
table, dependency distribution, `torch.multinomial`, or global RNG surface is
added merely because it might be useful later.

## Timing Jitter

For each source cell, use the analytically prepared latent-uniform plus
ideal-Gaussian displacement law from `rebuild.md`. Evaluate every destination
offset that can remain in the finite window; do not impose a tail cutoff.
Preflight enforces:

```text
2**-52 <= sigma / sample_period <= 64
2 <= sample_count <= 8192
sample_count * tensor_numel <= 2**63
```

The exact `z = 8` evaluator boundary, asymptotic-tail mapping, represented
symmetry, category-mass construction, local `1e-12` checks, and complete-law
L1 `1e-11` check are mandatory. The final combined drop category is the
no-draw remainder. Runtime uses increasing destination/address order and
aggregate multinomial redistribution. It neither expands PEs nor calls
Box-Muller. Exact `sigma == 0` skips the whole stage and stream.

## Dark Counts

Prepare the scalar per-cell mean in host binary64:

```text
lambda = rate_hz * sample_period_ps * 1e-12
```

Require it in `[0, 1e8]`. Positive values use the shared Poisson sampler at
row-major full-grid position, `source_quantum = 0`, and
`CHARGE_DARK_COUNTS`. Add the result to the working counts with checked
per-cell arithmetic. Exact zero rate skips the entire block and consumes no
draw. Dark counts occur before timing jitter, so present dark roots are
redistributed by jitter.

## Fixed-Generation Correlated Avalanches

Generation zero is the post-dark, post-jitter integer grid. For configured
`maximum_generations = K`, evaluate exactly generations `1..K`. `K=0` is
roots-only and requests no mechanism word. A provably extinct frontier may
avoid physical work only if the fixed positional/RNG semantics remain
unchanged.

Effective-mechanism preflight is contextual. `K=0` prepares no CT/AP delay or
recovery kernel and imposes no mechanism address, generation-storage, or
ledger-depth gate. A zero DiCT/DeCT mean likewise prepares no kernel and
imposes no such gate for that mode. Zero AP probability prepares neither its
delay nor optional recovery response and imposes no AP mechanism gate. These
paths still receive already-constructed structurally valid configs, but an
otherwise out-of-domain sampling/config pair for an unused mechanism must not
fail active-kernel numerical checks or require a seed.

For each generation, execute mechanisms in this exact order:

1. direct crosstalk;
2. delayed crosstalk;
3. afterpulses.

DiCT and DeCT each prepare their own fixed/exponential causal delay PMF and
analytic right tail. Retained destination rates use source-bin order and
separate Poisson streams; source-indexed overflow rates use their distinct
overflow streams. Every actual cell rate must be in `[0, 1e8]`. The two CT
rates are never superimposed. Retained and overflow addresses use the exact
generation-major `K * N` schema in `rebuild.md`.

AP uses its exponential delay law, exact analytic tail, at-most-one child law,
and aggregate multinomial categories in increasing retained offset order,
followed by overflow and the no-draw stop remainder. Addresses are
generation-major, fixed offset-category-major, then source-position-major over
the full `K * (S + 1) * N` lattice. Invalid source/offset cells remain reserved
rather than compacted.

When `afterpulse.recovery is None`, every retained AP has unit deposited
charge. When recovery is present, prepare the exact conditional
`rho_bar_ap[d]` and overflow response from the exponential phase-marginalized
law. Apply the category response before source/offset contributions collapse.
The retained S1 contribution is `count * rho_bar`; the S2 contribution is
`count * rho_bar**2`. Recovery changes neither the integer child count nor
future offspring laws.

Every retained child enters the single next unmarked frontier exactly once.
Overflow is diagnosed and removed. It never enters the frontier, ledgers, or
waveform. The exact count and real-arithmetic ledger identities from
`rebuild.md` are mandatory; floating validation uses the frozen accumulation
order and accepted dtype-aware error envelope.

## Count, Address, Ledger, And Storage Envelope

Use local/private integer values for the fixed limits; do not create public
package constants or config knobs.

Every source, working, frontier, mechanism, diagnostic, cumulative, binomial,
and Poisson count cell must remain in:

```text
0 <= count <= C_max
C_max = 2**53 - 1
```

Before every nonnegative addition prove `rhs <= C_max - lhs`. Reject one cell
above the bound without wrap. Do not sum the complete grid merely to impose an
aggregate ceiling.

Before allocation or RNG use prove each effective role's relevant product:

```text
timing jitter:  S * N <= 2**63
CT roles:       K * N <= 2**63
AP:             K * (S + 1) * N <= 2**63
```

Also check tensor shape/byte products against host/Torch representability. Do
not materialize a complete address lattice solely to prove a bound.

For dtype precision `p_d`, the ledger-depth proof uses:

```text
no recovered AP:       L = E*K + 1
recovered retained AP: L = E*K + S + 3
require L < 2**p_d
gamma_L = L / (2**p_d - L)
error <= gamma_L * T + L * eta_d
```

where `E` is the number of effective retained mechanisms contributing to the
ledger, `T` is the reference absolute-term sum, and `eta_d` is the accepted
subnormal allowance. Preserve `S2 <= S1 <= total_count` within this proved
envelope. A different reassociation or parallel reduction requires new Design
evidence.

## Charge Smearing

If smearing is absent or its relative sigma is exact zero, skip the entire
stage and stream. Otherwise every full-grid row-major position—including
zero-S2 cells—owns one scalar Box-Muller `z0` from `CHARGE_SMEARING` with
`source_quantum = 0`; discard `z1`.

Evaluate in the selected Charge dtype with ambient autocast disabled:

```text
scale = relative_sigma * sqrt(S2)
draw = S1 + scale * z0
charge = max(draw, 0)
```

Preflight applies the finite Box-Muller radius and target-dtype maximum guard
from `rebuild.md`, including the positive-sigma-rounds-to-zero and infinity
cases. Every accepted scale, excursion, pre-clipped draw, and result must be
finite. No value-dependent position compaction is allowed.

## Product Orchestration And Failure Effects

`_produce_charge(...)` follows the exact conditional-block shape in
`rebuild.md`. The lowercase local remains a tensor; only its final value is
wrapped as `Charge`. Smearing without correlation uses the unit-response
identity `S1 == S2`. Correlation always constructs S1 and S2 even if smearing
is absent. Every one of the 16 optional stage-presence combinations must be
valid when its active submodels satisfy contextual preflight.

Complete producer preflight includes exact source/config/dtype/device/sample
relationships, source count-domain validation, all effective kernel and sampler
bounds, requested storage shapes, and root-seed need. It completes before the
first random-word request or producer write. `seed=None` is accepted only when
the effective path is draw-free. A preflight failure returns no field, changes
neither source nor PyTorch global RNG, and begins no producer write.

Backend failure after work is launched has no rollback guarantee, but no
partially constructed `Charge` is returned. Unsupported CPU/CUDA alternatives
are rejected rather than moved through host memory. No code calls `.cpu()`,
`.numpy()`, `.tolist()`, source `.detach()`, a source device movement, or any
global Torch RNG API. The required fresh terminal conversion from integer
working counts into the requested floating Charge dtype is explicitly allowed;
it must not mutate, replace, or move the source field.

The completed `Charge`:

- is exactly `Charge`;
- uses the source's exact immutable axis tuple and axis instances;
- has the source shape and device;
- uses exactly requested `torch.float32` or `torch.float64`;
- is finite and nonnegative;
- has guaranteed-fresh storage independent of `photoelectrons.tensor`; and
- is nondifferentiable with respect to integer truth and stochastic count
  history.

The producer may use managed output/scratch tensors and backend intermediates.
It exposes none of them and writes through no alias after constructing the
semantic result.

## Exact Candidate Change Allowlist

Production changes are limited to:

```text
M  tensor_dslab/__init__.py
M  tensor_dslab/readout/__init__.py
M  tensor_dslab/readout/_random.py
M  tensor_dslab/readout/charge/__init__.py
M  tensor_dslab/readout/charge/types.py
A  tensor_dslab/readout/charge/_produce.py
R  tensor_dslab/readout/pure_waveform/_product.py
   tensor_dslab/readout/pure_waveform/_produce.py
R  tensor_dslab/readout/noise_waveform/_product.py
   tensor_dslab/readout/noise_waveform/_produce.py
R  tensor_dslab/readout/analog_waveform/_product.py
   tensor_dslab/readout/analog_waveform/_produce.py
R  tensor_dslab/readout/digitized_waveform/_product.py
   tensor_dslab/readout/digitized_waveform/_produce.py
```

Tests and evidence changes are limited to:

```text
M  tests/test_package_contracts.py
M  tests/test_readout_configs.py
M  tests/test_readout_random.py
M  tests/test_deterministic_waveform_products.py
M  tests/test_noise_waveform_product.py
M  tests/typing/stage_4_deterministic_waveform_products.py
M  tests/typing/stage_5_readout_rng_and_stochastic_noise.py
A  tests/test_readout_count_sampling.py
A  tests/test_charge_delay_preparation.py
A  tests/test_charge_timing_jitter.py
A  tests/test_charge_correlated_avalanches.py
A  tests/test_charge_product.py
A  tests/typing/stage_6_charge_simulation.py
```

Implementation may update `README.md` only to state the newly implemented
private Charge seam and preserve the public exclusion of
`simulate_readout(...)`. Review-owned closeout may update only this work order
and `docs/implementation/index.md`. No other architecture, parity, governance,
dependency, build, or public-document file changes are authorized. If a
necessary correction falls outside this allowlist, return it to Design.

Do not create `readout/simulation.py`, `charge/_product.py`, an extra sampler
module, placeholder package, public diagnostic type, workspace, allocator,
output-buffer abstraction, or sibling-repository change.

## Required Deterministic And Numerical Evidence

Tests must cover at least:

- exact `NormalDelayConfig` absence from every export and module namespace;
- exact two-member CT delay unions and all accepted/rejected constructor cases;
- exact module/callable rename, absence of retired paths/names, and unchanged
  Stage 4/5 deterministic/noise results;
- exact ten-member `_RngStream` sequence and collision-free values;
- independent fixed-word Poisson inversion/PTRS and binomial
  inversion/BTRS oracles at every crossover, acceptance path, boundary, and
  injected exhaustion path;
- at-least-80-decimal-digit local fixtures for the BTRS and PTRS log decisions,
  including the cancellation regression and uncertainty-band ownership;
- aggregate multinomial exact conservation, category order, zero/one/no-count
  identities, covariance, and no per-avalanche expansion;
- fixed and exponential delay PMFs/right tails, source-relative boundary loss,
  AP recovery categories/tails, and local `1e-12` plus complete-law `1e-11`
  agreement with independent high-precision fixtures;
- timing-jitter prepared symmetry, every retained destination, final drop
  remainder, edge loss, complete count conservation, increasing address order,
  and analytic mean/variance/covariance/displacement checks;
- source count ceiling, every checked-add boundary, Poisson-mean ceiling,
  exact address-product boundaries, ledger-depth boundaries, and no whole-grid
  count ceiling;
- all 16 optional Charge-stage combinations, all eight correlated-mechanism
  combinations, `K=0`, `K=1`, draw-free zero-effect modes, and exact mechanism
  ordering;
- contextual identity fixtures proving `K=0`, zero CT mean, and zero AP
  probability skip their delay/recovery preparation, seed, address,
  generation-storage, and ledger-depth gates even when the structurally valid
  unused config/sampling pair lies outside the corresponding active numerical
  domain;
- separate DiCT/DeCT retained and overflow roles, AP count/charge/S2
  bookkeeping, exact integer identities, dtype-aware ledger identities, and
  absorbing-boundary behavior;
- smearing word schedule, finite-envelope boundary, zero-S2 behavior, clipping,
  and float32/float64 result validity;
- every Charge path under at least two semantic axis orders, including a
  sample-not-last layout and a noncontiguous `torch.strided` Photoelectrons
  source, proving exact `SampleAxis`-class dimension lookup rather than
  hard-coded `dim=-1` and exact reuse of every source axis instance;
- source immutability, exact axes/device/shape, freshness, output storage
  independence, no post-exposure writes, and global-RNG immutability; and
- exact same-backend integer-history equality across float32/float64 Charge
  requests.

Independent references must not call production helpers to compute expected
values. Tests may use stdlib `decimal`, `fractions`, and frozen fixtures; they
must not add NumPy, SciPy, donor, or internet runtime dependencies.

## Statistical Validation Policy

Use the frozen seeds:

```text
0
1
0x0123456789abcdef
0xffffffffffffffff
```

Use exactly the sample allocations selected in `rebuild.md`:

```text
scalar and one-parent laws: 2**18 total, 2**16 per seed
aggregate Q=32 laws:        2**16 total, 2**14 per seed
small-grid K<=3 fixtures:   2**16 total, 2**14 per seed
completed Charge fixtures: 2**16 total, 2**14 per seed
```

For every predeclared statistic, accept only when:

```text
abs(observed - target) <= 8 * standard_error + delta

delta = (
    64
    * eps(dtype)
    * max(1, ceil(log2(accumulation_length)))
    * abs(reference_scale)
)
```

Frequency/tail cells require at least 256 expected hits and 256 expected
misses. Use exact identities for conservation, bypass, dtype, addressing, and
storage claims; statistical allowance must not weaken them. Record every
sample size, seed, formula, observed value, target, standard error, delta, and
final bound in the Validation report.

Stage 6 establishes conformance to the selected TensorDSLab probability model.
It does **not** establish complete IV-DSLab donor equivalence. IV timing-jitter,
recursive-cascade, and detector-level Charge margins remain deferred until
collaborators supply the calibration/observable-specific Delta required by
`docs/parity.md`. Their absence is not an implementation failure and must not
be replaced with an invented tolerance.

## Conditional CUDA Evidence

When CUDA is available, repeat focused raw-word, sampler, delay, Charge,
repeatability, source/device, and statistical checks in eager CUDA mode.
Raw Threefry words and integer/fixed-point decisions follow their exact
cross-backend contract; completed floating transcendental/ledger/Charge values
use the documented numerical/statistical comparison. Do not claim CPU/CUDA
bitwise equality for completed floating results.

If CUDA is unavailable, tests skip explicitly and the stage may close with an
eager CPU-only evidence qualification. No GPU execution, performance, fusion,
kernel-count, or memory claim follows from conditional skips.

## Static Typing And Package Evidence

Create `tests/typing/stage_6_charge_simulation.py` with positive
`typing.assert_type` evidence that `_produce_charge(...)` returns `Charge` for
representative deterministic and stochastic configs. Update Stage 4/5 typing
fixtures only for the required `_produce_*` rename. The typing corpus contains
no `Any`, cast, ignored diagnostic, private TensorCore import, or public
re-export of a private producer/RNG/sampler.

Package-contract tests must prove:

- `NormalDelayConfig` and every `_product_*` symbol/path are absent;
- public exports otherwise remain exact;
- `_produce_charge(...)`, `_produce_*` waveform seams, `_RngStream`, and all
  samplers/submodels remain private;
- product types do not import composition/orchestration;
- producer import direction remains acyclic; and
- importing `tensor_dslab` does not import TensorG4DS, TensorML, DSLab, G4DS,
  donor packages, NumPy, or SciPy.

Analyze against both the exact TensorCore source checkout and an independently
extracted archive of the exact pin.

## Verification Commands

Implementation and fixed-commit Validation must run at least:

```bash
git status --short --branch
git diff --check
git diff --check <design-dispatch-commit>..<candidate-commit>
git diff --name-status <design-dispatch-commit>..<candidate-commit>
git -C /Users/mbedard/Projects/TensorCore rev-parse HEAD
git -C /Users/mbedard/Projects/TensorCore status --short
git -C /Users/mbedard/Projects/TensorCore archive --format=zip --output=/tmp/tensorcore-stage6-b454d738.zip b454d738f6385ce6489d85492a618a3dab139bb6
shasum -a 256 /tmp/tensorcore-stage6-b454d738.zip
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/tmp/tensorcore-stage6-b454d738.zip python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest tests.test_readout_count_sampling tests.test_charge_delay_preparation tests.test_charge_timing_jitter tests.test_charge_correlated_avalanches tests.test_charge_product -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -c "import sys, tensor_dslab; print('tensor_g4ds' in sys.modules, 'tensor_ml' in sys.modules, 'dslab' in sys.modules, 'g4ds11' in sys.modules)"
env PATH=/Users/mbedard/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH pnpm dlx pyright@1.1.408 --version
```

Recreate the archive from the exact dependency commit for every fixed candidate
and record its SHA-256. Validation must extract it outside the repository and
run the static checker with that extracted package as the only TensorCore
analysis path. Use two temporary configs outside the repository: one whose sole
TensorCore `extraPaths` entry is the exact source checkout, and one whose sole
TensorCore entry is the extracted archive. Do not edit committed
`pyrightconfig.json` to switch evidence forms.

Every actual static-check invocation uses this verified launcher prefix:

```text
env PATH=/Users/mbedard/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH \
  pnpm dlx pyright@1.1.408 \
  --pythonpath /opt/miniconda3/bin/python
```

Append `--project` and the applicable temporary config. If the executor's
Python or bundled Node path differs, return to Design before substituting a
different launcher.

Also report exact Python, PyTorch, TensorCore, and checker versions; execution
mode and device; CUDA availability/skips; focused/full totals; all statistical
observations; export/private-surface evidence; no donor/global-RNG/host-
materialization path; exact allowlist/topology; and generated-artifact absence.

If the accepted statistical matrix cannot complete within the supported local
environment for a reason other than conditional CUDA absence, return the exact
runtime evidence to Design. Do not silently reduce sample sizes or loosen
acceptance bounds.

Do not run `torch.compile`, Triton, a profiler, allocator instrumentation,
kernel-count, fusion, or peak-memory tests as substitute closure evidence.

## Validation Report

Validation evaluates only a fixed committed candidate and returns:

- exact Design authority, candidate, parent, branch, and allowlist topology;
- exact TensorCore source/archive identity and dual-form results;
- deterministic/high-precision/fixed-word and statistical results;
- exact bypass, stream, address, count, ledger, dtype, device, freshness,
  source/global-RNG, import, and static-typing evidence;
- every CUDA/build/tooling qualification;
- findings ordered by severity; and
- one disposition: `Cleared`, `Returned / Implementation correction`,
  `Returned to Design`, or `Blocked`.

Validation does not change architecture or edit the feature branch. It sends
bounded test/code findings to Implementation. A finding that requires a new
scientific law, config, product, stream, dependency, public API, or scope change
returns to Design.

## Independent Review

Review begins only from a fixed Validation-cleared commit. It independently
checks:

- all source-of-truth contracts and exact candidate allowlist;
- the distribution algorithms and fixed-word/address mappings;
- numerical stability and decision-gate ownership;
- delay/recovery/timing/cascade/smearing scientific meaning;
- source immutability, result freshness, failure effects, private exposure,
  typing, imports, and package architecture;
- evidence independence and statistical power;
- absence of per-avalanche expansion, hidden fallback, clipping,
  renormalization, global RNG, host materialization, and unsupported public
  surface; and
- exact diff/topology/artifact cleanliness.

Review reports findings first, ordered by severity. It is read-only. It may
return fixes through Implementation and recheck the fixed candidate within the
finite loop. If no finding remains, Review issues `Review-cleared / Merge
authorized`, performs the clean `git merge --ff-only` into an otherwise
unchanged `main`, repeats the required post-merge gates, and records an
evidence-only closeout in this work order and the implementation index.

## Known Risks And Deferred Questions

- Eager aggregate samplers and causal category scans may launch many kernels
  and allocate substantial scratch. Performance and memory are intentionally
  unclaimed until real accelerator profiling.
- The fixed finite generation count is scientific truncation. Calibration of
  `K` and detector parameters remains collaborator work.
- The dense unmarked model omits finite-microcell occupancy, collision,
  recovery history, sibling phase covariance, and within-category AP recovery
  variation. Those are documented model boundaries, not implementation bugs.
- CPU-only evidence cannot establish CUDA execution or performance when no CUDA
  device is available.
- Complete IV detector-level statistical parity remains unestablished without
  predeclared collaborator-owned observables and calibration margins.
- Workspace reuse, allocation-free execution, compiled/fused samplers, stream
  leases, and output-buffer lifetime require a later measured optimization
  stage.

## Non-Goals And Forbidden Scope

Stage 6 does not implement or authorize:

- public `simulate_readout(...)` or `readout/simulation.py`;
- public Charge transforms, public RNG/sampler helpers, or public diagnostics;
- a new semantic field, collection mutation, retained private prerequisite, or
  persistence policy;
- TensorG4DS handoff, PE binning, detector-window construction, reconstruction,
  TensorML, DAG, cache, artifact, or IO work;
- `out=`, workspace, allocator, pool, exchange bank, lease, stream coordinator,
  generation-retirement engine, scheduler, or lifecycle service;
- TensorCore or sibling-repository changes;
- `NormalDelayConfig`, signed/noncausal delay, recursive same-bin closure,
  generation-wave execution, marked recovery state, finite-cell collision, or
  per-avalanche tables;
- `torch.poisson`, `torch.multinomial`, dependency/global RNG, reseeding,
  clipping, normal approximation, or exhaustion fallback;
- GPU fusion, performance, allocation-free, release, deployment,
  compatibility, conformance, or backward-compatibility claims; or
- push, PR, tag, package publication, or external state change.

## Return To Design Before

Stop the affected slice and return exact evidence before:

- changing a scientific equation, category order, delay law, recovery law,
  generation meaning, stream, address, bound, decision gate, accumulation
  order, parity classification, or statistical acceptance rule;
- adding a dependency, public name, semantic type, config, module outside the
  allowlist, public orchestration, output/workspace policy, or sibling edit;
- retaining `NormalDelayConfig` or a `_product_*` compatibility surface;
- weakening preflight, source/freshness/failure contracts, high-precision
  evidence, sample sizes, or tolerances to make a test pass;
- accepting an unsupported device or silent host materialization;
- encountering a TensorCore contradiction or exact-pin drift;
- exceeding the finite I/V loop, repeating a finding, or observing route,
  branch, baseline, topology, or dirty-state discrepancy; or
- requiring a merge other than Review's clean fast-forward.

## Merge And Closeout

After Review clearance, Review alone fast-forwards unchanged clean `main` to the
exact cleared candidate and repeats the required gates. No merge commit,
rebase, squash, cherry-pick, amend, force operation, or push is authorized.

Review may then make one evidence-only closeout commit changing only this work
order and `docs/implementation/index.md`. That record names the exact Design
authority, candidate chain, final main commit, TensorCore archive checksum,
test/checker totals, statistical observations, environment, residual
qualifications, and no-effects statement. It must not change production,
tests, README, metadata, architecture, parity, governance, or dependency bytes.

TensorDSLab Design independently reconciles the merged bytes and evidence,
repeats proportionate post-merge gates, and either records `Merged / Closed` or
returns a substantive discrepancy through the accepted role loop. Stage 6
closeout does not dispatch Stage 7, activate Coordination/Profile B, establish
conformance or compatibility, or authorize a push.
