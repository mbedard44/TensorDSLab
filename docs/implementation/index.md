# Implementation Stages

This directory holds TensorDSLab stage records and production work orders.
Documentation-only Design stages may stay in the Design thread. Production
work orders become handoff-ready instructions for the persistent
Implementation, Validation, and Review threads described in
[AGENTS](../../AGENTS.md).

Keep stages small enough that Implementation can finish, Validation can
critique, and Review can reason about the diff.

## Stages

- [Stage 0: Documentation Spine](stage_0_documentation_spine.md)
- [Stage 1: Post-Binned Readout MVP Architecture](stage_1_post_binned_readout_mvp_architecture.md)
  is a Design-complete documentation stage and itself dispatched no production
  work.

## Stage 2 Production Work Order

### [Stage 2: Package And Readout Collection Foundation](stage_2_package_and_readout_collection_foundation.md)

Status: Merged / Closed on `main` at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`, from the 2026-07-11 dispatch and
exact clean package baseline `d097cb3cdde185c6814116e886e7844ea3f55178`.
Fixed-commit Validation evaluated all three candidates in the bounded loop,
returned the first, and cleared the second and final candidates. Independent
Review returned narrow findings on the second, cleared the final bytes, and
performed the clean fast-forward plus post-merge verification. The persistent
logical roles remain available. Maintenance 1 and Stage 3 are separate records
below; Stage 2 itself dispatched no scientific transform.

The work order locks the flat package foundation under `tensor_dslab`, with
shared IDs in `tensor_dslab.common` and readout code in
`tensor_dslab.readout`; exact exported example/channel/sample axis IDs; exact
`ExampleId` and shared `ChannelId` coordinate backing; the count-only sample
axis; retained `SampleGrid`; the single primary
`ReadoutCollection(TensorCollection)`, conditional `DigitizedWaveformSpec`,
semantic reconstruction, exact `torch.int64`/common `torch.float32` or
`torch.float64`/`torch.int32` role dtypes, 1–16-bit truncated ADC
interpretation, destination preparation, public exports, and focused
construction tests. Required axes may occur in any layout order and are found
by axis-ID equality/index. `ChannelId` is common because readout and future
reconstruction reuse the same coordinate identity. Stage 2 implements no
physics transform, RNG, workspace, or full-chain builder. General collection
construction remains noncontiguous-capable, while every newly allocated public
target is contiguous in the existing semantic order without normalizing
retained fields.

## Maintenance 1 Production Work Order

### [Maintenance 1: Readout Surface Ownership](maintenance_1_readout_surface_ownership.md)

Status: Merged / Closed on `main` at
`3af8ab4acf834b07e3d027fb530e5f12934999a5`. It was dispatched from clean
`main` `cf0ccf0ad8fdee53767a374837276991decb1703` through committed Design
authority `d09cbad4a1538349e289523a9898f4e6dfd20a57`, cleared fixed-commit
Validation and independent Review, and was fast-forwarded without a push. The
work moves the semantic
`ReadoutCollection` record into `readout/types.py` and shortens the public
readout-axis Python symbols to `EXAMPLE_AXIS_ID`, `CHANNEL_AXIS_ID`,
`SAMPLE_AXIS_ID`, and `REQUIRED_AXIS_IDS`. It changes no ID string, collection
behavior, field API, TensorCore dependency, or scientific contract.

The work deliberately retains `readout/tensors.py` as the current home of four
readout-semantic reconstruction helpers. Whether those helpers should become
collection behavior or motivate an opt-in TensorCore reconstruction hook is a
separate Design question, not an Implementation decision in this work order.

## Stage 3 Production Work Order

### [Stage 3: TensorCore 0.7 Product Foundation](stage_3_tensorcore_0_7_product_foundation.md)

Status: Merged / Closed. Exact implementation candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde`, from committed Design/dispatch
base `fb4fd3753d336fd46203e122789caed32fb49d91`, passed fixed-commit Validation
and independent Review with no unresolved finding. Review's documentation-only
closeout and clean fast-forward produced `main`
`97e17c3177ac217aeb42a077db78f4bd223d51fa`; Design accepted that closeout on
2026-07-14 after independently repeating the post-merge package, dependency,
static-typing, import-isolation, and artifact checks. No push occurred.

Stage 3 is the clean structural migration from the historical TensorCore
`0.6` foundation to exact TensorCore `0.7.0` dependency
`b454d738f6385ce6489d85492a618a3dab139bb6`. It establishes common typed axes
and sampling, six direct final product-field leaves, product-owned config
records and deep validators, the unordered completed-result
`ReadoutCollection`, `ReadoutConfig`, deliberate exports, and focused runtime
and static-typing evidence.

The stage deletes old IDs, layouts, constants, sidecars, reconstruction,
selection, movement, invalidation, and output-buffer surfaces without aliases.
It creates no `simulation.py`, `_random.py`, product `_product.py`, scientific
algorithm, RNG, workspace, IO, source bridge, TensorML adapter, or future
placeholder. The exact production base remains
`3af8ab4acf834b07e3d027fb530e5f12934999a5`; Design committed the synchronized
rebuild authority at `fb4fd3753d336fd46203e122789caed32fb49d91`, verified the
persistent routes, and explicitly dispatched the work order before code began.

Review verified an exact 35-path candidate delta. Both the clean selected
TensorCore source and an independent exact-pin archive ran 51 tests: 49 passed
and 2 conditional CUDA tests were skipped. Pyright `1.1.408` reported no
findings against either dependency form. CUDA, `build`, and `hatchling` were
unavailable, so the closeout makes no GPU, editable-install, or wheel-build
claim. The exact archive SHA-256, environment, import isolation, and remaining
qualifications are recorded in the work order.

## Stage 4 Production Work Order

### [Stage 4: Deterministic Waveform Products](stage_4_deterministic_waveform_products.md)

Status: **Merged / Closed**. Exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da`, from committed Design/dispatch
base `b7af45741035821dfa94c8093bdeccea3320e26d`, passed fixed-commit Validation
and independent Review with no unresolved issue. Review's documentation-only
closeout and clean fast-forward produced `main`
`b3ebfcd9473537dd385195afea374bd2f426c6c0`; Design accepted that closeout on
2026-07-14 after independently repeating the post-merge package, dependency,
static-typing, import-isolation, and artifact gates. No push occurred.

The implementation adds exactly three private producers:
`_product_pure_waveform(...)` for both accepted TPC/Veto pulse models,
`_product_analog_waveform(...)`, and `_product_digitized_waveform(...)`.
It establishes the frozen equations, binary64 config preparation,
dtype/device/axes behavior, source immutability, guaranteed-fresh results,
pure/analog autograd, and nondifferentiable `torch.int32` digitization without
adding a public API.

Review verified the exact eight-path candidate delta. Against both the clean
TensorCore `0.7.0` source pin and its independent exact archive, 75 tests ran:
72 passed and 3 conditional CUDA tests skipped. Pyright `1.1.408` found no
issue against either dependency form, import isolation returned
`False False False False`, and diff/artifact checks passed. The exact archive
SHA-256, environment, scientific checks, and residual qualifications are
recorded in the work order. CUDA was unavailable, so the evidence makes no GPU
execution or performance claim; build tooling was unavailable, so it makes no
editable-install or wheel-build claim.

The whole noise producer, including exact zero noise, remains Stage 5 scope.
Stage 4 adds no public export, `simulate_readout(...)`, RNG, charge producer,
workspace, IO, or integration surface. It makes no kernel-count, fusion,
target-temporary, allocation-free, throughput, or GPU-performance claim.

## Stage 5 Production Work Order

### [Stage 5: Readout RNG And Stochastic Noise](stage_5_readout_rng_and_stochastic_noise.md)

Status: **Merged / Closed**. Exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd` is a linear descendant of committed
Design/dispatch authority `69b0472d246e107668a7ed253fa7c10bba22de8f`;
Candidate 1 has that authority as its exact parent, and the later candidates
are test-only corrections in the finite review loop. Fixed-commit Validation
and independent Review found no unresolved issue. Review fast-forwarded clean
`main` from `9ee84bf44a3a84e7e2d57d21362e79cc850f8e26` to the candidate and recorded
its evidence-only closeout at
`c6a506d3658b24197806b9e230480211a254a35a`. Design accepted the closeout on
2026-07-15 after independently repeating the source/archive package suites,
dual static-typing, import-isolation, dependency-identity, diff, and artifact
gates. No push occurred.

The focused slice adds only the private standard
`tensordslab.threefry4x32-20/v1` engine, exact fixed-point uniform conversion,
Box-Muller behavior actually consumed by noise, and the complete exact-zero,
IID-white, and PSD-shaped `_product_noise_waveform(...)` family. The private
central enum begins with exact streams `NOISE_WHITE = 0x0000_0001` and
`NOISE_PSD_COEFFICIENT = 0x0000_0002`; it assigns no Charge stream.

The acceptance mode is functionality-first vectorized eager CPU with
conditional eager CUDA. Raw words and fixed-point uniforms are exact across
accepted implementations; completed normal/PSD products are exactly
repeatable on one unchanged numerical execution stack and compare
statistically across backends.
The stage adds no public API, Charge-only distribution primitive,
`simulate_readout(...)`, compile/fusion/performance claim, workspace, IO, or
integration surface.

Review verified the exact seven-path candidate delta. Against both the clean
TensorCore `0.7.0` source pin and its independent exact archive, 109 tests ran:
104 passed and 5 conditional CUDA tests skipped. The focused RNG/noise run
executed 33 tests: 31 passed and 2 CUDA tests skipped. Pyright `1.1.408` found
no issue against either dependency form, import isolation returned
`False False False False`, and diff/artifact checks passed. The exact archive
SHA-256, frozen statistical observations, environment, and residual
qualifications are recorded in the work order. CUDA was unavailable, so this
is CPU-only evidence and makes no GPU execution or performance claim; build
tooling was unavailable, so it makes no editable-install or wheel-build claim.

## Stage 6 Production Work Order

### [Stage 6: Charge Simulation](stage_6_charge_simulation.md)

Status: **Merged / Closed**. Exact implementation and
Review-cleared candidate `fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58`
is a linear descendant of committed Design/dispatch authority
`21de93a239302a8c31edf3f7fec120ecb1eeea57`. Fixed-commit Validation cleared
the completed statistical-evidence candidate. Independent Review returned one
dark-count exact-ceiling finding, cleared the exact two-file correction, and
cleanly fast-forwarded unchanged `main` to the final candidate. No push
occurred.

The complete private slice implements `_produce_charge(...)`, aggregate
multinomial and hybrid Poisson sampling, dark counts, analytic timing jitter,
the fixed-generation DiCT/DeCT/AP cascade, S1/S2 ledgers, mechanism and
right-overflow diagnostics, charge smearing, all eight append-only Charge
streams, the per-cell `2**53 - 1` count ceiling, relational address/accumulator
bounds, and the frozen TensorDSLab-model statistical policy. It first retires
`NormalDelayConfig` without a shim and behavior-neutrally renames the four
transitional waveform producer modules/callables to `_produce.py` and
`_produce_*`.

The stage is eager and functionality-first. It permits ordinary private
scratch and backend intermediates and makes no allocation-free, compiler,
fusion, kernel-count, throughput, GPU-performance, public-orchestration, IO,
or integration claim. `Photoelectrons` remains immutable truth; all avalanche
state and diagnostics remain private; `simulate_readout(...)` remains Stage 7.

Against the clean exact TensorCore `0.7.0` source, ZIP, and newly extracted
pin, each full run executed 174 tests: 164 passed and 10 conditional CUDA tests
skipped. The focused Stage 6 run executed 65 tests: 60 passed and 5 CUDA tests
skipped. Pyright `1.1.408` reported zero findings against both dependency
forms, import isolation returned `False False False False`, and topology,
allowlist, diff, forbidden-surface, and artifact gates passed. The exact
archive SHA-256, statistical ledger, endpoint correction, environment, and
residual qualifications are recorded in the work order. Review's evidence-only
closeout produced `main` at
`ea979862b05f4ef543f6971c86641df317232479`; TensorDSLab Design accepted the
exact merged bytes and synchronized live documentation on 2026-07-15. Stage 7
remains undispatched.

## Dependency Evidence And Candidate Future Stages

These records include historical dependency evidence and nonoperative future
planning material. A linked draft is not a dispatch or cross-package
authority; only an explicitly recorded exact selection fixes a dependency.
Before Implementation starts, the owning package Design must close every
stated prerequisite, commit an exact work-order baseline, verify its package
routes, and obtain separate user authorization. When donor behavior is in
scope, the work order must name the comparison boundary, parity classification,
acceptance criteria, and intentional divergences defined in
[Parity](../parity.md).

### [Historical TensorCore Counter RNG And Distributions Consumer Proposal](proposed_tensorcore_counter_rng_and_distributions.md)

Status: **Fulfilled by TensorCore Stage 15 / Historical consumer proposal /
Never TensorCore authority**.

TensorDSLab records the demonstrated generic requirements for `RngKey`,
`CounterRng`, exact `Threefry4x32_R<20>`, `logical_positions`, public uniform,
Gaussian, Poisson, and binomial methods, plus the independently testable
`require_same_dtype()` relationship sub-slice. TensorCore independently
implemented and published that package-owned surface as version `0.9.0` at
exact commit `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. Its Stage 15 work
order and random architecture are authoritative. This proposal created none of
that authority and remains historical consumer evidence.

### [Maintenance 2: RNG And Product-Module Ownership Migration](maintenance_2_rng_and_product_module_ownership_migration.md)

Status: **Merged / Closed**. Exact final candidate and identical pre-closeout
merged `main`:
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63`. Dependency state:
**TensorCore `0.9.0` exact pin installed**.

Dependency: TensorDSLab Design selected published TensorCore `0.9.0` exact
commit `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, direct parent
`0e72f0e69cf9140b692d408e49a504cbdcb101b7`, after package-root and consumer
continuity probes. Independent source clones and archives reproduced exact
SHA-256 `f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd`.
The accepted implementation uses only the public package-root surface.

Implemented scope: split `types.py` ownership into product `config.py`/`field.py`
and readout `config.py`/`collection.py`; create private
`charge/effects` modules; retain Charge multinomial/category orchestration,
checked count helpers, and bookkeeping in `_counts.py`; add the ten
config-owned default keys; migrate stochastic functions from bare seeds and
private samplers to public TensorCore `CounterRng`
`gaussian(...)`, `poisson(...)`, and `binomial(...)`; preserve default-key
Stage 5/6 outputs; and remove `_RngStream`, `readout/_random.py`, and any
replacement `_rng.py` without compatibility shims. It also consolidates one
private scalar-to-dtype requirement helper and uses TensorCore
`require_same_dtype()` only for semantic field relationships. The work order
freezes the exact dependency, commands, allowlist, fixtures, lifecycle, and
stop conditions. Stage 7 remains undispatched and is not part of this
migration.

The exact candidate chain is Design authority
`daa046405f62ee324bc495867e796213bf6657a6` through implementation
`f6e1fc8c3d08152cf7ba603404a4d642628adfae`, two bounded test corrections
`5f6a8d56f0fefcd5606a8406da3a250c0f841b82` and
`f4e8eec9befaa107ceeb30c05ba1657eb7210bca`, and the Design-authorized
tests-only Review correction at the final candidate above. Fixed-commit
Validation and independent Review cleared the final bytes. Review
fast-forwarded clean `main` without a merge commit or push and repeated the
post-merge gates. The cumulative diff is 64 rename-aware files, 5,052
insertions, and 5,908 deletions, all within the frozen allowlist.

Final source and archive suites each ran 157 tests: 148 passed and 9
conditional CUDA tests skipped. The focused suite ran 148 tests: 139 passed
and 9 skipped; the supplemental proof module passed 9 of 9. Pyright `1.1.411`
reported zero diagnostics against both dependency forms, import isolation was
`False False False False`, and dependency, retired-surface, forbidden-call,
scope, protected-byte, diff, and artifact gates passed. The evidence
environment was Python `3.13.11`, PyTorch `2.12.1`, macOS `15.7.4` on arm64,
and eager CPU. CUDA was unavailable, so this closeout makes no GPU execution
or performance claim.
No production or test byte changed after Review clearance. Stage 7 remains
undispatched, Coordination remains Deferred, Profile B remains Disabled, and
conformance remains Not evaluated. This closeout establishes no broad
compatibility and authorizes no push.

### [Stage 7: Public Readout Orchestration](stage_7_public_readout_orchestration.md)

Status: **Merged / Closed** through exact Review-cleared implementation
candidate and merged `main`
`6dd55024685013fb9412a7247d3ddde7be1a3177`.

The Maintenance 2 prerequisite is satisfied at exact implementation candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63`, Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`, and TensorCore `0.9.0` pin
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`.

The work order freezes complete request/config/key preparation before product
execution, private product-owned prepared plans, exact typed prerequisite
planning, every producer at most once, requested-only retention, the complete
public `simulate_readout(..., rng=...)` surface, and focused composition/
storage/import/typing evidence. It adds no IO, persistence, workspace,
output-reuse, integration, or performance policy.

The user authorized production from exact Design authority
`254a624b39993c4dc0b9a2a832ebd07398ac5a24`. Fixed-commit Validation returned
Candidate 1 `0a028d715a75c2af5a38e6143815944403922737` for four bounded
test-proof corrections without a production or scope finding. Validation
cleared direct-child Candidate 2
`7ab98bea4c3db2b3a3d3710e18cb6dc1c96c0c06`, but Review returned it for two
additional bounded committed-test proof gaps without finding a
production, API, science, dependency, architecture, scope, import, typing, or
documentation-contract defect. Design authorized final direct-child Candidate
3 `e152aa05c0f960742e255ea40eb7e4591e628965` to prove zero replacement
`Photoelectrons` constructions and the exact empty ZeroNoise active-key set.
Validation returned Candidate 3 because the direct
ZeroNoise assertion observed the product module's preparer rather than the
binding used by public orchestration. With the ordinary I-to-V budget
exhausted, Design authorized exactly one exceptional direct-child supplemental
correction for that remaining proof.

Implementation independently reproduced the exact TensorCore ZIP hash and ran
the supplemental candidate focused source/archive suites at 30 tests with 27
passes and 3 conditional CUDA skips, both full suites at 188 tests with 176
passes and 12 conditional CUDA skips, and Pyright `1.1.411` with zero
diagnostics against both dependency forms. Review's exact mutation command now
reports `replacement_truth_mutant False 4` and
`phantom_zero_key_mutant False 1`. CUDA was unavailable, so the evidence is
eager CPU only. Validation cleared exact supplemental candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177`; independent Review cleared those
same bytes, fast-forwarded `main` without a merge commit, and repeated the
source/archive, typing, mutation, dependency, import/export, scope, privacy,
and hygiene gates post-merge. Final Design independently accepted the merged
topology, public surface, evidence, and qualifications and changed no
production or test byte. This closeout authorizes no push.

## Maintenance 3 Production Work Order

### [Maintenance 3: Environment-Qualified Stochastic Continuity](maintenance_3_environment_qualified_stochastic_continuity.md)

Status: **Merged / Closed** from exact committed Design authority
`37cd6403b66107ccd24acd7bf1e50c63f0599313` through exact Validation- and
Review-cleared candidate `dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0`,
Review's clean fast-forward of those unchanged bytes, and the exact
direct-child five-document Design closeout.

The first Stage 8 executable-input commit
`728840bf2858c861104d5f7bb3cdbb4e3e1361b5` correctly stopped before any
accepted Stage 8 focused/candidate result, benchmark, profiler, or measurement
because its protected suite applied Maintenance 2's exact macOS
completed-stochastic literals on a different Della Linux/x86_64 CPU stack. The
run reported 188 tests, 186 passes, two last-bit float32 failures, and no skips;
the existing conditional CUDA tests did execute, but the failing overall suite
produced no accepted Stage 8 result. The exact Stage 8 authority and executable
input remain immutable stopped evidence and are not candidates for merge or
reuse as accepted evidence.

Maintenance 3 changes only
`tests/test_rng_ownership_migration.py` plus synchronized documentation. It
preserves every historical literal and all fixed-point-uniform assertions
within their accepted exact scope. The exact recorded macOS stack continues to enforce the
literal completed Gaussian, Poisson, binomial, noise, and Charge payloads.
Another accepted stack executes the same requests independently twice and
requires exact replay within that unchanged numerical stack plus the existing
invariants and statistical contracts. It adds no ULP tolerance, Della-specific
golden, skip, expected failure, production change, TensorCore change, or
scientific change.

Fixed-commit Validation and independent Review cleared the same candidate.
Their recorded macOS source/archive suites passed with the expected twelve
conditional CUDA skips, and their separate full-A100 Della allocations passed
the complete source and archive suites at `188/188/0`. Review then
fast-forwarded clean `main` to the unchanged candidate. Final Design accepted
the evidence and synchronized the five authorized lifecycle documents without
changing production, tests, dependencies, metadata, governance, architecture,
or scientific contracts.

Stage 8 still requires a new Design authority and a complete evidence rerun
from scratch. At Maintenance 3 closeout, Maintenance 4 was the next required
baseline. Maintenance 4 and Maintenance 5 subsequently closed; the accepted
sequence now places Maintenance 6 next. Any later Stage 8 restart must begin
from the closed Maintenance 6 baseline. Maintenance 3 did not itself dispatch
or complete Stage 8.

## Maintenance 4 Production Work Order

### [Maintenance 4: Runtime Action Ownership](maintenance_4_runtime_action_ownership.md)

Status: **Merged / Closed** through exact Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda`, tree
`2d35a0e926b912f3fa846da97726e4e2490c4cc3`. Review fast-forwarded clean
`main` from `5fdd3fafe2c44357b09df2a04b88cb121f2d3638` to the exact
candidate without a merge commit. Final Design accepted the merged bytes and
synchronized the living documentation in a documentation-only direct child.
No push occurred.

Maintenance 4 is the completed behavior-preserving internal refactor from exact
clean package baseline `5fdd3fafe2c44357b09df2a04b88cb121f2d3638` while
retaining exact TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. It introduces non-exported
product `runtime/prepare.py`, `runtime/produce.py`, and
`runtime/validate.py` actions, replaces prepared `*Plan` values with final
frozen `*Runtime` records, shares one prepared `SamplingRuntime`, moves Charge
effects below `charge/runtime/effects/`, and makes
`readout.simulation.simulate_readout(...)` a thin
`prepare -> produce -> validate -> descendant -> retain` orchestrator.

The bounded candidate chain from Design authority
`36b9cd8338a0a5dad7e6c005d25e70c067a7e66d` was:

- Candidate 1: `0d9c322ea4db075af88b42fb7e7eeddd058e8d49`;
- Candidate 2: `8666f5dc9fa4e17e651a145afe98cb8873bb6a43`;
- Candidate 3: `cab01ca434d5c89f550e960fff8c1684fa7c2a8f`;
- final Design-authorized supplemental:
  `b3c7c907004741ba67b8b92a54bbdc8c85216dda`.

Validation returned the first two candidates for bounded committed-proof
defects and cleared Candidate 3. Independent Review returned Candidate 3 for
one consolidated relationship-proof gap. Design authorized the single exact
direct-child supplemental correction; Validation cleared it unchanged and
Review's single recheck found zero remaining findings.

Against independently reconstructed exact TensorCore source and canonical
archive forms, the final local focused suites passed at
`97 run / 88 passed / 9 conditional CUDA skips`, full discovery passed at
`198 / 185 / 13`, and Pyright `1.1.411` reported zero diagnostics in both
forms. Validation and Review each ran separate fresh full-A100 allocations;
each role passed focused source and archive suites at `97/97/0` and full source
and archive suites at `198/198/0`.

The exact public API, products, configs, science, stochastic addresses and
draw order, numerical results at accepted comparison boundaries, source and
storage contracts, autograd behavior, TensorCore dependency, and Maintenance
3 environment qualification remain unchanged. Privacy is export-driven:
runtime actions use clean internal names but appear in no supported facade.
The old private paths are removed without shims. The work authorizes no
renderer, IO/artifact surface, integration, workspace, optimization, Stage 8
restart, compatibility claim, or push. This was the correct historical
Maintenance 4 closeout boundary; the accepted sequence now places Maintenance
5 and then Maintenance 6 before any later Stage 8 authority.

### Later Integration And Artifact Stages

The exact TensorG4DS-to-`Photoelectrons` bridge, TensorML/Reconstruction
adapters, durable artifacts, and DAG/integration surfaces each require later
focused Design work. The bridge must own provenance, channel mapping, numeric
PE binning onto the bridge-selected compact `SampleAxis`, and
underflow/overflow accounting without native G4DS parsing or TensorG4DS
clustering. Model-facing field order and artifact identity are consumer/durable
contracts rather than implicit `ReadoutCollection` membership order.

## Maintenance 5 Compact-Axis And Sampling Migration

### [Maintenance 5 TensorCore 0.13 Compact Axes And Sampling](maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md)

Status: **Merged / Closed** through exact Review-cleared supplemental candidate
`81ad2f52fe4a1966e5b3a0ceb5063138e42e731f`, tree
`1c9ce87237544c32dee4b4f594e97ab929234475`. Review fast-forwarded clean
`main` from `a46899c4e3bacd6deec23ea64da5e68b382816e9` to those exact
bytes without a merge commit. Final Design accepted the merged result and
records closure in a documentation-only direct child. No push occurred.

Maintenance 5 atomically adopts published TensorCore `0.13.0` exact commit
`202d8b1bc6259b8453d3d377570417f2480d782b`, migrates
`ExampleAxis`/`ChannelAxis`/`SampleAxis` to the generic
`CountAxis`/`LabelAxis`/`RegularAxis` representations, and removes the
duplicated public `SamplingConfig` policy. Private sampling preparation derives
count, integer-picosecond period, and dimension once from the source
`SampleAxis`; the complete readout boundary retains example-local
`start == 0`.

This is a representation and API migration, not a scientific change. It keeps
the public `simulate_readout(...)` signature, all product equations and RNG
addresses, generated result contracts, arbitrary semantic axis order, and the
existing three-field `SamplingRuntime`. The exact Design authority commit and
verified execution routes are named in the dispatch handoff. It authorizes no
Pint, IO/artifact, integration, renderer, Stage 8 measurement, optimization,
compatibility shim, release claim, or push.

Implementation completed the bounded migration on
`codex/maintenance-5-tensorcore-0-13-compact-axes-and-sampling` from exact
Design authority `f734e745e5eda29bb88664c11d0a464f03c3f8e9`. Exact Candidate
1 `e6828e4309472238019844d9157b2b6bd96cd647` passed fixed-commit
Validation. Review returned one consolidated proof-only packet; Design
authorized one direct-child supplemental correction. The final candidate above
changed only three authorized test/typing paths from Candidate 1, with 67
insertions and 3 deletions. Validation cleared it unchanged and Review's single
recheck found zero remaining findings.

Against independently reconstructed exact TensorCore source and canonical
archive forms, the final local focused suites passed at
`169 run / 158 passed / 11 conditional CUDA skips`, full discovery passed at
`198 / 185 / 13`, Pyright `1.1.411` reported zero diagnostics, and the
external negative typing probe produced the same 12 expected diagnostics in
both forms. Validation and Review each ran separate fresh full-A100
allocations across PyTorch `2.11.0+cu126` and `2.12.1+cu126`; each role passed
focused source and archive suites at `169/169/0` and full source and archive
suites at `198/198/0` for both minors. Exact dependency, environment/inventory,
scope, protected-byte, import/privacy, mutation, post-merge, and repository
hygiene gates passed.

Closure changes no readout science, RNG address or call order, generated result
contract, supported-device boundary, persistence/integration surface, or
Stage 8 state. It authorizes no Pint implementation, IO/artifact work,
renderer, optimization, benchmark, release, compatibility claim, or push.
Any Stage 8 restart requires a separately accepted new Design authority from
the Maintenance 6 baseline after that maintenance closes; Maintenance 5 itself
dispatched no Pint or Stage 8 work.

## Maintenance 6 Physical-Configuration Migration

### [Maintenance 6 Pint Physical Configuration Boundary](maintenance_6_pint_physical_configuration_boundary.md)

Status: **Merged / Closed** through exact Review-cleared target
`0257fb477ee04556ebbe26351123ae610b5d7925`, tree
`b4f5703ca5b756dc27d876c1dd17ee56cb43b4e8`.

Maintenance 6 closed from exact clean Maintenance 5 Design-closeout baseline
`021694b9479d02546405f6a815aedf21c9c831a4`. It retains exact TensorCore
`0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b` and selects exact Pint
`0.25.3`.

The migration moves 26 physical fields across the existing 22 Config classes
to copied canonical scalar `pint.Quantity` values with unit-neutral names. One
private registry and one canonicalizer own recognition, conversion, defensive
copying, and exactly one TensorCore `Scalar.require(...)` normalization.
Preparation extracts each active magnitude once into plain unit-suffixed
Runtime facts. Producers, validators, product fields, collections, RNG
mechanics, and tensor kernels remain unit-free.

The stage also gives compact integer-picosecond `SampleAxis` one
`from_period(...)` constructor and four fresh Pint-valued accessors, while
retiring the redundant integer `start_ps`, `sample_period_ps`, and `stop_ps`
aliases. Its exact one-ULP integerization rule accepts conversion noise such as
`2 ns -> 2000.0000000000002 ps` without introducing general Config rounding.

A bounded TensorCore `0.13.0` golden-path cleanup removes annotation-only
Config membership checks plus duplicate admission checks inside private child
preparers/effect executors. It makes raw pulse, PSD, jitter, afterpulse, and
ADC preparation consume one-time extracted plain values. Pint
canonicalization, primitive value domains, genuine Config relationships,
public ingress validation, scientific checks, exact model dispatch,
generated-product validation, axes identity, storage freshness, and all
scientific/RNG/result behavior remain.

The work order freezes exact dependency artifacts, exports, Config census,
Runtime rename map, production/test/lifecycle allowlists, protected bytes,
complete local role evidence, explicit unavailable-CUDA qualifications, and
finite loop budgets.
Implementation was dispatched only for that exact bounded work order. It
authorized no IO/artifact, integration, Stage 8, performance, compatibility,
release, push, or sibling-package action.

Candidate 1 is one clean direct child of corrected Design authority
`6f7eba9b1c4e680930007836433f15e517288a9a`, at exact commit
`240e1492c466097b3059dfe9911ab338a4dd38a1` and tree
`1e5cae8c0e905c9638eb40e7f9d24fac950fee59`. Implementation's local
TensorCore source/archive by Pint wheel/sdist matrix passed the focused module
at `10/10`, full discovery at `207/194/13`, and Pyright `1.1.411` with zero
diagnostics in all four forms. Exact dependency artifacts, negative typing,
helper-retirement mutations, import/privacy, scope/protected-byte, diff,
artifact, and hygiene gates were also exercised.

The user later directed both packages to defer fresh cluster CUDA gates until
the new TensorCore and TensorDSLab surfaces are stable together. A
Design-owned documentation amendment therefore makes complete local
fixed-commit Validation and independent Review the Maintenance 6 closure gate;
both roles cleared the exact final target under that gate. The `13` CUDA skips
remain explicit and no accelerator claim follows.
Maintenance 6 stays pinned to TensorCore `0.13.0`. A separate later work order
will consider adoption of published TensorCore `0.15.0` exact commit
`0f974e9e7f52125bbe829e124beb24e69de811d3`, after which separately authorized
package-owned CUDA matrices may qualify only that exact integrated pairing.
Maintenance 7 now closes that adoption. Local `main` remains unpushed pending
later authorized gates.

## Maintenance 7 TensorCore 0.15 Adoption

### [Maintenance 7 TensorCore 0.15 Adoption](maintenance_7_tensorcore_0_15_adoption.md)

Status: **Merged / Closed** through exact Review-cleared and fast-forwarded
target `205182f0c7a4359cece79211ad22b47b522c34e3`, tree
`4c9f0ed2700b5683debb6e658ff2ec832e3d6acf`. Immutable production Candidate 1
is exact commit `68c2f62c2ce354dd6c92fde28b020c0ce71881d6`, tree
`a33750e4b4c094178ba4e65ffaaed530beb377d6`.

Maintenance 7 starts from exact clean local Maintenance 6 Design-closeout
baseline `65bb55bf98bb37a129a950d93a0bdb9b0d3f2971`, tree
`c76269e043c81b18243b8355327131eac68e3f0a`. It selects exact published
TensorCore `0.15.0` commit
`0f974e9e7f52125bbe829e124beb24e69de811d3`, tree
`587ff59711255c027a85cfef883422d40ea5dcda`.

The work order cleanly replaces `logical_positions(...)` with
`RngPositions`, migrates only exact-matching generic dtype/layout,
representability, allocation, shape-span, and count-domain mechanics to
TensorCore, and places the existing namespace plus all ten unchanged role keys
in one non-exported `readout/runtime/keys.py` table. Public stochastic Config
key fields and now-redundant runtime collision bookkeeping are removed; the
caller continues to select realizations through `CounterRng.seed`. Readout
composition, Pint/physical policy, product validation, scientific count
arithmetic, ledgers, role streams, raw address values, word schedules, and
public exports remain TensorDSLab-owned.
The sole remaining readout-domain structural requirement moves without a shim
from `readout/requirements.py` to the non-exported
`readout/runtime/requirements.py`, leaving the readout root organized around
its golden-path Config, collection, simulation, facade, and product packages.
Maintenance 7 also pins NumPy `2.3.5` for physical-configuration vector
storage, changes `quantities(...)` and the two PSD vector fields to one
canonical array-backed Pint Quantity each, and strips those arrays to plain
tuples during preparation. Runtime records and tensor execution remain Pint-
and NumPy-free. The ratified pulse narrowing makes both amplitudes strictly
positive magnitudes and applies fixed DS20k negative polarity exactly once in
preparation; calibrated rendered results remain exact.

The package Implementation/Validation/Review loop is local-only and makes no
new CUDA claim. Any integrated CUDA matrix remains separately authorized.
TensorDSLab remains unpushed.

Candidate 1 is one clean direct child of corrected Design authority
`ad6172b69fc86a97ba96f1751757ea33e59fef5d`; its exact commit and tree are
recorded in the immutable Validation handoff. Implementation's exact
TensorCore source/archive matrix passed the focused affected suite at
`185/173/12` and full discovery at `213/200/13`. Pyright `1.1.411` reported
zero diagnostics against both dependency forms, and the external negative
typing probe produced the same `16` required diagnostics in both forms.
Exact dependency artifacts, address/key/quantity mutations, import/privacy,
scope/protected-byte, diff, build, artifact, and hygiene gates were also
exercised. CUDA was unavailable locally and no accelerator claim follows.

Review found no production defect. It returned Candidate 1 because live
package sources still described caller-configured role keys, closure-wide key
admission, and the retired root `readout/requirements.py`. Design accepts that
P1 and authorizes one direct-child documentation correction across the exact
twelve live records named in the work order. Two merge-safety corrections were
then required, with the final exceptional one-file supplemental exact target
`205182f0c7a4359cece79211ad22b47b522c34e3` independently Validation- and
Review-cleared and fast-forwarded unchanged. The correction reopened no
implementation, dependency, science, RNG address, test, or public API byte and
authorized no cluster work or push.

## Maintenance 8 Provisional Design

### [Maintenance 8 Python 3.14 And TensorCore 0.16 Modernization](maintenance_8_python314_tensorcore_0_16_modernization.md)

Status: **Provisional Design / Undispatched / TensorCore 0.16 publication
pending**.

The record preserves the accepted one-stage modernization after locally closed
Maintenance 7: exact published TensorCore `0.16.0`, CPython `3.14.6`, PyTorch
`2.13.0`, NumPy `2.5.1`, Pint `0.25.3`, Hatchling `1.31.0`, Pyright
`1.1.411`, a semantic-only `21`-name TensorCore root, exact domain-owned
`7/3/7/15/3/4/1` subpackage/validation surfaces, bounded PEP 695 and
annotation-model cleanup, descriptive non-underscored type parameters,
intentional docstrings, and the deferred exact integrated CUDA/first-push
sequence. The seven retained TensorDSLab requirements migrate from the
TensorCore root to `tensor_core.tensor.validation`; additive
`require_index()` is dependency evidence rather than a new TensorDSLab
production dependency.

The naming convention distinguishes scoped type parameters from private
module-level aliases: `DataT` is a `TableColumn` payload parameter, `ColumnT`
is a concrete semantic column subtype, `FieldT` narrows a field factory, and
the private quantity-table alias remains `_QuantityField`. The record changes
no scientific, Pint, RNG, product, Config, Runtime, or public API contract.

Design must bind the exact published TensorCore commit, exact baseline,
allowlist, artifact identities, evidence commands, lifecycle wording, and loop
budget before obtaining separate user dispatch. The provisional record itself
authorizes no implementation, dependency update, cluster work, compatibility
claim, or push.

## Provisional Post-Maintenance-8 Planning

### [Provisional DS20k Veto Profile And Public Readout Demos](proposed_ds20k_veto_profile_and_public_readout_demos.md)

Status: **Provisional Design notes / Deferred until Maintenance 8 closes**.
This non-operative record preserves the accepted direction for a later
package-owned `ds20k_veto()` demonstration profile plus
`demos/readout.py` and `demos/readout.ipynb`. It records the provisional Veto
fixture values, illustrative dark-count/noise/ADC choices, preparation-owned
negative polarity, inline random `Photoelectrons` construction, manual and
profile Config paths, public-only execution, notebook hygiene, and explicit
non-goals.

The record does not expand Maintenance 8, authorize implementation, freeze a
post-Maintenance-8 baseline, promote a production calibration, create an IO or
TensorML adapter, issue cluster authority, or authorize a push. Design must
reconcile it against the exact closed Maintenance 8 bytes and obtain explicit
user dispatch before it becomes a production work order.

## Expected Stage Discipline

Each stage should stay scoped to its work order. If implementation reveals a
real contradiction in TensorDSLab product ownership, TensorCore axis/field
semantics, in-memory relationships, cache shape, or future integration
compatibility, stop and send the issue back to Design rather than widening the
branch.

Stage work should preserve the shared `AGENTS.md` and `CONTRIBUTING.md`
standard unless the stage explicitly changes repository-wide workflow or
engineering expectations. Meaningful deviations from the sibling-repository
shape should be documented in `docs/decisions.md`, the relevant architecture
doc, or the stage work order.

Documentation-only Design stages run the documentation checks named in their
stage record. Before production Review, Implementation should run the commands
named in the work order and report known risks or deferred items.
