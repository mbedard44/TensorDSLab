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

Status: **Review-cleared / fast-forward merged; Design acceptance pending** at
exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd`, from committed Design/dispatch
authority `69b0472d246e107668a7ed253fa7c10bba22de8f`. Fixed-commit Validation and
independent Review have no unresolved finding. Review fast-forwarded clean
`main` from `9ee84bf44a3a84e7e2d57d21362e79cc850f8e26` to the candidate and repeated
the required post-merge gates. Final Design acceptance remains outstanding.
No push occurred.

The focused slice adds only the private standard
`tensordslab.threefry4x32-20/v1` engine, exact fixed-point uniform conversion,
Box-Muller behavior actually consumed by noise, and the complete exact-zero,
IID-white, and PSD-shaped `_product_noise_waveform(...)` family. The private
central enum begins with exact streams `NOISE_WHITE = 0x0000_0001` and
`NOISE_PSD_COEFFICIENT = 0x0000_0002`; it assigns no Charge stream.

The acceptance mode is functionality-first vectorized eager CPU with
conditional eager CUDA. Raw words and fixed-point uniforms are exact across
accepted implementations; completed normal/PSD products are exactly
repeatable on the same backend/mode and compare statistically across backends.
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

## Candidate Future Stages

These are planning labels, not accepted work orders. Design must write a
focused stage document before Implementation starts any of them. When donor
behavior is in scope, the work order must name the comparison boundary, parity
classification, acceptance criteria, and intentional divergences defined in
[Parity](../parity.md).

### Stage 6: Charge Simulation

Possible goal: freeze the remaining Poisson/PMF/numerical-domain gates and
implement `_product_charge(...)` with private dark counts, timing jitter,
fixed-generation correlated avalanches, S1/S2 ledgers, overflow diagnostics,
and charge smearing. `Photoelectrons` remains immutable truth and all
intermediate avalanche state remains private.

### Stage 7: Public Readout Orchestration

Possible goal: implement request/config/seed preflight, exact prerequisite
planning, each prerequisite at most once, requested-only retention, and the
complete public `simulate_readout(...)` surface after every required product
producer exists. Do not add IO or persistence policy. Profile real GPU memory
and execution before proposing workspace or output-reuse architecture.

### Later Integration And Artifact Stages

The exact TensorG4DS-to-`Photoelectrons` bridge, TensorML/Reconstruction
adapters, durable artifacts, and DAG/integration surfaces each require later
focused Design work. The bridge must own provenance, channel mapping, numeric
PE binning under `SamplingConfig`, and underflow/overflow accounting without
native G4DS parsing or TensorG4DS clustering. Model-facing field order and
artifact identity are consumer/durable contracts rather than implicit
`ReadoutCollection` membership order.

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
