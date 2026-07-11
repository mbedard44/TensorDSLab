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

## Active Production Work Order

### [Stage 2: Package And Readout Collection Foundation](stage_2_package_and_readout_collection_foundation.md)

Status: Dispatched on 2026-07-11 from exact clean package baseline
`d097cb3cdde185c6814116e886e7844ea3f55178`. The package-owned persistent
Implementation, Validation, and Review routes are active for the bounded
production loop. No Stage 2 production result is accepted until a fixed
candidate passes Validation and independent Review and is cleanly merged.

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

## Candidate Future Stages

These are planning labels, not accepted work orders. Design must write a
focused stage document before Implementation starts any of them. When donor
behavior is in scope, the work order must name the comparison boundary, parity
classification, acceptance criteria, and intentional divergences defined in
[Parity](../parity.md).

### Stage 3: Deterministic Transforms And Execution Substrate

Possible goal: implement pulse-template pure rendering, constant baseline,
analog composition, clipping, and digitization with functional and
output-buffer paths over structurally immutable collection snapshots.
Preserve autograd for accepted functional analog transforms, treat retained
tensors as read-only, structurally share retained fields, and invalidate stale
descendants when a target field is transform-added or replaced. Add the exact
execution-signature and caller-owned scratch-only `ReadoutWorkspace`
substrates, including fixed-device/single-stream/non-reentrant leasing and
deterministic-tail scratch. Do not publish a placeholder full-chain builder
before the photoelectron-to-charge branch exists; keep its scheduling machinery
internal until Stage 4 can expose a complete configured chain.
Lock warmed preflight to contiguous sample-last participating storage, exact
destination/workspace/stream/lease signatures, and internally nonoverlapping
writable buffers. Functional paths may allocate explicit normalization;
ordinary `out` paths may allocate documented scratch or normalization; warmed
calls reject rather than permute, make contiguous, cast, move, reshape-copy, or
allocate a fallback. Do not add an execution-ready collection subclass or
storage sidecar. The focused Stage 3 work order must sketch the explicit
one-time source-preparation/materialization API (or an equally explicit caller
composition) that produces a validated sample-last contiguous semantic value
outside the repeated loop; Stage 2 intentionally provides no placeholder.

### Stage 4: Tensor-Native RNG And Workspace-Backed Charge Simulation

Possible goal: accept and implement the coordinate-addressed random-field
contract, timing jitter, and the public `simulate_charge(...)` transform.
Timing jitter is the only transform that replaces the binned photon-origin
primary PE seeds in `readout.photoelectrons`. Charge simulation consumes those
seeds, applies dark counts plus frozen-source crosstalk and afterpulses in
private integer scratch grids, applies aggregate charge smearing, and adds or
replaces the floating PE-equivalent `readout.charge` response. Intermediate
avalanche counts are not products. Extend the full-chain builder and workspace
signature with these algorithms, prove same-stream lease safety, and validate
the warmed TensorDSLab-managed tensor-storage-allocation-free mode for the
accepted charge kernels. Publish `build_readout_collection(...)` here with the
deterministic waveform/noise modes already accepted in Stage 3; Stage 5 may
extend its noise configurations without changing builder ownership or lifetime.
Add afterpulses only after Design resolves recovery-amplitude semantics.
Prove that count-domain ping-pong swaps references only within one exact
contiguous order/shape/dtype/device class, that different axis orders use
different workspaces, and that integer count, floating response, and ADC
domains remain separate buffers. Instrument stable target/scratch storage and
reject sample-not-last/noncontiguous warmed calls before RNG or writes.

### Stage 5: Stochastic Waveform Noise

Possible goal: implement Gaussian white noise and direct tensor FFT noise over
the accepted collection/RNG substrate, extend the full-chain workspace
signature and scratch preparation for those algorithms, and validate the three
execution modes without claiming backend-wide zero allocation.

### Stage 6: Readout Example Composition

Possible goal: define an optional thin `ReadoutExample` provenance/context
wrapper around the primary `ReadoutCollection` handoff without creating a
second product graph, upstream handoff, durable IO, or downstream adapters.

### Stage 7: TensorG4DS Handoff And Photoelectron-Binning Contract

Possible goal: accept one exact public TensorG4DS event/deposit/cluster product
and add the smallest TensorDSLab-owned semantic bridge needed to produce
`readout.photoelectrons`. The work order must define the potentially
one-to-many TensorG4DS `EventId` provenance-to-`ExampleId` mapping,
TensorDSLab-owned channel mapping, detector response, window/readout-grid
construction, and photon-origin primary-PE binning. It must require an
already-placed same-GPU input and output with no implicit host staging,
serialization, movement, cast, or detach, while recognizing that the bridge
constructs new example/channel/sample layouts rather than casting upstream
collections. It must bind and validate the accepted versioned TensorG4DS
position/time/energy unit contract and keep conversions explicit and
on-device. Native G4DS parsing and TensorG4DS clustering stay out of
TensorDSLab. No concrete TensorG4DS import or adapter type is accepted until
TensorG4DS itself freezes the required public GPU/device contract. The first
discrete bridge carries no end-to-end autograd promise and must reject
gradient-sensitive input rather than detaching silently. The work order must
also resolve empty upstream events and zero/one/many emitted examples without
inventing sentinel deposits, channels, or IDs.

### Stage 8: Durable Cache Contract

Possible goal: decide whether the first cache target is compatibility-oriented,
loader-compatible, or a new tensor-native format, then implement only the
accepted product reconstruction and storage boundary.

### Stage 9: Integration Surfaces

Possible goal: add DAG-compatible operation specs, executable doors, recipe
fragments, or downstream adapters only after in-memory products, TensorCore
contracts, and durable cache contracts are stable.

Any future TensorML adapter should use explicit field selection order as the
positional model ABI and account for generic selection/batching returning base
`TensorCollection`. It should not assume the `ReadoutCollection` class alone is
a model schema or pre-authorize TensorML `input_fields` / `output_fields`
changes.

## Expected Stage Discipline

Each stage should stay scoped to its work order. If implementation reveals a
real contradiction in TensorDSLab product ownership, TensorCore layout
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
