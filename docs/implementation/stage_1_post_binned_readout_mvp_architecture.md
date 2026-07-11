# Stage 1 Post-Binned Readout MVP Architecture

Status: Design-complete documentation stage. At Stage 1 completion, no
production implementation had been dispatched. Stage 2 subsequently completed
the required exact-base, role-separated production loop and is Merged / Closed
on `main` at `e8c62caf001ee7f58f766d7234747ed1d9a21e35`.

## Task

Establish the durable TensorCore integration and post-binned readout
architecture before creating package code, tests, caches, IO boundaries, DAG
surfaces, or downstream adapters.

The target product flow is:

```text
"readout.photoelectrons"
  -> optional timing-jitter replacement
  -> simulate_charge(
       dark counts
       + parallel frozen-snapshot crosstalk and afterpulses
       + aggregate charge smearing
     )
  -> "readout.charge"
  -> "readout.waveform.pure"

layout/sample grid -> "readout.waveform.noise"
pure + noise -> "readout.waveform.analog"
analog -> "readout.waveform.digitized"
```

## Workflow

This stage is owned directly by Design. It does not require the production
Implementation/Validation/Review loop unless the user asks for independent
documentation review.

The full role loop should be established before the first production-code work
order is dispatched.

## Baseline

Initial repository baseline:

```text
29c5589358b4ad38afe68596a4f77efc52464ee6
```

That commit is historical provenance for the Design pass, not a production
dispatch base. Phase 1 commits the synchronized Design record containing this
stage. A future production dispatch must still name the exact then-current
clean base and confirm that the accepted documentation has not drifted.

TensorCore `0.6.0` is the current design contract snapshot. This stage does not
add package metadata or pin a runtime dependency.

## Durable Source Of Truth

Stage 1 architecture lives in:

- [TensorCore Integration Architecture](../architecture/tensors.md);
- [Post-Binned Readout Architecture](../architecture/readout.md);
- [IV-DSLab Parity And Intentional Divergences](../parity.md);
- [Design](../design.md);
- [Decisions](../decisions.md);
- [Validation](../validation.md);
- repository-wide [Contributing](../../CONTRIBUTING.md) and
  [Agent Workflow](../../AGENTS.md).

This stage document records scope and readiness. It should not duplicate the
full architecture contracts.

## Accepted Architecture Outcomes

### TensorCore Boundary

- Use public root `tensor_core` imports.
- Extend only open `Id` and `TensorCollection` surfaces.
- Do not subclass sealed TensorCore primitives.
- Do not fork generic tensor, scalar, mapping, validation, selection, batching,
  movement, or output-buffer mechanics.
- Keep product labels separate from TensorCore field IDs.
- Place concrete domain packages directly under `tensor_dslab`, including
  `tensor_dslab.common` and `tensor_dslab.readout`; do not add an intermediate
  `tensor_dslab.domain` namespace.
- Own shared `ExampleId` and `ChannelId` coordinates in
  `tensor_dslab.common`. Readout and future reconstruction reuse the same exact
  `ChannelId` identity.

### Product Surface

- Use one primary `ReadoutCollection(TensorCollection)`, not one subclass per
  charge or waveform field.
- Treat it as a structurally immutable partial snapshot whose retained tensor
  payloads are read-only to transforms.
- Recognize exactly these field IDs in canonical topological order:

  ```text
  readout.photoelectrons
  readout.charge
  readout.waveform.pure
  readout.waveform.noise
  readout.waveform.analog
  readout.waveform.digitized
  ```

- Accept any nonempty subset. Public constructors reject noncanonical order;
  domain builders and transform-result helpers emit the canonical order
  filtered to the fields present.
- Require every present field to have the same exact layout and device and use
  `torch.strided` layout without requiring contiguous strides; require shared
  example, channel, sample, and accepted extra axes.
- Own and export exactly these required semantic axis identities:

  ```python
  READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")
  READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")
  READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")
  ```

- Treat those constants as identities, not tensor positions. Required axes may
  occur in any layout order and are located by `TensorAxisId` equality and
  `TensorAxes.index(...)`, never fixed dimensions or object identity.
- Require the example axis to be ID-backed by exact `ExampleId` coordinates,
  the channel axis by exact shared `ChannelId` coordinates, and the sample axis
  to be count-only. Every accepted extra axis occurs in every field and is
  shared.
- Retain typed regular-sample facts in `SampleGrid`. Do not add
  `ReadoutAxisRoles` or any caller-defined semantic axis-role sidecar.
- Require typed `DigitizedWaveformSpec` exactly with the digitized field so bit
  depth, voltage transfer, analog gain, quantization, and derived ADC bounds
  survive projection; remove the spec whenever the field is removed.
- Allow mixed numerical domains, but require all present floating roles
  (`readout.charge`, pure, noise, and analog) to share one exact floating
  dtype. The Stage 2 work order fixes photoelectrons to `torch.int64`, floating
  roles to common `torch.float32` or `torch.float64`, and digitized counts to
  `torch.int32`.
- Keep scientific transforms as free functions.
- Do not add a generic `Product` base, semantic `TensorField` subclasses, or
  per-field semantic collection subclasses.
- Define integer `readout.photoelectrons` as binned photon-origin primary PE
  seed counts and floating `readout.charge` as their aggregate PE-equivalent
  response after charge simulation.
- Do not expose intermediate dark-count, crosstalk, afterpulse, or aggregate
  avalanche-count grids as fields or products.
- Distinguish analog `readout.waveform.analog` from integer
  `readout.waveform.digitized` ADC counts.
- Allow runtime rank, axis order, and compatible extra axes.

A future `ReadoutExample` may be a thin provenance/context wrapper around the
collection. It is optional and is not the primary tensor handoff or a second
product graph.

### Transform Policy

- `out=None` returns a new `ReadoutCollection` snapshot by adding or replacing
  exactly one target field and does not mutate inputs.
- Transform-driven addition or replacement transitively removes every
  materialized descendant reachable from the target. Noise depends on the
  common layout/sample grid rather than charge values and remains valid across
  charge-only updates when those common sidecars do not change.
- Valid retained fields reuse the same frozen `TensorField` records and tensor
  objects. Transforms treat those shared payloads as read-only even though
  PyTorch tensors are not intrinsically immutable.
- Callers also treat every materialized field as read-only. Manual in-place
  PyTorch edits bypass invalidation and are outside the value-object contract.
- Projection or explicit removal may retain descendants because it changes no
  retained value and is not semantic field replacement.
- Differentiable deterministic transforms preserve autograd on the functional
  path.
- Supplied `out` is a buffer-reuse simulation path and returns that exact
  destination.
- `out` must have exactly the functional result's field set, canonical filtered
  order, sidecars, exact common layout/device, `torch.strided` layout,
  structurally shared retained fields, and role-correct target dtype.
- Only the fresh target buffer is writable, and it must not alias any source,
  retained tensor, or workspace scratch.
- Target allocation is field-scoped and zero-initialized in a valid value
  domain. Public construction must not create an invalid uninitialized
  `torch.empty_like` target before fill.
- No implicit movement, cast, detachment, replacement, or retained-field
  mutation is allowed.
- Ordinary functional and destination-reuse execution may allocate internal
  scratch. Only the accepted warmed workspace mode removes
  TensorDSLab-managed scratch-storage allocation from its steady-state
  contract.
- Complete config/result/workspace/lease/stream/gradient preflight before RNG
  consumption or target writes. Preflight failure leaves state unchanged;
  successful completion overwrites the full target, while launched backend
  failure has no transactional rollback guarantee.

The invalidation graph is:

```text
readout.photoelectrons -> readout.charge -> readout.waveform.pure
layout + sample grid -> readout.waveform.noise
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog
readout.waveform.analog -> readout.waveform.digitized
```

### Full-Chain Builder And Workspace Policy

- Keep semantic collection layout and warmed execution layout separate.
  `ReadoutCollection` accepts arbitrary axis order and noncontiguous
  `torch.strided` read-only values. Warmed `out + workspace` requires sample
  last and every participating source/generated-target/scratch tensor
  contiguous; writable tensors are internally nonoverlapping and disjoint.
  Do not add an execution-ready collection subclass or order/stride sidecar.
- Provide one local `build_readout_collection(...)` convenience builder over
  the free transforms. It requires `readout.photoelectrons` and runs optional
  timing jitter, charge simulation, pure and noise rendering, analog
  composition, and optional digitization in dependency order.
- The full configured result contains photoelectrons, charge, pure, noise, and
  analog, plus digitized exactly when configured. The builder recomputes this
  chain rather than opportunistically retaining an existing descendant;
  partial-output execution plans remain deferred.
- Prepare the exact full-result destination before execution, structurally
  sharing only an unchanged photoelectron field and zero-initializing every
  generated target contiguously in the existing semantic axis order without
  reordering retained fields. Validate the complete call before the first write, hold
  the destination exclusively for the full call, fully overwrite each target,
  and never expose intermediate stage snapshots.
- Preserve the one-target free-transform rule by forming exact stage-specific
  collection views over the prepared full-result field records rather than
  authorizing arbitrary multi-field mutation.
- Accept exactly three execution modes:
  `out=None, workspace=None` for an owned functional result;
  `out=destination, workspace=None` for destination reuse with ordinary
  scratch or documented normalization allocation; and
  `out=destination, workspace=workspace` for warmed
  steady-state TensorDSLab-managed tensor-storage-allocation-free execution.
  Supplying a workspace without `out` is invalid.
- Treat both supplied-`out` modes as simulation paths and reject
  gradient-sensitive use. The functional mode retains the accepted autograd
  behavior of its differentiable deterministic transforms.
- Make `ReadoutWorkspace` a caller-owned, explicitly supplied, scratch-only
  runtime resource. It never owns or aliases a public field, source, result,
  coordinate map, or mutable RNG stream. Returned collection lifetime is
  independent of workspace lifetime.
- Give each workspace one fixed exact signature: ordered layout axis IDs and
  sizes; exact device; count, common floating, digitized, and derived complex
  dtypes when used; enabled algorithm families; exact destination schema;
  stream; and every scratch-shape parameter. Derive required-axis positions
  from the ordered IDs by equality/index; do not store a role sidecar or
  semantic-position mapping.
  Current batch coordinates remain source values rather than cached workspace
  state. The warmed profile requires sample-last order and contiguous storage;
  no arbitrary stride tuple is needed because contiguous strides derive from
  ordered shape.
- Reject incompatible workspaces before writing. Do not resize, reallocate,
  permute, call `.contiguous()`, clone, reshape-copy, move, cast, normalize, or
  cache them implicitly. Allocate a new workspace for a new signature and
  close an old one only while idle. Prepare a non-ready source explicitly once
  outside the repeated loop.
- Bind one non-thread-safe, non-reentrant workspace to one execution stream and
  acquire one exclusive lease per full call. Same-stream order governs reuse;
  use one workspace per concurrent worker or stream. Cross-stream event pools
  and hidden synchronization are not part of the MVP.
- A functional output lives until ordinary references are released. A supplied
  destination remains a read-only snapshot until its caller explicitly reuses
  it. Overlap uses caller-owned destination ping-pong, and a destination is
  reused only after its previous consumer completes.
- Private ping-pong swaps references only within one contiguous exact
  order/shape/dtype/device storage class. Different axis orders use different
  workspaces, and integer count, floating response, and ADC domains use
  separate buffers.
- The warmed claim excludes Python record/view allocation, allocator
  bookkeeping, backend warm-up or plan caching, and opaque library scratch.
  It is not a backend-wide zero-allocation promise.

### Fixed-Grid Scientific Direction

- Timing jitter uses aggregate latent-sub-bin Gaussian redistribution and drops
  out-of-window counts. It is the only public transform that replaces
  `readout.photoelectrons`.
- Public `simulate_charge(...)` consumes photon-origin primary PE seeds without
  replacing them, adds or replaces `readout.charge`, and keeps all intermediate
  avalanche counts private.
- Inside charge simulation, dark counts use independent per-cell Poisson
  contributions.
- Inside charge simulation, crosstalk is bounded, first-generation,
  same-channel, and same-sample.
- Crosstalk and afterpulses use one frozen post-dark-count scratch snapshot.
- Standard exponential afterpulse delay is an intentional correction of the
  literal reciprocal-exponential expression in the audited IV source.
- Charge smearing completes `simulate_charge(...)` after the private integer
  effects and uses one aggregate `Normal(n, sqrt(n) * sigma)` draw clipped to
  zero. Its materialized result is the floating PE-equivalent charge response.
- Pure waveform uses causal same-length pulse convolution.
- Noise supports explicit constant, white, and direct tensor FFT models.
- Analog waveform is optional clipping of `pure + noise`.
- Digitization is a separate representation-changing field transform.

### RNG And Device Direction

- Separate scientific configs from runtime RNG/output/placement policy.
- Use coordinate-addressed random fields.
- Define the tensor-order-independent stochastic coordinate sequence as:
  required example-axis ID and coordinate; required channel-axis ID and
  coordinate; every other ID-backed shared axis ordered lexically by
  `axis_id.value` and paired with its coordinate; then
  `SampleGrid.sample_offset + local_sample_index`. Add an operation-local
  draw/counter coordinate when needed.
- Use the required axis IDs and stable coordinates rather than example,
  channel, or sample tensor indices.
- Allow extra count-only axes structurally, but require later stochastic
  transforms to reject any such axis without an accepted stable global-offset
  rule.
- Require accepted batching, chunking, and reordering invariance.
- Keep production transforms device-resident with no hidden Python-list or
  NumPy path.
- Do not require CPU/GPU bitwise RNG parity until a focused work order accepts
  an algorithm capable of providing it.

### Future TensorML Handoff

- `ReadoutCollection` is the primary tensor handoff, but class compatibility
  alone is not a model field schema.
- An explicit `TensorFieldSelection` order is the positional model-argument
  ABI and must be tested as such.
- A noncanonical model selection remains a base `TensorCollection`; do not
  reorder it to reconstruct `ReadoutCollection`.
- TensorCore generic field selection and batching return base
  `TensorCollection` records rather than preserving the semantic subclass.
- A future adapter may project fields and reconstruct semantic meaning at its
  own accepted boundary; Stage 1 does not depend on subclass preservation.
- Project fields before accelerator movement so only the requested subset is
  moved; reconstruct `ReadoutCollection` afterward only when semantic
  collection identity is required.
- Stock-loop models receiving projected fields accept `TensorCollection`
  unless a focused adapter reconstructs a canonical subset before
  `forward_pass`; a class check alone never proves field IDs.
- Do not request TensorML `input_fields` or `output_fields` changes now. Reopen
  that question only if a focused integration stage proves explicit projection
  insufficient.

## Remaining Production Gates And Stage 2 Handoff

### Afterpulse Recovery

Resolve unit-count fixed-grid afterpulses versus IV-style
`1 - exp(-delay / recovery_tau)` amplitude weighting. Afterpulse implementation
must not be dispatched until this is explicit.

### RNG Implementation

Choose the exact device-resident stateless/counter random-field algorithm,
backend support, and cross-device agreement and bitwise-identity requirements.

### Resolved Stage 2 Structural Device Contract

The Stage 2 work order accepts already-resident matching PyTorch devices for
collection structure, requires CPU tests, and repeats a structural slice on
CUDA when available. Each later transform work order must still name the
backends supported by its actual kernels.

Stage 2 keeps noncontiguous semantic collection construction but allocates
every newly generated public target contiguously in the collection's existing
axis order. It adds no workspace/preflight API and does not normalize retained
fields.

Stage 2 also fixes `AdcQuantization.TRUNCATE`, 1–16 bit ADC interpretation,
signed `torch.int32` in-memory codes, and the intended inclusive `[0, 40]` dB
gain range. Exact digitization fixtures remain Stage 3 work.

### Workspace And Execution Follow-Ups

Keep the exact physical scratch inventory and safe fusion, a compiled immutable
execution plan, partial-output plans, cross-stream event/lease pools, workspace
resizing, CUDA Graph capture, and any backend-wide zero-allocation claim open
until implementation and memory-instrumented validation provide evidence.
Any noncontiguous/stride-aware allocation-free execution profile also remains
a future focused Design decision; the MVP warmed profile is now fixed as
sample-last and contiguous.

These open implementation details do not reopen the accepted collection,
field, layout, ownership, fixed-grid, or scratch-only workspace architecture.

## Documentation Scope

This stage may update:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/overview.md
docs/design.md
docs/decisions.md
docs/parity.md
docs/validation.md
docs/architecture/tensors.md
docs/architecture/readout.md
docs/implementation/index.md
docs/implementation/stage_0_documentation_spine.md
docs/implementation/stage_1_post_binned_readout_mvp_architecture.md
docs/implementation/stage_2_package_and_readout_collection_foundation.md
```

## Non-Goals

- No `tensor_dslab/` package.
- No package metadata or dependency pin.
- No tests or fixtures.
- No copied donor code.
- No native G4DS parsing or TensorG4DS low-level analysis; those are upstream
  responsibilities. No TensorG4DS handoff, detector-window construction, or
  photoelectron binning in this post-binned stage.
- No cache schema, writer, loader, manifest, compaction, or migration.
- No DAG-compatible specs, recipes, executables, or campaign behavior.
- No TensorML adapter, model, objective, training, evaluation, or checkpoint
  surface.
- No semantic collection subclass per recognized readout field and no second
  loose product graph in `ReadoutExample`.
- No TensorML `input_fields` or `output_fields` API change.
- No TensorCore implementation change from this workspace.
- No hidden workspace cache, implicit workspace resize or movement,
  cross-stream workspace reuse, or scratch-backed returned tensor.

## Documentation Checks

Run:

```bash
git diff --check
```

Also check:

- local Markdown links resolve;
- headings are unique within each changed document;
- stale architecture alternatives are removed from active-contract prose;
- no package, test, cache, generated, or unrelated file was added;
- architecture, decisions, validation, and roadmap terminology agree;
- parity classifications, assumptions, and intentional divergences agree with
  the target readout contract.

## Completion And Next Stage

Stage 1 is complete: the synchronized Design documents are accepted, and the
remaining open scientific and implementation questions are recorded explicitly.

The next work is the
[Stage 2 package and `ReadoutCollection` foundation work order](stage_2_package_and_readout_collection_foundation.md),
which includes atomic field and full-result destination preparation. At Stage 1
completion it had not been dispatched; Design later supplied the required clean
base, target branch and files, public sketches, supported dtypes/devices,
minimum tests, Validation loop, and Review handoff. Stage 2 is now Merged /
Closed at `e8c62caf001ee7f58f766d7234747ed1d9a21e35`.
