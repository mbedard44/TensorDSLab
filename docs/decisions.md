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

### TensorCore `0.7` Is The Rebuild Target

The rebuild uses TensorCore's public `TensorAxis`, `TensorField`, and
`TensorCollection` ordinary-ABC roots. Exact final TensorDSLab leaf types carry
axis, product, and collection meaning. TensorCore owns universal
representation validation; TensorDSLab owns domain relationships and
scientific meaning.

The reviewed TensorCore Design reference is exact clean commit
`b454d738f6385ce6489d85492a618a3dab139bb6`. Selecting the exact dependency pin
and proving public imports, runtime construction, inherited-constructor static
typing, and result-storage contracts remain requirements of the first rebuild
production work order. TensorDSLab will not fork TensorCore or reproduce its
private mechanics.

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

`SamplingConfig` defines positive integer picosecond period, count, local-zero
start, and representable exclusive stop. It creates a regular `SampleAxis` of
canonical left-edge timestamp strings such as `"0ps"` and `"2000ps"`. Hot
paths use numeric config values and indices, not the coordinate strings.
Positional RNG likewise uses tensor positions and makes no coordinate-identity
or permutation-invariance promise.

The count-only sample representation and `SampleGrid` sidecar are retired.

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
`PhotoelectronsConfig` and no readout `_product.py`. A future TensorDSLab-owned
bridge will construct it from an exact accepted TensorG4DS product using the
caller's sampling policy. That bridge, not `simulate_readout`, owns provenance
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

`simulate_readout(photoelectrons, *, products, config, seed,
floating_dtype)` is the ordinary collaborator-facing simulation API. It
consumes the product iterable once, rejects empty/duplicate/unrecognized
requests, computes the typed transitive prerequisite closure, preflights the
entire effective request, executes each producer at most once, and retains
exactly the requested fields.

Request order has no meaning. Changing retention alone must not change a
common product value. Missing config, invalid seed, or another request-level
error fails before any random draw or tensor write.

Private `_product_*` functions construct semantic products. Private
`_simulate_*` functions implement scientific submodels inside a product
producer. There is no public sequential API through which callers can
accidentally feed one private avalanche contribution into another.

### Scientific Configuration Is Immutable And Compositional

`ReadoutConfig` contains one required `SamplingConfig` and optional exact
product configs. Each product owns its field and its config types. `None`
disables an optional submodel; closed unions of exact config types select real
alternative models. Product producers receive their exact config and shared
sampling facts rather than the whole configuration as a service locator.

There is no generic `Config` ABC without a polymorphic consumer, no string
algorithm selector, no product-level `persist` flag, and no mixing of
scientific choices with runtime allocation or stream control.

### The Package Tree Is Product-Centered

Shared `ExampleAxis`, `ChannelAxis`, and `SampleAxis` live in
`common/axes.py`; `SamplingConfig` lives in `common/sampling.py`.

`readout/types.py` contains only `ReadoutConfig` and `ReadoutCollection`.
`readout/simulation.py` owns public orchestration. Shared private readout
relationships and RNG live in `_requirements.py` and `_random.py` only when
implemented.

Each product subpackage owns a `types.py` containing its field and product
configs, and later a `_product.py` containing its private producer and
submodels. `photoelectrons` has only `types.py`. Product packages never import
the cross-product collection, config, or public simulation layer.

There are no global `configs`, `fields`, `builders`, or `validation` dumping
grounds, and no empty placeholder modules. Package roots deliberately export
the collaborator-facing classes and function.

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

### Public Validation Does Not Mean Adversarial Hardening

TensorDSLab validates supported public input relationships: exact product
requests, axes, shape, dtype, device, sampling, configs, scientific value
domains at trust boundaries, seed requirements, and numerical bounds. Cheap
intrinsic checks occur in semantic leaves; full-device scans occur at explicit
ingress or producer postconditions.

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

### Count-Only Sampling And Semantic-Coordinate RNG

The count-only sample axis, collection-level grid sidecar, and coordinate-
addressed random scheme are retired. Timestamp coordinates carry semantic
left-edge labels, while numeric sampling facts and position indices drive
kernels and RNG.

### Separate Avalanche Architecture Attempts

Same-bin recursive closure, causal-scan, generation-wave, and recovery-marked
alternatives are superseded as implementation directions. The fixed-generation
model in `architecture/rebuild.md` is the only active baseline.

## Open

### Exact TensorCore Consumer Pin And Probes

The first rebuild work order must select exact TensorCore candidate
`b454d738f6385ce6489d85492a618a3dab139bb6` and prove package-root imports,
inherited constructors, exact-leaf behavior, static result inference, and
operation-owned aliasing/freshness claims.

### Charge RNG And Supported Numerical Domain

Before stochastic Charge implementation, Design must close the numeric stream
table, Poisson sampler and crossover, per-quantum versus aggregate sampling,
execution dtype and raw-word budgets, rejection/exhaustion behavior,
repeatability modes, PMF preparation precision and tolerances, stable
normal-tail evaluation, supported generation/rate/count bounds, checked
overflow, and parity tolerances.

### Waveform-Tail Execution Evidence

The deterministic waveform stage must select scalar precision and execution
mode, prove equivalence with the frozen reference equations, instrument the
one-kernel/no-target-sized-temporary target for analog and digitized products,
and define the gate for a purpose-built fallback kernel. Cross-product fusion
remains excluded.

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
