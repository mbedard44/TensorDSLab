# TensorCore Integration Architecture

## Purpose And Authority

This page defines how the TensorDSLab rebuild uses TensorCore. Scientific
readout behavior belongs in [`readout.md`](readout.md) and the complete rebuild
contract in [`rebuild.md`](rebuild.md).

Closed Stage 3 through 6 evidence uses exact TensorCore `0.7.0` commit
`b454d738f6385ce6489d85492a618a3dab139bb6`. Maintenance 2 installed exact
published TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59` for public RNG/distribution and
same-dtype ownership. Those remain immutable historical baselines.

User-authorized Maintenance 5 selects published TensorCore `0.13.0` exact
commit `202d8b1bc6259b8453d3d377570417f2480d782b`, tree
`48fa9a28db6d043abc07d9963b2015983ca436ea`, as the next dependency target.
Maintenance 5 consumes its compact semantic axes and golden-path structural
boundary while preserving the public RNG and tensor relationships TensorDSLab
already uses. The dependency also exposes generic `Scalar`, but Maintenance 5
does not consume it; the deferred Pint stage owns any later use. Every
dependency change still requires an exact pin and TensorDSLab-owned
source/archive, typing, CPU, and CUDA consumer evidence. No broad compatibility
result follows.

Maintenance 3 is Merged / Closed through exact Review-cleared candidate
`dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0`. Maintenance 4 Runtime Action
Ownership is **Merged / Closed** through exact Review-cleared supplemental
candidate `b3c7c907004741ba67b8b92a54bbdc8c85216dda`. It preserves this
TensorCore boundary and the exact `0.9.0` dependency while reorganizing only
unexported TensorDSLab preparation, production, and validation ownership.
Maintenance 5 is the user-authorized next migration. Its work order and the
implementation index are the sole lifecycle records.

The previous TensorCore `0.6` ID/layout/sidecar architecture is historical and
is intentionally not preserved through aliases.

## TensorCore Boundary

TensorDSLab imports only TensorCore's public package-root surface. The relevant
semantic roots are:

```text
TensorAxis[CoordinateT]
  CountAxis              compact zero-based integer range
  RegularAxis            compact regular integer range
  LabelAxis              explicit unique string tuple
TensorField              tensor payload plus ordered semantic axes
TensorCollection         fields keyed by exact concrete TensorField type
```

TensorCore owns:

- abstract frozen/slotted roots and representation-state checks;
- exact range-backed Count/Regular coordinate mechanics;
- exact string-label mechanics;
- ordered coordinate/index and axis/dimension lookup;
- exact-type collection lookup;
- the generic `Scalar` root and constrained scalar leaves;
- focused numeric requirements; and
- `require_axis_signature`, `require_same_axes`, `require_same_device`, and
  `require_field_types`.

TensorCore `0.13.0` retains the generic immutable counter-RNG mechanics adopted
at `0.9.0`: exact key/seed/address validation, logical row-major positions,
Threefry raw-word continuity, fixed-point uniforms, parameterized Gaussian
draws, Poisson inversion/PTRS, binomial inversion/BTRS, sampler numerical
domains/exhaustion, and the count distributions' word schedules. It also
retains `require_same_dtype(*fields)`. TensorDSLab consumes only package-root
surfaces and never duplicates protected RNG or generic representation
mechanics.

TensorCore additionally exposes generic table roots and `TensorArtifact`.
Maintenance 5 does not use them to infer a TensorDSLab table backend, schema,
artifact, persistence format, or IO policy.

TensorDSLab owns:

- final semantic readout axes and their stronger domain narrowing;
- the integer-picosecond interpretation of `SampleAxis`;
- complete-input example-local start zero;
- final readout fields and collection;
- readout axis sets and intrinsic product dtypes;
- collection membership and cross-field coherence;
- deep value-domain checks at explicit trust boundaries;
- product dependencies, scientific configs, operations, and orchestration;
- placement of exact `RngKey` role identities on stochastic leaf configs,
  scientific position/category lattices, direct-uniform/Gaussian ordinals,
  draw-free scientific policy, multinomial ordering and final remainders,
  count accumulation, and ledgers;
- one private scalar-to-floating-dtype representation requirement and all raw
  tensor dtype/shape/device policy;
- result aliasing/freshness, device/dtype/layout, autograd, synchronization,
  failure, and multi-output relationships; and
- any future persistence or execution optimization.

TensorCore `0.13.0` has no generic selection, batching, movement, reduction,
addition, detachment, output-buffer, `out=`, workspace, storage, lease, or
lifecycle service. TensorDSLab does not recreate retired APIs merely to
preserve its pre-deployment implementation.

## Extension Contract

Each TensorDSLab semantic leaf directly inherits exactly one matching
TensorCore semantic or representation root, with no mixin:

```python
@final
class ExampleAxis(CountAxis):
    __slots__ = ()

    def _require(self) -> None:
        if self.count == 0:
            raise ValueError("ExampleAxis must be nonempty")
```

Every leaf:

- is a public `@final` class;
- declares `__slots__ = ()`;
- adds no stored fields;
- implements `_require()` for TensorDSLab narrowing; and
- inherits root construction, validation, identity/equality, and lookup.

Leaves do not reapply `@dataclass`, override construction, or add semantic
state. TensorCore's final `_validate()` phase always establishes universal
validity before the package-owned `_require()` phase.

TensorCore `0.13.0` and TensorDSLab intentionally rely on static checks, tests,
and Review for supported leaf structure. They retain runtime checks for
constructed data/scientific invariants, not hostile Python metaprogramming.
Malformed class objects, ignored type errors, private-module use,
monkey-patching, or subclassing a statically final leaf have no promised
runtime result or diagnostic category.

## Semantic Axes

TensorDSLab owns:

```text
ExampleAxis(CountAxis)
ChannelAxis(LabelAxis)
SampleAxis(RegularAxis)
```

Every readout field has exactly one of each and no other axis. The ordered axes
tuple is tensor dimension order. Semantic construction accepts any order;
operations locate dimensions by exact type:

```python
sample_dimension = field.dimension_of(SampleAxis)
sample_axis = field.axis(SampleAxis)
```

Exact axis types are unique in a field by TensorCore construction. TensorDSLab
checks the unordered readout type set so it does not accidentally impose one
dimension order through `require_axis_signature`.

Coordinates, indices, and dimensions remain distinct:

- a coordinate is the representation-specific semantic value at one position;
- an index is that zero-based position; and
- a dimension is the position of an axis in a field's ordered axes tuple.

`ExampleAxis(count=N)` exposes `range(0, N)` and carries no durable event
identity. `ChannelAxis(labels=(...))` retains explicit unique detector labels.
`SampleAxis(start, step, count)` exposes a compact regular `range` of integer
picosecond left edges. It requires nonnegative start, positive step, at least
two samples, and `start + step * count <= 2**63 - 1`.

The semantic sample axis may have nonzero start. Complete
`Photoelectrons`/readout preparation requires start zero and derives one
private `SamplingRuntime` directly from the source axis. There is no public
`SamplingConfig` or second period/count authority.

Dimension-preserving producers reuse the exact source axes tuple and exact
axis instances. This is stronger than merely reconstructing equal axes and is
part of each producer's return contract.

There are no `TensorAxisId` values, axis constants, ID-backed/count-only
sidecar distinction, coordinate object classes, `TensorLayout`, `TensorAxes`,
`shared_axes`, axis-role sidecar, or `SampleGrid`.

## Product Fields

The six direct final `TensorField` leaves are:

| Type | Intrinsic representation | Builder/deep-validator domain |
| --- | --- | --- |
| `Photoelectrons` | exact readout axes, `torch.int64`, `torch.strided` | nonnegative |
| `Charge` | exact readout axes, `torch.float32` or `torch.float64`, `torch.strided` | finite and nonnegative |
| `PureWaveform` | exact readout axes, `torch.float32` or `torch.float64`, `torch.strided` | finite |
| `NoiseWaveform` | exact readout axes, `torch.float32` or `torch.float64`, `torch.strided` | finite |
| `AnalogWaveform` | exact readout axes, `torch.float32` or `torch.float64`, `torch.strided` | finite |
| `DigitizedWaveform` | exact readout axes, `torch.int32`, `torch.strided` | nonnegative and bounded by config |

TensorCore guarantees that each payload is a `torch.Tensor`, its axes are an
exact tuple of valid axes, exact axis types are unique, and tensor shape equals
axis sizes. TensorDSLab leaf `_require()` methods add cheap intrinsic axis,
dtype, and Torch-layout restrictions.

A bare semantic constructor is not proof of a full scientific value domain.
Full-device finiteness, nonnegativity, or config-dependent ADC-bound scans run
through explicit product-owned runtime validators at untrusted ingress and
generated-product publication boundaries. This avoids hidden device
synchronization during every semantic reconstruction and keeps `field.py`
limited to semantic identity plus cheap intrinsic narrowing.

Exact concrete field type is the in-process product identity. There are no
`TensorFieldId` values, product-name strings, field-role constants, metadata
maps, or intermediate `ReadoutField` base class.

## ReadoutCollection

`ReadoutCollection` is a direct final `TensorCollection` leaf and an immutable
completed result for one product request. It accepts any nonempty subset of the
six exact product types.

Its intrinsic requirements are:

- every member has one recognized exact product type;
- all members have equal ordered axes;
- all members use the same exact device; and
- all present floating readout products use one common dtype.

TensorCore already rejects duplicate exact field types. Collection membership
is semantically unordered, so neither constructor order nor mapping iteration
defines dependencies, provenance, execution sequence, or model schema.

Access is exact-type directed:

```python
charge = readout.field(Charge)
charge_tensor = readout.tensor(Charge)
```

A missing unrequested type raises `KeyError`. The collection is not a workflow
object and has no public append, remove, replace, selection, invalidation, or
partial-validity state. The sole accepted schema declaration belongs on the
collection class and replaces loose field registries and canonical sequences.

## Construction And Dependency Ownership

Product relationships appear in private runtime-action signatures, not in
collection membership or TensorCore. Product configuration enters the
corresponding typed private preparer:

```python
def prepare_analog_waveform(
    config: AnalogWaveformConfig,
    *,
    floating_dtype: torch.dtype,
    device: torch.device,
) -> AnalogWaveformRuntime:
    ...


def produce_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    runtime: AnalogWaveformRuntime,
) -> AnalogWaveform:
    ...


def validate_analog_waveform(
    analog: AnalogWaveform,
    *,
    pure: PureWaveform,
    noise: NoiseWaveform,
) -> None:
    ...
```

`simulate_readout(...)` owns the complete request and dependency sequence. Its
private `prepare_readout(...)` builds one `ReadoutRuntime` composed from
optional product-owned Runtime records, retains exactly the requested
products, then constructs `ReadoutCollection` once. Complete preparation
precedes every product-production invocation. A prerequisite does not need to
be returned merely because it was computed.

Product packages do not receive a whole collection as a service locator. Each
preparer receives its exact config plus the source, sampling, device, shape, or
floating-dtype facts it needs. Each ProductRuntime is a concrete final frozen slotted
data carrier containing no Config, semantic product, mutable cache, method, or
Runtime base. Each producer receives explicit prerequisite fields and its
product-owned Runtime, plus one public TensorCore `CounterRng` when that
producer is stochastic-capable. Deterministic producers and helpers receive no
RNG. The stochastic effect uses the exact `RngKey` captured from its leaf
config during preparation.

`readout.runtime.sampling` owns one private `SamplingRuntime` containing Python
integer sample count, period, and dimension. Request preparation derives it
once from the source `SampleAxis` after enforcing complete-input start zero;
temporal ProductRuntime values that retain sampling facts reference that exact
object. This private source-bound execution state has no public companion
config or common sampling module.

Production imports no Config, performs no scientific preparation, and owns no
deep publication scan. Immediately after every generated field is constructed,
`simulate_readout(...)` invokes its product-owned `validate_<product>(...)`
before any descendant can consume it. Product runtime packages and their
`__init__.py` files are deliberately absent from public facades; direct deep
imports remain unsupported implementation access rather than a public API.

The package tree and dependency direction are fixed in
[`rebuild.md`](rebuild.md): common axes/sampling, unexported shared readout
requirements, product config/field modules, the source-bound sampling Runtime,
product runtime actions, readout config/collection, then public simulation.
There is no
generic field/config/builder/validation layer, Runtime ABC, action registry, or
reflection-based dependency graph.

## Functional Result Contract

TensorCore keeps a borrowed tensor reference and makes no universal freshness
or sharing guarantee. Every TensorDSLab operation returning semantic fields
classifies each successful path as one of:

- exact return;
- guaranteed storage-sharing result;
- sharing permitted but unspecified; or
- guaranteed fresh storage independent of named inputs.

The initial rebuild adopts a deliberately simple policy:

- requested `Photoelectrons` is an exact return of the named source field;
- every generated product has guaranteed-fresh storage independent of its
  named inputs;
- generated fields retained together are storage-independent from each other;
  and
- no initial public path uses guaranteed sharing or unspecified sharing.

Each operation additionally states exact result type, axes, dtype, device,
Torch layout/stride or contiguity where promised, autograd, synchronization,
failure effects, and any output-to-output relationship. Compact Python syntax
does not prove fusion, freshness, or absence of target-sized temporaries;
execution claims require instrumentation.

Frozen semantic records do not make PyTorch storage physically immutable.
Borrowed inputs are logically read-only. TensorDSLab enqueues all producer
writes before constructing and exposing the corresponding field and initiates
no later write through an alias. Private writable scratch remains exclusive
and unexposed and never enters a returned collection.

Constructing or returning a field is not itself an additional device
synchronization point. Accepted deep-value runtime validators use scalar
reductions that may synchronize CUDA. Outside those documented correctness
checks, ordinary current-stream PyTorch ordering applies; a cross-stream
consumer establishes its own event or stream dependency. Strong references
preserve lifetime, not write safety or stream ordering.

## No Initial Output-Buffer Layer

The rebuild starts with functional allocation. It has no public `out=`, valid
destination collection, `ReadoutWorkspace`, stream lease, allocator, pool, or
allocation-free claim.

If profiling later justifies reuse, writable destinations must remain raw,
exclusive, and semantically unexposed until all producer writes are enqueued.
Only then may TensorDSLab construct the completed field once. A future design
must not overwrite storage through an already exposed valid field or collection
and must not leak subsequently reusable scratch into a returned product.

## Device, Dtype, And Autograd

All present fields in a `ReadoutCollection` use one exact device. The public
builder accepts source tensors only on CPU or CUDA, even for a truth-only
request. It never silently moves, casts, detaches, calls `.cpu()` or `.numpy()`,
converts through Python lists, or serializes/reloads an existing source.
Generated products use their declared output dtypes.

The initial representation accepts ordinary `torch.strided` readout tensors
with any semantic axis order. A future measured execution profile may require
sample-last contiguous storage, but that is operation policy rather than
semantic collection identity.

Deterministic differentiable waveform operations preserve autograd where their
accepted implementation does. Discrete stochastic count simulation and ADC
digitization make no blanket differentiability promise. Exact behavior belongs
to each operation work order and test matrix.

The supported contracts cover ordinary PyTorch tensor behavior. TensorDSLab
does not exhaustively detect or normalize custom tensor subclasses and dispatch
modes that replace ordinary aliasing or operation semantics.

## Public Validation Boundary

Public `simulate_readout(...)` prepares the complete effective request before
its first RNG request, product-producer invocation, or semantic-output write.
It validates recognized unique product classes, required configs, source
sampling agreement, axes, shape, dtype, CPU/CUDA device, selected floating
dtype, a required `CounterRng` instance, exact config keys, closure-wide
duplicate-key rejection, representability, and all statically preparable
operation relationships needed by the closure.

TensorCore exposes no non-consuming concrete-algorithm capability query.
Deterministic closures validate nominal `CounterRng` membership and request no
values; stochastic closures perform no dummy probe. A real custom RNG backend
failure at the first genuine distribution request is an execution failure and
has no rollback guarantee.

Private product and scientific functions are internal independently testable
units. They may trust values that passed public preflight and do not need to
repeat the whole boundary or defend direct unsupported calls. Cheap
correctness-critical local assertions remain acceptable when they protect the
function's own valid-result contract.

The merged Maintenance 4 implementation preserves the Stage 7
public-operation postcondition through an explicit action boundary. A product
producer constructs and returns one local
field without interpreting Config or performing its deep scan. Orchestration
then invokes the corresponding product-owned runtime validator exactly once
with the exact generated field and its named direct prerequisite relationship
before any descendant or returned collection can consume that field. The
sequence is `produce -> validate -> descendant`, not “produce every field,
then validate them all.” This is not an intrinsic TensorCore constructor
invariant and may synchronize CUDA through scalar extraction.

`validate_photoelectrons(...)` remains an untrusted-ingress check during whole-
request preparation. Generated `validate_charge`,
`validate_pure_waveform`, `validate_noise_waveform`,
`validate_analog_waveform`, and `validate_digitized_waveform` own their exact
value domains. Digitized validation uses the prepared Runtime maximum code
rather than a Config. Charge performs one terminal finite/nonnegative scan and
preserves the accepted invalid-generated-result `RuntimeError` boundary. A
failed validation prevents descendants and final collection construction; the
failed local field never escapes.

Malformed supported data operands use TensorCore's documented `TypeError`
boundary; unsatisfied well-formed relationships use `ValueError` where
applicable. Malformed/off-path class-object calls have no promised behavior.
TensorDSLab does not promise error behavior for deliberate public-API drift.

## Persistence And Integration

Exact Python classes are process-local semantic identities, not durable
product labels. TensorCore now supplies a generic `TensorArtifact` extension
root, but TensorDSLab persistence remains deferred and will require a concrete
artifact, separately versioned labels, and the scientific configuration
necessary to interpret a payload.
The retired `DigitizedWaveformSpec` sidecar is not carried into the collection;
durable digitization-config association remains an explicit Design gate.

A future TensorG4DS bridge constructs dense truth `Photoelectrons` rather than
casting an upstream object into a downstream leaf. A future TensorML boundary
selects and orders exact products explicitly rather than relying on collection
iteration or subclass identity as a model ABI.

Maintenance 4 implements an internal Config-to-Runtime-to-Product seam but adds
no public renderer or model component. A reusable `PureWaveformRenderer`, its
buffer/state behavior, and its synchronization-free trusted forward boundary
remain a separate focused Design and implementation stage.

## Return To Design Before

Return to TensorDSLab Design before:

- changing the TensorCore version or using a non-root import;
- adding stored state or another inheritance layer to semantic leaves;
- restoring IDs, layouts, constants, sidecars, or compatibility shims;
- adding another axis or product type;
- exporting a Runtime action or adding a public renderer without its focused
  work order;
- adding generic movement, selection, reconstruction, or metadata locally;
- introducing a public destination/workspace/lifecycle surface;
- weakening fresh-result, no-post-exposure-write, device, dtype, or axes
  contracts;
- making a custom-tensor, cross-device, compatibility, persistence, or
  allocation-free claim; or
- changing the TensorG4DS or TensorML boundary.
