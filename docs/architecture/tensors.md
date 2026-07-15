# TensorCore Integration Architecture

## Purpose And Authority

This page defines how the TensorDSLab rebuild uses TensorCore. Scientific
readout behavior belongs in [`readout.md`](readout.md) and the complete rebuild
contract in [`rebuild.md`](rebuild.md).

The reviewed TensorCore target is version `0.7.0` at exact clean commit
`b454d738f6385ce6489d85492a618a3dab139bb6`. A production work order must
select the exact dependency pin and run TensorDSLab-owned consumer probes; this
document alone makes no compatibility claim.

The previous TensorCore `0.6` ID/layout/sidecar architecture is historical and
is intentionally not preserved through aliases.

## TensorCore Boundary

TensorDSLab imports only TensorCore's public package-root surface. The relevant
semantic roots are:

```text
TensorAxis
  coordinates: tuple[str, ...]

TensorField
  tensor: torch.Tensor
  axes: tuple[TensorAxis, ...]

TensorCollection
  fields keyed by exact concrete TensorField type
```

TensorCore owns:

- abstract frozen/slotted roots and their universal representation checks;
- ordered coordinate/index and axis/dimension lookup;
- exact-type collection lookup;
- constrained scalar wrappers;
- focused numeric requirements; and
- `require_axis_signature`, `require_same_axes`, `require_same_device`, and
  `require_field_types`.

TensorDSLab owns:

- final semantic readout axes, fields, and collection;
- readout axis sets and intrinsic product dtypes;
- collection membership and cross-field coherence;
- deep value-domain checks at explicit trust boundaries;
- product dependencies, scientific configs, operations, and orchestration;
- result aliasing/freshness, device/dtype/layout, autograd, synchronization,
  failure, and multi-output relationships; and
- any future persistence or execution optimization.

TensorCore `0.7` has no generic selection, batching, movement, reduction,
addition, detachment, metadata, output-buffer, `out=`, workspace, storage,
lease, registry, publication, or lifecycle API. TensorDSLab does not recreate
retired generic APIs merely to preserve its pre-deployment implementation.

## Extension Contract

Each TensorDSLab semantic leaf has exactly its matching TensorCore root in
`__bases__`, with no mixin or other base:

```python
@final
class ExampleAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
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

Ordinary ABC inheritance intentionally relies on static checks, tests, and
Review for package-authored leaf shape. TensorDSLab does not add a metaclass,
runtime-finality guard, lineage scanner, or other adversarial defense against a
caller who deliberately violates the public contract.

## Semantic Axes

TensorDSLab owns three final readout axis types:

```text
ExampleAxis
ChannelAxis
SampleAxis
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

Each coordinate is a unique nonempty exact string, and tuple order defines
index order. Coordinates, indices, and dimensions remain distinct:

- a coordinate is the semantic string at one axis position;
- an index is that zero-based position; and
- a dimension is the position of an axis in a field's ordered axes tuple.

`ExampleAxis` and `ChannelAxis` contain stable package-owned labels.
`SampleAxis` contains regular increasing left-edge timestamps generated from
`SamplingConfig`, using canonical strings such as `"0ps"` and `"2000ps"`.
Kernels use the numeric sampling values and integer indices; they do not parse
semantic labels in the hot path.

Dimension-preserving producers reuse the exact source axes tuple and exact
axis instances. This is stronger than merely reconstructing equal axes and is
part of each producer's return contract.

There are no `TensorAxisId` values, axis constants, ID-backed/count-only
distinction, coordinate object classes, `TensorLayout`, `TensorAxes`,
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
at untrusted ingress and producer postconditions. This avoids hidden device
synchronization during every semantic reconstruction.

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

Product relationships appear in private function signatures, not in collection
membership or TensorCore:

```python
def _produce_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    config: AnalogWaveformConfig,
) -> AnalogWaveform:
    ...
```

`simulate_readout(...)` owns the complete request and dependency plan. It uses
private mutable local assembly state, retains exactly the requested products,
then constructs `ReadoutCollection` once. A prerequisite does not need to be
returned merely because it was computed.

Product packages do not receive a whole collection as a service locator. Each
producer receives explicit prerequisite fields, its exact config, shared
sampling facts when numerically relevant, and private RNG inputs when needed.

The package tree and dependency direction are fixed in
[`rebuild.md`](rebuild.md): common axes/sampling, private shared readout
requirements, product types, product producers, cross-product types, then
public simulation. There is no generic field/config/builder/validation layer.

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

Constructing or returning a field is not a device synchronization point.
Ordinary current-stream PyTorch ordering applies; a cross-stream consumer
establishes its own event or stream dependency. Strong references preserve
lifetime, not write safety or stream ordering.

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
builder never silently moves, casts, detaches, calls `.cpu()` or `.numpy()`,
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

Public `simulate_readout(...)` preflights the complete effective request before
its first random draw or tensor write. It validates recognized unique product
classes, required configs, source sampling agreement, axes, shape, dtype,
device, selected floating dtype, seed requirements, representability, and all
operation relationships needed by the closure.

Private product and scientific functions are internal independently testable
units. They may trust values that passed public preflight and do not need to
repeat the whole boundary or defend direct unsupported calls. Cheap
correctness-critical local assertions remain acceptable when they protect the
function's own valid-result contract.

Malformed supported operands use TensorCore's documented `TypeError` boundary;
unsatisfied well-formed relationships use `ValueError` where applicable.
TensorDSLab does not promise error behavior for deliberate public-API drift.

## Persistence And Integration

Exact Python classes are process-local semantic identities, not durable
product labels. Persistence is deferred and will require separately versioned
labels plus the scientific configuration necessary to interpret a payload.
The retired `DigitizedWaveformSpec` sidecar is not carried into the collection;
durable digitization-config association remains an explicit Design gate.

A future TensorG4DS bridge constructs dense truth `Photoelectrons` rather than
casting an upstream object into a downstream leaf. A future TensorML boundary
selects and orders exact products explicitly rather than relying on collection
iteration or subclass identity as a model ABI.

## Return To Design Before

Return to TensorDSLab Design before:

- changing the TensorCore version or using a non-root import;
- adding stored state or another inheritance layer to semantic leaves;
- restoring IDs, layouts, constants, sidecars, or compatibility shims;
- adding another axis or product type;
- adding generic movement, selection, reconstruction, or metadata locally;
- introducing a public destination/workspace/lifecycle surface;
- weakening fresh-result, no-post-exposure-write, device, dtype, or axes
  contracts;
- making a custom-tensor, cross-device, compatibility, persistence, or
  allocation-free claim; or
- changing the TensorG4DS or TensorML boundary.
