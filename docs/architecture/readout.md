# Post-Binned Readout Architecture

Status: active Design contract for the first TensorDSLab MVP. Stage 2's
structural package-and-collection foundation is Merged / Closed on `main` at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. Scientific transform execution
remains undispatched. Maintenance 1 changes only public-name and module
ownership. Its feature-branch form is candidate evidence before Review's clean
fast-forward; if the updated surface is read on `main`, that merge gate has
completed.

## Purpose

This page defines the in-memory, tensor-native post-binned readout collection,
field, and transform contract. It promotes reviewed scientific behavior from
DSLab and IV-DSLab without inheriting their package layouts, fixed ranks,
global state, CPU-list paths, cache machinery, or orchestration surfaces.

[IV-DSLab Parity And Intentional Divergences](../parity.md) classifies donor
comparisons and records the evidence, assumptions, exclusions, and accepted
differences behind this target contract.

The first readout boundary is already-binned integer photoelectrons. Native
G4DS parsing and TensorG4DS low-level analysis are upstream
responsibilities, not later readout features. The typed TensorG4DS handoff,
provenance/channel mapping, event placement, detector-window construction, and
photoelectron binning are later TensorDSLab integration stages.

## Scope

The target field flow is:

```text
readout.photoelectrons
  -> timing jitter
  -> readout.photoelectrons
  -> SiPM charge response
       -> primary avalanche counts plus dark counts
       -> frozen source snapshot
            -> crosstalk contribution
            -> afterpulse contribution
       -> aggregate charge smearing
  -> readout.charge
  -> readout.waveform.pure

ReadoutCollection layout/sample grid
  -> readout.waveform.noise

readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog
  -> optional readout.waveform.digitized
```

The flow describes field dependencies and scientific order. It is not a DAG,
campaign, cache, or scheduling contract.

## Physical Interpretation And Reference Plane

The product names follow the simulated photosensor-to-DAQ signal path, but
they do not claim a one-to-one tensor for every hardware component.

DarkSide-20k SiPMs are arrays of SPAD microcells operated above breakdown. A
photon absorption initiates a charge avalanche; there is no physically stored
population of free photoelectrons waiting for a later collective avalanche.
Dark noise originates from thermally or field-generated carriers, while
crosstalk and afterpulsing are additional correlated avalanches. DarkSide
therefore also uses PE as a response-equivalent charge unit. These distinctions
are described in the collaboration's
[SiPM production paper](https://link.springer.com/article/10.1140/epjc/s10052-025-14196-9)
and
[Tile production paper](https://link.springer.com/article/10.1140/epjc/s10052-025-14940-1).

TensorDSLab uses `readout.photoelectrons` for the binned, photon-origin primary
PE seeds supplied to the readout simulation. The name is a simulation-boundary
concept, not a claim that the tensor contains pre-avalanche free carriers.
Timing jitter may redistribute those seeds. Dark counts, crosstalk, and
afterpulses do not replace that field; they live in an ephemeral avalanche-count
workspace inside `simulate_charge(...)`, which produces `readout.charge`.

`readout.charge` is the aggregate floating PE-equivalent SiPM response per
readout channel and sample. It is not yet charge in coulombs and it is not
resolved by individual SPAD microcell or SiPM. DarkSide Tiles sum the currents
from 24 SiPMs and expose an amplified signal, while PDUs and channel mapping
introduce further aggregation. The exact TensorDSLab channel map and an exact
Tile/PDU transfer function remain deferred; see the collaboration's
[cryogenic photosensor overview](https://arxiv.org/abs/2502.09558).

The pure and noise waveforms are signal-only and noise-only simulation
components at one shared analog reference plane, not sequential hardware
outputs. Their composed `readout.waveform.analog` value is the modeled voltage
presented to the digitization transform. `readout.waveform.digitized` is the
direct ADC-code result. The narrower term *digitized* is retained because the
DarkSide DAQ distinguishes digitized waveforms from later digital processing;
see the
[DAQ architecture paper](https://arxiv.org/abs/2502.15651).

## Non-Goals

- No native G4DS parsing or TensorG4DS low-level analysis.
- No TensorG4DS dependency or handoff adapter in the post-binned readout
  package.
- No detector PE-hit product implementation.
- No photoelectron binning from sparse hits.
- No durable cache, manifest, compaction, or compatibility format.
- No condition-database or detector-channel-map runtime dependency.
- No DAG operation specs, recipes, executable doors, or campaign policy.
- No downstream TensorML adapter or model architecture.
- No reconstruction, trigger, ZLE, hit finding, or baseline-subtracted analysis
  product.
- No fixed global tensor rank, singleton batch axis, or channel/sample axis
  order.
- No NumPy, Python-list, Numba, pyFFTW, or remote-data hot path.

## Collection And Field Surface

The primary in-memory type is one concrete
`ReadoutCollection(TensorCollection)`. It is a structurally immutable,
partially materialized readout snapshot, not a target-only result wrapper.
Collection topology and `TensorField` records are immutable. Although PyTorch
tensor storage remains mutable, callers and transforms treat every materialized
collection field as read-only. An atomic transform may write only its fresh
target in an explicitly supplied destination; the full-chain builder may write
only the prepared produced-field set in its exclusively borrowed destination.
Manual in-place edits bypass descendant invalidation and are outside this
value-object contract.

The recognized fields, in canonical order, are:

| Role | Exact `TensorFieldId` | Value domain | Durable producer label |
| --- | --- | --- | --- |
| photoelectrons | `TensorFieldId("readout.photoelectrons")` | finite nonnegative integer photon-origin PE counts | `readout.photoelectrons` |
| charge response | `TensorFieldId("readout.charge")` | finite nonnegative floating PE-equivalent response amplitudes | `readout.charge` |
| pure waveform | `TensorFieldId("readout.waveform.pure")` | finite floating signal-only mV values | `readout.waveform.pure` |
| noise waveform | `TensorFieldId("readout.waveform.noise")` | finite floating baseline/noise-only mV values | `readout.waveform.noise` |
| analog waveform | `TensorFieldId("readout.waveform.analog")` | finite floating composed `pure + noise` mV values | `readout.waveform.analog` |
| digitized waveform | `TensorFieldId("readout.waveform.digitized")` | nonnegative integer ADC counts | `readout.waveform.digitized` |

The exact readout field-ID constants, their canonical registry, and the three
canonical readout axis-ID constants are owned and publicly exported by
`tensor_dslab.readout`, with their definitions housed in
`tensor_dslab.readout.ids`. Shared coordinate types are a different boundary:
`tensor_dslab.common` owns and exports `ExampleId` and `ChannelId`, and future
reconstruction products reuse the same `ChannelId` values. `common`, `readout`,
and future scientific packages are direct subpackages of the package root;
there is no intermediate domain namespace.

Every collection contains any nonempty subset of these fields. Its explicit
field order is the canonical table order filtered to the fields that are
present. Unknown field IDs, an empty field set, and another ordering are
invalid. The field mapping cannot represent duplicate keys; duplicate-ID
validation belongs at ordered-ID and selection boundaries.

The field ID gives a tensor its in-memory semantic role. A durable producer
label names a produced artifact or cache entry. Those remain separate
namespaces even where their string spellings currently coincide.

The distinction between photoelectrons and charge response remains scientific,
not class-based. The first is a binned integer primary-seed product. The second
includes the modeled SiPM avalanche effects and floating response variation.
Analog voltage values and digitized ADC values likewise occupy distinct
recognized fields even though they share a collection layout.

There is no generic TensorDSLab `Product` base. Readout physics remains in free
functions over `ReadoutCollection`.
The stable `ReadoutCollection` value type lives in
`tensor_dslab.readout.types`; `tensor_dslab.readout.tensors` retains only the
semantic reconstruction, projection, selection, and movement helpers.

## Canonical Axes And Semantic Sidecars

The three core readout axes have exact public identities. Their positions are
resolved from each field's `TensorLayout`; tensor dimension order remains
arbitrary.

Current public value sketch:

```python
class AdcQuantization(StrEnum):
    TRUNCATE = "truncate"


EXAMPLE_AXIS_ID = TensorAxisId("example")
CHANNEL_AXIS_ID = TensorAxisId("channel")
SAMPLE_AXIS_ID = TensorAxisId("sample")
REQUIRED_AXIS_IDS = IdSequence(
    (EXAMPLE_AXIS_ID, CHANNEL_AXIS_ID, SAMPLE_AXIS_ID)
)


@dataclass(frozen=True, slots=True)
class SampleGrid:
    sample_period_ns: PositiveFloat
    origin_ns: FiniteFloat
    sample_offset: NonnegativeInteger


@dataclass(frozen=True, slots=True)
class DigitizedWaveformSpec:
    bit_depth: PositiveInteger
    voltage_pp_mv: PositiveFloat
    voltage_offset_mv: FiniteFloat
    analog_gain_db: FiniteFloat
    quantization: AdcQuantization
```

The axis constants are TensorCore values. Code compares them with `==`, not
Python object identity; a freshly constructed `TensorAxisId("sample")`
resolves the same semantic axis.

The focused Stage 2 work order supplies the exact constructor and export
spelling. The durable meanings are:

- every layout contains the exact `example`, `channel`, and `sample` axis IDs
  above exactly once, in any order;
- the example and channel axes are ID-backed by exact `ExampleId` and
  `ChannelId` coordinates, respectively;
- the sample axis is count-only;
- sample count is derived from the TensorCore layout;
- sample period and origin are typed physical values, not IDs or metadata-only
  strings;
- sample offset is the zero-based ordinal of the first stored sample in its
  containing readout grid; it is typed grid metadata, not a TensorCore
  coordinate or durable row identity;
- extra axes are allowed when an operation can preserve them; because every
  field has the same exact layout, every accepted extra axis is common to all
  present fields;
- additional ID-backed shared axes participate in stochastic keys after
  example and channel, ordered lexically by `axis_id.value`, independent of
  tensor layout order;
- the collection carries no configurable stochastic-axis membership list;
- additional count-only axes are valid structural collection axes, but a
  stochastic transform rejects them until Design accepts a typed containing-
  grid offset contract for each such axis;
- `DigitizedWaveformSpec` records the transfer facts required to interpret and
  validate ADC counts; `adc_min` is zero and `adc_max` is derived as
  `2**bit_depth - 1`;
- accepted bit depth is from 1 through 16 inclusive, and
  `analog_gain_db` is from 0 through 40 dB inclusive. The gain bound corrects
  the donor's impossible out-of-range conjunction intentionally;
- the digitized spec is present exactly when
  `readout.waveform.digitized` is present. Projecting or invalidating that
  field drops the spec, while a digitized-only projection retains it;
- the accepted MVP quantization member is `AdcQuantization.TRUNCATE`, applied
  after gain and clipping map the value into a nonnegative interval.

TensorCore metadata may carry small descriptive provenance. It must not be the
only location of sample-grid values, field units, or RNG coordinate policy.

## Semantic Layout And Warmed Readout Profile

Arbitrary axis order is a semantic feature and is not itself a ping-pong
problem. TensorLayout is the sole axis-order source of truth; swapping two
compatible buffers swaps references and never permutes tensor values. General
`ReadoutCollection` construction therefore continues to accept arbitrary
required-axis order and noncontiguous `torch.strided` read-only payloads. It
does not add an axis-order/stride sidecar or execution-ready subclass, and it
does not reject expanded or internally overlapping read-only sources merely
for their storage arrangement.

The warmed `out + workspace` MVP profile is intentionally stricter:

- `SAMPLE_AXIS_ID` is last;
- every source, generated public target, and private scratch tensor
  participating in the configured call is contiguous;
- writable targets and scratch are internally nonoverlapping and
  storage-disjoint; and
- ordered axes/sizes, shape, device, role dtypes, algorithms, destination
  schema, stream, and exclusive workspace lease match exactly.

Readout kernels primarily operate along time: timing redistribution, delayed
afterpulses, pulse convolution, FFT noise, composition, clipping, and
digitization. Contiguous `(..., sample)` storage makes each waveform a single
consecutive block, allows a no-copy `view(-1, sample_count)` after strict
preflight, and aligns naturally with length-last convolution/FFT and
neighboring-sample GPU access. The arbitrary leading-axis order remains
semantic; each order defines a different workspace signature.

Warmed preflight completes before RNG consumption or writes. It rejects rather
than permuting, calling `.contiguous()`, cloning, casting, moving, performing a
reshape-copy, or allocating fallback storage. Contiguous strides derive from
ordered shape, so the MVP signature does not need arbitrary stride tuples.
Functional execution may explicitly allocate normalization while preserving
accepted autograd; ordinary destination reuse may use documented allocating
scratch/normalization and claims no allocation freedom. Non-ready inputs are
explicitly prepared once outside the repeated loop. Any future
allocation-free noncontiguous profile requires a focused measured contract.

## Construction Invariants

Every `ReadoutCollection` constructor validates:

- the exact semantic collection class and TensorCore primitive types;
- a nonempty subset of the six exact recognized field IDs;
- canonical filtered field order;
- one exact ordered `TensorLayout` shared by every present field;
- `shared_axes` equal to every common-layout axis ID in layout order;
- one common device shared by every present tensor;
- `torch.strided` layout for every present tensor, without requiring
  contiguous strides;
- exact `EXAMPLE_AXIS_ID`, `CHANNEL_AXIS_ID`, and
  `SAMPLE_AXIS_ID` axes in that shared layout, in arbitrary order;
- exact `ExampleId` and `ChannelId` coordinate classes and uniqueness;
- sample-axis size and count-only semantics;
- field-role-specific dtype, unit, representation, and value domain;
- one common floating dtype across every present `readout.charge`,
  `readout.waveform.pure`, `readout.waveform.noise`, and
  `readout.waveform.analog` field, selected from `torch.float32` and
  `torch.float64`;
- exact `torch.int64` dtype for photoelectrons and exact `torch.int32` dtype
  for digitized ADC counts;
- finite sample-grid values, positive sample period, and nonnegative sample
  offset;
- a valid `DigitizedWaveformSpec` if and only if the digitized field is
  present, with bit depth, gain, quantization, transfer values, and ADC bounds
  validated as above;
- immutable semantic sidecars;
- allowed extra-axis policy.

Initial value-domain expectations are:

| Field ID | Value domain |
| --- | --- |
| `readout.photoelectrons` | finite, nonnegative photon-origin integer counts |
| `readout.charge` | finite, nonnegative floating PE-equivalent response amplitudes |
| `readout.waveform.pure`, `readout.waveform.noise`, `readout.waveform.analog` | finite floating mV values |
| `readout.waveform.digitized` | nonnegative integer ADC counts within configured range |

Collection construction is placement-neutral: it accepts tensors on any
PyTorch device when all fields share that exact device, but does not claim that
every later kernel supports every PyTorch device.
Stage 2 requires CPU coverage and conditionally exercises CUDA when available.
Dense noncontiguous tensors are accepted by the semantic constructor because
`torch.strided`, not contiguity, is the collection storage-layout contract;
the warmed profile is stricter as defined above. Mixed floating and integer domains
are valid, but all present floating roles share one of the two accepted
floating dtypes. When a transform adds the first floating field, its accepted
runtime dtype policy must select one explicitly; when any floating field is
already present, a new floating target must match it. Constructors must not
infer scientific meaning from dtype alone. A valid projection need not contain
a field's scientific dependencies.

## Snapshot Dependencies And Invalidation

The accepted dependency graph is:

```text
readout.photoelectrons -> readout.charge -> readout.waveform.pure
common layout/sample grid -> readout.waveform.noise
readout.waveform.pure + readout.waveform.noise
  -> readout.waveform.analog -> readout.waveform.digitized
```

A transform consumes one snapshot and produces a full result snapshot. It
retains unaffected source fields by structural sharing, adds or replaces its
target field, and removes descendants made stale by that addition or
replacement. The result preserves the exact shared layout and exactly equal
sample-grid sidecar. The invalidation table is exact:

| Added or replaced target | Remove these stale descendants if present |
| --- | --- |
| `readout.photoelectrons` | `readout.charge`, `readout.waveform.pure`, `readout.waveform.analog`, `readout.waveform.digitized` |
| `readout.charge` | `readout.waveform.pure`, `readout.waveform.analog`, `readout.waveform.digitized` |
| `readout.waveform.pure` | `readout.waveform.analog`, `readout.waveform.digitized` |
| `readout.waveform.noise` | `readout.waveform.analog`, `readout.waveform.digitized` |
| `readout.waveform.analog` | `readout.waveform.digitized` |
| `readout.waveform.digitized` | none |

One central TensorDSLab result builder owns this table, canonical field
ordering, retained-field structural sharing, and target insertion. Individual
transforms must not hand-code divergent result policies.

Addition is not an invalidation exception. For example, adding newly computed
`readout.charge` to a partial snapshot that already contains
`readout.waveform.analog` and `readout.waveform.digitized` removes both of
those reachable descendants even when no charge field was previously present.

This invalidation is a transform-result rule, not a constructor ancestry rule.
Explicit projection or field removal does not invalidate any retained
descendant. Such an operation returns another nonempty `ReadoutCollection`,
keeps canonical filtered order, the same layout and sample-grid sidecar,
and structurally shares every retained `TensorField` record. Field-specific
sidecars follow their retained fields. For example, a projected collection
containing only `readout.waveform.analog` is valid, but composing a new analog
waveform still requires both pure and noise fields in the input
collection. Presence records materialized state, not derivation history or
proof of parity.

Field-specific sidecars follow their fields. Any result that removes
`readout.waveform.digitized` also removes `DigitizedWaveformSpec`. A digitizing
transform adds or replaces both the digitized field and its spec atomically.
Projection preserves the spec only when it preserves the digitized field.

For a target-preserving transform such as timing jitter, an accepted
zero-effect config is an exact identity of the target field values, not an
exception to snapshot construction. It consumes no semantically relevant
random draws, but the returned collection still uses the ordinary full-result
schema and conservative descendant invalidation. `simulate_charge(...)`
crosses from integer photoelectrons to a floating charge-response field, so
disabling all optional avalanche effects and using zero smearing produces an
exact integer-to-float response conversion rather than returning the source
field. A later source-return optimization would require a focused contract for
both functional and `out=` paths.

## Public Transform Shape

The target operation surface uses free functions:

```python
def apply_timing_jitter(
    collection: ReadoutCollection,
    config,
    *,
    rng,
    out: ReadoutCollection | None = None,
    workspace=None,
) -> ReadoutCollection: ...

def simulate_charge(
    collection: ReadoutCollection,
    config,
    *,
    rng,
    out: ReadoutCollection | None = None,
    workspace=None,
) -> ReadoutCollection: ...

def render_pure_waveform(
    collection: ReadoutCollection,
    config,
    *,
    out: ReadoutCollection | None = None,
    workspace=None,
) -> ReadoutCollection: ...

def render_noise_waveform(
    collection: ReadoutCollection,
    config,
    *,
    rng=None,
    out: ReadoutCollection | None = None,
    workspace=None,
) -> ReadoutCollection: ...

def compose_analog_waveform(
    collection: ReadoutCollection,
    config=None,
    *,
    out: ReadoutCollection | None = None,
    workspace=None,
) -> ReadoutCollection: ...

def digitize_waveform(
    collection: ReadoutCollection,
    config,
    *,
    out: ReadoutCollection | None = None,
    workspace=None,
) -> ReadoutCollection: ...
```

This is a concrete API direction, not yet production spelling. An
implementation work order may refine parameter names or split a config only if
the architecture meaning remains unchanged.

`simulate_charge(...)` owns the complete photoelectron-to-charge response
boundary. Its typed config carries the dark-count, crosstalk, afterpulse, and
charge-smearing submodels; optional effects may be disabled explicitly. The
transform constructs private integer avalanche scratch, adds dark counts,
freezes the shared crosstalk/afterpulse source, adds each enabled contribution
exactly once, and converts the result into the floating `readout.charge`
target. When a compatible workspace is supplied, that scratch comes from its
exclusive lease; otherwise the transform may allocate it per call. Low-level
contribution samplers and intermediate avalanche tensors remain private.
Exposing avalanche counts as a recognized product would require a focused
Design decision rather than overloading
`readout.photoelectrons`.

Each transform requires and targets these recognized fields:

| Transform | Required input fields | Added or replaced target |
| --- | --- | --- |
| timing jitter | `readout.photoelectrons` | `readout.photoelectrons` |
| SiPM charge simulation | `readout.photoelectrons` | `readout.charge` |
| pure rendering | `readout.charge` | `readout.waveform.pure` |
| noise rendering | collection layout and sample grid | `readout.waveform.noise` |
| analog composition | `readout.waveform.pure`, `readout.waveform.noise` | `readout.waveform.analog` |
| digitization | `readout.waveform.analog` | `readout.waveform.digitized` |

Noise rendering uses the collection's shared layout and sample grid as its
rendering reference; it does not read another field's values. No transform
hides source loading, example assembly, device movement, cache IO, or DAG
behavior.

Every atomic transform performs all source, config, RNG, target, alias,
workspace, stream, and gradient validation before consuming random values or
writing its target. A preflight failure leaves source, destination, RNG state,
and workspace generations unchanged. After successful backend completion the
target is completely overwritten. The contract does not promise transactional
rollback after a launched CPU kernel fails partway or an asynchronous
accelerator fault occurs.

## Scientific Config Versus Runtime Control

Scientific configs contain physical/model choices:

- timing width and edge policy;
- dark-count rate;
- crosstalk model parameters;
- afterpulse fire/delay/recovery model;
- charge-smearing width and negative policy;
- pulse-template constants;
- noise model and spectrum values;
- analog clipping;
- digitizer voltage/gain/bit parameters.

Runtime controls are explicit call inputs:

- RNG seed and namespace;
- output destination;
- optional prepared scratch workspace or exclusive workspace lease;
- device movement or dtype conversion;
- execution/chunking policy.

Runtime controls must not be encoded as TensorCore IDs, coordinates, field IDs,
axis IDs, product labels, or hidden mutable config state.

The target RNG record is TensorDSLab-owned:

```python
@dataclass(frozen=True, slots=True)
class RngSpec:
    seed: int
    namespace: str
```

Operation role is part of the deterministic random-field domain. A caller
namespace separates independent simulation streams without changing physics
config.

## Readout Workspace

`ReadoutWorkspace` is the reusable scratch layer for the simulation hot path.
It is a mutable caller-owned runtime resource, not a `ReadoutCollection`,
TensorCore primitive, semantic sidecar, scientific config, ID, coordinate,
product label, cache record, or durable artifact. It owns no public output
field and retains no source or RNG state. No returned collection, field,
sidecar, or metadata may reference its storage.

Non-final preparation shape:

```python
workspace = ReadoutWorkspace.allocate(
    source,
    config,
    stream=stream,
)
```

Preparation derives one fixed execution signature containing:

- exact ordered axis IDs and sizes; axis positions are derived from that order
  and the three canonical readout axis IDs rather than stored as a second role
  mapping;
- exact tensor shape, device, and accelerator index;
- photoelectron-count, common-floating, digitized, and derived complex dtypes;
- enabled algorithm families and each option that changes scratch geometry,
  such as FFT size, pulse support, or delayed-effect bucket capacity;
- exact configured destination schema; and
- one synchronous CPU execution domain or one exact CUDA stream.

Warmed compatibility additionally requires sample-last order and contiguous
participating source, generated output, and scratch tensors. Writable tensors
are internally nonoverlapping and storage-disjoint. The signature stores no
arbitrary stride tuple because contiguous strides derive from ordered shape.

Coordinate values may vary only where the accepted implementation proves that
scratch is positional and current-source coordinates still provide every RNG
key; semantic layout and coordinate-map compatibility remain exact at public
collection boundaries. Numeric scientific parameters that do not change
scratch geometry may reuse a workspace. The exact signature comparison belongs
to a typed runtime record, not ad hoc shape checks.

The MVP workspace is non-resizable, non-reentrant, and exclusive to one
in-flight call. A supplied mismatch fails before RNG consumption or output
writes. The builder does not silently allocate, grow, shrink, move, cast,
rebind, or replace it. There is no process-global, thread-local, device-global,
or hidden LRU workspace cache. Same-stream sequential CUDA reuse is ordered by
that stream; nested, concurrent, or different-stream reuse is rejected. Use a
separate workspace for every concurrent worker or stream. Event-mediated
cross-stream leasing is deferred.

Workspace scratch may use uninitialized storage because it never appears in a
valid public collection. Every operation declares scratch read/write liveness
and completely writes a slot generation before any read. Failed execution may
poison affected generations until explicit reset or workspace replacement; the
first work order must define that state transition without claiming recovery
from an invalid CUDA context.

The logical private charge inventory includes at least:

```text
count_source        # primary avalanches plus dark counts; frozen for fan-out
count_total         # source plus completed secondary contributions
contribution        # reusable crosstalk/afterpulse contribution slot
operation scratch   # probability, index, FFT, convolution, or RNG-specific
```

Crosstalk and afterpulse contributions may reuse `contribution` sequentially
only after each has been accumulated into `count_total`; both must read the
unchanged `count_source`. This is not a promise that two physical ping-pong
buffers suffice. Kernel fusion may reduce storage, while timing,
multinomial/delay, convolution, FFT, or RNG kernels may require more. The exact
physical inventory is an implementation work-order decision proven by slot
liveness tests.

Any scratch-slot reuse is limited to one compatible storage class: exact shape,
ordered axes, device, dtype, contiguous tensor layout, and algorithm capacity.
Ping-pong within such a class swaps references only; it never permutes or
copies values, and a different axis order requires a different workspace. The
integer-count to floating-charge boundary, floating-waveform to integer-ADC
boundary, and simultaneously materialized public fields always use distinct
storage.

## Full-Chain Readout Builder

`build_readout_collection(...)` is the ordinary local domain composition
surface above atomic transforms. It owns the accepted operation order,
stage-specific result schemas, scratch-slot assignment, public target
selection, descendant replacement, and final collection assembly. It does not
load a source, move or cast tensors, perform cache IO, choose campaign shards,
schedule retries, or own any Projects/dag behavior.

Non-final public sketch:

```python
destination = build_readout_output_buffer(
    source,
    floating_dtype=floating_dtype,
    replace_photoelectrons=timing_enabled,
    digitized_waveform_spec=digitized_waveform_spec,
)
workspace = ReadoutWorkspace.allocate(source, config, stream=stream)

result = build_readout_collection(
    source,
    config,
    rng=rng,
    out=destination,
    workspace=workspace,
)
```

The first builder requires `readout.photoelectrons` and executes the complete
configured chain in accepted order:

```text
optional timing jitter
  -> charge simulation
  -> pure rendering + noise rendering
  -> analog composition
  -> optional digitization
```

Its final canonical field set is photoelectrons, charge, pure, noise, and
analog, plus digitized exactly when digitization is configured. A general
partial-output execution plan is deferred. When jitter is disabled, the
photoelectron field may be the exact source record; when enabled it is a fresh
public target. Every other produced field has distinct public storage, and
pre-existing derived source fields are recomputed rather than trusted.

The full destination is a caller-owned, valid `ReadoutCollection` prepared
before execution. Its factory zero-initializes newly allocated fields in
contiguous storage using the source's existing semantic axis order; it neither
reorders nor materializes retained fields. A full destination is warmed-ready
only when sample is already last and every participating retained tensor is
also contiguous. A reused destination may contain its prior valid result
because the builder completely overwrites every produced field. The builder
makes stage-specific exact collection views over its field records so every
atomic transform still writes one target and sees exact retained records. It schedules
the last compatible write directly into each public target and assembles the
final collection without cloning tensors. Public fields that coexist in the
result never ping-pong through one payload; automatic reuse applies only to
private scratch after its final reader.

Builder modes are:

```text
out=None, workspace=None
  -> allocating functional composition
  -> explicit order/stride normalization may allocate
  -> accepted deterministic autograd behavior is preserved

out supplied, workspace=None
  -> caller-owned public destinations are reused
  -> documented scratch or normalization allocation remains allowed

out supplied, compatible workspace supplied, warmed signature
  -> caller-owned public destinations and named scratch are reused
  -> steady-state TensorDSLab-managed tensor-storage allocation-free path
```

Workspace-without-`out` is invalid. Supplying `out` selects the non-autograd
simulation path and must reject gradient-sensitive use before any mutation.
The builder returns the exact supplied destination. That collection remains
stable until its caller explicitly resubmits it as writable `out`; reuse
authorizes overwrite and ends
the prior result's stable snapshot lifetime. Overlapped consumers therefore
need caller-managed output banks. A reusable destination is valid only for its
exact semantic layout, coordinate maps, sidecars, device, and dtypes. Reusing
raw payload storage across batches with different coordinates would require a
future leased output-pool/reconstruction contract. A workspace never serves as
such a bank.

The allocation-free phrase excludes lightweight Python records and tensor
views, allocator bookkeeping, first-use warm-up, and opaque PyTorch/CUDA
library plan or scratch behavior. Absolute allocator-free execution, growable
workspaces, compiled immutable plans, leased output pools, cross-stream events,
partial-output plans, and CUDA Graph capture require later focused decisions.

After warmed preflight, no implicit permutation, `.contiguous()`, clone, cast,
movement, copying reshape, or fallback allocation is permitted. Preparing a
non-ready source is an explicit one-time materialization outside the repeated
builder loop.

## Fixed-Grid Charge-Response Order

The first-MVP response order is accepted as:

```text
photoelectrons = already-binned photon-origin integer counts
jittered_photoelectrons = timing_jitter(photoelectrons)

simulate_charge(jittered_photoelectrons):
primary_avalanches = integer_copy(jittered_photoelectrons)
post_dark = primary_avalanches + dark_counts(layout, sample_grid)
snapshot = frozen(post_dark)
crosstalk_addition = crosstalk_contribution(snapshot)
afterpulse_addition = afterpulse_contribution(snapshot)
final_counts = snapshot + crosstalk_addition + afterpulse_addition
charge = smear(final_counts)
```

Crosstalk and afterpulses read the same snapshot. Neither contribution feeds
the other, and generated counts do not recursively generate more counts in the
first fixed-grid model. The public `simulate_charge(...)` transform owns this
composition so callers cannot accidentally serialize the two effects or
publish an intermediate avalanche grid as photoelectrons. An implementation
may fuse tensor kernels only when it preserves all externally observable
behavior required by this TensorDSLab contract.

This order intentionally differs from IV-DSLab's sparse-PE execution order and
recursive queue. The fixed-grid order is selected for bounded output shape,
reviewable behavior, and GPU execution.

## Timing Jitter

Timing jitter redistributes photon-origin integer PE counts across the sample
axis. It does not create a sparse post-readout PE table or add sensor-origin
avalanches.

For source sample `s`, target sample `t`, shift `k = t - s`, sample period `T`,
latent source-bin phase `U ~ Uniform(0, T)`, and
`J ~ Normal(0, sigma_ns)`:

```text
p(target=t) = P(k*T <= U + J < (k + 1)*T)
```

The latent phase matters because the already-binned input no longer carries
sub-bin PE time. The output is an aggregate categorical/multinomial
redistribution, not a nearest-bin shift from a bin center.

Accepted policies:

- `sigma_ns == 0` is exact target-field identity and consumes no random draws;
- counts shifted outside the sampled window are dropped;
- count conservation holds only after including the explicit dropped bucket;
- no stale `nearest` policy field appears in the config;
- the target remains `readout.photoelectrons`.

## Dark Counts

Dark counts add independent integer counts per eligible channel/sample cell:

```text
lambda_per_cell = rate_hz * sample_period_ns * 1e-9
contribution ~ Poisson(lambda_per_cell)
```

The first config uses one global nonnegative rate. Typed per-channel variation
is deferred. A zero rate contributes exact zeros and consumes no random draws.
Dark-count avalanches exist only in the private charge-response workspace; they
are never written back into `readout.photoelectrons`.

## Crosstalk

The first fixed-grid crosstalk model is bounded, first-generation,
same-channel, and same-sample. For source count `n`:

```text
contribution ~ Poisson(mean_additional_counts_per_source * n)
```

The config should name this value as a Poisson mean coefficient rather than
mislabeling it as a Bernoulli probability. Cross-channel coupling and recursive
branching require a later scientific model and fixture.

## Afterpulses

The fixed-grid delay model uses a fire probability and an exponential delay.
For source sample `s`, target sample `t >= s`, `k = t - s`, latent phase
`U ~ Uniform(0, T)`, and `D ~ Exponential(mean_delay_ns)`:

```text
p(target=t) = p_fire * P(k*T <= U + D < (k + 1)*T)
```

The aggregate buckets are:

- no afterpulse with probability `1 - p_fire`;
- every in-range delayed target sample;
- an explicit dropped-out-of-range delayed bucket.

Delayed counts outside the sampled window are dropped. The first model is
same-channel and first-generation. Crosstalk and afterpulse counts are private
response intermediates rather than materialized collection fields.

The audited IV source literally uses a reciprocal-exponential delay expression,
not an ordinary exponential with mean `ap_tau`. TensorDSLab intentionally uses
the standard mean-delay model above as a reviewed scientific correction. This
is an intentional divergence, not exact or distributional parity with the
literal IV delay implementation.

One scientific choice remains open before afterpulse implementation: IV-DSLab
weights an afterpulse by the recovery factor
`1 - exp(-delay / recovery_tau)`, while the later DSLab fixed-grid path emits a
unit count. Recovery weighting would cross from integer existence into
fractional amplitude and may change its relationship to charge smearing. The
first implementation work order must either:

- accept unit-count afterpulses explicitly; or
- define a typed recovery-amplitude stage and its ordering.

Do not silently drop the recovery model while claiming exact parity with
IV-DSLab afterpulse amplitudes.

## Charge Smearing

Charge smearing is the final internal boundary from aggregate avalanche counts
to floating PE-equivalent response amplitudes. For populated count `n`:

```text
draw = Normal(mean=n, sigma=sqrt(n) * sigma)
amplitude = max(draw, 0)
```

Accepted policies:

- one aggregate draw per populated cell;
- zero counts remain exactly zero;
- zero sigma is an exact integer-to-float conversion;
- negative draws use `clip_to_zero`;
- smearing occurs after all accepted integer existence effects;
- the enclosing `simulate_charge(...)` target is `readout.charge`; no
  intermediate count target is added to the collection.

Before clipping, the aggregate draw has the same probability distribution as
summing equal independent unit-charge Gaussian variations. It does not
reproduce heterogeneous per-quantum or afterpulse-recovery weights.

## Pure Waveform

The first physical rendering path is causal pulse-template convolution, not the
historical gain-only placeholder.

The reviewed FEB-SNR-style template is:

```text
h(t) = exp(-t / (tau_fall + tau_rise)) - exp(-t / tau_rise)
```

It is sampled on the collection sample grid for a finite configured support and
normalized by its sampled maximum to the configured amplitude. Rendering uses
full causal convolution truncated to the input sample count, followed by
configured gain and sign. Baseline belongs to `readout.waveform.noise`; adding
it here would violate the signal-only `readout.waveform.pure` field contract
and could count it twice during analog composition.

Accepted policies:

- output length and all non-sample axes match the input;
- no implicit padding or persistent prebuffer field;
- no eventwise sub-bin amplitude correction is claimed because the input does
  not carry sub-bin PE time;
- no latent-phase-marginalized amplitude correction is applied in the first
  MVP; return to Design if parity validation finds unacceptable peak or area
  bias;
- `out=None` remains differentiable with respect to the floating input charge;
- the transform stays on the input device.

Tensor-valued differentiable pulse parameters are not part of the first config
contract and would require a focused public-surface decision.

A delta/gain-only mode may exist as an explicit test or simplified model. It
does not establish numerical or exact parity with the reviewed donor pulse
renderer.

## Noise Waveform

The architecture recognizes three explicit noise models:

1. constant baseline;
2. Gaussian white noise;
3. direct one-sided FFT noise.

Constant baseline is deterministic and requires no RNG.

White noise uses zero-mean Gaussian samples with a typed RMS or an explicit
`pe_amplitude_mv / snr` derivation. Each sample is addressed by semantic
coordinates plus sample index; it must not use one sequential per-channel
stream whose values change when the sample range is sliced or chunked.

Direct FFT noise uses:

```text
one-sided spectrum matching sample_count
  -> deterministic semantic phases
  -> torch.fft.irfft(..., n=sample_count)
  -> explicit normalization
  -> explicit scale
```

Accepted FFT policies:

- spectrum length is exactly `sample_count // 2 + 1`;
- frequency metadata matches the sample count and period;
- DC is real and either forced to zero or rejected when nonzero;
- even-length Nyquist is real;
- interior bins receive coordinate-addressed phases;
- no quiet crop, pad, truncate, resample, long baseline bank, or random crop;
- no remote spectrum download or persistent precomputed bank;
- normalization and units are explicit config values.

Power-spectral-density rebinning is a possible later typed input boundary. It
must conserve integrated target-bin power and must not become implicit file
loading inside waveform rendering.

## Analog Waveform

Analog composition is:

```text
analog = pure + noise
analog = optional_analog_clip(analog)
```

Both required fields come from the same `ReadoutCollection`. Construction
already guarantees their matching canonical axes, sample grid, exact layout,
shape, device, `torch.strided` layout, common floating dtype, and mV
representation. No implicit broadcast across missing axes is accepted for the
first implementation.

`out=None` is differentiable with respect to the pure and noise field tensors.
Analog clipping uses the ordinary PyTorch gradient behavior of the selected
operation. The result is the modeled readout-channel voltage at the declared
input reference plane of `digitize_waveform(...)`, before the digitizer-specific
gain recorded by `DigitizedWaveformSpec`; it is not claimed to be a separate
waveform emitted independently by every Tile and PDU stage.

## Digitization

Digitization is a separate transform and field role:

```text
gained_mv = waveform_mv * 10 ** (analog_gain_db / 20)
vmin_mv = voltage_offset_mv - voltage_pp_mv / 2
vmax_mv = vmin_mv + voltage_pp_mv
clipped_mv = clamp(gained_mv, vmin_mv, vmax_mv)
adc = quantize((clipped_mv - vmin_mv) / voltage_pp_mv * (2**bits - 1))
```

The config states the quantization rule explicitly. The accepted MVP policy is
`AdcQuantization.TRUNCATE`, matching the donor for values mapped inside the
nonnegative ADC interval.

Accepted policies:

- analog composition and clipping happen before digitization;
- pure and noise components are never digitized separately and then added;
- output shape/layout/canonical axes/sample grid are preserved;
- the target is `readout.waveform.digitized` with ADC-count semantics;
- bit depth, voltage range, offset, gain, quantization, output dtype, and ADC
  bounds are validated at the transform boundary; the stable transfer facts
  are retained as `DigitizedWaveformSpec` and revalidated at collection
  construction;
- bit depth is in `[1, 16]`, `analog_gain_db` is in `[0, 40]`, the output dtype
  is `torch.int32`, and out-of-range analog values clamp before conversion;
- digitization is not declared differentiable.

## Deterministic Random Fields

Stochastic transforms are coordinate-addressed. Their result must not depend on
mapping iteration order, channel reordering, unrelated batch members, or how an
ID-backed batch is chunked.

There is no per-collection stochastic-axis declaration. The canonical axis
contract determines the key payload.

RNG keys include:

- seed and caller namespace;
- operation role;
- the `EXAMPLE_AXIS_ID` and its `ExampleId` coordinate;
- the `CHANNEL_AXIS_ID` and its `ChannelId` coordinate, never a
  channel index;
- every other ID-backed shared axis paired with its coordinate, ordered
  lexically by `axis_id.value` rather than tensor layout order;
- `sample_offset + local_sample_index` for the count-only sample axis;
- an operation-local draw/counter index when more than one draw is required.

This is the one canonical coordinate payload order: example, channel, lexical
additional ID-backed axes, then the containing-grid sample ordinal. A
count-only extra axis remains valid structural layout, but a stochastic
transform rejects it until that axis has an accepted typed offset contract;
using its transient local position would break chunking invariance.
Lexical ordering applies only to the labeled extra-axis key entries; it never
sorts tensor dimensions or any axis's coordinate `IdSequence`.

Selecting or chunking a contiguous sample range must advance both physical
origin and sample offset consistently. The offset exists to preserve random
field identity across sample-axis chunks; it does not turn the count-only
sample axis into an ID-backed axis.

Do not treat a per-quantum loop index as durable identity. Aggregate count
distributions and fixed-shape random fields are preferred over ragged quantum
axes.

Required reproducibility levels:

- exact repeatability for the same inputs/config/RNG spec on the same supported
  backend;
- invariance under accepted axis reordering, ID-backed batching, and chunking;
- cross-backend distributional agreement with accepted probability kernels, with
  finite-sample statistical validation as evidence;
- no cross-device bitwise guarantee until a work order accepts one RNG
  algorithm capable of providing it.

Production GPU paths must not round-trip random fields through Python lists or
NumPy. A CPU reference may exist for fixture generation only when the work
order names it explicitly.

## Atomic Output Buffer And Aliasing Contract

All transforms use the architecture-wide call convention:

```text
out=None:
  derive the exact full-result field schema
  allocate only a new target TensorField and collection record
  structurally share every unaffected retained source TensorField
  leave all source records and tensors unchanged
  allow explicit allocating order/stride normalization

out=destination:
  require the exact full-result ReadoutCollection schema
  require retained fields to be the exact source TensorField records
  write only the target field tensor
  return that exact destination
  allow documented per-call scratch or normalization allocation when no workspace is supplied

out=destination, workspace=compatible exclusive lease:
  preserve the same exact destination and one-target rules
  obtain every declared TensorDSLab scratch slot from the workspace
  perform no TensorDSLab-managed target or named-scratch tensor allocation
```

For target field `T`, the exact result schema is the source field set, minus
the stale descendants named for `T`, with `T` added or replaced, then ordered
by the canonical field registry. This full schema applies equally to
functional and `out=` execution. A transform never returns only its target
field and never quietly retains a stale descendant.

For `out=destination`, validation additionally requires:

- exact `ReadoutCollection` type, canonical fields and order, shared layout,
  device, `torch.strided` layout, and sample-grid values equal to the
  source;
- the exact expected field-specific sidecar state: a retained digitized field
  shares its source `DigitizedWaveformSpec`, an invalidated digitized field has
  no spec, and a new/replaced digitized target carries the spec derived from
  its validated config;
- the expected role-specific integer or common floating target dtype;
- every unaffected retained field to be the identical `TensorField` record
  from the source collection, not a copied or merely equal record;
- the target tensor to be internally nonoverlapping and its field/tensor not to
  alias any source field, other live output, or workspace storage;
- only the target tensor to be writable by the transform.

All of those checks, plus config, RNG, workspace-signature, lease, stream, and
gradient checks, complete before RNG consumption or the first target write. A
preflight failure leaves the destination bitwise unchanged. Successful
execution overwrites every target element. This is not a transactional rollback
contract for faults after backend execution begins.

Public `build_readout_result_buffer(...)` must construct that exact
field-scoped destination with structurally shared retained fields and one
zero-initialized contiguous target field of the required role, shape, dtype,
device, and existing semantic axis order. It must not reorder or materialize
retained fields, and it must not use an uninitialized `torch.empty_like`
tensor or allocate an all-field
collection merely to produce one target: every public `ReadoutCollection` is
valid from construction onward, and arbitrary uninitialized storage may
violate finite, nonnegative, or ADC-bound invariants. The transform overwrites
the complete target tensor before returning the destination; partial writes are
invalid.

The first implementation allows no target/source aliasing. Timing
redistribution, convolution, the shared crosstalk/afterpulse snapshot, dtype
transitions, and digitization all make blanket in-place behavior unsafe.
Structural sharing of retained structurally immutable field records is
required and is not treated as target aliasing. Source tensor storage is
read-only to the transform, not intrinsically immutable.

Changing the target/source alias rule requires Design acceptance and a behavior
test. `out=` alone controls only the public destination and does not promise
zero internal scratch allocation. The prepared hot-path claim additionally
requires a compatible exclusive workspace lease. Workspace storage must be
disjoint from every source, retained, target, and public-output storage range;
a returned collection never borrows it.

Functional deterministic paths preserve the documented PyTorch autograd
behavior through the newly computed target while structurally sharing retained
fields. The `out=` path carries no autograd guarantee. Gradient-sensitive
callers must use `out=None`; `out=` should reject gradient-sensitive use rather
than silently detach or imply differentiability.

## Device Execution

`ReadoutCollection` construction is placement-neutral and accepts a coherent
snapshot on any PyTorch device without turning construction into a
kernel-support claim. CPU construction and collection behavior are mandatory;
CUDA coverage is conditional on available hardware. Each later transform work
order must name its own supported device matrix, and accepting a CUDA-resident
collection does not promise that an as-yet-unimplemented kernel can execute
it.

Every production transform:

- operates on the collection's common device;
- allocates its new target tensor on that device when `out` is omitted;
- rejects destination or workspace device/dtype/signature mismatch;
- uses only the workspace's bound execution stream when a lease is supplied;
- performs no implicit `.cpu()`, `.numpy()`, or Python-list conversion;
- preserves the exact shared layout, canonical axes, accepted common extra
  axes, and canonical field order;
- uses vectorized PyTorch, scatter/gather, convolution, FFT, or other
  device-resident mechanics appropriate to the operation.

Order/stride support is execution-mode-specific. Functional operation must
preserve semantic results for arbitrary accepted axis order and may allocate
explicit normalization. Ordinary `out` behavior must document any supported
target strides and normalization. Warmed execution supports only the strict
contiguous sample-last profile and must reject all other layouts before RNG or
writes.

CPU-only behavior may be used as a small scientific reference, not as the
hidden implementation behind a nominal GPU transform.

Sample-last is not imposed on future reconstruction. A Readout-to-
Reconstruction bridge should select fields and exact `ChannelId` coordinates,
explicitly reorder/materialize once into the reconstruction algorithm's
preferred storage—potentially channel-last—and construct a
reconstruction-owned value before its hot path.

## Validation Matrix

Future implementation tests must cover:

- construction of every valid field role and representative nonempty subsets
  from explicit TensorCore records;
- empty, unknown, or wrong-order fields and missing canonical axes; field
  duplication is not representable in a Python mapping;
- exact canonical axis IDs, shared layout, common device, and
  `torch.strided` invariants, including valid noncontiguous tensors;
- exact `torch.int64` photoelectrons, common `torch.float32`/`torch.float64`
  floating roles, exact `torch.int32` digitized output, and rejection of mixed
  floating dtypes;
- conditional `DigitizedWaveformSpec` presence, transfer validation, ADC
  bounds, `[1, 16]` bit depth, inclusive `[0, 40]` dB gain,
  `AdcQuantization.TRUNCATE`, projection retention, and invalidation removal;
- wrong channel-coordinate class;
- general collection and functional shape/order semantics with at least two
  axis orders;
- functional semantic equality for arbitrary accepted axis order/strides;
- warmed acceptance for at least two sample-last leading-axis orders using
  distinct exact workspaces;
- preflight rejection of sample-not-last or noncontiguous participating
  source/output tensors, internally overlapping writable targets, wrong
  workspace axis order, and any attempted hidden normalization before RNG or
  writes;
- preservation of allowed extra axes;
- exact `readout.photoelectrons` target identity for zero-width timing jitter,
  plus exact integer-to-float `readout.charge` conversion when all optional
  response effects and smearing are disabled;
- malformed numeric configs, including bool rejection;
- functional target allocation, retained-field record identity, stale-
  descendant invalidation, and no source mutation;
- projection/removal without descendant invalidation;
- supplied destination identity, exact result schema, retained-field identity,
  zero-initialized valid target allocation, complete target-only writes, alias
  rejection, and incompatible destination failures;
- bitwise-unchanged destinations, source tensors, RNG position, and workspace
  generations after every preflight failure;
- scratch/output/source storage disjointness, including overlapping tensor
  views and conservative common-storage rejection;
- exact workspace signature, stable slot identities, write-before-read reset,
  no implicit resize/reconfiguration, and poisoned-slot failure behavior;
- exclusive non-reentrant same-stream workspace reuse, wrong-stream and
  concurrent-use rejection, and independent-workspace concurrency;
- `build_readout_collection(...)` field schema, accepted order, direct public
  target landing, and equality with explicit atomic composition;
- output lifetime: workspace reuse cannot mutate a prior result, while explicit
  reuse of the same caller destination authorizes overwrite;
- warmed steady-state stability of TensorDSLab-owned target/scratch storage,
  with backend allocator and library scratch reported only as diagnostics;
- memory instrumentation proving no `.contiguous()`, clone, copying reshape,
  cast, move, permutation materialization, or fallback target/scratch storage
  allocation occurs after warmed preflight;
- gradient preservation for functional pure and analog transforms;
- clear preflight rejection of gradient-sensitive `out` or workspace use;
- same-coordinate stochastic repeatability;
- reordering/batching/chunking RNG invariance;
- exact example/channel/lexical-extra/sample-ordinal RNG payload order and
  rejection of stochastic execution with an extra count-only axis;
- timing and afterpulse probability buckets and drop behavior;
- shared crosstalk/afterpulse source-snapshot behavior;
- charge-smearing aggregate statistics and clipping;
- end-to-end `simulate_charge(...)` behavior, including unchanged retained
  photoelectrons and private intermediate avalanche submodel checks;
- pulse-template and truncated-convolution fixtures;
- white and FFT noise normalization/shape/phase rules;
- analog sum/clip order;
- digitization gain/clip/quantization order and ADC bounds;
- parity fixtures name their comparison boundary, classification, assumptions,
  acceptance criteria, and intentional divergences;
- CPU and available accelerator behavior within accepted tolerances;
- forbidden imports and no host-list/NumPy hot path.

## Donor Behavior Promoted

TensorDSLab promotes these reviewed ideas:

- fixed-grid post-binned photoelectron and SiPM-response effects;
- photoelectron timing followed inside charge simulation by dark counts and
  parallel frozen-snapshot crosstalk/afterpulses;
- aggregate-bin charge smearing after integer effects;
- latent sub-bin timing and afterpulse probability models;
- drop-out-of-window acquisition semantics;
- same-length causal pulse rendering;
- direct tensor FFT noise rather than bank/crop mechanics;
- analog `pure + noise` composition before clipping/digitization;
- clip-before-integer-conversion ADC safety.

Promotion means architectural or scientific adoption. It does not by itself
establish exact, numerical, distributional, or statistical parity; those claims
and their qualifications live in `docs/parity.md`.

## Donor Behavior Rejected Or Deferred

- IV global RNG state and implicit call-order reproducibility;
- recursive unbounded PE-row growth;
- package-load condition DB and channel-map globals;
- post-readout sparse PE products as the core stochastic boundary;
- fixed `(1, channel, sample)` rank;
- CPU list conversion behind tensor products;
- stale nearest-bin config fields;
- channel-index RNG identity when channel coordinates exist;
- sequential per-channel white-noise streams;
- gain-only rendering presented as physical pulse parity;
- analog and digitized values sharing one field role or field ID;
- ADC unsigned-wrap accidents;
- remote spectrum downloads, large binary fixtures, or persistent FFT banks;
- reconstruction, trigger, and analysis preprocessing inside readout.

## First Production Slices

The architecture should be implemented in bounded stages:

1. Package, TensorCore-backed `ReadoutCollection`, atomic-target, and full-output
   preparation foundation, with semantic noncontiguous support and contiguous
   newly allocated targets.
2. Deterministic transforms plus the exact-signature workspace and internal
   builder-execution substrate, including strict sample-last/contiguous warmed
   preflight, without a placeholder public full-chain API.
3. Tensor-native RNG, photoelectron timing, workspace-backed SiPM charge
   response, and publication of the complete local builder.
4. White and direct FFT noise with measured scratch/allocation behavior.
5. Typed `ReadoutExample` composition after the local execution contract is
   stable.

Afterpulse implementation must wait for the recovery-amplitude decision. A
work order may otherwise defer one operation without weakening the collection
foundation.

## Return To Design Before

- changing the fixed-grid effect order;
- allowing recursive or cross-channel correlated-noise growth;
- exposing an intermediate avalanche-count collection field or writing
  sensor-origin avalanches back into `readout.photoelectrons`;
- silently choosing unit versus recovery-weighted afterpulses;
- changing drop-out-of-window policy;
- allowing negative `readout.charge` values;
- merging analog and digitized waveform field roles;
- introducing implicit movement, casting, detachment, or host conversion;
- returning a collection backed by workspace scratch or silently recycling a
  caller-owned source/output tensor;
- adding hidden workspace caches, resize, cross-stream sharing, leased outputs,
  partial-output planning, or a stronger backend-wide zero-allocation claim;
- weakening or broadening the warmed sample-last/contiguous profile without a
  focused measured Design contract;
- requiring sample-last or contiguity for general semantic collection
  construction, or imposing one execution order on every TensorDSLab domain;
- using transient indices where stable coordinates exist;
- adding source, cache, DAG, reconstruction, or downstream adapter scope;
- changing recognized readout field IDs/order or durable product labels.
