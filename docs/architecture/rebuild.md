# TensorDSLab Rebuild Architecture

Status: accepted Design architecture for the TensorCore `0.7.0` rebuild. It
does not dispatch implementation, change the installed dependency, replace
production bytes, or make a compatibility claim. Production replacement
requires a focused work order, fixed dependency evidence, Validation,
independent Review, and the ordinary merge gate.

Within this architecture, the fixed-`K` algorithm under
[Fixed-Generation Correlated-Avalanche Baseline](#fixed-generation-correlated-avalanche-baseline)
is the sole active correlated-avalanche implementation baseline. The separate
avalanche-algorithm architecture pages have been removed. A work order may
implement only this baseline unless a new explicit user and TensorDSLab Design
decision first changes this page.

The architecture pass was started from clean TensorDSLab `main` at
`3af8ab4acf834b07e3d027fb530e5f12934999a5`. The TensorCore reference examined
for this design is clean TensorCore `0.7.0` `main` at
`b454d738f6385ce6489d85492a618a3dab139bb6`. That exact commit contains the
operative ordinary-ABC semantic roots and the Stage 13 operation-owned
aliasing/freshness documentation contract. Selecting the exact package pin and
passing TensorDSLab-owned consumer probes remain explicit implementation-work-
order gates; this Design review is not a broad compatibility claim.

Stage 2 and Maintenance 1 remain valid historical evidence for the current
package. This is a clean pre-deployment redesign, not a compatibility
layer. Governance Core remains Adopted; conformance remains Not evaluated;
Coordination remains Deferred; and Profile B remains Disabled.

## Design Thesis

TensorDSLab should present a small class-and-function API to collaborators who
do not need to understand TensorCore internals, field registries, dependency
maps, partial pipeline state, or buffer scheduling.

The ordinary workflow is:

```python
readout = simulate_readout(
    photoelectrons,
    products=[
        AnalogWaveform,
        DigitizedWaveform,
    ],
    config=config,
    seed=1234,
)

analog = readout.field(AnalogWaveform)
digitized = readout.field(DigitizedWaveform)
```

The returned collection contains exactly the products requested by the caller:

```python
readout.field_types == frozenset(
    {
        AnalogWaveform,
        DigitizedWaveform,
    }
)
```

The builder computes prerequisites privately and at most once. A prerequisite
does not become a collection member unless it was also requested.

```text
requested DigitizedWaveform
  -> requires AnalogWaveform
       -> requires PureWaveform and NoiseWaveform
            -> PureWaveform requires Charge
                 -> Charge requires truth Photoelectrons
```

The architecture is:

```text
dense truth Photoelectrons
  -> request-aware private typed product producers
  -> one public simulate_readout(...)
  -> one immutable completed ReadoutCollection
       containing exactly the requested products
```

Exact Python classes carry in-process axis, product, and collection meaning.
There is no parallel namespace of axis IDs, field IDs, semantic constants,
product-name strings, canonical sequences, or dependency registries.

The public API separates three concerns:

```text
config    -> scientific model choices
products  -> final in-memory retention policy
seed      -> positional random realization
builder   -> dependency planning and execution
```

Durable persistence and IO are deferred entirely from this rebuild.

## Goals

- Make the normal API understandable from product names and function
  signatures alone.
- Use TensorCore's `TensorAxis`, `TensorField`, and `TensorCollection` directly.
- Replace loose semantic constants with exact final TensorDSLab types.
- Keep the readout input and all generated payloads dense, tensor-native, and
  resident on the caller-selected device.
- Preserve `Photoelectrons` as truth rather than replacing it with an
  electronics-smeared value.
- Express product dependencies through typed producer calls rather than a
  public workflow graph.
- Let callers retain only the products they consume.
- Keep scientific configuration exact, immutable, and compositional.
- Preserve accepted detector/readout behavior and parity classifications where
  their comparison boundaries still apply.
- Leave explicit later boundaries for TensorG4DS, TensorML, reconstruction,
  artifacts, and measured execution optimization beyond the accepted
  product-local waveform-tail fusion target.

## Non-Goals

- Backward compatibility with the current pre-deployment TensorCore `0.6`
  representation.
- Compatibility aliases for retired IDs, constants, sidecars, or helper
  modules.
- Passing CPU-resident jagged G4DS tables into the readout builder.
- Native G4DS file parsing or TensorG4DS deposit clustering.
- Persistence, cache formats, artifact stores, or write policy.
- TensorML model, training, metric, or checkpoint ownership.
- Projects/dag scheduling, retries, fan-out, or campaign policy.
- A generic `Config(ABC)` without a real polymorphic consumer.
- A public workspace, allocator, stream lease, or `out=` surface before
  profiling demonstrates the need.
- An exact-until-extinction or recovery-marked correlated-avalanche cascade.
  The selected charge path uses one caller-bounded fixed-generation process;
  `maximum_generations=1` is its first-generation case rather than a separate
  algorithm.
- Bitwise parity merely because an older implementation produced a particular
  RNG stream.

## Ecosystem And Input Boundary

The intended chain remains:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared substrate, not another data-flow stage.

- G4DS owns native simulation output.
- TensorG4DS owns native ingestion, CPU-resident jagged source tables, and
  low-level tensor processing such as deposit clustering.
- TensorDSLab owns the dense downstream detector/readout products defined here
  and future reconstruction products.
- TensorML owns model-facing schemas, models, training, and evaluation.

`simulate_readout(...)` accepts a dense TensorDSLab
`Photoelectrons` field. It does not accept a native G4DS table, an Awkward
Array, a jagged PE table, or an untyped mapping of columns. Jagged tables remain
on the CPU and outside the GPU readout hot path.

A separate future TensorDSLab-owned bridge will consume one exact accepted
TensorG4DS product and construct the dense truth `Photoelectrons` field. That
bridge owns event/channel mapping and realizes the caller's exact
`SamplingConfig` as numeric PE bins and a timestamp-backed `SampleAxis`. It
must not infer the window from observed hits or apply readout timing jitter:
jitter is an electronics response effect inside charge simulation, not truth
construction.

The production handoff target keeps dense payloads on one explicit accelerator
device and does not silently call `.cpu()`, `.numpy()`, convert through Python
lists, serialize/reload, cast, move, or detach.

## TensorCore `0.7` Consumer Contract

The rebuild targets TensorCore's three semantic roots:

```text
TensorAxis
  coordinates: tuple[str, ...]

TensorField
  tensor: torch.Tensor
  axes: tuple[TensorAxis, ...]

TensorCollection
  immutable fields keyed by exact TensorField subtype
```

Every TensorDSLab semantic leaf:

- has `__bases__ == (matching_tensor_core_root,)`, with no mixin or other base;
- is a public `@final` class;
- declares `__slots__ = ()`;
- adds no stored fields;
- implements `_require()` for TensorDSLab semantic narrowing; and
- inherits TensorCore construction, validation, immutability, and lookup.

Leaves do not reapply `@dataclass`. They use the ordinary inherited root
constructor. Direct inheritance, `@final`, empty slots, fieldlessness, and
inherited root behavior are TensorDSLab static-analysis, test, and Review
obligations rather than runtime lineage enforcement supplied by TensorCore.

TensorCore owns universal representation validation. TensorDSLab owns axis,
product, collection, dtype, device, scientific, and operation relationships.
Exact concrete class identity replaces runtime axis and field IDs.

TensorCore `0.7` has no layout object, metadata mapping, generic selection,
generic movement, output buffer, workspace, persistence, or lifecycle API.
Exact Python type identity is an in-process contract, not a durable artifact
identifier.

Before implementation, TensorDSLab must select and test an exact TensorCore
dependency. That checkpoint includes runtime construction, package-root
imports, static constructor typing, exact-leaf validation, and public
relationship helpers. It must provide explicit evidence for ordinary-ABC
inherited constructor signatures and concrete result inference with the
selected static checker. A generic gap returns to TensorCore Design with a
minimal reproducer rather than being patched through a downstream fork.

TensorCore establishes neither universal freshness nor universal storage
sharing. Every TensorDSLab operation returning one or more fields must classify
each successful path using TensorCore's exact vocabulary:

- exact return;
- guaranteed storage-sharing result;
- sharing permitted but unspecified; or
- guaranteed fresh storage independent of named inputs.

The owning operation separately specifies subtype, dtype, device, layout and
strides, axes, autograd, synchronization, failure effects, and any promised
output-to-output storage relationship. Constructing or returning a field is
not a device synchronization point. TensorCore provides no runtime overlap
scanner, copy-on-write layer, lease, workspace, or stream-ordering service.

TensorDSLab validates documented public inputs, scientific configuration, and
cheap correctness-critical operation relationships. It does not harden the
package against callers who leave the public contract by subclassing final
semantic leaves, modifying classes, bypassing inherited construction, invoking
private functions directly, mutating exposed tensors, or installing custom
dispatch behavior. Such use is unsupported and may fail naturally or produce
invalid results; it does not require a stable error category, an eager guard,
or adversarial test coverage.

## Selected Rebuild Package Shape

Status: accepted Design contract for the rebuild. A production work order may
materialize only the files needed by its implementation slice; this complete
tree is the ownership target, not authorization to create placeholders.

```text
tensor_dslab/
  __init__.py

  common/
    __init__.py
    axes.py                  # ExampleAxis, ChannelAxis, SampleAxis
    sampling.py              # SamplingConfig and canonical sample-grid facts

  readout/
    __init__.py
    types.py                 # ReadoutConfig and ReadoutCollection only
    simulation.py            # public simulate_readout() orchestration
    _requirements.py         # shared private readout requirements
    _random.py               # shared private readout RNG, when implemented

    photoelectrons/
      __init__.py
      types.py               # Photoelectrons; no config or producer

    charge/
      __init__.py
      types.py               # Charge and Charge-related configs
      _product.py            # _product_charge() and private submodels

    pure_waveform/
      __init__.py
      types.py               # field, wrapper config, TPC/Veto model configs
      _product.py            # _product_pure_waveform()

    noise_waveform/
      __init__.py
      types.py               # field and zero/white/PSD configs
      _product.py            # _product_noise_waveform()

    analog_waveform/
      __init__.py
      types.py               # field, config, and saturation config
      _product.py            # _product_analog_waveform()

    digitized_waveform/
      __init__.py
      types.py               # DigitizedWaveform and its config
      _product.py            # _product_digitized_waveform()
```

The tree is organized around semantic products rather than implementation
layers. Each product's `types.py` owns its exact `TensorField` leaf and that
product's public configuration types. Its `_product.py`, once implemented,
owns the private `_product_*` builder and any `_simulate_*` scientific
submodels needed by that product. Product-specific deep validation remains
with the product; `_requirements.py` contains only relationships genuinely
shared across readout products.

`readout/types.py` contains exactly the two cross-product composition types:
`ReadoutConfig` and `ReadoutCollection`. It is not a miscellaneous record
module. `ReadoutConfig` composes `SamplingConfig` and optional product configs;
`ReadoutCollection` composes the requested product fields. Product packages
never import `readout.types`.

`ExampleAxis`, `ChannelAxis`, and `SampleAxis` belong in
`tensor_dslab.common.axes`. `SamplingConfig` belongs in
`tensor_dslab.common.sampling` because the future TensorG4DS bridge, readout,
and future Reconstruction may share the same sample-grid contract. This does
not add source binning to the current readout package.

`Photoelectrons` is an already-produced dense truth input. Its package has no
`PhotoelectronsConfig` and no `_product.py`; source construction and PE binning
remain deferred to the future TensorG4DS bridge. `simulate_readout(...)`
borrows the supplied field and validates its realized `SampleAxis` against the
caller's `SamplingConfig`.

`_requirements.py` and `_random.py` are private modules. Privacy is an API and
compatibility boundary, not a runtime access-control mechanism. The public RNG
surface remains the root `seed`; readout-specific streams, counters, engines,
and distribution samplers stay in `readout._random`. Move generic mechanics to
`common` only after a second TensorDSLab domain needs the exact same accepted
contract.

The dependency direction is acyclic:

```text
tensor_core
  -> tensor_dslab.common
  -> readout._requirements
  -> product types
  -> product _product modules and prerequisite product types
  -> readout.types
  -> readout.simulation
  -> deliberate package-root exports
```

Product packages do not import `ReadoutConfig`, `ReadoutCollection`, or
`simulate_readout(...)`. Product `_product.py` modules import only their own
types, explicit prerequisite product types, shared sampling facts, private RNG
when needed, and focused requirements. `readout.simulation` is the sole layer
allowed to import the complete product graph and orchestrate it.

The physical module path does not define public visibility. Package
`__init__.py` files and `__all__` deliberately re-export the collaborator-
facing classes, configs, and `simulate_readout(...)`; collaborators need not
import from nested product modules. `simulation.py`, rather than a generic
`api.py`, names the behavior it owns. Do not add global `configs/`,
`fields.py`, `builders.py`, or `validation.py` dumping grounds.

Create `_random.py`, any `_product.py`, or another future module only when an
accepted implementation stage gives it real behavior. Do not create empty
files to reserve this tree.

### Exact Foundation Symbol Inventory

The first rebuild foundation stage freezes this concrete ownership inventory.
It may add module-private implementation details needed to express these
contracts, but it must not move public types between modules, introduce a
second registry, or create a later behavior module as a placeholder.

| Module | Public symbols owned by the foundation stage | Shared private symbols |
| --- | --- | --- |
| `common/axes.py` | `ExampleAxis`, `ChannelAxis`, `SampleAxis` | none |
| `common/sampling.py` | `SamplingConfig` | none; this common module validates its own config directly |
| `readout/_requirements.py` | none | `_require_readout_structure`, `_require_dtype`, `_require_floating_dtype`, `_require_exact`, `_require_optional_exact`, `_require_one_of_exact` |
| `readout/photoelectrons/types.py` | `Photoelectrons` | product-local `_require_valid_values` |
| `readout/charge/types.py` | `Charge`, `TimingJitterConfig`, `DarkCountConfig`, `FixedDelayConfig`, `ExponentialDelayConfig`, `NormalDelayConfig`, `DirectCrosstalkConfig`, `DelayedCrosstalkConfig`, `AfterpulseRecoveryConfig`, `AfterpulseConfig`, `CorrelatedAvalancheConfig`, `ChargeSmearingConfig`, `ChargeConfig` | product-local `_require_valid_values` |
| `readout/pure_waveform/types.py` | `PureWaveform`, `TpcFebSnrPulseConfig`, `VetoPduPulseConfig`, `PureWaveformConfig` | product-local `_require_valid_values` |
| `readout/noise_waveform/types.py` | `NoiseWaveform`, `ZeroNoiseConfig`, `WhiteNoiseConfig`, `PsdNoiseConfig`, `NoiseWaveformConfig` | product-local `_require_valid_values` |
| `readout/analog_waveform/types.py` | `AnalogWaveform`, `AnalogSaturationConfig`, `AnalogWaveformConfig` | product-local `_require_valid_values` |
| `readout/digitized_waveform/types.py` | `DigitizedWaveform`, `DigitizedWaveformConfig` | product-local `_require_valid_values` accepting the exact config |
| `readout/types.py` | `ReadoutConfig`, `ReadoutCollection` | none |

The shared private requirement functions exist only where two or more product
modules need the exact same relationship. Product-specific value-domain scans
remain private to their product under the consistently scoped
`_require_valid_values` name rather than becoming a generic validation layer.
That name is fixed for the foundation work order and tests but remains private,
not a downstream compatibility surface.

Every product subpackage root re-exports only its public row above. The
`common` and `readout` roots compose those deliberate exports, and the package
root re-exports the collaborator-facing axes, sampling/config types, product
field types, `ReadoutConfig`, and `ReadoutCollection`. Generic TensorCore names
are never re-exported. Importing the public package must not import a future
simulation, RNG, compiler, TensorG4DS, TensorML, or IO dependency.

The following behavior symbols belong to later focused stages and are not
created by the foundation stage:

- `readout.simulation.simulate_readout`;
- `_product_charge`, `_product_pure_waveform`, `_product_noise_waveform`,
  `_product_analog_waveform`, and `_product_digitized_waveform` in their
  corresponding product `_product.py` modules;
- Charge's private `_simulate_*` scientific submodels; and
- the still-private, gate-dependent contents of `readout._random`.

This separation makes the Stage 3 exit testable: all accepted semantic leaves,
configs, collection composition, imports, and constructor contracts exist,
while no scientific simulation behavior or empty architectural scaffolding
does.

The rebuild retires:

- `common/ids.py`, `ExampleId`, and `ChannelId`;
- `readout/ids.py`;
- `TensorAxisId`, `TensorFieldId`, and `IdSequence` values;
- module-level axis/field constants and product registries;
- `TensorLayout`, `shared_axes`, and layout reconstruction;
- count-only sample semantics and `SampleGrid`;
- `DigitizedWaveformSpec` as a collection sidecar;
- ordered partial pipeline snapshots and descendant invalidation;
- `readout/tensors.py` reconstruction/projection helpers; and
- public mutation of a collection through atomic add-or-replace transforms.

The package root and subpackage roots export only deliberate public classes and
functions. Historical work orders remain unchanged as records of what they
implemented.

## Semantic Axes

```python
from typing import final

from tensor_core import TensorAxis


@final
class ExampleAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
            raise ValueError("ExampleAxis must be nonempty")


@final
class ChannelAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.coordinates:
            raise ValueError("ChannelAxis must be nonempty")


@final
class SampleAxis(TensorAxis):
    __slots__ = ()

    def _require(self) -> None:
        if len(self.coordinates) < 2:
            raise ValueError("SampleAxis requires at least two timestamps")

        times_ps: list[int] = []
        for coordinate in self.coordinates:
            if not coordinate.endswith("ps"):
                raise ValueError("SampleAxis timestamps must end in 'ps'")
            magnitude = coordinate[:-2]
            if not (
                magnitude == "0"
                or (
                    magnitude
                    and magnitude[0] != "0"
                    and magnitude.isascii()
                    and magnitude.isdigit()
                )
            ):
                raise ValueError("noncanonical SampleAxis timestamp")
            time_ps = int(magnitude)
            if time_ps > (1 << 63) - 1:
                raise ValueError("SampleAxis timestamp exceeds int64")
            times_ps.append(time_ps)

        period_ps = times_ps[1] - times_ps[0]
        if period_ps <= 0:
            raise ValueError("SampleAxis timestamps must increase")
        if any(
            right - left != period_ps
            for left, right in zip(times_ps, times_ps[1:])
        ):
            raise ValueError("SampleAxis timestamps must be uniformly spaced")
        if times_ps[-1] + period_ps > (1 << 63) - 1:
            raise ValueError("SampleAxis exclusive stop exceeds int64")

    @property
    def start_ps(self) -> int:
        return int(self.coordinates[0][:-2])

    @property
    def sample_period_ps(self) -> int:
        return int(self.coordinates[1][:-2]) - self.start_ps

    @property
    def stop_ps(self) -> int:
        return int(self.coordinates[-1][:-2]) + self.sample_period_ps
```

There are no corresponding axis-ID constants. Code locates dimensions by exact
type:

```python
sample_dimension = field.dimension_of(SampleAxis)
sample_axis = field.axis(SampleAxis)
```

### Coordinate Contract

Every coordinate is an exact, unique, nonempty string. Tuple order is tensor
index order. Positional RNG uses that index and does not use the coordinate
string as random identity.

- `ExampleAxis` contains stable TensorDSLab example keys.
- `ChannelAxis` contains stable detector/readout channel keys.
- `SampleAxis` contains canonical time-ordered truth-bin timestamp strings
  constructed from the accepted sampling policy.

`SamplingConfig` owns the numeric policy from which a regular timestamp axis
is realized. The ordinary construction path has this shape:

```python
sampling = SamplingConfig(
    sample_period_ps=PositiveInteger(2_000),
    sample_count=PositiveInteger(8_192),
)
samples = sampling.build_axis()
```

TensorDSLab uses one bin convention everywhere: every stored bin coordinate is
the inclusive left edge, every bin is left-closed and right-open, and the final
exclusive stop is carried or derived separately. Public bin arrays therefore
never mix left edges with centers or terminal right edges. This applies to
sample timestamps, upstream numeric PE bins, PSD frequency bins, and later
histogram-like scientific inputs unless a focused Design change says otherwise.

The config first defines numeric left edges plus the exclusive window stop for
upstream PE binning and then generates canonical semantic left-edge coordinates
such as `"0ps"`, `"2000ps"`, and `"4000ps"`. A full window starts at
example-local zero. The exact timestamp grammar is ASCII
`^(0|[1-9][0-9]*)ps$`: lowercase `ps`, no sign, whitespace, decimal point,
exponent, alternate unit, or leading zero. Direct `SampleAxis(...)`
construction remains available for semantic reconstruction, but `_require()`
requires at least two coordinates, nonnegative signed-int64 values, strict
increase, one positive integer-picosecond spacing, and a derived exclusive
stop no greater than `2**63 - 1`.

This is the low-level construction surface for fixtures, custom sources, and
the future TensorG4DS bridge. It is not a second readout builder.
`simulate_readout(...)` receives `Photoelectrons` with a complete
`SampleAxis`, requires its already-validated start, period, and size to match
`config.sampling`, reuses that exact axis object for every generated field, and
never creates, rebases, or replaces it. Because construction has already
validated the complete regular tuple, agreement is an O(1) check of size,
`start_ps == 0`, and `sample_period_ps`; it neither rebuilds nor reparses the
full coordinate tuple in the repeated readout path.

The earlier count-only representation was never used and is not carried into
the rebuild. `SampleGrid` is retired. Timestamp strings describe dense readout
bins, not individual G4 hits.

TensorCore does not parse or chronologically validate strings. TensorDSLab's
accepted `SampleAxis._require()` contract owns the grammar, signed-int64
domain, chronological order, positive uniform spacing, and derivable period
and stop. Charge simulation, pulse convolution, and
power-spectral-density-shaped noise synthesis consume numeric `SamplingConfig`
values and tensor indices rather than parsing coordinate strings. Operation
preflight may impose additional algorithm-specific limits on an already-valid
period.

A shared `SampleAxis` means every example in one dense tensor uses the same
relative readout-bin coordinates. A complete bridge-produced example window
starts at zero. With left-edge coordinates and period `dt`, its timestamps are
`0, dt, ..., (sample_count - 1) * dt`, bin `i` represents
`[i * dt, (i + 1) * dt)`, and the exclusive window stop is
`sample_count * dt`. Per-example absolute G4 origins and trigger position, if
needed, belong in explicit bridge provenance rather than ambiguous sample
coordinates.

After the future bridge has normalized an upstream hit to an exact accepted
example-local integer-picosecond value, bin assignment is:

```text
if 0 <= time_ps < sampling.window_stop_ps:
    sample_index = time_ps // sampling.sample_period_ps.value
else:
    drop and account for the hit
```

This freezes boundary ownership but not the upstream conversion of floating G4
time into exact numeric picoseconds. That rounding/normalization policy remains
part of the focused TensorG4DS bridge contract.

A later contiguous selection of at least two samples preserves the selected
timestamp strings instead of rebasing them, so a valid subaxis may start above
zero while remaining relative to the original example origin. A singleton
selection cannot reconstruct a period-bearing `SampleAxis` in the MVP and
remains a non-readout semantic result until a separate timing-association
contract exists. The zero-start rule is therefore a full-window bridge
postcondition, not a universal `SampleAxis` invariant. The MVP public builder
accepts only a full source axis matching its zero-start `SamplingConfig`; making
a selected subwindow a new simulation input requires a later boundary/halo
policy rather than silently treating it as a complete window.

### Axis Order

Every readout field contains exactly one `ExampleAxis`, one `ChannelAxis`, and
one `SampleAxis`. Semantic construction accepts those axes in any order; the
tuple order remains tensor dimension order. Builders reuse the exact source
axis instances and locate dimensions by exact type. A different valid axis
tuple order is nevertheless a different positional RNG schema; the builder
does not attempt to reproduce the same draws across a tensor permutation.

The local relationship check is order-independent and does not accidentally
call TensorCore's ordered `require_axis_signature(...)`:

```python
def _require_readout_structure(field: TensorField) -> None:
    axis_types = tuple(type(axis) for axis in field.axes)
    accepted = frozenset({ExampleAxis, ChannelAxis, SampleAxis})
    if len(axis_types) != 3 or frozenset(axis_types) != accepted:
        raise ValueError(
            "readout fields require exactly example, channel, and sample axes"
        )
    if field.tensor.layout is not torch.strided:
        raise ValueError("readout fields require dense strided tensors")
```

The shared `_require_floating_dtype(...)` relationship accepts exactly
`torch.float32` or `torch.float64`. No product leaf expands that set to
`torch.float16` or `torch.bfloat16`.

The supported MVP contract and its storage, aliasing, compiled-kernel, and
fresh-result evidence cover ordinary `torch.Tensor` behavior. Custom tensor
subclasses and dispatch modes are outside that contract. TensorDSLab does not
need a defensive runtime guard that recognizes or rejects every unsupported
Torch extension.

The upstream bridge should ordinarily construct
`(ExampleAxis, ChannelAxis, SampleAxis)` so samples are last for temporal GPU
kernels. A future warmed execution profile may require sample-last contiguous
storage without changing semantic identity.

## Product Fields

TensorDSLab defines six direct final `TensorField` leaves:

| Product | Intrinsic leaf contract | Producer/deep-validation postcondition | Meaning |
| --- | --- | --- | --- |
| `Photoelectrons` | `torch.int64`, exact readout axes | nonnegative | dense binned photon-origin truth PE counts |
| `Charge` | `torch.float32` or `torch.float64` | finite and nonnegative | aggregate PE-equivalent SiPM response |
| `PureWaveform` | `torch.float32` or `torch.float64` | finite | signal-only waveform in mV |
| `NoiseWaveform` | `torch.float32` or `torch.float64` | finite | noise-only voltage excursion about zero in mV |
| `AnalogWaveform` | `torch.float32` or `torch.float64` | finite | zero-referenced composed analog waveform in mV |
| `DigitizedWaveform` | `torch.int32` | nonnegative and producer-bounded by its digitizer config | immediate ADC-code output |

`DigitizedWaveform`, not `DigitalWaveform`, remains the accepted name.
`DigitalWaveform` is reserved for a possible later firmware/filter/trigger/
compression product.

Every field uses `torch.strided` tensor layout. Operations preserve existing
input fields and perform no implicit movement or in-place cast. Newly generated
products intentionally use their declared output dtype: truth `torch.int64`
becomes floating `Charge`, and floating `AnalogWaveform` becomes
`torch.int32` ADC codes. Charge, pure, noise, and analog use one
builder-selected `torch.float32` or `torch.float64` dtype.

Illustrative leaves are:

```python
@final
class Photoelectrons(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_dtype(self, torch.int64)


@final
class Charge(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_floating_dtype(self)


@final
class PureWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_floating_dtype(self)


@final
class NoiseWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_floating_dtype(self)


@final
class AnalogWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_floating_dtype(self)


@final
class DigitizedWaveform(TensorField):
    __slots__ = ()

    def _require(self) -> None:
        _require_readout_structure(self)
        _require_dtype(self, torch.int32)
```

Repeated field relationships live in private functions. There is no artificial
`ReadoutField` base and no loose dtype/dependency/name mapping.

### Truth Meaning Of `Photoelectrons`

`Photoelectrons` is the dense, binned photon-origin truth input to readout
simulation. It never includes timing jitter, dark counts, crosstalk,
afterpulses, or charge smearing.

When configured effectively, timing jitter is a private step of charge
production after any private dark-count avalanches have been added. It
redistributes the then-current working counts but does not replace or mutate
the source field. Therefore:

- requesting `Photoelectrons` retains the exact input field;
- requesting `Charge` uses a private working-count representation that is
  jittered only when that stage executes;
- requesting both returns unjittered truth beside the derived charge; and
- the original input remains unchanged in every case.

### Structural And Deep Validation

TensorCore always validates tensor/axis shape and semantic lineage.
TensorDSLab leaf construction validates cheap intrinsic facts such as exact
axes, dtype, and `torch.layout`. It does not hide a full-device
`isfinite().all()` or nonnegativity scan inside every construction.

The exact class identifies declared semantic role and representation; it is
not proof that arbitrary caller-supplied values satisfy every scientific
postcondition. Public builders guarantee their output domains. Untrusted
ingress—including the source `Photoelectrons` supplied to the public builder
and any future artifact load—runs the explicit product-specific deep validator
before scientific execution.

This avoids repeated accelerator synchronization without weakening the named
trust boundary. Documentation must not claim that a bare constructor proves a
config-dependent ADC maximum. Builder postconditions and deep validators are
tested separately.

## `ReadoutCollection`

`ReadoutCollection` is an immutable completed result for one explicit product
request. It accepts any nonempty subset of the six recognized product types.
It is not a partially executed pipeline and exposes no add, replace, or
invalidation lifecycle.

Here *completed* means that collection membership is final and contains no
workflow state. It does not mean every variable calibration fact is stored in
the sidecar-free collection record. In particular, a digitized-only result is a
valid completed in-process result only while its caller separately retains the
exact `DigitizedWaveformConfig` needed to interpret it. Durable or
independently transported digitized values remain blocked on the explicit
association in Design gate 4.

Membership is semantically unordered. A collection requires:

- at least one recognized exact product type;
- no unrecognized type;
- at most one field of each exact type, already enforced generically by
  TensorCore;
- equal ordered axes on every present field;
- the same exact device on every present field;
- one common dtype among all present floating readout fields; and
- every product's intrinsic leaf contract.

The accepted schema is declared once on the owning collection class:

```python
@final
class ReadoutCollection(TensorCollection):
    __slots__ = ()

    @classmethod
    def accepted_field_types(
        cls,
    ) -> frozenset[type[TensorField]]:
        return frozenset(
            {
                Photoelectrons,
                Charge,
                PureWaveform,
                NoiseWaveform,
                AnalogWaveform,
                DigitizedWaveform,
            }
        )

    def _require(self) -> None:
        if not self.field_types:
            raise ValueError("ReadoutCollection must be nonempty")
        require_field_types(
            self,
            required=frozenset(),
            optional=self.accepted_field_types(),
        )

        fields = tuple(self.fields.values())
        require_same_axes(*fields)
        require_same_device(*fields)

        floating_dtypes = {
            field.tensor.dtype
            for field in fields
            if field.tensor.is_floating_point()
        }
        if len(floating_dtypes) > 1:
            raise ValueError("readout floating fields must share one dtype")
```

This one class-owned method is the unavoidable accepted-schema declaration. It
replaces module constants, field IDs, canonical-order registries, floating-role
registries, and descendant maps. Its returned set has no order semantics.

Ordinary access is type-directed:

```python
analog = readout.field(AnalogWaveform)
analog_tensor = readout.tensor(AnalogWaveform)
```

A missing unrequested product raises `KeyError`:

```python
readout.field(Charge)  # computed as a prerequisite but not retained
```

Typed convenience properties may be added later only if they materially help
collaborators and remain thin exact-type lookups.

## Product Requests

`products` is a required keyword-only iterable of exact product classes. A
caller may pass a list, tuple, generator, or another iterable; its iteration
order has no semantic meaning.

The builder:

1. consumes the iterable exactly once;
2. rejects an empty request;
3. requires exact classes accepted by `ReadoutCollection`;
4. rejects duplicates before converting membership to a set;
5. computes the transitive prerequisite closure;
6. preflights every required config and runtime relationship;
7. executes each required product producer at most once; and
8. retains only requested fields.

Unknown, duplicate, empty, or unsatisfied requests fail before an RNG draw or
tensor write.

The planner is ordinary typed code, not a public dependency registry. A
conceptual implementation can derive booleans from the requested set:

```python
need_digitized = DigitizedWaveform in requested
need_analog = AnalogWaveform in requested or need_digitized
need_pure = PureWaveform in requested or need_analog
need_noise = NoiseWaveform in requested or need_analog
need_charge = Charge in requested or need_pure
```

`Photoelectrons` is always available as the source, but it is retained only
when explicitly requested. Each requested combination remains a completed
result because no collection member represents workflow state.

Changing retention alone must not change the value of a product common to two
requests. RNG design and operation scheduling must therefore isolate product
random fields from unrelated requested branches.

## Scientific Configuration

`ReadoutConfig` composes one required shared sampling policy with optional
exact product configs. Every product producer with scientific choices accepts
its exact config type. A time-dependent builder also receives the shared exact
`SamplingConfig`; no subfunction receives the whole `ReadoutConfig` as a
service locator.

Concrete configs are normal domain value classes:

```text
@final
@dataclass(frozen=True, slots=True, kw_only=True)
```

They validate exact component types, compose other configs, and may use
TensorCore constrained scalars. `None` disables an optional submodel.
Alternative algorithms use closed unions of exact config classes rather than
string switches.

The hierarchy is:

```text
ReadoutConfig
├── SamplingConfig
├── ChargeConfig | None
│   ├── DarkCountConfig | None
│   ├── TimingJitterConfig | None
│   ├── CorrelatedAvalancheConfig | None
│   │   ├── maximum_generations
│   │   ├── DirectCrosstalkConfig | None
│   │   ├── DelayedCrosstalkConfig | None
│   │   └── AfterpulseConfig | None
│   │       └── AfterpulseRecoveryConfig | None
│   └── ChargeSmearingConfig | None
├── PureWaveformConfig | None
│   └── model: TpcFebSnrPulseConfig | VetoPduPulseConfig
├── NoiseWaveformConfig | None
│   └── model: ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig
├── AnalogWaveformConfig | None
│   └── AnalogSaturationConfig | None
└── DigitizedWaveformConfig | None
```

This is the sole charge hierarchy selected by the rebuild. It does not reserve
a second first-generation crosstalk/afterpulse surface. A caller who wants one
offspring generation chooses `maximum_generations=1`; increasing `K` extends
the same coupled algorithm. No second recursive surface or competing
avalanche-algorithm document is part of the rebuild contract.

The two product wrappers deliberately use the same vocabulary:

```python
PureWaveformConfig(model=TpcFebSnrPulseConfig(...))
NoiseWaveformConfig(model=PsdNoiseConfig(...))
```

The exact model class selects the accepted algorithm without a string switch,
registry, loose type alias, or marker ABC. TPC and Veto pulse response have
different equations and parameter schemas, so they earn separate exact model
classes. Noise generation, analog composition, saturation, and digitization
remain shared algorithms whose calibrated values may differ by detector
subsystem.

### Scalar MVP Calibration

Every scientific calibration value in an MVP config is scalar and applies
uniformly to the complete channel axis for one `simulate_readout(...)` call.
The same configured pulse parameters, noise model and power, analog limits,
ADC transfer, and eventual charge-response parameters apply to every channel
and example in that invocation. This is parameter homogeneity, not output
equality: source values and position-addressed stochastic realizations may
differ independently at every tensor position.

One `PureWaveformConfig.model` likewise applies to the complete channel axis.
The MVP therefore has a caller precondition that each invocation is
homogeneous with respect to both its TPC or Veto electronics-response family
and its calibration values. TPC and Veto, or two differently calibrated
channel groups, are simulated in separate invocations. Generic `ChannelAxis`
strings carry no trusted family or calibration provenance, so preflight does
not infer parameters from coordinate text and performs no per-channel lookup
or implicit parameter broadcasting.

Future channel-varying calibration should use an explicit, strongly typed,
device-resident tensor representation whose channel axis is validated against
the simulated data. It should be passed as a deliberate scientific input and
prepared before the hot path. It should not be smuggled into these frozen
scalar configs as a mutable `torch.Tensor`, a channel-keyed dictionary, or a
large tuple whose ordering must be interpreted privately. The exact future
type, supported parameter axes, movement/lifetime rules, and composition with
scalar configs require a focused Design stage; the MVP does not reserve a
placeholder class or pretend that arbitrary broadcasting is supported.

There is no `PhotoelectronsConfig`: the cross-product `SamplingConfig` defines
the shared numeric readout window used to create dense truth and every later
time-dependent product. `Photoelectrons` remains an already-constructed truth
input to `simulate_readout(...)`; the future TensorG4DS bridge receives
the same `SamplingConfig` when it bins the upstream jagged PE table. Timing
jitter belongs to `ChargeConfig` because it is a private electronics-response
step used only to produce charge.

`SamplingConfig` is always required because it defines the realized source
grid. Top-level product configs are optional so a caller can configure only
the requested computation. Missing required product config is a
request-specific preflight error. A `None` nested inside an existing product
config disables that submodel; a `None` top-level product config means that
product cannot be built by this invocation.

Illustrative definitions are shown together below even though their production
owners are split across `tensor_dslab.common.sampling`, `readout.types`, and
the corresponding product package's `types.py`. TPC and Veto pulse-model
configs initially live with `PureWaveform` in
`readout.pure_waveform.types`; split them further inside that product package
only when real implementation size or behavior justifies it:

```python
def _require_exact(value: object, expected: type[object], field: str) -> None:
    if type(value) is not expected:
        raise TypeError(f"{field} must be exactly {expected.__name__}")


def _require_optional_exact(
    value: object | None,
    expected: type[object],
    field: str,
) -> None:
    if value is not None:
        _require_exact(value, expected, field)


def _require_one_of_exact(
    value: object,
    expected: tuple[type[object], ...],
    field: str,
) -> None:
    if type(value) not in expected:
        names = ", ".join(item.__name__ for item in expected)
        raise TypeError(f"{field} must be exactly one of: {names}")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingConfig:
    sample_period_ps: PositiveInteger
    sample_count: PositiveInteger

    def __post_init__(self) -> None:
        _require_exact(
            self.sample_period_ps,
            PositiveInteger,
            "SamplingConfig.sample_period_ps",
        )
        _require_exact(
            self.sample_count,
            PositiveInteger,
            "SamplingConfig.sample_count",
        )
        if self.sample_count.value < 2:
            raise ValueError("SamplingConfig.sample_count must be at least 2")
        if (
            self.sample_period_ps.value * self.sample_count.value
            > (1 << 63) - 1
        ):
            raise ValueError("SamplingConfig.window_stop_ps exceeds int64")

    @property
    def window_stop_ps(self) -> int:
        return self.sample_period_ps.value * self.sample_count.value

    def build_axis(self) -> SampleAxis:
        period = self.sample_period_ps.value
        return SampleAxis(
            coordinates=tuple(
                f"{index * period}ps"
                for index in range(self.sample_count.value)
            )
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TimingJitterConfig:
    sigma_ns: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.sigma_ns,
            NonnegativeFloat,
            "TimingJitterConfig.sigma_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DarkCountConfig:
    rate_hz: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.rate_hz,
            NonnegativeFloat,
            "DarkCountConfig.rate_hz",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FixedDelayConfig:
    delay_ns: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.delay_ns,
            NonnegativeFloat,
            "FixedDelayConfig.delay_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExponentialDelayConfig:
    mean_delay_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_delay_ns,
            PositiveFloat,
            "ExponentialDelayConfig.mean_delay_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NormalDelayConfig:
    location_ns: NonnegativeFloat
    sigma_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.location_ns,
            NonnegativeFloat,
            "NormalDelayConfig.location_ns",
        )
        _require_exact(
            self.sigma_ns,
            PositiveFloat,
            "NormalDelayConfig.sigma_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DirectCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig | NormalDelayConfig

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DirectCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig, NormalDelayConfig),
            "DirectCrosstalkConfig.delay",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DelayedCrosstalkConfig:
    mean_offspring_per_parent: NonnegativeFloat
    delay: FixedDelayConfig | ExponentialDelayConfig | NormalDelayConfig

    def __post_init__(self) -> None:
        _require_exact(
            self.mean_offspring_per_parent,
            NonnegativeFloat,
            "DelayedCrosstalkConfig.mean_offspring_per_parent",
        )
        _require_one_of_exact(
            self.delay,
            (FixedDelayConfig, ExponentialDelayConfig, NormalDelayConfig),
            "DelayedCrosstalkConfig.delay",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseRecoveryConfig:
    time_constant_ns: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.time_constant_ns,
            PositiveFloat,
            "AfterpulseRecoveryConfig.time_constant_ns",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AfterpulseConfig:
    probability: Probability
    mean_delay_ns: PositiveFloat
    recovery: AfterpulseRecoveryConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(
            self.probability,
            Probability,
            "AfterpulseConfig.probability",
        )
        _require_exact(
            self.mean_delay_ns,
            PositiveFloat,
            "AfterpulseConfig.mean_delay_ns",
        )
        _require_optional_exact(
            self.recovery,
            AfterpulseRecoveryConfig,
            "AfterpulseConfig.recovery",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CorrelatedAvalancheConfig:
    maximum_generations: NonnegativeInteger
    direct_crosstalk: DirectCrosstalkConfig | None = None
    delayed_crosstalk: DelayedCrosstalkConfig | None = None
    afterpulse: AfterpulseConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(
            self.maximum_generations,
            NonnegativeInteger,
            "CorrelatedAvalancheConfig.maximum_generations",
        )
        _require_optional_exact(
            self.direct_crosstalk,
            DirectCrosstalkConfig,
            "CorrelatedAvalancheConfig.direct_crosstalk",
        )
        _require_optional_exact(
            self.delayed_crosstalk,
            DelayedCrosstalkConfig,
            "CorrelatedAvalancheConfig.delayed_crosstalk",
        )
        _require_optional_exact(
            self.afterpulse,
            AfterpulseConfig,
            "CorrelatedAvalancheConfig.afterpulse",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeSmearingConfig:
    relative_sigma: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.relative_sigma,
            NonnegativeFloat,
            "ChargeSmearingConfig.relative_sigma",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeConfig:
    dark_count: DarkCountConfig | None = None
    timing_jitter: TimingJitterConfig | None = None
    correlated_avalanches: CorrelatedAvalancheConfig | None = None
    smearing: ChargeSmearingConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_exact(
            self.dark_count,
            DarkCountConfig,
            "ChargeConfig.dark_count",
        )
        _require_optional_exact(
            self.timing_jitter,
            TimingJitterConfig,
            "ChargeConfig.timing_jitter",
        )
        _require_optional_exact(
            self.correlated_avalanches,
            CorrelatedAvalancheConfig,
            "ChargeConfig.correlated_avalanches",
        )
        _require_optional_exact(
            self.smearing,
            ChargeSmearingConfig,
            "ChargeConfig.smearing",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TpcFebSnrPulseConfig:
    fast_time_constant_ns: PositiveFloat
    slow_time_constant_ns: PositiveFloat
    support_time_ns: PositiveFloat
    peak_voltage_mv_per_pe: FiniteFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.fast_time_constant_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.fast_time_constant_ns",
        )
        _require_exact(
            self.slow_time_constant_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.slow_time_constant_ns",
        )
        _require_exact(
            self.support_time_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.support_time_ns",
        )
        _require_exact(
            self.peak_voltage_mv_per_pe,
            FiniteFloat,
            "TpcFebSnrPulseConfig.peak_voltage_mv_per_pe",
        )
        if (
            self.slow_time_constant_ns.value
            <= self.fast_time_constant_ns.value
        ):
            raise ValueError(
                "slow time constant must exceed fast time constant"
            )
        if self.peak_voltage_mv_per_pe.value == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class VetoPduPulseConfig:
    gaussian_center_ns: FiniteFloat
    gaussian_width_ns: PositiveFloat
    edge_offset_1_ns: FiniteFloat
    edge_width_1_ns: PositiveFloat
    edge_offset_2_ns: FiniteFloat
    edge_width_2_ns: PositiveFloat
    support_time_ns: PositiveFloat
    peak_voltage_mv_per_pe: FiniteFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.gaussian_center_ns,
            FiniteFloat,
            "VetoPduPulseConfig.gaussian_center_ns",
        )
        _require_exact(
            self.gaussian_width_ns,
            PositiveFloat,
            "VetoPduPulseConfig.gaussian_width_ns",
        )
        _require_exact(
            self.edge_offset_1_ns,
            FiniteFloat,
            "VetoPduPulseConfig.edge_offset_1_ns",
        )
        _require_exact(
            self.edge_width_1_ns,
            PositiveFloat,
            "VetoPduPulseConfig.edge_width_1_ns",
        )
        _require_exact(
            self.edge_offset_2_ns,
            FiniteFloat,
            "VetoPduPulseConfig.edge_offset_2_ns",
        )
        _require_exact(
            self.edge_width_2_ns,
            PositiveFloat,
            "VetoPduPulseConfig.edge_width_2_ns",
        )
        _require_exact(
            self.support_time_ns,
            PositiveFloat,
            "VetoPduPulseConfig.support_time_ns",
        )
        _require_exact(
            self.peak_voltage_mv_per_pe,
            FiniteFloat,
            "VetoPduPulseConfig.peak_voltage_mv_per_pe",
        )
        if self.peak_voltage_mv_per_pe.value == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PureWaveformConfig:
    model: TpcFebSnrPulseConfig | VetoPduPulseConfig

    def __post_init__(self) -> None:
        _require_one_of_exact(
            self.model,
            (TpcFebSnrPulseConfig, VetoPduPulseConfig),
            "PureWaveformConfig.model",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ZeroNoiseConfig:
    """Select the exact all-zero noise algorithm."""


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class WhiteNoiseConfig:
    rms_mv: PositiveFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.rms_mv,
            PositiveFloat,
            "WhiteNoiseConfig.rms_mv",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PsdNoiseConfig:
    frequency_left_edges_hz: tuple[NonnegativeFloat, ...]
    frequency_stop_hz: PositiveFloat
    power_density_mv2_per_hz: tuple[NonnegativeFloat, ...]

    def __post_init__(self) -> None:
        if type(self.frequency_left_edges_hz) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.frequency_left_edges_hz must be a tuple"
            )
        if type(self.power_density_mv2_per_hz) is not tuple:
            raise TypeError(
                "PsdNoiseConfig.power_density_mv2_per_hz must be a tuple"
            )
        if not self.frequency_left_edges_hz:
            raise ValueError("a PSD requires at least one frequency bin")
        if len(self.frequency_left_edges_hz) != len(
            self.power_density_mv2_per_hz
        ):
            raise ValueError("PSD left-edge and density counts must match")
        for edge in self.frequency_left_edges_hz:
            _require_exact(
                edge,
                NonnegativeFloat,
                "PsdNoiseConfig.frequency_left_edges_hz",
            )
        _require_exact(
            self.frequency_stop_hz,
            PositiveFloat,
            "PsdNoiseConfig.frequency_stop_hz",
        )
        if self.frequency_left_edges_hz[0].value != 0.0:
            raise ValueError("PSD frequency coverage must start at zero")
        if any(
            right.value <= left.value
            for left, right in zip(
                self.frequency_left_edges_hz,
                self.frequency_left_edges_hz[1:],
            )
        ):
            raise ValueError(
                "PSD frequency left edges must be strictly increasing"
            )
        if (
            self.frequency_left_edges_hz[-1].value
            >= self.frequency_stop_hz.value
        ):
            raise ValueError("PSD frequency stop must exceed its final left edge")
        for density in self.power_density_mv2_per_hz:
            _require_exact(
                density,
                NonnegativeFloat,
                "PsdNoiseConfig.power_density_mv2_per_hz",
            )
        if not any(
            density.value > 0.0
            for density in self.power_density_mv2_per_hz
        ):
            raise ValueError("use ZeroNoiseConfig for an all-zero PSD")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NoiseWaveformConfig:
    model: ZeroNoiseConfig | WhiteNoiseConfig | PsdNoiseConfig

    def __post_init__(self) -> None:
        _require_one_of_exact(
            self.model,
            (
                ZeroNoiseConfig,
                WhiteNoiseConfig,
                PsdNoiseConfig,
            ),
            "NoiseWaveformConfig.model",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogSaturationConfig:
    minimum_mv: FiniteFloat | None = None
    maximum_mv: FiniteFloat | None = None

    def __post_init__(self) -> None:
        _require_optional_exact(
            self.minimum_mv,
            FiniteFloat,
            "AnalogSaturationConfig.minimum_mv",
        )
        _require_optional_exact(
            self.maximum_mv,
            FiniteFloat,
            "AnalogSaturationConfig.maximum_mv",
        )
        if self.minimum_mv is None and self.maximum_mv is None:
            raise ValueError("analog saturation requires at least one bound")
        if (
            self.minimum_mv is not None
            and self.maximum_mv is not None
            and self.minimum_mv.value >= self.maximum_mv.value
        ):
            raise ValueError("analog saturation minimum must be below maximum")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AnalogWaveformConfig:
    saturation: AnalogSaturationConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_exact(
            self.saturation,
            AnalogSaturationConfig,
            "AnalogWaveformConfig.saturation",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DigitizedWaveformConfig:
    bit_depth: PositiveInteger
    input_min_mv: FiniteFloat
    input_max_mv: FiniteFloat
    analog_gain_db: NonnegativeFloat

    def __post_init__(self) -> None:
        _require_exact(
            self.bit_depth,
            PositiveInteger,
            "DigitizedWaveformConfig.bit_depth",
        )
        _require_exact(
            self.input_min_mv,
            FiniteFloat,
            "DigitizedWaveformConfig.input_min_mv",
        )
        _require_exact(
            self.input_max_mv,
            FiniteFloat,
            "DigitizedWaveformConfig.input_max_mv",
        )
        _require_exact(
            self.analog_gain_db,
            NonnegativeFloat,
            "DigitizedWaveformConfig.analog_gain_db",
        )
        if self.bit_depth.value > 16:
            raise ValueError("bit_depth must be between 1 and 16")
        if self.input_min_mv.value >= self.input_max_mv.value:
            raise ValueError("ADC input minimum must be below maximum")
        if self.analog_gain_db.value > 40.0:
            raise ValueError("analog_gain_db must be between 0 and 40")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutConfig:
    sampling: SamplingConfig
    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(
            self.sampling,
            SamplingConfig,
            "ReadoutConfig.sampling",
        )
        _require_optional_exact(
            self.charge,
            ChargeConfig,
            "ReadoutConfig.charge",
        )
        _require_optional_exact(
            self.pure_waveform,
            PureWaveformConfig,
            "ReadoutConfig.pure_waveform",
        )
        _require_optional_exact(
            self.noise_waveform,
            NoiseWaveformConfig,
            "ReadoutConfig.noise_waveform",
        )
        _require_optional_exact(
            self.analog_waveform,
            AnalogWaveformConfig,
            "ReadoutConfig.analog_waveform",
        )
        _require_optional_exact(
            self.digitized_waveform,
            DigitizedWaveformConfig,
            "ReadoutConfig.digitized_waveform",
        )
```

These names establish ownership, not calibration defaults. Exact calibrated
values are accepted by focused scientific work orders. A sampling policy is
never inferred from hit extrema or tensor shape: doing so would discard empty
tail bins and make dense batch shape data-dependent. `ZeroNoiseConfig` selects
a real deterministic algorithm; `AnalogWaveformConfig(saturation=None)`
selects the exact linear `pure + noise` transfer. Neither exists merely to
reserve a future hierarchy. The sketch elides repetitive exact-component-type
checks; production `__post_init__` methods must reject wrong config/scalar
classes before reading their values.

`DirectCrosstalkConfig` and `DelayedCrosstalkConfig` are distinct even when
their configured delay families happen to match. Each owns its own
nonnegative Poisson mean and exact causal delay model; neither name is merely a
synonym for offset zero or a later bin. A zero mean is a draw-free identity.
An exact fixed delay of zero produces an in-bin edge after uniform
phase marginalization, while fixed, exponential, and zero-clipped normal delay
models may place children in later bins. `NormalDelayConfig.location_ns` is
the location of its latent Gaussian, not the mean of the resulting clipped
physical delay. Its `sigma_ns` must be strictly positive; an exact zero-width
law is expressed with `FixedDelayConfig`. `afterpulse=None` disables AP. A present
`AfterpulseConfig(recovery=None)` retains AP with unit deposited charge, and a
present recovery record selects the exact exponential recovery response
documented below. None of these records carries a persistence or execution
policy.

Every physical-delay model in the readout simulation is causal and must define
a nonnegative realized delay. That is a shared scientific and preparation
invariant, not a universal silent clamp: fixed inputs are validated,
exponential laws have nonnegative support by definition, and the normal family
explicitly includes clipping in its declared law. Common kernel preflight must
reject any prepared negative-offset mass or underflow category rather than
repairing an invalid model. Timing jitter is a signed displacement and is not
a physical-delay model, so this invariant does not apply to it.

The caller-facing spectral contract is only `PsdNoiseConfig`. Callers provide
one left edge per PSD bin, one separate exclusive frequency stop, and one-sided
absolute power density in `mV^2/Hz`; they do not provide FFT frequencies, FFT
amplitudes, complex coefficients, FFT length, or an implementation scale. For
left edges `f[i]`, density `S[i]` applies on `[f[i], f[i + 1])`, and the final
density applies on `[f[-1], frequency_stop_hz)`. The PSD grid is independent of
the requested record length. Request preflight uses `SamplingConfig` to
construct left-closed/right-open target integration intervals, requires source
coverage from zero through Nyquist, and integrates the piecewise-constant
supplied PSD over those intervals. TensorDSLab may use an inverse real FFT
privately to synthesize the fixed-length noise tensor, but that mechanism is
absent from the public config and does not change the input's PSD semantics.
The integrated PSD outside the deliberately suppressed target DC cell
determines the noise variance; there is no second RMS or SNR scale in
`PsdNoiseConfig` and discarded DC power is not redistributed.
Density values are interval densities, not point samples evaluated at their
left-edge coordinates.
The supplied PSD describes the effective noise at the common analog reference
plane after any intended front-end/anti-alias response. Source coverage above
Nyquist is not folded or silently aliased into band; it is outside the target
discrete process. Modeling that analog transfer explicitly would be a separate
accepted submodel.

The MVP sampling contract fixes example-local start at zero, left-closed and
right-open bins, and dropping hits outside `[0, window_stop_ps)`. Callers
therefore choose only period and count. The future bridge computes numeric bin
indices from those values before it constructs the semantic timestamp axis;
validation must report `underflow_hit_count` for normalized times below zero
and `overflow_hit_count` for times at or beyond the exclusive stop rather than
hiding either loss. These names are distinct from arithmetic overflow. Absolute
G4 origin and trigger alignment remain explicit bridge provenance, not
additional meanings hidden in `SamplingConfig`.

Picoseconds are the single numeric *time* execution unit in this architecture.
Preflight normalizes nanosecond-valued jitter, pulse, and afterpulse config
values to floating picoseconds once before any RNG draw or tensor write.
Waveform voltage is expressed in mV, PSD frequency left edges and exclusive
stop in Hz, and absolute one-sided PSD values in `mV^2/Hz`; the PSD rebinning
boundary performs the explicit frequency/time conversion implied by
`SamplingConfig`. Kernels never combine values expressed in different units
and never parse unit-bearing coordinate strings.

### No Generic `Config(ABC)`

The rebuild does not introduce a universal `Config(ABC)` or marker base. These
records share conventions, not one useful substitutable behavior. A generic
base would encourage APIs typed as `Config`, weaken exact product/config
pairing, and add inheritance/dataclass complexity without guaranteeing correct
validation.

If a real polymorphic consumer appears later, add the narrowest protocol or
abstract type at that boundary. Serialization alone does not justify a base;
artifact codecs can operate on exact config types.

### Runtime Inputs Are Not Scientific Config

The following are explicit builder/runtime inputs or derived facts:

- requested retained products;
- root seed and private fixed numeric operation streams;
- floating dtype;
- source axes, timestamps, shape, and device;
- destination storage, future workspace, and stream;
- chunking and scheduling; and
- future persistence and IO policy.

Future typed calibration tensors are scientific inputs rather than execution
controls, but they also do not belong inside the MVP's immutable scalar config
records. This distinction leaves room for channel-aligned GPU parameters
without making ordinary scalar configuration difficult for collaborators.

Subsystem-specific named scientific presets may be classmethods such as
`ReadoutConfig.darkside20k_tpc_nominal()` and
`ReadoutConfig.darkside20k_veto_nominal()` only after each exact calibration is
reviewed. A single nominal preset must not blur the distinct response families.
Do not add loose default constants or a scientifically unqualified `default()`.

## Private Product Builders

Private product operations are exact and independently testable with valid
fixtures, but they receive already-preflighted values from
`simulate_readout(...)`. They do not repeat the public boundary or promise a
supported direct-call API:

```python
def _product_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    seed: int | None,
    floating_dtype: torch.dtype,
) -> Charge:
    ...


def _product_pure_waveform(
    charge: Charge,
    *,
    sampling: SamplingConfig,
    config: PureWaveformConfig,
) -> PureWaveform:
    ...


def _product_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: NoiseWaveformConfig,
    seed: int | None,
    floating_dtype: torch.dtype,
) -> NoiseWaveform:
    ...


def _product_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    config: AnalogWaveformConfig,
) -> AnalogWaveform:
    ...


def _product_digitized_waveform(
    analog: AnalogWaveform,
    *,
    config: DigitizedWaveformConfig,
) -> DigitizedWaveform:
    ...
```

The naming split is intentional: `_product_*` constructs one semantic product,
while `_simulate_*` names a private scientific submodel used inside a product
builder. Neither family is public; `simulate_readout(...)` remains the one
ordinary collaborator-facing simulation API.

The two pointwise waveform-tail producers own their product arithmetic directly.
Do not add `_apply_analog_saturation(...)`, `_digitize(...)`, or another
one-line Python wrapper merely to rename either expression. Here *kernel*
means the fused backend execution of a product producer, not another layer in
the Python API.

After structural and config preflight, `_product_analog_waveform(...)` evaluates
one elementwise product expression:

```text
analog[i] = clamp(
    pure[i] + noise[i],
    analog_minimum,
    analog_maximum,
)
```

Either bound may be absent. When both are absent, the expression is simply
addition. The producer returns one new `AnalogWaveform` with guaranteed fresh
storage independent of `pure.tensor` and `noise.tensor`; it must not materialize
a target-sized sum merely to clamp it in a second eager operation.

`_product_digitized_waveform(...)` computes its scalar transfer constants once
during preflight:

```text
maximum_code = 2**bit_depth - 1
gain = 10**(analog_gain_db / 20)
span = input_max_mv - input_min_mv
slope = gain * maximum_code / span
intercept = -input_min_mv * maximum_code / span
```

It then evaluates one elementwise product expression:

```text
digitized[i] = int32(clamp(
    analog[i] * slope + intercept,
    0,
    maximum_code,
))
```

The clamp occurs in floating point before conversion. Because every clamped
value is nonnegative, float-to-`torch.int32` conversion implements the accepted
truncation rule without a separate target-sized `trunc` tensor. The producer
returns one new `DigitizedWaveform` with guaranteed fresh storage independent
of `analog.tensor`; no target-sized gained, clipped, or scaled waveform is a
semantic or private intermediate.

The normal materialized waveform tail therefore has exactly two semantic
product steps:

```text
PureWaveform + NoiseWaveform
  -> _product_analog_waveform(...)
  -> AnalogWaveform
  -> _product_digitized_waveform(...)
  -> DigitizedWaveform
```

This remains true when only `DigitizedWaveform` is retained: the analog product
is computed once as a private prerequisite and is then allowed to become
unreachable. Do not fuse across the `AnalogWaveform` product boundary in the
MVP. Such fusion would require a separate proof that requesting or retaining
`AnalogWaveform` cannot change digitized values, product execution, autograd,
or lifetime behavior.

`_product_noise_waveform(...)` uses `Photoelectrons` only as the authoritative
axes/device/shape reference; it does not read PE counts as a noise input.

Private charge subfunctions receive their exact config:

```python
def _simulate_dark_counts(
    counts: torch.Tensor,
    *,
    sampling: SamplingConfig,
    config: DarkCountConfig,
    seed: int | None,
) -> torch.Tensor:
    ...


def _simulate_timing_jitter(
    counts: torch.Tensor,
    *,
    sample_dimension: int,
    sampling: SamplingConfig,
    config: TimingJitterConfig,
    seed: int | None,
) -> torch.Tensor:
    ...


def _simulate_correlated_avalanches(
    seed_avalanches: torch.Tensor,
    *,
    sample_dimension: int,
    sampling: SamplingConfig,
    floating_dtype: torch.dtype,
    config: CorrelatedAvalancheConfig,
    seed: int | None,
) -> _CorrelatedAvalancheResult:
    ...


def _simulate_charge_smearing(
    charge_pe: torch.Tensor,
    charge_square_sum: torch.Tensor,
    *,
    config: ChargeSmearingConfig,
    seed: int | None,
) -> torch.Tensor:
    ...
```

Only operations whose numerical behavior depends on sample timing receive
`SamplingConfig`. Operations that shift values along the sample axis also
receive the already-resolved numeric `sample_dimension`; hot-path code does not
look up timestamp strings. The coupled cascade additionally receives the
selected floating dtype because it constructs the S1/S2 ledgers. No subfunction
receives `ReadoutConfig` merely to reach one nested value. A private simulation
accepts `seed=None` only on its documented draw-free identity path; complete
public preflight rejects `None` before any effective stochastic branch. Every
stochastic leaf uses the same public root seed with its own globally unique
fixed numeric operation stream; product producers never share a mutable
sequential stream between leaves.
`_product_charge(...)` is the private typed product producer named by the
scientific contract. The public way to request that result is
`simulate_readout(..., products=[Charge], ...)`.

## Public Builder

The target signature is:

```python
def simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    seed: int | None = None,
    floating_dtype: torch.dtype = torch.float32,
) -> ReadoutCollection:
    ...
```

The public surface accepts an ordinary root seed rather than a TensorDSLab RNG
object. It must be an exact non-boolean Python `int` in `[0, 2**64)`. `None` is
valid only when the effective requested closure is deterministic. Fixed
private numeric operation streams isolate random draws; users do not construct
or manage them. The MVP exposes neither a custom RNG object nor a public
`torch.Generator` parameter.

Conceptual orchestration:

```python
requested_items = tuple(products)  # consume exactly once
_require_nonempty_unique_product_request(requested_items)
requested = frozenset(requested_items)

need_digitized = DigitizedWaveform in requested
need_analog = AnalogWaveform in requested or need_digitized
need_pure = PureWaveform in requested or need_analog
need_noise = NoiseWaveform in requested or need_analog
need_charge = Charge in requested or need_pure

_preflight_request(
    photoelectrons,
    requested=requested,
    config=config,
    seed=seed,
    floating_dtype=floating_dtype,
)

charge = (
    _product_charge(
        photoelectrons,
        sampling=config.sampling,
        config=_require_config(config.charge, Charge),
        seed=seed,
        floating_dtype=floating_dtype,
    )
    if need_charge
    else None
)

pure = (
    _product_pure_waveform(
        _require_value(charge, Charge),
        sampling=config.sampling,
        config=_require_config(config.pure_waveform, PureWaveform),
    )
    if need_pure
    else None
)

noise = (
    _product_noise_waveform(
        photoelectrons,
        sampling=config.sampling,
        config=_require_config(config.noise_waveform, NoiseWaveform),
        seed=seed,
        floating_dtype=floating_dtype,
    )
    if need_noise
    else None
)

analog = (
    _product_analog_waveform(
        _require_value(pure, PureWaveform),
        _require_value(noise, NoiseWaveform),
        config=_require_config(config.analog_waveform, AnalogWaveform),
    )
    if need_analog
    else None
)

digitized = (
    _product_digitized_waveform(
        _require_value(analog, AnalogWaveform),
        config=_require_config(
            config.digitized_waveform,
            DigitizedWaveform,
        ),
    )
    if need_digitized
    else None
)

retained = tuple(
    value
    for value in (
        photoelectrons,
        charge,
        pure,
        noise,
        analog,
        digitized,
    )
    if value is not None and type(value) in requested
)
return ReadoutCollection(fields=retained)
```

The fixed local tuple gives equivalent request sets the same mechanical
construction order while remaining nonsemantic to `ReadoutCollection`.
Caller iterable order never affects field values, `field_types`, mechanical
mapping iteration, or the documented collection contract. The tuple is private
assembly code, not a public canonical field sequence or registry.

Preflight completes before the first RNG draw or tensor write. It validates:

- source deep-value validity;
- product request type, uniqueness, and nonemptiness;
- exact `SamplingConfig` validity and agreement between its period/count policy
  and the validated source `SampleAxis`, without regenerating all coordinates;
- every config required by the transitive closure;
- timestamp grammar and timing suitability for enabled operations;
- source device and supported tensor layout;
- `floating_dtype` is exactly `torch.float32` or `torch.float64` when the
  closure generates a floating product; and
- analog saturation bounds and digitizer `maximum_code`, `gain`, `span`,
  `slope`, and `intercept` are valid and representable for the selected
  execution dtype before either waveform-tail producer launches; and
- exact seed type/range, a present seed when an effective enabled submodel is
  stochastic, and positional RNG/backend compatibility.

`ReadoutConfig(sampling=sampling)` is a valid uniform config argument for a
truth-only request, and `seed=None` is valid whenever the requested closure is
deterministic. Irrelevant product configs and runtime controls are neither
consumed nor allowed to perturb common product values.

A request such as this fails before work:

```python
simulate_readout(
    photoelectrons,
    products=[DigitizedWaveform],
    config=config_without_digitizer,
    seed=1234,
)
# ValueError: DigitizedWaveform requires digitization configuration
```

A stochastic closure without a seed also fails before work:

```python
simulate_readout(
    photoelectrons,
    products=[Charge],
    config=stochastic_charge_config,
    seed=None,
)
# ValueError: this readout request requires a seed
```

The builder performs no IO, loading, persistence, DAG scheduling, or implicit
movement/cast of an existing input field. Generated products use their declared
output dtypes.

## Scientific Chain

The selected rebuild computation is:

```text
`Photoelectrons` payload
  -> optional dark-count avalanche seeds
  -> optional private timing redistribution
  -> optional fixed-K coupled correlated-avalanche simulation
       -> integer count frontier for branching
       -> floating S1 deposited-charge ledger
       -> floating S2 charge-square-sum ledger
  -> optional terminal charge smearing from S1 and S2
  -> completed `Charge` product
  -> PureWaveform

truth field axes/device/shape + NoiseWaveformConfig
  -> NoiseWaveform

PureWaveform + NoiseWaveform
  -> AnalogWaveform
  -> optional DigitizedWaveform
```

### Timing Jitter Inside Charge Simulation

When its block executes, timing jitter redistributes the then-current private
primary-avalanche working counts after any effective dark-count block. It
therefore affects photon-origin truth seeds and any dark-count seeds that are
present without mutating or relabeling the public truth `Photoelectrons` field.
For source bin `s`, target bin `t`, shift
`k = t - s`, sample period `T`, latent source-bin phase
`U ~ Uniform([0, T))`, and jitter `J ~ Normal(0, sigma)`, the latent source
time is `s * T + U`; `s * T` is the source bin's left edge. Preflight first
normalizes `T` and `sigma` to floating picoseconds:

```text
target bin t = [t*T, (t + 1)*T)
p(target=t) = P(t*T <= s*T + U + J < (t + 1)*T)
            = P(k*T <= U + J < (k + 1)*T)
```

The latent phase is required because dense truth counts no longer retain each
PE's sub-bin time. The implementation samples aggregate target/drop buckets
rather than materializing a jagged row per PE.

Accepted policies:

- `sigma_ns == 0` is an exact logical identity and consumes no jitter draws;
- out-of-window shifted truth and dark-count seeds are dropped;
- conservation includes an explicit dropped bucket; and
- the truth `Photoelectrons` tensor is never mutated or replaced.

Moving jitter into charge simulation changes the old public timing-transform
boundary. The synchronized parity document compares truth `Photoelectrons` to
the private jitter diagnostic or to requested `Charge`; it does not claim that
jitter produces a new public `Photoelectrons` value.

### Charge Response

The private order is:

```python
charge = photoelectrons.tensor
charge_square_sum: torch.Tensor | None = None

if (
    config.dark_count is not None
    and config.dark_count.rate_hz.value != 0.0
):
    charge = _simulate_dark_counts(charge)

if (
    config.timing_jitter is not None
    and config.timing_jitter.sigma_ns.value != 0.0
):
    charge = _simulate_timing_jitter(charge)

if config.correlated_avalanches is not None:
    correlated = _simulate_correlated_avalanches(charge)
    charge = correlated.S1
    charge_square_sum = correlated.S2

if (
    config.smearing is not None
    and config.smearing.relative_sigma.value != 0.0
):
    charge = charge.to(dtype=floating_dtype)
    charge = _simulate_charge_smearing(
        charge,
        charge if charge_square_sum is None else charge_square_sum,
    )

return Charge(
    tensor=charge.to(dtype=floating_dtype),
    axes=photoelectrons.axes,
)
```

Lowercase `charge` remains a `torch.Tensor` throughout `_product_charge(...)`.
It is the single evolving payload, not a `Charge` instance and not a durable
product identity. Each entered optional block completely replaces that tensor
with its stage result. When an enclosing condition is false, the block performs
no call or assignment and `charge` remains the exact preceding tensor. The
correlated block alone needs a private `_CorrelatedAvalancheResult`; it replaces
`charge` with the result's floating `S1` ledger and retains `S2` only as
`charge_square_sum` for a possible later smearing block. The function returns
the completed uppercase `Charge` product directly, so one local name never
changes between tensor and product types.

Valid chains therefore include, among others:

```text
truth -> dark counts -> timing jitter -> correlated avalanches -> smearing -> Charge
truth -> smearing -> Charge
truth -> timing jitter -> correlated avalanches -> Charge
truth -> dark counts -> smearing -> Charge
truth -> Charge
```

If smearing runs without a preceding correlated stage, every root in `charge`
has unit response, so one floating conversion supplies both `S1` and `S2`
without constructing a correlated result. If smearing is absent, the terminal
product construction performs the required floating conversion directly. A
configured smearing model draws only after every selected correlated generation
has updated both ledgers.

Dark counts use independent per-cell Poisson counts:

```text
lambda_per_cell = rate_hz * sampling.sample_period_ps.value * 1e-12
dark_count ~ Poisson(lambda_per_cell)
```

A zero rate is an exact zero contribution: the dark block is skipped, no
`dark` variable is bound, and no dark-count draw is consumed.

For an unbounded homogeneous Poisson dark process, independent timing
displacement preserves the homogeneous law. The finite MVP window is more
subtle: it generates seeds only inside the window and drops values jittered
outside, so there is no compensating influx from dark events just beyond an
edge. The MVP accepts and measures this boundary truncation rather than
claiming exact edge invariance. A later stationary-boundary treatment may use
a configured halo and crop, but it is not silently inferred here.

This order improves the isolated dark-count-plus-jitter comparison with the IV
donor when gate, latent phase, binning, and drop policies agree. It does not
make the full chain equivalent. IV-DSLab creates recursive correlated
avalanches before its later independent PE timing operation, whereas the
rebuild jitters truth and dark seeds first and applies its prepared causal edge
kernels during the coupled cascade. Parent-child timing covariance therefore
remains an intentional parity difference.

### Fixed-Generation Correlated-Avalanche Baseline

Status: sole active correlated-avalanche algorithmic baseline for the rebuild.
It remains Design-only and does not dispatch implementation or add another
public API. It is the only avalanche algorithm available to a rebuild work
order and cannot be replaced without a new user-authorized TensorDSLab Design
decision.

The selected model is a synchronous, fixed-generation, unit-count
branching process with separate floating deposited-charge response over the
complete dense post-binned grid. Generation zero is the private integer seed
grid after dark-count addition and timing jitter. A caller-selected nonnegative
`maximum_generations = K` gives the number of offspring generations evaluated
after those roots:

```text
K = 0  -> roots only
K = 1  -> roots plus their direct children
K = 2  -> roots, children, and grandchildren
```

`K` is scientific configuration because changing it changes the returned
distribution. The algorithm does not infer it from an extinction
test, tolerance, finite-window bound, or hardware execution plan. A smarter
selection or exact-termination policy is deferred. Implementations may avoid
physical work for a provably zero frontier, but doing so must not change the
fixed-`K` semantics.

Each generation is organized around two collaborator-facing questions:

1. **How many avalanches does the current frontier produce?**
2. **Which sample bins receive those avalanches?**

The separate DiCT, DeCT, and AP draws answer both questions. Their retained
children are then summed into the next frozen frontier; their right-overflow
children are recorded but do not continue.

`K=0` performs no correlated-mechanism draw even when mechanism configs are
present. Such configs still validate structurally, but they do not make an
otherwise deterministic request require a seed. Likewise, a zero DiCT/DeCT
mean or zero AP probability is a draw-free identity for that mechanism.

The exact `CorrelatedAvalancheConfig` and mechanism records are defined in the
configuration section above. Structural `None` disables a mechanism. With all
three absent, the operation is the exact identity for every `K` and needs no
generation storage or stochastic draw. Its logical final frontier is the root
grid when `K = 0` and zero when `K >= 1`, even when an optimized identity path
materializes neither value.

The recursive state distinguishes avalanche multiplicity from deposited
charge:

```text
F[g, t]       : int64 avalanche count in generation g and sample t
S1[t]         : floating accumulated pre-smearing PE-equivalent charge
S2[t]         : floating accumulated sum of squared response weights
```

Every successful primary, dark-count, DiCT, DeCT, or AP avalanche contributes
exactly one integer count to its generation frontier. `F[g]`, never `S1` or
`S2`, determines every enabled offspring law.

`S1`, `S2`, all prepared rate fields, and AP charge diagnostics use the exact
`floating_dtype` selected for `Charge`; no accumulator silently widens or
narrows. Config-derived CDF/PMF preparation may use host binary64 arithmetic
before one explicit checked cast because configs are small semantic records,
not payload tensors. Preflight must prove every prepared probability and rate
finite and representable in the selected execution dtype. A nonfinite ledger or
checked `int64` count overflow is a hard algorithm failure and never a valid
partial `Charge` result.

The response model makes this explicit microcell assumption:

- every DiCT or DeCT child triggers a microcell distinct from its parent that
  has not previously fired in the modeled cascade and is fully recovered, so
  the child deposits exactly one PE-equivalent charge; and
- every AP child retriggers its parent microcell, so it still contributes one
  avalanche count but deposits delay-dependent recovery-weighted charge.

The CT rule is a low-occupancy, effectively unlimited-fresh-cell
approximation. Crosstalk collisions, finite microcell exhaustion, or a CT seed
landing on a previously fired or recovering destination require microcell
identity and recovery state, which the dense post-binned frontier does not
carry and which remain outside this initial model. Recovery-dependent source
emission is also excluded: even an AP-born avalanche has the same future
offspring law as every other unit-count avalanche.

This is **unmarked recursion with recovery-weighted AP deposited charge**, not
recovery-dependent marked recursion. The recovery weight is consumed into the
floating charge accumulator and is not carried in `F[g + 1]` or used to scale
that avalanche's later DiCT, DeCT, or AP production. Every root, dark count,
DiCT, DeCT, and AP parent therefore uses the same recovery-independent DiCT
mean, DeCT mean, and AP fire probability. If recovery classes are ever used to
refine within-offset AP charge heterogeneity, they are transient sampling
categories that collapse immediately into the ordinary unmarked `F[g + 1]`;
they are not recursive state. The unit-count rule is what permits the three
integer contribution tensors to merge into one next-generation parent
frontier.

`F[g]` is frozen for the complete generation update. No child produced by one
mechanism may feed DiCT, DeCT, or AP again until `F[g + 1]` is complete. This
also applies to a zero-offset child that occupies its parent's sample bin:
sample offset zero does not mean genealogical generation zero.

For that current frontier, the three mechanisms are conditionally independent
and are drawn separately. DiCT and DeCT are two physical or calibrated modes of
correlated crosstalk; they are not synonyms for same-bin and later-bin
children. Each mode `m` in `{direct, delayed}` has its own fixed mean offspring
count `lambda_m` and its own exact `FixedDelayConfig`,
`ExponentialDelayConfig`, or `NormalDelayConfig`. These are
sampling-independent causal physical-delay models. Preflight combines each
one with the exact `SamplingConfig` to prepare an integer-offset PMF
`q_m[d; sampling]`.

```text
FixedDelayConfig:       Delta_m = delay_ns
ExponentialDelayConfig: Delta_m ~ Exponential(mean_delay_ns)
NormalDelayConfig:      X_m ~ Normal(location_ns, sigma_ns)
                        Delta_m = max(X_m, 0)
```

The normal family is a **zero-clipped (rectified) normal**, not a
truncated-and-renormalized normal and not a folded `abs(X_m)` distribution. It
therefore has a deliberate prompt atom:

```text
P(Delta_m = 0) = Phi(-location_ns / sigma_ns)

F_Delta(x) = 0,                                      x < 0
F_Delta(x) = Phi((x - location_ns) / sigma_ns),       x >= 0
```

Negative latent Gaussian mass maps to an exact zero physical delay. No
strictly-positive epsilon is introduced. In particular, a latent location of
zero assigns one half of the delay probability to that zero atom. This can be a
substantial prompt component when `location_ns` and `sigma_ns` are comparable,
so calibration must select the family and parameters consciously. A future
truncated-normal law would require a different config and scientific decision
because it has no zero atom and a different positive-delay density.

The binned phase policy is independent per parent-child edge. Conceptually,
every realized CT child edge or fired AP edge receives a fresh
`U_edge ~ Uniform([0, T))`. It is independent of every sibling edge, mechanism,
and generation, is integrated into the prepared category law, and is never
stored or inherited. For sample period `T` and physical edge delay `Delta_m`:

```text
q_m[d; sampling]
    = P(d * T <= U_edge + Delta_m < (d + 1) * T),  d in Z_{≥0}
```

Both CT modes are causal: `Delta_m >= 0`, so their prepared kernels have no
negative-offset or underflow category. A fixed zero-delay DiCT model therefore
lands in offset zero with probability one; a cross-bin DiCT model requires an
explicitly nonzero causal delay. The clipped-normal zero atom likewise lands
in offset zero with probability one after phase marginalization. Preflight
derives its complete offset PMF and analytic right tail from the clipped law;
the aggregate simulation does not draw a target-sized normal tensor and clamp
it on the hot path. Every prepared delay kernel must have nonnegative support
and satisfy the accepted PMF-plus-right-tail normalization tolerance, or
preflight fails before RNG consumption or writes. The selected TensorDSLab
timing policy does not fold a later independent child-jitter draw into
`Delta_m`. IV's later independent jitter of parent and child rows can produce a
signed post-binned relative displacement, but that is a donor timing divergence
rather than an alternate selected CT kernel. The independent-edge closure
preserves each one-edge offset PMF but intentionally omits covariance from a
shared hidden parent phase among siblings, mechanisms, or successive
generations. Sharing one phase per parent would require marked per-parent state
and would make the aggregate independent destination-Poisson equations below
inexact; it is not part of this dense unmarked model.

Per-mode Poisson thinning and within-mode source superposition give:

```text
R_m[g + 1, u]
    = lambda_m * sum_t F[g, t] * q_m[u - t; sampling]

A_direct_crosstalk[g + 1, u]
    ~ Poisson(R_direct[g + 1, u])

A_delayed_crosstalk[g + 1, u]
    ~ Poisson(R_delayed[g + 1, u])
```

The selected algorithm keeps these as two explicit Poisson draws. It does not
sample `Poisson(R_direct + R_delayed)` and does not recover the modes with a
conditional Binomial split, even though that alternative has the same
conditional joint law. Separate draws keep the scientific bookkeeping and RNG
roles direct. A disabled mode or zero rate produces exact zeros without a draw.

No Gamma latent intensity, Gamma-Poisson mixture, or negative-binomial
offspring law surrounds either supplied mean. Adding one would be a different
scientific model requiring a new Design decision and parity classification.

The audited IV source uses configured `direct_ct = 0.3` as
`lambda_direct = 0.3`: a fixed mean number of DiCT offspring in the following
genealogical generation per unit parent. TensorDSLab does not sample a Gamma
latent intensity around `0.3`, and does not reinterpret `0.3` as the probability
of at least one child. “Fixed” means that the configured scalar is not itself a
per-cell random variable; the IV comparison preset is exactly `0.3`, while later
calibrated presets may use a different reviewed scalar.

For a DiCT-only process with one root on an unbounded retained domain, the
expected total population in generation `g`, summed over destination bins, is
`lambda_direct**g`; its expected position profile is
`lambda_direct**g * convolution_power(q_direct, g)`. With
`lambda_direct = 0.3`, `K = 1` adds mean `0.3`, while the untruncated expected
additional progeny is `0.3 / (1 - 0.3) = 0.428571...`. Finite-window retained
means are smaller and position-dependent because of edge losses. The generation
loop obtains the larger cascade mean only by recursively processing realized
children; it never replaces the fixed direct-offspring mean with `0.428571...`.

DeCT is an optional TensorDSLab model rather than an IV-parity claim. For either
causal crosstalk mode, exact thinning places children beyond the right window
edge in a separate mode-specific overflow bucket. For source bin `t`:

```text
q_m_overflow[t]  = sum(q_m[d; sampling], d >= S - t)

A_m_overflow[g + 1, t]
    ~ Poisson(lambda_m * F[g, t] * q_m_overflow[t])
```

Each mode retains its own overflow draw and diagnostic; modes are not
superimposed there either. `direct_crosstalk=None` or
`delayed_crosstalk=None` disables the corresponding mode structurally and
requires no mode draw or physical zero contribution buffer.

The initial finite-window algorithm uses an absorbing right boundary. Every CT
or AP overflow child is counted and removed before the next generation; its
descendants are not simulated. Because every selected displacement is causal,
an overflow child cannot later return to the retained window. Given the same
in-window roots, this absorbing rule therefore agrees with simulating the
unbounded causal cascade and cropping its retained prefix.

The mathematical CT and AP offset PMFs are defined over nonnegative integer
offsets. A tensor-prepared mode plan must provide `q[0]` through `q[S - 1]`
plus its exact right tail, or obtain the same values from exact CDF differences.
No prepared kernel may collapse possibly retainable offsets into a tail,
truncate an infinite-support law, or silently renormalize it.

AP is also optional. Every parent produces at most one direct AP child with
probability `p_ap`. Conditional on firing, its nonnegative delay offset follows
`q_ap`, prepared from the ordinary
`Delta_ap ~ Exponential(AfterpulseConfig.mean_delay_ns)` law and the same
independent-edge phase closure. The exponential law satisfies the same shared
causal-delay invariant by construction, and its prepared kernel must likewise
have no underflow category and must preserve its complete right tail. For source
bin `t` containing `Q` parents, define
`q_ap_overflow[t] = 1 - sum(q_ap[d], d=0..S-1-t)`. The exact direct outcome law
is:

```text
(A[t, 0], ..., A[t, S - 1 - t], A[t, overflow], A[t, stop])
    ~ Multinomial(
        Q,
        p_ap * q_ap[0],
        ...,
        p_ap * q_ap[S - 1 - t],
        p_ap * q_ap_overflow[t],
        1 - p_ap,
    )
```

Retained AP categories are shifted into their destination bins to form
`A[g + 1]`; `A[overflow]` enters the AP overflow bucket. Every retained AP child
is one unit-count avalanche and receives the full common offspring law in the
next generation. The overflow probability includes both explicit kernel-tail
mass and offsets that cross the right edge from the current source bin. A no-AP
outcome and a fired-but-overflowed AP remain different diagnostics.

AP deposited charge is derived from the same realized multinomial categories,
not from an independent mean field. `afterpulse=None` disables AP completely.
`afterpulse.recovery=None` keeps AP count, timing, and future branching enabled
but gives every retained AP unit deposited charge. A present exact
`AfterpulseRecoveryConfig(time_constant_ns=tau_recovery)` selects:

```text
rho(Delta) = 1 - exp(-Delta / tau_recovery)
```

The recovery response changes deposited charge only. Let `Delta` be the
physical AP delay, `U_edge` the same independent edge-phase marginalization
used to assign this AP outcome, and
`J = floor((U_edge + Delta) / T)` the offset before window clipping. Preflight
prepares:

```text
q_ap[d]
    = P(J = d | AP fired)

h_ap[d]
    = E[rho(Delta) * 1{J = d} | AP fired]

rho_bar_ap[d]
    = h_ap[d] / q_ap[d]  when q_ap[d] > 0
```

`rho_bar_ap[d]` is the conditional mean recovery for that offset category. It
is not `rho(d * T)`, a bin-center evaluation, or a function of an unrelated
timing-jitter displacement. A zero-probability category is never sampled and
needs no recovery division. The response treats `Delta` as the delay
from the AP's immediate parent avalanche and assumes that parent reset the
relevant microcell. It does not reconstruct full same-cell firing history from
channel-level bins; that requires microcell-resolved state.

If `A[g + 1, t, d]` denotes the logical category count drawn from source bin
`t` before shifting, the selected binned response is:

```text
A_ap[g + 1, u]
    = sum(A[g + 1, t, d], t + d = u)

C_ap[g + 1, u]
    = sum(A[g + 1, t, d] * rho_bar_ap[d], t + d = u)
```

The full source-by-offset tensor need not be materialized: one exact
multinomial category scan or fused sampler may accumulate the integer
destination count and floating destination charge together. Applying the
recovery weight before source/offset categories collapse is essential because
different delays can reach the same destination bin.

This is the conditional expected physical charge given the realized binned AP
categories. It preserves the exact mean, AP fire/delay fluctuations,
multinomial cross-bin covariance, and covariance between the AP count frontier,
AP charge, and later count-driven generations. It omits only recovery-amplitude
variation within one offset category, in addition to the timing-latent
covariances already excluded by the unit-count binned model.

When `recovery is None`, preflight uses `rho_bar_ap[d] = 1` for every nonzero
category. Mathematically, `afterpulse_charge` and
`afterpulse_charge_square_sum` both reduce to retained `afterpulse_count` in the
charge dtype, while overflow remains excluded from the retained ledgers. The
implementation's generation-wise floating accumulations use the dtype-aware
ledger tolerances below rather than promising bitwise equality with one final
cast of the cumulative integer count.

For comparison, if `Lambda[t] = p_ap * F[g, t]`, then

```text
E[C_ap[g + 1, u] | F[g]]
    = sum_t Lambda[t] * h_ap[u - t]
```

is a useful analytic validation oracle. Setting the simulated `C_ap` equal to
that convolution would be a different mean-field approximation: it could
deposit positive expected charge when the sampled AP count is zero and would
remove the AP count/charge covariance still available after binning. It is not
the selected event-level algorithm. Likewise, a Poisson AP count is not
interchangeable with the selected bounded multinomial law because it permits
more than one direct AP child per parent.

For overflow-charge diagnostics, preflight also prepares:

```text
h_ap_overflow[t] = sum(h_ap[d], d >= S - t)

rho_bar_ap_overflow[t]
    = h_ap_overflow[t] / q_ap_overflow[t]
      when q_ap_overflow[t] > 0
```

It applies that source-position-dependent conditional mean recovery to the same
sampled overflow count. A zero-probability tail is exact zero and needs no
division. `afterpulse=None` disables AP and every AP count, charge, squared
charge sum, and overflow diagnostic structurally.

The generation update keeps every birth mechanism explicit. DiCT and DeCT use
their two separate Poisson draws and accumulators; AP retains its separate
bounded categorical law. Cross-mode rate superposition and a conditional mode
split are not part of the selected algorithm.

The symmetric `draw_direct_crosstalk`, `draw_delayed_crosstalk`, and
`draw_afterpulses` names below label sampler roles inside
`_simulate_correlated_avalanches(...)`. They are not public transforms or
additional product producers, and an implementation may inline or fuse them
while preserving their separate laws, streams, and diagnostics.

```python
plan = prepare_correlated_avalanche_plan(
    sample_dimension=sample_dimension,
    sampling=sampling,
    floating_dtype=floating_dtype,
    config=config,
)

if plan.all_mechanisms_disabled:
    return identity_avalanche_result(
        seed_avalanches,
        maximum_generations=config.maximum_generations,
        seed_S1=to_charge_dtype(seed_avalanches, dtype=floating_dtype),
        seed_S2=to_charge_dtype(seed_avalanches, dtype=floating_dtype),
    )

frontier_count = seed_avalanches
total_count = copy_to_fresh_total(seed_avalanches)
S1 = to_charge_dtype(seed_avalanches, dtype=floating_dtype)
S2 = to_charge_dtype(seed_avalanches, dtype=floating_dtype)

direct_crosstalk_count = (
    zeros_like(seed_avalanches)
    if plan.direct_crosstalk is not None
    else None
)
direct_crosstalk_overflow_count = (
    zeros_like(seed_avalanches)
    if plan.direct_crosstalk is not None
    else None
)
delayed_crosstalk_count = (
    zeros_like(seed_avalanches)
    if plan.delayed_crosstalk is not None
    else None
)
delayed_crosstalk_overflow_count = (
    zeros_like(seed_avalanches)
    if plan.delayed_crosstalk is not None
    else None
)
afterpulse_count = (
    zeros_like(seed_avalanches)
    if plan.afterpulse is not None
    else None
)
afterpulse_overflow_count = (
    zeros_like(seed_avalanches)
    if plan.afterpulse is not None
    else None
)
afterpulse_charge = (
    zeros_like_charge(seed_avalanches, dtype=floating_dtype)
    if plan.afterpulse is not None
    else None
)
afterpulse_overflow_charge = (
    zeros_like_charge(seed_avalanches, dtype=floating_dtype)
    if plan.afterpulse is not None
    else None
)
afterpulse_charge_square_sum = (
    zeros_like_charge(seed_avalanches, dtype=floating_dtype)
    if plan.afterpulse is not None
    else None
)

for generation_index in range(config.maximum_generations.value):
    children_count = logical_zero

    if plan.direct_crosstalk is not None:
        (
            new_direct_crosstalk_count,
            new_direct_crosstalk_overflow_count,
        ) = draw_direct_crosstalk(
            frontier_count,
            plan.direct_crosstalk,
        )
        direct_crosstalk_count += new_direct_crosstalk_count
        direct_crosstalk_overflow_count += (
            new_direct_crosstalk_overflow_count
        )
        direct_charge = to_charge_dtype(
            new_direct_crosstalk_count,
            dtype=floating_dtype,
        )
        S1 += direct_charge
        S2 += direct_charge
        children_count += new_direct_crosstalk_count

    if plan.delayed_crosstalk is not None:
        (
            new_delayed_crosstalk_count,
            new_delayed_crosstalk_overflow_count,
        ) = draw_delayed_crosstalk(
            frontier_count,
            plan.delayed_crosstalk,
        )
        delayed_crosstalk_count += new_delayed_crosstalk_count
        delayed_crosstalk_overflow_count += (
            new_delayed_crosstalk_overflow_count
        )
        delayed_charge = to_charge_dtype(
            new_delayed_crosstalk_count,
            dtype=floating_dtype,
        )
        S1 += delayed_charge
        S2 += delayed_charge
        children_count += new_delayed_crosstalk_count

    if plan.afterpulse is not None:
        (
            new_afterpulse_count,
            new_afterpulse_overflow_count,
            new_afterpulse_charge,
            new_afterpulse_overflow_charge,
            new_afterpulse_charge_square_sum,
        ) = draw_afterpulses(
            frontier_count,
            plan.afterpulse,
        )
        afterpulse_count += new_afterpulse_count
        afterpulse_overflow_count += new_afterpulse_overflow_count
        afterpulse_charge += new_afterpulse_charge
        afterpulse_overflow_charge += new_afterpulse_overflow_charge
        afterpulse_charge_square_sum += new_afterpulse_charge_square_sum
        S1 += new_afterpulse_charge
        S2 += new_afterpulse_charge_square_sum
        children_count += new_afterpulse_count

    frontier_count = materialize_or_reuse(children_count)
    total_count += frontier_count
```

The unsuffixed mechanism names above are accumulated over all evaluated
offspring generations. Retained count and charge diagnostics use the readout
grid and are indexed by destination bin. Each `*_overflow_count` and
`afterpulse_overflow_charge` uses the same logical grid but is indexed by the
source bin whose child crossed the right boundary. There is no underflow
diagnostic because every selected edge law is causal. Overflow never enters
`frontier_count`, `total_count`, `S1`, `S2`, or a waveform. DiCT and DeCT need no
separate persistent charge diagnostic: their deposited charge is exactly the
floating conversion of their retained count.

The all-disabled identity result reports the root grid as its logical final
frontier for `K=0` and an exact zero frontier for `K>=1`; it may represent the
latter structurally without allocating a public tensor. All diagnostics for an
absent mechanism are structurally absent rather than materialized zeros.

`logical_zero` above is an absent mathematical contribution, not an allocated
zero tensor, and `materialize_or_reuse` stands only for the later execution
plan's choice of an already required generation buffer. Thus a structurally
disabled mechanism has no sampler call, count/charge contribution buffer, or
overflow buffer. With all mechanisms disabled, the implementation takes the
identity path before this loop. The names show scientific roles, not a required
one-buffer-per-name execution plan; later workspace design may safely reuse
storage only after preserving every simultaneously live count and charge role.

The pseudocode names mathematical integer samplers. It does not prescribe
PyTorch's floating distribution APIs, TensorDSLab stream numbers, positional
addresses, raw-word budgets, fusion, or scratch scheduling. Adapting the fixed
algorithm to the package RNG convention is a later design step. Avalanche
counts remain nonnegative `int64` throughout the branching simulation; rates,
probabilities, and deposited charge use separate floating computation. No
floating charge value is converted back into a parent count or sampler
parameter. The eventual implementation must detect or preclude integer
overflow rather than permit wrapping.

For homogeneous parameters, the unwindowed mean reproduction per unit parent
is `lambda_direct + lambda_delayed + p_ap`. Fixed `K` makes the algorithm finite
even when that value is at least one, but does not make explosive count growth
safe or well calibrated. Supported parameter bounds and resource-risk policy
remain focused-stage gates rather than an implicit change to `K`.

The frontier and two ledgers above are semantic roles, not a literal
three-buffer ceiling. Whenever the correlated stage executes, its algorithm
always accumulates the private numerical `S2` ledger; that ledger is not
conditioned on whether smearing is enabled. If the complete correlated stage is
skipped, no correlated result or `S2` tensor is constructed. A later effective
smearing stage instead derives the unit-response identity `S1 == S2` from the
then-current integer `charge` tensor. No gain-smearing draw occurs inside the
generation loop. After generation `K`, the terminal charge rule consumes the
completed ledgers.

Recovery-weighted AP deposits make the old unit-count smearing scale
`sqrt(total_count) * relative_sigma` insufficient. The selected independent
multiplicative per-avalanche Gaussian gain model with category response weights
`w_i` uses:

```text
S1[t] = sum_i w_i                  = charge_pe[t]
S2[t] = sum_i w_i**2               = charge_square_sum[t]

draw[t] | {w_i}
    ~ Normal(S1[t], relative_sigma * sqrt(S2[t]))

Charge[t] = max(draw[t], 0)
```

Roots, DiCT, and DeCT add one to both `S1` and `S2`; a realized AP category adds
`rho_bar_ap[d]` to `S1` and `rho_bar_ap[d]**2` to `S2`. Under the ideal
mathematical Gaussian model, those two ledgers are sufficient for the aggregate
gain distribution and reduce to the existing `sqrt(n)` rule for unit weights.
The selected finite-lattice implementation draws one aggregate digital normal;
it does not claim bitwise or distributional identity with summing separately
rounded per-avalanche digital normal draws. The model also does not restore the
intentionally omitted within-category recovery variance. `S2` is private
numerical scratch, not a physical response, branching tensor, product,
collection sidecar, or input to offspring sampling. A later Design may decide
not to expose it, but that does not remove it from this algorithm. Using only
`sqrt(total_count)` or `sqrt(charge_pe)` after fractional AP deposits would
instead select a different smearing law.

If `ChargeConfig.smearing is None`, or if its configured `relative_sigma` is
zero, the smearing block is skipped and consumes no smearing draw. The terminal
`Charge` is constructed directly from the then-current tensor, with the one
explicit floating conversion required when the last executed stage still holds
integer root counts. Otherwise `_simulate_charge_smearing(...)` evaluates the
normal law above in the selected floating dtype and applies the documented
nonnegative clipping. No count, rate, or later offspring law reads the smeared
result.

Within `draw_afterpulses(...)`, the corresponding private output is
`afterpulse_charge_square_sum`. It accumulates each realized category as
`category_count * rho_bar_ap[d]**2` before category collapse. Squaring the
already aggregated `afterpulse_charge` would add cross terms and is not this
quantity. If the response config selects unit AP charge instead,
`afterpulse_charge` and `afterpulse_charge_square_sum` both reduce
mathematically to `afterpulse_count` in the charge dtype; generation-wise
floating accumulation still follows the dtype-aware ledger rule. The sampler
surface does not change.

Contribution identity is the mechanism on the incoming birth edge, not the
root ancestry. If an AP-born avalanche later produces a DiCT child, the parent
is counted in `afterpulse_count` and the new child in
`direct_crosstalk_count`. This preserves mechanism-resolved multiplicity
without carrying a growing lineage state. Charge uses that same incoming-edge
identity: the AP parent's recovery-weighted deposit remains in
`afterpulse_charge`, while its unit-charge DiCT child contributes through the
floating conversion of `direct_crosstalk_count`.

Subject to absent mechanisms being mathematical zero, the count invariant for
the selected absorbing retained-window process is:

```text
total_count
    == seed_avalanches
     + direct_crosstalk_count
     + delayed_crosstalk_count
     + afterpulse_count
```

Before smearing, the corresponding mathematical deposited-charge identity is:

```text
S1
    == to_charge_dtype(seed_avalanches)
     + to_charge_dtype(direct_crosstalk_count)
     + to_charge_dtype(delayed_crosstalk_count)
     + afterpulse_charge
```

The unconditional mathematical charge-square-sum identity is:

```text
S2
    == to_charge_dtype(seed_avalanches)
     + to_charge_dtype(direct_crosstalk_count)
     + to_charge_dtype(delayed_crosstalk_count)
     + afterpulse_charge_square_sum
```

The integer count identity is checked exactly. The S1 and S2 identities define
the scientific ledgers in real arithmetic, but the implementation accumulates
them in the selected `float32` or `float64` dtype. Validation therefore uses a
frozen accumulation order plus dtype- and bound-aware tolerances; it must not
assert exact recomputed floating equality unless an implementation stage proves
that stronger property for its exact reduction plan.

Here `afterpulse_charge_square_sum` accumulates
`A[g + 1, t, d] * rho_bar_ap[d]**2` before delay categories collapse. It cannot
be reconstructed afterward from `afterpulse_count` and
`afterpulse_charge`, and it is not `afterpulse_charge**2`.

When the correlated stage executes, its private algorithm result retains the
three mechanism-specific count totals, the final integer frontier after
generation `K`, separate DiCT, DeCT, and AP overflow counts, the retained AP
charge and charge-square-sum, and the AP overflow-charge diagnostic needed for
validation and truncation. The common `S1` and `S2` ledgers are always present
in that result. These are private diagnostic/state values, not
`ReadoutCollection` fields, durable products, or independently ordered public
transforms. Only the terminally finalized `Charge` is a recognized field. The
public charge path invokes `_simulate_correlated_avalanches(...)` at most once
inside `_product_charge(...)`, and only when that stage's enclosing condition
is true. The final frontier is the included generation `K` whose children were
not evaluated; it is a truncation indicator, not an estimate of the complete
omitted population.

The original IV-DSLab donor implements a real recursive avalanche tree in
`Projects/iv-dslab-main_db_PB/src/dselec/sipm.py`. Its default database path
uses the `_db` variants and includes DiCT plus AP; the non-database path also
contains the disabled-by-default prompt PHCT mechanism:

- `_poissonian_loop(...)` recursively samples Poisson offspring;
- `_add_phct(...)` and `_add_dict(...)` collapse same-type prompt recursion;
- `_add_ap(...)` creates delayed descendants and leaves
  afterpulse-of-afterpulse recursion enabled; and
- `_add_corr_noise(...)` walks a growing PE queue, so crosstalk and
  afterpulse descendants can seed further effects. Dark counts enter that
  queue before the correlated effects.

The audited `PEType` set has no DeCT member. `PHCT` is a `TODO`/`FIXME`, is
disabled by default, creates same-raw-time unit-charge rows, and is absent from
the database path; it is not evidence of an implemented delayed-crosstalk
model. IV's AP rows are delayed and recovery weighted, but that does not make
them DeCT. TensorDSLab therefore treats DeCT as absent from the donor rather
than calibrating it from IV's AP path. Any physical DeCT model remains a new,
separately calibrated inter-microcell mechanism with no IV-parity claim.

IV copies a DiCT descendant's raw continuous time from its source row, but it
later applies an independent Gaussian timing jitter to every parent and
generated row before binning. DiCT is therefore same-time only at IV's
pre-jitter table boundary. At a post-jitter binned comparison boundary, the
relative displacement contains `J_child - J_parent` and may produce a signed,
sampling-period-dependent offset kernel. Because an internally unrolled IV
subtree also shares one pre-jitter time, independently marking each binned edge
with the marginal kernel does not preserve every sibling or multigeneration
timing covariance.

Literal donor parity needs care. IV gives an AP row recovery-weighted charge and
then multiplies both that row's later DiCT coefficient and AP fire probability
by the same fractional value. Its prompt DiCT helper also freezes a source
charge-weighted offspring coefficient throughout an internally unrolled tree,
even though the emitted DiCT descendants have unit charge. Charge smearing is
applied only after the correlated-noise queue, so the smeared value does not
feed branching.

The first source-side scaling is not inherently unphysical. A partially
recovered source avalanche has lower multiplication charge and may emit fewer
secondary photons or carriers, reducing the number of DiCT or DeCT seeds. The
inter-microcell nature of crosstalk instead means that a seed which triggers a
different, recovered destination microcell normally produces a full-charge
child. IV does assign unit charge to those emitted DiCT children. Its specific
oddity is carrying the original fractional source coefficient through the
helper's whole hidden DiCT subtree rather than letting each unit-charge child
own its direct-offspring law.

This architecture deliberately chooses cleaner unit-count, recovery-independent
branching while retaining recovery only as AP deposited-charge response. That
is an intentional divergence from IV's charge-dependent future branching. The
donor's source-charge-frozen DiCT unrolling is classified as a historical
artifact, not a target for TensorDSLab. The selected model must not recover IV
behavior by feeding an aggregate charge bin, a conditional recovery weight,
or a transient recovery category back into any offspring sampler.

The previous tensor-path DSLab made the opposite, deliberate simplification.
`Projects/dslab/dslab/domain/readout/kernels/charge/__init__.py` freezes the
count grid after its enabled timing-jitter and dark-count steps, while
`crosstalk.py` and `afterpulses.py` independently produce first-generation
contributions. Its charge config fixes generation depth to one, and
`Projects/dslab/tests/test_readout_iv_grid_effects.py` tests frozen-snapshot
additivity. That is donor/comparator history, not the rebuild implementation
surface. The rebuild obtains a first-generation approximation only by choosing
`K=1` on the same coupled `_simulate_correlated_avalanches(...)` path. It does
not preserve separate `_contribute_crosstalk(...)` and
`_contribute_afterpulses(...)` functions or let their results bypass the shared
integer frontier and S1/S2 ledgers.

There are analytic and tensor-native routes worth evaluating. Ignoring spatial
or temporal placement, a DiCT-only Galton-Watson tree with Poisson mean
offspring `lambda` has the Borel total-progeny distribution:

```text
P(T=n) = exp(-lambda*n) * (lambda*n) ** (n - 1) / n!
E[T]   = 1 / (1 - lambda)
Var[T] = lambda / (1 - lambda) ** 3
```

Multiple fixed seeds give the Borel-Tanner distribution. A direct total-progeny
sampler could therefore avoid explicit count recursion provided `lambda < 1`
and numerical-tail policy is explicit, but it does not by itself assign the
sampling-dependent DiCT offsets or reproduce their joint timing covariance. A
same-bin scalar sampler is only the delta-at-zero kernel special case.

For delayed afterpulsing with mean offspring-time kernel `H`, the expected
descendant response is the renewal series:

```text
R = H + H*H + H*H*H + ...
R_hat = (I - H_hat) ** -1 * H_hat
```

For `H(t) = eta * beta * exp(-beta*t)`, this mean response has the closed form
`R(t) = eta * beta * exp(-beta * (1 - eta) * t)`. A coupled channel/time model
has the corresponding Neumann series `I + H + H**2 + ...`; convergence
requires scalar mean reproduction below one or, for a multitype model, an
integrated offspring matrix with spectral radius below one.

These closed forms describe means and selected moments. They do not by
themselves reproduce the random cascade's variance, tails, joint timing, or
channel correlations. A fixed channel topology can be represented by a
multitype kernel. Finite microcell count, saturation, collisions,
recovery-dependent offspring laws, and occupancy- or state-dependent topology
break the simple count-branching closure. Recovery-weighted AP deposited
charge alone does not: it is a separate linear response kernel over the same
unmarked avalanche cascade.

The fixed-`K` generation loop intentionally truncates descendant chains by
genealogical depth regardless of their realized offset. A future optimization
or scientific alternative must begin as a new Design proposal rather than
entering implementation as an undocumented substitution.

The scientific transition law, config ownership, recovery response, causal
window policy, and diagnostic roles are closed above. The focused algorithm
stage must still freeze the precise Poisson and multinomial samplers, their
numeric stream assignments and raw-word budgets, prepared-PMF precision and
tail tolerances, supported count/rate/`K` bounds, checked-overflow mechanics,
and concrete parity tolerances. Fusion and scratch scheduling remain measured
implementation decisions; smarter automatic `K` selection would be a later
scientific Design change. Because descendants from every mechanism feed every
enabled mechanism in the following generation, the private boundary remains
one coupled `_simulate_correlated_avalanches(...)` operation rather than three
sequential public effects.

Algorithm-level validation should cover `K = 0` and `K = 1` off-by-one cases;
the DiCT-only per-generation Poisson law, `lambda_direct**g` integrated mean,
and convolution-power position profile; sampling-period and phase-policy PMF
fixtures for both CT modes; nonnegative offset support, PMF-plus-right-tail
normalization, exact absence of underflow, and separate per-mode overflow
accounting; distinct DiCT and DeCT Poisson draws and accounting; sibling and
multigeneration timing covariance as named approximation diagnostics; AP's
one-child bound, destination multinomial covariance, and separate stop/overflow
outcomes; `q_ap`, `h_ap`, and `rho_bar_ap` fixtures from the joint latent phase
and physical-delay law; AP charge and `afterpulse_charge_square_sum` from the
same realized category counts; conditional charge mean and count/charge
covariance; the named missing
within-category recovery variance; exact unit deposited charge for fresh-cell
DiCT/DeCT children; parent-cell recovery weighting for AP; recovery-independent
cross-feeding in the following generation; all eight mechanism enablement
combinations; the exact integer count invariant; dtype-aware validation of the
mathematical S1/S2 component identities and their unit-weight reduction;
final-frontier semantics; and checked `int64` overflow behavior. These tests
need no final TensorDSLab stream assignment or downstream smearing decision to
establish the scientific transition law.

Research references for that stage are:

- M. Dwass, [*The total progeny in a branching process and a related random
  walk*](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/total-progeny-in-a-branching-process-and-a-related-random-walk/B9505DF87F411C873D3AA511F21EBF8C),
  for total-progeny validation evidence for Poisson branching limits.
- S. Vinogradov, [*Analytical models of probability distribution and excess
  noise factor of Solid State Photomultiplier signals with
  crosstalk*](https://arxiv.org/abs/1109.2014), including the Poisson branching
  and Borel model; journal DOI
  [10.1016/j.nima.2011.11.086](https://doi.org/10.1016/j.nima.2011.11.086).
- A. G. Hawkes and D. Oakes, [*A cluster process representation of a
  self-exciting process*](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/cluster-process-representation-of-a-selfexciting-process/E836A3D07D808068E2F9F3E7E366B081),
  the immigration-birth/Poisson-cluster foundation for the renewal view.
- A. Para, [*Afterpulsing in Silicon Photomultipliers: Impact on the
  Photodetectors Characterization*](https://arxiv.org/abs/1503.01525).
- J. Rosado et al., [*Modeling crosstalk and afterpulsing in silicon
  photomultipliers*](https://arxiv.org/abs/1409.4564), relevant to spatial
  crosstalk and recovery-aware delayed effects.
- J. Rosado and S. Hidalgo, [*Characterization and modeling of crosstalk and
  afterpulsing in Hamamatsu silicon
  photomultipliers*](https://arxiv.org/abs/1509.02286), which distinguishes
  low-amplitude same-pixel afterpulses during recovery from full-amplitude
  delayed crosstalk in other pixels.
- V. Moya and J. Rosado, [*Understanding the Nonlinear Response of
  SiPMs*](https://arxiv.org/abs/2401.06581), whose recovery-aware Monte Carlo
  scales correlated-noise seed production from a recovering source avalanche
  while separately modeling whether a seed triggers its destination pixel.
- Y. Guan et al., [*Study of Silicon Photomultiplier External
  Cross-Talk*](https://arxiv.org/abs/2312.12901), additional evidence for
  Borel-family fired-pixel models and for treating inter-device topology as a
  deliberate model choice.
- Y. Liu, X. Liu, and B. Xu, [*Closed-Form Analytical Charge Response Model
  for Silicon Photomultipliers with Recursive Correlated
  Avalanches*](https://arxiv.org/abs/2605.27340). This May 2026 preprint is a
  recent research lead and is treated here as unreviewed evidence, not an
  accepted TensorDSLab authority.

### Pure Waveform

`PureWaveformConfig.model` selects one electronics response for the complete
channel axis, using one scalar parameter set for the entire invocation.

For the MVP, TensorDSLab provisionally adopts the two mathematical pulse-shape
families implemented by IV-DSLab. This is an implementation/parity decision,
not a claim that collaborators have finished validating either model as the
best detector description. A later focused scientific decision may revise a
model or its calibration without reopening the typed product/config
architecture. The donor audit found no equation-level defect analogous to the
ADC cast-before-clip bug.

TensorDSLab expresses the IV TPC FEB-SNR response using its actual fast and
slow exponential constants:

```text
h_tpc(t) = exp(-t / tau_slow) - exp(-t / tau_fast)
tau_fast = IV sipm.feb_snr.tau_r
tau_slow = IV sipm.feb_snr.tau_r + IV sipm.feb_snr.tau_l
```

This avoids calling IV's `tau_l` the slow or fall time constant: the donor
places `tau_l + tau_r`, not `tau_l`, in the slow exponential denominator.
`TpcFebSnrPulseConfig` therefore stores `fast_time_constant_ns` and
`slow_time_constant_ns` directly and requires `slow > fast`.

The provisionally adopted IV Veto PDU response is:

```text
x = t - gaussian_center
h_veto(t) = exp(-x**2 / (2 * gaussian_width**2))
            / sqrt(2 * pi * gaussian_width**2)
            * (1 + erf(
                (x - edge_offset_1) / (sqrt(2) * edge_width_1)
            ))
            * (1 + erf(
                (x - edge_offset_2) / (sqrt(2) * edge_width_2)
            ))
```

The exact donor-to-config mapping is:

| TensorDSLab config field | IV-DSLab parameter | Mapping |
| --- | --- | --- |
| `TpcFebSnrPulseConfig.fast_time_constant_ns` | `sipm.feb_snr.tau_r` | direct |
| `TpcFebSnrPulseConfig.slow_time_constant_ns` | `sipm.feb_snr.tau_r`, `sipm.feb_snr.tau_l` | sum |
| `VetoPduPulseConfig.gaussian_center_ns` | `veto.offset1` | direct |
| `VetoPduPulseConfig.gaussian_width_ns` | `veto.fall_time` | renamed to its actual equation role |
| `VetoPduPulseConfig.edge_offset_1_ns` | `veto.offset2` | direct |
| `VetoPduPulseConfig.edge_width_1_ns` | `veto.rise_time1` | renamed to its actual equation role |
| `VetoPduPulseConfig.edge_offset_2_ns` | `veto.offset3` | direct |
| `VetoPduPulseConfig.edge_width_2_ns` | `veto.rise_time2` | renamed to its actual equation role |

IV's calibrated numbers remain provisional parity-fixture evidence, not public
TensorDSLab defaults. In particular, the donor's scalar Veto amplitude and its
condition-database path do not override the MVP decision that one explicit
scalar config applies to every channel in one invocation.

For an 8 ns parity fixture, the audited donor values translate to:

| Model | TensorDSLab fixture values |
| --- | --- |
| TPC FEB-SNR | `fast_time_constant_ns=83`, `slow_time_constant_ns=383`, `support_time_ns=3000`, `peak_voltage_mv_per_pe=-7` |
| Veto PDU | `gaussian_center_ns=232.89`, `gaussian_width_ns=507.72`, `edge_offset_1_ns=-81.92`, `edge_width_1_ns=147.28`, `edge_offset_2_ns=-176.50`, `edge_width_2_ns=45.69`, `support_time_ns=2020.27`, `peak_voltage_mv_per_pe=-14.5912372` |

The Veto support reproduces the donor's retained samples from 0 through
2016 ns at 8 ns spacing; it does not promote the donor's support heuristic as a
general rule. These values belong in parity fixtures until collaborators
approve named calibration presets.

For sample period `T` and template index `j`, each model is point-sampled as
`h[j] = h(j * T)` at left-edge times satisfying
`0 <= j * T < support_time`; `support_time` is the separate exclusive stop.
Preflight rejects a model/sampling combination whose exclusive support produces
no samples or whose sampled extremum is nonfinite or zero; normalization must
never divide by an unresolved template. The sampled template is normalized by
the magnitude of its sampled extremum, scaled once by its signed
`peak_voltage_mv_per_pe`, convolved causally with PE-equivalent charge, and
truncated to the input sample count. Negative-going detector pulses therefore
use a negative configured peak voltage; there is no second gain or inversion
switch. Output axes and shape match charge. Baseline is not part of the
signal-only `PureWaveform`.

TensorDSLab intentionally standardizes the discretization around those donor
equations:

- IV analytically normalizes the continuous TPC curve but normalizes the
  sampled Veto curve. TensorDSLab normalizes both by the magnitude of the
  sampled extremum so `peak_voltage_mv_per_pe` means the realized discrete
  peak. At IV's 8 ns TPC fixture this changes the peak by about 66 parts per
  million.
- IV derives TPC support as `10 * max(tau_l, tau_r)` even though its actual slow
  constant is `tau_l + tau_r`, and derives Veto support through a heuristic
  strict crop. TensorDSLab instead requires one explicit exclusive
  `support_time_ns` and applies the repository-wide left-edge convention.
- IV scales the positive template and inverts after convolution. TensorDSLab
  applies the signed configured peak once, producing the same negative-going
  parity waveform without a second inversion switch.
- The first MVP applies none of IV's eventwise fractional-bin exponential
  amplitude correction. That correction reuses parameters inconsistently and
  is not an exact fractional delay of either adopted pulse equation.

These are documented tensor-path discretization corrections, not alternative
pulse-shape equations. Literal donor support values may be used by parity
fixtures, but they are not package defaults. Gate off-by-one behavior,
pre-window-tail loss, and donor noise/amplitude coupling remain outside the
pure-waveform equation contract and must not be copied accidentally.

### Noise Waveform

Recognized models are:

1. exact zero noise;
2. position-addressed IID Gaussian white noise with ensemble mean zero and
   explicit RMS; and
3. exact-record-zero-mean Gaussian noise shaped by a caller-supplied one-sided
   PSD.

`PsdNoiseConfig` supplies arbitrary strictly increasing frequency left edges,
one exclusive `frequency_stop_hz`, and piecewise-constant absolute density in
`mV^2/Hz`. Left-edge and density counts are equal. It must start at zero, its
stop must exceed the final left edge, and it must cover the Nyquist frequency
implied by `SamplingConfig`; it need not already match the fixed-length
synthesis grid. Preflight integrates source density over left-closed/right-open
target intervals into the pre-suppression `Q` cells, conserving represented
source power before the accepted DC-cell discard constructs `P`.
The PSD is the effective post-front-end, post-anti-alias noise at the shared
analog reference plane. Any supplied coverage above Nyquist is ignored
deliberately rather than folded into band.
Raw FFT amplitudes and complex coefficients are not accepted inputs. No
persistent baseline bank, random crop, spectrum download, SNR-to-amplitude
coupling, or per-call spectral file loading belongs to the noise producer.

For sample rate `fs`, record length `N`, spacing `df = fs / N`, and
`K = floor(N / 2)`, the private Fourier basis frequencies are `c[k] = k * df`
for `k = 0, ..., K`. They are spectral-line/basis frequencies, not bin edges
or necessarily geometric centers of their endpoint cells. The separate
power-integration cells are represented by this left-edge array and exclusive
stop:

```text
target_left_edge[0] = 0
target_left_edge[k] = (k - 1/2) * df,  k = 1, ..., K
target_stop = fs / 2
```

Pre-suppression target cell `Q[k]` receives the PSD integral over
`[target_left_edge[k], target_left_edge[k + 1])`, with the final cell ending at
`target_stop`. The pre-suppression DC cell therefore owns `[0, df / 2)`, which
the accepted policy later discards. For even `N`, the pre-suppression real
Nyquist cell owns a half-width final interval. For odd `N`, there is no
Nyquist coefficient; the highest complex coefficient receives a full-width
final cell that ends at `fs / 2`. Source coverage means exactly
`frequency_stop_hz >= fs / 2`; equality is sufficient because the exclusive
endpoint has zero measure. Preflight rejects a supplied PSD whose retained
in-band power after the zero-DC policy is zero, even if it contains positive
density only above Nyquist or only inside the discarded DC cell.
Rebinning is interval-overlap integration, not interpolation:

```text
Q[k] = sum_i S[i] * max(
    0,
    min(source_right[i], target_right[k])
    - max(source_left[i], target_left_edge[k]),
)

P[0] = 0
P[k] = Q[k],  k = 1, ..., K
```

Within the accepted numerical tolerance, `sum_k(Q[k])` equals the supplied PSD
integral over `[0, fs / 2)`. Synthesis then deliberately sets the DC-cell power
to zero. It therefore retains `sum_{k=1}^K(P[k])`, discards exactly the
integrated power over `[0, df / 2)`, and performs no redistribution or global
renormalization. This is a finite record-length DC notch, not removal of only
the measure-zero point `f = 0`.

The private real DC coefficient is exactly zero, so every synthesized PSD
record has zero sample mean up to inverse-transform roundoff. It cannot add a
record-wide random voltage offset when `NoiseWaveform` is composed with the
zero-baseline `PureWaveform`; the digitizer transfer remains the sole owner of
the quiescent ADC-code placement. This exact-record rule applies specifically
to `PsdNoiseConfig`. An IID `WhiteNoiseConfig` realization may have an ordinary
finite-sample mean fluctuation and is not silently demeaned, because demeaning
would change its accepted IID covariance.

PSD-shaped noise uses one exact Gaussian one-sided coefficient law. For each
leading-index waveform row, let all `u[k]`, `v[k]`, and `z` values below be
mutually independent real standard-normal variates generated in the selected
output floating dtype. Define the interior index set as
`I = {1, ..., floor((N - 1) / 2)}`. The private one-sided coefficients are:

```text
X[0] = 0 + 0j

X[k] = (N / 2) * sqrt(P[k]) * (u[k] + i * v[k]),  k in I

if N is even:
    X[N / 2] = N * sqrt(P[N / 2]) * z + 0j
```

The odd-`N` terminal coefficient belongs to `I` and remains complex. The
even-`N` Nyquist coefficient is real. Its imaginary component and the DC
imaginary component are exactly zero rather than values left for `irfft` to
ignore. The implementation constructs the two parts of every interior
coefficient from two explicit real standard-normal
draws; it must not silently substitute a native complex-normal draw whose real
and imaginary components each have variance `1 / 2`.

The output is normatively:

```python
noise = torch.fft.irfft(X, n=N, dim=-1, norm="backward")
```

`N` is always explicit because odd record length cannot be recovered uniquely
from one-sided coefficient count. For floating `torch.float32` output, `X` is
`torch.complex64`; for `torch.float64`, it is `torch.complex128`.
`norm="backward"` is the Fourier normalization convention, not an autograd
instruction: the paired forward transform is unscaled and the inverse applies
the factor `1 / N`. With `K = floor(N / 2)`, the exact odd/even inverse
equations are:

```text
N = 2*K:
    x[n] = (1 / N) * (
        2 * Re(sum_{k=1}^{K-1}(X[k] * exp(i * 2*pi*k*n/N)))
        + X[K] * (-1)**n
    )

N = 2*K + 1:
    x[n] = (2 / N) * Re(
        sum_{k=1}^{K}(X[k] * exp(i * 2*pi*k*n/N))
    )
```

`X[0]` is absent from both right-hand sides because it is exactly zero. The
factor of two comes from the omitted negative-frequency conjugate of each
interior coefficient. DC and even-length Nyquist are self-conjugate and do not
receive that factor. This explains the `N / 2` interior scale and `N` Nyquist
scale above.

The frozen statistical contract is:

```text
E[x[n]] = 0
mean_n(x[n]) = 0                         # up to transform roundoff
Var[x[n]] = sum_{k=1}^K(P[k])

N = 2*K:
    Cov(x[n], x[n + lag]) =
        sum_{k=1}^{K-1}(P[k] * cos(2*pi*k*lag/N))
        + P[K] * (-1)**lag

N = 2*K + 1:
    Cov(x[n], x[n + lag]) =
        sum_{k=1}^{K}(P[k] * cos(2*pi*k*lag/N))
```

For every paired coefficient,
`Var(Re(X[k])) = Var(Im(X[k])) = N**2 * P[k] / 4` and
`E[abs(X[k])**2] = N**2 * P[k] / 2`. The even-length real Nyquist coefficient
has variance `N**2 * P[N / 2]`. Parseval's identity consequently gives
`E[mean_n(x[n]**2)] = sum_{k=1}^K(P[k])`.

The covariance is circular in the sample index. Different waveform rows use
independent coefficient variates in the MVP, so their cross-covariance is zero
in expectation. Realized finite-record power fluctuates stochastically; only
its expectation equals retained integrated PSD power. The builder performs no
post-`irfft` demeaning, unit-standard-deviation normalization, power
normalization, discarded-DC redistribution, or independent `scale_mv`. Tiny
nonzero numerical sample means caused by transform roundoff are not corrected.

The DC notch has an observable finite-record consequence. For an accepted PSD
record (`N >= 2`) with constant input density `S`, the retained variance is
`S * fs * (N - 1) / (2 * N)`, every nonzero circular lag has covariance
`-S * df / 2`, and the corresponding correlation is `-1 / (N - 1)`. A
fixed-length flat-PSD realization is therefore exactly zero-sum and weakly
anticorrelated, not IID white. Callers who want IID samples select
`WhiteNoiseConfig`; the two accepted models are deliberately distinct.

The private coefficient frequencies `k / (N * dt)` are therefore never exposed
as semantic bin coordinates or caller-supplied edges.

The MVP synthesizes exactly the configured `N = sampling.sample_count` samples.
An exact-length inverse transform therefore realizes a periodic finite record
with circular/circulant covariance across the two window edges. It performs no
hidden longer-record generation, padding, overlap, crop, or baseline-bank
selection. This boundary is validated and recorded as an intentional
tensor-path divergence from IV's long-baseline generation and random crop. A
later padded/cropped model would be a different accepted noise algorithm, not
a quiet implementation substitution.

White-noise positions are the final waveform's logical tensor positions.
PSD-synthesis random positions instead belong to a defined private one-sided
coefficient shape: every source dimension except the sample dimension,
preserved in its current order, followed by the target frequency dimension.
That intermediate shape/order is part of the PSD-noise synthesis algorithm
version; its flat positions are not interchangeable with flat positions in the
time-domain output waveform. Consistent with the scalar-calibration rule, the
MVP uses the same white-noise RMS or PSD for every channel while drawing the
channels independently. Per-channel spectra and cross-channel spectral
correlation require a later typed tensor-calibration/input contract.

### Analog Waveform

```text
analog[i] = clamp(
    pure[i] + noise[i],
    optional_minimum_mv,
    optional_maximum_mv,
)
```

Pure and noise must have equal axes, device, dtype, shape, and mV reference
plane. One optional pair of scalar saturation bounds applies to every channel
and example. No implicit broadcast or coordinate-dependent limit lookup is
accepted. This clamp models physical analog/front-end saturation. It belongs
inside `_product_analog_waveform(...)` and is distinct from the finite ADC code
range. An absent lower or upper bound leaves that side unbounded; with no
bounds the equation reduces to `pure + noise`.

The MVP introduces no deterministic analog baseline or pedestal.
`PureWaveform` is a signal excursion from 0 mV, `NoiseWaveform` is a stochastic
voltage fluctuation about 0 mV, and `AnalogWaveform` is their zero-referenced
composition. PSD-shaped noise has an exactly zero synthesized DC coefficient;
IID white noise remains ensemble-centered and may have an ordinary
finite-record mean fluctuation. Neither case carries a configured pedestal.

If a later detector model needs a physical front-end bias, a channel-dependent
quiescent voltage, or a separately retained baseline waveform, that effect
belongs explicitly in the analog stage before saturation. It may become an
`AnalogWaveformConfig` component or a distinct typed input/product after a
focused Design decision; it must not be hidden in `NoiseWaveform`. Time-varying
baseline wander may remain a named stochastic noise submodel when that is its
actual physical meaning.

### Digitization

The `AnalogWaveform` is expressed at the pre-digitizer-gain mV reference plane.
`input_min_mv` and `input_max_mv` describe the digitizer's post-gain analog
input range. Preflight computes:

```text
maximum_code = 2**bit_depth - 1
gain = 10**(analog_gain_db / 20)
span = input_max_mv - input_min_mv
slope = gain * maximum_code / span
intercept = -input_min_mv * maximum_code / span
```

The product producer then evaluates:

```text
digitized[i] = int32(clamp(
    analog[i] * slope + intercept,
    0,
    maximum_code,
))
```

Bit depth is in `[1, 16]`, analog gain is in `[0, 40]` dB, and output is
nonnegative `torch.int32`. Clipping precedes conversion; unsigned wraparound is
forbidden. One scalar gain and voltage-transfer range applies to every channel
and example. Conversion of the nonnegative clamped value truncates toward zero,
which is the accepted ADC quantization rule. No separate pedestal is needed:
an asymmetric input range determines the code corresponding to 0 mV.
That nonzero zero-voltage code is an ADC transfer property, not a baseline
voltage added to `AnalogWaveform`. Digitization is not declared
differentiable.

For scientific readability and validation, the algebraically equivalent
unfused reference transfer remains:

```text
gained_mv = analog_mv * gain
clipped_mv = clamp(gained_mv, input_min_mv, input_max_mv)
scaled_code = (
    (clipped_mv - input_min_mv)
    / span
    * maximum_code
)
reference_code = int32(clamp(scaled_code, 0, maximum_code))
```

The production expression using precomputed `slope` and `intercept` is the
normative execution form. Validation covers exact endpoints, code-transition
neighborhoods, and the accepted dtype/backend arithmetic rather than assuming
cross-backend bitwise identity under floating multiply-add reassociation. Its
pre-conversion clamp intentionally fixes IV-DSLab's cast-before-clip
wraparound defect.

Because TensorCore leaves are fieldless, a bare `DigitizedWaveform` does not
carry variable bit depth, gain, or voltage transfer. The builder guarantees its
codes against the supplied `DigitizedWaveformConfig`, and the caller retains
that config for interpretation. Before durable IO or an independent
cross-process digitized handoff, a focused design must bind the config or an
equivalent versioned calibration record to the artifact.

## RNG And Positional Repeatability

Randomness is a runtime input, not scientific config. The public builder
accepts one root `seed`; every stochastic leaf owns one globally unique fixed
numeric operation stream. Existing stream assignments never depend on the
requested subset or execution order and must not be renumbered when a later
operation is added. Do not derive assignments with `Enum.auto()`, tuple
position, Python `hash()`, or branch-dependent sequential consumption. The
assignments are encapsulated by the private RNG implementation and are not
exported as public module constants.

The random engine is general over arbitrary tensor rank and shape. Its logical
address is:

```text
root seed
globally unique numeric operation stream
logical flat tensor position
local source-quantum ordinal, when required
local raw-word ordinal
```

For shape `(n0, n1, ..., nk)`, logical flat position is the ordinary row-major
flattening of the tensor's current dimension order, conceptually
`(((i0 * n1 + i1) * n2 + i2) * ... + ik)`. It is not a physical storage offset
and does not depend on strides. No axis class, coordinate string, timestamp,
product label, or physical time enters the random address. An operation may
still locate the sample dimension during preflight and use numeric sample
period or index for its physics; that is separate from random identity.

A rank-zero scalar has one logical position, `0`. A shape with any zero-sized
dimension has no logical positions and consumes no draws. Valid readout fields
have nonempty axes, so scalar and empty cases belong to private generic RNG
primitive tests rather than the public readout collection contract.

Conceptually, a one-raw-word-per-position random field is:

```python
flat_position = logical_flat_positions(tensor.shape)
random_bits = counter_random(
    seed=root_seed,
    stream=operation_stream,
    position=flat_position,
    raw_word_ordinal=0,
)
random_field = random_bits.reshape(tensor.shape)
```

The GPU implementation should derive logical position from thread/index
arithmetic; the conceptual `flat_position` value does not require allocating
an `arange` tensor in a warmed kernel.

An iterative stochastic role may extend this same positional rule with a
virtual leading iteration dimension when it must distinguish repeated draws at
the same tensor position. For a fixed per-iteration private lattice of size
`N`, zero-based global iteration `j`, and row-major local position `u`, use:

```text
p = j * N + u
```

The virtual dimension is conceptual and is never materialized. Its meaning,
the role's exact private shape and dimension order, and `N` are frozen with the
stochastic algorithm. Preflight uses checked host arithmetic and requires the
maximum processed iteration count `G` to satisfy `G * N <= 2**63`. It never
uses a block-local iteration, active-only compaction, semantic coordinate,
timestamp, label, or execution order. Ordinary noniterative tensor roles are
the special case `G = 1` and `p = u`.

The fixed-`K` correlated-avalanche simulation uses this rule directly. For the
draws that produce offspring generation `g + 1`, `j = g` with
`0 <= g < K`. Each DiCT, DeCT, and AP-category draw role has its own fixed
numeric stream and fixed per-generation lattice; a delay-category dimension,
when required, is part of that role's frozen row-major lattice. Terminal
smearing owns a separate noniterative stream and product-grid lattice. A
zero frontier requests no words and may skip physical work, but it never
compacts later positions or changes another role's address. Preflight requires
`K * N <= 2**63` independently for every enabled generation role. Exact stream
numbers, Poisson/multinomial raw-word schedules, and exhaustion behavior remain
the focused RNG gate; generation identity itself is no longer open.

### Private Raw Engine And Address Schema

The accepted raw engine is standard Random123 `Threefry4x32` with exactly 20
rounds. The normative external definition is Random123 `1.14.0` at commit
[`726a093`](https://github.com/DEShawResearch/random123/commit/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13),
specifically its
[`threefry.h`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/include/Random123/threefry.h)
`Threefry4x32_R<20>` algorithm and
[`kat_vectors`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/tests/kat_vectors)
word order. This freezes the standard rotation constants, key-injection
schedule, output tuple order, and `0x1bd1_1bda` parity constant. A reduced-round
variant, PyTorch-internal approximation, or different output-word order is not
the same algorithm.

The durable private algorithm identifier is:

```text
tensordslab.threefry4x32-20/v1
```

This identifier is a repeatability boundary, not a public parameter or
TensorCore identity. The public API continues to accept only the root `seed`.
Callers never construct a generator, key, counter, stream, or RNG object.

For root seed `s`, fixed private stream `g`, logical flat position `p`,
source-quantum ordinal `q`, and zero-based raw-word ordinal `r`, define:

```text
low32(x)  = x & 0xffff_ffff
high32(x) = (x >> 32) & 0xffff_ffff

block = r // 4
lane  = r % 4
```

Schema v1 packs words numerically as:

```text
key[0] = low32(s)
key[1] = high32(s)
key[2] = g
key[3] = 0x54445331  # ASCII "TDS1"

counter[0] = low32(p)
counter[1] = high32(p)
counter[2] = q
counter[3] = block

words = Threefry4x32_20(counter, key)
raw_word = words[lane]
```

The low/high split is numerical and independent of host byte order. `lane`
selects Random123 output tuple member `v[0]` through `v[3]` in its declared
order. The exact accepted bounds are:

```text
0 <= seed < 2**64
0 <= stream < 2**32
0 <= logical_flat_position < 2**63
0 <= source_quantum_ordinal < 2**32
0 <= raw_word_ordinal < 2**34
```

The counter encoding itself has two position words, but the accepted execution
contract uses the stricter signed Torch indexing/`numel` range. A per-quantum
algorithm therefore accepts at most `2**32` source quanta in one cell, with
ordinals `0` through `2**32 - 1`. Each block supplies four lanes, so the
raw-word bound makes `block` an exact unsigned 32-bit value. Input counts remain
`torch.int64`; the smaller per-cell quantum population limit is checked
explicitly rather than reached through a narrowing cast.

This encoding is injective over the accepted address domain: equal encoded
key, counter, and lane recover the same seed, stream, position, quantum, and
raw-word ordinal. That is an address-collision statement, not an output-value
uniqueness claim. Distinct addresses may naturally return the same 32-bit raw
word.

Every independently specified random field or stochastic substep owns one
private exact-valued 32-bit stream. Cell-level operations use `q = 0`;
per-source-quantum operations use the ordinal within that source cell. One
stream must not mix those two meanings. Streams live as explicitly assigned
members of a private type such as `_RngStream`, never as exported loose
constants, and never derive from `Enum.auto()`, Python `hash()`, declaration
order, requested-product order, or execution order. The exact stream table is
frozen with the corresponding stochastic algorithms; this architecture does
not prematurely assign numbers to the still-deferred charge distribution
models. Once assigned, a stream is never renumbered or repurposed.

Threefry operates on mathematical unsigned 32-bit words. The reference Torch
implementation may carry each word in `torch.int64`, provided every live word
stays in `[0, 2**32)`, every modular addition is masked to 32 bits, every right
shift receives an already-masked nonnegative value, and rotations use explicit
shifts and masks. Under those rules, the accepted Threefry additions and
rotations fit signed 64-bit intermediates. The implementation must not rely on
signed overflow, host endianness, physical strides, or implementation-defined
signed right shift. A later native-unsigned CPU, Triton, or CUDA kernel is an
implementation substitution only if it returns the same standard raw words.

The integer core has a strong boundary: every supported eager, compiled, CPU,
and CUDA implementation must return identical four-word output for an
identical key and counter. This claim becomes true for a backend only after it
passes the authoritative known-answer and cross-implementation tests. The
initial fixed Random123 oracles include:

```text
counter: 00000000 00000000 00000000 00000000
key:     00000000 00000000 00000000 00000000
output:  9c6ca96a e17eae66 fc10ecd4 5256a7d8

counter: ffffffff ffffffff ffffffff ffffffff
key:     ffffffff ffffffff ffffffff ffffffff
output:  2a881696 57012287 f6c7446e a16a6732

counter: 243f6a88 85a308d3 13198a2e 03707344
key:     a4093822 299f31d0 082efa98 ec4e6c89
output:  59cd1dbb b8879579 86b5d00c ac8b6d84
```

Raw-word identity does not automatically make floating distribution transforms
bitwise identical across backends. Uniform conversion, normal pairing,
exponential and Bernoulli transforms, Poisson sampling, transcendental
functions, arithmetic dtype, and rejection behavior are separate frozen
algorithm contracts. Until their evidence supports something stronger,
completed stochastic products retain the same-backend repeatability and
cross-backend statistical-parity boundary documented below.

The implementation must not read or mutate PyTorch's global RNG state, create
a `torch.Generator`, use `torch.poisson` as the normative sampler, or depend on
private PyTorch RNG operations. The engine will belong privately to
TensorDSLab when its focused stage is dispatched; selecting it here neither
changes TensorCore nor creates a Random123 runtime dependency.

### MVP Uniform And Distribution Primitives

The MVP uses the conventional precision-matched Random123 fixed-point
conversions rather than a widened `float64` inverse-transform path for
`float32` products. The normative conversion reference is Random123 `1.14.0`
[`u01fixedpt.h`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/include/Random123/u01fixedpt.h).
This keeps the GPU path simple and makes the finite precision and tail limits
explicit.

For one raw word `w0`, the `float32` conversions are:

```text
m24 = w0 >> 8

U32[0, 1) = float32(m24) * 2**-24
U32(0, 1) = float32(m24 + 1) * 0x1.fffffep-25
```

The open multiplier is exactly `2**-24 - 2**-48`. `U32[0, 1)` ranges from
zero through `1 - 2**-24`; `U32(0, 1)` ranges from
`2**-24 - 2**-48` through `1 - 2**-24`. The lower eight raw bits are discarded
and never reused.

A `float64` uniform consumes two consecutive raw words. The earlier word is
the numerical high word, independent of host byte order:

```text
m53 = w0 * 2**21 + (w1 >> 11)

U64[0, 1) = float64(m53) * 2**-53
U64(0, 1) = float64(m53 + 1) * 0x1.fffffffffffffp-54
```

This arithmetic assembly stays within signed `torch.int64`; it does not first
construct an overflowing unsigned 64-bit carrier. The open multiplier is
exactly `2**-53 - 2**-106`. `U64[0, 1)` ranges from zero through
`1 - 2**-53`; `U64(0, 1)` ranges from `2**-53 - 2**-106` through
`1 - 2**-53`. The lower eleven bits of `w1` are discarded and never reused.

Bounded phase and interpolation, including the Box-Muller angle, use `[0, 1)`.
Any logarithm uses `(0, 1)`, so neither logarithmic infinity nor an artificial
exact-zero inverse-transform result is possible. The endpoint-safe Random123
open mapping is intentional: the superficially simpler midpoint expression
`(m + 0.5) * 2**-p` can round its upper value to exactly one in the target
floating dtype.

Bernoulli sampling bypasses floating uniforms. From the accepted finite
binary64 configuration probability `p`, preflight computes:

```text
T = round_ties_to_even(p * 2**32)
fires = raw_word < T
```

`T` is an integer in `[0, 2**32]`, and the realized probability is exactly
`T / 2**32`, within `2**-33` of the represented configuration probability.
Threshold zero returns false without requesting a word; threshold `2**32`
returns true without requesting a word. Every interior threshold consumes one
assigned raw word. This avoids the systematic downward bias of a floored
threshold and gives both `float32` and `float64` consumers the same Bernoulli
law.

An ordinary exponential variate of configured mean `tau` is:

```text
delay = -tau * log(U(0, 1))
```

The operation evaluates in its accepted execution dtype; a `float32` operation
does not silently widen to `float64`. The finite uniform lattice bounds a
`float32` delay below about `16.64 * tau` and a `float64` delay below about
`36.74 * tau`. An operation whose scientifically relevant window reaches that
limit must classify and validate the tail truncation rather than call the
sampled law an exact continuous exponential.

This inverse-transform primitive applies only to an operation that explicitly
samples one continuous exponential variate. The fixed-`K` correlated-avalanche
path does not use it for CT or AP edge placement: preflight integrates the
physical delay law and latent phase into complete categorical offset
probabilities plus an analytic right tail, and the runtime samples those
prepared categories. Its exponential-law qualification is therefore numerical
CDF/PMF preparation and categorical sampling, not the finite inverse-transform
tail above.

Standard-normal variates use ordinary Box-Muller in the selected execution
dtype:

```text
radius = sqrt(-2 * log(U(0, 1)))
angle = 2 * pi * U[0, 1)
z0 = radius * cos(angle)
z1 = radius * sin(angle)
```

One exact `(stream, logical_position, source_quantum, base_raw_word)` address
owns the ordered pair `(z0, z1)`. A `float32` pair consumes two consecutive raw
words; a `float64` pair consumes four. A scalar-normal consumer always uses
`z0` and discards `z1`. A natural two-component consumer at the same address,
such as the real and imaginary standard-normal components of one PSD
coefficient, uses `z0` and then `z1`. The spare result is never cached or
reassigned to another tensor position, source quantum, branch, or stochastic
substep. This intentionally favors a simple stable positional contract over
packing adjacent output positions into one pair.

The maximum Box-Muller radius is about `5.77` for `float32` and `8.57` for
`float64`. The `float32` cutoff is an accepted bounded-MVP approximation, not a
claim of an unbounded mathematical Gaussian. It must be included in tail-aware
validation and in the synchronized `docs/parity.md` comparison; evidence that
rare threshold observables are sensitive to it is the trigger for a separately
versioned widened or tail-complete normal algorithm.

Uniform conversions and Bernoulli integer comparisons must reproduce the same
target-dtype values bit-for-bit on every supported implementation. Box-Muller
and exponential outputs retain the same-backend repeatability boundary because
`log`, square root, sine, and cosine may differ across backends or compiler
modes. Selecting these primitives does not create a CPU/CUDA bitwise guarantee
for completed stochastic products.

Variable-count operations assign a source quantum the deterministic address
`(source_flat_position, quantum_ordinal, raw_word_ordinal)` before
redistribution. `quantum_ordinal` is the zero-based ordinal within its source
cell and never depends on parallel execution order. A distribution-level draw
may consume more than one raw word, so `raw_word_ordinal` must not be renamed or
treated as a one-word draw number.

Preflight rejects an invalid seed, unsupported stream, position/shape overflow,
source population outside the quantum bound, or statically known raw-word
overflow before any stochastic product write. A variable-attempt rejection
sampler cannot preflight its realized attempt count. Exhausting its 32-bit block
coordinate is a deterministic hard algorithm failure: it returns no valid
product and must never wrap, reseed, reuse an earlier address, change
algorithms, clamp the sample, or emit a biased fallback.

Required behavior is:

- exact repeatability for the same root seed, input values, tensor shape,
  dimension order, coordinate order, config, dtype, algorithm/version, and
  supported backend;
- unchanged common product values when unrelated products are added to or
  removed from `products`, because their numeric operation streams are fixed;
- zero-effect configs consume no relevant draws;
- no hidden global RNG; and
- cross-backend distributional/statistical agreement without an assumed
  bitwise guarantee.

Coordinate values are semantic metadata but are not random identities.
Relabeling or reordering coordinate metadata without moving payload values
leaves the positional random bits unchanged but changes their semantic
interpretation. Explicitly reindexing a payload to preserve coordinate
identity generally moves an item to a different position and therefore gives
it a different draw. Tensor-dimension permutation likewise changes the
position-to-semantic mapping. TensorDSLab promises no coordinate-identity or
permutation invariance; a meaningful semantic comparison first reindexes both
tensors explicitly.

Selection and arbitrary chunking are also not invariant. Positional addresses
restart at zero in every builder invocation. Calls of any shapes using the same
root seed and operation stream therefore reuse the same underlying random
prefix over their overlapping logical flat-position range; equal `numel`
reuses the complete flat sequence before reshaping. MVP callers must supply
distinct seeds for statistically independent invocations. A future
chunk-stable execution surface would require explicit global positional
offsets and a focused Design contract; TensorDSLab does not infer those
offsets from semantic labels.

The positional engine is deliberately rank- and domain-agnostic, so it may
become a TensorCore candidate if another tensor package needs the same tested
contract. The rebuild initially keeps it private to TensorDSLab: generic shape
alone does not justify expanding TensorCore before the algorithm,
distribution, backend, and repeatability evidence exists.

## Functional, Memory, And Lifetime Contract

The initial rebuild adopts TensorCore's operation-owned result taxonomy. Its
public and private field-returning paths are deliberately stricter than the
generic root:

- the source `Photoelectrons` field is borrowed read-only;
- requesting `Photoelectrons` classifies that member as an **exact return** of
  the named source field;
- every product producer classifies its new result as **guaranteed fresh
  storage independent of named inputs**;
- generated fields retained together in one result are also guaranteed
  storage-independent from one another;
- the initial `simulate_readout(...)` surface has no guaranteed-storage-sharing
  result and no sharing-permitted-but-unspecified result path;
- every dimension-preserving product reuses the exact source `axes` tuple and
  exact immutable axis instances rather than reconstructing merely equal axes;
- private mutable scratch and assembly state never enter the collection;
- no unrequested prerequisite receives a collection-owned reference, so it is
  reclaimable after the builder returns when ordinary Python/Torch reachability
  permits;
- autograd may retain intermediates when a differentiable result requires
  them;
- no operation silently detaches, moves, casts, or host-materializes an
  existing input field;
- deterministic differentiable waveform operations preserve autograd where
  accepted; and
- stochastic count simulation and digitization make no blanket autograd claim.

TensorCore frozen records do not make Torch storage physically immutable.
Callers must not mutate tensors held by fields or collections through any
alias while they remain observable. Private functions that return raw tensors
rather than fields are outside TensorCore's field-result taxonomy, but their
scratch remains exclusively owned and cannot be exposed as a semantic value
that TensorDSLab later overwrites.

TensorDSLab initiates or enqueues every producer write before constructing and
exposing the corresponding result field, and it initiates no later write
through any alias to that storage. Accelerator work uses the current PyTorch
stream for the input device and returns without an implicit host
synchronization. Same-stream consumers inherit ordinary stream ordering;
cross-stream consumers must establish their own event/stream dependency before
reading. Strong references preserve lifetime, not write safety or stream
ordering.

Request selection reduces the long-lived result footprint; it does not by
itself promise a lower peak during construction. The simple functional planner
may keep local prerequisite references until final assembly, and autograd may
retain them longer through the result graph.

The pointwise waveform tail has one narrower MVP execution target. Each of
`_product_analog_waveform(...)` and `_product_digitized_waveform(...)` should emit
one fused accelerator kernel that reads each product input once and writes its
guaranteed-fresh product output without a target-sized temporary. The first implementation
uses the direct Torch expressions owned by those producers and lets
`torch.compile`/the selected backend perform product-local fusion. The
uncompiled equations remain the correctness reference.

One-kernel/no-target-sized-temporary behavior is an evidence-backed execution
claim, not something inferred from compact Python syntax. The implementation
work order must inspect the compiled graph and use accelerator profiling and
memory instrumentation on representative shapes. If the selected compiler
cannot prove this contract, the affected implementation slice returns to
Design and may use one purpose-built Triton or CUDA kernel without changing the
public API or adding a decorative Python wrapper. Product outputs themselves
remain guaranteed-fresh fields relative to their named inputs, so this does
not create an allocation-free chain claim.

Fusion across product boundaries, earlier release of arbitrary prerequisites,
and scratch reuse remain later measured execution optimizations with their own
retention, value-invariance, autograd, and lifetime tests.

The initial public API has no destination collection, public `out=`, or
`ReadoutWorkspace`. After real GPU profiling, a focused design may add reusable
scratch, output banks, stream binding, contiguity profiles, and allocation
instrumentation. It may not revive the old contract that overwrote a target
inside an already exposed valid semantic field or collection: any reusable
writable destination must remain raw, exclusive, and unexposed until all
producer writes have been initiated or enqueued, after which TensorDSLab may
construct the completed field exactly once. Advanced execution must preserve
request semantics and may not leak subsequently reusable scratch into retained
fields.

## Persistence And IO Are Deferred

This architecture defines no writer, persistence request, path, artifact format,
cache, schema, overwrite behavior, or default retained-for-disk product.
`products` means only fields retained in the returned in-memory collection.

Scientific configs contain no `persist` flags. A future artifact design will
let users choose which present product fields to write, and it must record the
scientific config/calibration needed to interpret them. Exact Python types will
map to separately versioned durable labels at that boundary.

## TensorG4DS Bridge

The future TensorDSLab-owned bridge must freeze:

- the exact accepted TensorG4DS input type and commit;
- event/provenance-to-`ExampleAxis` mapping;
- detector-channel-to-`ChannelAxis` mapping;
- dense photon-origin PE binning from the numeric left edges and exclusive stop
  implied by the exact caller-supplied `SamplingConfig`;
- normalization of each example to the accepted provenance origin followed by
  `sampling.build_axis()` with local start zero;
- `underflow_hit_count` and `overflow_hit_count` reporting for normalized hits
  outside the accepted half-open window, separately from arithmetic-overflow
  rejection;
- input/output dtype and device matrices;
- gradient rejection or preservation; and
- same-device tests proving no silent host staging.

The bridge returns a deeply validated dense truth `Photoelectrons` field. It
does not return a `ReadoutCollection`, apply timing jitter, parse native G4DS
files, expose CPU jagged storage to the readout builder, or relabel a
TensorG4DS value by subclass cast.

Conceptually, the future bridge is called with the same shared policy later
contained by `ReadoutConfig`:

```python
photoelectrons = build_photoelectrons(
    pe_hits,
    sampling=config.sampling,
)
```

For local hit time `t_ps`, the MVP numeric bin index is
`floor(t_ps / sampling.sample_period_ps.value)` and is retained only when it
lies in `[0, sampling.sample_count.value)`. The bridge does not infer the window
extent from the largest observed hit because empty tail bins are part of the
configured dense shape. The focused bridge design must freeze how the upstream
G4 floating representation is normalized to canonical integer picoseconds at
exact bin boundaries.

## TensorML And Reconstruction Boundaries

`ReadoutCollection` membership is unordered. A model-facing schema is an
explicit ordered tuple of product types:

```python
model_inputs = (AnalogWaveform,)
```

TensorML or a focused adapter resolves each type explicitly and owns positional
model order. It does not infer an ABI from request or collection insertion
order, move a full collection when one field is needed, or assume a field not
requested is available.

Future reconstruction may reuse `ExampleAxis` and `ChannelAxis` but owns its
field classes, collection classes, geometry, and preferred storage. A bridge
constructs new semantic leaves; it does not mutate or relabel readout fields.

## Parity Contract

`docs/parity.md` remains the authority for donor comparisons. This Design pass
synchronizes its rebuild-facing boundaries by replacing retired field IDs and
public atomic transforms with exact product requests and private diagnostic
seams.

Examples:

- public charge comparison becomes
  `simulate_readout(..., products=[Charge]) -> Charge`;
- isolated timing redistribution is a private diagnostic inside charge
  simulation and never a replacement truth field;
- pure/noise/analog/digitized comparisons request the corresponding type; and
- retention, prerequisite lifetime, and collection membership do not alter a
  scientific classification.

The scientific targets retained by this architecture are:

- conditional statistical timing-jitter parity under the binned latent-phase
  assumption, including redistribution means, variances, edge loss, and named
  tails rather than equality of the donor's finite digital normal law;
- conditional distributional homogeneous dark-count parity, including the
  finite-gate loss when private dark-count avalanches are jittered out of the
  configured window;
- intentional full-chain divergence where IV jitters recursive generated
  avalanches but the rebuild jitters truth and dark roots before its causal
  fixed-`K` edge-placement loop;
- the ideal unit-parent Poisson generation law as an analytic DiCT multiplicity
  oracle using the fixed IV mean, without Gamma-intensity or negative-binomial
  offspring; the implemented donor comparison remains statistical because the
  complete recursive and digital laws differ;
- post-binned DiCT placement only as a marginal/statistical comparison
  under an explicit `SamplingConfig`-to-`q_direct` mapping, aligned edge policy,
  and explicit divergence from IV's later independent parent/child timing
  jitter; the selected causal edge-placement law does not claim IV's signed
  post-binned displacement, hidden sibling timing, or multigeneration timing
  covariance;
- no IV-parity claim for optional DeCT, which is a TensorDSLab model with its
  own calibration and validation boundary;
- a full-cascade statistical comparison, rather than an accepted
  eventwise parity claim, reported as a function of `K`, with unit-count
  branching, recovery-weighted AP deposited charge, and finite-depth
  truncation classified explicitly;
- ordinary exponential afterpulse delay as an intentional donor correction,
  integrated during preflight into complete offset categories and an analytic
  right tail; the aggregate AP path does not use the finite inverse-transform
  primitive;
- fresh uniform within-bin afterpulse phase after aggregate jitter as an
  intentional binned approximation, without eventwise parent-child timing
  parity;
- AP charge from conditional mean recovery applied to
  the same realized delay-category counts, preserving binned count/charge
  covariance while intentionally omitting within-category recovery variance;
- intentional divergence from IV's recovery-weighted future branching: the
  rebuild uses only integer avalanche multiplicity for reproduction and
  never feeds deposited charge back into an offspring law;
- exact unit AP charge when `AfterpulseConfig.recovery is None`, and the
  composed exponential recovery response when it is present;
- statistical charge parity for named observables;
- equation-level TPC FEB-SNR pulse parity over aligned sample times using the
  explicit IV `tau_r -> tau_fast`, `tau_r + tau_l -> tau_slow` mapping, with
  sampled-peak normalization, explicit support, half-open windowing, and
  omission of IV's fractional-bin amplitude correction classified separately;
- equation-level Veto PDU Gaussian/two-erf pulse parity over aligned sample
  times using the explicit renamed parameter mapping, likewise separating the
  accepted sampled-support/sign convention from IV's heuristic crop,
  fractional-bin correction, and gate-boundary defects;
- exact zero-noise behavior plus statistical white-noise and PSD-shaped-noise
  parity after the documented PSD-to-target-grid integration; the finite
  precision-matched uniform lattice and Box-Muller radius cutoff preclude a
  literal unbounded-Gaussian or donor-digital-law distributional claim unless
  the donor comparison uses the exact same conversion and transform;
  IV raw spectral amplitudes are comparison evidence only after an explicit
  offline amplitude-squared-to-PSD-shape mapping plus an independently accepted
  absolute-power calibration; without that calibration, the absolute-PSD model
  is an intentional donor divergence rather than a fabricated parity claim;
  IV frequency coordinates label spectral lines and must not be copied into
  `frequency_left_edges_hz`; offline conversion must construct explicit
  interval left edges plus an exclusive stop, or fit an interval PSD, before
  comparison;
- intentional exact-length circular-covariance divergence from IV's long
  synthesized baseline and random-crop boundary;
- forced-zero synthesized DC in agreement with IV's broad zero-DC intent,
  while explicitly classifying the discarded fixed-record `[0, df / 2)` PSD
  power and making no absolute-power parity claim for that cell;
- exact analog composition and optional physical saturation at the
  `AnalogWaveform` product boundary; and
- exact in-range ADC codes under the frozen affine execution form, with
  intentional pre-conversion clipping divergence from IV's out-of-range
  integer wraparound.

Post-binned statistical parity remains acceptable without eventwise or bitwise
identity when tensor rebasing or RNG streams differ. Every production claim
still names observables, assumptions, units, tolerances, and fixtures. No donor
runtime becomes a production dependency.

## Validation Strategy

The rebuild validation matrix includes:

- exact TensorCore dependency, package-root imports, and ordinary-ABC static
  construction and exact lookup inference;
- static and Review evidence that every public semantic leaf has exactly one
  matching root in `__bases__`, with no mixin or other base, is `@final`,
  declares `__slots__ = ()`, adds no stored
  annotation or field, does not reapply `@dataclass` or override root behavior,
  and implements `_require(self) -> None`;
- public-boundary rejection of malformed documented inputs and unsatisfied
  supported relationships, without adversarial runtime policing of unsupported
  subclassing, class mutation, constructor bypass, or direct private calls;
- nonempty unique example/channel coordinates and at least two unique sample
  timestamps;
- exact `SamplingConfig` component types, positive period, count of at least
  two, and `window_stop_ps <= 2**63 - 1`;
- canonical ASCII `^(0|[1-9][0-9]*)ps$` acceptance plus rejection of signs,
  whitespace, decimals, exponents, alternate units, leading zeros, and values
  or derived stops outside signed int64;
- strict timestamp order, positive uniform integer-picosecond spacing, derived
  start/period/stop, a regular nonzero-start semantic subaxis, and O(1)
  full-source agreement on size, zero start, and configured period;
- exact sample-boundary fixtures proving `N` stored left-edge timestamps and no
  stop coordinate, assignment of times `0` and `i * T` to their corresponding
  bins, exclusion of negative times and the terminal time `N * T`, and separate
  `underflow_hit_count`/`overflow_hit_count` accounting;
- exact three-axis membership in any accepted order;
- tensor shape/axis agreement and `torch.strided` layout;
- intrinsic dtype checks and explicit deep-value validation;
- collection rejection of empty or unknown membership;
- collection same-axis, same-device, and common-floating-dtype coherence;
- one-pass product iterable consumption;
- empty, duplicate, non-class, base-class, and foreign-class request rejection;
- request order having no result semantics;
- exact transitive config preflight before RNG or writes;
- proof that every MVP calibration value is scalar and applies uniformly to
  all example/channel positions in one invocation, without channel-coordinate
  lookup, implicit parameter broadcasting, or tensors hidden inside configs;
- `_product_analog_waveform(...)` reference checks for no saturation, each
  one-sided bound, and two-sided bounds, including exact bound values and proof
  that the input fields remain unchanged;
- zero-input waveform fixtures proving that neither `PureWaveform`,
  `NoiseWaveform`, nor `AnalogWaveform` receives a deterministic pedestal, and
  that the digitizer's zero-voltage code follows only from its accepted affine
  transfer;
- digitizer preflight checks for finite representable `maximum_code`, `gain`,
  `span`, `slope`, and `intercept`, followed by endpoint, asymmetric-zero-code,
  code-transition-neighborhood, truncation, `torch.int32`, and pre-cast
  clamping fixtures with no unsigned wraparound;
- equivalence of each direct waveform-tail production expression to its
  unfused reference equation under the accepted dtype/backend numerical
  contract;
- static and runtime proof that the waveform tail contains only the two owning
  product producers, materializes `AnalogWaveform` even when it is an
  unretained prerequisite, and does not introduce decorative pointwise Python
  wrappers or cross-product fusion;
- conditional accelerator compiler-graph, profiler, and memory evidence that
  each waveform-tail product producer emits one fused backend kernel and no
  target-sized temporary; absent such evidence, no fusion claim is made and
  the affected implementation returns to Design;
- exact `PureWaveformConfig.model` rejection of foreign/base values,
  exact dispatch to both accepted pulse models, and proof that one selected
  model and scalar parameter set are applied uniformly without inferring
  physical family or calibration from channel strings;
- exact TPC FEB-SNR and Veto PDU donor-equation oracles, including
  `tau_fast = tau_r`, `tau_slow = tau_r + tau_l`, `tau_slow > tau_fast`, and
  every renamed Gaussian/two-erf parameter mapping;
- sampled-template normalization/support oracles, signed peak scaling, causal
  convolution alignment, same-length truncation, and no hidden
  gain/inversion/baseline;
- parity fixtures proving the separately classified differences from IV's TPC
  continuous normalization, heuristic TPC/Veto support, post-convolution
  inversion, fractional-bin amplitude correction, and gate-edge behavior;
- pulse-support fixtures proving inclusion exactly when `j * T < support_time`,
  exclusion at the support stop, and preflight rejection of an empty,
  nonfinite, or zero-extremum sampled template;
- exact `PsdNoiseConfig` left-edge/density count agreement, zero-start, strict
  left-edge order, final-left-edge/stop ordering, nonnegative finite absolute
  density, constructor-level nonzero supplied power, and request-preflight
  nonzero retained power after DC suppression;
- PSD coverage through the `SamplingConfig` Nyquist frequency, deterministic
  overlap integration onto the target one-sided frequency intervals,
  pre-suppression power conservation, and explicit DC-cell discard without
  redistribution or raw FFT amplitudes;
- PSD boundary fixtures with coincident source/target left edges, the final
  exclusive stop, exact odd- and even-`N` target-cell left-edge arrays, no
  double ownership at shared boundaries, and proof that Fourier basis
  frequencies never populate the public left-edge tuple;
- exact small odd- and even-`N` coefficient-to-time-domain oracles for
  `torch.fft.irfft(..., n=N, dim=-1, norm="backward")`, including cosine-only
  and sine-only interior bases, the `N / 2` interior scale, the `N` real
  Nyquist scale and alternating output, explicit real/imaginary
  standard-normal component convention, complex dtype, and absence of a DC draw;
- endpoint and degenerate-record fixtures: `N=1` is outside the accepted
  `SamplingConfig` domain and would retain no PSD power after DC suppression,
  `N=2` is a real-Nyquist-only process, odd-`N` terminal imaginary components
  affect the output, even-`N` Nyquist imaginary components are exactly zero,
  and zero-power cells produce exact zero coefficients;
- proof that no native complex-normal variance convention, implicit FFT
  normalization, post-transform demeaning, standard-deviation normalization,
  power normalization, or DC-power redistribution changes the frozen law;
- PSD-shaped-noise ensemble checks for an exactly zero real DC coefficient and
  per-record sample mean, retained expected variance, the accepted odd/even
  circular covariance equation, target spectral shape outside the discarded
  DC cell, real Nyquist behavior, fluctuating finite-record power, independent
  channel rows, zero cross-row covariance in expectation, Parseval mean-square
  power, coefficient moments, and the flat-density `-1 / (N - 1)` nonzero-lag
  correlation caused by the documented DC notch; ensemble variance oracles use
  population mean-square/`correction=0`, not an unrelated sample correction;
- exact non-boolean root-seed validation in `[0, 2**64)`, deterministic
  closure acceptance of `None`, and stochastic-closure rejection of `None`
  before any draw or write;
- exact Random123 `Threefry4x32_R<20>` known-answer vectors on every supported
  scalar, vectorized, eager, compiled, CPU, and conditional CUDA engine path,
  with expected words held as independent fixed fixtures rather than generated
  by the implementation under test;
- exact schema-v1 key/counter/lane packing for zero and maximum seeds, differing
  seed halves, every accepted private stream, positions around the `2**32`
  split and at the accepted maximum, zero and maximum quantum ordinals, and
  raw-word ordinals `0`, `1`, `2`, `3`, `4`, and the accepted maximum;
- explicit lane-three-to-next-block-lane-zero rollover, numerical low/high word
  order, the `0x54445331` domain tag, and deterministic rejection of every
  position, population, and raw-word bound violation without narrowing casts;
- exact `float32` and `float64` closed-open and open-open conversion oracles for
  zero, maximum, and representative raw words, including endpoint exclusion,
  numerical two-word order, discarded-bit behavior, and no reuse of discarded
  bits;
- Bernoulli ties-to-even threshold construction, exact threshold-boundary word
  comparisons, quantized probability error no greater than `2**-33`, and
  draw-free threshold-zero and threshold-`2**32` results;
- Box-Muller raw-word schedule and ordered cosine/sine components at one exact
  positional address, scalar-consumer spare-result discard, two-component PSD
  use, native-dtype execution, same-backend repeatability, component moments
  and covariance, and explicit `float32`/`float64` radial cutoffs;
- exponential endpoint, mean, and finite-tail fixtures in each accepted dtype,
  with no hidden widened `float64` path for a `float32` operation;
- globally unique fixed numeric operation-stream assignments that do not
  change with the requested subset, enabled branches, or later appended
  operations;
- arbitrary-rank and arbitrary-shape positional RNG oracles, including scalar
  and empty results where the selected backend accepts them;
- row-major logical flat positions derived from current dimension order rather
  than physical storage offsets, including noncontiguous-view fixtures;
- for every accepted iterative stochastic role, exact
  virtual-leading-iteration addressing `p = j * N + u`, checked
  `G * N <= 2**63`, global rather
  than block-local iteration identity, and no activity-compacted positions;
- deterministic source-quantum and raw-word ordinals, collision-free address
  encoding across every accepted root-seed/stream/position/quantum/raw-word
  tuple, and the distinction between address uniqueness and ordinary repeated
  32-bit output values;
- exact raw-word agreement between the scalar oracle and all accepted
  vectorized or compiled implementations, plus proof that TensorDSLab neither
  reads nor mutates PyTorch global RNG state and does not construct a
  `torch.Generator`;
- proof that axis classes, coordinate strings, and timestamps do not enter the
  random address or hot-path RNG inputs;
- exact same-backend repeatability only for an unchanged positional schema,
  values, config, dtype, algorithm/version, and seed;
- proof that coordinate relabeling alone preserves positional bits while
  changing semantic association, with no coordinate-identity or
  tensor-permutation invariance claim;
- selection/chunk noninvariance and same-seed separate invocations of different
  shapes explicitly shown to reuse the random prefix over overlapping flat
  positions;
- every prerequisite executed at most once;
- exactly requested final membership;
- unrequested prerequisite absence;
- source `Photoelectrons` identity and immutability when retained;
- all 16 structural presence/absence combinations of the dark, jitter,
  correlated, and smearing stages, proving that a skipped block constructs no
  identity result or replacement tensor, smearing without correlation uses the
  unit-response `S1 == S2` identity, lowercase `charge` remains a tensor, and
  its final value alone becomes the uppercase `Charge` payload;
- exact private orchestration with `_simulate_dark_counts(...)` before
  `_simulate_timing_jitter(...)` whenever both blocks execute;
- timing jitter affecting the then-current private working counts, including
  dark roots when present, but never the public truth field;
- `K=0` roots-only and `K=1` direct-child off-by-one fixtures for the one
  coupled `_simulate_correlated_avalanches(...)` path;
- all eight DiCT/DeCT/AP enablement combinations, including exact
  all-disabled identity and no-draw behavior;
- one frozen unmarked integer frontier per generation, with every retained
  child entering the next frontier exactly once and no charge value entering
  an offspring law;
- separate DiCT and DeCT rate fields, ordinary Poisson draws, streams,
  accumulators, and right-overflow diagnostics, with no rate superposition,
  conditional mode split, Gamma latent, or negative-binomial substitution;
- analytic fixed-, exponential-, and zero-clipped-normal-delay PMFs under
  independent per-edge uniform phase marginalization, plus dtype-aware
  preparation tolerances, including the clipped-normal zero atom, zero-delay
  DiCT staying in-bin, no shared or inherited phase, and no correlated-stage
  underflow;
- clipped-normal fixtures proving
  `P(Delta = 0) = Phi(-location_ns / sigma_ns)`, PMF-plus-right-tail
  normalization without silent renormalization, agreement with an explicit
  edge-level Monte Carlo reference, and convergence toward
  `FixedDelayConfig(location_ns)` as positive `sigma_ns` approaches zero;
- AP's one-child multinomial law, shared realized categories for integer count
  and deposited charge, and separate stop, retained, and right-overflow
  accounting;
- `recovery=None` reducing AP `count`, `charge`, and `charge_square_sum` to the
  unit-weight law, plus configured recovery changing deposited charge without
  changing descendant probabilities;
- the exact integer mechanism-count invariant and dtype-aware validation of the
  mathematical S1/S2 ledger identities, including
  `afterpulse_charge_square_sum` as the sum of category weights squared rather
  than `afterpulse_charge**2`;
- all overflow excluded from the retained frontier, total count, `S1`, `S2`,
  terminal `Charge`, and waveforms;
- final-frontier truncation semantics after generation `K`, checked count
  overflow, and no partial valid result after an algorithm failure;
- independent-edge prepared PMFs compared with an explicit edge-level Monte
  Carlo reference, with the intentionally omitted shared-phase and
  within-category recovery variances measured as named approximation
  boundaries;
- zero dark rate and zero jitter as exact identities without unnecessary RNG
  consumption;
- jitter conservation through explicit retained and dropped buckets;
- dark-only finite-window ensembles matching
  `mean_t = lambda * sum_s(P[s, t])`, Poisson variance and zero probability,
  expected edge depletion, and expected total drop;
- conditional isolated IV dark-count-plus-jitter comparison under matching
  gate and drop conventions, while explicitly excluding full recursive-chain
  timing equivalence;
- common output invariance when unrelated requested products change;
- private product-builder and submodel fixtures;
- public single-product and multi-product composition fixtures;
- an exact returned source `Photoelectrons` field when requested, guaranteed-
  fresh generated fields independent of every named input, and pairwise
  storage-independent generated result fields;
- no new write initiated or enqueued after a field becomes observable, plus
  same-stream and explicit cross-stream ordering behavior;
- ordinary-`torch.Tensor` execution evidence, with custom tensor subclasses and
  dispatch modes explicitly unsupported rather than exhaustively detected;
- no silent CPU, NumPy, list, move, input-cast, or detach path;
- transform-specific scientific/parity tests;
- CPU tests and conditional CUDA tests with accurate qualifications;
- stale `0.6` names and compatibility aliases absent; and
- import isolation from TensorG4DS, TensorML, Projects/dag, and IO backends.

## Rebuild Migration

This Design pass accepted the architecture, synchronized the live Design
documents, and wrote the focused Stage 3 structural-foundation work order.
Stage 3 subsequently selected exact TensorCore `0.7.0` commit
`b454d738f6385ce6489d85492a618a3dab139bb6`, passed fixed-pin consumer probes,
and merged the product/config/collection foundation. Historical work orders
and governance records remain unchanged.

The completed structural step was:

- commit synchronized Design authority, verify the persistent role routes,
  replace the `0.6` package with the typed axes, sampling, product/config, and
  collection foundation, then clear fixed-commit Validation, independent
  Review, fast-forward merge, and Design closeout.

The remaining production sequence is:

1. Implement deterministic product producers under focused work orders.
2. Freeze and implement the private RNG and stochastic-noise contracts under a
   focused work order.
3. Freeze the remaining Poisson/multinomial stream, numerical-tail, and
   count-bound contracts, then implement the fixed-`K` charge simulation in
   parity-scoped slices.
4. Publish request-aware `simulate_readout(...)` only after every required
   producer exists and its complete closure can be preflighted.
5. Profile real GPU memory and execution before designing workspace/output
   reuse.
6. Design the exact TensorG4DS-to-truth-Photoelectrons bridge.
7. Design explicit TensorML/reconstruction adapters.
8. Design durable artifacts only after in-memory contracts stabilize.

Each production slice uses the repository Implementation/Validation/Review
loop and fixed-commit evidence. No compatibility alias preserves `0.6`.

## Supersession Ledger

The accepted rebuild architecture and completed Stage 3 foundation replace the
following historical `0.6` contracts.

| Historical `0.6` contract | Implemented Stage 3 or accepted rebuild target |
| --- | --- |
| TensorCore `0.6` ID/layout records | TensorCore `0.7` semantic roots |
| `TensorAxisId`, `TensorFieldId`, `IdSequence` | exact final leaf classes |
| `ExampleId`, `ChannelId` objects | strings scoped by exact axis type |
| `TensorLayout` plus `shared_axes` | ordered axes directly on each field |
| three required axes plus optional extra shared axes | exactly three readout axis types |
| sealed generic `TensorField` | six direct TensorDSLab product leaves |
| loose axis/field constants and registries | class-owned schema and typed calls |
| count-only sample plus collection-sidecar `SampleGrid` | shared `SamplingConfig` policy plus its realized timestamp-backed `SampleAxis` |
| `DigitizedWaveformSpec` sidecar | builder config held externally; artifact binding deferred |
| partial ordered pipeline snapshots | request-selected completed unordered results |
| descendant invalidation | immutable one-shot construction |
| public timing transform replacing photoelectrons | truth photoelectrons; private charge-only jitter |
| public atomic collection transforms | private typed product producers plus one public request API |
| generic selection/movement plus reconstruction | explicit downstream operations when needed |
| `readout/tensors.py` | retired |
| global config/field/builder modules | product-owned `types.py` and `_product.py` modules plus `readout.simulation` |
| immediate public workspace architecture | functional first; optimize after measurement |
| field-ID model selection | explicit ordered product-type selection |
| field-ID parity boundaries | product-request/builder parity boundaries |
| semantic-coordinate RNG identity | fixed numeric operation streams plus logical flat tensor positions |

This synchronization pass updated:

- `AGENTS.md`;
- `CONTRIBUTING.md`;
- `docs/overview.md`;
- `docs/design.md`;
- `docs/decisions.md`;
- `docs/architecture/tensors.md`;
- `docs/architecture/readout.md`;
- `docs/parity.md`;
- `docs/validation.md`; and
- `docs/implementation/index.md`.

It also created the Stage 3 work order, which was later implemented, validated,
independently reviewed, fast-forwarded, and accepted as Merged / Closed.
Governance records and earlier completed work orders remain historical records.

## Closed Decisions And Remaining Design Gates

The rebuild package tree and import ownership are closed. Shared semantic axes
and sampling live in `tensor_dslab.common`; each readout product owns its field,
configs, validation, and eventual private product builder; `readout.types`
contains only `ReadoutConfig` and `ReadoutCollection`; and
`readout.simulation` owns the one public orchestration function. The private
readout requirements and RNG modules are not public APIs. `Photoelectrons` is
an already-produced input with neither a config nor a producer. Reopening this
tree requires a concrete import-cycle, cohesion, or implementation-size finding
rather than a preference for layer-oriented grouping.

The MVP sampling and semantic timestamp contract is closed. One exact
`SamplingConfig` owns positive integer `sample_period_ps`, `sample_count >= 2`,
and a signed-int64 `window_stop_ps`. Its full axis begins at zero, contains
exactly the `N` canonical ASCII left-edge timestamps `i * T` in lowercase
picoseconds, and omits the exclusive `N * T` stop. Every `SampleAxis` is a
regular, increasing, period-bearing signed-int64 time axis; only complete
simulation inputs additionally require zero start and exact config agreement.
Kernels use numeric config values and indices, while upstream floating-time
normalization remains a TensorG4DS-bridge decision.

The two pulse-shape equations and their donor-to-config parameter mappings are
closed provisionally for the MVP by the IV adoption above. Later collaborator
review may motivate a new scientific model or calibration, but it does not
block implementation of these explicitly provisional equations and must not
silently change them inside an implementation stage.

The complete PSD-shaped-noise mathematical contract is also closed for the
MVP: absolute one-sided interval density, overlap integration onto fixed-length
odd/even target cells, explicit `[0, df / 2)` DC-cell discard without
redistribution, Gaussian one-sided coefficients, two independent real
standard-normal components per interior coefficient,
`torch.fft.irfft(..., n=N, dim=-1, norm="backward")`, the documented endpoint
scales, retained expected variance, circular covariance, independent rows, and
no post-transform normalization or hidden longer-record crop. The later RNG
stream table assigns the private PSD coefficient stream, but the accepted
precision-matched uniforms and Box-Muller pair now define how its two ordered
standard-normal components are generated. Neither choice reopens this PSD law.

Waveform baseline ownership is closed as well. The MVP has no deterministic
analog pedestal: pure and noise are zero-referenced voltage components, analog
is their optionally saturated sum, and the digitizer's affine transfer owns
the nonzero ADC code corresponding to 0 mV.

The private raw RNG core and positional address encoding are closed. RNG schema
`tensordslab.threefry4x32-20/v1` uses the exact standard Random123
`Threefry4x32_R<20>` word algorithm, numeric seed/stream/domain-tag key packing,
logical-position/quantum/raw-word-block counter packing, lane selection, and
accepted bounds specified above. This closes raw-bit generation only; it does
not select the stream numbers for deferred algorithms.

The generic MVP distribution layer is also closed: precision-matched
Random123-style `float32` and `float64` closed-open/open-open conversions,
ties-to-even 32-bit Bernoulli thresholds, native-dtype exponential inversion,
and address-local ordered Box-Muller pairs. The documented finite exponential
and normal tails are accepted bounded-MVP approximations and parity
qualifications, not hidden claims of unbounded continuous support.

Stage 3 completed the TensorCore selection, inherited-constructor typing,
public-import, and fixed consumer-probe gate at exact commit
`b454d738f6385ce6489d85492a618a3dab139bb6`.

The remaining gates are:

1. Exact private numeric stream table, Poisson algorithm and crossover,
   operation-specific per-quantum versus exact aggregate sampling choices,
   Charge-specific execution-dtype and raw-word-budget choices,
   rejection/exhaustion behavior, and supported execution-mode repeatability
   evidence. The Threefry engine, address packing, generic uniform conversion,
   Bernoulli threshold, exponential inversion, and Box-Muller mapping are no
   longer open in this gate.
2. Exact fixed/exponential/zero-clipped-normal offset-PMF preparation
   precision, stable normal-CDF and right-tail evaluation, normalization and
   tail-rounding tolerance; supported `maximum_generations`, rate,
   source-count, and accumulated-count bounds; and hard checked-overflow
   behavior. These are execution-support limits, not alternate scientific
   laws.
3. Waveform-tail execution acceptance: scalar constant dtype/precision,
   compiler/execution mode, equivalence to the frozen unfused reference,
   one-kernel/no-target-sized-temporary instrumentation, and the fallback gate
   for a purpose-built kernel. Cross-product analog/digitized fusion remains
   excluded.
4. Digitization-config association for independent/durable consumers.
5. Exact TensorG4DS source and dense truth-binning bridge, including provenance
   origin, left-edge construction, exact boundary assignment at `0`, `i * T`,
   and exclusive `N * T`, plus `underflow_hit_count` and
   `overflow_hit_count` accounting.
6. Whether typed collection convenience properties materially improve the API.

The fixed-`K` correlated-avalanche model itself is closed at the scientific
algorithm level: exact config ownership, independent per-edge phase closure,
ordinary separate DiCT/DeCT Poisson laws, AP's bounded categorical law,
fixed/exponential/zero-clipped-normal CT delay families, optional composed
exponential recovery response, unmarked cross-feeding, S1/S2 ledgers, terminal
smearing rule, causal right-overflow policy, and private diagnostic vocabulary
are selected above. The remaining Charge gates are the sampler/RNG and
supported-numerical-domain items 1 and 2, plus concrete parity tolerances. No
work order may substitute a same-bin closure, generation-wave plan, marked
recovery process, Gamma-Poisson law, or separate public mechanism pipeline for
this baseline.

`Config(ABC)`, product-level `persist` flags, jagged builder input, and public
truth-replacing timing jitter are deliberately omitted. Persistence remains a
future focused design.

## Collaborator Example

Illustrative pseudocode, using already-supplied validated `photoelectrons` and
`config` values:

```python
readout = simulate_readout(
    photoelectrons,
    products=[
        AnalogWaveform,
        DigitizedWaveform,
    ],
    config=config,
    seed=1234,
)

analog: AnalogWaveform = readout.field(AnalogWaveform)
digitized: DigitizedWaveform = readout.field(DigitizedWaveform)

assert readout.field_types == frozenset(
    {
        AnalogWaveform,
        DigitizedWaveform,
    }
)
```

Requesting truth beside a derived product is unambiguous:

```python
readout = simulate_readout(
    photoelectrons,
    products=[Photoelectrons, Charge],
    config=config,
    seed=1234,
)

assert readout.field(Photoelectrons) is photoelectrons
charge = readout.field(Charge)  # reflects every effective configured charge stage
```

The builder computes what is necessary, retains exactly what was requested,
and never changes the truth input.
