# Contributing

TensorDSLab should be developed like professional scientific software: clear
ownership boundaries, typed public APIs, deterministic behavior, focused tests,
small coherent changes, and documentation that explains contracts rather than
narrating code.

## Governance And Delivery Maturity

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`, bound to
exact candidate `d634401a853915edeb4f83df4a4943b3553deced` and recorded in the
[package governance records](docs/governance/index.md). Conformance remains
`Not evaluated`, Coordination remains `Deferred`, and Profile B remains
`Disabled`.

The current identity and maturity are:

```text
Project/display name: TensorDSLab
Python import: tensor_dslab (accepted on main through Maintenance 5)
Distribution name: tensor-dslab (accepted metadata; not published or released)
Delivery maturity: active development / pre-deployment
Package maturity: Maintenance 6 Merged / Closed
Next production gate: Maintenance 7 TensorCore 0.15 adoption;
User-authorized / Dispatched
Stage 8: separately stopped; any restart requires a new Design authority after
Maintenance 6
```

Stage 1 is Design-complete, and Stage 2 is Merged / Closed on `main` at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. That historical stage established
the first package metadata, structural package, and focused test suite. Its
evidence remains limited to the exact repository/dependency commits, Python
and PyTorch versions, CPU execution mode, and conditional CUDA skips recorded
in the Stage 2 work order. No wheel or published artifact was built, and
TensorDSLab makes no deployability, release-readiness, backward-compatibility,
or broad cross-package compatibility claim.

Maintenance 1 was separately dispatched to correct only readout public-name
and module ownership. Before Review's clean fast-forward, its feature-branch
form is candidate evidence; if the updated surface is read on `main`, that
merge gate has completed. It changes no collection behavior, TensorCore pin,
scientific contract, deployment state, or compatibility finding.

The TensorCore `0.7` rebuild in `docs/architecture/rebuild.md` is implemented
through Stage 3. Exact implementation candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde` cleared fixed-commit Validation
and independent Review; Review's clean fast-forward and post-merge closeout
produced `97e17c3177ac217aeb42a077db78f4bd223d51fa`, and Design accepted the
result after an independent post-merge audit on clean `main`
`5ff13eb3c0735abfda454a334be59faac35259c2`. Stage 2 and Maintenance 1 remain
historical TensorCore `0.6` evidence.

Stage 4 is Merged / Closed through exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da` and Review's clean fast-forward
closeout `b3ebfcd9473537dd385195afea374bd2f426c6c0`. It implements exactly the
private pure, analog, and digitized producers under a functionality-first
acceptance gate. Fixed-commit Validation, independent Review, and Design's
post-merge audit found no unresolved issue.

Stage 5 is Merged / Closed through exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd` and Review's evidence-only
closeout `c6a506d3658b24197806b9e230480211a254a35a`. It implements the private
positional Threefry reference, noise-consumed fixed-point uniforms and
Box-Muller pair, and complete exact-zero, IID-white, and caller-supplied PSD
noise producer. Its fixed-commit Validation, independent Review, and Design
post-merge audit found no unresolved issue. CUDA was unavailable, so the
evidence is eager CPU-only and makes no GPU execution or performance claim.
Stage 6 is Merged / Closed through exact implementation candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58` and Review's evidence-only
closeout `ea979862b05f4ef543f6971c86641df317232479`. It implements the complete
private Charge producer, aggregate samplers, and fixed-generation correlated-
avalanche slice. Fixed-commit Validation, independent Review, and Design's
post-merge audit found no unresolved issue. CUDA was unavailable, so its
evidence is eager CPU-only and makes no GPU execution or performance claim.
Stage 7 public orchestration is Merged / Closed through exact Review-cleared
candidate `6dd55024685013fb9412a7247d3ddde7be1a3177`. Its fixed-commit
Validation, independent Review, and post-merge verification found no unresolved
issue. CUDA was unavailable, so its evidence is eager CPU-only; measured GPU
characterization, optimization, and integration remain later work.

Maintenance 3 Environment-Qualified Stochastic Continuity is Merged / Closed
through exact Review-cleared candidate
`dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0` and its five-document Design
closeout. It qualifies completed stochastic literals by numerical stack and
changes no production, dependency, RNG, or scientific contract. Maintenance 4
Runtime Action Ownership is **Merged / Closed** through exact Review-cleared
supplemental candidate `b3c7c907004741ba67b8b92a54bbdc8c85216dda` under
`docs/implementation/maintenance_4_runtime_action_ownership.md`. It implements
the internal behavior-preserving split into product-owned Runtime, prepare,
produce, and validate actions without adding a public surface. Separate fresh
Validation and Review full-A100 source/archive runs cleared the exact final
bytes. The first Stage 8 attempt remains stopped evidence and is not executable
authority.

Maintenance 5 TensorCore 0.13 Compact Axes And Sampling is **Merged / Closed**
through exact Review-cleared supplemental candidate
`81ad2f52fe4a1966e5b3a0ceb5063138e42e731f` and Design closeout
`021694b9479d02546405f6a815aedf21c9c831a4`. It adopts exact TensorCore
`0.13.0`, installs compact count/label/regular semantic axes, removes
`SamplingConfig`, and derives execution sampling once from the source
`SampleAxis`. Maintenance 6 is Merged / Closed through exact Review-cleared
target `0257fb477ee04556ebbe26351123ae610b5d7925`: collaborators configure
physical values with canonical copied Pint quantities, while preparation
extracts plain execution values and production and validation remain
unit-free. Local `main` remains unpushed pending the separate TensorCore
`0.15.0` adoption and exact integrated CUDA gates.

[Maintenance 7](docs/implementation/maintenance_7_tensorcore_0_15_adoption.md)
is User-authorized and Dispatched. It adopts exact published TensorCore
`0.15.0`, replaces `logical_positions(...)` with validated `RngPositions`,
uses TensorCore's generic validation parts where the contracts match, and
centralizes the unchanged readout RNG namespace. It also makes pulse Configs
store positive amplitude magnitudes and applies fixed DS20k negative polarity
once in preparation, preserving calibrated rendered results. Streams,
addresses, other science, Pint ownership, products, and facades remain
unchanged. Its local package gate makes no fresh CUDA claim; separately
authorized integrated CUDA evidence follows only after the adoption closes.

The `tensor-dslab` distribution spelling is accepted package metadata, not an
installed, published, or released distribution claim. GPU residency
and no-silent-host-materialization requirements are TensorDSLab Design
constraints for future boundaries, not evidence that any TensorG4DS,
TensorCore, or TensorML baseline is compatible. A breaking change affecting
multiple repositories requires every affected package Design authority and a
synchronized migration plan. Compatibility shims, aliases, or deprecation
windows require demonstrated value and explicit Design acceptance.

## Repository Identity

TensorDSLab is a clean-slate, tensor-native detector data-lab package in this
intended data flow:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared tensor substrate rather than a pipeline stage. The
diagram is not an import graph and does not claim that the still-deferred
TensorG4DS-to-TensorDSLab or TensorDSLab-to-TensorML boundaries already exist.

TensorDSLab owns post-TensorG4DS detector/readout semantics and future cache
contracts:

- mapping an accepted future TensorG4DS public product into TensorDSLab
  provenance, coordinates, detector-window/readout-grid semantics, and binned
  photon-origin photoelectrons when a focused integration stage accepts that
  bridge;
- building the typed `ReadoutCollection`, its recognized product fields, and
  optional thin readout provenance/context records;
- defining future reconstruction examples and reconstruction products;
- defining TensorCore-backed semantic collections, fields, and domain
  transforms where a stage accepts a tensor-native contract;
- writing, validating, loading, and performing deterministic storage-level
  compaction of strict durable caches only after in-memory product contracts
  are stable;
- exposing DAG-compatible executables, operation specs, or recipe fragments
  only after local product and cache contracts are stable.

TensorDSLab does not own native G4DS parsing or simulation execution,
TensorG4DS low-level products or algorithms such as deposit clustering,
generic TensorCore primitives, downstream source adaptation for model
training, model assembly, training loops, evaluation loops, metrics,
checkpoints, campaign orchestration, scheduler behavior, repair, or retries.

Projects/dag owns scheduling and fan-in for future cache compaction, including
campaign/cross-shard grouping, retries, repair, and execution policy.
TensorDSLab owns only the deterministic storage primitive over caller-supplied
complete compatible products.

The production integration target keeps tensor payloads resident on one
explicit accelerator device across TensorG4DS, TensorDSLab, and TensorML.
Boundary code must not silently call `.cpu()`, `.numpy()`, serialize/reload, or
otherwise materialize payload data on the host as a package handoff. A
TensorG4DS-to-TensorDSLab bridge is a semantic transformation and may create
new tensors on that device; same-device residency is not a claim that the
input and output axes or storage are interchangeable or that every transform
is zero-copy. TensorCore axes, TensorDSLab configs, and other small semantic
records remain ordinary host-side objects. Device movement is always explicit.

The first discrete TensorG4DS bridge carries no end-to-end autograd promise.
It must not detach silently and should reject gradient-sensitive inputs unless
a focused differentiable detector contract is accepted. Later deterministic
waveform work must state and verify its exact autograd result contract.

Historical predecessor code, if consulted outside this repository, is
parts-bin material only. Promote scientific facts, product semantics, cache
guarantees, algorithms, fixtures, and tests deliberately into TensorDSLab docs
and tests. Do not preserve old package layouts, helper frameworks,
compatibility shims, DAG wiring, or representation shortcuts by default.
Use `docs/parity.md` to classify every donor comparison and intentional
divergence; promoting a donor idea does not itself establish parity.

## Build Philosophy

Define the MVP early, but build toward it from the inside out. The first
accepted MVP direction is the post-binned tensor-native readout path:
already-binned photon-origin primary photoelectrons, aggregate SiPM charge
simulation, waveform products, analog waveform composition, and optional
digitization.

Native G4DS parsing belongs upstream of TensorDSLab. The typed TensorG4DS
handoff, detector-window construction, photoelectron binning, IO boundaries,
durable cache formats, table/array codecs, manifest rules,
compaction, package-local CLIs, DAG-compatible operation specs, recipes,
executable doors, and downstream adapter contracts should not shape the first
post-binned readout module boundaries.

Early implementation stages should be judged by whether the local field
dependency graph and collection contract are typed, deterministic, testable,
and easy to reason about. Compatibility with external orchestration or
downstream training packages is deferred until the local TensorDSLab contracts
are stable.

Scientific configs should describe physics and readout behavior. Exact
stochastic leaf configs may own immutable TensorCore `RngKey` role identities.
Invocation seeds, RNG algorithm instances, device, dtype, movement, output
storage, accelerator stream, and execution/chunking policy are runtime
concerns. Do not hide persistence, placement, allocation, mutation, mutable RNG
state, or device-stream policy inside scientific config records.

## Sibling Repository Shape

TensorDSLab should feel like a sibling of TensorML and TensorCore in engineering
style: explicit boundaries, staged implementation, typed records, small APIs,
clear docs, and disciplined review gates. Use the shared style, but do not copy
another repository's domain boundaries blindly.

The tree below is a menu of accepted surfaces, not a requirement to create
empty files:

```text
TensorDSLab/
  AGENTS.md
  CONTRIBUTING.md
  README.md
  docs/
    overview.md
    design.md
    decisions.md
    parity.md
    validation.md
    architecture/
      common.md
      detector.md
      readout.md
      reconstruction.md
      caches.md
      tensors.md
    implementation/
      index.md
      stage_<n>_<name>.md
  pyproject.toml              # when package metadata is accepted
  tensor_dslab/
    common/
      axes.py                 # ExampleAxis, ChannelAxis, SampleAxis
    readout/
      config.py               # ReadoutConfig
      collection.py           # ReadoutCollection
      simulation.py           # implemented Stage 7 public orchestration
      requirements.py         # non-exported shared readout relationships
      runtime/
        prepare.py            # ReadoutRuntime and prepare_readout
        sampling.py           # SamplingRuntime and prepare_sampling
      photoelectrons/
        field.py
        runtime/validate.py
      charge/
        config.py
        field.py
        runtime/
          prepare.py
          produce.py
          validate.py
          effects/
            counts.py
            delays.py
            dark_counts.py
            timing_jitter.py
            correlated_avalanches.py
            smearing.py
      pure_waveform/
        config.py
        field.py
        runtime/{prepare.py,produce.py,validate.py}
      noise_waveform/
        config.py
        field.py
        runtime/{prepare.py,produce.py,validate.py}
      analog_waveform/
        config.py
        field.py
        runtime/{prepare.py,produce.py,validate.py}
      digitized_waveform/
        config.py
        field.py
        runtime/{prepare.py,produce.py,validate.py}
  tests/
```

The project/display folder is `TensorDSLab`; the Python import package is
`tensor_dslab`. Do not create a flat TitleCase Python package that imports
as `TensorDSLab`; keep semantic subpackages directly below the import root.

Do not create placeholder modules to reserve architecture. Add a module only
when there is a real concept, behavior, or contract to house.

This product-centered tree is the implemented Maintenance 5 baseline. Its
product and runtime structure is the merged Maintenance 4 implementation;
Maintenance 5 removes only the redundant public `common/sampling.py` policy
owner while retaining the private source-derived sampling runtime. It combines the public
ownership established by Maintenance 2 and Stage 7 with non-exported product
runtime actions. Maintenance 2 is Merged / Closed
through exact candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`; it realizes the product/module
ownership paths and removes the former Stage 5/6 `types.py`, `_RngStream`, and
`readout/_random.py` surfaces without shims. Stage 7 completes
`readout/simulation.py` and the public orchestration surface.
TensorCore fulfilled the historical consumer proposal in published version
`0.9.0` at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. TensorDSLab's Maintenance 2
implementation pins that dependency. Its accepted evidence is eager CPU-only:
157 tests ran, 148 passed, 9 conditional CUDA tests skipped, and Pyright
reported no diagnostics against either exact dependency form.
Product `field.py` modules own final field leaves and their cheap intrinsic
TensorCore `_require()` narrowing; product `config.py` modules own configs.
Each generated product's non-exported `runtime/` package owns one concrete
final frozen ProductRuntime plus explicit `prepare_*`, `produce_*`, and `validate_*`
actions. Charge's scientific submodels, multinomial/category orchestration,
and count bookkeeping remain private under `charge/runtime/effects`.
`Photoelectrons` is already-produced dense truth and has neither a config,
preparer, producer, nor Runtime record; it owns only its field and runtime deep
validator. There is no global `configs/`, `fields.py`, `builders.py`, generic
`validation.py`, Runtime/Action base, registry, or product framework.

## TensorCore Backbone

Stage 2 used TensorCore `0.6` and Stage 3 selected exact TensorCore `0.7.0`
commit `b454d738f6385ce6489d85492a618a3dab139bb6`; both remain historical
evidence only. Maintenance 2 selected exact TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. Closed Maintenance 5 adopts
published TensorCore `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b`. Every selection and fixed
TensorDSLab consumer probe is exact-baseline package evidence, not a broad
compatibility claim.

Import TensorCore public names from the root `tensor_core` package. Do not
import implementation modules, fork generic behavior, or re-export generic
TensorCore helpers through `tensor_dslab.common`.

Maintenance 5 uses TensorCore's compact semantic-axis roots plus its existing
field and collection roots:

```text
TensorAxis[CoordinateT]
  CountAxis(*, count: int)
  RegularAxis(*, start: int, step: int, count: int)
  LabelAxis(*, labels: tuple[str, ...])
TensorField(tensor: torch.Tensor, axes: tuple[TensorAxis, ...])
TensorCollection(fields: Iterable[TensorField])
```

TensorCore also owns constrained scalar records, universal representation
validation, exact-type lookup, generic relationship helpers, table roots, and
the generic `TensorArtifact` extension point. It does not own TensorDSLab
domain products, scientific configuration, artifact format, or IO policy, and
it has no `0.6` ID/layout/metadata model, generic selection/movement API,
output-buffer/workspace API, or lifecycle service. Maintenance 5 adopts no
table or artifact surface.

Maintenance 6 uses TensorCore's public `Scalar.require(...)` contract exactly
once per canonicalized physical Config field. TensorCore remains independent
of Pint and owns no registry, unit, quantity, canonical physical dimension, or
TensorDSLab Config policy. Do not wrap canonical magnitudes in Scalar objects;
store fresh Pint quantities at the public Config boundary and plain built-in
numbers in Runtime records.

The exact TensorCore `0.9.0` dependency selected for Maintenance 2 exposed
public `RngKey`, `CounterRng`, `Threefry4x32`, `logical_positions`, and
`require_same_dtype` at
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. TensorCore owns generic
key/seed/address validation, Threefry word continuity, fixed-point uniforms,
parameterized Gaussian draws, Poisson inversion/PTRS, binomial
inversion/BTRS, sampler numerical domains and exhaustion, and the count
distributions' internal word schedules. TensorDSLab owns config placement of
stochastic roles, scientific position/category lattices, direct
uniform/Gaussian ordinals, draw-free scientific policy, multinomial
orchestration, final remainders, count accumulation, and ledgers. Use only
public TensorCore package-root RNG surfaces; do not copy or import protected
raw-word or promoted distribution mechanics. Published TensorCore `0.13.0`
preserves those public RNG contracts while adding compact axes, table roots,
`TensorArtifact`, `Scalar`, and the golden-path runtime simplification. Closed
Maintenance 2 evidence remains scoped to exact `0.9.0`, and closed Stage 3
through 6 evidence remains scoped to `0.7.0`. TensorDSLab uses
`require_same_dtype` only for semantic-field relationships and retains raw
tensor requirements plus its private scalar-to-dtype representation helper.

Each TensorDSLab semantic leaf has exactly one matching root in `__bases__`,
with no mixin or other base, is `@final`, declares `__slots__ = ()`, adds no
stored fields, does not reapply
`@dataclass`, inherits the root constructor and behavior, and implements only
its domain `_require()` narrowing. `ExampleAxis`, `ChannelAxis`, and
`SampleAxis` inherit `CountAxis`, `LabelAxis`, and `RegularAxis` respectively;
the six readout products are `TensorField` leaves; and `ReadoutCollection` is
the `TensorCollection` leaf. Do not invent an intermediate `ReadoutField`,
generic `Product`, or wrapper hierarchy.

Because these are ordinary ABC extension points, static analysis, focused
tests, and Review enforce our leaf declarations. Runtime code validates
documented public inputs and correctness-critical supported relationships. It
does not exhaustively defend against callers who subclass final leaves, mutate
classes, bypass inherited construction, call private functions directly,
mutate exposed tensors, or install custom Torch dispatch behavior. Such use is
unsupported and has no stable error contract.

`ReadoutCollection` is a completed immutable result containing any nonempty
unordered subset of `Photoelectrons`, `Charge`, `PureWaveform`,
`NoiseWaveform`, `AnalogWaveform`, and `DigitizedWaveform`. It requires equal
ordered axes, one device, and one common dtype across floating products. It is
not a partial pipeline snapshot and has no add/replace/invalidation lifecycle.
The accepted schema lives once on the collection class rather than in IDs,
constants, canonical-order maps, or descendant registries.

TensorCore does not guarantee universal freshness or sharing. The owning
TensorDSLab operation classifies every field result as exact return, guaranteed
storage sharing, sharing permitted but unspecified, or guaranteed fresh
storage independent of named inputs, and separately owns subtype, dtype,
device, axes, strides, autograd, synchronization, failure effects, and
output-to-output relationships. The MVP returns requested `Photoelectrons`
exactly and makes every generated product fresh and pairwise independent.

Constructing a field is semantic exposure. Producers initiate or enqueue all
writes before constructing that field and never write through an alias
afterward. The initial public API has no destination collection, `out=`,
workspace, or allocation-free claim. Any later reusable destination remains
raw, exclusive, and unexposed until its writes are enqueued and the semantic
field is constructed once.

### Coordinates, Indices, And Axes

A coordinate is the value represented at one zero-based tensor index.
Coordinates and indices are not interchangeable. Exact axis class supplies
semantic scope, and the ordered axis tuple on a field is the complete
tensor-dimension order. Code locates a dimension by exact axis type rather
than by a loose axis-name constant.

`ExampleAxis(CountAxis)` holds identity-free local zero-based ordinals.
`ChannelAxis(LabelAxis)` holds unique nonempty detector-label strings.
`SampleAxis(RegularAxis)` holds compact integer-picosecond left edges through
`start`, `step`, and `count`; the exclusive stop is not itself a coordinate.
The complete Photoelectrons/readout boundary requires `start == 0`, while the
semantic axis may represent a valid nonzero-start regular subgrid. Private
sampling preparation derives numeric execution facts once from this exact
source axis. There is no duplicate `SamplingConfig` or timestamp parsing.
Maintenance 6 adds package-owned Pint conveniences at Config and SampleAxis
boundaries only; axes remain canonical integers and Runtime/tensor execution
remains unit-free.

Every readout field has exactly one example, channel, and sample axis, in any
order. The tuple order is tensor dimension order and therefore also defines
the MVP positional RNG schema. Reordering a tensor is not promised to preserve
random draws. The future source bridge should normally construct
example/channel/sample order so temporal kernels receive sample-last input.

Do not persist transient tensor indices as durable identity. Exact Python axis
and field classes are in-process identities only; durable labels and coordinate
provenance require a separately accepted artifact contract.

## Product Semantics

TensorDSLab should preserve the data-flow and ownership chain unless Design
accepts a focused change:

```text
G4DS native products
  -> TensorG4DS typed tensor-native products
  -> deferred TensorG4DS-to-TensorDSLab bridge
       -> explicit provenance and coordinate mapping
       -> detector-window/readout-grid construction
       -> photon-origin PE binning
  -> dense TensorDSLab Photoelectrons truth field
  -> simulate_readout(...)
  -> request-selected ReadoutCollection
  -> deferred reconstruction and TensorML boundaries
```

This sequence is a product-flow and ownership rule, not a Python import graph
or scheduling policy. Projects/dag may
fan out, retry, cache, stream, compact, or parallelize work, but local readout
builders should not hide the upstream semantic bridge or downstream model
adaptation. As package dependencies, core `tensor_dslab.common` and
`tensor_dslab.readout` remain TensorCore-only. A future downstream-owned bridge
may import an exact accepted public TensorG4DS type; TensorG4DS must not import
TensorDSLab.

The primary readout tensor boundary is `ReadoutCollection`, not a loose product
tuple or required dataclass adapter. Exact final Python classes provide
in-process axis and product identity. Source event IDs remain upstream
provenance; the future bridge maps the selected batch to local ordinal
`ExampleAxis` positions and maps detector channels to string labels on
`ChannelAxis`. Durable event identity is a separate deferred artifact or bridge
contract and must not be inferred from local example ordinals or class names.

`Photoelectrons` is the already-produced dense, binned, photon-origin truth
input. It never includes dark counts, timing jitter, correlated avalanches, or
smearing. Requesting it returns the exact source field. Charge production uses
private working tensors and never mutates or relabels the truth field.

The product graph is:

```text
Photoelectrons -> Charge -> PureWaveform
Photoelectrons axes/device/shape + source SampleAxis -> NoiseWaveform
PureWaveform + NoiseWaveform -> AnalogWaveform -> DigitizedWaveform
```

The implemented Stage 7 `simulate_readout(...)` contract
computes the requested transitive closure privately, executes each producer at
most once, and retains exactly the caller-requested products. Prerequisites
need not be returned. `products` is in-memory retention only; persistence and
IO remain deferred.

Consumer-facing adapters are deferred. TensorDSLab should first make the local
typed product graph coherent enough that future consumers can depend on it
without parsing raw `.fil`, table, array, manifest, or private representation
details.

## Domain Organization

Domain code should communicate through typed in-memory objects. Persistence,
caches, artifacts, tables, arrays, tensors, executables, operations, and
recipes are bridges around the domain model, not replacements for it.

The readout domain is organized around products rather than generic layers.
Each product package owns `field.py` for its final `TensorField` leaf and
`config.py` for its configs where applicable. Each generated product owns one
non-exported `runtime/` package with `prepare.py`, `produce.py`, and
`validate.py`; those modules own the exact Config-to-Runtime,
Runtime-to-Product, and Product-to-validation actions. Charge's focused
scientific submodels, multinomial/category orchestration, and count bookkeeping
live under non-exported `charge/runtime/effects`. Do not create global
`configs/`, `fields.py`, `builders.py`, `validation.py`, or `tensors.py`
dumping grounds, or a generic Runtime/Action framework.

`readout/config.py` contains exactly `ReadoutConfig`, and
`readout/collection.py` contains exactly `ReadoutCollection`. Implemented
Stage 7 `readout/simulation.py` continues to own public
`simulate_readout(...)`; in the merged Maintenance 4 implementation it is a
thin owner of the public signature, topological produce/validate sequence,
exact retention, and final collection construction.
`readout/runtime/prepare.py` owns request parsing, dependency and config
closure, key uniqueness, and composition of product Runtime values.
`readout/runtime/sampling.py` is a small private source-derived dependency leaf
shared by request and product preparation. `readout/requirements.py` and every
product runtime/effect module are non-exported implementation modules. The
current implementation has no `readout/_random.py` or replacement `_rng.py`;
generic RNG mechanics come from TensorCore. Under Maintenance 5,
`common/axes.py` is the sole common sampling-coordinate owner and
`common/sampling.py` is removed without a shim.

Keep import direction acyclic: TensorCore, common, shared readout requirements,
product configs/fields, the private source-bound sampling runtime, product
runtime actions, readout config/collection and whole-request preparation,
readout simulation, then deliberate root exports. Product runtime modules do not
import `ReadoutConfig`, `ReadoutRuntime`, `ReadoutCollection`, simulation, or
`simulate_readout(...)`; produce modules import neither Configs nor validators.

Physical module location does not decide public visibility. Package facades
and `__all__` deliberately export the collaborator-facing axes, fields,
configs, collection, and `simulate_readout(...)`. Runtime paths are ordinary
importable Python modules but are unsupported and carry no compatibility
promise; runtime `__init__.py` files export nothing. Cross-module runtime names
therefore omit leading underscores, while TensorCore's semantic-leaf
`_require()` hook and genuinely module-local helpers retain them. Never create
placeholder modules merely to reserve a future tree.

## Target Domain Simulation Surface

Closed Stage 7 exposes one public readout action:

```python
readout = simulate_readout(
    photoelectrons,
    products=[AnalogWaveform, DigitizedWaveform],
    config=config,
    rng=Threefry4x32(seed=1234),
)
```

The `products` iterable is required, consumed once, semantically unordered,
and contains exact recognized classes. Reject empty requests, duplicates,
base/foreign types, and missing transitive configs before any RNG request,
product-producer invocation, or semantic-output write. Product-owned private
Runtime records must complete the entire closure before the first RNG request,
production call, or semantic-output write; `readout.runtime.prepare` composes
those values without duplicating scientific equations. One shared
`SamplingRuntime` binds the source sample dimension and sampling facts once.
Execute each prerequisite at most once and retain exactly the requested
fields. Each exact product config enters its typed preparer; the producer
receives the trusted ProductRuntime rather than the config again.

`ReadoutConfig` composes only optional product-specific configs. Sampling is
source structure, not scientific configuration, so `ReadoutConfig()` is the
valid truth-only configuration. Config absence is structural. Product configs
use exact model types rather than string switches. There is no generic
`Config` ABC merely for uniformity and no `PhotoelectronsConfig`.

The accepted runtime action lifecycle is explicit:

```text
Config plus execution facts -> prepare_<product> -> ProductRuntime
prerequisites plus ProductRuntime (+ RNG) -> produce_<product> -> Product
Product plus minimal prepared facts -> validate_<product> -> None
```

Preparation interprets already-validated config, extracts canonical physical
magnitudes, derives scientific operands, proves contextual bounds, and
materializes required device constants. Under Maintenance 6, unit recognition,
conversion, canonical copying, and scalar-domain validation happen once in the
public Config constructor; preparation never carries Pint into Runtime.
Production performs
tensor execution, accepted RNG requests, narrow dynamic guards, and exactly
one semantic-field construction; it imports no Config or validator. Validation
performs the completed product's deep publication scan without repair,
construction, movement, or config interpretation. `simulate_readout(...)`
executes `produce -> validate -> descendant` for every generated product and
passes the exact result plus named direct prerequisites to its validator.
Photoelectrons deep validation remains part of whole-request preflight.

Charge's `simulate_*` effect actions privately apply enabled dark counts,
timing jitter, fixed-generation correlated avalanches, and smearing without
mutating truth. Pure/noise produce zero-referenced components; analog composes
them; digitization applies its ADC transfer. Runtime paths are not public APIs
and may trust preconditions established by the public boundary.

Stage 6's `_produce.py` / `_produce_*` convention remains exact historical
evidence. The merged Maintenance 4 implementation removed those private paths
without an alias, compatibility shim, or restoration of the earlier
`_product.py` / `_product_*` names.

The builder does not load sources, perform IO, move or normalize existing
inputs, persist products, or own DAG scheduling. A producer's declared fresh
generated-product dtype conversion is not input normalization. The builder
requires one immutable TensorCore
`CounterRng`, even for a deterministic closure; there is no simultaneous
`seed=` or ambient mutable generator. Deterministic private producers receive
no RNG. Stochastic-capable Charge and noise producers receive it and select
the exact `RngKey` owned by their leaf config. TensorCore exposes no
non-consuming algorithm-capability query: Stage 7 validates nominal
`CounterRng` membership, performs no dummy draw, and treats a real custom RNG
backend failure at the first genuine distribution request as an execution
failure.

Default keys use namespace `0x54445331` and append-only streams:

```text
WhiteNoiseConfig.rng_key                     1
PsdNoiseConfig.rng_key                       2
DarkCountConfig.rng_key                      3
DirectCrosstalkConfig.retained_rng_key       4
DirectCrosstalkConfig.overflow_rng_key       5
DelayedCrosstalkConfig.retained_rng_key      6
DelayedCrosstalkConfig.overflow_rng_key      7
TimingJitterConfig.rng_key                   8
AfterpulseConfig.rng_key                     9
ChargeSmearingConfig.rng_key                10
```

Keys are exact immutable config fields and may be deliberately overridden.
Do not use loose constants, `IntEnum`, `auto()`, hashes, declaration order,
requested-product order, or mutable/global generators. Stage 7 rejects one key
assigned to different stochastic roles in the requested closure before any
RNG request, producer invocation, or semantic-output write.

Stage 5/6 implemented the same default addresses through private `_RngStream`
and `readout/_random.py`; those bytes remain historical evidence. The
closed Maintenance 2 implementation preserves their default-key outputs
against selected TensorCore `0.9.0` and removes the old module and enum without
shims.

Charge timing/AP redistribution uses TensorDSLab-owned aggregate multinomial
factorization through calls to TensorCore's public `rng.binomial(...)`.
TensorDSLab prepares stable current-category and later-category masses, fixes
category order, and assigns the final no-draw remainder. TensorCore owns exact
no-draw degeneracies, strict reflection, the frozen one-uniform forward-CDF
inversion below `n * p_star = 10`, and the frozen cancellation-resistant BTRS
mapping at and above that crossover. BTRD and the cancellation-prone three-log
grouping are not accepted v1 mappings. Aggregate counts are supported through
the per-cell ceiling `2**53 - 1`; the stabilized BTRS log bound owns a central
`1e-6` and complete-support mixed absolute/relative high-precision oracle gate.
Multinomial preparation must never update a remaining probability by repeated
subtraction from one or recover a tiny complement as `1-p`.

Timing jitter analytically prepares the latent-uniform plus ideal-Gaussian law
through the frozen log-domain one-sided tail evaluator in `rebuild.md`. Its
initial domain is `2**-52 <= sigma / T <= 64`,
`2 <= sample_count <= 8192`, and `S * N <= 2**63`; exact zero sigma is a
separate identity. It scans every possibly in-window target in increasing
order, uses `TimingJitterConfig.rng_key` whose default stream is `8`, and leaves one final
no-draw drop remainder. The local absolute tolerance is `1e-12` and the
complete source-law L1 tolerance is `1e-11`. Negative probabilities, clipping,
residual assignment, normalization, per-PE normals, Box-Muller jitter, and an
arbitrary timing-tail cutoff are forbidden. Correctness-first quadratic
sample-count work is accepted until a later measured optimization preserves
the same law. Dark counts and the four
crosstalk roles call TensorCore's public `rng.poisson(...)`: exact-zero no-draw,
one-uniform CDF inversion below mean `10`, and Hoermann PTRS from `10` through
`1e8`. Discrete probabilities, rate fields, and sampler control use binary64
independently of the requested `Charge` dtype. TensorCore returns integer
counts; TensorDSLab's physical charge ledgers retain the requested product
dtype. Never substitute
per-avalanche expansion, `torch.poisson`, a normal approximation, global RNG,
reseed-on-exhaustion, clipping, or a fallback algorithm. The exact caps,
addressing, repeatability boundary, and validation oracles live in
`docs/architecture/rebuild.md`.

The active Charge numeric envelope is relational. Every source, working,
frontier, mechanism, overflow, and cumulative count cell is no greater than
`2**53 - 1`; nonnegative additions are checked before execution and there is no
whole-grid population cap. Jitter, CT, and AP addresses respectively prove
`S*N <= 2**63`, `K*N <= 2**63`, and `K*(S+1)*N <= 2**63`. The requested-dtype
ledger depth must satisfy the frozen `L < 2**p_d` relation, and enabled
smearing proves finiteness against the maximum dtype-specific Box-Muller
radius. Do not replace these with wrapping arithmetic, an arbitrary `K`, a
silent clamp, or a guessed memory ceiling.

The MVP accepts exactly `FixedDelayConfig | ExponentialDelayConfig` for each
crosstalk mode. `NormalDelayConfig` is retired despite its historical Stage 3
implementation; Stage 6 removed its class, union memberships, exports, and
tests without a compatibility shim. Do not restore an unsupported public
config. A later calibrated delay family requires a new explicit type and
Design decision.

Fixed and exponential phase-marginalized delay preparation is closed Design.
The fixed law accepts every finite nonnegative delay, uses an exact two-point
mapping, and has no PMF tolerance. Exponential delay and recovery use the
bounded ratio/sample domains, binary64 evaluation branches, analytic right
tails, and `1e-12` local / `1e-11` complete-law tolerances normative in
`docs/architecture/rebuild.md`. Implementations must not replace either law
with latent per-edge draws, clipping, a cutoff, residual assignment,
renormalization, or subtraction-derived tiny tails.

White RMS and PSD cells are prepared in Python binary64, PSD overlaps use
`math.fsum`, and executed values round once into the selected output dtype.
White RMS must remain in the selected dtype's positive normal range. The
represented values define ideal-normal target moments; the finite Box-Muller
lattice is not renormalized, and conservative host bounds reject subnormal
white scales and overflow-prone scales. Noise results are fresh,
source-payload-independent, and
`requires_grad=False`. The accepted reference path is eager CPU with
conditional eager CUDA. Raw words and fixed-point uniforms are exact across
accepted implementations; completed normal/PSD products compare exactly only
within one unchanged numerical execution stack and statistically across
backends. For completed transcendental values, unchanged includes the
OS/architecture, Python and PyTorch build, backend/device implementation,
execution mode, dtype, and relevant math settings—not merely the word
`CPU` or `CUDA`.

Literal floating-point continuity fixtures must name the exact numerical stack
that owns them. On that stack, retain exact payload assertions. On another
accepted stack, prove exact replay within that unchanged stack plus the
accepted structural, invariant, analytic, and statistical contracts. Do not
silently substitute a post-observation ULP tolerance, a new platform-specific
golden table, a skip, or an expected failure. Raw-word and fixed-point-uniform
claims retain their separately documented exact scope. Do not turn private
helper signatures into public or compatibility contracts.

Every field-returning path must adopt TensorCore's operation-owned result
taxonomy: exact return, guaranteed storage-sharing, sharing permitted but
unspecified, or guaranteed fresh storage independent of named inputs. It must
also specify concrete type, dtype, device, axes, strides/layout, autograd,
synchronization, failure effects, and output-to-output relationships.

The MVP borrows source `Photoelectrons` read-only and returns it exactly when
requested. Every generated product has guaranteed-fresh storage independent of
named inputs, and generated fields retained together are pairwise
storage-independent. Dimension-preserving products reuse the exact source axis
tuple and exact immutable axis instances. No operation silently detaches,
moves, casts, or host-materializes an existing field.

Callers must not mutate tensors exposed through fields or collections. A
producer initiates or enqueues all writes before it constructs and exposes a
field, and TensorDSLab initiates no later write through any alias to that
storage. Field construction alone is not an additional synchronization point,
but accepted deep-value validation and producer postconditions use scalar
reductions that may synchronize CUDA. Outside those documented correctness
checks, same-stream consumers use ordinary stream ordering and cross-stream
consumers establish their own dependency.

The initial API has no public `out=`, destination collection, workspace,
allocator, lease, or allocation-free promise. Private scratch remains
exclusive and is never exposed as a semantic value that will later be
overwritten. Any future reuse design must keep writable storage raw and
unexposed until all writes are enqueued, then construct the completed semantic
field exactly once.

Sample-last is an upstream recommendation for readout performance, not a
universal semantic order. Future Reconstruction and TensorML adapters own their
explicit product selection, axis reordering/materialization, positional schema,
and result contracts; TensorCore `0.13` supplies no implicit generic selection
or movement layer.

## Common Code

`common/` should stay dependency-light and semantic. Maintenance 5 uses it for
`ExampleAxis`, `ChannelAxis`, and `SampleAxis` because source construction,
readout, and future Reconstruction may share those coordinate contracts.
Sampling execution facts are derived privately from the source axis rather
than represented by another public common config. Good later candidates
include small value objects and validation primitives used by multiple real
domains.

Do not put representation dependencies in top-level `common/` merely because
multiple domains use tables, arrays, tensors, Parquet, NPZ, JSON, or another
format. Prefer domain-local `tables.py`, `arrays.py`, or `tensors.py`, or a
cache/artifact-local helper, until a shared semantic abstraction proves itself.

Avoid expanding `common/` because two modules happen to look similar. Wait
until the concept is actually shared.

Use small semantic quantity wrappers for stable public records when a scalar
field needs finite signed, strictly positive, or nonnegative numeric
semantics. Place wrappers domain-local when only one domain needs them; promote
them to `common/` only when multiple real domains share the same quantity
vocabulary or a common public record intentionally uses them.

## Public Surface Discipline

Package `__init__.py` files should re-export documented public surfaces
deliberately. They should not expose private representation, persistence,
normalization, or validation-helper functions by accident.

Privacy for readout runtime actions is defined by these deliberate facades,
not by making Python submodules inaccessible. Runtime packages import and
export nothing from their `__init__.py` files; direct deep imports remain
unsupported implementation use and receive no compatibility promise.

Public names should be stable and intention-revealing. When public names move
or change:

- update the relevant docs in the same stage;
- run targeted stale-name searches;
- keep historical mentions only when clearly framed as historical,
  superseded, or deferred;
- do not add compatibility wrappers or aliases unless Design accepts a
  compatibility window.

Downstream code should consume typed upstream objects. It should not parse
another domain's raw persisted files directly. If a cache or artifact is the
bridge, the owning domain must provide the loader that reconstructs typed
objects.

### Public Verb Vocabulary

Use consistent verbs for public and semipublic module-level APIs:

- `build_*` constructs an in-memory domain object or representation from
  already-available inputs. It does not perform filesystem IO or durable side
  effects.
- `read_*` parses or decodes a durable or boundary representation into a typed
  representation record or bridge record.
- `write_*` persists a representation to a durable output boundary.
- `load_*` crosses from durable or boundary storage into the typed domain
  object that downstream code should consume.
- `validate_*` reports contract violations without repairing inputs.
- `compact_*` is reserved for strict storage-level reduction over complete
  compatible durable products.
- `render_*` or `build_*_tensor*` may be used for explicit TensorCore-backed
  tensor rendering when a stage accepts that boundary.
- `assemble_*` packages already-built typed products into one coherent
  in-memory example or container. It does not load, write, or invoke DAG
  behavior.
- `compute_*` derives a result from already-available inputs without durable
  side effects.

Prefer the verb based on the return value and boundary crossed:

- If a function reads storage and returns the domain product, use `load_*`.
- If a function reads storage or in-memory boundary data and returns a
  representation or bridge record, use `read_*`.
- If a function creates an in-memory JSON, table, array, tensor, or other
  representation-shaped value, use `build_*`, not `write_*`.
- If a function writes files, cache entries, artifact files, or other durable
  outputs, use `write_*`.

`from_*` and `to_*` are acceptable for methods or very local/private
conversions. Public module-level bridge functions should prefer
`build/read/write/load`.

Reserve `parse_*` for textual grammars or user input. Avoid `serialize_*` and
`deserialize_*` unless the project is explicitly implementing a serialization
layer. Do not use `get_*` for functions that hide IO, construction,
validation, or expensive computation.

`validate_*` functions must not repair, fill, cast, write, or conceal missing
upstream work. `assemble_*` functions must not call loaders, writers, cache
APIs, or DAG APIs.

`simulate_readout(...)` is the public scientific
orchestration function. It consumes already-produced dense `Photoelectrons`,
plans the requested product closure, and returns one completed
`ReadoutCollection`. It must not load data, perform durable IO, move the source,
normalize an existing input through a cast, persist products, or invoke DAG
APIs. A producer's declared fresh generated-product dtype conversion remains
required.

## Deferred Integration Surfaces

Projects/dag owns campaign orchestration, sharding, scheduling, concrete DAG
construction, execution policy, repair, retries, status, and fanout/fanin.
TensorDSLab may later expose stable public surfaces for operation specs,
executable adapters, artifact/cache requirements, output validation, and
recipe fragments.

Stage 7 local product dependency planning inside `simulate_readout(...)` is
TensorDSLab scientific orchestration, not Projects/dag campaign orchestration.

For compaction, TensorDSLab owns a strict deterministic storage-level operation
over complete compatible caller-supplied products. Projects/dag owns discovering
or scheduling shards, campaign fan-in, retries, repair, and cross-shard
execution policy.

Use these optional package directories only when the project needs them:

- `operations/` for DAG-compatible operation specs;
- `recipes/` for reusable composition fragments;
- `executables/` for CLI, DAG, or task adapters.

Do not add DAG-compatible modules, downstream adapters, package-local workflow
CLIs, or cache-driven integration surfaces before local TensorDSLab contracts
are accepted. Keep product producers dependency-light and campaign-
orchestration-free; local composition belongs only in
`simulate_readout(...)`.

## Parts-Bin Rule

Historical predecessor code is donor material only. Reuse scientific facts,
small algorithms, naming lessons, fixtures, tests, and accepted cache semantics
after review. Do not preserve old package layouts, helper frameworks, local DAG
mechanics, compatibility wrappers, or representation shortcuts by default.

When promoting donor code or behavior:

- write down the accepted reason in the relevant implementation or decision
  doc;
- name the donor snapshot and source symbol, comparison boundary, parity
  classification, assumptions, observables, acceptance criteria, exclusions,
  and intentional divergences in `docs/parity.md`;
- adapt names to TensorDSLab's tensor-native design;
- remove compatibility baggage unless Design accepts a compatibility window;
- add tests around the promoted contract.

## Engineering Standard

Prefer boring, explicit, maintainable code over cleverness. The design should
be easy for a future contributor to reconstruct from module names, type
signatures, tests, and architecture documents.

Good changes should:

- preserve documented ownership boundaries;
- keep public APIs small, typed, and intention-revealing;
- use concrete typed records instead of unstructured dictionaries where a
  stable concept exists;
- keep implementation details private until they are real extension points;
- make behavior deterministic unless nondeterminism is explicit;
- include tests that protect behavior and invariants;
- update source-of-truth documentation when public contracts or accepted
  semantics change.

## Boundary-First Validation

TensorDSLab should move toward validated-once, trusted-downstream records.

Validate strongly when data enters or re-enters the TensorDSLab/TensorCore
typed path:

- the future TensorG4DS-to-TensorDSLab semantic bridge and its provenance,
  coordinate, unit, dtype, axes, and device contract;
- user configs;
- normalization through TensorCore constrained-scalar requirements for
  meaningful numeric config or artifact values;
- construction of detector, readout, reconstruction, cache, table, array, and
  tensor records;
- construction of TensorCore axes, fields, and collections;
- explicit product-specific deep validation for untrusted field values; and
- `simulate_readout(...)` whole-request preparation before any RNG request,
  product-producer invocation, or semantic-output write.

Once an object has crossed into a valid native record, hot-path functions
should avoid repeatedly revalidating full object graphs. Product builders may
still perform narrow function-specific checks, but should not repeat full
device scans or materialize/parse compact semantic coordinates inside kernels.
Product preparation
owns statically derivable scientific and relationship checks; production owns
only tensor/RNG execution and correctness-critical dynamic guards; explicit
product validation owns the completed-result deep scan before any descendant
or collection publication consumes it.

Use TensorCore scalar requirements at config, source, and artifact boundaries
where numeric constraints are meaningful. Store a Scalar wrapper only when
that wrapper is itself the accepted public representation; Maintenance 6
instead stores canonical Pint quantities and plain Runtime magnitudes. Numeric
requirements should reject bool. Plain `bool` is appropriate for boolean
fields.

Prefer these migration directions as real code is introduced:

- repeated ad hoc numerical checks become constrained scalar records;
- repeated broad validation in hot paths moves to constructor invariants and
  boundary validators;
- recursive inner-loop object checks become trusted typed inputs plus narrow
  operation preconditions; and
- semantic leaf construction remains cheap while product-owned runtime
  validators own value-domain scans and `simulate_readout(...)` invokes each
  validator immediately after production.

For the accepted Maintenance 6 boundary, whole-request
`prepare_readout(...)` owns public ingress, request closure, dtype/device, RNG
capability, and key admission. Config construction owns Pint
canonicalization, unwrapped primitive domains, and genuine local
relationships, but does not repeat annotated wrapper, key, nested-Config,
optional, or union membership. Private child preparers do not repeat those
exact-type admission checks; they own contextual extraction, exact model
dispatch, scientific/representability checks, and Runtime construction.
Private effect executors may trust exact Runtime and primitive types reachable
only through that typed path, but must retain count, envelope, address,
allocation, and scientific-law guards. Generic TensorCore relationship
helpers should be used where their contract matches exactly, not where
TensorDSLab requires arbitrary axis order, exact source-axis object reuse, an
absolute dtype domain, or storage freshness.

Do not add runtime policing for callers who deliberately leave the public API.
Unsupported final-leaf subclassing, class mutation, constructor bypass,
private-function calls, exposed-tensor mutation, and custom dispatch behavior
do not require exhaustive guards or stable error categories.

## Scope Discipline

Implement only the accepted documentation stage or production work order. Do
not broaden package ownership, public APIs, cache shape, TensorCore contracts,
DAG semantics, or downstream integration implicitly.

If work reveals a contradiction in the accepted design, stop and route it to
Design. Do not patch around it by inventing architecture inside implementation.

## Code Expectations

- Use a short module context docstring when ownership or boundary is not
  obvious from the module path and public types. Do not add filler docstrings
  to tiny cohesive modules.
- Type public functions, methods, dataclass fields, and any real module
  constants.
- Avoid `Any`, unbounded `dict`, and stringly typed public interfaces unless
  the boundary is intentionally JSON-like or there is a documented reason.
- Prefer frozen slotted dataclasses for stable configuration records. Do not
  reapply `@dataclass` to TensorCore semantic leaves.
- Use concrete final frozen slotted product Runtime dataclasses for prepared private
  execution operands. They must not inherit a Runtime base, retain Configs or
  semantic products, or expose execution methods or mutable caches.
- Use `value` for primitive payloads on constrained scalar records.
- Define semantic axes, fields, and collections as direct final fieldless
  TensorCore leaves with inherited construction. A leaf adds no stored state
  or overridden root mechanics; it may expose only the explicitly accepted
  nonstored conveniences owned by its domain contract.
- `Enum`, `Literal`, `Protocol`, and generics remain useful for non-hot-path
  contracts when they make a public boundary clearer.
- Keep modules cohesive; split a module when it owns more than one meaningful
  boundary.
- Keep product preparation, production, and deep validation in their exact
  owning runtime modules; extract only genuinely identical behavior into the
  narrowest private owner rather than a generic action framework or broad
  helper module.
- Keep comments sparse and useful. Explain non-obvious intent, not mechanics.
- Do not hide cross-domain, TensorCore, cache, or adapter behavior behind broad
  helper modules.

## Test Expectations

Tests should prove intended behavior, not mirror implementation structure.

Good tests should:

- exercise success paths and meaningful failure modes;
- protect invariants and boundary conditions;
- cover serialization and round-trip behavior where persistence is involved;
- prove deterministic ordering where order is part of the contract;
- prove validation rejects malformed artifacts, caches, or domain objects;
- assert public exports and retired-name absence when a stage performs a clean
  public API transplant;
- include import-isolation or dependency-scan smoke tests when a stage extracts
  or layers packages;
- prove every semantic leaf has exactly one matching TensorCore root in
  `__bases__`, with no mixin or other base, is final and fieldless, preserves
  inherited construction, and implements the
  accepted `_require()` narrowing;
- for Maintenance 5, prove exact `ExampleAxis(CountAxis)`,
  `ChannelAxis(LabelAxis)`, and `SampleAxis(RegularAxis)` bases, inherited
  constructors, range-backed nonmaterializing example/sample coordinates,
  exact channel labels, arbitrary semantic axis order, shape agreement,
  source-derived sampling, complete-input zero start, `SamplingConfig` and
  `common.sampling` absence, and retirement of the off-path
  `field(TensorField)` exception assertion;
- for Maintenance 6, prove exact Pint `0.25.3` identity, one private registry,
  two deliberate quantity-construction exports, canonical defensive copies,
  one TensorCore `Scalar.require(...)` normalization per physical field,
  complete physical/dimensionless Config coverage, and explicit unhashability
  of every public Config;
- for Maintenance 6, prove `SampleAxis` remains integer-backed while its
  constructor/access conveniences return fresh quantities, Runtime records
  recursively contain no Config/Pint object, each active physical operand is
  extracted exactly once, and producers and validators perform no unit work;
- for Maintenance 6, kill mutations that restore duplicate private admission
  guards or Config-bearing numerical helpers, while preserving scientific,
  relationship, axes-identity, storage, dtype/device, and generated-product
  postcondition checks;
- prove intrinsic leaf dtype/structure checks separately from explicit deep
  scientific value validation;
- prove `ReadoutCollection` accepts exactly nonempty unordered recognized
  subsets with equal ordered axes, one device, and one floating dtype;
- prove product-request one-pass consumption, duplicate/unrecognized
  rejection, transitive config preflight, order irrelevance, execute-once
  prerequisites, and exact requested retention;
- for Maintenance 2, prove exact TensorCore public RNG imports, the ten
  config-owned default keys and overrides, config equality/`repr`, deterministic
  producers omitting RNG, stochastic-capable producers using `CounterRng`,
  public `uniform`, `gaussian`, `poisson`, and `binomial` use, default-key
  product continuity, Charge-local multinomial/count-bookkeeping ownership,
  semantic-only `require_same_dtype` use, the private scalar representation
  helper, and absence of `_RngStream`, `readout/_random.py`, and replacement
  `_rng.py`;
- for Stage 7, prove required `CounterRng` even on deterministic closures,
  draw-free deterministic execution, and closure-wide duplicate-role-key
  rejection before RNG requests, producer invocation, or semantic-output
  writes;
- prove source `Photoelectrons` exact return and immutability, guaranteed-fresh
  generated fields, pairwise generated-output independence, exact source-axis
  reuse, and no post-exposure TensorDSLab writes;
- retain closed Stage 7 evidence that every generated product passes its exact
  product-specific deep postcondition before downstream use or retention and
  that a failed postcondition exposes no field or partial collection;
- for Maintenance 4, prove every requested product Runtime is prepared before
  the first RNG request or production call, all temporal runtimes share the
  exact one `SamplingRuntime` where stored, and optional Runtime presence is
  the closure execution signal rather than duplicated `need_*` flags;
- for Maintenance 4, prove every generated product follows exact
  `prepare -> produce -> validate` ownership, Photoelectrons has only its
  ingress validator, every validator receives the exact produced field and
  named direct prerequisites once, and a failed validation prevents every
  descendant and final collection;
- for Maintenance 4, prove producers import no Config or validator, Runtime
  records retain no Config, semantic product, collection, stored callable,
  mutable cache, or framework state, and no Runtime/action/helper enters a
  public facade export;
- for Maintenance 4, prove the retired `_requirements.py`, product
  `_produce.py`, and `charge/effects/` live paths are absent without aliases,
  while every runtime/effect `__init__.py` imports and exports nothing;
- preserve exact same-stack product values, TensorCore RNG distribution calls,
  keys, positions, quanta, ordinals, counts and ordering, no-draw behavior,
  storage, axes, dtype/device, failure, and autograd contracts across the
  Maintenance 4 ownership refactor;
- prove deterministic waveform operations preserve accepted autograd behavior
  and stochastic/digitized paths make only their documented claims;
- prove complete request preparation precedes RNG consumption, producer
  invocation, and semantic-output writes, while launched backend failures make
  no rollback promise;
- prove no source IO, persistence, host staging, device movement,
  in-place/source replacement, input normalization, or campaign orchestration
  occurs in `simulate_readout(...)`; this does not prohibit a producer's
  declared fresh generated-product dtype conversion, including
  `Photoelectrons[torch.int64]` to floating Charge;
- do not add adversarial tests for unsupported final-leaf subclassing, class
  mutation, constructor bypass, private calls, exposed-tensor mutation, or
  custom dispatch merely to harden outside the public contract;
- use small fixtures that make behavior visible;
- avoid depending on private implementation details unless testing a private
  helper is the only focused way to cover an edge case.

When a test would be expensive, slow, or fragile, prefer a smaller invariant
test plus one representative integration test.

## Documentation Expectations

Documentation should state contracts, boundaries, non-goals, and examples that
guide implementation. It should not become a second copy of the source code.
Use the `docs/` spine for design, validation, decisions, architecture
contracts, and staged work orders.

Update docs when a change affects:

- public APIs;
- TensorCore-backed axes, fields, collections, relationships, or shapes;
- product ownership;
- cache files, manifest shape, durable guarantees, or compaction rules;
- validation rules;
- donor comparison boundaries, parity classifications, assumptions,
  tolerances, or intentional divergences;
- operation, recipe, or executable surfaces;
- implementation stages or accepted decisions.

Keep the relevant source of truth synchronized:

- `docs/implementation/...` for stage work orders, scope, public surfaces,
  invariants, and non-goals;
- `docs/architecture/<domain>.md` for public domain contracts, cache shapes,
  builders, validation boundaries, and representation bridges;
- `docs/architecture/tensors.md` for the TensorCore consumer contract,
  semantic tensor products, axis/field rules, result sharing/freshness,
  placement, synchronization, exposure, lifetimes, and cross-repository
  coordination items;
- `docs/parity.md` for donor evidence, comparison classes, assumptions,
  tolerances, fixture provenance, and intentional divergences;
- `docs/architecture/common.md` for shared primitives and cross-domain rules;
- `docs/design.md` for end-to-end domain flow and ownership boundaries;
- `docs/decisions.md` for accepted, renamed, superseded, or explicitly
  deferred semantic choices;
- `docs/validation.md` for validation cases, fixtures, failure modes, and
  tolerances;
- `README.md`, `AGENTS.md`, or `CONTRIBUTING.md` for workflow, onboarding, or
  repository-wide engineering expectations.

## Documentation-Only Design Checks

Documentation-only Design changes remain in the Design thread unless the user
requests independent Validation or Review. At minimum, run:

```bash
git diff --check
```

Also run targeted link, heading, and stale-term searches for the changed
contracts. Do not create placeholder code or tests merely to exercise a docs
stage.

When package-governance records change, also run the state,
manifest, rule-coverage, source-anchor, dormant-trigger, deviation, raw-ID,
changed-file-allowlist, and forbidden-claim checks defined in
[Validation](docs/validation.md#governance-adoption-checks). Runtime, import,
dependency, export, environment, and post-merge commands are active for the
accepted package surfaces through Stage 7. Integration and later scientific-
runtime commands remain dormant until their corresponding surfaces exist.

## Before Production Review

Before asking for Review, provide:

- branch and commit, when the repository is initialized as git;
- scope implemented;
- files changed;
- docs updated or checked;
- tests added or changed;
- commands run;
- known risks or deferred follow-ups.

At minimum, run:

```bash
git diff --check
```

For Python changes after the package exists, also run the relevant test suite:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

For future DAG-compatible changes, also run the accepted operation-spec or
domain-module validation command.

Never stage generated caches, local outputs, or unrelated files.
