# Agent Workflow

This repository uses role-separated Codex ownership. Design owns architecture,
decisions, validation expectations, and future work orders directly. Stage 2
completed the first production Implementation/Validation/Review loop on
2026-07-11. Documentation-only Design work outside a dispatched production
work order may remain in Design unless the user requests independent Validation
or Review.

The production workflow is:

```text
Design -> Implementation + Validation -> Review -> Implementation fixes -> Review recheck
```

Before production implementation starts, establish one persistent thread per
role for each active TensorDSLab workspace:

- Design
- Implementation
- Validation
- Review

Tasks and stages are passed through handoffs, not represented as new permanent
threads. Documentation-only Design work does not need to simulate the
Implementation/Validation/Review loop. Once code is in scope, the role split
keeps architecture, implementation, behavioral validation, and independent
critique separate enough that each thread can do its job without blurring
ownership.

When a stage spans multiple repositories, such as TensorDSLab, TensorG4DS,
TensorCore, TensorML, G4DS/g4ds11, or Projects/dag, keep each workspace's role
threads explicit in the handoff. A role from one repository must not silently
own implementation, validation, review, or merge work in another repository.
Any exception must be explicit in the handoff and accepted by the user and
every affected package Design authority.

Agents should also follow `CONTRIBUTING.md`, which defines repository-wide
engineering standards. Start with `docs/overview.md` for the documentation map.
Design work orders should cite the relevant `CONTRIBUTING.md` standards when a
stage touches TensorCore axes/fields/collections, TensorDSLab product semantics,
in-memory product relationships, durable cache shape, validation boundaries,
public typing, coordinates versus indices, artifacts, result storage, or future
integration boundaries.
They should also cite `docs/parity.md` when promoting donor behavior, changing
a comparison boundary, or accepting a statistical approximation or intentional
divergence.
Validation and Review should treat violations of accepted `CONTRIBUTING.md`
standards as real findings, not style-only comments.

## Governance Authority And State

TensorDSLab Design owns this package's architecture, public contracts,
ownership boundaries, accepted dependencies, documentation, work orders,
governance adoption, conformance findings, routing, and deviations. A
cross-package proposal binds TensorDSLab only after every affected package
Design authority ratifies the same immutable proposal. Coordination agreement,
Moderator synthesis, tests, work orders, and similarity among package documents
are evidence; none creates package architecture.

Package sources take precedence for TensorDSLab architecture and contracts. If
package and cross-package sources disagree, stop the affected work or routing,
identify the conflicting records, return the contradiction to every affected
Design authority, and resume only from an explicit resolution and synchronized
baseline. `AGENTS.md` governs roles, handoffs, routing, work-order gates, and
verification responsibilities. `CONTRIBUTING.md` governs engineering quality,
API design, typing, validation, testing, documentation, and code style.

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`, bound to
exact accepted candidate `d634401a853915edeb4f83df4a4943b3553deced`. The
current package states are:

```text
package_adoption_state: Adopted
conformance_finding: Not evaluated
coordination_status: Deferred
registry_storage_profile: Disabled
```

These states are independent. Package adoption does not constitute conformance,
routing activation, Coordination activation, production authority, or Stage 2
dispatch.

Design may operate alone for documentation-only work. Design, Implementation,
Validation, and Review are persistent logical roles per workspace after
activation. Production dispatch requires every execution role
named by the work order to be Active and verified. A dormant, stale, missing,
or discrepant route does not authorize dispatch; procedural routing returns to
Design.

Coordination is an optional representation role and remains Deferred.
TensorDSLab Design is its procedural fallback. Coordination may represent only
an accepted package position and may not ratify architecture, command Design,
dispatch implementation, replace D/I/V/R, edit production or tests by virtue
of the role, or own merge authority. Later activation requires a concrete
recurring cross-package need, an accepted charter and Design-return path, an
adopted routing/privacy procedure, verified route and fallback, no routing
discrepancy, and explicit Design and user authorization.

The Ecosystem Moderator is neutral and procedural. It may distribute
authorized packets, collect package positions, synthesize agreements and
objections, and maintain authorized procedural records. It may not represent
TensorDSLab, vote or break ties, ratify architecture, command Design, dispatch
package execution roles, modify this repository, broaden package ownership,
conceal objections, or infer consent from silence.

TensorDSLab Design owns package routing and discrepancy resolution. Stable
logical package, workspace, role, and work-order keys are primary. Raw platform
route identifiers are optional private attributes and must not appear in
committed package records. Profile B is disabled and not instantiated: do not
create `.agents`, an ignore rule, a committed route table, a private live-route
store, or a Moderator cache because the common core was ratified, a candidate
was prepared, or package adoption was issued. A discrepancy pauses only the
affected routing and returns to Design. Profile B requires a later focused
Design decision covering the private path, ignore policy,
permissions/operators, sharing,
replacement/history/deletion, verification, and discrepancy procedure.

TensorDSLab is in active development and pre-deployment. It makes no
deployability, release-readiness, backward-compatibility, or broad
cross-package compatibility claim. Later compatibility evidence is limited to
exact named commits, environment, device/backend, and execution mode. The
same-device and no-silent-host-materialization Design constraints are not proof
of an implemented or compatible package handoff.

## Project Mode

TensorDSLab is a clean-slate, tensor-native detector data-lab package. It
consumes accepted TensorG4DS tensor-native products and turns them into typed
readout and future reconstruction products, while using TensorCore's semantic
axis, field, collection, constrained-scalar, validation, and relationship
roots directly.

The intended chain is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

This is the intended data flow, not an import graph or a claim that every
boundary is implemented. TensorCore is the shared substrate across the three
tensor packages. TensorDSLab must not parse native G4DS files or implement
TensorG4DS low-level analysis such as deposit clustering. A later focused
integration stage may add a narrow TensorDSLab-owned adapter over an accepted
public TensorG4DS product.

TensorDSLab owns its post-TensorG4DS detector/readout semantics, readout and
future reconstruction products, and future durable cache contracts. It should
not own native G4DS ingestion, TensorG4DS deposit/cluster products or
algorithms, generic TensorCore primitives, TensorML
model/training/evaluation surfaces, checkpoint policy, metric reporting, or
campaign orchestration.

The first accepted MVP direction is the post-binned tensor-native readout
path: already-binned photon-origin primary photoelectrons, the aggregate SiPM
charge response, waveform products, analog waveform composition, and optional
digitization. Native G4DS parsing is permanently upstream of TensorDSLab.
Defer the typed TensorG4DS handoff, detector-window construction,
photoelectron binning, IO, cache compatibility, DAG compatibility, and
TensorML integration until the post-binned contract is stable.

Historical predecessor code, if consulted outside this repository, is parts-bin
material only. It may provide scientific facts, algorithms, fixtures, tests,
and cautionary examples, but it does not define current architecture by
default. Do not copy old package layouts, helper framework shape,
compatibility baggage, or DAG-facing mechanics into TensorDSLab by default.
Promote only reviewed behavior that fits the tensor-native design and is
recorded in TensorDSLab docs. Every promoted donor behavior must name its
comparison boundary and parity classification or intentional divergence in
`docs/parity.md`.

TensorML is a style and workflow reference, not a detector data-lab domain
template. Replace TensorML process semantics with TensorDSLab product and cache
semantics when adapting docs or patterns. TensorCore is the source of truth for
generic tensor vocabulary and contracts.

Stage 2 is Merged / Closed at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`, and Maintenance 1 is Merged /
Closed at `3af8ab4acf834b07e3d027fb530e5f12934999a5`. They remain historical
TensorCore `0.6` evidence.

Stage 3 is Merged / Closed through exact implementation candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde` and Review's clean fast-forward
closeout `97e17c3177ac217aeb42a077db78f4bd223d51fa`; Design's accepted final
closeout is clean `main` at
`5ff13eb3c0735abfda454a334be59faac35259c2`. It implements the TensorCore
`0.7` product/config/collection foundation described by
`docs/architecture/rebuild.md`. Fixed-commit Validation, independent Review,
and Design's post-merge audit found no unresolved issue. The evidence is
CPU-only because CUDA was unavailable, and no wheel or editable-install claim
was made because the required build tooling was absent.

Stage 4 is Merged / Closed through exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da` and Review's clean fast-forward
closeout `b3ebfcd9473537dd385195afea374bd2f426c6c0`. It implements exactly the
private pure, analog, and digitized waveform producers under the
functionality-first contract in
`docs/implementation/stage_4_deterministic_waveform_products.md`. Fixed-commit
Validation, independent Review, and Design's post-merge audit found no
unresolved issue. The evidence is CPU-only because CUDA was unavailable, and
it makes no GPU-performance, fusion, editable-install, or wheel-build claim.
Stage 5 is Merged / Closed through exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd` and Review's evidence-only
closeout `c6a506d3658b24197806b9e230480211a254a35a`. It implements the private
`tensordslab.threefry4x32-20/v1` reference RNG plus exact-zero, IID-white, and
caller-supplied PSD noise under
`docs/implementation/stage_5_readout_rng_and_stochastic_noise.md`.
Fixed-commit Validation, independent Review, and Design's post-merge audit
found no unresolved issue. The evidence is eager CPU-only because CUDA was
unavailable; it makes no GPU execution, performance, fusion, editable-install,
or wheel-build claim. Stage 6 Charge Simulation is Merged / Closed through
exact implementation candidate `fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58`
and Review's evidence-only closeout
`ea979862b05f4ef543f6971c86641df317232479` under
`docs/implementation/stage_6_charge_simulation.md`. It implements the complete
private Charge producer and aggregate count-sampler slice. Fixed-commit
Validation, independent Review, and Design's post-merge audit found no
unresolved issue. The evidence is eager CPU-only because CUDA was unavailable;
it makes no GPU execution, performance, fusion, install, or wheel-build claim.
Maintenance 2 RNG and product-module ownership migration is Merged / Closed.
Its exact implementation candidate is
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63`, and Design's documentation-only
closeout is `9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. It installs the
product-owned module split, public TensorCore RNG/distribution use, exact
config-owned role keys, and exact TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The accepted evidence is eager
CPU-only because CUDA was unavailable. Stage 7 public readout orchestration is
Merged / Closed through exact Review-cleared implementation candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177` under
`docs/implementation/stage_7_public_readout_orchestration.md`. It implements
the public `simulate_readout(...)` surface, whole-request preparation,
execute-once prerequisite planning, exact requested retention, and generated-
product postconditions. Its accepted evidence is likewise eager CPU-only
because CUDA was unavailable. Maintenance 3 Environment-Qualified Stochastic
Continuity is Merged / Closed through exact Review-cleared candidate
`dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0` and its five-document Design
closeout under
`docs/implementation/maintenance_3_environment_qualified_stochastic_continuity.md`.
It corrects only the applicability of Maintenance 2's exact macOS eager-CPU
stochastic literals: fixed-point uniforms retain their accepted
cross-implementation exact scope, recorded completed-value literals remain
exact on their recorded stack, and other accepted stacks prove exact
same-stack replay plus existing invariants and statistics. Fixed-commit
Validation and independent Review cleared the unchanged candidate on the
recorded macOS stack and in separate full-A100 Della allocations. It changes
no production, dependency, RNG, or scientific contract.

Maintenance 4 Runtime Action Ownership is **Merged / Closed** through exact
Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda` under
`docs/implementation/maintenance_4_runtime_action_ownership.md`. It replaces
product-local `_produce.py` bundles and `*Plan` records with non-exported
product `runtime/` packages, concrete `*Runtime` records, and explicit
`prepare_*`, `produce_*`, and `validate_*` actions. Fixed-commit Validation and
independent Review cleared the exact final bytes locally and in separate fresh
full-A100 source/archive allocations. It changes no public facade, scientific
equation, stochastic address, result law, dependency, or supported device
boundary, and it makes no performance or Stage 8 claim.

Maintenance 5 TensorCore 0.13 Compact Axes And Sampling is **Merged / Closed**
through exact Review-cleared supplemental candidate
`81ad2f52fe4a1966e5b3a0ceb5063138e42e731f` and Design closeout
`021694b9479d02546405f6a815aedf21c9c831a4` under
`docs/implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md`.
It adopts exact published TensorCore `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b`, compact
`CountAxis`/`LabelAxis`/`RegularAxis` representation roots with final
TensorDSLab semantic leaves, and sampling derived once from the source
`SampleAxis`. It cleanly removes `SamplingConfig` without introducing Pint,
IO/artifacts, scientific changes, or a Stage 8 claim.

Maintenance 6 Pint Physical Configuration Boundary is the
**Design-complete / Undispatched** next package gate under
`docs/implementation/maintenance_6_pint_physical_configuration_boundary.md`.
It retains exact TensorCore `0.13.0`, selects exact Pint `0.25.3`, moves public
physical Config values to copied canonical scalar quantities, and extracts
plain unit-suffixed execution facts exactly once during preparation. Runtime
records, producers, validators, tensor payloads, collections, RNG addressing,
and scientific equations remain Pint-free. Its bounded TensorCore
golden-path cleanup removes duplicate private admission guards while
preserving public ingress, scientific, relationship, storage, and generated-
product checks. Production implementation remains undispatched.

The first Stage 8 real-CUDA attempt correctly stopped before any accepted
measurement when its protected-suite gate over-applied those macOS literals to
the frozen Della Linux/x86_64 stack. Exact Stage 8 authority
`84802c1f2c89a6a5deeec305ce7bb2cd9ad2e829` and executable input
`728840bf2858c861104d5f7bb3cdbb4e3e1361b5` remain immutable stopped evidence
and are not executable authority. Stage 8 requires a new Design authority after
Maintenance 6 before any rerun. Later GPU
characterization and integration production remain undispatched.

If implementation reveals a concrete contradiction in the accepted design, stop
and send the issue back to Design. Do not silently widen architecture, create
placeholder package trees, add DAG-facing surfaces, rename public concepts,
fork TensorCore, or copy donor code into production modules inside an
implementation thread.

## Package Shape And Imports

Use the ecosystem naming convention:

```text
Project/display folder: TensorDSLab
Python import package:  tensor_dslab
```

The checkout root is the project folder. The `tensor_dslab/` directory is the
Python import package. Do not create a
flat TitleCase Python package that imports as `TensorDSLab`.

The implemented Maintenance 5 baseline uses this product-centered readout
tree. It retains Maintenance 4's product/runtime structure while removing the
redundant public sampling-config module:

```text
tensor_dslab/
  common/
    axes.py
  readout/
    config.py
    collection.py
    requirements.py
    simulation.py
    runtime/
      prepare.py
      sampling.py
    photoelectrons/
      field.py
      runtime/
        validate.py
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
```

This is the implemented Maintenance 5 package tree, not permission to create
additional placeholders. Maintenance 6 may add only the accepted
`common/units.py` behavior and the exact allowlisted physical-config/runtime
changes in its work order; it does not reorganize this product tree.
Maintenance 2 realized the preceding product/module ownership
migration without compatibility shims, and Stage 7 completed
`readout/simulation.py`; their private `_produce.py`, `*Plan`,
`_requirements.py`, and `charge/effects/` paths remain exact historical facts.
Maintenance 2 pins TensorCore's published generic RNG
and `require_same_dtype` surface at exact version `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The migration is Merged / Closed
at the exact candidate and Design closeout recorded above.

Materialize only modules with real behavior accepted by the active work order.
Every generated product owns its final `TensorField` leaf, public configs, and
one non-exported `runtime/` package containing explicit preparation,
production, and deep-validation actions. `Photoelectrons` remains the
already-produced truth input and owns only its field and runtime deep
validator. `readout.runtime.prepare` owns whole-request preparation;
`readout.runtime.sampling` owns the private shared sampling execution facts;
and `readout.simulation` remains the thin owner of the unchanged public
`simulate_readout(...)` signature, topological action sequence, exact
retention, and final collection construction. Shared semantic axes remain in
`common`; Maintenance 5 makes the source `SampleAxis` the sole sampling
authority and removes `SamplingConfig`. Maintenance 6 keeps those compact
integer axes, adds deliberate Pint construction/access conveniences at the
TensorDSLab boundary, and never sends quantities into `SamplingRuntime` or
tensor execution. Charge-specific multinomial/category
orchestration, count-domain helpers, and scientific effects remain private
under `charge/runtime/effects/`.

Runtime privacy is export-driven. Runtime modules and records remain ordinary
importable Python implementation details, but no Runtime, action, requirement,
or effect name is exported by a package facade or carries a compatibility
promise. Cross-module runtime actions omit a leading underscore; TensorCore's
semantic-leaf `_require()` hook and genuinely module-local helpers retain their
established underscore meaning. Runtime `__init__.py` files export nothing.

Every ProductRuntime and `SamplingRuntime` is a concrete final frozen slotted
dataclass with no Runtime base, Config, semantic product, collection, mutable
cache, or execution method. Preparation, production, and validation remain
explicit product-named actions rather than an ABC, registry, reflection layer,
or generic graph. Extract genuinely identical checks or preparation mechanics
into the narrowest private owner, while keeping product-specific semantics and
errors explicit; do not create broad `utils.py` or `helpers.py` modules merely
to make signatures look uniform.

Under the accepted Maintenance 6 target, Config construction owns public
quantity recognition, dimension conversion, canonical copying, and scalar
domain validation. Whole-request `prepare_readout(...)` owns public ingress,
closure, dtype, device, RNG capability, and stochastic-key admission. Private
child preparers trust that typed admission and own only contextual extraction,
model dispatch, scientific/representation checks, and Runtime construction.
Producers consume typed prerequisites plus plain Runtime facts and perform
tensor/RNG execution only. Validators remain immediate product and
relationship postconditions. Do not duplicate parent admission checks inside
private child actions, and do not remove scientific, axes-identity, freshness,
device/dtype, or generated-product checks merely because TensorCore supplies
generic roots.

Keep import direction acyclic: TensorCore, common, shared readout requirements,
product configs/fields, the source-bound sampling runtime, product runtime
actions, readout config/collection and whole-request preparation, readout simulation,
then deliberate package-root exports. Product runtime modules must not import
`ReadoutConfig`, `ReadoutRuntime`, `ReadoutCollection`, `simulation`, or
`simulate_readout(...)`. A producer imports no Config or validator. Generic
counter generation, logical positions, uniforms, parameterized Gaussian
draws, Poisson sampling, and binomial sampling belong to TensorCore;
TensorDSLab must not retain or rename `_random.py`.

Stage 6 behavior-neutrally renamed all four transitional waveform modules,
callables, imports, and tests from `_product.py` / `_product_*` to
`_produce.py` / `_produce_*`. Those paths remain exact closed Stage 6/7 and
Maintenance 2 evidence. Maintenance 4 supersedes them in the merged live tree,
without restoring `_product.py`, adding an alias, or providing a compatibility
shim for either private convention.

Merged Stage 6 implements the aggregate multinomial and hybrid Poisson
contracts selected in `docs/architecture/rebuild.md`. The five Poisson roles,
timing jitter, AP, and charge smearing have fixed append-only
`_RngStream` values through `CHARGE_SMEARING = 0x0000_000A`; discrete
probabilities, rates, and sampler control use binary64 independently of the
requested Charge dtype.
Conditional binomials use stable prepared current/later-category masses, the
selected exact small-mean inversion, and the cancellation-resistant large-mean
BTRS mapping with its central `1e-6` and complete-support mixed
absolute/relative high-precision log-bound gates; they do not
repeatedly subtract categories from one or recover a tiny complement as
`1-p`. The earlier BTRD direction and cancellation-prone three-log grouping
are retired. Active Charge count cells are bounded by `2**53 - 1`, additions
are checked before execution, and `K` is limited only by the accepted
role-address and requested-dtype accumulator-depth relations. Do not substitute per-
avalanche expansion, dependency distribution samplers, `torch.poisson`, a
normal approximation, global RNG, clipping, residual assignment,
renormalization, reseeding, or another exhaustion fallback. This is accepted
and implemented eager-reference behavior through exact Stage 6 candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58`. CUDA was unavailable, so the
accepted evidence makes no GPU execution or cross-backend claim.

That enum/module arrangement is retained only as closed Stage 5/6 evidence.
The closed Maintenance 2 implementation preserves the same default addresses
as exact
config-owned TensorCore `RngKey` values, uses TensorCore for generic RNG and
count-distribution mechanics, keeps Charge multinomial orchestration and count
bookkeeping in `readout/charge/effects/_counts.py`, and removes `_RngStream`,
`readout/_random.py`, and any replacement `_rng.py` without shims.
The merged Maintenance 4 implementation moves that Charge-owned behavior
intact to
`readout/charge/runtime/effects/counts.py`; it does not change the historical
Maintenance 2 record or any RNG behavior.

The active MVP crosstalk delay union is exactly
`FixedDelayConfig | ExponentialDelayConfig`. Although Stage 3 historically
implemented and exported `NormalDelayConfig`, Stage 6 removed that class, both
union memberships, all three export layers, and its tests without a
compatibility shim. Do not restore it or revive its zero-clipped law. Any later
normal, lognormal, tabulated, or other delay family requires a new calibrated
scientific and API decision. Closed Stage 3 records remain historical and are
not rewritten.

Fixed and exponential phase-marginalized delay preparation is frozen in
`docs/architecture/rebuild.md`. Fixed delay accepts every finite nonnegative
value and uses its exact represented two-point law with no PMF tolerance.
Exponential delay and AP recovery use their documented bounded ratio/sample
domains, analytic right tails, stable binary64 branches, and `1e-12` local /
`1e-11` complete-law tolerances. Do not replace these mappings with per-edge
latent draws, cutoff tails, clipping, residual assignment, renormalization, or
subtraction-derived overflow.

Timing jitter specifically integrates the latent-uniform plus ideal-Gaussian
law into binary64 destination probabilities during preflight. Its frozen log-
tail evaluator supports `2**-52 <= sigma / T <= 64`,
`2 <= sample_count <= 8192`, and `S * N <= 2**63`, with exact zero sigma as a
separate identity, `1e-12` local probability tolerance, and `1e-11` complete-
source-law L1 tolerance. Runtime scans every in-window target bin in increasing
order through aggregate conditional binomials and leaves one combined drop
category as the final no-draw count remainder. It must not draw a normal per
PE, call Box-Muller for jitter, impose an arbitrary Gaussian tail cutoff, clip
or normalize its law, or trade correctness for subquadratic sample-count work
without a later focused Design decision.

`Photoelectrons` is an already-produced dense truth input. It has no
`PhotoelectronsConfig`, no TensorDSLab readout preparer, producer, or Runtime
record, and only one explicit runtime deep validator. Source construction and
PE binning remain part of the future TensorG4DS bridge.

Runtime commands launched from the project root should use the project root on
`PYTHONPATH` so absolute `tensor_dslab.*` imports resolve:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

Maintenance 5 production imports stay absolute, such as:

```python
from tensor_core import CountAxis, LabelAxis, RegularAxis, TensorField
from tensor_dslab import Photoelectrons, ReadoutCollection
from tensor_dslab.common import ChannelAxis, ExampleAxis, SampleAxis
```

Do not rewrite imports to relative forms to satisfy editor-only diagnostics.
Editor analysis tools should mirror the runtime path by including the project
root on their analysis path.

Do not create placeholder modules to reserve architecture. Add a module only
when there is a real TensorDSLab concept, behavior, or contract to house.

## TensorCore Boundary

Maintenance 5 adopts exact published TensorCore `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b` and these semantic roots:

```text
TensorAxis[CoordinateT]
  CountAxis(*, count: int)
  RegularAxis(*, start: int, step: int, count: int)
  LabelAxis(*, labels: tuple[str, ...])
TensorField(tensor: torch.Tensor, axes: tuple[TensorAxis, ...])
TensorCollection(fields: Iterable[TensorField])
```

Production imports come from the public `tensor_core` package root. TensorCore
owns universal representation validation, constrained scalars, exact-type
lookup, generic relationship helpers, generic table roots, and the
`TensorArtifact` extension point. Maintenance 5 adopts no table, artifact, IO,
or Pint surface. Maintenance 6 consumes TensorCore's public
`Scalar.require(...)` normalization at the package-owned quantity boundary,
but Pint recognition, dimensions, canonical units, registry ownership, and
physical policy remain entirely TensorDSLab-owned. TensorCore has no retired
`0.6` ID/layout/metadata model,
generic selection or movement API, output-buffer/workspace API, or lifecycle
service. TensorDSLab must not recreate retired IDs, layouts, constants,
sidecars, compatibility shims, or generic operations.

`ExampleAxis`, `ChannelAxis`, and `SampleAxis` are direct final fieldless
leaves of `CountAxis`, `LabelAxis`, and `RegularAxis`, respectively. The six
product types are direct final fieldless `TensorField` leaves.
`ReadoutCollection` is a direct final fieldless `TensorCollection` leaf. Each
leaf has exactly its matching root in `__bases__`, with no mixin or other base.
Every semantic leaf uses inherited root construction, `@final`,
`__slots__ = ()`, no added stored fields, and one TensorDSLab `_require()`
narrowing hook. Do not reapply `@dataclass`, introduce an intermediate
semantic base, or override generic root behavior.

These are ordinary Python ABC extension points. TensorDSLab verifies its own
leaf declarations through static analysis, focused tests, and Review. Runtime
code validates documented public inputs and cheap correctness-critical
relationships; it does not police callers who subclass final leaves, mutate
classes, bypass constructors, call private functions, mutate exposed tensors,
or install custom Torch dispatch behavior. Such behavior is unsupported and
has no promised error category.

Supported exact-type axis and field lookup remains unchanged, and a missing
supported exact collection key raises `KeyError`. Malformed or off-path
class-object arguments have no promised runtime result or diagnostic category.
Maintenance 5 therefore deletes the current test that
`collection.field(TensorField)` must raise `TypeError`; it does not replace it
with another exception promise.

Coordinates are representation-specific and scoped by exact axis type. Axis
tuple order is tensor-dimension order. Code locates dimensions by exact axis
class, not loose names or constants. `ExampleAxis` exposes identity-free local
integer ordinals; `ChannelAxis` exposes exact string detector labels; and
`SampleAxis` compactly represents integer-picosecond left edges through
`start`, `step`, and `count`. The complete Photoelectrons/readout boundary
requires `start == 0`, while a nonzero-start semantic subgrid remains valid.
Numeric kernels use tensor indices plus one source-derived `SamplingRuntime`;
they do not materialize or parse semantic coordinates. There is no
`SamplingConfig`, `ExampleId`, `ChannelId`, `TensorAxisId`, `TensorFieldId`,
`IdSequence`, `TensorLayout`, `SampleGrid`, or `DigitizedWaveformSpec` in the
implemented Maintenance 5 baseline.

Every field contains exactly one example, channel, and sample axis in any
order, uses `torch.strided`, and reuses the exact source axis tuple for
dimension-preserving results. `Photoelectrons` is `torch.int64`; `Charge`,
`PureWaveform`, `NoiseWaveform`, and `AnalogWaveform` use one common
`torch.float32` or `torch.float64`; `DigitizedWaveform` is `torch.int32`.
`ReadoutCollection` accepts any nonempty unordered subset of those exact six
product types, with equal ordered axes, one device, and one common floating
dtype. It is a completed requested result, not a partial pipeline snapshot; it
has no add, replace, descendant-invalidation, or reconstruction lifecycle.

TensorCore establishes neither universal freshness nor universal storage
sharing. Every TensorDSLab field-returning operation classifies each successful
path as exact return, guaranteed storage-sharing, sharing permitted but
unspecified, or guaranteed fresh storage independent of named inputs. The MVP
classifies requested source `Photoelectrons` as an exact return and every
generated product as guaranteed fresh and pairwise storage-independent. Every
operation also owns dtype, device, axes, layout/strides, autograd,
synchronization, failure effects, and output-to-output relationships.

No write may begin through an alias after a semantic field has been constructed
and exposed. Producers initiate or enqueue all writes before constructing the
result field, and never later write through an alias to its storage. The public
MVP has no `out=`, destination collection, workspace, allocator, or stream
lease. Any later reusable destination remains raw, exclusive, and unexposed
until writes have been enqueued and the semantic field is constructed exactly
once. TensorCore contract changes still require TensorCore Design acceptance;
TensorDSLab does not fork it.

The historical Maintenance 2 dependency exposed public `RngKey`, `CounterRng`,
`Threefry4x32`, `logical_positions`, and `require_same_dtype` at exact
TensorCore `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. TensorCore owns generic
counter/address validation, Threefry word continuity, fixed-point uniforms,
parameterized Gaussian draws, Poisson inversion/PTRS, binomial
inversion/BTRS, sampler numerical domains, exhaustion, and those
distributions' internal word schedules. TensorDSLab owns exact
stochastic-role key placement in leaf configs, scientific position/category
lattices, direct-uniform/Gaussian ordinals, multinomial ordering and final
remainders, draw-free scientific policy, count accumulation, and ledgers.
Import only public TensorCore package-root names; do not copy or import
protected RNG or promoted distribution mechanics. Published TensorCore
`0.13.0` preserves that accepted public RNG surface while adding compact axes,
table roots, `TensorArtifact`, `Scalar`, and the golden-path runtime
simplification. Closed Maintenance 2 evidence remains scoped to exact `0.9.0`;
closed Stage 3 through 6 evidence remains scoped to `0.7.0`. TensorDSLab uses
`require_same_dtype` only for semantic-field relationships and retains raw
tensor checks plus its private scalar-to-dtype representation helper.

## Product Relationships And Boundaries

TensorDSLab should preserve this data-flow and ownership rule unless Design
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

The external chain is data flow, not package dependency flow. Core readout and
common modules depend only on TensorCore. A future downstream-owned bridge may
import an exact accepted public TensorG4DS type; TensorG4DS must never import
TensorDSLab to construct downstream identities. The bridge is a semantic
transformation, not a subclass cast or an assumption that TensorG4DS and
TensorDSLab axes are interchangeable.

The production integration target keeps tensor payloads resident on one
explicit accelerator device across TensorG4DS, TensorDSLab, and TensorML.
Boundary code must not silently call `.cpu()`, `.numpy()`, serialize/reload, or
otherwise use host materialization as the package handoff. New computations
may allocate new tensors on that same device, and TensorCore axes and other
small semantic records may remain ordinary host-side objects. Device movement
is always explicit. Because TensorG4DS has not yet frozen a public GPU output
contract, the exact accepted input type, dtype/axis matrix, and
device-preservation tests belong to the future integration work order; they do
not add a TensorG4DS dependency to the local readout foundation.

The discrete TensorG4DS bridge carries no end-to-end autograd promise and must
not detach silently. Its first work order should reject gradient-sensitive
inputs unless Design accepts a separate differentiable detector surface. This
does not weaken functional autograd for accepted deterministic waveform
transforms later in TensorDSLab.

The primary readout tensor handoff is `ReadoutCollection`, not a loose product
tuple or a required dataclass adapter. Runtime product and axis identity is the
exact concrete Python class. `ExampleAxis` coordinates are local integer
ordinals, `ChannelAxis` coordinates are detector-label strings, and
`SampleAxis` coordinates are regular integer-picosecond left edges.
TensorG4DS event values remain upstream provenance; the future bridge maps the
selected batch to local example ordinals rather than pretending those ordinals
are durable event identity. Durable provenance and serialization remain
deferred and must not be inferred from local ordinals or Python class names
without a focused artifact contract.

Consumer-facing adapters are deferred. TensorDSLab should first make the local
typed product graph coherent enough that future consumers can depend on it
without parsing raw `.fil`, table, array, manifest, or private representation
details.

The following is the implemented Stage 7 completed-readout contract. The
public readout operation consumes an already-produced dense truth field:

```text
Photoelectrons
  -> simulate_readout(products=..., config=..., rng=...)
  -> ReadoutCollection containing exactly the requested products
```

`Photoelectrons` contains binned photon-origin primary PE truth. It never
contains dark counts, timing jitter, correlated avalanches, or charge
smearing, and the builder never mutates or replaces it. Charge production uses
private working values in physical order: truth, optional dark counts,
optional timing jitter, optional correlated-avalanche simulation, and optional
smearing. Intermediate count, charge-ledger, and diagnostic values are private
implementation state rather than fields or durable products.

`Charge` is the finite floating aggregate PE-equivalent response per
readout channel and sample. It is not an SI-coulomb measurement and does not
claim an explicit individual-SPAD output. Pure and noise waveforms are
signal-only and noise-only components at one shared analog reference plane;
they are not sequential hardware products. Their composition produces the
analog waveform consumed by digitization.

`simulate_readout(...)` requires an explicit nonempty iterable of exact product
classes. It consumes the iterable once, rejects duplicates and unknown classes,
computes the transitive prerequisite closure, and completes one private typed
Runtime for every required product before any RNG request, production call, or
semantic-output write. The merged Maintenance 4 implementation prepares one
shared `SamplingRuntime` and composes the optional product Runtime values in one
`ReadoutRuntime`; Runtime presence is the execution signal rather than a
duplicated set of `need_*` flags. It then executes each producer at most once
and retains exactly the requested fields. Product preparers own their
scientific/contextual equations; `readout.simulation` composes no scientific
equation. Request order has no collection semantics. Unrequested prerequisites
remain private local values.

The accepted Stage 7 signature requires keyword-only `rng: CounterRng`, even
for deterministic closures; there is no simultaneous `seed=` parameter.
Deterministic closures request no values. Preflight rejects one `RngKey`
assigned to distinct stochastic roles in the requested closure before any RNG
request, producer invocation, or semantic-output write. TensorCore exposes no
non-consuming concrete-algorithm capability query: the public boundary accepts
nominal `CounterRng` membership, performs no dummy draw, and treats a real
custom RNG/backend incompatibility at its first genuine distribution request
as a dynamic execution failure.

The computational graph is:

```text
Photoelectrons -> Charge -> PureWaveform
Photoelectrons axes/device/shape + source SampleAxis -> NoiseWaveform
PureWaveform + NoiseWaveform -> AnalogWaveform -> DigitizedWaveform
```

`ReadoutConfig` composes only optional product configs. Sampling is source
structure rather than duplicated public configuration, so `ReadoutConfig()` is
the valid truth-only configuration. Config absence is structural. A requested
product requires the configs in its transitive closure; an unrequested branch
does not. Product configs describe science, not persistence, device movement,
allocation, mutation, accelerator streams, invocation seeds, or campaign
policy. Exact stochastic leaf configs may own immutable `RngKey` role
identities. `products` controls only final in-memory retention. IO is deferred.

The initial builder is functional. It borrows `Photoelectrons` read-only,
returns that exact field when requested, and creates guaranteed-fresh generated
products. Generated products retained together are storage-independent. The
builder does not mutate sources, silently move/cast/detach/host-materialize
inputs, expose private scratch, or write through any alias after exposing a
semantic result. Declared fresh generated-product dtype conversion is not an
input cast. Every supported statically preparable failure occurs before
the first RNG request, producer invocation, or semantic-output write. Dynamic
execution failures return no partial collection or exposed failed field but
carry no rollback guarantee for private allocations or completed local
prerequisites. Deep ingress validation and producer postconditions may
synchronize CUDA through scalar reductions; field construction itself adds no
synchronization point.

Stage 7 accepts source devices of exactly CPU or CUDA, including truth-only
requests. It validates `floating_dtype` only when the closure generates a
floating product; a truth-only request does not consume that control.

Stage 7 closed the result trust boundary deferred by Stage 4. Maintenance 4
preserves that boundary through explicit product-owned `validate_*` actions:
`simulate_readout(...)` performs `produce -> validate -> descendant` for each
generated product, passing the exact result and named direct prerequisites to
the validator; Photoelectrons deep validation remains in whole-request
preflight. Product `field.py` modules retain only their cheap intrinsic
TensorCore `_require()` narrowing. Invalid generated fields do not escape and
no downstream producer runs; these deep postcondition scans may synchronize
CUDA.

Every generated dimension-preserving field reuses the source's exact immutable
axis tuple and axis instances. Axis order may vary semantically; upstream
construction should ordinarily use example/channel/sample order so samples are
last for temporal kernels. Positional RNG addresses use tensor indices in the
actual dimension order, not semantic coordinate values, and therefore do not
promise permutation or arbitrary chunking invariance.

The completed MVP public surface has no atomic public transforms, mutation
lifecycle, generic projection/reconstruction helpers, `out=`, workspace,
allocation-free claim, or public stream policy. A later optimization stage
starts from measured GPU evidence and must preserve request, freshness,
exposure, synchronization, and lifetime contracts rather than reviving Stage
2's preconstructed writable collection model.

Projects/dag owns campaign fanout and fanin, scheduling, retry, repair,
compiled DAG objects, scheduler-visible grouping, status, and cross-shard
orchestration. TensorDSLab may later expose DAG-compatible executables,
operation specs, and recipe fragments only after local product and cache
contracts are accepted. Local dependency planning inside
`simulate_readout(...)` is TensorDSLab scientific orchestration, not campaign
orchestration.

For future caches, TensorDSLab owns deterministic storage-level compaction over
caller-supplied complete compatible products. Projects/dag owns scheduling,
fan-in, retries, repair, and campaign or cross-shard compaction orchestration.

## Validation Boundaries

TensorDSLab should move toward boundary-first validation:

```text
external/source/config/artifact boundary
  -> validate/coerce into strong typed objects
  -> construct TensorDSLab semantic leaves and collections
  -> hot path trusts those records
```

Validate strongly when data crosses into TensorDSLab or TensorCore native
records and when constructing new typed axes, products, collections, configs,
or artifacts. Leaf construction checks cheap intrinsic structure. Explicit
deep validation owns device-wide scientific scans at untrusted ingress and
producer postconditions. Do not repeatedly revalidate already constructed
graphs or materialize/parse compact semantic coordinates inside hot loops.

Use constrained scalar wrappers for meaningful numeric config/source/artifact
values where constraints matter. Tensor-local positive counts should use
TensorCore-owned `PositiveInteger`. Numeric wrappers should reject bool. Do not
add generic bool wrappers by default.

Implementation should validate supported public use, not adversarial attempts
to escape it. Unsupported subclassing, constructor bypass, class mutation,
private-function calls, exposed-tensor mutation, and custom dispatch require no
exhaustive detection or stable error behavior.

## Thread Roles

### Design

Design owns the target behavior.

Design should:

- define feature scope, invariants, accepted contracts, and non-goals;
- update design, decision, and architecture documentation when contracts
  change;
- produce implementation work orders;
- ratify, condition, revise, reject, or defer cross-package proposals;
- decide package governance adoption, conformance findings, deviations, and
  routing state;
- resolve package routing, conformance, architecture, and scope disputes;
- say what would require coming back to Design.

Design should not implement production code for the feature branch unless the
user explicitly delegates that exception. Design may directly edit
documentation during a documentation-only Design stage; those edits do not
require an Implementation handoff unless the user asks for that review loop.

### Implementation

Implementation owns the feature branch and is the default code-writing role.

Implementation should:

- make production code, test, and docs-sync changes required by the work order;
- keep the diff scoped to the work order;
- apply fixes requested by Validation and Review;
- keep the branch coherent and committed when asked;
- report commands run, known risks, and unresolved questions;
- stop and return to Design when a requested change would alter accepted
  architecture, ownership, scope, or non-goals.

Other threads should not modify production code or tests unless the user
explicitly delegates that exception. By default, they send findings or
suggested patches back to Implementation.

### Validation

Validation owns behavioral confidence.

Validation should:

- derive a test strategy from the design contract;
- identify missing edge cases and invariants;
- check whether tests prove behavior rather than mirror implementation details;
- exercise external integration paths when those paths are part of the behavior
  contract;
- interpret test failures;
- send concrete test gaps or suggested test cases to Implementation;
- dispatch the fixed branch and commit to Review when Validation clears.

Validation should not broaden scope or reopen architecture. By default,
Validation does not edit the feature branch; Implementation applies any test
changes. If a Validation finding would require changing the accepted stage
architecture or scope, Validation should send the issue back to Design rather
than asking Implementation to widen the branch.

### Review

Review owns final independent critique.

Review should:

- review the final or near-final diff for correctness, maintainability, typing,
  API fit, and scope control;
- verify external compatibility gates named by the work order;
- report findings first, ordered by severity;
- cite exact file and line references where possible;
- distinguish blockers from follow-up polish;
- send findings back to Implementation for fixes unless resolving them would
  require changing the accepted architecture or stage scope;
- issue explicit clearance on the fixed commit or identify the remaining
  blockers.

Review is read-only by default. It should not rewrite the branch unless the
user explicitly asks it to. If a Review finding requires an architecture or
scope change, Review should route it to Design instead of asking Implementation
to patch around the work order.

## Production Work Order Handoff

This section applies when Design dispatches implementation or another
state-changing stage. A documentation-only Design pass may remain in the Design
thread while it is being discussed.

Design should dispatch production work only after the source-of-truth work
order is committed and the base branch is clean, unless the user explicitly
accepts an exploratory exception. A dispatch must satisfy the complete
work-order checklist below and use Active, verified execution routes.

A Design work order should include:

- a stable package-owned work-order key and task; by default, the committed
  `docs/implementation/stage_<number>_<slug>.md` path is the key;
- exact Design and document baseline;
- base branch or commit and target branch;
- target files, packages, and public surfaces;
- source-of-truth docs to keep synchronized;
- invariants and validation rules;
- donor reference, comparison boundary, parity classification, and intentional
  divergences when donor behavior is in scope;
- scope and non-goals;
- minimum tests and verification commands;
- verified persistent Implementation, Validation, and Review routes;
- a finite Implementation/Validation loop budget;
- package-owned work-order state vocabulary and its source;
- known risks or open questions;
- stale-routing, architecture, scope, and other escalation or stop conditions;
- Review and clean-closeout expectations;
- what requires coming back to Design.

The strongest work orders include concrete code or test sketches when code or
tests are in scope. Avoid vague requests such as "add coverage"; say which
module, test name, public imports, helper boundaries, assertions, and forbidden
shortcuts matter. If a stage is docs-only, audit-only, or test-only, say that
clearly and repeat that production behavior changes require Design escalation.

When a work order or architecture doc names a public surface, it should include
a concrete sketch of that surface unless the surface is explicitly deferred.
Sketch dataclasses, functions, modules, validation helpers, and expected tests
with enough detail that Implementation can execute without inventing the
contract and Review can compare the diff against a specific target.

## Production Implementation And Validation Loop

After Design sends a production-code work order, Implementation and Validation
may iterate until the branch is stable:

```text
Implementation builds -> Validation tests/critiques -> Implementation fixes
```

Implementation and Validation may message each other automatically only when
the work order explicitly authorizes the loop, provides Active and verified
logical routes, and defines the finite budget. Raw platform route identifiers
remain private routing attributes and are not work-order identity. This loop is
bounded:

- maximum three Implementation-to-Validation dispatches;
- maximum three Validation-to-Implementation dispatches;
- each message must be specific and actionable;
- no architecture changes or scope expansion;
- no branch ownership changes;
- stop early when Validation reports no blocking findings;
- stop if a required route becomes stale, Deferred, missing, or discrepant;
- stop and ask the user or Design if the same issue repeats twice, the loop
  budget is exhausted, or a Design decision is needed.

Expected message shapes:

```text
Implementation -> Validation:
branch/commit, scope, files changed, docs updated/checked, commands run,
invariants to attack

Validation -> Implementation:
severity-ordered findings, missing tests, edge cases, suggested test cases

Implementation -> Validation:
fixes made, new commit, verification run, deferred items

Validation -> Review:
fixed branch/commit, validation scope, commands run, residual risks,
cleared for Review

Validation -> Implementation:
remaining blockers, or architecture/scope issue routed to Design
```

Ready for Review means:

- scoped behavior is implemented;
- Validation findings are resolved or explicitly deferred;
- tests requested by Validation pass;
- applicable design, decision, implementation, validation, review, and
  architecture docs are synchronized with any changed public contracts;
- generated files, caches, and unrelated outputs are not staged;
- the branch is committed when the work order asks for a commit;
- the handoff names a fixed branch and commit rather than a moving target,
  unless the user accepted an exploratory exception and the handoff reports
  that risk;
- the handoff lists commands run and remaining risks.

When Validation clears a fixed branch, Validation should dispatch that branch
and commit to Review. If Validation does not clear, it should send actionable
findings back to Implementation. If those findings would require changing the
accepted architecture, non-goals, or stage scope, Validation should stop and
ask Design instead.

## Production Review Gate

Send a production branch to Review only after the implementation/validation
loop is quiet. Review should not be asked to review a moving target unless the
request is explicitly an early design or architecture review.
Documentation-only Design changes do not require this gate unless the user
asks for an independent review.

While reviewing, Review remains read-only and should send findings back to
Implementation rather than rewriting the branch. If a Review finding would
require changing accepted architecture, non-goals, or stage scope, Review
should route it to Design instead of Implementation. After Implementation
fixes Review findings, Review should recheck the fixed branch and commit.

After Review reports no remaining findings on a fixed branch, Review owns the
closeout merge:

- fast-forward merge the cleared branch into the work order's target base
  branch, normally `main`;
- run the post-merge verification commands named by the work order;
- report the resulting base branch HEAD, commands run, and any residual risk;
- ask Design to open discussion of the next stage or direction.

If the merge is not a clean fast-forward, verification fails, the worktree is
dirty, or the target branch is ambiguous, Review should stop and report the
blocker instead of resolving conflicts, rewriting history, force-pushing, or
changing implementation code.

## Documentation Synchronization Gate

Implementation, Validation, and Review should treat documentation drift as a
real review item. Every stage should leave the source-of-truth docs aligned
with the implemented contract.

Before Review, check whether the change requires updates to:

- `docs/implementation/stage_*.md` when a stage work order, scope, handoff, or
  accepted implementation surface changed;
- `docs/architecture/common.md`, when present, for shared helpers, cache
  schema, durable representation shape, manifest behavior, validation
  behavior, or cross-domain contracts;
- `docs/architecture/<domain>.md` when a domain public contract, product shape,
  builder/loader contract, validation rule, or representation bridge changed;
- `docs/architecture/tensors.md` when TensorCore integration, semantic axes or
  fields, result sharing/freshness, placement, synchronization, exposure, or
  lifetime contracts changed;
- `docs/parity.md` when donor references, comparison boundaries, fixtures,
  tolerances, RNG comparisons, or intentional-divergence claims changed;
- `docs/design.md` when end-to-end domain flow or ownership boundaries changed;
- `docs/decisions.md` when a semantic choice was accepted, renamed,
  superseded, or explicitly deferred;
- `docs/validation.md` for expected behavior, validation cases, fixtures,
  failure modes, or numeric tolerances changed;
- `README.md`, `AGENTS.md`, or `CONTRIBUTING.md` when workflow, onboarding, or
  repository-wide expectations changed.

Before Review, the Implementation handoff must identify documentation updated,
documentation checked but unchanged with a reason, verification commands run,
residual risks, and intentionally deferred items.

Update `docs/governance/` when package adoption state, conformance evidence,
semantic rule mappings, deviations, routing posture, Coordination status, or
the adopted Governance Core version changes. Governance records must
distinguish a proposed decision from an issued package decision.

Implementation handoffs should explicitly say which docs were updated, or why
no docs update was needed. Validation and Review should run targeted stale-name
searches when a public term is renamed. Keep legitimate historical mentions
only when they are clearly framed as historical, deferred, or superseded.

## Verification Baseline

For documentation-only Design changes, run at minimum:

```bash
git diff --check
```

Also run targeted link, heading, and stale-term checks appropriate to the
change.

Before production Review, run the smallest relevant verification set for the
change. At minimum, run:

```bash
git diff --check
```

For Python changes after the package exists, also run the relevant test suite:

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

For future DAG-compatible changes, also validate the DAG repo-facing operation
specs with the repository's accepted command.

If a repository requires a specific environment, use that environment and
report the exact command.
