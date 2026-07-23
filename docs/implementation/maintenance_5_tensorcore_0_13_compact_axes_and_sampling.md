# Maintenance 5 TensorCore 0.13 Compact Axes And Sampling Work Order

Status: **Design-complete / User-authorized / Undispatched**.

Stable work-order key:
`TensorDSLab/maintenance-5-tensorcore-0-13-compact-axes-and-sampling`.

This is the package-authoritative TensorDSLab Design work order for one
breaking, pre-deployment dependency and semantic-axis migration. It adopts the
published TensorCore `0.13.0` compact-axis and golden-path contracts, replaces
materialized example/sample coordinate strings with their accepted compact
representations, and makes the source `SampleAxis` the sole sampling authority.

The user authorized Design to finish and dispatch this work. Production starts
only from the exact clean containing Design commit named in the dispatch
handoff. A commit cannot contain its own hash.

## Objective

Perform one atomic migration:

```text
TensorCore 0.9.0
direct TensorAxis string leaves
SamplingConfig + matching SampleAxis
  ->
TensorCore 0.13.0
CountAxis / LabelAxis / RegularAxis representation roots
source-derived SamplingRuntime
```

The maintenance must:

- pin exact published TensorCore `0.13.0`;
- define `ExampleAxis` as an identity-free local ordinal count axis;
- retain detector channel identity as explicit string labels on
  `ChannelAxis`;
- define `SampleAxis` as a compact regular integer-picosecond axis;
- retain arbitrary tensor-dimension axis order;
- remove `SamplingConfig`, `ReadoutConfig.sampling`, and the duplicate
  source/config agreement check without a shim;
- derive the private `SamplingRuntime` once from the exact source axis;
- retain the complete-readout requirement that sample time starts at zero;
- preserve every scientific equation, stochastic address, tensor operation,
  result relationship, and supported CPU/CUDA boundary; and
- adopt TensorCore `0.13.0`'s supported golden-path lookup boundary without
  creating new runtime policing.

This is not the Pint migration. The axis stores plain canonical integers and
the existing public `*_ps` integer properties remain integer-valued.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`.

The exact clean package baseline is the Maintenance 4 Design closeout:

```text
repository:          TensorDSLab
reference:           main
commit:              a46899c4e3bacd6deec23ea64da5e68b382816e9
tree:                79fcd2497834931a43fe48cdb6aa9f8eee534b28
package version:     0.1.0
Python requirement:  >=3.11
```

Maintenance 4 is Merged / Closed through exact Review-cleared supplemental
candidate `b3c7c907004741ba67b8b92a54bbdc8c85216dda`. Its production, tests,
and metadata are contained unchanged in the starting baseline above.

The exact published dependency target is:

```text
repository:       https://github.com/mbedard44/TensorCore.git
reference:        origin/main
commit:           202d8b1bc6259b8453d3d377570417f2480d782b
direct parent:    f62506b6d2f6926db446e2d163f26870575c9419
tree:             48fa9a28db6d043abc07d9963b2015983ca436ea
package version:  0.13.0
Python:           >=3.11
Torch:            >=2.11,<2.13
root exports:     30
installed files:  14
```

Design independently recreated the exact no-prefix Git ZIP:

```text
SHA-256:  ed804c71a617a79a63b53be86157e2045322d6c6868ca3766dc75d5526cb8b09
bytes:    373491
```

Implementation, Validation, and Review must independently reconstruct and
verify the exact dependency source and archive. A locally similar checkout,
moving branch, version-only constraint, tag, PyPI package, or unpinned Git
reference is not the selected dependency.

The Design branch is:

```text
codex/maintenance-5-tensorcore-0-13-compact-axes-design
```

The Implementation branch is:

```text
codex/maintenance-5-tensorcore-0-13-compact-axes-and-sampling
```

Implementation creates the latter from the exact Design authority. Raw
platform route identifiers remain private and must not enter repository files.

Package governance remains:

```text
package_adoption_state:    Adopted
conformance_finding:       Not evaluated
coordination_status:       Deferred
registry_storage_profile:  Disabled
maintenance_4:             Merged / Closed
maintenance_5:             lifecycle-dependent state above
stage_8:                   Stopped / new authority required after this work
```

Coordination is not an execution route. This work authorizes no push.

## Applicable Contracts And Source Precedence

Implementation, Validation, and Review must read and reconcile:

- `AGENTS.md` for role, route, handoff, finite-loop, and merge rules;
- `CONTRIBUTING.md`, especially TensorCore Backbone, Coordinates Versus
  Indices, Boundary-First Validation, Public Surface Discipline, Public
  Typing, Test Expectations, and Scope Discipline;
- [TensorCore Integration](../architecture/tensors.md);
- [Readout Architecture](../architecture/readout.md);
- [Rebuild Architecture](../architecture/rebuild.md);
- [Design](../design.md);
- [Decisions](../decisions.md);
- [Validation](../validation.md);
- [Parity](../parity.md);
- the closed Stage 3 through Stage 7 and Maintenance 2 through 4 work orders
  as immutable historical implementation/evidence records; and
- TensorCore's public API and architecture at exact commit
  `202d8b1bc6259b8453d3d377570417f2480d782b`.

This work order controls the Maintenance 5 implementation slice. Living
architecture documents control current accepted target meaning. Closed work
orders preserve the exact older `0.7`/`0.9` contracts they implemented and
must not be rewritten. If operative sources conflict, stop and return the
exact contradiction to TensorDSLab Design.

This migration changes representation and public construction, not donor
physics or a comparison boundary. `docs/parity.md` records the exact
old-to-new `T`/`N` comparison fixture, but no parity classification changes.

## Design Finding

TensorCore `0.13.0` now provides the generic representations TensorDSLab had
been encoding manually:

```text
TensorAxis[CoordinateT]
  CountAxis   -> range(0, count)
  RegularAxis -> range(start, start + step * count, step)
  LabelAxis   -> tuple[str, ...]
```

The current TensorDSLab design duplicates sampling state:

```text
SamplingConfig(sample_period_ps, sample_count)
SampleAxis("0ps", "2000ps", ...)
```

Request preparation then proves that two independently constructed values
agree. The compact `SampleAxis(start, step, count)` already carries every
sampling fact used by readout execution. Retaining both values would create
two authorities and preserve a public class with no independent role.

The selected replacement is therefore not an adapter over the old API.
`SamplingConfig` is removed cleanly and the source axis becomes authoritative.

## Exact TensorCore Consumer Boundary

TensorDSLab imports supported TensorCore names only from the package root.
Maintenance 5 consumes:

```text
TensorAxis
CountAxis
RegularAxis
LabelAxis
TensorField
TensorCollection
RngKey
CounterRng
Threefry4x32
existing public relationship and numeric requirements
```

TensorCore `0.13.0` also exposes `TableColumn`, `TableField`,
`TableCollection`, `TensorArtifact`, and `Scalar`. Their existence is
acknowledged so TensorDSLab does not make a false generic-capability claim.
Maintenance 5 does not add a TensorDSLab table, artifact, IO surface, scalar
constraint, or Pint boundary.

TensorCore continues to enforce constructed-value invariants:

- scalar value domains and normalized primitives;
- axis representation state;
- semantic-root membership;
- exact semantic-type uniqueness;
- tensor shape versus ordered axis sizes;
- table row-count equality; and
- well-typed axes/device/dtype/field-set relationships.

TensorCore `0.13.0` deliberately leaves supported Python structure to typing,
tests, documentation, and Review. TensorDSLab follows that golden path:

- its semantic leaves are publicly `@final`;
- each leaf is fieldless and directly inherits one accepted root;
- it adds no metaclass, registry, subclass scanner, or runtime-finality guard;
- supported exact-type field and axis lookup remains unchanged;
- a missing supported exact collection key still raises `KeyError`; and
- malformed/off-path class-object arguments have no promised result or
  exception category.

The current assertion that `collection.field(TensorField)` must raise
`TypeError` is retired. It is not replaced by a different off-path exception
assertion.

## Exact Public Axis Contract

### `ExampleAxis`

The target declaration is structurally:

```python
@final
class ExampleAxis(CountAxis):
    __slots__ = ()

    def _require(self) -> None:
        if self.count == 0:
            raise ValueError("ExampleAxis must be nonempty")
```

Its inherited public constructor is:

```text
ExampleAxis(*, count: int)
```

Its coordinates are exact zero-based local ordinals:

```text
0, 1, ..., count - 1
```

`coordinates` is an exact `range`, not a tuple. It is O(1) representation
state even for a very large accepted count. `coordinate_at` and `index_of`
return/accept exact integers under TensorCore's contract.

An `ExampleAxis` carries no durable event identity, provenance, or promise
that two collections contain the same real-world examples merely because
their axes compare equal. Upstream and downstream boundary code owns durable
identity where needed.

### `ChannelAxis`

The target declaration is structurally:

```python
@final
class ChannelAxis(LabelAxis):
    __slots__ = ()

    def _require(self) -> None:
        if not self.labels:
            raise ValueError("ChannelAxis must be nonempty")
```

Its inherited public constructor is:

```text
ChannelAxis(*, labels: tuple[str, ...])
```

TensorCore owns exact tuple, exact nonempty string, and uniqueness checks.
TensorDSLab adds only the nonempty-axis requirement. Channel labels retain
their current identity-bearing meaning and order.

### `SampleAxis`

The target declaration is structurally:

```python
@final
class SampleAxis(RegularAxis):
    __slots__ = ()

    def _require(self) -> None:
        ...
```

Its inherited public constructor is:

```text
SampleAxis(*, start: int, step: int, count: int)
```

The TensorDSLab narrowing is exact:

- `start >= 0`;
- `step > 0`;
- `count >= 2`; and
- `start + step * count <= 2**63 - 1`.

TensorCore first requires exact built-in integers, rejects `bool`, checks
`0 <= count <= 2**63 - 1`, and rejects zero step. TensorDSLab does not
duplicate those generic checks.

The represented sample coordinates are integer picosecond left edges:

```python
range(start, start + step * count, step)
```

The stop is exclusive. `coordinates` is an exact nonmaterializing `range`.
There is no stored tuple, timestamp string, unit string, cached quantity, Pint
object, or parsing path.

The existing integer convenience properties remain public and are defined
directly from inherited state:

```python
axis.start_ps             == axis.start
axis.sample_period_ps     == axis.step
axis.stop_ps              == axis.start + axis.step * axis.count
```

All three return exact Python integers. A later Pint stage may add separate
physical convenience accessors without changing this stored state.

`SampleAxis` permits a valid nonzero-start regular subgrid. The complete
TensorDSLab `Photoelectrons`/readout boundary separately requires
`sample_axis.start == 0`, because readout time is example-local.

### Equality, hashing, and order

TensorCore owns equality and hashing by exact concrete semantic type plus full
representation state:

```text
ExampleAxis: type + count
ChannelAxis: type + labels
SampleAxis:  type + start + step + count
```

Equal represented values from another semantic axis type are not equal.

Every readout field still contains exactly one `ExampleAxis`, one
`ChannelAxis`, and one `SampleAxis` in arbitrary tensor-dimension order.
Operations locate dimensions by exact axis type. Dimension-preserving
generated fields retain the exact source axis tuple and exact axis objects.

## Sampling Ownership

### Retired public surface

Maintenance 5 removes without an alias, deprecation shim, or compatibility
module:

```text
tensor_dslab.common.sampling
SamplingConfig
SamplingConfig.build_axis()
ReadoutConfig.sampling
```

`SamplingConfig` disappears from `tensor_dslab.common.__all__`,
`tensor_dslab.__all__`, module imports, tests, typing fixtures, README
examples, and living architecture.

The pre-deployment package accepts this clean break. It does not support both
the old string/sampling model and the new compact model simultaneously.

### `ReadoutConfig`

`ReadoutConfig` contains only the five optional exact generated-product
configs:

```python
@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutConfig:
    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None
```

Its existing exact optional-component checks remain. `ReadoutConfig()` is the
minimal valid config for a truth-only request. Product closure still rejects a
missing config for every generated product that must execute.

### Source-derived runtime

`SamplingRuntime` remains one private final frozen slotted record:

```python
@final
@dataclass(frozen=True, slots=True)
class SamplingRuntime:
    sample_count: int
    sample_period_ps: int
    sample_dimension: int
```

Maintenance 5 does not add redundant start or stop fields because execution
does not consume them. The exact source `SampleAxis` already owns them.

The preparer becomes:

```python
def prepare_sampling(
    photoelectrons: Photoelectrons,
) -> SamplingRuntime:
    sample_dimension = photoelectrons.dimension_of(SampleAxis)
    sample_axis = photoelectrons.axis(SampleAxis)
    if sample_axis.start != 0:
        raise ValueError("sample-axis start must be zero")
    return SamplingRuntime(
        sample_count=sample_axis.count,
        sample_period_ps=sample_axis.step,
        sample_dimension=sample_dimension,
    )
```

The implementation may use an exact equivalent spelling. It must not
materialize `coordinates`, parse a string, construct a second axis, or accept
a second sampling policy.

`prepare_readout(...)` calls `prepare_sampling(photoelectrons)` after exact
public input type/config/RNG validation and before any RNG request, product
producer, or semantic-output write. A truth-only request also passes through
this boundary. A nonzero-start source therefore fails before RNG activity or
output effects.

The generated Charge, pure-waveform, and noise preparers continue to consume
the same plain `SamplingRuntime` fields. Their production modules need no
change.

## Public API Consequences

The public simulation callable keeps its exact signature:

```text
simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype = torch.float32,
) -> ReadoutCollection
```

The deliberate breaking changes are:

```text
ExampleAxis(coordinates=(...))       -> ExampleAxis(count=...)
ChannelAxis(coordinates=(...))       -> ChannelAxis(labels=(...))
SampleAxis(coordinates=(...))        -> SampleAxis(start=..., step=..., count=...)
ReadoutConfig(sampling=..., ...)     -> ReadoutConfig(...)
```

The package root continues to export the three semantic axes, five product
config families, six product fields, `ReadoutCollection`, `ReadoutConfig`, and
`simulate_readout`. It no longer exports `SamplingConfig`. TensorDSLab does not
re-export `CountAxis`, `RegularAxis`, `LabelAxis`, `Scalar`, TensorCore RNG
types, table roots, or `TensorArtifact`.

## Scientific, RNG, And Result Continuity

The frozen comparison fixture is:

```text
old source:
  SamplingConfig(period=T, count=N)
  SampleAxis("0ps", "Tps", ..., "(N-1)Tps")

new source:
  SampleAxis(start=0, step=T, count=N)
```

With identical tensor payload, tensor axis order, channel label order,
product configs, dtype, device, and `CounterRng`:

- `SamplingRuntime.sample_count` remains `N`;
- `SamplingRuntime.sample_period_ps` remains `T`;
- `SamplingRuntime.sample_dimension` is unchanged;
- every prepared scientific scalar/tensor remains unchanged;
- RNG keys, logical positions, ordinals, requests, and draw-free branches
  remain unchanged;
- same-stack completed tensor values remain exact;
- generated product dtype/device/axes/storage/autograd contracts remain
  unchanged; and
- requested `Photoelectrons` remains the exact supplied field.

Example coordinates were never part of the frozen positional RNG address.
Replacing labels with local ordinals therefore changes no RNG value when
tensor order and shape are unchanged.

The migration must not update stochastic golden values, reference equations,
scientific tolerances, distribution statistics, or accepted environment
qualification merely to make tests pass. Any observed value drift is a stop
condition and returns to Design with a minimal reproducer.

## Production And Dependency Scope

The implementation production/dependency allowlist is exactly:

```text
pyproject.toml
tensor_dslab/__init__.py
tensor_dslab/common/__init__.py
tensor_dslab/common/axes.py
tensor_dslab/common/sampling.py                 (delete)
tensor_dslab/readout/config.py
tensor_dslab/readout/runtime/prepare.py
tensor_dslab/readout/runtime/sampling.py
```

No other `tensor_dslab/` production path may change.

The exact expected production effects are:

- one dependency-pin replacement;
- one axis-leaf migration;
- one public module deletion;
- two facade export removals;
- one `ReadoutConfig` field/check removal;
- one source-derived sampling preparer; and
- one updated preparer call site.

In particular, no Charge effect, waveform producer, noise algorithm,
validator, field, collection, RNG, requirement helper, or simulation function
body changes.

## Test And Typing Scope

The implementation test/support allowlist is exactly:

```text
tests/readout_fixtures.py
tests/test_charge_delay_preparation.py
tests/test_charge_product.py
tests/test_deterministic_waveform_products.py
tests/test_noise_waveform_product.py
tests/test_package_contracts.py
tests/test_readout_axes_and_sampling.py
tests/test_readout_collection.py
tests/test_readout_configs.py
tests/test_readout_product_types.py
tests/test_readout_simulation.py
tests/test_rng_ownership_migration.py
tests/test_runtime_action_ownership.py
tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py
tests/typing/maintenance_4_runtime_action_ownership.py
tests/typing/stage_3_semantic_leaf_contracts.py
tests/typing/stage_4_deterministic_waveform_products.py
tests/typing/stage_7_public_readout_orchestration.py
```

These historical-named typing fixtures remain active package-wide probes; the
filenames are not renamed.

`tests/readout_fixtures.py` may replace its test-only direct `TensorAxis` leaf
with one fieldless representation-root leaf and update heterogeneous generic
annotations for TensorCore `0.13.0`. That is test support, not a public
TensorDSLab type.

Tests whose intended axes mismatch previously changed only example labels must
change a meaningful compact state: example count/shape, channel labels, or
sample state as appropriate. Equal count axes deliberately compare equal.

No other test path may change without a returned Design finding and explicit
scope amendment.

## Required Focused Runtime Evidence

Focused tests must prove all of the following against exact TensorCore source
and archive forms.

### Axis representation and construction

- exact bases:
  `ExampleAxis.__bases__ == (CountAxis,)`,
  `ChannelAxis.__bases__ == (LabelAxis,)`, and
  `SampleAxis.__bases__ == (RegularAxis,)`;
- every leaf is statically final, fieldless, slotted, and lacks `__dict__`;
- inherited constructors are keyword-only and have the accepted signatures;
- downstream `_require()` runs after TensorCore's generic validation;
- exact integer inputs are required for count/start/step and `bool` is
  rejected at runtime;
- `ExampleAxis(count=0)` and `ChannelAxis(labels=())` fail;
- a zero-size generic TensorCore representation remains generic-valid;
- `SampleAxis` rejects negative start, nonpositive step, count below two, and
  exclusive-stop overflow;
- exact signed-int64 exclusive-stop boundary values pass/fail correctly;
- a nonzero-start valid `SampleAxis` constructs successfully;
- Count/Sample `coordinates` are exact `range` values and large accepted axes
  construct without coordinate materialization;
- Channel coordinates retain exact label tuple identity/value semantics;
- `coordinate_at` and `index_of` return the expected int/int/str types and
  preserve TensorCore's TypeError/IndexError/KeyError categories;
- equality and hashing use exact leaf type plus full representation state;
  and
- the three integer `SampleAxis` properties return exact values and exclusive
  stop.

### Source-derived sampling

- `prepare_sampling(photoelectrons)` derives the exact sample dimension,
  count, and step from arbitrary accepted axis order;
- it reuses no config and materializes no coordinate tuple;
- a valid nonzero-start `SampleAxis` can exist on a semantic field but fails
  the complete readout/preparation boundary;
- truth-only public requests enforce start zero before producer, RNG, or
  collection effects;
- different valid source count/step values propagate into the one shared
  `SamplingRuntime` consumed by every required temporal ProductRuntime;
- no duplicate count/period agreement test remains because there is no second
  policy;
- source axes and tensors remain unchanged; and
- generated products reuse the exact source axes tuple and instances.

### Public API and golden path

- `SamplingConfig` is absent from both public facades and its defining module
  path is absent;
- no alias, dynamic fallback, or compatibility import exists;
- `ReadoutConfig()` is valid and all optional component exact-type checks
  remain;
- generated closure config requirements remain unchanged;
- the `simulate_readout(...)` function object, signature, and export identity
  remain unchanged;
- supported exact collection lookup and missing-product `KeyError` remain;
- the old abstract-root `collection.field(TensorField)` exception assertion
  is absent and no substitute diagnostic promise exists; and
- TensorDSLab production directly imports no TensorCore private module,
  NumPy, Pint, or downstream package.

### Continuity

- every existing reference equation, stochastic invariant, exact-zero
  no-draw path, same-stack replay, source immutability, storage freshness,
  axes identity, dtype/device, validation, and autograd gate remains;
- a focused old-state/new-state comparison uses the same `T` and `N` and
  proves identical prepared runtime values;
- existing exact same-stack stochastic literals remain unchanged; and
- full source/archive discovery passes without modifying science thresholds.

## Static Typing Evidence

Pyright `1.1.411` in standard project mode must report zero diagnostics
against both exact dependency forms.

Positive probes must establish:

```text
ExampleAxis(count=...).coordinate_at(...)     -> int
ExampleAxis(...).index_of(...)                -> int
SampleAxis(...).coordinate_at(...)            -> int
SampleAxis(...).index_of(...)                 -> int
ChannelAxis(labels=...).coordinate_at(...)    -> str
ChannelAxis(...).index_of(...)                -> int
field.axis(ExampleAxis)                       -> ExampleAxis
field.axis(ChannelAxis)                       -> ChannelAxis
field.axis(SampleAxis)                        -> SampleAxis
field.coordinate_at(ExampleAxis, ...)         -> int
field.coordinate_at(ChannelAxis, ...)         -> str
field.coordinate_at(SampleAxis, ...)          -> int
ReadoutConfig()                               -> ReadoutConfig
prepare_sampling(photoelectrons)              -> SamplingRuntime
```

Static negative probes cover expressible constructor-domain errors such as
float/string values, old `coordinates=`, old `sampling=`, and retired
`SamplingConfig` imports. Runtime tests, not typing, prove `bool` rejection
because Python typing treats `bool` as an `int` subtype.

## Exact Dependency And Package Gates

Every role must verify:

- exact TensorCore commit, parent, tree, version, 30-name root export order,
  and 14-file installed topology;
- exact direct-reference pin in `pyproject.toml`;
- independently recreated canonical archive hash;
- byte equality of TensorCore production files between source and archive;
- public-root-only TensorCore imports from TensorDSLab production;
- fresh-process package isolation from TensorG4DS, TensorML, G4DS, and Pint;
- source/AST scans proving TensorDSLab production directly imports neither
  NumPy nor TensorCore private implementation paths (transitive Torch and
  TensorCore implementation modules in `sys.modules` are not isolation
  failures);
- TensorDSLab package-root/readout/common export identity and order;
- retired module/import absence;
- exact production/test allowlists;
- protected historical/governance/parity-classification bytes;
- no raw task IDs or private route facts;
- `git diff --check`; and
- no cache, bytecode, build, distribution, or egg-info artifacts.

The TensorDSLab package version remains `0.1.0`. This exact Git dependency
migration is not a package-index release or compatibility certification.

## Local And Real-CUDA Evidence

Implementation must run, from one clean fixed candidate:

- focused source and archive suites for every changed contract;
- complete source and archive test discovery;
- Pyright against source and archive forms;
- package/export/import isolation and forbidden-surface scans;
- exact dependency and archive gates; and
- mutation or equivalent concrete proofs for source-derived sampling,
  compact-axis narrowing, off-path test retirement, and unchanged positional
  science.

Validation reruns the complete fixed matrix independently. It must use one
fresh full-A100 allocation and run the exact candidate against both supported
PyTorch minors (`2.11` and `2.12`) and both TensorCore source/archive forms.
Every conditional CUDA test must execute; no unavailable-CUDA skip is accepted
inside that allocation.

Independent Review repeats the material local/static/source/archive gates and
uses its own fresh full-A100 allocation across the same two PyTorch minors.
Review must not reuse Validation's allocation or evidence tree.

Each role reports exact Python, PyTorch, CUDA runtime, GPU, driver, and relevant
library identities plus exact test totals. These are functional correctness
and dependency-adoption results, not benchmark or performance evidence.

No Stage 8 benchmark, profiler, allocation measurement, compile/fusion claim,
or performance threshold runs under this work order.

## Documentation And Historical Records

The containing Design authority synchronizes:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md
docs/implementation/proposed_pint_physical_configuration_boundary.md
docs/overview.md
docs/parity.md
docs/validation.md
```

The Pint candidate remains nonoperative and later. It is reconciled to treat
closed Maintenance 5 as its future package-local prerequisite and the
published TensorCore `0.13.0` Scalar/axis surface as available evidence.

During implementation, only the lifecycle/evidence portions of this work
order and `docs/implementation/index.md` may change. All other synchronized
Design documents are protected candidate inputs.

The following remain immutable historical records:

- every closed Stage 0 through Stage 7 work order;
- every closed Maintenance 1 through Maintenance 4 work order;
- the proposed historical TensorCore RNG consumer document;
- `docs/governance/**`;
- closed commit identities, test totals, and old exact dependency statements
  in their historical contexts; and
- parity donor classifications.

Living docs may describe supersession, but historical records are not
backfilled with the new API.

## Explicit Non-Goals

Maintenance 5 does not authorize:

- Pint, a unit registry, quantities, unit strings, or physical config
  migration;
- any use of TensorCore `Scalar.require()` or `Scalar.accepts()` in
  TensorDSLab production;
- TensorDSLab tables, `TensorArtifact`, IO, persistence, cache, or
  serialization;
- TensorG4DS ingestion, PE binning, provenance, or package integration;
- TensorML adapters, models, renderers, training, or dependency alignment;
- a public output buffer, `out=`, workspace, allocator, pool, stream lease, or
  allocation-free claim;
- a Stage 8 restart, benchmark, profiler, optimization, compilation, Triton,
  or custom CUDA;
- scientific equations, distributions, RNG keys/addresses, call order,
  stochastic laws, tolerances, or parity reclassification;
- new products, configs, transforms, reconstruction, or public operations;
- automatic axis selection, slicing, alignment, representation conversion, or
  materialization;
- event identity on `ExampleAxis`;
- integer channel-label support;
- a compatibility shim or dual axis model;
- a release, deployment, conformance, broad compatibility, or performance
  claim;
- sibling-repository edits; or
- a push.

## Stop Conditions

Stop the affected slice and return exact evidence to Design if:

- the package baseline, authority, dependency identity, branch, or route is
  discrepant;
- the worktree contains overlapping user changes;
- TensorCore source/archive bytes or public contracts differ from the fixed
  target;
- the migration requires a production path outside the exact allowlist;
- a product/effect implementation must change to complete the migration;
- exact old/new `T`/`N` comparison changes a prepared value, RNG call, or
  completed same-stack tensor value;
- TensorCore generic behavior must be copied, wrapped, or patched locally;
- a supported API needs a behavior not fixed here;
- a required local, typing, archive, or CUDA gate cannot run;
- CUDA evidence exposes a scientific, dependency, device, or execution
  discrepancy;
- protected historical, governance, or parity-classification bytes would
  change;
- a raw task ID or private route would enter a repository file; or
- completion requires Pint, IO, integration, Stage 8, or another excluded
  scope.

Do not silently widen the work order. A concrete TensorCore defect returns to
both package Design authorities with a minimal reproducer; it does not
authorize a TensorCore edit from TensorDSLab.

## Execution Roles And Finite Loop

The persistent package-owned routes are:

```text
TensorDSLab/default/Design
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Design verifies all routes as Active, current, return-capable, and bound to
this workspace/key before dispatch. Coordination remains Deferred and is not
used.

The lifecycle vocabulary is:

```text
Design-complete / User-authorized / Undispatched
Dispatched / Active
Implementation candidate / Validation pending
Validation returned / Implementation correction active
Validation cleared / Review pending
Review returned / bounded correction decision pending
Review cleared / fast-forward pending
Merged / Design acceptance pending
Merged / Closed
Stopped / new Design authority required
```

Implementation creates one clean fixed candidate and sends it unchanged to
Validation. The ordinary finite loop permits at most:

```text
Implementation -> Validation candidates: 3
Validation -> Implementation returns:     3
```

Each corrected candidate is a clean direct child of the returned candidate.
Validation dispatches only one exact cleared commit to Review.

Review performs an independent fixed-commit audit. It may return one complete
in-scope finding packet to Design. Design decides whether one bounded
supplemental correction is justified; no supplemental loop is implicit.

Review alone may fast-forward clean unambiguous `main` with
`git merge --ff-only` after exact clearance. There is no merge commit, squash,
rebase, amend, force operation, or push.

Final Design performs an evidence-only audit and synchronizes only the
authorized lifecycle/evidence portions of this work order and the
implementation index. Any production, test, dependency, or architecture
finding returns to a new fixed candidate rather than being repaired in
closeout.

## Completion Boundary

Maintenance 5 is complete only when:

- the exact dependency pin is published TensorCore `0.13.0` commit
  `202d8b1bc6259b8453d3d377570417f2480d782b`;
- all three semantic axes implement the exact compact representation contract;
- `SamplingConfig`, its module, exports, config field, and compatibility
  surfaces are absent;
- one source-derived `SamplingRuntime` supplies every temporal preparer;
- complete readout input still requires sample start zero before effects;
- public API/export and golden-path lookup evidence is exact;
- scientific/RNG/result continuity is proved;
- exact source/archive local, typing, and separate Validation/Review A100
  evidence passes;
- changed paths stay within the exact allowlists;
- protected history/governance/parity classifications remain unchanged;
- Review clears and fast-forwards the unchanged candidate;
- final Design accepts the merged bytes and records exact evidence; and
- the final worktree is clean.

Closure authorizes no Pint work, IO, integration, renderer, Stage 8 restart,
optimization, release, or push.
