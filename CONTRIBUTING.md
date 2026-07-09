# Contributing

TensorDSLab should be developed like professional scientific software: clear
ownership boundaries, typed public APIs, deterministic behavior, focused tests,
small coherent changes, and documentation that explains contracts rather than
narrating code.

## Repository Identity

TensorDSLab is a clean-slate, tensor-native detector data-lab package. It sits
between g4ds11 detector simulation and future consumers:

```text
g4ds11 -> TensorDSLab -> future consumers
```

TensorDSLab owns detector data-lab products and future cache contracts:

- loading accepted g4ds11 detector-simulation output into typed source records;
- building typed detector examples and detector products;
- building typed readout examples and readout products;
- defining future reconstruction examples and reconstruction products;
- rendering domain products into TensorCore-backed tensor records where a stage
  accepts a tensor-native contract;
- writing, validating, loading, and compacting strict durable caches only after
  in-memory product contracts are stable;
- exposing DAG-compatible executables, operation specs, or recipe fragments
  only after local product and cache contracts are stable.

TensorDSLab does not own generic TensorCore primitives, downstream source
adaptation for model training, model assembly, training loops, evaluation
loops, metrics, checkpoints, campaign orchestration, scheduler behavior,
repair, retries, or g4ds11 simulation execution.

Historical predecessor code, if consulted outside this repository, is
parts-bin material only. Promote scientific facts, product semantics, cache
guarantees, algorithms, fixtures, and tests deliberately into TensorDSLab docs
and tests. Do not preserve old package layouts, helper frameworks,
compatibility shims, DAG wiring, or representation shortcuts by default.

## Build Philosophy

Define the MVP early, but build toward it from the inside out. The first
accepted MVP direction is the post-binned tensor-native readout path:
already-binned charge, stochastic charge transforms, waveform products,
physical waveform composition, and optional digitization.

Source PE-hit parsing, detector-window construction, charge binning, IO
boundaries, durable cache formats, table/array codecs, manifest rules,
compaction, package-local CLIs, DAG-compatible operation specs, recipes,
executable doors, and downstream adapter contracts should not shape the first
post-binned readout module boundaries.

Early implementation stages should be judged by whether the local product graph
is typed, deterministic, testable, and easy to reason about. Compatibility with
external orchestration or downstream training packages is deferred until the
local TensorDSLab contracts are stable.

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
  tensor_dslab/               # when production package code is accepted
    domain/
      common/                 # shared IDs, quantities, validation only when real
      g4ds11/                 # source boundary, only when real
      detector/               # detector examples and products
      readout/                # readout examples and products
      reconstruction/         # future reconstruction products
      caches/                 # durable/cache/load/write bridge, if not domain-local
    executables/              # future DAG/task adapters, only when accepted
    operations/               # future DAG operation specs, only when accepted
    recipes/                  # future reusable composition fragments, only when accepted
  tests/
```

The project/display folder is `TensorDSLab`; the Python import package should
be `tensor_dslab`. Do not create a flat TitleCase Python package that imports
as `TensorDSLab`.

Do not create placeholder modules to reserve architecture. Add a module only
when there is a real concept, behavior, or contract to house.

## TensorCore Backbone

TensorDSLab should use TensorCore for generic tensor mechanics, not fork or
mirror them. Code that needs generic tensor identity, layout, field,
collection, selection, batching, movement, validation, or pure operation
helpers should import those surfaces from `tensor_core`.

TensorCore owns:

- `Id`, `TensorAxisId`, and `TensorFieldId`;
- `IdSequence` for ordered same-type ID values;
- `PositiveInteger` for tensor-local positive counts such as batch size or
  axis size;
- `TensorAxis`, `TensorAxes`, `TensorLayout`, `TensorField`, and
  `TensorCollection`;
- tensor selection records such as `TensorFieldSelection` and
  `TensorAxisSelection`;
- generic tensor builders, validators, mapping helpers, batching helpers,
  movement helpers, and pure tensor operations.

TensorDSLab owns domain IDs and records such as row identity, source
provenance, detector examples, readout examples, reconstruction examples,
product labels, domain configs, and domain-specific tensor rendering rules.
Future cache manifests belong here only after the in-memory product model is
accepted. TensorDSLab domain IDs may appear as TensorCore coordinates, but they
should not become TensorCore-owned primitives.

TensorCore is the dense tensor spine. TensorDSLab should give TensorCore
records detector/readout product meaning instead of competing with the generic
tensor substrate. Scripts and runtime builders may choose the concrete
TensorCore layout shape, but TensorDSLab must make product roles, field roles,
semantic axis roles, sample metadata, and stochastic coordinate inputs
explicit.

Concrete product wrapper classes are optional. Do not add a generic `Product`
base, ToyProduct-like example pattern, or wrapper hierarchy unless a focused
Design stage explains why that shape is better for TensorDSLab than direct
product functions or smaller role records.

### Coordinates, Indices, And Layouts

TensorCore terminology is strict:

- a coordinate is a stable `Id` value associated with an ID-backed axis;
- an index is a zero-based integer tensor position along an axis;
- a layout is ordered axes plus coordinate-to-index maps for ID-backed axes.

Coordinates and indices are not interchangeable. Do not use IDs as array or
tensor positions, and do not persist transient tensor, table, or array indices
as durable identity. Diagnostics, artifacts, and reports should keep reporting
semantic IDs when an axis is ID-backed.

Axis records describe the dimensions of a compiled layout. A `*Axis` describes
one dimension; a `*Axes` record describes the ordered collection of axes that
define the full tensor/layout shape. All tensor dimensions should be explicit
in an axes object, including numeric/bin dimensions such as time samples.

ID-backed axes use ordered ID sequences. IDs identify points in the abstract
space defined by their axis; IDs should not encode ranges, bin edges, units, or
physical interpretation. Normalize external quantities in TensorDSLab boundary
configs or builders before constructing IDs.

`TensorAxes` is the ordering source of truth. `TensorLayout` owns those ordered
axes and carries coordinate maps only for ID-backed axes. Count-only axes do
not need layout map entries because integer positions are already the native
position. A count-only axis is strictly zero-based and continuous.

## Product Semantics

TensorDSLab should preserve the typed product chain unless Design accepts a
focused change:

```text
g4ds11 native output
  -> DetectorExample
  -> ReadoutExample
  -> ReconstructionExample
  -> future consumer-facing tensor/product views
```

This sequence is a dependency rule, not a scheduling policy. Projects/dag may
fan out, retry, cache, stream, compact, or parallelize work, but TensorDSLab
should not hide upstream loading inside downstream builders.

The domain-to-domain boundary is the example object, not a loose product tuple.
Source event IDs, such as g4ds11 event IDs, are provenance and should not be
used as row identity unless a stage explicitly accepts that policy.

Producer product labels are durable TensorDSLab product labels. They are not
automatically TensorCore `TensorFieldId` values. Keep the namespaces explicit:

- `detector.pe_hits`;
- `readout.charge`;
- `readout.waveform.pure`;
- `readout.waveform.noise`;
- `readout.waveform.physical`;
- future reconstruction labels.

Consumer-facing adapters are deferred. TensorDSLab should first make the local
typed product graph coherent enough that future consumers can depend on it
without parsing raw `.fil`, table, array, manifest, or private representation
details.

## Domain Organization

Domain code should communicate through typed in-memory objects. Persistence,
caches, artifacts, tables, arrays, tensors, executables, operations, and
recipes are bridges around the domain model, not replacements for it.

Use these module names when they fit real behavior:

- `types.py` owns stable public records, type aliases, and domain value
  objects.
- `configs.py` owns public configuration records when configuration is
  nontrivial enough to deserve a boundary.
- `builders.py` owns in-memory construction when construction is meaningfully
  separate from loading, writing, or representation conversion.
- `validation.py` owns domain invariants, validation reports, issue codes, and
  validation errors.
- `artifacts.py` owns the durable/cache/load/write bridge for the domain.
- `tables.py` owns table schemas, row conversion, table conversion, and
  table-level parsing helpers when table representation exists.
- `arrays.py` owns array conversion, axes, shapes, and payload rendering when
  array representation exists.
- `tensors.py` owns TensorCore rendering when tensors exist as a public product
  or backend representation.
- `exports.py` may be added when a project has a real external export/catalog
  surface, such as DAG operation discovery, cache publication, or stable
  adapter metadata.

Do not use `exports.py` as a dumping ground for ordinary package re-exports.
Public imports still belong in deliberate package `__init__.py` surfaces.

`artifacts.py` should be readable as the domain bridge contract. If it becomes
mostly schema literals, row codecs, array plumbing, tensor plumbing, or backend
mechanics, split those mechanics into `tables.py`, `arrays.py`, or
`tensors.py` as appropriate.

Do not create placeholder `configs.py`, `builders.py`, `artifacts.py`,
`tables.py`, `arrays.py`, `tensors.py`, or `exports.py` modules. Split or add a
module only when the behavior is real enough to make the boundary useful.

## Domain Transform Surfaces

TensorDSLab domain surfaces may expose functions or instance methods when the
operation is intrinsic to that product. Post-binned readout operations such as
timing jitter, dark counts, crosstalk, afterpulses, charge smearing, waveform
rendering, physical waveform composition, and digitization are product
semantics, not generic TensorCore operations.

Domain transforms should use explicit output buffers:

```python
new_charge = time_jitter(charge, jitter)
time_jitter(charge, jitter, out=charge)

scratch = empty_like(charge)
afterpulses(charge, afterpulse, out=scratch)
dark_counts(scratch, dark, out=charge)

pure = render_pure_waveform(charge, pulse)
render_pure_waveform(charge, pulse, out=pure)
```

If `out` is omitted, the method allocates and returns a new product. If `out`
is supplied, the method writes into `out` and returns `out`. `out` must be the
correct TensorDSLab product type with compatible TensorCore layout, device,
dtype, semantic axis roles, and product meaning.

Do not use a persistent ambient mutation mode for core transforms. Do not
encode mutation or allocation policy as a TensorCore `Id`, `TensorFieldId`,
`TensorAxisId`, coordinate, or product label. This is runtime output policy,
not tensor identity.

## Common Code

`common/` should stay dependency-light and semantic. Good candidates include
shared IDs, small value objects, shared exceptions, and validation primitives
that are used by multiple real domains.

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

## Deferred Integration Surfaces

Projects/dag owns campaign orchestration, sharding, scheduling, concrete DAG
construction, execution policy, repair, retries, status, and fanout/fanin.
TensorDSLab may later expose stable public surfaces for operation specs,
executable adapters, artifact/cache requirements, output validation, and
recipe fragments.

Use these optional package directories only when the project needs them:

- `operations/` for DAG-compatible operation specs;
- `recipes/` for reusable composition fragments;
- `executables/` for CLI, DAG, or task adapters.

Do not add DAG-compatible modules, downstream adapters, package-local workflow
CLIs, or cache-driven integration surfaces before local TensorDSLab contracts
are accepted. Keep core domain code dependency-light and orchestration-free.

## Parts-Bin Rule

Historical predecessor code is donor material only. Reuse scientific facts,
small algorithms, naming lessons, fixtures, tests, and accepted cache semantics
after review. Do not preserve old package layouts, helper frameworks, local DAG
mechanics, compatibility wrappers, or representation shortcuts by default.

When promoting donor code or behavior:

- write down the accepted reason in the relevant implementation or decision
  doc;
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

- source adapters and native g4ds11 input readers;
- user configs;
- construction of public ID objects;
- construction of constrained scalar wrappers for meaningful numeric config or
  artifact values;
- construction of detector, readout, reconstruction, cache, table, array, and
  tensor records;
- construction of TensorCore axes, layouts, fields, collections, and
  selections.

Once an object has crossed into a valid native record, hot-path functions
should avoid repeatedly revalidating full object graphs. Product builders and
tensor renderers may still perform narrow function-specific checks, but they
should not rediscover layout validity, identity validity, or mapping
immutability every inner-loop call.

Use scalar wrappers at config, source, and artifact boundaries where
constraints are meaningful. Numeric wrappers should reject bool. Plain `bool`
is appropriate for boolean fields.

## Code Expectations

- Use a short module context docstring when ownership or boundary is not
  obvious from the module path and public types.
- Type public functions, methods, dataclass fields, and module constants.
- Avoid `Any`, unbounded `dict`, and stringly typed public interfaces unless
  the boundary is intentionally JSON-like or there is a documented reason.
- Prefer dataclasses for stable records.
- Use `value` for primitive payloads on ID and scalar-wrapper records.
- Prefer frozen runtime wrapper classes for IDs used on tensor-facing hot
  paths.
- Keep modules cohesive; split a module when it owns more than one meaningful
  boundary.
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
- TensorCore-backed tensor axes, layouts, fields, selections, or shapes;
- product ownership;
- cache files, manifest shape, durable guarantees, or compaction rules;
- validation rules;
- operation, recipe, or executable surfaces;
- implementation stages or accepted decisions.

Keep the relevant source of truth synchronized:

- `docs/implementation/...` for stage work orders, scope, public surfaces,
  invariants, and non-goals;
- `docs/architecture/<domain>.md` for public domain contracts, cache shapes,
  builders, validation boundaries, and representation bridges;
- `docs/architecture/common.md` for shared primitives and cross-domain rules;
- `docs/design.md` for end-to-end domain flow and ownership boundaries;
- `docs/decisions.md` for accepted, renamed, superseded, or explicitly
  deferred semantic choices;
- `docs/validation.md` for validation cases, fixtures, failure modes, and
  tolerances;
- `README.md`, `AGENTS.md`, or `CONTRIBUTING.md` for workflow, onboarding, or
  repository-wide engineering expectations.

## Before Review

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
