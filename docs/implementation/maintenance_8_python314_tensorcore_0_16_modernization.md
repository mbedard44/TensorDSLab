# Maintenance 8 Python 3.14 And TensorCore 0.16 Modernization

Status: **Provisional Design / Undispatched / TensorCore 0.16 publication
pending**.

This document preserves the accepted TensorDSLab Maintenance 8 direction. It
is not yet an operative production work order, fixed dependency selection,
implementation authority, compatibility claim, cluster authority, release, or
push authorization. Design will freeze the exact TensorCore commit, baseline,
allowlist, evidence commands, artifact identities, loop budget, and route
disposition only after TensorCore publishes its final `0.16.0` bytes.

Stable provisional key:

```text
TensorDSLab/maintenance-8-python314-tensorcore-0-16-modernization
```

## Purpose

Modernize TensorDSLab once, coherently, before the package's first push:

1. adopt the exact published TensorCore `0.16.0` dependency;
2. select the current-stable Python, Torch, NumPy, Pint, build, and typing
   stack;
3. migrate precise validation imports to TensorCore's domain-owned modules;
4. use the Python 3.14 annotation model and bounded PEP 695 syntax;
5. standardize descriptive type-parameter names without leading underscores;
6. add intentional module and supported-public-symbol docstrings; and
7. preserve every accepted TensorDSLab public, Pint, scientific, RNG,
   preparation, production, validation, and result contract.

This is a dependency, syntax, documentation, and packaging modernization. It
must not become an opportunistic scientific or architectural rewrite.

## Accepted Starting Point And Prerequisites

The current TensorDSLab local baseline is Maintenance 7's exact corrected
Design closeout:

```text
commit: 5ff9620ae2538778efadf3b3ad5345ba3548ac5e
tree:   9575333cf9703e4f7ce6872844616294cedbae79
```

Maintenance 7 is locally Merged / Closed, adopts exact published TensorCore
`0.15.0`, and remains unpushed. Its complete production, tests, dependency,
and evidence history must remain intact.

Maintenance 8 may become operative only after all of these are true:

1. TensorCore closes and publishes the final accepted `0.16.0` commit on its
   live GitHub `main`;
2. TensorCore returns the exact commit, tree, version, metadata, package
   topology, export surfaces, source/archive identities, and residual
   qualifications;
3. the published bytes include the accepted domain-owned validation topology,
   Python 3.14/Torch 2.13 modernization, intentional docstrings, and
   descriptive PEP 695 type-parameter naming;
4. TensorDSLab Design independently audits that exact dependency against the
   Maintenance 7 baseline;
5. Design converts this provisional record into a fixed, self-consistent work
   order with exact path and evidence boundaries; and
6. the user explicitly authorizes dispatch after the fixed work order is
   committed.

No unpublished TensorCore local commit is a dependency target. TensorDSLab
must not pin a commit until the corresponding live remote ref is independently
verified.

## Exact Intended Environment

The intended fixed Maintenance 8 environment is:

```text
CPython:   3.14.6
PyTorch:   2.13.0
NumPy:     2.5.1
Pint:      0.25.3
Hatchling: 1.31.0
Pyright:   1.1.411
```

The intended metadata changes are:

```toml
[build-system]
requires = ["hatchling==1.31.0"]
build-backend = "hatchling.build"

[project]
requires-python = ">=3.14"
dependencies = [
    "numpy==2.5.1",
    "pint==0.25.3",
    "torch>=2.13,<2.14",
    "tensor-core @ git+https://github.com/mbedard44/TensorCore.git@<exact-published-0.16-commit>",
]
```

The final work order must replace the placeholder with the exact published
TensorCore `0.16.0` commit. It must freeze the accepted source and artifact
identities for TensorCore, Pint, NumPy, TensorDSLab, and the build frontend.

TensorDSLab currently has no package-owned CI workflow. Maintenance 8 must not
create placeholder CI merely to mirror TensorCore. If Design later accepts a
real TensorDSLab CI surface, that requires a separate explicit scope decision.

## TensorCore 0.16 Adoption

TensorCore `0.16.0` deliberately removes both the central
`tensor_core.validation` package and every requirement re-export from the
curated package root. The root remains canonical only for semantic classes and
types:

```python
from tensor_core import (
    CounterRng,
    FiniteFloat,
    PositiveFloat,
    RngKey,
    RngPositions,
    Scalar,
    TensorField,
)
```

All retained requirements use their precise owning domains. Maintenance 8
must migrate the exact imports:

```python
from tensor_core.tensor.validation import (
    require_field_dtype,
    require_field_layout,
    require_field_types,
    require_representable_float,
    require_same_axes,
    require_same_device,
    require_same_dtype,
    require_shape_span,
    require_tensor_allocation,
)
from tensor_core.random.validation import require_count_tensor
```

The seven retained requirements currently imported from the TensorCore root
occur `27` times in the Maintenance 7 source: `20` production imports and `7`
test imports. Every occurrence of `require_field_dtype`,
`require_field_layout`, `require_field_types`,
`require_representable_float`, `require_same_axes`, `require_same_device`, and
`require_same_dtype` must move to `tensor_core.tensor.validation`.

The corresponding root and retired-package imports must disappear without
aliases or local forwarders:

```python
from tensor_core import require_field_dtype, require_same_dtype
from tensor_core.validation import require_shape_span
from tensor_core.validation.random import require_count_tensor
```

The final dependency audit must verify these exact public surfaces:

- exact `21`-name semantic-class/type `tensor_core` root;
- exact `7`-name `tensor_core.scalar` surface;
- exact `3`-name `tensor_core.scalar.validation` surface;
- exact `7`-name `tensor_core.tensor` surface;
- exact `15`-name `tensor_core.tensor.validation` surface;
- exact `3`-name `tensor_core.table` surface;
- exact `4`-name `tensor_core.random` surface;
- exact one-name `tensor_core.random.validation` surface;
- exact `26`-file domain-owned package topology; and
- no `tensor_core.validation` compatibility package, root requirement alias,
  or forwarding assignment.

These counts are the accepted consumer boundary of the pre-publication
TensorCore Design candidate. The dependency pin remains unset until the exact
same contract is implemented, closed, published, and independently verified
on live `origin/main`.

TensorCore `0.16.0` also adds public
`tensor_core.tensor.validation.require_index(...)` for strict nonnegative
indices used internally by axes, `TensorField.axis_at()`, and
`RngPositions.select()`. Maintenance 8 must prove the dependency surface and
preserved strict-index behavior, but TensorDSLab production does not import or
call `require_index()` directly.

The free `require_positive_integer()` and `require_nonnegative_integer()`
functions are absent from the final scalar-validation surface. TensorDSLab
does not consume them. `PositiveInteger` and `NonnegativeInteger` remain
semantic root imports and preserve their exact construction, `require()`,
`accepts()`, normalization, and diagnostic behavior.

Maintenance 8 adopts these exact import/export and additive-index changes; it
does not alter them. Every retained validator preserves its accepted
signature, candidate domain, return, exception category, validation order,
allocation/no-effect promise, range law, reduction, host-extraction, and
synchronization boundary.

## Validation Ownership Preserved

TensorCore continues to own reusable generic mechanics, including:

- primitive scalar requirements;
- Torch float representability;
- exact field dtype and layout requirements;
- structural tensor, shape, device, and dimension requirements;
- address/layout span and ordinary allocation preflight;
- the fixed CounterRng count-tensor domain; and
- generic axis, field, collection, table, and RNG representation invariants.

TensorDSLab continues to own:

- Pint recognition, registry ownership, canonical unit conversion, and
  defensive canonical quantity copies;
- readout-axis composition and complete readout structure;
- Config physical and cross-field relationships;
- exact product closure and public readout preparation;
- fixed readout RNG role policy and package-owned keys;
- Charge scientific count ledgers and checked accumulation;
- pulse/noise/ADC scientific preparation;
- product-specific `validate_<product>` postconditions;
- axes identity, device/dtype relationships, freshness, and nonaliasing; and
- the topological `simulate_readout(...)` execution contract.

Maintenance 8 must not revive local copies of generic TensorCore requirements
or move TensorDSLab scientific requirements upstream merely because both are
expressed as validation functions.

## Python 3.14 Syntax Modernization

The syntax migration is mechanical and contract-preserving.

### PEP 695 type parameters

Use scoped PEP 695 parameters for generic classes and functions. Every type
parameter must have a descriptive semantic name without a leading underscore.
Do not use a bare `T` when the role has a useful name.

Accepted naming examples:

```python
class Scalar[ScalarT: (int, float)]:
    ...


class TensorAxis[CoordinateT]:
    ...


class TableColumn[DataT]:
    values: DataT


def column[ColumnT: TableColumn[Any]](
    self,
    column_type: type[ColumnT],
) -> ColumnT:
    ...


def make_product[FieldT: TensorField](
    field_type: type[FieldT],
) -> FieldT:
    return field_type(tensor=tensor, axes=axes)
```

The names distinguish different roles:

- `DataT` is the stored payload type selected by `TableColumn`;
- `ColumnT` is one concrete semantic `TableColumn` subtype selected by exact
  lookup;
- `CoordinateT` is one axis coordinate type;
- `AxisT` is one concrete semantic axis subtype;
- `FieldT` is one concrete semantic field subtype; and
- `ScalarT` is the normalized primitive owned by one Scalar specialization.

The `Scalar`, `TensorAxis`, and table examples are TensorCore dependency-side
prerequisites, not TensorDSLab production edits. Maintenance 8 verifies them
against the exact published dependency and applies the same convention to
TensorDSLab's own scoped parameters.

`ColumnDataT` is unnecessarily repetitive in the scope
`TableColumn[ColumnDataT]`; `DataT` is preferred there. `ColumnT` must not be
used for the payload because it denotes the column object type at exact-type
lookup boundaries.

TensorDSLab's current active legacy generic:

```python
FieldT = TypeVar("FieldT", bound=TensorField)


def make_product(
    field_type: type[FieldT],
) -> FieldT:
    return field_type(tensor=tensor, axes=axes)
```

becomes the scoped `make_product[FieldT: TensorField](...) -> FieldT` form.
The standalone `TypeVar` import and declaration are removed.

### Private type aliases

A type parameter and a module-level type alias have different privacy rules.
The quantity-field table descriptor is a private implementation detail, so its
leading underscore remains correct and the alias may use Python 3.14 syntax:

```python
type _QuantityField = tuple[
    str,
    str,
    type[Scalar[float]],
]
```

`_QuantityField` is private because the alias itself lives in module scope and
must not be imported as supported API. By contrast, `FieldT`, `DataT`, and
`ColumnT` are lexically scoped parameters and do not leak a new module
attribute; a privacy underscore adds noise rather than useful information.

The final static audit must:

- find no active `typing.TypeVar` or `typing.Generic` scaffolding where PEP
  695 directly expresses the existing contract;
- find no leading underscore on a PEP 695 class/function type parameter;
- retain a leading underscore on private module-level aliases;
- preserve all generic bounds and return narrowing; and
- avoid creating or widening any generic surface merely to use new syntax.

Type-parameter spelling improves source readability. Exact raw
`__annotations__` objects, annotation evaluation internals, type-parameter
runtime representations, and qualified definition provenance are not
compatibility promises.

### Annotation model

Remove blanket:

```python
from __future__ import annotations
```

from active production, test, and typing-fixture modules where Python 3.14's
annotation model makes it unnecessary. Do not preserve the import
ceremonially.

All supported annotations must still resolve through
`typing.get_type_hints(...)` under Python 3.14. `TYPE_CHECKING` must not hide a
runtime name required to resolve a supported signature.

### Overrides

Add `typing.override` to real inherited implementations where it clarifies the
accepted extension contract, including:

- TensorDSLab semantic-axis `_require()` hooks;
- TensorDSLab product-field and collection `_require()` hooks; and
- concrete test doubles that implement accepted abstract TensorCore methods.

Do not add `@override` to unrelated same-named functions or use it to imply a
new subclass relationship.

## Docstring Contract

Maintenance 8 standardizes intentional documentation in the production
package.

Every tracked production Python module receives a concise triple-quoted module
docstring describing its ownership or semantic role. This includes private
runtime packages and intentionally empty `__init__.py` files; their docstrings
should explain the boundary rather than merely repeat the path.

Every supported public class or function exported through:

- `tensor_dslab`;
- `tensor_dslab.common`;
- `tensor_dslab.readout`; or
- an accepted product facade

receives its own concise semantic docstring. A class must own its docstring;
inherited ABC text or dataclass-generated text does not satisfy this rule.

Non-obvious public operations and important abstract, effect, validation,
synchronization, representation, or numerical boundaries receive concise
contract docstrings. Private helpers receive docstrings only when they own a
non-obvious invariant. Maintenance 8 must not pad obvious private functions
with repetitive narration or freeze exact prose as a numerical/API
compatibility surface.

Required evidence must check:

```python
assert type_.__dict__.get("__doc__")
assert function.__doc__
```

It must not accept inherited class documentation as proof.

## Preserved Config And Pint Model

Maintenance 8 preserves Maintenance 6 and 7's Config model exactly:

- public physical values remain canonical copied Pint quantities;
- scalar and vector quantity tables remain separate;
- optional fields are canonicalized only when non-`None`;
- vector Config state remains one array-backed Pint Quantity;
- preparation extracts vector magnitudes into plain tuples exactly once;
- Runtime records, producers, validators, tensors, and collections remain
  Pint- and NumPy-free;
- positive pulse amplitude magnitudes remain positive public quantities;
- preparation applies fixed DS20k negative polarity exactly once; and
- dtype-rounded-zero guards and calibrated waveform continuity remain.

The private `_QuantityField` alias may adopt the new `type` statement, but its
structure and the canonicalization behavior remain unchanged.

Maintenance 8 adds no Config ABC, serialization root, profile loader,
reflection registry, builder framework, or replacement canonicalization
layer.

## Preserved RNG And Scientific Contracts

Maintenance 8 preserves:

- package-owned namespace `0x54445331` and all ten fixed role keys;
- `CounterRng.seed` as the public realization selector;
- exact `RngPositions` construction and transform plans;
- role stream/address values, key values, positional coordinates, word
  schedules, traversal order, and global-state isolation;
- established distribution, jitter, delay, cascade, smearing, noise, pulse,
  analog, and digitization laws;
- exact same-stack replay and the environment-qualified stochastic-continuity
  policy; and
- all product axes, dtype, device, freshness, and storage relationships.

There is no namespace cleanup, new role, RNG algorithm, distribution, probing,
stateful generator, scientific recalibration, or performance rewrite in
Maintenance 8.

## Expected Change Families

The final fixed work order should bound changes to these families:

1. dependency and tool metadata;
2. TensorCore precise-import migration;
3. mechanical Python 3.14 syntax and typing modernization;
4. intentional production docstrings;
5. tests and typing fixtures required to prove the preserved contracts;
6. synchronized current architecture, API, validation, overview, contribution,
   and implementation records; and
7. exact lifecycle/evidence updates.

The final allowlist must be derived from the exact published TensorCore
handoff and the exact Maintenance 7 baseline. This provisional record does not
authorize every repository file to change.

## Required Evidence

The fixed work order should require, at minimum:

### Dependency and environment

- exact Python `3.14.6`;
- exact PyTorch `2.13.0` and accepted metadata line;
- exact NumPy `2.5.1`;
- exact Pint `0.25.3`;
- exact Hatchling `1.31.0`;
- exact Pyright `1.1.411`;
- exact published TensorCore `0.16.0` commit/tree/version;
- exact source/archive/wheel identities where applicable; and
- clean source/archive package-byte comparison.

### Functional preservation

- complete focused and full TensorDSLab source/archive suites;
- complete TensorCore dependency suite in the accepted local environment;
- exact `21/7/3/7/15/3/4/1` dependency export tuples and import-isolation
  checks;
- proof that all `13` former requirement re-exports are absent from the
  TensorCore root, the `11` retained requirements are available only from
  their accepted domain modules, and the two sign-specific free requirements
  are absent completely;
- proof that `require_index()` is present in
  `tensor_core.tensor.validation`, absent from the TensorCore root, and
  preserves strict nonnegative axis/field/RngPositions index behavior;
- proof that TensorDSLab production does not import `require_index()`
  directly;
- proof that the seven retained TensorDSLab requirements use only
  `tensor_core.tensor.validation`;
- proof that the removed free positive/nonnegative integer requirements are
  absent while the two Scalar leaves preserve exact behavior;
- Config scalar/vector quantity normalization and registry-copy evidence;
- prepared Runtime and producer Pint/NumPy privacy;
- exact `RngPositions`, namespace/key, address, and no-global-RNG evidence;
- full readout orchestration and product-validator evidence; and
- all accepted environment-qualified stochastic continuity checks.

### Typing and syntax

- Pyright `1.1.411` with `pythonVersion` `3.14`, zero positive diagnostics;
- a frozen negative typing fixture with exact intended diagnostics;
- positive narrowing for semantic axes, fields, collections, Configs,
  Runtimes, and `make_product[FieldT]`;
- no active legacy `TypeVar`/`Generic` scaffolding where PEP 695 applies;
- no underscored PEP 695 parameters;
- preserved private module-level aliases;
- resolved `typing.get_type_hints(...)` for supported signatures;
- correct `@override` use; and
- no raw annotation-representation compatibility assertion.

### Documentation and package shape

- every production module has an intentional module docstring;
- every supported exported class/function has its own nonempty docstring;
- class checks use `__dict__.get("__doc__")`;
- all Markdown fences parse and executable examples run;
- all relative links resolve;
- retired TensorCore validation imports are absent;
- no compatibility alias, placeholder module, generated cache, bytecode,
  build, dist, or egg-info artifact remains; and
- final branch, topology, scope, protected bytes, and worktree cleanliness
  pass.

The final work order must freeze exact test totals only after the exact
dependency and implementation bytes exist. This provisional record does not
invent future totals.

## CUDA And First-Push Sequence

Maintenance 8's package loop closes locally with complete CPU, typing,
artifact, documentation, import, and hygiene evidence. It makes no fresh
accelerator claim.

After exact TensorCore `0.16.0` and exact TensorDSLab Maintenance 8 are both
locally closed:

1. freeze the exact integrated TensorCore/TensorDSLab pairing;
2. issue separate explicit cluster authority;
3. run package-owned full-suite CUDA matrices for TensorCore and TensorDSLab
   against that same pairing and the accepted current Torch line;
4. record separate package dispositions and any environment qualifications;
5. resolve every real finding through the owning package; and
6. only then consider TensorDSLab's first push.

The later CUDA work is functional compatibility evidence, not a performance,
deployment, release-readiness, compilation, or broad-backend claim. No old
TensorCore/TensorDSLab pairing may substitute for the exact integrated
baseline.

## Explicit Non-Goals

Maintenance 8 does not add or change:

- DS20k profiles or demos;
- IO, TensorArtifact adoption, persistence, caches, or serialization;
- TensorG4DS or TensorML adapters;
- product, Config, Runtime, producer, validator, or public orchestration
  architecture;
- public exports or package version unless separately justified and frozen;
- Pint units, registry policy, Config fields, or physical defaults;
- RNG namespaces, role keys, addresses, schedules, algorithms, or numerical
  laws;
- table products or downstream table adoption;
- axes, field, or collection semantics;
- scientific models, coefficients, approximations, thresholds, or parity;
- free-threaded Python support;
- compile/fusion/performance work;
- compatibility shims for retired TensorCore precise imports; or
- a push, tag, package-index release, deployment, or broad compatibility
  claim.

The provisional DS20k Veto profile and demos remain a separate later stage
under
[Provisional DS20k Veto Profile And Public Readout Demos](proposed_ds20k_veto_profile_and_public_readout_demos.md).

## Promotion Checklist

Before dispatch, Design must replace this provisional status with a
self-consistent fixed work order that:

1. names the exact published TensorCore `0.16.0` commit and artifacts;
2. names the exact TensorDSLab authority commit and tree;
3. freezes the exact changed-path allowlist and protected bytes;
4. freezes the exact source/archive/wheel and environment commands;
5. freezes the exact public/module docstring census;
6. freezes the exact typing fixtures and expected diagnostics;
7. freezes the Implementation/Validation/Review loop and candidate budget;
8. uses merge-safe lifecycle wording;
9. keeps CUDA and push authority separate; and
10. obtains explicit user dispatch.

If the published TensorCore handoff contradicts any dependency, import,
typing, docstring, runtime, scientific, or evidence assumption here, stop and
return the conflict to TensorDSLab and TensorCore Design before dispatch.
