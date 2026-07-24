# Maintenance 8 Python 3.14 And TensorCore 0.16 Modernization

Status: **Design-complete / User dispatch pending**.

This is the fixed TensorDSLab Maintenance 8 production work order. It binds
the exact published TensorCore `0.16.0` dependency, exact locally closed
TensorDSLab Maintenance 7 production baseline, complete changed-path
allowlist, protected bytes, evidence matrix, loop budget, and merge authority.
It is not Implementation dispatch until the user explicitly authorizes the
route. It grants no cluster, push, release, deployment, or broad compatibility
authority.

Stable key:

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

## Accepted Starting Point And Dependency Authority

The current TensorDSLab local baseline is Maintenance 7's exact corrected
Design closeout:

```text
commit: 5ff9620ae2538778efadf3b3ad5345ba3548ac5e
tree:   9575333cf9703e4f7ce6872844616294cedbae79
```

Maintenance 7 is locally Merged / Closed, adopts exact published TensorCore
`0.15.0`, and remains unpushed. Its complete production, tests, dependency,
and evidence history must remain intact.

Those publication prerequisites are now satisfied. The exact dependency
authority is:

```text
repository:              https://github.com/mbedard44/TensorCore.git
live refs/heads/main:    e05324699892a8bcea024375720bfae1ed9569cc
publication tree:        0414a99ac6096035213479e195a0b095d4b1b12e
publication parent:      7588ccb718aba1cd2b3e3456bb5eb09d1fbc592e
package version:         0.16.0
implementation anchor:   0c1475258e7ede60f1b607db8017dc13a7fafc02
implementation tree:     1147f0523b78e21e7454f8728f6fcacef1db9df7
```

The publication descendants after the implementation anchor are
documentation-only. An exact path comparison proves that `tensor_core/`,
`tests/`, and `pyproject.toml` are byte-identical between the implementation
anchor and published containing commit.

TensorDSLab Design independently verified the live GitHub branch at exact
`e053246...`, the clean synchronized local TensorCore checkout, version
`0.16.0`, Python/Torch metadata, ordered `21`-name root API, exact `26` package
files, and the accepted domain-owned validation modules. A canonical no-prefix
Git ZIP created from exact `e053246...` is `476239` bytes with SHA-256
`f9267b74fec35a57591cbf7b2fe2ec28cb9023442080aa7b21a8ab31a38cd6c8`.
TensorCore's independently accepted build evidence records wheel SHA-256
`51df309d3389c2ea4336d676e2e05a3bfe4585e2f65bd699d79acfe9265761f5`
and source-archive SHA-256
`a22870ad38de3b6792c6e9929e1e01ca1e29ca72923857a76d8e1c4ae44ec089`.

Implementation and both evidence roles must independently reconstruct the
exact source checkout and canonical Git ZIP. A different commit, tree,
version, package topology, export tuple, archive hash, or live remote ref is a
hard stop. No unpublished TensorCore commit is an accepted dependency target.

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
    "tensor-core @ git+https://github.com/mbedard44/TensorCore.git@e05324699892a8bcea024375720bfae1ed9569cc",
]
```

The exact Python, Torch, NumPy, Pint, Hatchling, and Pyright versions are
required inputs. Each evidence role must record the exact resolved artifact
filenames and SHA-256 values it uses. TensorDSLab candidate wheel and sdist
hashes cannot exist before the candidate; each role must build fresh isolated
artifacts, report their hashes, inspect their metadata, and prove extracted
`tensor_dslab/` package bytes equal the fixed source bytes.

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

These counts are the accepted consumer boundary of exact published commit
`e053246...`. The dependency pin must name that commit and no other.

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

## Exact Implementation Scope

The candidate may change only these families:

1. dependency and tool metadata;
2. TensorCore precise-import migration;
3. mechanical Python 3.14 syntax and typing modernization;
4. intentional production docstrings;
5. tests and typing fixtures required to prove the preserved contracts;
6. synchronized current architecture, API, validation, overview, contribution,
   and implementation records; and
7. exact lifecycle/evidence updates.

Because the docstring contract applies to every production module, all `59`
current production Python modules are deliberately in scope. This is not
blanket repository authority: every changed production hunk must be one of
the exact dependency-import, Python 3.14 syntax, `@override`, or truthful
docstring changes defined here. The Design baseline has one intentional module
docstring and `58` production modules without one; the candidate must reach
`59/59`. The supported public-symbol census is exactly `32` classes and `3`
functions across the unchanged `35/5/30` package/common/readout facades.

### Exact changed-path allowlist

Metadata and type checking:

```text
pyproject.toml
pyrightconfig.json
```

Production:

```text
tensor_dslab/__init__.py
tensor_dslab/common/__init__.py
tensor_dslab/common/axes.py
tensor_dslab/common/units.py
tensor_dslab/readout/__init__.py
tensor_dslab/readout/analog_waveform/__init__.py
tensor_dslab/readout/analog_waveform/config.py
tensor_dslab/readout/analog_waveform/field.py
tensor_dslab/readout/analog_waveform/runtime/__init__.py
tensor_dslab/readout/analog_waveform/runtime/prepare.py
tensor_dslab/readout/analog_waveform/runtime/produce.py
tensor_dslab/readout/analog_waveform/runtime/validate.py
tensor_dslab/readout/charge/__init__.py
tensor_dslab/readout/charge/config.py
tensor_dslab/readout/charge/field.py
tensor_dslab/readout/charge/runtime/__init__.py
tensor_dslab/readout/charge/runtime/effects/__init__.py
tensor_dslab/readout/charge/runtime/effects/correlated_avalanches.py
tensor_dslab/readout/charge/runtime/effects/counts.py
tensor_dslab/readout/charge/runtime/effects/dark_counts.py
tensor_dslab/readout/charge/runtime/effects/delays.py
tensor_dslab/readout/charge/runtime/effects/smearing.py
tensor_dslab/readout/charge/runtime/effects/timing_jitter.py
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/charge/runtime/produce.py
tensor_dslab/readout/charge/runtime/validate.py
tensor_dslab/readout/collection.py
tensor_dslab/readout/config.py
tensor_dslab/readout/digitized_waveform/__init__.py
tensor_dslab/readout/digitized_waveform/config.py
tensor_dslab/readout/digitized_waveform/field.py
tensor_dslab/readout/digitized_waveform/runtime/__init__.py
tensor_dslab/readout/digitized_waveform/runtime/prepare.py
tensor_dslab/readout/digitized_waveform/runtime/produce.py
tensor_dslab/readout/digitized_waveform/runtime/validate.py
tensor_dslab/readout/noise_waveform/__init__.py
tensor_dslab/readout/noise_waveform/config.py
tensor_dslab/readout/noise_waveform/field.py
tensor_dslab/readout/noise_waveform/runtime/__init__.py
tensor_dslab/readout/noise_waveform/runtime/prepare.py
tensor_dslab/readout/noise_waveform/runtime/produce.py
tensor_dslab/readout/noise_waveform/runtime/validate.py
tensor_dslab/readout/photoelectrons/__init__.py
tensor_dslab/readout/photoelectrons/field.py
tensor_dslab/readout/photoelectrons/runtime/__init__.py
tensor_dslab/readout/photoelectrons/runtime/validate.py
tensor_dslab/readout/pure_waveform/__init__.py
tensor_dslab/readout/pure_waveform/config.py
tensor_dslab/readout/pure_waveform/field.py
tensor_dslab/readout/pure_waveform/runtime/__init__.py
tensor_dslab/readout/pure_waveform/runtime/prepare.py
tensor_dslab/readout/pure_waveform/runtime/produce.py
tensor_dslab/readout/pure_waveform/runtime/validate.py
tensor_dslab/readout/runtime/__init__.py
tensor_dslab/readout/runtime/keys.py
tensor_dslab/readout/runtime/prepare.py
tensor_dslab/readout/runtime/requirements.py
tensor_dslab/readout/runtime/sampling.py
tensor_dslab/readout/simulation.py
```

Tests and typing fixtures:

```text
tests/readout_fixtures.py
tests/test_charge_correlated_avalanches.py
tests/test_charge_count_orchestration.py
tests/test_charge_delay_preparation.py
tests/test_charge_product.py
tests/test_charge_timing_jitter.py
tests/test_deterministic_waveform_products.py
tests/test_noise_waveform_product.py
tests/test_package_contracts.py
tests/test_pint_physical_configuration.py
tests/test_readout_axes_and_sampling.py
tests/test_readout_collection.py
tests/test_readout_configs.py
tests/test_readout_product_types.py
tests/test_readout_simulation.py
tests/test_rng_ownership_migration.py
tests/test_runtime_action_ownership.py
tests/test_tensorcore_0_15_adoption.py
tests/test_tensorcore_0_16_modernization.py
tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py
tests/typing/maintenance_4_runtime_action_ownership.py
tests/typing/maintenance_6_pint_physical_configuration_boundary.py
tests/typing/stage_3_semantic_leaf_contracts.py
tests/typing/stage_4_deterministic_waveform_products.py
tests/typing/stage_7_public_readout_orchestration.py
```

`tests/test_tensorcore_0_15_adoption.py` must be cleanly replaced by
`tests/test_tensorcore_0_16_modernization.py`; they may not coexist in the
candidate. `tests/__init__.py` is protected.

Synchronized current records:

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
docs/implementation/maintenance_8_python314_tensorcore_0_16_modernization.md
docs/overview.md
docs/validation.md
```

The maximum candidate scope is therefore `98` logical paths: `2` metadata,
`59` production, `25` test/typing, and `12` current-document paths. A candidate
need not touch an allowlisted test or current record when no truthful change
is required. It must add truthful module docstrings to the exact `58` modules
that lack them; the already-documented package root need not change unless
another authorized mechanical edit requires it. A changed path outside this
list is a hard stop.

### Protected bytes

The following are protected:

- `LICENSE`, `tensor_dslab/py.typed`, `tests/__init__.py`,
  `docs/parity.md`, and every non-allowlisted path;
- every closed historical implementation work order;
- `docs/implementation/proposed_ds20k_veto_profile_and_public_readout_demos.md`;
- all package/facade export tuples and public call signatures except
  annotation spelling mechanically required by Python 3.14;
- all scientific constants, Pint canonical units, RNG namespace/key/address
  values, equations, thresholds, loop order, dtype/device behavior, and
  result relationships; and
- all CUDA, push, release, IO, profile/demo, TensorML, TensorG4DS, artifact,
  cache, serialization, and performance surfaces.

Mechanical edits must not be used to reformat unrelated code. If a required
change falls outside the exact allowlist or contradicts a protected byte,
Implementation stops and returns the exact conflict to Design.

## Required Evidence

The candidate must provide all of the following:

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
  TensorCore root, the `10` TensorDSLab-consumed requirements are available
  only from
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
- the exact docstring census is `59` production modules, `32` supported
  public classes, and `3` supported public functions;
- class checks use `__dict__.get("__doc__")`;
- all Markdown fences parse and executable examples run;
- all relative links resolve;
- retired TensorCore validation imports are absent;
- no compatibility alias, placeholder module, generated cache, bytecode,
  build, dist, or egg-info artifact remains; and
- final branch, topology, scope, protected bytes, and worktree cleanliness
  pass.

Implementation records the first exact candidate totals; Validation and Review
must independently reproduce them. Totals are evidence, not permission to
accept missing tests, extra skips, or silently narrowed discovery.

## Normalized Commands

Each role uses fresh exact TensorCore source and canonical-ZIP forms and a
Python `3.14.6` environment containing the exact dependency versions above.
Paths are role-private evidence inputs and must not enter committed records.
The normalized commands are:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<TensorDSLab>:<TensorCore-form> \
  <python-3.14.6> -B -m unittest discover -s tests -v

<pyright-1.1.411> --pythonversion 3.14 --pythonpath <python-3.14.6>

<python-3.14.6> -B -m build --wheel --sdist
```

The focused suite is the exact union of the modernized TensorCore dependency,
package-contract, axes/sampling, Pint/config, RNG ownership, count
orchestration, runtime-action ownership, readout simulation, and typing
contract tests named by the candidate. The full discovery command above is
mandatory in both dependency forms before and after the candidate commit.
Pyright runs against both forms from fresh role-private configurations whose
only dependency-path difference is the exact TensorCore source versus archive
root.

The artifact gate installs the fresh TensorDSLab wheel into an isolated
Python `3.14.6` environment with no project root on `sys.path`, verifies exact
metadata pins and `35/5/30` TensorDSLab facade identities, checks downstream
import isolation, runs the focused suite, and compares extracted wheel/sdist
package bytes with source. Editable installs are not evidence.

## Role Route And Candidate Budget

The persistent TensorDSLab roles are:

```text
Design -> Implementation -> Validation -> Review
```

Implementation produces one immutable direct-child candidate of the fixed
Design authority and dispatches it to persistent Validation. Validation
independently reconstructs all dependency forms and may return a bounded
finding packet to Implementation. The ordinary budget is:

```text
Implementation -> Validation candidate submissions: 3
Validation -> Implementation returns:              3
```

Validation dispatches an unchanged cleared candidate to persistent Review.
Review independently repeats the required functional, typing, artifact,
scope, documentation, privacy, and hygiene evidence. Review alone may
fast-forward a cleared candidate to the governed local `main`, using
`git merge --ff-only`. Review may not push.

An exhausted candidate route, Design-owned documentation contradiction,
dependency discrepancy, environment mismatch, unavailable exact Python/Torch
input, unexpected skip, scientific difference, or need for cluster evidence
returns to Design. No supplemental candidate or allocation is implicit.
Design owns a later evidence-only lifecycle closeout over the exact twelve
current records only after Review's unchanged fast-forward.

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

## Dispatch Gate

The Design authority is the exact commit that introduces this fixed work-order
state. Its hash and tree are carried in the immutable dispatch handoff because
a Git commit cannot truthfully contain its own hash. Implementation must prove
that exact authority as its direct parent before editing.

The work order is ready for user review but remains undispatched. Explicit user
authorization is still required. Dispatch must name the exact authority
commit/tree, confirm all execution roles are Active, and state that CUDA and
push remain unauthorized. If any later TensorCore or TensorDSLab byte
contradicts this contract, stop and return the conflict to both package Design
authorities.
