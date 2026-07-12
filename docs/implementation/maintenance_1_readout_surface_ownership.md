# Maintenance 1 Readout Surface Ownership Work Order

Status: production dispatched from committed Design authority
`d09cbad4a1538349e289523a9898f4e6dfd20a57` on 2026-07-11. A feature-branch
copy is candidate evidence and does not itself imply fixed-commit Validation
or independent Review clearance; if this updated record is read on `main`,
Review's clean fast-forward gate has completed. The user accepted this exact
ownership correction and no broader surface.

## Objective

Correct two public-organization problems without changing readout behavior:

1. move the semantic `ReadoutCollection` value type from
   `tensor_dslab.readout.tensors` to `tensor_dslab.readout.types`; and
2. remove the redundant `READOUT_` prefix from the readout package's four
   public axis-ID symbols.

The resulting public axis API is exactly:

```python
EXAMPLE_AXIS_ID = TensorAxisId("example")
CHANNEL_AXIS_ID = TensorAxisId("channel")
SAMPLE_AXIS_ID = TensorAxisId("sample")
REQUIRED_AXIS_IDS = IdSequence(
    (EXAMPLE_AXIS_ID, CHANNEL_AXIS_ID, SAMPLE_AXIS_ID)
)
```

These remain exact `TensorAxisId` and `IdSequence` values. Callers should use
the domain namespace when ambiguity is possible:

```python
from tensor_dslab import readout

sample_dimension = layout.axes.index(readout.SAMPLE_AXIS_ID)
```

Do not replace them with enum members, caller-configurable role records,
strings, an aggregate axis sidecar, or class attributes on
`ReadoutCollection`.

## Authority And Baseline

Starting clean `main`:

```text
cf0ccf0ad8fdee53767a374837276991decb1703
```

Accepted Stage 2 implementation commit:

```text
e8c62caf001ee7f58f766d7234747ed1d9a21e35
```

TensorCore remains exact version `0.6.0` at:

```text
dc554994061183776f23f65860a0594516074f2e
```

Target branch:

```text
codex/readout-surface-ownership
```

Target merge branch: `main`.

Logical execution routes:

```text
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Coordination remains Deferred and is not used. The Implementation/Validation
loop permits at most two dispatches in each direction. Review receives only a
fixed Validation-cleared commit and owns the clean fast-forward plus post-merge
verification.

## Required Production Changes

### `tensor_dslab/readout/types.py`

Move `ReadoutCollection` here unchanged in behavior:

- retain `@final` and the frozen, slotted, keyword-only dataclass contract;
- retain direct `TensorCollection` inheritance;
- retain the function-local validation import in `__post_init__`;
- retain `layout`, `device`, and semantic dimension properties; and
- use the new short axis-ID symbols.

This module owns stable public readout records and domain value objects:
`SampleGrid`, `DigitizedWaveformSpec`, `AdcQuantization`, and
`ReadoutCollection`.

### `tensor_dslab/readout/tensors.py`

Remove the class definition and import `ReadoutCollection` from `types.py`.
Keep only the existing readout-semantic reconstruction, projection, selection,
and movement functions. Their public names and behavior do not change in this
maintenance work.

The retained module is explicit evidence for a later Design decision. It is
not permission to add generic tensor helpers locally.

### Other Production Imports

Update `builders.py`, `validation.py`, and `readout/__init__.py` to import
`ReadoutCollection` from `types.py`. The package-level public import remains:

```python
from tensor_dslab.readout import ReadoutCollection
```

Update `ids.py`, internal users, package exports, and tests to use exactly:

```text
EXAMPLE_AXIS_ID
CHANNEL_AXIS_ID
SAMPLE_AXIS_ID
REQUIRED_AXIS_IDS
```

Remove these old public names completely:

```text
READOUT_EXAMPLE_AXIS_ID
READOUT_CHANNEL_AXIS_ID
READOUT_SAMPLE_AXIS_ID
READOUT_REQUIRED_AXIS_IDS
```

TensorDSLab is pre-deployment and has made no backward-compatibility guarantee.
Do not add aliases, deprecation shims, `__getattr__` fallbacks, or duplicate
registries. Do not rename any field-ID Python symbol or change any axis/field
string value.

## Documentation Duties

Synchronize current public-contract and onboarding sources that name the old
axis symbols or module ownership, including as applicable:

- `AGENTS.md`;
- `CONTRIBUTING.md`;
- `README.md`;
- `docs/overview.md`;
- `docs/design.md`;
- `docs/decisions.md`;
- `docs/architecture/tensors.md`;
- `docs/architecture/readout.md`;
- `docs/validation.md`; and
- `docs/implementation/index.md`.

The closed Stage 0, Stage 1, and Stage 2 work orders are historical execution
records. Do not mechanically rewrite their accepted-time code sketches. This
maintenance record supersedes their Python module/name spellings while leaving
their scientific and behavioral evidence intact. Governance adoption and rule
map records remain byte-unchanged.

## Tests And Verification

Add focused assertions that prove:

- `ReadoutCollection.__module__ == "tensor_dslab.readout.types"`;
- the package root exports all four new short axis symbols as exact TensorCore
  value types;
- none of the four retired names is exported or present in `__all__`;
- importing `types`, `validation`, `tensors`, `builders`, and the package root
  in fresh processes exposes no circular import; and
- every pre-existing behavioral test remains unchanged in meaning.

Run at least:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/tmp/tensorcore-dc554994.zip python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/tmp/tensorcore-dc554994.zip python -c "import tensor_dslab.readout.types, tensor_dslab.readout.validation, tensor_dslab.readout.tensors, tensor_dslab.readout.builders, tensor_dslab.readout"
```

Run `pyright` if available. Report exact Python, PyTorch, TensorCore, and CUDA
environment evidence. Conditional CUDA skips remain a qualification, not a GPU
claim.

## Non-Goals

- No scientific transform, RNG, workspace, full-chain builder, cache, source,
  reconstruction, TensorG4DS, TensorML, DAG, or integration work.
- No behavior, constructor, dtype, layout, sidecar, invalidation, destination,
  aliasing, autograd, or lifetime change.
- No field-ID rename or durable product-label change.
- No TensorCore change, compatibility shim, sibling-repository edit, or
  compatibility finding.
- No attempt to eliminate `tensors.py` in this maintenance work.
- No release, deployment, zero-copy, allocation-free, GPU, conformance, or
  backward-compatibility claim.
- No Coordination/Profile B/routing activation and no push.

## Return-To-Design Boundary

Return to Design if implementation requires:

- changing any TensorCore contract;
- overriding generic `TensorCollection` operations with stronger
  preconditions;
- moving the remaining semantic helpers out of `tensors.py`;
- adding an axis namespace, enum, wrapper, role record, or compatibility alias;
- changing field IDs, sidecars, constructor semantics, or public behavior; or
- widening any production or documentation path beyond this ownership/name
  correction.

## Deferred Generalization Question

After this maintenance work, Design should separately decide whether
`tensors.py` should disappear through:

1. readout-semantic methods or functions with a more precise domain home; or
2. a new opt-in TensorCore reconstruction hook that lets semantic subclasses
   preserve valid sidecars without changing TensorCore's current exact-base
   operation contracts.

`SampleGrid` updates after sample selection, digitized-sidecar pruning after
field projection, and readout canonical-order validation remain TensorDSLab
semantics even if TensorCore later reduces the generic reconstruction
boilerplate.
