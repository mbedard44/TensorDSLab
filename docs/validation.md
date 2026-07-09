# Validation

Validation should prove behavior and contract boundaries, not mirror
implementation structure.

## Current Baseline

TensorDSLab currently has a documentation-only baseline. There is no production
package, test suite, cache schema, package metadata, or accepted runtime API.

For documentation-only stages, the expected check is:

```bash
git diff --check
```

If the repository is not yet initialized as a git repository, use an equivalent
whitespace and link sanity check and report that `git diff --check` was not
available.

## Future Validation Direction

Validation should attack TensorDSLab's typed in-memory product and TensorCore
boundaries first:

- source readers validate accepted g4ds11 input records before constructing
  TensorDSLab-native values;
- row identity and source provenance stay distinct;
- detector products preserve deterministic ordering and explicit identity;
- readout products preserve accepted TensorDSLab semantics while making tensor axes,
  layouts, coordinates, and indices explicit;
- post-binned readout transforms preserve the accepted operation order for
  timing jitter, dark counts, crosstalk, afterpulses, charge smearing,
  waveform rendering, physical composition, and optional digitization;
- TensorDSLab product surfaces defer concrete tensor shape to runtime
  TensorCore layouts while validating required field roles, semantic axis
  roles, device, dtype, and product meaning at boundaries;
- transforms using `out=` write into the supplied product and return that same
  product, while transforms without `out` allocate a new product without
  mutating the input product;
- reconstruction products enter only through accepted future contracts;
- TensorCore-backed renderings use explicit `TensorAxis`, `TensorAxes`,
  `TensorLayout`, `TensorField`, and `TensorCollection` contracts;
- product labels are not confused with TensorCore `TensorFieldId` values;
- durable IO, cache writers, loaders, validators, and compaction stay deferred
  until accepted in a later work order;
- downstream and orchestration compatibility stay deferred until local product
  contracts are stable.

## Boundary-First Checks

Validation should prefer tests that prove data is validated at boundaries and
trusted downstream:

```text
external/source/config/artifact boundary
  -> validate/coerce into typed records
  -> construct TensorDSLab and TensorCore records
  -> product builders and tensor renderers trust those records
```

Tests should include malformed-boundary cases where practical: duplicate IDs,
empty ID sequences, invalid tensor shapes, out-of-order or inconsistent cache
rows, missing product labels, invalid manifest paths, bool values passed to
numeric wrappers, and coordinates used where tensor indices are required.

## DAG Boundary Checks

Before DAG compatibility is accepted, Validation should flag accidental
campaign orchestration inside TensorDSLab:

- scheduler policy;
- retry or repair behavior;
- compiled DAG objects;
- campaign fanout/fanin;
- cross-shard orchestration policy;
- hidden dependencies on Projects/dag inside core domain modules.

Those concerns belong to Projects/dag unless Design accepts a specific adapter
surface.

## Future Command Shape

After the package exists, local checks should run from the project root with
the project root on `PYTHONPATH`:

```bash
git diff --check
PYTHONPATH=. python -m unittest discover -s tests
```

For future DAG-compatible changes, add the accepted operation-spec validation
command to this file and to the relevant stage work order.
