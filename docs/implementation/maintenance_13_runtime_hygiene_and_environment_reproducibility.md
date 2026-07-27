# Maintenance 13 Runtime Hygiene And Environment Reproducibility

Status is self-effecting. While these exact bytes are absent from `main`, they
are an **Implementation candidate under fixed-commit Validation and Review
gates**, with the latest completed same-byte handoff determining the pending
role. If these exact bytes appear unchanged on `main` after complete
Validation, independent Review, final same-byte Design approval, and Review's
clean fast-forward, the status is **Merged / Closed**.

Stable key:
`TensorDSLab/maintenance-13-runtime-hygiene-and-environment-reproducibility`

## Purpose

Remove two small production-maintenance defects, centralize one genuinely
shared Runtime preparation operation, replace one brittle package-topology
test, and restore the repository-root environment creator to the current
published package contract.

This maintenance is deliberately narrow. It:

- removes the unused `checked_rate_product()` function, its unused
  `MAX_POISSON_MEAN` constant, and the now-unused `math` import;
- extracts the duplicated physical-kernel conditioning alignment from Charge
  and PureWaveform preparation into one private
  `align_quantity_kernel(...)` action;
- replaces the exact production-module-count assertion with required-path and
  retired-path package invariants;
- corrects `create_environment.sh` for exact TensorCore `0.21.0` and the
  current geometry-requiring `ds20k_veto(...)` signature;
- proves that a fresh environment installs and imports the noneditable package
  rather than accidentally importing the repository checkout; and
- changes no public API, scientific law, physical coefficient, RNG identity,
  stochastic result, product, Config, Runtime record, dependency, version, or
  supported device boundary.

Maintenance 14 is a separate tests-only curation stage. It may split the large
noise test module and consolidate repetitive generated tests only after this
maintenance closes. Maintenance 13 must not absorb that broader test rewrite.
Living-documentation curation is also separate.

This work follows:

- [CONTRIBUTING](../../CONTRIBUTING.md), especially the requirements to
  extract only genuinely identical behavior into the narrowest private owner,
  preserve the Config/Runtime/production split, keep imports acyclic, and make
  tests prove contracts rather than private implementation volume;
- [Readout Architecture](../architecture/readout.md) for product preparation,
  physical-kernel ownership, and Pint-free Runtime state;
- [Tensor Architecture](../architecture/tensors.md) for semantic field/kernel
  roles, snapshots, coordinate identity, and device materialization;
- [Validation](../validation.md) for exact dependency, typing, product,
  import-isolation, and environment evidence; and
- [Parity](../parity.md) for the unchanged Maintenance 12 scientific
  comparison boundary.

No donor behavior, approximation, or new parity classification is selected.
`docs/parity.md` remains byte-identical.

## Exact Design Baseline

The exact draft baseline is the published clean TensorDSLab main:

```text
repository:
    https://github.com/mbedard44/TensorDSLab.git
branch / local main / origin/main / live GitHub main:
    c8de1528d1ed57d3e86a9c37d1ad307127a23feb
tree:
    1d58e398428f35600e9bc582366c846c90d5f47c
package version:
    0.1.0
Python:
    >=3.14
Torch:
    >=2.13,<2.14
NumPy:
    2.5.1
Pint:
    0.25.3
TensorCore:
    exact published 0.21.0 containing/implementation commit
    78d0891bf6c0fefbcad4abe09980867c54202a9e
TensorCore tree:
    af5c4f6d693fa25cf767f3aaae31a47d86cf3a8d
```

The baseline full eager-CPU suite was independently repeated from this exact
checkout with CPython `3.14.6`, the exact sibling TensorCore source, and
bytecode disabled:

```text
371 tests run
368 passed
3 conditional unavailable-CUDA skips
```

## Implementation Candidate Evidence

Implementation prepared one coherent direct descendant of the exact Design
authority. Its changed-path set is exactly the ten-path allowlist in this work
order. The focused local gate used CPython `3.14.6`, PyTorch `2.13.0`, NumPy
`2.5.1`, Pint `0.25.3`, exact TensorCore `0.21.0` source commit
`78d0891bf6c0fefbcad4abe09980867c54202a9e`, and Pyright `1.1.411`.

The focused count, alignment, package-contract, fake-Conda environment-script,
Charge, PureWaveform, and public-readout matrix passed:

```text
147 tests run
147 passed
0 skipped
```

Pyright reported zero errors, warnings, or informations. `bash -n` passed and
the environment creator retained exact mode `100755`. Separate process-local
mutants that omitted conditioning-coordinate reorder or conditioning-dimension
permutation were both rejected by the committed combined-alignment proof.
Source, scope, protected-byte, facade, retired-symbol, diff, and artifact
hygiene gates passed before candidate dispatch.

Per the final-candidate evidence cadence, Implementation did not create a real
Conda environment, rebuild package artifacts, run the complete source/archive
suite, or claim final package/environment clearance. Those gates belong to
Validation on the exact immutable candidate. CUDA was unavailable
(`torch.cuda.is_available() is False`), no CUDA test was run, and no
accelerator claim follows.

This is the current baseline. The historical Maintenance 12 `383/380/3`
result predates removal of the provisional random notebook and its dedicated
tests and must not be used as a Maintenance 13 acceptance total.

The accepted TensorCore artifacts remain:

```text
canonical commit-bound source archive:
    tensorcore-stage29-78d0891.tar
size:
    1,003,520 bytes
SHA-256:
    0f8ca6a5270845c272e941ef928a325f1a0e57aa7fe81c965d04086a5823363f
wheel:
    tensor_core-0.21.0-py3-none-any.whl
size:
    51,644 bytes
SHA-256:
    29ff9dc4f0fcead0120da2b3c1993dae2bc6c79106c757cc90fd2a446c4f8bc6
```

Maintenance 13 changes neither the dependency pin nor any TensorCore
consumer contract.

## Current Defects

### Dead count code

`tensor_dslab/readout/charge/runtime/counts.py` currently contains:

```python
import math

MAX_POISSON_MEAN = 1.0e8


def checked_rate_product(
    counts: torch.Tensor,
    factor: float,
    field: str,
) -> torch.Tensor:
    ...
```

Repository-wide source search finds no consumer of `checked_rate_product` or
`MAX_POISSON_MEAN`. The active dark-count and branching paths already own
their contextual Poisson-domain checks. The dead function and constant are not
an accepted extension point or public surface.

### Duplicated conditioning alignment

Charge preparation and PureWaveform preparation independently perform the
same operation:

1. resolve a physical kernel's semantic roles against the source field;
2. compare conditioning-axis coordinate domains;
3. map kernel coordinate order to source coordinate order;
4. select and reorder the kernel's conditioning dimensions into source-field
   dimension order;
5. leave the trailing operation geometry unchanged; and
6. materialize one contiguous tensor on the execution device and dtype.

The two implementations differ only in product-specific dtype choice and
diagnostic wording. Keeping both copies risks drift in coordinate permutation,
device movement, contiguity, or validation order.

### Brittle module-count gate

`tests/test_package_contracts.py` currently requires exactly `59` production
Python modules. Adding one cohesive private Runtime action therefore fails the
test even when all required and retired paths, facades, imports, and ownership
rules remain correct.

An exact module count is not a package contract. Required modules, retired
paths, public facades, empty Runtime facades, import isolation, and an acyclic
dependency direction are the meaningful invariants.

### Stale environment smoke

`create_environment.sh` correctly:

- defaults to the environment name `tensor_dslab`;
- accepts one alternate environment name;
- refuses to replace an existing named environment;
- creates exact Python `3.14.6`;
- installs the local noneditable `.[demos]` project; and
- prints commands that the user can run in the parent shell.

Its final smoke check is stale in two concrete ways:

```text
expected TensorCore version:
    0.19.0
current exact dependency:
    0.21.0

called profile:
    ds20k_veto()
current supported signature:
    ds20k_veto(*, sample_axis, channel_axis=None, example_axis=None)
```

The smoke also executes from the caller's current directory. When the script
is launched from the repository root, an import can resolve the source
checkout instead of proving the just-installed noneditable artifact.

## Frozen Non-Goals

Maintenance 13 does not:

- change `QuantityKernel`, its Pint canonicalization boundary, or any physical
  kernel class;
- change `ChargeConfig`, `PureWaveformConfig`, `ReadoutConfig`,
  `ds20k_veto(...)`, or `simulate_readout(...)`;
- move `DarkCountRate` to Photoelectrons or introduce an Axioelectrons
  product;
- add, remove, rename, or reorder a public export;
- change a Config or Runtime dataclass field;
- change a product field, collection, dtype, device, axes, storage, autograd,
  or freshness contract;
- change probability normalization, branching, afterpulse, dark-count,
  smearing, pulse-convolution, noise, analog, or digitization behavior;
- change RngKey roles, RngElements, RngAddress domains/quanta, word requests,
  seeds, traversal, or completed stochastic values;
- change TensorCore, the exact TensorCore pin, Python, Torch, NumPy, Pint,
  Hatchling, Pyright, or demo-tool versions;
- add an environment manager, lockfile, container, CI workflow, package-index
  publication, editable install, or shell-activation mechanism;
- reorganize the test suite beyond the exact focused files below;
- split `tests/test_noise_waveform_product.py`;
- consolidate the repetitive QuantityKernel or Runtime test families reserved
  for Maintenance 14;
- rewrite current architecture pages or historical work orders; or
- run or claim CUDA, accelerator performance, deployment, calibration,
  compatibility, release readiness, or production readiness.

## Target Production Changes

### Count-owner cleanup

Edit `tensor_dslab/readout/charge/runtime/counts.py` only to remove:

```text
import math
MAX_POISSON_MEAN
checked_rate_product(...)
```

Retain without semantic or diagnostic change:

```python
MAX_COUNT = (1 << 53) - 1


def checked_add(
    left: torch.Tensor,
    right: torch.Tensor,
    field: str,
) -> torch.Tensor:
    ...
```

The contextual `1.0e8` Poisson checks in active preparation remain in their
current owners. This maintenance does not centralize every numerical ceiling
merely because the unused constant existed.

Required evidence:

- production source contains no `checked_rate_product`;
- production source contains no `MAX_POISSON_MEAN`;
- `counts.py` no longer imports `math`;
- every active `MAX_COUNT` and `checked_add` consumer is unchanged; and
- existing count-ceiling, checked-addition, fail-before-words, branching, and
  dark-count tests remain exact.

### Shared private alignment action

Add:

```text
tensor_dslab/readout/runtime/kernel.py
```

with one ordinary non-exported action:

```python
def align_quantity_kernel(
    kernel: QuantityKernel,
    *,
    field: TensorField,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Align one physical kernel to a semantic execution field."""
```

The exact implementation must use public TensorCore contracts. It may import:

```python
import torch
from tensor_core import TensorField
from tensor_core.tensor.validation import require_kernel_dimensions

from tensor_dslab.common.kernel import QuantityKernel
```

It must not import a Config, Runtime record, Distribution, producer,
validator, profile, collection, RNG class, or product orchestration module.
`tensor_dslab/readout/runtime/__init__.py` remains empty and exports nothing.
The new action is an implementation detail and is not re-exported through any
facade.

The operation is exact:

1. Call `require_kernel_dimensions(field, kernel)` before materialization.
   This proves that every conditioning role and every operation target role is
   present in the field.
2. Start from the kernel-owned defensive tensor snapshot. Do not access
   `kernel.quantity` and do no Pint work.
3. For each conditioning axis in kernel order:
   - resolve the exact axis role against `field`;
   - require matching coordinate cardinality;
   - require a complete one-to-one coordinate correspondence;
   - map the kernel coordinate order into the field coordinate order with
     `index_of(...)`;
   - select only that conditioning dimension; and
   - record the corresponding field dimension.
4. Sort the conditioning dimensions into ascending field-dimension order and
   apply the same stable permutation to the leading conditioning dimensions.
5. Preserve every operation dimension, its order, its values, and its
   row-major geometry as trailing dimensions.
6. Perform one final `.to(device=field.tensor.device, dtype=dtype)` and return
   a contiguous tensor.
7. Return the conditioning field dimensions in the same ascending order as
   the returned tensor's leading conditioning dimensions.

The helper must not:

- broadcast or expand the tensor;
- select one source coordinate;
- interpret an offset, anchor, causality rule, boundary, probability,
  intensity, rate, pulse coefficient, or scientific law;
- construct a Runtime or Distribution;
- move or mutate the source field or kernel;
- normalize, clamp, repair, or reshape operation values;
- derive operation target dimensions or offset products;
- retain a Config, Quantity, field, kernel, or mutable cache; or
- silently copy through CPU after the final target-device materialization.

The helper owns one shared generic diagnostic family. Errors must identify the
concrete kernel class and distinguish:

- a required semantic role absent from the execution field; and
- conditioning coordinates that do not exactly correspond to the field axis.

Private diagnostic prose may be unified; exception categories and validation
precedence must remain stable. Missing roles and coordinate mismatches fail
before target-device materialization.

### Product preparation delegation

Change:

```text
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/pure_waveform/runtime/prepare.py
```

to import and call `align_quantity_kernel(...)`.

Charge preparation passes `torch.float64`. PureWaveform preparation passes the
already-admitted waveform `floating_dtype`.

Charge retains ownership of:

- dark-count exposure and Poisson ceiling;
- conditioning broadcast into the source shape;
- timing-jitter operation offsets and Runtime construction;
- branching operation target dimensions;
- row-major offset products;
- branching/smearing contextual bounds; and
- `ChargeRuntime`, `TimingJitterRuntime`, and `BranchingRuntime`.

PureWaveform retains ownership of:

- Pulse operation-axis interpretation;
- sample offsets;
- requested waveform dtype policy; and
- `PureWaveformRuntime`.

The two product modules must no longer contain their own coordinate-selection
loop, conditioning permutation, or direct
`require_kernel_dimensions(...)` call. This is a focused deduplication, not a
new generic effect or Runtime framework.

## Target Package-Contract Change

Rename the existing test method according to its real contract and remove only
the exact module-count assertion:

```python
def test_required_and_retired_production_paths(self) -> None:
    ...
```

The test must positively require the current cohesive ownership paths,
including:

```text
tensor_dslab/common/axis.py
tensor_dslab/common/kernel.py
tensor_dslab/readout/runtime/kernel.py
tensor_dslab/readout/charge/runtime/counts.py
tensor_dslab/readout/charge/runtime/branching.py
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/pure_waveform/runtime/prepare.py
```

It must retain or strengthen exact absence of retired paths, including:

```text
tensor_dslab/common/axes.py
tensor_dslab/readout/charge/runtime/effects/
tensor_dslab/readout/charge/effects/
tensor_dslab/readout/_random.py
tensor_dslab/readout/_rng.py
```

It must not replace `59` with `60` or introduce another exact production-file,
module, line, or import-edge count. Validation may report observed topology,
but harmless cohesive additions must not fail merely because a census changed.

The exact public facade sequences remain:

```text
tensor_dslab:
    31
tensor_dslab.common:
    6
tensor_dslab.readout:
    25
tensor_dslab.readout.charge:
    8
tensor_dslab.readout.pure_waveform:
    3
```

Existing exact facade, export-identity, Runtime-empty-facade, module-docstring,
public-docstring, import-isolation, and retired-surface tests remain.

## Environment Creator Repair

`create_environment.sh` remains executable mode `100755` and retains its
supported command:

```text
./create_environment.sh [environment-name]
```

The script continues to:

- default to exact name `tensor_dslab`;
- accept at most one nonempty alternate name;
- use `${CONDA_EXE:-conda}`;
- refuse an existing named environment rather than modify it;
- create exact Python `3.14.6` from conda-forge;
- install the repository noneditably with `.[demos]`;
- avoid activating or modifying the caller's shell;
- avoid registering or selecting a Jupyter kernel; and
- print the later activation, repository, and demo commands.

The corrected smoke must run from a fresh temporary directory outside the
repository so the checkout cannot shadow the installed package. The script
must clean that temporary directory on both success and failure.

Inside the created environment, the smoke must prove:

```text
Python:
    3.14.6
tensor-dslab distribution:
    0.1.0
tensor-core distribution:
    0.21.0
TensorCore PEP 610 vcs_info.commit_id:
    78d0891bf6c0fefbcad4abe09980867c54202a9e
tensor_dslab import:
    resolves from the created environment's installed site-packages
    and not from the repository root
```

The profile smoke must construct explicit geometry:

```python
sample_axis = SampleAxis.from_period(
    period=quantity(2, "ns"),
    count=8,
)

config = ds20k_veto(sample_axis=sample_axis)

assert type(config) is ReadoutConfig
```

The small count is a smoke-only execution input. It does not change the
profile, demo, or scientific configuration.

The exact dependency commit must be read from the installed TensorCore
distribution's `direct_url.json` / PEP 610 `vcs_info.commit_id`. A
`version("tensor-core") == "0.21.0"` assertion alone is insufficient because
it cannot distinguish the selected immutable Git commit from other package
bytes with the same version.

The script must not:

- clone or use the sibling TensorCore checkout;
- set `PYTHONPATH`;
- import TensorDSLab from the project root during its smoke;
- install editable package state;
- overwrite an existing environment;
- install a floating TensorCore branch, tag, or version;
- write environment paths into tracked files; or
- run the full package suite or notebook as part of ordinary environment
  creation.

## Focused Tests

Add:

```text
tests/test_runtime_kernel_alignment.py
tests/test_environment_script.py
```

### Alignment tests

The shared alignment tests must cover:

- a global scalar kernel with no conditioning axes;
- one ExampleAxis-conditioned kernel;
- one ChannelAxis-conditioned kernel;
- combined ExampleAxis and ChannelAxis conditioning;
- kernel conditioning axes in a different order from source field dimensions;
- conditioning coordinates in a different order from the source coordinates;
- a source field whose semantic axes are not in profile-construction order;
- one operation axis and a multidimensional operation geometry, proving all
  operation dimensions remain trailing and unchanged;
- exact returned conditioning dimensions;
- requested `float64` and waveform floating dtype;
- target device, contiguity, no gradient, and fresh materialization where
  conversion requires it;
- source and kernel immutability;
- missing-role and coordinate-mismatch failures before materialization; and
- both Charge and PureWaveform preparation delegating to the shared action.

At least one fixture must require both coordinate reordering and conditioning
dimension permutation. A mutant that omits either operation must fail.

Existing end-to-end scientific and product tests remain the result oracle.
The new tests must not duplicate their statistical sample loops.

### Environment-script tests

The script-contract tests must use a role-private fake Conda executable and
temporary directories. They must prove, without network access:

- usage and nonempty-name admission;
- exact default and alternate environment names;
- existing-environment refusal before create/install;
- exact Python create request;
- exact noneditable local `.[demos]` install request;
- no `-e` / `--editable`, `PYTHONPATH`, activation, kernel registration, or
  environment replacement;
- exact current TensorCore version and commit assertions in the smoke;
- explicit `SampleAxis` construction before `ds20k_veto(...)`;
- an external smoke working directory;
- cleanup of temporary smoke state; and
- executable mode plus `bash -n`.

The fake executable is contract evidence, not a substitute for the one real
final-candidate environment gate below.

## Protected Bytes And Contracts

The Implementation candidate must leave these bytes unchanged:

```text
pyproject.toml
tensor_dslab/__init__.py
tensor_dslab/common/
tensor_dslab/readout/config.py
tensor_dslab/readout/collection.py
tensor_dslab/readout/simulation.py
tensor_dslab/readout/profiles.py
tensor_dslab/readout/runtime/keys.py
tensor_dslab/readout/runtime/prepare.py
tensor_dslab/readout/runtime/sampling.py
tensor_dslab/readout/charge/config.py
tensor_dslab/readout/charge/kernel.py
tensor_dslab/readout/charge/runtime/branching.py
tensor_dslab/readout/charge/runtime/produce.py
tensor_dslab/readout/pure_waveform/config.py
tensor_dslab/readout/pure_waveform/kernel.py
tensor_dslab/readout/pure_waveform/runtime/produce.py
tensor_dslab/readout/noise_waveform/
tensor_dslab/readout/analog_waveform/
tensor_dslab/readout/digitized_waveform/
demos/
README.md
CONTRIBUTING.md
AGENTS.md
docs/api.md
docs/architecture/
docs/design.md
docs/decisions.md
docs/overview.md
docs/parity.md
docs/validation.md
all historical docs/implementation records
```

Tests outside the exact allowlist below are protected. Maintenance 14 owns
later test consolidation and must not be preimplemented here.

## Exact Candidate Allowlist

The complete Implementation candidate may change only:

```text
create_environment.sh
tensor_dslab/readout/runtime/kernel.py
tensor_dslab/readout/charge/runtime/counts.py
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/pure_waveform/runtime/prepare.py
tests/test_environment_script.py
tests/test_package_contracts.py
tests/test_runtime_kernel_alignment.py
docs/implementation/index.md
docs/implementation/maintenance_13_runtime_hygiene_and_environment_reproducibility.md
```

The maximum scope is ten paths. The work order and index may receive only
candidate identity, focused Implementation evidence, and self-effecting
lifecycle synchronization. No other current or historical document is
allowlisted.

Any required path outside this list is a Design stop. Implementation must not
infer permission from conceptual similarity.

## Evidence Cadence

This maintenance adopts final-candidate-oriented evidence rather than
rebuilding every expensive environment and artifact after every narrow
correction.

### Implementation handoff

Implementation must provide:

- exact commit/tree/parent and clean branch identity;
- exact allowlist diff and `git diff --check` / `git show --check`;
- focused count, alignment, package-contract, environment-script, Charge,
  PureWaveform, and public-readout tests;
- `bash -n create_environment.sh` and exact mode `100755`;
- Pyright zero diagnostics;
- source scans for retired count symbols, duplicated alignment mechanics,
  forbidden environment behavior, public export drift, and scope;
- proof that no package dependency or protected byte changed; and
- no real Conda environment requirement unless Implementation elects one for
  diagnosis.

Implementation must not claim final package or environment clearance from
that focused handoff.

### Validation final-candidate gate

Validation runs one complete gate against the final immutable candidate:

1. verify exact commit/tree/parent, allowlist, protected bytes, diff checks,
   mode bits, and clean state;
2. verify exact TensorCore `0.21.0` source/archive identity and public consumer
   surface without rebuilding unrelated dependency evidence twice;
3. run the focused tests against exact TensorCore source and extracted
   canonical archive;
4. run the complete TensorDSLab suite once in each accepted dependency form,
   recording run/pass/conditional-skip totals without requiring the historical
   `371` method count;
5. run Pyright once in each dependency form and retain the exact dependency
   negative fixture;
6. build final wheel and sdist once, prove source/artifact package equality,
   and run isolated-wheel import plus truth-only readout smoke;
7. run `bash -n`, fake-Conda script tests, mode/privacy/path scans, and
   import-isolation;
8. create one fresh role-private real Conda environment through the exact
   candidate script from outside the repository;
9. prove installed Python/package versions, exact TensorCore PEP 610 commit,
   installed-site-packages import, profile smoke, and CPU demo execution;
10. remove the role-private environment and prove repository/environment
    hygiene; and
11. record the unavailable-CUDA qualification without running CUDA.

If a correction changes only committed evidence prose after this complete
gate, Validation performs an exact byte/scope/truth recheck and does not
recreate the environment, rebuild artifacts, or rerun production tests whose
inputs are byte-identical. Any production, test, script, dependency, or
artifact-input change requires the affected gate and, where relevant, the
complete final-candidate gate again.

### Independent Review

Review is risk-based. It must independently inspect:

- the exact shared alignment semantics and both product delegations;
- retained scientific and product ownership;
- dead-code removal with active ceilings preserved;
- replacement of the brittle census by meaningful path invariants;
- the environment script's external import proof and exact Git dependency
  identity;
- focused mutation evidence for omitted coordinate reorder/permutation and
  source-tree import shadowing; and
- exact scope, protected bytes, typing, privacy, and cleanliness.

Review may rely on exact unchanged complete Validation evidence rather than
rerunning the full source/archive/artifact/environment matrix without a
specific risk or discrepancy. A finding returns a concrete contract or proof
gap; it does not automatically authorize unrelated test or documentation
cleanup.

## Lifecycle And Merge

The exact Design authority is the committed form of this work order and its
synchronized index entry accepted by the user. The persistent Implementation,
Validation, and Review routes must verify their roles and exact package state
before acting on their respective gates.

The production route is:

```text
Design authority
    -> focused Implementation candidate and handoff
    -> one complete final-candidate Validation gate
    -> risk-based independent Review
    -> final same-byte Design approval
    -> Review-owned clean git merge --ff-only to local main
    -> identity, diff, and cleanliness recheck
```

Candidate commits are immutable. A correction is a new direct child and
reruns only the evidence affected by its changed bytes, subject to the final
Validation rule above.

The Status section and index must use a self-effecting lifecycle rule:

- before complete Validation, Review clearance, final Design approval, and the
  exact clean fast-forward, the applicable state is the latest completed
  same-byte handoff; and
- after that exact fast-forward, Maintenance 13 is Merged / Closed on local
  main.

No separate evidence-only closeout commit is required merely to restate the
successful fast-forward. Review records the accepted exact evidence in its
handoff and verifies post-merge identity. A later documentation correction is
required only for an actual false living claim.

Push is not part of this work order. A later ordinary push may be separately
authorized after local closeout; no repeated production gate is implied if
the pushed bytes are the exact already-cleared local main.

## Hard Stops

Stop and return to Design if:

- the helper requires a public export, Runtime base, registry, callback,
  Config reflection, Distribution factory, or generic effect framework;
- exact source behavior requires changing a physical kernel, Config, Runtime
  record, producer, validator, product, or public signature;
- coordinate alignment cannot be shared without changing operation-axis or
  scientific semantics;
- any stochastic word/address/result changes;
- `create_environment.sh` cannot verify the installed TensorCore commit
  through standard installed metadata;
- a required fix changes `pyproject.toml`, another dependency, or a protected
  path;
- the real environment requires `PYTHONPATH`, an editable install, or the
  sibling TensorCore checkout;
- Maintenance 14 test consolidation is needed to make this candidate pass; or
- CUDA, deployment, release, compatibility, or documentation-curation scope
  becomes necessary.

## Completion Criteria

Maintenance 13 is complete only when:

- the three dead count artifacts are absent and active count behavior is
  unchanged;
- exactly one private shared conditioning-alignment implementation exists;
- Charge and PureWaveform preparation both delegate to it;
- operation geometry and product-specific Runtime ownership remain intact;
- exact public facades and signatures remain unchanged;
- the package contract proves required/retired paths without an exact module
  count;
- the environment script proves Python `3.14.6`, TensorDSLab `0.1.0`,
  TensorCore `0.21.0` at exact commit `78d0891...`, installed-site-packages
  import, and current profile construction;
- focused, full, typing, artifact, import-isolation, fake-Conda, and one real
  fresh-environment gates pass;
- the candidate contains only the ten allowlisted paths;
- unavailable CUDA remains explicit and unclaimed;
- independent Review returns no finding;
- Design approves the exact same bytes; and
- Review fast-forwards the exact candidate cleanly to local main.

## Deferred Maintenance 14

Maintenance 14 begins only from the exact closed Maintenance 13 main. Its
provisional tests-only scope is:

- split `tests/test_noise_waveform_product.py` into cohesive flat modules;
- replace the seventy repetitive physical-kernel construction methods with
  compact table-driven boundary evidence;
- remove the ten repeated Runtime methods and their no-op
  `assertEqual(index, index)` assertions;
- preserve substantive scientific, statistical, RNG, conditioning,
  convolution, preflight, device, typing, and facade evidence; and
- explicitly permit the total test-method count to decrease.

Maintenance 14 receives its own exact Design work order, allowlist, and
acceptance gate. Nothing in this Maintenance 13 authority dispatches it.
