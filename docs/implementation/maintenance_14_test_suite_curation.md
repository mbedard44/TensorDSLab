# Maintenance 14 Test Suite Curation

Status: **Implementation Candidate 1; fixed-commit Validation pending**.

This status describes the feature-branch state while the candidate is absent
from local `main`. If the exact candidate tree later appears unchanged on
local `main` through the authorized same-byte workflow, that presence records
Review's clean fast-forward and this maintenance is **Merged / Closed**.

Stable key:
`TensorDSLab/maintenance-14-test-suite-curation`

## Purpose

Make the TensorDSLab test suite easier to navigate and maintain without
changing what the package proves.

This maintenance is tests-only. It:

- splits the `1,291`-line noise test monolith into cohesive flat test modules;
- moves shared noise fixtures and independent numerical oracles into one
  private test-support module;
- replaces seventy generated physical-kernel construction methods with one
  explicit table-driven contract test containing seven named subtests;
- replaces ten generated Runtime methods with one meaningful compiled-Runtime
  facts test and removes their no-op `assertEqual(index, index)` assertions;
- preserves every substantive noise, kernel, Runtime, scientific, statistical,
  RNG, conditioning, preflight, device, dtype, typing, facade, and CUDA-skip
  obligation in scope; and
- explicitly permits the discovered test-method total to decrease when the
  decrease comes only from removing redundant generated copies.

It changes no production byte, public API, dependency, environment script,
demo, notebook, scientific law, RNG identity, product contract, or living
architecture page.

This work follows:

- [CONTRIBUTING](../../CONTRIBUTING.md), especially the requirements that tests
  prove observable contracts, avoid unnecessary duplication, preserve
  scientific comparison boundaries, and keep helpers in the narrowest
  meaningful owner;
- [Validation](../validation.md) for deterministic, statistical, typing,
  import, and device evidence;
- [Parity](../parity.md) for the unchanged scientific comparison boundary;
- [Readout Architecture](../architecture/readout.md) for product and Runtime
  ownership; and
- [Maintenance 13](maintenance_13_runtime_hygiene_and_environment_reproducibility.md)
  for the exact closed production baseline and its final-candidate evidence
  cadence.

No donor behavior, approximation, test-strength reduction, or new parity
classification is selected. `docs/parity.md` remains byte-identical.

## Exact Design Baseline

Maintenance 14 starts only from exact locally closed Maintenance 13:

```text
branch / local main:
    af23b129de41601f825f08a50f7783980e6e9551
tree:
    ecf20f237ca685ff697111211d260dfadec4ab9b
exact parent / Maintenance 13 Design authority:
    dded046893365d3489cd3b2b9b2402c3f65e2c4c
published origin/main at Design time:
    c8de1528d1ed57d3e86a9c37d1ad307127a23feb
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

The accepted Maintenance 13 complete source and extracted-archive evidence is:

```text
384 tests run
381 passed
3 conditional unavailable-CUDA skips
```

Pyright reported zero diagnostics. The exact dependency negative fixture
retained `82` intended diagnostics. Maintenance 13 also passed its focused
source/archive, artifact, isolated-wheel, environment-script, installed import,
and fresh real Conda-environment gates.

Candidate 1 preserves the eighteen noise methods with exact class/method
identities and byte-identical method bodies in the required `7 / 4 / 4 / 3`
partition. Its focused source gate passed `79` tests: `78` passed and the one
established CUDA case was skipped because CUDA was unavailable. Pyright
`1.1.411` reported zero errors, warnings, or informations. The private support
module defines the twenty shared helpers/oracles exactly once, owns no
`TestCase` and no `test*` callable, and the generated kernel/Runtime families
and no-op assertion are absent. Complete source/archive and negative-typing
clearance remain Validation-owned.

Those results are baseline evidence, not a frozen test-count contract.
Maintenance 14 deliberately removes seventy-eight redundant discovered
methods: sixty-nine duplicates from the seventy-method kernel family and nine
duplicates from the ten-method Runtime family. Validation must report the
candidate's actual discovered totals, but no test may assert a repository-wide
method, file, line, or module count.

The exact accepted TensorCore artifacts remain:

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

Maintenance 14 changes neither the dependency pin nor a TensorCore consumer
contract. No TensorCore coordination or renewed dependency adoption is
required.

## Current Test-Suite Defects

### Noise responsibilities are hidden in one large module

`tests/test_noise_waveform_product.py` is `1,291` physical lines. It owns four
different evidence responsibilities:

```text
NoiseProductBranchTest:
    product branches, dtype/layout/device behavior, deterministic addresses,
    exact-zero behavior, white-noise equations and bounds

PsdPreparationTest:
    PSD cell integration, fsum conservation, Nyquist/DC coverage, finite
    accumulation, and fail-before-RNG admission

PsdSynthesisTest:
    odd/even spectral construction, coefficient mapping, terminal-bin
    behavior, repeatability, and row independence

NoiseStatisticalContractTest:
    frozen white and PSD ensembles plus conditional CUDA behavior
```

All four are valid test owners, but their coexistence behind one filename makes
the suite difficult to scan. The first `326` lines also mix reusable fixtures,
test-only RNG machinery, config constructors, independent reference
calculations, and assertion helpers with the first test class.

The fix is organizational. It is not permission to simplify the numerical
oracles, shrink statistical ensembles, loosen tolerances, change seeds, or
replace independent expected-value calculations with production helpers.

### Kernel construction is multiplied seventy times

`tests/test_kernel_geometry_and_quantity.py` already contains one
`test_all_concrete_kernel_leaves_are_fieldless_and_immutable()` table over the
seven public physical-kernel leaves. At module scope it separately generates
seventy methods by cycling over the same seven construction cases:

```python
for _case_index in range(70):
    ...
    setattr(
        QuantityKernelContractTest,
        f"test_quantity_kernel_construction_{_case_index:02d}",
        _construction_case,
    )
```

The changing scalar magnitude in those copies does not establish seventy
distinct contracts. One named subtest per concrete leaf is sufficient to prove
the common constructor facts while the existing boundary methods retain
invalid-domain, geometry, conditioning, convolution, canonical-quantity, and
immutability evidence.

### Runtime facts are multiplied ten times

`tests/test_runtime_action_ownership.py` generates ten identical methods. Each
prepares the same Runtime and checks the same two facts:

```python
runtime.charge.correlated_avalanche_generations == 0
runtime.pure_waveform.sample_offsets == (0,)
```

Every copy ends with:

```python
self.assertEqual(index, index)
```

That assertion proves only the loop variable equals itself. One clearly named
test should retain the two Runtime facts. The existing frozen/slotted,
privacy, dtype/device, and no-execution-method tests remain separate.

## Frozen Non-Goals

Maintenance 14 does not:

- edit any file under `tensor_dslab/`;
- edit `pyproject.toml`, `create_environment.sh`, `README.md`, a demo, or a
  notebook;
- add, remove, rename, or reorder a public export;
- change a function, class, dataclass field, signature, diagnostic, Config,
  Runtime, product, collection, profile, or simulation action;
- change a quantity, axis, conditioning, alignment, convolution, boundary,
  dtype, device, storage, autograd, or freshness contract;
- change probability normalization, branching, afterpulse, dark-count,
  smearing, pulse, noise, analog, or digitization behavior;
- change an RngKey, RngElements, RngAddress, domain, quantum, word request,
  seed, traversal, completed value, or stochastic/statistical acceptance
  boundary;
- alter a scientific fixture, sample size, seed, tolerance, high-precision
  oracle, expected value, conditional skip, or fail-before-words assertion;
- rewrite a test to assert production output against the same production
  helper that generated it;
- add a generic testing framework, base TestCase, registry, callback,
  parameterization dependency, snapshot plugin, or fixture package;
- introduce nested product test packages or mirror the complete production
  package tree under `tests/`;
- curate tests outside the exact files named below;
- change living API, design, overview, architecture, validation, parity,
  governance, or historical implementation records;
- rebuild the Conda environment or execute notebooks merely because tests were
  reorganized; or
- run or claim CUDA, accelerator performance, deployment, calibration,
  compatibility, release readiness, or production readiness.

Living-documentation curation remains a separate maintenance. Additional test
consolidation, including large modules not named here, requires later evidence
rather than opportunistic edits.

## Target Noise Test Topology

Delete:

```text
tests/test_noise_waveform_product.py
```

Add exactly:

```text
tests/_noise_waveform_support.py
tests/test_noise_waveform_branches.py
tests/test_noise_waveform_psd_preparation.py
tests/test_noise_waveform_psd_synthesis.py
tests/test_noise_waveform_statistics.py
```

The target deliberately remains flat. Four test modules correspond to four
cohesive evidence responsibilities. The underscore-prefixed support module is
a private test implementation detail and is not itself discovered as a test
module.

No compatibility import, forwarding module, empty retired file, dynamic
`sys.modules` alias, wildcard export, or nested `tests/noise_waveform/` package
is permitted.

### Private noise support

`tests/_noise_waveform_support.py` owns only definitions genuinely shared by
two or more target noise modules, or substantial independent oracle machinery
whose duplication would weaken maintainability. It may own:

- the frozen seed and axis-order cases;
- concise Pint construction conveniences used by multiple noise modules;
- `_FailingRng`;
- source, axis, sampling, and Config factories;
- direct TensorCore address/reference-draw construction used as an independent
  oracle;
- PSD cell-overlap, expected-power, coefficient, synthesis, and high-precision
  reference calculations;
- exact tensor/statistical assertion helpers; and
- other existing shared definitions moved byte-for-behavior from the retired
  module.

It must have an intentional module docstring and must not contain:

- a `unittest.TestCase` subclass;
- any callable whose name begins with `test`;
- a production API, facade export, or supported downstream import;
- package mutation at import time;
- test discovery manipulation;
- a replacement random engine or scientific algorithm used by production;
- a dependency on one of the four new test modules; or
- a new tolerance, seed, sample size, expected value, or source of truth.

Each target test module imports only the support definitions it actually uses.
No wildcard import is permitted. Test-only helper names remain underscored.

### Noise branches

`tests/test_noise_waveform_branches.py` owns the existing
`NoiseProductBranchTest` and exactly these seven methods:

```text
test_every_model_dtype_axis_order_and_noncontiguous_source
test_zero_is_fresh_exact_rng_inert_and_never_calls_rng
test_private_preparer_relies_on_public_dtype_and_device_admission
test_white_matches_finite_lattice_equation_without_demeaning
test_white_is_repeatable_stream_isolated_and_source_value_independent
test_float32_stochastic_products_ignore_ambient_cpu_autocast
test_white_normal_range_and_conservative_upper_bound
```

Their executable bodies, subtest domains, seeds, configs, exact comparisons,
error categories, RNG traces, and assertions remain behaviorally identical.

### PSD preparation

`tests/test_noise_waveform_psd_preparation.py` owns the existing
`PsdPreparationTest` and exactly these four methods:

```text
test_odd_even_cells_overlap_fsum_conservation_and_one_rounding
test_coverage_dc_only_above_nyquist_and_rounded_zero_reject_before_rng
test_nonfinite_accumulation_guard_rejects_before_rng
test_finite_accumulation_guard_accepts_limit_and_rejects_nextafter_before_rng
```

The module continues to prove independent frequency-cell integration, stable
summation, complete represented power, dtype rounding, DC and Nyquist policy,
finite accumulation limits, and failure before any word request.

### PSD synthesis

`tests/test_noise_waveform_psd_synthesis.py` owns the existing
`PsdSynthesisTest` and exactly these four methods:

```text
test_small_odd_even_reference_coefficients_dc_and_zero_power
test_odd_terminal_imaginary_and_isolated_cosine_sine_bases
test_two_sample_psd_is_real_nyquist_only
test_psd_repeatability_no_crop_no_normalization_and_row_independence
```

The independent coefficient and inverse-transform calculations remain
test-owned. Production preparation or production synthesis may not become the
expected-value oracle.

### Statistical and conditional-device evidence

`tests/test_noise_waveform_statistics.py` owns the existing
`NoiseStatisticalContractTest` and exactly these three methods:

```text
test_frozen_white_ensemble
test_frozen_odd_even_psd_ensembles
test_cuda_all_models_and_same_backend_repeatability
```

The frozen ensemble sizes, seeds, moment/spectral statistics, tolerances,
dtype cases, conditional CUDA guard, and same-backend replay requirements
remain exact. The split must not turn a conditional CUDA test into an
unconditional skip or a CPU-only substitute.

### Exact noise preservation proof

The eighteen noise test methods move one-for-one. Their class names and method
names do not change. Implementation and Validation must compare the baseline
and candidate AST/source inventory and prove:

- the exact `7 / 4 / 4 / 3` class-method partition;
- no method was dropped, added, duplicated, renamed, skipped, or decorated
  differently;
- control flow, assertions, constants, subtest arguments, patch targets, and
  exception expectations are unchanged except for imports or qualification
  required by the support-module move;
- all shared helpers have one target owner;
- no oracle delegates its expected value to the production action under test;
  and
- running the four target modules reproduces the baseline noise module's
  pass/conditional-skip disposition on the same environment.

Mechanical movement may change line numbers, import statements, and a helper's
module-qualified patch target. It must not change the observed test input or
acceptance condition.

## QuantityKernel Test Consolidation

Edit only `tests/test_kernel_geometry_and_quantity.py`.

Delete the complete module-level `range(70)` / `setattr(...)` method-generation
block and its `_case_index` / `case_index` machinery. The source must contain
no dynamically attached `test_quantity_kernel_construction_*` method.

Retain `_LEAF_CASES` as one explicit seven-entry table covering exactly:

```text
DarkCountRate
SmearingWidth
TimingJitter
DirectCrosstalk
DelayedCrosstalk
Afterpulse
Pulse
```

Fold the meaningful generated-construction assertions into the existing
`test_all_concrete_kernel_leaves_are_fieldless_and_immutable()` method. For
each leaf, under a named `subTest(leaf=leaf.__name__)`, the test must prove:

```text
exact constructed type
empty conditioning axes
exact supplied operation axes
owned tensor dtype torch.float64
requires_grad is False
absence of __dict__
rejection of arbitrary attribute assignment
```

Scalar leaves continue to receive a valid scalar quantity and operation-valued
leaves continue to receive their explicit valid tensor-shaped quantity. One
valid case per leaf is sufficient. Do not generate extra copies by changing
only a scalar magnitude or test index.

Every other existing method in `QuantityKernelContractTest` remains, including
the quantity snapshot, read-only quantity view, operation-target uniqueness,
invalid domains/geometry, Config generation relationship, coordinate
permutation, unequal-coordinate rejection, combined conditioning,
missing-role failure, and pulse-convolution proof.

The consolidation must not weaken the all-seven leaf set into a shared base
class assertion or check only a subset selected by inheritance.

## Runtime Test Consolidation

Edit only `tests/test_runtime_action_ownership.py`.

Delete the complete module-level `range(10)` / `setattr(...)` method-generation
block. Delete every no-op:

```python
self.assertEqual(index, index)
```

Add one ordinary class method:

```python
def test_compiled_runtime_contains_expected_execution_facts(self) -> None:
    ...
```

It calls `_prepared()` once and proves:

```python
runtime.charge.correlated_avalanche_generations == 0
runtime.pure_waveform.sample_offsets == (0,)
```

The four existing ownership methods remain:

```text
test_runtime_records_are_frozen_slotted_dataclasses
test_runtime_contains_no_config_quantity_kernel_or_distribution
test_pulse_is_materialized_once_in_requested_dtype
test_runtime_records_have_no_execution_methods
```

No production reflection, Config retention, extra Runtime field, or execution
method is added merely to make the consolidated test convenient.

## Protected Bytes And Contracts

Every production, dependency, environment, demo, and living-document byte is
protected:

```text
tensor_dslab/
pyproject.toml
create_environment.sh
demos/
README.md
AGENTS.md
CONTRIBUTING.md
LICENSE
docs/api.md
docs/architecture/
docs/design.md
docs/decisions.md
docs/overview.md
docs/parity.md
docs/validation.md
all historical docs/implementation records
```

Also protected:

```text
tests/__init__.py
tests/typing/
tests/test_environment_script.py
tests/test_runtime_kernel_alignment.py
tests/test_package_contracts.py
every other test path not listed in the exact allowlist
```

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

The exact TensorCore pin, source/archive identities, package topology, product
module topology, production import graph, no-cycle result, Runtime-empty
facades, and retired-path invariants remain unchanged. They may be verified
from parent identity rather than rewritten into new candidate tests.

## Exact Candidate Allowlist

The complete Implementation candidate may change only:

```text
tests/test_noise_waveform_product.py
tests/_noise_waveform_support.py
tests/test_noise_waveform_branches.py
tests/test_noise_waveform_psd_preparation.py
tests/test_noise_waveform_psd_synthesis.py
tests/test_noise_waveform_statistics.py
tests/test_kernel_geometry_and_quantity.py
tests/test_runtime_action_ownership.py
docs/implementation/index.md
docs/implementation/maintenance_14_test_suite_curation.md
```

The old noise module must be deleted and the five new noise paths must be
added. The maximum changed-path scope is ten paths. A move-expanded endpoint
inventory may report the retired and added paths separately but must not widen
the allowlist.

The work order and index may receive only exact candidate identity, focused
evidence, and self-effecting lifecycle synchronization. No other current or
historical document is allowlisted.

Any required change outside this list is a Design stop. Implementation must
not infer authority from another untidy test, nearby documentation, or a
potential future cleanup.

## Test And Static Acceptance

### Focused source gate

Implementation runs, with bytecode disabled, at least:

```text
tests.test_noise_waveform_branches
tests.test_noise_waveform_psd_preparation
tests.test_noise_waveform_psd_synthesis
tests.test_noise_waveform_statistics
tests.test_kernel_geometry_and_quantity
tests.test_runtime_action_ownership
tests.test_runtime_kernel_alignment
tests.test_package_contracts
```

The source form must:

- pass every non-CUDA test;
- retain only the established conditional CUDA skips;
- discover exactly eighteen noise methods in the `7 / 4 / 4 / 3` partition;
- discover one table-driven all-leaf construction/immutability method with
  seven explicit subtests;
- discover one compiled-Runtime execution-facts method;
- discover no `test_quantity_kernel_construction_*` methods;
- discover no `test_compiled_runtime_00` through
  `test_compiled_runtime_09` methods;
- contain no `assertEqual(index, index)` or equivalent no-op assertion; and
- contain no duplicate noise helper or oracle definition across target files.

### Complete package gate

The final immutable candidate must pass:

- the complete TensorDSLab source-form suite;
- the complete suite from an extracted canonical commit-bound TensorDSLab
  archive;
- Pyright `1.1.411` in Python `3.14` mode against source and archive;
- the exact TensorCore dependency negative fixture with `82` intended
  diagnostics and no incidental diagnostic;
- import isolation, public facade, retired-path, module-docstring, and package
  contract tests already present in the unchanged suite;
- diff, scope, mode, privacy, bytecode, cache, build, dist, and egg-info hygiene
  checks; and
- Markdown link/fence checks for the two changed documentation records.

Validation records actual run/pass/conditional-skip totals. A lower total is
expected from the exact redundant-method removal and is not a finding. A
missing substantive method, assertion, subtest leaf, scientific case, or
conditional-device gate is a finding.

### Baseline-to-candidate proof

Validation must use the exact parent `af23b129...` as the comparison source and
produce a machine-readable or reviewable inventory proving:

- all eighteen noise class/method identities are retained once;
- the old and new noise methods have behaviorally equivalent AST/source after
  allowing only import/support qualification and file-location changes;
- the frozen seeds, axes, configs, dtypes, sample sizes, statistical
  tolerances, expected values, exception categories, patch targets, address
  identities, and conditional skip predicates are unchanged;
- all seven `_LEAF_CASES` entries remain and every common constructor assertion
  is present in the consolidated method;
- both Runtime facts remain in the one replacement method;
- the only intentional method-count decrease is the redundant generated-copy
  removal; and
- every protected production and test byte is parent-identical.

If a purely syntactic AST comparison cannot account for an import move,
Validation must inspect and report the exact normalized difference rather than
silently accepting it.

## Evidence Cadence

This tests-only maintenance uses evidence proportional to changed risk.

### Implementation handoff

Implementation provides:

- exact commit/tree/parent and clean branch identity;
- exact ten-path-or-smaller allowlist diff and both diff checks;
- the focused source gate;
- the baseline-to-candidate class/method/helper inventory;
- Pyright zero diagnostics;
- source scans for generated tests, no-op assertions, duplicate helpers,
  forbidden production changes, and scope;
- exact protected-byte identity; and
- no wheel, sdist, real Conda environment, notebook execution, or CUDA
  requirement.

Implementation must not claim complete package clearance from the focused
handoff.

### Validation final-candidate gate

Validation runs one complete gate on the final immutable candidate:

1. verify exact commit/tree/parent, allowlist, protected bytes, deletions and
   additions, diff checks, and clean state;
2. verify the exact TensorCore `0.21.0` dependency source/archive identity;
3. run the focused source gate and the exact baseline-to-candidate preservation
   proof;
4. run the complete source suite once and the complete extracted-archive suite
   once;
5. run Pyright in both forms and the exact dependency negative fixture;
6. verify public imports/facades, archive test inclusion, documentation,
   privacy, and repository hygiene; and
7. record the existing unavailable-CUDA qualification without running CUDA.

Validation does not rebuild the TensorDSLab wheel or sdist, create a Conda
environment, reinstall dependencies, execute `create_environment.sh`, execute
the demo script or notebook, or reconstruct unrelated artifact evidence when
all artifact-input package/metadata/demo/environment bytes are proven
parent-identical. The extracted commit-bound archive is sufficient to prove
the reorganized tests remain runnable outside the checkout.

If a correction changes a test executable byte, rerun its focused gate, the
preservation proof affected by it, and the complete source/archive suite. If a
correction changes only truthful evidence prose, perform an exact
byte/scope/truth recheck without rerunning unchanged executable evidence.

### Independent Review

Review is risk-based. It must independently inspect:

- the exact noise responsibility split and private support ownership;
- one-for-one preservation of all eighteen substantive noise methods;
- independence of numerical/statistical oracles from production code;
- exact all-seven QuantityKernel coverage after generated-copy removal;
- exact preservation of both compiled Runtime facts;
- absence of dynamic method attachment and no-op assertions;
- unchanged protected production, dependency, environment, demo, and other
  test bytes;
- complete source/archive/typing evidence consistency; and
- exact scope, privacy, diff, and cleanliness.

Review may rely on exact complete Validation execution without rebuilding
artifacts or environments absent a concrete discrepancy. A finding must name a
lost or weakened obligation, incorrect split, scope violation, or evidence
defect; it does not authorize unrelated test or documentation curation.

## Lifecycle And Merge

The exact Design authority is the committed form of this work order and its
synchronized index entry accepted by the user. The persistent Implementation,
Validation, and Review routes must verify their role, workspace, exact parent,
and return path before acting.

The route is:

```text
Design authority
    -> focused Implementation candidate and handoff
    -> one complete final-candidate Validation gate
    -> risk-based independent Review
    -> final same-byte Design approval
    -> Review-owned clean git merge --ff-only to local main
    -> identity, diff, and cleanliness recheck
```

Candidate commits are immutable. A correction is a new direct child and reruns
only the evidence affected by changed bytes, subject to the complete
final-candidate rule above.

Before the complete same-byte gate and clean fast-forward, the status is the
latest completed fixed-byte handoff. After the exact cleared candidate appears
unchanged on local `main`, Maintenance 14 is **Merged / Closed**.

No separate evidence-only closeout commit is required merely to restate a
successful fast-forward. Push is not part of this work order. A later ordinary
push may be separately authorized without repeating package gates if the
pushed bytes are the exact already-cleared local main.

## Hard Stops

Stop and return to Design if:

- preserving a noise test requires changing its scientific input, oracle,
  tolerance, seed, sample size, expected value, or conditional-device rule;
- the split creates cyclic test imports, duplicate helpers, or a test
  discovery dependency on import order;
- a target test can pass after its production behavior is materially changed
  because an expected-value calculation now delegates to that behavior;
- a kernel leaf, Runtime fact, substantive assertion, or failure-before-words
  boundary would be lost;
- a production, dependency, metadata, environment, demo, notebook, typing
  fixture, living document, or non-allowlisted test must change;
- a new test framework or dependency appears necessary;
- exact TensorCore `0.21.0` source/archive forms disagree;
- the current unavailable-CUDA tests require semantic edits merely to pass on
  CPU;
- CUDA, deployment, release, compatibility, or documentation-curation scope
  becomes necessary; or
- the candidate cannot remain an exact descendant of closed Maintenance 13.

## Completion Criteria

Maintenance 14 is complete only when:

- the old noise monolith is absent;
- the four cohesive noise test modules and one private support module are the
  only replacement paths;
- all eighteen noise methods and every substantive obligation are retained
  exactly once;
- shared fixtures and independent oracles have one narrow owner;
- the seventy generated kernel methods are gone and one seven-subtest method
  proves their meaningful common contract;
- the ten generated Runtime methods and no-op assertions are gone and one
  ordinary method proves both retained facts;
- no other test or production responsibility changes;
- focused and complete source/archive suites pass;
- Pyright and the exact negative fixture pass;
- protected-byte, import, facade, diff, documentation, privacy, and hygiene
  gates pass;
- the candidate stays inside the exact ten-path allowlist;
- unavailable CUDA remains explicit and unclaimed;
- independent Review returns no finding;
- Design approves the exact same bytes; and
- Review fast-forwards the exact candidate cleanly to local main.
