# Maintenance 3 Environment-Qualified Stochastic Continuity Work Order

Status: **Candidate / Validation pending** while these exact bytes are absent
from `main`. If they are present unchanged on `main`, Review's fast-forward is
complete and the state is **Merged / Design acceptance pending** until this
work order and the implementation index record **Merged / Closed**.

Stable work-order key:
`TensorDSLab/maintenance-3-environment-qualified-stochastic-continuity`.

This focused maintenance corrects the applicability of two existing
TensorDSLab eager-CPU continuity fixtures. It changes tests and synchronized
documentation only. It does not change TensorDSLab production code,
TensorCore, the selected dependency, random addresses, distribution
algorithms, scientific equations, or accepted stochastic results.

## Objective

Make the frozen Maintenance 2 stochastic literals enforce exactly the
environment boundary under which they were recorded, while retaining the
portable and same-stack guarantees already accepted by TensorCore and
TensorDSLab.

After this maintenance:

- Threefry raw words and fixed-point uniform conversion remain exact wherever
  their accepted TensorCore contract says they are exact;
- the Maintenance 2 completed-distribution and completed-product hexadecimal
  literals remain unchanged and exact on their recorded reference stack;
- another accepted stack proves exact replay on that same unchanged stack,
  together with the existing dtype, shape, range, finiteness, storage,
  immutability, and statistical contracts, rather than being required to
  reproduce macOS transcendental rounding;
- no portable ULP tolerance, alternate Della golden profile, skip, expected
  failure, or silent weakening is introduced; and
- Stage 8 may later restart from a clean corrected package baseline and run
  its real-CUDA evidence from the beginning.

This is a test-contract correction. The stopped Stage 8 run found no evidence
of an RNG-address, RNG-word, scientific, production, dependency, or CUDA
implementation defect.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`.
The user accepted this focused correction and its Stage 8 restart sequence on
2026-07-21. That authorization does not broaden any exclusion below.

The exact starting package state is:

```text
repository:              TensorDSLab
reference:               main
starting commit:         e7207a0cafe9ff4d68253518aabc0e462173e69d
Stage 7 implementation:  6dd55024685013fb9412a7247d3ddde7be1a3177
package version:         0.1.0
```

The selected dependency remains:

```text
repository:       TensorCore
commit:           4708bf2ca063a1bcd37a30a342733b9e3dbe9f59
direct parent:    0e72f0e69cf9140b692d408e49a504cbdcb101b7
package version:  0.9.0
archive SHA-256:  f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd
```

The Design branch is:

```text
codex/maintenance-3-environment-qualified-rng-fixtures-design
```

The Implementation branch is fixed as:

```text
codex/maintenance-3-environment-qualified-rng-fixtures
```

Implementation creates that branch from the exact committed Design authority.
Raw platform route identifiers remain private and must not enter committed
files.

Package governance remains:

```text
package_adoption_state:     Adopted
conformance_finding:        Not evaluated
coordination_status:        Deferred
registry_storage_profile:   Disabled
maintenance_3:              topology-dependent lifecycle described below
stage_8:                    Stopped / superseding authority required
```

Design committed exact authority
`37cd6403b66107ccd24acd7bf1e50c63f0599313` as the direct child of starting
`main`, reverified the persistent package-owned Implementation, Validation,
and Review routes, and explicitly dispatched the fixed Implementation branch.
While this candidate is absent from `main`, its state is **Candidate /
Validation pending** and neither later gate is implied. If Review
fast-forwards these exact bytes to `main`, the state becomes **Merged / Design
acceptance pending**; only the later two-document Design closeout may record
**Merged / Closed**.

The permitted execution-state vocabulary is:

```text
Undispatched
Dispatched / Active
Candidate / Validation pending
Returned
Validation-cleared / Review pending
Review-cleared / Merge authorized
Merged / Design acceptance pending
Merged / Closed
Returned to Design
Blocked
```

Lifecycle documentation must use the applicable state without treating user
authorization, a candidate commit, a passing local command, or role activity
as clearance by a later gate.

## Stage 8 Stop Evidence

The earlier Stage 8 Design authority is exact commit
`84802c1f2c89a6a5deeec305ce7bb2cd9ad2e829`. Its frozen executable-input
commit is exact direct child
`728840bf2858c861104d5f7bb3cdbb4e3e1361b5`. Both remain immutable evidence of
the stopped attempt; neither is a Maintenance 3 starting point or an accepted
Stage 8 result candidate.

On the frozen Della environment, before any Stage 8 focused suite, candidate
suite, benchmark, profiler, or accepted measurement, the unchanged package
suite reported:

```text
tests run:       188
passed:          186
failed:          2
skipped:         0
elapsed:         84.475 seconds
suite-log SHA:   3880ab5dc96c79febaea8d95e45d80b7a7bb7014b5b561fb468215c706492a9d
```

The two failures were:

1. the fourth float32 Gaussian value was `3f06517c` rather than the frozen
   macOS value `3f06517b`; and
2. its completed float32 white-noise consequence was `3e499156` rather than
   `3e499154`.

The preceding fixed-point uniform assertions executed exactly. Because each
`unittest` method stopped at its first failed assertion, this evidence does
not claim that later assertions in either method executed.

The stopped environment was:

```text
OS / architecture:       RHEL 9.8 / x86_64
Python:                  3.11.15
PyTorch:                 2.12.1+cu126
execution:               eager CPU for the failed fixtures
allocated device:        NVIDIA A100-PCIE-40GB, CC 8.0, MIG disabled
torch CUDA runtime:      12.6
driver / compatibility:  610.43.02 / 13.3
```

The observed last-bit differences are consistent with a different accepted
CPU implementation of the float32 `log`, square-root, sine, or cosine steps in
Box-Muller. They are not promoted to normative Della literals, and their
observed size does not create a portable ULP bound.

## Source Precedence And Design Finding

Implementation, Validation, and Review must read and reconcile:

- `AGENTS.md` and `CONTRIBUTING.md`;
- [Rebuild Architecture](../architecture/rebuild.md), especially RNG And
  Positional Repeatability;
- [Validation](../validation.md);
- [Parity](../parity.md);
- [Maintenance 2](maintenance_2_rng_and_product_module_ownership_migration.md),
  especially Behavior And Continuity Contract and Frozen Eager-CPU Continuity
  Fixtures;
- the stopped Stage 8 authority at exact commit
  `84802c1f2c89a6a5deeec305ce7bb2cd9ad2e829`; and
- TensorCore's public random architecture and API at exact commit
  `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`.

Maintenance 2 explicitly binds its literal fixture table to:

```text
Python:     3.13.11
PyTorch:    2.12.1
OS:         macOS 15.7.4 arm64
execution:  eager CPU
```

It also states that completed transcendental results outside the accepted
exact boundary remain statistical. TensorCore `0.9.0` independently states
that completed Gaussian, Poisson, and binomial results repeat on one unchanged
backend and eager execution mode, where unchanged includes the Torch minor,
runtime, and device implementation; it does not promise completed-distribution
bitwise identity across different backend implementations.

The stopped Stage 8 work order correctly required a platform-sensitive
protected-suite failure to return to Design. Its simultaneous `188/188`
unchanged-suite requirement over-applied the Maintenance 2 macOS literals to a
different Linux/x86_64 stack. Maintenance 3 corrects that test applicability
before any Stage 8 measurement. It does not revise historical Maintenance 2
evidence.

## Accepted Continuity Classes

### Universal Exact Values

The selected TensorCore contract continues to own exact Threefry raw-word and
fixed-point-uniform behavior over its accepted device matrix. TensorDSLab's
public-consumer continuity fixture keeps its exact fixed-point uniform literals
active on every accepted test stack.

TensorDSLab must not import or call a private TensorCore block generator,
uniform helper, implementation class, or protected module to prove this. The
public `CounterRng`/`Threefry4x32`, `RngKey`, `logical_positions`, and public
distribution methods are the only downstream RNG surfaces.

### Reference-Stack Exact Literals

The existing Maintenance 2 hexadecimal literals remain byte-for-byte
unchanged. They are asserted only when the actual test environment exactly
matches every recorded identity component:

```text
Python version:       3.13.11
PyTorch version:      2.12.1
PyTorch CUDA build:   none (`torch.version.cuda is None`)
operating system:     Darwin / macOS 15.7.4
machine architecture: arm64
execution:            eager CPU
```

Partial matches do not activate the literal branch. A future platform profile
requires a separate predeclared Design decision; Maintenance 3 does not add
one.

The reference-only class includes every completed operation in the frozen
tables whose evaluation uses implementation-dependent transcendentals or is
downstream of them:

- Gaussian values;
- Poisson and binomial results;
- completed white and PSD noise; and
- completed Charge.

The profile gate is test support, not package API or production runtime
behavior. It uses only public Python/platform and PyTorch version information.
It must not inspect private Torch dispatch state or modify CPU capability,
threading, math-library, autocast, compiler, or deterministic-algorithm
settings to chase the old bits.

The intended exact test-only predicate is equivalent to:

```python
(
    sys.version_info[:3] == (3, 13, 11)
    and str(torch.__version__) == "2.12.1"
    and torch.version.cuda is None
    and platform.system() == "Darwin"
    and platform.mac_ver()[0] == "15.7.4"
    and platform.machine() == "arm64"
)
```

The request tensors themselves establish eager CPU execution. Git commits and
package archive hashes remain workflow gates rather than runtime test inputs.

### Other Accepted Stacks

On a stack that does not exactly match the recorded Maintenance 2 profile,
every completed fixture still executes. It must prove:

- exact `torch.equal` replay from two independently constructed equal RNG
  instances and independently executed calls on the same unchanged stack;
- exact dtype, shape, device, and requested count behavior;
- finite Gaussian/noise/Charge values;
- exact `torch.int64`, shape, and CPU device for Poisson and binomial results;
- Poisson values in `0 <= result <= 2**53 - 1`;
- elementwise binomial values in `0 <= result <= counts`;
- source and configuration immutability;
- generated-product freshness and documented storage independence; and
- every existing analytic, invariant, and statistical contract elsewhere in
  the package suite.

The second result must be produced independently. Aliasing one result,
comparing a tensor with itself, cloning/copying the first result, reusing a
cached expected tensor, or deriving the second expectation from the first is
not evidence.

Validation and Review run the relevant test process more than once on each
evidence stack. Exact replay is scoped to an unchanged stack; it is not a new
cross-OS, cross-architecture, cross-PyTorch-build, CPU/CUDA, compiled/eager, or
device-model bitwise promise.

### Cross-Backend Completed Values

Completed Gaussian/noise and completed count/distribution consumers retain
their already accepted invariant and statistical CPU/CUDA comparison. Raw
words and fixed-point uniforms retain their exact accepted cross-backend
contract. Maintenance 3 adds no direct CPU/CUDA bitwise requirement for a
completed transcendental value.

## Required Test Correction

Modify only `tests/test_rng_ownership_migration.py` among existing test files.
Keep its nine test methods and the full package's 188-test count unchanged.
Private test-only constants or helpers may be introduced in that file when
they make the two contracts explicit.

### Public TensorCore Distribution Continuity

`test_public_tensorcore_distribution_continuity` must:

1. retain the exact public fixed-point uniform payload assertions on every
   accepted stack;
2. execute Gaussian, Poisson, and binomial requests twice through independent
   equal public RNG instances on every stack;
3. require exact same-stack replay plus their public result invariants;
4. additionally assert the unchanged Maintenance 2 completed-distribution
   literals only on the exact reference profile; and
5. preserve keys, seed, positions, quantum, ordinal, count, dtypes, means, and
   masses exactly.

It must not call a private TensorCore surface, add an independently maintained
downstream copy of Box-Muller/PTRS/BTRS, accept an alternate algorithm, or
turn the observed Della payloads into expected values.

### Completed Noise And Charge Continuity

`test_completed_noise_and_charge_eager_cpu_continuity` must:

1. preserve every existing source, sampling, config, key, seed, dtype, and
   product request;
2. execute white noise, PSD noise, and completed Charge independently twice on
   every stack, using fresh equal RNG objects and fresh producer calls;
3. require exact same-stack replay and the existing field/value/storage/source
   invariants;
4. additionally assert the unchanged Maintenance 2 hexadecimal product
   literals only on the exact reference profile; and
5. leave all other product and sampler tests unchanged.

The test may not obtain replay by returning the first product twice, sharing
one produced tensor, or weakening a fresh-storage assertion.

### Proof Strength

Implementation's handoff and Validation's report must show that the committed
tests would reject at least these wrong behaviors:

- applying the macOS completed literals unconditionally on Della;
- suppressing the literal checks on the exact recorded macOS stack;
- changing any existing Maintenance 2 literal;
- making the two same-stack results nonidentical;
- comparing one result object or tensor with itself;
- cloning or copying the first result instead of executing the second request;
- replacing an exact fixed-point-uniform assertion with replay-only evidence;
- accepting a global one- or two-ULP tolerance;
- adding a Della-specific completed-value golden branch; and
- skipping or marking either existing test as an expected failure.

This may be demonstrated by focused mutation probes, exact branch evidence
from the two frozen environments, call-count/spy evidence that every completed
request reaches its public distribution or product producer exactly twice
through distinct equal RNG instances, structural inspection, or an equally
strong combination. Mutation code is role-private and is not committed.

## Evidence Environments

### Recorded Reference Stack

The local reference environment is frozen as:

```text
Python:       3.13.11
PyTorch:      2.12.1
OS:           macOS 15.7.4
architecture: arm64
execution:    eager CPU
CUDA:         unavailable
```

Against independently recreated exact TensorCore source and archive forms,
run at least:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:<tensorcore-root> \
/opt/miniconda3/bin/python -B -m unittest \
  tests.test_rng_ownership_migration

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:<tensorcore-root> \
/opt/miniconda3/bin/python -B -m unittest discover -s tests -v
```

Expected unchanged totals are:

```text
focused module:  9 run / 9 passed / 0 skipped
full suite:      188 run / 176 passed / 12 conditional CUDA skips
```

The reference run must prove that the literal branch actually executed; a
generic-replay-only pass on this stack is a failure.

### Frozen Della Stack

Use the same privacy-safe Della runtime and exact inventory identities already
verified by the stopped Stage 8 attempt:

```text
Python:                       3.11.15
PyTorch:                      2.12.1+cu126
OS / architecture:           RHEL 9.8 / x86_64
torch CUDA runtime:           12.6
GPU:                          NVIDIA A100-PCIE-40GB, CC 8.0, MIG disabled
driver / compatibility:      610.43.02 / 13.3
cuDNN:                        91002 / 9.10.2.21
cuFFT:                        11.3.0.4
source conda-explicit SHA:    4cee93f2ccbc1d3b22a94a8bd1d7f3efe7a9edfaa80e57e0e32188ce73ceb6cb
source pip-freeze SHA:        a7d3683e1f5c72189464bd99a567aa6eb4f8810163d4952a2731d3b5072abe77
clone conda-explicit SHA:     4cee93f2ccbc1d3b22a94a8bd1d7f3efe7a9edfaa80e57e0e32188ce73ceb6cb
clone pip-freeze SHA:         256a147f19056ff30d6b54be742ab87f544c1be5b3243b5354a38d6f6ed6db3a
```

Run the same focused and complete commands against independently reconstructed
exact TensorCore source and canonical archive forms. Every existing
conditional CUDA test must execute:

```text
focused module:  9 run / 9 passed / 0 skipped
full suite:      188 run / 188 passed / 0 skipped
```

Run the focused module in at least two fresh processes. Record that the generic
same-stack branch executed and that the Maintenance 2 reference-literal branch
did not. Do not run a Stage 8 benchmark, profiler, or new Stage 8 CUDA module in
this maintenance.

Implementation, Validation, and Review each use a separate allocated compute
job and independently reconstruct the immutable candidate plus exact
TensorCore source and archive forms. Each role verifies all four frozen runtime
inventory hashes before and after its commands. A changed inventory, candidate,
dependency form, or allocation stops that role's evidence.

All Della execution occurs inside an allocated compute job, never on a login
node. Reuse of the previously verified private transfer and environment
procedure does not make its private paths or raw route identifiers durable
package data.

### Static And Package Gates

Run Pyright `1.1.411` in standard mode against both exact dependency forms with
Python target `3.11`, using the already accepted separate static-tool runtime.
Require zero errors, warnings, and informations.

Also require:

- exact TensorCore commit, parent, version, package-root exports, and canonical
  archive hash;
- source/archive import isolation;
- `git diff --check`;
- no test-count change;
- no production, dependency, metadata, governance, or closed-work-order
  change;
- no new skip, expected failure, tolerance, alternate golden, private
  TensorCore import/call, global RNG, or environment mutation;
- no bytecode, cache, build, distribution, or egg-info artifact; and
- a clean fixed candidate.

## Design Authority And Candidate Allowlists

Relative to starting `main`, Design's committed authority changes exactly the
ten documentation paths below:

```text
AGENTS.md
CONTRIBUTING.md
docs/architecture/rebuild.md
docs/architecture/readout.md
docs/decisions.md
docs/implementation/index.md
docs/implementation/maintenance_3_environment_qualified_stochastic_continuity.md
docs/overview.md
docs/parity.md
docs/validation.md
```

Relative to that exact committed Design authority, Implementation's complete
candidate may change only:

```text
M  docs/implementation/index.md
M  docs/implementation/maintenance_3_environment_qualified_stochastic_continuity.md
M  tests/test_rng_ownership_migration.py
```

The cumulative starting-main-to-candidate set is therefore exactly:

```text
AGENTS.md
CONTRIBUTING.md
docs/architecture/rebuild.md
docs/architecture/readout.md
docs/decisions.md
docs/implementation/index.md
docs/implementation/maintenance_3_environment_qualified_stochastic_continuity.md
docs/overview.md
docs/parity.md
docs/validation.md
tests/test_rng_ownership_migration.py
```

Design's eight architecture/workflow documents outside the work order and
index are frozen after the authority commit. Implementation owns only the
bounded test correction plus truthful lifecycle synchronization in the work
order and index. If another path is required, stop and return to Design before
editing it.

## Protected Bytes And Exclusions

The following are protected:

- every `tensor_dslab/**` production byte;
- every existing test other than
  `tests/test_rng_ownership_migration.py`;
- `pyproject.toml`, package metadata, and the exact TensorCore pin;
- all closed work orders, including Maintenance 2;
- `docs/governance/**`; and
- the stopped Stage 8 authority and executable-input commits.

This maintenance adds no:

- public or private production API;
- scientific or statistical target change;
- RNG key, seed, address, word schedule, distribution, or affine mapping
  change;
- TensorCore edit, dependency move, fork, private import, or local generic RNG
  implementation;
- Della completed-value profile or portable ULP promise;
- skip, expected failure, xfail, retry-until-pass, environment override, or
  hidden exclusion;
- CUDA optimization, compile/fusion, benchmark, profiler, workspace, output
  reuse, or performance claim;
- TensorG4DS, TensorML, Reconstruction, IO, persistence, cache, or DAG work;
- compatibility, conformance, release, deployment, or production-readiness
  claim; or
- push.

## Branch, Loop, And Role Contract

Use the persistent package-owned roles:

```text
Design -> Implementation + Validation -> Review -> permitted corrections -> Review recheck
```

Implementation owns the feature branch and returns one immutable candidate.
Validation evaluates that exact commit against both dependency forms and both
evidence environments. The ordinary finite loop permits at most three
Implementation-to-Validation candidate dispatches and at most three
Validation-to-Implementation returns. Either exhausted bound, the same
substantive finding returned twice without convergence, an architecture
conflict, protected-byte need, or environment mismatch returns to Design.

Review begins only after Validation clears a fixed commit. Review independently
reconstructs the dependency forms, reruns the applicable local and Della gates,
checks the environment-branch behavior, and either returns exact findings or
fast-forwards clean `main` with `git merge --ff-only`. Review does not push.
If Review returns a finding, Design must authorize its exact bounded scope;
Implementation owns the correction, Validation clears the new fixed bytes, and
read-only Review rechecks them. Review does not edit the candidate.

Final Design performs an evidence-only closeout after the merge. That closeout
must be an exact direct child of the Review-cleared merged candidate and may
modify only the work order and implementation index. It changes no test,
production, dependency, metadata, governance, or other documentation byte and
records exact parent, name-status, diff, evidence, final state, and no-effects.

No Stage 8 measurement or new Stage 8 executable candidate may run in parallel
with this maintenance.

## Stop Conditions

Stop and return exact evidence to Design if:

- the starting commit, dependency, archive, branch, route, or environment is
  discrepant;
- the exact recorded macOS profile cannot execute its literal branch;
- Della cannot execute the generic same-stack branch with all existing CUDA
  tests active;
- any original Maintenance 2 literal would need to change;
- any same-stack replay is nondeterministic;
- exact fixed-point uniform values differ;
- a production, TensorCore, dependency, science, API, metadata, governance, or
  closed-work-order change appears necessary;
- a ULP tolerance, alternate Della golden, skip, expected failure, runtime
  override, or private API appears necessary;
- the full test count changes unexpectedly;
- the correction budget is exhausted; or
- completion would require excluded work.

Do not patch around a stop with a different runtime, forced CPU capability,
post-observation threshold, conditional exclusion, or hidden fallback.

## Completion Boundary And Stage 8 Handoff

Maintenance 3 is complete only when:

- the exact reference stack executes the unchanged literal payload branch;
- the exact Della stack executes the same-stack replay branch;
- source/archive focused, complete, and static gates pass in their applicable
  environments;
- the full package retains the 188-test count;
- no production, dependency, science, API, governance, or historical evidence
  byte changed;
- Validation and independent Review clear one immutable candidate;
- Review fast-forwards the exact candidate to `main` without a merge commit or
  push; and
- final Design records Merged / Closed with exact evidence and qualifications.

Completion authorizes no Stage 8 work by itself. Design must create a new
Stage 8 authority from the corrected merged `main`, explicitly supersede
`84802c1f...` and `728840bf...`, re-freeze its executable input, reverify the
Della environment, and separately dispatch the restart. Every correctness,
statistical, benchmark, profiler, and measurement result must then be produced
again from scratch; the stopped run contributes only diagnostic evidence.
The separately authorized restart receives a fresh finite role-loop budget
because the stopped attempt never reached Validation. Maintenance 3 Della
passes remain Maintenance 3 evidence only and do not count as Stage 8 results.
