# Proposed TensorCore Stage 15 Counter RNG And Distributions Work Order

Status: **Fulfilled by TensorCore Stage 15 / Historical consumer proposal /
Never TensorCore authority**.

TensorCore's adopted stable work-order key:
`TensorCore/stage-15-counter-rng-and-distributions`.

This document is the TensorDSLab-owned statement of demonstrated consumer
requirements that preceded TensorCore Stage 15. TensorCore independently
reconciled those requirements into its package-authoritative
`TensorCore/stage-15-counter-rng-and-distributions` work order and published
version `0.9.0` at exact `origin/main` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, with direct parent
`0e72f0e69cf9140b692d408e49a504cbdcb101b7`. The authoritative accepted
contracts are TensorCore's `docs/implementation/stage_15_counter_rng_and_distributions.md`
and `docs/architecture/random.md` at that exact commit. This historical
consumer proposal never authorized TensorCore work and is not an alternative
runtime specification. Proposal-time future-tense language below is retained
as historical evidence except where an exact accepted-contract correction is
needed for downstream continuity.

## Objective

Add the smallest generic positional RNG surface that can replace the complete
TensorDSLab Stage 5/6 private counter engine and count samplers without moving
detector science into TensorCore:

- immutable `RngKey` role identifiers;
- immutable stateless `CounterRng` invocation objects;
- the exact Random123 `Threefry4x32_R<20>` concrete implementation;
- row-major `logical_positions(...)` over arbitrary tensor shapes;
- public `uniform(...)`, `gaussian(...)`, `poisson(...)`, and `binomial(...)`
  methods; and
- the focused generic relationship requirement
  `require_same_dtype(*fields)`.

`require_same_dtype()` is an independent, separately testable sub-slice of the
same proposed dependency revision because it directly parallels TensorCore's
existing `require_same_axes()` and `require_same_device()` helpers. It must not
be coupled to RNG internals or become an opening for a broad validator
expansion. TensorCore Design may split it into a separate work order if its
package review finds a concrete lifecycle reason; TensorDSLab Maintenance 2
still waits for one cumulative commit containing both prerequisites.

The implementation is functionality-first and reference-first. Before this
proposal can be adopted, TensorCore Design must transplant the complete exact
eager executable mapping into package-authoritative TensorCore bytes; this
consumer record identifies the required mapping and continuity evidence but is
not itself a substitute for that TensorCore specification. The eventual work
order makes no compiler, fusion, kernel-count, target-temporary,
allocation-free, throughput, or accelerator-performance claim.

## Authority And Package Boundary

TensorCore Design owns every TensorCore architecture, API, implementation,
validation, review, version, and merge decision. TensorDSLab Design owns only
the consumer requirements and continuity boundary recorded here.

Before this proposal can become a TensorCore work order, TensorCore Design
must:

1. reverify the closed Stage 14 containing commit;
2. reverify the exact clean operative TensorCore `main`;
3. reconcile this proposal with TensorCore's `AGENTS.md`, `CONTRIBUTING.md`,
   public API, documentation, and package version;
4. decide the final stage number, design branch, implementation branch,
   allowed-file list, and exact source-of-truth commit;
5. record whether the proposed public signatures below are accepted unchanged
   or return every discrepancy to TensorDSLab Design; and
6. obtain a separate user authorization before dispatch.

The TensorDSLab evidence baseline for this draft is Stage 6 closed `main` at
`245e8155f66f51d061c680b8b220356689b24b60`, plus the uncommitted
documentation-only `codex/counter-rng-architecture` overlay. The exact
TensorCore baseline observed during this revision is clean local `main`
at `78e60abad96ab72eafb6a244c662f89d13f17599`, three commits ahead of
`origin/main`, with Stage 14 closed locally and package version `0.8.0`. The
earlier `b454d738...` state is historical evidence, not an alternative current
baseline. This downstream proposal still does not issue a TensorCore work
order: TensorCore Design must select `78e60ab...` or a later clean `main`
explicitly in its package-authoritative record.

If TensorCore Design selects that observed baseline, the proposed execution
coordinates are:

```text
starting main:         78e60abad96ab72eafb6a244c662f89d13f17599
design branch:         codex/stage-15-counter-rng-and-distributions-design
implementation branch: codex/stage-15-counter-rng-and-distributions
target version:        0.9.0
```

These coordinates remain proposed until TensorCore Design adopts its own
committed work order and the user separately authorizes dispatch.

Governance state is unchanged in both packages. Coordination remains Deferred
and is not an execution route. Raw task identifiers must not appear in either
repository.

## Demonstrated Generic Ownership

TensorCore owns:

- validation and packing of generic seed, key, position, quantum, and
  distribution-ordinal coordinates;
- exact counter-word generation;
- fixed-point conversion to floating uniforms;
- the Box-Muller mapping and parameterized Gaussian affine transformation;
- generic Poisson inversion/PTRS and binomial inversion/BTRS mechanics;
- deterministic attempt/term exhaustion;
- generic sampler input/result representability ceilings;
- result shape, dtype, device, freshness, and generic failure behavior; and
- generic same-dtype semantic-field relationships.

TensorDSLab retains:

- the scientific meaning and default assignment of each `RngKey`;
- which tensor lattice and dimension order defines positions for an operation;
- virtual generation/category position construction;
- physical rates, means, masses, delay kernels, and recovery laws;
- complete multinomial category order and final no-draw remainder;
- operation-level checked addition and accumulator-depth limits, configured
  generation limits, mechanism diagnostics, S1/S2 ledgers, and detector
  failure policy layered above TensorCore's generic sampler limits; and
- product dependency planning and simulation orchestration.

TensorCore must not import TensorDSLab, TensorML, TensorG4DS, or another
consumer to implement or test this stage. Consumer evidence may be translated
into neutral fixed fixtures, but no consumer class, config, stream name,
detector namespace, or product meaning becomes TensorCore API.

TensorCore's existing exclusion of DataLoader/domain/campaign “samplers”
remains intact. Here, Poisson and binomial sampling are generic numerical
operations over tensor-shaped positional addresses; they do not create data
pipelines, batches, datasets, schedules, or domain orchestration.

## Proposed Public Surface

The required public shape is:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import torch

from tensor_core.roots import TensorField


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RngKey:
    namespace: int
    stream: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterRng(ABC):
    seed: int

    @abstractmethod
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        ...

    @final
    def uniform(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        dtype: torch.dtype,
        quantum: int = 0,
        ordinal: int = 0,
        count: int = 1,
        include_zero: bool = True,
    ) -> torch.Tensor:
        ...

    @final
    def gaussian(
        self,
        *,
        mean: float | torch.Tensor,
        standard_deviation: float | torch.Tensor,
        key: RngKey,
        positions: torch.Tensor,
        dtype: torch.dtype,
        quantum: int = 0,
        ordinal: int = 0,
        count: int = 1,
    ) -> torch.Tensor:
        ...

    @final
    def poisson(
        self,
        *,
        mean: float | torch.Tensor,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int = 0,
    ) -> torch.Tensor:
        ...

    @final
    def binomial(
        self,
        *,
        counts: torch.Tensor,
        success_mass: float | torch.Tensor,
        failure_mass: float | torch.Tensor | None = None,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int = 0,
    ) -> torch.Tensor:
        ...


@final
class Threefry4x32(CounterRng):
    __slots__ = ()

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        ...


def logical_positions(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    ...


def require_same_dtype(*fields: TensorField) -> None:
    ...
```

The sketch is public-shape pseudocode. TensorCore production uses its accepted
package-internal `TensorField` and `_require_field` imports.

TensorCore Design owns the protected counter hook and private module layout.
The accepted `_generate_block(...)` hook returns same-device
`torch.int64`-carried unsigned 32-bit words with shape
`positions.shape + (4,)`; TensorCore Design must freeze its complete subclass
obligations before adoption. Malformed custom overrides are unsupported rather
than an adversarial runtime-policing target.
The proposed production shape is one real public `tensor_core/random.py`
module plus package-root exports; raw-word helpers and distribution internals
remain protected/private. There is no public `standard_normal`, `random_block`,
raw-counter constructor, raw-word function, sampler-control record, stream
enum, or mutable generator.

The five proposed package-root additions are:

```text
RngKey
CounterRng
Threefry4x32
logical_positions
require_same_dtype
```

Against TensorCore `0.8.0`'s exact 18-name export surface, the proposed
cumulative surface contains 23 names. TensorCore Design must rederive and
record the exact complete export tuple against its selected baseline rather
than inheriting this count blindly.

## Value Records And Address Domain

`RngKey` is an immutable value record. `namespace` and `stream` are exact
non-boolean Python `int` values in `[0, 2**32)`. Equality and hashing use those
two fields. TensorCore does not interpret either component.

`CounterRng` is an immutable stateless ABC carrying one exact non-boolean
Python `int` seed in `[0, 2**64)`. Equality requires the same exact concrete RNG
class and equal seed; different algorithms are not equal merely because their
seeds match. Hashing follows the frozen-dataclass seed hash and is consistent
with equality; hash collisions between different concrete RNG classes are
permitted. It owns no mutable cursor, cache, tensor, workspace, generator,
device, or stream. Every conforming subclass must remain immutable and stateless;
immutable algorithm-definition fields require a separately accepted concrete
type and participate in that type's equality. Reusing one instance replays the
same positional realization and is safe for concurrent logical reuse subject
to ordinary backend execution rules. Subclasses select algorithms; they do not
silently change one concrete class's accepted word mapping.

The protected block hook accepts an actual `RngKey`, an ordinary
`torch.strided` `torch.int64` tensor of nonnegative positions, exact non-boolean
`quantum` and `block` integers in `[0, 2**32)`, and returns a same-device
ordinary `torch.strided` `torch.int64` tensor with shape
`positions.shape + (4,)`. Every returned value represents one unsigned 32-bit
word in `[0, 2**32)`. Public final distribution methods own request validation;
they may trust a conforming protected override after cheap structural checks.
Tensor subclasses, custom Torch dispatch, mutable subclass state, a
shape/device/dtype mismatch, or out-of-range hook values are unsupported
extension behavior and receive no stable exception promise.

The accepted address components are:

```text
0 <= seed < 2**64
0 <= namespace < 2**32
0 <= stream < 2**32
0 <= logical_flat_position < 2**63
0 <= source_quantum_ordinal < 2**32
0 <= raw_word_ordinal < 2**34
```

For seed `s`, key `(namespace=d, stream=g)`, position `p`, quantum `q`, and
raw-word ordinal `r`:

```text
low32(x)  = x & 0xffff_ffff
high32(x) = (x >> 32) & 0xffff_ffff
block = r // 4
lane  = r % 4

key[0] = low32(s)
key[1] = high32(s)
key[2] = g
key[3] = d

counter[0] = low32(p)
counter[1] = high32(p)
counter[2] = q
counter[3] = block

raw_word = Threefry4x32_20(counter, key)[lane]
```

The split is numerical and independent of host byte order. Equal packed
address and lane recover every accepted address component; distinct addresses
may naturally return equal 32-bit values.

## `logical_positions(...)`

`logical_positions(...)` validates an exact tuple of exact non-boolean
nonnegative Python dimensions and an accepted Torch device, checks
`prod(max(dimension, 1))` is strictly less than `2**63`, then returns a fresh contiguous
`torch.int64` tensor on that device. The proposed initial device matrix is
eager CPU and CUDA; TensorCore Design must ratify it before dispatch.

For shape `(n0, ..., nk)`, values are row-major logical flat positions reshaped
to that exact shape. They depend on shape and dimension order, not coordinates,
strides, contiguity, or storage offsets. Shape `()` returns scalar position
zero. Any zero-sized dimension returns an empty tensor and consumes no random
words. A functionality-first implementation may materialize the complete
position tensor; later fusion may derive identical positions without changing
their meaning.

## Threefry Word Contract

`Threefry4x32` means standard Random123 `Threefry4x32_R<20>`. The normative
external definition is Random123 `1.14.0` at immutable commit
[`726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`](https://github.com/DEShawResearch/random123/commit/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13),
including
[`threefry.h`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/include/Random123/threefry.h),
its rotation constants, key-injection schedule, `0x1bd1_1bda` parity constant,
output tuple order, and published
[`kat_vectors`](https://github.com/DEShawResearch/random123/blob/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13/tests/kat_vectors).

At minimum, exact tests include:

```text
counter: 00000000 00000000 00000000 00000000
key:     00000000 00000000 00000000 00000000
output:  9c6ca96a e17eae66 fc10ecd4 5256a7d8

counter: ffffffff ffffffff ffffffff ffffffff
key:     ffffffff ffffffff ffffffff ffffffff
output:  2a881696 57012287 f6c7446e a16a6732

counter: 243f6a88 85a308d3 13198a2e 03707344
key:     a4093822 299f31d0 082efa98 ec4e6c89
output:  59cd1dbb b8879579 86b5d00c ac8b6d84
```

Any later incompatible algorithm, round count, or address packing requires a
new concrete type/version. It must not silently change `Threefry4x32`.

## Distribution Contract

Every method validates its complete generic request before counter generation
or output writes. Every successful path returns a fresh, contiguous,
non-aliasing `torch.strided` tensor on the positions device. Inputs are never
mutated, moved, cast, detached, or broadcast beyond the explicitly documented
final variate dimension. Failure before backend launch has no RNG or output
effect; the stateless RNG itself has no state to roll back.

Public floating dtypes are exactly `torch.float32` and `torch.float64`.
Floating results use the requested dtype. Poisson and binomial return
`torch.int64`. Distribution methods do not read or modify PyTorch global RNG,
construct `torch.Generator`, call `torch.poisson`, or use private PyTorch RNG
operations.

Every public request accepts an actual `RngKey` and an ordinary
`torch.strided` `torch.int64` positions tensor containing only nonnegative
values. Positions may have arbitrary rank and valid strides; their values, not
physical storage offsets, are the logical addresses. Tensor law operands and
counts are likewise ordinary `torch.strided` tensors; tensor subclasses and
custom Torch dispatch are outside the initial contract. All tensor operands
are borrowed read-only and must not require gradients. Arbitrary valid
ordinary strided read layouts, including expanded or internally overlapping
views, are accepted; input-to-input storage aliasing is permitted because no
input is written. `quantum`, `ordinal`, and `count` are exact
non-boolean Python integers. `quantum` lies in `[0, 2**32)`, `ordinal` is
nonnegative within the distribution-specific raw-word slice, and `count` is
positive. `include_zero` is exactly `bool`. Floating `dtype` is exactly
`torch.float32` or `torch.float64`. Appended result shape spans use
`prod(max(dimension, 1))` and are strictly less than `2**63`. Unsupported device families fail before word generation or
result allocation.

All package writes needed to construct a successful result are initiated or
enqueued before return, and TensorCore performs no later write through an alias
to returned storage. Normal backend stream ordering applies. The API neither
promises result completion at Python return nor synchronizes merely to wait for
generated values. Eager validation of device-resident positions, counts, or
law values may synchronize when required to decide a documented public error
before word generation; TensorCore Design must record any backend-specific
qualification rather than silently changing failure timing.

### Uniform

`include_zero=True` selects the accepted closed-open fixed-point conversion;
`False` selects open-open conversion. Closed-open includes exact zero and
excludes one. Open-open excludes both endpoints. `count == 1` returns exactly
`positions.shape`; larger counts append one final count dimension.

Distribution ordinals count returned variates:

```text
float32 uniform v -> raw word v
float64 uniform v -> raw words 2v, 2v + 1
```

Checked slice bounds require `ordinal + count <= 2**34` for `float32` and
`ordinal + count <= 2**33` for `float64`, plus a valid appended output shape
and signed Torch `numel`.

Let `w0` and `w1` be exact unsigned 32-bit words carried by `torch.int64`.
With autocast disabled, the fixed-point mappings are exactly:

```text
float32 closed-open: float32(w0 >> 8) * float32(2**-24)
float32 open-open:   float32(0.5 + float32(w0 >> 9)) * float32(2**-23)

float64 closed-open:
    float64(w0 * 2**21 + (w1 >> 11)) * float64(2**-53)
float64 open-open:
    float64(0.5 + float64(w0 * 2**20 + (w1 >> 12))) * float64(2**-52)
```

The stated grouping and target-dtype conversions are normative; an algebraic
rewrite is accepted only if it produces the same values over the complete raw
word domain on every accepted backend.

### Gaussian

The public name is `gaussian`, with explicit mean and standard deviation.
There is no public standard-normal method. Internal standard-normal variates
use the accepted ordered Box-Muller pair mapping:

```text
pair      = variate_ordinal // 2
component = variate_ordinal % 2

float32 pair j -> raw words 2j, 2j + 1
float64 pair j -> raw words 4j, 4j + 1, 4j + 2, 4j + 3
```

Order is `z0(pair 0)`, `z1(pair 0)`, `z0(pair 1)`, `z1(pair 1)`, and every
slice equals stacking its scalar requests, including odd starts. Scalar
parameters are exact finite Python `float` values rounded once to the output
dtype. Tensor parameters have exactly `positions.shape`, requested dtype, and
positions device. They do not carry the appended count dimension and may not
require gradients. Standard deviation is nonnegative after representation.

For each pair, `u_radius` uses the open-open uniform mapping and `u_angle`
uses the closed-open mapping in the requested dtype. With autocast disabled:

```text
tau    = requested_dtype(math.tau)
radius = sqrt(requested_dtype(-2) * log(u_radius))
angle  = tau * u_angle
z0     = radius * cos(angle)
z1     = radius * sin(angle)
```

The multiplication inside the square root, tau rounding, angle
multiplication, and component order are normative. `count == 1` removes the
appended variate dimension exactly as Uniform does. Checked pair/word bounds
must account for an odd starting ordinal and the final requested component,
not merely `count // 2`. As with Uniform, `ordinal + count <= 2**34` for
`float32` and `ordinal + count <= 2**33` for `float64`.

The frozen affine branches are:

```text
standard_deviation == 0                -> mean
standard_deviation == 1 and mean == 0  -> Z
mean == 0                              -> standard_deviation * Z
otherwise                              -> mean + standard_deviation * Z
```

The general path multiplies before adding. A `gaussian(...)` invocation still
requests and generates the complete ordinary Gaussian word lattice when
represented standard deviation is zero, then returns the represented mean
exactly; zero scale is not an internal word-free shortcut. A consumer obtains
draw-free behavior only by skipping the complete call before TensorCore.
TensorCore checks a conservative maximum-radius finite-output envelope before
word generation. Returned values are finite and `requires_grad=False`;
stochastic autograd is deferred.

### Poisson

`positions` fixes output shape and device. A scalar mean is an exact Python
`float`; a tensor mean is exact-shaped/device-matched `torch.float64`.
Accepted means are finite in `[0, 1e8]`.

The executable mapping is the exact TensorDSLab Stage 6 mapping:

- exact zero is a word-free zero result;
- `0 < mean < 10` uses the frozen inversion recurrence, strict CDF comparison,
  and 64-term exhaustion rule;
- `10 <= mean <= 1e8` uses the frozen Hoermann PTRS constants, proposal,
  quick decisions, full log decision, one four-word block per attempt, and
  64-attempt exhaustion rule; and
- mixed tensors preserve each cell's original positional identity rather than
  compacting addresses by active algorithm branch.

TensorCore may reorganize the implementation but may not substitute
`torch.poisson`, a normal approximation, clipping, reseeding, a biased
fallback, or a different uncertainty decision. The high-precision gates and
fixed-word fixtures selected in TensorDSLab
`docs/architecture/rebuild.md`, sections “Selected RNG Distribution
Contracts,” “Poisson Count Sampling,” and “Validation Strategy,” are required
consumer evidence to reconcile into the package-authoritative TensorCore work
order.

Every PTRS proposal is inspected while still in binary64. A finite
nonnegative proposal above `2**53 - 1` raises `RuntimeError` immediately,
before integer conversion and before quick-accept, quick-reject, or full-log
decisions. It is not retried as an ordinary rejection. Every successful result
is an exact nonnegative integer no larger than `2**53 - 1` before conversion to
`torch.int64`. This is a generic sampler representability limit;
TensorDSLab's later accumulation and ledger ceilings remain downstream rules.

### Binomial

`counts` and `positions` are exact-shape/device-matched `torch.int64`; counts
are in `[0, 2**53 - 1]`. Scalar masses are exact Python `float`; tensor masses
are exact-shaped/device-matched `torch.float64`. Masses are finite and in
`[0, 1]`.

With `failure_mass=None`, `success_mass` is the represented probability and
TensorCore derives `1 - success_mass`. With both masses present, TensorCore
uses them directly as relative success and failure masses without first
forming a cancellation-prone complement. They need not sum to one. Both may
be zero only where `counts == 0`; otherwise their sum must be finite and
positive.

Zero counts, zero success probability, and unit success probability are
word-free. Nontrivial cells use the exact Stage 6 reduction/reflection,
small-mean inversion, stabilized BTRS proposal and acceptance mapping,
high-precision uncertainty gates, fixed one-block-per-attempt addressing, and
64-term/attempt exhaustion rules. Results remain within `[0, counts]` and
preserve original positional identity across mixed paths.

The inversion recurrence follows the accepted production association exactly:
`(probability * count_ratio) * mass_ratio`. The stale reassociated test-oracle
form `probability * (count_ratio * mass_ratio)` is not a continuity source; it
can cross a fixed-word acceptance boundary.

TensorCore does not expose multinomial categories. TensorDSLab continues to
perform sequential physical-category factorization through public
`binomial(...)` calls and assigns the final no-draw remainder.

## `require_same_dtype(...)` Independent Sub-Slice

The helper mirrors TensorCore's existing relationship conventions:

```python
def require_same_dtype(*fields: TensorField) -> None:
    checked = tuple(
        _require_field(field, f"require_same_dtype.fields[{index}]")
        for index, field in enumerate(fields)
    )
    if len(checked) <= 1:
        return
    expected = checked[0].tensor.dtype
    for field in checked[1:]:
        if field.tensor.dtype != expected:
            raise ValueError("fields must have the same dtype")
```

Every operand is validated before relationship comparison. Zero or one valid
operand is a no-op; well-formed unequal dtypes raise `ValueError`. The helper
does not coerce, cast, move, detach, reconstruct, allocate payload tensors, or
mutate anything. It makes no dtype allowlist, floating-only, shape, axes,
device, layout, stride, storage, value, gradient, or scientific-coherence
claim.

Do not add `require_same_shape()` for semantic fields in this work order:
equal valid TensorCore axes already determine equal field shape. Do not add a
raw-tensor compatibility helper, generalized predicate registry, coercion
framework, or dtype-policy object.

## Expected Package Shape

Subject to TensorCore Design reconciliation, the implementation should add one
real module:

```text
tensor_core/
  random.py
```

and update:

```text
tensor_core/__init__.py
tensor_core/validation.py
tests/test_counter_rng_contracts.py
tests/test_tensor_core_contracts.py
docs/api.md
docs/architecture/random.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/stage_15_counter_rng_and_distributions.md
docs/integration.md
docs/overview.md
docs/quickstart.md
docs/validation.md
AGENTS.md
CONTRIBUTING.md
README.md
CHANGELOG.md
pyproject.toml
```

TensorCore Design must convert this expected list into an exact allowlist at
the selected baseline. It should create no placeholder distribution modules,
consumer adapter, registry, RNG service, stream catalog, workspace, allocator,
or backend-specific public module.

`docs/architecture/random.md` is the proposed normative home for the complete
generic RNG contract. `docs/architecture/tensors.md` receives only the narrow
same-dtype relationship and documentation-map synchronization; it should not
become a random-number manual.

## Required TensorCore Design Closure Before Adoption

This consumer proposal deliberately does not let Implementation choose the
remaining TensorCore-owned details. TensorCore Design must replace each item
below with one exact contract before adopting or dispatching its work order:

1. the exact Gaussian finite-output envelope, including scalar/tensor
   evaluation order and the selected maximum Box-Muller radius for each dtype;
2. the initial accepted device matrix and exact unsupported-device behavior;
3. the protected `CounterRng` algorithm-extension hook, its input/output
   contract, and the unsupported behavior of malformed downstream overrides;
4. the exact logical-position and result-numel boundary, resolving every
   current `<= 2**63` versus `< 2**63` wording difference against Torch's
   signed element-count domain;
5. the exact `TypeError`/`ValueError`/`RuntimeError` taxonomy for malformed
   requests, unsupported devices, representability failures, and deterministic
   sampler exhaustion;
6. the complete uniform and Gaussian formulas, evaluation dtypes/order,
   endpoint conventions, append/slice behavior, zero-scale full-word behavior,
   and word schedules in package-authoritative TensorCore documentation;
7. the complete Poisson and binomial equations, constants, word/attempt
   schedules, proposal and acceptance ordering, high-precision uncertainty
   gates, representability checks, and exhaustion rules, copied from the exact
   closed TensorDSLab Stage 5/6 mapping rather than left as a downstream
   citation;
8. the complete immutable/stateless subclass and protected-hook obligations,
   including equality/hash semantics, the structural trust boundary, and
   concurrent reuse;
9. the law-tensor autograd boundary for Gaussian, Poisson, and binomial,
   retaining the proposed initial rule that tensor law operands requiring
   gradients are rejected and stochastic results have `requires_grad=False`;
10. the independent `require_same_dtype()` sub-slice's exact validation-first
    precedence and agreement with existing relationship-helper exception and
    import conventions.

Any other unresolved public or numerical choice returns to both package Design
authorities rather than becoming an Implementation decision.

## Required Evidence

The package-authoritative work order must require at least:

- exact `RngKey` and seed type/range/immutability tests;
- abstract/concrete construction and static typing for `CounterRng` and
  `Threefry4x32`;
- exact-class equality, hash consistency with permitted cross-class
  collisions, and immutable/stateless concurrent-reuse tests;
- malformed/out-of-range key, positions, quantum, block, ordinal, count, and
  `include_zero` boundary tests;
- conforming protected-hook structural tests plus abstractness/type-checking
  obligations, without promising adversarial override policing;
- Random123 known-answer tests and an independent scalar oracle;
- schema-v1 key/counter/lane packing boundaries, rollover, and injectivity;
- arbitrary-rank, scalar, empty, noncontiguous-source-derived, and maximum
  logical-position fixtures;
- exact float32/float64 closed-open and open-open conversion oracles;
- exact Box-Muller word schedule, ordered components, scalar slicing, affine
  branches, odd ordinal/count boundaries, output-shape slices, finite
  envelopes, zero-scale full-word requests, and Gaussian moments/covariance;
- Poisson scalar/fixed-word/high-precision/statistical fixtures across zero,
  inversion, crossover, PTRS, endpoint, mixed-path, and exhaustion cases;
- binomial scalar/fixed-word/high-precision/statistical fixtures across zero,
  unit, inversion, BTRS, reflection, mass-pair, large-count cancellation,
  mixed-path, and exhaustion cases;
- complete result shape/dtype/device/contiguity/freshness/non-aliasing and input
  immutability tests, including word-free paths;
- ordinary law/count tensor layout, dtype, shape, device, gradient, broadcast,
  and alias tests, with tensor subclasses/custom dispatch explicitly
  unqualified rather than exhaustively detected;
- `require_same_dtype()` zero/one/equal/mismatch/malformed tests, including a
  malformed operand after an earlier dtype mismatch to prove all operands are
  validated first, plus payload/input identity preservation;
- proof that successful return enqueues every package write and no package
  alias is written afterward, without implying host synchronization;
- proof that the public package exports exactly the accepted names and imports
  no downstream package;
- proof that no public raw-word or sampler-internal surface leaked;
- proof that PyTorch global RNG state is unchanged and forbidden native/private
  RNG calls are absent;
- strict static-typing probes for all public signatures;
- the complete supported TensorCore dependency matrix, including every
  supported PyTorch minor (currently 2.11 and 2.12); and
- eager CPU evidence plus conditional eager CUDA evidence when CUDA is
  available, with every unavailable accepted backend explicitly qualified.

TensorCore's tests own generic word/distribution correctness. The later
TensorDSLab Maintenance 2 work order owns default detector-key continuity and
completed-product equality; TensorCore tests must not import TensorDSLab to
claim that evidence.

## Version And Compatibility Proposal

Against the observed closed `0.8.0` baseline, the proposed additive public
surface would ordinarily become pre-1.0 version `0.9.0`. TensorCore Design owns
that decision against its exact selected starting commit.

The stage makes no broad source, binary, downstream, cross-version,
cross-backend, performance, deployment, or release-readiness claim. Exact
same-algorithm word identity and the named distribution mapping are contracts;
completed floating transcendental results are same-backend/mode repeatable and
may receive only statistical cross-backend guarantees where documented.

## Non-Goals And Forbidden Scope

This proposed work order does not authorize:

- TensorDSLab edits or dependency movement;
- public Bernoulli, standard-normal, exponential, categorical, multinomial,
  shuffle, permutation, choice, or raw-bit APIs;
- mutable/global RNG, `torch.Generator`, or PyTorch RNG adapters;
- a Random123, NumPy, JAX, native-RNG, or other new runtime dependency;
- host materialization as a hidden implementation route; Random123 remains
  normative/reference evidence only;
- detector stream enums, default detector keys, physical delay or rate models,
  scientific position planners, or product dependency graphs;
- generic checked Charge addition, count ledgers, allocation planning, or
  overflow diagnostics;
- partial-axis agreement, deep tensor-value scans, overflow-safe int64 totals,
  range/count axes, or scalar-to-target-dtype public helpers;
- selection, batching, movement, reconstruction, output buffers, workspaces,
  leases, registries, lifecycle services, IO, or artifacts beyond an already
  operative separately accepted TensorCore baseline;
- compiler/fusion/custom-kernel work; or
- a push, package publication, conformance finding, Coordination activation,
  or Profile B activation.

## Stop Conditions

Return to TensorCore Design and TensorDSLab Design before implementation if:

- Stage 14 or another accepted TensorCore change leaves the proposed baseline,
  version, package shape, or public signatures ambiguous;
- the exact Stage 5/6 executable mapping cannot be transplanted without a
  changed default-key result;
- a proposed generic contract would require TensorCore to import or understand
  a TensorDSLab scientific concept;
- implementation appears to require a public raw-word or sampler-internal API;
- `CounterRng` cannot remain immutable/stateless under the accepted design;
- the work requires a different RNG algorithm, packing, sampler, exhaustion,
  probability, or high-precision decision policy;
- another package-authoritative source conflicts with this consumer proposal;
  or
- the exact package baseline, persistent routes, allowed files, or user
  dispatch authority is absent or discrepant.

Unavailable CUDA, build tooling, or hosted CI is a recorded qualification
unless TensorCore Design makes it an explicit dispatch prerequisite; it does
not authorize invented evidence or a performance claim.

## Downstream Handoff Gate

TensorDSLab may not dispatch Maintenance 2 merely because this draft exists.
The handoff becomes eligible only after TensorCore Design has:

1. adopted a package-authoritative work order;
2. implemented it through TensorCore's normal role-separated workflow;
3. cleared fixed-commit Validation, independent Review, final Design approval,
   and clean merge to `main`;
4. identified the exact merged commit and package version;
5. supplied the public API and test evidence without a downstream import; and
6. reported every qualification relevant to TensorDSLab's selected devices
   and eager execution modes.

TensorDSLab Design then independently selects or rejects that exact commit,
fills every dependency placeholder in its Maintenance 2 draft, and seeks a
separate user dispatch. No automatic cross-package adoption is implied.
