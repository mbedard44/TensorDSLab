# Validation

Validation proves public contracts, scientific behavior, and package ownership
at the boundary that owns each claim. Tests should exercise supported use and
must not mirror private implementation structure or harden the package against
callers who deliberately leave the public API.

## Current State And Next Gate

Stage 2 is Merged / Closed at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. Maintenance 1 is Merged / Closed
at `3af8ab4acf834b07e3d027fb530e5f12934999a5`. Those commits remain the
historical TensorCore `0.6` production baseline.

[Stage 3: TensorCore 0.7 Product Foundation](implementation/stage_3_tensorcore_0_7_product_foundation.md)
is Design-complete / Undispatched. It is the next production gate and replaces
the live `0.6` structural package with the product-centered TensorCore `0.7`
foundation described in [Rebuild Architecture](architecture/rebuild.md). Until
Stage 3 passes fixed-commit Validation, independent Review, and merge, the
rebuild is a Design target rather than implemented behavior.

Documentation-only Design work remains in Design unless the user requests an
independent documentation Validation or Review. At minimum, run:

```bash
git diff --check
```

Also check local Markdown links, heading and code-fence balance, stale `0.6`
names in live documents, accidental placeholder files, and consistency among
architecture, design, decisions, parity, validation, and implementation-stage
records.

## Governance Adoption Checks

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`.
Conformance remains `Not evaluated`, Coordination remains `Deferred`, and
Profile B remains `Disabled`. Validate the
[governance index](governance/index.md),
[adoption record](governance/adoption_0_1_0.md),
[overlay](governance/overlay.md),
[semantic rule map](governance/rule_map_0_1_0.md), and `docs/decisions.md`
against these checks:

- verify Governance Core manifest-file SHA-256
  `45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8`
  and all eight entries;
- verify Council Charter manifest-file SHA-256
  `343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916`
  and all three entries;
- verify governed Design base
  `151b61fdc36475498219ee5fe7b045a3a72c2d09`, exact accepted candidate
  `d634401a853915edeb4f83df4a4943b3553deced`, its exact parent, and its
  authorized nine-path scope;
- map `OP-01` through `OP-13` and `ENG-01` through `ENG-12` exactly once and
  retain six `Adopted`, 19 `Stronger local rule`, no whole-rule
  Not-applicable disposition, and no accepted deviation;
- retain exact absence evidence and a focused activation trigger for every
  dormant surface;
- confirm durable files contain no raw task identifier and no `.agents`,
  route, registry, or cache state was created;
- reject claims of conformance, Active Coordination, enabled Profile B,
  deployability, release readiness, backward compatibility, broad
  compatibility, or implemented integration; and
- treat compatibility evidence as exact-baseline evidence only. Same-device
  residency and no-silent-host-materialization Design constraints do not prove
  an implemented package handoff.

Changing the TensorCore dependency and package structure under a focused
Stage 3 work order does not alter the adopted governance record or create a
conformance finding.

## Boundary-First Validation

The rebuild follows this order:

```text
public constructor/config values
  -> TensorCore constrained scalars and semantic roots
  -> TensorDSLab axes and product fields
  -> ReadoutCollection completed results
  -> future product operations and simulate_readout(...)
  -> future TensorG4DS, TensorML, and durable boundaries
```

TensorCore validates universal representation invariants before calling a
leaf's `_require()`. TensorDSLab leaf construction validates cheap intrinsic
semantics. Full-device finite/nonnegative/bounded scans belong to explicit
product-specific deep validators at untrusted ingress and producer validation
boundaries; they do not run invisibly in every field constructor.

## Stage 3 Package And Dependency Checks

Stage 3 must prove:

- the project remains `TensorDSLab` and the import package remains
  `tensor_dslab`;
- `tensor_dslab.common` and `tensor_dslab.readout` remain directly under the
  import root, with product packages directly under `readout`;
- `pyproject.toml` selects exact TensorCore `0.7.0` commit
  `b454d738f6385ce6489d85492a618a3dab139bb6`;
- all TensorCore imports come from the public `tensor_core` package root;
- no retired TensorCore module, compatibility alias, copied helper, local
  fork, or generic TensorCore re-export exists;
- no production import reaches TensorG4DS, TensorML, DSLab, IV-DSLab,
  Projects/dag, G4DS/g4ds11, NumPy, or an IO backend;
- package imports do not transitively load those deferred dependencies;
- every created module owns real Stage 3 behavior; and
- `simulation.py`, `_random.py`, and every `_product.py` remain absent because
  Stage 3 implements no operation or orchestration behavior.

The exact TensorCore public surface used by this stage is limited to
`TensorAxis`, `TensorField`, `TensorCollection`, the accepted constrained
scalars, and public relationship requirements. Retired `Id`, axis/field IDs,
`IdSequence`, `TensorLayout`, selection, movement, reconstruction, output-
buffer, and like-allocation surfaces must be absent from live production and
tests.

## Ordinary-ABC Semantic Leaf Checks

The three axes, six product fields, and `ReadoutCollection` each have
`__bases__ == (matching_tensor_core_root,)`, directly inheriting exactly one
root with no mixin or other base. Static, runtime, and Review evidence must
prove every leaf:

- is decorated with `@final`;
- declares `__slots__ = ()`;
- adds no stored annotation or dataclass field;
- does not reapply `@dataclass`;
- does not override TensorCore construction, `_validate`, equality, hashing,
  or lookup behavior;
- implements `_require(self) -> None`; and
- inherits the exact root constructor signature.

Constructor probes must establish:

```text
ExampleAxis(coordinates: tuple[str, ...])
Photoelectrons(tensor: torch.Tensor, axes: tuple[TensorAxis, ...])
ReadoutCollection(*, fields=...)
```

Use the actual inherited signatures exposed by TensorCore and require the
selected static checker to infer concrete results from each constructor and
from typed `field(...)`, `tensor(...)`, `axis(...)`, and `dimension_of(...)`
calls. Tests must not require a runtime-finality guard or adversarially probe
subclassing of final classes, class mutation, constructor bypass, direct
private calls, or custom Torch dispatch. Those uses are unsupported.

## Axis And Sampling Checks

### `ExampleAxis` And `ChannelAxis`

Tests should prove exact-string coordinate validation is inherited from
TensorCore and each TensorDSLab axis additionally rejects an empty coordinate
tuple. Coordinate tuple order is tensor index order. Coordinate labels are not
RNG hot-path identities.

### `SampleAxis`

Tests should prove:

- at least two coordinates are required;
- timestamps use exact ASCII grammar `^(0|[1-9][0-9]*)ps$`;
- signs, whitespace, leading zeros, decimals, exponents, uppercase or
  alternate units, and values above signed-int64 are rejected;
- timestamps increase strictly at one positive uniform integer-picosecond
  period;
- the derived exclusive stop is at most `2**63 - 1`;
- `start_ps`, `sample_period_ps`, and `stop_ps` are correct;
- direct construction of a regular nonzero-start subaxis remains valid; and
- the coordinate tuple contains left edges only and never the terminal right
  edge.

### `SamplingConfig`

Tests should prove:

- `sample_period_ps` and `sample_count` require exact `PositiveInteger`
  wrappers;
- the count is at least two;
- `sample_period_ps * sample_count <= 2**63 - 1`;
- `window_stop_ps` is exact;
- `build_axis()` produces exactly `sample_count` canonical zero-start left-edge
  timestamps; and
- the returned `SampleAxis` agrees with the configured count, period, and
  exclusive stop.

Stage 3 does not implement PE binning. Boundary fixtures for future binning
must continue to treat bins as left-closed/right-open, include `0` and every
`i * period`, exclude negative time and `window_stop_ps`, and account for
underflow and overflow separately when that bridge is implemented.

## Shared Private Requirement Checks

`tensor_dslab.readout._requirements` contains only these shared private
helpers:

- `_require_readout_structure`;
- `_require_dtype`;
- `_require_floating_dtype`;
- `_require_exact`;
- `_require_optional_exact`; and
- `_require_one_of_exact`.

Focused tests should prove their supported relationship behavior, including
the `TypeError` distinction for malformed types and `ValueError` distinction
for well-typed values that violate a relationship. The readout-structure
helper requires exactly one `ExampleAxis`, `ChannelAxis`, and `SampleAxis` in
any order and `torch.strided` storage. It does not require contiguity, a fixed
dimension order, or an exact base `torch.Tensor` type.

`_require_floating_dtype` accepts exactly `torch.float32` or `torch.float64`.
It rejects `torch.float16`, `torch.bfloat16`, and every non-floating dtype.

`common.sampling` must validate directly and must not import private readout
requirements. Product-specific value scans remain in their owning product
modules rather than becoming a shared role registry.

## Product Field Checks

Stage 3 defines exactly these direct final leaves:

| Exact type | Constructor invariant | Explicit deep-value invariant |
| --- | --- | --- |
| `Photoelectrons` | `torch.int64`, exact readout axes, `torch.strided` | nonnegative |
| `Charge` | `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite and nonnegative |
| `PureWaveform` | exactly `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite |
| `NoiseWaveform` | exactly `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite |
| `AnalogWaveform` | exactly `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite |
| `DigitizedWaveform` | `torch.int32`, exact readout axes, `torch.strided` | nonnegative and at most `2**bit_depth - 1` for its exact config |

Each product module owns one private `_require_valid_values(...)` function for
its exact field; the digitized variant also accepts the exact
`DigitizedWaveformConfig` needed to derive its upper bound. Tests must separate
cheap constructor validation from these explicit full-value checks.

Test at least two valid axis orders, exact shape/axis agreement, missing,
duplicate, or foreign axes, sparse/non-strided rejection, noncontiguous
strided acceptance, correct and incorrect dtypes, CPU tensors, and conditional
CUDA tensors when available. General semantic construction remains
placement-neutral and makes no GPU-kernel claim.

`Photoelectrons` is an already-produced dense truth input. Stage 3 creates no
`PhotoelectronsConfig`, source producer, PE-binning function, or TensorG4DS
adapter. `DigitizedWaveform`, not `DigitalWaveform`, is the accepted product
name; truncation is fixed by the future producer and no quantization enum or
sidecar exists.

## Configuration Checks

Every config is a public `@final`, frozen, slotted, keyword-only dataclass.
There is no generic `Config` ABC. Tests should prove exact component wrapper
types, closed exact-class unions, immutable composition, and every local range
or relationship in [Scientific Configuration](architecture/rebuild.md#scientific-configuration).

At minimum, cover:

- `TimingJitterConfig`, `DarkCountConfig`, `FixedDelayConfig`,
  `ExponentialDelayConfig`, `NormalDelayConfig`, `DirectCrosstalkConfig`,
  `DelayedCrosstalkConfig`, `AfterpulseRecoveryConfig`, `AfterpulseConfig`,
  `CorrelatedAvalancheConfig`, `ChargeSmearingConfig`, and `ChargeConfig`;
- `TpcFebSnrPulseConfig`, `VetoPduPulseConfig`, and the exact two-model
  `PureWaveformConfig` union;
- `ZeroNoiseConfig`, `WhiteNoiseConfig`, `PsdNoiseConfig`, and the exact
  three-model `NoiseWaveformConfig` union;
- `AnalogSaturationConfig` and `AnalogWaveformConfig`;
- `DigitizedWaveformConfig`, including bit depth 1 through 16, strict input
  voltage ordering, and gain from 0 through 40 dB; and
- `ReadoutConfig`, including required exact `SamplingConfig` and optional exact
  product-config components.

PSD tests at this structural stage cover tuple type, nonempty equal-length
left-edge/density arrays, zero start, strict edge order, exclusive stop, finite
nonnegative density, and rejection of an all-zero supplied PSD in favor of
`ZeroNoiseConfig`. They do not implement or validate FFT synthesis.

Base classes, foreign objects, and wrong scalar wrappers are unsupported config
values and must fail at the documented public constructor. Subclassing a final
config is itself unsupported and needs no separate adversarial probe. This is
ordinary public-input validation, not a promise to police callers who mutate
classes or bypass construction.

## `ReadoutCollection` Checks

Tests should prove:

- any nonempty subset of the six exact product types is accepted;
- empty or unrecognized membership is rejected;
- duplicate exact product types are rejected by TensorCore;
- membership order has no semantic meaning;
- `field_types` is the exact frozenset of present classes;
- `field(Product)` and `tensor(Product)` infer and return the exact product or
  tensor, while a missing product raises `KeyError`;
- every present field has equal ordered axes and the same exact device;
- all present floating products have one common dtype;
- mixed integer and floating role dtypes remain valid where each leaf permits
  them;
- the collection retains the exact supplied field records and tensor
  references;
- collection membership is immutable and collection equality remains object
  identity; and
- there is one `ReadoutCollection`, no per-product collection subclass,
  canonical-order registry, descendant map, sidecar, lifecycle state, or
  mutation API.

`ReadoutCollection.accepted_field_types()` is the sole class-owned schema
declaration and returns one unordered frozenset containing all six exact
classes. A completed collection may contain only a requested product subset;
it is not an ordered partial-pipeline snapshot.

## Public Surface And Import Checks

Verify deliberate `__all__` values and object identity across:

- each product package root;
- `tensor_dslab.common`;
- `tensor_dslab.readout`; and
- the top-level `tensor_dslab` collaborator API.

The top-level package should expose the three axes, `SamplingConfig`, six
products, all public product configs, `ReadoutConfig`, and
`ReadoutCollection`. It must not re-export TensorCore generic classes or
scalars, private requirements, private validators, retired `0.6` names, or a
placeholder simulation function.

Fresh-process imports should prove every product package, `readout.types`, the
readout root, and the package root are acyclic. Product packages must not
import `ReadoutConfig`, `ReadoutCollection`, or future orchestration. The
complete product graph may be imported only by the cross-product composition
layer and deliberate export layers.

## Static Typing Checks

The selected checker must analyze package and tests against the exact
TensorCore pin. Positive probes should require concrete inference for:

- each inherited axis and field constructor;
- `SamplingConfig.build_axis() -> SampleAxis`;
- `ReadoutCollection(fields=...) -> ReadoutCollection`;
- `readout.field(Charge) -> Charge`;
- `readout.tensor(Charge) -> torch.Tensor`;
- `charge.axis(SampleAxis) -> SampleAxis`; and
- `charge.dimension_of(SampleAxis) -> int`.

The work order must report the exact checker/version or explicitly qualify its
absence. Manual review is not a substitute for the fixed ordinary-ABC static
probe required to select the dependency.

## Result Taxonomy, Storage, And Device Scope

Stage 3 introduces no field-returning operation, so TensorCore's operation-
owned exact-return/storage-sharing/guaranteed-fresh taxonomy has no production
operation to classify yet. Construction retains the exact caller tensor by
TensorCore contract; it does not claim a copied or fresh payload. Axis lookup
returns the exact stored axis. `SamplingConfig.build_axis()` returns an axis,
not a field result.

Every later field-returning operation must classify each successful path and
separately document subtype, dtype, device, axes, autograd, synchronization,
failure effects, and output-to-output sharing. No future operation may enqueue
writes after publishing its semantic field. These later requirements do not
authorize Stage 3 to create `out=`, output-buffer, workspace, lease, movement,
selection, or lifecycle APIs.

Stage 3 requires CPU construction tests and conditional CUDA construction
tests. CUDA absence is a recorded skip and no GPU behavior claim. Tests must
prove device mismatches within one collection fail rather than move silently.
No code path may silently call `.cpu()`, `.numpy()`, `.tolist()`, detach, cast,
or import NumPy as a reference implementation.

## Later Scientific Validation

Stage 3 does not implement scientific producers, RNG, request planning, or
`simulate_readout(...)`. The detailed future acceptance matrix remains
normative in [Rebuild Validation Strategy](architecture/rebuild.md#validation-strategy),
including:

- product-request closure and retention invariance;
- TensorDSLab positional Threefry RNG and distribution primitives;
- dark counts, timing jitter, fixed-generation correlated avalanches, S1/S2
  charge ledgers, recovery weighting, overflow, and smearing;
- TPC FEB-SNR and Veto PDU pulse models;
- white, zero, and PSD-shaped noise;
- analog composition/saturation and digitization;
- operation-owned freshness, stream ordering, and future GPU fusion evidence;
  and
- future TensorG4DS, TensorML, Reconstruction, and durable boundaries.

Those checks activate only under focused production work orders. They must
not be weakened merely because Stage 3 establishes the semantic types. The
sole active correlated-avalanche baseline is the fixed-maximum-generation
model in `rebuild.md`; deleted exploratory algorithm documents are not
implementation sources.

## Parity And Donor-Fixture Rules

[IV-DSLab Parity](parity.md) defines the comparison taxonomy, audited donor
baseline, accepted divergences, and operation-level claims. Tests must not
import or execute IV-DSLab or DSLab at runtime. Every promoted fixture must
name its donor source/snapshot, comparison boundary, parity classification,
units, axes, dtype, operation order, RNG or probability contract, edge policy,
acceptance criterion, and intentional divergences.

Golden fixtures remain small, reviewable, and TensorDSLab-owned. One fixture
does not prove distributional parity. Do not preserve donor global state,
unsigned wraparound, CPU-list conversion, singleton-batch assumptions,
condition-DB loading, remote downloads, or apparent bugs merely for literal
parity.

## Ownership And Scope Checks

Validation and Review should reject accidental introduction of:

- source Photoelectrons production, native G4DS parsing, TensorG4DS clustering,
  a TensorG4DS adapter, or PE binning in Stage 3;
- scientific product builders, RNG, request planning, `simulate_readout`,
  workspace, output buffer, stream, lease, selection, movement, or lifecycle
  behavior;
- durable cache, manifest, IO, scheduler, retry, campaign, or DAG surfaces;
- TensorML model/training/evaluation or Reconstruction concepts;
- global config, field, builder, validation, or registry dumping-ground
  modules;
- a `PhotoelectronsConfig`, generic `Config` or product ABC, per-product
  collection subclass, or compatibility shim;
- placeholder packages or modules;
- generated caches, outputs, or unrelated files; and
- release, deployment, backward-compatibility, broad compatibility,
  conformance, GPU-kernel, zero-copy, or allocation-free claims.

## Stage 3 Command Baseline

The focused work order defines the exact commands. At minimum, run from the
project root against the selected TensorCore source and an independently
archived exact-pin checkout:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
pyright
```

Also run fixed public-import, retired-name, dependency/import-isolation,
ordinary-ABC signature, and static inference probes. Report exact Python,
PyTorch, TensorCore, static-checker, and CUDA evidence. A missing required
static checker, dependency mismatch, dirty fixed candidate, or unexplained
conditional skip prevents Stage 3 clearance.
