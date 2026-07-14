# Stage 3 TensorCore 0.7 Product Foundation Work Order

Status: Merged / Closed. Exact implementation candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde`, from committed Design/dispatch
base `fb4fd3753d336fd46203e122789caed32fb49d91`, passed fixed-commit Validation
and independent Review with no unresolved finding. Review's documentation-only
closeout and clean fast-forward produced `main`
`97e17c3177ac217aeb42a077db78f4bd223d51fa`. TensorDSLab Design accepted the
closeout on 2026-07-14 after independently repeating the post-merge package,
dependency, static-typing, import-isolation, and artifact checks. No push
occurred.

Review evidence: the candidate has exact parent
`fb4fd3753d336fd46203e122789caed32fb49d91` and an exact 35-path authorized
delta comprising 19 additions, 9 modifications, and 7 deletions. TensorCore
was clean at exact pin `b454d738f6385ce6489d85492a618a3dab139bb6`, version
`0.7.0`. An independent archive of that commit had SHA-256
`649c4daac3b953397371cb64647dcaf9a7ca7a857b32fae58c4ec4a856c79796`.
Both the source checkout and archived pin ran 51 tests: 49 passed and 2
conditional CUDA tests were skipped because CUDA was unavailable. Pyright
`1.1.408` reported 0 errors, warnings, or informational findings against both
the source checkout and extracted archive. Import isolation returned
`False False False False`; `git diff --check`, exact exports, static leaf
contracts, retired-surface absence, and artifact checks passed.

The evidence environment was Python `3.13.11`, PyTorch `2.12.1`, and macOS
`15.7.4` on arm64. CUDA and a CUDA runtime were unavailable, so this is CPU
evidence and makes no GPU claim. The `build` and `hatchling` modules were
unavailable, so no editable-install or wheel-build claim is made. The
Review-owned closeout changes only this work order and the implementation
index; the cleared production, test, and metadata bytes remain exact candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde`.

## Objective

Replace the pre-deployment TensorCore `0.6` structural package with the
smallest complete TensorCore `0.7` product-centered semantic foundation:

- exact TensorCore `0.7.0` dependency selection;
- three common semantic axes and the shared sampling policy;
- six exact readout product-field leaves;
- product-owned public scientific config records;
- one completed-result `ReadoutCollection` and one cross-product
  `ReadoutConfig`;
- focused shared private requirements and product-owned deep-value checks;
- deliberate package exports; and
- CPU, conditional CUDA, import, dependency, runtime, and static-typing
  evidence.

This is a clean structural migration. It deliberately deletes the `0.6`
IDs/layout/constants/sidecar/reconstruction/output-preparation package rather
than preserving compatibility aliases.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`. The exact accepted
pre-rebuild production baseline is clean `main` at:

```text
3af8ab4acf834b07e3d027fb530e5f12934999a5
```

That commit includes the closed Stage 2 foundation and Maintenance 1 surface
ownership. It is the governed predecessor for the production bytes being
replaced. Completed Stage 0, Stage 1, Stage 2, Maintenance 1, and governance
records remain historical evidence and are not rewritten to pretend they
implemented TensorCore `0.7`.

The Stage 3 dependency candidate is clean TensorCore `0.7.0` at exact commit:

```text
b454d738f6385ce6489d85492a618a3dab139bb6
```

The implementation branch must begin from the clean committed TensorDSLab
Design authority that contains this work order and the synchronized rebuild
documents. The dispatch handoff must record that exact commit before work
starts. A dirty tree, divergent branch, missing authority commit, changed
TensorCore bytes, or discrepant route returns the stage to Design.

The exact change allowlist is evaluated from that recorded Design dispatch
commit to each implementation candidate. The documentation synchronization
between `3af8ab4` and the Design dispatch commit is Design-owned authority, not
an Implementation candidate delta. Final reporting records all three points:
governed production predecessor, committed Design/dispatch base, and fixed
implementation candidate.

TensorDSLab remains active-development and pre-deployment. This breaking
replacement makes no backward-compatibility, release, deployment, conformance,
broad compatibility, or GPU-execution claim.

## Source Of Truth

Implementation, Validation, and Review must read and reconcile:

- [Agent Workflow](../../AGENTS.md);
- [Contributing](../../CONTRIBUTING.md);
- [Rebuild Architecture](../architecture/rebuild.md), especially TensorCore
  `0.7`, selected package shape, axes, products, configuration,
  `ReadoutCollection`, validation, migration, and supersession;
- [TensorCore Integration](../architecture/tensors.md);
- [Readout Architecture](../architecture/readout.md);
- [Validation](../validation.md);
- [IV-DSLab Parity](../parity.md); and
- TensorCore `0.7.0` `docs/api.md`, `docs/architecture/tensors.md`,
  `docs/integration.md`, and package-root implementation at the exact selected
  commit.

The rebuild architecture controls current product semantics. Historical work
orders explain prior bytes but do not override this stage. If these live
sources disagree about an in-scope contract, stop before editing the affected
slice and return the contradiction to Design.

## Dispatch And Role Loop

Before dispatch, Design must privately verify these persistent logical routes
for the active workspace:

```text
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Coordination remains Deferred and is not used. Raw platform route identifiers
must not appear in committed files.

The bounded production loop is:

```text
Design dispatch
  -> Implementation candidate
  -> fixed-commit Validation
  -> independent fixed-commit Review
  -> Implementation corrections, if any
  -> Validation/Review recheck
  -> Review clean fast-forward
  -> post-merge verification
  -> Design closeout
```

Validation may prepare tests and independent oracles while Implementation
works, but it evaluates only a fixed candidate commit. Review does not edit the
candidate. If three candidate cycles do not clear the same bounded scope, or a
finding requires a Design choice, return the stage to Design rather than
widening the branch.

Suggested implementation branch:

```text
codex/stage-3-tensorcore-0-7-product-foundation
```

No push is authorized by this work order.

## TensorCore Selection Checkpoint

Before changing TensorDSLab production, Implementation must independently
verify:

- TensorCore checkout is clean at exact commit
  `b454d738f6385ce6489d85492a618a3dab139bb6`;
- package version is `0.7.0`;
- package-root `__all__` exactly matches its operative API document;
- `TensorAxis`, `TensorField`, and `TensorCollection` are ordinary ABC roots;
- downstream leaves inherit root constructor signatures and generic lookup
  inference under the selected static checker;
- public relationship helpers have the documented malformed-type versus
  unsatisfied-relationship behavior; and
- no retired `0.6` symbol or module is being imported indirectly.

Update `pyproject.toml` to the exact Git dependency:

```text
tensor-core @ git+https://github.com/mbedard44/TensorCore.git@b454d738f6385ce6489d85492a618a3dab139bb6
```

Do not edit TensorCore, vendor its code, use private imports, or introduce a
downstream compatibility layer. A genuine generic gap stops only the affected
slice and returns a minimal reproducer to TensorDSLab Design for routing to
TensorCore Design. Stage 3 must not infer a gap merely because TensorCore no
longer provides a removed generic operation.

## Exact Production Tree For This Stage

After Stage 3, the production package contains only these in-scope modules:

```text
tensor_dslab/
  __init__.py
  py.typed

  common/
    __init__.py
    axes.py
    sampling.py

  readout/
    __init__.py
    _requirements.py
    types.py

    photoelectrons/
      __init__.py
      types.py

    charge/
      __init__.py
      types.py

    pure_waveform/
      __init__.py
      types.py

    noise_waveform/
      __init__.py
      types.py

    analog_waveform/
      __init__.py
      types.py

    digitized_waveform/
      __init__.py
      types.py
```

Do not create `readout/simulation.py`, `readout/_random.py`, or any product
`_product.py`. Their accepted owners are reserved by architecture, but this
stage has no real behavior for them. Do not create empty packages or modules
for future TensorG4DS, TensorML, Reconstruction, IO, caches, operations,
recipes, or executables.

## Exact Symbol Inventory And Ownership

### `tensor_dslab/common/axes.py`

Define exactly these public direct `TensorAxis` leaves:

- `ExampleAxis`;
- `ChannelAxis`; and
- `SampleAxis`.

Every leaf is `@final`, has `__slots__ = ()`, adds no dataclass field, and
inherits TensorCore construction and lookup. `ExampleAxis` and `ChannelAxis`
require nonempty coordinate tuples. `SampleAxis` implements the exact
timestamp grammar, signed-int64 bound, positive uniform period, and derived
`start_ps`, `sample_period_ps`, and `stop_ps` properties specified by
`rebuild.md`.

### `tensor_dslab/common/sampling.py`

Define exactly one public type:

- `SamplingConfig`.

It is a `@final`, frozen, slotted, keyword-only dataclass with exact
`PositiveInteger` fields `sample_period_ps` and `sample_count`, count at least
two, signed-int64 exclusive-stop validation, `window_stop_ps`, and
`build_axis() -> SampleAxis`. This common module validates directly; it must
not import `readout._requirements`.

### `tensor_dslab/readout/_requirements.py`

Define only these shared private helper surfaces:

```python
def _require_readout_structure(field: TensorField) -> None: ...
def _require_dtype(field: TensorField, expected: torch.dtype) -> None: ...
def _require_floating_dtype(field: TensorField) -> None: ...
def _require_exact(
    value: object,
    expected: type[object],
    field: str,
) -> None: ...
def _require_optional_exact(
    value: object | None,
    expected: type[object],
    field: str,
) -> None: ...
def _require_one_of_exact(
    value: object,
    expected: tuple[type[object], ...],
    field: str,
) -> None: ...
```

The structure helper accepts exactly one `ExampleAxis`, one `ChannelAxis`, and
one `SampleAxis` in any dimension order and requires `torch.strided`. It does
not require contiguity, a fixed dimension order, or an exact base Torch tensor
type. `_require_floating_dtype` accepts exactly `torch.float32` or
`torch.float64`; it does not accept `torch.float16`, `torch.bfloat16`, or a
non-floating dtype. The config helpers validate documented public exact-class
composition; they do not attempt to defend against class mutation or
constructor bypass.

### `tensor_dslab/readout/photoelectrons/types.py`

Define:

- public `Photoelectrons(TensorField)`; and
- private `_require_valid_values(field: Photoelectrons) -> None`.

The leaf requires exact readout structure and `torch.int64`. The explicit deep
validator requires nonnegative values. There is no `PhotoelectronsConfig` and
no source producer.

### `tensor_dslab/readout/charge/types.py`

Define public:

- `Charge`;
- `TimingJitterConfig`;
- `DarkCountConfig`;
- `FixedDelayConfig`;
- `ExponentialDelayConfig`;
- `NormalDelayConfig`;
- `DirectCrosstalkConfig`;
- `DelayedCrosstalkConfig`;
- `AfterpulseRecoveryConfig`;
- `AfterpulseConfig`;
- `CorrelatedAvalancheConfig`;
- `ChargeSmearingConfig`; and
- `ChargeConfig`.

Define private `_require_valid_values(field: Charge) -> None`. `Charge`
requires exact readout structure and exact `torch.float32` or `torch.float64`;
the explicit deep validator requires finite nonnegative values. Config fields,
exact unions, optional composition, and ranges are exactly those in
`rebuild.md`. This stage defines no sampler, PMF, stream, avalanche state, or
charge producer.

### `tensor_dslab/readout/pure_waveform/types.py`

Define public:

- `PureWaveform`;
- `TpcFebSnrPulseConfig`;
- `VetoPduPulseConfig`; and
- `PureWaveformConfig`.

Define private `_require_valid_values(field: PureWaveform) -> None`.
`PureWaveform` requires exactly `torch.float32` or `torch.float64`; the deep
validator requires finite values. The wrapper config accepts exactly one of
the two exact pulse model classes. Stage 3 validates config relationships but
implements neither pulse equation.

### `tensor_dslab/readout/noise_waveform/types.py`

Define public:

- `NoiseWaveform`;
- `ZeroNoiseConfig`;
- `WhiteNoiseConfig`;
- `PsdNoiseConfig`; and
- `NoiseWaveformConfig`.

Define private `_require_valid_values(field: NoiseWaveform) -> None`.
`NoiseWaveform` requires exactly `torch.float32` or `torch.float64`; the deep
validator requires finite values. Structural PSD validation implements the
accepted left-edge/density input contract but no spectral synthesis.

### `tensor_dslab/readout/analog_waveform/types.py`

Define public:

- `AnalogWaveform`;
- `AnalogSaturationConfig`; and
- `AnalogWaveformConfig`.

Define private `_require_valid_values(field: AnalogWaveform) -> None`.
`AnalogWaveform` requires exactly `torch.float32` or `torch.float64`; the deep
validator requires finite values. No addition, saturation, or product producer
is implemented.

### `tensor_dslab/readout/digitized_waveform/types.py`

Define public:

- `DigitizedWaveform`; and
- `DigitizedWaveformConfig`.

Define private:

```python
def _require_valid_values(
    field: DigitizedWaveform,
    config: DigitizedWaveformConfig,
) -> None: ...
```

The leaf requires exact `torch.int32`. The explicit deep validator requires
nonnegative codes no greater than `2**bit_depth - 1` for the exact supplied
config. The config accepts bit depth 1 through 16, finite strictly ordered
input bounds, and inclusive gain 0 through 40 dB. Truncation is the fixed
future producer rule; do not add a quantization enum or
`DigitizedWaveformSpec` sidecar.

### `tensor_dslab/readout/types.py`

Define exactly two public cross-product composition types:

- `ReadoutConfig`; and
- `ReadoutCollection`.

`ReadoutConfig` is a `@final`, frozen, slotted, keyword-only dataclass. It
requires exact `SamplingConfig` and accepts optional exact configs for charge,
pure, noise, analog, and digitized products.

`ReadoutCollection` is one direct final `TensorCollection` leaf. Its class-
owned

```python
@classmethod
def accepted_field_types(cls) -> frozenset[type[TensorField]]: ...
```

returns the exact unordered frozenset of all six field classes. `_require()`
enforces nonempty recognized
membership, equal ordered axes, same exact device, and one common dtype among
all present floating products. TensorCore already enforces one member per
exact field type.

The collection accepts any nonempty product subset. It does not encode order,
workflow state, prerequisites, descendant invalidation, configs, persistence,
or a sidecar.

### Package Initializers

Each product package root re-exports only the public field and configs it owns.
`tensor_dslab.common` re-exports its three axes and `SamplingConfig`.
`tensor_dslab.readout` re-exports the six products, all public product configs,
`ReadoutConfig`, and `ReadoutCollection`. The top-level `tensor_dslab` root
re-exports the complete collaborator-facing TensorDSLab surface from common
and readout.

Private requirements and `_require_valid_values` functions are not re-exported.
Do not re-export TensorCore roots, scalar wrappers, relationship helpers, or
retired TensorDSLab names.

Apart from deliberate `__all__` export declarations, the foundation adds no
loose module-level schema, axis, field, dtype, range, regex, dependency, or
product constants. Fixed semantic facts live on the owning class or inside
the focused requirement that enforces them.

## Exact Invariants

### Semantic Leaves

Every TensorDSLab axis, field, and collection leaf:

- has `__bases__ == (matching_tensor_core_root,)`, directly inheriting exactly
  that one root with no mixin or other base;
- is public `@final`;
- declares `__slots__ = ()`;
- adds no stored field or annotation;
- does not reapply `@dataclass`;
- implements `_require(self) -> None`; and
- inherits TensorCore constructor, universal `_validate`, equality/identity,
  hashing, and lookup behavior.

TensorCore root validation runs before domain `_require()`. Do not override a
root final method or duplicate universal tensor/axis/collection mechanics.

The exact inherited constructor state is:

```text
ExampleAxis/ChannelAxis/SampleAxis(coordinates: tuple[str, ...])
each product field(tensor: torch.Tensor, axes: tuple[TensorAxis, ...])
ReadoutCollection(*, fields: Iterable[TensorField])
```

Axis and field constructors accept positional or keyword arguments according
to the inherited TensorCore dataclass signature; the collection `fields`
argument is keyword-only. TensorDSLab must not wrap these constructors merely
to change their spelling.

### Axes And Fields

Every readout field contains exactly three semantic axes, one each of
`ExampleAxis`, `ChannelAxis`, and `SampleAxis`, in arbitrary order matching
tensor dimension order. Extra axes are not accepted in this MVP. All tensors
use `torch.strided`; noncontiguous strided tensors remain structurally valid.

Constructors perform no implicit movement, cast, contiguity conversion,
detach, clone, host materialization, or full-device value scan. Product deep
validators run only when explicitly invoked by a future ingress/producer
boundary or focused tests. Each deep validator returns `None` on success,
raises `ValueError` for an invalid supported payload, never changes the field
or tensor, and may synchronize the device because it is an explicit trust
boundary rather than a hot constructor path. Private callers are responsible
for passing the exact owning field/config types; these helpers do not add
defensive guards for unsupported direct misuse.

### Configs

All configs use exact component types and closed exact-class unions from the
accepted architecture. `None` disables an optional submodel. There is no
generic config base, string algorithm switch, global registry, loose default,
product persistence flag, tensor-valued channel calibration, or inferred
sampling policy.

Every record below is `@final` and
`@dataclass(frozen=True, slots=True, kw_only=True)`. Component relationships
use the exact accepted classes rather than value-compatible foreign objects.
Subclassing a final config is unsupported and needs no adversarial probe.

| Owner/type | Exact constructor fields | Additional invariant |
| --- | --- | --- |
| `SamplingConfig` | `sample_period_ps: PositiveInteger`; `sample_count: PositiveInteger` | count at least 2; exclusive stop at most signed-int64 |
| `TimingJitterConfig` | `sigma_ns: NonnegativeFloat` | none beyond wrapper |
| `DarkCountConfig` | `rate_hz: NonnegativeFloat` | none beyond wrapper |
| `FixedDelayConfig` | `delay_ns: NonnegativeFloat` | none beyond wrapper |
| `ExponentialDelayConfig` | `mean_delay_ns: PositiveFloat` | none beyond wrapper |
| `NormalDelayConfig` | `location_ns: NonnegativeFloat`; `sigma_ns: PositiveFloat` | none beyond wrappers |
| `DirectCrosstalkConfig` | `mean_offspring_per_parent: NonnegativeFloat`; `delay: FixedDelayConfig \| ExponentialDelayConfig \| NormalDelayConfig` | exact delay-model union |
| `DelayedCrosstalkConfig` | `mean_offspring_per_parent: NonnegativeFloat`; `delay: FixedDelayConfig \| ExponentialDelayConfig \| NormalDelayConfig` | exact delay-model union |
| `AfterpulseRecoveryConfig` | `time_constant_ns: PositiveFloat` | none beyond wrapper |
| `AfterpulseConfig` | `probability: Probability`; `mean_delay_ns: PositiveFloat`; `recovery: AfterpulseRecoveryConfig \| None = None` | exact optional recovery |
| `CorrelatedAvalancheConfig` | `maximum_generations: NonnegativeInteger`; `direct_crosstalk: DirectCrosstalkConfig \| None = None`; `delayed_crosstalk: DelayedCrosstalkConfig \| None = None`; `afterpulse: AfterpulseConfig \| None = None` | exact optional components |
| `ChargeSmearingConfig` | `relative_sigma: NonnegativeFloat` | none beyond wrapper |
| `ChargeConfig` | `dark_count: DarkCountConfig \| None = None`; `timing_jitter: TimingJitterConfig \| None = None`; `correlated_avalanches: CorrelatedAvalancheConfig \| None = None`; `smearing: ChargeSmearingConfig \| None = None` | exact optional components |
| `TpcFebSnrPulseConfig` | `fast_time_constant_ns: PositiveFloat`; `slow_time_constant_ns: PositiveFloat`; `support_time_ns: PositiveFloat`; `peak_voltage_mv_per_pe: FiniteFloat` | slow exceeds fast; peak is nonzero |
| `VetoPduPulseConfig` | `gaussian_center_ns: FiniteFloat`; `gaussian_width_ns: PositiveFloat`; `edge_offset_1_ns: FiniteFloat`; `edge_width_1_ns: PositiveFloat`; `edge_offset_2_ns: FiniteFloat`; `edge_width_2_ns: PositiveFloat`; `support_time_ns: PositiveFloat`; `peak_voltage_mv_per_pe: FiniteFloat` | peak is nonzero |
| `PureWaveformConfig` | `model: TpcFebSnrPulseConfig \| VetoPduPulseConfig` | exact two-model union |
| `ZeroNoiseConfig` | no fields | selects the real deterministic zero-noise model |
| `WhiteNoiseConfig` | `rms_mv: PositiveFloat` | none beyond wrapper |
| `PsdNoiseConfig` | `frequency_left_edges_hz: tuple[NonnegativeFloat, ...]`; `frequency_stop_hz: PositiveFloat`; `power_density_mv2_per_hz: tuple[NonnegativeFloat, ...]` | exact nonempty tuples; equal lengths; first edge zero; edges strictly increasing; stop exceeds last edge; at least one positive density |
| `NoiseWaveformConfig` | `model: ZeroNoiseConfig \| WhiteNoiseConfig \| PsdNoiseConfig` | exact three-model union |
| `AnalogSaturationConfig` | `minimum_mv: FiniteFloat \| None = None`; `maximum_mv: FiniteFloat \| None = None` | at least one bound; when both exist minimum is below maximum |
| `AnalogWaveformConfig` | `saturation: AnalogSaturationConfig \| None = None` | exact optional saturation |
| `DigitizedWaveformConfig` | `bit_depth: PositiveInteger`; `input_min_mv: FiniteFloat`; `input_max_mv: FiniteFloat`; `analog_gain_db: NonnegativeFloat` | bit depth at most 16; minimum below maximum; gain at most 40 dB |
| `ReadoutConfig` | `sampling: SamplingConfig`; `charge: ChargeConfig \| None = None`; `pure_waveform: PureWaveformConfig \| None = None`; `noise_waveform: NoiseWaveformConfig \| None = None`; `analog_waveform: AnalogWaveformConfig \| None = None`; `digitized_waveform: DigitizedWaveformConfig \| None = None` | exact required sampling and optional product configs |

The TensorCore scalar wrappers already guarantee the lower bounds implied by
their names. Do not duplicate those generic scalar implementations locally.
The config records express scientific choices only; requested products,
floating execution dtype, source axes/device, seed, persistence, workspace,
and stream are not config fields.

### Collection

Membership is type-directed and semantically unordered. The collection keeps
the exact supplied field records. It does not reorder, clone, select, move,
batch, reconstruct, add, replace, or invalidate them. Exact concrete classes
are the in-process semantic keys; they are not durable identifiers.

## Import Direction

The allowed production dependency direction is:

```text
tensor_core
  -> tensor_dslab.common
  -> readout._requirements
  -> product types
  -> readout.types
  -> deliberate package exports
```

Product packages do not import `readout.types` or sibling product packages.
Only `readout.types` and export layers import the full product graph. This
stage contains no orchestrator. Circular imports or a need for a global config
or field registry are findings, not reasons to hide imports at runtime.

## Retired Production Surface

Delete these files with no alias, shim, warning, or empty placeholder:

```text
tensor_dslab/common/ids.py
tensor_dslab/readout/builders.py
tensor_dslab/readout/ids.py
tensor_dslab/readout/tensors.py
tensor_dslab/readout/validation.py
```

Retire from production and public exports every `0.6` concept they carried,
including:

- `ExampleId`, `ChannelId`;
- all axis IDs, field IDs, ID sequences, constants, and registries;
- `TensorLayout` and shared-axis reconstruction;
- `SampleGrid`, `DigitizedWaveformSpec`, and `AdcQuantization`;
- projection, selection, movement, reconstruction, descendant invalidation,
  result-buffer construction, and output-buffer construction; and
- partial ordered pipeline snapshots.

Do not retain stale-name import failures as compatibility tests. Test exact
absence from modules, `__all__`, repository production searches, and fresh
processes.

## Exact Change Allowlist

Implementation may add, modify, or delete only these production/metadata
paths:

```text
README.md
pyproject.toml
tensor_dslab/__init__.py
tensor_dslab/common/__init__.py
tensor_dslab/common/axes.py
tensor_dslab/common/ids.py                         # delete
tensor_dslab/common/sampling.py
tensor_dslab/readout/__init__.py
tensor_dslab/readout/_requirements.py
tensor_dslab/readout/analog_waveform/__init__.py
tensor_dslab/readout/analog_waveform/types.py
tensor_dslab/readout/builders.py                   # delete
tensor_dslab/readout/charge/__init__.py
tensor_dslab/readout/charge/types.py
tensor_dslab/readout/digitized_waveform/__init__.py
tensor_dslab/readout/digitized_waveform/types.py
tensor_dslab/readout/ids.py                        # delete
tensor_dslab/readout/noise_waveform/__init__.py
tensor_dslab/readout/noise_waveform/types.py
tensor_dslab/readout/photoelectrons/__init__.py
tensor_dslab/readout/photoelectrons/types.py
tensor_dslab/readout/pure_waveform/__init__.py
tensor_dslab/readout/pure_waveform/types.py
tensor_dslab/readout/tensors.py                    # delete
tensor_dslab/readout/types.py
tensor_dslab/readout/validation.py                 # delete
```

Implementation may replace/delete the obsolete Stage 2 tests and add only
these focused test/support paths:

```text
tests/readout_fixtures.py
tests/test_package_contracts.py
tests/test_readout_axes_and_sampling.py
tests/test_readout_collection.py
tests/test_readout_configs.py
tests/test_readout_output_preparation.py            # delete
tests/test_readout_product_types.py
tests/test_readout_tensor_operations.py             # delete
tests/typing/stage_3_semantic_leaf_contracts.py
```

Review closeout may update status/evidence only in this work order and
`docs/implementation/index.md`. Any other production, test, metadata, or
documentation path requires return to Design and an amended committed work
order before work continues.

## Required Test Design

### Package And Dependency

Test exact package exports, module ownership, import isolation, retired-name
absence, exact dependency commit, no private TensorCore imports, and no
deferred dependency imports. Use fresh processes for package-root and each
product-package import. Assert every public class's exact `__module__` matches
the owner named in this work order and every re-export is the same class
object, not an aliasing wrapper or duplicate definition.

### TensorCore And Static Contract

Test abstract roots, inherited constructor signatures, exact leaf
`__bases__ == (matching_tensor_core_root,)` with no mixin, `@final`, empty
slots, no added stored fields, no reapplied dataclass, root lookup behavior,
and exact type inference. The static probe
must include `assert_type` or checker-equivalent evidence for:

```python
axis = SampleAxis(coordinates=("0ps", "2000ps"))
photoelectrons = Photoelectrons(
    tensor=torch.zeros((1, 1, 2), dtype=torch.int64),
    axes=(ExampleAxis(coordinates=("e0",)),
          ChannelAxis(coordinates=("c0",)), axis),
)
readout = ReadoutCollection(fields=(photoelectrons,))
```

It must prove concrete results for axis/field/collection construction,
`field(...)`, `tensor(...)`, `axis(...)`, and `dimension_of(...)`.

### Axes And Sampling

Cover every accepted and rejected timestamp grammar case, signed-int64 edge,
uniformity, at-least-two constraint, nonzero-start direct subaxis, all derived
properties, config range edges, left-edge construction, and exact config-to-
axis agreement.

### Fields And Deep Validators

Cover both representative axis orders, tensor/axis shape agreement, exact
axes, every accepted/rejected dtype, sparse rejection, noncontiguous strided
acceptance, private deep finite/nonnegative/bounded checks, CPU, conditional
CUDA construction, and no implicit movement/cast/copy. Include a focused
fixture showing that a structurally valid field containing a bad scientific
value can be constructed without an implicit full-device scan and is then
rejected by its explicit owning `_require_valid_values(...)` boundary.

### Configs

Cover each config's exact component types, valid boundaries, just-outside
values, optional composition, exact model union, frozen/slotted behavior, and
PSD structural edge cases. No test may pretend a config implies an implemented
scientific algorithm.

### Collection

Cover all six singleton memberships, representative multi-product subsets,
all six together, empty/foreign/duplicate rejection, unordered input,
same-axis/device/common-floating-dtype coherence, exact field/tensor retention,
typed lookup, missing lookup, identity equality, and immutability.

### Unsupported-Use Boundary

Do not add adversarial tests for subclassing final leaves, monkeypatching
classes, calling private functions with fabricated invalid objects, mutating
exposed tensors, or custom Torch dispatch. Continue to test malformed
documented public constructor values and supported relationships normally.

## Result Taxonomy And Memory Scope

Stage 3 defines semantic records, not product operations. It therefore has no
field-returning operation to classify as exact return, guaranteed storage-
sharing, sharing permitted but unspecified, or guaranteed fresh. TensorCore
construction retains the exact caller tensor reference; that is constructor
behavior rather than a product-result freshness claim. Axis lookup returns the
exact stored axis. `SamplingConfig.build_axis()` returns an axis, not a field.

Do not create a producer merely to exercise the taxonomy. Every later
field-returning operation must state its classification and independently
document dtype, device, axes, autograd, synchronization, failure effects, and
output-to-output sharing. Stage 3 adds no `out=`, workspace, destination,
lease, stream, selection, movement, copy, or lifecycle contract.

## Documentation Duties

Implementation updates `README.md` to describe the exact implemented Stage 3
surface, its already-produced `Photoelectrons` input boundary, and its explicit
scientific-operation exclusions. Do not advertise `simulate_readout(...)`
before it exists.

Implementation and Review may record only concrete status/evidence in this
work order and the implementation index. Architecture, parity, decisions,
governance, and completed historical work orders must remain unchanged. If
code requires a semantic correction to the synchronized Design baseline,
return to Design rather than editing authority inside the implementation loop.

## Verification Commands

Implementation and fixed-commit Validation must run at least:

```bash
git diff --check
git -C /Users/mbedard/Projects/TensorCore rev-parse HEAD
git -C /Users/mbedard/Projects/TensorCore status --short
git -C /Users/mbedard/Projects/TensorCore archive --format=zip --output=/tmp/tensorcore-b454d738.zip b454d738f6385ce6489d85492a618a3dab139bb6
shasum -a 256 /tmp/tensorcore-b454d738.zip
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/tmp/tensorcore-b454d738.zip python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -c "import sys, tensor_dslab; print('tensor_g4ds' in sys.modules, 'tensor_ml' in sys.modules, 'dslab' in sys.modules, 'g4ds11' in sys.modules)"
pyright
```

Record the archive SHA-256 and independently confirm it was built from exact
TensorCore commit `b454d738f6385ce6489d85492a618a3dab139bb6` before using it.
Also run focused commands that prove:

- exact inherited signatures;
- exact public `__all__` values and nested-module ownership;
- no retired production name or module remains;
- every product package imports independently in a fresh process;
- product packages do not import the cross-product composition layer;
- no `simulation.py`, `_random.py`, or `_product.py` exists;
- no `__pycache__`, `.pyc`, generated output, or unrelated file remains; and
- wheel metadata records the exact dependency.

Run an editable install and wheel build only if they can be performed without
changing the fixed source tree; report the exact result and environment.
Report exact Python, PyTorch, TensorCore, static-checker, and CUDA versions.
Conditional CUDA skips are qualifications, not failures or GPU claims. A
missing `pyright` or equivalent accepted static checker prevents clearance of
the mandatory ordinary-ABC typing checkpoint.

## Validation And Review Evidence

Validation returns a fixed-commit report containing:

- exact candidate and parent commits;
- changed-path reconciliation against this allowlist;
- exact TensorCore source and archived-pin evidence;
- every command and result;
- test count, passes, failures, errors, and conditional skips;
- static inference evidence;
- public import and isolation evidence;
- absence/deletion evidence;
- device/environment qualifications; and
- findings classified by severity and returned without editing the candidate.

Independent Review inspects the exact Validation-cleared bytes, architecture
conformance, package cohesion, public simplicity, import direction, test
quality, deleted-surface completeness, unsupported-use boundary, and absence
of scientific or future placeholders. Review either clears that exact commit
or returns actionable findings to Implementation.

Only Review may perform the final clean `git merge --ff-only` after clearance.
Review then reruns the required suite and fixed dependency/import evidence from
`main`, confirms a clean tree, and records the pre-merge and post-merge commits.
No push occurs.

## Non-Goals

- No scientific producer or private product builder.
- No `simulate_readout(...)`, request parser, prerequisite planner, retained-
  product policy, or partial public skeleton.
- No RNG, seed, stream, counter, Threefry, Poisson, normal, exponential,
  Bernoulli, PMF, or stochastic execution.
- No dark count, timing jitter, correlated avalanche, charge smearing, pulse,
  convolution, noise synthesis, analog composition, saturation, or
  digitization execution.
- No `out=`, destination, output bank, workspace, allocation-free, zero-copy,
  lease, stream, generation, scheduler, or pool.
- No field selection, movement, batching, reduction, reconstruction, mutation,
  descendant invalidation, or TensorCore operation replacement.
- No source Photoelectrons construction, PE binning, TensorG4DS import/adapter,
  native G4DS parsing, or detector-window bridge.
- No IO, artifact, persistence, cache, DAG, TensorML, Reconstruction, campaign,
  or deployment surface.
- No generic `Config` ABC, `ReadoutField` base, per-product collection,
  quantization enum, digitized sidecar, field/axis constant, or product
  registry.
- No TensorCore repository or API change.
- No placeholder file from the complete future architecture.

## Return To Design Before

Return before:

- selecting any TensorCore commit other than the exact candidate;
- changing a public TensorCore contract or requiring a private import;
- changing an axis grammar, field dtype, product meaning, config name,
  component type, range, or ownership;
- changing the six recognized products or collection coherence rules;
- adding an extra axis, fixed axis order, count-only sample representation,
  sidecar, durable ID, registry, or compatibility alias;
- moving a product config out of its owning `types.py`;
- adding a public or private operation, orchestrator, RNG, placeholder, or
  future integration module;
- weakening the explicit deep-value boundary by hiding full-device scans in
  every leaf constructor;
- adding defensive runtime hardening for unsupported subclass/class-mutation/
  constructor-bypass/custom-dispatch behavior;
- broadening the allowlist or modifying historical/governance records;
- accepting a dirty candidate, unexplained generated file, dependency drift,
  route discrepancy, or missing mandatory static checker; or
- making a GPU, compatibility, release, deployment, conformance, zero-copy, or
  allocation-free claim.

## Merge And Closeout Criteria

Stage 3 is Merged / Closed only when:

1. the exact TensorCore checkpoint passes;
2. every changed path is authorized and every retired path is absent;
3. package/runtime/static tests pass at one fixed candidate against both the
   selected source checkout and independently archived exact pin;
4. Validation has no unresolved finding;
5. independent Review clears the same bytes;
6. Review performs a clean fast-forward to `main` and repeats post-merge
   verification;
7. `main` is clean and contains no generated artifacts;
8. this work order and the implementation index record exact closeout
   commits, environment, counts, skips, and qualifications; and
9. Design accepts the closeout report.

Closure establishes only the TensorCore `0.7` semantic product/config
foundation. Scientific implementation, GPU execution, TensorG4DS and TensorML
integration, durable artifacts, conformance, Coordination, and Profile B
remain unchanged and undispatched.
