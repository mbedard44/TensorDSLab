# Stage 4 Deterministic Waveform Products Work Order

Status: **Review-cleared / fast-forward merged; Design acceptance pending** at
exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da`. The user authorized production
execution on 2026-07-14 from committed Design/dispatch base
`b7af45741035821dfa94c8093bdeccea3320e26d`; fixed-commit Validation and
independent Review found no unresolved finding. Review cleanly fast-forwarded
`main` from `5ff13eb3c0735abfda454a334be59faac35259c2` to the exact candidate and
repeated the required post-merge gates. Final Design acceptance remains the
last closeout gate. No push occurred.

Review verified that the candidate has the exact Design parent and an exact
eight-path implementation delta comprising five additions and three
modifications. The full fast-forward from the prior `main` integrates the
eleven-path committed Design authority and the eight-path candidate as a
nineteen-path linear change. The feature branch remains fixed at the candidate.
Every protected initializer, product/config type, metadata file, architecture
source, governance record, sibling repository, and deferred producer remained
unchanged across the implementation candidate.

TensorCore was clean at exact `0.7.0` pin
`b454d738f6385ce6489d85492a618a3dab139bb6`. Review independently archived
that commit; the ZIP commit comment named the exact pin and its SHA-256 was
`649c4daac3b953397371cb64647dcaf9a7ca7a857b32fae58c4ec4a856c79796`.
Before and after the merge, both the source checkout and exact archive ran 75
tests: 72 passed and 3 conditional CUDA tests skipped, with no failure or
error. The focused Stage 4 module ran 22 tests: 21 passed and its one CUDA
test skipped. Independent numerical Review also checked 545 binary64 support
cases, all six semantic axis orders in both floating dtypes, noncontiguous
convolution, randomized analog/ADC references, ambient CPU autocast, donor
hashes, freshness, and autograd without finding a discrepancy.

Pyright `1.1.408` reported 0 errors, warnings, or informational findings
against both the TensorCore source checkout and the independently extracted
archive. Import isolation returned `False False False False`; `git diff
--check`, exact public/private surfaces, dependency/import direction, and
generated-artifact scans passed. The evidence environment was Python
`3.13.11`, PyTorch `2.12.1`, macOS `15.7.4` on arm64, and eager execution.
CUDA and a CUDA runtime were unavailable, and MPS was unavailable, so this is
CPU evidence and makes no GPU or performance claim. The `build` and
`hatchling` modules were unavailable, so no editable-install or wheel-build
claim is made.

This Review-owned closeout changes only this work order and the implementation
index. Cleared production, tests, README, metadata, and synchronized Design
bytes remain exactly those merged at candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da`.

## Objective

Implement the smallest complete deterministic waveform slice on top of the
closed TensorCore `0.7` product foundation:

- `_product_pure_waveform(...)` for both accepted TPC FEB-SNR and Veto PDU
  pulse models;
- `_product_analog_waveform(...)` for direct composition and optional physical
  saturation; and
- `_product_digitized_waveform(...)` for the accepted endpoint-guarded affine
  ADC transfer and truncating `torch.int32` conversion.

The stage is functionality-first. It establishes scientific equations,
functional tensor behavior, exact product results, storage freshness,
autograd behavior, and reference evidence. It makes no kernel-count,
fusion, target-temporary, allocation-free, throughput, or accelerator-
performance claim.

The complete noise producer is deferred. Stage 4 does **not** create
`_product_noise_waveform(...)`, even for `ZeroNoiseConfig`. Zero, white, and
PSD noise remain one coherent `NoiseWaveformConfig` product family and become
executable together under the focused Stage 5 work order. Analog tests use an
already-constructed valid `NoiseWaveform` fixture.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`.

The exact clean TensorDSLab starting baseline is `main` at:

```text
5ff13eb3c0735abfda454a334be59faac35259c2
```

That commit contains the closed Stage 3 implementation and Design closeout.
The exact Stage 3 implementation candidate remains:

```text
9250192587d1e05e71f09c9cda4ba9d0bce09bde
```

The documentation-only Stage 4 Design branch is:

```text
codex/stage-4-deterministic-waveform-design
```

Before dispatch, Design must commit this work order and every synchronized
Design source, verify a clean tree, and record that exact committed Design
authority as the implementation base. Implementation must not begin from the
uncommitted Design tree or directly from `5ff13eb3`; the dispatch commit is the
exact parent of the implementation branch.

The exact TensorCore dependency remains clean TensorCore `0.7.0` at:

```text
b454d738f6385ce6489d85492a618a3dab139bb6
```

No dependency move or TensorCore edit is in scope. TensorDSLab remains active-
development and pre-deployment. This stage makes no release, deployment,
backward-compatibility, conformance, broad compatibility, or GPU-performance
claim.

## Source Of Truth

Implementation, Validation, and Review must read and reconcile:

- [Agent Workflow](../../AGENTS.md);
- [Contributing](../../CONTRIBUTING.md), especially product ownership,
  boundary-first validation, result freshness, semantic exposure, autograd,
  and testing standards;
- [Rebuild Architecture](../architecture/rebuild.md), especially the private
  product-builder signatures, TPC and Veto pulse equations, analog equation,
  digitizer equation, functional/storage contract, and validation matrix;
- [Readout Architecture](../architecture/readout.md);
- [TensorCore Integration](../architecture/tensors.md);
- [Validation](../validation.md);
- [IV-DSLab Parity](../parity.md); and
- TensorCore `0.7.0` `docs/api.md`, `docs/architecture/tensors.md`,
  `docs/integration.md`, and public package implementation at the exact pin.

The functionality-first decision in this work order governs Stage 4 execution
acceptance. Synchronized live Design documents must describe fusion and
target-temporary elimination as later measured optimization rather than a
Stage 4 closure condition. If any live source still requires those performance
claims for this stage, stop before dispatch and return the contradiction to
Design.

Historical work orders and donor repositories are evidence. They do not
override the accepted rebuild equations or authorize copied architecture.

## Dispatch And Finite Role Loop

Before dispatch, Design must privately verify the persistent logical routes:

```text
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Each must be Active, current for this workspace, and able to return to Design.
Coordination remains Deferred and is not used. Raw route identifiers must not
appear in committed files.

The authorized loop is:

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

The loop allows at most three Implementation-to-Validation dispatches and at
most three Validation-to-Implementation returns. Review is read-only and
reviews a fixed Validation-cleared commit. A repeated finding, exhausted
budget, stale route, scope expansion, or required architecture choice returns
the work to Design.

Suggested implementation branch:

```text
codex/stage-4-deterministic-waveform-products
```

No push is authorized by this work order.

## Selected Stage Decisions

Stage 4 freezes these decisions:

1. There are exactly three new product producers: pure, analog, and
   digitized waveform.
2. Noise production, including exact zero noise, is deferred in full to
   Stage 5 rather than represented by a temporarily partial three-model
   dispatcher.
3. Production uses ordinary eager PyTorch. Target-sized eager intermediates
   are permitted. `torch.compile`, Triton, custom CUDA, compiler graphs,
   profiler traces, kernel counts, and memory instrumentation are neither
   required nor accepted as closure evidence.
4. Pulse configuration, sample-time, raw pulse, and sampled-extremum
   preparation use Python binary64 arithmetic. The normalized coefficient
   prefix is materialized exactly once in the input `Charge` dtype and device.
5. No field payload is moved to the host. Host-side preparation reads only
   immutable scalar config and sampling values.
6. Convolution and every analog and digitizer payload expression execute in
   the input tensor's dtype and device.
7. Analog bounds and ADC execution scalars are checked and rounded once in the
   field dtype before payload calculation, then used as exact zero-dimensional
   tensors on the field device.
8. Digitization uses an affine open interior plus inclusive comparisons
   against dtype-rounded pre-gain thresholds; this prevents upper-endpoint
   code loss from floating rounding.
9. Every generated field has guaranteed-fresh storage independent of every
   named input, reuses the exact source axes tuple and axis instances, and is
   constructed only after all producer writes have been initiated or
   enqueued.
10. Pure and analog production preserve ordinary PyTorch autograd. Integer
   digitization is explicitly nondifferentiable.
11. Arbitrary accepted semantic axis order and noncontiguous `torch.strided`
   inputs remain supported. This stage imposes no sample-last or contiguity
   execution profile.

Changing one of these decisions requires Design, not an Implementation
workaround.

## Exact Private Surfaces

### Shared Sampling Agreement

Stage 4 may add exactly this shared helper to
`tensor_dslab/readout/_requirements.py`:

```python
def _require_sampling(
    field: TensorField,
    sampling: SamplingConfig,
) -> None:
    ...
```

It is a narrow O(1) relationship check. It requires exact `SamplingConfig`,
locates the exact `SampleAxis`, and checks:

- sample-axis size equals `sampling.sample_count.value`;
- sample-axis start is zero; and
- sample-axis period equals `sampling.sample_period_ps.value`.

It must not call `sampling.build_axis()`, compare every coordinate, require a
fixed dimension, require contiguity, move data, parse timestamps in a payload
loop, or perform a full-device value scan. It raises `TypeError` for malformed
exact config type and `ValueError` for a valid but disagreeing field/grid
relationship. It is private and is not re-exported.

If Implementation can satisfy the same already-accepted relationship without
adding a shared helper or duplicating it, it may omit this modification. It
must not invent a broader generic validation layer.

### Pure Waveform Producer

Create `tensor_dslab/readout/pure_waveform/_product.py` with this exact main
signature:

```python
def _product_pure_waveform(
    charge: Charge,
    *,
    sampling: SamplingConfig,
    config: PureWaveformConfig,
) -> PureWaveform:
    ...
```

The producer receives an already-constructed, deeply trusted `Charge` and
accepted exact configs from future public preflight. It owns the narrow
sampling relationship and config-derived pulse preflight required for its own
calculation. Direct private calls with wrong semantic types, bypassed config
construction, mutated classes, custom dispatch tensors, or scientifically
invalid payloads are unsupported and need no defensive test matrix.

The output contract is:

| Property | Required result |
| --- | --- |
| exact type | `PureWaveform` |
| shape | exactly `charge.shape` |
| axes | the exact `charge.axes` tuple and exact axis instances |
| dtype | exactly `charge.tensor.dtype` (`torch.float32` or `torch.float64`) |
| device | exactly `charge.tensor.device` |
| Torch layout | `torch.strided`; no exact stride or contiguity promise |
| storage | guaranteed fresh and non-aliasing with `charge.tensor` |
| input effects | no mutation, move, cast, detach, host materialization, or write |
| autograd | preserves the differentiable dependency on `charge.tensor` |
| synchronization | ordinary current-stream ordering; no explicit device synchronization |
| exposure | construct the field only after its tensor writes are enqueued |

The config-derived pulse coefficients do not require gradients.

#### Binary64 Pulse Preparation

Use `sampling.sample_period_ps.value` as the numeric period. Convert it once to
Python binary64 nanoseconds for the config equations. Sample raw pulse values
at integer left-edge indices satisfying:

```text
t_j = j * sample_period_ns
0 <= t_j < support_time_ns
```

The exclusive support stop is normative. Deriving the count with a ceiling is
an implementation detail; exact inclusion is governed by the binary64
left-edge comparison above. Preflight requires
`1 <= template_sample_count <= 2**63 - 1` and rejects a nonfinite or
out-of-range derivation before materializing a tensor.

Evaluate the complete configured sampled support in Python binary64 to obtain
the normalization extremum. Reject an empty, nonfinite, or all-zero sampled
template. Normalize by the magnitude of the complete sampled extremum and
apply the signed `peak_voltage_mv_per_pe` exactly once:

```text
coefficient[j] = (
    raw[j]
    / max_k(abs(raw[k]))
    * peak_voltage_mv_per_pe
)
```

Only the prefix `j < min(template_sample_count, sampling.sample_count.value)`
can contribute to the returned finite record. Materializing only that
normalized prefix is permitted, but normalization must still use the full
configured support. Before device materialization, scalar preflight must prove
that every retained coefficient remains finite in `charge.tensor.dtype` and
that the normalized full-support extremum remains nonzero and finite in that
dtype. Create one coefficient tensor directly in `charge.tensor.dtype` on
`charge.tensor.device`. CPU scalar conversion used solely for this config-
derived representability check is permitted; a second coefficient-vector
tensor is not. Do not call `.cpu()`, `.numpy()`, `.tolist()`, or otherwise
materialize the charge payload on the host.

No cache, global template constant, persistent template bank, channel lookup,
or hidden model default is introduced. Ordinary backend/resource failure for
an impractically large but structurally valid support carries no rollback or
resource-guarantee promise.

#### TPC FEB-SNR Equation

For exact `TpcFebSnrPulseConfig`, use:

```text
raw_tpc(t) = (
    exp(-t / slow_time_constant_ns)
    - exp(-t / fast_time_constant_ns)
)
```

The config already requires `slow_time_constant_ns > fast_time_constant_ns`.
The slow constant is the actual IV denominator
`tau_r + tau_l`; it is not IV's `tau_l` alone.

#### Veto PDU Equation

For exact `VetoPduPulseConfig`, define:

```text
x = t - gaussian_center_ns

raw_veto(t) = (
    exp(-x**2 / (2 * gaussian_width_ns**2))
    / sqrt(2 * pi * gaussian_width_ns**2)
    * (1 + erf(
        (x - edge_offset_1_ns)
        / (sqrt(2) * edge_width_1_ns)
    ))
    * (1 + erf(
        (x - edge_offset_2_ns)
        / (sqrt(2) * edge_width_2_ns)
    ))
)
```

Use Python `math` binary64 functions for config-derived sampling. Do not add
SciPy, NumPy, a donor import, or a hidden alternate equation.

#### Causal Convolution

Locate the sample dimension by exact `SampleAxis` type. Evaluate the same-
length causal convolution in the charge dtype/device:

```text
pure[..., t] = sum(
    charge[..., t - j] * coefficient[j]
    for j in 0 .. min(t, coefficient_count - 1)
)
```

The notation places sample last only for readability. The implementation must
work for any accepted axis order and for noncontiguous strided input. It may
use views, reordering views, reshaping that allocates, ordinary eager
`torch.nn.functional` operations, and target-sized intermediates. It must
restore the source semantic dimension order in the result and must not impose
a public stride guarantee.

There is no baseline, second gain, post-convolution inversion, fractional-bin
amplitude correction, stochastic step, or output-length extension. An all-
zero charge produces exact all-zero signal values.

### Analog Waveform Producer

Create `tensor_dslab/readout/analog_waveform/_product.py` with exactly:

```python
def _product_analog_waveform(
    pure: PureWaveform,
    noise: NoiseWaveform,
    *,
    config: AnalogWaveformConfig,
) -> AnalogWaveform:
    ...
```

Before payload arithmetic, require equal ordered axes, shape, device, and
exact dtype. Use TensorCore's public `require_same_axes(...)` and
`require_same_device(...)` where applicable; do not duplicate or privately
import TensorCore validation. Explicit dtype equality remains TensorDSLab's
product relationship. Broadcasting between unequal product shapes is not
accepted.

Evaluate directly:

```text
analog[i] = clamp(
    pure[i] + noise[i],
    optional minimum_mv,
    optional maximum_mv,
)
```

Either bound may be absent. With both absent, the expression is addition.
Bounds are scalar config values applied uniformly to every tensor position.
The clamp is physical analog/front-end saturation. Do not add a pedestal,
baseline, channel calibration, or ADC-range clamp.

Before adding the payloads, convert every present bound through the common
input dtype on the CPU, reject a nonfinite conversion, and, when both bounds
are present, require the rounded minimum to remain strictly below the rounded
maximum. Materialize those exact rounded values as zero-dimensional tensors in
the common input dtype/device and use them for the clamp. This config-only
preflight must complete before `pure + noise`; it does not inspect, move, or
synchronize either input payload.

The output contract is:

| Property | Required result |
| --- | --- |
| exact type | `AnalogWaveform` |
| shape/axes | exactly `pure.shape` and the exact `pure.axes` tuple/instances |
| dtype/device | exactly the common input dtype/device |
| Torch layout | `torch.strided`; no exact stride or contiguity promise |
| storage | guaranteed fresh and non-aliasing with both named inputs |
| input effects | neither input is mutated, moved, cast, detached, or host-materialized |
| autograd | preserves differentiable dependencies on both inputs; no custom derivative promise exactly at clamp boundaries |
| synchronization | ordinary current-stream ordering; no explicit device synchronization |
| exposure | field construction follows enqueue of all result writes |

Do not add `_apply_analog_saturation(...)` or another one-expression Python
wrapper. Eager addition followed by eager clamp may materialize an intermediate
in Stage 4; that is accepted and carries no performance claim.

### Digitized Waveform Producer

Create `tensor_dslab/readout/digitized_waveform/_product.py` with exactly:

```python
def _product_digitized_waveform(
    analog: AnalogWaveform,
    *,
    config: DigitizedWaveformConfig,
) -> DigitizedWaveform:
    ...
```

Compute scalar transfer constants once with Python binary64 arithmetic:

```text
maximum_code = 2**bit_depth - 1
gain = 10**(analog_gain_db / 20)
span = input_max_mv - input_min_mv
slope = gain * maximum_code / span
intercept = -input_min_mv * maximum_code / span
lower_input_mv = input_min_mv / gain
upper_input_mv = input_max_mv / gain
```

Before payload arithmetic, require every derived binary64 value to be finite,
require `span > 0` and `slope > 0`, and require every scalar used in field-
dtype arithmetic to remain finite and nondegenerate when represented in
`analog.tensor.dtype`. Concretely, CPU scalar conversion to that dtype must
remain finite; `gain`, `span`, and `slope` must remain strictly positive;
`maximum_code` must remain exact; and the rounded input-domain thresholds must
satisfy `lower_input_mv < upper_input_mv`. `intercept` may be zero. Preserve
the exact rounded threshold, `maximum_code`, slope, and intercept values from
that preflight and materialize them as zero-dimensional tensors in the analog
dtype/device. Those device scalars, rather than independently wrapped Python
values, are the payload execution constants. This preflight must not read or
move the analog payload. A scalar-preflight failure occurs before any product
tensor write.

The affine interior is:

```text
interior[i] = clamp(
    analog[i] * slope + intercept,
    0,
    maximum_code,
)

code_float[i] =
    0,                         if analog[i] <= lower_input_mv
    maximum_code,              if analog[i] >= upper_input_mv
    interior[i],               otherwise

digitized[i] = int32(code_float[i])
```

The inclusive threshold guards are normative. They make both configured ADC
endpoints exact in the accepted field-dtype execution even when floating
multiply/add rounding would place the affine upper endpoint just below
`maximum_code`. Compare directly in the pre-gain analog input domain; do not
recompute an endpoint decision through `analog * gain`. The floating clamp and
endpoint selection occur before integer conversion. Because selected values
are nonnegative, ordinary float-to-`torch.int32` conversion provides the
accepted truncation-toward-zero rule for the open interior. Do not cast before
clipping, use `torch.uint16`, add an ADC pedestal, or add a quantization enum.

For readable validation, the equivalent reference equation is:

```text
gained_mv = analog_mv * gain
clipped_mv = clamp(gained_mv, input_min_mv, input_max_mv)
scaled_code = (
    (clipped_mv - input_min_mv)
    / span
    * maximum_code
)
interior_reference_code = int32(clamp(scaled_code, 0, maximum_code))
```

Apply the same `analog <= lower_input_mv` and
`analog >= upper_input_mv` endpoint guards around
`interior_reference_code`. The gained/
clipped and affine interior equations are algebraically equivalent in real
arithmetic; they are not promised to be floating-point identical near code
transitions. The guarded affine form is the production execution contract.
Tests near code transitions must choose values deliberately and account for
the exact accepted field-dtype scalar values rather than inferring cross-
backend bitwise identity.

The output contract is:

| Property | Required result |
| --- | --- |
| exact type | `DigitizedWaveform` |
| shape/axes | exactly `analog.shape` and exact `analog.axes` tuple/instances |
| dtype | exact `torch.int32` |
| device | exactly `analog.tensor.device` |
| Torch layout | `torch.strided`; no exact stride or contiguity promise |
| storage | guaranteed fresh and non-aliasing with `analog.tensor` |
| input effects | no mutation, movement, cast, detach, or host materialization |
| autograd | digitized result is nondifferentiable and has `requires_grad=False` |
| synchronization | ordinary current-stream ordering; no explicit device synchronization |
| exposure | field construction follows enqueue of all result writes |

Do not add `_digitize(...)`, `_apply_adc_transfer(...)`, or another one-line
wrapper.

## Shared Functional, Failure, And Trust Contract

These private producers are independently testable implementation seams, not
supported collaborator APIs. They receive native fields/configs that future
public preflight has already validated. Stage 4 therefore adds only narrow
operation-specific relationship and scalar checks; it does not repeat full-
device source scans or harden direct private calls.

The accepted product value domain is finite field-dtype execution over deeply
valid inputs in a representable numerical range. Tests invoke each existing
product-local `_require_valid_values(...)` on returned representative results.
The producers must not add hidden full-device `.item()` postcondition scans
that synchronize every call. The later public `simulate_readout(...)` work
order must close any remaining ingress/result trust boundary before publishing
these producers through a public operation.

Every preflight failure precedes its producer's payload computation and leaves
named inputs unchanged. No RNG state exists in this stage. Once a backend
operation has launched, an exceptional failure has no transactional rollback
guarantee, but no caller-owned output target or semantic result has been
published. A successful producer constructs its semantic field once and
initiates no later write through any alias to its storage.

CPU execution follows ordinary synchronous Torch behavior. Accelerator
execution is enqueued on the current stream and returns without explicit host
synchronization. Same-stream consumers inherit ordinary ordering. Cross-stream
consumers establish their own dependency. CUDA absence is an accurate test
skip and creates no GPU claim.

## Exact Public API And Import Boundary

Stage 4 adds no public symbol. Do not modify any package `__init__.py` or
`__all__`. In particular, none of these becomes public:

```text
_require_sampling
_product_pure_waveform
_product_analog_waveform
_product_digitized_waveform
```

Product producer imports follow the accepted acyclic direction:

```text
TensorCore/common/private shared requirements
  -> prerequisite product types
  -> owning product producer
```

No producer imports `ReadoutConfig`, `ReadoutCollection`, a package root,
`simulate_readout`, another downstream producer, or a deferred integration
package. Use only public `tensor_core` package-root imports.

Stage 4 does not create `readout/simulation.py`, `readout/_random.py`,
`charge/_product.py`, `noise_waveform/_product.py`, or
`photoelectrons/_product.py`.

## Exact Candidate Change Allowlist

From the exact committed Design dispatch base to the fixed implementation
candidate, only these paths may change:

```text
M  README.md
M  tensor_dslab/readout/_requirements.py              # only if _require_sampling is used
A  tensor_dslab/readout/pure_waveform/_product.py
A  tensor_dslab/readout/analog_waveform/_product.py
A  tensor_dslab/readout/digitized_waveform/_product.py
M  tests/test_package_contracts.py
A  tests/test_deterministic_waveform_products.py
A  tests/typing/stage_4_deterministic_waveform_products.py
```

If `_require_sampling` is unnecessary without duplication, leave
`_requirements.py` byte-identical and report that narrower candidate. No other
production, test, metadata, config, initializer, typing, documentation, or
dependency file may change in the implementation candidate.

`README.md` must state precisely that three private deterministic producers
exist while `simulate_readout(...)`, charge, all noise production, RNG, and
integration remain unavailable. It must not tell ordinary collaborators to
import private modules.

After the exact candidate clears Validation and Review, Review may update only:

```text
docs/implementation/stage_4_deterministic_waveform_products.md
docs/implementation/index.md
```

Those closeout edits record exact evidence/status only and must not change the
cleared production, tests, README, architecture, parity, decisions,
governance, or prior work-order bytes.

The following remain protected and unchanged throughout the implementation
candidate and closeout unless Design issues a replacement work order:

- every product `types.py` and public config;
- every package initializer and public export tuple;
- `ReadoutConfig` and `ReadoutCollection`;
- `pyproject.toml`, `pyrightconfig.json`, `LICENSE`, and `py.typed`;
- all governance records and completed work orders;
- architecture, design, decisions, parity, and validation authority; and
- TensorCore and every sibling repository.

## Required Tests

### Focused Runtime Test Module

Create `tests/test_deterministic_waveform_products.py`. It owns small local
fixtures and independent scalar/reference helpers needed by this stage. Do not
create a generic producer-test framework or modify shared fixtures merely to
avoid a few explicit lines.

At minimum, implement tests equivalent to:

```text
test_tpc_pure_waveform_matches_binary64_reference
test_veto_pure_waveform_matches_binary64_reference
test_pure_waveform_support_is_left_closed_right_open
test_pure_waveform_normalizes_over_complete_support_before_record_crop
test_pure_waveform_is_causal_same_length_and_zero_baseline
test_pure_waveform_handles_alternate_axis_order_and_noncontiguous_input
test_pure_waveform_reuses_axes_is_fresh_and_preserves_autograd
test_sampling_relationship_rejects_size_start_and_period_mismatch
test_analog_waveform_matches_unbounded_and_each_saturation_form
test_analog_waveform_rejects_axis_device_or_dtype_disagreement
test_analog_waveform_rejects_nonfinite_or_collapsed_dtype_bounds_before_addition
test_analog_waveform_is_fresh_preserves_inputs_and_autograd
test_digitized_waveform_matches_guarded_affine_reference
test_digitized_waveform_endpoints_zero_code_and_gain
test_digitized_waveform_rejects_nonfinite_or_collapsed_dtype_scalars_before_mapping
test_digitized_waveform_truncates_at_code_transitions
test_digitized_waveform_clips_before_int32_conversion_without_wraparound
test_digitized_waveform_is_fresh_int32_and_nondifferentiable
test_deterministic_products_run_conditionally_on_cuda
```

Equivalent organization is acceptable, but every named behavior must be
visible in focused assertions.

The endpoint fixtures must include at least one nontrivial-gain case in each
floating dtype for which the unguarded affine expression evaluates just below
`maximum_code`; the guarded producer must still return the exact maximum code
at its dtype-rounded `upper_input_mv`. Preflight fixtures use finite binary64
values that become nonfinite or collapse after `torch.float32` conversion and
prove rejection before analog addition or digitizer mapping.

Test both `torch.float32` and `torch.float64`. Test the default
example/channel/sample order and at least one alternate order. At least one
pure producer case must start from a noncontiguous valid `Charge`, and at least
one analog case must use noncontiguous valid inputs. Do not require the output
to reproduce the source strides or be contiguous.

For each producer prove:

- exact returned semantic class;
- exact shape, dtype, device, axes tuple identity, and axis-instance identity;
- guaranteed-fresh storage through a public Torch storage-identity check;
- unchanged input values and tensor version counters;
- absence of implicit movement, cast, detach, or host conversion;
- explicit product deep validator success on representative results; and
- no later invocation mutates an earlier returned result.

Do not mutate an exposed input tensor merely to test unsupported caller
behavior. Inspect storage identity and producer effects instead.

Pure and analog autograd tests use floating leaf inputs with
`requires_grad=True`, call `backward()`, and compare the resulting input
gradients with an independent differentiable reference away from clamp
boundaries. At least one representative float64 case for each differentiable
producer also passes `torch.autograd.gradcheck(...)`. Digitization tests prove
`requires_grad is False` and do not claim or fabricate a straight-through
estimator.

### Numerical Tolerances

For pure waveform value and gradient comparison over the named 8 ns donor
fixtures and the focused bounded matrices in this work order, use:

| dtype | relative tolerance | absolute tolerance |
| --- | ---: | ---: |
| `torch.float32` | `2e-5` | `2e-6` mV |
| `torch.float64` | `1e-12` | `1e-12` mV |

These tolerances cover field-dtype convolution arithmetic after binary64
coefficient preparation for those named cases. They are not a package-wide
forward-error bound over arbitrary support lengths or pulse amplitudes. The
dedicated impulse/template fixtures must also assert exact zeros where the
causal equation is zero and the exact dtype-rounded realized sampled-template
peak magnitude/sign when the record contains that peak, so a small nonzero
pulse cannot pass as all zero under the absolute tolerance. Any additional
fixture scales its absolute tolerance with the configured peak magnitude and
records that choice rather than reusing `2e-6` blindly. Gradient comparison
uses the same numeric tolerances after interpreting the absolute unit as mV
per PE-equivalent input. These tolerances do not authorize a different
equation, normalization, support edge, sign, or sample alignment. A failure
outside the tolerance is a finding; Implementation may not widen it without
Design.

Use exact equality for integer ADC results, exact dtype/device/type/axes
claims, source immutability, and simple analog fixtures whose expected eager
operations are exactly representable. Otherwise use zero relative/absolute
tolerance against the same accepted eager analog reference operation on the
same backend. Conditional CUDA uses the same dtype-specific pure tolerances
but makes no CPU/CUDA bitwise claim.

### IV-DSLab Parity Fixture Duties

The audited IV source for pulse equations is:

```text
Projects/iv-dslab-main_db_PB/src/dselec/waveform.py
SHA-256: 5eb5b29e6958184e520b2151877a678f6d98cdbe6e53cbf9d1b4c4e64e0f82b5
```

The audited calibration source is:

```text
Projects/iv-dslab-main_db_PB/data/config_files/dselec.ini
SHA-256: fd42244bb4405dc328496efb8043fff522584a1922b811246670ac0e940e1c64
```

Tests must not import or execute IV-DSLab or DSLab. Check in small,
reviewable TensorDSLab-owned literal checkpoints or expected arrays and record
their donor path/hash, units, sampling, equation mapping, and intentional
differences in a nearby comment or fixture docstring.

At 8 ns sampling, exercise these reviewed parameter mappings:

| Model | TensorDSLab fixture values |
| --- | --- |
| TPC FEB-SNR | `fast_time_constant_ns=83`, `slow_time_constant_ns=383`, `support_time_ns=3000`, `peak_voltage_mv_per_pe=-7` |
| Veto PDU | `gaussian_center_ns=232.89`, `gaussian_width_ns=507.72`, `edge_offset_1_ns=-81.92`, `edge_width_1_ns=147.28`, `edge_offset_2_ns=-176.50`, `edge_width_2_ns=45.69`, `support_time_ns=2020.27`, `peak_voltage_mv_per_pe=-14.5912372` |

The comparison is numerical parity for the mapped equation family, not
literal donor output identity. Tests must retain the accepted differences:

- sampled-extremum normalization for both models;
- explicit half-open support;
- signed peak applied once;
- no donor post-convolution inversion switch;
- no eventwise fractional-bin correction; and
- same-length tensor-native causal convolution.

For the IV-aligned pulse checkpoints, use `rtol=1e-4` and
`atol=1e-5` mV. That fixture tolerance accommodates the documented roughly
66-part-per-million TPC sampled-versus-continuous normalization difference; it
does not replace the tighter TensorDSLab-equation tolerances above.

Analog composition is exact conditional parity for aligned inputs and bounds.
Open-interior digitization is exact after representation mapping when its
floating input lands away from an ambiguous transition. Inclusive endpoint
saturation is an intentional numerical correction where literal affine/donor
operation order would lose one code to rounding. Pre-conversion clipping is an
intentional divergence from IV unsigned wraparound and must have explicit out-
of-range fixtures.

### Package And Absence Tests

Modify `tests/test_package_contracts.py` only to:

- remove the three newly real `_product.py` paths from the future-placeholder
  absence assertion;
- retain explicit absence of `noise_waveform/_product.py`,
  `charge/_product.py`, `photoelectrons/_product.py`, `readout/_random.py`, and
  `readout/simulation.py`;
- prove the three producer names and `_require_sampling` are absent from
  product, readout, and package-root public exports;
- prove producer modules import without TensorG4DS, TensorML, DSLab, IV-DSLab,
  G4DS/g4ds11, NumPy, SciPy, IO, or orchestration dependencies; and
- preserve the public-TensorCore-import and acyclic product-import checks.

Do not weaken any Stage 3 leaf, config, collection, metadata, or retired-name
test.

### Static Typing Probe

Create `tests/typing/stage_4_deterministic_waveform_products.py` with positive
`typing.assert_type` probes requiring:

```text
_product_pure_waveform(...) -> PureWaveform
_product_analog_waveform(...) -> AnalogWaveform
_product_digitized_waveform(...) -> DigitizedWaveform
```

The probe must construct or receive exact Stage 3 semantic/config types and
analyze against the exact TensorCore pin. It must not use `Any`, casts, ignored
diagnostics, private TensorCore imports, or public re-exports of the producer
functions.

## Verification Commands

Implementation and fixed-commit Validation must run at least:

```bash
git status --short --branch
git diff --check
git diff --name-status <design-dispatch-commit>..<candidate-commit>
git -C /Users/mbedard/Projects/TensorCore rev-parse HEAD
git -C /Users/mbedard/Projects/TensorCore status --short
git -C /Users/mbedard/Projects/TensorCore archive --format=zip --output=/tmp/tensorcore-stage4-b454d738.zip b454d738f6385ce6489d85492a618a3dab139bb6
shasum -a 256 /tmp/tensorcore-stage4-b454d738.zip
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/tmp/tensorcore-stage4-b454d738.zip python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -c "import sys, tensor_dslab; print('tensor_g4ds' in sys.modules, 'tensor_ml' in sys.modules, 'dslab' in sys.modules, 'g4ds11' in sys.modules)"
pyright
```

Recreate the archive from the exact commit for every fixed candidate; do not
reuse an unverified old archive. Record its SHA-256. Validation must also
extract the exact archive outside the repository and run the static checker
with that extracted package as the only TensorCore analysis path. The source-
checkout and extracted-archive typing results must both be recorded.

Also report:

- exact Python, PyTorch, TensorCore, and static-checker versions;
- eager execution mode and whether CUDA is available;
- total test count, passes, failures, errors, and conditional skips;
- focused test counts for both pulse models, both floating dtypes, alternate
  axes, noncontiguous inputs, autograd, saturation, and ADC boundaries;
- exact public export tuples and private-producer absence from them;
- no production `.cpu()`, `.numpy()`, `.tolist()`, detach, input cast, device
  move, donor import, NumPy/SciPy import, hidden RNG, or global-state use;
- no `__pycache__`, `.pyc`, `.pyo`, coverage output, build artifact, or other
  generated file in the final tree; and
- a clean candidate and final `main` worktree.

If CUDA is unavailable, conditional accelerator tests skip and no GPU result
or performance claim is made. If Pyright or an equivalent Design-accepted
checker is unavailable, static typing clearance is blocked. A build or
editable install may be reported if available without changing source, but it
is not a Stage 4 closure requirement because metadata is unchanged.

Do **not** run `torch.compile`, profiler, kernel-count, allocator, or peak-
memory tests as substitute closure evidence. Such exploratory evidence may be
reported separately but cannot change Stage 4 scope or status.

## Validation Report

Validation evaluates only a fixed committed candidate and returns:

- exact Design base, candidate commit, parent, branch, and changed-path
  reconciliation;
- exact TensorCore source and independently archived-pin evidence;
- every verification command and exact result;
- scientific/reference results and tolerances for both pulse models;
- analog and digitizer boundary results;
- autograd, freshness, axes, dtype, device, layout, exposure, and input-effect
  evidence;
- static typing and import-isolation evidence;
- CPU/CUDA and environment qualifications;
- all findings ordered by severity; and
- an explicit `Cleared` or `Returned to Implementation/Design` disposition.

Validation does not edit the feature branch. An equation, precision,
normalization, support-edge, autograd, public-surface, or scope question goes
to Design rather than being improvised in a test.

## Independent Review

Review examines the exact Validation-cleared bytes and reports findings first.
At minimum it verifies:

- every changed path is allowlisted and every deferred path remains absent;
- the three signatures and import directions are exact;
- Python binary64 preparation is config-derived only and no input payload is
  host-materialized;
- output dtype/device/axes/freshness and semantic-exposure contracts are
  actually established by code and tests;
- pure convolution is causal, same-length, axis-order-independent, and uses
  the complete-support sampled-extremum normalization;
- TPC/Veto equations, mappings, support, signed peak, and parity fixtures are
  faithful to the accepted sources;
- analog composition has no baseline, scalar bounds are preflighted before
  addition, and digitization applies its dtype-rounded endpoint guards and
  floating clamp before `int32`;
- autograd is preserved only where claimed;
- private functions remain private and public exports are byte-for-byte
  unchanged;
- eager functionality is not presented as fusion or performance evidence;
- tests are independent enough to catch equation/order errors rather than
  restating implementation helpers; and
- no protected architecture, governance, metadata, dependency, or sibling
  repository changed.

Review clears one exact commit or returns actionable findings to
Implementation. It must return architecture or scope findings to Design.

## Non-Goals And Forbidden Scope

- No `_product_noise_waveform(...)`, including no zero-only partial producer.
- No white-noise or PSD synthesis, FFT, RNG, seed handling, stream assignment,
  `_random.py`, Threefry, uniform, normal, or other stochastic primitive.
- No `Charge` production, dark counts, timing jitter, correlated avalanches,
  charge smearing, or `charge/_product.py`.
- No `simulate_readout(...)`, request parsing, dependency planner, final
  retention, public transform, `simulation.py`, or partial public skeleton.
- No public producer export, convenience method, product method, new config,
  changed config, sidecar, enum, registry, ID, constant, or collection change.
- No analog baseline/pedestal, per-channel calibration, tensor-valued config,
  hidden gain, alternate pulse equation, fractional-bin correction, or
  cross-product fusion.
- No `torch.compile`, custom autograd, straight-through digitizer, Triton,
  CUDA/C++ extension, custom kernel, compiler fallback, kernel-count promise,
  target-temporary prohibition, allocation-free claim, or performance gate.
- No `out=`, destination, workspace, allocator, pool, lease, stream API,
  mutable field, selection, movement, batching, reconstruction, or lifecycle
  surface.
- No `.cpu()`, `.numpy()`, `.tolist()`, host materialization of an input
  payload, silent movement, input cast, detach, or explicit device
  synchronization.
- No source `Photoelectrons` construction, PE binning, TensorG4DS bridge,
  TensorML/Reconstruction adapter, IO, cache, persistence, artifact, DAG,
  campaign, deployment, or compatibility surface.
- No TensorCore or sibling-repository edit, private TensorCore import, fork,
  shim, or dependency move.
- No donor runtime or dependency on IV-DSLab, DSLab, NumPy, SciPy, Numba,
  pyFFTW, or external condition/calibration loading.
- No adversarial hardening/tests for final-leaf subclassing, class mutation,
  constructor bypass, direct private misuse, exposed-tensor mutation, or
  custom Torch dispatch.
- No broad GPU, compatibility, conformance, release, deployment, or scientific
  detector-validity claim.

## Return To Design Before

Return before making any change that would:

- alter one frozen equation, parameter mapping, unit, sampled support edge,
  binary64 preparation rule, normalization rule, sign, convolution alignment,
  saturation meaning, ADC transfer, or truncation rule;
- add or remove a producer, including promoting zero noise into Stage 4;
- require a public API/export or `ReadoutCollection` orchestration decision;
- change any Stage 3 field/config/axis/collection contract;
- change TensorCore, its pin, or a sibling repository;
- require NumPy/SciPy, donor runtime, payload host materialization, input
  movement/casting/detachment, or explicit device synchronization;
- weaken freshness, axes identity, source immutability, semantic exposure,
  or autograd contracts;
- require contiguity/sample-last execution or reject a supported arbitrary
  semantic axis order merely for implementation convenience;
- make performance, fusion, memory, GPU, compatibility, or deployment a
  closure condition;
- widen a numerical tolerance or supported equation domain to make a failing
  reference test pass;
- broaden the path allowlist or edit protected/historical/governance sources;
- proceed with a dirty or divergent base, stale/missing role, missing static
  checker, unexplained artifact, or nonreproducible fixed dependency; or
- exceed the finite Implementation/Validation loop.

## Merge And Closeout

Stage 4 becomes **Merged / Closed** only when:

1. Design commits the synchronized authority and explicitly dispatches its
   exact commit;
2. the implementation candidate has that exact parent and only allowlisted
   changes;
3. source-checkout and independent exact-pin archive tests pass on one fixed
   candidate;
4. mandatory static typing passes against both dependency forms;
5. all scientific/reference, functional, storage, autograd, package, and
   absence tests pass with accurate CUDA qualifications;
6. Validation has no unresolved finding;
7. independent Review clears the exact same bytes;
8. Review performs `git merge --ff-only` to clean `main` and repeats the
   required post-merge checks;
9. Review changes only this work order and the implementation index to record
   exact candidate, merge, commands, counts, environment, skips, residual
   qualifications, and no-push state;
10. final `main` is clean and contains no generated artifact; and
11. TensorDSLab Design accepts the closeout.

Closure establishes only three private deterministic waveform producers.
Noise production, RNG, charge, public readout orchestration, measured GPU
optimization, cross-package integration, durable artifacts, conformance,
Coordination, Profile B, deployment, and push remain unchanged and
undispatched.
