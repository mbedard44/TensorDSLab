# Stage 5 Readout RNG And Stochastic Noise Work Order

Status: **Design-complete / Undispatched**. This documentation-only work order
defines a future production slice. It does not authorize Implementation,
Validation, Review, merge, or push. Design must explicitly dispatch the exact
committed authority after privately verifying every required route.

## Objective

Implement the smallest complete stochastic waveform slice on top of the closed
TensorCore `0.7` product foundation and Stage 4 deterministic producers:

- the private positional `tensordslab.threefry4x32-20/v1` raw engine;
- only the fixed-point uniform conversion and Box-Muller behavior consumed by
  noise;
- the complete `_product_noise_waveform(...)` family for exact zero, IID white,
  and caller-supplied PSD-shaped noise; and
- focused eager CPU plus conditional eager CUDA evidence.

The stage is functionality-first. It establishes exact raw-word identity,
position-addressed stochastic meaning, the accepted noise laws, fresh result
storage, source independence, nondifferentiable generated noise, and reference
evidence. It makes no compiler, fusion, kernel-count, target-temporary,
allocation-free, throughput, or accelerator-performance claim.

Stage 5 implements no Bernoulli, exponential, Poisson, categorical,
multinomial, rejection, source-quantum, or iterative-generation consumer.
Those mechanics remain later Charge work even where their scientific equations
are already selected in architecture.

## Authority And Exact Baselines

Package authority is `TensorDSLab/default/Design`.

The exact clean TensorDSLab starting baseline is `main` at:

```text
9ee84bf44a3a84e7e2d57d21362e79cc850f8e26
```

That commit is the accepted Stage 4 Design closeout. The exact Stage 4
implementation candidate remains:

```text
3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da
```

The documentation-only Stage 5 Design branch is:

```text
codex/stage-5-rng-and-noise-design
```

The target production branch after explicit dispatch is:

```text
codex/stage-5-readout-rng-and-stochastic-noise
```

This committed work-order path is the stable work-order key and the normative
source for the Stage 5 lifecycle vocabulary. The current package/state snapshot
is:

```text
package_adoption_state: Adopted
conformance_finding: Not evaluated
coordination_status: Deferred
registry_storage_profile: Disabled
stage_4: Merged / Closed
stage_5: Design-complete / Undispatched
```

The only permitted Stage 5 execution states are:

```text
Design-complete / Undispatched
Dispatched / Active
Implementation candidate / Validation pending
Returned / Implementation correction
Validation-cleared / Review pending
Review-cleared / Merge authorized
Merged / Design acceptance pending
Merged / Closed
Returned to Design
Blocked
```

`Returned to Design` and `Blocked` terminate the current execution attempt and
authorize no scope expansion. Design alone dispatches the stage and accepts its
final closeout; Implementation, Validation, and Review report the intermediate
dispositions defined here.

The exact committed on-disk package sources at the named TensorDSLab baseline
are authoritative. Earlier exported or contextual copies that still describe
Stage 4 as undispatched are stale evidence and do not override clean `main`,
the Stage 4 closeout chain, or this user-authorized documentation pass. This
work order itself still authorizes no Stage 5 production.

Before dispatch, Design must commit this work order and every synchronized live
Design source, verify a clean tree, and name that exact commit in the dispatch.
Implementation must branch from that committed Design authority, not directly
from `9ee84bf` or an uncommitted documentation tree.

The exact TensorCore dependency remains clean TensorCore `0.7.0` at:

```text
b454d738f6385ce6489d85492a618a3dab139bb6
```

No dependency move or TensorCore edit is in scope. TensorDSLab remains
active-development and pre-deployment. This work order makes no release,
deployment, backward-compatibility, conformance, broad compatibility, or GPU
performance claim.

## Source Of Truth

Implementation, Validation, and Review must read and reconcile:

- [Agent Workflow](../../AGENTS.md);
- [Contributing](../../CONTRIBUTING.md), especially boundary-first validation,
  product ownership, private-surface discipline, result freshness, random-field
  rules, semantic exposure, and testing standards;
- [Rebuild Architecture](../architecture/rebuild.md), especially Noise
  Waveform, RNG And Positional Repeatability, Functional/Memory/Lifetime, and
  Validation Strategy;
- [Readout Architecture](../architecture/readout.md);
- [Validation](../validation.md);
- [IV-DSLab Parity](../parity.md), especially the white-noise, PSD-noise, and
  RNG backend boundaries;
- TensorCore `0.7.0` `docs/api.md`, `docs/architecture/tensors.md`,
  `docs/integration.md`, and public package implementation at the exact pin;
- Random123 `1.14.0` commit
  [`726a093`](https://github.com/DEShawResearch/random123/commit/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13),
  especially `threefry.h`, `u01fixedpt.h`, and its independent known-answer
  vectors;
- the [Random123 paper](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  as the counter-based parallel-RNG rationale; and
- the [JAX PRNG design](https://docs.jax.dev/en/latest/jep/263-prng.html) as
  supporting functional/array-oriented Threefry rationale.

Random123 is a normative algorithm/reference source, not a runtime dependency.
The paper and JAX design are rationale evidence; they do not override
TensorDSLab's exact schema, streams, bounds, or execution contract.
Historical work orders and donor repositories are evidence and do not override
this focused scope.

If a live source still requires compiled execution, Charge-only distributions,
public RNG, coordinate-addressed random identity, exact CPU/CUDA completed
Gaussian values, or an exactly zero floating `irfft` sample mean as Stage 5
closure, stop before dispatch and return the contradiction to Design.

## Dispatch And Finite Role Loop

Before dispatch, Design must privately verify these persistent logical routes:

```text
TensorDSLab/default/Implementation
TensorDSLab/default/Validation
TensorDSLab/default/Review
```

Each must be Active, current for this workspace, and able to return to Design.
Coordination remains Deferred and is not used. Raw route identifiers must not
appear in committed files.

The authorized loop after explicit dispatch is:

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
reviews a fixed Validation-cleared commit. A repeated finding, exhausted loop,
route discrepancy, architecture contradiction, dirty baseline, or required
scope expansion returns to Design.

## Selected Stage Decisions

Stage 5 freezes these decisions:

1. The private raw algorithm is standard Random123 `Threefry4x32_R<20>` under
   durable private identifier `tensordslab.threefry4x32-20/v1`.
2. One central private `Enum` owns globally unique numeric stochastic roles.
   Stage 5 assigns only white-noise and PSD-coefficient roles; stream zero is
   unassigned and no Charge range is reserved.
3. Random identity is numeric and positional: root seed, operation stream,
   row-major logical flat position in current dimension order, source quantum,
   and raw-word ordinal. Semantic axes, IDs, coordinate strings, timestamps,
   physical strides, requested-product order, and execution order do not enter
   the address.
4. Stage 5 uses only `q = 0`. Its exact raw-word schedules cover fixed-point
   uniforms and Box-Muller. Source-quantum and iteration consumers remain
   deferred.
5. White and PSD scalar preparation uses Python binary64, `math.fsum` for PSD
   overlap contributions, and one rounding into the selected output dtype.
   The represented dtype-rounded normal-range RMS and represented powers define
   ideal-normal target statistics; the finite Box-Muller lattice is not
   renormalized.
6. Zero noise is seed-inert. White and PSD noise require a valid seed. No
   branch reads or mutates PyTorch global RNG or constructs `torch.Generator`.
7. The accepted execution mode is vectorized eager Torch on CPU, with an eager
   CUDA path accepted only if conditional evidence runs. Compiled, Triton,
   custom-kernel, and MPS paths are outside this stage.
8. Threefry words and fixed-point uniforms are exact across accepted CPU/CUDA
   implementations. Box-Muller and completed noise are exactly repeatable on
   the same accepted backend/mode/environment and statistically agree across
   backends; they are not promised bitwise equal across CPU and CUDA.
9. Every result is a fresh `NoiseWaveform` with exact source axes, shape, and
   device, the requested floating dtype, and `requires_grad=False`. The source
   payload is not read.
10. Detailed private helper decomposition is Implementation-owned. This work
    order freezes behavior, addresses, streams, and product results—not helper
    names, argument order, tuple-versus-stacked returns, or whether eager
    logical-position tensors are materialized.

## Exact Private Product Surface

Create exactly one product producer:

```python
def _product_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: NoiseWaveformConfig,
    seed: int | None,
    floating_dtype: torch.dtype,
) -> NoiseWaveform:
    ...
```

It lives only in:

```text
tensor_dslab/readout/noise_waveform/_product.py
```

It is private and must not appear in any `__init__.py`, `__all__`, package-root
surface, method, convenience wrapper, or public documentation example.

The producer consumes `Photoelectrons` only as the exact axes/device/shape
template. It does not inspect the integer payload, infer calibration from
coordinates, or modify truth. It locates `SampleAxis` by exact axis type rather
than a fixed tensor dimension.

## Private RNG Ownership

Create:

```text
tensor_dslab/readout/_random.py
```

This module owns the private enum, Threefry word engine, schema-v1 address
packing, fixed-point uniform conversion, and Box-Muller mechanics used by
Stage 5. It must not import TensorCore, TensorDSLab products/configs/axes,
TensorG4DS, TensorML, donor packages, NumPy, SciPy, or private Torch RNG APIs.

The exact private enum is:

```python
@unique
class _RngStream(Enum):
    NOISE_WHITE = 0x0000_0001
    NOISE_PSD_COEFFICIENT = 0x0000_0002
```

Use `Enum`, not `IntEnum`; numeric packing reads `.value`. Do not use
`Enum.auto()`, hashes, tuple/declaration positions, exported constants, aliases,
or a second stream registry. Existing values are never renumbered or reused.

The module may expose private helpers to sibling implementation modules and
tests, but their names/signatures are not architecture. Implementation should
choose the smallest coherent decomposition. Tests must anchor raw words,
address behavior, distributions, and product results rather than making an
incidental helper signature a compatibility surface.

## Raw Engine And Address Contract

For seed `s`, stream value `g`, logical position `p`, source quantum `q`, and
raw-word ordinal `r`:

```text
low32(x)  = x & 0xffff_ffff
high32(x) = (x >> 32) & 0xffff_ffff

block = r // 4
lane  = r % 4

key = (
    low32(s),
    high32(s),
    g,
    0x54445331,
)

counter = (
    low32(p),
    high32(p),
    q,
    block,
)

raw_word = Threefry4x32_20(counter, key)[lane]
```

The accepted domain is:

```text
0 <= seed < 2**64
0 <= stream < 2**32
0 <= logical_flat_position < 2**63
0 <= source_quantum_ordinal < 2**32
0 <= raw_word_ordinal < 2**34
```

Stage 5 always uses `q = 0`. Seed splitting occurs in Python before device
materialization so values above signed int64 are preserved. The numerical
low/high order is independent of host byte order.

The private raw layer nevertheless implements and validates schema packing over
the full declared `q` and `r` widths, including their maxima and overflow. That
is generic key/counter/lane evidence only. It creates no source-population,
per-quantum distribution, iteration, or Charge consumer; every Stage 5 noise
product address still has `q = 0` and uses only the normal schedule.

Torch may carry mathematical uint32 words in nonnegative `torch.int64`, but
every modular addition is masked to 32 bits, rotations use explicit masked
shifts, and no signed overflow, physical stride, host endianness, or
implementation-defined signed right shift may affect the result. A lane-three
to next-block lane-zero transition is the ordinary `r=3` to `r=4` schedule,
not wraparound.

The mandatory independent Random123 word oracles include:

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

Expected words are fixed fixtures from Random123, not values regenerated by
the production code under test.

## Fixed-Point Uniform And Box-Muller Contract

The exact `float32` conversions from one raw word `w0` are:

```text
m24 = w0 >> 8
m23 = w0 >> 9

U32[0, 1) = float32(m24) * 2**-24
U32(0, 1) = (float32(0.5) + float32(m23)) * float32(2**-23)
```

The exact `float64` conversions from consecutive numerical words `w0`, `w1`
are:

```text
m53 = w0 * 2**21 + (w1 >> 11)
m52 = w0 * 2**20 + (w1 >> 12)

U64[0, 1) = float64(m53) * 2**-53
U64(0, 1) = (float64(0.5) + float64(m52)) * float64(2**-52)
```

These are the exact Random123 `u01fixedpt.h` lattices and evaluation order. The
open-open forms use the midpoint of each 23- or 52-bit cell and range from
`2**-24` through `1 - 2**-24` for `float32`, and from `2**-53` through
`1 - 2**-53` for `float64`. Discarded low bits are never reused. A logarithm
consumes the open-open value; the Box-Muller angle consumes the closed-open
value. Box-Muller evaluates in the selected execution dtype with ambient
autocast disabled:

```text
radius = sqrt(-2 * log(U(0, 1)))
angle = tau * U[0, 1)
z0 = radius * cos(angle)
z1 = radius * sin(angle)
```

Prepare `tau` from Python binary64 `math.tau` and round it exactly once to one
scalar in the selected dtype/device. The angle is one multiplication of that
scalar by the closed-open uniform, not a reassociable `2 * pi * U` sequence.
Apply the exact `-2` factor to `log(U)` before square root. Ambient autocast is
disabled throughout.

For `float32`, radius uses raw word `r=0` and angle uses `r=1`. For `float64`,
radius uses numerical words `(r=0, r=1)` and angle uses `(r=2, r=3)`. A scalar
normal consumer uses `z0` and discards `z1`; a PSD interior coefficient uses
the ordered `(z0, z1)` as real and imaginary components. The spare value is not
cached or reassigned.

The finite uniform lattice bounds the maximum radius to approximately `5.77`
for `float32` and `8.57` for `float64`. This is an accepted bounded-MVP normal
approximation and parity qualification. Stage 5 does not silently widen
`float32` normal generation to `float64`.

## Exact Noise Addresses

For white noise, logical positions are the final output tensor's ordinary
row-major flat positions in its current dimension order:

```text
stream = NOISE_WHITE
p = 0, ..., output.numel() - 1
q = 0
```

For PSD noise, let `N = sampling.sample_count`,
`F = floor(N / 2) + 1`, and let `row` flatten every non-sample source dimension
in its existing relative order:

```text
stream = NOISE_PSD_COEFFICIENT
p = row * F + k,  k = 1, ..., floor(N / 2)
q = 0
```

`k=0` has no address and requests no draw. Positive-frequency zero-power cells
retain their fixed positions; an eager implementation may evaluate their
normal values and then force the coefficients to exact zero. The even Nyquist
position uses `z0` and discards `z1`. Interior positions use ordered
`z0 + i*z1`.

Positions derive from logical indices, not physical storage offsets. The
producer accepts arbitrary semantic axis order and noncontiguous source
storage because it reads no source payload. Dimension permutation, coordinate
reordering, arbitrary selection, and separate chunk invocation carry no random
invariance promise. Same-seed invocations of equal or overlapping shape reuse
the same positional prefix; callers use distinct seeds when invocations must be
independent.

## Contextual Preflight And Failure Contract

Intrinsic config validity remains owned by the exact frozen Stage 3 config
constructors. The private product performs only contextual checks needed for
correct execution:

- exact source/sample-axis agreement with `SamplingConfig`;
- exact output dtype `torch.float32` or `torch.float64`;
- source device type CPU or CUDA;
- an exact non-boolean Python `int` seed in `[0, 2**64)` when non-`None`;
- a required seed for white/PSD and optional valid seed for zero;
- checked output/coefficient shapes and logical-position bounds;
- recognized Stage 5 private stream and fixed normal raw-word schedule;
- represented white RMS finite and in the dtype's positive normal range; and
- PSD Nyquist coverage, prepared-cell validity, and positive retained
  representable power after DC suppression.

The supported numerical domain is closed by conservative host bounds rather
than a device-wide finite-value scan. Let:

```text
normal_guard = 8.0   for torch.float32
normal_guard = 16.0  for torch.float64
```

These exceed the accepted maximum Box-Muller radii. White preflight requires:

```text
torch.finfo(dtype).tiny <= represented_rms_mv
normal_guard * represented_rms_mv <= torch.finfo(dtype).max
```

`torch.finfo(dtype).tiny` is the smallest positive normal value. Stage 5
rejects represented subnormal RMS values because output multiplication would
have a materially quantized law that is not covered by the frozen
ideal-normal/lattice acceptance contract.

After PSD rounding, preflight requires:

```text
N * normal_guard * math.fsum(sqrt(P[k]) for k = 1, ..., K)
    <= torch.finfo(dtype).max
```

The PSD bound conservatively covers coefficient construction and complete
inverse-transform accumulation. Any nonfinite host evaluation or failed bound
is rejected before random-word generation or target-sized allocation. Tests
cover both accepted boundary-near cases and rejection just beyond each bound.

The producer does not defend against fabricated private objects,
constructor bypass, class mutation, or direct misuse outside documented
internal calls. Unsupported MPS/Meta/other devices are rejected; no fallback
moves data through CPU.

All contextual checks and host numeric preparation complete before any
target-sized output/random-position allocation or random-word generation.
Small host scalar/list preparation is ordinary preflight and is not an
allocation-free claim. A preflight failure returns no field and leaves the
source and PyTorch global RNG state unchanged. A backend failure after work is
launched has no transactional rollback promise, but no partially constructed
semantic field is returned.

## Zero Noise

`ZeroNoiseConfig` returns a fresh exact all-zero tensor with the source shape,
axes, and device and the requested floating dtype. It accepts `seed=None` or a
valid seed, never calls a random helper, never creates logical positions, and
never reads global RNG. An invalid non-`None` seed is still rejected as malformed
runtime input.

Zero is part of the same complete product family. Do not create a separate
zero producer, public identity transform, singleton cache, or shared zero
storage.

## White Noise

`WhiteNoiseConfig.rms_mv` is converted through Python binary64 and rounded once
to the requested dtype before device payload work. The represented value must
remain finite and at least the dtype's smallest positive normal value. Each
output position receives:

```text
noise[p] = represented_rms_mv * z0[p]
```

The result is independent over logical positions under the accepted finite
Box-Muller lattice and targets ensemble mean zero and the represented RMS. The
discrete open-uniform lattice and target-dtype transcendentals do not make its
digital variance exactly one, and the producer does not renormalize to force
it. The producer also does not demean a realized record, normalize its
finite-sample power, add a pedestal, or couple channels. A finite record may
have an ordinary nonzero sample mean.

## PSD Preparation

The caller supplies piecewise-constant absolute one-sided density in
`mV^2/Hz` with strictly increasing source left edges, an exclusive stop, and
coverage through the `SamplingConfig` Nyquist frequency. Values above Nyquist
are ignored, not aliased or folded.

Sampling is stored in integer picoseconds. Preflight converts explicitly in
Python binary64:

```text
fs_hz = 1e12 / sampling.sample_period_ps.value
df_hz = fs_hz / N
nyquist_hz = fs_hz / 2
```

All values must be finite and positive. Coverage with
`frequency_stop_hz == nyquist_hz` is sufficient because the exclusive endpoint
has zero measure. With `K = floor(N/2)`, target cells are:

```text
target_left_edge[0] = 0
target_left_edge[k] = (k - 1/2) * df_hz,  k = 1, ..., K
target_stop = fs_hz / 2
```

The final cell ends at `target_stop`. For each target cell, preflight computes
all source-overlap contributions in Python binary64 and combines them with
`math.fsum`:

```text
Q[k] = fsum_i(
    S[i] * max(
        0,
        min(source_right[i], target_right[k])
        - max(source_left[i], target_left[k]),
    )
)

P[0] = 0
P[k] = Q[k],  k = 1, ..., K
```

DC suppression occurs in binary64 before each retained `P[k]` is rounded once
into the output floating dtype. These represented powers define the
ideal-standard-normal target coefficient moments, variance, covariance, and
Validation's numerical oracle. The accepted finite Box-Muller lattice targets
but does not exactly equal those ideal digital moments.
Preflight rejects nonfinite preparation, insufficient Nyquist coverage, and
zero retained represented power. It performs no interpolation, DC
redistribution, global renormalization, or raw-FFT-amplitude interpretation.

## PSD Synthesis

For each private waveform row, with mutually independent finite-lattice
Box-Muller components targeting standard normals and
`I = {1, ..., floor((N - 1)/2)}`:

```text
X[0] = 0 + 0j

X[k] = (N / 2) * sqrt(P[k]) * (z0[k] + i*z1[k]),  k in I

if N is even:
    X[N / 2] = N * sqrt(P[N / 2]) * z0[N / 2] + 0j
```

`float32` output uses `complex64`; `float64` uses `complex128`. The odd terminal
coefficient remains complex; even Nyquist is real. DC and even-Nyquist
imaginary components are exact zero. Zero-power coefficients are exact zero.

The normative inverse is:

```python
torch.fft.irfft(X, n=N, dim=-1, norm="backward")
```

`N` is explicit. The result targets the ideal-standard-normal expected variance
and circular covariance from the represented powers specified in `rebuild.md`,
with the frozen finite-lattice/numerical allowance. Different rows are
independent in expectation. Realized finite-record power fluctuates. The
DC coefficient is exactly zero; the sample-domain record mean is zero only up
to inverse-FFT roundoff and is never corrected by post-transform demeaning.

There is no longer-record generation, crop, baseline bank, padding, overlap,
per-channel PSD, cross-channel covariance model, native complex-normal
shortcut, post-FFT normalization, or independent scale parameter.

## Functional, Storage, Autograd, And Device Contract

Every successful branch returns exactly `NoiseWaveform` with:

- exact source `axes` tuple and immutable axis objects;
- exact source shape and device;
- requested exact `torch.float32` or `torch.float64` dtype;
- `torch.strided` storage;
- guaranteed-fresh storage independent of the named source input;
- finite values; and
- `requires_grad=False` with no `grad_fn`.

The finite-value guarantee is scoped to the conservative represented numerical
domain above. The producer must not add a payload-wide `.item()` finite scan or
implicit CUDA synchronization as a postcondition.

The producer does not promise a particular output stride or contiguity for
arbitrary sample-axis order. It may use eager views, moves of dimensions, and
fresh intermediates. It must not call `.cpu()`, `.numpy()`, `.tolist()`, move,
cast, detach, synchronize, or read the `Photoelectrons` payload.

Generated noise has no differentiable tensor input. Later addition may still
preserve a `PureWaveform` gradient path independently. Ambient autocast is
disabled for random conversion, Box-Muller, coefficient construction, and FFT
payload arithmetic.

On CUDA, work is enqueued on the current input-device stream and returned
without implicit synchronization. Same-stream consumers follow ordinary Torch
ordering. This is not a public stream, event, lease, or lifetime API.

Same exact package/Torch/backend/eager mode, source shape and dimension order,
config, dtype, and seed must return `torch.equal` completed noise. CPU/CUDA raw
words and fixed-point uniforms must be exact. CPU/CUDA completed normals and
PSD waveforms require only the frozen statistical agreement because
transcendental and FFT implementations may differ.

## Exact Public API And Import Boundary

Stage 5 adds no public symbol. All existing `__init__.py` files and `__all__`
tuples remain byte-identical. Ordinary users still cannot call
`simulate_readout(...)`; they should not be directed to import private
producers or RNG helpers.

Production imports use only public `tensor_core` package-root names where
TensorCore is needed. `_random.py` has no TensorCore dependency. The noise
producer may import sibling public semantic types plus the private readout RNG
module, but must not import downstream product producers, `ReadoutConfig`,
`ReadoutCollection`, a package root, or a deferred integration package.

## Exact Candidate Change Allowlist

From the exact committed Design dispatch base to the fixed implementation
candidate, only these paths may change:

```text
M  README.md
A  tensor_dslab/readout/_random.py
A  tensor_dslab/readout/noise_waveform/_product.py
M  tests/test_package_contracts.py
A  tests/test_readout_random.py
A  tests/test_noise_waveform_product.py
A  tests/typing/stage_5_readout_rng_and_stochastic_noise.py
```

No other production, test, metadata, config, initializer, typing,
documentation, dependency, or governance file may change in the implementation
candidate.

`README.md` must say precisely that the private complete noise producer and its
private RNG prerequisites exist while `simulate_readout(...)`, charge
production, public RNG, IO, and integrations remain unavailable. It must not
teach ordinary collaborators to import private modules.

After the exact candidate clears Validation and Review, Review may update only:

```text
docs/implementation/stage_5_readout_rng_and_stochastic_noise.md
docs/implementation/index.md
```

Those closeout edits record evidence and status only. They must not alter the
cleared production/tests/README, synchronized architecture, parity, decisions,
governance, metadata, dependency, or prior work-order bytes.

Protected throughout the implementation candidate and closeout are:

- every existing field/config/axis/collection type and deep validator;
- every package initializer and public export tuple;
- every Stage 4 producer and test;
- `pyproject.toml`, `pyrightconfig.json`, `LICENSE`, and `py.typed`;
- architecture, design, decisions, parity, and validation authority;
- governance records and completed work orders; and
- TensorCore and every sibling repository.

## Required Runtime Evidence

### Independent Raw Engine Evidence

`tests/test_readout_random.py` must include an independent scalar Threefry
oracle or literal algorithm checkpoints that do not call the production engine
to compute expected words. It must cover:

- all fixed Random123 known-answer vectors above;
- exact registry equality to only `NOISE_WHITE -> 0x0000_0001` and
  `NOISE_PSD_COEFFICIENT -> 0x0000_0002`, with `_RngStream` an `Enum` rather
  than `IntEnum`, no alias, no zero member, and no additional member;
- seed low/high halves, both exact streams, schema domain tag, counter word
  order, lane order, and `r=3`/`r=4` rollover;
- representative and boundary logical positions around `2**32`, zero/maximum
  quantum and raw-word ordinals, and the accepted maxima without narrowing;
- invalid non-boolean seeds and every seed/stream/position/quantum/raw-word
  schema-bound overflow, without adding a source-population consumer;
- exact float32/float64 open-open and closed-open conversion fixtures at zero,
  maximum, representative words, discarded-bit boundaries, and adjacent
  midpoint cells around `0.5`;
- Box-Muller raw-word schedule, ordered cosine/sine components, spare discard,
  target dtype, finite radial cutoff, and same-backend repeatability;
- arbitrary-rank logical row-major positions independent of physical strides;
- exact proof that semantic axis/coordinate/timestamp values do not enter RNG;
- exact address-set evidence that PSD requests every `row * F + k` for
  `k=1,...,K` and never requests the DC address `row * F`;
- no read or mutation of global Torch RNG and no `torch.Generator`; and
- conditional CUDA exact raw-word and fixed-point-uniform agreement when CUDA
  is available.

Do not require a stable public/private helper signature. Tests may adapt to the
small implementation-owned helper decomposition while preserving these
behavioral oracles.

### Noise Product Evidence

`tests/test_noise_waveform_product.py` must cover both floating dtypes, all six
readout axis orders, alternate valid shapes, noncontiguous sources, and:

- exact `ZeroNoiseConfig` values, freshness, seed-inert behavior, valid optional
  seed, invalid-seed rejection, and proof that no RNG helper runs;
- white represented-RMS preparation, independent finite-lattice equation, no
  demeaning, fixed-seed repeatability, stream isolation, source-payload
  independence, exact smallest-normal acceptance, subnormal rejection,
  conservative upper-bound execution, and overflow-guard rejection before RNG;
- PSD Nyquist coverage, exact odd/even target cells, coincident boundaries,
  exclusive stop, `math.fsum` overlap preparation, pre-suppression conservation,
  exact DC discard, one dtype rounding, rounded-zero rejection, conservative
  accumulation-bound acceptance, and overflow-guard rejection before RNG;
- small odd/even full-product references proving coefficient scaling, complex
  dtype, terminal-coefficient behavior, exact zero DC and zero-power
  coefficients, explicit `n=N`, and `norm="backward"`;
- represented expected variance, covariance, Parseval mean-square power,
  independent rows, real Nyquist, finite-record power fluctuation, and the flat
  PSD `-1/(N-1)` nonzero-lag correlation;
- DC coefficient exactness while sample-domain mean uses a deterministic
  dtype/FFT-roundoff bound rather than exact equality;
- no longer-record crop, native complex-normal convention, post-demeaning,
  normalization, pedestal, or per-channel variation;
- exact type, axes identity, shape, device, dtype, `torch.strided`, finite
  values, guaranteed-fresh storage, and `requires_grad=False`;
- no source payload read/mutation, host materialization, move, cast, detach,
  explicit synchronization, global RNG access, or generator construction; and
- CPU execution plus conditional CUDA execution with accurate qualification.

### Statistical Validation Policy

Stochastic tests use analytic estimator uncertainty derived from the represented
dtype-rounded ideal-normal target plus the finite-lattice allowance below, not
an arbitrary percentage or a normality-test p-value. Freeze these four seeds:

```text
0
1
0x0123_4567_89ab_cdef
0xffff_ffff_ffff_ffff
```

Do not replace or tune them after observing a candidate.

The primary ensemble fixtures are frozen before implementation:

- white uses a 1000 ps period, `N = 32`,
  `WhiteNoiseConfig(rms_mv=PositiveFloat(1.0))`, canonical
  `(ExampleAxis, ChannelAxis, SampleAxis)` order, 64 examples, and 32 channels;
- odd PSD uses the same 1000 ps period, `N = 31`, canonical axis order, 64
  examples, and 64 channels;
- even PSD differs only by `N = 32`; and
- both PSD fixtures use one flat `PsdNoiseConfig` interval with
  `NonnegativeFloat(0.0)` as its left edge,
  `PositiveFloat(500_000_000.0)` as its stop, and
  `NonnegativeFloat(2.0e-9)` as its density in `mV^2/Hz`.

Thus white contributes exactly `2**16` logical values per seed and
`M = 2**18` across the four seeds. Each PSD fixture contributes exactly
`2**12` waveform rows per seed and `M = 2**14` across the four seeds. These
primary counts and fixtures must not be enlarged, refactored, or replaced in
response to an observed candidate result; separate deterministic and stress
fixtures may add evidence but do not replace the frozen gates.

For each dtype, white validation uses the exact primary fixture above. With
represented target RMS `sigma`, flatten each seed's result in logical row-major
order and pair `(0, 1), (2, 3), ...` without crossing a seed boundary. Thus
`M_cov = M/2`. Required bounds are:

```text
abs(mean(x))
    <= 6 * sigma / sqrt(M) + delta(sigma, 1)

abs(mean(x**2) - sigma**2)
    <= 6 * sigma**2 * sqrt(2 / M) + delta(sigma**2, 1)

abs(mean(x[2*j] * x[2*j + 1]))
    <= 6 * sigma**2 / sqrt(M_cov) + delta(sigma**2, 1)
```

For each dtype and each frozen odd/even PSD fixture, use its exact `M = 2**14`
rows. Check sample index `n = 0` at circular lags `0`, `1`, and `2`. For ideal
target covariance `C_lag` and variance `C0`, the Gaussian product standard
error is:

```text
SE(C_lag) = sqrt((C0**2 + C_lag**2) / M)
```

Each checked variance/covariance must lie within
`8 * SE(C_lag) + delta(max(C0, abs(C_lag)), N)`. These exact lags cover the
flat-PSD `-1/(N-1)` nonzero-lag case and distinguish the even Nyquist
alternation. Eight standard errors are the accepted familywise guard for this
predeclared PSD family; do not add or remove checks after seeing results.

Every statistical estimator casts the generated values to `torch.float64` on
their existing device before reduction and uses population raw moments, never
centering or Bessel correction. Waveform covariance is exactly
`mean(x[:, 0] * x[:, lag % N])`; white covariance uses the disjoint flat pairs
above. Only the final reduced scalar may be observed by the test on the host.

Coefficient ensemble evidence is reconstructed from the returned waveform; it
does not capture a private intermediate or depend on a helper signature. Cast
the waveform to `torch.float64` on its existing device and compute exactly:

```python
observed_X = torch.fft.rfft(
    waveform.tensor.to(dtype=torch.float64),
    n=N,
    dim=sample_dimension,
    norm="backward",
)
```

The test may observe only reduced scalars on the host. For exact nonzero
interior coefficient `k = 3`, with
`v = N**2 * P[k] / 4`, real and imaginary sample variances use
`SE = v * sqrt(2/M)`, their covariance uses `SE = v/sqrt(M)`, and
`abs(X[k])**2` with target `2*v` uses `SE = 2*v/sqrt(M)`. An even Nyquist
coefficient with target variance `v_n = N**2 * P[N/2]` uses
`SE = v_n * sqrt(2/M)`. These are raw second moments or raw cross-products in
`torch.float64`; apply the same eight-SE rule with `delta(v, N)`,
`delta(2*v, N)`, or `delta(v_n, N)` as appropriate. Pair rows `(0, 1),
(2, 3), ...` within each seed at sample index `n = 0` for cross-row
covariance, so `M_cross = M/2` and `SE = C0/sqrt(M_cross)`, with
`delta(C0, N)`. Parseval first computes each row's population
`mean(x**2)` in `torch.float64` and then the raw mean across rows. Its standard
error is:

```text
SE_parseval = sqrt(
    (
        sum(P[k]**2 for k in interior)
        + (2 * P[N/2]**2 if N is even else 0)
    ) / M
)
```

and its gate is `8 * SE_parseval + delta(C0, N)` around target `C0`.

Here the frozen lattice/numerical allowance is:

```text
delta(scale, length) =
    64 * torch.finfo(dtype).eps
       * max(1, ceil(log2(length)))
       * abs(scale)
```

For a zero-target covariance, use the corresponding positive variance scale,
not zero. This allowance covers the accepted fixed-point Box-Muller departure
from exact unit variance and ordinary target-dtype arithmetic; it does not
authorize post-generation normalization.

Validation independently reconstructs these estimators and observed bounds.
Exact deterministic fixtures catch sign, endpoint, normalization, and scale
errors more tightly than ensembles. Do not use KS, Shapiro-Wilk, or another
arbitrary p-value gate as the primary unit-test oracle. A failing frozen plan is
a finding; seeds, counts, formulas, and multipliers may not be tuned to pass.

### Numerical Tolerances

Use exact equality for raw words, address packing, fixed-point uniform values
when represented exactly, zero-noise payloads, dtype/device/type/axes claims,
storage nonaliasing, and source/global-RNG immutability.

For Box-Muller, coefficient, and inverse-FFT deterministic references, let
`scale` be the maximum absolute nonzero reference value and use:

```text
atol = 64 * torch.finfo(dtype).eps
          * max(1, ceil(log2(N)))
          * max(scale, torch.finfo(dtype).tiny)
rtol = 0
```

Use `N=1` for one Box-Muller pair and the actual record length for coefficient
and inverse-FFT fixtures. Expected zeros are asserted exactly, and every
fixture separately proves that its expected nonzero component remains nonzero.
This bound must not hide a swapped component, missing factor of two, DC
redistribution, or odd/even endpoint defect. Implementation may not widen it to
make a fixed output pass.

Cross-backend completed-noise checks use statistical agreement, not direct
`allclose`, while exact same-backend repeatability uses `torch.equal`.

## Package And Static-Typing Evidence

Modify `tests/test_package_contracts.py` only to:

- remove the newly real `_random.py` and noise `_product.py` paths from future
  absence assertions;
- retain absence of `charge/_product.py`, `photoelectrons/_product.py`, and
  `readout/simulation.py`;
- prove `_RngStream`, RNG helpers, and `_product_noise_waveform` are absent from
  every public export tuple and package root;
- prove the new modules import without TensorG4DS, TensorML, IV-DSLab, DSLab,
  G4DS/g4ds11, NumPy, SciPy, IO, or orchestration dependencies;
- prove `_random.py` has no TensorCore or product-domain import; and
- preserve every Stage 3/4 package, type, config, collection, import-direction,
  metadata, and retired-name check.

Create `tests/typing/stage_5_readout_rng_and_stochastic_noise.py` with positive
`typing.assert_type` evidence that the exact producer call returns
`NoiseWaveform` for all three accepted config models. It must use exact public
semantic/config types, contain no `Any`, cast, ignored diagnostic, private
TensorCore import, or public re-export of a private producer/RNG helper, and
analyze against both exact TensorCore dependency forms.

## Verification Commands

Implementation and fixed-commit Validation must run at least:

```bash
git status --short --branch
git diff --check
git diff --check <design-dispatch-commit>..<candidate-commit>
git diff --name-status <design-dispatch-commit>..<candidate-commit>
git -C /Users/mbedard/Projects/TensorCore rev-parse HEAD
git -C /Users/mbedard/Projects/TensorCore status --short
git -C /Users/mbedard/Projects/TensorCore archive --format=zip --output=/tmp/tensorcore-stage5-b454d738.zip b454d738f6385ce6489d85492a618a3dab139bb6
shasum -a 256 /tmp/tensorcore-stage5-b454d738.zip
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/tmp/tensorcore-stage5-b454d738.zip python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -c "import sys, tensor_dslab; print('tensor_g4ds' in sys.modules, 'tensor_ml' in sys.modules, 'dslab' in sys.modules, 'g4ds11' in sys.modules)"
env PATH=/Users/mbedard/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH pnpm dlx pyright@1.1.408 --version
```

Recreate the archive from the exact dependency commit for every fixed candidate
and record its SHA-256. Validation must also extract that archive outside the
repository and run the static checker with the extracted package as the only
TensorCore analysis path. Use two temporary configs outside the repository:
one whose sole TensorCore
`extraPaths` entry is the exact source checkout, and one whose sole TensorCore
entry is the extracted archive. Do not edit committed `pyrightconfig.json` to
switch evidence forms.

Every actual static-check invocation uses this verified launcher prefix:

```text
env PATH=/Users/mbedard/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH \
  pnpm dlx pyright@1.1.408 \
  --pythonpath /opt/miniconda3/bin/python
```

Append `--project` and the applicable temporary config path. If the executor's
Python or bundled Node location differs from these verified paths, return to
Design before substituting a different launcher.

Also report:

- exact Python, PyTorch, TensorCore, and static-checker versions;
- eager execution mode and exact device/backend evidence;
- whether CUDA is available and every conditional skip;
- full and focused test totals, passes, failures, errors, and skips;
- exact statistical seeds/sample sizes/formulas/observed values/bounds;
- exact public export tuples and private-symbol absence;
- source and global-RNG immutability evidence;
- no `.cpu()`, `.numpy()`, `.tolist()`, source detach/cast/move, implicit host
  payload staging, donor import, NumPy/SciPy import, global RNG use, or
  `torch.Generator` construction;
- no `__pycache__`, `.pyc`, `.pyo`, coverage output, build artifact, or other
  generated file; and
- a clean candidate and final `main` worktree.

If CUDA is unavailable, conditional CUDA checks skip and Stage 5 may close with
an explicit CPU-only evidence qualification and no CUDA behavior/performance
claim. A missing Design-accepted static checker blocks clearance. Build or
editable-install evidence may be reported if available without changing
source, but metadata is unchanged and it is not a closure requirement.

Do not run `torch.compile`, profiler, allocator, peak-memory, kernel-count, or
fusion tests as substitute closure evidence.

## Validation Report

Validation evaluates only a fixed committed candidate and returns:

- exact Design base, candidate commit, parent, branch, and allowlist
  reconciliation;
- exact TensorCore source and independent archive identity/evidence;
- every verification command and exact result;
- independent raw-word/address/uniform oracle results;
- deterministic and statistical white/PSD results with declared formulas;
- zero/no-draw, source/global-RNG immutability, freshness, autograd, axes,
  device, dtype, and failure results;
- conditional CUDA outcome and exact qualification;
- static typing, public/private surface, import, artifact, and documentation
  findings; and
- one disposition: cleared for Review or returned to Implementation with
  concrete findings.

Validation may suggest tests but does not edit the implementation branch. A
finding that requires a different equation, stream, address, execution mode,
config, dependency, public API, or path returns to Design rather than widening
Implementation.

## Independent Review

Review independently inspects the fixed Validation-cleared commit and verifies:

- exact parent/topology and candidate allowlist;
- no self-generated expected Threefry words or implementation-derived oracle;
- standard Threefry round/key/rotation/output order and masked int64 carrier
  safety;
- exact streams, address injectivity, word schedules, and positional meaning;
- no semantic-coordinate, stride, global-state, mutable-generator, or
  branch-order dependence;
- exact white and PSD represented laws, odd/even endpoints, DC policy, and
  dtype/FFT normalization;
- statistical evidence designed before interpretation and capable of detecting
  common scaling/covariance defects;
- zero seed-inertness, source independence/immutability, fresh storage,
  `requires_grad=False`, and failure ordering;
- eager-only scope, accurate CUDA qualification, and absence of performance
  inference;
- public API/export and dependency boundaries;
- protected file identity, documentation consistency, artifacts, and clean
  worktree; and
- whether the candidate is safe for clean fast-forward.

Review returns findings to Implementation unless the correction changes Design
authority or scope. Review does not rewrite production or tests.

## Known Risks And Deferred Questions

- Stage 4's environment had no CUDA runtime, so Stage 5 may close with CPU-only
  evidence and conditional CUDA skips. That would not establish a CUDA claim.
- The fixed-point Box-Muller law has finite tails and moments that differ
  slightly from an ideal Gaussian. The frozen statistical allowance covers the
  accepted MVP; sensitivity of rare detector thresholds is a trigger for a
  separately versioned normal algorithm.
- Eager positional tensors, transcendental operations, and FFTs may allocate
  intermediates and launch many kernels. Performance, fusion, and memory
  optimization await measurement after functional acceptance.
- Completed CPU and CUDA Gaussian/PSD values may differ bitwise. Only raw words
  and fixed-point uniforms have a cross-backend exactness target.
- The private helper decomposition is intentionally unfrozen. Any finding that
  behavior cannot be implemented without a public/stable helper contract
  returns to Design.
- Charge stream assignments, count samplers, PMF preparation, and supported
  population/generation bounds remain unresolved Stage 6 work.
- Pyright `1.1.408` is the accepted static checker from Stage 4. Pre-dispatch
  verification on 2026-07-14 passed against both exact TensorCore source and
  independently extracted archive forms with the launcher above. The bundled
  Node directory is not on the default shell path; omitting the prefix fails
  with `node: not found`. Missing or changed tooling blocks production
  clearance rather than weakening the gate.

## Non-Goals And Forbidden Scope

- No Bernoulli, exponential, Poisson, categorical, multinomial, rejection,
  source-quantum, iterative-generation, dark-count, jitter, crosstalk,
  afterpulse, smearing, Charge producer, or Charge stream assignment.
- No `simulate_readout(...)`, request parser/planner, retention, dependency
  closure, public transform, `simulation.py`, or partial public skeleton.
- No public producer/RNG/generator, config/type change, new public enum,
  constant, registry, product, field, sidecar, or collection behavior.
- No `torch.Generator`, Torch global RNG, private Torch RNG API,
  `torch.randn`, donor runtime, NumPy, SciPy, random spectral download, IO, or
  persisted noise bank.
- No compiled execution, `torch.compile`, Triton, CUDA/C++ extension, custom
  kernel, fusion, profiler, kernel-count, target-temporary, allocation-free,
  or performance claim.
- No exact CPU/CUDA completed Gaussian/PSD equality claim.
- No longer-record generation/crop, overlap-add, baseline bank, analog pedestal,
  per-channel PSD, tensor-valued calibration, cross-channel spectral
  correlation, alternate FFT normalization, native complex-normal shortcut,
  post-demeaning, post-normalization, or DC redistribution.
- No `out=`, workspace, destination bank, allocator, pool, lease, public stream,
  movement, selection, batching, reconstruction, or lifecycle surface.
- No input payload host materialization, source move/cast/detach/mutation, or
  explicit synchronization.
- No Photoelectrons producer/binning, TensorG4DS bridge, TensorML/Reconstruction
  adapter, cache, persistence, artifact, DAG, campaign, or integration surface.
- No TensorCore/sibling edit, private TensorCore import, fork, shim, or
  dependency move.
- No adversarial hardening/tests for unsupported subclassing, class mutation,
  constructor bypass, direct private misuse, custom Torch dispatch, or exposed
  tensor mutation.
- No broad GPU, scientific detector-validity, compatibility, conformance,
  release, deployment, or backward-compatibility claim.

## Return To Design Before

Return before any change that would:

- alter Threefry rounds, constants, key/counter packing, lane order, bounds,
  stream values, uniform conversion, Box-Muller ordering, raw-word schedule, or
  finite-tail classification;
- alter white RMS meaning, PSD cells/coverage/DC policy/coefficient scales,
  FFT length/norm, covariance, or no-pedestal semantics;
- add a distribution, stochastic role, stream, producer, public API/export,
  config/type, or dependency;
- require a stable private helper API rather than the frozen behavior;
- reject arbitrary semantic axis order or noncontiguous source storage merely
  for implementation convenience;
- read/mutate/host-materialize/move/cast/detach the source payload or touch
  global RNG;
- make compiled, performance, fusion, memory, CUDA, compatibility, or
  deployment evidence a closure condition;
- weaken freshness, axes identity, finite-value, dtype, device, seed,
  repeatability, or `requires_grad=False` contracts;
- tune statistical seeds/sample sizes/tolerances after observing a failure;
- broaden the allowlist or edit protected/historical/governance sources;
- change TensorCore, its pin, or a sibling repository;
- proceed with a dirty/divergent baseline, stale/missing route, missing static
  checker, unexplained artifact, or nonreproducible dependency; or
- exceed the finite Implementation/Validation loop.

## Merge And Closeout

Stage 5 becomes **Merged / Closed** only when:

1. Design explicitly dispatches the exact committed synchronized authority;
2. the implementation candidate has that exact parent and only allowlisted
   changes;
3. source-checkout and independently archived exact-pin suites pass on one
   fixed candidate;
4. mandatory static typing passes against both dependency forms;
5. raw engine, address, uniform, normal, zero, white, PSD, functional, storage,
   autograd, package, and absence evidence clears;
6. fixed-commit Validation returns no unresolved finding;
7. independent Review returns no unresolved finding and approves a clean
   fast-forward;
8. Review fast-forwards only the cleared candidate, runs post-merge checks, and
   records exact evidence in only the two authorized closeout documents;
9. final `main` is clean and generated-artifact free; and
10. TensorDSLab Design accepts the closeout.

No push occurs unless separately authorized. A merged implementation without
final Design acceptance is not a closed stage.
