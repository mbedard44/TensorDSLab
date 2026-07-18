# Maintenance 2 RNG And Product-Module Ownership Migration Work Order

Status: **Merged / Closed**.

Dependency state: **TensorCore `0.9.0` exact pin installed**.

Stable work-order key:
`TensorDSLab/maintenance-2-rng-and-product-module-ownership-migration`.

This is the package-authoritative TensorDSLab Design work order. TensorCore has
published the required RNG and `require_same_dtype()` surface, and this
document freezes the TensorDSLab-local commands, exact allowlist, continuity
fixtures, ownership boundary, lifecycle, and stop conditions. Its containing
Design authority was committed at
`daa046405f62ee324bc495867e796213bf6657a6`, the persistent routes were
reverified, and the user separately authorized production execution. No push
was authorized or performed.

## Final Design Closeout

TensorDSLab Design accepted Maintenance 2 on 2026-07-17. The exact linear
candidate chain was:

```text
daa046405f62ee324bc495867e796213bf6657a6
  -> f6e1fc8c3d08152cf7ba603404a4d642628adfae
  -> 5f6a8d56f0fefcd5606a8406da3a250c0f841b82
  -> f4e8eec9befaa107ceeb30c05ba1657eb7210bca
  -> 89a188abe330c06aa0b54c27cd61ac32a4fe9f63
```

Candidate 1 implemented the complete ownership migration. Candidate 2 added
the complete config-owned-key and public-distribution proof after Validation
returned a test-evidence gap. Candidate 3 proved that the retained and
overflow crosstalk keys must differ by value rather than object identity.
Independent Review found no production defect, but did not clear Candidate 3
because two explicit work-order proofs were still absent: configured
exact-zero dark-count, timing-jitter, and smearing branches had not been
observably proven draw-free, and TensorCore's zero-dimension address-span rule
had no downstream consumer probe.

The original finite loop had reached its three Implementation-to-Validation
dispatches, so Design accepted the findings and authorized one supplemental,
tests-only Review-correction loop. The final candidate is the exact direct
child of Candidate 3 and changes only
`tests/test_rng_ownership_migration.py` by 83 insertions. It proves all six
exact-zero Charge cases across `torch.float32` and `torch.float64` with
distinct failing RNGs, exact type/dtype/value, fresh storage, and source
immutability. It also proves the public `logical_positions(...)` and public
Gaussian result-shape span checks at the exact `2**63` boundary, including
rejection before any RNG word request. The unpublished intermediate
`69b01e7a169b8e308b3cbf82ccda4d4a1f7a17d8` is not in the accepted ancestry.

Validation and independent Review cleared exact final candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63`. Review cleanly fast-forwarded
unchanged `main` from `245e8155f66f51d061c680b8b220356689b24b60` to that
same commit and repeated the post-merge gates. The feature branch and
pre-closeout `main` both resolved to the final candidate. The cumulative
Design-authority-to-candidate diff is 64 rename-aware files, 5,052 insertions,
and 5,908 deletions, entirely within the frozen allowlist; protected history
and governance bytes are unchanged.

The selected dependency is TensorCore `0.9.0` at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, direct parent
`0e72f0e69cf9140b692d408e49a504cbdcb101b7`. Independent source clones and
archives reproduced SHA-256
`f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd`.
The final evidence was:

```text
supplemental proof module:  9 run,   9 passed,   0 skipped
focused source suite:     148 run, 139 passed,   9 CUDA skips
full source suite:        157 run, 148 passed,   9 CUDA skips
full archive suite:       157 run, 148 passed,   9 CUDA skips
Pyright source/archive:     0 errors, 0 warnings, 0 informations
import isolation:          False False False False
```

Review recorded the same full-suite and Pyright results before and after the
fast-forward; the merge changed no candidate byte.

The evidence environment was Python `3.13.11`, PyTorch `2.12.1`, macOS
`15.7.4` on arm64, and eager CPU. `torch.version.cuda` was `None`, CUDA
availability was `False`, and the device count was zero. Design independently
recreated the exact source and archive, repeated the focused and both full
suites, repeated both Pyright configurations, and rechecked the exact 23-name
TensorCore root export, dependency identity, imports, retired surfaces,
forbidden calls, topology, scope, diff, and artifact cleanliness before this
closeout.

The containing evidence-only commit, identified externally by `HEAD`, is the
final closeout authority. It has the merged candidate as its exact parent and
changes only this work order and the implementation index. No cleared
production, test, package metadata,
README, architecture, parity, validation, governance, dependency, or
scientific or API byte changes after Review clearance. Package adoption
remains Adopted, conformance remains Not evaluated, Coordination remains
Deferred, Profile B remains Disabled, and Stage 7 remains Undispatched. This
closeout makes no CUDA, GPU-performance, allocation, release, deployment,
compatibility, conformance, or backward-compatibility claim and authorizes no
push.

## Objective

Perform one behavior-preserving ownership migration against the selected
TensorCore `0.9.0` generic primitives:

- select and pin one exact merged TensorCore commit;
- consume only its package-root `RngKey`, `CounterRng`, `Threefry4x32`,
  `logical_positions`, and `require_same_dtype` surfaces;
- move TensorDSLab's ten stochastic role identities into the exact leaf configs
  that own them;
- delete the TensorDSLab-owned generic counter engine and distribution
  samplers without compatibility shims;
- preserve the frozen Stage 5/6 default-address and same-eager-backend output
  mappings;
- split ambiguous `types.py` modules into explicit config, field, and
  collection ownership;
- split the monolithic Charge implementation into one thin product producer
  plus focused private effect modules; and
- consolidate repeated scalar-to-floating-dtype representation in one private
  TensorDSLab requirement helper.

This is an ownership and package-structure migration, not a scientific stage.
Every accepted Charge, noise, waveform, axis, product, collection, config,
freshness, parity, and failure contract remains unchanged unless this work
order explicitly says otherwise.

## Authority, Evidence Baseline, And Dependency Gate

Package authority is `TensorDSLab/default/Design`.

The current evidence baseline is Stage 6 closed `main`:

```text
245e8155f66f51d061c680b8b220356689b24b60
```

The documentation-only Design branch is:

```text
codex/counter-rng-architecture
```

The exact clean production parent is
`245e8155f66f51d061c680b8b220356689b24b60`. The exact Design/work-order
authority is the containing commit of this document; a commit cannot embed its
own hash, so the private dispatch must name and reverify that commit. The
selected implementation branch is
`codex/maintenance-2-rng-and-product-module-ownership`.

The TensorCore dependency gate is satisfied by the following immutable
selection:

```text
repository:       https://github.com/mbedard44/TensorCore.git
reference:        origin/main
commit:           4708bf2ca063a1bcd37a30a342733b9e3dbe9f59
direct parent:    0e72f0e69cf9140b692d408e49a504cbdcb101b7
package version:  0.9.0
Python:           >=3.11
Torch:            >=2.11,<2.13
archive SHA-256:  f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd
```

TensorCore's authoritative sources are
`docs/implementation/stage_15_counter_rng_and_distributions.md` and
`docs/architecture/random.md` at that exact commit. TensorCore completed its
package-owned Implementation, Validation, independent Review, final Design,
Review-closeout, and publication workflow; clean local `main`, local
`origin/main`, and remote `refs/heads/main` were independently observed at the
same commit. TensorDSLab Design also verified the package-root surface and ran
the existing 174-test TensorDSLab suite against both the source checkout and
an exact Git archive: 164 passed, 10 conditional CUDA skips, and no failures.
The root exports exactly 23 names and includes all five required additions:
`RngKey`, `CounterRng`, `Threefry4x32`, `logical_positions`, and
`require_same_dtype`.
Focused continuity probes matched all ten legacy role addresses, both
floating Gaussian mappings, and representative Poisson and binomial results
on eager CPU. These are exact-baseline consumer observations, not a blanket
compatibility or deployment claim.

The exact Maintenance 2 dependency line is:

```toml
"tensor-core @ git+https://github.com/mbedard44/TensorCore.git@4708bf2ca063a1bcd37a30a342733b9e3dbe9f59"
```

Before dispatch, TensorDSLab Design must commit this complete Design overlay;
reverify a clean linear topology; privately reverify the persistent
logical `TensorDSLab/default/Implementation`, `TensorDSLab/default/Validation`,
and `TensorDSLab/default/Review` routes; and obtain a separate user dispatch.

The two packages do not implement these stages concurrently. TensorCore roles
clear TensorCore work; TensorDSLab roles clear TensorDSLab work. Coordination
is Deferred and is not an execution route.

The package governance state remains:

```text
package_adoption_state: Adopted
conformance_finding: Not evaluated
coordination_status: Deferred
registry_storage_profile: Disabled
stage_6: Merged / Closed
maintenance_2: Merged / Closed
stage_7: Undispatched
```

## Applicable Contracts

Implementation, Validation, and Review must read and reconcile:

- `AGENTS.md`;
- `CONTRIBUTING.md`, especially TensorCore Backbone, Package Shape And Imports,
  RNG/config ownership, boundary-first validation, public imports, functional
  result contracts, scope discipline, tests, and documentation;
- [Rebuild Architecture](../architecture/rebuild.md), especially Semantic Axes,
  Product Fields, Scientific Configuration, Private Product Builders, RNG And
  Positional Repeatability, Functional/Memory/Lifetime Contract, Validation
  Strategy, Rebuild Migration, and Closed Decisions And Remaining Design
  Gates;
- [TensorCore Integration](../architecture/tensors.md);
- [Readout Architecture](../architecture/readout.md);
- [Parity](../parity.md);
- [Validation](../validation.md);
- the closed Stage 3, Stage 4, Stage 5, and Stage 6 work orders as historical
  evidence for their exact implemented bytes; and
- the exact TensorCore Stage 15 work order, architecture, and closeout at
  `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`.

Closed work orders are not rewritten. Live architecture and package docs are
synchronized to the new ownership and dependency after implementation.

## Target Package Shape

The final readout tree is:

```text
tensor_dslab/
  common/
    __init__.py
    axes.py
    sampling.py

  readout/
    __init__.py
    _requirements.py
    config.py
    collection.py

    photoelectrons/
      __init__.py
      field.py

    charge/
      __init__.py
      config.py
      field.py
      _produce.py
      effects/
        __init__.py
        _counts.py
        _delays.py
        _dark_counts.py
        _timing_jitter.py
        _correlated_avalanches.py
        _smearing.py

    pure_waveform/
      __init__.py
      config.py
      field.py
      _produce.py

    noise_waveform/
      __init__.py
      config.py
      field.py
      _produce.py

    analog_waveform/
      __init__.py
      config.py
      field.py
      _produce.py

    digitized_waveform/
      __init__.py
      config.py
      field.py
      _produce.py
```

`Photoelectrons` remains an already-produced truth input. It has no config and
no producer. The other five product packages own one public field class, their
public configs, and their private producer where applicable.

Delete without aliases:

```text
tensor_dslab/readout/types.py
tensor_dslab/readout/photoelectrons/types.py
tensor_dslab/readout/charge/types.py
tensor_dslab/readout/pure_waveform/types.py
tensor_dslab/readout/noise_waveform/types.py
tensor_dslab/readout/analog_waveform/types.py
tensor_dslab/readout/digitized_waveform/types.py
tensor_dslab/readout/_random.py
```

Do not create `readout/_rng.py`, a compatibility package, or re-export modules
at any retired path. Package-root and accepted package `__init__` imports keep
the collaborator-facing class names stable; direct imports from the retired
module paths intentionally fail. This pre-deployment cleanup makes no backward-
compatibility promise.

`effects/__init__.py` is an empty-export package marker. Callers and sibling
modules import the needed private function from its exact private module; the
package does not aggregate aliases or create a second internal API.

## Ownership After The Split

### Readout Composition

`readout/config.py` owns only `ReadoutConfig`.

`readout/collection.py` owns only `ReadoutCollection` and its intrinsic
completed-result coherence. It continues to accept any nonempty unordered
subset of the exact six product field types, requires common ordered axes and
device, and requires a common dtype only among the present floating products.
It deliberately does not compare `Photoelectrons` (`torch.int64`) or
`DigitizedWaveform` (`torch.int32`) against floating products through
`require_same_dtype()`.

### Product Packages

Each `field.py` owns exactly its final direct `TensorField` leaf and its field-
local requirements. Each `config.py` owns the product's public config records
and constrained scalar composition. Product `__init__.py` files deliberately
export the accepted collaborator-facing classes; package-root exports remain
deliberate.

No product package imports `ReadoutConfig`, `ReadoutCollection`, or future
`simulate_readout(...)`. Import direction remains acyclic.

### Charge Producer And Effects

`charge/_produce.py` becomes the thin product orchestrator. It owns the
physical sequence and final `Charge` construction:

```text
truth
  -> optional dark counts
  -> optional timing jitter
  -> optional correlated avalanches
  -> optional charge smearing
  -> Charge
```

It does not own generic RNG algorithms or the detailed implementation of each
effect.

The private effect package owns:

- `_counts.py`: the TensorDSLab Charge count domain, checked nonnegative
  additions/subtractions, allocation/rate/address guards, local ordered
  multinomial orchestration through repeated public `rng.binomial(...)` calls,
  and final no-draw remainder;
- `_delays.py`: fixed/exponential delay and afterpulse-recovery plans,
  preparation, stable analytic kernels, and their numeric checks;
- `_dark_counts.py`: dark-count plan/preflight and simulation;
- `_timing_jitter.py`: integrated jitter law preparation and aggregate
  redistribution;
- `_correlated_avalanches.py`: the fixed-generation DiCT/DeCT/AP frontier,
  mechanism diagnostics, S1/S2 result, and right-overflow accounting; and
- `_smearing.py`: terminal charge-smearing preflight and simulation.

Effect modules may import private shared count/delay mechanics and public
config types. They must not import `_produce.py`, construct `Charge`, depend on
readout composition, or create a circular import. `_produce.py` alone imports
and sequences the effect functions.

`_counts.py` does not implement Threefry, fixed-point uniforms, Box-Muller,
Poisson inversion/PTRS, or binomial inversion/BTRS. Those are TensorCore-owned.
It also does not expose a public multinomial primitive: TensorDSLab retains the
physical category order, stable current/later masses, sequential conditional
calls, conservation, and final remainder.

## TensorCore Imports And Dependency Pin

All generic names come only from the selected package root:

```python
from tensor_core import (
    CounterRng,
    RngKey,
    Threefry4x32,
    logical_positions,
    require_same_dtype,
)
```

Production may import only the names it actually uses. It must not import
`tensor_core.random`, `tensor_core.validation`, a protected TensorCore helper,
or a private module to bypass the package boundary. TensorDSLab does not wrap
or re-export these generic names from its own package root.

`pyproject.toml` must replace only the TensorCore direct reference with the
exact `4708bf2ca063a1bcd37a30a342733b9e3dbe9f59` line frozen above. Validation
independently recreates a ZIP with
`git archive --format=zip 4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`,
requires SHA-256
`f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd`,
and proves source/archive identity. A version string alone is insufficient.

`Threefry4x32` is expected to be the ordinary collaborator-facing RNG choice,
but Maintenance 2 does not add a public simulation function or construct a
hidden default RNG. Callers of private test/producer surfaces supply an
accepted `CounterRng` instance explicitly.

## Config-Owned Keys

Every independently stochastic leaf config owns its immutable `RngKey`.
Default namespace is exact ASCII `TDS1`:

```text
0x54445331
```

The append-only default mapping is:

| Config field | Stream |
|---|---:|
| `WhiteNoiseConfig.rng_key` | `0x0000_0001` |
| `PsdNoiseConfig.rng_key` | `0x0000_0002` |
| `DarkCountConfig.rng_key` | `0x0000_0003` |
| `DirectCrosstalkConfig.retained_rng_key` | `0x0000_0004` |
| `DirectCrosstalkConfig.overflow_rng_key` | `0x0000_0005` |
| `DelayedCrosstalkConfig.retained_rng_key` | `0x0000_0006` |
| `DelayedCrosstalkConfig.overflow_rng_key` | `0x0000_0007` |
| `TimingJitterConfig.rng_key` | `0x0000_0008` |
| `AfterpulseConfig.rng_key` | `0x0000_0009` |
| `ChargeSmearingConfig.rng_key` | `0x0000_000A` |

Callers may override any leaf field with another exact `RngKey`. Config
construction preserves ordinary immutable equality and repr. Each crosstalk
config requires its retained and overflow keys to differ.

Deterministic, delay, recovery, saturation, pulse-shape, ADC, product-wrapper,
and composite configs own no key. `CorrelatedAvalancheConfig` does not collect
or duplicate the keys of its leaves. Closure-wide detection of one key reused
by distinct stochastic roles in the requested transitive closure belongs to
Stage 7 request preflight and is not added here. That later collision set
includes structurally present key-bearing configs even when their numeric
parameters make them no-ops.

Delete `_RngStream` completely. Never derive keys from declaration order,
request order, execution order, `Enum.auto()`, Python `hash()`, or branch-
dependent sequential consumption.

## Function Signatures And RNG Use

Stochastic-capable product producers require the immutable invocation RNG:

```python
def _produce_noise_waveform(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: NoiseWaveformConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> NoiseWaveform:
    ...


def _produce_charge(
    photoelectrons: Photoelectrons,
    *,
    sampling: SamplingConfig,
    config: ChargeConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> Charge:
    ...
```

The keyword-only order above is the accepted post-maintenance signature:
sampling, config, RNG, then floating dtype. It changes only the retired
`seed: int | None` slot to required `rng: CounterRng`; Implementation does not
redesign unrelated parameters. Relevant stochastic effect functions likewise
accept the same `CounterRng` and select the exact key from their leaf config.

Deterministic product producers and deterministic preparation helpers receive
no RNG merely for signature uniformity:

```text
_produce_pure_waveform
_produce_analog_waveform
_produce_digitized_waveform
delay/recovery/pulse/ADC/scalar preparation
```

There is no `seed=` compatibility parameter. `CounterRng.seed` replaces the
bare invocation seed; `RngKey` replaces the central stream enum. The RNG is
immutable and stateless, so functions never advance or return it.

An exact-zero or disabled branch makes no distribution request:

- `ZeroNoiseConfig` returns fresh exact zeros without calling `rng`;
- `WhiteNoiseConfig` cannot represent zero RMS and an all-zero PSD is invalid;
  either invalid config fails preflight and creates no producer RNG branch;
- absent or exact-zero dark count, timing jitter, crosstalk, afterpulse, or
  smearing skips that complete effect; and
- deterministic product producers never inspect the RNG.

The stochastic-capable producer still requires a valid `CounterRng` even when
its selected branch is draw-free. Stage 7 later requires one RNG for every
public request, including a deterministic closure.

## Distribution Calls

TensorDSLab consumes only the public methods:

```text
white noise
  -> rng.gaussian(mean=0.0, standard_deviation=rms, ordinal=0, count=1)

PSD coefficient
  -> rng.gaussian(mean=0.0, standard_deviation=1.0, ordinal=0, count=2)

charge smearing
  -> rng.gaussian(mean=S1, standard_deviation=sigma*sqrt(S2),
                  ordinal=0, count=1)

dark counts and retained/overflow DiCT/DeCT
  -> rng.poisson(...)

timing-jitter and afterpulse categories
  -> rng.binomial(success_mass=current_mass,
                  failure_mass=later_mass, ...)
```

TensorCore's accepted address-capacity check uses
`prod(max(dimension, 1)) < 2**63` for every positions and result shape,
including shapes with a zero-sized dimension. TensorDSLab tests this public
contract and does not substitute an ordinary zero-valued shape product.

Every call supplies the config-owned key and the TensorDSLab-owned position
lattice. Aggregate count roles use `quantum=0`. TensorDSLab uses public
`logical_positions(...)` for materialized row-major grids and checked virtual
position arithmetic for iteration/category lattices without materializing
prohibited full address banks.

Do not compact positions by active values, parse coordinate labels, use storage
offsets, or change generation/category ordering. Do not derive a small
binomial complement when the physical law has already prepared independent
current and later masses.

The migration may reorganize calls but must preserve the exact Stage 5/6
address, variate-ordinal, operation-order, and floating operation-order
contracts. In particular, replacing a standard-normal helper with
`gaussian(...)` must retain the accepted multiply/add branch and PSD component
order. Within an invoked Gaussian field, a cell whose represented standard
deviation is zero still requests its complete ordinary Gaussian words before
returning the represented mean; only a skipped scientific branch is draw-free.

For binomial inversion, continuity follows the production association
`(probability * count_ratio) * mass_ratio`. The stale reassociated Stage 6
test-oracle expression `probability * (count_ratio * mass_ratio)` must be
deleted with the generic sampler tests and must not generate Maintenance 2
golden values.

## Shared Scalar Representation Helper

Add exactly one private helper to `readout/_requirements.py`:

```python
def _require_representable_float(
    value: float | int,
    *,
    dtype: torch.dtype,
    field: str,
) -> float:
    ...
```

It:

- accepts only a Python `float` or non-boolean Python `int` already produced by
  accepted config/derived preparation;
- accepts exactly `torch.float32` or `torch.float64`;
- represents the scalar once through an ordinary CPU scalar tensor of that
  dtype;
- returns the represented exact Python `float`;
- raises `TypeError` for malformed value or dtype and `ValueError` if the
  represented result is nonfinite; and
- performs no positivity, range, ordering, unit, detector, or output-envelope
  policy.

Use it to replace the duplicated private scalar-rounding helpers in pure,
noise, analog, and digitized waveform producers and the equivalent Charge
recovery/smearing scalar conversions. Callers retain every stronger local
check. Do not replace legitimate zero/one/`nextafter` tensors or device-local
payload constants whose construction has a different algorithmic purpose.

This helper remains private TensorDSLab code. It is not re-exported and does
not become a TensorCore API in this maintenance stage.

## `require_same_dtype(...)` Use

Use TensorCore's public `require_same_dtype(...)` only for semantic field
relationships:

- `AnalogWaveform` production compares `PureWaveform` and `NoiseWaveform`; and
- `ReadoutCollection` compares only the present subset of `Charge`,
  `PureWaveform`, `NoiseWaveform`, and `AnalogWaveform`.

Raw private scratch, law, count, and ledger tensors retain operation-specific
shape/dtype/device checks inside TensorDSLab. Do not force raw tensors into
temporary semantic fields merely to call a generic requirement.

Equal axes already imply equal valid field shape, but removing an existing
redundant private shape check is authorized only when tests prove no accepted
behavior or error boundary changes. This maintenance does not add
`require_same_shape()` or another TensorCore helper.

## Behavior And Continuity Contract

This migration preserves:

- exact Stage 5/6 schema-v1 address packing;
- default namespace and streams `1` through `10`;
- exact eager-CPU raw words, fixed-point uniform values, Gaussian component
  order, Poisson counts, and binomial counts for frozen fixtures;
- exact same-backend/mode completed `NoiseWaveform` and `Charge` outputs for
  the frozen migration fixtures;
- every accepted scientific distribution, equation, category order, delay
  kernel, recovery law, count ceiling, overflow rule, S1/S2 mapping, and
  ordinary smearing behavior, with only the contextual extreme representation
  domain intersected as recorded below;
- source immutability, generated-product freshness and pairwise independence,
  axes/device/dtype behavior, no silent movement/casting/detachment/host
  materialization, and no global RNG effect;
- the public package-root class/config exports; and
- the Stage 5/6 eager functionality-first execution mode and qualifications.

Golden expected values must be literal frozen data or independently computed
oracles. Candidate tests must not calculate their expected values by retaining
or copying the deleted TensorDSLab RNG implementation. Integer views of
floating tensors may be used for exact eager-CPU bit-pattern continuity.

Cross-backend completed transcendental results remain statistical where the
accepted contract is statistical. This work order does not manufacture a CUDA
claim when CUDA is unavailable.

## Design-Return Charge-Smearing Envelope Resolution

Design returned this correction before candidate 1 was committed or dispatched
to Validation, then reauthorized the existing Implementation candidate. At
that resolution point, the candidate ordinal remained 1 and no
Implementation-to-Validation or Validation-to-Implementation budget had been
consumed.

Candidate execution correctly stopped when TensorCore `0.9.0` rejected a
float32 maximum-ledger Gaussian law before word generation. TensorCore was
enforcing its published conservative affine-law envelope correctly. The
apparent `K=0` conflict came from representing the binary64 expression for the
Stage 6 ledger bound to nearest in float32, which rounded above the real bound
and therefore constructed a value the Stage 6 proof did not cover.

Maintenance 2 resolves that mismatch without changing TensorCore. Let

```text
B_real = C_max*(1 + gamma_L) + L*eta_d
```

be the existing outward Stage 6 bound returned by `_ledger_envelope(...)`.
Same-device compatibility preflight and any defensive represented-ledger
comparison derive the greatest finite value `B_d` representable in the
requested floating dtype such that `B_d <= B_real`. Every realized ledger is
itself target-dtype representable and no target-dtype value lies between `B_d`
and `B_real`, so `B_d` still bounds every represented ledger. A nearest
target-dtype representation of the Python `B_real` value is not used as an
accepted ledger bound when it lies above `B_real`. The existing upward-rounded
Stage 6 mathematical ledger bound and smearing-envelope check over `B_real`
remain unchanged.

For positive smearing, `_prepare_smearing_sigma(...)` keeps that Stage 6
worst-ledger check and additionally evaluates the same-device target-dtype
operations `sqrt(B_d)` then `represented_sigma * sqrt(B_d)`. It requires that
actual prepared scale, combined with TensorCore's exact documented radius for
the dtype, satisfy TensorCore's represented Gaussian finite-output
envelope. This model-specific compatibility preflight completes before any
Charge effect, RNG request, or write. Runtime retains the exact direct call:

```python
rng.gaussian(
    mean=S1,
    standard_deviation=represented_sigma * torch.sqrt(S2),
    key=config.rng_key,
    positions=positions,
    dtype=S1.dtype,
    quantum=0,
    ordinal=0,
    count=1,
)
```

There is no local standard-normal affine path, wrapper, fallback, conditional
sampler branch, TensorCore edit, or clipping change.

The correction preserves the frozen `K=0`, `L=1` adjacent boundaries. The
maximum valid represented float32 ledger is `0x1.0000000000000p+53`; float32
relative sigma accepts `0x1.f61fea0000000p+98` and rejects its immediate
neighbor `0x1.f61fec0000000p+98`. Float64 remains accepted
`0x1.51e4a059b7cf4p+994` and rejected
`0x1.51e4a059b7cf5p+994`. Only a contextual extreme can narrow when the public
Gaussian prepared-scale envelope is stricter: the verified `L=24` float32
pair accepts `0x1.f61fd20000000p+98` and rejects its immediate neighbor
`0x1.f61fd40000000p+98`.

The ordinary frozen Charge fixture at relative sigma `0.1`, every other
completed-value fixture, all scientific distributions and equations, default
keys and addresses, variate ordinals, category and accumulation order, and
physical operation order remain unchanged. The contextual extreme values have
no calibrated detector interpretation and this correction creates no broader
science, compatibility, or execution claim.

## Frozen Eager-CPU Continuity Fixtures

The literal fixtures below are bound to:

```text
TensorDSLab base: 245e8155f66f51d061c680b8b220356689b24b60
TensorCore:       4708bf2ca063a1bcd37a30a342733b9e3dbe9f59
Python:           3.13.11
PyTorch:          2.12.1
OS:               macOS 15.7.4 arm64
execution:        eager CPU
seed:             0x0123456789ABCDEF
namespace:        0x54445331
```

The common role-fixture positions are the `torch.int64` values
`[0, 1, 2, 4294967299]`; `quantum=0`. Hex values are exact flattened
row-major IEEE payload bit patterns in the named dtype.

| Stream | Public request | Exact result |
|---:|---|---|
| `1` | Uniform, include zero, count `1`, `float32` | `3ecdc482 3f7ace57 3f28f330 3ed8ba00` |
| `1` | same, `float64` | `3fd9b8905cff7a8c 3fef59caf40108e5 3fe51e6600fe5534 3fdb17402dca4490` |
| `1` | Uniform, exclude zero, count `1`, `float32` | `3ecdc482 3f7ace57 3f28f331 3ed8ba02` |
| `1` | same, `float64` | `3fd9b8905cff7a8e 3fef59caf40108e5 3fe51e6600fe5535 3fdb17402dca4492` |
| `1` | Gaussian, mean `0`, standard deviation `0.75`, count `1`, `float32` | `3f81741c 3e184d06 3f25bf13 3f06517b` |
| `1` | same, `float64` | `3fcf75d99582d78a 3f89fe5e8e452724 3fe178009eac0f2d bfdc45b21a9f07da` |
| `2` | Gaussian, mean `0`, standard deviation `1`, count `2`, `float32` | `bfe0b1d4 bf186554 bf01b88f 4018c5a8 bef8a544 3f1953a0 3f9c0e27 3fb37fb6` |
| `2` | same, `float64` | `bff73a00f4cae65c bff2713778ea1903 c002d5befcdee864 3fe488e59e06900e bfe2b75e6f86914b 3fe01435868a425e 3ffabc480d2e0237 bfea0354fa54a537` |
| `3` | Poisson means `[0, 0.75, 9.5, 25]` | `[0, 2, 12, 30]` |
| `4` | same | `[0, 0, 8, 25]` |
| `5` | same | `[0, 1, 7, 29]` |
| `6` | same | `[0, 2, 11, 17]` |
| `7` | same | `[0, 4, 4, 26]` |
| `8` | Binomial counts `[0, 3, 20, 100]`, success masses `[0, 0.25, 0.9, 0.2]`, failure masses `[0, 0.75, 0.1, 0.8]` | `[0, 1, 19, 17]` |
| `9` | same | `[0, 1, 16, 23]` |
| `10` | Gaussian means `[0.25, -1, 3.5, 0]`, standard deviations `[0.5, 0.25, 1.5, 2]`, count `1`, `float32` | `3e936e95 bfab1848 400b2629 406d6d29` |
| `10` | same, `float64` | `3fe0d430c98c26d9 bff0364ad8f2cf8c 400aefb6b143ae4d 4001eb62399da026` |

Completed-product fixtures share `SamplingConfig(sample_period_ps=2000,
sample_count=4)`, one example coordinate `"event-0"`, one channel coordinate
`"channel-0"`, canonical sample coordinates `("0ps", "2000ps", "4000ps",
"6000ps")`, and `Photoelectrons=[[[3, 0, 1, 2]]]`.

White noise with `rms_mv=0.75`:

```text
float32  3f81741c 3e184d06 3f25bf13 3e499154
float64  3fcf75d99582d78a 3f89fe5e8e452724 3fe178009eac0f2d 3f851d1a3e211f04
```

PSD noise with frequency left edges `[0, 100000000]` Hz, stop `250000000`
Hz, and densities `[1e-8, 2e-8]` mV-squared/Hz:

```text
float32  bfa40c51 c03bf3d0 3e486878 4080b9b9
float64  c01058090a5b3d75 bfd20487b1837bec 4006399d2ea79a9c 3ff96e0bb87e9f95
```

The complete Charge fixture enables every stochastic Charge role with dark
rate `5e8` Hz, jitter sigma `1` ns, maximum generations `2`, direct crosstalk
mean `0.6` with exponential mean delay `2.5` ns, delayed crosstalk mean `0.4`
with exponential mean delay `4` ns, afterpulse probability `0.35` with mean
delay `3` ns and recovery time constant `5` ns, and relative smearing sigma
`0.1`:

```text
float32  40a08b0b 40b571a7 40f01acc 4161c37e
float64  40147e5936eabbbe 4017f2b37dd7adc7 401edf582a1ee0b5 402c390c96702ab5
```

Implementation must reproduce every literal through public TensorCore calls
and the migrated product paths. These bit patterns are exact only on the
recorded same-environment eager-CPU boundary. CUDA and other supported
backends retain their accepted statistical/transcendental comparison policy.

## Test Migration

TensorCore owns generic algorithm tests after the dependency gate:

- Random123 raw-word known-answer and schema packing;
- fixed-point uniform conversion;
- Box-Muller and Gaussian affine mapping;
- generic Poisson inversion/PTRS;
- generic binomial inversion/BTRS and exhaustion; and
- generic shape/dtype/device/freshness contracts.

TensorDSLab deletes `tests/test_readout_random.py`. It renames and narrows
`tests/test_readout_count_sampling.py` to
`tests/test_charge_count_orchestration.py`, retaining only TensorDSLab-owned
category planning, conditional multinomial ordering, conservation, checked
count behavior, and scientific position schedules. New
`tests/test_rng_ownership_migration.py` owns the ten config-key fixtures,
public TensorCore call continuity, recording/failing RNG doubles, and retired-
surface absence gates. No test copies protected TensorCore mechanics.

Required TensorDSLab evidence includes:

- exact config key defaults, overrides, equality/repr, append-only numeric
  mapping, and crosstalk retained/overflow inequality;
- literal default-role address/output fixtures for all ten roles through the
  public TensorCore methods without protected imports;
- frozen eager-CPU uniform/Gaussian/Poisson/binomial consumer results;
- small complete `NoiseWaveform` and `Charge` bit-pattern fixtures on the
  recorded eager-CPU environment;
- a local concrete `CounterRng` test double implementing TensorCore's exact
  protected `_generate_block(*, key, positions, quantum, block)` hook, whose
  returned tensor is same-device strided `torch.int64`, has shape
  `positions.shape + (4,)`, and does not require gradients; recording and
  failing forms prove every disabled or exact-zero branch requests no words
  without overriding TensorCore's final public distribution methods;
- proof that stochastic-capable producers require `CounterRng`, deterministic
  producers omit it, and every bare TensorDSLab producer/effect `seed=`
  parameter and compatibility shim is absent;
- public TensorCore `uniform`, `gaussian`, `poisson`, `binomial`, and
  `logical_positions` call-contract probes;
- aggregate multinomial/category order, final remainder, conservation,
  covariance, and positional identity tests;
- all existing Charge delay, timing-jitter, cascade, ledger, overflow,
  smearing, scientific/statistical, failure, source/global-RNG immutability,
  axes, dtype, device, and freshness evidence;
- exact proof that `B_d` is the greatest target-dtype ledger not exceeding
  `B_real`; preserved `K=0` float32/float64 adjacent endpoints; the contextual
  `L=24` float32 adjacent pair; and maximum positive/negative radius outcomes
  through public `gaussian(...)`, with the negative outcome clipped to zero;
- proof that rejection at the contextual neighbor occurs before the local
  concrete `CounterRng` test double's protected `_generate_block(...)` hook and
  before any earlier enabled Charge effect can consume words;
- all existing pure/noise/analog/digitized product evidence;
- focused `_require_representable_float()` type, dtype, finite, rounding,
  endpoint, and caller-owned stronger-policy regressions;
- `require_same_dtype()` use for Analog and only the collection's floating
  subset;
- exact module inventory, old-module absence in fresh processes, empty-export
  `effects` package, import-cycle scan, and no compatibility shim;
- package-root export and import-isolation checks;
- deletion of
  `tests/typing/stage_5_readout_rng_and_stochastic_noise.py` and
  `tests/typing/stage_6_charge_simulation.py`, replaced by exact
  `tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py`;
  Stage 3/4 typing fixtures remain protected unless an authorized import-path
  correction is proven necessary;
- the complete suite against both the selected TensorCore source checkout and
  an independently extracted exact archive;
- zero-diagnostic Pyright `1.1.411` evidence in the protected repository's
  standard mode against both dependency forms;
- conditional eager CUDA tests where available; and
- diff, forbidden-import/call, generated-artifact, bytecode, and final
  cleanliness gates.

## Frozen Verification Environment And Commands

The accepted local evidence environment is `/opt/miniconda3/bin/python`,
Python `3.13.11`, PyTorch `2.12.1`, macOS `15.7.4` arm64, eager CPU, and
Pyright `1.1.411`. CUDA is currently unavailable. Conditional CUDA tests run
when CUDA is available; otherwise every skip is reported and no GPU claim is
made. The committed `pyrightconfig.json` remains in `standard` mode. This
maintenance does not silently strengthen repository-wide typing policy to
strict mode.

Every role uses these immutable values, substituting only its own private
temporary directories and the exact authority/candidate commits:

```bash
TC_OBJECT_SOURCE=/Users/mbedard/Projects/TensorCore
TC_REMOTE=https://github.com/mbedard44/TensorCore.git
TC_COMMIT=4708bf2ca063a1bcd37a30a342733b9e3dbe9f59
TC_PARENT=0e72f0e69cf9140b692d408e49a504cbdcb101b7
TC_ARCHIVE_SHA=f793ef3645ab44175e445feb94444a90e01ccc34d01fc467db36bd81ad0606bd
PYRIGHT_NODE=/Users/mbedard/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin
PYRIGHT_PYTHON=/opt/miniconda3/bin/python
```

`M2_BASE` is the exact containing Design-authority commit named by the private
dispatch. `M2_CANDIDATE` is the fixed candidate being evaluated. A role must
create its own clean detached source clone rather than using the mutable shared
TensorCore checkout as an evidence worktree:

```bash
git ls-remote "$TC_REMOTE" refs/heads/main
git -C "$TC_OBJECT_SOURCE" cat-file -e "$TC_COMMIT^{commit}"

EVIDENCE_ROLE=implementation
TC_SOURCE="$(mktemp -d "/tmp/tensorcore-m2-${EVIDENCE_ROLE}-source.XXXXXX")"
git clone --no-checkout "$TC_OBJECT_SOURCE" "$TC_SOURCE"
git -C "$TC_SOURCE" checkout --detach "$TC_COMMIT"
test "$(git -C "$TC_SOURCE" rev-parse HEAD)" = "$TC_COMMIT"
test "$(git -C "$TC_SOURCE" rev-parse HEAD^)" = "$TC_PARENT"
test -z "$(git -C "$TC_SOURCE" status --porcelain=v1)"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_SOURCE python -c "from pathlib import Path; import tensor_core; root=Path('$TC_SOURCE').resolve(); loaded=Path(tensor_core.__file__).resolve(); assert loaded.is_relative_to(root); required=('RngKey','CounterRng','Threefry4x32','logical_positions','require_same_dtype'); assert all(hasattr(tensor_core, name) for name in required); print(loaded); print(tensor_core.__all__)"

python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
env PATH=$PYRIGHT_NODE:$PATH pnpm dlx pyright@1.1.411 --version
```

The first command must report remote `main` at exact `TC_COMMIT`. The final
candidate dependency pin is checked literally:

```bash
python -c "import tomllib; expected='tensor-core @ git+https://github.com/mbedard44/TensorCore.git@4708bf2ca063a1bcd37a30a342733b9e3dbe9f59'; data=tomllib.load(open('pyproject.toml','rb')); found=[item for item in data['project']['dependencies'] if item.startswith('tensor-core @ ')]; assert found == [expected], found"
```

Implementation runs the focused source-checkout suite during development and
both commands before fixing every candidate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_SOURCE python -m unittest \
  tests.test_rng_ownership_migration \
  tests.test_charge_count_orchestration \
  tests.test_charge_delay_preparation \
  tests.test_charge_timing_jitter \
  tests.test_charge_correlated_avalanches \
  tests.test_charge_product \
  tests.test_noise_waveform_product \
  tests.test_deterministic_waveform_products \
  tests.test_readout_configs \
  tests.test_readout_product_types \
  tests.test_readout_collection \
  tests.test_package_contracts \
  -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_SOURCE python -m unittest discover -s tests -v
```

Validation sets `EVIDENCE_ROLE=validation`; Review sets
`EVIDENCE_ROLE=review` and independently repeats this archive block without
reusing Validation's paths or files:

```bash
TC_ARCHIVE_ROOT="$(mktemp -d "/tmp/tensorcore-m2-${EVIDENCE_ROLE}-archive.XXXXXX")"
TC_ARCHIVE_ZIP="${TC_ARCHIVE_ROOT}.zip"
git -C "$TC_SOURCE" archive --format=zip --output="$TC_ARCHIVE_ZIP" "$TC_COMMIT"
test "$(shasum -a 256 "$TC_ARCHIVE_ZIP" | awk '{print $1}')" = "$TC_ARCHIVE_SHA"
unzip -q "$TC_ARCHIVE_ZIP" -d "$TC_ARCHIVE_ROOT"
test "$(python -c "import tomllib; print(tomllib.load(open('$TC_ARCHIVE_ROOT/pyproject.toml','rb'))['project']['version'])")" = "0.9.0"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_ARCHIVE_ROOT python -c "from pathlib import Path; import tensor_core; root=Path('$TC_ARCHIVE_ROOT').resolve(); loaded=Path(tensor_core.__file__).resolve(); assert loaded.is_relative_to(root); required=('RngKey','CounterRng','Threefry4x32','logical_positions','require_same_dtype'); assert all(hasattr(tensor_core, name) for name in required); print(loaded); print(tensor_core.__all__)"
```

Validation, Review, and post-merge closeout run the complete suite against
both dependency forms and record exact totals, skips, environment, and device:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_SOURCE python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_ARCHIVE_ROOT python -m unittest discover -s tests -v
```

For static analysis, each evidence role derives temporary standard-mode
configs from the protected committed `pyrightconfig.json`:

```bash
TDSLAB_ROOT="$(git rev-parse --show-toplevel)"
cd "$TDSLAB_ROOT"
test "$(git rev-parse HEAD)" = "$M2_CANDIDATE"
test -z "$(git status --porcelain=v1)"

SOURCE_PYRIGHT_CONFIG="/tmp/tensordslab-m2-${EVIDENCE_ROLE}-source-pyright.json"
ARCHIVE_PYRIGHT_CONFIG="/tmp/tensordslab-m2-${EVIDENCE_ROLE}-archive-pyright.json"

jq --arg root "$TDSLAB_ROOT" --arg source "$TC_SOURCE" \
  'del(.include) | .extraPaths = [$root, $source]' \
  pyrightconfig.json > "$SOURCE_PYRIGHT_CONFIG"
jq --arg root "$TDSLAB_ROOT" --arg archive "$TC_ARCHIVE_ROOT" \
  'del(.include) | .extraPaths = [$root, $archive]' \
  pyrightconfig.json > "$ARCHIVE_PYRIGHT_CONFIG"

env PATH=$PYRIGHT_NODE:$PATH pnpm dlx pyright@1.1.411 \
  --pythonpath "$PYRIGHT_PYTHON" --project "$SOURCE_PYRIGHT_CONFIG" \
  tensor_dslab tests
env PATH=$PYRIGHT_NODE:$PATH pnpm dlx pyright@1.1.411 \
  --pythonpath "$PYRIGHT_PYTHON" --project "$ARCHIVE_PYRIGHT_CONFIG" \
  tensor_dslab tests
```

Both commands must report zero errors and zero warnings. Absolute CLI targets
are deliberately not placed in a temporary config's `include`; Pyright
resolves `include` relative to the config location.

Fresh-process import and retired-surface probes run against both dependency
forms by substituting `$TC_SOURCE` and `$TC_ARCHIVE_ROOT`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_SOURCE python -c "import sys, tensor_dslab; names=('tensor_g4ds','tensor_ml','dslab','g4ds11'); result=tuple(name in sys.modules for name in names); print(*result); assert result == (False, False, False, False); generic=('RngKey','CounterRng','Threefry4x32','logical_positions','require_same_dtype'); assert all(name not in vars(tensor_dslab) for name in generic)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_ARCHIVE_ROOT python -c "import sys, tensor_dslab; names=('tensor_g4ds','tensor_ml','dslab','g4ds11'); result=tuple(name in sys.modules for name in names); print(*result); assert result == (False, False, False, False); generic=('RngKey','CounterRng','Threefry4x32','logical_positions','require_same_dtype'); assert all(name not in vars(tensor_dslab) for name in generic)"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_SOURCE python -c "import importlib.util; retired=('tensor_dslab.readout._random','tensor_dslab.readout._rng','tensor_dslab.readout.types','tensor_dslab.readout.photoelectrons.types','tensor_dslab.readout.charge.types','tensor_dslab.readout.pure_waveform.types','tensor_dslab.readout.noise_waveform.types','tensor_dslab.readout.analog_waveform.types','tensor_dslab.readout.digitized_waveform.types'); found=tuple(name for name in retired if importlib.util.find_spec(name) is not None); assert not found, found"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:$TC_SOURCE python -c "import tensor_dslab.readout.charge.effects as effects; assert getattr(effects, '__all__', ()) == (); public=tuple(name for name in vars(effects) if not name.startswith('_')); assert not public, public"
```

Every candidate runs these forbidden production scans:

```bash
test ! -e tensor_dslab/readout/_random.py
test ! -e tensor_dslab/readout/_rng.py
test -z "$(find tensor_dslab/readout -type f -name 'types.py' -print)"
! rg -n '(^|[[:space:]])(from|import)[[:space:]]+tensor_core\.' tensor_dslab
! rg -n '\b(_RngStream|_threefry4x32|_random_block|_raw_word|_uniform_closed_open|_uniform_open_open|_standard_normal_pair|_sample_poisson|_sample_conditional_binomial)\b' tensor_dslab
! rg -n '\bseed[[:space:]]*:' tensor_dslab/readout
! rg -n 'torch\.(rand|rand_like|randn|randn_like|randint|randint_like|randperm|poisson|multinomial|normal|manual_seed|seed)\(' tensor_dslab
! rg -n 'torch\.Generator\(' tensor_dslab
! rg -n '\.(cpu|numpy|tolist|detach)\(' tensor_dslab
! rg -n '(^|[[:space:]])(from|import)[[:space:]]+(numpy|scipy|dslab|g4ds11|tensor_g4ds|tensor_ml)(\.|[[:space:]])' tensor_dslab
! rg -n '\bNormalDelayConfig\b' tensor_dslab tests
```

Tests may name retired symbols only as negative assertions; the local RNG scan
therefore targets production. Final fixed-commit, allowlist, and hygiene gates
are:

```bash
test "$(git rev-parse HEAD)" = "$M2_CANDIDATE"
test -z "$(git status --porcelain=v1)"
git merge-base --is-ancestor "$M2_BASE" "$M2_CANDIDATE"
git diff --check "$M2_BASE"..."$M2_CANDIDATE"
git diff --name-status "$M2_BASE"..."$M2_CANDIDATE"
git diff --stat "$M2_BASE"..."$M2_CANDIDATE"
test -z "$(find tensor_dslab tests \( -type d -name '__pycache__' -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) -print)"
test ! -e .pytest_cache
test ! -e .coverage
test ! -e htmlcov
test ! -e build
test ! -e dist
test -z "$(find . -maxdepth 2 -type d -name '*.egg-info' -print)"
test -z "$(git status --porcelain=v1)"
```

Validation and Review compare the literal `git diff --name-status` endpoints
against the exact allowlist below. Review uses an independent checkout and
remains read-only until it authorizes and performs a clean fast-forward.
Post-merge runs the full source/archive suites, both Pyright forms, import
isolation, diff, artifact, and final-cleanliness gates on unchanged merged
bytes. An unavailable accelerator or build tool is a reported qualification,
not permission to claim evidence.

## Exact Candidate Allowlist

The following is the complete allowed name-status inventory relative to the
future committed Design-authority commit. Any changed path outside it is a stop
condition. A rename may appear as delete-plus-add under a different Git
similarity threshold without changing its allowed endpoints.

Production and metadata:

```text
M  pyproject.toml
M  tensor_dslab/__init__.py
M  tensor_dslab/readout/__init__.py
M  tensor_dslab/readout/_requirements.py
M  tensor_dslab/readout/analog_waveform/__init__.py
M  tensor_dslab/readout/analog_waveform/_produce.py
M  tensor_dslab/readout/charge/__init__.py
M  tensor_dslab/readout/charge/_produce.py
M  tensor_dslab/readout/digitized_waveform/__init__.py
M  tensor_dslab/readout/digitized_waveform/_produce.py
M  tensor_dslab/readout/noise_waveform/__init__.py
M  tensor_dslab/readout/noise_waveform/_produce.py
M  tensor_dslab/readout/photoelectrons/__init__.py
M  tensor_dslab/readout/pure_waveform/__init__.py
M  tensor_dslab/readout/pure_waveform/_produce.py

A  tensor_dslab/readout/config.py
A  tensor_dslab/readout/collection.py
A  tensor_dslab/readout/analog_waveform/config.py
A  tensor_dslab/readout/analog_waveform/field.py
A  tensor_dslab/readout/charge/config.py
A  tensor_dslab/readout/charge/field.py
A  tensor_dslab/readout/charge/effects/__init__.py
A  tensor_dslab/readout/charge/effects/_counts.py
A  tensor_dslab/readout/charge/effects/_delays.py
A  tensor_dslab/readout/charge/effects/_dark_counts.py
A  tensor_dslab/readout/charge/effects/_timing_jitter.py
A  tensor_dslab/readout/charge/effects/_correlated_avalanches.py
A  tensor_dslab/readout/charge/effects/_smearing.py
A  tensor_dslab/readout/digitized_waveform/config.py
A  tensor_dslab/readout/digitized_waveform/field.py
A  tensor_dslab/readout/noise_waveform/config.py
A  tensor_dslab/readout/noise_waveform/field.py
A  tensor_dslab/readout/pure_waveform/config.py
A  tensor_dslab/readout/pure_waveform/field.py

D  tensor_dslab/readout/_random.py
D  tensor_dslab/readout/types.py
D  tensor_dslab/readout/analog_waveform/types.py
D  tensor_dslab/readout/charge/types.py
D  tensor_dslab/readout/digitized_waveform/types.py
D  tensor_dslab/readout/noise_waveform/types.py
D  tensor_dslab/readout/pure_waveform/types.py

R  tensor_dslab/readout/photoelectrons/types.py
   tensor_dslab/readout/photoelectrons/field.py
```

Runtime and typing evidence:

```text
M  tests/test_charge_correlated_avalanches.py
M  tests/test_charge_delay_preparation.py
M  tests/test_charge_product.py
M  tests/test_charge_timing_jitter.py
M  tests/test_deterministic_waveform_products.py
M  tests/test_noise_waveform_product.py
M  tests/test_package_contracts.py
M  tests/test_readout_collection.py
M  tests/test_readout_configs.py
M  tests/test_readout_product_types.py

A  tests/test_rng_ownership_migration.py
A  tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py

D  tests/test_readout_random.py
D  tests/typing/stage_5_readout_rng_and_stochastic_noise.py
D  tests/typing/stage_6_charge_simulation.py

R  tests/test_readout_count_sampling.py
   tests/test_charge_count_orchestration.py
```

Live documentation that may be synchronized to candidate facts:

```text
M  AGENTS.md
M  CONTRIBUTING.md
M  README.md
M  docs/architecture/readout.md
M  docs/architecture/rebuild.md
M  docs/architecture/tensors.md
M  docs/decisions.md
M  docs/design.md
M  docs/implementation/index.md
M  docs/implementation/maintenance_2_rng_and_product_module_ownership_migration.md
M  docs/overview.md
M  docs/parity.md
M  docs/validation.md
```

Everything else is protected. Important explicit protected paths include
`LICENSE`, `pyrightconfig.json`, `tensor_dslab/py.typed`, all
`tensor_dslab/common/` files, `tests/readout_fixtures.py`,
`tests/test_readout_axes_and_sampling.py`, the Stage 3/4 typing fixtures,
`docs/governance/**`, `docs/physics/correlated_avalanches.md`, every closed
Stage 0 through Stage 6 work order, Maintenance 1, and the historical
TensorCore consumer proposal.

Historical Stage 0 through Stage 6 work orders, governance records, and closed
decision records remain byte-identical unless a later explicit Design
correction authorizes a documentary amendment. This work does not rewrite
history to pretend TensorCore owned the original Stage 5/6 implementation.

## Dispatch And Role Loop

Production execution was dispatched from exact Design authority
`daa046405f62ee324bc495867e796213bf6657a6` after clean-topology and persistent-
route reverification plus separate user authorization. Lifecycle is determined
by repository topology: absence of these exact bytes from `main` means they
remain a candidate in the fixed-commit loop; presence unchanged on `main`
means Review's clean fast-forward completed and Design acceptance remains
pending; only the two-document `Merged / Closed` closeout completes final
acceptance. The only permitted Maintenance 2 execution states are:

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
authorize no scope expansion. Design alone dispatches Maintenance 2 and accepts
its final closeout. Implementation, Validation, and Review report the
intermediate dispositions above.

The production loop is:

```text
TensorDSLab Design
  -> TensorDSLab Implementation
  -> fixed-commit TensorDSLab Validation
  -> independent fixed-commit TensorDSLab Review
  -> bounded Implementation correction when returned
  -> Validation/Review recheck of changed bytes
  -> Review clean fast-forward
  -> final TensorDSLab Design acceptance
```

The loop is explicitly authorized only after dispatch and is bounded by:

- at most three Implementation-to-Validation dispatches;
- at most three Validation-to-Implementation returns;
- stopping early when Validation reports no blocking finding;
- returning to Design if the same issue recurs twice, either budget is
  exhausted, a route becomes stale/missing/discrepant, or a Design decision is
  required;
- fixed-commit Validation and read-only fixed-commit Review; and
- re-running the applicable same-byte gates after every candidate change.

No TensorCore role clears a TensorDSLab candidate, and no TensorDSLab role
edits or clears TensorCore. Review alone may perform the clean fast-forward
merge after exact fixed-commit clearance; Design alone accepts the final
merged state.

## Merge And Evidence-Only Closeout

After exact fixed-candidate clearance, Review alone fast-forwards an otherwise
unchanged clean `main` to that candidate and repeats the frozen post-merge
gates on the unchanged bytes. Review reports the resulting `main` commit,
commands, totals, environment, residual qualifications, and no-effects
statement to Design; Review makes no closeout commit.

Design then independently reconciles the merged bytes and Review evidence and
repeats proportionate post-merge checks. A substantive discrepancy returns the
work to the accepted finite loop and authorizes no documentary closeout. If
Design accepts the unchanged merged candidate, Design makes exactly one
evidence-only closeout commit whose parent is that candidate and whose only
changed paths are:

```text
M  docs/implementation/maintenance_2_rng_and_product_module_ownership_migration.md
M  docs/implementation/index.md
```

The closeout records `Merged / Closed`, the exact Design authority, fixed
candidate and identical pre-closeout merged-`main` commit, the selected
TensorCore commit and archive checksum, all test/checker totals and
environments, residual qualifications, and the no-effects statement. The
containing commit is the closeout authority and is identified externally by
`HEAD` after creation; the document does not attempt to embed its own hash. It
must not change production, tests, metadata, README, architecture, parity,
governance, or any dependency byte. Before accepting that closeout, Design
verifies:

```bash
test "$(git rev-parse HEAD^)" = "$M2_CANDIDATE"
test -z "$(git status --porcelain=v1)"
git diff --check "$M2_CANDIDATE"..HEAD
git diff --name-status "$M2_CANDIDATE"..HEAD
test -z "$(git diff --name-only "$M2_CANDIDATE"..HEAD -- pyproject.toml tensor_dslab tests pyrightconfig.json)"
```

The name-status output must be exactly the two modified documentation paths
above. The evidence-only commit changes no cleared production or test byte and
does not require another Implementation/Validation/Review cycle. Design's
acceptance of that commit is the transition from `Merged / Design acceptance
pending` to `Merged / Closed`.

## Non-Goals And Forbidden Scope

Maintenance 2 does not authorize:

- `simulate_readout(...)`, `readout/simulation.py`, request dependency closure,
  requested-product retention, or any other Stage 7 public orchestration;
- closure-wide duplicate-key checking, a key registry, automatic key
  assignment, or key derivation;
- any scientific law, parameter, equation, probability, delay, recovery,
  cascade, count ceiling, ledger, overflow, PSD, pulse, analog, or digitization
  change;
- `NormalDelayConfig` or another delay family;
- a public multinomial, Bernoulli, exponential, standard-normal, normal, or
  raw-bit API;
- a local `_rng.py`, RNG wrapper/facade, `seed=` shim, `types.py` shim, old-name
  re-export, or compatibility alias;
- TensorDSLab re-export of TensorCore RNG classes/functions;
- TensorCore repository edits or cross-package implementation from this work
  order;
- partial-axis agreement, overflow-safe generic reductions, range/count axes,
  or another speculative TensorCore extraction;
- workspace, `out=`, output reuse, allocator, pool, stream lease, generation
  lifetime, compiler, fusion, Triton/CUDA kernel, or GPU-performance work;
- public IO, persistence, cache, artifact, TensorG4DS bridge, TensorML adapter,
  reconstruction, DAG, or campaign behavior;
- release, deployment, backward-compatibility, broad compatibility,
  conformance, or zero-copy claims;
- Coordination/Profile B/routing activation; or
- a push.

## Stop Conditions

Return to TensorDSLab Design without broadening implementation if:

- TensorCore's merged public API, device matrix, address packing, Gaussian
  pair/ordinal mapping, floating operation order, Poisson mapping, binomial
  mapping, mass semantics, exhaustion rule, result contract, or exceptions
  differ from the accepted consumer contract;
- the selected TensorCore commit is not clean, merged, Review/Design-cleared,
  exactly archived, or reproducible from the dependency pin;
- the selected commit is not reachable through the exact dependency
  URL/reference declared in `pyproject.toml`;
- implementation needs a protected/private TensorCore import, raw-word API,
  local RNG wrapper, or copied generic sampler;
- any default-key fixture or same-backend Stage 5/6 continuity fixture changes;
- scientific/statistical behavior, source/global-RNG immutability, freshness,
  axes/device/dtype, or failure effects drift;
- the module split creates an import cycle or needs broader ownership than the
  exact tree above;
- Stage 7 collision/preflight/orchestration logic appears necessary;
- a dirty or overlapping worktree, route discrepancy, unexpected baseline,
  dependency/archive mismatch, or unapproved file is found;
- an execution role attempts to modify the sibling repository or use one
  package's clearance for the other; or
- the finite correction loop is exhausted.

CUDA absence, compiler absence, or build-tool absence is a qualification, not
a blocker unless the final dispatch explicitly makes it one. It never permits
an accelerator, fusion, installation, or wheel claim.

## Completion Boundary

Maintenance 2 is complete only when one unchanged fixed candidate has:

- selected and verified the exact TensorCore commit;
- completed the exact module split and deletion policy;
- migrated every generic RNG/distribution call to the public TensorCore API;
- preserved all ten default-role and completed-product continuity fixtures;
- retained all TensorDSLab scientific planning and bookkeeping locally;
- cleared the complete source/archive runtime, typing, import, documentation,
  diff, and artifact gates;
- received fixed-commit Validation, independent Review, clean fast-forward,
  and final TensorDSLab Design acceptance; and
- left Stage 7, optimization, integration, governance, Coordination, and push
  state unchanged.

Only that closeout makes Stage 7 eligible for a separately drafted and
separately dispatched work order. It does not itself create the public
simulation API.
