# Maintenance 7 TensorCore 0.15 Adoption Work Order

Status: **Design-complete / User authorization pending / Undispatched**.

Stable work-order key:
`TensorDSLab/maintenance-7-tensorcore-0-15-adoption`.

## Authority And Baseline

This work order is based on exact clean local TensorDSLab `main` after the
Maintenance 6 Design closeout:

```text
commit: 65bb55bf98bb37a129a950d93a0bdb9b0d3f2971
tree:   c76269e043c81b18243b8355327131eac68e3f0a
```

The baseline is intentionally unpushed. Maintenance 6 is Merged / Closed and
retains exact TensorCore `0.13.0` plus Pint `0.25.3`. Its complete local
evidence has `13` unavailable-CUDA skips and makes no accelerator claim.

TensorCore has published exact `0.15.0` on GitHub:

```text
repository:      https://github.com/mbedard44/TensorCore.git
commit:          0f974e9e7f52125bbe829e124beb24e69de811d3
tree:            587ff59711255c027a85cfef883422d40ea5dcda
parent:          983a37fd5996c24c1fad3eeec994d365686dac6f
package version: 0.15.0
Python:          >=3.11
Torch:           >=2.11,<2.13
root exports:    34
package files:   25
```

An independently generated exact `git archive --format=zip` of that commit is
`439969` bytes with SHA-256
`4af0210adf23d6e81a1da725889e223c5b151bb37779b7820ea4281c0a43c2fb`.
Implementation, Validation, and Review must independently reconstruct and
verify the exact dependency source and archive rather than trusting this
Design-side observation alone.

This work order follows the TensorCore, public typing, validation-boundary,
same-device, stochastic-address, product-semantics, and testing standards in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md). The dependency/RNG migration changes
no donor boundary. The positive-amplitude/fixed-negative-polarity narrowing
keeps the calibrated rendered parity result but changes how caller calibration
expresses that sign, so Design synchronizes
[`docs/parity.md`](../parity.md) explicitly.

## Objective

Adopt the exact published TensorCore `0.15.0` dependency and remove
TensorDSLab's remaining generic validation and raw RNG-position mechanics:

```text
TensorCore 0.15
  owns generic dtype/layout/representability/allocation/count requirements
  owns validated immutable RngPositions and trusted position transforms

TensorDSLab
  owns readout axis composition
  owns physical/Pint normalization
  owns scientific count arithmetic and ledgers
  owns stochastic role keys and one readout RNG namespace
  owns product prepare / produce / validate actions
```

The dependency and RNG migration must be behavior-preserving on supported
TensorDSLab paths. The one separately ratified public scientific/API narrowing
in this maintenance is the pulse-amplitude convention: both pulse Configs
accept a strictly positive peak-voltage magnitude, and pure-waveform
preparation applies the fixed DS20k negative polarity exactly once. Existing
calibrated negative-going rendered outputs remain exact. No scientific
equation, role stream number, positional address, Threefry word, distribution
law, stochastic traversal order, exact-zero draw behavior, product meaning,
field name, canonical unit, or Pint ownership boundary otherwise changes.

## Dependency Adoption

Replace the exact `pyproject.toml` TensorCore pin:

```text
202d8b1bc6259b8453d3d377570417f2480d782b
```

with:

```text
0f974e9e7f52125bbe829e124beb24e69de811d3
```

Keep:

- TensorDSLab version `0.1.0`;
- exact Pint `0.25.3`;
- Python `>=3.11`;
- the existing unbounded TensorDSLab `torch` dependency spelling;
- Hatch build metadata and package selection; and
- the exact 35-name package-root, 30-name readout, and 5-name common exports.

TensorDSLab imports public TensorCore roots through `tensor_core` and the
supported validation parts bin through `tensor_core.validation` or the one
accepted precise leaf `tensor_core.validation.random`. It must not import
TensorCore-private modules or `_values` from `RngPositions`.

## Generic Validation Ownership

### Field and representability requirements

Delete TensorDSLab's local implementations of:

```text
require_dtype
require_floating_dtype
require_representable_float
```

Use the exact TensorCore `0.15.0` surfaces:

```python
require_field_dtype(field, torch.int64)
require_field_dtype(field, torch.int32)
require_field_dtype(field, torch.float32, torch.float64)
require_representable_float(value, dtype=dtype, field=field)
```

The six semantic field `_require()` hooks continue to call TensorDSLab's
domain-owned `require_readout_structure(...)`, then call
`require_field_dtype(...)` directly. All preparation/effect modules that need
eager scalar representability import `require_representable_float` directly
from TensorCore. TensorDSLab does not re-export or wrap either helper.

`require_readout_structure(...)` remains in
`tensor_dslab.readout.requirements`. It continues to require exactly one
`ExampleAxis`, `ChannelAxis`, and `SampleAxis` in arbitrary dimension order.
Its generic dense-layout clause composes:

```python
require_field_layout(field, torch.strided)
```

TensorCore does not own the unordered readout axis set. The change may adopt
TensorCore's generic private diagnostic wording for dtype/layout mismatch, but
the existing TypeError-versus-ValueError relationship categories and every
supported acceptance/rejection case remain fixed.

### Charge count domain

Delete the local free-standing `require_count_domain(...)` implementation.
Every applicable Charge preparation/effect caller directly uses:

```python
from tensor_core.validation.random import require_count_tensor
```

The shared law is exact for TensorDSLab's existing active count domain:

- exact ordinary `torch.Tensor`;
- `torch.strided`;
- `torch.int64`;
- `requires_grad=False`;
- empty accepted without a scan;
- nonempty inclusive values in `[0, 2**53 - 1]`;
- same-object return; and
- one combined nonempty reduction and host-scalar extraction.

This migration deliberately replaces the prior two-reduction local scan with
TensorCore's one-reduction boundary. It may change private diagnostic wording
and strengthens malformed structural rejection; it changes no accepted count,
scientific ceiling, device, or count result.

The following remain TensorDSLab-owned in
`readout/charge/runtime/effects/counts.py`:

- `MAX_COUNT = 2**53 - 1`;
- `MAX_POISSON_MEAN`;
- `checked_add(...)`;
- `checked_subtract(...)`;
- `checked_rate_product(...)`;
- `draw_ordered_categories(...)`; and
- the readout-specific positional basis builder, renamed only if necessary to
  accept and return `RngPositions`.

Those functions enforce Charge accumulation, remainder, Poisson-rate,
category-order, and scientific ceiling policy. They are not generic
TensorCore validation.

### Allocation and shape-span requirements

Delete the local free-standing `require_tensor_allocation(...)`. Charge
preparation directly imports:

```python
from tensor_core.validation import require_tensor_allocation
```

and calls it with:

```text
upper = 2**63
```

for both source and output shapes. On the supported readout path, all three
semantic axes are nonempty, so TensorCore's ordinary-numel contract is
equivalent to the retired local helper:

- reject `numel >= 2**63`;
- reject `numel * element_size >= 2**63`; and
- return exact built-in `numel`.

Noise preparation replaces `_require_position_count(...)` with the exact
zero-as-one address-lattice check:

```python
require_shape_span(shape, "noise output", upper=2**63)
```

This is address-span policy, not an allocation-success promise. No helper
queries available memory or performs a device allocation.

### Helpers intentionally retained downstream

Do not promote, delete, or weaken:

- `require_readout_structure(...)`;
- request product and config closure;
- distinct stochastic-role key policy;
- CPU/CUDA readout device-family policy;
- Pint recognition, registry isolation, dimensions, canonical units, and
  `Scalar.require(...)` use;
- SampleAxis and complete-input start/count/period rules;
- Charge accumulation and ledger checks;
- scientific probability/rate/envelope/address bounds;
- exact axes identity, dtype/device relationships, storage freshness, and
  generated-product validation; or
- the explicit `prepare_<product>`, `produce_<product>`, and
  `validate_<product>` action boundary.

Do not use `require_dimension(...)` in places that currently require a
nonnegative sample dimension. TensorCore accepts Torch-style negative
dimensions, so its generic contract is not an exact replacement there.

## Prepared RNG Positions

Remove every `logical_positions` import and call. There is no TensorDSLab
wrapper, alias, compatibility shim, union annotation, raw accessor, or
fallback. Public TensorCore distribution calls receive exact `RngPositions`.

### Exact production transform map

| TensorDSLab role | Exact TensorCore `RngPositions` construction |
| --- | --- |
| white noise | `RngPositions.from_shape(shape, device=device)` |
| PSD coefficient bank | `RngPositions.from_shape((row_count, frequency_count), device=device).slice(1, 1, None)` |
| dark counts | `RngPositions.from_shape(counts.shape, device=counts.device)` |
| charge smearing | `RngPositions.from_shape(charge.shape, device=charge.device)` |
| sample-last Charge basis | `RngPositions.from_shape(shape, device=device).movedim(sample_dimension, -1)` |
| timing-jitter source | `basis.select(-1, source).offset(target * tensor_numel)` |
| DiCT/DeCT generation | `basis.offset(generation_index * tensor_numel)` |
| AP category | `basis.select(-1, source).offset((generation_index * (sample_count + 1) + category) * tensor_numel)` |

All offsets are exact nonnegative built-in integers. Existing preflight proves
the corresponding cached exclusive upper bound is at most `2**63`, including
the accepted equality boundary. No production path needs `from_tensor`,
signed offset, reshape/view/expand, broadcasting, general permutation,
concatenation, stacking, advanced indexing, arbitrary arithmetic, or raw
position access.

`draw_ordered_categories(...)` accepts
`tuple[RngPositions, ...]`. The internal readout position-basis function
returns `RngPositions`. Product/effect tests may observe the raw tensor only
through the protected `CounterRng._generate_block(...)` test hook. They must
not reach into `RngPositions._values`.

### Determinism and alias rules

For every migrated production route:

- the raw position tensor delivered to `_generate_block(...)` has the same
  dtype, device, shape, values, ordering, and non-renumbering behavior as the
  retired `logical_positions` expression;
- key, quantum, ordinal, count, dtype, and distribution arguments are exact;
- same-seed raw words and completed products remain exact on the same accepted
  backend/execution stack;
- Maintenance 3's environment-qualified stochastic-literal boundary remains
  unchanged;
- no global Torch RNG state is read or mutated;
- caller/source tensors remain immutable;
- `RngPositions.from_tensor(...)` snapshot behavior is tested as a TensorCore
  consumer contract but is not used by TensorDSLab production; and
- disabled and exact-zero paths still request no words and avoid unnecessary
  position construction.

## TensorDSLab RNG Namespace

Add one non-exported package-policy module:

```text
tensor_dslab/readout/rng_keys.py
```

with exactly one current value:

```python
RNG_NAMESPACE = 0x54445331
```

Both Charge and Noise config modules import that value for their default
`RngKey` objects. Remove product-local `_RNG_NAMESPACE` and production
hardcodings of `0x54445331`. Keep all ten append-only stream values exactly
unchanged:

```text
0x0000_0001  noise white
0x0000_0002  noise PSD coefficient
0x0000_0003  dark count
0x0000_0004  direct crosstalk retained
0x0000_0005  direct crosstalk overflow
0x0000_0006  delayed crosstalk retained
0x0000_0007  delayed crosstalk overflow
0x0000_0008  timing jitter
0x0000_0009  afterpulse
0x0000_000A  charge smearing
```

The module and constant are export-private: neither is added to
`tensor_dslab.readout.__all__` or `tensor_dslab.__all__`. Historical
documentation and expected-value tests may continue to spell the literal when
they are recording the frozen wire/address value. Production config bytes must
contain the literal only in `rng_keys.py`.

## Product And Runtime Boundaries

Maintenance 7 preserves Maintenance 4 and Maintenance 6 ownership:

```text
public Config
  -> prepare_<product>(...)  # interpretation and preflight
  -> ProductRuntime         # plain execution facts
  -> produce_<product>(...) # tensor/RNG execution
  -> validate_<product>(...)# immediate product/relationship postconditions
```

No Config ABC, generic Runtime ABC, registry, reflection graph, product
protocol, compatibility layer, or cross-product fusion is introduced.
Producers still import no Config or validator. Runtime records remain Pint-
free, Config-free, final, frozen, and slotted. The public
`simulate_readout(...)` signature and request-closure semantics are unchanged.

## Positive Pulse Amplitude And Fixed DS20k Polarity

`TpcFebSnrPulseConfig.peak_voltage_per_photoelectron` and
`VetoPduPulseConfig.peak_voltage_per_photoelectron` denote amplitude
magnitudes. Both Configs canonicalize that field in `mV` through
`PositiveFloat`, not `FiniteFloat` plus a separate zero check. Zero, signed
zero, and negative caller quantities are rejected during Config construction.
The field name and public `Quantity` representation remain unchanged.

All Quantity-bearing Configs use one TensorDSLab-private
`_canonicalize_quantity_fields(...)` mechanism from `common/units.py`.
Each Config calls that helper exactly once from `__post_init__` and supplies
two deliberately separate declarations:

- `quantity_fields`: `(name, unit, constraint)` rows that own physical
  conversion and scalar-domain normalization; and
- `tuple_fields`: names whose accepted non-`None` representation must be
  exactly `tuple`.

The helper has this fixed private shape:

```python
def _canonicalize_quantity_fields(
    config: object,
    *,
    quantity_fields: tuple[
        tuple[str, str, type[Scalar[float]]],
        ...,
    ],
    tuple_fields: tuple[str, ...] = (),
) -> None:
    owner = type(config).__name__

    for name in tuple_fields:
        parameter = getattr(config, name)
        if parameter is not None and type(parameter) is not tuple:
            raise TypeError(f"{owner}.{name} must be a tuple")

    for name, unit, constraint in quantity_fields:
        parameter = getattr(config, name)
        if parameter is None:
            continue

        is_tuple = name in tuple_fields
        values = parameter if is_tuple else (parameter,)
        canonical = tuple(
            _canonical_quantity(
                value,
                unit=unit,
                field=(
                    f"{owner}.{name}[{index}]"
                    if is_tuple
                    else f"{owner}.{name}"
                ),
                constraint=constraint,
            )
            for index, value in enumerate(values)
        )
        object.__setattr__(
            config,
            name,
            canonical if is_tuple else canonical[0],
        )
```

The scalar-as-one-item sequence makes `_canonical_quantity(...)` occur in only
one helper branch. The helper then restores the declared scalar-versus-tuple
shape. It does not coerce lists or other sequences to tuples. It skips every
optional `None` value before unpacking or canonicalization and preserves that
field as `None`. A declared tuple is canonicalized element by element, in
order, into a newly owned tuple; every failing element uses the exact indexed
field label. Empty tuples pass this generic mechanics layer so that the owning
Config can apply its own nonempty relationship if required.

Tuple-valued physical fields remain `tuple[Quantity, ...]`; they do not become
one array-valued `Quantity`. Under the selected Pint version, constructing a
Quantity from a tuple may promote its magnitude to a mutable NumPy array when
NumPy is available, while accepted environments need not contain NumPy.
Array-valued quantities would therefore weaken frozen-Config storage,
environment independence, exact built-in scalar validation, independently
convertible element units, and indexed diagnostics. The helper accepts and
canonicalizes only scalar-magnitude Quantity elements and reassembles an exact
immutable tuple.

For example, the TPC Config declaration is:

```python
_canonicalize_quantity_fields(
    self,
    quantity_fields=(
        ("fast_time_constant", "ns", PositiveFloat),
        ("slow_time_constant", "ns", PositiveFloat),
        ("support_time", "ns", PositiveFloat),
        ("peak_voltage_per_photoelectron", "mV", PositiveFloat),
    ),
)
```

`VetoPduPulseConfig` uses the same exact `(name, unit, constraint)` shape for
its eight fields. This removes its field-name-dependent unit conditional.
`DigitizedWaveformConfig` uses the same declaration-table shape for its two
required scalar voltage quantities. `AnalogSaturationConfig` uses the same
table for its two optional scalar voltage quantities; the helper skips `None`
without requiring an optional marker in each row. Single-Quantity Configs use
the same helper with one declaration row rather than retaining a second direct
canonicalization pattern.

`PsdNoiseConfig` keeps tuple representation separate from element quantity
semantics. Its one helper call supplies a tuple-admission table:

```python
tuple_fields=(
    "frequency_left_edges",
    "power_density",
),
```

A separate `(name, unit, constraint)` table owns both tuple-element and scalar
quantity canonicalization:

```python
quantity_fields=(
    ("frequency_left_edges", "Hz", NonnegativeFloat),
    ("frequency_stop", "Hz", PositiveFloat),
    ("power_density", "mV ** 2 / Hz", NonnegativeFloat),
),
```

The helper recognizes the two tuple names, unpacks their elements, applies
every declared unit and constraint with indexed diagnostics, and reassembles
fresh tuples. It treats `frequency_stop` as a scalar through the same
quantity table. PSD nonemptiness, equal tuple lengths, ordered coverage,
exclusive stop, and nonzero-power requirements remain explicit relationship
checks in `PsdNoiseConfig.__post_init__` after the shared representation and
canonicalization mechanics. The tuple table must not be merged with the
quantity table: immutable sequence representation and physical element
semantics are distinct Config responsibilities.

`__post_init__`, not `__new__`, remains the canonicalization owner: the
dataclass-generated initializer must first assign the caller's values, after
which Config construction normalizes representation and checks genuine local
relationships. This is not a Config ABC, metadata/reflection framework, or
requirement that structurally different Quantity fields use an inappropriate
scalar loop.

`prepare_pure_waveform(...)` extracts the positive magnitude once and creates
the signed execution fact exactly once:

```python
signed_peak_voltage_mv_per_pe = -canonical_magnitude(
    model.peak_voltage_per_photoelectron
)
```

That signed negative value enters the existing representability and coefficient
equations. The prepared Runtime remains Config-free and Pint-free. The existing
represented-zero check remains mandatory because a valid strictly positive
binary64 magnitude can still vanish when represented in the requested Torch
dtype. No producer applies another sign, gain, or inversion.

The calibrated TPC and Veto Config fixtures change from negative caller
quantities to equal positive magnitudes. Their prepared kernels,
`PureWaveform`, downstream `AnalogWaveform`, and downstream
`DigitizedWaveform` values remain exact on the same accepted execution stack.
Formerly accepted positive rendered polarity is deliberately retired: positive
and negative Config inputs are not two supported detector modes. The single
negative-going polarity is part of TensorDSLab's DS20k DAQ model.

## Exact Production Scope

Implementation may change exactly these production/metadata paths:

```text
pyproject.toml
tensor_dslab/common/units.py
tensor_dslab/readout/rng_keys.py
tensor_dslab/readout/requirements.py
tensor_dslab/readout/analog_waveform/config.py
tensor_dslab/readout/analog_waveform/field.py
tensor_dslab/readout/charge/config.py
tensor_dslab/readout/charge/field.py
tensor_dslab/readout/charge/runtime/prepare.py
tensor_dslab/readout/charge/runtime/effects/correlated_avalanches.py
tensor_dslab/readout/charge/runtime/effects/counts.py
tensor_dslab/readout/charge/runtime/effects/dark_counts.py
tensor_dslab/readout/charge/runtime/effects/smearing.py
tensor_dslab/readout/charge/runtime/effects/timing_jitter.py
tensor_dslab/readout/digitized_waveform/config.py
tensor_dslab/readout/digitized_waveform/field.py
tensor_dslab/readout/noise_waveform/config.py
tensor_dslab/readout/noise_waveform/field.py
tensor_dslab/readout/noise_waveform/runtime/prepare.py
tensor_dslab/readout/noise_waveform/runtime/produce.py
tensor_dslab/readout/photoelectrons/field.py
tensor_dslab/readout/pure_waveform/config.py
tensor_dslab/readout/pure_waveform/field.py
tensor_dslab/readout/pure_waveform/runtime/prepare.py
tensor_dslab/readout/analog_waveform/runtime/prepare.py
tensor_dslab/readout/digitized_waveform/runtime/prepare.py
```

No other production or metadata path is authorized. The target adds exactly
one package file and removes no package file. Package-root and product-facade
bytes are protected; every common/readout path not named above is protected.

Implementation may update only focused tests that exercise the changed
contracts. The candidate handoff must enumerate their exact paths. Expected
areas are:

```text
tests/test_charge_correlated_avalanches.py
tests/test_charge_count_orchestration.py
tests/test_charge_product.py
tests/test_deterministic_waveform_products.py
tests/test_noise_waveform_product.py
tests/test_package_contracts.py
tests/test_pint_physical_configuration.py
tests/test_readout_configs.py
tests/test_readout_product_types.py
tests/test_rng_ownership_migration.py
tests/test_readout_simulation.py
tests/test_runtime_action_ownership.py
tests/typing/maintenance_4_runtime_action_ownership.py
tests/typing/maintenance_2_rng_and_product_module_ownership_migration.py
tests/typing/maintenance_6_pint_physical_configuration_boundary.py
tests/typing/stage_4_deterministic_waveform_products.py
tests/typing/stage_7_public_readout_orchestration.py
```

Adding a new focused `tests/test_tensorcore_0_15_adoption.py` is preferred when
it makes the dependency, validation-delegation, namespace, and RngPositions
proof clearer than extending unrelated historical modules. Tests may not
alter protected scientific fixtures merely to make the migration pass.

## Required Focused Evidence

### Dependency and package

- exact TensorCore commit, tree, parent, version, 34 root exports, 19
  `tensor_core.validation` exports, one precise public
  `tensor_core.validation.random` export, and 25 package files;
- live GitHub `refs/heads/main` equal to the exact accepted commit before
  candidate work begins;
- exact source/archive package-byte equality and archive SHA-256;
- exact `pyproject.toml` pin and unchanged Pint/Torch/build metadata;
- isolated source and archive imports with no sibling package loaded;
- unchanged TensorDSLab 35/30/5 facade exports;
- exactly one new non-exported `readout/rng_keys.py`;
- no `logical_positions`, local duplicate of an adopted TensorCore generic
  helper, compatibility alias, wrapper, or private TensorCore import in
  production; and
- clean wheel/source import and tracked-package topology evidence.

### Validation ownership

- all six fields delegate exact dtype policy to `require_field_dtype`;
- readout structure delegates only generic layout policy while retaining the
  exact downstream axis set;
- every representability caller uses TensorCore's exact helper and retains its
  scientific postcondition checks;
- Charge source/effect count-domain calls use the exact public
  `require_count_tensor` binding;
- empty counts incur no reduction and a nonempty validation incurs exactly one
  combined reduction/host extraction;
- Charge allocation calls use `upper=2**63` and retain the exact accepted
  positive-readout boundaries;
- noise address-span preparation uses `require_shape_span`;
- mutants that restore or call the retired local helpers fail focused
  evidence; and
- product validators, axes identity, storage freshness, and scientific checks
  remain independently mutation-sensitive.

### RNG positions and namespace

- every row-major base, move, select, slice, and offset has exact raw-tensor
  agreement with the frozen mathematical expression;
- PSD drops only the DC position column without renumbering;
- timing-jitter, generation, and AP category addresses preserve every exact
  offset, including maximum accepted bounds;
- cached-bound rejection happens before overflow arithmetic;
- public distributions reject raw tensors and accept `RngPositions`;
- protected `_generate_block` observation receives raw tensors unchanged;
- `from_tensor` makes a defensive same-device snapshot;
- same-seed raw words, white/PSD noise, dark, jitter, correlated-avalanche,
  and smearing outputs preserve the accepted same-stack contract;
- exact-zero/disabled branches make no draw and no observationally meaningful
  position request;
- default config keys retain namespace and all ten stream values; and
- production contains exactly one `0x54445331` literal, in
  `readout/rng_keys.py`, while expected-value tests/docs remain free to record
  that frozen number.

### Pulse amplitude and polarity

- both pulse Configs use `PositiveFloat` for the canonical `mV` amplitude
  magnitude and contain no duplicate explicit zero check;
- exact positive quantities are accepted while zero, signed zero, and negative
  quantities are rejected with the canonical field-bearing scalar boundary;
- every Quantity-bearing Config calls the one private helper exactly once with
  an exact `(name, unit, constraint)` quantity table;
- Config modules no longer import or call `_canonical_quantity(...)` directly;
  that singular primitive remains module-local to `common/units.py`;
- optional scalar or tuple fields skip `None` before unpacking and
  canonicalization, preserve `None`, and need no per-row optional marker;
- scalar fields remain scalars, declared tuple fields require exact tuple
  representation, and every tuple element is canonicalized in order into a
  fresh tuple with an exact indexed diagnostic;
- one array-valued Pint Quantity, a list of quantities, or a tuple containing
  a non-scalar-magnitude Quantity is rejected rather than normalized into the
  accepted tuple-of-scalar-Quantity representation;
- `PsdNoiseConfig` passes one exact tuple-admission table separately from its
  quantity table, while all PSD relationship checks remain explicit after
  generic mechanics;
- `__post_init__` remains the sole Config canonicalization/local-relationship
  hook and neither Config defines `__new__`;
- pure-waveform preparation applies one and only one negative sign after
  magnitude extraction and before dtype representability;
- a nonzero magnitude that vanishes in the requested floating dtype remains a
  pre-production error;
- calibrated TPC/Veto kernels and negative-going pure, analog, and digitized
  outputs remain exact after converting their Config fixtures from negative
  values to equal positive magnitudes; and
- mutants that accept nonpositive amplitudes, omit or duplicate the fixed sign,
  retain the old explicit zero checks, restore the Veto unit conditional, or
  bypass one declaration-table entry fail focused evidence; and
- mutants that merge or bypass PSD tuple admission, accept or coerce a
  non-tuple sequence or array-valued Quantity, reject or canonicalize `None`,
  change a scalar into a tuple, omit/reorder/drop a tuple element, skip one
  indexed quantity constraint, reuse a non-indexed tuple diagnostic, or
  conflate element validation with a PSD relationship fail focused evidence.

### Typing

Pyright `1.1.411` standard mode must remain zero-diagnostic for production and
positive fixtures. Negative fixtures must prove:

- CounterRng distributions reject `torch.Tensor` positions statically;
- exact `RngPositions` construction and transform result types;
- `require_field_dtype`, `require_representable_float`,
  `require_tensor_allocation`, `require_shape_span`, and
  `require_count_tensor` call signatures;
- `RNG_NAMESPACE` is an exact `int` but is absent from package facades; and
- public TensorDSLab signatures and field/collection lookup types are
  unchanged.

Runtime tests own bool rejection where Python typing treats bool as int.

## Full Local Validation And Independent Review

Implementation, Validation, and Review use the persistent package routes and
the repository workflow:

```text
Design -> Implementation -> Validation -> Review
```

The ordinary loop budget is at most three Implementation-to-Validation
candidate submissions and at most three Validation-to-Implementation returns.
Each submission is an immutable clean commit. Exhaustion or a contract/scope
contradiction returns to Design.

For each exact candidate:

1. reconstruct exact TensorCore `0.15.0` source and archive;
2. verify exact Pint `0.25.3` wheel and sdist-built forms remain unchanged;
3. run the focused adoption/affected-product suites in every required
   dependency form;
4. run complete unittest discovery from exact source and archive forms;
5. run Pyright and the frozen negative typing probe;
6. run exact dependency, export, import-isolation, retired-surface,
   protected-byte, diff, artifact, and hygiene gates;
7. record exact test totals, environment, Torch version, execution mode,
   skipped tests, and artifact hashes; and
8. leave the candidate clean and unpushed.

Validation independently reconstructs inputs and reruns the complete fixed
gate. Review performs a read-only architecture/API/science/RNG/typing audit
and independently reruns the complete local evidence before any fast-forward.
Only one unchanged Validation-cleared and Review-cleared commit may be
fast-forwarded with `git merge --ff-only`.

### Accelerator scheduling

No Della or other cluster submission occurs inside the Maintenance 7
Implementation/Validation/Review loop. Local CUDA-unavailable skips remain
explicit, and this maintenance makes no fresh accelerator claim.

After exact Maintenance 7 local merge and Design closeout, Design will issue a
separate fixed integrated-CUDA evidence authority. TensorCore and TensorDSLab
then run their own package-owned complete two-Torch-minor CUDA matrices against
the exact integrated TensorCore `0.15.0` plus closed TensorDSLab pairing. A
result from an older pairing does not qualify the new baseline. The matrices
are functional correctness evidence only, not Stage 8 performance,
deployment, release, or broad-backend certification.

## Documentation Scope

Design may synchronize:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_7_tensorcore_0_15_adoption.md
docs/overview.md
docs/parity.md
docs/validation.md
```

Implementation may update only the new work order and implementation index for
candidate lifecycle/evidence. Final Design closeout may update the exact live
status records named by a pre-merge tree search and frozen before Review
fast-forward. The Design-ratified `docs/parity.md` amplitude/polarity wording
is part of the immutable authority baseline and is protected from
Implementation edits. Closed work orders, governance records, and history are
also protected.

## Stop Conditions

Return to Design without widening the candidate if:

- exact TensorCore `0.15.0` source/archive identity or published ref differs;
- a generic helper contract does not match the accepted TensorDSLab path;
- RngPositions cannot express an existing exact production address without a
  raw accessor or unsupported transform;
- any raw position, key, word, schedule, same-stack output, scientific law,
  no-draw path, axis identity, storage, or autograd behavior drifts;
- positive-magnitude Config input cannot reproduce the exact accepted
  negative-going calibrated waveform after one preparation-owned sign;
- the represented-zero safeguard would have to be removed or moved into tensor
  execution;
- the one-namespace cleanup requires a facade export, compatibility shim, or
  generic RNG module;
- a protected production, test, documentation, dependency, parity,
  governance, or history path must change;
- local evidence cannot distinguish restored local helpers, duplicate count
  scans, raw-position use, renumbering, or altered offsets; or
- adoption reveals a concrete need to combine the deferred CUDA authority
  with implementation correction.

## Non-Goals

Maintenance 7 does not:

- change readout science, probability laws, limits, or public products beyond
  the exact positive-amplitude/fixed-negative-polarity narrowing;
- change Pint fields, canonical units, Config construction, or Runtime facts
  beyond the private table-driven Quantity-field canonicalization and
  preparation-owned signed peak described above;
- add a Config, Runtime, renderer, artifact, IO, persistence, table, bridge,
  reconstruction, or TensorML surface;
- add or change an RNG algorithm, distribution, key, stream, seed, quantum,
  ordinal, word schedule, or global RNG interaction;
- expose RngPositions or TensorCore validators through TensorDSLab facades;
- add a general tensor-wrapper or arbitrary position transform;
- optimize, fuse, compile, benchmark, profile, or run Stage 8;
- push TensorDSLab, create a release/tag, publish a package, or claim
  compatibility; or
- edit TensorCore or another package.

## Acceptance

Maintenance 7 is complete only when:

- exact TensorCore `0.15.0` is pinned and independently verified;
- every authorized generic helper and RngPositions migration is complete with
  no local compatibility layer;
- the single non-exported TensorDSLab RNG namespace source is installed;
- both pulse Configs store positive canonical amplitude magnitudes,
  preparation applies fixed negative polarity exactly once, and calibrated
  negative-going outputs remain exact;
- all supported outputs, addresses, scientific and storage contracts remain
  unchanged;
- complete local fixed-commit Validation and independent Review clear one
  immutable candidate;
- Review fast-forwards that exact candidate locally;
- Design performs the bounded evidence-only closeout; and
- local `main` remains unpushed for the separate exact integrated CUDA gate.
