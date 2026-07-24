# Provisional DS20k Veto Profile And Public Readout Demos

Status: **Provisional Design notes / Deferred until Maintenance 8 closes**.
This document preserves an accepted post-Maintenance-8 direction. It is not an
operative work order, implementation authority, dependency decision,
calibration claim, dispatch, compatibility claim, cluster authority, or push
authorization.

The provisional planning label is Maintenance 9. Design will assign the exact
stable work-order key, authority commit, baseline commit, package version,
allowlist, evidence matrix, and loop budget only after TensorCore `0.16.0` and
TensorDSLab Maintenance 8 close.

## Purpose

Add one deliberately provisional package-owned DS20k Veto readout profile and
two executable public demonstrations:

```text
tensor_dslab/
  readout/
    profiles.py
demos/
  readout.py
  readout.ipynb
```

The profile makes a complete representative Veto readout configuration
available without hiding the ordinary compositional Config model. The demos
teach:

1. how a completed `Photoelectrons` field is built explicitly from semantic
   axes and an ordinary `torch.int64` tensor;
2. how the complete readout configuration is built explicitly;
3. how the same provisional configuration is obtained from `ds20k_veto()`;
4. how an experimental configuration is derived with
   `dataclasses.replace(...)`;
5. how `simulate_readout(...)` produces and retains typed products; and
6. where future FIL/TensorG4DS input and TensorML output adapters would replace
   the current demonstration boundaries.

This stage is intentionally separate from
[Maintenance 8](maintenance_8_python314_tensorcore_0_16_modernization.md).
Maintenance 8 remains the focused TensorCore `0.16.0`, Python `3.14`, PyTorch
`2.13`, syntax, dependency, validation-import, and docstring modernization.

## Sequencing

Before this proposal becomes an operative work order:

1. TensorCore must close and publish its accepted `0.16.0` modernization.
2. TensorDSLab Maintenance 8 must adopt that exact dependency and close
   locally.
3. Design must re-audit the exact post-Maintenance-8 public constructors,
   exports, typing, package topology, documentation tooling, and first-push /
   integrated-CUDA schedule.
4. Design must convert these notes into a fixed work order and obtain explicit
   user dispatch.

The later integrated-CUDA and first-push gate should bind the exact final
TensorCore/TensorDSLab pairing selected after the local stages stabilize. This
proposal itself makes no accelerator claim.

## Public Profile Surface

Add one real module:

```python
from tensor_dslab.readout.profiles import ds20k_veto
```

The provisional exact module export is:

```python
__all__ = ("ds20k_veto",)
```

Do not re-export the factory from the `tensor_dslab` or
`tensor_dslab.readout` package roots unless the final post-Maintenance-8 audit
finds a concrete golden-path need. Do not add `ds20k_tpc`, `ds20k_veto_v1`, a
profile constant, registry, class hierarchy, builder, override engine, loader,
serialized schema, or profiles package.

The phrase “use a profile” means call a parameterless factory. Nothing is
loaded, parsed, selected, cached, or read from external state.

## Provisional Factory Contract

`ds20k_veto()`:

- takes no arguments;
- returns exact `ReadoutConfig`;
- constructs a fresh complete Config tree and fresh canonical Pint quantities
  on every call;
- uses only public Config classes, TensorCore constrained scalars, and
  TensorDSLab quantity helpers;
- reads no file, environment variable, network resource, device state, or
  mutable application registry;
- owns no products, input axes, source data, RNG seed, dtype, device,
  requested-product set, retention, IO, or TensorML policy;
- performs no tensor allocation, preparation, production, or RNG activity; and
- is documented as a provisional demonstration profile, not a production
  calibration.

The provisional factory body is conceptually:

```python
def ds20k_veto() -> ReadoutConfig:
    """Return a fresh provisional DS20k Veto demonstration profile."""

    return ReadoutConfig(
        charge=ChargeConfig(
            dark_count=DarkCountConfig(
                rate=quantity(100.0, "kHz"),
            ),
        ),
        pure_waveform=PureWaveformConfig(
            model=VetoPduPulseConfig(
                gaussian_center=quantity(232.89, "ns"),
                gaussian_width=quantity(507.72, "ns"),
                edge_offset_1=quantity(-81.92, "ns"),
                edge_width_1=quantity(147.28, "ns"),
                edge_offset_2=quantity(-176.50, "ns"),
                edge_width_2=quantity(45.69, "ns"),
                support_time=quantity(2020.27, "ns"),
                peak_voltage_per_photoelectron=quantity(
                    14.5912372,
                    "mV",
                ),
            )
        ),
        noise_waveform=NoiseWaveformConfig(
            model=WhiteNoiseConfig(
                rms=quantity(0.25, "mV"),
            ),
        ),
        analog_waveform=AnalogWaveformConfig(),
        digitized_waveform=DigitizedWaveformConfig(
            bit_depth=PositiveInteger(12),
            input_minimum=quantity(-20.0, "mV"),
            input_maximum=quantity(2.0, "mV"),
            analog_gain_db=NonnegativeFloat(0.0),
        ),
    )
```

The final work order must use the post-Maintenance-8 import topology and
docstring conventions while preserving this returned Config content unless
Design records an explicit later change.

## Provisional Scientific Status

The factory is useful because it is explicit about which values are inherited
parity fixtures and which are demonstration choices:

| Surface | Provisional value | Status |
| --- | --- | --- |
| Veto Gaussian center | `232.89 ns` | audited donor/parity fixture |
| Veto Gaussian width | `507.72 ns` | audited donor/parity fixture |
| Veto first edge offset | `-81.92 ns` | audited donor/parity fixture |
| Veto first edge width | `147.28 ns` | audited donor/parity fixture |
| Veto second edge offset | `-176.50 ns` | audited donor/parity fixture |
| Veto second edge width | `45.69 ns` | audited donor/parity fixture |
| Veto support | `2020.27 ns` | audited 8 ns donor/parity fixture |
| Veto peak magnitude | `14.5912372 mV` | audited donor/parity fixture |
| Dark-count rate | `100 kHz` | illustrative demo choice |
| Timing jitter | disabled | illustrative simplification |
| Correlated avalanches | disabled | illustrative simplification |
| Charge smearing | disabled | illustrative simplification |
| White-noise RMS | `0.25 mV` | illustrative demo choice |
| Analog saturation | disabled | illustrative linear composition |
| ADC bit depth | `12` | illustrative demo choice |
| ADC input interval | `[-20, 2] mV` | illustrative demo choice |
| Analog gain | `0 dB` | illustrative demo choice |

The name `ds20k_veto()` identifies the represented detector/readout family. It
does not make these provisional values an approved run calibration. API docs,
the factory docstring, the demo narrative, and `docs/parity.md` must state that
distinction plainly.

Later promotion or replacement of the provisional values is an intentional
scientific/API change. It must be recorded through Design rather than slipped
in as ordinary maintenance.

## Polarity

The public Veto amplitude is the strictly positive magnitude
`14.5912372 mV`. Current accepted preparation applies the fixed DS20k
negative-going polarity exactly once:

```python
signed_peak_voltage_mv_per_pe = -canonical_magnitude(
    model.peak_voltage_per_photoelectron
)
```

The profile must not store a negative amplitude or expose a polarity switch.
Focused evidence must prove that the manual Config and `ds20k_veto()` both
prepare the same negative represented extremum in `float32` and `float64`,
including the existing represented-zero guard.

## Explicit Photoelectrons Construction

Do not hide the synthetic source behind
`make_synthetic_photoelectrons()` or `_make_synthetic_photoelectrons()`.
The notebook receives one dedicated construction cell, and the script places
the equivalent statements directly in `main()`.

The provisional construction is:

```python
source_generator = torch.Generator(device="cpu")
source_generator.manual_seed(11)

axes = (
    ExampleAxis(count=2),
    ChannelAxis(
        labels=(
            "veto-0",
            "veto-1",
            "veto-2",
            "veto-3",
        )
    ),
    SampleAxis.from_period(
        period=quantity(8.0, "ns"),
        count=320,
    ),
)

shape = tuple(axis.size for axis in axes)
draws = torch.randint(
    low=0,
    high=512,
    size=shape,
    dtype=torch.int64,
    generator=source_generator,
)
counts = torch.where(
    draws < 2,
    draws + 1,
    torch.zeros_like(draws),
)

photoelectrons = Photoelectrons(
    tensor=counts,
    axes=axes,
)
```

This yields sparse deterministic counts in `{0, 1, 2}` without mutating
global Torch RNG state. Seed `11` owns only synthetic input construction.
`Threefry4x32(seed=17)` separately owns the TensorDSLab readout realization.

The chosen `8 ns` sample period matches the audited Veto parity fixture, and
`320` samples provide a `2.56 us` window that can display the approximately
`2020.27 ns` pulse support. The channel labels are illustrative and do not
claim an implemented detector mapping.

The demo should inspect the constructed source through public APIs and show:

- `Photoelectrons.tensor` is exact `torch.int64`;
- the ordered semantic axes are exact
  `(ExampleAxis, ChannelAxis, SampleAxis)`;
- axis objects and sizes match the tensor;
- counts are nonnegative; and
- the input is already binned on the readout sample grid.

## Two Configuration Paths

The notebook uses adjacent cells.

The first spells out the complete `ReadoutConfig` literal shown in the factory
contract and binds it as `manual_config`. The second uses:

```python
from tensor_dslab.readout.profiles import ds20k_veto

profile_config = ds20k_veto()
```

Both are ordinary `ReadoutConfig` values. The notebook should explain the
profile’s provisional status and then use `profile_config` for the main
simulation.

An optional later cell may demonstrate an explicit user-owned experiment:

```python
from dataclasses import replace

experimental_config = replace(
    profile_config,
    noise_waveform=NoiseWaveformConfig(
        model=WhiteNoiseConfig(
            rms=quantity(0.5, "mV"),
        )
    ),
)
```

Do not add `overrides=...`, string paths, dictionaries, mutation, implicit
merging, profile tracking, or a package-owned experiment registry.

The executable script should use `ds20k_veto()` directly so it remains the
concise canonical workflow. It need not repeat the full manual Config literal;
the notebook owns that pedagogical comparison.

## Public Readout Workflow

The full request uses supported root imports and requests all six products:

```python
readout = simulate_readout(
    photoelectrons,
    products=(
        Photoelectrons,
        Charge,
        PureWaveform,
        NoiseWaveform,
        AnalogWaveform,
        DigitizedWaveform,
    ),
    config=profile_config,
    rng=Threefry4x32(seed=17),
    floating_dtype=torch.float32,
)
```

The dark-count and white-noise roles make the readout RNG active. The fixed
package-owned role keys remain private; the demo neither imports nor hardcodes
their namespace or stream values.

The demo retrieves products only through exact typed lookup, inspects semantic
axes, shape, dtype, and device, and explains:

- source `Photoelectrons` uses `torch.int64`;
- Charge and nondigitized waveforms use the requested floating dtype;
- `DigitizedWaveform` uses `torch.int32`;
- generated fields preserve the exact source axes and device; and
- `ReadoutCollection` retains exactly the requested product types.

It also runs a concise selected-product request:

```python
digitized_only = simulate_readout(
    photoelectrons,
    products=(DigitizedWaveform,),
    config=ds20k_veto(),
    rng=Threefry4x32(seed=17),
)
```

The narrative explains that prerequisites execute but only
`DigitizedWaveform` is retained.

## Script And Notebook Boundaries

`demos/readout.py` is an ordinary executable script with `main()` and:

```python
if __name__ == "__main__":
    main()
```

It:

- runs from the repository root;
- defaults to eager CPU;
- uses no input file, network, GPU, environment mutation, or output file;
- prints concise product, shape, dtype, axes, and device information;
- includes small public-contract assertions; and
- imports no private runtime, key, requirement, preparation, production,
  validation, or effect surface.

`demos/readout.ipynb` presents:

1. purpose and explicit provisional-profile status;
2. supported imports;
3. inline random `Photoelectrons` construction;
4. source and axis inspection;
5. manual Config composition;
6. `ds20k_veto()` composition;
7. full readout execution;
8. product and collection inspection;
9. selected-product retention;
10. an optional `dataclasses.replace(...)` experiment; and
11. future FIL/TensorG4DS and TensorML replacement boundaries.

Any nonexistent FIL or TensorML adapter appears only in Markdown or commented
pseudocode. The notebook imports neither package and makes no compatibility
claim.

The first version should omit plotting unless the exact post-Maintenance-8
documentation environment already provides accepted notebook-only Matplotlib
tooling. Matplotlib must not become a runtime dependency merely for the demo.

The committed notebook has no execution counts, outputs, embedded plots,
timestamps, paths, machine identifiers, or binary state. Validation executes a
temporary copy.

## Anticipated Evidence

The final work order should freeze exact post-Maintenance-8 commands and test
paths. At minimum, evidence should prove:

- `ds20k_veto() -> ReadoutConfig` statically and at runtime;
- exact nested Config types, optional membership, values, units, and scalar
  wrappers;
- fresh Config and Quantity identities across factory calls;
- no import-time construction, tensor allocation, RNG activity, registry
  mutation, file, environment, network, or device dependency in the factory;
- exact equivalence between the profile and an independently constructed
  literal Config, without self-comparison;
- identical prepared Runtime facts and exact same-stack products from the
  profile and literal under the same source, dtype, device, and seed;
- positive public Veto amplitude and one preparation-owned negative polarity
  application;
- exact dark-count and white-noise role use with no public key exposure;
- fixed-seed same-stack repeatability without promoting cross-platform
  stochastic literals;
- successful repository-root execution of `demos/readout.py`;
- successful temporary-copy execution of the notebook;
- public imports only;
- expected product membership, shapes, dtypes, axes, and device;
- exact selected-product retention;
- no filesystem or network output;
- cleared committed notebook state;
- documentation links and repository hygiene; and
- unchanged production science outside the new profile values and demo.

Because `ds20k_veto()` is a supported public scientific factory and the demos
are executable code, the promoted stage requires the normal
Design/Implementation/Validation/Review loop even though the values are
provisional and much of the scope is documentation-oriented.

## Explicit Non-Goals

- production calibration or run-conditions claim;
- `ds20k_tpc`;
- canonical Veto channel mapping;
- Config ABC;
- generic profile root or registry;
- loader, parser, serialized profile, YAML, JSON, TOML, or pickle;
- profile constant or singleton;
- user override engine;
- FIL parsing or TensorG4DS adaptation;
- TensorML adapter or input-contract decision;
- IO, artifact, persistence, cache, or deployment policy;
- device, dtype, seed, product, or retention policy inside the profile;
- new readout effect, equation, RNG role, key, address, or schedule;
- CUDA, performance, release, deployment, or broad compatibility claim; and
- Maintenance 8 scope expansion.

## Open Items Before Promotion

The post-Maintenance-8 Design pass must freeze:

- exact TensorDSLab and TensorCore baseline commits and trees;
- target package version and metadata;
- post-Maintenance-8 imports, exports, type syntax, and docstrings;
- exact profile and demo production/test/documentation allowlists;
- exact notebook execution tooling as development/documentation-only input;
- whether plotting remains excluded;
- exact local CPU, typing, wheel/archive, and documentation matrices;
- exact relation to the deferred integrated-CUDA and first-push gates;
- finite Implementation/Validation correction budgets; and
- the final stable work-order key and lifecycle wording.

Until those items are fixed and the user dispatches the resulting work order,
this document remains planning evidence only.
