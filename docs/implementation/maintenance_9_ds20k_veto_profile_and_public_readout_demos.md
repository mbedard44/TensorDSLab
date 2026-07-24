# Maintenance 9 DS20k Veto Profile And Public Readout Demos

Status: **Design-complete / User-authorized / Package loop active**.

This is the fixed TensorDSLab Maintenance 9 production work order. It binds
the exact locally closed Maintenance 8 baseline, exact published TensorCore
`0.16.0`, one deliberately provisional package-owned DS20k Veto profile, one
environment-creation script, one readout script, one public notebook with
plots, the complete changed-path allowlist, protected bytes, evidence matrix,
loop budget, and merge authority. The user authorized the package route
through the exact lifecycle authority carried by the dispatch handoff. It
grants no cluster, push, release, deployment, production-calibration, or broad
compatibility authority.

Stable key:

```text
TensorDSLab/maintenance-9-ds20k-veto-profile-and-public-readout-demos
```

Exact starting baseline:

```text
TensorDSLab commit: f213c387c5de0b9f508a233ab43336f5dc5439ea
TensorDSLab tree:   72b338e40427d75ee4a949e8f61e10a36848b5f3
TensorCore commit:  e05324699892a8bcea024375720bfae1ed9569cc
TensorCore tree:    0414a99ac6096035213479e195a0b095d4b1b12e
package version:    0.1.0
Python:             3.14.6
PyTorch:            2.13.0
NumPy:              2.5.1
Pint:               0.25.3
Hatchling:          1.31.0
Pyright:            1.1.411
```

Maintenance 8 is Merged / Closed at this exact baseline. The separately
preserved provisional Design commit
`111c083e237aac4d1eae301faba44512da53b49b` is planning evidence only; it is
not an ancestor, authority, candidate, or implementation input for this work
order.

## Purpose

Add one deliberately provisional package-owned DS20k Veto readout profile and
two executable public demonstrations:

```text
tensor_dslab/
  readout/
    profiles.py
demos/
  create_environment.sh
  readout.py
  readout.ipynb
```

The profile makes a complete representative Veto readout configuration
available without hiding the ordinary compositional Config model. The demos
teach:

1. how to create and activate the ordinary `tensor_dslab` Conda environment
   used for both project work and the public demos;
2. how a completed `Photoelectrons` field is built explicitly from semantic
   axes and an ordinary `torch.int64` tensor;
3. how the complete readout configuration is built explicitly;
4. how the same provisional configuration is obtained from `ds20k_veto()`;
5. how an experimental configuration is derived with
   `dataclasses.replace(...)`;
6. how `simulate_readout(...)` produces and retains typed products;
7. how `PureWaveform + NoiseWaveform = AnalogWaveform` and the analog waveform
   becomes `DigitizedWaveform` for one selected trace; and
8. where future FIL/TensorG4DS input and TensorML output adapters would replace
   the current demonstration boundaries.

This stage is intentionally separate from the closed
[Maintenance 8](maintenance_8_python314_tensorcore_0_16_modernization.md). It
does not reopen that dependency, syntax, validation-import, docstring,
scientific, or lifecycle scope.

## Completed Post-Maintenance-8 Audit

Design re-audited the exact closed baseline before freezing this work order:

- every profile literal maps directly onto the current public Config
  constructor and scalar/quantity boundary;
- `Photoelectrons` remains directly constructible from public TensorCore-backed
  axes and an ordinary `torch.int64` tensor;
- exact typed `ReadoutCollection.field(...)` lookup exposes every plotted
  product without a private Runtime or validator import;
- `SampleAxis` supplies the exact compact start, step, and count needed to
  construct the plotting coordinate without materializing stored labels;
- unsaturated analog production is exactly `torch.add(pure, noise)`;
- digitization remains the configured affine/clamp/truncate map from the
  analog field to `torch.int32` codes;
- the package root remains exact `35` names and need not export the profile;
- adding one precise public `tensor_dslab.readout.profiles` module creates no
  import cycle or facade change; and
- plotting can be isolated to the notebook and optional demonstration tooling,
  without making Matplotlib or Jupyter a TensorDSLab runtime dependency.

The later integrated-CUDA and first-push gate must bind the exact final
TensorCore/TensorDSLab pairing after Maintenance 9 closes locally. This work
order itself makes no accelerator claim.

## Governing Standards

Implementation, Validation, and Review must apply the current
[CONTRIBUTING.md](../../CONTRIBUTING.md) standards for public API design,
typing, exact dependency use, semantic axes and fields, in-memory product
relationships, validation ownership, documentation, and test quality.

The demonstrations consume only the current public in-memory boundaries
defined by the [readout architecture](../architecture/readout.md) and
[tensor architecture](../architecture/tensors.md). They do not create a new
input adapter, artifact, persistence, or integration contract.

The retained Veto pulse values and every illustrative choice must be classified
truthfully in [parity.md](../parity.md). A donor or historical value is
evidence only at its stated comparison boundary; the provisional profile does
not promote an unreviewed calibration or statistical approximation.

## Public Profile Surface

Add one real module:

```python
from tensor_dslab.readout.profiles import ds20k_veto
```

The exact supported precise-module export is:

```python
__all__ = ("ds20k_veto",)
```

Do not re-export the factory from the `tensor_dslab` or
`tensor_dslab.readout` package roots. Do not add `ds20k_tpc`,
`ds20k_veto_v1`, a profile constant, registry, class hierarchy, builder,
override engine, loader, serialized schema, or profiles package.

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

The exact factory content is:

```python
def ds20k_veto() -> ReadoutConfig:
    """Return a fresh provisional DS20k Veto demonstration profile."""

    frequency_left_edges = (
        0.0,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        62.5,
    )
    power_density = (
        4.0e-8,
        7.0e-8,
        6.0e-8,
        3.0e-8,
        7.0e-9,
        1.0e-9,
        2.0e-10,
        5.0e-11,
        0.0,
    )

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
            model=PsdNoiseConfig(
                frequency_left_edges=quantities(
                    frequency_left_edges,
                    "MHz",
                ),
                frequency_stop=quantity(250.0, "MHz"),
                power_density=quantities(
                    power_density,
                    "mV ** 2 / Hz",
                ),
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

Implementation must use the closed Maintenance 8 import topology and
docstring conventions and must preserve this exact returned Config content.
Any value or nested Config change returns to Design.

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
| PSD frequency coverage | `[0, 250] MHz` | illustrative 2 ns demo choice |
| PSD nonzero support | below `62.5 MHz` | IV-DSLab-like demonstration shape |
| Integrated PSD RMS | approximately `0.5 mV` | illustrative demo choice |
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

### Provisional PSD rationale

The demo uses a `2 ns` sample period: a `500 MHz` sample rate and `250 MHz`
Nyquist limit. The PSD therefore covers exactly through `250 MHz`. Its last
left edge is `62.5 MHz` with zero density, so the profile assumes no noise
power beyond the historical spectrum while remaining explicit through
Nyquist.

The piecewise-constant source spectrum integrates to `0.255125 mV**2`, or
approximately `0.5051 mV` RMS before the current zero-DC-bin treatment. The
current `2 ns` / `1280`-sample finite grid retains approximately `0.4973 mV`
RMS in `float32`, so the intended description remains approximately `0.5 mV`.
This is an IV-DSLab-like toy spectrum—strong low-frequency noise and a rapid
rolloff—not a detector calibration or a required measured-realization RMS.

The profile stores both vector fields as canonical array-backed Pint
quantities:

```python
psd_noise = PsdNoiseConfig(
    frequency_left_edges=quantities(
        frequency_left_edges,
        "MHz",
    ),
    frequency_stop=quantity(250.0, "MHz"),
    power_density=quantities(
        power_density,
        "mV ** 2 / Hz",
    ),
)
```

Preparation alone converts those quantities to unit-free execution facts,
integrates source bins over the demo's finite real-FFT bins, and constructs the
Pint-free Runtime. The profile does not own a sampling axis: its `250 MHz`
coverage is selected to match the demo, and ordinary preparation rejects a
future source whose Nyquist frequency exceeds that coverage.

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
        period=quantity(2.0, "ns"),
        count=1280,
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

The chosen `2 ns` sample period gives the PSD an exact `250 MHz` Nyquist
limit. `1280` samples preserve the `2.56 us` demonstration window and can
display the approximately `2020.27 ns` pulse support. The Veto pulse
parameters remain the audited donor/parity fixture; the finer demo sampling
does not promote a new calibration. The channel labels are illustrative and
do not claim an implemented detector mapping.

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

The dark-count and PSD-noise roles make the readout RNG active. The fixed
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

## Waveform Plot Contract

The notebook must include one dedicated plotting cell after the full readout
request. It selects exactly example `0` and channel index `0` from the known
demo axis order and presents one four-panel figure:

1. `PureWaveform` in millivolts;
2. `NoiseWaveform` in millivolts;
3. `AnalogWaveform` in millivolts, overlaid with a dashed independently
   recomposed `PureWaveform + NoiseWaveform`; and
4. `DigitizedWaveform` as a post-step ADC-code trace.

The cell must prove the unsaturated profile relationship before plotting:

```python
import matplotlib.pyplot as plt

pure = readout.field(PureWaveform)
noise = readout.field(NoiseWaveform)
analog = readout.field(AnalogWaveform)
digitized = readout.field(DigitizedWaveform)

sample_axis = photoelectrons.axis(SampleAxis)
sample_indices = torch.arange(sample_axis.size, dtype=torch.float64)
time_ns = (
    sample_axis.start + sample_axis.step * sample_indices
) * 1.0e-3

example_index = 0
channel_index = 0
selection = (example_index, channel_index, slice(None))

pure_trace = pure.tensor[selection].detach().cpu()
noise_trace = noise.tensor[selection].detach().cpu()
analog_trace = analog.tensor[selection].detach().cpu()
digitized_trace = digitized.tensor[selection].detach().cpu()
recomposed_trace = pure_trace + noise_trace

assert torch.equal(analog_trace, recomposed_trace)

figure, plot_axes = plt.subplots(
    4,
    1,
    figsize=(12, 10),
    sharex=True,
)
plot_axes[0].plot(time_ns.numpy(), pure_trace.numpy(), color="tab:blue")
plot_axes[0].set_ylabel("Pure [mV]")

plot_axes[1].plot(time_ns.numpy(), noise_trace.numpy(), color="tab:orange")
plot_axes[1].set_ylabel("Noise [mV]")

plot_axes[2].plot(
    time_ns.numpy(),
    analog_trace.numpy(),
    color="tab:green",
    label="AnalogWaveform",
)
plot_axes[2].plot(
    time_ns.numpy(),
    recomposed_trace.numpy(),
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="Pure + Noise",
)
plot_axes[2].set_ylabel("Analog [mV]")
plot_axes[2].legend()

plot_axes[3].step(
    time_ns.numpy(),
    digitized_trace.numpy(),
    where="post",
    color="tab:red",
)
plot_axes[3].set_ylabel("ADC code")
plot_axes[3].set_xlabel("Time [ns]")

for axis in plot_axes:
    axis.grid(alpha=0.25)

figure.suptitle(
    "PureWaveform + NoiseWaveform = AnalogWaveform → DigitizedWaveform"
)
figure.tight_layout()
plt.show()
plt.close(figure)
```

The exact colors and ordinary Matplotlib layout calls are presentation
details, not scientific contracts. The four product identities, selected
trace, physical/code units, exact unsaturated addition assertion, panel order,
and analog-to-digitized narrative are required.

The explicit `detach().cpu().numpy()` calls are a notebook visualization
boundary. They do not alter TensorDSLab's same-device execution contract or
authorize silent host materialization inside production. The notebook remains
eager CPU by default and saves no figure or other file.

## Demonstration Tooling

Maintenance 9 adds one exact optional package extra:

```toml
[project.optional-dependencies]
demos = [
    "ipykernel==7.3.0",
    "matplotlib==3.11.1",
    "nbclient==0.11.0",
    "nbformat==5.10.4",
]
```

The ordinary TensorDSLab dependency list is otherwise byte-identical.
`tensor_dslab` production imports none of these packages, and a core install
without the `demos` extra remains supported. The optional extra provides the
plotting kernel and the exact programmatic notebook-execution surface; it does
not select a Jupyter frontend.

The fixed direct demonstration-tool artifacts for the accepted macOS arm64 /
CPython 3.14 Design endpoint are:

| Artifact | SHA-256 |
| --- | --- |
| `matplotlib-3.11.1-cp314-cp314-macosx_11_0_arm64.whl` | `ac104be2768ffdd8655db9e71b768cbb45f2b9aa7b450cf1595e8f65d3822319` |
| `ipykernel-7.3.0-py3-none-any.whl` | `897eb64da762549ef610698fca5e9675195ec6ac8ec7f19d81ce1ca20c876057` |
| `nbclient-0.11.0-py3-none-any.whl` | `ef7fa0d59d6e1d41103933d8a445a18d5de860ca6b613b87b8574accdb3c2895` |
| `nbformat-5.10.4-py3-none-any.whl` | `3b48d6c8fbca4b299bf3982ea7db1af21580e4fec269ad087b9e81588891200b` |

Implementation and each evidence role must record the complete resolved
transitive inventory and reject drift. These packages are demonstration and
evidence inputs, not TensorDSLab runtime-science dependencies.

## Conda Demo Environment

Add one executable convenience script:

```text
demos/create_environment.sh
```

The supported invocation is:

```bash
./demos/create_environment.sh [environment-name]
```

The optional positional name exists only so Implementation, Validation, and
Review can construct disjoint role-private evidence environments. The exact
ordinary project-environment default is:

```text
tensor_dslab
```

The script:

- uses a Bash shebang, `set -euo pipefail`, and mode `100755`;
- accepts zero or one positional argument and rejects every other invocation;
- resolves the repository root from its own physical location, so it works
  from any current directory and records no absolute repository path;
- uses `${CONDA_EXE:-conda}` as the Conda executable;
- fails before mutation when Conda is unavailable or the requested named
  environment already exists;
- creates a new isolated environment with exact Python `3.14.6` and `pip`,
  using `--no-default-packages`, `--override-channels`, and only
  `conda-forge`;
- installs the current local repository non-editably with the exact `demos`
  extra by running the new environment's Python through `conda run`;
- uses noninteractive Conda and pip operation and never installs with
  `--user`;
- performs one import/version smoke check inside the completed environment;
- prints the next commands to
  `conda activate <environment-name>`, change to the detected repository root,
  and run `python demos/readout.py`; and
- exits nonzero on any incomplete creation, installation, or smoke check.

The script deliberately does not call `conda activate`: an executed child
process cannot change its parent shell. It does not call `conda init`, source a
shell profile, edit `.condarc` or a shell startup file, register a Jupyter
kernel, remove or overwrite an existing environment, launch a frontend, run
the simulation implicitly, or promise offline operation. Network access is an
explicit setup prerequisite because the exact package dependencies may need to
be fetched.

The default `tensor_dslab` environment is the ordinary project environment,
not a separate demo-only environment. Installing `.[demos]` makes that same
environment capable of running the notebook while leaving the package's core
dependency contract unchanged.

Conda is a bootstrap tool rather than a TensorDSLab dependency. The accepted
Design endpoint uses Conda `26.1.0`; evidence must record the actual Conda
version and complete created-environment inventory. The supported script
contract is its argument/default behavior and resulting environment, not exact
console prose or the incidental Conda package-build solution.

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

The script and notebook require no GPU, CUDA runtime, accelerator driver, or
accelerator-specific package choice. They create the source on the CPU,
retain every generated product on CPU, and do not inspect or select CUDA even
when it is available. The complete public demonstration must pass in a
CPU-only environment with `torch.cuda.is_available() is False`.

`demos/readout.ipynb` presents:

1. an opening Markdown setup section telling the user to run
   `./demos/create_environment.sh`, then `conda activate tensor_dslab`, before
   launching or selecting the notebook kernel;
2. purpose and explicit provisional-profile status;
3. a first executable Python cell that reports the active Python executable,
   Python and Torch versions, and proves an ordinary default tensor is on CPU;
4. supported imports;
5. inline random `Photoelectrons` construction;
6. source and axis inspection;
7. manual Config composition;
8. `ds20k_veto()` composition;
9. full readout execution;
10. product and collection inspection;
11. the required four-panel waveform plot and exact additive assertion;
12. selected-product retention;
13. an optional `dataclasses.replace(...)` experiment; and
14. future FIL/TensorG4DS and TensorML replacement boundaries.

Environment creation and activation are deliberately user actions before
notebook execution. No notebook code cell invokes Conda, creates or mutates an
environment, activates a shell, installs a package, changes interpreter, or
launches another notebook process.

Any nonexistent FIL or TensorML adapter appears only in Markdown or commented
pseudocode. The notebook imports neither package and makes no compatibility
claim.

The committed notebook has no execution counts, outputs, embedded plots,
timestamps, paths, machine identifiers, or binary state. Validation executes a
temporary copy with the exact `demos` extra. That temporary execution must
produce at least one Matplotlib display output; no exact PNG/SVG bytes become
a compatibility or scientific contract.

## Exact Implementation Scope

The candidate may change only:

1. `pyproject.toml` to add the exact optional `demos` extra;
2. one precise public profile module;
3. one executable readout script, one executable environment-creation script,
   and one clean executable notebook;
4. the focused tests and typing fixture required to prove them; and
5. the exact synchronized current records listed below.

### Design-owned allowlist amendment

Implementation stopped before any candidate commit or Validation dispatch at
exact authority `8a8961cfff0d446a83aef3836b384f4f0dfb328b`, with loop accounting
`I->V 0/3` and `V->I 0/3`. The required new production module makes the
existing Maintenance 8 census in
`tests/test_tensorcore_0_16_modernization.py` fail truthfully at its frozen
`59`-module count. That test also owns the existing public-function census,
which enumerates only the three facade exports and therefore cannot prove the
new precise-module-only `ds20k_veto()` function.

The corrected authority adds exactly that existing census test to the
allowlist and requires its precise-module-aware update. The preserved dirty
Implementation worktree contains exactly five changed or untracked authorized
paths:

```text
pyproject.toml
tensor_dslab/readout/profiles.py
demos/readout.py
demos/readout.ipynb
tests/typing/maintenance_9_ds20k_veto_profile_and_public_readout_demos.py
```

Neither allowlisted runtime test has been edited. The authority correction is
documentation-only and disjoint from those five paths. Implementation may
fast-forward its exact authority while preserving them only after the usual
clean-index, exact-parent, disjointness, and post-fast-forward status gates
pass. This procedural correction consumes no candidate or return slot and
changes no profile, demonstration, dependency, scientific, API, or evidence
contract.

### Design-owned Conda-script amendment

At the user's request, Design issued a second procedural pause while
Implementation was still uncommitted at exact corrected authority
`37cbb53477699d7030e3f174258b6de69e87f358`, tree
`12c009f060ab2f367cb89c1680c334688dfe0922`. No command was running.
Candidate 1 had not been committed or dispatched, so loop accounting remained
`I->V 0/3` and `V->I 0/3`.

The preserved exact dirty set had advanced to eight authorized paths:

```text
pyproject.toml
tensor_dslab/readout/profiles.py
demos/readout.py
demos/readout.ipynb
tests/test_package_contracts.py
tests/test_readout_profiles_and_demos.py
tests/test_tensorcore_0_16_modernization.py
tests/typing/maintenance_9_ds20k_veto_profile_and_public_readout_demos.py
```

The focused source/archive suites were each `24/24/0`; the full source/archive
suites were each `224` run / `211` passed / `13` conditional CUDA skips; and
real temporary-copy notebook execution had passed in both forms before the
pause. This evidence is implementation progress, not candidate clearance.

The corrected authority adds only `demos/create_environment.sh` to the
allowlist and freezes its setup/evidence boundary above. The authority
correction is documentation-only and disjoint from the eight preserved paths.
Implementation may fast-forward and resume only after exact-parent,
clean-index, disjointness, unchanged-dirty-set, and post-fast-forward gates
pass. This amendment consumes no candidate or return slot and changes no
profile, notebook, plotting, dependency, production, scientific, API, RNG,
Pint, or CUDA contract.

The user then clarified the notebook teaching boundary while Candidate 1
remained uncommitted: the opening environment instructions are Markdown, the
user creates and activates `tensor_dslab` before notebook execution, and the
first executable cell only verifies the already-active interpreter and CPU
boundary. This refinement changes no path allowlist, dependency, environment
script, product, plot, science, RNG, Pint, or CUDA contract.

### Design-owned notebook-guidance amendment

The notebook clarification reached Implementation while the already-authorized
real `demos/create_environment.sh` evidence command was in its pip-install
phase after successful Conda environment creation. Implementation allowed that
single in-flight invocation to reach a clean successful boundary rather than
leave an indeterminate partial environment, then started no further command.

At the pause, Implementation remained uncommitted at exact authority
`a602eae0e0979c3e0c0ee1034d5c3287ef18572a`, tree
`eb6bf271b1fc4dda90976c83d478f33112a46851`, with a clean index and exactly
nine authorized dirty paths:

```text
pyproject.toml
tensor_dslab/readout/profiles.py
demos/create_environment.sh
demos/readout.py
demos/readout.ipynb
tests/test_package_contracts.py
tests/test_readout_profiles_and_demos.py
tests/test_tensorcore_0_16_modernization.py
tests/typing/maintenance_9_ds20k_veto_profile_and_public_readout_demos.py
```

Candidate accounting remained `I->V 0/3` and `V->I 0/3`. The amended focused
source suite was `26/26/0`; Bash syntax and mode were clear; and one real
role-private environment had been successfully created from outside the
repository using Conda `26.1.0`, exact Python `3.14.6`, a non-editable local
`.[demos]` installation, and the accepted dependency/tool versions. This is
implementation progress, not candidate clearance.

This final documentation-only clarification is disjoint from all nine
preserved paths. Implementation may fast-forward and resume only after
exact-parent, clean-index, disjointness, unchanged-dirty-set, and
post-fast-forward gates pass. It may then update only the already-allowlisted
notebook/test/document consequences, finish the interrupted evidence plan,
clean up its role-private environment after recording its inventory, and
freeze Candidate 1 through the ordinary route. This amendment consumes no
candidate or return slot.

### Exact changed-path allowlist

Metadata:

```text
pyproject.toml
```

Production:

```text
tensor_dslab/readout/profiles.py
```

Demonstrations:

```text
demos/create_environment.sh
demos/readout.py
demos/readout.ipynb
```

Tests and typing:

```text
tests/test_package_contracts.py
tests/test_readout_profiles_and_demos.py
tests/test_tensorcore_0_16_modernization.py
tests/typing/maintenance_9_ds20k_veto_profile_and_public_readout_demos.py
```

Synchronized current records:

```text
AGENTS.md
CONTRIBUTING.md
README.md
docs/api.md
docs/architecture/readout.md
docs/architecture/rebuild.md
docs/architecture/tensors.md
docs/decisions.md
docs/design.md
docs/implementation/index.md
docs/implementation/maintenance_9_ds20k_veto_profile_and_public_readout_demos.md
docs/overview.md
docs/parity.md
docs/validation.md
```

The maximum candidate scope is exactly `23` logical paths: one metadata path,
one production path, three demo paths, four test/typing paths, and fourteen
current-document paths. A candidate need not touch an allowlisted current
record when no truthful change is required. Any changed path outside this list
is a hard stop.

### Protected bytes

The following remain protected:

- `LICENSE`, `tensor_dslab/py.typed`, `tests/__init__.py`, every closed
  historical work order, and every non-allowlisted path;
- the exact `35/5/30` package/common/readout facade tuples and all existing
  public call signatures;
- every existing Config, axis, field, collection, Runtime, preparation,
  production, validation, requirement, key, and effect byte;
- every accepted scientific equation, threshold, RNG namespace/key/address,
  iteration order, dtype/device rule, product relationship, Pint unit, and
  canonicalization contract;
- the exact TensorCore dependency and core runtime dependency list; and
- IO, artifact, TensorG4DS, TensorML, CUDA, performance, deployment, and
  first-push surfaces.

Implementation must not reformat unrelated files, add compatibility aliases,
or use the demo as authority to refactor production. A required change outside
this scope returns the exact contradiction to Design.

## Required Evidence

The candidate must prove:

- `ds20k_veto() -> ReadoutConfig` statically and at runtime;
- exact one-name `tensor_dslab.readout.profiles.__all__`, with no package-root
  or `tensor_dslab.readout` re-export;
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
- exact dark-count and PSD-noise role use with no public key exposure;
- exact `2 ns` sample period, `250 MHz` Nyquist coverage, source PSD table,
  zero power above `62.5 MHz`, source integral `0.255125 mV**2`, and the
  approximately `0.5 mV` prepared RMS scale without a realization-RMS promise;
- fixed-seed same-stack repeatability without promoting cross-platform
  stochastic literals;
- successful repository-root execution of `demos/readout.py`;
- exact CPU-only demonstration execution with every source/result tensor on
  CPU and no GPU, CUDA runtime, driver, device discovery, or accelerator
  selection requirement;
- `bash -n demos/create_environment.sh`, exact executable mode, relative-path
  operation, zero/one-argument behavior, exact `tensor_dslab` default name,
  and rejection of malformed invocation;
- fresh real creation through `demos/create_environment.sh` under a unique
  role-owned name, followed by exact Python/dependency/import/inventory checks;
- successful `conda run` script and notebook execution inside that created
  environment, followed by role-owned cleanup after evidence capture;
- exact non-editable local `.[demos]` installation, no user install, no
  existing-environment mutation, and working printed
  activation/change-directory/run guidance;
- no `conda activate`, `conda init`, shell-profile/Conda-config mutation,
  kernel registration, frontend launch, implicit simulation, absolute
  repository path, or environment removal in the committed script;
- successful `nbclient` execution of a temporary notebook copy under the exact
  `demos` extra;
- exact opening Markdown setup instructions, no executable notebook
  environment-management command, and a first Python cell that reports the
  active interpreter/tool versions and proves the ordinary default tensor is
  on CPU without inspecting or selecting CUDA;
- public imports only;
- expected product membership, shapes, dtypes, axes, and device;
- exact unsaturated tensor equality between `AnalogWaveform` and the
  independently recomposed `PureWaveform + NoiseWaveform`;
- one four-panel figure with the required product/panel order, physical/code
  labels, one selected example/channel trace, and at least one temporary
  Matplotlib display output;
- no plot-file write and no exact image-byte assertion;
- exact selected-product retention;
- no filesystem or network output;
- cleared committed notebook state;
- exact optional-extra metadata and proof that core installation/import does
  not require or import Matplotlib/Jupyter packages;
- exact production-module count `60`, supported public-class count `32`,
  supported public-function count `4`, and own module/symbol docstrings;
- a precise-module-aware update to the existing Maintenance 8 census: it must
  enumerate all `60` production modules, preserve the unchanged facade
  identities, and include `ds20k_veto` through exact
  `tensor_dslab.readout.profiles.__all__`;
- exact supported public-function identities
  `{quantity, quantities, simulate_readout, ds20k_veto}`, with no private,
  imported-but-unexported, alias, or duplicate function admitted;
- unchanged `35/5/30` facade identities and downstream import isolation;
- exact source/wheel/sdist package bytes and isolated wheel behavior;
- full TensorDSLab source/archive suites with no new skip;
- exact TensorCore `0.16.0` source/archive identity and complete dependency
  suite;
- Pyright `1.1.411` with Python `3.14`, zero positive diagnostics, and the
  frozen dependency negative fixture with no incidental diagnostic;
- truthful `docs/parity.md` classification separating retained Veto fixture
  facts from illustrative PSD/dark-count/ADC/demo choices;
- documentation links and repository hygiene; and
- unchanged production science outside the new profile values and demo.

Because `ds20k_veto()` is a supported public scientific factory and the demos
are executable code, this stage requires the normal
Design/Implementation/Validation/Review loop even though the values are
provisional and much of the scope is documentation-oriented.

The accepted pre-candidate baseline is `215` TensorDSLab tests with `13`
conditional unavailable-CUDA skips and `84` TensorCore tests with two such
skips. Candidate totals may increase only through the allowlisted focused
tests. Missing baseline tests, additional skips, silently narrowed discovery,
or a plotting dependency imported by production is a finding.

## Normalized Commands

Each role uses fresh exact TensorCore source and canonical-ZIP forms and fresh
Python `3.14.6` core and `demos` environments. Role-private paths and resolved
transitive inventories must not enter committed records.

```bash
bash -n demos/create_environment.sh

./demos/create_environment.sh <role-private-environment-name>

<conda> run --name <role-private-environment-name> \
  python demos/readout.py

<conda> run --name <role-private-environment-name> \
  python -B -m unittest tests.test_readout_profiles_and_demos -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<TensorDSLab>:<TensorCore-form> \
  <python-3.14.6> -B -m unittest \
  tests.test_readout_profiles_and_demos \
  tests.test_package_contracts -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<TensorDSLab>:<TensorCore-form> \
  <python-3.14.6> -B -m unittest discover -s tests -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<TensorDSLab>:<TensorCore-form> \
  <python-3.14.6> -B demos/readout.py

<pyright-1.1.411> --pythonversion 3.14 --pythonpath <python-3.14.6>

<python-3.14.6> -B -m build --wheel --sdist

<conda> env remove --yes --name <role-private-environment-name>
```

The focused profile/demo test loads the committed notebook with exact
`nbformat`, proves its cleared state, executes a temporary copy with exact
`nbclient` and `ipykernel`, and uses a noninteractive Matplotlib backend while
retaining display-output evidence. Full discovery, Pyright, script execution,
and notebook execution are mandatory against both exact TensorCore forms
before and after the candidate commit.

The artifact gate installs the fresh TensorDSLab wheel first without optional
extras and then with the exact `demos` extra. The core form proves
Matplotlib/Jupyter absence and clean TensorDSLab import. The demonstration form
repeats script/notebook execution with no project-root package shadowing.
The Conda-script environment is a separate user-workflow gate over the local
source tree; it does not replace the isolated wheel/sdist evidence. Editable
installs are not evidence.

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
- Matplotlib, Jupyter, or plotting imports from `tensor_dslab` production;
- automatic parent-shell activation, shell-profile or Conda-configuration
  mutation, environment replacement/removal, or Jupyter-kernel registration;
- Conda lockfile, cross-platform environment solution, offline installer, or
  Conda package publication;
- stored notebook outputs, execution counts, generated figures, or image-byte
  golden files;
- multi-example/channel dashboard, interactivity, widgets, animation, styling
  framework, or saved plot artifact;
- CUDA, performance, release, deployment, or broad compatibility claim; and
- Maintenance 8 scope expansion.

## Role Route And Candidate Budget

The persistent TensorDSLab route is:

```text
Design -> Implementation -> Validation -> Review
```

Implementation produces one immutable direct-child candidate of the exact
Design authority and dispatches it to persistent Validation. Validation
independently reconstructs the core/demo environments and both TensorCore
forms and may return a bounded finding packet to Implementation. The ordinary
budget is:

```text
Implementation -> Validation candidate submissions: 3
Validation -> Implementation returns:              3
```

Validation dispatches an unchanged cleared candidate to persistent Review.
Review independently repeats the functional, typing, profile, notebook,
plotting, artifact, scope, documentation, privacy, and hygiene evidence.
Review alone may fast-forward a cleared candidate to governed local `main`
with `git merge --ff-only`. Review may not push.

An exhausted candidate route, Design-owned documentation contradiction,
dependency/tool discrepancy, failed notebook kernel, unavailable exact input,
unexpected skip, scientific difference, or need for cluster evidence returns
to Design. No supplemental candidate, allocation, or tool substitution is
implicit. Design owns a later evidence-only lifecycle closeout over the exact
fourteen current records only after Review's unchanged fast-forward.

## CUDA And First-Push Sequence

Maintenance 9 closes locally with CPU, typing, profile, script/notebook,
plotting, artifact, documentation, import, and hygiene evidence and makes no
fresh accelerator claim.

After the exact Maintenance 9 closeout:

1. freeze exact published TensorCore `0.16.0` plus exact closed TensorDSLab
   Maintenance 9;
2. issue separate explicit integrated-CUDA authority;
3. run package-owned full-suite CUDA matrices for both packages against that
   same pairing and accepted PyTorch line;
4. record separate package dispositions and environment qualifications;
5. resolve every real finding through the owning package; and
6. only then consider TensorDSLab's first push.

The integrated CUDA work is functional compatibility evidence, not a
performance, deployment, release-readiness, compilation, notebook-rendering,
or broad-backend claim.

## Dispatch Gate

The exact Design authority is the lifecycle-only direct-child commit that
records user authorization while preserving the fixed work-order contract.
Its hash and tree are carried in the immutable dispatch handoff because a Git
commit cannot truthfully contain its own hash. Implementation must prove that
exact authority as its direct parent before editing.

The user authorized the persistent Implementation/Validation/Review route.
The package loop is active until an unchanged Review fast-forward and Design
closeout; this branch/main-neutral status remains true throughout intermediate
candidate states. The dispatch packet must name the exact authority
commit/tree, confirm all execution roles are Active, and state that cluster
work and push remain unauthorized.
