# Maintenance 17 Application-Neutral Readout Quickstart

Status: **Self-effecting under the frozen exact-byte route**.

Before the complete same-byte package gate and clean local-`main`
fast-forward, the applicable state is the latest completed exact-byte handoff
under the route below. After the exact Validation-cleared, Review-cleared, and
Design-approved candidate appears unchanged on local `main` through Review's
clean fast-forward, Maintenance 17 is **Merged / Closed**.

Stable key:
`TensorDSLab/maintenance-17-application-neutral-readout-quickstart`

## Purpose

Add one small, runnable notebook that introduces TensorDSLab to a newcomer by
building and executing a complete readout example from the package's public
parts.

Maintenance 15 correctly removed the old package-owned DS20k profile,
`Readout` collection, `simulate_readout()` function, and bundled readout
demos. Collaboration packages now own profiles and workflow orchestration.
That application boundary remains correct.

The missing piece is a package-neutral quickstart. A collaborator should be
able to open one notebook and see:

- how TensorDSLab axes describe one example, three sensors, and physical time;
- how a Product Spec describes the tensor that will be produced;
- how a Kernel carries a configurable coefficient and its literal geometry;
- how a Config combines one output Spec with the Kernels for one Product;
- how each Product's `create()` classmethod performs the complete
  prepare-produce-validate path;
- how three sensor channels are processed together in one tensor operation;
- how a hand-supplied power spectral density produces realistic electronic
  noise; and
- how the separate Products may be composed into one useful workflow without
  making that workflow part of the TensorDSLab API.

The notebook is a friendly demonstration, not a developer design document.
Every code cell must be introduced by simple prose that tells a new user what
the next cell does and why it is present. It must not narrate the Design
process, mention maintenance history, explain internal Runtime mechanics, or
sound like a handoff between engineering roles.

## Governing Sources

Implementation, Validation, and Review must read:

- [AGENTS](../../AGENTS.md);
- [CONTRIBUTING](../../CONTRIBUTING.md);
- [Overview](../overview.md);
- [Design](../design.md);
- [Tensor architecture](../architecture/tensors.md);
- [Validation](../validation.md);
- [Maintenance 15 executable work order](maintenance_15_execution_work_order.md);
- [Maintenance 15 architecture record](maintenance_15_spec_composed_products_and_application_boundary.md);
- [Maintenance 16](maintenance_16_declarative_requirements_and_kernel_ownership.md).

The current package sources and this work order take precedence over
historical notebook bytes. Historical Maintenance 9 and 10 demos may be
consulted for presentation lessons and for the exact recognizable
pulse-template/time-grid numbers selected below. Their DS20k profile,
`simulate_readout()`, `ReadoutCollection`, old axes, and former Config shape
are retired and must not return.

The selected pulse numbers remain illustrative package-demo inputs. Reusing
them does not restore the former profile, claim calibration, promote donor
behavior, or select a new parity boundary. `docs/parity.md` remains unchanged.

## Exact Baseline

Maintenance 17 starts from exact locally closed Maintenance 16:

```text
local main / Maintenance 16 Candidate 2:
    1b3122084e296c1162e9f54a2ddb4d984e0c35eb
tree:
    2755f122a8f1b2d5a2722db3b7df8a4e778cc26d
exact parent / amended Maintenance 16 authority:
    ab24fa57e71bb4e2ab3e02c4ccc0055003a8fd77
package version:
    0.2.0
```

Maintenance 16 accepted:

```text
TensorDSLab source:
    57 tests run / 57 passed / 0 skipped
TensorDSLab extracted archive:
    57 tests run / 57 passed / 0 skipped
Pyright positive source and archive:
    0 errors / 0 warnings / 0 informations
TensorDSLab negative typing source and archive:
    exactly 12 intended errors
mutation matrix:
    30 / 30 killed
package topology:
    65 package files / 64 Python modules
test topology:
    23 Python test/support files
    57 discovered methods
    19 TestCase classes
    3,160 Python lines
public root facade:
    61 names
```

Its exact artifacts were:

```text
wheel:
    tensor_dslab-0.2.0-py3-none-any.whl
wheel size:
    54,815 bytes
wheel SHA-256:
    c703a160a9c1e2e8eaf04101349e4dc86a69e06fdc21cf372d5cf1b3bb5fa9ee
sdist:
    tensor_dslab-0.2.0.tar.gz
sdist size:
    551,784 bytes
sdist SHA-256:
    dd5ed3da99c264d49725c7033bcec11becc0978b479846fb4c8f833991fbc4ca
```

The accepted evidence is eager CPU. CUDA was unavailable and remains
unclaimed.

These figures are baseline evidence, not candidate identities. Maintenance 17
adds one notebook, one focused test module, optional demo metadata, and living
documentation; test totals and sdist bytes will therefore change. Production
package bytes and the public facade must not change.

## Pre-Candidate Environment-Test Amendment

Implementation stopped before committing a candidate when the first complete
source run exposed one exact protected-test contradiction:

```text
selected Maintenance 17 environment contract:
    create_environment.sh installs ${repository_root}[demos]
protected Maintenance 16 test:
    tests/test_environment_script.py requires "[demos]" to be absent
complete stopped source run:
    60 tests run / 59 passed / 1 failed
sole failure:
    EnvironmentScriptTests.test_core_only_smoke_and_exact_dependency
```

The notebook, optional dependency group, and environment contract were already
coherent in focused execution. The failure was caused only by the prior
test freezing the superseded core-only install spelling. No implementation
candidate, Validation return, or loop slot was consumed.

This amended Design authority changes no notebook, dependency, environment, or
application boundary. It adds exactly one existing test path to the
Implementation allowlist:

```text
tests/test_environment_script.py
```

Implementation must rename or rewrite the one environment test so it requires
the exact `[demos]` install spelling while retaining its existing package
version, TensorCore version/commit, and retired-`SampleAxis` assertions. It may
add a direct check for all four exact optional demo dependencies when useful,
but it must not weaken the installed-site-packages, PEP 610, cleanup, or
no-shadowing contracts owned elsewhere.

Every preserved dirty implementation byte remains provisional and uncommitted.
Implementation may fast-forward its branch with those nonoverlapping changes
preserved, add only this bounded test correction, rerun its complete assigned
gate, and then freeze Candidate 1 as the exact direct child of this amended
authority.

## Post-Validation Recognizable-Pulse Amendment

Implementation froze immutable Candidate 1 at exact:

```text
candidate:
    95c630b6f99cc3a7a44588fea537fa4bf0e687be
tree:
    9737ae68ebcc3e512164364a0342e6e30fb6572d
exact parent / first amended authority:
    b9303f858a77104d07ad6dedb54d9624c42276f6
```

Independent Validation cleared those exact bytes, including source/archive
`60/60`, Pyright zero, the exact `20/20` first-candidate mutation matrix,
deterministic artifacts, isolated installed-package execution, one fresh real
Conda workflow, and independent visual inspection. Review began its read-only
audit, but Design placed an immediate hold before Review issued a disposition
or performed any merge after the user requested one recognizable presentation
refinement.

Candidate 1 remains immutable, Validation-cleared, unmerged, and unpushed. It
is superseded as the merge target because the user-owned refinement arrived
before Review issued a disposition. Review's preserved pre-hold observations
below are mandatory replacement corrections even though no final Candidate 1
Review disposition was issued.

Before stopping, Review preserved three independent observations that the
replacement must close:

1. the focused execution proof collected Product units but did not assert
   them, so a process-local `PureWaveform` unit drift from `mV` to `V`
   survived while the plot retained its hard-coded `mV` label;
2. the notebook named three sensors but did not explicitly explain that three
   channels were selected to demonstrate one tensor-wide operation; and
3. strict Pyright reported an error where the test helper `_execute()` was
   annotated as returning `object` and its caller accessed `.cells`.

Candidate 2 must assert the exact units of all six Products, bind every plotted
unit label to the matching asserted Product unit, add one plain newcomer
sentence explaining the three-channel choice, and give `_execute()` its exact
`nbformat.NotebookNode` return type or an equally precise supported type. It
must add no cast, ignore, broad `Any`, or weakened typing proof.

The replacement must preserve the new application-neutral, public
spec-composed Product construction while using the exact pulse-template
settings familiar from the former readout notebook/profile:

```text
time spacing:
    2 ns
time samples:
    5,000
time window:
    10,000 ns
pulse support:
    2,020.27 ns
pulse coefficient count:
    ceil(2,020.27 ns / 2 ns) = 1,011
pulse center shift:
    232.89 ns
Gaussian width:
    507.72 ns
first error-function location / width:
    -81.92 ns / 147.28 ns
second error-function location / width:
    -176.50 ns / 45.69 ns
normalized peak scale:
    -14.5912372 mV / avalanche
```

This is a numerical presentation choice only. The replacement must express
the values directly as a public `PulseResponse`; it must not import or restore
the former profile, `SampleAxis`, readout package, orchestration function, or
old Config surface.

The longer Time axis implies the exact cold-path PSD grid:

```text
RFFT bins:
    5,000 // 2 + 1 = 2,501
frequency spacing:
    1 / (5,000 * 2 ns) = 0.1 MHz
```

The four familiar source deposits at samples `100`, `1300`, `2500`, and
`3700`, with respective magnitudes `1`, `2`, `3`, and `4`, should be
distributed across the three sensor channels so every channel is active and
the final deposit still retains the complete 1,011-coefficient pulse support
inside the window.

The replacement authority is a documentation-only direct child of immutable
Candidate 1. Implementation must freeze Candidate 2 as the direct child of
that replacement authority. Relative to Candidate 1, executable correction
scope is bounded to:

```text
M  demos/readout.ipynb
M  tests/test_readout_demo.py
```

The work-order/index lifecycle records may change only as authorized here.
Every other Candidate 1 byte, including production, metadata, dependency,
environment, and living-document integration bytes, must remain exact.

Because the notebook's executed values and plot change, Candidate 2 requires
renewed focused and complete source/archive execution, deterministic replay,
the exact amended `24`-mutant matrix, typing, visual QA, deterministic artifact
reconciliation, isolated installed-package execution, documentation/privacy
checks, and cleanliness. Candidate 1's fresh real-Conda dependency,
site-packages, PEP 610, and cleanup evidence may carry forward because
`pyproject.toml`, `create_environment.sh`, exact dependency versions, and
package bytes remain identical; the replacement notebook must still execute
outside the checkout in an isolated installed-artifact environment.

## Selected Boundary

Maintenance 17 adds exactly one package-level demonstration:

```text
demos/readout.ipynb
```

It does not add:

- `demos/random.ipynb`;
- a script duplicate of the notebook;
- committed plot images;
- a DS20k, Silex, 3DPi, or other collaboration profile;
- collaboration-specific axes or detector constants;
- a generic `Readout`, `ReadoutConfig`, or `ReadoutCollection`;
- `simulate_readout()`;
- a workflow base class, graph, registry, callback, plugin, or factory;
- a public Config builder;
- a PSD generator;
- Product IO;
- an application package;
- a compatibility alias or forwarding module; or
- a new `tensor_dslab` export.

The notebook composes public Product APIs as ordinary user code. It does not
make the demonstrated sequence a package-owned workflow contract.

`CONTRIBUTING.md` must be updated narrowly. It should continue to prohibit
embedded application profiles, orchestration, generic result collections, and
compatibility surfaces while allowing this exact application-neutral
newcomer notebook. The living maturity record should identify Maintenance 16
as locally closed and Maintenance 17 as the active gate.

## Audience And Voice

The intended reader is a new collaborator who has never used TensorDSLab.

Notebook prose must:

- use short, ordinary sentences;
- explain the user-visible idea before each code cell;
- say what objects are being made rather than which internal owner validates
  them;
- introduce terminology only when it first becomes useful;
- keep a casual, welcoming tone;
- explain why three channels are included;
- explain why the first two Products intentionally match;
- explain why the noise is supplied through a PSD;
- explain that application repositories may wrap the same Product calls in
  their own one-shot workflow; and
- finish with a short statement that the separate Product values can be kept,
  plotted, or passed into another Product as the application requires.

Notebook prose must not:

- mention this Maintenance number;
- mention candidates, Review, Validation, work orders, Design threads, or
  architecture debates;
- call the notebook a technical specification;
- claim detector calibration;
- call the values DS20k, Silex, 3DPi, or realistic detector constants;
- imply GPU execution, parallel speedup, or measured performance;
- explain private RNG addresses or private preparation records;
- discuss retired APIs; or
- use conversational phrases that address the repository owner personally.

The notebook title is:

```markdown
# Readout quickstart
```

The opening paragraph should state plainly that the notebook builds one small
readout example from public TensorDSLab pieces. It should say that the numbers
are illustrative and that application packages usually provide their own
axes, kernels, and workflow wrappers.

## Notebook Structure

The notebook uses a clear numbered progression. The exact Markdown wording may
be edited for naturalness, but the section order and code responsibilities are
fixed:

1. Imports
2. Axes
3. Photoelectrons
4. Charge
5. Pure waveform
6. Noise waveform
7. Analog waveform
8. Digitized waveform
9. Shared shape
10. Product views

Every code cell has a preceding Markdown explanation. A section may contain
multiple code cells when separating construction, execution, and a small
illustrative check makes the notebook easier to read. Code should use blank
lines generously and avoid placing unrelated construction on one line.

The notebook does not begin with a non-executable pseudocode summary. The
first code appears beneath `## 1. Imports`, after one plain-language
introduction.

## Imports

Use only public TensorCore and TensorDSLab surfaces plus the ordinary notebook
presentation dependencies.

The imports should be grouped visibly:

```python
import math

import matplotlib.pyplot as plt
import torch

from tensor_core import (
    CountCoordinates,
    LabelCoordinates,
    NonnegativeInteger,
    OffsetAxis,
    OffsetCoordinates,
    RegularCoordinates,
    Threefry4x32,
)

from tensor_dslab import (
    AnalogGain,
    AnalogGainSpec,
    AnalogWaveform,
    AnalogWaveformConfig,
    AnalogWaveformKernels,
    AnalogWaveformSpec,
    BitDepth,
    BitDepthSpec,
    ChannelAxis,
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    DigitizedWaveformKernels,
    DigitizedWaveformSpec,
    ExampleAxis,
    FrequencyAxis,
    InputMaximum,
    InputMaximumSpec,
    InputMinimum,
    InputMinimumSpec,
    NoiseWaveform,
    NoiseWaveformConfig,
    NoiseWaveformKernels,
    NoiseWaveformSpec,
    Photoelectrons,
    PhotoelectronsSpec,
    PowerSpectralDensity,
    PowerSpectralDensitySpec,
    PulseResponse,
    PulseResponseSpec,
    PureWaveform,
    PureWaveformConfig,
    PureWaveformKernels,
    PureWaveformSpec,
    TimeAxis,
    unit_registry,
)
```

No private module import, wildcard import, application import, NumPy import,
IPython-only state mutation, environment creation, package installation,
filesystem write, network access, or dynamic source-path insertion is
permitted.

The notebook may select one fixed Matplotlib style from Matplotlib itself.
It must not depend on a local style file.

## Axes

The example is eager CPU with:

```text
examples:
    1
channels:
    3
channel labels:
    sensor-0, sensor-1, sensor-2
time samples:
    5,000
time spacing:
    2 ns
window:
    10,000 ns
RFFT frequency bins:
    2,501
frequency spacing:
    0.1 MHz
```

Construct the axes explicitly and separately:

```python
device = torch.device("cpu")
field_dtype = torch.float32

example_axis = ExampleAxis(
    coordinates=CountCoordinates(count=1),
)

channel_axis = ChannelAxis(
    coordinates=LabelCoordinates(
        labels=("sensor-0", "sensor-1", "sensor-2"),
    ),
)

time_axis = TimeAxis(
    coordinates=RegularCoordinates(
        start=0,
        step=1,
        count=5000,
    ),
    coordinate_scale=2.0,
    unit=unit_registry.Unit("ns"),
)

frequency_axis = FrequencyAxis(
    coordinates=RegularCoordinates(
        start=0,
        step=1,
        count=2501,
    ),
    coordinate_scale=0.1,
    unit=unit_registry.Unit("MHz"),
)

axes = (
    example_axis,
    channel_axis,
    time_axis,
)
```

The accompanying prose should explain:

- an Example axis permits batching;
- a Channel axis names the three sensors;
- three channels are used so a newcomer can see that the same Product call
  processes multiple sensors together in one tensor;
- a Time axis gives the last tensor dimension its physical spacing;
- all Products below use the same ordered domain; and
- the Frequency axis describes the prepared PSD bins but is not itself a
  Product dimension.

The physical frequency relationship is exact for the displayed binary64
values:

```text
2501 == 5000 // 2 + 1
0.1 MHz == 1 / (5000 * 2 ns)
```

The notebook must not calculate or infer this grid in Product execution.
Constructing it explicitly in the notebook illustrates the cold-path
"punchcard" responsibility.

## Photoelectrons

Construct `PhotoelectronsSpec` explicitly with:

```text
axes:
    axes
device:
    cpu
dtype:
    torch.int64
unit:
    avalanche
```

Construct one tensor of shape `(1, 3, 5000)` directly in the notebook. Use
`torch.zeros(...)` followed by a small number of explicit indexed deposits so
the source is readable. Include at least one nonzero deposit in each channel
and use different locations and integer magnitudes across channels.

Use the familiar four deposit indices and magnitudes from the former
quickstart, distributed across the three sensors:

```text
sensor-0:
    1 avalanche at sample 100
    4 avalanches at sample 3700
sensor-1:
    2 avalanches at sample 1300
sensor-2:
    3 avalanches at sample 2500
```

The exact selected indices must leave the complete pulse support visible
inside the finite time window. No random source generation is used.

Construct:

```python
photoelectrons = Photoelectrons(
    tensor=photoelectron_values,
    spec=photoelectrons_spec,
)
```

The prose should explain that this Product represents an already-produced
source and that real application packages may obtain it from simulation,
measurement, or another transformation.

## Charge

This demo intentionally selects no Charge mechanisms:

```python
charge_spec = ChargeSpec(
    axes=axes,
    device=device,
    dtype=field_dtype,
    unit=unit_registry.Unit("avalanche"),
)

charge_config = ChargeConfig(
    spec=charge_spec,
    kernels=ChargeKernels(members=()),
    correlated_avalanche_generations=NonnegativeInteger(value=0),
)

rng = Threefry4x32(seed=2026)

charge = Charge.create(
    sources=(photoelectrons,),
    config=charge_config,
    rng=rng,
)
```

The prose must say that an empty `ChargeKernels` collection leaves the source
counts unchanged apart from the requested floating representation. This is a
deliberate minimal starting point; users may later add timing jitter,
crosstalk, afterpulse, dark-count, or smearing Kernels.

Do not claim that Charge and Photoelectrons are universally identical.

## Pure Waveform

Construct one global `PulseResponse` with:

```text
conditioning axes:
    ()
operation axes:
    one OffsetAxis relative_to TimeAxis
offset coordinates:
    a contiguous causal support beginning at 0
dtype:
    field_dtype
unit:
    mV / avalanche
```

Use the exact recognizable pulse settings frozen by the amendment. Define the
support and transparent tensor arithmetic directly in the notebook:

```python
pulse_support_ns = 2020.27
pulse_coefficient_count = math.ceil(
    pulse_support_ns / time_axis.coordinate_scale
)
pulse_offsets = torch.arange(
    pulse_coefficient_count,
    dtype=field_dtype,
    device=device,
)
pulse_time_ns = pulse_offsets * time_axis.coordinate_scale
pulse_x = pulse_time_ns - 232.89

pulse_gaussian = torch.exp(
    -(pulse_x**2) / (2.0 * 507.72**2)
) / math.sqrt(2.0 * math.pi * 507.72**2)
pulse_first = 1.0 + torch.erf(
    (pulse_x - (-81.92)) / (math.sqrt(2.0) * 147.28)
)
pulse_second = 1.0 + torch.erf(
    (pulse_x - (-176.50)) / (math.sqrt(2.0) * 45.69)
)

pulse_raw = pulse_gaussian * pulse_first * pulse_second
pulse_values = (
    pulse_raw / torch.max(torch.abs(pulse_raw)) * -14.5912372
)
```

The operation `OffsetAxis` uses exact offsets `tuple(range(1011))`. The
coefficients must be finite, signed, nonzero, and stored at `field_dtype`.
The numerical template is illustrative and recognizable; it is not a
calibration claim. Do not hide its construction in a helper function, profile,
or imported application factory.

Then construct:

```python
pulse_response = PulseResponse(...)

pure_waveform_config = PureWaveformConfig(
    spec=PureWaveformSpec(
        axes=axes,
        device=device,
        dtype=field_dtype,
        unit=unit_registry.Unit("mV"),
    ),
    kernels=PureWaveformKernels(
        members=(pulse_response,),
    ),
)

pure_waveform = PureWaveform.create(
    sources=(charge,),
    config=pure_waveform_config,
)
```

The prose should explain that the PulseResponse maps avalanches into voltage
samples and that its empty conditioning geometry makes one response shared by
all three sensors. It should also say plainly that the selected numerical
template is familiar from earlier TensorDSLab examples but remains
illustrative.

## Noise Waveform

Use a caller-prepared `PowerSpectralDensity`, not `WhiteNoiseRms`.

The PSD must be literal notebook state. Do not add or call a PSD-generator
function. Define three named rows:

```python
psd_sensor_0 = torch.tensor(...)
psd_sensor_1 = torch.tensor(...)
psd_sensor_2 = torch.tensor(...)
```

Each row must:

- have exactly 2,501 values;
- use `field_dtype`;
- have exactly zero at the DC bin;
- be finite and nonnegative;
- have positive non-DC power;
- use simple, reviewable values;
- differ visibly from the other rows; and
- avoid pretending to be calibrated detector data.

Compact literal repetition is allowed when it remains transparent, for
example concatenating explicitly sized constant bands. A helper function,
formula hidden behind a callback, downloaded dataset, or random PSD is not
allowed.

Expand Candidate 1's same illustrative band values to the longer grid:

```text
sensor-0:
    1 zero DC bin + 625 bins at 0.012 + 1,875 bins at 0.004
sensor-1:
    1 zero DC bin + 938 bins at 0.008 + 1,562 bins at 0.016
sensor-2:
    1 zero DC bin + 1,250 bins at 0.020 + 1,250 bins at 0.006
```

Each row therefore has exactly `2,501` values without introducing a generator
or hidden interpolation rule.

Combine the rows explicitly:

```python
psd_values = torch.stack(
    (
        psd_sensor_0,
        psd_sensor_1,
        psd_sensor_2,
    ),
)
```

The PSD Spec is:

```python
psd_spec = PowerSpectralDensitySpec(
    conditioning_axes=(channel_axis,),
    operation_axes=(frequency_axis,),
    device=device,
    dtype=field_dtype,
    unit=unit_registry.Unit("mV ** 2"),
)
```

The `PowerSpectralDensity` tensor shape is `(3, 2501)`. The Channel axis is
conditioning geometry: each sensor receives its own row. The Frequency axis
is operation geometry: each row describes the one-sided frequency bins used
to draw the time-domain noise.

Construct and execute:

```python
power_spectral_density = PowerSpectralDensity(
    tensor=psd_values,
    spec=psd_spec,
)

noise_waveform_config = NoiseWaveformConfig(
    spec=NoiseWaveformSpec(
        axes=axes,
        device=device,
        dtype=field_dtype,
        unit=unit_registry.Unit("mV"),
    ),
    kernels=NoiseWaveformKernels(
        members=(power_spectral_density,),
    ),
)

noise_waveform = NoiseWaveform.create(
    sources=(),
    config=noise_waveform_config,
    rng=rng,
)
```

The prose should explain that `NoiseWaveform` has no source Product in this
example. Its Config and RNG are enough. The public preparation path verifies
that the frequency count and spacing match the output Time axis before any
noise is drawn.

Reuse of one immutable `Threefry4x32` object is intentional because the
Products use separate package-owned RNG roles. The notebook need not explain
the private role details.

## Analog Waveform

Use the smallest composition Config:

```python
analog_waveform_config = AnalogWaveformConfig(
    spec=AnalogWaveformSpec(
        axes=axes,
        device=device,
        dtype=field_dtype,
        unit=unit_registry.Unit("mV"),
    ),
    kernels=AnalogWaveformKernels(members=()),
)

analog_waveform = AnalogWaveform.create(
    sources=(
        pure_waveform,
        noise_waveform,
    ),
    config=analog_waveform_config,
)
```

The prose should explain that AnalogWaveform combines its source Products.
Empty saturation Kernels keep the example focused on composition. Do not add
the redundant equality assertion comparing AnalogWaveform to its two inputs.

## Digitized Waveform

Construct four global coefficient Kernels explicitly:

```text
BitDepth:
    scalar exact signed-integer tensor, value 12, dtype torch.int16
InputMinimum:
    scalar floating tensor, value -80, unit mV
InputMaximum:
    scalar floating tensor, value 20, unit mV
AnalogGain:
    scalar positive floating tensor, value 1, dimensionless
```

The wider illustrative interval keeps the restored pulse template visible
without rail clipping in this fixed example. The exact values must be stated
in prose as demo choices, not calibration. `AnalogGain` is linear, not
decibels.

All four Kernel Specs use:

```text
conditioning_axes:
    ()
operation_axes:
    ()
device:
    cpu
```

Construct:

```python
digitized_waveform_config = DigitizedWaveformConfig(
    spec=DigitizedWaveformSpec(
        axes=axes,
        device=device,
        dtype=torch.int32,
        unit=unit_registry.Unit(""),
    ),
    kernels=DigitizedWaveformKernels(
        members=(
            bit_depth,
            input_minimum,
            input_maximum,
            analog_gain,
        ),
    ),
)

digitized_waveform = DigitizedWaveform.create(
    sources=(analog_waveform,),
    config=digitized_waveform_config,
)
```

The prose should explain that the scalar Kernels apply globally here, while a
collaboration may condition the same coefficient types on Channel or Example
axes when its hardware varies.

## Shared Shape

Use one short assertion cell for one pedagogical point:

```python
expected_shape = tuple(axis.size for axis in axes)

assert photoelectrons.tensor.shape == expected_shape
assert charge.tensor.shape == expected_shape
assert pure_waveform.tensor.shape == expected_shape
assert noise_waveform.tensor.shape == expected_shape
assert analog_waveform.tensor.shape == expected_shape
assert digitized_waveform.tensor.shape == expected_shape
```

The preceding prose should explain that every Product in this example uses
the same `(example, channel, time)` domain even though each Product represents
a different physical quantity or transformation.

Do not add a large assertion inventory. Product construction already performs
its semantic validation.

The notebook test, rather than the newcomer assertion cell, must lock this
exact Product-unit/plot-label mapping:

```text
Photoelectrons:
    avalanche / "Photoelectrons"
Charge:
    avalanche / "Charge (avalanche)"
PureWaveform:
    mV / "Pure (mV)"
NoiseWaveform:
    mV / "Noise (mV)"
AnalogWaveform:
    mV / "Analog (mV)"
DigitizedWaveform:
    dimensionless / "ADC code"
```

## Product Views

Create one vertically stacked Matplotlib figure with six panels:

1. Photoelectrons
2. Charge
3. PureWaveform
4. NoiseWaveform
5. AnalogWaveform
6. DigitizedWaveform

Plot all three sensors in every panel with a stable color mapping:

```text
sensor-0:
    tab:blue
sensor-1:
    tab:orange
sensor-2:
    tab:green
```

The figure contract is:

- one Example is plotted;
- all panels share the physical time x-axis;
- x values are copied to ordinary CPU presentation values only at the plotting
  boundary;
- each Product tensor is copied to ordinary CPU presentation values only at
  that same boundary;
- Photoelectrons and Charge use `step(..., where="post")`;
- PureWaveform, NoiseWaveform, and AnalogWaveform use line plots;
- DigitizedWaveform uses `step(..., where="post")`;
- each Product panel has a plain y label with its represented unit or code
  meaning;
- only the last panel has the `Time (ns)` x label;
- one figure-level legend identifies all three sensors;
- panels do not repeat legends;
- no black Charge-spike overlay is repeated over later Products;
- no misleading extra legend entry is created;
- the title or caption says these are illustrative values;
- layout is readable at normal notebook width; and
- `plt.show()` produces the notebook's only required display figure.

The Markdown before the plot should explain simply:

- the first two panels match because no Charge mechanisms were enabled;
- the pulse panel shows the deterministic sensor response;
- the noise panel shows the PSD-driven electronic component;
- the analog panel combines the two waveform sources; and
- the final panel shows integer ADC codes.

Do not describe CPU-list conversion as an internal architecture triumph. One
short sentence that plotting libraries receive ordinary CPU values is enough.

## Notebook Metadata And Determinism

The notebook must be a clean `nbformat` 4 document with:

- a Python 3 kernelspec;
- language metadata for Python;
- no absolute interpreter path;
- no environment name;
- no project-root path;
- no execution timestamps;
- no widgets metadata;
- no attachments;
- no hidden source;
- no trusted-path assumption;
- cleared committed outputs and execution counts; and
- stable JSON serialization.

The committed notebook is source-only. Validation executes fresh copies.

The notebook uses:

```python
rng = Threefry4x32(seed=2026)
```

The exact seed is illustrative and stable. Two immediate executions on the
same accepted numerical stack must produce identical Product tensors and an
equivalent figure structure. Validation need not require byte-identical PNG
rendering across unrelated Matplotlib or platform stacks.

## Optional Demo Dependencies

Restore one exact optional dependency group:

```toml
[project.optional-dependencies]
demos = [
    "ipykernel==7.3.0",
    "matplotlib==3.11.1",
    "nbclient==0.11.0",
    "nbformat==5.10.4",
]
```

These packages support the notebook and its committed execution tests. They
are not imported by `tensor_dslab`, do not enter the core dependency list,
and do not change the package version.

`create_environment.sh` must install:

```text
${repository_root}[demos]
```

instead of the core-only repository root. Its existing exact environment,
TensorCore PEP 610, site-packages, no-shadowing, cleanup, and smoke contracts
remain. Its smoke may additionally prove the demo dependencies are installed,
but it must not execute the notebook during the ordinary environment creation
path.

## Living Documentation

Update only the minimum living pages needed to make the notebook discoverable
and the boundary truthful:

- `README.md`: add a short Quickstart-notebook link and state that it builds
  an application-neutral example by hand;
- `CONTRIBUTING.md`: replace the blanket demo prohibition with the exact
  package-neutral demonstration boundary and update maturity state;
- `docs/overview.md`: list the notebook as a newcomer entry point without
  restoring a package-owned workflow;
- `docs/design.md`: clarify that application workflow ownership does not
  prohibit a package-neutral composition demonstration;
- `docs/validation.md`: describe notebook validation and isolated execution;
  and
- `AGENTS.md`: synchronize the Maintenance 17 current package rule narrowly
  enough that future agents do not delete the accepted notebook as a
  Maintenance 15 violation.

Do not rewrite historical Maintenance records. Do not restore the old
selected package shape. The M17 current rule supersedes only the blanket
prohibition on any demo, not the application boundary.

## Test Reconciliation

Add one focused test module:

```text
tests/test_readout_demo.py
```

It should be cohesive and small. It must not copy the notebook's full
scientific implementation into a second hand-maintained source file.

The test module owns:

1. notebook inventory and metadata;
2. Markdown-before-every-code-cell structure;
3. public-import and retired-surface absence;
4. exact axis/channel/sample/frequency construction;
5. exact six-Product straight-line order;
6. exact light shape-assertion block;
7. PSD row/count/DC/conditioning/operation geometry;
8. no `Readout`, profile, `simulate_readout`, helper PSD generator, filesystem,
   network, installation, or path mutation;
9. clean fresh execution;
10. deterministic immediate replay on the accepted stack;
11. exact Product types/shapes/devices/dtypes/units, including assertions that
    bind all plotted unit labels to the corresponding Product Specs;
12. source tensor immutability;
13. three-sensor distinction;
14. six-panel plot structure, plot styles, stable colors, one figure legend,
    and no repeated per-axis legends;
15. absence of the rejected Analog equality assertion;
16. absence of `random.ipynb`; and
17. source-only committed notebook outputs.

Use notebook parsing and execution through `nbformat` and `nbclient`.
Inspection of cells and the resulting namespace may use narrow test-owned
instrumentation. Do not create a production notebook runner.

The test may execute the notebook once per focused run and reuse the result
inside one test method or a test-owned fixture. Avoid multiplying slow kernel
launches across many near-identical tests.

The complete repository test suite should remain navigable and under the
existing Maintenance 15 ceilings:

```text
tracked test/support Python files:
    <= 150
discovered unittest methods:
    <= 150
physical Python test lines:
    <= 6,000
```

Do not freeze an exact repository-wide module count as a permanent
architecture invariant.

Update `tests/test_environment_script.py` only to replace its obsolete
core-only installation assertion with the selected exact demo-extra
installation assertion. Preserve the remaining exact dependency and retired
surface checks.

## Required Mutants

Committed evidence must kill these private process-local mutants or exact
equivalents:

1. replace three channel labels with one;
2. use a Time-axis count other than `5000`;
3. use a frequency bin count other than `N // 2 + 1`;
4. use a frequency spacing other than `1 / (N * dt)`;
5. make one PSD DC value nonzero;
6. make all three PSD rows identical;
7. remove Channel conditioning from the PSD;
8. replace the PSD branch with WhiteNoiseRms;
9. restore a DS20k/application profile import;
10. restore `simulate_readout` or a `Readout` collection;
11. omit the Markdown explanation immediately before one code cell;
12. omit one Product from the six-Product shape proof;
13. add the rejected Analog equality assertion;
14. collapse the plot to one sensor;
15. change one sensor's color between panels;
16. add a legend to every panel;
17. add black Charge markers to later waveform panels;
18. commit executed notebook outputs;
19. add `random.ipynb`;
20. execute the notebook with checkout shadowing during the isolated artifact
    gate;
21. change any selected historical pulse constant or replace the
    Gaussian-double-error-function equation family; and
22. truncate the pulse support from the exact `1011` causal coefficients;
23. change one Product unit while retaining its plotted unit label; and
24. remove the newcomer explanation for why three channels are used.

Each mutant must fail a named committed proof for the intended reason.
Validation reports the exact kill matrix. Mutation bytes remain private and
must not modify the immutable candidate.

## Exact Scope

Implementation may change only:

```text
A  demos/readout.ipynb
A  tests/test_readout_demo.py
M  tests/test_environment_script.py
M  pyproject.toml
M  create_environment.sh
M  README.md
M  AGENTS.md
M  CONTRIBUTING.md
M  docs/overview.md
M  docs/design.md
M  docs/validation.md
M  docs/implementation/index.md
M  docs/implementation/maintenance_17_application_neutral_readout_quickstart.md
```

The implementation candidate is the direct child of the exact Design
authority containing this work order. The work-order and index lifecycle bytes
may be changed only as expressly authorized by the self-effecting status rule;
all substantive contract bytes remain Design-owned.

Protected and byte-identical:

```text
tensor_dslab/**
tests/** except tests/test_readout_demo.py and tests/test_environment_script.py
docs/parity.md
docs/architecture/**
docs/decisions.md
docs/governance/**
docs/implementation/** except this work order and index
LICENSE
.github/**
```

No rename outside the allowlist is permitted. No generated image, executed
notebook copy, build product, environment, bytecode, cache, coverage file, or
editor metadata may be committed.

If a public API or production-code change appears necessary, stop and return
the contradiction to Design.

## Implementation Evidence

Implementation reports on one fixed commit:

- exact commit, tree, parent, branch, and cleanliness;
- exact changed-path inventory and line counts;
- notebook cell inventory and metadata;
- focused notebook tests;
- complete source-form suite;
- positive Pyright;
- exact negative typing diagnostics;
- notebook execution in the development environment;
- relative-link and fence checks for changed Markdown;
- `bash -n create_environment.sh`;
- `git diff --check` and `git show --check`;
- public-facade equality to the baseline;
- production-package byte equality to the baseline;
- protected-path equality;
- no private or application import;
- artifact/cache/bytecode hygiene; and
- truthful CPU-only qualification.

Implementation does not build the final artifacts, create the normative fresh
Conda environment, contact Review, merge, or push.

The fixed implementation bytes containing this record were exercised with
CPython `3.14.6`, PyTorch `2.13.0`, NumPy `2.5.1`, Pint `0.25.3`, and exact
TensorCore `0.22.0` source. Focused notebook evidence passes `3/3`; the
complete source-form suite passes `60/60`; positive Pyright reports zero
diagnostics; and the unchanged negative fixture reports exactly `12` intended
errors. The source-only notebook has `20` cells (`10` Markdown and `10` code),
two immediate CPU executions agree, and visual inspection confirms one
readable six-panel figure with stable sensor colors, visible pulses and noise,
non-railed ADC codes, and no title/legend overlap. The environment script
passes syntax and exact demo-extra/dependency checks. Scope, public-facade,
production-byte, link/fence, privacy, diff, and artifact-hygiene evidence is
reported with the fixed commit handoff. These facts are Implementation
evidence only; they do not claim independent Validation, Review, merge,
artifact, fresh-environment, or CUDA clearance.

## Validation Gate

Validation independently checks the exact immutable candidate in both exact
TensorCore source and canonical extracted-archive dependency forms.

The complete gate includes:

1. exact commit/tree/parent/ref and clean fixed-candidate identity;
2. exact allowlist and protected-byte equality;
3. complete notebook inventory, metadata, source-only state, and prose/code
   alternation;
4. complete public-import and application-isolation audit;
5. focused notebook source/archive execution;
6. two immediate deterministic notebook executions on one accepted stack;
7. exact six Product values, types, shapes, devices, dtypes, and units;
8. exact PSD/time/frequency relationship and three distinct channel rows;
9. source immutability;
10. visual inspection of the rendered six-panel figure;
11. programmatic plot-structure checks;
12. all 24 required mutants killed;
13. complete TensorDSLab source/archive suites;
14. Pyright `1.1.411` positive source/archive with zero diagnostics;
15. exact unchanged negative-fixture diagnostic count in both forms;
16. exact TensorCore `0.22.0` identity;
17. two independent deterministic wheel builds;
18. two independent deterministic sdist builds;
19. wheel/sdist/source package equality;
20. sdist notebook/test/metadata equality;
21. isolated installation of the exact TensorCore and TensorDSLab wheels with
    the exact demo dependencies;
22. isolated notebook execution outside the checkout with imports resolving
    from site-packages;
23. one fresh real Conda environment through exact
    `create_environment.sh`, including cleanup;
24. exact package metadata and TensorCore PEP 610 commit;
25. changed-document links, anchors, fences, and code fences;
26. privacy and retired-surface scans;
27. no committed or residual generated artifacts; and
28. final fixed-candidate cleanliness.

The notebook and Matplotlib execution are CPU-only. No CUDA job is required.
Unavailable CUDA does not create a skip because the notebook never requests
CUDA.

Visual inspection should confirm that:

- all six panels are readable;
- the three sensors are distinguishable;
- colors remain stable;
- pulses are visible inside the selected time window;
- noise is visible but does not completely obscure the pulse response;
- ADC codes are not universally pinned at a rail;
- labels and legend do not overlap; and
- the figure does not suggest detector calibration.

If the figure is unreadable, that is a real demo finding even if the notebook
executes.

## Review Gate

Review reads this complete work order and the immutable Validation handoff,
then independently performs a focused risk-based audit:

- exact identity, ancestry, scope, and protected bytes;
- no production or public-facade change;
- no restored application/profile/orchestration surface;
- newcomer prose and Markdown-before-code structure;
- direct public Product construction and exact six-Product sequence;
- three-channel literal source and PSD geometry;
- time/frequency consistency;
- absence of hidden factories and retired APIs;
- exact light assertion boundary;
- figure structure and visual readability;
- isolated installed-package notebook execution or a justified direct reuse
  of exact Validation evidence under evidence economy;
- Pyright and complete-suite evidence reconciliation;
- deterministic artifact identities;
- environment and import isolation;
- living-documentation consistency;
- privacy and repository hygiene; and
- CPU-only qualification.

Review returns findings to Design. Review does not edit the candidate.

After Review reports clear, Design must approve the exact same commit/tree.
Only then may Review run:

```bash
git merge --ff-only <approved-candidate>
```

Review verifies post-merge commit/tree/parent, exact byte identity, diff
checks, branch state, and cleanliness. No push is authorized by this work
order.

## Candidate Loop

The ordinary finite loop is:

```text
Implementation candidates:
    at most 3
Validation returns:
    at most 3
Review returns:
    at most 2
```

Every candidate is immutable. A correction is a new direct child of the last
immutable candidate or an exact Design amendment as routed by Design. No
amend, rebase, squash, merge commit, or moving-branch evidence is accepted.

The notebook may require visual tuning. A Validation return for a genuinely
unreadable figure consumes an ordinary return; it is not silently fixed in
Review.

## Completion

Maintenance 17 is complete only when:

- the exact candidate passes complete Validation;
- independent Review reports clear;
- Design approves the exact same commit/tree;
- Review fast-forwards that exact commit unchanged to clean local `main`;
- `demos/readout.ipynb` is present and `demos/random.ipynb` is absent;
- the notebook executes from an isolated installed artifact;
- all six Products and all three sensors are shown clearly;
- application ownership remains outside TensorDSLab;
- production package and public facade bytes remain unchanged; and
- no unresolved finding remains.

The self-effecting lifecycle rule is:

> Before the complete same-byte gate and clean local-main fast-forward, the
> applicable status is the latest completed exact-byte handoff. After the
> exact Validation-cleared, Review-cleared, Design-approved candidate appears
> unchanged on local `main` through Review's clean fast-forward, Maintenance
> 17 is Merged / Closed.

No separate evidence-only closeout commit is required.

## Explicit Non-Goals

Maintenance 17 does not:

- change a scientific equation;
- change any RNG key, address, word, or result law;
- change any Product, Spec, Kernel, Config, or requirement implementation;
- change TensorCore;
- add calibration;
- add a collaboration profile;
- add an application workflow API;
- add IO or persistence;
- add a PSD generator;
- add reconstruction or TensorML integration;
- add CUDA evidence;
- claim performance, release, deployment, compatibility, or production
  readiness;
- publish a package;
- push Git history; or
- authorize later demo or application surfaces by analogy.
