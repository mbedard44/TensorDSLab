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
consulted only for presentation lessons. Their DS20k profile,
`simulate_readout()`, `ReadoutCollection`, old axes, and former Config shape
are retired and must not return.

No donor behavior, parity promotion, or scientific approximation is selected.
`docs/parity.md` remains unchanged.

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
    256
time spacing:
    2 ns
window:
    512 ns
RFFT frequency bins:
    129
frequency spacing:
    1.953125 MHz
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
        count=256,
    ),
    coordinate_scale=2.0,
    unit=unit_registry.Unit("ns"),
)

frequency_axis = FrequencyAxis(
    coordinates=RegularCoordinates(
        start=0,
        step=1,
        count=129,
    ),
    coordinate_scale=1.953125,
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
- a Time axis gives the last tensor dimension its physical spacing;
- all Products below use the same ordered domain; and
- the Frequency axis describes the prepared PSD bins but is not itself a
  Product dimension.

The physical frequency relationship is exact for the displayed binary64
values:

```text
129 == 256 // 2 + 1
1.953125 MHz == 1 / (256 * 2 ns)
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

Construct one tensor of shape `(1, 3, 256)` directly in the notebook. Use
`torch.zeros(...)` followed by a small number of explicit indexed deposits so
the source is readable. Include at least one nonzero deposit in each channel
and use different locations and integer magnitudes across channels.

The deposits should be separated far enough that the pulse shapes can be
seen. A suitable illustrative arrangement is:

```text
sensor-0:
    1 avalanche near sample 32
    2 avalanches near sample 144
sensor-1:
    2 avalanches near sample 64
    1 avalanche near sample 176
sensor-2:
    3 avalanches near sample 96
    1 avalanche near sample 208
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
    torch.float32
unit:
    mV / avalanche
```

Use a support long enough for the pulse shape to be visible on the
`256 x 2 ns` grid. A support in the range 40–64 samples is appropriate.
The coefficients must be finite, signed, nonzero, and illustrative rather
than calibration data. A simple negative pulse with a quick leading edge and
slower recovery is preferred.

The construction may use transparent Torch arithmetic to form the literal
coefficient tensor in the notebook, for example a difference of exponentials
over one explicitly named offset tensor. It must not hide the construction in
a helper function or profile.

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
all three sensors.

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

- have exactly 129 values;
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

The `PowerSpectralDensity` tensor shape is `(3, 129)`. The Channel axis is
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
    scalar floating tensor, unit mV
InputMaximum:
    scalar floating tensor, unit mV
AnalogGain:
    scalar positive floating tensor, dimensionless
```

Use an illustrative input interval wide enough that the plotted Analog
waveforms are not mostly clipped. The exact values must be stated in prose as
demo choices, not calibration. `AnalogGain` is linear, not decibels.

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
11. exact Product types/shapes/devices/dtypes/units;
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
2. reduce the Time axis to 32 samples;
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
19. add `random.ipynb`; and
20. execute the notebook with checkout shadowing during the isolated artifact
    gate.

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
12. all 20 required mutants killed;
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
