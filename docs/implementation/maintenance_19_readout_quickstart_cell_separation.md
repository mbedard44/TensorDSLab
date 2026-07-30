# Maintenance 19 Readout Quickstart Cell Separation

Status: **Active / Design authority prepared**.

Stable key:
`TensorDSLab/maintenance-19-readout-quickstart-cell-separation`

## Purpose

Restructure the application-neutral readout quickstart so that each code cell
has one obvious storytelling role:

```text
mathematical or tensor-value preparation
    lives in its own cell

TensorCore / TensorDSLab semantic construction
    lives in its own cell

assertions
    live in their own cell

presentation preparation
    lives in its own cell

plot rendering
    lives in its own cell
```

The maintenance changes presentation structure only. It does not change any
accepted input, numerical law, Product, public API, plot, or execution result.

## Governing Sources

Implementation, Validation, and Review must read:

- [AGENTS](../../AGENTS.md);
- [CONTRIBUTING](../../CONTRIBUTING.md);
- [Overview](../overview.md);
- [Design](../design.md);
- [Validation](../validation.md);
- [Maintenance 15 architecture](maintenance_15_spec_composed_products_and_application_boundary.md);
- [Maintenance 15 execution work order](maintenance_15_execution_work_order.md);
- [Maintenance 17](maintenance_17_application_neutral_readout_quickstart.md);
- [Maintenance 18](maintenance_18_encoded_waveform_raw_zle.md).

The selected objective is:

> A newcomer should be able to skim the semantic construction cells and see
> the TensorDSLab story—Fields, Kernels, Specs, Configs, and Product
> classmethods—without mathematical derivations or tensor-preparation details
> interrupting that story.

The selected objective is not:

> Hide the numerical preparation or replace the recognizable illustrative
> pulse and PSD with unexplained durable assets.

## Exact Baseline

Maintenance 19 starts from exact locally closed Maintenance 18:

```text
local main:
    b77fe74155d0903a727ead6da451f06b8b3ef652
tree:
    94b3ccc8248b0119ddec682c51c277190ea83c50
package version:
    0.2.0
```

The exact TensorCore dependency remains published `0.22.0`:

```text
commit:
    19bfae35fbc773b55cac7bcd659dda57c4dee6d6
tree:
    53aa10520a50c0714e79c685d814cbae1b6f7740
```

The baseline readout notebook is source-only and has:

```text
22 cells:
    11 Markdown
    11 code
Products:
    Photoelectrons
    Charge
    PureWaveform
    NoiseWaveform
    AnalogWaveform
    DigitizedWaveform
    EncodedWaveform
plot:
    one seven-panel display
```

Maintenance 18 accepted complete source/archive evidence at `70/70/0`,
positive Pyright with zero diagnostics, and an exact negative fixture with
fifteen intended diagnostics. Those figures are baselines, not permission to
weaken or delete a test.

## Selected Notebook Grammar

The notebook must remain strictly alternating Markdown/code. Every code cell
has one immediately preceding plain-language Markdown cell that tells a
newcomer what the next cell does.

The target is exactly `34` cells:

```text
17 Markdown
17 code
```

The exact code-cell sequence and stable IDs are:

```text
imports-code
axes-code
photoelectron-values-code
photoelectrons-code
charge-code
pulse-math-code
pure-waveform-code
psd-values-code
noise-waveform-code
analog-waveform-code
digitizer-values-code
digitized-waveform-code
encoding-values-code
encoded-waveform-code
shared-shape-code
plot-preparation-code
product-views-code
```

The corresponding sections may use casual newcomer-facing titles. They must
retain an obvious order and correct numbering.

## Exact Cell Ownership

### Imports

`imports-code` retains only the ordinary public imports required by the
notebook. `math` remains because the illustrative pulse calculation remains
visible. No private TensorDSLab import, application package, profile,
filesystem, network, NumPy, installation action, or compatibility surface is
introduced.

### Axes

`axes-code` constructs the device, floating Field dtype, ExampleAxis,
ChannelAxis, TimeAxis, FrequencyAxis, and shared ordered Product axes. It is a
semantic-construction cell. It performs no pulse, PSD, digitizer, ZLE, or plot
calculation.

### Photoelectrons

`photoelectron-values-code` prepares only the source tensor values and their
defensive before-image used by the execution proof. It contains no
Photoelectrons Spec or Product construction.

`photoelectrons-code` constructs `PhotoelectronsSpec` and
`Photoelectrons`. It does not place deposits or mutate tensor values.

### Charge

`charge-code` constructs `ChargeSpec`, `ChargeConfig`, the empty
`ChargeKernels`, the seeded public RNG, and `Charge.create(...)`. There is no
separate numerical preparation for this intentionally minimal Product.

### Pulse and PureWaveform

`pulse-math-code` contains the complete existing illustrative
Gaussian/double-error-function calculation:

- `2020.27 ns` support;
- `1011` coefficients on the two-nanosecond grid;
- constants `232.89`, `507.72`, `-81.92`, `147.28`, `-176.50`, and `45.69`;
- exact normalization to the existing `-14.5912372 mV / avalanche` peak; and
- the resulting `pulse_values`.

It contains no `OffsetAxis`, `PulseResponse`, `PulseResponseSpec`,
`PureWaveformConfig`, `PureWaveformKernels`, `PureWaveformSpec`, or
`PureWaveform.create(...)`.

`pure-waveform-code` begins with the semantic pulse operation axis and then
shows the concise story:

```python
pulse_time_axis = OffsetAxis(
    coordinates=OffsetCoordinates(offsets=tuple(range(1011))),
    relative_to=TimeAxis,
)

pulse_response = PulseResponse(
    tensor=pulse_values,
    spec=PulseResponseSpec(
        conditioning_axes=(),
        operation_axes=(pulse_time_axis,),
        device=device,
        dtype=field_dtype,
        unit=unit_registry.Unit("mV / avalanche"),
    ),
)

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

That cell contains no `torch.exp`, `torch.erf`, pulse normalization, or other
pulse derivation.

### PSD and NoiseWaveform

`psd-values-code` contains the existing three channel-conditioned PSD tensor
rows and their stack. It contains no PowerSpectralDensity Spec/Kernel,
NoiseWaveform Config, or Product construction.

`noise-waveform-code` constructs `PowerSpectralDensitySpec`,
`PowerSpectralDensity`, `NoiseWaveformSpec`, `NoiseWaveformKernels`,
`NoiseWaveformConfig`, and `NoiseWaveform.create(...)`. It contains no
`torch.cat`, `torch.full`, or `torch.stack`.

### AnalogWaveform

`analog-waveform-code` retains the concise semantic Config and
`AnalogWaveform.create(...)` call. No new assertion is added.

### DigitizedWaveform

`digitizer-values-code` prepares only the four scalar tensor values:

```text
bit depth
input minimum
input maximum
analog gain
```

`digitized-waveform-code` wraps those values in their exact public Kernel
Specs/Kernels, constructs the Product Spec/Config/Kernels collection, and
calls `DigitizedWaveform.create(...)`. It contains no direct
`torch.tensor(...)` call.

### EncodedWaveform

`encoding-values-code` prepares only the five scalar policy tensors:

```text
trigger threshold
release threshold
required time-over samples
pre-trigger samples
post-trigger samples
```

`encoded-waveform-code` wraps those values in their exact public Kernel
Specs/Kernels, constructs `EncodedWaveformSpec` and
`EncodedWaveformConfig`, and calls `EncodedWaveform.create(...)`. It contains
no direct `torch.tensor(...)` call.

### Assertions

`shared-shape-code` remains the sole assertion cell. It contains exactly the
existing `expected_shape` preparation plus the seven light Product-shape
assertions in their existing order. No scientific equality, recomposition, or
redundant Product-validation assertion is added elsewhere.

### Presentation

`plot-preparation-code` prepares labels, stable sensor colors, CPU time
values, the ordered Product tuple, y-axis labels, and step-panel selection. It
contains no `plt.subplots`, `plot`, `step`, figure legend, display call, or
Product construction.

`product-views-code` owns Matplotlib style, figure creation, the seven-panel
render loop, presentation-only suppression gaps, labels, title, and one
figure-level legend. It produces the notebook's only display.

Every Product tensor crosses to ordinary CPU values only within the
presentation boundary.

## Preserved Numerical And Visual Contract

The restructuring must preserve exactly:

- one Example and three sensor channels;
- `5000` Time bins at `2 ns`;
- `2501` one-sided Frequency bins at `0.1 MHz`;
- source deposits `(sensor, index, value)`:
  `(0,100,1)`, `(0,3700,4)`, `(1,1300,2)`, `(2,2500,3)`;
- the pulse equation, support, coefficient count, values, dtype, device,
  operation offsets, Unit, and illustrative classification;
- the three PSD rows, band counts, dtype, device, axes, Unit, and values;
- the same seeded Charge/noise execution and exact seven Product values;
- the `12`-bit, `[-80,20] mV`, unit-gain digitizer;
- the exact raw-ZLE policy `2500 / 2800 / 3 / 25 / 50`;
- `torch.int32` digitized/encoded values and suppression code `-1`;
- seven `(1,3,5000)` Products;
- the exact retained EncodedWaveform intervals;
- one deterministic seven-panel plot with three stable channel colors;
- blank presentation-only ZLE gaps;
- the existing Product units/y-label relationships; and
- all newcomer-facing non-calibration language.

The notebook must remain source-only:

```text
all execution_count values:
    null
all code-cell outputs:
    empty
attachments:
    absent
```

## Exact Scope

Implementation may change exactly:

```text
demos/readout.ipynb
tests/test_readout_demo.py
docs/implementation/index.md
docs/implementation/maintenance_19_readout_quickstart_cell_separation.md
```

The two documentation paths identify the authority and lifecycle. The
executable candidate should normally change only the notebook and its focused
proof; it must not rewrite Design-owned authority prose.

Every other path is protected, including:

- `tensor_dslab/**`;
- every other `tests/**` path;
- `pyproject.toml`;
- `create_environment.sh`;
- all dependency, environment, package-version, and artifact metadata;
- every other demo;
- all living architecture, parity, API, overview, design, validation, and
  contribution records; and
- every historical implementation record.

No compatibility module, alias, forwarder, public export, data asset, stored
executed notebook, plot image, cache, bytecode, build output, or environment
residue may enter the repository.

## Implementation Route

Implementation uses:

```text
branch:
    codex/maintenance-19-readout-demo-cell-separation
exact parent:
    this Design authority
ordinary candidate ceiling:
    2
Validation return ceiling:
    2
Review return ceiling:
    1
```

Implementation must:

1. bind the exact authority commit/tree/parent and clean branch;
2. modify only the notebook and focused proof;
3. preserve stable unique notebook IDs and public imports;
4. update structural tests to prove the exact seventeen-cell grammar and
   separation boundaries;
5. execute the focused proof;
6. execute the complete source suite and positive/negative typing gates;
7. execute one temporary copy of the notebook outside the repository;
8. save one temporary PNG outside the repository for Design/user inspection;
9. verify the committed notebook remains source-only and byte-unchanged by
   execution;
10. verify diff/scope/privacy/artifact hygiene and branch cleanliness; and
11. commit one immutable direct-child candidate for Design routing.

Implementation must not contact Validation or Review directly.

## Focused Proof

`tests/test_readout_demo.py` must preserve its existing numerical, execution,
plot, public-import, source-immutability, and Product-contract proof while
updating only structural expectations made obsolete by the cell split.

It must additionally prove:

- exact `34 / 17 / 17` source inventory;
- strict Markdown/code alternation;
- the exact ordered stable code-cell IDs;
- the pulse mathematical cell contains the pulse calculation and none of the
  semantic pulse/PureWaveform classes;
- the PureWaveform semantic cell contains the exact public semantic sequence
  and none of the pulse mathematics;
- PSD preparation and NoiseWaveform semantic construction are separated;
- scalar digitizer tensor preparation and digitizer Kernel/Product
  construction are separated;
- scalar ZLE tensor preparation and encoded Kernel/Product construction are
  separated;
- assertions occur only in `shared-shape-code`;
- plot preparation and plot rendering are separated; and
- Product construction order remains exact.

Do not add dynamic source rewriting, broad `Any`, casts, ignores, or generated
test methods merely to satisfy these obligations.

## Validation

Validation receives one immutable Design-dispatched candidate and must run:

1. exact identity/tree/parent/ref/direct-child and two-path executable scope;
2. notebook inventory, stable IDs, source-only metadata, public-import, and
   cell-separation proofs;
3. focused source and exact extracted-archive demo tests;
4. complete source and exact extracted-archive suites;
5. positive Pyright and the unchanged exact fifteen-diagnostic negative
   fixture in source/archive forms;
6. two immediate CPU executions proving deterministic seven-Product and plot
   identity;
7. one independent visual inspection of the rendered PNG;
8. byte equality for every protected production, dependency, metadata,
   environment, and non-demo-test path;
9. documentation links/fences, privacy, artifact/cache/bytecode/build
   hygiene; and
10. final exact detached cleanliness.

Artifact rebuilding and a fresh Conda environment are not required because
package, dependency, metadata, environment, and production bytes are
protected and the notebook runs directly from source/archive. Validation must
not infer an artifact, installation, CUDA, performance, calibration,
compatibility, release, or publication claim.

## Review And Merge

Review remains independent and read-only until:

1. Validation clears one exact immutable candidate;
2. Design dispatches that exact candidate to Review;
3. Review returns CLEAR with zero findings; and
4. Design issues final same-byte approval.

Review then owns only:

```text
git merge --ff-only <exact-approved-candidate>
```

on clean governed local `main`, followed by exact identity, tree, lineage,
diff/check, worktree-cleanliness, and origin-relation verification.

No merge commit, rewrite, amend, rebase, squash, push, tag, publication,
branch deletion, CUDA run, or closeout edit is authorized.

## Completion

Maintenance 19 is complete only when:

- the exact source-only `34`-cell notebook implements the selected visual
  grammar;
- its numerical Products and rendered plot remain unchanged;
- focused and complete Validation clear the same immutable bytes;
- independent Review returns zero findings;
- Design approves those same bytes; and
- Review fast-forwards them unchanged to clean local `main`.

The final report must include a rendered temporary plot for user inspection
without committing notebook outputs or any generated image.
