# Maintenance 20 Product-Local Readout Plots

Status: **Active / Design authority prepared**.

Stable key:
`TensorDSLab/maintenance-20-product-local-readout-plots`

## Purpose

Restructure the application-neutral readout quickstart so that every Product
is plotted immediately after it is created.

The selected newcomer story is:

```text
explain one Product
    -> create that Product
    -> inspect that Product
    -> continue to the next transformation
```

The notebook must no longer defer all seven visualizations to one combined
figure at the bottom. It must also make overlapping sensor traces easier to
distinguish by using thinner, semi-transparent lines and a separate legend
below the x-axis of every plot.

This maintenance changes presentation structure only. It changes no Product
value, scientific law, public API, package code, dependency, metadata, or
supported execution boundary.

## Governing Sources

Implementation, Validation, and Review must read:

- [AGENTS](../../AGENTS.md);
- [CONTRIBUTING](../../CONTRIBUTING.md);
- [Overview](../overview.md);
- [Design](../design.md);
- [Validation](../validation.md);
- [Maintenance 15 architecture](maintenance_15_spec_composed_products_and_application_boundary.md);
- [Maintenance 17](maintenance_17_application_neutral_readout_quickstart.md);
- [Maintenance 18](maintenance_18_encoded_waveform_raw_zle.md); and
- [Maintenance 19](maintenance_19_readout_quickstart_cell_separation.md).

Maintenance 19's cell-ownership rule remains governing:

```text
mathematical or tensor-value preparation
    lives in its own cell

TensorCore / TensorDSLab semantic construction
    lives in its own cell

assertions
    live in their own cell

presentation setup and presentation calls
    remain visibly separate from Product construction
```

## Exact Baseline

Maintenance 20 starts from exact locally closed Maintenance 19:

```text
local main:
    fce1f352ae417a009e52c3ab889ba7881e42468b
tree:
    dc538012906d0af1908b479d635a3bce9bb43edf
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

The baseline source-only notebook has:

```text
34 cells:
    17 Markdown
    17 code
Products:
    Photoelectrons
    Charge
    PureWaveform
    NoiseWaveform
    AnalogWaveform
    DigitizedWaveform
    EncodedWaveform
presentation:
    one combined seven-panel display at the bottom
```

Maintenance 19 independently proved complete source/archive evidence at
`70/70/0`, positive Pyright with zero diagnostics, the unchanged exact
fifteen-diagnostic negative fixture, exact preservation of all seven Product
digests, and a byte-identical render relative to Maintenance 18.

## Selected Notebook Grammar

The notebook remains strictly alternating Markdown/code. Every code cell has
one immediately preceding plain-language Markdown cell.

The target is exactly `46` cells:

```text
23 Markdown
23 code
```

The exact code-cell sequence and stable IDs are:

```text
imports-code
axes-code
plotting-code
photoelectron-values-code
photoelectrons-code
photoelectrons-view-code
charge-code
charge-view-code
pulse-math-code
pure-waveform-code
pure-waveform-view-code
psd-values-code
noise-waveform-code
noise-waveform-view-code
analog-waveform-code
analog-waveform-view-code
digitizer-values-code
digitized-waveform-code
digitized-waveform-view-code
encoding-values-code
encoded-waveform-code
encoded-waveform-view-code
shared-shape-code
```

There is no final combined `plot-preparation-code` or `product-views-code`
cell.

## Plotting Setup

`plotting-code` is an ordinary presentation cell after the axes have been
constructed and before the first Product values are prepared.

It owns:

- the existing Matplotlib style;
- channel labels;
- the stable blue/orange/green sensor colors;
- ordinary CPU time values;
- one small private notebook-local `plot_product(...)` function; and
- the common visual policy for all seven figures.

The helper is presentation-only. It must not construct, prepare, validate, or
modify a TensorDSLab Product. It accepts one already-created Product and:

1. creates one figure and one axes;
2. copies that Product's three sensor traces to ordinary CPU lists;
3. optionally maps the EncodedWaveform suppression code to `NaN` for display;
4. draws each sensor with the existing stable color;
5. applies the requested thin and semi-transparent style;
6. labels the x-axis and y-axis;
7. places one three-column sensor legend below the x-axis; and
8. displays exactly that one Product figure.

The exact common curve style is:

```python
alpha=0.72
linewidth=0.9
```

The exact legend policy is:

```python
loc="upper center"
bbox_to_anchor=(0.5, -0.24)
ncol=3
frameon=False
```

Each figure uses a wide, short single-panel shape suitable for one waveform:

```python
figsize=(11, 3.6)
```

The figure may use constrained layout or an explicit bottom margin, but
independent visual inspection must prove that the legend is fully visible,
centered below the x-axis, and does not overlap the x-axis label or plotted
data.

The helper uses step drawing for:

```text
Photoelectrons
Charge
DigitizedWaveform
EncodedWaveform
```

It uses ordinary line drawing for:

```text
PureWaveform
NoiseWaveform
AnalogWaveform
```

## Product-Local Views

Every Product semantic-construction cell is immediately followed by a short
Markdown explanation and one view cell:

```text
photoelectrons-code
photoelectrons-view-code

charge-code
charge-view-code

pure-waveform-code
pure-waveform-view-code

noise-waveform-code
noise-waveform-view-code

analog-waveform-code
analog-waveform-view-code

digitized-waveform-code
digitized-waveform-view-code

encoded-waveform-code
encoded-waveform-view-code
```

Each view cell contains one concise call to the presentation helper for the
Product just constructed. It must not construct another Product, prepare
mathematical inputs, add assertions, or access a later Product.

The exact view order and labels remain:

```text
Photoelectrons
Charge (avalanche)
Pure (mV)
Noise (mV)
Analog (mV)
ADC code
Retained ADC code
```

`encoded-waveform-view-code` alone supplies the exact suppression code to the
helper so that suppressed regions remain blank. No Product tensor is mutated.

The final combined seven-panel renderer and figure-level legend are retired
without a compatibility cell.

## Preserved Cell Ownership

The following Maintenance 19 separations remain exact:

- `photoelectron-values-code` owns source tensor values;
- `photoelectrons-code` owns Photoelectrons Spec/Product construction;
- `pulse-math-code` owns the complete Gaussian/double-error-function
  calculation;
- `pure-waveform-code` owns OffsetAxis, PulseResponse, Config, and
  `PureWaveform.create(...)`;
- `psd-values-code` owns the three PSD rows and stack;
- `noise-waveform-code` owns PSD semantics and `NoiseWaveform.create(...)`;
- `digitizer-values-code` owns exactly four scalar tensors;
- `digitized-waveform-code` owns digitizer Kernel/Product construction and no
  direct `torch.tensor(...)`;
- `encoding-values-code` owns exactly five scalar tensors;
- `encoded-waveform-code` owns ZLE Kernel/Product construction and no direct
  `torch.tensor(...)`; and
- `shared-shape-code` remains the only assertion cell with exactly seven
  Product-shape assertions.

The assertion section moves to the end because every Product is available
there. It does not create a plot.

## Preserved Numerical Contract

The restructuring must preserve exactly:

- one Example and three sensor channels;
- `5000` Time bins at `2 ns`;
- `2501` one-sided Frequency bins at `0.1 MHz`;
- source deposits `(sensor, index, value)`:
  `(0,100,1)`, `(0,3700,4)`, `(1,1300,2)`, `(2,2500,3)`;
- the complete recognizable pulse equation, support, values, dtype, device,
  operation offsets, Unit, and illustrative classification;
- the three PSD rows, band counts, dtype, device, axes, Unit, and values;
- the same seeded Charge/noise execution and exact seven Product values;
- the `12`-bit, `[-80,20] mV`, unit-gain digitizer;
- the raw-ZLE policy `2500 / 2800 / 3 / 25 / 50`;
- `torch.int32` digitized/encoded values and suppression code `-1`;
- seven `(1,3,5000)` Products;
- the exact retained EncodedWaveform intervals;
- stable blue/orange/green sensor identity across all figures;
- blank presentation-only ZLE gaps;
- exact Product units and y-axis labels; and
- all newcomer-facing non-calibration language.

Product tensors, Product digests, retained support, source immutability,
freshness, dtype, device, Units, axes, and stochastic replay must remain
exactly Maintenance-19-identical.

## Source-Only And External Execution Contract

The committed notebook remains source-only:

```text
all execution_count values:
    null
all code-cell outputs:
    empty
attachments:
    absent
```

Implementation executes a temporary copy outside the repository and retains:

- one temporary executed notebook containing the seven local plot outputs;
- the seven individual PNG outputs; and
- one optional contact sheet assembled outside the repository for convenient
  Design/user inspection.

No executed notebook, PNG, contact sheet, cache, or other generated byte enters
the repository.

## Exact Scope

Design authority may change exactly:

```text
docs/implementation/index.md
docs/implementation/maintenance_19_readout_quickstart_cell_separation.md
docs/implementation/maintenance_20_product_local_readout_plots.md
```

Implementation may change exactly:

```text
demos/readout.ipynb
tests/test_readout_demo.py
```

Every other path is protected, including:

- `tensor_dslab/**`;
- every other `tests/**` path;
- `pyproject.toml`;
- `create_environment.sh`;
- all dependency, environment, package-version, and artifact metadata;
- every other demo;
- all other living architecture, parity, API, overview, design, validation,
  contribution, governance, and historical implementation records; and
- the three exact Design authority records after dispatch.

No compatibility module, alias, forwarder, public export, data asset, stored
executed notebook, plot image, cache, bytecode, build output, or environment
residue may enter the repository.

## Implementation Route

Implementation uses:

```text
branch:
    codex/maintenance-20-product-local-readout-plots
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
3. preserve unique stable notebook IDs and public imports;
4. update structural tests to prove the exact twenty-three-cell grammar,
   seven local view calls, and absence of the combined renderer;
5. preserve the complete numerical/Product oracle;
6. execute focused and complete source tests plus positive/negative typing;
7. execute one temporary notebook copy outside the repository;
8. save the executed notebook and seven PNGs outside the repository;
9. independently inspect all seven plots and the legend placement;
10. verify committed source bytes remain unchanged by execution;
11. verify diff/scope/privacy/artifact hygiene and branch cleanliness; and
12. commit one immutable direct-child candidate for Design routing.

Implementation must not contact Validation or Review directly.

## Focused Proof

`tests/test_readout_demo.py` must preserve its complete numerical, execution,
public-import, source-immutability, Product-contract, and plot-content oracle
while updating structural expectations made obsolete by the presentation
change.

It must prove:

- exact `46 / 23 / 23` source inventory;
- strict Markdown/code alternation;
- exact ordered stable code-cell IDs;
- all Maintenance 19 math/semantic/assertion separations;
- `plotting-code` owns the helper and exact common alpha, linewidth, figure
  size, legend placement, labels, colors, and host conversion;
- Product construction does not occur in `plotting-code`;
- each of the seven view cells calls the helper exactly once for its
  immediately preceding Product;
- exact view order, y-axis labels, and step/line choices;
- only the EncodedWaveform view supplies suppression handling;
- view cells contain no Product construction, assertions, mathematical input
  preparation, or later-Product access;
- there are exactly seven display outputs and seven PNG outputs after
  execution;
- every figure has exactly one axes and three stable-color sensor lines;
- every axes uses `alpha == 0.72` and `linewidth == 0.9`;
- every axes has its own three-entry legend below the x-axis;
- EncodedWaveform alone contains visible suppression gaps;
- no combined seven-panel figure, figure-level legend, `products` tuple,
  `y_labels` tuple, or `step_panels` set remains; and
- the seven Product values/digests and retained support remain exact.

Do not add dynamic source rewriting, broad `Any`, casts, ignores, generated
test methods, or a plotting dependency in package code.

## Validation

Validation receives one immutable Design-dispatched candidate and must run:

1. exact identity/tree/parent/ref/direct-child and two-path executable scope;
2. notebook inventory, stable IDs, source-only metadata, public imports, and
   cell-ownership proofs;
3. focused source and exact extracted-archive demo tests;
4. complete source and exact extracted-archive suites;
5. positive Pyright and the unchanged exact fifteen-diagnostic negative
   fixture in source/archive forms;
6. two immediate CPU executions proving deterministic seven-Product identity,
   seven display outputs, and seven PNG outputs;
7. direct equality of every Product digest and retained interval against exact
   Maintenance 19;
8. independent visual inspection of all seven figures, including curve
   transparency, thin lines, stable colors, non-overlapping legends below
   each x-axis, visible signals/noise, non-railed ADC, and encoded gaps;
9. byte equality for every protected production, dependency, metadata,
   environment, documentation, and non-demo-test path;
10. privacy, artifact/cache/bytecode/build hygiene; and
11. final exact detached cleanliness.

Artifact rebuilding and a fresh Conda environment are not required because
package, dependency, metadata, environment, and production bytes are
protected. Validation must not infer an artifact, installation, CUDA,
performance, calibration, compatibility, release, or publication claim.

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

Maintenance 20 is complete only when:

- each Product is plotted immediately in its own notebook section;
- all seven figures use the selected thin, semi-transparent curve style;
- every figure has its own legend below the x-axis;
- the source-only notebook preserves every numerical Product result;
- focused and complete Validation clear the same immutable bytes;
- independent Review returns zero findings;
- Design approves those same bytes; and
- Review fast-forwards them unchanged to clean local `main`.

The final report must link the source-only notebook and the temporary executed
notebook and show a temporary contact sheet or the individual rendered plots
without committing generated outputs.
