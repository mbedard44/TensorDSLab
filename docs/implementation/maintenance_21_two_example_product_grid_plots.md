# Maintenance 21 Two-Example Product Grid Plots

Status: **Merged / Closed** through exact Review-cleared and fast-forwarded
Candidate 1 `98db48138800b8500d587389e5e159c11f252a73`, tree
`b75ee59c6383a293658e0eaa8f8aa60942110a6d`.

Stable key:
`TensorDSLab/maintenance-21-two-example-product-grid-plots`

## Purpose

Make the application-neutral readout quickstart demonstrate a genuinely
batched three-dimensional Product domain:

```text
(Example, Channel, Time)
```

The notebook will use:

```text
2 independent examples
3 sensor channels
5000 time samples
```

Every Product therefore has exact shape:

```text
(2, 3, 5000)
```

Each Product remains plotted immediately after it is constructed. Its local
figure becomes a `3 x 2` grid:

```text
columns:
    Example 0
    Example 1

rows:
    sensor-0
    sensor-1
    sensor-2
```

Each subplot contains one trace. A Product uses one stable color across all
six of its subplots, while every other Product uses a different color.
Legends are omitted because each subplot contains only the one Product named
by the figure title. Column titles, row labels, and the Product-level figure
title communicate the complete layout directly.

There is no final combined or summary plot. The newcomer sees each Product
where it enters the story and never has to decode a second presentation
vocabulary at the bottom of the notebook.

This maintenance changes the demonstration's Example extent, second-example
source values, derived Product values, and presentation layout. It changes no
production code, public API, Product law, scientific equation, dependency,
metadata, environment, supported device, or package execution boundary.

## Governing Sources

Implementation, Validation, and Review must read:

- [AGENTS](../../AGENTS.md);
- [CONTRIBUTING](../../CONTRIBUTING.md);
- [Overview](../overview.md);
- [Design](../design.md);
- [Validation](../validation.md);
- [Maintenance 15 architecture](maintenance_15_spec_composed_products_and_application_boundary.md);
- [Maintenance 17](maintenance_17_application_neutral_readout_quickstart.md);
- [Maintenance 18](maintenance_18_encoded_waveform_raw_zle.md);
- [Maintenance 19](maintenance_19_readout_quickstart_cell_separation.md);
  and
- [Maintenance 20](maintenance_20_product_local_readout_plots.md).

Maintenance 15 remains the Product architecture authority. Maintenance 19's
cell-ownership rule remains governing:

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

Maintenance 20 remains the presentation-sequencing authority:

```text
explain one Product
    -> create that Product
    -> inspect that Product
    -> continue to the next transformation
```

Maintenance 21 changes only the batch extent, the second realization, and the
internal layout/color vocabulary of those already-local Product figures.

## Exact Baseline

Maintenance 21 starts from exact locally closed Maintenance 20:

```text
local main:
    77201b415d00237b4a62951d6e13faa1751a1057
tree:
    7293508c667a9da2939e125beae41192336951ca
package version:
    0.2.0
source notebook SHA-256:
    c58f2cce0ebb19955e6604c6b02a0f1ba3fd8af150bf5eb3474b17ed1fe3d9d9
```

The exact TensorCore dependency remains published `0.22.0`:

```text
commit:
    19bfae35fbc773b55cac7bcd659dda57c4dee6d6
tree:
    53aa10520a50c0714e79c685d814cbae1b6f7740
```

The baseline notebook has:

```text
46 cells:
    23 Markdown
    23 code
Products:
    Photoelectrons
    Charge
    PureWaveform
    NoiseWaveform
    AnalogWaveform
    DigitizedWaveform
    EncodedWaveform
Product domain:
    (1, 3, 5000)
presentation:
    seven Product-local single-axes figures
    three overlaid sensor traces per figure
    one sensor legend below each x-axis
```

Maintenance 20 independently passed:

```text
focused source/archive:
    3 / 3 / 0 each
complete source/archive:
    70 / 70 / 0 each
positive Pyright:
    0 errors / 0 warnings / 0 informations
negative fixture:
    exactly 15 intended diagnostics
Validation:
    CLEAR
Review:
    CLEAR
local ff-only merge:
    exact
```

## Selected Notebook Grammar

The source-only notebook retains the exact Maintenance 20 inventory:

```text
46 cells
23 Markdown
23 code
strict Markdown/code alternation
```

The exact ordered code-cell IDs remain:

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

Every Product semantic-construction cell remains immediately followed by its
short explanation and local view cell. The final code cell remains the shared
shape assertion cell. No final plot cell is added.

## Example Semantics

`ExampleAxis` contains independent waveform realizations. It does not contain
consecutive DAQ slices and must not imply cross-example temporal state.

The exact axis construction becomes conceptually:

```python
example_axis = ExampleAxis(
    coordinates=CountCoordinates(count=2),
)
```

Both examples share:

- the same ChannelAxis;
- the same TimeAxis;
- the same FrequencyAxis relationship;
- the same PulseResponse;
- the same channel-conditioned PowerSpectralDensity;
- the same digitizer Kernels;
- the same ZLE policy Kernels; and
- one batched Product call at each transformation.

They differ through their source Photoelectrons values and their independent
addressed stochastic realization.

The notebook prose must say plainly that one Product call processes both
examples and all three channels together. It must not describe the examples
as adjacent windows, linked records, or sequential state.

## Exact Source Deposits

Example 0 retains the complete recognizable Maintenance 20 fixture:

```text
(example, sensor, sample, value)

(0, 0,  100, 1)
(0, 0, 3700, 4)
(0, 1, 1300, 2)
(0, 2, 2500, 3)
```

Example 1 uses a distinct complete realization:

```text
(1, 0,  800, 2)
(1, 1, 2000, 4)
(1, 1, 3500, 1)
(1, 2, 2900, 3)
```

Every deposit begins early enough for the exact `1011`-coefficient
PulseResponse to remain inside the `5000`-sample window.

The source-value Markdown names both realizations as illustrative and
independent. The source tensor is still prepared in
`photoelectron-values-code`, and the copied before-image still proves source
immutability after the complete chain.

## Product Shape And Numerical Contract

The exact shared Product shape becomes:

```text
(2, 3, 5000)
```

The notebook retains exactly seven light shape assertions in
`shared-shape-code`.

The following Maintenance 20 facts remain unchanged:

- TimeAxis: `5000` bins at `2 ns`;
- FrequencyAxis: `2501` bins at `0.1 MHz`;
- ChannelAxis labels: `sensor-0`, `sensor-1`, `sensor-2`;
- exact Gaussian/double-error-function PulseResponse mathematics;
- exact `2020.27 ns` pulse support and `1011` operation offsets;
- exact peak normalization `-14.5912372 mV / avalanche`;
- exact three PowerSpectralDensity rows and channel conditioning;
- seed `2026`;
- exact `12`-bit, `[-80, 20] mV`, unit-gain digitizer;
- exact raw-ZLE policy `2500 / 2800 / 3 / 25 / 50`;
- exact `torch.int32` DigitizedWaveform and EncodedWaveform dtypes;
- exact EncodedWaveform suppression code `-1`;
- exact Units for all Fields and quantity Kernels;
- source immutability;
- fresh Product storage;
- public-only imports; and
- illustrative/non-calibration language.

Example 0's complete Product slices must remain byte/numerically identical to
the exact Maintenance 20 Product values. Implementation must freeze the new
complete two-example Product digests and exact Example 1 encoded intervals in
the focused proof. Validation and Review must independently reproduce them
from the immutable candidate rather than accepting implementation output.

No cross-example state, source mixing, RNG address collision, ZLE state,
padding, or retained support is permitted.

## Product Grid Layout

`plotting-code` remains the one presentation-helper owner.

For each already-created Product, `plot_product(...)` creates exactly:

```text
one figure
six axes
three rows
two columns
one trace per axes
```

The selected orientation is:

```text
column 0:
    Example 0
column 1:
    Example 1

row 0:
    sensor-0
row 1:
    sensor-1
row 2:
    sensor-2
```

The helper must request a non-squeezed two-dimensional axes structure so the
layout remains explicit:

```python
plt.subplots(
    3,
    2,
    squeeze=False,
    ...
)
```

Each axes reads exactly:

```python
product.tensor[example_index, channel_index]
```

It never sums, flattens, concatenates, permutes, averages, or otherwise
combines Example or Channel coordinates.

The Product's exact presentation label is the figure-level title:

```text
Photoelectrons
Charge (avalanche)
Pure (mV)
Noise (mV)
Analog (mV)
ADC code
Retained ADC code
```

The exact column titles are:

```text
Example 0
Example 1
```

The exact row labels are:

```text
sensor-0
sensor-1
sensor-2
```

The bottom row supplies the shared x-axis label:

```text
Time (ns)
```

Other axes may omit the repeated x-axis label. Tick labels may be shared in
the ordinary Matplotlib way.

No legend is created at the axes or figure level. Each axes has one trace,
the figure title already names the Product, and the row/column labels identify
the exact tensor coordinates.

Independent visual inspection must prove:

- all six axes are visible;
- row and column labels are readable;
- no label/title overlaps plotted data;
- no axes is cropped;
- the two examples are visibly distinct where their source/scientific values
  differ;
- pulses, noise, digitizer values, and ZLE gaps remain readable; and
- the grid tells the same tensor-domain story at ordinary notebook width.

The selected figure dimensions may be tuned during implementation, but the
committed proof must freeze the final exact `figsize`, layout/margin policy,
and title/label placement. A likely starting point is:

```python
figsize=(13, 8.5)
sharex=True
squeeze=False
```

If visual QA requires a small adjustment, Implementation may choose the
smallest common figure-level adjustment. Product-specific layout branches are
forbidden.

## Product Colors

Each Product uses one color across all six coordinate panels. The exact
selected palette is:

```text
Photoelectrons:
    tab:blue
Charge:
    tab:orange
PureWaveform:
    tab:green
NoiseWaveform:
    tab:red
AnalogWaveform:
    tab:purple
DigitizedWaveform:
    tab:brown
EncodedWaveform:
    tab:pink
```

All seven colors are distinct. A sensor or example does not change the
Product color.

The Maintenance 20 common curve style remains:

```python
alpha=0.72
linewidth=0.9
```

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

`EncodedWaveform` alone supplies suppression handling. The helper maps the
exact suppression code to `NaN` only in its ordinary CPU presentation values.
No Product tensor is mutated.

## Product-Local View Calls

Each view cell remains one concise call for the Product just constructed.
Conceptually:

```python
plot_product(
    photoelectrons,
    title="Photoelectrons",
    color="tab:blue",
    step=True,
)
```

The exact parameter spelling may use `title` or another narrowly equivalent
presentation name, but the focused proof must freeze it consistently.

Each view cell:

- accesses only its already-created Product;
- supplies exactly that Product's title and color;
- selects the exact step/line policy;
- supplies suppression handling only for EncodedWaveform;
- contains no Product construction;
- contains no mathematical preparation;
- contains no assertion;
- contains no direct Matplotlib call; and
- accesses no later Product.

No final combined plot, summary figure, contact sheet, or repeated Product
view appears in the committed notebook.

## Preserved Cell Ownership

The following separations remain exact:

- `photoelectron-values-code` owns both examples' source tensor values;
- `photoelectrons-code` owns Photoelectrons Spec/Product construction;
- `pulse-math-code` owns the complete pulse calculation;
- `pure-waveform-code` owns PulseResponse/Config/Product semantics;
- `psd-values-code` owns the three PSD rows and stack;
- `noise-waveform-code` owns PSD/Config/Product semantics;
- `digitizer-values-code` owns exactly four scalar tensors;
- `digitized-waveform-code` owns digitizer Kernel/Product construction and no
  direct `torch.tensor(...)`;
- `encoding-values-code` owns exactly five scalar tensors;
- `encoded-waveform-code` owns ZLE Kernel/Product construction and no direct
  `torch.tensor(...)`; and
- `shared-shape-code` remains the only assertion cell and contains exactly
  seven Product-shape assertions.

The plot helper and seven plot calls remain presentation-only.

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

- one temporary executed notebook with seven Product-local display outputs;
- seven individual PNG outputs;
- an optional temporary contact sheet; and
- one machine-readable execution summary sufficient to bind shapes, Product
  digests, Example 0 preservation, Example 1 support, and figure geometry.

The preferred paths are:

```text
/private/tmp/tensordslab-maintenance-21-readout-executed.ipynb
/private/tmp/tensordslab-maintenance-21-photoelectrons.png
/private/tmp/tensordslab-maintenance-21-charge.png
/private/tmp/tensordslab-maintenance-21-pure-waveform.png
/private/tmp/tensordslab-maintenance-21-noise-waveform.png
/private/tmp/tensordslab-maintenance-21-analog-waveform.png
/private/tmp/tensordslab-maintenance-21-digitized-waveform.png
/private/tmp/tensordslab-maintenance-21-encoded-waveform.png
```

No executed notebook, image, contact sheet, cache, bytecode, or other
generated byte enters the repository.

## Exact Scope

Design authority may change exactly:

```text
docs/implementation/index.md
docs/implementation/maintenance_20_product_local_readout_plots.md
docs/implementation/maintenance_21_two_example_product_grid_plots.md
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
- every other demo;
- all dependency, metadata, package-version, artifact, and environment bytes;
- every other living or historical documentation record; and
- the three exact Design authority records after dispatch.

No compatibility module, public export, data asset, stored output, cache,
bytecode, build output, or environment residue may enter the repository.

## Implementation Route

Implementation uses:

```text
branch:
    codex/maintenance-21-two-example-product-grids
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

1. bind the exact authority commit/tree/parent and clean required branch;
2. modify only the notebook and focused proof;
3. preserve the exact `46 / 23 / 23` source-only grammar and stable IDs;
4. construct the exact two-example source fixture;
5. preserve Example 0's complete Maintenance 20 Product slices;
6. implement one `3 x 2` grid per Product with the exact orientation;
7. assign the exact one-color-per-Product palette;
8. remove all legends and retain no final summary plot;
9. update the focused proof substantively;
10. freeze exact complete two-example Product digests and Example 1 encoded
    intervals in the proof;
11. run focused and complete source tests;
12. run positive Pyright and the unchanged negative fixture;
13. execute at least two temporary notebook copies outside the repository;
14. retain and inspect all seven rendered figures;
15. verify the committed notebook remains source-only and byte-unchanged;
16. verify exact scope, protected bytes, privacy, and artifact hygiene; and
17. commit one immutable direct-child candidate for Design routing.

Implementation must stop and return to Design if:

- batching changes Example 0 scientific values;
- any Product fails the exact `(2, 3, 5000)` domain;
- independent examples require cross-example production state;
- the grid cannot remain readable at ordinary notebook width;
- a legend or final summary plot is required to identify the data; or
- implementation needs production, dependency, metadata, environment, or
  non-focused-test changes.

Implementation must not contact Validation or Review directly.

## Focused Proof

`tests/test_readout_demo.py` must preserve its complete public-import,
source-only, scientific, Product, source-immutability, and execution oracle
while replacing obsolete one-example and single-axes expectations.

It must prove:

- exact `46 / 23 / 23` inventory and strict alternation;
- exact stable code-cell IDs;
- `ExampleAxis` has exact CountCoordinates extent `2`;
- exact eight source deposits across the two examples;
- all seven Products have exact shape `(2, 3, 5000)`;
- Example 0's Product slices equal exact Maintenance 20;
- complete new Product digests are frozen;
- exact Example 1 EncodedWaveform intervals are frozen;
- no cross-example support or value mixing occurs;
- plotting-code is the only helper owner;
- the helper creates an exact `3 x 2`, non-squeezed axes grid;
- axes index Product values positionally as
  `[example_index, channel_index]`;
- column titles are exact `Example 0` and `Example 1`;
- row labels are exact sensor labels;
- the figure-level title names the Product;
- bottom-row x-labels are exact `Time (ns)`;
- every figure has exactly six axes and six lines;
- every axes has exactly one line;
- no axes or figure legend exists;
- exact alpha and linewidth apply to all forty-two lines;
- each Product uses its exact selected color across all six axes;
- all seven Products use distinct selected colors;
- exact step/line policy is retained;
- only EncodedWaveform uses suppression masking;
- there are exactly seven display and PNG outputs;
- no final combined/summary plot vocabulary remains;
- all value/semantic/assertion/presentation ownership boundaries remain; and
- every Product view remains immediately local to its construction.

The proof must inspect real Matplotlib axes/artists after execution, including
figure shape, line count, line colors, line styles/draw styles, titles,
labels, legend absence, and renderer-space bounds.

Do not add dynamic source rewriting, generated tests, broad `Any`, casts,
ignores, type suppressions, or plotting dependencies to package code.

## Validation

Validation receives one immutable Design-dispatched candidate and runs:

1. exact commit/tree/parent/ref/direct-child identity;
2. exact two-path executable scope and all protected-byte equality;
3. exact source-only notebook inventory, IDs, imports, and ownership;
4. focused source and exact extracted-archive demo tests;
5. complete source and exact extracted-archive test suites;
6. positive Pyright and the exact fifteen-diagnostic negative fixture in both
   dependency forms;
7. two immediate independent CPU executions;
8. exact deterministic equality between both executions;
9. exact `(2, 3, 5000)` shape and source fixture;
10. direct comparison of Example 0 slices to exact Maintenance 20;
11. independent reproduction of complete Product digests and both examples'
    EncodedWaveform intervals;
12. explicit independence checks in which an unrelated example/channel source
    change does not alter another lane;
13. exact seven-display/seven-PNG output count;
14. exact `3 x 2` grid geometry, artist counts, colors, style, titles, labels,
    legend absence, step/line policy, and Encoded-only masking;
15. independent visual inspection of all seven figures;
16. privacy, artifact/cache/bytecode/build hygiene; and
17. final exact detached cleanliness.

Artifact rebuilding and a fresh Conda environment are not required because
package, dependency, metadata, environment, and production bytes are
protected.

Validation must not infer artifact, installation, CUDA, performance,
calibration, compatibility, release, or publication claims.

## Review And Merge

Review remains independent and read-only until:

1. Validation clears one exact immutable candidate;
2. Design dispatches that exact candidate to Review;
3. Review returns CLEAR with zero findings; and
4. Design issues final same-byte approval.

Review's risk-based obligations include:

- exact identity, two-path scope, and protected-byte checks;
- semantic review of Example independence and positional axes indexing;
- direct review of the newcomer-facing two-example explanation;
- direct review of one-color-per-Product readability;
- independent execution of the focused proof in both dependency forms;
- direct numerical comparison of Example 0 to Maintenance 20;
- direct visual inspection of every Product grid;
- confirmation that legend removal loses no semantic information;
- confirmation that no final summary plot remains; and
- evidence-economy acceptance or targeted rerun of complete Validation
  evidence.

After final same-byte approval, Review owns only:

```text
git merge --ff-only <exact-approved-candidate>
```

on clean governed local `main`, followed by exact identity, tree, lineage,
diff/check, worktree-cleanliness, and origin-relation verification.

No merge commit, amend, rewrite, rebase, squash, push, tag, branch deletion,
CUDA run, artifact rebuild, environment creation, or publication is
authorized.

## Completion

Maintenance 21 is complete only when:

- the notebook demonstrates two independent examples and three channels;
- all seven Products have exact `(2, 3, 5000)` shape;
- each Product is inspected immediately in a `3 x 2` coordinate grid;
- every axes contains exactly one trace;
- each Product has one unique stable color;
- figure titles, column titles, and row labels replace legends cleanly;
- no final combined or summary plot exists;
- Example 0 remains exact Maintenance 20;
- both examples' values and supports are independently reproducible;
- the committed notebook remains source-only;
- complete fixed-byte Validation clears;
- independent Review returns zero findings;
- Design approves the same immutable bytes; and
- Review fast-forwards them unchanged to clean local `main`.

The final report must link the source-only and temporary executed notebooks
and show a temporary contact sheet or the individual seven grids.

CUDA, push, publication, compatibility, calibration, application IO,
reconstruction, performance, release, and deployment remain unauthorized.
