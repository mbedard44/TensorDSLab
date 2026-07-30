# Maintenance 23 Delayed Crosstalk And Global PSD Quickstart

Status: **Active Design authority; implementation undispatched**.

Stable key:
`TensorDSLab/maintenance-23-delayed-crosstalk-global-psd-quickstart`

## Purpose

Refine the sole newcomer-facing readout demonstration so its scientific story
is both more representative and easier to scan:

```text
Photoelectrons
    -> Charge with illustrative delayed crosstalk
    -> PureWaveform
    -> NoiseWaveform from one global PSD
    -> AnalogWaveform
    -> DigitizedWaveform
    -> EncodedWaveform
```

The notebook will demonstrate two complementary Kernel geometries:

```text
DelayedCrosstalk
    unconditioned
    one Time-relative operation axis

PowerSpectralDensity
    unconditioned
    one Frequency operation axis
```

Both Kernels apply to every example and sensor lane through ordinary
TensorDSLab broadcasting. The resulting stochastic realizations remain
independent by their existing package-owned addresses; sharing one coefficient
law does not make the generated lane values identical.

The notebook will also use only Product names as section headings. Preparation
math, semantic construction, assertions, and local plots remain separate code
blocks inside their owning Product section.

This maintenance changes only the supported readout notebook and its focused
proof. It changes no package production byte, public API, Product law, RNG
address, dependency, metadata, environment, supported device, or application
ownership boundary.

## Exact Baseline

Maintenance 23 starts from exact locally closed Maintenance 22:

```text
local main:
    ea59807e4fd18b4e120ce5675c5efacb7adbbd73

tree:
    cc9533fbed8412cf03119ec565c4eaafe74167bb

package version:
    0.2.0

committed notebook SHA-256:
    5f423a6c90d093e10af5d2f8e7f5dcfe1070cf369d195b28b27f2d58c66be57b
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

committed execution counts:
    1..23

committed displays:
    seven Product-local PNG outputs

Product domain:
    (2, 3, 5000)

Charge mechanisms:
    none

PowerSpectralDensity:
    three Channel-conditioned rows
```

Maintenance 22 independently cleared:

```text
focused source/archive:
    3 / 3 / 0 each

complete source/archive:
    70 / 70 / 0 each

positive Pyright:
    0 errors / 0 warnings / 0 informations

negative fixture:
    exactly 15 intended diagnostics

required output mutants:
    18 / 18 killed

Validation:
    CLEAR

Review:
    CLEAR

local ff-only merge:
    exact
```

## Governing Standards

Implementation, Validation, and Review must apply:

- [AGENTS](../../AGENTS.md) for role separation, exact-byte routing,
  application ownership, privacy, and local-main merge authority;
- [CONTRIBUTING](../../CONTRIBUTING.md) for deterministic evidence, focused
  tests, documentation, and honest qualification;
- [Overview](../overview.md) for the current package and demonstration map;
- [Design](../design.md) for application-owned workflow composition;
- [Validation](../validation.md) for notebook replay and visual evidence;
- [Maintenance 15 architecture](maintenance_15_spec_composed_products_and_application_boundary.md)
  for direct Product/Spec/Kernel/Config composition;
- [Maintenance 16](maintenance_16_declarative_requirements_and_kernel_ownership.md)
  for coefficient-Kernel ownership;
- [Maintenance 18](maintenance_18_encoded_waveform_raw_zle.md) for the exact
  EncodedWaveform result;
- [Maintenance 19](maintenance_19_readout_quickstart_cell_separation.md) for
  preparation/semantic/assertion/presentation separation;
- [Maintenance 21](maintenance_21_two_example_product_grid_plots.md) for the
  two-example, three-channel Product grids; and
- [Maintenance 22](maintenance_22_committed_readout_notebook_outputs.md) for
  normalized committed outputs and clear-before-replay evidence.

The governing cell-ownership rule remains:

```text
ordinary tensor-value or mathematical preparation
    -> its own code block

TensorCore/TensorDSLab semantic construction
    -> its own code block

assertions
    -> their own code block

presentation preparation and view calls
    -> visibly separate from Product construction
```

The notebook remains ordinary application code that composes independent
Products. It does not introduce a profile, Readout collection, workflow
factory, package orchestration layer, or one-shot readout API.

## Selected Product Story

The exact Product sequence remains:

```text
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
EncodedWaveform
```

Every Product continues to have:

```text
axes:
    (ExampleAxis, ChannelAxis, TimeAxis)

shape:
    (2, 3, 5000)

device:
    cpu
```

The source fixture, TimeAxis, FrequencyAxis, PulseResponse mathematics,
digitizer policy, encoding policy, seed, plotting geometry, and Product colors
remain unchanged unless this work order explicitly replaces them.

The notebook continues to make no calibration claim. The delayed-crosstalk and
PSD coefficients are intentionally illustrative.

## Product-Only Heading Contract

The notebook retains one document title:

```text
# Readout quickstart
```

Below it, the complete ordered level-two heading inventory is exactly:

```text
## Photoelectrons
## Charge
## PureWaveform
## NoiseWaveform
## AnalogWaveform
## DigitizedWaveform
## EncodedWaveform
```

No other section heading is permitted.

In particular, these former headings become ordinary unheaded explanatory
paragraphs inside the surrounding story:

```text
Axes
Plotting setup
Photoelectron values
Pulse mathematics
PSD values
Digitizer values
Encoding values
Shared shape
```

The imports, shared axes, and plotting helper remain before the first Product
section. They use prose, whitespace, and code-block separation without
inventing another section taxonomy.

Each Product section may contain several Markdown/code pairs. The absence of a
subheading does not merge responsibilities:

```text
Product heading and preparation explanation
preparation code
semantic-construction explanation
semantic Product code
local-view explanation
local-view code
```

Products with no separate preparation block keep only the blocks they need.

The final shared-shape paragraph and assertion cell remain after
EncodedWaveform without another heading.

## Exact Cell Grammar

Charge gains one dedicated preparation code block so delayed-crosstalk
mathematics never shares the semantic construction cell.

The exact target inventory becomes:

```text
48 cells
24 Markdown
24 code
strict Markdown/code alternation
```

All existing cell IDs remain stable. The new cells are exactly:

```text
delayed-crosstalk-values-explanation
delayed-crosstalk-values-code
```

They are inserted immediately after `charge-explanation` and immediately
before `charge-code`.

The exact ordered code-cell IDs become:

```text
imports-code
axes-code
plotting-code
photoelectron-values-code
photoelectrons-code
photoelectrons-view-code
delayed-crosstalk-values-code
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

The committed execution counts become exact top-to-bottom integers:

```text
1..24
```

Only these seven code cells own outputs:

```text
photoelectrons-view-code
charge-view-code
pure-waveform-view-code
noise-waveform-view-code
analog-waveform-view-code
digitized-waveform-view-code
encoded-waveform-view-code
```

Each owns exactly one normalized `display_data` output with:

```text
data:
    image/png
    text/plain

metadata:
    {}
```

Every other code cell owns no output.

## Illustrative Delayed Crosstalk

### Geometry

The Charge section adds one unconditioned `DelayedCrosstalk` Kernel.

Its exact conditioning axes are:

```python
conditioning_axes=()
```

Its operation geometry is one exact `OffsetAxis` relative to `TimeAxis`:

```python
delayed_time_axis = OffsetAxis(
    coordinates=OffsetCoordinates(
        offsets=tuple(range(1, 251)),
    ),
    relative_to=TimeAxis,
)
```

The offsets are exact source-sample displacements:

```text
1..250 samples
2..500 ns on the exact 2 ns TimeAxis
```

Offset zero is absent. This is delayed, not direct, crosstalk.

The Kernel is global: no ExampleAxis or ChannelAxis is a conditioning axis.
The same expected-offspring law applies to all six lanes.

### Preparation mathematics

`delayed-crosstalk-values-code` owns only ordinary Torch/Python preparation.
It computes the exact binary64 coefficient vector:

```python
delayed_offsets = tuple(range(1, 251))
delayed_times_ns = (
    torch.arange(
        1,
        251,
        dtype=torch.float64,
        device=device,
    )
    * time_axis.coordinate_scale
)
delayed_crosstalk_values = torch.exp(
    -delayed_times_ns / 150.0
)
delayed_crosstalk_values *= (
    0.15 / delayed_crosstalk_values.sum()
)
```

The exact represented facts on the frozen stack are:

```text
dtype:
    torch.float64

shape:
    (250,)

sum:
    0.14999999999999997

first coefficient:
    0.0020602220776112065

last coefficient:
    7.448286214784992e-05

contiguous tensor SHA-256:
    a49cd5c4c14d606904a321aa48ba7a8d861f3f37ea03ecdbc0d4492d4d780b70
```

The coefficients are expected Poisson offspring intensities. They are not
independent Bernoulli probabilities. Their sum is the first-generation
expected offspring mean.

The profile is intentionally illustrative and uncalibrated. The notebook may
say that plainly. It must not imply a DarkSide20k, Silex, 3Dπ, IV-DSLab, or
hardware measurement.

### Semantic construction

`charge-code` owns:

- `ChargeSpec`;
- `OffsetAxis`;
- `DelayedCrosstalkSpec`;
- `DelayedCrosstalk`;
- `ChargeKernels`;
- `ChargeConfig`; and
- `Charge.create`.

Conceptually:

```python
delayed_crosstalk = DelayedCrosstalk(
    tensor=delayed_crosstalk_values,
    spec=DelayedCrosstalkSpec(
        conditioning_axes=(),
        operation_axes=(delayed_time_axis,),
        device=device,
        dtype=torch.float64,
        unit=unit_registry.Unit(""),
    ),
)

charge_config = ChargeConfig(
    spec=ChargeSpec(
        axes=axes,
        device=device,
        dtype=field_dtype,
        unit=unit_registry.Unit("avalanche"),
    ),
    kernels=ChargeKernels(
        members=(delayed_crosstalk,),
    ),
    correlated_avalanche_generations=NonnegativeInteger(value=2),
)

charge = Charge.create(
    sources=(photoelectrons,),
    config=charge_config,
    rng=rng,
)
```

The Config uses exactly two correlated-avalanche generations. The existing
package-owned delayed-crosstalk RNG role and address construction remain
unchanged.

The old empty `ChargeKernels` and zero-generation configuration are removed.
The notebook must not describe Charge as unchanged Photoelectrons.

The focused proof must show:

- exact Kernel shape, dtype, Unit, axes, and coefficient digest;
- exact mean `0.15` within the existing binary64 tolerance;
- exact offsets `1..250`;
- exact `relative_to is TimeAxis`;
- empty conditioning axes;
- exact two-generation policy;
- source immutability;
- deterministic replay on the frozen stack;
- at least one delayed descendant in the fixed seeded demonstration; and
- no descendant appears before its parent sample or outside the represented
  TimeAxis.

## One Global Power Spectral Density

### Representation

The three Channel-conditioned PSD rows are retired from the notebook.

The replacement is one global tensor:

```text
shape:
    (2501,)

dtype:
    torch.float32

device:
    cpu

unit:
    mV ** 2
```

Its Spec has:

```python
conditioning_axes=()
operation_axes=(frequency_axis,)
```

The same coefficient tensor therefore broadcasts across both examples and all
three sensor lanes. The addressed RNG still produces distinct stochastic
realizations for different lanes.

The notebook prose should say only that this is one illustrative global PSD
shared by every example and sensor. It must not discuss a source file,
conversion procedure, FFT, sampling-system provenance, calibration, or why
part of the vector is zero.

### Exact value payload

`psd-values-code` owns the ordinary value construction. It contains one
Design-supplied exact 626-value decimal float32 literal prefix and appends
zeros to the exact FrequencyAxis extent:

```python
psd_values = torch.cat(
    (
        torch.tensor(
            (
                # exact 626-value Design payload
            ),
            dtype=field_dtype,
            device=device,
        ),
        torch.zeros(
            frequency_axis.size - 626,
            dtype=field_dtype,
            device=device,
        ),
    )
)
```

The exact complete represented tensor facts are:

```text
shape:
    (2501,)

nonzero cells:
    625

DC:
    0.0

positive represented cells:
    indices 1..625

zero represented cells:
    index 0 and indices 626..2500

sum in Python binary64 over represented float32 cells:
    11.251378314314934

contiguous tensor SHA-256:
    928921a257e50cbe1720358a8b929249e5bff9383cb3c63be1a67769c4c09a47
```

The exact 626-value payload is a private Design-to-Implementation input. Its
decimal serialization is not a package data asset, durable artifact, or public
scientific contract. The immutable candidate freezes the resulting notebook
source and the complete tensor hash; Validation and Review independently
reconstruct and hash the tensor rather than trusting an operational handoff
file.

The notebook must contain no:

- `psd_sensor_0`;
- `psd_sensor_1`;
- `psd_sensor_2`;
- `torch.stack` for PSD rows;
- Channel-conditioned PSD Spec;
- per-sensor PSD explanation; or
- claim that equal PSD coefficients imply equal realized noise.

### Semantic construction

`noise-waveform-code` owns:

```python
power_spectral_density = PowerSpectralDensity(
    tensor=psd_values,
    spec=PowerSpectralDensitySpec(
        conditioning_axes=(),
        operation_axes=(frequency_axis,),
        device=device,
        dtype=field_dtype,
        unit=unit_registry.Unit("mV ** 2"),
    ),
)
```

It then constructs `NoiseWaveformKernels`, `NoiseWaveformConfig`, and
`NoiseWaveform.create` exactly through the current public API.

The focused proof must show:

- exact `(2501,)` Kernel shape;
- exact empty conditioning tuple;
- exact one FrequencyAxis operation tuple;
- exact dtype/device/Unit;
- exact complete tensor digest and sum;
- exact DC and positive non-DC admission;
- no ChannelAxis in the Kernel Spec;
- distinct realized noise lane values under the existing addressed RNG; and
- deterministic exact same-stack replay.

## Preserved Product Inputs

The following exact values remain unchanged:

```text
ExampleAxis:
    CountCoordinates(count=2)

ChannelAxis:
    sensor-0
    sensor-1
    sensor-2

TimeAxis:
    5000 bins
    2 ns coordinate scale

FrequencyAxis:
    2501 bins
    0.1 MHz coordinate scale

source deposits:
    (0, 0,  100, 1)
    (0, 0, 3700, 4)
    (0, 1, 1300, 2)
    (0, 2, 2500, 3)
    (1, 0,  800, 2)
    (1, 1, 2000, 4)
    (1, 1, 3500, 1)
    (1, 2, 2900, 3)

PulseResponse:
    exact Maintenance 17/21 mathematics
    1011 offsets
    -14.5912372 mV / avalanche peak

RNG seed:
    2026

digitizer:
    12 bit
    -80 mV minimum
    +20 mV maximum
    unit gain

raw ZLE:
    trigger 2500
    release 2800
    required time-over 3
    pre-trigger 25
    post-trigger 50
    suppression code -1
```

`Photoelectrons` values remain exact Maintenance 22. Charge and every
downstream Product are expected to change because the delayed-crosstalk law
and PSD law change.

Implementation must freeze new complete Product and per-example digests,
DigitizedWaveform extrema, and EncodedWaveform retained intervals in the
focused proof. Validation must reproduce them independently.

No old downstream digest, image hash, ADC range, or retained interval is
carried forward merely because the source fixture remains unchanged.

## Plot And Presentation Contract

The exact Maintenance 21/22 Product-local presentation remains:

```text
seven figures
one figure immediately after each Product
one 3 x 2 grid per figure
columns by Example
rows by Channel
one trace per axes
no axes legend
no figure legend
no final combined or summary plot
```

The exact Product colors remain:

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

The common curve style remains:

```text
alpha:
    0.72

linewidth:
    0.9
```

The step/line policy remains:

```text
step:
    Photoelectrons
    Charge
    DigitizedWaveform
    EncodedWaveform

line:
    PureWaveform
    NoiseWaveform
    AnalogWaveform
```

EncodedWaveform alone maps suppression codes to presentation-only `NaN`.

Independent visual inspection must confirm that:

- delayed Charge descendants are visible without obscuring the source story;
- the spread into PureWaveform and AnalogWaveform is recognizable;
- the global PSD produces visible but non-obscuring noise;
- distinct lanes remain visibly distinct despite the shared PSD law;
- the digitizer is not universally pinned to either rail;
- EncodedWaveform retains visible samples and visible suppression gaps;
- all forty-two axes remain readable;
- all headings and Product-local explanations remain newcomer-oriented; and
- committed images match the exact Product section immediately above them.

## Assertions

`shared-shape-code` remains the only assertion cell.

It contains exactly the seven light shape assertions:

```python
assert photoelectrons.tensor.shape == expected_shape
assert charge.tensor.shape == expected_shape
assert pure_waveform.tensor.shape == expected_shape
assert noise_waveform.tensor.shape == expected_shape
assert analog_waveform.tensor.shape == expected_shape
assert digitized_waveform.tensor.shape == expected_shape
assert encoded_waveform.tensor.shape == expected_shape
```

It does not assert Product equations, stochastic values, Kernel contents,
plotting facts, or provenance.

Tests own those detailed proofs. The newcomer notebook uses assertions only to
illustrate that every Product shares the same tensor domain.

## Committed Output Contract

The candidate commits one normalized top-to-bottom execution of the revised
notebook.

The notebook must contain:

```text
24 code execution counts:
    exact 1..24

seven and only seven display_data outputs
seven and only seven embedded PNGs
no stream output
no execute_result
no error output
no attachment
no widget state
no execution timing
no timestamp
no username
no hostname
no absolute path
no task or route identifier
```

Every cell metadata mapping is exactly `{}`. Output metadata is exactly `{}`.
Notebook metadata retains only the accepted public kernelspec and
language-info values.

Implementation must clear every committed count and output in a deep copy
before each scientific replay. A replay must never consume, trust, or compare
scientific values from stored outputs.

At least two independent clean-kernel executions must reproduce:

- all seven complete Product digests;
- all fourteen per-example Product digests;
- all fixed configuration and Kernel facts;
- exact source immutability;
- exact delayed-crosstalk tensor digest;
- exact global-PSD tensor digest;
- exact digitizer extrema;
- exact EncodedWaveform retained intervals;
- exact seven output objects; and
- exact seven decoded PNG hashes.

The committed notebook is then the exact normalized output of one such clean
top-to-bottom execution.

## Exact Scope

Design authority may change exactly:

```text
docs/implementation/index.md
docs/implementation/maintenance_23_delayed_crosstalk_global_psd_quickstart.md
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
- `README.md`;
- `AGENTS.md`;
- `CONTRIBUTING.md`;
- every living architecture/validation/parity record;
- every historical work order;
- dependency, metadata, package-version, artifact, and environment bytes; and
- the two exact Design authority records after dispatch.

No public export, compatibility module, profile, workflow factory, data asset,
standalone PNG, second notebook, cache, bytecode, build output, or environment
residue may enter the repository.

## Implementation Route

Implementation uses:

```text
branch:
    codex/maintenance-23-delayed-crosstalk-global-psd

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
2. read the complete work order and exact current notebook proof;
3. modify only the notebook and focused proof;
4. insert the exact delayed-crosstalk preparation pair;
5. preserve every other stable cell ID and exact ordered Product story;
6. install only the seven exact Product headings;
7. construct the unconditioned DelayedCrosstalk Kernel;
8. use exact two-generation Charge branching;
9. construct the exact Design-supplied global PSD tensor;
10. remove all channel-conditioned PSD rows and vocabulary;
11. preserve the seven light shape assertions;
12. preserve the exact seven Product-local `3 x 2` grids;
13. substantively update the focused proof;
14. run focused and complete source tests;
15. run positive Pyright and the exact negative fixture;
16. clear and execute at least two independent temporary notebook copies;
17. freeze new Product, Kernel, support, and image identities;
18. inspect all seven figures independently;
19. normalize and commit one exact executed notebook;
20. verify exact scope, protected bytes, privacy, and artifact hygiene; and
21. commit one immutable direct-child candidate for Design routing.

Implementation must stop and return to Design if:

- the exact 626-value payload cannot reproduce the frozen PSD hash;
- DelayedCrosstalk cannot remain unconditioned and Time-relative;
- the global PSD cannot broadcast through the current public Product API;
- two generations require a production, Config, RNG, or public-surface change;
- the seeded demonstration produces no visible delayed descendants;
- global PSD noise hides the pulse/Charge story;
- any Product leaves exact shape `(2, 3, 5000)`;
- committed outputs cannot be normalized without private/transient metadata;
- the required changes need any protected path; or
- any accepted TensorDSLab/TensorCore contract is contradicted.

Implementation must not contact Validation or Review directly.

## Focused Proof

`tests/test_readout_demo.py` must be reconciled, not merely layered with more
obsolete assertions.

It must prove:

- exact `48 / 24 / 24` inventory and strict alternation;
- exact stable old IDs plus the one exact new Markdown/code pair;
- exact ordered code-cell inventory;
- exact seven Product-only level-two headings;
- absence of every former non-Product heading;
- imports/axes/plot setup remain before the first Product section;
- exact delayed-crosstalk preparation/semantic separation;
- exact delayed coefficient shape, dtype, values, sum, offsets, and digest;
- exact empty DelayedCrosstalk conditioning tuple;
- exact Time-relative OffsetAxis operation tuple;
- exact two-generation Config policy;
- exact global PSD shape, dtype, values, sum, DC law, zero count, and digest;
- exact empty PSD conditioning tuple;
- exact FrequencyAxis operation tuple;
- absence of all three retired sensor-PSD names and row stack;
- one shared PSD law with distinct realized lane noise;
- exact source deposits and source immutability;
- exact seven Product shapes `(2, 3, 5000)`;
- newly frozen complete and per-example Product digests;
- newly frozen ADC extrema and EncodedWaveform retained intervals;
- at least one seeded delayed descendant;
- no cross-example or cross-channel descendant leakage;
- exact Product-local plot geometry, colors, styles, titles, and labels;
- exact seven display owners and exact output types;
- exact counts `1..24`;
- newly frozen exact seven PNG hashes;
- exact normalized cell/output/notebook metadata;
- clear-before-replay behavior;
- no final combined/summary plot;
- no extra assertion outside `shared-shape-code`; and
- every changed scientific and visual fact is reproduced from execution rather
  than inferred from source spelling alone.

The proof must not add:

- broad `Any`;
- cast;
- ignore;
- type suppression;
- generated test source;
- dynamic notebook rewriting except the existing deliberate clear-before-run
  execution copy and bounded adversarial mutants;
- donor import;
- filesystem/network dependency;
- application package import; or
- production import of demo tooling.

## Required Adversarial Evidence

Validation must independently kill at least these focused mutations:

1. remove the delayed-crosstalk Kernel;
2. change its conditioning axes to ChannelAxis;
3. include offset zero;
4. change the final offset from `250`;
5. change the decay constant from `150.0`;
6. change the expected offspring mean from `0.15`;
7. change generations from `2` to `1`;
8. restore empty ChargeKernels;
9. restore the old three channel PSD rows;
10. condition the global PSD on ChannelAxis;
11. make the PSD tensor `(1, 2501)` rather than `(2501,)`;
12. change one represented PSD cell;
13. make all generated noise lanes equal after broadcasting;
14. restore any former non-Product heading;
15. remove one Product heading;
16. merge delayed math into the semantic Charge cell;
17. merge PSD values into the semantic NoiseWaveform cell;
18. add an assertion outside the shared-shape cell;
19. remove one delayed descendant through a stale output snapshot;
20. retain one old Product digest or image hash;
21. clear one committed output;
22. add one output to a non-view cell;
23. bypass clear-before-replay;
24. accept an image by count without exact decoded bytes; and
25. introduce a private path, task identifier, or execution timing metadata.

Mutations exist only in private temporary copies. Candidate bytes must remain
immutable.

## Validation Route

Validation receives one immutable Design-dispatched candidate and runs:

1. exact commit/tree/parent/ref/direct-child identity;
2. exact two-path executable scope and all protected-byte equality;
3. exact notebook inventory, IDs, headings, counts, outputs, and metadata;
4. focused source and exact extracted-archive demo tests;
5. complete source and exact extracted-archive test suites;
6. positive Pyright and the exact fifteen-diagnostic negative fixture in both
   dependency forms;
7. independent reconstruction and hashing of delayed-crosstalk values;
8. independent reconstruction and hashing of the complete global PSD;
9. all twenty-five required focused mutants;
10. two immediate clean-kernel CPU executions;
11. exact equality of execution summaries and output objects;
12. exact Product/per-example digests and lane independence;
13. exact ADC extrema and EncodedWaveform retained intervals;
14. exact seven committed decoded PNG hashes;
15. original-resolution inspection of every Product grid;
16. exact normalized committed-output comparison;
17. public-import/downstream-isolation checks;
18. privacy, cache, bytecode, build, and artifact hygiene; and
19. final exact detached cleanliness.

Artifact rebuilding and a fresh Conda environment are not required because
package, dependency, metadata, environment, and production bytes are
protected.

Validation must not infer artifact, installation, CUDA, performance,
calibration, compatibility, release, deployment, or publication claims.

Validation returns CLEAR or one consolidated finding packet to Design only.
It must not contact Review.

## Review And Merge

Review remains independent and read-only until:

1. Validation clears one exact immutable candidate;
2. Design dispatches that exact candidate to Review;
3. Review returns CLEAR with zero findings; and
4. Design issues final same-byte approval.

Review's risk-based obligations include:

- exact identity, scope, and protected-byte checks;
- direct review of the Product-only heading story;
- direct review of preparation/semantic separation;
- direct semantic review of unconditioned delayed-crosstalk geometry;
- direct semantic review of unconditioned global PSD geometry;
- independent focused execution against exact TensorCore source/archive;
- exact Kernel and complete Product digest reproduction;
- direct lane-independence review;
- direct visual inspection of all seven figures;
- confirmation that the new Charge effect is visible but not misleading;
- confirmation that the global PSD is presented without false calibration or
  provenance claims;
- exact committed-output normalization review; and
- evidence-economy acceptance or targeted rerun of complete Validation
  evidence.

After final same-byte approval, Review owns only:

```text
git merge --ff-only <exact-approved-candidate>
```

on clean governed local `main`, followed by exact identity, tree, lineage,
diff/check, worktree-cleanliness, notebook hash/output verification, and
origin-relation verification.

No merge commit, amend, rewrite, rebase, squash, push, tag, branch deletion,
CUDA run, artifact rebuild, environment creation, or publication is
authorized.

If governed local `main` contains an uncommitted notebook edit when final merge
authority is issued, Review must stop without discarding, stashing, rewriting,
or overwriting it. Design must reconcile that user-owned byte before merge.

## Completion

Maintenance 23 is complete only when:

- the notebook has exactly seven Product section headings;
- every preparation/semantic/assertion/presentation boundary remains visible;
- Charge uses the exact global delayed-crosstalk Kernel;
- Charge uses exactly two correlated-avalanche generations;
- NoiseWaveform uses one exact global PSD Kernel;
- neither coefficient Kernel has conditioning axes;
- all seven Products retain exact `(2, 3, 5000)` shape;
- global coefficient laws still produce independently addressed lane values;
- every Product plot remains local, readable, and committed;
- complete scientific and visual identities reproduce on the frozen stack;
- complete fixed-byte Validation clears;
- independent Review returns zero findings;
- Design approves the same immutable bytes; and
- Review fast-forwards them unchanged to clean local `main`.

CUDA, push, publication, compatibility, calibration, application IO,
reconstruction, performance, release, and deployment remain unauthorized.
