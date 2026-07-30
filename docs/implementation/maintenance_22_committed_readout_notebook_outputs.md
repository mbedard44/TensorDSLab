# Maintenance 22 Committed Readout Notebook Outputs

Status: **Self-effecting under the frozen exact-byte route**.

Before the complete same-byte gate and clean local fast-forward, this status
means the latest completed fixed-byte handoff. After exact unchanged
Validation-cleared, Review-cleared, and Design-approved bytes reach clean
local `main` through Review's `git merge --ff-only`, this same status resolves
to **Merged / Closed**. No post-merge closeout edit is required or authorized.

Stable key:
`TensorDSLab/maintenance-22-committed-readout-notebook-outputs`

## Purpose

Commit the seven generated Product plots in
`demos/readout.ipynb` so a newcomer can open the notebook and immediately see
the complete two-example, three-sensor readout story without first running a
kernel.

Maintenance 21 deliberately committed a source-only notebook and generated
plots only in temporary execution copies. The user has now selected the
opposite presentation boundary for this one supported newcomer notebook:

```text
committed notebook source
    + exact deterministic execution counts
    + seven embedded Product figures
```

The notebook remains executable ordinary public-API user code. Committed
outputs are a reproducible presentation snapshot, not a scientific artifact,
calibration record, cache, durable Product representation, or package workflow
result.

## Exact Baseline

Maintenance 22 starts from exact locally closed Maintenance 21:

```text
commit:
    e71974239c4ec04d6b138ade5fe346c47f85172f

tree:
    0c660682d92c0c60c404e848ee3eee193ccf3257

exact parent / Review-cleared Maintenance 21 Candidate 1:
    98db48138800b8500d587389e5e159c11f252a73
```

The baseline notebook is:

```text
demos/readout.ipynb

SHA-256:
    281d385e60747d15e3ae85b26dd69def9f64484b430b10273f81b6423c961038

cells:
    46

Markdown:
    23

code:
    23

committed outputs:
    0
```

Maintenance 22 changes no Product source, public API, dependency, metadata,
environment script, scientific equation, numerical input, RNG address,
Product value, axis, Spec, Kernel, Config, plot construction, or notebook
source text.

## Governing Standards

Implementation, Validation, and Review must apply:

- `AGENTS.md` for role separation, exact-byte routing, application ownership,
  privacy, and local-main merge authority;
- `CONTRIBUTING.md` for deterministic evidence, focused tests, documentation,
  and honest qualification;
- `docs/overview.md` for the current newcomer demonstration boundary;
- `docs/validation.md` for notebook execution, artifact, and visual evidence;
- Maintenance 15 for direct Product composition and application ownership;
- Maintenance 18 for the exact EncodedWaveform result;
- Maintenance 19 for math/semantic/assertion/presentation cell ownership;
- Maintenance 20 for Product-local inspection;
- Maintenance 21 for the two-example `3 x 2` Product-grid contract.

Historical source-only requirements remain truthful evidence for their exact
older candidates. They are not edited. This Maintenance supersedes that
presentation rule only for the current `demos/readout.ipynb`.

## Selected Representation

### One notebook, not a second checked-in artifact

The repository continues to contain exactly one supported readout notebook:

```text
demos/readout.ipynb
```

Do not add:

- a second `readout_executed.ipynb`;
- checked-in standalone PNG files;
- an output directory;
- an HTML export;
- a PDF export;
- an image attachment;
- a notebook cache;
- a demo data file; or
- a generated manifest.

The seven PNG payloads live only in the seven corresponding notebook
`display_data` outputs.

### Exact cell grammar is unchanged

The committed notebook retains exactly:

```text
46 cells
23 Markdown cells
23 code cells
strict Markdown/code alternation
the exact Maintenance 21 cell IDs and order
```

Every source string, cell ID, cell type, and Markdown explanation is
Maintenance-21-identical. Output commitment must not be used as authority to
rewrite or reformat source.

### Exact execution counts

The 23 code cells have exact execution counts:

```text
1, 2, 3, ..., 23
```

Markdown cells have no execution count.

Counts record one clean top-to-bottom execution. They do not imply an
interactive editing history.

### Exact output ownership

Only these seven code cells own committed outputs:

```text
photoelectrons-view-code
charge-view-code
pure-waveform-view-code
noise-waveform-view-code
analog-waveform-view-code
digitized-waveform-view-code
encoded-waveform-view-code
```

Each owns exactly one output:

```text
output_type:
    display_data

data keys:
    image/png
    text/plain

text/plain:
    <Figure size 1300x850 with 6 Axes>

metadata:
    {}
```

The other 16 code cells own no output.

The notebook contains no:

- `stream`;
- `execute_result`;
- `error`;
- widget state;
- JavaScript;
- HTML;
- SVG;
- external image reference;
- cell attachment; or
- filesystem/network link to a generated plot.

### Exact seven embedded images

One clean execution on the frozen local stack produces the seven images in
Product order:

```text
Photoelectrons
    3f5814fa5c83b856ecdb5a15def832251cd0a5cff91d3dfbda21bd44bceb9139

Charge
    680c6231d009a9d6c590ee060bd5ddd1d4aad79950411e677a1e3da53cc2854c

PureWaveform
    924a95d4c97e00d3cc6fde320e03b548cf323d24c0ddc8e7285560fa35809a1f

NoiseWaveform
    69bf7451031461ad9a3baad9c70f6b53b7f733a1df3a7c5213e0849bae64e08b

AnalogWaveform
    74f5b46839ad9ee3263caba3fac21705fb1e3606a2e25e0d688cb1c655b040c3

DigitizedWaveform
    fdd07c0a2a187aebe7b10239f297efafa96ba9ced98377278c626dc5d12479f0

EncodedWaveform
    dc7df1661bd9e5fca09a886ebae8d84f56b07c010f62066890a7a98f64e52e0d
```

Hashes are over decoded PNG bytes, not base64 text.

These exact hashes bind only the recorded eager-CPU dependency stack:

```text
CPython 3.14.6
PyTorch 2.13.0
NumPy 2.5.1
Pint 0.25.3
Matplotlib 3.11.1
ipykernel 7.3.0
nbclient 0.11.0
nbformat 5.10.4
TensorCore 0.22.0 at exact accepted commit
```

They are not a cross-version Matplotlib compatibility promise.

### Normalized metadata

Every cell metadata mapping is exactly empty:

```text
{}
```

Notebook metadata retains only the existing public `kernelspec` and
`language_info` values. It contains no:

- execution timing;
- timestamp;
- username;
- hostname;
- absolute path;
- task/route identifier;
- widget state;
- editor state;
- kernel identifier;
- environment path; or
- private operational record.

The notebook contains no attachments.

### Fresh execution remains authoritative

Committed output is a convenience snapshot. Tests must still execute a fresh
deep copy from the committed source.

Before fresh execution, the test harness explicitly clears execution counts
and outputs from that copy. This proves the result from code, not from the
stored snapshot. The fresh run must recreate:

- the exact seven Product values and digests;
- the exact two-example lane semantics;
- the exact EncodedWaveform support;
- the exact seven figures;
- the exact decoded PNG hashes; and
- the exact normalized committed output structure.

The committed notebook source itself must remain byte-unchanged during every
test execution.

## Newcomer Contract

The committed plots exist solely to improve first-open readability.

Living documentation must explain:

- the notebook contains two independent examples and three sensors;
- all seven Products are built through public TensorDSLab APIs;
- every Product is shown in its own `3 x 2` grid;
- the committed figures let readers inspect the story before running;
- rerunning the notebook deterministically refreshes the same demonstration on
  the frozen stack; and
- the figures are illustrative, not calibrated detector output.

Do not describe the committed images as:

- canonical data;
- durable artifacts;
- expected cross-platform PNG bytes;
- performance evidence;
- detector calibration;
- a package-owned workflow;
- a compatibility promise; or
- a release asset.

## Exact Implementation Scope

Implementation may change exactly:

```text
AGENTS.md
CONTRIBUTING.md
README.md
demos/readout.ipynb
docs/design.md
docs/overview.md
docs/validation.md
tests/test_readout_demo.py
```

The Design authority itself changes exactly:

```text
docs/implementation/index.md
docs/implementation/maintenance_22_committed_readout_notebook_outputs.md
```

No other path is authorized.

In particular, protect every byte under:

```text
tensor_dslab/
pyproject.toml
create_environment.sh
tests/typing/
```

Protect every other test, demo, documentation, governance, historical
work-order, dependency, metadata, and environment path.

## Implementation Requirements

Implementation must:

1. start from the exact Design authority;
2. execute the exact notebook once from top to bottom on the frozen local
   eager-CPU stack;
3. retain only normalized execution counts and the seven exact `display_data`
   outputs;
4. remove all transient execution metadata;
5. make no notebook source, ID, type, or order change;
6. update the focused proof from source-only admission to the exact committed
   output contract;
7. make fresh replay explicitly clear stored outputs in its private copy;
8. compare fresh replay output structure and decoded PNG hashes with the
   committed snapshot;
9. update the exact six living/current documentation paths in scope;
10. preserve every Product value, digest, axis, Unit, dtype, device, support,
    and plot artist contract;
11. run focused and complete source-form tests;
12. run positive Pyright and the unchanged exact negative fixture;
13. record the fixed notebook SHA-256 and byte size;
14. commit one immutable direct-child candidate; and
15. return only to Design.

Implementation must not contact Validation or Review, merge, push, execute
CUDA, build a release, or broaden scope.

## Focused Test Contract

`tests/test_readout_demo.py` must continue to own exactly three discovered
test methods. It may refactor private helpers but must not add dynamic or
generated tests.

The source/metadata test must prove:

- exact `46 / 23 / 23` grammar and IDs;
- exact source equality to Maintenance 21;
- code execution counts exactly `1..23`;
- exactly seven output-owning view cells;
- exactly one `display_data` output per view cell;
- exact `image/png` plus `text/plain` keys;
- exact figure text/plain value;
- exact empty output/cell metadata;
- exact decoded PNG hashes;
- zero outputs on every other code cell;
- zero attachments;
- no forbidden private/transient metadata; and
- public-import and newcomer prose contracts remain exact.

The scientific/source test continues to prove all Maintenance 21 Product,
axis, Kernel, Config, shape, plot, masking, and ownership facts.

The execution test must:

1. deep-copy the committed notebook;
2. clear every code-cell count and output in that copy;
3. append its private probe;
4. execute from top to bottom twice;
5. require identical summaries;
6. require all exact Product and lane digests;
7. require the exact seven plot structures;
8. compare the first 46 executed cells' normalized output structure with the
   committed notebook;
9. compare decoded PNG hashes with the exact seven committed hashes; and
10. prove the repository notebook bytes did not change.

## Required Adversarial Evidence

At minimum, the fixed suite must fail for each private mutation:

1. clear all committed outputs;
2. clear one committed Product output;
3. change one PNG byte;
4. move one image to the wrong Product cell;
5. add an output to a preparation/semantic/assertion cell;
6. replace `display_data` with `execute_result`;
7. add a `stream` output;
8. add an `error` output;
9. remove `text/plain`;
10. change the figure text/plain value;
11. make one execution count `None`;
12. duplicate or skip one execution count;
13. add execution timing metadata;
14. add a private absolute path;
15. add an attachment;
16. preserve stale committed images while changing executable source;
17. let fresh replay consume stored output rather than clearing it; and
18. weaken the exact seven-image hash boundary to image-count-only admission.

Mutation bytes remain private temporary evidence only.

## Validation Gate

Validation receives one immutable candidate from Design and must independently:

### Identity and scope

- bind exact commit, tree, parent, and feature ref;
- prove direct-child topology from the Design authority;
- prove the exact allowlist and protected-byte equality;
- run `git show --check` and cumulative `git diff --check`;
- verify the checkout remains detached, clean, and immutable.

### Notebook structure

- independently parse the committed notebook;
- verify exact cells, IDs, source, counts, output owners, output types, data
  keys, text, metadata, attachment absence, and decoded image hashes;
- verify the committed notebook contains no private/transient state;
- compare all source strings and IDs byte-for-byte with Maintenance 21.

### Runtime and typing

- run focused source and exact extracted-archive forms;
- run complete source and exact extracted-archive forms;
- require the existing test inventory and all tests pass;
- run positive Pyright in both forms with zero diagnostics;
- require the unchanged negative fixture diagnostic count and sites;
- execute two privately cleared notebook copies and compare complete summaries,
  normalized outputs, and decoded PNG hashes.

### Visual inspection

- independently decode and inspect all seven committed PNGs at original
  resolution;
- verify the same `3 x 2` grids, distinct Product colors, titles, Example
  columns, sensor rows, time labels, pulses/noise/ADC/support gaps, and absence
  of legends/final summary plot;
- verify no image is blank, truncated, corrupt, swapped, stale, or mismatched
  with its Product cell.

### Artifacts

- build two deterministic wheels and sdists from independent exact archives;
- require pairwise byte identity;
- prove wheel package bytes are candidate-exact;
- prove the sdist contains the exact committed notebook and focused proof;
- install exact TensorCore and TensorDSLab wheels outside every checkout;
- execute the exact sdist notebook from the isolated environment;
- prove installed-path/no-shadowing behavior;
- compare its seven output hashes with the committed snapshot where the exact
  frozen plotting stack applies.

A fresh Conda environment is unnecessary if dependency, metadata, and
environment-script bytes are exact Maintenance-21-identical. Carry-forward
must be stated narrowly and supported by byte equality.

### Documentation and hygiene

- verify every changed living link and Markdown fence;
- verify current docs describe committed outputs and two examples accurately;
- verify historical source-only work orders remain unchanged;
- scan for private paths, raw route IDs, outputs outside the notebook, caches,
  bytecode, build residue, and partial environments;
- rebind exact final identity and cleanliness.

### Qualification

Evidence is eager CPU-only. CUDA is unavailable and outside scope.

Validation returns `CLEAR` or consolidated findings to Design only. It must
not contact Review, edit the candidate, merge, push, or infer compatibility.

## Review Gate

After Validation `CLEAR`, Design may dispatch the exact unchanged candidate to
the persistent Review route.

Review must independently:

- bind exact identity, topology, and allowlist;
- inspect the notebook's committed-output representation;
- compare source and scientific facts with Maintenance 21;
- decode and inspect all seven committed figures;
- ensure fresh replay, not stored output, remains the scientific oracle;
- audit exact image hashing without overclaiming cross-stack compatibility;
- verify current documentation and historical-record boundaries;
- use evidence economy only for exact unchanged package/dependency inputs;
- return `CLEAR` or one consolidated finding set to Design.

Review remains read-only before final same-byte Design approval.

After Review `CLEAR`, Design may approve only the exact Validation- and
Review-cleared bytes. Review then owns the sole authorized merge:

```text
git merge --ff-only <exact-approved-candidate>
```

from exact clean governed local `main`, followed by identity, tree, parent,
lineage, notebook/output/hash, diff/check, lifecycle, origin-relation, and
cleanliness verification.

## Finite Return Loop

The route admits:

```text
Implementation candidates:
    at most 3 ordinary candidates

Validation returns:
    at most 3

Review returns:
    at most 2
```

Any contradiction in notebook determinism, image-byte stability on the frozen
stack, test scope, artifact inclusion, or protected bytes returns to Design.
Do not silently weaken exact evidence or add a second output representation.

## Explicit Non-Goals

Maintenance 22 does not:

- change any TensorDSLab production byte;
- change a Product, Spec, Kernel, Config, axis, Unit, dtype, or device law;
- change scientific inputs or results;
- add a package workflow, profile, orchestration surface, or cache;
- add a second notebook or standalone plot;
- add IO, reconstruction, calibration, performance, or compatibility claims;
- alter dependencies, metadata, or environment setup;
- qualify CUDA or another backend;
- push, tag, publish, release, or deploy.

## Completion

Maintenance 22 is complete only when:

- the exact single notebook contains the normalized top-to-bottom execution;
- all seven exact Product figures are visible on first open;
- every source byte and scientific result remains Maintenance-21-identical;
- the committed/fresh output structures and exact PNG hashes agree;
- focused and complete source/archive evidence clears;
- positive and negative typing boundaries clear;
- deterministic artifact and isolated sdist-notebook evidence clears;
- current documentation truthfully supersedes the source-only rule;
- fixed-byte Validation returns `CLEAR`;
- independent Review returns `CLEAR`;
- Design approves the exact same bytes; and
- Review cleanly fast-forwards them unchanged to local `main`.

No push, tag, publication, release, deployment, CUDA, calibration,
performance, or broad compatibility claim follows.
