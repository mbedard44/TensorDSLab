# Maintenance 10 IV-Like Digitizer And Demo Source

Status: Active fixed work order; execution requires exact role handoffs

Stable key:
`TensorDSLab/maintenance-10-iv-like-digitizer-and-demo-source`

## Authority And Baseline

This focused maintenance starts from exact clean Maintenance 9 closeout
`3466c2d59be359ffa537848c8abba4d1405f338f`, tree
`36e254a15bc616ef592e59ca233ce8fe33cdfc28`. It retains exact published
TensorCore `0.16.0` containing commit
`e05324699892a8bcea024375720bfae1ed9569cc`, CPython `3.14.6`, PyTorch
`2.13.0`, NumPy `2.5.1`, Pint `0.25.3`, Hatchling `1.31.0`, and Pyright
`1.1.411`.

The user accepted this refinement before TensorDSLab's first GitHub push:

- use an IV-DSLab-like illustrative digitizer in the provisional
  `ds20k_veto()` profile;
- move the project environment creator from `demos/create_environment.sh` to
  repository-root `create_environment.sh`;
- extend the demo source window to `10000 ns` at the existing `2 ns`
  period;
- replace the sparse random upstream source with four explicit, separated
  `1`, `2`, `3`, and `4` PE deposits; and
- retain the executed CPU notebook, waveform-chain plot, and black Charge
  overlays as reviewed development-preview evidence.

The earlier unmerged development-preview candidates
`bc7729c01c28478f6a2cdf670a1a6e9def243b9b` and
`d2221fde7f7151043e1c5208061ede6152493926` are superseded stopped evidence.
Neither was cleared, merged, or pushed. Their branches and commits must not be
rewritten or used as the Maintenance 10 base.

This work order is authoritative for the exact refinement only. It cites
`CONTRIBUTING.md` for public Config, semantic product, validation, typing,
documentation, and role-separation standards and `docs/parity.md` for the
required provisional/non-calibration classification.

## Exact Target

### Provisional Profile Digitizer

`tensor_dslab.readout.profiles.ds20k_veto()` must retain its no-argument
signature, precise-module-only export, fresh complete Config-tree guarantee,
Pint ownership, pulse/noise/charge/analog values, and provisional status.
Only its `DigitizedWaveformConfig` changes:

```python
DigitizedWaveformConfig(
    bit_depth=PositiveInteger(16),
    input_minimum=quantity(-3900.0, "mV"),
    input_maximum=quantity(100.0, "mV"),
    analog_gain_db=NonnegativeFloat(3.5218),
)
```

These are illustrative IV-DSLab-like values, not an approved detector
calibration, donor-equivalence result, or durable hardware profile. The exact
binary64 magnitudes and canonical Pint quantities are part of the provisional
factory contract until a later focused profile decision changes them.

No digitization equation, gain convention, endpoint rule, dtype, Config class,
Runtime, producer, validator, facade, or product type changes. The existing
digitizer still applies:

```text
gain = 10 ** (analog_gain_db / 20)
maximum_code = 2 ** bit_depth - 1
```

with the already accepted input-threshold, clamp, truncation, and signed
`torch.int32` result rules.

### Explicit Demo Source

Both `demos/readout.py` and the source-construction cell in
`demos/readout.ipynb` must construct the same semantic source:

```python
axes = (
    ExampleAxis(count=2),
    ChannelAxis(labels=("veto-0", "veto-1", "veto-2", "veto-3")),
    SampleAxis.from_period(period=quantity(2.0, "ns"), count=5000),
)
shape = tuple(axis.size for axis in axes)
counts = torch.zeros(shape, dtype=torch.int64, device="cpu")
counts[0, 0, 100] = 1
counts[0, 0, 1300] = 2
counts[0, 0, 2500] = 3
counts[0, 0, 3700] = 4
photoelectrons = Photoelectrons(tensor=counts, axes=axes)
```

The exact source facts are:

- ordered shape `(2, 4, 5000)`;
- source period `2000 ps`;
- exclusive source stop `10_000_000 ps`, described as `10000 ns`;
- deposits at `200`, `2600`, `5000`, and `7400 ns`;
- respective values `1`, `2`, `3`, and `4` PE;
- exactly four nonzero source cells and total source count `10`;
- no source RNG, `torch.randint`, global RNG mutation, implicit device,
  accelerator query, IO, or hidden source factory.

The selected demonstration trace is example `0`, channel `0`, so all four
explicit deposits are visible. The active provisional `100 kHz` dark-count
model may add seeded Charge cells; tests and narration must distinguish the
four exact upstream `Photoelectrons` deposits from later modeled Charge
events. They must not claim that the completed `Charge` trace contains only
four nonzero cells.

### Project Environment Script

Move `demos/create_environment.sh` to repository-root
`create_environment.sh` with executable mode `100755`. The old path must be
absent, without a forwarding script, symlink, alias, duplicate, or deprecation
shim.

The script's environment contract remains unchanged:

- optional positional environment name, defaulting to `tensor_dslab`;
- exact CPython `3.14.6`;
- Conda Forge only, no default packages;
- non-editable installation of the local `.[demos]` extra;
- refusal to replace an existing environment;
- exact dependency/profile smoke check;
- no shell activation, kernel registration, CUDA selection, or project
  mutation; and
- printed follow-up activation, repository-change, and demo-run commands.

Because the script now lives at repository root, its resolved
`repository_root` is its own directory, not its parent. It must work when
invoked from any current directory as:

```bash
./create_environment.sh [environment-name]
```

The notebook opening Markdown and every live setup instruction/proof must use
`./create_environment.sh`. Historical Maintenance 9 records retain the old
path as a truthful record of the closed stage and must not be rewritten.

### Manual/Profile Equivalence

The notebook's manual `ReadoutConfig` must use the exact profile digitizer
above. Its complete recursively normalized value signature must equal a fresh
`ds20k_veto()` result. The script and notebook must both use a fresh profile
and `Threefry4x32(seed=17)` for readout execution.

The profile remains responsible only for the Config tree. Source axes,
source values, products, device, floating dtype, RNG instance/seed, retention,
and visualization remain explicit caller-owned choices.

### Plot And Stored Notebook

The notebook must retain exactly four aligned panels:

```text
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
```

It must still prove exact
`AnalogWaveform == PureWaveform + NoiseWaveform` before plotting. The Analog
panel displays only `AnalogWaveform`, with no legend and no redundant
`Pure + Noise` curve. Secondary axes on Pure, Analog, and Digitized display
faint black delta-like spikes for every nonzero completed Charge cell. Noise
has no Charge overlay. All Charge axes, curves, labels, ticks, and visible
spines use black; no purple Charge styling is accepted.

The x-axis must cover the full source window in nanoseconds. Stored notebook
outputs must come from a real execution in the exact `tensor_dslab` Conda
environment created by `create_environment.sh`, using CPU only. The
committed notebook must have:

- exactly `23` cells and `11` code cells;
- consecutive execution counts `1` through `11`;
- exactly seven intentional outputs;
- one embedded PNG and zero error outputs;
- no per-cell execution timestamps;
- no home path, temporary path, private route, user identity, token, or
  credential; and
- byte-identical output under one immediate replay after the Matplotlib cache
  is warm.

The final code-source and whole-notebook SHA-256 values are implementation
evidence and must be frozen in the candidate test and closeout record only
after the final bytes execute successfully.

## Provisional Parity Classification

`docs/parity.md` must update only the Maintenance 9 provisional-profile table
and append a bounded Maintenance 10 refinement note:

- ADC bit depth: `16`;
- ADC input interval: `[-3900, 100] mV`;
- analog gain: `3.5218 dB`;
- sampling: start `0`, period `2 ns`, count `5000`, exclusive stop
  `10000 ns`;
- toy source: explicit `1`, `2`, `3`, and `4` PE deposits at the exact times
  above plus the fixed readout seed.

The digitizer values are classified **Deferred** illustrative IV-DSLab-like
choices. The source/grid/plot remain **Not applicable** to detector
calibration and donor-event parity. Nothing in this maintenance upgrades
existing waveform numerical parity, establishes digitizer calibration,
claims IV eventwise parity, or changes any scientific equation.

## Public And Package Boundaries

Preserve exactly:

- TensorDSLab package/readout/common facade counts `35/5/30`;
- precise `tensor_dslab.readout.profiles.__all__ == ("ds20k_veto",)`;
- exact production-module count `60`;
- exact public-class/function census `32/4`;
- TensorCore public topology and imports;
- NumPy/Pint privacy at Runtime, producer, and validator boundaries;
- all product/axis/collection identities and public signatures;
- all fixed RNG namespace, key, address, word, and result-law behavior;
- `demos` as the only optional Matplotlib/Jupyter dependency boundary; and
- CPU-only demo operation without CUDA inspection or selection.

Do not add a profile registry, override engine, source factory, calibration
store, environment mutation cell, file IO, artifact, TensorG4DS/TensorML
adapter, alternate ADC implementation, compatibility shim, or new export.

## Exact Candidate Allowlist

Implementation may change only:

- `tensor_dslab/readout/profiles.py`
- `create_environment.sh`
- `demos/create_environment.sh` (deletion half of the exact move)
- `demos/readout.py`
- `demos/readout.ipynb`
- `tests/test_readout_profiles_and_demos.py`
- `docs/api.md`
- `docs/parity.md`
- `docs/implementation/index.md`
- `docs/implementation/maintenance_10_iv_like_digitizer_and_demo_source.md`

The production profile, environment-script move endpoints, and two demo paths
own implementation behavior. The one test path owns committed proof.
Documentation changes synchronize only the exact provisional values,
source/grid, setup path, qualification, evidence, and lifecycle-neutral
candidate state.

All other production, test, demo, dependency, metadata, architecture,
governance, historical-work-order, parity sections, and package bytes are
protected. If a required check is owned by a protected path, Implementation
must stop and return the exact contradiction to Design rather than widen
scope.

## Implementation Requirements

Implementation must:

1. start from the exact committed Design authority and remain its clean direct
   child for Candidate 1;
2. update the profile, both demo forms, synchronized public API/parity text,
   exact environment-script move, and the one exact proof module;
3. run `bash -n create_environment.sh`, verify mode `100755`, independently
   exercise its fake-Conda contract, and create one fresh role-private real
   environment from outside the repository;
4. execute `demos/readout.py` from the project root in the real
   `tensor_dslab` environment;
5. execute `demos/readout.ipynb` in place with nbclient, remove execution
   timestamp metadata, verify one PNG/zero errors, and replay it immediately;
6. visually inspect the embedded figure for the four amplitude levels, full
   `10000 ns` window, black Charge spikes, Analog-only panel, and legibility;
7. freeze exact source/notebook hashes only after the final replay;
8. run the focused and full accepted source/archive suites, exact Pyright,
   package/import/privacy, Markdown/link/fence, diff, and hygiene gates; and
9. commit one immutable candidate and dispatch it to persistent Validation.

No cluster allocation is authorized. The candidate must record the existing
`13` TensorDSLab and two TensorCore unavailable-CUDA skips without creating an
accelerator claim.

## Required Proof

Committed tests must prove:

- exact profile digitizer wrapper types, canonical units/magnitudes, freshness,
  and complete manual/profile signature equality;
- exact root environment-script path/mode/content, absent old path, correct
  repository-root resolution, unchanged fake-Conda behavior, and updated
  notebook setup instruction;
- exact script/notebook AST for `torch.zeros(...)` and the four assignments;
- shape `(2, 4, 5000)`, exact SampleAxis state/stop, source tensor identity,
  exact nonzero coordinates/values, total `10`, and unchanged global RNG;
- script/notebook source equivalence and CPU-only execution;
- exact public requests, axes, dtype, device, retention, and seeded replay;
- exact analog recomposition and plot source/forbidden styling;
- exact notebook cells, executions, outputs, PNG/error count, hashes, privacy,
  and immediate replay identity;
- unchanged public/import/package contracts; and
- mutation resistance against one changed digitizer value, one missing or
  moved source deposit, restored random source construction, shortened sample
  grid, manual/profile drift, Analog overlay/legend restoration, or nonblack
  Charge styling.

Tests must not assert completed Charge nonzero-cell count because the accepted
dark-count law owns additional seeded events.

## Validation And Review

The bounded role loop is:

```text
Implementation -> Validation -> Review
```

Candidate submissions and Validation returns are limited to three each.
Validation must independently reconstruct the exact dependency source/archive
forms and execute the candidate's focused/full CPU, typing, notebook, script,
contract, privacy, and hygiene matrix. Review independently rechecks exact
candidate bytes, proof strength, visual output, scope, and evidence.

On CLEAR, Review may fast-forward the exact unchanged candidate to local
`main`. Design may then create one evidence-only direct-child closeout limited
to this work order and `docs/implementation/index.md`; Review must verify and
fast-forward that exact closeout before publication.

## Qualified Development Preview Push

After the exact candidate and its evidence-only closeout are independently
cleared and fast-forwarded, the user authorizes one ordinary non-force push of
exact local `main` to `origin/main` as a pre-deployment development preview.
Design must verify local `HEAD`, local `main`, the tracking ref, and live
GitHub `refs/heads/main` all resolve to the exact pushed commit.

The publication is:

- eager-CPU qualified on the exact named stack;
- CUDA-unqualified because the accepted conditional CUDA cases remain
  unavailable;
- API-unstable and suitable only for collaborators pinning the exact commit;
- not a tag, GitHub release object, package-index publication, deployment,
  calibration, accelerator claim, performance claim, or broad compatibility
  certification.

The deferred integrated TensorCore/TensorDSLab CUDA matrices remain required
before any later accelerator or broad compatibility claim. No tag, release
object, PyPI upload, force push, cluster submission, or post-publication byte
change is authorized here.
