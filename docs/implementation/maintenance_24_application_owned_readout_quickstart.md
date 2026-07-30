# Maintenance 24 Application-Owned Readout Quickstart

Status: **Active / Implementation authorized**.

Stable key:
`TensorDSLab/maintenance-24-application-owned-readout-quickstart`

## Purpose

Add one short final section to the supported newcomer notebook showing how an
application package may compose the seven already-demonstrated Products into
one explicit readout workflow.

The section teaches this boundary:

```text
TensorDSLab:
    reusable Product, Spec, Kernel, Config, and TensorCollection types

application package:
    ReadoutConfig
    Readout
    quickstart_profile()
    explicit Product dependency order
```

The example remains ordinary notebook code. None of these three illustrative
application names enters `tensor_dslab`, a package facade, or a reusable demo
module.

## Exact Baseline

Maintenance 24 starts from synchronized local/live `main`:

```text
commit:
    adbd3ded3bbbc90fb7fee2c2ecb486307e815014

tree:
    f2300d47076bd2eb4a3f19e7b4fbab7acaf904a3

notebook SHA-256:
    9c5ba35300f64edb421a53bff55b7df4777140f057050e67c36885c4506bd055

notebook source-projection SHA-256:
    b9fccd51e0a2afed827a9e21e7da5808c947fd659dae9af5ee7c89ded92db90f
```

The baseline notebook has exactly:

```text
48 cells:
    24 Markdown
    24 code

execution counts:
    1..24

stored displays:
    seven existing Product-local PNG outputs
```

TensorCore remains exact published `0.22.0` at commit
`19bfae35fbc773b55cac7bcd659dda57c4dee6d6`.

## Selected Application Contract

### ReadoutConfig

The notebook defines one frozen, slotted `ReadoutConfig` dataclass holding the
exact six generated-Product Config types:

```text
ChargeConfig
PureWaveformConfig
NoiseWaveformConfig
AnalogWaveformConfig
DigitizedWaveformConfig
EncodedWaveformConfig
```

`Photoelectrons` remains the caller-supplied source and therefore has no entry
in this application Config.

### Readout

The notebook defines:

```python
class Readout(TensorCollection[TensorField[Any]]):
    ...
```

The collection admits exactly:

```text
Photoelectrons
Charge
PureWaveform
NoiseWaveform
AnalogWaveform
DigitizedWaveform
EncodedWaveform
```

Its `create()` method explicitly calls the six generated Product
`create()` methods in dependency order. It does not use recursion, reflection,
a registry, a generic dependency resolver, or hidden package orchestration.

The application method receives the caller's `Photoelectrons`,
`ReadoutConfig`, and `CounterRng` explicitly. Randomness does not become Config
state.

### quickstart_profile

The notebook defines:

```python
def quickstart_profile() -> ReadoutConfig:
    ...
```

It returns a fresh `ReadoutConfig` container holding the exact same six
Product Config instances constructed in the preceding sections. The name is
deliberately application-neutral. A future detector application may provide a
detector-specific profile such as `ds20k_veto()` in its own repository.

### Exact reproduction

One final call:

```python
readout = Readout.create(
    photoelectrons=photoelectrons,
    config=readout_config,
    rng=rng,
)
```

must reproduce the exact Specs and tensor values of the seven Products created
earlier in the notebook. The new Products are fresh values; the proof compares
structural Spec equality and exact tensor equality rather than object
identity.

## Notebook Presentation

The section title is:

```text
## Application-Owned Workflow
```

Every code cell has one immediately preceding short Markdown explanation in
the same plain language as the rest of the notebook.

The section uses five code blocks:

1. define `ReadoutConfig`;
2. define `Readout`;
3. define `quickstart_profile()`;
4. create the complete `Readout`; and
5. prove Config-instance reuse and exact Product reproduction.

The section adds no plot, display, figure, output payload, data asset, second
notebook, package import, or final summary visualization. The seven existing
committed PNG payloads remain byte-identical.

## Governing Boundary Amendment

`AGENTS.md` and `CONTRIBUTING.md` must distinguish:

```text
forbidden:
    package-owned workflow/profile API
    tensor_dslab.readout
    production import of notebook code
    generic orchestration machinery

permitted:
    one explicitly application-owned teaching composition
    defined locally in demos/readout.ipynb
    built only from public TensorCore and TensorDSLab APIs
```

This focused exception changes no Maintenance 15 production ownership rule:
real application packages continue to own their workflow classes, profiles,
retention, IO, and integration.

## Exact Scope

The implementation candidate may change exactly:

```text
AGENTS.md
CONTRIBUTING.md
demos/readout.ipynb
tests/test_readout_demo.py
```

The two Design records are this work order and
`docs/implementation/index.md`.

No `tensor_dslab/**`, dependency, metadata, environment, typing-fixture,
scientific, RNG, Product, Kernel, Config, Spec, or other demo/test byte may
change.

## Required Evidence

Implementation must prove:

- exact allowed-path scope and clean diff checks;
- the notebook retains strict Markdown/code alternation and unique stable IDs;
- all five new code cells have immediately preceding simple Markdown;
- `ReadoutConfig` has exactly the six selected Config fields;
- `Readout` requires exactly the seven selected Product member types;
- `Readout.create()` calls the six generated Products explicitly in the frozen
  dependency order;
- `quickstart_profile()` retains the exact six existing Config instances;
- the one-shot collection reproduces every earlier Spec and tensor exactly;
- the RNG remains an explicit `Readout.create()` input;
- no generic recursion, registry, reflection, graph, package workflow, or
  detector-specific profile appears;
- the complete notebook has exactly seven display outputs and no new plot;
- all seven stored PNG payloads remain byte-identical;
- a privately cleared top-to-bottom notebook execution passes;
- focused and complete source tests pass;
- positive Pyright remains clean and the negative fixture is unchanged;
- public imports, privacy, protected bytes, and repository hygiene pass.

## Non-Goals

This maintenance adds no:

- public `Readout`, `ReadoutConfig`, profile, or workflow export;
- `tensor_dslab.readout` package;
- generic Product base or orchestration engine;
- recursive or reflective dependency discovery;
- application repository;
- detector calibration or DS20k profile;
- IO, persistence, cache, records, or reconstruction;
- plot, figure, stored display, or standalone image;
- dependency, version, environment, CUDA, compatibility, release, or
  publication claim.

## Stop Conditions

Stop and return to Design if the application example requires:

- a production-package change;
- hidden Config mutation;
- RNG state inside `ReadoutConfig`;
- generic dependency discovery;
- altered scientific Product values;
- changed existing plot bytes; or
- weakening the direct Product or application-ownership boundary.
