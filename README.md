# TensorDSLab

TensorDSLab is a clean-slate, tensor-native detector data-lab package.

It is intended to sit between g4ds11 detector simulation and future consumers:

```text
g4ds11 -> TensorDSLab -> future consumers
```

The package should define detector, readout, reconstruction, in-memory product,
and tensor-rendering contracts while using TensorCore as the generic tensor
identity, layout, field, collection, selection, batching, movement, validation,
and pure operation backbone.

The first MVP direction starts at already-binned charge and focuses on the
post-binned readout path: stochastic charge transforms, waveform products,
physical waveform composition, and optional digitization. Source parsing,
charge binning, durable cache, and external integration boundaries come later.

## Current Status

This repository is in initial Design documentation mode. There is not yet a
production package, test suite, cache schema, or implementation branch.

Start with:

- [Agent Workflow](AGENTS.md)
- [Contributing](CONTRIBUTING.md)
- [Overview](docs/overview.md)
- [Design](docs/design.md)
- [Implementation Stages](docs/implementation/index.md)

## Intended Package Shape

```text
Project/display folder: TensorDSLab
Python import package:  tensor_dslab
```

When production code is accepted, local smoke and test commands should run from
the project root with `PYTHONPATH=.`.
