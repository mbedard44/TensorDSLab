# TensorDSLab

TensorDSLab is a clean-slate, tensor-native detector data-lab package. Its
intended ecosystem data flow is:

```text
G4DS -> TensorG4DS -> TensorDSLab -> TensorML
```

TensorCore is the shared semantic tensor substrate rather than another
pipeline stage. TensorDSLab owns downstream readout and future reconstruction
meaning; it does not parse native G4DS files or own TensorG4DS deposit
clustering, TensorML training, durable IO, or campaign orchestration.

## Implemented Foundation

The package currently provides the TensorCore `0.7` semantic foundation for
post-binned readout:

- `ExampleAxis`, `ChannelAxis`, and regular timestamp-backed `SampleAxis`;
- `SamplingConfig` for zero-start, integer-picosecond left-edge samples;
- `Photoelectrons`, `Charge`, `PureWaveform`, `NoiseWaveform`,
  `AnalogWaveform`, and `DigitizedWaveform` fields;
- product-owned immutable scientific configuration records; and
- `ReadoutCollection`, an immutable completed result containing any nonempty
  unordered subset of the six products.

The exact concrete axis, field, and collection classes carry in-process
semantic identity. There are no parallel axis IDs, field IDs, layout objects,
product registries, or collection sidecars.

```python
import torch
from tensor_core import PositiveInteger

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    Photoelectrons,
    ReadoutCollection,
    SamplingConfig,
)

sampling = SamplingConfig(
    sample_period_ps=PositiveInteger(2_000),
    sample_count=PositiveInteger(4),
)
axes = (
    ExampleAxis(coordinates=("event-0",)),
    ChannelAxis(coordinates=("tile-0",)),
    sampling.build_axis(),
)
photoelectrons = Photoelectrons(
    tensor=torch.zeros((1, 1, 4), dtype=torch.int64),
    axes=axes,
)
readout = ReadoutCollection(fields=(photoelectrons,))

assert readout.field(Photoelectrons) is photoelectrons
```

`Photoelectrons` is an already-produced dense, binned photon-origin truth
input. TensorDSLab does not yet construct it from TensorG4DS data.

## Explicit Exclusions

This foundation does not yet implement `simulate_readout(...)`, scientific
product producers, RNG, PE binning, TensorG4DS or TensorML adapters, IO,
caches, `out=`, workspaces, movement/selection helpers, or an allocation-free
execution path. It makes no GPU-execution, release, deployment,
backward-compatibility, conformance, or broad cross-package compatibility
claim.

Start with [the documentation overview](docs/overview.md) and the
[rebuild architecture](docs/architecture/rebuild.md). Local tests run from the
project root with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
