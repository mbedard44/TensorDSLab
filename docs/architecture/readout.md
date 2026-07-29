# Product-Composition Architecture

Maintenance 15 is the current candidate contract. Its detailed authority is
the [architecture record](../implementation/maintenance_15_spec_composed_products_and_application_boundary.md)
and [executable work order](../implementation/maintenance_15_execution_work_order.md).

## Boundary

TensorDSLab is a reusable Product library:

```text
ordered source Products + one Product Config -> one Product
```

It is not an application-owned readout graph. Applications decide which
sources exist, which Products run, in what order, and which results are
retained. The former generic readout Config, collection, orchestration,
profile, and demos are absent without compatibility aliases.

## Preparation And Execution

Each generated Product follows:

```text
prepare(source Specs, Config) -> fresh prepared same-type Config
produce(source Fields, prepared Config[, RNG]) -> Product
validate(Product, source Fields, prepared Config)
```

Preparation validates source count, exact quantity Specs, semantic role sets,
coordinates, device, units, coefficient membership, geometry, dtype policy,
allocation/count ceilings, and RNG-address capacity. It aligns Kernel
conditioning coordinates and dimension order, performs coefficient unit and
device conversion, and retains exact positional source-Spec provenance.

Production first rechecks generic quantity-source admission and positional
structural equality. Only then may it convert tensors, allocate, perform
arithmetic, or request RNG words. It consumes no Pint and performs no role or
coordinate discovery. Validation applies the same source binding and the
completed Product/storage relationships.

## Scientific Products

- `Charge` performs ordered integer avalanche-source accumulation, literal
  multinomial timing displacement, dark-count Poisson draws, fixed-generation
  collapsed destination-rate branching, and optional Gaussian smearing.
- `PureWaveform` performs literal signed `PulseResponse` convolution.
- `NoiseWaveform` implements exact zero, Gaussian white, and prepared-bin PSD
  synthesis.
- `AnalogWaveform` performs ordered unit-aware addition and optional literal
  saturation.
- `DigitizedWaveform` applies literal bounds, gain, integer BitDepth, truncation,
  and clipping.
- `EncodedWaveform` applies deterministic raw negative-going trigger/time-over,
  release-hysteresis, padding, and dense-union support to one exact
  DigitizedWaveform. It preserves retained codes and uses its Spec-owned
  negative suppression code elsewhere.

Encoded-waveform preparation requires one exact source, identical ordered
axes/shape/device/unit/signed dtype, one regular step-one TimeAxis, and aligned
non-Time int64 policies. Production flattens non-Time dimensions into
independent lanes, derives support with tensor prefix/reduction/event
operations, and restores the original dimension order. It has no RNG, donor
call, host-list state, record object, or reconstruction responsibility.

The eight private stochastic roles retain namespace `0x54445331` and streams
one through eight. Products use only public TensorCore `RngElements`,
`RngAddress`, Distributions, and `CounterRng`.

The candidate is CPU-qualified only. It makes no CUDA, application,
compatibility, calibration, release, or production-readiness claim.
