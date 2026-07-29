# Public API

Maintenance 15 replaces the embedded readout workflow with direct reusable
Products. The supported root facade has exactly 75 ordered names and the
subpackage facades are deliberate precision paths.

## Shared Representations

- `ExampleAxis`, `ChannelAxis`
- `QuantityAxis`, `TimeAxis`, `FrequencyAxis`
- `QuantityFieldSpec`, `QuantityKernelSpec`
- `quantity`, `unit_registry`

`QuantityAxis` composes exact integer TensorCore Coordinates with
`coordinate_scale: float` and one package-registry Pint Unit. Physical
coordinate magnitude is `coordinate * coordinate_scale * unit`.

## Products

The exact Product/Spec pairs are:

| Product | Spec |
|---|---|
| `Photoelectrons` | `PhotoelectronsSpec` |
| `Charge` | `ChargeSpec` |
| `PureWaveform` | `PureWaveformSpec` |
| `NoiseWaveform` | `NoiseWaveformSpec` |
| `AnalogWaveform` | `AnalogWaveformSpec` |
| `DigitizedWaveform` | `DigitizedWaveformSpec` |
| `EncodedWaveform` | `EncodedWaveformSpec` |

`Photoelectrons` exposes `validate(product=...)`. Each generated Product
exposes keyword-only `create`, `prepare`, `produce`, and `validate`.
`Charge` and `NoiseWaveform` require a public TensorCore `CounterRng` for
creation and production. Deterministic Products expose no RNG argument.

`DigitizedWaveformSpec` and `EncodedWaveformSpec` admit exactly
`torch.int8`, `torch.int16`, `torch.int32`, or `torch.int64`.
`EncodedWaveformSpec` additionally requires an explicit negative
`suppression_code` representable by its dtype. An `EncodedWaveform` value is
either that exact code or a nonnegative retained ADC code.

Preparation accepts ordered `QuantityFieldSpec` sources and one exact
Product-specific Config. It returns a fresh Config of the same exact type.
Production and validation require positional structural equality with the
retained source-Spec tuple. Structurally equal replacement Spec objects are
valid; changed units, axes, coordinates, device, dtype, semantic Spec type,
count, or order are not.

## Physical Kernels And Config Punchcards

The public coefficient leaves are `TimingJitter`, `DirectCrosstalk`,
`DelayedCrosstalk`, `Afterpulse`, `DarkCountRate`, `SmearingWidth`,
`PulseResponse`, `WhiteNoiseRms`, `PowerSpectralDensity`, `AnalogMinimum`,
`AnalogMaximum`, `BitDepth`, `InputMinimum`, `InputMaximum`, and
`AnalogGain`, each with its exact semantic Spec.

Six exact typed collections compose those kernels:
`ChargeKernels`, `PureWaveformKernels`, `NoiseWaveformKernels`,
`AnalogWaveformKernels`, `DigitizedWaveformKernels`, and
`EncodedWaveformKernels`. The encoded collection contains exact int64
`TriggerThresholdCode`, `ReleaseThresholdCode`,
`RequiredTimeOverSamples`, `PreTriggerSamples`, and `PostTriggerSamples`
Kernels. The corresponding
Configs contain only the output Spec, the typed collection, and Charge's
generation count. Prepared execution facts are private immutable state on a
fresh same-type Config; there is no Runtime or Plan type.

## Retired Surface

The package exports no `tensor_dslab.readout` namespace, `ReadoutConfig`,
`ReadoutCollection`, `SampleAxis`, `QuantityKernel`, `quantities`,
`ds20k_veto`, or `simulate_readout`. No alias or forwarding shim is provided.
Applications own workflow composition and retention.
