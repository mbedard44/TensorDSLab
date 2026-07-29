# Scientific Parity And Intentional Divergences

## Maintenance 15 Comparison Boundary

Maintenance 15 preserves the selected literal-kernel Product laws while
deliberately replacing the workflow and representation boundary. Evidence is
therefore attached to individual Product equations, exact private RNG roles,
same-stack replay, and tensor relationships—not to the retired generic readout
or to old completed end-to-end bytes.

| Product/law | Maintenance 15 classification |
|---|---|
| ordered avalanche-source accumulation | Preserved |
| literal complete timing multinomial | Preserved |
| dark-count destination Poisson | Preserved |
| fixed-generation collapsed-rate direct/delayed/afterpulse Poisson | Preserved |
| Gaussian charge smearing `sigma * sqrt(count)` with zero clipping | Preserved |
| literal signed PulseResponse convolution | Preserved |
| exact-zero and Gaussian white noise | Preserved |
| prepared-bin real PSD synthesis | Preserved |
| ordered analog addition and literal saturation | Preserved |
| linear gain/bounds/BitDepth ADC mapping | Preserved |
| eight role keys in namespace `0x54445331`, streams 1–8 | Preserved |
| generic readout orchestration and request retention | Intentionally retired |
| DS20k profile and bundled readout demonstrations | Intentionally retired to application ownership |
| Runtime/Plan execution representation | Intentionally retired |

## Scientific Details

Charge sources are exact nonnegative `torch.int64` avalanche-compatible
magnitudes bounded by `2**53 - 1`. Timing probabilities are neither normalized
nor repaired. Every branching mechanism in one generation reads the same
immutable frontier; pooled children become the next frontier only after all
mechanisms complete. Out-of-window destinations are discarded. Afterpulse
offspring contribute full charge and have no recovery factor.

White noise uses the configured RMS. PSD input is already integrated per-bin
power with exact zero DC, one regular FrequencyAxis, RFFT count
`N // 2 + 1`, and reciprocal physical spacing. PulseResponse polarity is
literal and is applied exactly once. Digitization preserves integer BitDepth
rather than homogenizing it with floating coefficients.

Kernel conditioning coordinates may be reordered and conditioning dimensions
permuted during preparation. Source axes are not reordered: they must have
equivalent complete semantic state, though their tensor dimension order may
differ.

## Qualifications

Exact stochastic completed values are qualified to the named stack and
same-stack replay. Maintenance 15 changes orchestration and address capacity
composition enough that historical end-to-end literals are not continuity
goldens. Independent statistical/analytic oracles govern current science.

The candidate evidence is CPU-only. It establishes no donor-wide equivalence,
CUDA behavior, performance, calibration, release, deployment, compatibility,
or production-readiness claim.
