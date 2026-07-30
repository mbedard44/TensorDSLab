# Reconstruction Product Architecture

Status: **Design baseline; not an implementation work order.**

TensorDSLab reconstruction is a reusable sequence of scientific Product
transformations:

```text
EncodedWaveform -> SignalWaveform -> Hits -> PulseMembership
```

Applications own the workflow graph, source loading, retention, IO, and the
choice of which downstream pulse Products to calculate. TensorDSLab owns each
individual Product contract and scientific transformation. No package-level
`reconstruct(...)`, reconstruction Config, result collection, or fixed DS20k
workflow follows from this design.

The progression is intentionally architectural rather than algorithmic.
Signal extraction, hit finding, and pulse finding will each receive a focused
design and implementation work order. IV-DSLab is a scientific donor and
cautionary reference, not architectural authority.

## Design Principles

- Real detector reconstruction begins with acquired ZLE data represented as
  `EncodedWaveform`; it never assumes that a complete `DigitizedWaveform`
  exists.
- Simulation may produce `EncodedWaveform` from `DigitizedWaveform`, but
  reconstruction receives only the encoded result.
- Products represent scientifically meaningful information transformations,
  not incidental DAQ record structure or one donor package's function
  boundaries.
- Dense tensors retain common detector and time coordinates so reconstruction
  can execute tensor-natively without converting Products into ragged event
  records.
- An intermediate such as `FilteredWaveform` becomes a public Product only if
  it is independently meaningful, reusable, or inspectable. Algorithmic
  scratch remains private to the Product that needs it.
- Preparation, production, and validation retain the Maintenance 15
  boundaries in [Product composition](readout.md): preparation owns alignment
  and policy, production owns tensor math, and applications own composition.

## Shape Vocabulary

```text
E    ExampleAxis, one independent waveform realization
D... one or more detector/readout axes, commonly ChannelAxis
T    shared TimeAxis
P    reconstructed PulseAxis capacity
```

Conceptual shapes name semantic roles rather than fixing tensor dimension
order. Each exact Spec determines its own ordering. A Product name also does
not require `D...` to be exactly one `ChannelAxis`; focused Product design
must state the detector roles it admits and any roles it reduces.

| Product | Conceptual shape | Question answered |
|---|---:|---|
| `EncodedWaveform` | `(E, D..., T)` | Which ADC samples were acquired, and what codes were recorded? |
| `SignalWaveform` | `(E, D..., T)` | What nonnegative signal magnitude was reconstructed at each usable detector-time sample? |
| `Hits` | `(E, D..., T)` | Where were localized hits reconstructed, and what weight was assigned to them? |
| `PulseMembership` | `(E, T)` | Which detector-wide reconstructed pulse, if any, owns each time sample? |

The first three sample-aligned Products preserve detector and time roles.
`PulseMembership` deliberately reduces the admitted detector roles and retains
one detector-wide temporal segmentation per example.

## `EncodedWaveform`

`EncodedWaveform` is the dense tensor-native representation of ZLE acquisition:

```text
suppression_code < 0
                    suppressed / no ADC code available to reconstruction
0 ... maximumCode   retained ADC code
```

Its Spec therefore selects a signed integer dtype and one explicit negative
`suppression_code` representable by that dtype. The sentinel is not derived
from the minimum code observed in a particular tensor and need not be the
storage dtype's minimum.

In simulation, `EncodedWaveform` may share the exact detector and `TimeAxis`
coordinates of its source `DigitizedWaveform`. In experimental reconstruction,
an application constructs the same Product from DAQ ZLE records. No suppressed
ADC value is retained in `EncodedWaveform`, regardless of construction path.

Touching half-open ZLE intervals are canonically indistinguishable from one
merged interval:

```text
[240, 270) + [270, 300) -> [240, 300)
```

There is no intervening time sample at which a boundary can be encoded.
Hardware record boundaries are consequently absent from `EncodedWaveform`
unless future DS20k evidence shows that reconstruction must reset a
scientifically meaningful calculation at those boundaries.

The configured negative sentinel describes ordinary valid ZLE suppression, not
corrupt input or a DAQ coverage gap. The initial boundary should reject those
exceptional conditions or handle them explicitly in the application loader
rather than silently mapping them to suppression.

## `SignalWaveform`

`SignalWaveform` contains reconstructed, nonnegative, sample-aligned signal on
the support where reconstruction is valid. It must distinguish:

```text
unusable / insufficient sample
observed sample with reconstructed zero signal
observed sample with positive reconstructed signal
```

The exact signal unit, support rule near ZLE boundaries, and unusable-sample
encoding remain part of the focused `SignalWaveform` design. `NaN` is the
leading floating-point sentinel candidate because it is outside the finite
nonnegative signal domain and propagates through accidental arithmetic.

Signed baseline-subtracted, matched-filter, or deconvolution responses may be
private production scratch. A public `FilteredWaveform` is not selected merely
because an implementation uses a filter internally.

## `Hits`

`Hits` localizes the continuous `SignalWaveform` into discrete reconstructed
detections while retaining its detector and time roles. Its dense semantic
domains are:

```text
unusable sentinel
0                   usable sample with no reconstructed hit
positive value      reconstructed hit weight at a localized time
```

`Hits` is reconstruction-side information analogous to `Charge`, but it is not
an exact inversion of the forward readout transformation:

```text
Charge
  -> waveform formation
  -> digitization and ZLE
  -> SignalWaveform
  -> Hits
```

The leading weight candidate is calibrated integrated signal assigned to the
hit and stored at a selected anchor, such as the hit peak. That choice would
allow pulse integrals and patterns to reduce `Hits` directly. The focused
`Hits` design must still freeze the unit, anchor, interval partition, boundary
behavior, and whether pulse-shape quantities instead require
`SignalWaveform`.

## `PulseMembership`

`PulseMembership` is an integer relationship Product over example and time:

```text
-1          no reconstructed pulse
0 ... P-1   direct PulseAxis index
```

For every example, it must satisfy:

- labels are local to that example;
- active labels form the prefix `0, 1, ..., n_pulses - 1`;
- labels are assigned in temporal order;
- each active label occupies exactly one contiguous temporal run;
- every active label is less than the configured PulseAxis capacity `P`; and
- at most one pulse owns a time sample.

Thus `PulseMembership[e, t] = k` directly selects pulse slot `k` in every
pulse-indexed Product. There is no one-based identifier namespace or `k - 1`
translation.

Ordinary ZLE suppression does not introduce a second invalid-membership
sentinel. Absence of acquired samples remains evidence supplied by the ZLE
decision and can validly result in no reconstructed pulse. Genuine missing or
corrupt DAQ coverage remains an ingress concern until concrete data requires a
recoverable in-Product state.

The initial representation assumes non-overlapping detector-wide pulse
regions. The focused design must freeze the pulse-finding statistic and how a
fixed dense PulseAxis capacity is selected before production. Overflow must be
rejected rather than silently dropping, merging, or reallocating pulses.

## Pulse-Level Product Family

`PulseMembership` establishes the relationship between shared time and
PulseAxis slots. Pulse-level Products combine that relationship with the
appropriate lower-level magnitude:

| Product | Conceptual shape | Meaning |
|---|---:|---|
| `PulseIntegral` | `(E, P)` | Total reconstructed signal assigned to each pulse |
| `PulsePattern` | `(E, P, D...)` | Distribution of reconstructed signal over detector elements |
| `PromptFraction` | `(E, P)` | Fraction of pulse signal inside the accepted prompt window |
| `PulseType` | `(E, P)` | Accepted pulse classification, such as S1-like or S2-like |
| later `PulseX`, `PulseY` | `(E, P)` | Reconstructed transverse position where applicable |

Conceptually:

```text
Hits or SignalWaveform
  + PulseMembership
      -> PulseIntegral
      -> PulsePattern
      -> PromptFraction
      -> PulseType
      -> later position Products
```

Whether `PulseIntegral` and `PromptFraction` reduce `Hits`,
`SignalWaveform`, or both depends on the final hit-weight semantics. Empty
padded PulseAxis slots must be distinguishable in each pulse-level Product;
their exact sentinel and validation rules belong to those focused designs.

Pulse starts, ends, and durations are derivable from transitions in
`PulseMembership`. Pulse-indexed timing Products may later be useful
projections for downstream interfaces, but they would not introduce new
reconstruction information.

The `Pulse*` names form a semantic family. They do not imply one mixed-unit
`PulseFeatures` tensor or a mandatory package-owned collection.

## Implementation Sequence

Reconstruction should proceed through independent focused stages:

1. freeze and implement `EncodedWaveform`, including simulated encoding and
   experimental construction/validation boundaries;
2. freeze and implement `SignalWaveform`;
3. freeze and implement `Hits`;
4. freeze and implement `PulseMembership`; and
5. add pulse-level Products individually once their lower-level quantity
   contracts are stable.

Each stage must define one Product's exact Spec, Config, preparation,
production, validation, scientific law, donor comparison boundary, and
focused evidence. This page does not authorize production implementation.
