# How TensorDSLab Simulates Correlated Avalanches

This guide explains the physical and statistical ideas behind TensorDSLab's
correlated-avalanche model. It is written for readers who want to understand
the simulation without first learning its implementation details.

The exact Design contract remains the
[fixed-generation correlated-avalanche baseline](../architecture/rebuild.md#fixed-generation-correlated-avalanche-baseline).
The detailed comparison with IV-DSLab remains in
[IV-DSLab parity](../parity.md#crosstalk-and-correlated-noise-recursion). This
guide summarizes those sources; it does not replace them or claim that the
selected model is already implemented in production.

## The Thirty-Second Explanation

An avalanche in a silicon photomultiplier (SiPM) can produce additional
avalanches. A SiPM contains many microscopic avalanche cells, called
microcells:

- direct crosstalk (DiCT) can trigger another microcell promptly;
- delayed crosstalk (DeCT) can trigger another microcell after a delay; and
- afterpulsing (AP) can retrigger the parent avalanche's microcell after a
  delay.

Those children can produce children of their own. TensorDSLab models this as a
sequence of genealogical generations on the already-binned readout grid.

Instead of creating one table row for every avalanche, TensorDSLab combines
equivalent parents using probability identities and samples aggregate counts
directly in tensor cells. This preserves the selected model's statistical
fluctuations and relationships; it does not merely replace random counts with
their averages.

The aggregation is distributionally exact for the declared binned branching
model. The model itself still makes physical approximations, described below.

## Where This Fits In The Readout Chain

```text
Photoelectron truth
    |
    v
optional dark counts
    |
    v
optional timing jitter
    |
    v
optional correlated avalanches
    |
    v
optional charge smearing
    |
    v
Charge
```

`Photoelectrons` is immutable photon-origin truth. Dark counts, timing jitter,
correlated avalanches, and smearing are private parts of Charge production;
they do not replace or mutate that truth field. The correlated stage begins
from the private integer avalanche grid produced by the preceding enabled
steps.

## A Small Vocabulary

| Term | Meaning |
| --- | --- |
| root | An avalanche present before correlated-avalanche reproduction begins. |
| parent | An avalanche whose direct children are being simulated. |
| child | A DiCT, DeCT, or AP avalanche produced by one parent generation. |
| frontier | The avalanches allowed to produce children in the current update. |
| generation | One genealogical level of the cascade, not one time bin. |
| sample bin | One discrete time bin in the readout grid. |
| retained | Located inside the readout window. |
| overflow | Produced beyond the right edge of the readout window. |
| PE-equivalent charge | The response of one fully recovered avalanche, used as the unit of Charge. |

A child can occupy the same sample bin as its parent and still belong to the
next genealogical generation.

## The Three Mechanisms

| Mechanism | Physical picture | Children produced per parent in one generation | Timing | Deposited charge | Can reproduce later? |
| --- | --- | --- | --- | --- | --- |
| DiCT | Light from one avalanche triggers another microcell | Poisson-distributed count | Configured causal delay | One PE-equivalent | Yes |
| DeCT | Another microcell is triggered after a physical delay | Poisson-distributed count | Configured causal delay | One PE-equivalent | Yes |
| AP | The parent avalanche's microcell retriggers | At most one, with configured probability | Exponential delay model | One or recovery weighted | Yes |

DiCT and DeCT are separate physical or calibrated crosstalk modes. “Direct”
and “delayed” do not simply mean “same sample bin” and “later sample bin.” Each
mode has its own causal delay model, and bin placement follows from that model
and the sampling period.

The MVP treats every crosstalk child as a fresh, fully recovered microcell.
An afterpulse retriggers its parent's microcell and may therefore deposit less
than one PE-equivalent charge. That recovery reduction affects deposited
charge only: the afterpulse still counts as one avalanche and has the same
future offspring laws as every other avalanche.

## Why Aggregate Tensor Sampling Works

Assume that one tensor cell contains `Q` statistically equivalent parent
avalanches.

For a crosstalk mode with mean direct-offspring count `lambda`, each parent has
an independent Poisson number of children. Adding those independent
populations gives another Poisson population:

```text
Q independent Poisson(lambda) populations
    -> one Poisson(Q * lambda) population
```

If the probability that one child lands at offset `d` is `q[d]`, Poisson
thinning gives the destination count directly:

```text
children at offset d ~ Poisson(Q * lambda * q[d])
```

Contributions from different source bins that reach the same destination can
also be added into that destination's Poisson rate before drawing. No
individual child record is required.

For destination bin `u`, the superposition looks like this:

```text
Poisson(Q[t0] * lambda * q[u - t0])
  + Poisson(Q[t1] * lambda * q[u - t1])
  + Poisson(Q[t2] * lambda * q[u - t2])
  + ...
      -> Poisson(lambda * sum_t(Q[t] * q[u - t])) at destination u
```

The algorithm first forms the one aggregate rate for destination `u`, then
draws its count once. It does not draw or store the contributing children from
each parent separately.

Afterpulsing has a different combinatorial structure. Each parent can produce
at most one AP child in that generation, so the `Q` parents are divided among
mutually exclusive outcomes:

```text
retained delay offsets, right overflow, or no afterpulse
```

Their aggregate outcome is therefore multinomial. This retains the
competition among AP outcomes: assigning one parent to one destination means
that same parent cannot also produce another AP child somewhere else.

For one source bin, the parent-by-parent choices combine as follows:

```text
parent 1 --choose one--> [offset 0, offset 1, ..., overflow, no AP]
parent 2 --choose one--> [offset 0, offset 1, ..., overflow, no AP]
   ...
parent Q --choose one--> [offset 0, offset 1, ..., overflow, no AP]
                                   |
                                   v
(A[0], A[1], ..., A[overflow], A[no AP])
    ~ Multinomial(
          Q,
          [p * q[0], p * q[1], ..., p * q[overflow], 1 - p],
      )
```

Every parent appears in exactly one category:

```text
A[0] + A[1] + ... + A[overflow] + A[no AP] = Q
```

The retained categories from different source bins are then shifted to their
absolute destination bins and added:

```text
source bin t0 multinomial --shift by t0--+
source bin t1 multinomial --shift by t1--+--> AP counts by destination bin
source bin t2 multinomial --shift by t2--+
```

This preserves the one-AP-per-parent rule and the competition among outcomes
without constructing one record for every parent or AP child.

## Why This Algorithm Maps Naturally To GPUs

The algorithm is built around dense arrays whose dimensions are examples,
channels, and sample bins. Within one generation, the same small set of
operations is applied across that complete grid:

```text
current dense frontier F[g]
    |
    +--> form DiCT and DeCT rate arrays across destination cells
    +--> draw Poisson counts across destination cells
    +--> draw AP multinomial categories across source cells
    +--> shift and add retained children into destination cells
    +--> accumulate newly deposited charge and overflow diagnostics
    |
    v
next dense frontier F[g + 1]
```

For readouts containing very large numbers of G4DS-derived photoelectrons, this
aggregate tensor algorithm is expected to scale more naturally on GPUs than
IV-DSLab's recursive CPU algorithm: TensorDSLab performs batched draws and
reductions over dense arrays instead of recursively following every individual
avalanche lineage. The size of any speedup is deliberately left to later
measurement rather than asserted by this Design guide.

Examples and channels are independent dimensions of this work. Sample cells
within a mechanism can also be processed in parallel once the current frontier
is fixed. A simulation with many examples and channels therefore exposes a
large amount of parallel work even when each waveform contains a relatively
modest number of sample bins.

The statistical aggregation is what makes this practical. A per-avalanche
simulation would continually create variable-length child lists, append rows,
follow individual lineages, and regroup their results into time bins. Those
jagged operations are a poor match for dense GPU memory and would encourage
CPU-side control or repeated allocation. TensorDSLab instead keeps fixed-shape
integer frontier arrays and floating charge arrays on the execution device.

The random draws are position addressed: the draw for one logical tensor cell
does not depend on consuming random numbers for earlier cells. Independent
cells can therefore request their random values in parallel without sharing a
mutable global generator sequence.

There is still one unavoidable causal boundary. Generation `g + 1` cannot be
simulated until every retained child of generation `g` has been combined into
the next frontier:

```text
parallel work for generation 0
    -> generation boundary
parallel work for generation 1
    -> generation boundary
parallel work for generation 2
    -> ...
```

Target workloads contain many example-channel cells, often far more than the
configured number of generations. In that regime, the design places the
dominant parallelism inside each generation while retaining the scientific
generation order. Nothing in the statistical model requires the frontier or
charge tensors to leave the execution device between generations.

This is an architectural reason to expect effective GPU acceleration, not a
measured performance claim. A production stage must still measure kernel
launches, memory traffic, rejection-sampler divergence, late-generation
sparsity, and CPU/CUDA statistical agreement. Optimization may change how the
operations are fused or scheduled, but it must not change the probability
laws, frontier rule, or requested generation semantics described here.

## How One Generation Works

Every generation answers two questions:

1. **How many avalanches are produced by this frontier?**
2. **Which sample bins receive those avalanches?**

Conceptually, one update looks like this:

```text
current frontier
    |
    +--> DiCT / DeCT / AP draws
             |
             +--> retained child counts --> next frontier
             |              |
             |              +------------> add new child charge
             |
             +--> right overflow --------> diagnostics only
```

Before the first update, the roots deposit their charge into the cumulative
ledger exactly once.

The update proceeds as follows:

1. Freeze the current frontier.
2. Draw DiCT, DeCT, and AP outcomes from that frontier.
3. Place retained children in their destination sample bins.
4. Record children that cross the right edge as overflow.
5. Add only the newly retained children's deposited charge to the cumulative
   charge ledger.
6. Add their integer counts to form the next frontier.
7. Repeat until the configured generation count has been evaluated.

> **The frontier rule:** each generation simulates children only for the
> avalanches in the current frontier. Avalanches from earlier generations have
> already had their opportunity to produce children and are never processed
> again. Their deposited charge remains in the cumulative charge total, but
> their counts do not remain in the branching frontier. Charge is recorded
> once when an avalanche is born; it is not added again when that avalanche
> later serves as a parent.

For `maximum_generations = K`:

```text
K = 0  -> roots only
K = 1  -> roots and their children
K = 2  -> roots, children, and grandchildren
```

Generation `K` contributes count and charge, but its own children are not
simulated.

## A Worked Array Example

The following numbers are deliberately simple and illustrative. They are not
TensorDSLab defaults, calibration recommendations, or a fixed expected
outcome.

Suppose the readout contains five sample bins:

| Quantity | Illustrative choice |
| --- | --- |
| initial frontier | `[2, 0, 1, 0, 0]` |
| DiCT | mean `0.4` children per parent, same-bin placement |
| DeCT | mean `0.5` children per parent, one-bin delay |
| AP | probability `0.5` per parent, exponential mean delay equal to one sample period |
| AP recovery | disabled, so every retained AP has unit charge |

The initial frontier is:

```text
sample bin                  0    1    2    3    4
generation 0 frontier F0   2    0    1    0    0
```

The roots deposit their unit charge before reproduction:

```text
cumulative charge = charge deposited by F0
```

### Draw The DiCT Children

Same-bin placement leaves the source positions unchanged. The aggregate DiCT
rate array is:

```text
R_direct = 0.4 * F0
         = [0.8, 0, 0.4, 0, 0]
```

TensorDSLab draws one Poisson count at every destination. One possible
realization is:

```text
Poisson(R_direct) -> [1, 0, 0, 0, 0]
```

### Draw The DeCT Children

The one-bin delay shifts the aggregate rate to the right:

```text
R_delayed = shift_right(0.5 * F0, 1)
          = [0, 1.0, 0, 0.5, 0]
```

One possible realization is:

```text
Poisson(R_delayed) -> [0, 1, 0, 1, 0]
```

### Draw The AP Children

For the illustrated ratio of sample period to exponential mean delay, the
conditional AP offset probabilities, rounded for display, are:

```text
offset                   0      1      2      3      4     5 or more
P(offset | AP fired)   0.368  0.400  0.147  0.054  0.020    0.011
```

The `0.5` AP probability multiplies those offset probabilities, while the
remaining probability `0.5` is the no-AP outcome.

At sample bin 0, two parents are divided among all retained destinations,
right overflow, and no AP. One possible draw is:

```text
Multinomial(
    parents = 2,
    outcomes = [
        bin 0: 0.184,
        bin 1: 0.200,
        bin 2: 0.0735,
        bin 3: 0.027,
        bin 4: 0.010,
        overflow: 0.0055,
        no AP: 0.500,
    ],
)
-> [bin 2: 1, no AP: 1, every other category: 0]
```

At sample bin 2, only offsets 0, 1, and 2 remain inside the window. All longer
offsets combine into right overflow:

```text
Multinomial(
    parents = 1,
    outcomes = [
        bin 2: 0.184,
        bin 3: 0.200,
        bin 4: 0.0735,
        overflow: 0.0425,
        no AP: 0.500,
    ],
)
-> [bin 4: 1, every other category: 0]
```

The resulting AP contribution is:

```text
[0, 0, 1, 0, 1]
```

### Form The Next Frontier

Only retained children are added:

```text
sample bin                  0  1  2  3  4  | overflow
─────────────────────────────────────────────────────
DiCT children               1  0  0  0  0  |    0
DeCT children               0  1  0  1  0  |    0
AP children                 0  0  1  0  1  |    0
                            ───────────────
generation 1 frontier F1   1  1  1  1  1  |    —
```

At this point:

```text
cumulative charge += charge deposited by F1
next reproduction uses F1 only
```

The next update draws children from `F1` only. For illustration, suppose those
draws produce:

```text
sample bin                  0  1  2  3  4  | overflow
─────────────────────────────────────────────────────
DiCT children               0  1  0  0  0  |    0
DeCT children               0  0  1  0  1  |    1
AP children                 0  0  0  1  0  |    0
                            ───────────────
generation 2 frontier F2   0  1  1  1  1  |    —
```

If `K = 2`, the simulation stops here. `F2` contributes to the result, but the
simulation does not draw children from it.

The cumulative retained avalanche count is:

```text
F0 + F1 + F2 = [3, 2, 3, 2, 2]
```

That cumulative array contributes to Charge, but it was never used as a
frontier. In particular, the second generation update branched from:

```text
F1 = [1, 1, 1, 1, 1]
```

not from:

```text
F0 + F1 = [3, 1, 2, 1, 1]
```

This example uses unit-charge APs. With AP recovery enabled, the integer
frontiers are formed in exactly the same way, while the AP contribution to the
floating charge ledger can be less than one PE-equivalent per avalanche.

## Avalanche Count And Deposited Charge Are Different

TensorDSLab keeps branching state separate from charge response:

- the integer frontier determines how many parents can reproduce next;
- the accumulated charge weight records the response deposited so far; and
- the accumulated squared charge weight supplies the variance needed by later
  aggregate gain smearing.

The squared-charge ledger is the sum of the individual response weights
squared, `sum(w_i**2)`. It is not the square of the aggregate charge,
`sum(w_i)**2`.

Roots, DiCT children, and DeCT children contribute one avalanche count and one
PE-equivalent charge. An AP contributes one avalanche count but can contribute
a recovery-dependent fractional charge. Its fractional charge is never carried
into the next frontier and never reduces its future DiCT, DeCT, or AP
probability.

This is why TensorDSLab cannot use one cumulative floating Charge tensor as
both the detector response and the branching population.

## How Continuous Delays Become Sample Bins

The dense input identifies the sample bin of each root but no longer contains
its exact arrival time inside that bin. TensorDSLab represents this missing
sub-bin phase with a fresh independent value for every parent-child edge:

```text
U     ~ Uniform([0, sample period))
delay = configured nonnegative physical delay

destination offset = floor((U + delay) / sample period)
```

Preflight combines this phase policy with the physical delay law to obtain the
probability of every integer sample offset. The hot path samples those prepared
categories; it does not reconstruct individual continuous event times.

Crosstalk may use fixed or exponential delays. AP uses an exponential delay
model. The earlier zero-clipped normal proposal is not part of the MVP; a later
distributed family beyond the ordinary exponential would require calibration
and a new Design decision. All selected delays are causal, so there is no left
underflow. A child beyond the final sample enters a right-overflow diagnostic
and is removed from later generations. Because it cannot travel backward in
time, it cannot return to the retained window.

Fixed delays become at most two adjacent offset probabilities. Exponential
delays use analytic probabilities and right tails, so the implementation does
not truncate a long tail or recover overflow by subtracting many rounded
probabilities from one. When AP recovery is configured, its mean deposited
charge is integrated over the same delay category without changing how many
afterpulses were produced or where they landed.

The independent phase policy preserves the selected one-edge offset
distribution. It intentionally omits correlations that would arise from
siblings sharing one hidden parent phase.

## Population Growth And Statistical Correlations

Ignoring finite-window loss, `lambda_direct` and `lambda_delayed` are the
configured mean crosstalk child counts per parent, while `p_afterpulse` is the
probability of one AP child. The mean number of children per parent is:

```text
m = lambda_direct + lambda_delayed + p_afterpulse
```

For a homogeneous unbounded model with `N0` roots, the expected population in
generation `g` is:

```text
E[Ng] = N0 * m**g
```

This is useful intuition, not a replacement for simulation. The actual
generation populations fluctuate, their timing distributions evolve, and
finite-window overflow reduces the retained population.

The expected generation population decreases when `m < 1` and increases when
`m > 1`; individual sampled generations can fluctuate in either direction. A
fixed `K` makes execution finite, but it does not make an explosively
configured model physically plausible or numerically safe.

The result is correlated even though the three mechanisms are drawn
independently once the current frontier is fixed:

- a larger realized frontier tends to create larger later generations;
- descendants share random ancestry;
- AP destination outcomes compete because one parent can produce at most one
  direct AP; and
- realized AP counts jointly affect charge and the population available for
  later branching.

Aggregate tensor sampling preserves these relationships within the selected
model.

## Assumptions That Make Aggregation Valid

The initial model assumes:

- the same probabilities and delay settings apply to every simulated channel;
- parents follow identical, conditionally independent offspring laws;
- crosstalk has an effectively unlimited supply of fresh microcells;
- microcell identity, topology, collisions, occupancy, and saturation are not
  tracked;
- children enter one shared next frontier without carrying microcell identity
  or recovery history;
- AP recovery changes deposited charge but not later offspring laws;
- every parent-child edge receives an independent sub-bin phase;
- APs assigned to one binned delay category use that category's average
  recovery weight;
- all mechanisms draw from one frozen frontier before the next generation is
  formed; and
- the caller selects a finite maximum generation count.

These assumptions are what allow many equivalent parents to be combined. A
model with cell identities, recovery-dependent branching, collisions, shared
hidden phase, or occupancy-dependent probabilities would require additional
state and different combinatorics.

## What Is Preserved And What Is Approximated

Within the selected model, aggregation preserves:

- Poisson crosstalk fluctuations and timing allocation;
- AP fire and delay fluctuations;
- multinomial competition among AP destinations, overflow, and no AP;
- coupling between AP count, AP charge, and later generations;
- generation-to-generation branching fluctuations; and
- separate mechanism and overflow accounting.

The model approximates or omits:

- exact continuous sub-bin histories;
- phase correlations shared by siblings or successive generations;
- recovery variation among APs assigned to the same offset category;
- finite-cell collisions, exhaustion, topology, and recovery history;
- recovery-dependent future offspring production; and
- every generation after the configured `K`.

“Exact aggregate simulation” therefore means exact for this declared
statistical model. It does not mean exact microscopic detector simulation.

## Relationship To IV-DSLab

TensorDSLab targets statistical comparison of named post-binned observables,
not eventwise or bitwise reproduction of IV-DSLab.

Important intentional differences include:

- explicit fixed-`K` generation semantics rather than IV's growing recursive
  queue and source-dependent recursion quirks;
- a separate DeCT mechanism with no audited IV counterpart;
- an exponential AP delay model rather than IV's literal
  reciprocal-exponential expression;
- recovery-weighted AP charge without recovery-dependent future branching;
  and
- causal destination placement with explicit absorbing right overflow.

The full evidence and comparison classifications are maintained in
[IV-DSLab parity](../parity.md).

## How We Establish Confidence

Validation should combine simple identities, analytic statistics, and explicit
small references:

- all mechanisms disabled gives the exact identity;
- `K = 0` produces roots only;
- `K = 1` produces exactly one offspring generation;
- one-mechanism tests recover the expected zero probability, mean, variance,
  timing profile, and overflow fraction;
- AP outcomes conserve parents across retained offsets, overflow, and no AP;
- prepared delay probabilities plus their right tail normalize to one;
- generation populations follow analytic expectations over repeated trials;
- a small explicit per-avalanche reference agrees statistically with the
  aggregate algorithm for supported assumptions; and
- means, variances, tails, temporal covariance, and named post-binned
  observables are compared with IV-DSLab where parity is claimed.

The aggregate algorithm should be trusted because these distributions and
conservation laws are demonstrated, not merely because a GPU implementation is
fast.

## Further Reading

- [Rebuild architecture](../architecture/rebuild.md#fixed-generation-correlated-avalanche-baseline):
  exact selected algorithms, configuration, numeric contracts, and remaining
  implementation gates.
- [IV-DSLab parity](../parity.md#crosstalk-and-correlated-noise-recursion):
  audited donor behavior, statistical comparison boundaries, and intentional
  divergences.
- [Validation](../validation.md): package-wide validation philosophy and
  accepted scientific evidence boundaries.
