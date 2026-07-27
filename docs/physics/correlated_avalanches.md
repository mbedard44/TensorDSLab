# How TensorDSLab Simulates Correlated Avalanches

This guide explains the Maintenance 12 literal-kernel correlated-avalanche
model. The normative package contract is the
[rebuild architecture](../architecture/rebuild.md), and deliberate comparison
changes are classified in [parity](../parity.md).

When these bytes are absent from `main`, they describe a fixed-commit
candidate. If present unchanged on `main`, Review's fast-forward has completed;
final Design acceptance remains pending until the Maintenance 12 work order
and implementation index record **Merged / Closed**.

## The Short Version

An avalanche in a silicon photomultiplier can trigger additional avalanches:

- direct crosstalk acts promptly;
- delayed crosstalk acts at a positive sample displacement; and
- afterpulsing acts at a positive sample displacement.

TensorDSLab represents each mechanism as a literal expected-offspring kernel.
For one fixed genealogical generation, every enabled mechanism reads the same
immutable integer frontier, maps that frontier into destination-rate bins, and
issues one tensor-valued Poisson draw. The pooled children become the next
frontier. A child never becomes a parent in the generation in which it is
born.

Every retained child contributes exactly one avalanche count, one
PE-equivalent to the S1 ledger, and one to the S2 ledger. The Maintenance 12
model has no recovery weight, separate afterpulse occurrence draw, conditional
afterpulse delay draw, or overflow product.

## Literal Geometry

Each public physical kernel combines:

- a copied canonical CPU `float64` quantity tensor;
- optional conditioning axes that select source roles such as channel or
  example; and
- operation `OffsetAxis` values that describe destination displacement.

The sample-relative operation axis is required:

- `DirectCrosstalk` permits offsets greater than or equal to zero;
- `DelayedCrosstalk` requires strictly positive offsets; and
- `Afterpulse` requires strictly positive offsets.

Direct and delayed kernels may also carry other concrete operation axes. An
example-relative operation axis is forbidden because different examples are
independent events.

Preparation resolves roles against the exact source axes, validates coordinate
coverage, applies coordinate permutations, and materializes a plain
target-device floating tensor. Public kernel storage remains literal; it is
not expanded to product shape.

## One Generation

Let `F_g(s)` be the current integer frontier at source cell `s` for generation
`g`, and let `K_m(s, d)` be the aligned expected-offspring coefficient from
source `s` to destination `d` for mechanism `m`.

The in-window destination mean is

```text
lambda_m,g(d) = sum_s F_g(s) * K_m(s, d)
```

Only represented in-window destinations contribute. An operation-axis offset
that crosses the finite sample window is discarded before the distribution
request; it is not renormalized or reported separately.

The mechanism then draws

```text
C_m,g(d) ~ Poisson(lambda_m,g(d))
```

independently under its fixed private RNG role. The next frontier is

```text
F_(g+1)(d) = sum_m C_m,g(d)
```

and the retained correlated-avalanche count is the sum of all `C_m,g`.

## Fixed Depth And Same-Frontier Independence

`ChargeConfig.correlated_avalanche_generations` is the number of branching
rounds. Branching kernels are present exactly when that value is positive.
Generation zero consumes the post-dark, post-timing primary frontier.

All mechanisms in a generation consume that same frontier. Neither direct
crosstalk nor delayed crosstalk nor afterpulse can observe a sibling
mechanism's newly drawn children until the next generation. This fixed-depth
rule makes traversal order irrelevant and prevents an accidental same-round
cascade.

## Count, S1, And S2 Ledgers

The Charge producer tracks:

- integer avalanche count;
- S1, the sum of deposited PE-equivalent charge; and
- S2, the sum of squared deposited charge.

For the active Maintenance 12 full-charge branching law, every retained child
adds one to all three ledgers. This equality is deliberate. S1 and S2 remain
separate because later smearing consumes both and because a future separately
authorized scientific model could introduce non-unit deposits.

The producer performs checked accumulation and rejects an invalid prepared
domain before a distribution request or result write. It never clips a sampled
count, silently wraps integer arithmetic, or normalizes a physical rate.

## Addressing

The eight active private stochastic keys use namespace `0x54445331` and compact
streams:

```text
1 white noise
2 PSD noise
3 dark counts
4 timing jitter
5 direct crosstalk
6 delayed crosstalk
7 afterpulse
8 charge smearing
```

Branching uses one shared generation-root address shape and selects the stable
generation coordinate without renumbering. Mechanisms remain disjoint through
their keys. Reusing the same immutable `CounterRng` replays the same result and
does not mutate Torch's global RNG state.

## Deliberate Rebaseline

Maintenance 11 used an at-most-one afterpulse occurrence law, conditional
delay allocation, and optional recovery-weighted deposited charge. Maintenance
12 intentionally replaces that model with literal expected-offspring rates and
one full-charge Poisson draw per generation. Completed Maintenance 11
stochastic tensors therefore are historical evidence, not continuity goldens.

This change is not a claim of donor equivalence or approved detector
calibration. Evidence is based on independent destination-rate oracles,
Poisson mean/variance, fixed-depth recursion, boundary discard, ledger
identities, replay, storage, and downstream product contracts. Current
qualification is CPU-only.
