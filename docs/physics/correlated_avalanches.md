# Correlated Avalanches

Maintenance 15 retains the literal-kernel fixed-generation model selected by
Maintenance 12, now behind the direct `Charge` Product boundary.

`DirectCrosstalk`, `DelayedCrosstalk`, and `Afterpulse` are dimensionless
expected-offspring TensorKernels. Direct crosstalk may use prompt nonnegative
time offsets or other semantic offset roles. Delayed crosstalk and afterpulse
require one strictly positive TimeAxis offset dimension.

For one generation:

1. every enabled mechanism reads the same immutable integer frontier;
2. each literal coefficient maps source counts to retained in-window
   destination Poisson means;
3. each mechanism performs one tensor-valued addressed Poisson draw;
4. mechanism children are checked and pooled;
5. the pooled children are added to total Charge and become the next
   generation frontier.

Children never feed back in the generation in which they are born.
Out-of-window destinations contribute no mean or count. Afterpulse children
carry one full avalanche; there is no recovery weight, occurrence Bernoulli,
conditional delay allocation, overflow Product, or total-first
Poisson-plus-Multinomial replacement.

The public generation count is fixed before execution. Direct, delayed, and
afterpulse roles retain streams 5, 6, and 7 in namespace `0x54445331`.
Preparation proves geometry, units, devices, destination-mean ceilings, count
ceilings, and address capacity before stochastic execution.

Independent tests compare first- and later-generation destination means with
analytic Poisson-branching expectations, prove exact same-seed replay, and
prove finite-window discard.
