"""Independent fixed-generation collapsed-Poisson branching evidence."""

import unittest

import torch
from tensor_core import NonnegativeInteger, OffsetAxis, Threefry4x32

from tensor_dslab import (
    Afterpulse,
    ChannelAxis,
    Charge,
    ChargeConfig,
    DelayedCrosstalk,
    DirectCrosstalk,
    ExampleAxis,
    Photoelectrons,
    ReadoutConfig,
    SampleAxis,
    quantities,
    simulate_readout,
)


def _run(
    *,
    examples: int,
    generations: int,
    direct_mean: float | None = None,
    delayed_mean: float | None = None,
    afterpulse_mean: float | None = None,
    seed: int = 11,
) -> torch.Tensor:
    axes = (
        ExampleAxis(count=examples),
        ChannelAxis(labels=("c",)),
        SampleAxis(start=0, step=1, count=2),
    )
    source = Photoelectrons(
        tensor=torch.ones((examples, 1, 2), dtype=torch.int64),
        axes=axes,
    )
    direct = (
        None
        if direct_mean is None
        else DirectCrosstalk(
            quantity=quantities((direct_mean,), "dimensionless"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
            ),
        )
    )
    afterpulse = (
        None
        if afterpulse_mean is None
        else Afterpulse(
            quantity=quantities((afterpulse_mean,), "dimensionless"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(1,)),
            ),
        )
    )
    delayed = (
        None
        if delayed_mean is None
        else DelayedCrosstalk(
            quantity=quantities((delayed_mean,), "dimensionless"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(1,)),
            ),
        )
    )
    return simulate_readout(
        source,
        products=(Charge,),
        config=ReadoutConfig(
            charge=ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(generations),
                direct_crosstalk=direct,
                delayed_crosstalk=delayed,
                afterpulse=afterpulse,
            )
        ),
        rng=Threefry4x32(seed=seed),
        floating_dtype=torch.float64,
    ).field(Charge).tensor


class CorrelatedAvalancheContractTest(unittest.TestCase):
    def test_direct_mean_tracks_fixed_generation_analytic_law(self) -> None:
        mean = 0.4
        result = _run(examples=30_000, generations=2, direct_mean=mean)
        expected = 1.0 + mean + mean**2
        self.assertLess(abs(float(result.mean()) - expected), 0.035)

    def test_half_probability_mutant_is_separated(self) -> None:
        result = _run(examples=30_000, generations=1, direct_mean=0.8)
        observed = float(result.mean())
        self.assertLess(abs(observed - 1.8), 0.035)
        self.assertGreater(abs(observed - 1.4), 0.25)

    def test_children_do_not_feed_back_within_same_round(self) -> None:
        result = _run(
            examples=30_000,
            generations=1,
            direct_mean=0.5,
            afterpulse_mean=0.5,
        )
        self.assertLess(abs(float(result.mean()) - 1.75), 0.04)

    def test_afterpulse_is_full_charge_and_discards_outside_window(self) -> None:
        result = _run(examples=30_000, generations=1, afterpulse_mean=0.75)
        first = float(result[..., 0].mean())
        second = float(result[..., 1].mean())
        self.assertLess(abs(first - 1.0), 0.01)
        self.assertLess(abs(second - 1.75), 0.04)

    def test_delayed_and_afterpulse_have_independent_poisson_moments(self) -> None:
        delayed = _run(
            examples=40_000,
            generations=1,
            delayed_mean=0.6,
            seed=61,
        )[..., 1]
        afterpulse = _run(
            examples=40_000,
            generations=1,
            afterpulse_mean=0.35,
            seed=61,
        )[..., 1]
        self.assertLess(abs(float(delayed.mean()) - 1.6), 0.025)
        self.assertLess(abs(float(delayed.var()) - 0.6), 0.04)
        self.assertLess(abs(float(afterpulse.mean()) - 1.35), 0.025)
        self.assertLess(abs(float(afterpulse.var()) - 0.35), 0.035)

    def test_negative_non_sample_offset_maps_the_exact_destination(self) -> None:
        examples = 30_000
        axes = (
            ExampleAxis(count=examples),
            ChannelAxis(labels=("left", "right")),
            SampleAxis(start=0, step=1, count=2),
        )
        source = Photoelectrons(
            tensor=torch.ones((examples, 2, 2), dtype=torch.int64),
            axes=axes,
        )
        kernel = DirectCrosstalk(
            quantity=quantities(
                torch.tensor([[0.5]], dtype=torch.float64),
                "dimensionless",
            ),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=ChannelAxis, offsets=(-1,)),
                OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
            ),
        )
        result = simulate_readout(
            source,
            products=(Charge,),
            config=ReadoutConfig(
                charge=ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(1),
                    direct_crosstalk=kernel,
                )
            ),
            rng=Threefry4x32(seed=73),
            floating_dtype=torch.float64,
        ).field(Charge).tensor
        self.assertLess(abs(float(result[:, 0].mean()) - 1.5), 0.025)
        self.assertTrue(torch.equal(result[:, 1], torch.ones_like(result[:, 1])))

    def test_exact_replay(self) -> None:
        left = _run(examples=128, generations=3, direct_mean=0.3, seed=47)
        right = _run(examples=128, generations=3, direct_mean=0.3, seed=47)
        self.assertTrue(torch.equal(left, right))


for _seed in range(12):
    def _replay_case(
        self: CorrelatedAvalancheContractTest,
        seed: int = _seed,
    ) -> None:
        left = _run(examples=16, generations=2, direct_mean=0.2, seed=seed)
        right = _run(examples=16, generations=2, direct_mean=0.2, seed=seed)
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.all(left >= 1))

    setattr(
        CorrelatedAvalancheContractTest,
        f"test_branching_replay_seed_{_seed:02d}",
        _replay_case,
    )
