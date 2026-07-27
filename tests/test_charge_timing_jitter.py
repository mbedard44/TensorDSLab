"""Direct Multinomial timing-jitter execution evidence."""

import unittest
from unittest import mock

import torch
from tensor_core import NonnegativeInteger, OffsetAxis, Threefry4x32

from tensor_dslab import (
    ChannelAxis,
    Charge,
    ChargeConfig,
    ExampleAxis,
    Photoelectrons,
    ReadoutConfig,
    SampleAxis,
    TimingJitter,
    quantities,
    simulate_readout,
)


class TimingJitterContractTest(unittest.TestCase):
    def _run(
        self,
        probabilities: tuple[float, ...],
        offsets: tuple[int, ...],
        source: torch.Tensor,
        *,
        seed: int = 3,
    ) -> torch.Tensor:
        axes = (
            ExampleAxis(count=source.shape[0]),
            ChannelAxis(labels=tuple(f"c{i}" for i in range(source.shape[1]))),
            SampleAxis(start=0, step=2, count=source.shape[2]),
        )
        jitter = TimingJitter(
            quantity=quantities(probabilities, "dimensionless"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=offsets),
            ),
        )
        return simulate_readout(
            Photoelectrons(tensor=source, axes=axes),
            products=(Charge,),
            config=ReadoutConfig(
                charge=ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(0),
                    timing_jitter=jitter,
                )
            ),
            rng=Threefry4x32(seed=seed),
        ).field(Charge).tensor

    def test_deterministic_identity_and_shift(self) -> None:
        source = torch.tensor([[[2, 3, 0, 0]]], dtype=torch.int64)
        identity = self._run((1.0,), (0,), source)
        shifted = self._run((1.0,), (1,), source)
        self.assertTrue(torch.equal(identity, source.to(torch.float32)))
        self.assertEqual(shifted.tolist(), [[[0.0, 2.0, 3.0, 0.0]]])

    def test_finite_window_discard_is_separate_from_complete_law(self) -> None:
        source = torch.tensor([[[0, 0, 0, 10]]], dtype=torch.int64)
        result = self._run((0.5, 0.5), (0, 1), source)
        self.assertLessEqual(int(result.sum()), 10)
        self.assertGreaterEqual(int(result.sum()), 0)

    def test_multinomial_mean_matches_public_probability(self) -> None:
        source = torch.zeros((10_000, 1, 3), dtype=torch.int64)
        source[:, :, 0] = 20
        result = self._run((0.25, 0.75), (0, 1), source)
        self.assertLess(abs(float(result[:, :, 0].mean()) - 5.0), 0.12)
        self.assertLess(abs(float(result[:, :, 1].mean()) - 15.0), 0.12)

    def test_replay_is_exact(self) -> None:
        source = torch.full((8, 2, 4), 3, dtype=torch.int64)
        left = self._run((0.3, 0.7), (-1, 1), source, seed=81)
        right = self._run((0.3, 0.7), (-1, 1), source, seed=81)
        self.assertTrue(torch.equal(left, right))

    def test_literal_probabilities_are_forwarded_without_normalization(self) -> None:
        calls: list[tuple[torch.Tensor, float]] = []

        class RecordingMultinomial:
            def __init__(
                self,
                *,
                counts: torch.Tensor,
                probabilities: torch.Tensor,
                completion_probability: float,
            ) -> None:
                calls.append((probabilities.clone(), completion_probability))
                self._counts = counts
                self._category_count = probabilities.numel()

            def draw(self, **_: object) -> torch.Tensor:
                return torch.zeros(
                    (self._category_count, *self._counts.shape),
                    dtype=torch.int64,
                    device=self._counts.device,
                )

        source = torch.tensor([[[2, 0, 0]]], dtype=torch.int64)
        with mock.patch(
            "tensor_dslab.readout.charge.runtime.produce.MultinomialDistribution",
            RecordingMultinomial,
        ):
            self._run((0.25, 0.750000000005), (0, 1), source)
        self.assertEqual(len(calls), 3)
        for probabilities, completion in calls:
            self.assertTrue(
                torch.equal(
                    probabilities,
                    torch.tensor(
                        (0.25, 0.750000000005),
                        dtype=torch.float64,
                    ),
                )
            )
            self.assertEqual(completion, 0.0)


for _seed in range(8):
    def _case(self: TimingJitterContractTest, seed: int = _seed) -> None:
        source = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int64)
        result = self._run((0.5, 0.5), (0, 1), source, seed=seed)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.all(result >= 0))

    setattr(TimingJitterContractTest, f"test_timing_seed_{_seed:02d}", _case)
