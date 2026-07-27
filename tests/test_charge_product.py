"""Charge product construction, smearing, and storage evidence."""

import unittest
from typing import override

import torch
from tensor_core import (
    CounterRng,
    NonnegativeInteger,
    OffsetAxis,
    RngKey,
    Threefry4x32,
)

from tensor_dslab import (
    ChannelAxis,
    Charge,
    ChargeConfig,
    DarkCountRate,
    DirectCrosstalk,
    ExampleAxis,
    Photoelectrons,
    ReadoutConfig,
    SampleAxis,
    SmearingWidth,
    quantities,
    quantity,
    simulate_readout,
)


class _FailingRng(CounterRng):
    __slots__ = ()

    @override
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        raise AssertionError("unexpected RNG request")


def _source(
    values: torch.Tensor,
) -> Photoelectrons:
    axes = (
        ExampleAxis(count=values.shape[0]),
        ChannelAxis(labels=tuple(f"c{i}" for i in range(values.shape[1]))),
        SampleAxis(start=0, step=2000, count=values.shape[2]),
    )
    return Photoelectrons(tensor=values, axes=axes)


def _run(
    source: Photoelectrons,
    config: ChargeConfig,
    *,
    rng: CounterRng,
    dtype: torch.dtype = torch.float32,
) -> Charge:
    return simulate_readout(
        source,
        products=(Charge,),
        config=ReadoutConfig(charge=config),
        rng=rng,
        floating_dtype=dtype,
    ).field(Charge)


class ChargeProductContractTest(unittest.TestCase):
    def test_deterministic_charge_is_fresh_exact_and_draw_free(self) -> None:
        values = torch.tensor([[[1, 2, 3]]], dtype=torch.int64)
        source = _source(values)
        result = _run(
            source,
            ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0)
            ),
            rng=_FailingRng(seed=0),
        )
        self.assertTrue(torch.equal(result.tensor, values.to(torch.float32)))
        self.assertIs(result.axes, source.axes)
        self.assertNotEqual(
            result.tensor.untyped_storage().data_ptr(),
            source.tensor.untyped_storage().data_ptr(),
        )

    def test_zero_dark_and_zero_smearing_are_draw_free(self) -> None:
        source = _source(torch.tensor([[[2, 0, 1]]], dtype=torch.int64))
        result = _run(
            source,
            ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0),
                dark_counts=DarkCountRate(
                    quantity=quantity(0, "Hz"),
                    conditioning_axes=(),
                    operation_axes=(),
                ),
                smearing_width=SmearingWidth(
                    quantity=quantity(0, "dimensionless"),
                    conditioning_axes=(),
                    operation_axes=(),
                ),
            ),
            rng=_FailingRng(seed=0),
        )
        self.assertEqual(result.tensor.tolist(), [[[2.0, 0.0, 1.0]]])

    def test_dark_count_mean_matches_rate_and_exposure(self) -> None:
        source = _source(torch.zeros((30_000, 1, 2), dtype=torch.int64))
        result = _run(
            source,
            ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0),
                dark_counts=DarkCountRate(
                    quantity=quantity(2.5e8, "Hz"),
                    conditioning_axes=(),
                    operation_axes=(),
                ),
            ),
            rng=Threefry4x32(seed=17),
            dtype=torch.float64,
        )
        observed_mean = float(result.tensor.mean())
        observed_variance = float(result.tensor.var())
        self.assertLess(abs(observed_mean - 0.5), 0.02)
        self.assertLess(abs(observed_variance - 0.5), 0.03)
        self.assertGreater(abs(observed_mean - 1.0), 0.4)

    def test_smearing_mean_and_variance_follow_relative_width(self) -> None:
        source = _source(torch.full((40_000, 1, 2), 4, dtype=torch.int64))
        result = _run(
            source,
            ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0),
                smearing_width=SmearingWidth(
                    quantity=quantity(0.25, "dimensionless"),
                    conditioning_axes=(),
                    operation_axes=(),
                ),
            ),
            rng=Threefry4x32(seed=29),
            dtype=torch.float64,
        )
        self.assertLess(abs(float(result.tensor.mean()) - 4.0), 0.015)
        self.assertLess(abs(float(result.tensor.var()) - 0.25), 0.015)

    def test_source_and_global_rng_are_unchanged(self) -> None:
        values = torch.full((16, 1, 2), 3, dtype=torch.int64)
        source = _source(values.clone())
        global_before = torch.random.get_rng_state().clone()
        source_before = source.tensor.clone()
        _run(
            source,
            ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0),
                smearing_width=SmearingWidth(
                    quantity=quantity(0.1, "dimensionless"),
                    conditioning_axes=(),
                    operation_axes=(),
                ),
            ),
            rng=Threefry4x32(seed=2),
        )
        self.assertTrue(torch.equal(source.tensor, source_before))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))

    def test_count_and_poisson_ceilings_fail_before_a_word_request(self) -> None:
        too_large = _source(
            torch.tensor([[[1 << 53, 0]]], dtype=torch.int64)
        )
        with self.assertRaises(ValueError):
            _run(
                too_large,
                ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(0)
                ),
                rng=_FailingRng(seed=0),
            )

        with self.assertRaises(ValueError):
            _run(
                _source(torch.zeros((1, 1, 2), dtype=torch.int64)),
                ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(1 << 63),
                    direct_crosstalk=DirectCrosstalk(
                        quantity=quantities((0.0,), "dimensionless"),
                        conditioning_axes=(),
                        operation_axes=(
                            OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
                        ),
                    ),
                ),
                rng=_FailingRng(seed=0),
            )

        source = _source(torch.zeros((1, 1, 2), dtype=torch.int64))
        with self.assertRaises(ValueError):
            _run(
                source,
                ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(0),
                    dark_counts=DarkCountRate(
                        quantity=quantity(5.0000001e16, "Hz"),
                        conditioning_axes=(),
                        operation_axes=(),
                    ),
                ),
                rng=_FailingRng(seed=0),
            )

        large_source = _source(
            torch.tensor([[[(1 << 53) - 1, 0]]], dtype=torch.int64)
        )
        with self.assertRaises(RuntimeError):
            _run(
                large_source,
                ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(1),
                    direct_crosstalk=DirectCrosstalk(
                        quantity=quantities((1.0e-7,), "dimensionless"),
                        conditioning_axes=(),
                        operation_axes=(
                            OffsetAxis(
                                relative_to=SampleAxis,
                                offsets=(0,),
                            ),
                        ),
                    ),
                ),
                rng=_FailingRng(seed=0),
            )


for _seed in range(12):
    def _replay(
        self: ChargeProductContractTest,
        seed: int = _seed,
    ) -> None:
        source = _source(torch.full((8, 1, 2), 2, dtype=torch.int64))
        config = ChargeConfig(
            correlated_avalanche_generations=NonnegativeInteger(0),
            smearing_width=SmearingWidth(
                quantity=quantity(0.2, "dimensionless"),
                conditioning_axes=(),
                operation_axes=(),
            ),
        )
        left = _run(source, config, rng=Threefry4x32(seed=seed))
        right = _run(source, config, rng=Threefry4x32(seed=seed))
        self.assertTrue(torch.equal(left.tensor, right.tensor))

    setattr(ChargeProductContractTest, f"test_smearing_replay_{_seed:02d}", _replay)
