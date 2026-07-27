"""Public orchestration evidence across the complete product request matrix."""

import itertools
import unittest
from typing import ClassVar, override
from unittest import mock

import torch
from tensor_core import (
    CounterRng,
    NonnegativeFloat,
    NonnegativeInteger,
    OffsetAxis,
    PositiveInteger,
    RngKey,
    Threefry4x32,
)
from tensor_dslab.readout import simulation

from tensor_dslab import (
    AnalogWaveform,
    AnalogWaveformConfig,
    ChannelAxis,
    Charge,
    ChargeConfig,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    Pulse,
    PureWaveform,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    ZeroNoiseConfig,
    quantities,
    quantity,
    simulate_readout,
)


PRODUCTS = (
    Photoelectrons,
    Charge,
    PureWaveform,
    NoiseWaveform,
    AnalogWaveform,
    DigitizedWaveform,
)


def _fixture() -> tuple[Photoelectrons, ReadoutConfig]:
    axes = (
        ExampleAxis(count=2),
        ChannelAxis(labels=("a", "b")),
        SampleAxis(start=0, step=2000, count=4),
    )
    source = Photoelectrons(
        tensor=torch.tensor(
            [
                [[1, 0, 2, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [2, 0, 0, 0]],
            ],
            dtype=torch.int64,
        ),
        axes=axes,
    )
    config = ReadoutConfig(
        charge=ChargeConfig(
            correlated_avalanche_generations=NonnegativeInteger(0)
        ),
        pure_waveform=PureWaveformConfig(
            pulse=Pulse(
                quantity=quantities((-2.0, -1.0), "mV"),
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(relative_to=SampleAxis, offsets=(0, 1)),
                ),
            )
        ),
        noise_waveform=NoiseWaveformConfig(model=ZeroNoiseConfig()),
        analog_waveform=AnalogWaveformConfig(),
        digitized_waveform=DigitizedWaveformConfig(
            bit_depth=PositiveInteger(12),
            input_minimum=quantity(-20, "mV"),
            input_maximum=quantity(5, "mV"),
            analog_gain_db=NonnegativeFloat(0),
        ),
    )
    return source, config


class _FailingRng(CounterRng):
    __slots__ = ()
    calls: ClassVar[int] = 0

    @override
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls += 1
        raise AssertionError(
            f"unexpected RNG request: {key=}, {quantum=}, {block=}"
        )


class _OneShotProducts:
    def __init__(self, products: tuple[type, ...]) -> None:
        self.products = products
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("products iterable consumed more than once")
        return iter(self.products)


class ReadoutSimulationContractTest(unittest.TestCase):
    def test_all_63_nonempty_product_subsets(self) -> None:
        source, config = _fixture()
        for mask in range(1, 1 << len(PRODUCTS)):
            requested = tuple(
                product
                for index, product in enumerate(PRODUCTS)
                if mask & (1 << index)
            )
            result = simulate_readout(
                source,
                products=requested,
                config=config,
                rng=Threefry4x32(seed=mask),
            )
            self.assertEqual(result.field_types, frozenset(requested))

    def test_request_order_invariance_and_requested_only_retention(self) -> None:
        source, config = _fixture()
        left = simulate_readout(
            source,
            products=(Charge, PureWaveform),
            config=config,
            rng=Threefry4x32(seed=1),
        )
        right = simulate_readout(
            source,
            products=(PureWaveform, Charge),
            config=config,
            rng=Threefry4x32(seed=1),
        )
        self.assertTrue(torch.equal(left.field(Charge).tensor, right.field(Charge).tensor))
        self.assertTrue(
            torch.equal(left.field(PureWaveform).tensor, right.field(PureWaveform).tensor)
        )

    def test_all_product_relationship_storage_and_global_state_contract(self) -> None:
        source, config = _fixture()
        source_before = source.tensor.clone()
        global_before = torch.random.get_rng_state().clone()
        result = simulate_readout(
            source,
            products=PRODUCTS,
            config=config,
            rng=Threefry4x32(seed=11),
        )
        self.assertIs(result.field(Photoelectrons), source)
        pointers: list[int] = []
        for field_type in PRODUCTS:
            field = result.field(field_type)
            self.assertIs(field.axes, source.axes)
            self.assertEqual(field.tensor.shape, source.tensor.shape)
            pointers.append(field.tensor.untyped_storage().data_ptr())
        self.assertEqual(len(set(pointers)), len(pointers))
        self.assertIs(result.field(Charge).tensor.dtype, torch.float32)
        self.assertIs(result.field(PureWaveform).tensor.dtype, torch.float32)
        self.assertIs(result.field(NoiseWaveform).tensor.dtype, torch.float32)
        self.assertIs(result.field(AnalogWaveform).tensor.dtype, torch.float32)
        self.assertIs(result.field(DigitizedWaveform).tensor.dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                result.field(AnalogWaveform).tensor,
                result.field(PureWaveform).tensor
                + result.field(NoiseWaveform).tensor,
            )
        )
        self.assertTrue(torch.equal(source.tensor, source_before))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))

    def test_products_iterable_is_consumed_once(self) -> None:
        source, config = _fixture()
        products = _OneShotProducts((Photoelectrons, Charge))
        result = simulate_readout(
            source,
            products=products,
            config=config,
            rng=_FailingRng(seed=0),
        )
        self.assertEqual(products.iterations, 1)
        self.assertEqual(result.field_types, frozenset((Photoelectrons, Charge)))

    def test_truth_identity_and_source_immutability(self) -> None:
        source, config = _fixture()
        before = source.tensor.clone()
        result = simulate_readout(
            source,
            products=(Photoelectrons,),
            config=config,
            rng=Threefry4x32(seed=0),
        )
        self.assertIs(result.field(Photoelectrons), source)
        self.assertTrue(torch.equal(source.tensor, before))

    def test_invalid_requests_fail_before_execution(self) -> None:
        source, config = _fixture()
        with self.assertRaises(ValueError):
            simulate_readout(
                source,
                products=(),
                config=config,
                rng=Threefry4x32(seed=0),
            )

    def test_truth_preflight_failure_has_no_rng_producer_or_collection_effect(
        self,
    ) -> None:
        source, config = _fixture()
        invalid = Photoelectrons(
            tensor=source.tensor.clone(),
            axes=source.axes,
        )
        invalid.tensor[0, 0, 0] = -1
        _FailingRng.calls = 0
        with (
            mock.patch.object(
                simulation,
                "produce_charge",
                side_effect=AssertionError("producer must not run"),
            ),
            mock.patch.object(
                simulation,
                "ReadoutCollection",
                side_effect=AssertionError("collection must not be built"),
            ),
        ):
            with self.assertRaises(ValueError):
                simulate_readout(
                    invalid,
                    products=(Photoelectrons,),
                    config=config,
                    rng=_FailingRng(seed=0),
                )
        self.assertEqual(_FailingRng.calls, 0)
        with self.assertRaises(ValueError):
            simulate_readout(
                source,
                products=(Charge, Charge),
                config=config,
                rng=Threefry4x32(seed=0),
            )


for _mask in range(1, 17):
    def _subset_case(
        self: ReadoutSimulationContractTest,
        mask: int = _mask,
    ) -> None:
        source, config = _fixture()
        requested = tuple(
            product
            for index, product in enumerate(PRODUCTS)
            if mask & (1 << index)
        )
        result = simulate_readout(
            source,
            products=requested,
            config=config,
            rng=Threefry4x32(seed=mask),
        )
        self.assertEqual(result.field_types, frozenset(requested))

    setattr(ReadoutSimulationContractTest, f"test_request_subset_{_mask:02d}", _subset_case)
