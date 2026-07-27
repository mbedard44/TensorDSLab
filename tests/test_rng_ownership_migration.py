"""Compact package-owned RNG key and address evidence."""

import unittest
from typing import ClassVar, override

import torch
from tensor_core import (
    CounterRng,
    NonnegativeInteger,
    OffsetAxis,
    RngElements,
    RngKey,
    Threefry4x32,
)

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
from tensor_dslab.readout.runtime.addresses import branching_generation_address
from tensor_dslab.readout.runtime.keys import (
    AFTERPULSE_RNG_KEY,
    CHARGE_SMEARING_RNG_KEY,
    DARK_COUNT_RNG_KEY,
    DELAYED_CROSSTALK_RNG_KEY,
    DIRECT_CROSSTALK_RNG_KEY,
    PSD_NOISE_RNG_KEY,
    RNG_NAMESPACE,
    TIMING_JITTER_RNG_KEY,
    WHITE_NOISE_RNG_KEY,
)


class _RecordingRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[list[tuple[RngKey, torch.Tensor, int, int]]] = []

    @override
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls.append((key, positions.clone(), quantum, block))
        return Threefry4x32(seed=self.seed)._generate_block(  # pyright: ignore[reportPrivateUsage]
            key=key,
            positions=positions,
            quantum=quantum,
            block=block,
        )


class RngOwnershipContractTest(unittest.TestCase):
    def test_compact_key_table_is_exact(self) -> None:
        keys = (
            WHITE_NOISE_RNG_KEY,
            PSD_NOISE_RNG_KEY,
            DARK_COUNT_RNG_KEY,
            TIMING_JITTER_RNG_KEY,
            DIRECT_CROSSTALK_RNG_KEY,
            DELAYED_CROSSTALK_RNG_KEY,
            AFTERPULSE_RNG_KEY,
            CHARGE_SMEARING_RNG_KEY,
        )
        self.assertEqual(tuple(key.namespace for key in keys), (RNG_NAMESPACE,) * 8)
        self.assertEqual(tuple(key.stream for key in keys), tuple(range(1, 9)))

    def test_generation_address_preserves_elements_and_selects_generation(self) -> None:
        elements = RngElements.from_shape((2, 3), device="cpu")
        address = branching_generation_address(
            elements,
            key=DIRECT_CROSSTALK_RNG_KEY,
            maximum_generations=4,
            generation_index=2,
        )
        self.assertEqual(address.shape, ())
        self.assertEqual(address.element_shape, (2, 3))

    def test_obsolete_overflow_keys_are_absent(self) -> None:
        import tensor_dslab.readout.runtime.keys as keys

        self.assertFalse(hasattr(keys, "DIRECT_CROSSTALK_RETAINED_RNG_KEY"))
        self.assertFalse(hasattr(keys, "DELAYED_CROSSTALK_RETAINED_RNG_KEY"))

    def test_branching_execution_uses_exact_keys_and_generation_coordinates(
        self,
    ) -> None:
        axes = (
            ExampleAxis(count=2),
            ChannelAxis(labels=("c",)),
            SampleAxis(start=0, step=1, count=3),
        )
        source = Photoelectrons(
            tensor=torch.ones((2, 1, 3), dtype=torch.int64),
            axes=axes,
        )

        def kernel(
            leaf: type[DirectCrosstalk]
            | type[DelayedCrosstalk]
            | type[Afterpulse],
            offset: int,
        ) -> DirectCrosstalk | DelayedCrosstalk | Afterpulse:
            return leaf(
                quantity=quantities((1.1,), "dimensionless"),
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(relative_to=SampleAxis, offsets=(offset,)),
                ),
            )

        direct = kernel(DirectCrosstalk, 0)
        delayed = kernel(DelayedCrosstalk, 1)
        afterpulse = kernel(Afterpulse, 1)
        assert isinstance(direct, DirectCrosstalk)
        assert isinstance(delayed, DelayedCrosstalk)
        assert isinstance(afterpulse, Afterpulse)
        _RecordingRng.calls = []
        simulate_readout(
            source,
            products=(Charge,),
            config=ReadoutConfig(
                charge=ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(2),
                    direct_crosstalk=direct,
                    delayed_crosstalk=delayed,
                    afterpulse=afterpulse,
                )
            ),
            rng=_RecordingRng(seed=2),
        )
        streams = tuple(call[0].stream for call in _RecordingRng.calls)
        self.assertEqual(streams, (5, 6, 7, 5, 6, 7))
        generation_zero = _RecordingRng.calls[0][1]
        generation_one = _RecordingRng.calls[3][1]
        self.assertTrue(torch.equal(generation_zero, torch.arange(6)))
        self.assertTrue(torch.all(generation_one >= 6))
        self.assertTrue(
            all(call[2:] == (0, 0) for call in _RecordingRng.calls)
        )


for _generation in range(12):
    def _address_case(
        self: RngOwnershipContractTest,
        generation: int = _generation,
    ) -> None:
        count = 12
        address = branching_generation_address(
            RngElements.from_shape((2,), device="cpu"),
            key=AFTERPULSE_RNG_KEY,
            maximum_generations=count,
            generation_index=generation,
        )
        self.assertTrue(address.is_complete)
        self.assertEqual(address.element_shape, (2,))

    setattr(RngOwnershipContractTest, f"test_generation_address_{_generation:02d}", _address_case)
