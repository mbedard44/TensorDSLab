"""Independent physical-kernel, geometry, and alignment evidence."""

from typing import cast, final, override
import unittest

import numpy as np
import pint
from pint import Quantity
import torch
from tensor_core import LabelAxis, NonnegativeInteger, OffsetAxis, Threefry4x32

from tensor_dslab import (
    Afterpulse,
    ChannelAxis,
    Charge,
    ChargeConfig,
    DarkCountRate,
    DelayedCrosstalk,
    DirectCrosstalk,
    ExampleAxis,
    Photoelectrons,
    Pulse,
    PureWaveform,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    SmearingWidth,
    TimingJitter,
    quantities,
    quantity,
    simulate_readout,
)


def _axes(
    *,
    channels: tuple[str, ...] = ("a", "b"),
    samples: int = 5,
) -> tuple[ExampleAxis, ChannelAxis, SampleAxis]:
    return (
        ExampleAxis(count=2),
        ChannelAxis(labels=channels),
        SampleAxis(start=0, step=2000, count=samples),
    )


def _source(
    tensor: torch.Tensor | None = None,
    *,
    axes: tuple | None = None,
) -> Photoelectrons:
    exact_axes = _axes() if axes is None else axes
    values = (
        torch.zeros(tuple(axis.size for axis in exact_axes), dtype=torch.int64)
        if tensor is None
        else tensor
    )
    return Photoelectrons(tensor=values, axes=exact_axes)


class QuantityKernelContractTest(unittest.TestCase):
    def test_scalar_numpy_tuple_and_tensor_inputs_are_snapshotted(self) -> None:
        scalar = DarkCountRate(
            quantity=quantity(2, "kHz"),
            conditioning_axes=(),
            operation_axes=(),
        )
        self.assertEqual(scalar.tensor.dtype, torch.float64)
        self.assertEqual(scalar.tensor.device.type, "cpu")
        self.assertEqual(float(scalar.tensor), 2000.0)

        values = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
        timing = TimingJitter(
            quantity=quantities(values, "dimensionless"),
            conditioning_axes=(ExampleAxis(count=1),),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0, 1)),
            ),
        )
        values.zero_()
        self.assertEqual(timing.tensor.tolist(), [[0.25, 0.75]])
        self.assertEqual(timing.tensor.dtype, torch.float64)

        vector = DarkCountRate(
            quantity=quantities((1.0, 2.0), "kHz"),
            conditioning_axes=(ChannelAxis(labels=("a", "b")),),
            operation_axes=(),
        )
        self.assertEqual(vector.tensor.tolist(), [1000.0, 2000.0])

        external_registry = pint.UnitRegistry(cache_folder=None)
        numpy_source = np.array([[1.0, 2.0], [3.0, 4.0]])
        numpy_kernel = DirectCrosstalk(
            quantity=cast(
                Quantity,
                external_registry.Quantity(
                    numpy_source,
                    "dimensionless",
                ),
            ),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=ChannelAxis, offsets=(0, 1)),
                OffsetAxis(relative_to=SampleAxis, offsets=(0, 1)),
            ),
        )
        numpy_source.fill(0)
        self.assertEqual(
            numpy_kernel.tensor.tolist(),
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_quantity_view_is_canonical_and_read_only(self) -> None:
        width = SmearingWidth(
            quantity=quantity(25, "percent"),
            conditioning_axes=(),
            operation_axes=(),
        )
        self.assertAlmostEqual(float(width.quantity.magnitude), 0.25)
        self.assertEqual(str(width.quantity.units), "dimensionless")
        with self.assertRaises(ValueError):
            width.quantity.magnitude[...] = 1.0
        with self.assertRaises(TypeError):
            hash(width)
        self.assertNotIn("tensor(", repr(width))

    def test_operation_target_roles_are_unique(self) -> None:
        with self.assertRaises(ValueError):
            DirectCrosstalk(
                quantity=quantities(
                    torch.ones((1, 1), dtype=torch.float64),
                    "dimensionless",
                ),
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
                    OffsetAxis(relative_to=SampleAxis, offsets=(1,)),
                ),
            )

    def test_leaf_domains_and_geometry_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            DarkCountRate(
                quantity=quantity(-1, "Hz"),
                conditioning_axes=(),
                operation_axes=(),
            )
        with self.assertRaises(ValueError):
            TimingJitter(
                quantity=quantities((0.2, 0.2), "dimensionless"),
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(relative_to=SampleAxis, offsets=(0, 1)),
                ),
            )
        with self.assertRaises(ValueError):
            Afterpulse(
                quantity=quantities((0.1,), "dimensionless"),
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
                ),
            )

    def test_config_generation_relationship_is_fail_closed(self) -> None:
        direct = DirectCrosstalk(
            quantity=quantities((0.1,), "dimensionless"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
            ),
        )
        with self.assertRaises(ValueError):
            ChargeConfig(correlated_avalanche_generations=NonnegativeInteger(1))
        with self.assertRaises(ValueError):
            ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0),
                direct_crosstalk=direct,
            )

    def test_conditioning_coordinate_permutation_is_applied(self) -> None:
        axes = _axes(channels=("left", "right"), samples=4)
        rate = DarkCountRate(
            quantity=quantities((0.0, 1.0e12), "Hz"),
            conditioning_axes=(ChannelAxis(labels=("right", "left")),),
            operation_axes=(),
        )
        config = ReadoutConfig(
            charge=ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0),
                dark_counts=rate,
            )
        )
        result = simulate_readout(
            _source(axes=axes),
            products=(Charge,),
            config=config,
            rng=Threefry4x32(seed=9),
        ).field(Charge)
        self.assertTrue(torch.all(result.tensor[:, 0] > 0))
        self.assertTrue(torch.equal(result.tensor[:, 1], torch.zeros_like(result.tensor[:, 1])))

    def test_equal_length_different_coordinates_are_rejected(self) -> None:
        axes = _axes(channels=("left", "right"))
        rate = DarkCountRate(
            quantity=quantities((1.0, 2.0), "Hz"),
            conditioning_axes=(ChannelAxis(labels=("left", "other")),),
            operation_axes=(),
        )
        with self.assertRaises(ValueError):
            simulate_readout(
                _source(axes=axes),
                products=(Charge,),
                config=ReadoutConfig(
                    charge=ChargeConfig(
                        correlated_avalanche_generations=NonnegativeInteger(0),
                        dark_counts=rate,
                    )
                ),
                rng=Threefry4x32(seed=1),
            )

    def test_combined_example_channel_conditioning_is_aligned(self) -> None:
        axes = _axes(channels=("left", "right"), samples=2)
        rates = DarkCountRate(
            quantity=quantities(
                torch.tensor(
                    [[0.0, 5.0e11], [1.0e12, 0.0]],
                    dtype=torch.float64,
                ),
                "Hz",
            ),
            conditioning_axes=(
                ChannelAxis(labels=("right", "left")),
                ExampleAxis(count=2),
            ),
            operation_axes=(),
        )
        result = simulate_readout(
            _source(axes=axes),
            products=(Charge,),
            config=ReadoutConfig(
                charge=ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(0),
                    dark_counts=rates,
                )
            ),
            rng=Threefry4x32(seed=31),
        ).field(Charge).tensor
        self.assertTrue(torch.equal(result[0, 1], torch.zeros_like(result[0, 1])))
        self.assertTrue(torch.equal(result[1, 0], torch.zeros_like(result[1, 0])))
        self.assertGreater(float(result[0, 0].mean()), 1.0)
        self.assertGreater(float(result[1, 1].mean()), 0.2)

    def test_missing_conditioning_role_is_rejected(self) -> None:
        @final
        class ForeignAxis(LabelAxis):
            __slots__ = ()

            @override
            def _require(self) -> None:
                return

        source = _source()
        rate = DarkCountRate(
            quantity=quantities((1.0, 2.0), "Hz"),
            conditioning_axes=(ForeignAxis(labels=("a", "b")),),
            operation_axes=(),
        )
        with self.assertRaises(ValueError):
            simulate_readout(
                source,
                products=(Charge,),
                config=ReadoutConfig(
                    charge=ChargeConfig(
                        correlated_avalanche_generations=NonnegativeInteger(0),
                        dark_counts=rate,
                    )
                ),
                rng=Threefry4x32(seed=1),
            )

    def test_pulse_convolution_has_one_polarity_and_finite_window(self) -> None:
        axes = _axes(samples=4)
        source = _source(
            torch.tensor(
                [[[2, 0, 0, 0], [0, 0, 0, 0]]] * 2,
                dtype=torch.int64,
            ),
            axes=axes,
        )
        pulse = Pulse(
            quantity=quantities((-2.0, -1.0), "mV"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0, 2)),
            ),
        )
        result = simulate_readout(
            source,
            products=(PureWaveform,),
            config=ReadoutConfig(
                charge=ChargeConfig(
                    correlated_avalanche_generations=NonnegativeInteger(0)
                ),
                pure_waveform=PureWaveformConfig(pulse=pulse),
            ),
            rng=Threefry4x32(seed=0),
        ).field(PureWaveform)
        self.assertEqual(result.tensor[0, 0].tolist(), [-4.0, 0.0, -2.0, 0.0])


_LEAF_CASES = (
    (DarkCountRate, "Hz", (), ()),
    (SmearingWidth, "dimensionless", (), ()),
    (
        TimingJitter,
        "dimensionless",
        (1.0,),
        (OffsetAxis(relative_to=SampleAxis, offsets=(0,)),),
    ),
    (
        DirectCrosstalk,
        "dimensionless",
        (0.1,),
        (OffsetAxis(relative_to=SampleAxis, offsets=(0,)),),
    ),
    (
        DelayedCrosstalk,
        "dimensionless",
        (0.1,),
        (OffsetAxis(relative_to=SampleAxis, offsets=(1,)),),
    ),
    (
        Afterpulse,
        "dimensionless",
        (0.1,),
        (OffsetAxis(relative_to=SampleAxis, offsets=(1,)),),
    ),
    (
        Pulse,
        "mV",
        (-1.0,),
        (OffsetAxis(relative_to=SampleAxis, offsets=(0,)),),
    ),
)


for _case_index in range(70):
    def _construction_case(
        self: QuantityKernelContractTest,
        case_index: int = _case_index,
    ) -> None:
        leaf, unit, raw, operation_axes = _LEAF_CASES[
            case_index % len(_LEAF_CASES)
        ]
        magnitude = (
            quantity(float(case_index + 1), unit)
            if not operation_axes
            else quantities(raw, unit)
        )
        kernel = leaf(
            quantity=magnitude,
            conditioning_axes=(),
            operation_axes=operation_axes,
        )
        self.assertIs(type(kernel), leaf)
        self.assertEqual(kernel.conditioning_axes, ())
        self.assertEqual(kernel.operation_axes, operation_axes)
        self.assertEqual(kernel.tensor.dtype, torch.float64)
        self.assertFalse(kernel.tensor.requires_grad)

    setattr(
        QuantityKernelContractTest,
        f"test_quantity_kernel_construction_{_case_index:02d}",
        _construction_case,
    )
