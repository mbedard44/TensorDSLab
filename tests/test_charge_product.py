import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    CountCoordinates,
    CounterRng,
    LabelCoordinates,
    NonnegativeInteger,
    RegularCoordinates,
    TensorField,
    Threefry4x32,
)
from typing import override

from tensor_dslab import (
    AnalogWaveform,
    ChannelAxis,
    Charge,
    ChargeConfig,
    ChargeKernels,
    ChargeSpec,
    DarkCountRate,
    DarkCountRateSpec,
    ExampleAxis,
    Photoelectrons,
    PhotoelectronsSpec,
    QuantityFieldSpec,
    SmearingWidth,
    SmearingWidthSpec,
    TimeAxis,
    unit_registry,
)
from tests._product_support import analog_config, charge_config, source


class _FailingRng(CounterRng):
    @override
    def _generate_block(
        self,
        *,
        key,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        raise AssertionError("source binding must precede every RNG request")


class _ApplicationSpec(QuantityFieldSpec[tuple]):
    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        pass


class _OtherApplicationSpec(QuantityFieldSpec[tuple]):
    __slots__ = ()

    @override
    def _require_quantity_field_spec(self) -> None:
        pass


class _ApplicationField(TensorField[_ApplicationSpec]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        pass


class _OtherApplicationField(TensorField[_OtherApplicationSpec]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        pass


class ChargeProductTests(unittest.TestCase):
    def test_deterministic_charge_preserves_counts_and_source(self) -> None:
        source_field = source()
        before = source_field.tensor.clone()
        result = Charge.create(
            sources=(source_field,),
            config=charge_config(),
            rng=Threefry4x32(seed=7),
        )
        self.assertTrue(torch.equal(result.tensor, before.to(torch.float32)))
        self.assertTrue(torch.equal(source_field.tensor, before))
        self.assertNotEqual(
            result.tensor.untyped_storage().data_ptr(),
            source_field.tensor.untyped_storage().data_ptr(),
        )

    def test_staged_equal_spec_reuse_and_changed_spec_rejection(self) -> None:
        original = source()
        config = Charge.prepare(
            source_specs=(original.spec,),
            config=charge_config(),
        )
        equal_spec = PhotoelectronsSpec(
            axes=tuple(original.spec.axes),
            device=original.spec.device,
            dtype=original.spec.dtype,
            unit=original.spec.unit,
        )
        self.assertIsNot(equal_spec, original.spec)
        self.assertEqual(equal_spec, original.spec)
        equal = type(original)(tensor=original.tensor.clone(), spec=equal_spec)
        product = Charge.produce(
            sources=(equal,), config=config, rng=Threefry4x32(seed=1)
        )
        Charge.validate(product=product, sources=(equal,), config=config)
        self.assertIs(config._source_specs[0], original.spec)

        changed_axes = (
            original.spec.axes[0],
            ChannelAxis(coordinates=LabelCoordinates(labels=("a", "changed"))),
            original.spec.axes[2],
        )
        changed_spec = PhotoelectronsSpec(
            axes=changed_axes,
            device=original.spec.device,
            dtype=original.spec.dtype,
            unit=original.spec.unit,
        )
        changed = Photoelectrons(
            tensor=original.tensor.clone(),
            spec=changed_spec,
        )
        changed_unit_spec = PhotoelectronsSpec(
            axes=original.spec.axes,
            device=original.spec.device,
            dtype=original.spec.dtype,
            unit=unit_registry.Unit("milliavalanche"),
        )
        changed_unit = Photoelectrons(
            tensor=original.tensor.clone(),
            spec=changed_unit_spec,
        )
        for invalid in (changed, changed_unit):
            with self.subTest(spec=invalid.spec):
                with (
                    patch(
                        "tensor_dslab.charge.runtime.produce.torch.zeros",
                        side_effect=AssertionError(
                            "source binding must precede allocation"
                        ),
                    ),
                    self.assertRaises(ValueError),
                ):
                    Charge.produce(
                        sources=(invalid,),
                        config=config,
                        rng=_FailingRng(seed=0),
                    )
                with self.assertRaises(ValueError):
                    Charge.validate(
                        product=product,
                        sources=(invalid,),
                        config=config,
                    )

    def test_staged_binding_covers_dtype_device_semantic_type_count_and_order(
        self,
    ) -> None:
        output_config = analog_config()
        axes = output_config.spec.axes
        first_spec = _ApplicationSpec(
            axes=axes,
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit=unit_registry.Unit("mV"),
        )
        second_spec = _ApplicationSpec(
            axes=axes,
            device=torch.device("cpu"),
            dtype=torch.float64,
            unit=unit_registry.Unit("mV"),
        )
        first = _ApplicationField(
            tensor=torch.ones(first_spec.shape, dtype=first_spec.dtype),
            spec=first_spec,
        )
        second = _ApplicationField(
            tensor=torch.ones(second_spec.shape, dtype=second_spec.dtype),
            spec=second_spec,
        )
        prepared = AnalogWaveform.prepare(
            source_specs=(first_spec, second_spec),
            config=output_config,
        )
        product = AnalogWaveform.produce(
            sources=(first, second),
            config=prepared,
        )
        AnalogWaveform.validate(
            product=product,
            sources=(first, second),
            config=prepared,
        )

        changed_dtype_spec = _ApplicationSpec(
            axes=axes,
            device=torch.device("cpu"),
            dtype=torch.float64,
            unit=unit_registry.Unit("mV"),
        )
        changed_dtype = _ApplicationField(
            tensor=torch.ones(changed_dtype_spec.shape, dtype=torch.float64),
            spec=changed_dtype_spec,
        )
        changed_device_spec = _ApplicationSpec(
            axes=axes,
            device=torch.device("meta"),
            dtype=torch.float32,
            unit=unit_registry.Unit("mV"),
        )
        changed_device = _ApplicationField(
            tensor=torch.empty(
                changed_device_spec.shape,
                dtype=torch.float32,
                device="meta",
            ),
            spec=changed_device_spec,
        )
        changed_semantic_spec = _OtherApplicationSpec(
            axes=axes,
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit=unit_registry.Unit("mV"),
        )
        changed_semantic = _OtherApplicationField(
            tensor=torch.ones(changed_semantic_spec.shape),
            spec=changed_semantic_spec,
        )
        invalid_sources = (
            (changed_dtype, second),
            (changed_device, second),
            (changed_semantic, second),
            (first,),
            (second, first),
        )
        for supplied in invalid_sources:
            with self.subTest(sources=tuple(type(value) for value in supplied)):
                with (
                    patch(
                        "tensor_dslab.analog_waveform.runtime.produce.torch.zeros",
                        side_effect=AssertionError(
                            "source binding must precede arithmetic and allocation"
                        ),
                    ),
                    self.assertRaises(ValueError),
                ):
                    AnalogWaveform.produce(
                        sources=supplied,
                        config=prepared,
                    )
                with self.assertRaises(ValueError):
                    AnalogWaveform.validate(
                        product=product,
                        sources=supplied,
                        config=prepared,
                    )

    def test_charge_preparation_rejects_unit_and_device_mismatch(self) -> None:
        target = charge_config()
        incompatible = _ApplicationSpec(
            axes=target.spec.axes,
            device=target.spec.device,
            dtype=torch.int64,
            unit=unit_registry.Unit("mV"),
        )
        wrong_device = _ApplicationSpec(
            axes=target.spec.axes,
            device=torch.device("meta"),
            dtype=torch.int64,
            unit=unit_registry.Unit("avalanche"),
        )
        for supplied in (incompatible, wrong_device):
            with self.subTest(spec=supplied):
                with self.assertRaises(ValueError):
                    Charge.prepare(
                        source_specs=(supplied,),
                        config=target,
                    )

    def test_dark_count_and_smearing_statistics_replay(self) -> None:
        example = ExampleAxis(coordinates=CountCoordinates(count=30000))
        time = TimeAxis(
            coordinates=RegularCoordinates(start=0, step=1, count=1),
            coordinate_scale=1.0,
            unit=unit_registry.Unit("s"),
        )
        source_spec = PhotoelectronsSpec(
            axes=(example, time),
            device=torch.device("cpu"),
            dtype=torch.int64,
            unit=unit_registry.Unit("avalanche"),
        )
        source_field = Photoelectrons(
            tensor=torch.zeros(source_spec.shape, dtype=torch.int64),
            spec=source_spec,
        )
        dark = DarkCountRate(
            tensor=torch.tensor(0.5, dtype=torch.float64),
            spec=DarkCountRateSpec(
                conditioning_axes=(),
                operation_axes=(),
                device=torch.device("cpu"),
                dtype=torch.float64,
                unit=unit_registry.Unit("avalanche / s"),
            ),
        )
        dark_config = ChargeConfig(
            spec=ChargeSpec(
                axes=source_spec.axes,
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("avalanche"),
            ),
            kernels=ChargeKernels(members=(dark,)),
            correlated_avalanche_generations=NonnegativeInteger(value=0),
        )
        left = Charge.create(
            sources=(source_field,),
            config=dark_config,
            rng=Threefry4x32(seed=29),
        )
        right = Charge.create(
            sources=(source_field,),
            config=dark_config,
            rng=Threefry4x32(seed=29),
        )
        self.assertTrue(torch.equal(left.tensor, right.tensor))
        self.assertLess(abs(float(left.tensor.mean()) - 0.5), 0.04)

        count_spec = PhotoelectronsSpec(
            axes=(example,),
            device=torch.device("cpu"),
            dtype=torch.int64,
            unit=unit_registry.Unit("avalanche"),
        )
        count_source = Photoelectrons(
            tensor=torch.full(count_spec.shape, 100, dtype=torch.int64),
            spec=count_spec,
        )
        width = SmearingWidth(
            tensor=torch.tensor(0.1, dtype=torch.float64),
            spec=SmearingWidthSpec(
                conditioning_axes=(),
                operation_axes=(),
                device=torch.device("cpu"),
                dtype=torch.float64,
                unit=unit_registry.Unit(""),
            ),
        )
        smear_config = ChargeConfig(
            spec=ChargeSpec(
                axes=count_spec.axes,
                device=torch.device("cpu"),
                dtype=torch.float32,
                unit=unit_registry.Unit("avalanche"),
            ),
            kernels=ChargeKernels(members=(width,)),
            correlated_avalanche_generations=NonnegativeInteger(value=0),
        )
        smeared = Charge.create(
            sources=(count_source,),
            config=smear_config,
            rng=Threefry4x32(seed=31),
        )
        self.assertLess(abs(float(smeared.tensor.mean()) - 100.0), 0.08)
        self.assertLess(abs(float(smeared.tensor.var()) - 1.0), 0.08)
