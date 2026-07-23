from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from inspect import signature
from pathlib import Path
from typing import ClassVar, get_args, get_origin, get_type_hints
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    CounterRng,
    FiniteFloat,
    NonnegativeFloat,
    PositiveFloat,
    PositiveInteger,
    RngKey,
    TensorField,
    Threefry4x32,
)

import tensor_dslab
import tensor_dslab.readout as readout
from tensor_dslab import (
    quantities,
    quantity,
    AnalogSaturationConfig,
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
    PsdNoiseConfig,
    PureWaveform,
    PureWaveformConfig,
    ReadoutCollection,
    ReadoutConfig,
    SampleAxis,
    TpcFebSnrPulseConfig,
    ZeroNoiseConfig,
    simulate_readout,
)
from tensor_dslab.readout import simulation
from tensor_dslab.readout.analog_waveform.runtime.prepare import (
    AnalogWaveformRuntime,
)
from tensor_dslab.readout.analog_waveform.runtime.validate import (
    validate_analog_waveform,
)
from tensor_dslab.readout.charge.runtime.effects.correlated_avalanches import (
    CorrelatedAvalancheRuntime,
)
from tensor_dslab.readout.charge.runtime.effects.dark_counts import (
    DarkCountRuntime,
)
from tensor_dslab.readout.charge.runtime.effects.delays import (
    AfterpulseRuntime,
    DelayRuntime,
)
from tensor_dslab.readout.charge.runtime.effects.smearing import (
    ChargeSmearingRuntime,
)
from tensor_dslab.readout.charge.runtime.effects.timing_jitter import (
    TimingJitterRuntime,
)
from tensor_dslab.readout.charge.runtime.prepare import ChargeRuntime
from tensor_dslab.readout.charge.runtime.validate import validate_charge
from tensor_dslab.readout.digitized_waveform.runtime.prepare import (
    DigitizedWaveformRuntime,
)
from tensor_dslab.readout.digitized_waveform.runtime.validate import (
    validate_digitized_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    NoiseWaveformRuntime,
    PsdNoiseRuntime,
    WhiteNoiseRuntime,
    ZeroNoiseRuntime,
)
from tensor_dslab.readout.noise_waveform.runtime.validate import (
    validate_noise_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.prepare import (
    PureWaveformRuntime,
)
from tensor_dslab.readout.pure_waveform.runtime.validate import (
    validate_pure_waveform,
)
import tensor_dslab.readout.runtime.prepare as readout_preparer
from tensor_dslab.readout.runtime.prepare import ReadoutRuntime, prepare_readout
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


PRODUCT_TYPES: tuple[type[TensorField], ...] = (
    Photoelectrons,
    Charge,
    PureWaveform,
    NoiseWaveform,
    AnalogWaveform,
    DigitizedWaveform,
)

RUNTIME_TYPES = (
    SamplingRuntime,
    ReadoutRuntime,
    ChargeRuntime,
    PureWaveformRuntime,
    NoiseWaveformRuntime,
    ZeroNoiseRuntime,
    WhiteNoiseRuntime,
    PsdNoiseRuntime,
    AnalogWaveformRuntime,
    DigitizedWaveformRuntime,
    DelayRuntime,
    AfterpulseRuntime,
    DarkCountRuntime,
    TimingJitterRuntime,
    CorrelatedAvalancheRuntime,
    ChargeSmearingRuntime,
)


def _ns(value: int | float):
    return quantity(value, "ns")


def _hz(value: int | float):
    return quantity(value, "Hz")


def _mv(value: int | float):
    return quantity(value, "mV")


def _density(value: int | float):
    return quantity(value, "mV ** 2 / Hz")


class _FailingRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[int] = 0

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls += 1
        raise AssertionError("deterministic runtime requested RNG words")


def _source() -> Photoelectrons:
    axes = (
        ExampleAxis(count=1),
        ChannelAxis(labels=("channel-0",)),
        SampleAxis(start=0, step=2_000, count=4),
    )
    return Photoelectrons(
        tensor=torch.tensor([[[1, 0, 2, 1]]], dtype=torch.int64),
        axes=axes,
    )


def _config(*, psd: bool = False) -> ReadoutConfig:
    noise = (
        NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges=(_hz(0.0),),
                frequency_stop=_hz(300_000_000.0),
                power_density=(_density(1.0e-9),),
            )
        )
        if psd
        else NoiseWaveformConfig(model=ZeroNoiseConfig())
    )
    return ReadoutConfig(
        charge=ChargeConfig(),
        pure_waveform=PureWaveformConfig(
            model=TpcFebSnrPulseConfig(
                fast_time_constant=_ns(1.0),
                slow_time_constant=_ns(2.0),
                support_time=_ns(6.0),
                peak_voltage_per_photoelectron=_mv(-1.0),
            )
        ),
        noise_waveform=noise,
        analog_waveform=AnalogWaveformConfig(
            saturation=AnalogSaturationConfig(
                minimum=_mv(-10.0),
                maximum=_mv(10.0),
            )
        ),
        digitized_waveform=DigitizedWaveformConfig(
            bit_depth=PositiveInteger(12),
            input_minimum=_mv(-20.0),
            input_maximum=_mv(20.0),
            analog_gain_db=NonnegativeFloat(0.0),
        ),
    )


def _runtime_tensors(value: object) -> tuple[torch.Tensor, ...]:
    found: list[torch.Tensor] = []

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            found.append(item)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, tuple):
            for member in item:
                visit(member)

    visit(value)
    return tuple(found)


def _contains_type(annotation: object, candidates: tuple[type[object], ...]) -> bool:
    if annotation in candidates:
        return True
    origin = get_origin(annotation)
    return origin is not None and any(
        _contains_type(argument, candidates) for argument in get_args(annotation)
    )


class RuntimeActionOwnershipTest(unittest.TestCase):
    def test_exact_runtime_tree_and_retired_paths(self) -> None:
        expected = (
            "tensor_dslab/readout/runtime/prepare.py",
            "tensor_dslab/readout/runtime/sampling.py",
            "tensor_dslab/readout/photoelectrons/runtime/validate.py",
            "tensor_dslab/readout/charge/runtime/prepare.py",
            "tensor_dslab/readout/charge/runtime/produce.py",
            "tensor_dslab/readout/charge/runtime/validate.py",
            "tensor_dslab/readout/pure_waveform/runtime/prepare.py",
            "tensor_dslab/readout/pure_waveform/runtime/produce.py",
            "tensor_dslab/readout/pure_waveform/runtime/validate.py",
            "tensor_dslab/readout/noise_waveform/runtime/prepare.py",
            "tensor_dslab/readout/noise_waveform/runtime/produce.py",
            "tensor_dslab/readout/noise_waveform/runtime/validate.py",
            "tensor_dslab/readout/analog_waveform/runtime/prepare.py",
            "tensor_dslab/readout/analog_waveform/runtime/produce.py",
            "tensor_dslab/readout/analog_waveform/runtime/validate.py",
            "tensor_dslab/readout/digitized_waveform/runtime/prepare.py",
            "tensor_dslab/readout/digitized_waveform/runtime/produce.py",
            "tensor_dslab/readout/digitized_waveform/runtime/validate.py",
        )
        for path in expected:
            self.assertTrue(Path(path).is_file(), path)

        retired = (
            "tensor_dslab/readout/_requirements.py",
            "tensor_dslab/readout/charge/effects",
            "tensor_dslab/readout/charge/_produce.py",
            "tensor_dslab/readout/pure_waveform/_produce.py",
            "tensor_dslab/readout/noise_waveform/_produce.py",
            "tensor_dslab/readout/analog_waveform/_produce.py",
            "tensor_dslab/readout/digitized_waveform/_produce.py",
        )
        for path in retired:
            self.assertFalse(Path(path).exists(), path)

        runtime_markers = tuple(
            Path("tensor_dslab/readout").glob("runtime/__init__.py")
        ) + tuple(Path("tensor_dslab/readout").glob("*/runtime/__init__.py"))
        runtime_markers += (
            Path("tensor_dslab/readout/charge/runtime/effects/__init__.py"),
        )
        self.assertEqual(len(runtime_markers), 8)
        for marker in runtime_markers:
            self.assertEqual(marker.read_bytes(), b"", str(marker))

    def test_runtime_records_are_private_final_frozen_slotted_values(self) -> None:
        public_modules = (tensor_dslab, readout)
        for runtime_type in RUNTIME_TYPES:
            with self.subTest(runtime=runtime_type.__name__):
                self.assertEqual(runtime_type.__bases__, (object,))
                self.assertTrue(getattr(runtime_type, "__final__", False))
                self.assertTrue(is_dataclass(runtime_type))
                dataclass_params = getattr(runtime_type, "__dataclass_params__")
                self.assertTrue(getattr(dataclass_params, "frozen"))
                self.assertNotIn("__dict__", runtime_type.__slots__)
                annotations = get_type_hints(runtime_type)
                annotation_text = " ".join(
                    repr(value) for value in annotations.values()
                )
                for forbidden in (
                    "Config",
                    "Callable",
                    "list[",
                    "dict[",
                    "set[",
                    "Any",
                ):
                    self.assertNotIn(forbidden, annotation_text)
                semantic_types: tuple[type[object], ...] = (
                    *PRODUCT_TYPES,
                    ReadoutCollection,
                )
                self.assertFalse(
                    any(
                        _contains_type(annotation, semantic_types)
                        for annotation in annotations.values()
                    )
                )
                for module in public_modules:
                    self.assertNotIn(runtime_type.__name__, module.__all__)
                    self.assertFalse(hasattr(module, runtime_type.__name__))

        self.assertNotIn("requested", {field.name for field in fields(ReadoutRuntime)})

    def test_action_modules_and_execution_sequence_are_exact(self) -> None:
        actions = {
            "charge": ("prepare_charge", "produce_charge", "validate_charge"),
            "pure_waveform": (
                "prepare_pure_waveform",
                "produce_pure_waveform",
                "validate_pure_waveform",
            ),
            "noise_waveform": (
                "prepare_noise_waveform",
                "produce_noise_waveform",
                "validate_noise_waveform",
            ),
            "analog_waveform": (
                "prepare_analog_waveform",
                "produce_analog_waveform",
                "validate_analog_waveform",
            ),
            "digitized_waveform": (
                "prepare_digitized_waveform",
                "produce_digitized_waveform",
                "validate_digitized_waveform",
            ),
        }
        for product, names in actions.items():
            for action, module_leaf in zip(names, ("prepare", "produce", "validate")):
                with self.subTest(product=product, action=action):
                    module_name = (
                        f"tensor_dslab.readout.{product}.runtime.{module_leaf}"
                    )
                    module = __import__(module_name, fromlist=(action,))
                    value = getattr(module, action)
                    self.assertEqual(value.__module__, module_name)
                    self.assertNotIn(action, tensor_dslab.__all__)
                    self.assertNotIn(action, readout.__all__)

        photo_runtime = Path("tensor_dslab/readout/photoelectrons/runtime")
        self.assertEqual(
            {path.name for path in photo_runtime.glob("*.py")},
            {"__init__.py", "validate.py"},
        )

        tree = ast.parse(Path("tensor_dslab/readout/simulation.py").read_text())
        lifecycle_names = {
            "prepare_readout",
            "produce_charge",
            "validate_charge",
            "produce_pure_waveform",
            "validate_pure_waveform",
            "produce_noise_waveform",
            "validate_noise_waveform",
            "produce_analog_waveform",
            "validate_analog_waveform",
            "produce_digitized_waveform",
            "validate_digitized_waveform",
            "ReadoutCollection",
        }
        calls: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id in lifecycle_names:
                calls.append((node.lineno, function.id))
        observed = tuple(name for _, name in sorted(calls))
        self.assertEqual(
            observed,
            (
                "prepare_readout",
                "produce_charge",
                "validate_charge",
                "produce_pure_waveform",
                "validate_pure_waveform",
                "produce_noise_waveform",
                "validate_noise_waveform",
                "produce_analog_waveform",
                "validate_analog_waveform",
                "produce_digitized_waveform",
                "validate_digitized_waveform",
                "ReadoutCollection",
            ),
        )

    def test_producers_and_validators_keep_narrow_ownership(self) -> None:
        product_roots = (
            "charge",
            "pure_waveform",
            "noise_waveform",
            "analog_waveform",
            "digitized_waveform",
        )
        semantic_names = {
            "Charge",
            "PureWaveform",
            "NoiseWaveform",
            "AnalogWaveform",
            "DigitizedWaveform",
        }
        for product in product_roots:
            produce_path = Path(
                f"tensor_dslab/readout/{product}/runtime/produce.py"
            )
            produce_source = produce_path.read_text()
            produce_tree = ast.parse(produce_source)
            imports = {
                node.module
                for node in ast.walk(produce_tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            with self.subTest(product=product, owner="produce"):
                self.assertFalse(any(name.endswith(".config") for name in imports))
                self.assertFalse(any(name.endswith(".validate") for name in imports))
                self.assertNotIn("validate_", produce_source)
                self.assertNotIn("ReadoutRuntime", produce_source)
                self.assertNotIn("ReadoutCollection", produce_source)

            validate_path = Path(
                f"tensor_dslab/readout/{product}/runtime/validate.py"
            )
            validate_tree = ast.parse(validate_path.read_text())
            constructed = {
                node.func.id
                for node in ast.walk(validate_tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            with self.subTest(product=product, owner="validate"):
                self.assertTrue(constructed.isdisjoint(semantic_names))
                self.assertFalse(
                    any(
                        isinstance(node, ast.ImportFrom)
                        and node.module is not None
                        and node.module.endswith(".config")
                        for node in ast.walk(validate_tree)
                    )
                )

        for forbidden_name in ("utils.py", "helpers.py", "framework.py"):
            self.assertEqual(
                tuple(Path("tensor_dslab/readout").rglob(forbidden_name)),
                (),
            )

    def test_request_membership_and_shared_sampling_are_single_owned(self) -> None:
        source = _source()
        captured_sampling: list[SamplingRuntime] = []
        sample_dimension_calls: list[tuple[TensorField, type[SampleAxis]]] = []
        original = readout_preparer.prepare_sampling
        original_dimension_of = TensorField.dimension_of

        def capture_sampling(*args, **kwargs):  # type: ignore[no-untyped-def]
            runtime = original(*args, **kwargs)
            captured_sampling.append(runtime)
            return runtime

        def capture_dimension(
            field: TensorField,
            axis_type: type[SampleAxis],
        ) -> int:
            sample_dimension_calls.append((field, axis_type))
            return original_dimension_of(field, axis_type)

        with patch.object(
            ReadoutCollection,
            "accepted_field_types",
            wraps=ReadoutCollection.accepted_field_types,
        ) as accepted, patch.object(
            readout_preparer,
            "prepare_sampling",
            side_effect=capture_sampling,
        ) as prepare_sampling_call, patch.object(
            TensorField,
            "dimension_of",
            autospec=True,
            side_effect=capture_dimension,
        ):
            requested, runtime = prepare_readout(
                source,
                products=PRODUCT_TYPES,
                config=_config(),
                rng=_FailingRng(seed=0),
                floating_dtype=torch.float32,
            )

        accepted.assert_called_once_with()
        prepare_sampling_call.assert_called_once()
        self.assertEqual(requested, frozenset(PRODUCT_TYPES))
        self.assertEqual(len(captured_sampling), 1)
        self.assertEqual(
            sample_dimension_calls,
            [(source, SampleAxis), (source, SampleAxis)],
        )
        sampling = captured_sampling[0]
        assert runtime.charge is not None
        assert runtime.pure_waveform is not None
        assert runtime.noise_waveform is not None
        self.assertIs(runtime.charge.sampling, sampling)
        self.assertIs(runtime.pure_waveform.sampling, sampling)
        self.assertIs(runtime.noise_waveform.sampling, sampling)

        alternate_axes = (
            SampleAxis(start=0, step=4_000, count=6),
            ExampleAxis(count=1),
            ChannelAxis(labels=("channel-0",)),
        )
        alternate_source = Photoelectrons(
            tensor=torch.ones((6, 1, 1), dtype=torch.int64),
            axes=alternate_axes,
        )
        _, alternate_runtime = prepare_readout(
            alternate_source,
            products=(DigitizedWaveform,),
            config=_config(),
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float32,
        )
        assert alternate_runtime.charge is not None
        assert alternate_runtime.pure_waveform is not None
        assert alternate_runtime.noise_waveform is not None
        alternate_sampling = alternate_runtime.charge.sampling
        self.assertEqual(alternate_sampling.sample_count, 6)
        self.assertEqual(alternate_sampling.sample_period_ps, 4_000)
        self.assertEqual(alternate_sampling.sample_dimension, 0)
        self.assertIs(
            alternate_runtime.pure_waveform.sampling,
            alternate_sampling,
        )
        self.assertIs(
            alternate_runtime.noise_waveform.sampling,
            alternate_sampling,
        )

    def test_prepared_tensor_values_and_storage_are_read_only(self) -> None:
        source = _source()
        requested, runtime = prepare_readout(
            source,
            products=PRODUCT_TYPES,
            config=_config(psd=True),
            rng=Threefry4x32(seed=17),
            floating_dtype=torch.float64,
        )
        runtime_tensors = _runtime_tensors(runtime)
        self.assertGreaterEqual(len(runtime_tensors), 8)
        snapshots = tuple(
            (tensor, tensor.clone(), tensor.untyped_storage().data_ptr())
            for tensor in runtime_tensors
        )

        with patch.object(
            simulation,
            "prepare_readout",
            return_value=(requested, runtime),
        ):
            result = simulate_readout(
                source,
                products=PRODUCT_TYPES,
                config=_config(psd=True),
                rng=Threefry4x32(seed=17),
                floating_dtype=torch.float64,
            )
        self.assertEqual(result.field_types, frozenset(PRODUCT_TYPES))
        for tensor, values, storage in snapshots:
            self.assertEqual(tensor.untyped_storage().data_ptr(), storage)
            self.assertTrue(torch.equal(tensor, values))

    def test_pure_kernel_is_prepared_once_and_consumed_directly(self) -> None:
        prepare_source = Path(
            "tensor_dslab/readout/pure_waveform/runtime/prepare.py"
        ).read_text()
        produce_source = Path(
            "tensor_dslab/readout/pure_waveform/runtime/produce.py"
        ).read_text()
        self.assertEqual(prepare_source.count(".flip("), 1)
        self.assertIn(".reshape(1, 1, coefficient_count)", prepare_source)
        self.assertNotIn(".flip(", produce_source)
        self.assertNotIn("coefficients", produce_source)

        source = _source()
        _, runtime = prepare_readout(
            source,
            products=(PureWaveform,),
            config=_config(),
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float32,
        )
        assert runtime.charge is not None
        assert runtime.pure_waveform is not None
        kernel = runtime.pure_waveform.kernel
        before = kernel.clone()
        storage = kernel.untyped_storage().data_ptr()
        with patch.object(
            simulation,
            "prepare_readout",
            return_value=(frozenset((PureWaveform,)), runtime),
        ), patch(
            "tensor_dslab.readout.pure_waveform.runtime.produce.functional.conv1d",
            wraps=torch.nn.functional.conv1d,
        ) as convolution:
            result = simulate_readout(
                source,
                products=(PureWaveform,),
                config=_config(),
                rng=_FailingRng(seed=0),
                floating_dtype=torch.float32,
            )
        convolution.assert_called_once()
        self.assertIs(convolution.call_args.args[1], kernel)
        self.assertIs(type(result.field(PureWaveform)), PureWaveform)
        self.assertEqual(kernel.untyped_storage().data_ptr(), storage)
        self.assertTrue(torch.equal(kernel, before))

    def test_charge_has_one_terminal_scan_and_failure_precedes_descendants(
        self,
    ) -> None:
        produce_source = Path(
            "tensor_dslab/readout/charge/runtime/produce.py"
        ).read_text()
        validate_source = Path(
            "tensor_dslab/readout/charge/runtime/validate.py"
        ).read_text()
        self.assertNotIn("isfinite", produce_source)
        self.assertEqual(validate_source.count("torch.isfinite("), 1)

        source = _source()
        invalid = Charge(
            tensor=torch.full(source.shape, float("nan"), dtype=torch.float32),
            axes=source.axes,
        )
        with patch.object(
            simulation,
            "produce_charge",
            return_value=invalid,
        ) as charge_producer, patch.object(
            simulation,
            "produce_pure_waveform",
        ) as pure_producer, patch.object(
            simulation,
            "produce_noise_waveform",
        ) as noise_producer, patch.object(
            simulation,
            "ReadoutCollection",
        ) as collection:
            with self.assertRaisesRegex(RuntimeError, "invalid terminal value"):
                simulate_readout(
                    source,
                    products=(AnalogWaveform,),
                    config=_config(),
                    rng=_FailingRng(seed=0),
                )
        charge_producer.assert_called_once()
        pure_producer.assert_not_called()
        noise_producer.assert_not_called()
        collection.assert_not_called()

    def test_validators_own_relationships_and_alias_rejection(self) -> None:
        source = _source()
        _, runtime = prepare_readout(
            source,
            products=(Charge, NoiseWaveform),
            config=_config(),
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float32,
        )
        assert runtime.charge is not None
        assert runtime.noise_waveform is not None

        other_axes = (
            ExampleAxis(count=1),
            ChannelAxis(labels=("other-channel",)),
            SampleAxis(start=0, step=2_000, count=4),
        )
        wrong_charge = Charge(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=other_axes,
        )
        with self.assertRaisesRegex(RuntimeError, "axes and shape"):
            validate_charge(wrong_charge, source=source, runtime=runtime.charge)
        wrong_charge_dtype = Charge(
            tensor=torch.zeros(source.shape, dtype=torch.float64),
            axes=source.axes,
        )
        with self.assertRaisesRegex(RuntimeError, "prepared floating dtype"):
            validate_charge(
                wrong_charge_dtype,
                source=source,
                runtime=runtime.charge,
            )

        short_axes = (
            source.axis(ExampleAxis),
            source.axis(ChannelAxis),
            SampleAxis(start=0, step=2_000, count=2),
        )
        short_charge = Charge(
            tensor=torch.zeros((1, 1, 2), dtype=torch.float32),
            axes=short_axes,
        )
        with self.assertRaisesRegex(RuntimeError, "axes and shape"):
            validate_charge(short_charge, source=source, runtime=runtime.charge)

        wrong_axes_noise = NoiseWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=other_axes,
        )
        with self.assertRaisesRegex(ValueError, "axes and shape"):
            validate_noise_waveform(
                wrong_axes_noise,
                source=source,
                runtime=runtime.noise_waveform,
            )
        wrong_noise = NoiseWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float64),
            axes=source.axes,
        )
        with self.assertRaisesRegex(ValueError, "prepared floating dtype"):
            validate_noise_waveform(
                wrong_noise,
                source=source,
                runtime=runtime.noise_waveform,
            )
        short_noise = NoiseWaveform(
            tensor=torch.zeros((1, 1, 2), dtype=torch.float32),
            axes=short_axes,
        )
        with self.assertRaisesRegex(ValueError, "axes and shape"):
            validate_noise_waveform(
                short_noise,
                source=source,
                runtime=runtime.noise_waveform,
            )

        charge = Charge(
            tensor=torch.ones(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        wrong_axes_pure = PureWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=other_axes,
        )
        with self.assertRaisesRegex(ValueError, "axes and shape"):
            validate_pure_waveform(wrong_axes_pure, source=charge)
        short_pure = PureWaveform(
            tensor=torch.zeros((1, 1, 2), dtype=torch.float32),
            axes=short_axes,
        )
        with self.assertRaisesRegex(ValueError, "axes and shape"):
            validate_pure_waveform(short_pure, source=charge)
        wrong_dtype_pure = PureWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float64),
            axes=source.axes,
        )
        with self.assertRaisesRegex(ValueError, "preserve Charge dtype"):
            validate_pure_waveform(wrong_dtype_pure, source=charge)
        aliased_pure = PureWaveform(tensor=charge.tensor, axes=charge.axes)
        with self.assertRaisesRegex(ValueError, "fresh storage"):
            validate_pure_waveform(aliased_pure, source=charge)

        noise = NoiseWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        pure = PureWaveform(
            tensor=torch.ones(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        wrong_axes_analog = AnalogWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=other_axes,
        )
        with self.assertRaisesRegex(ValueError, "prerequisite axes and shape"):
            validate_analog_waveform(
                wrong_axes_analog,
                pure=pure,
                noise=noise,
            )
        short_analog = AnalogWaveform(
            tensor=torch.zeros((1, 1, 2), dtype=torch.float32),
            axes=short_axes,
        )
        with self.assertRaisesRegex(ValueError, "prerequisite axes and shape"):
            validate_analog_waveform(
                short_analog,
                pure=pure,
                noise=noise,
            )
        aliased_analog = AnalogWaveform(tensor=charge.tensor, axes=charge.axes)
        with self.assertRaisesRegex(ValueError, "fresh storage"):
            validate_analog_waveform(
                aliased_analog,
                pure=aliased_pure,
                noise=noise,
            )

        source_analog = AnalogWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        wrong_axes_digitized = DigitizedWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.int32),
            axes=other_axes,
        )
        with self.assertRaisesRegex(ValueError, "axes and shape"):
            validate_digitized_waveform(
                wrong_axes_digitized,
                source=source_analog,
                maximum_code=(1 << 12) - 1,
            )
        short_digitized = DigitizedWaveform(
            tensor=torch.zeros((1, 1, 2), dtype=torch.int32),
            axes=short_axes,
        )
        with self.assertRaisesRegex(ValueError, "axes and shape"):
            validate_digitized_waveform(
                short_digitized,
                source=source_analog,
                maximum_code=(1 << 12) - 1,
            )
        aliased_digitized = DigitizedWaveform(
            tensor=source_analog.tensor.view(torch.int32),
            axes=source_analog.axes,
        )
        with self.assertRaisesRegex(ValueError, "fresh storage"):
            validate_digitized_waveform(
                aliased_digitized,
                source=source_analog,
                maximum_code=(1 << 12) - 1,
            )

        _, float64_runtime = prepare_readout(
            source,
            products=(Charge, NoiseWaveform),
            config=_config(),
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float64,
        )
        assert float64_runtime.charge is not None
        assert float64_runtime.noise_waveform is not None
        aliased_charge = Charge(
            tensor=source.tensor.view(torch.float64),
            axes=source.axes,
        )
        with self.assertRaisesRegex(RuntimeError, "fresh storage"):
            validate_charge(
                aliased_charge,
                source=source,
                runtime=float64_runtime.charge,
            )
        aliased_noise = NoiseWaveform(
            tensor=source.tensor.view(torch.float64),
            axes=source.axes,
        )
        with self.assertRaisesRegex(ValueError, "fresh storage"):
            validate_noise_waveform(
                aliased_noise,
                source=source,
                runtime=float64_runtime.noise_waveform,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_runtime_relationships_execute_on_cuda(self) -> None:
        cpu = _source()
        source = Photoelectrons(tensor=cpu.tensor.cuda(), axes=cpu.axes)
        result = simulate_readout(
            source,
            products=PRODUCT_TYPES,
            config=_config(),
            rng=Threefry4x32(seed=17),
            floating_dtype=torch.float32,
        )
        self.assertTrue(
            all(
                field.tensor.device.type == "cuda"
                for field in result.fields.values()
            )
        )
        _, runtime = prepare_readout(
            source,
            products=(Charge, NoiseWaveform),
            config=_config(),
            rng=Threefry4x32(seed=17),
            floating_dtype=torch.float32,
        )
        assert runtime.charge is not None
        assert runtime.noise_waveform is not None
        cpu_charge = Charge(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        with self.assertRaisesRegex(RuntimeError, "source device"):
            validate_charge(cpu_charge, source=source, runtime=runtime.charge)
        cpu_noise = NoiseWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        with self.assertRaisesRegex(ValueError, "source device"):
            validate_noise_waveform(
                cpu_noise,
                source=source,
                runtime=runtime.noise_waveform,
            )
        cuda_charge = Charge(
            tensor=torch.zeros(source.shape, dtype=torch.float32, device="cuda"),
            axes=source.axes,
        )
        cpu_pure = PureWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        with self.assertRaisesRegex(ValueError, "Charge device"):
            validate_pure_waveform(cpu_pure, source=cuda_charge)
        cuda_pure = PureWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32, device="cuda"),
            axes=source.axes,
        )
        cuda_noise = NoiseWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32, device="cuda"),
            axes=source.axes,
        )
        cpu_analog = AnalogWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32),
            axes=source.axes,
        )
        with self.assertRaisesRegex(ValueError, "same device"):
            validate_analog_waveform(
                cpu_analog,
                pure=cuda_pure,
                noise=cuda_noise,
            )
        cuda_analog = AnalogWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.float32, device="cuda"),
            axes=source.axes,
        )
        cpu_digitized = DigitizedWaveform(
            tensor=torch.zeros(source.shape, dtype=torch.int32),
            axes=source.axes,
        )
        with self.assertRaisesRegex(ValueError, "source device"):
            validate_digitized_waveform(
                cpu_digitized,
                source=cuda_analog,
                maximum_code=(1 << 12) - 1,
            )


if __name__ == "__main__":
    unittest.main()
