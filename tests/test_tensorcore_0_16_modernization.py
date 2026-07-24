import ast
from dataclasses import fields
import importlib.util
import inspect
import math
from pathlib import Path
from typing import cast, ClassVar, get_type_hints, override
import unittest
from unittest.mock import patch

import numpy as np
import pint
from pint import Quantity
import torch
import tensor_core
import tensor_core.random as tensor_random
import tensor_core.random.validation as random_validation
import tensor_core.scalar as scalar
import tensor_core.scalar.validation as scalar_validation
import tensor_core.table as table
import tensor_core.tensor as tensor
import tensor_core.tensor.validation as tensor_validation
from tensor_core import (
    CounterRng,
    RngKey,
    RngPositions,
    TensorField,
)
from tensor_core.tensor.validation import (
    require_field_dtype,
    require_field_layout,
    require_representable_float,
)
from tensor_core.tensor.validation import (
    require_shape_span,
    require_tensor_allocation,
)
from tensor_core.tensor.validation import require_index
from tensor_core.random.validation import require_count_tensor

import tensor_dslab
import tensor_dslab.common as common
import tensor_dslab.readout as readout
from tensor_dslab import (
    quantities,
    quantity,
    ChannelAxis,
    Charge,
    ExampleAxis,
    Photoelectrons,
    PsdNoiseConfig,
    PureWaveformConfig,
    SampleAxis,
    TpcFebSnrPulseConfig,
)
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    PsdNoiseRuntime,
    prepare_noise_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.prepare import (
    prepare_pure_waveform,
)
from tensor_dslab.readout.runtime import keys
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


_ROOT_EXPORTS = (
    "TensorArtifact",
    "TensorAxis",
    "CountAxis",
    "RegularAxis",
    "LabelAxis",
    "TensorCollection",
    "TensorField",
    "TableColumn",
    "TableCollection",
    "TableField",
    "RngKey",
    "CounterRng",
    "Threefry4x32",
    "Scalar",
    "FiniteFloat",
    "NonnegativeFloat",
    "NonnegativeInteger",
    "PositiveFloat",
    "PositiveInteger",
    "Probability",
    "RngPositions",
)
_TENSOR_VALIDATION_EXPORTS = (
    "require_axis_signature",
    "require_device",
    "require_dimension",
    "require_field_dtype",
    "require_field_layout",
    "require_field_types",
    "require_index",
    "require_representable_float",
    "require_same_axes",
    "require_same_device",
    "require_same_dtype",
    "require_shape",
    "require_shape_span",
    "require_tensor",
    "require_tensor_allocation",
)
_FIXED_KEYS = (
    keys.WHITE_NOISE_RNG_KEY,
    keys.PSD_NOISE_RNG_KEY,
    keys.DARK_COUNT_RNG_KEY,
    keys.DIRECT_CROSSTALK_RETAINED_RNG_KEY,
    keys.DIRECT_CROSSTALK_OVERFLOW_RNG_KEY,
    keys.DELAYED_CROSSTALK_RETAINED_RNG_KEY,
    keys.DELAYED_CROSSTALK_OVERFLOW_RNG_KEY,
    keys.TIMING_JITTER_RNG_KEY,
    keys.AFTERPULSE_RNG_KEY,
    keys.CHARGE_SMEARING_RNG_KEY,
)


class _RecordingRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[list[torch.Tensor]] = []

    @override
    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        del key, quantum, block
        type(self).calls.append(positions.clone())
        return torch.zeros(
            positions.shape + (4,),
            dtype=torch.int64,
            device=positions.device,
        )


def _axes() -> tuple[ExampleAxis, ChannelAxis, SampleAxis]:
    return (
        ExampleAxis(count=1),
        ChannelAxis(labels=("channel-0",)),
        SampleAxis(start=0, step=2_000, count=4),
    )


class TensorCore016ModernizationTest(unittest.TestCase):
    def test_dependency_exports_and_precise_random_validation_are_exact(self) -> None:
        self.assertEqual(tensor_core.__all__, _ROOT_EXPORTS)
        self.assertEqual(
            scalar.__all__,
            (
                "Scalar",
                "FiniteFloat",
                "NonnegativeFloat",
                "NonnegativeInteger",
                "PositiveFloat",
                "PositiveInteger",
                "Probability",
            ),
        )
        self.assertEqual(
            scalar_validation.__all__,
            ("require_exact_integer", "require_finite_real", "require_integer"),
        )
        self.assertEqual(
            tensor.__all__,
            (
                "TensorAxis",
                "CountAxis",
                "RegularAxis",
                "LabelAxis",
                "TensorCollection",
                "TensorField",
                "TensorArtifact",
            ),
        )
        self.assertEqual(tensor_validation.__all__, _TENSOR_VALIDATION_EXPORTS)
        self.assertEqual(
            table.__all__,
            ("TableColumn", "TableCollection", "TableField"),
        )
        self.assertEqual(
            tensor_random.__all__,
            ("RngKey", "CounterRng", "Threefry4x32", "RngPositions"),
        )
        self.assertEqual(random_validation.__all__, ("require_count_tensor",))
        for name in (*_TENSOR_VALIDATION_EXPORTS, *scalar_validation.__all__):
            self.assertNotIn(name, tensor_core.__all__)
            self.assertFalse(hasattr(tensor_core, name))
        for name in (
            "require_nonnegative_integer",
            "require_positive_integer",
        ):
            self.assertFalse(hasattr(tensor_core, name))
            self.assertFalse(hasattr(scalar_validation, name))
        self.assertIs(tensor_validation.require_field_dtype, require_field_dtype)
        self.assertIs(tensor_validation.require_field_layout, require_field_layout)
        self.assertIs(
            tensor_validation.require_representable_float,
            require_representable_float,
        )
        self.assertIsNone(importlib.util.find_spec("tensor_core.validation"))

    def test_generic_requirement_signatures_and_boundaries_are_owned_upstream(
        self,
    ) -> None:
        axes = _axes()
        charge = Charge(
            tensor=torch.zeros((1, 1, 4), dtype=torch.float32),
            axes=axes,
        )
        require_field_dtype(charge, torch.float32, torch.float64)
        require_field_layout(charge, torch.strided)
        self.assertEqual(
            require_representable_float(
                16_777_217,
                dtype=torch.float32,
                field="value",
            ),
            16_777_216.0,
        )
        self.assertEqual(
            require_shape_span((2, 3, 4), "shape", upper=1 << 63),
            24,
        )
        self.assertEqual(
            require_tensor_allocation(
                (2, 3, 4),
                "allocation",
                element_size=8,
                upper=1 << 63,
            ),
            24,
        )
        counts = torch.tensor((0, (1 << 53) - 1), dtype=torch.int64)
        self.assertIs(require_count_tensor(counts, "counts"), counts)
        with self.assertRaises(ValueError):
            require_count_tensor(
                torch.tensor(((1 << 53),), dtype=torch.int64),
                "counts",
            )

    def test_strict_index_requirement_is_dependency_owned_only(self) -> None:
        self.assertEqual(require_index(0, "index", size=1), 0)
        for value in (True, -1, 1, 1.0, "0"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, IndexError)):
                    require_index(value, "index", size=1)

        axes = _axes()
        field = Photoelectrons(
            tensor=torch.zeros((1, 1, 4), dtype=torch.int64),
            axes=axes,
        )
        self.assertIs(field.axis_at(1), axes[1])
        self.assertEqual(axes[2].coordinate_at(1), 2_000)
        selected = RngPositions.from_shape((2, 3), device="cpu").select(1, 1)
        self.assertEqual(selected.shape, (2,))
        for operation in (
            lambda: field.axis_at(True),
            lambda: field.axis_at(-1),
            lambda: axes[2].coordinate_at(True),
            lambda: axes[2].coordinate_at(-1),
            lambda: RngPositions.from_shape((2, 3), device="cpu").select(1, True),
            lambda: RngPositions.from_shape((2, 3), device="cpu").select(1, -1),
        ):
            with self.assertRaises((TypeError, IndexError)):
                operation()
        for path in Path("tensor_dslab").rglob("*.py"):
            self.assertNotIn("require_index", path.read_text(), str(path))

    def test_python314_syntax_and_docstring_contracts_are_exact(self) -> None:
        production = tuple(sorted(Path("tensor_dslab").rglob("*.py")))
        self.assertEqual(len(production), 59)
        for path in production:
            tree = ast.parse(path.read_text(), filename=str(path))
            self.assertTrue(ast.get_docstring(tree, clean=False), str(path))
            self.assertNotIn(
                "from __future__ import annotations",
                path.read_text(),
                str(path),
            )

        active = (*production, *sorted(Path("tests").rglob("*.py")))
        for path in active:
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, "TypeVar", str(path))
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "typing"
                ):
                    self.assertNotIn(
                        "Generic",
                        (alias.name for alias in node.names),
                        str(path),
                    )

        units_tree = ast.parse(Path("tensor_dslab/common/units.py").read_text())
        aliases = tuple(
            node.name.id
            for node in units_tree.body
            if isinstance(node, ast.TypeAlias)
            and isinstance(node.name, ast.Name)
        )
        self.assertEqual(aliases, ("_QuantityField",))

        modules = (tensor_dslab, common, readout)
        objects = {
            id(getattr(module, name)): getattr(module, name)
            for module in modules
            for name in module.__all__
        }
        public_classes = tuple(
            value for value in objects.values() if inspect.isclass(value)
        )
        public_functions = tuple(
            value for value in objects.values() if inspect.isfunction(value)
        )
        self.assertEqual(len(public_classes), 32)
        self.assertEqual(len(public_functions), 3)
        for public_class in public_classes:
            self.assertTrue(
                public_class.__dict__.get("__doc__"),
                public_class.__name__,
            )
            get_type_hints(public_class)
        for public_function in public_functions:
            self.assertTrue(public_function.__doc__, public_function.__name__)
            get_type_hints(public_function)

    def test_requirements_relocation_and_direct_readout_module_shape(self) -> None:
        self.assertIsNone(
            importlib.util.find_spec("tensor_dslab.readout.requirements")
        )
        self.assertIsNotNone(
            importlib.util.find_spec(
                "tensor_dslab.readout.runtime.requirements"
            )
        )
        direct_modules = tuple(
            sorted(path.name for path in Path("tensor_dslab/readout").glob("*.py"))
        )
        self.assertEqual(
            direct_modules,
            ("__init__.py", "collection.py", "config.py", "simulation.py"),
        )
        source = Path(
            "tensor_dslab/readout/runtime/requirements.py"
        ).read_text()
        tree = ast.parse(source)
        functions = tuple(
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        )
        self.assertEqual(functions, ("require_readout_structure",))
        self.assertIn("require_field_layout", source)

    def test_rng_positions_factories_and_transforms_preserve_raw_addresses(
        self,
    ) -> None:
        base = RngPositions.from_shape((2, 3, 4), device="cpu")
        transformed = base.movedim(1, -1).select(0, 1).slice(0, 1, 4).offset(97)
        self.assertEqual(transformed.shape, (3, 3))
        _RecordingRng.calls = []
        _RecordingRng(seed=0).uniform(
            key=keys.WHITE_NOISE_RNG_KEY,
            positions=transformed,
            dtype=torch.float64,
        )
        expected = (
            torch.arange(24, dtype=torch.int64)
            .reshape(2, 3, 4)
            .movedim(1, -1)
            .select(0, 1)
            .narrow(0, 1, 3)
            + 97
        )
        self.assertTrue(torch.equal(_RecordingRng.calls[0], expected))

        caller = torch.tensor((3, 5, 8), dtype=torch.int64)
        snapshot = RngPositions.from_tensor(caller)
        caller.fill_(0)
        _RecordingRng.calls = []
        _RecordingRng(seed=0).uniform(
            key=keys.WHITE_NOISE_RNG_KEY,
            positions=snapshot,
            dtype=torch.float64,
        )
        self.assertTrue(
            torch.equal(
                _RecordingRng.calls[0],
                torch.tensor((3, 5, 8), dtype=torch.int64),
            )
        )

    def test_fixed_key_source_is_unique_and_has_each_literal_once(self) -> None:
        self.assertEqual(
            tuple((key.namespace, key.stream) for key in _FIXED_KEYS),
            tuple((0x54445331, stream) for stream in range(1, 11)),
        )
        self.assertEqual(len(set(_FIXED_KEYS)), 10)
        source = Path("tensor_dslab/readout/runtime/keys.py").read_text()
        self.assertEqual(source.count("0x54445331"), 1)
        for stream in range(1, 10):
            self.assertEqual(source.count(f"0x0000_000{stream}"), 1)
        self.assertEqual(source.count("0x0000_000A"), 1)
        for module in (tensor_dslab, readout):
            self.assertFalse(hasattr(module, "RNG_NAMESPACE"))

    def test_vector_quantity_copy_rank_dtype_and_indexed_domains(self) -> None:
        vector = quantities((1, 2.5), "Hz")
        self.assertIs(type(vector.magnitude), np.ndarray)
        self.assertEqual(vector.magnitude.dtype, np.dtype(np.float64))
        self.assertEqual(vector.magnitude.ndim, 1)
        self.assertFalse(vector.magnitude.flags.writeable)
        self.assertEqual(vector.magnitude.tolist(), [1.0, 2.5])

        external = pint.UnitRegistry(cache_folder=None)
        source = np.array((0.0, 1.0), dtype=np.float64)
        configured = PsdNoiseConfig(
            frequency_left_edges=cast(
                Quantity,
                external.Quantity(source, "Hz"),
            ),
            frequency_stop=quantity(2.0, "Hz"),
            power_density=quantities((1.0, 2.0), "mV ** 2 / Hz"),
        )
        source[0] = 99.0
        self.assertEqual(configured.frequency_left_edges.magnitude.tolist(), [0.0, 1.0])
        self.assertFalse(
            configured.frequency_left_edges.magnitude.flags.writeable
        )

        invalid_magnitudes = (
            np.array(1.0),
            np.zeros((1, 1)),
            np.array((True, False)),
            np.array((1.0 + 0.0j, 2.0 + 0.0j)),
            np.array((object(), object()), dtype=object),
        )
        for magnitude in invalid_magnitudes:
            with self.subTest(dtype=magnitude.dtype, rank=magnitude.ndim):
                with self.assertRaises((TypeError, ValueError)):
                    PsdNoiseConfig(
                        frequency_left_edges=cast(
                            Quantity,
                            external.Quantity(magnitude, "Hz"),
                        ),
                        frequency_stop=quantity(2.0, "Hz"),
                        power_density=quantities(
                            (1.0, 2.0),
                            "mV ** 2 / Hz",
                        ),
                    )
        with self.assertRaisesRegex(ValueError, r"power_density\[1\]"):
            PsdNoiseConfig(
                frequency_left_edges=quantities((0.0, 1.0), "Hz"),
                frequency_stop=quantity(2.0, "Hz"),
                power_density=cast(
                    Quantity,
                    external.Quantity(
                        np.array((1.0, float("nan"))),
                        "mV ** 2 / Hz",
                    ),
                ),
            )

    def test_psd_strips_each_vector_once_and_runtime_is_numpy_free(self) -> None:
        model = PsdNoiseConfig(
            frequency_left_edges=quantities((0.0, 100_000_000.0), "Hz"),
            frequency_stop=quantity(250_000_000.0, "Hz"),
            power_density=quantities(
                (1.0e-8, 2.0e-8),
                "mV ** 2 / Hz",
            ),
        )
        with patch(
            "tensor_dslab.readout.noise_waveform.runtime.prepare."
            "canonical_magnitudes",
            wraps=lambda value: tuple(float(item) for item in value.magnitude),
        ) as stripped:
            runtime = prepare_noise_waveform(
                tensor_dslab.NoiseWaveformConfig(model=model),
                sampling=SamplingRuntime(
                    sample_count=4,
                    sample_period_ps=2_000,
                    sample_dimension=2,
                ),
                shape=(1, 1, 4),
                floating_dtype=torch.float32,
                device=torch.device("cpu"),
            )
        self.assertEqual(stripped.call_count, 2)
        self.assertIs(type(runtime.model), PsdNoiseRuntime)
        for path in (
            Path("tensor_dslab/readout/noise_waveform/runtime/prepare.py"),
            Path("tensor_dslab/readout/noise_waveform/runtime/produce.py"),
            Path("tensor_dslab/readout/noise_waveform/runtime/validate.py"),
        ):
            imported = {
                node.module
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.ImportFrom)
            }
            names = {
                alias.name
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertNotIn("numpy", imported)
            self.assertNotIn("numpy", names)

    def test_positive_pulse_magnitude_gets_one_fixed_negative_polarity(self) -> None:
        model = TpcFebSnrPulseConfig(
            fast_time_constant=quantity(1.0, "ns"),
            slow_time_constant=quantity(2.0, "ns"),
            support_time=quantity(6.0, "ns"),
            peak_voltage_per_photoelectron=quantity(2.0, "mV"),
        )
        runtime = prepare_pure_waveform(
            PureWaveformConfig(model=model),
            sampling=SamplingRuntime(
                sample_count=4,
                sample_period_ps=2_000,
                sample_dimension=2,
            ),
            floating_dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.assertEqual(float(torch.min(runtime.kernel)), -2.0)
        self.assertEqual(float(torch.max(runtime.kernel)), 0.0)
        source = Path(
            "tensor_dslab/readout/pure_waveform/runtime/prepare.py"
        ).read_text()
        self.assertEqual(
            source.count(
                "signed_peak_voltage_mv_per_pe = -canonical_magnitude("
            ),
            2,
        )
        for config_type in (TpcFebSnrPulseConfig,):
            self.assertEqual(
                tuple(field.name for field in fields(config_type)),
                (
                    "fast_time_constant",
                    "slow_time_constant",
                    "support_time",
                    "peak_voltage_per_photoelectron",
                ),
            )


if __name__ == "__main__":
    unittest.main()
