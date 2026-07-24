from __future__ import annotations

import ast
from dataclasses import fields
import importlib.util
import math
from pathlib import Path
from typing import cast, ClassVar
import unittest
from unittest.mock import patch

import numpy as np
import pint
from pint import Quantity
import torch
import tensor_core
from tensor_core import (
    CounterRng,
    RngKey,
    RngPositions,
    TensorField,
    require_field_dtype,
    require_field_layout,
    require_representable_float,
)
from tensor_core.validation import (
    require_shape_span,
    require_tensor_allocation,
)
from tensor_core.validation.random import require_count_tensor

import tensor_dslab
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
    "require_axis_signature",
    "require_exact_integer",
    "require_field_dtype",
    "require_field_layout",
    "require_finite_real",
    "require_field_types",
    "require_integer",
    "require_nonnegative_integer",
    "require_positive_integer",
    "require_representable_float",
    "require_same_axes",
    "require_same_device",
    "require_same_dtype",
)
_VALIDATION_EXPORTS = (
    "require_axis_signature",
    "require_device",
    "require_dimension",
    "require_exact_integer",
    "require_field_dtype",
    "require_field_layout",
    "require_finite_real",
    "require_field_types",
    "require_integer",
    "require_nonnegative_integer",
    "require_positive_integer",
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


class TensorCore015AdoptionTest(unittest.TestCase):
    def test_dependency_exports_and_precise_random_validation_are_exact(self) -> None:
        import tensor_core.validation as validation
        import tensor_core.validation.random as random_validation

        self.assertEqual(tensor_core.__all__, _ROOT_EXPORTS)
        self.assertEqual(validation.__all__, _VALIDATION_EXPORTS)
        self.assertEqual(random_validation.__all__, ("require_count_tensor",))
        self.assertIs(tensor_core.require_field_dtype, require_field_dtype)
        self.assertIs(tensor_core.require_field_layout, require_field_layout)
        self.assertIs(
            tensor_core.require_representable_float,
            require_representable_float,
        )

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
