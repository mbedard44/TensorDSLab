from __future__ import annotations

import ast
from inspect import isabstract, signature
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

import torch
from tensor_core import TensorAxis, TensorCollection, TensorField

import tensor_dslab
import tensor_dslab.common as common
import tensor_dslab.readout as readout
from tensor_dslab import (
    AnalogWaveform,
    ChannelAxis,
    Charge,
    DigitizedWaveform,
    ExampleAxis,
    NoiseWaveform,
    Photoelectrons,
    PureWaveform,
    ReadoutCollection,
    SampleAxis,
)


class PackageContractTest(unittest.TestCase):
    def test_package_metadata_selects_exact_tensorcore_candidate(self) -> None:
        metadata = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(metadata["build-system"]["build-backend"], "hatchling.build")
        project = metadata["project"]
        self.assertEqual(project["name"], "tensor-dslab")
        self.assertEqual(project["version"], "0.1.0")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(
            project["dependencies"],
            [
                "torch",
                "tensor-core @ git+https://github.com/mbedard44/TensorCore.git@b454d738f6385ce6489d85492a618a3dab139bb6",
            ],
        )
        self.assertEqual(
            metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"],
            ["tensor_dslab"],
        )
        self.assertTrue(Path("LICENSE").is_file())
        self.assertTrue(Path("tensor_dslab/py.typed").is_file())

    def test_exact_common_readout_and_package_exports(self) -> None:
        self.assertEqual(
            common.__all__,
            ("ChannelAxis", "ExampleAxis", "SampleAxis", "SamplingConfig"),
        )
        expected_readout = (
            "AfterpulseConfig",
            "AfterpulseRecoveryConfig",
            "AnalogSaturationConfig",
            "AnalogWaveform",
            "AnalogWaveformConfig",
            "Charge",
            "ChargeConfig",
            "ChargeSmearingConfig",
            "CorrelatedAvalancheConfig",
            "DarkCountConfig",
            "DelayedCrosstalkConfig",
            "DigitizedWaveform",
            "DigitizedWaveformConfig",
            "DirectCrosstalkConfig",
            "ExponentialDelayConfig",
            "FixedDelayConfig",
            "NoiseWaveform",
            "NoiseWaveformConfig",
            "NormalDelayConfig",
            "Photoelectrons",
            "PsdNoiseConfig",
            "PureWaveform",
            "PureWaveformConfig",
            "ReadoutCollection",
            "ReadoutConfig",
            "TimingJitterConfig",
            "TpcFebSnrPulseConfig",
            "VetoPduPulseConfig",
            "WhiteNoiseConfig",
            "ZeroNoiseConfig",
        )
        self.assertEqual(readout.__all__, expected_readout)
        self.assertEqual(
            tensor_dslab.__all__,
            (
                "AfterpulseConfig",
                "AfterpulseRecoveryConfig",
                "AnalogSaturationConfig",
                "AnalogWaveform",
                "AnalogWaveformConfig",
                "ChannelAxis",
                "Charge",
                "ChargeConfig",
                "ChargeSmearingConfig",
                "CorrelatedAvalancheConfig",
                "DarkCountConfig",
                "DelayedCrosstalkConfig",
                "DigitizedWaveform",
                "DigitizedWaveformConfig",
                "DirectCrosstalkConfig",
                "ExampleAxis",
                "ExponentialDelayConfig",
                "FixedDelayConfig",
                "NoiseWaveform",
                "NoiseWaveformConfig",
                "NormalDelayConfig",
                "Photoelectrons",
                "PsdNoiseConfig",
                "PureWaveform",
                "PureWaveformConfig",
                "ReadoutCollection",
                "ReadoutConfig",
                "SampleAxis",
                "SamplingConfig",
                "TimingJitterConfig",
                "TpcFebSnrPulseConfig",
                "VetoPduPulseConfig",
                "WhiteNoiseConfig",
                "ZeroNoiseConfig",
            ),
        )
        for name in tensor_dslab.__all__:
            self.assertTrue(hasattr(tensor_dslab, name), name)

    def test_product_package_exports_and_module_ownership(self) -> None:
        expected = {
            "tensor_dslab.readout.photoelectrons": ("Photoelectrons",),
            "tensor_dslab.readout.charge": (
                "AfterpulseConfig",
                "AfterpulseRecoveryConfig",
                "Charge",
                "ChargeConfig",
                "ChargeSmearingConfig",
                "CorrelatedAvalancheConfig",
                "DarkCountConfig",
                "DelayedCrosstalkConfig",
                "DirectCrosstalkConfig",
                "ExponentialDelayConfig",
                "FixedDelayConfig",
                "NormalDelayConfig",
                "TimingJitterConfig",
            ),
            "tensor_dslab.readout.pure_waveform": (
                "PureWaveform",
                "PureWaveformConfig",
                "TpcFebSnrPulseConfig",
                "VetoPduPulseConfig",
            ),
            "tensor_dslab.readout.noise_waveform": (
                "NoiseWaveform",
                "NoiseWaveformConfig",
                "PsdNoiseConfig",
                "WhiteNoiseConfig",
                "ZeroNoiseConfig",
            ),
            "tensor_dslab.readout.analog_waveform": (
                "AnalogSaturationConfig",
                "AnalogWaveform",
                "AnalogWaveformConfig",
            ),
            "tensor_dslab.readout.digitized_waveform": (
                "DigitizedWaveform",
                "DigitizedWaveformConfig",
            ),
        }
        for module_name, exports in expected.items():
            with self.subTest(module=module_name):
                module = __import__(module_name, fromlist=("__all__",))
                self.assertEqual(module.__all__, exports)
                for name in exports:
                    value = getattr(module, name)
                    self.assertIs(value, getattr(tensor_dslab, name))
                    self.assertEqual(value.__module__, f"{module_name}.types")

        self.assertEqual(ChannelAxis.__module__, "tensor_dslab.common.axes")
        self.assertEqual(ExampleAxis.__module__, "tensor_dslab.common.axes")
        self.assertEqual(SampleAxis.__module__, "tensor_dslab.common.axes")
        self.assertEqual(
            tensor_dslab.SamplingConfig.__module__,
            "tensor_dslab.common.sampling",
        )
        self.assertEqual(
            ReadoutCollection.__module__,
            "tensor_dslab.readout.types",
        )
        self.assertEqual(
            tensor_dslab.ReadoutConfig.__module__,
            "tensor_dslab.readout.types",
        )

    def test_semantic_leaves_are_direct_final_fieldless_roots(self) -> None:
        leaf_groups = (
            ((ExampleAxis, ChannelAxis, SampleAxis), TensorAxis),
            (
                (
                    Photoelectrons,
                    Charge,
                    PureWaveform,
                    NoiseWaveform,
                    AnalogWaveform,
                    DigitizedWaveform,
                ),
                TensorField,
            ),
            ((ReadoutCollection,), TensorCollection),
        )
        for leaves, root in leaf_groups:
            for leaf in leaves:
                with self.subTest(leaf=leaf.__name__):
                    self.assertEqual(leaf.__bases__, (root,))
                    self.assertEqual(leaf.__slots__, ())
                    self.assertTrue(getattr(leaf, "__final__", False))
                    self.assertNotIn("__annotations__", leaf.__dict__)
                    self.assertNotIn("__dataclass_fields__", leaf.__dict__)
                    self.assertNotIn("__dataclass_params__", leaf.__dict__)
                    self.assertIn("_require", leaf.__dict__)
                    self.assertIs(leaf.__init__, root.__init__)
                    self.assertIs(leaf._validate, root._validate)
                    self.assertEqual(signature(leaf), signature(root))

    def test_tensorcore_roots_are_abstract_and_lookup_is_typed(self) -> None:
        self.assertTrue(isabstract(TensorAxis))
        self.assertTrue(isabstract(TensorField))
        self.assertTrue(isabstract(TensorCollection))

        sample = SampleAxis(coordinates=("0ps", "2000ps"))
        photoelectrons = Photoelectrons(
            tensor=torch.zeros((1, 1, 2), dtype=torch.int64),
            axes=(
                ExampleAxis(coordinates=("e0",)),
                ChannelAxis(coordinates=("c0",)),
                sample,
            ),
        )
        collection = ReadoutCollection(fields=(photoelectrons,))
        self.assertIs(photoelectrons.axis(SampleAxis), sample)
        self.assertEqual(photoelectrons.dimension_of(SampleAxis), 2)
        self.assertIs(collection.field(Photoelectrons), photoelectrons)
        self.assertIs(collection.tensor(Photoelectrons), photoelectrons.tensor)

    def test_tensorcore_names_are_not_reexported(self) -> None:
        retired_or_generic = (
            "Id",
            "IdSequence",
            "TensorAxis",
            "TensorAxisId",
            "TensorCollection",
            "TensorField",
            "TensorFieldId",
            "TensorLayout",
            "PositiveInteger",
        )
        for module in (tensor_dslab, common, readout):
            for name in retired_or_generic:
                self.assertNotIn(name, module.__all__)
                self.assertFalse(hasattr(module, name), f"{module.__name__}.{name}")

    def test_retired_files_and_future_placeholders_are_absent(self) -> None:
        absent = (
            "tensor_dslab/common/ids.py",
            "tensor_dslab/readout/builders.py",
            "tensor_dslab/readout/ids.py",
            "tensor_dslab/readout/tensors.py",
            "tensor_dslab/readout/validation.py",
            "tensor_dslab/readout/simulation.py",
            "tensor_dslab/readout/_random.py",
            "tensor_dslab/readout/photoelectrons/_product.py",
            "tensor_dslab/readout/charge/_product.py",
            "tensor_dslab/readout/pure_waveform/_product.py",
            "tensor_dslab/readout/noise_waveform/_product.py",
            "tensor_dslab/readout/analog_waveform/_product.py",
            "tensor_dslab/readout/digitized_waveform/_product.py",
        )
        for path in absent:
            self.assertFalse(Path(path).exists(), path)

    def test_production_uses_only_public_tensorcore_imports(self) -> None:
        for path in Path("tensor_dslab").rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    if node.module.startswith("tensor_core"):
                        self.assertEqual(node.module, "tensor_core", str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name.startswith("tensor_core."),
                            str(path),
                        )

    def test_product_types_do_not_import_composition_layer(self) -> None:
        for path in Path("tensor_dslab/readout").glob("*/types.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            self.assertNotIn("tensor_dslab.readout.types", imported, str(path))

    def test_fresh_process_imports_are_acyclic_and_isolated(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        modules = (
            "tensor_dslab.common",
            "tensor_dslab.readout.photoelectrons",
            "tensor_dslab.readout.charge",
            "tensor_dslab.readout.pure_waveform",
            "tensor_dslab.readout.noise_waveform",
            "tensor_dslab.readout.analog_waveform",
            "tensor_dslab.readout.digitized_waveform",
            "tensor_dslab.readout.types",
            "tensor_dslab.readout",
            "tensor_dslab",
        )
        deferred = ("tensor_g4ds", "tensor_ml", "dslab", "g4ds11")
        for module_name in modules:
            with self.subTest(module=module_name):
                code = (
                    f"import {module_name}, sys; "
                    f"assert not any(name in sys.modules for name in {deferred!r})"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", code],
                    check=False,
                    capture_output=True,
                    env=environment,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
