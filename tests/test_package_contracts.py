from __future__ import annotations

import ast
from inspect import Parameter, isabstract, signature
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

import torch
import tensor_core
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
    simulate_readout,
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
                "tensor-core @ git+https://github.com/mbedard44/TensorCore.git@4708bf2ca063a1bcd37a30a342733b9e3dbe9f59",
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
            "simulate_readout",
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
                "simulate_readout",
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
        field_names = {
            "AnalogWaveform",
            "Charge",
            "DigitizedWaveform",
            "NoiseWaveform",
            "Photoelectrons",
            "PureWaveform",
        }
        for module_name, exports in expected.items():
            with self.subTest(module=module_name):
                module = __import__(module_name, fromlist=("__all__",))
                self.assertEqual(module.__all__, exports)
                for name in exports:
                    value = getattr(module, name)
                    self.assertIs(value, getattr(tensor_dslab, name))
                    owner = "field" if name in field_names else "config"
                    self.assertEqual(value.__module__, f"{module_name}.{owner}")

        self.assertEqual(ChannelAxis.__module__, "tensor_dslab.common.axes")
        self.assertEqual(ExampleAxis.__module__, "tensor_dslab.common.axes")
        self.assertEqual(SampleAxis.__module__, "tensor_dslab.common.axes")
        self.assertEqual(
            tensor_dslab.SamplingConfig.__module__,
            "tensor_dslab.common.sampling",
        )
        self.assertEqual(
            ReadoutCollection.__module__,
            "tensor_dslab.readout.collection",
        )
        self.assertEqual(
            tensor_dslab.ReadoutConfig.__module__,
            "tensor_dslab.readout.config",
        )
        self.assertIs(simulate_readout, readout.simulate_readout)
        self.assertEqual(
            simulate_readout.__module__,
            "tensor_dslab.readout.simulation",
        )

    def test_public_simulation_signature_is_exact(self) -> None:
        parameters = signature(simulate_readout).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "photoelectrons",
                "products",
                "config",
                "rng",
                "floating_dtype",
            ),
        )
        self.assertIs(
            parameters["photoelectrons"].kind,
            Parameter.POSITIONAL_OR_KEYWORD,
        )
        for name in ("products", "config", "rng", "floating_dtype"):
            self.assertIs(parameters[name].kind, Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["products"].default, Parameter.empty)
        self.assertIs(parameters["config"].default, Parameter.empty)
        self.assertIs(parameters["rng"].default, Parameter.empty)
        self.assertIs(parameters["floating_dtype"].default, torch.float32)
        self.assertNotIn("seed", parameters)

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
            "CounterRng",
            "RngKey",
            "Threefry4x32",
            "logical_positions",
            "require_same_dtype",
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
            "tensor_dslab/readout/types.py",
            "tensor_dslab/readout/_random.py",
            "tensor_dslab/readout/_rng.py",
            "tensor_dslab/readout/photoelectrons/types.py",
            "tensor_dslab/readout/charge/types.py",
            "tensor_dslab/readout/pure_waveform/types.py",
            "tensor_dslab/readout/noise_waveform/types.py",
            "tensor_dslab/readout/analog_waveform/types.py",
            "tensor_dslab/readout/digitized_waveform/types.py",
            "tensor_dslab/readout/photoelectrons/_product.py",
            "tensor_dslab/readout/charge/_product.py",
            "tensor_dslab/readout/pure_waveform/_product.py",
            "tensor_dslab/readout/noise_waveform/_product.py",
            "tensor_dslab/readout/analog_waveform/_product.py",
            "tensor_dslab/readout/digitized_waveform/_product.py",
            "tensor_dslab/readout/_requirements.py",
            "tensor_dslab/readout/photoelectrons/_produce.py",
            "tensor_dslab/readout/charge/_produce.py",
            "tensor_dslab/readout/pure_waveform/_produce.py",
            "tensor_dslab/readout/noise_waveform/_produce.py",
            "tensor_dslab/readout/analog_waveform/_produce.py",
            "tensor_dslab/readout/digitized_waveform/_produce.py",
            "tensor_dslab/readout/charge/effects",
        )
        for path in absent:
            self.assertFalse(Path(path).exists(), path)

    def test_runtime_actions_records_and_sampling_remain_private(
        self,
    ) -> None:
        private_names = (
            "_RngStream",
            "_require_representable_float",
            "_require_sampling",
            "_produce_charge",
            "_produce_pure_waveform",
            "_produce_noise_waveform",
            "_produce_analog_waveform",
            "_produce_digitized_waveform",
            "_ReadoutPlan",
            "_ChargePlan",
            "_PureWaveformPlan",
            "_NoiseWaveformPlan",
            "_AnalogWaveformPlan",
            "_DigitizedWaveformPlan",
            "_prepare_readout",
            "_prepare_charge",
            "_prepare_pure_waveform",
            "_prepare_noise_waveform",
            "_prepare_analog_waveform",
            "_prepare_digitized_waveform",
            "SamplingRuntime",
            "ReadoutRuntime",
            "ChargeRuntime",
            "PureWaveformRuntime",
            "NoiseWaveformRuntime",
            "AnalogWaveformRuntime",
            "DigitizedWaveformRuntime",
            "prepare_readout",
            "prepare_sampling",
            "prepare_charge",
            "prepare_pure_waveform",
            "prepare_noise_waveform",
            "prepare_analog_waveform",
            "prepare_digitized_waveform",
            "produce_charge",
            "produce_pure_waveform",
            "produce_noise_waveform",
            "produce_analog_waveform",
            "produce_digitized_waveform",
            "validate_photoelectrons",
            "validate_charge",
            "validate_pure_waveform",
            "validate_noise_waveform",
            "validate_analog_waveform",
            "validate_digitized_waveform",
        )
        public_modules = (
            tensor_dslab,
            readout,
            __import__(
                "tensor_dslab.readout.pure_waveform",
                fromlist=("__all__",),
            ),
            __import__(
                "tensor_dslab.readout.noise_waveform",
                fromlist=("__all__",),
            ),
            __import__(
                "tensor_dslab.readout.analog_waveform",
                fromlist=("__all__",),
            ),
            __import__(
                "tensor_dslab.readout.digitized_waveform",
                fromlist=("__all__",),
            ),
        )
        for module in public_modules:
            for name in private_names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertNotIn(name, module.__all__)
                    self.assertFalse(hasattr(module, name))

    def test_production_uses_only_public_tensorcore_imports(self) -> None:
        public_names = frozenset(tensor_core.__all__)
        for path in Path("tensor_dslab").rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    if node.module.startswith("tensor_core"):
                        self.assertEqual(node.module, "tensor_core", str(path))
                        for alias in node.names:
                            self.assertNotEqual(alias.name, "*", str(path))
                            self.assertIn(alias.name, public_names, str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name.startswith("tensor_core."),
                            str(path),
                        )

    def test_product_modules_do_not_import_composition_layer(self) -> None:
        paths = tuple(Path("tensor_dslab/readout").glob("*/config.py")) + tuple(
            Path("tensor_dslab/readout").glob("*/field.py")
        )
        self.assertEqual(len(paths), 11)
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            for composition_module in (
                "tensor_dslab.readout.config",
                "tensor_dslab.readout.collection",
                "tensor_dslab.readout.simulation",
                "tensor_dslab.readout.types",
            ):
                self.assertNotIn(composition_module, imported, str(path))

    def test_product_runtime_imports_are_private_and_acyclic(self) -> None:
        runtime_paths = tuple(
            sorted(Path("tensor_dslab/readout").glob("*/runtime/*.py"))
        )
        self.assertEqual(len(runtime_paths), 22)
        forbidden_prefixes = (
            "dag",
            "dask",
            "dselec",
            "dslab",
            "g4ds",
            "g4ds11",
            "io",
            "iv_dslab",
            "numpy",
            "prefect",
            "ray",
            "scipy",
            "tensor_g4ds",
            "tensor_ml",
        )
        forbidden_tensor_dslab_modules = {
            "tensor_dslab",
            "tensor_dslab.readout.simulation",
            "tensor_dslab.readout.config",
            "tensor_dslab.readout.collection",
            "tensor_dslab.readout.types",
        }
        for path in runtime_paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
            for imported in imports:
                with self.subTest(path=str(path), imported=imported):
                    self.assertNotIn(imported, forbidden_tensor_dslab_modules)
                    self.assertFalse(
                        any(
                            imported == prefix
                            or imported.startswith(f"{prefix}.")
                            for prefix in forbidden_prefixes
                        )
                    )
    def test_runtime_packages_are_empty_and_effects_do_not_import_producer(
        self,
    ) -> None:
        runtime_packages = (
            "tensor_dslab.readout.runtime",
            "tensor_dslab.readout.photoelectrons.runtime",
            "tensor_dslab.readout.charge.runtime",
            "tensor_dslab.readout.charge.runtime.effects",
            "tensor_dslab.readout.pure_waveform.runtime",
            "tensor_dslab.readout.noise_waveform.runtime",
            "tensor_dslab.readout.analog_waveform.runtime",
            "tensor_dslab.readout.digitized_waveform.runtime",
        )
        for module_name in runtime_packages:
            with self.subTest(module=module_name):
                module = __import__(module_name, fromlist=("__name__",))
                self.assertFalse(hasattr(module, "__all__"))
                package_path = Path(*module_name.split("."), "__init__.py")
                self.assertEqual(package_path.read_text(), "")
        effect_paths = tuple(
            sorted(
                Path("tensor_dslab/readout/charge/runtime/effects").glob("*.py")
            )
        )
        self.assertEqual(len(effect_paths), 7)
        for path in effect_paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            self.assertNotIn(
                "tensor_dslab.readout.charge.runtime.produce",
                imported,
            )
            self.assertNotIn("tensor_dslab.readout.config", imported)
            self.assertNotIn("tensor_dslab.readout.collection", imported)

    def test_fresh_process_imports_are_acyclic_and_isolated(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        modules = (
            "tensor_dslab.common",
            "tensor_dslab.readout.photoelectrons",
            "tensor_dslab.readout.photoelectrons.field",
            "tensor_dslab.readout.photoelectrons.runtime",
            "tensor_dslab.readout.photoelectrons.runtime.validate",
            "tensor_dslab.readout.charge",
            "tensor_dslab.readout.charge.config",
            "tensor_dslab.readout.charge.field",
            "tensor_dslab.readout.charge.runtime",
            "tensor_dslab.readout.charge.runtime.prepare",
            "tensor_dslab.readout.charge.runtime.produce",
            "tensor_dslab.readout.charge.runtime.validate",
            "tensor_dslab.readout.charge.runtime.effects",
            "tensor_dslab.readout.charge.runtime.effects.counts",
            "tensor_dslab.readout.charge.runtime.effects.delays",
            "tensor_dslab.readout.charge.runtime.effects.dark_counts",
            "tensor_dslab.readout.charge.runtime.effects.timing_jitter",
            "tensor_dslab.readout.charge.runtime.effects.correlated_avalanches",
            "tensor_dslab.readout.charge.runtime.effects.smearing",
            "tensor_dslab.readout.pure_waveform",
            "tensor_dslab.readout.pure_waveform.config",
            "tensor_dslab.readout.pure_waveform.field",
            "tensor_dslab.readout.pure_waveform.runtime",
            "tensor_dslab.readout.pure_waveform.runtime.prepare",
            "tensor_dslab.readout.pure_waveform.runtime.produce",
            "tensor_dslab.readout.pure_waveform.runtime.validate",
            "tensor_dslab.readout.noise_waveform",
            "tensor_dslab.readout.noise_waveform.config",
            "tensor_dslab.readout.noise_waveform.field",
            "tensor_dslab.readout.noise_waveform.runtime",
            "tensor_dslab.readout.noise_waveform.runtime.prepare",
            "tensor_dslab.readout.noise_waveform.runtime.produce",
            "tensor_dslab.readout.noise_waveform.runtime.validate",
            "tensor_dslab.readout.analog_waveform",
            "tensor_dslab.readout.analog_waveform.config",
            "tensor_dslab.readout.analog_waveform.field",
            "tensor_dslab.readout.analog_waveform.runtime",
            "tensor_dslab.readout.analog_waveform.runtime.prepare",
            "tensor_dslab.readout.analog_waveform.runtime.produce",
            "tensor_dslab.readout.analog_waveform.runtime.validate",
            "tensor_dslab.readout.digitized_waveform",
            "tensor_dslab.readout.digitized_waveform.config",
            "tensor_dslab.readout.digitized_waveform.field",
            "tensor_dslab.readout.digitized_waveform.runtime",
            "tensor_dslab.readout.digitized_waveform.runtime.prepare",
            "tensor_dslab.readout.digitized_waveform.runtime.produce",
            "tensor_dslab.readout.digitized_waveform.runtime.validate",
            "tensor_dslab.readout.config",
            "tensor_dslab.readout.collection",
            "tensor_dslab.readout.requirements",
            "tensor_dslab.readout.runtime",
            "tensor_dslab.readout.runtime.sampling",
            "tensor_dslab.readout.runtime.prepare",
            "tensor_dslab.readout.simulation",
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

        retired_modules = (
            "tensor_dslab.readout._random",
            "tensor_dslab.readout._rng",
            "tensor_dslab.readout.types",
            "tensor_dslab.readout.photoelectrons.types",
            "tensor_dslab.readout.charge.types",
            "tensor_dslab.readout.pure_waveform.types",
            "tensor_dslab.readout.noise_waveform.types",
            "tensor_dslab.readout.analog_waveform.types",
            "tensor_dslab.readout.digitized_waveform.types",
            "tensor_dslab.readout._requirements",
            "tensor_dslab.readout.charge.effects",
            "tensor_dslab.readout.photoelectrons._produce",
            "tensor_dslab.readout.charge._produce",
            "tensor_dslab.readout.pure_waveform._produce",
            "tensor_dslab.readout.noise_waveform._produce",
            "tensor_dslab.readout.analog_waveform._produce",
            "tensor_dslab.readout.digitized_waveform._produce",
        )
        for module_name in retired_modules:
            with self.subTest(retired=module_name):
                code = (
                    "import importlib; "
                    f"name={module_name!r}; "
                    "\ntry: importlib.import_module(name)"
                    "\nexcept ModuleNotFoundError as error: "
                    "assert error.name == name, (name, error.name)"
                    "\nelse: raise AssertionError(name)"
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
