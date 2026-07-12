from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

from tensor_core import Id, IdSequence, TensorAxisId, TensorFieldId

import tensor_dslab
import tensor_dslab.common as common
import tensor_dslab.readout as readout
import tensor_dslab.readout.ids as readout_ids
import tensor_dslab.readout.types as readout_types
from tensor_dslab.common import ChannelId, ExampleId


class PackageContractTest(unittest.TestCase):
    def test_package_metadata_matches_stage2_contract(self) -> None:
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
                "tensor-core @ git+https://github.com/mbedard44/TensorCore.git@dc554994061183776f23f65860a0594516074f2e",
            ],
        )
        self.assertEqual(
            metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"],
            ["tensor_dslab"],
        )
        self.assertTrue(Path("LICENSE").is_file())
        self.assertTrue(Path("tensor_dslab/py.typed").is_file())

    def test_public_readout_imports_resolve(self) -> None:
        expected = {
            "AdcQuantization",
            "DigitizedWaveformSpec",
            "READOUT_ANALOG_WAVEFORM_FIELD_ID",
            "CHANNEL_AXIS_ID",
            "READOUT_CHARGE_FIELD_ID",
            "READOUT_DIGITIZED_WAVEFORM_FIELD_ID",
            "EXAMPLE_AXIS_ID",
            "READOUT_FIELD_IDS",
            "READOUT_NOISE_WAVEFORM_FIELD_ID",
            "READOUT_PHOTOELECTRONS_FIELD_ID",
            "READOUT_PURE_WAVEFORM_FIELD_ID",
            "REQUIRED_AXIS_IDS",
            "SAMPLE_AXIS_ID",
            "ReadoutCollection",
            "SampleGrid",
            "build_readout_output_buffer",
            "build_readout_result_buffer",
            "move_readout_collection",
            "project_readout_fields",
            "require_valid_readout_collection",
            "select_readout_indices",
        }
        self.assertEqual(set(readout.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(readout, name), name)
        self.assertEqual(common.__all__, ("ChannelId", "ExampleId"))
        self.assertEqual(tensor_dslab.__doc__, "TensorDSLab package.")

    def test_tensorcore_symbols_are_not_reexported(self) -> None:
        for module in (tensor_dslab, common, readout):
            for name in ("Id", "TensorCollection", "TensorField", "TensorAxis"):
                self.assertFalse(hasattr(module, name), f"{module.__name__}.{name}")

    def test_readout_collection_module_ownership(self) -> None:
        self.assertIs(readout.ReadoutCollection, readout_types.ReadoutCollection)
        self.assertEqual(
            readout.ReadoutCollection.__module__,
            "tensor_dslab.readout.types",
        )

    def test_axis_exports_and_retired_names(self) -> None:
        expected_axes = (
            ("EXAMPLE_AXIS_ID", "example"),
            ("CHANNEL_AXIS_ID", "channel"),
            ("SAMPLE_AXIS_ID", "sample"),
        )
        for name, value in expected_axes:
            axis_id = getattr(readout, name)
            self.assertIs(type(axis_id), TensorAxisId)
            self.assertEqual(axis_id, TensorAxisId(value))
            self.assertIs(axis_id, getattr(readout_ids, name))

        self.assertIs(type(readout.REQUIRED_AXIS_IDS), IdSequence)
        self.assertIs(readout.REQUIRED_AXIS_IDS, readout_ids.REQUIRED_AXIS_IDS)
        self.assertEqual(
            readout.REQUIRED_AXIS_IDS.ids,
            (
                readout.EXAMPLE_AXIS_ID,
                readout.CHANNEL_AXIS_ID,
                readout.SAMPLE_AXIS_ID,
            ),
        )

        retired_names = (
            "READOUT_EXAMPLE_AXIS_ID",
            "READOUT_CHANNEL_AXIS_ID",
            "READOUT_SAMPLE_AXIS_ID",
            "READOUT_REQUIRED_AXIS_IDS",
        )
        for name in retired_names:
            self.assertNotIn(name, readout.__all__)
            self.assertFalse(hasattr(readout, name), name)
            self.assertFalse(hasattr(readout_ids, name), name)

    def test_readout_modules_import_in_fresh_processes(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for module_name in (
            "tensor_dslab.readout.types",
            "tensor_dslab.readout.validation",
            "tensor_dslab.readout.tensors",
            "tensor_dslab.readout.builders",
            "tensor_dslab.readout",
        ):
            with self.subTest(module_name=module_name):
                completed = subprocess.run(
                    [sys.executable, "-c", f"import {module_name}"],
                    check=False,
                    capture_output=True,
                    env=environment,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_only_public_tensorcore_root_is_imported(self) -> None:
        for path in Path("tensor_dslab").rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    if node.module.startswith("tensor_core"):
                        self.assertEqual(node.module, "tensor_core", str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("tensor_core."), str(path))

    def test_stage2_imports_no_tensor_g4ds_g4ds_or_tensor_ml_package(self) -> None:
        for package_name in ("tensor_g4ds", "tensor_ml", "dslab", "g4ds11"):
            self.assertNotIn(package_name, sys.modules)

    def test_coordinate_ids_extend_id_but_axis_and_field_ids_do_not(self) -> None:
        self.assertTrue(issubclass(ExampleId, Id))
        self.assertTrue(issubclass(ChannelId, Id))
        self.assertIs(type(readout.SAMPLE_AXIS_ID), TensorAxisId)
        self.assertIs(type(readout.READOUT_CHARGE_FIELD_ID), TensorFieldId)
        self.assertFalse(issubclass(ExampleId, TensorAxisId))


if __name__ == "__main__":
    unittest.main()
