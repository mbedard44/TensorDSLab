"""Focused source and execution proof for the public readout quickstart."""

import ast
import copy
import json
from pathlib import Path
import unittest

from nbclient import NotebookClient
import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "demos" / "readout.ipynb"
SUMMARY_PREFIX = "READOUT_DEMO_SUMMARY="


PROBE_SOURCE = f"""
import hashlib
import json


def tensor_digest(tensor):
    values = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


product_values = (
    photoelectrons,
    charge,
    pure_waveform,
    noise_waveform,
    analog_waveform,
    digitized_waveform,
)
summary = {{
    "types": [type(product).__name__ for product in product_values],
    "shapes": [list(product.tensor.shape) for product in product_values],
    "devices": [product.tensor.device.type for product in product_values],
    "dtypes": [str(product.tensor.dtype) for product in product_values],
    "units": [str(product.spec.unit) for product in product_values],
    "digests": [tensor_digest(product.tensor) for product in product_values],
    "source_unchanged": bool(torch.equal(
        photoelectron_values,
        photoelectron_values_before,
    )),
    "photo_charge_equal": bool(torch.equal(
        photoelectrons.tensor.to(charge.tensor.dtype),
        charge.tensor,
    )),
    "channel_labels": list(channel_axis.coordinates.labels),
    "time_size": time_axis.size,
    "time_scale": time_axis.coordinate_scale,
    "frequency_size": frequency_axis.size,
    "frequency_scale": frequency_axis.coordinate_scale,
    "psd_shape": list(power_spectral_density.tensor.shape),
    "psd_conditioning": [
        type(axis).__name__
        for axis in power_spectral_density.conditioning_axes
    ],
    "psd_operation": [
        type(axis).__name__
        for axis in power_spectral_density.operation_axes
    ],
    "psd_dc_zero": bool(
        (power_spectral_density.tensor[:, 0] == 0).all()
    ),
    "psd_rows_distinct": bool(
        not torch.equal(psd_sensor_0, psd_sensor_1)
        and not torch.equal(psd_sensor_0, psd_sensor_2)
        and not torch.equal(psd_sensor_1, psd_sensor_2)
    ),
    "sensor_products_distinct": [
        bool(
            not torch.equal(product.tensor[0, 0], product.tensor[0, 1])
            and not torch.equal(product.tensor[0, 0], product.tensor[0, 2])
        )
        for product in product_values
    ],
    "figure_axes": len(figure.axes),
    "line_counts": [len(axis.lines) for axis in plot_axes],
    "colors": [
        [line.get_color() for line in axis.lines]
        for axis in plot_axes
    ],
    "drawstyles": [
        [line.get_drawstyle() for line in axis.lines]
        for axis in plot_axes
    ],
    "axis_legends": [
        axis.get_legend() is not None
        for axis in plot_axes
    ],
    "figure_legends": len(figure.legends),
    "xlabels": [axis.get_xlabel() for axis in plot_axes],
    "ylabels": [axis.get_ylabel() for axis in plot_axes],
    "adc_min": int(digitized_waveform.tensor.min()),
    "adc_max": int(digitized_waveform.tensor.max()),
}}
print({SUMMARY_PREFIX!r} + json.dumps(summary, sort_keys=True))
"""


class ReadoutDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.code_cells = tuple(
            cell for cell in cls.notebook.cells if cell.cell_type == "code"
        )
        cls.markdown_cells = tuple(
            cell for cell in cls.notebook.cells if cell.cell_type == "markdown"
        )
        cls.code_source = "\n".join(cell.source for cell in cls.code_cells)

    @classmethod
    def _execute(cls) -> tuple[dict, object]:
        notebook = copy.deepcopy(cls.notebook)
        notebook.cells.append(nbformat.v4.new_code_cell(PROBE_SOURCE))
        executed = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
        ).execute(cwd=str(ROOT))
        probe = executed.cells[-1]
        streams = tuple(
            output.text
            for output in probe.outputs
            if output.output_type == "stream"
        )
        summary_lines = tuple(
            line
            for stream in streams
            for line in stream.splitlines()
            if line.startswith(SUMMARY_PREFIX)
        )
        if len(summary_lines) != 1:
            raise AssertionError(f"missing execution summary: {streams!r}")
        return (
            json.loads(summary_lines[0][len(SUMMARY_PREFIX) :]),
            executed,
        )

    def test_source_inventory_metadata_and_public_boundary(self) -> None:
        self.assertEqual(self.notebook.nbformat, 4)
        self.assertEqual(len(self.notebook.cells), 20)
        self.assertEqual(len(self.markdown_cells), 10)
        self.assertEqual(len(self.code_cells), 10)
        self.assertEqual(
            len({cell.id for cell in self.notebook.cells}),
            len(self.notebook.cells),
        )
        self.assertEqual(
            self.notebook.metadata.kernelspec.name,
            "python3",
        )
        self.assertEqual(
            self.notebook.metadata.language_info.name,
            "python",
        )
        metadata_text = json.dumps(self.notebook.metadata, sort_keys=True)
        for forbidden in (
            "/Users/",
            "/private/",
            "tensor_dslab",
            "widgets",
            "timestamp",
        ):
            self.assertNotIn(forbidden, metadata_text)
        for cell in self.notebook.cells:
            self.assertNotIn("attachments", cell)
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

        self.assertTrue(
            self.markdown_cells[0].source.startswith("# Readout quickstart")
        )
        for index, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                self.assertGreater(index, 0)
                previous = self.notebook.cells[index - 1]
                self.assertEqual(previous.cell_type, "markdown")
                self.assertTrue(previous.source.strip())

        trees = tuple(ast.parse(cell.source) for cell in self.code_cells)
        imports = tuple(
            node
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        for node in imports:
            if isinstance(node, ast.Import):
                self.assertTrue(
                    all(
                        alias.name in ("matplotlib.pyplot", "torch")
                        for alias in node.names
                    )
                )
            else:
                self.assertIn(node.module, ("tensor_core", "tensor_dslab"))
        self.assertFalse(
            any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for tree in trees
                for node in ast.walk(tree)
            )
        )
        for forbidden in (
            "simulate_readout",
            "ReadoutConfig",
            "ReadoutCollection",
            "WhiteNoiseRms",
            "ds20k",
            "sys.path",
            "pip install",
            "requests",
            "urllib",
        ):
            self.assertNotIn(forbidden, self.code_source)
        self.assertFalse((ROOT / "demos" / "random.ipynb").exists())

    def test_exact_product_axis_psd_and_shape_source(self) -> None:
        self.assertIn(
            'labels=("sensor-0", "sensor-1", "sensor-2")',
            self.code_source,
        )
        self.assertIn("count=256", self.code_source)
        self.assertIn("coordinate_scale=2.0", self.code_source)
        self.assertIn("count=129", self.code_source)
        self.assertIn("coordinate_scale=1.953125", self.code_source)
        self.assertIn(
            "conditioning_axes=(channel_axis,)",
            self.code_source,
        )
        self.assertIn(
            "operation_axes=(frequency_axis,)",
            self.code_source,
        )
        self.assertIn("psd_sensor_0 = torch.cat", self.code_source)
        self.assertIn("psd_sensor_1 = torch.cat", self.code_source)
        self.assertIn("psd_sensor_2 = torch.cat", self.code_source)
        self.assertNotIn("torch.rand", self.code_source)

        assignments: list[tuple[str, str]] = []
        for cell in self.code_cells:
            for node in ast.parse(cell.source).body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                ):
                    function = node.value.func
                    if isinstance(function, ast.Name):
                        call_name = function.id
                    elif (
                        isinstance(function, ast.Attribute)
                        and isinstance(function.value, ast.Name)
                    ):
                        call_name = f"{function.value.id}.{function.attr}"
                    else:
                        continue
                    assignments.append((node.targets[0].id, call_name))
        sequence = tuple(
            assignment
            for assignment in assignments
            if assignment[0]
            in (
                "photoelectrons",
                "charge",
                "pure_waveform",
                "noise_waveform",
                "analog_waveform",
                "digitized_waveform",
            )
        )
        self.assertEqual(
            sequence,
            (
                ("photoelectrons", "Photoelectrons"),
                ("charge", "Charge.create"),
                ("pure_waveform", "PureWaveform.create"),
                ("noise_waveform", "NoiseWaveform.create"),
                ("analog_waveform", "AnalogWaveform.create"),
                ("digitized_waveform", "DigitizedWaveform.create"),
            ),
        )

        shape_cell = next(
            cell
            for cell in self.code_cells
            if cell.id == "shared-shape-code"
        )
        shape_tree = ast.parse(shape_cell.source)
        assertions = tuple(
            node for node in shape_tree.body if isinstance(node, ast.Assert)
        )
        self.assertEqual(len(assertions), 6)
        products = []
        for assertion in assertions:
            self.assertIsInstance(assertion.test, ast.Compare)
            comparison = assertion.test
            assert isinstance(comparison, ast.Compare)
            self.assertIsInstance(comparison.left, ast.Attribute)
            shape_attribute = comparison.left
            assert isinstance(shape_attribute, ast.Attribute)
            self.assertEqual(shape_attribute.attr, "shape")
            self.assertIsInstance(shape_attribute.value, ast.Attribute)
            tensor_attribute = shape_attribute.value
            assert isinstance(tensor_attribute, ast.Attribute)
            self.assertEqual(tensor_attribute.attr, "tensor")
            self.assertIsInstance(tensor_attribute.value, ast.Name)
            product_name = tensor_attribute.value
            assert isinstance(product_name, ast.Name)
            products.append(product_name.id)
            self.assertEqual(len(comparison.comparators), 1)
            self.assertIsInstance(comparison.comparators[0], ast.Name)
            comparator = comparison.comparators[0]
            assert isinstance(comparator, ast.Name)
            self.assertEqual(comparator.id, "expected_shape")
        self.assertEqual(
            tuple(products),
            (
                "photoelectrons",
                "charge",
                "pure_waveform",
                "noise_waveform",
                "analog_waveform",
                "digitized_waveform",
            ),
        )

        analog_assertions = tuple(
            node
            for cell in self.code_cells
            for node in ast.walk(ast.parse(cell.source))
            if isinstance(node, ast.Assert)
            and "analog_waveform.tensor.shape" not in ast.unparse(node)
            and "analog_waveform" in ast.unparse(node)
        )
        self.assertEqual(analog_assertions, ())

    def test_clean_execution_replays_products_and_plot_structure(self) -> None:
        first, first_notebook = self._execute()
        second, second_notebook = self._execute()
        self.assertEqual(first, second)

        self.assertEqual(
            first["types"],
            [
                "Photoelectrons",
                "Charge",
                "PureWaveform",
                "NoiseWaveform",
                "AnalogWaveform",
                "DigitizedWaveform",
            ],
        )
        self.assertEqual(first["shapes"], [[1, 3, 256]] * 6)
        self.assertEqual(first["devices"], ["cpu"] * 6)
        self.assertEqual(
            first["dtypes"],
            [
                "torch.int64",
                "torch.float32",
                "torch.float32",
                "torch.float32",
                "torch.float32",
                "torch.int32",
            ],
        )
        self.assertTrue(first["source_unchanged"])
        self.assertTrue(first["photo_charge_equal"])
        self.assertEqual(
            first["channel_labels"],
            ["sensor-0", "sensor-1", "sensor-2"],
        )
        self.assertEqual(first["time_size"], 256)
        self.assertEqual(first["time_scale"], 2.0)
        self.assertEqual(first["frequency_size"], 129)
        self.assertEqual(first["frequency_scale"], 1.953125)
        self.assertEqual(first["psd_shape"], [3, 129])
        self.assertEqual(first["psd_conditioning"], ["ChannelAxis"])
        self.assertEqual(first["psd_operation"], ["FrequencyAxis"])
        self.assertTrue(first["psd_dc_zero"])
        self.assertTrue(first["psd_rows_distinct"])
        self.assertTrue(all(first["sensor_products_distinct"]))

        self.assertEqual(first["figure_axes"], 6)
        self.assertEqual(first["line_counts"], [3] * 6)
        expected_colors = ["tab:blue", "tab:orange", "tab:green"]
        self.assertEqual(first["colors"], [expected_colors] * 6)
        self.assertEqual(
            first["drawstyles"],
            [
                ["steps-post"] * 3,
                ["steps-post"] * 3,
                ["default"] * 3,
                ["default"] * 3,
                ["default"] * 3,
                ["steps-post"] * 3,
            ],
        )
        self.assertEqual(first["axis_legends"], [False] * 6)
        self.assertEqual(first["figure_legends"], 1)
        self.assertEqual(first["xlabels"], ["", "", "", "", "", "Time (ns)"])
        self.assertGreater(first["adc_min"], 0)
        self.assertLess(first["adc_max"], (1 << 12) - 1)

        for executed in (first_notebook, second_notebook):
            original_cells = executed.cells[:-1]
            errors = tuple(
                output
                for cell in original_cells
                if cell.cell_type == "code"
                for output in cell.outputs
                if output.output_type == "error"
            )
            displays = tuple(
                output
                for cell in original_cells
                if cell.cell_type == "code"
                for output in cell.outputs
                if output.output_type == "display_data"
            )
            self.assertEqual(errors, ())
            self.assertEqual(len(displays), 1)
            self.assertIn("image/png", displays[0].data)


if __name__ == "__main__":
    unittest.main()
