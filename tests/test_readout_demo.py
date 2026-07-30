"""Focused source and execution proof for the public readout quickstart."""

import ast
import copy
import json
from pathlib import Path
import unittest

from nbclient import NotebookClient
import nbformat
from nbformat.notebooknode import NotebookNode


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
    encoded_waveform,
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
    "deposits": [
        [
            int(position[1]),
            int(position[2]),
            int(photoelectrons.tensor[tuple(position)]),
        ]
        for position in torch.nonzero(
            photoelectrons.tensor,
            as_tuple=False,
        ).tolist()
    ],
    "time_size": time_axis.size,
    "time_scale": time_axis.coordinate_scale,
    "frequency_size": frequency_axis.size,
    "frequency_scale": frequency_axis.coordinate_scale,
    "pulse_support_ns": pulse_support_ns,
    "pulse_coefficient_count": pulse_coefficient_count,
    "pulse_offset_count": pulse_time_axis.size,
    "pulse_minimum": float(pulse_values.min()),
    "pulse_finite": bool(torch.isfinite(pulse_values).all()),
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
    "psd_band_counts": [
        [
            int((psd_sensor_0 == 0.0).sum()),
            int((psd_sensor_0 == 0.012).sum()),
            int((psd_sensor_0 == 0.004).sum()),
        ],
        [
            int((psd_sensor_1 == 0.0).sum()),
            int((psd_sensor_1 == 0.008).sum()),
            int((psd_sensor_1 == 0.016).sum()),
        ],
        [
            int((psd_sensor_2 == 0.0).sum()),
            int((psd_sensor_2 == 0.020).sum()),
            int((psd_sensor_2 == 0.006).sum()),
        ],
    ],
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
    "input_minimum": float(input_minimum.tensor),
    "input_maximum": float(input_maximum.tensor),
    "bit_depth": int(bit_depth.tensor),
    "analog_gain": float(analog_gain.tensor),
    "zle_policy": [
        int(trigger_threshold_code.tensor),
        int(release_threshold_code.tensor),
        int(required_time_over_samples.tensor),
        int(pre_trigger_samples.tensor),
        int(post_trigger_samples.tensor),
    ],
    "suppression_code": encoded_waveform.spec.suppression_code,
    "encoded_support": [
        [
            [start, end]
            for start, end in zip(
                [
                    index
                    for index in range(time_axis.size)
                    if (
                        encoded_waveform.tensor[0, channel_index, index]
                        != encoded_waveform.spec.suppression_code
                        and (
                            index == 0
                            or encoded_waveform.tensor[
                                0, channel_index, index - 1
                            ]
                            == encoded_waveform.spec.suppression_code
                        )
                    )
                ],
                [
                    index
                    for index in range(1, time_axis.size + 1)
                    if (
                        encoded_waveform.tensor[
                            0, channel_index, index - 1
                        ]
                        != encoded_waveform.spec.suppression_code
                        and (
                            index == time_axis.size
                            or encoded_waveform.tensor[
                                0, channel_index, index
                            ]
                            == encoded_waveform.spec.suppression_code
                        )
                    )
                ],
            )
        ]
        for channel_index in range(channel_axis.size)
    ],
    "encoded_plot_gaps": [
        any(math.isnan(float(value)) for value in line.get_ydata())
        for line in plot_axes[6].lines
    ],
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
    def _execute(cls) -> tuple[dict, NotebookNode]:
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
        self.assertEqual(len(self.notebook.cells), 34)
        self.assertEqual(len(self.markdown_cells), 17)
        self.assertEqual(len(self.code_cells), 17)
        self.assertEqual(
            tuple(cell.cell_type for cell in self.notebook.cells),
            ("markdown", "code") * 17,
        )
        self.assertEqual(
            tuple(cell.id for cell in self.code_cells),
            (
                "imports-code",
                "axes-code",
                "photoelectron-values-code",
                "photoelectrons-code",
                "charge-code",
                "pulse-math-code",
                "pure-waveform-code",
                "psd-values-code",
                "noise-waveform-code",
                "analog-waveform-code",
                "digitizer-values-code",
                "digitized-waveform-code",
                "encoding-values-code",
                "encoded-waveform-code",
                "shared-shape-code",
                "plot-preparation-code",
                "product-views-code",
            ),
        )
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
                        alias.name in ("math", "matplotlib.pyplot", "torch")
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
        markdown_source = "\n".join(
            cell.source for cell in self.markdown_cells
        )
        self.assertIn(
            "one Product call processes several sensors together in one tensor",
            markdown_source,
        )
        self.assertIn(
            "illustrative rather than calibrated",
            markdown_source,
        )

    def test_exact_product_axis_psd_and_shape_source(self) -> None:
        self.assertIn(
            'labels=("sensor-0", "sensor-1", "sensor-2")',
            self.code_source,
        )
        self.assertIn("count=5000", self.code_source)
        self.assertIn("coordinate_scale=2.0", self.code_source)
        self.assertIn("count=2501", self.code_source)
        self.assertIn("coordinate_scale=0.1", self.code_source)
        for source_line in (
            "photoelectron_values[0, 0, 100] = 1",
            "photoelectron_values[0, 0, 3700] = 4",
            "photoelectron_values[0, 1, 1300] = 2",
            "photoelectron_values[0, 2, 2500] = 3",
            "pulse_support_ns = 2020.27",
            "pulse_time_ns - 232.89",
            "2.0 * 507.72**2",
            "math.pi * 507.72**2",
            "pulse_x - (-81.92)",
            "math.sqrt(2.0) * 147.28",
            "pulse_x - (-176.50)",
            "math.sqrt(2.0) * 45.69",
            "torch.max(torch.abs(pulse_raw)) * -14.5912372",
            "OffsetCoordinates(offsets=tuple(range(1011)))",
            "torch.full((625,), 0.012",
            "torch.full((1875,), 0.004",
            "torch.full((938,), 0.008",
            "torch.full((1562,), 0.016",
            "torch.full((1250,), 0.020",
            "torch.full((1250,), 0.006",
            "bit_depth_value = torch.tensor(12",
            "input_minimum_value = torch.tensor(\n    -80.0",
            "input_maximum_value = torch.tensor(\n    20.0",
            "analog_gain_value = torch.tensor(\n    1.0",
            "trigger_threshold_value = torch.tensor(\n    2500",
            "release_threshold_value = torch.tensor(\n    2800",
            "required_time_over_value = torch.tensor(\n    3",
            "pre_trigger_value = torch.tensor(\n    25",
            "post_trigger_value = torch.tensor(\n    50",
            "suppression_code=-1",
            "encoded_waveform = EncodedWaveform.create(",
        ):
            with self.subTest(source_line=source_line):
                self.assertIn(source_line, self.code_source)
        self.assertEqual(self.code_source.count("torch.erf("), 2)
        self.assertEqual(
            self.code_source.count("pulse_gaussian = torch.exp("),
            1,
        )
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
        self.assertIn(
            "The last panel leaves suppressed regions blank",
            "\n".join(cell.source for cell in self.markdown_cells),
        )
        self.assertIn(
            'if value == encoded_waveform.spec.suppression_code',
            self.code_source,
        )
        self.assertIn('float("nan")', self.code_source)

        photoelectron_values_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "photoelectron-values-code"
        )
        photoelectrons_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "photoelectrons-code"
        )
        self.assertNotIn("Photoelectrons", photoelectron_values_source)
        self.assertNotIn("PhotoelectronsSpec", photoelectron_values_source)
        self.assertIn("PhotoelectronsSpec", photoelectrons_source)
        self.assertIn("photoelectrons = Photoelectrons(", photoelectrons_source)
        self.assertNotIn("photoelectron_values[", photoelectrons_source)

        pulse_math_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "pulse-math-code"
        )
        pure_waveform_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "pure-waveform-code"
        )
        for calculation in (
            "pulse_support_ns = 2020.27",
            "pulse_gaussian = torch.exp(",
            "torch.erf(",
            "pulse_raw = ",
            "pulse_values = ",
        ):
            self.assertIn(calculation, pulse_math_source)
        for semantic_name in (
            "OffsetAxis",
            "PulseResponse",
            "PulseResponseSpec",
            "PureWaveformConfig",
            "PureWaveformKernels",
            "PureWaveformSpec",
            "PureWaveform.create",
        ):
            self.assertNotIn(semantic_name, pulse_math_source)
        pure_sequence = (
            "pulse_time_axis = OffsetAxis(",
            "pulse_response = PulseResponse(",
            "pure_waveform_config = PureWaveformConfig(",
            "pure_waveform = PureWaveform.create(",
        )
        self.assertTrue(all(item in pure_waveform_source for item in pure_sequence))
        self.assertEqual(
            tuple(pure_waveform_source.index(item) for item in pure_sequence),
            tuple(
                sorted(
                    pure_waveform_source.index(item)
                    for item in pure_sequence
                )
            ),
        )
        for calculation in ("torch.exp(", "torch.erf(", "pulse_raw"):
            self.assertNotIn(calculation, pure_waveform_source)

        psd_values_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "psd-values-code"
        )
        noise_waveform_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "noise-waveform-code"
        )
        for calculation in ("torch.cat(", "torch.full(", "torch.stack("):
            self.assertIn(calculation, psd_values_source)
            self.assertNotIn(calculation, noise_waveform_source)
        for semantic_name in (
            "PowerSpectralDensitySpec",
            "PowerSpectralDensity(",
            "NoiseWaveformConfig",
            "NoiseWaveformKernels",
            "NoiseWaveformSpec",
            "NoiseWaveform.create",
        ):
            self.assertNotIn(semantic_name, psd_values_source)
            self.assertIn(semantic_name, noise_waveform_source)

        digitizer_values_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "digitizer-values-code"
        )
        digitized_waveform_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "digitized-waveform-code"
        )
        self.assertEqual(digitizer_values_source.count("torch.tensor("), 4)
        self.assertNotIn("DigitizedWaveform", digitizer_values_source)
        for semantic_name in (
            "BitDepth(",
            "InputMinimum(",
            "InputMaximum(",
            "AnalogGain(",
            "DigitizedWaveformConfig",
            "DigitizedWaveformKernels",
            "DigitizedWaveformSpec",
            "DigitizedWaveform.create",
        ):
            self.assertIn(semantic_name, digitized_waveform_source)
        self.assertNotIn("torch.tensor(", digitized_waveform_source)

        encoding_values_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "encoding-values-code"
        )
        encoded_waveform_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "encoded-waveform-code"
        )
        self.assertEqual(encoding_values_source.count("torch.tensor("), 5)
        self.assertNotIn("EncodedWaveform", encoding_values_source)
        for semantic_name in (
            "TriggerThresholdCode(",
            "ReleaseThresholdCode(",
            "RequiredTimeOverSamples(",
            "PreTriggerSamples(",
            "PostTriggerSamples(",
            "EncodedWaveformConfig",
            "EncodedWaveformKernels",
            "EncodedWaveformSpec",
            "EncodedWaveform.create",
        ):
            self.assertIn(semantic_name, encoded_waveform_source)
        self.assertNotIn("torch.tensor(", encoded_waveform_source)

        assertion_cells = tuple(
            cell.id
            for cell in self.code_cells
            if any(
                isinstance(node, ast.Assert)
                for node in ast.walk(ast.parse(cell.source))
            )
        )
        self.assertEqual(assertion_cells, ("shared-shape-code",))

        plot_preparation_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "plot-preparation-code"
        )
        product_views_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "product-views-code"
        )
        for preparation in (
            "channel_labels = ",
            "sensor_colors = ",
            "time_ns = ",
            "products = ",
            "y_labels = ",
            "step_panels = ",
        ):
            self.assertIn(preparation, plot_preparation_source)
        for rendering in (
            "plt.style.use(",
            "plt.subplots(",
            "plot_axis.step(",
            "plot_axis.plot(",
            "figure.legend(",
            "plt.show()",
        ):
            self.assertNotIn(rendering, plot_preparation_source)
            self.assertIn(rendering, product_views_source)
        self.assertNotIn("products = ", product_views_source)
        self.assertNotIn(".create(", plot_preparation_source)
        self.assertNotIn(".create(", product_views_source)

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
                "encoded_waveform",
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
                ("encoded_waveform", "EncodedWaveform.create"),
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
        self.assertEqual(len(assertions), 7)
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
                "encoded_waveform",
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
                "EncodedWaveform",
            ],
        )
        self.assertEqual(first["shapes"], [[1, 3, 5000]] * 7)
        self.assertEqual(first["devices"], ["cpu"] * 7)
        self.assertEqual(
            first["dtypes"],
            [
                "torch.int64",
                "torch.float32",
                "torch.float32",
                "torch.float32",
                "torch.float32",
                "torch.int32",
                "torch.int32",
            ],
        )
        expected_units = [
            "avalanche",
            "avalanche",
            "millivolt",
            "millivolt",
            "millivolt",
            "dimensionless",
            "dimensionless",
        ]
        expected_ylabels = [
            "Photoelectrons",
            "Charge (avalanche)",
            "Pure (mV)",
            "Noise (mV)",
            "Analog (mV)",
            "ADC code",
            "Retained ADC code",
        ]
        self.assertEqual(first["units"], expected_units)
        self.assertTrue(first["source_unchanged"])
        self.assertTrue(first["photo_charge_equal"])
        self.assertEqual(
            first["channel_labels"],
            ["sensor-0", "sensor-1", "sensor-2"],
        )
        self.assertEqual(
            first["deposits"],
            [
                [0, 100, 1],
                [0, 3700, 4],
                [1, 1300, 2],
                [2, 2500, 3],
            ],
        )
        self.assertEqual(first["time_size"], 5000)
        self.assertEqual(first["time_scale"], 2.0)
        self.assertEqual(first["frequency_size"], 2501)
        self.assertEqual(first["frequency_scale"], 0.1)
        self.assertEqual(first["pulse_support_ns"], 2020.27)
        self.assertEqual(first["pulse_coefficient_count"], 1011)
        self.assertEqual(first["pulse_offset_count"], 1011)
        self.assertAlmostEqual(first["pulse_minimum"], -14.5912372, places=5)
        self.assertTrue(first["pulse_finite"])
        self.assertEqual(first["psd_shape"], [3, 2501])
        self.assertEqual(first["psd_conditioning"], ["ChannelAxis"])
        self.assertEqual(first["psd_operation"], ["FrequencyAxis"])
        self.assertTrue(first["psd_dc_zero"])
        self.assertTrue(first["psd_rows_distinct"])
        self.assertEqual(
            first["psd_band_counts"],
            [
                [1, 625, 1875],
                [1, 938, 1562],
                [1, 1250, 1250],
            ],
        )
        self.assertTrue(all(first["sensor_products_distinct"]))
        self.assertEqual(first["input_minimum"], -80.0)
        self.assertEqual(first["input_maximum"], 20.0)
        self.assertEqual(first["bit_depth"], 12)
        self.assertEqual(first["analog_gain"], 1.0)
        self.assertEqual(first["zle_policy"], [2500, 2800, 3, 25, 50])
        self.assertEqual(first["suppression_code"], -1)
        self.assertEqual(
            first["encoded_support"],
            [
                [[280, 362], [3721, 4261]],
                [[1363, 1749]],
                [[2541, 3085]],
            ],
        )
        self.assertEqual(first["encoded_plot_gaps"], [True, True, True])

        self.assertEqual(first["figure_axes"], 7)
        self.assertEqual(first["line_counts"], [3] * 7)
        expected_colors = ["tab:blue", "tab:orange", "tab:green"]
        self.assertEqual(first["colors"], [expected_colors] * 7)
        self.assertEqual(
            first["drawstyles"],
            [
                ["steps-post"] * 3,
                ["steps-post"] * 3,
                ["default"] * 3,
                ["default"] * 3,
                ["default"] * 3,
                ["steps-post"] * 3,
                ["steps-post"] * 3,
            ],
        )
        self.assertEqual(first["axis_legends"], [False] * 7)
        self.assertEqual(first["figure_legends"], 1)
        self.assertEqual(
            first["xlabels"],
            ["", "", "", "", "", "", "Time (ns)"],
        )
        self.assertEqual(first["ylabels"], expected_ylabels)
        self.assertEqual(
            tuple(zip(first["units"], first["ylabels"])),
            tuple(zip(expected_units, expected_ylabels)),
        )
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
