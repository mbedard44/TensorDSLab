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
plot_axes = [figure.axes[0] for figure in product_figures]
layout_bounds = []
for figure, axis in zip(product_figures, plot_axes):
    figure.draw_without_rendering()
    renderer = figure._get_renderer()
    layout_bounds.append(
        {{
            "figure": list(figure.bbox.bounds),
            "xlabel": list(
                axis.xaxis.label.get_window_extent(renderer).bounds
            ),
            "legend": list(
                axis.get_legend().get_window_extent(renderer).bounds
            ),
        }}
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
    "figure_count": len(product_figures),
    "figure_axes": [len(figure.axes) for figure in product_figures],
    "line_counts": [len(axis.lines) for axis in plot_axes],
    "colors": [
        [line.get_color() for line in axis.lines]
        for axis in plot_axes
    ],
    "drawstyles": [
        [line.get_drawstyle() for line in axis.lines]
        for axis in plot_axes
    ],
    "alphas": [
        [line.get_alpha() for line in axis.lines]
        for axis in plot_axes
    ],
    "linewidths": [
        [line.get_linewidth() for line in axis.lines]
        for axis in plot_axes
    ],
    "axis_legends": [
        axis.get_legend() is not None
        for axis in plot_axes
    ],
    "figure_legends": [
        len(figure.legends) for figure in product_figures
    ],
    "legend_labels": [
        [text.get_text() for text in axis.get_legend().get_texts()]
        for axis in plot_axes
    ],
    "legend_locations": [
        axis.get_legend()._loc for axis in plot_axes
    ],
    "legend_columns": [
        axis.get_legend()._ncols for axis in plot_axes
    ],
    "legend_frames": [
        axis.get_legend().get_frame_on() for axis in plot_axes
    ],
    "layout_bounds": layout_bounds,
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
        self.assertEqual(len(self.notebook.cells), 46)
        self.assertEqual(len(self.markdown_cells), 23)
        self.assertEqual(len(self.code_cells), 23)
        self.assertEqual(
            tuple(cell.cell_type for cell in self.notebook.cells),
            ("markdown", "code") * 23,
        )
        self.assertEqual(
            tuple(cell.id for cell in self.code_cells),
            (
                "imports-code",
                "axes-code",
                "plotting-code",
                "photoelectron-values-code",
                "photoelectrons-code",
                "photoelectrons-view-code",
                "charge-code",
                "charge-view-code",
                "pulse-math-code",
                "pure-waveform-code",
                "pure-waveform-view-code",
                "psd-values-code",
                "noise-waveform-code",
                "noise-waveform-view-code",
                "analog-waveform-code",
                "analog-waveform-view-code",
                "digitizer-values-code",
                "digitized-waveform-code",
                "digitized-waveform-view-code",
                "encoding-values-code",
                "encoded-waveform-code",
                "encoded-waveform-view-code",
                "shared-shape-code",
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
        functions = tuple(
            node
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        self.assertEqual(len(functions), 1)
        self.assertIsInstance(functions[0], ast.FunctionDef)
        self.assertEqual(functions[0].name, "plot_product")
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
            "leaves suppressed regions blank",
            "\n".join(cell.source for cell in self.markdown_cells),
        )
        self.assertIn(
            "if value == suppression_code",
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

        plotting_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "plotting-code"
        )
        for plotting_contract in (
            'plt.style.use("seaborn-v0_8-whitegrid")',
            "channel_labels = ",
            'sensor_colors = ("tab:blue", "tab:orange", "tab:green")',
            "float(time_axis.quantity_at(index).to(\"ns\").magnitude)",
            "def plot_product(",
            "product.tensor[0, channel_index].detach().cpu().tolist()",
            "figsize=(11, 3.6)",
            "plot_axis.step(",
            'where="post"',
            "plot_axis.plot(",
            "alpha=0.72",
            "linewidth=0.9",
            'plot_axis.set_xlabel("Time (ns)")',
            "plot_axis.set_ylabel(y_label)",
            'loc="upper center"',
            "bbox_to_anchor=(0.5, -0.24)",
            "ncol=3",
            "frameon=False",
            "figure.subplots_adjust(bottom=0.29)",
            "plt.show()",
        ):
            with self.subTest(plotting_contract=plotting_contract):
                self.assertIn(plotting_contract, plotting_source)
        self.assertEqual(plotting_source.count("alpha=0.72"), 2)
        self.assertEqual(plotting_source.count("linewidth=0.9"), 2)
        for retired_combined_contract in (
            "products = ",
            "y_labels = ",
            "step_panels = ",
            "figure.legend(",
            "plot-preparation-code",
            "product-views-code",
        ):
            self.assertNotIn(retired_combined_contract, self.code_source)
        self.assertNotIn(".create(", plotting_source)

        plotting_tree = ast.parse(plotting_source)
        helper_definitions = tuple(
            node
            for node in plotting_tree.body
            if isinstance(node, ast.FunctionDef)
        )
        self.assertEqual(len(helper_definitions), 1)
        helper = helper_definitions[0]
        self.assertEqual(helper.name, "plot_product")
        self.assertEqual(
            tuple(argument.arg for argument in helper.args.args),
            ("product",),
        )
        self.assertEqual(
            tuple(argument.arg for argument in helper.args.kwonlyargs),
            ("y_label", "step", "suppression_code"),
        )

        view_contracts = (
            (
                "photoelectrons-code",
                "photoelectrons-view-code",
                "photoelectrons",
                "Photoelectrons",
                True,
                None,
            ),
            (
                "charge-code",
                "charge-view-code",
                "charge",
                "Charge (avalanche)",
                True,
                None,
            ),
            (
                "pure-waveform-code",
                "pure-waveform-view-code",
                "pure_waveform",
                "Pure (mV)",
                False,
                None,
            ),
            (
                "noise-waveform-code",
                "noise-waveform-view-code",
                "noise_waveform",
                "Noise (mV)",
                False,
                None,
            ),
            (
                "analog-waveform-code",
                "analog-waveform-view-code",
                "analog_waveform",
                "Analog (mV)",
                False,
                None,
            ),
            (
                "digitized-waveform-code",
                "digitized-waveform-view-code",
                "digitized_waveform",
                "ADC code",
                True,
                None,
            ),
            (
                "encoded-waveform-code",
                "encoded-waveform-view-code",
                "encoded_waveform",
                "Retained ADC code",
                True,
                "encoded_waveform.spec.suppression_code",
            ),
        )
        all_cell_ids = tuple(cell.id for cell in self.notebook.cells)
        for (
            construction_id,
            view_id,
            product_name,
            y_label,
            step,
            suppression,
        ) in view_contracts:
            with self.subTest(view_id=view_id):
                construction_index = all_cell_ids.index(construction_id)
                self.assertEqual(
                    all_cell_ids[construction_index + 2],
                    view_id,
                )
                view_cell = self.notebook.cells[construction_index + 2]
                view_tree = ast.parse(view_cell.source)
                self.assertEqual(len(view_tree.body), 1)
                expression = view_tree.body[0]
                self.assertIsInstance(expression, ast.Expr)
                assert isinstance(expression, ast.Expr)
                self.assertIsInstance(expression.value, ast.Call)
                call = expression.value
                assert isinstance(call, ast.Call)
                self.assertIsInstance(call.func, ast.Name)
                assert isinstance(call.func, ast.Name)
                self.assertEqual(call.func.id, "plot_product")
                self.assertEqual(len(call.args), 1)
                self.assertIsInstance(call.args[0], ast.Name)
                assert isinstance(call.args[0], ast.Name)
                self.assertEqual(call.args[0].id, product_name)
                keywords = {
                    keyword.arg: keyword.value for keyword in call.keywords
                }
                self.assertEqual(
                    set(keywords),
                    {"y_label", "step"}
                    | ({"suppression_code"} if suppression else set()),
                )
                y_label_node = keywords["y_label"]
                self.assertIsInstance(y_label_node, ast.Constant)
                assert isinstance(y_label_node, ast.Constant)
                self.assertEqual(y_label_node.value, y_label)
                step_node = keywords["step"]
                self.assertIsInstance(step_node, ast.Constant)
                assert isinstance(step_node, ast.Constant)
                self.assertIs(step_node.value, step)
                if suppression is not None:
                    self.assertEqual(
                        ast.unparse(keywords["suppression_code"]),
                        suppression,
                    )
                self.assertNotIn(".create(", view_cell.source)
                self.assertNotIn("assert ", view_cell.source)
                self.assertNotIn("torch.", view_cell.source)

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
        self.assertEqual(
            first["digests"],
            [
                "8cd46f0f4ff164b63fc0068e22d7705cab82adbee84c804849925f436d247fc7",
                "66113e0e8952d8e3906217ca6efee0ade16884ff2a2ec095192faf0b72dda5b4",
                "7233c98dd4237a425703855271ecd85fa4a78440ffbf34f1b6fb8b3593481f45",
                "0f8798c98ec544dcbcfd57579d0f6ee9c1af2e24e10caf6832919f438be51208",
                "e625e37451c37424d345e470ca6426543e56a19dd2bfdf570ad5c3660bad39ad",
                "4bc693b1c8870cecc1827afcf6da88883908b5bc202250ba99c9beaa0d1d4780",
                "00644cce41594031fa44018ed27362c60234154e3dddafb6b6b517c87ca8335d",
            ],
        )
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

        self.assertEqual(first["figure_count"], 7)
        self.assertEqual(first["figure_axes"], [1] * 7)
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
        self.assertEqual(first["alphas"], [[0.72] * 3] * 7)
        self.assertEqual(first["linewidths"], [[0.9] * 3] * 7)
        self.assertEqual(first["axis_legends"], [True] * 7)
        self.assertEqual(first["figure_legends"], [0] * 7)
        self.assertEqual(
            first["legend_labels"],
            [["sensor-0", "sensor-1", "sensor-2"]] * 7,
        )
        self.assertEqual(first["legend_locations"], [9] * 7)
        self.assertEqual(first["legend_columns"], [3] * 7)
        self.assertEqual(first["legend_frames"], [False] * 7)
        self.assertEqual(first["xlabels"], ["Time (ns)"] * 7)
        self.assertEqual(first["ylabels"], expected_ylabels)
        for layout in first["layout_bounds"]:
            _, _, figure_width, figure_height = layout["figure"]
            xlabel_x, xlabel_y, xlabel_width, xlabel_height = layout[
                "xlabel"
            ]
            legend_x, legend_y, legend_width, legend_height = layout[
                "legend"
            ]
            for x, y, width, height in (
                (xlabel_x, xlabel_y, xlabel_width, xlabel_height),
                (legend_x, legend_y, legend_width, legend_height),
            ):
                self.assertGreaterEqual(x, 0.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(x + width, figure_width)
                self.assertLessEqual(y + height, figure_height)
            self.assertLessEqual(legend_y + legend_height, xlabel_y)
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
            self.assertEqual(len(displays), 7)
            for display in displays:
                self.assertIn("image/png", display.data)


if __name__ == "__main__":
    unittest.main()
