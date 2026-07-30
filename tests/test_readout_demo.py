"""Focused source and execution proof for the public readout quickstart."""

import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest

from matplotlib import font_manager
from nbclient import NotebookClient
import nbformat
from nbformat.notebooknode import NotebookNode
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "demos" / "readout.ipynb"
SUMMARY_PREFIX = "READOUT_DEMO_SUMMARY="
SOURCE_PROJECTION_SHA256 = (
    "b9fccd51e0a2afed827a9e21e7da5808c947fd659dae9af5ee7c89ded92db90f"
)
COMMITTED_NOTEBOOK_SHA256 = (
    "9c5ba35300f64edb421a53bff55b7df4777140f057050e67c36885c4506bd055"
)
FIGURE_TEXT = "<Figure size 1300x850 with 6 Axes>"
OUTPUT_HASHES = {
    "photoelectrons-view-code": (
        "3f5814fa5c83b856ecdb5a15def832251cd0a5cff91d3dfbda21bd44bceb9139"
    ),
    "charge-view-code": (
        "fa0b3be699497ee279410261d095a67a729e25f5a1a59c3cea6af52c2116688f"
    ),
    "pure-waveform-view-code": (
        "6df6a5ad87287b2cff09f3d0ce30d3901da1b9d026ee80dc433da3efdef34c2c"
    ),
    "noise-waveform-view-code": (
        "150a39b12bc3ac4c36c49baa64257d2758e76458cbe7ff88848062d27f2f3a3e"
    ),
    "analog-waveform-view-code": (
        "5101ea7c7b0a69712210f3d198a0fba220262cf30a1f386acdd9e9d841ed116a"
    ),
    "digitized-waveform-view-code": (
        "acacf7d175b10409051db6792aae50d2c0e9fabf52720aaae9755dccfc8d64e5"
    ),
    "encoded-waveform-view-code": (
        "242e6b8bf753219460a4fa8147a47a7081f907ef1dde0755cbb56acddf1fc0d3"
    ),
}


def _source_projection_hash(notebook: NotebookNode) -> str:
    projection = [
        (cell.cell_type, cell.id, cell.source) for cell in notebook.cells
    ]
    encoded = json.dumps(
        projection,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _decoded_png_hash(output: NotebookNode) -> str:
    png_bytes = base64.b64decode(output.data["image/png"])
    return hashlib.sha256(png_bytes).hexdigest()


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
plot_axes = [list(figure.axes) for figure in product_figures]
layout_bounds = []
for figure, axes_group in zip(product_figures, plot_axes):
    figure.draw_without_rendering()
    renderer = figure._get_renderer()
    text_bounds = []
    for axis in axes_group:
        for text_artist in (
            axis.title,
            axis.xaxis.label,
            axis.yaxis.label,
        ):
            if text_artist.get_text():
                text_bounds.append(
                    list(text_artist.get_window_extent(renderer).bounds)
                )
    if figure._suptitle is not None:
        text_bounds.append(
            list(figure._suptitle.get_window_extent(renderer).bounds)
        )
    layout_bounds.append(
        {{
            "figure": list(figure.bbox.bounds),
            "axes": [
                list(axis.get_window_extent(renderer).bounds)
                for axis in axes_group
            ],
            "text": text_bounds,
        }}
    )
summary = {{
    "types": [type(product).__name__ for product in product_values],
    "shapes": [list(product.tensor.shape) for product in product_values],
    "devices": [product.tensor.device.type for product in product_values],
    "dtypes": [str(product.tensor.dtype) for product in product_values],
    "units": [str(product.spec.unit) for product in product_values],
    "digests": [tensor_digest(product.tensor) for product in product_values],
    "example_digests": [
        [
            tensor_digest(product.tensor[example_index])
            for example_index in range(example_axis.size)
        ]
        for product in product_values
    ],
    "source_unchanged": bool(torch.equal(
        photoelectron_values,
        photoelectron_values_before,
    )),
    "channel_labels": list(channel_axis.coordinates.labels),
    "deposits": [
        [
            int(position[0]),
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
    "delayed_offsets": list(delayed_offsets),
    "delayed_shape": list(delayed_crosstalk.tensor.shape),
    "delayed_dtype": str(delayed_crosstalk.tensor.dtype),
    "delayed_unit": str(delayed_crosstalk.spec.unit),
    "delayed_sum": float(delayed_crosstalk.tensor.sum()),
    "delayed_first": float(delayed_crosstalk.tensor[0]),
    "delayed_last": float(delayed_crosstalk.tensor[-1]),
    "delayed_digest": tensor_digest(delayed_crosstalk.tensor),
    "delayed_conditioning": [
        type(axis).__name__ for axis in delayed_crosstalk.conditioning_axes
    ],
    "delayed_operation": [
        type(axis).__name__ for axis in delayed_crosstalk.operation_axes
    ],
    "delayed_relative_to": delayed_time_axis.relative_to.__name__,
    "generations": charge_config.correlated_avalanche_generations.value,
    "descendant_positions": [
        [int(value) for value in position]
        for position in torch.nonzero(
            charge.tensor
            > photoelectrons.tensor.to(charge.tensor.dtype),
            as_tuple=False,
        ).tolist()
    ],
    "descendant_count": int(
        (
            charge.tensor
            > photoelectrons.tensor.to(charge.tensor.dtype)
        ).sum()
    ),
    "descendant_total": float(
        (
            charge.tensor
            - photoelectrons.tensor.to(charge.tensor.dtype)
        ).sum()
    ),
    "earliest_source": {{
        f"{{example_index}}:{{channel_index}}": int(
            torch.nonzero(
                photoelectrons.tensor[example_index, channel_index] > 0,
                as_tuple=False,
            ).min()
        )
        for example_index in range(example_axis.size)
        for channel_index in range(channel_axis.size)
    }},
    "charge_integer": bool(torch.equal(
        charge.tensor,
        charge.tensor.floor(),
    )),
    "pulse_support_ns": pulse_support_ns,
    "pulse_coefficient_count": pulse_coefficient_count,
    "pulse_offset_count": pulse_time_axis.size,
    "pulse_minimum": float(pulse_values.min()),
    "pulse_finite": bool(torch.isfinite(pulse_values).all()),
    "psd_shape": list(power_spectral_density.tensor.shape),
    "psd_dtype": str(power_spectral_density.tensor.dtype),
    "psd_device": power_spectral_density.tensor.device.type,
    "psd_unit": str(power_spectral_density.spec.unit),
    "psd_digest": tensor_digest(power_spectral_density.tensor),
    "psd_sum": sum(
        float(value) for value in power_spectral_density.tensor
    ),
    "psd_nonzero": int(torch.count_nonzero(power_spectral_density.tensor)),
    "psd_zero": int((power_spectral_density.tensor == 0).sum()),
    "psd_dc": float(power_spectral_density.tensor[0]),
    "psd_positive_prefix": bool(
        (power_spectral_density.tensor[1:626] > 0).all()
    ),
    "psd_zero_suffix": bool(
        (power_spectral_density.tensor[626:] == 0).all()
    ),
    "psd_conditioning": [
        type(axis).__name__
        for axis in power_spectral_density.conditioning_axes
    ],
    "psd_operation": [
        type(axis).__name__
        for axis in power_spectral_density.operation_axes
    ],
    "noise_lane_digests": [
        tensor_digest(noise_waveform.tensor[example_index, channel_index])
        for example_index in range(example_axis.size)
        for channel_index in range(channel_axis.size)
    ],
    "figure_count": len(product_figures),
    "figure_axes": [len(figure.axes) for figure in product_figures],
    "line_counts": [
        [len(axis.lines) for axis in axes_group]
        for axes_group in plot_axes
    ],
    "colors": [
        [
            [line.get_color() for line in axis.lines]
            for axis in axes_group
        ]
        for axes_group in plot_axes
    ],
    "drawstyles": [
        [
            [line.get_drawstyle() for line in axis.lines]
            for axis in axes_group
        ]
        for axes_group in plot_axes
    ],
    "alphas": [
        [
            [line.get_alpha() for line in axis.lines]
            for axis in axes_group
        ]
        for axes_group in plot_axes
    ],
    "linewidths": [
        [
            [line.get_linewidth() for line in axis.lines]
            for axis in axes_group
        ]
        for axes_group in plot_axes
    ],
    "axis_legends": [
        [
            axis.get_legend() is not None
            for axis in axes_group
        ]
        for axes_group in plot_axes
    ],
    "figure_legends": [
        len(figure.legends) for figure in product_figures
    ],
    "figure_titles": [
        figure._suptitle.get_text()
        if figure._suptitle is not None
        else ""
        for figure in product_figures
    ],
    "axis_titles": [
        [axis.get_title() for axis in axes_group]
        for axes_group in plot_axes
    ],
    "xlabels": [
        [axis.get_xlabel() for axis in axes_group]
        for axes_group in plot_axes
    ],
    "ylabels": [
        [axis.get_ylabel() for axis in axes_group]
        for axes_group in plot_axes
    ],
    "layout_bounds": layout_bounds,
    "adc_min": int(digitized_waveform.tensor.min()),
    "adc_max": int(digitized_waveform.tensor.max()),
    "adc_example_extrema": [
        [
            int(digitized_waveform.tensor[index].min()),
            int(digitized_waveform.tensor[index].max()),
        ]
        for index in range(example_axis.size)
    ],
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
            [
                [start, end]
                for start, end in zip(
                    [
                        index
                        for index in range(time_axis.size)
                        if (
                            encoded_waveform.tensor[
                                example_index, channel_index, index
                            ]
                            != encoded_waveform.spec.suppression_code
                            and (
                                index == 0
                                or encoded_waveform.tensor[
                                    example_index,
                                    channel_index,
                                    index - 1,
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
                                example_index,
                                channel_index,
                                index - 1,
                            ]
                            != encoded_waveform.spec.suppression_code
                            and (
                                index == time_axis.size
                                or encoded_waveform.tensor[
                                    example_index,
                                    channel_index,
                                    index,
                                ]
                                == encoded_waveform.spec.suppression_code
                            )
                        )
                    ],
                )
            ]
            for channel_index in range(channel_axis.size)
        ]
        for example_index in range(example_axis.size)
    ],
    "encoded_plot_gaps": [
        [
            any(math.isnan(float(value)) for value in axis.lines[0].get_ydata())
            for axis in plot_axes[6]
        ]
    ],
}}
print({SUMMARY_PREFIX!r} + json.dumps(summary, sort_keys=True))
"""


class ReadoutDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        font_manager.findfont("DejaVu Sans")
        cls.notebook_bytes = NOTEBOOK_PATH.read_bytes()
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.code_cells = tuple(
            cell for cell in cls.notebook.cells if cell.cell_type == "code"
        )
        cls.markdown_cells = tuple(
            cell for cell in cls.notebook.cells if cell.cell_type == "markdown"
        )
        cls.code_source = "\n".join(cell.source for cell in cls.code_cells)

    @classmethod
    def _cleared_copy(cls) -> NotebookNode:
        notebook = copy.deepcopy(cls.notebook)
        for cell in notebook.cells:
            cell.metadata = {}
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
        return notebook

    @classmethod
    def _execute(cls) -> tuple[dict, NotebookNode]:
        notebook = cls._cleared_copy()
        code_cells = tuple(
            cell for cell in notebook.cells if cell.cell_type == "code"
        )
        if any(
            cell.execution_count is not None or cell.outputs
            for cell in code_cells
        ):
            raise AssertionError("fresh replay retained committed execution")
        notebook.cells.append(nbformat.v4.new_code_cell(PROBE_SOURCE))
        executed = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            record_timing=False,
        ).execute(cwd=str(ROOT))
        if NOTEBOOK_PATH.read_bytes() != cls.notebook_bytes:
            raise AssertionError("fresh replay changed the committed notebook")
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
        self.assertEqual(len(self.notebook.cells), 48)
        self.assertEqual(len(self.markdown_cells), 24)
        self.assertEqual(len(self.code_cells), 24)
        self.assertEqual(
            tuple(cell.cell_type for cell in self.notebook.cells),
            ("markdown", "code") * 24,
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
                "delayed-crosstalk-values-code",
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
            hashlib.sha256(self.notebook_bytes).hexdigest(),
            COMMITTED_NOTEBOOK_SHA256,
        )
        self.assertEqual(len(self.notebook_bytes), 845140)
        self.assertEqual(
            _source_projection_hash(self.notebook),
            SOURCE_PROJECTION_SHA256,
        )
        self.assertEqual(
            set(self.notebook.metadata),
            {"kernelspec", "language_info"},
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
        self.assertEqual(
            tuple(cell.execution_count for cell in self.code_cells),
            tuple(range(1, 25)),
        )
        observed_output_hashes = {}
        for cell in self.notebook.cells:
            self.assertNotIn("attachments", cell)
            self.assertEqual(cell.metadata, {})
            if cell.cell_type == "code":
                if cell.id not in OUTPUT_HASHES:
                    self.assertEqual(cell.outputs, [])
                    continue
                self.assertEqual(len(cell.outputs), 1)
                output = cell.outputs[0]
                self.assertEqual(output.output_type, "display_data")
                self.assertEqual(
                    set(output.data),
                    {"image/png", "text/plain"},
                )
                self.assertEqual(output.data["text/plain"], FIGURE_TEXT)
                self.assertEqual(output.metadata, {})
                observed_output_hashes[cell.id] = _decoded_png_hash(output)
        self.assertEqual(observed_output_hashes, OUTPUT_HASHES)
        test_module = ast.parse(Path(__file__).read_text())
        exact_hash_assertions = tuple(
            node
            for node in ast.walk(test_module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "assertEqual"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "observed_output_hashes"
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == "OUTPUT_HASHES"
        )
        self.assertEqual(len(exact_hash_assertions), 1)

        cleared = self._cleared_copy()
        cleared_code_cells = tuple(
            cell for cell in cleared.cells if cell.cell_type == "code"
        )
        self.assertTrue(
            all(
                cell.execution_count is None and cell.outputs == []
                for cell in cleared_code_cells
            )
        )

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
            "One `Photoelectrons` Product holds both independent examples "
            "and all three sensors in one tensor",
            markdown_source,
        )
        self.assertIn("independent waveform realizations", markdown_source)
        self.assertIn("not adjacent windows or sequential state", markdown_source)
        self.assertIn(
            "illustrative rather than calibrated",
            markdown_source,
        )
        level_two_headings = tuple(
            line
            for cell in self.markdown_cells
            for line in cell.source.splitlines()
            if line.startswith("## ")
        )
        self.assertEqual(
            level_two_headings,
            (
                "## Photoelectrons",
                "## Charge",
                "## PureWaveform",
                "## NoiseWaveform",
                "## AnalogWaveform",
                "## DigitizedWaveform",
                "## EncodedWaveform",
            ),
        )
        all_cell_ids = tuple(cell.id for cell in self.notebook.cells)
        charge_boundary = (
            "charge-explanation",
            "delayed-crosstalk-values-code",
            "charge-semantics-explanation",
            "charge-code",
        )
        charge_start = all_cell_ids.index("charge-explanation")
        self.assertEqual(
            all_cell_ids[charge_start : charge_start + 4],
            charge_boundary,
        )
        self.assertNotIn(
            "delayed-crosstalk-values-explanation",
            all_cell_ids,
        )

    def test_exact_product_axis_psd_and_shape_source(self) -> None:
        self.assertIn(
            'labels=("sensor-0", "sensor-1", "sensor-2")',
            self.code_source,
        )
        self.assertIn("coordinates=CountCoordinates(count=2)", self.code_source)
        self.assertIn("count=5000", self.code_source)
        self.assertIn("coordinate_scale=2.0", self.code_source)
        self.assertIn("count=2501", self.code_source)
        self.assertIn("coordinate_scale=0.1", self.code_source)
        for source_line in (
            "photoelectron_values[0, 0, 100] = 1",
            "photoelectron_values[0, 0, 3700] = 4",
            "photoelectron_values[0, 1, 1300] = 2",
            "photoelectron_values[0, 2, 2500] = 3",
            "photoelectron_values[1, 0, 800] = 2",
            "photoelectron_values[1, 1, 2000] = 4",
            "photoelectron_values[1, 1, 3500] = 1",
            "photoelectron_values[1, 2, 2900] = 3",
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
            "delayed_offsets = tuple(range(1, 251))",
            "torch.arange(\n        1,\n        251,",
            "* time_axis.coordinate_scale",
            "-delayed_times_ns / 150.0",
            "0.15 / delayed_crosstalk_values.sum()",
            "conditioning_axes=()",
            "operation_axes=(delayed_time_axis,)",
            "correlated_avalanche_generations=NonnegativeInteger(value=2)",
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
        self.assertNotIn("conditioning_axes=(channel_axis,)", self.code_source)
        self.assertIn(
            "operation_axes=(frequency_axis,)",
            self.code_source,
        )
        self.assertIn("conditioning_axes=()", self.code_source)
        self.assertNotIn("psd_sensor_0", self.code_source)
        self.assertNotIn("psd_sensor_1", self.code_source)
        self.assertNotIn("psd_sensor_2", self.code_source)
        self.assertNotIn("torch.stack", self.code_source)
        self.assertNotIn("torch.rand", self.code_source)
        self.assertIn(
            "noisy baseline regions are suppressed",
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

        delayed_values_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "delayed-crosstalk-values-code"
        )
        charge_source = next(
            cell.source
            for cell in self.code_cells
            if cell.id == "charge-code"
        )
        for calculation in (
            "delayed_offsets = tuple(range(1, 251))",
            "delayed_times_ns = ",
            "torch.arange(",
            "torch.exp(",
            "0.15 / delayed_crosstalk_values.sum()",
        ):
            self.assertIn(calculation, delayed_values_source)
            self.assertNotIn(calculation, charge_source)
        for semantic_name in (
            "OffsetAxis",
            "DelayedCrosstalkSpec",
            "DelayedCrosstalk(",
            "ChargeConfig",
            "ChargeKernels",
            "ChargeSpec",
            "Charge.create",
        ):
            self.assertNotIn(semantic_name, delayed_values_source)
            self.assertIn(semantic_name, charge_source)
        reconstructed_delayed = torch.exp(
            -torch.arange(1, 251, dtype=torch.float64) * 2.0 / 150.0
        )
        reconstructed_delayed *= 0.15 / reconstructed_delayed.sum()
        self.assertEqual(reconstructed_delayed.shape, (250,))
        self.assertEqual(reconstructed_delayed.dtype, torch.float64)
        self.assertEqual(
            tuple(range(1, 251)),
            tuple(range(1, reconstructed_delayed.numel() + 1)),
        )
        self.assertEqual(
            float(reconstructed_delayed.sum()),
            0.14999999999999997,
        )
        self.assertEqual(
            float(reconstructed_delayed[0]),
            0.0020602220776112065,
        )
        self.assertEqual(
            float(reconstructed_delayed[-1]),
            7.448286214784992e-05,
        )
        self.assertEqual(
            hashlib.sha256(
                reconstructed_delayed.numpy().tobytes()
            ).hexdigest(),
            "a49cd5c4c14d606904a321aa48ba7a8d861f3f37ea03ecdbc0d4492d4d780b70",
        )

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
        for calculation in ("torch.cat(", "torch.tensor(", "torch.zeros("):
            self.assertIn(calculation, psd_values_source)
            self.assertNotIn(calculation, noise_waveform_source)
        self.assertNotIn("torch.full(", psd_values_source)
        self.assertNotIn("torch.stack(", psd_values_source)
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
        psd_tree = ast.parse(psd_values_source)
        psd_tensor_calls = tuple(
            node
            for node in ast.walk(psd_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "tensor"
        )
        self.assertEqual(len(psd_tensor_calls), 1)
        psd_prefix = ast.literal_eval(psd_tensor_calls[0].args[0])
        self.assertIsInstance(psd_prefix, tuple)
        assert isinstance(psd_prefix, tuple)
        self.assertEqual(len(psd_prefix), 626)
        reconstructed_psd = torch.cat(
            (
                torch.tensor(psd_prefix, dtype=torch.float32),
                torch.zeros(1875, dtype=torch.float32),
            )
        )
        self.assertEqual(reconstructed_psd.shape, (2501,))
        self.assertEqual(reconstructed_psd.dtype, torch.float32)
        self.assertEqual(int(torch.count_nonzero(reconstructed_psd)), 625)
        self.assertEqual(float(reconstructed_psd[0]), 0.0)
        self.assertTrue(bool((reconstructed_psd[1:626] > 0).all()))
        self.assertTrue(bool((reconstructed_psd[626:] == 0).all()))
        self.assertEqual(
            sum(float(value) for value in reconstructed_psd),
            11.251378314314934,
        )
        self.assertEqual(
            hashlib.sha256(reconstructed_psd.numpy().tobytes()).hexdigest(),
            "928921a257e50cbe1720358a8b929249e5bff9383cb3c63be1a67769c4c09a47",
        )

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
            "float(time_axis.quantity_at(index).to(\"ns\").magnitude)",
            "def plot_product(",
            "product.tensor[example_index, channel_index]",
            "plt.subplots(\n        3,\n        2,",
            "figsize=(13, 8.5)",
            "sharex=True",
            "squeeze=False",
            "plot_axis.step(",
            'where="post"',
            "plot_axis.plot(",
            "alpha=0.72",
            "linewidth=0.9",
            'plot_axis.set_title(f"Example {example_index}")',
            'plot_axis.set_xlabel("Time (ns)")',
            "plot_axis.set_ylabel(channel_label)",
            "figure.suptitle(title)",
            "left=0.08",
            "right=0.98",
            "bottom=0.08",
            "top=0.91",
            "hspace=0.18",
            "wspace=0.12",
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
            ".legend(",
            "sensor_colors",
            "y_label",
            "plot-preparation-code",
            "product-views-code",
            "combined",
            "summary figure",
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
            ("title", "color", "step", "suppression_code"),
        )

        view_contracts = (
            (
                "photoelectrons-code",
                "photoelectrons-view-code",
                "photoelectrons",
                "Photoelectrons",
                "tab:blue",
                True,
                None,
            ),
            (
                "charge-code",
                "charge-view-code",
                "charge",
                "Charge (avalanche)",
                "tab:orange",
                True,
                None,
            ),
            (
                "pure-waveform-code",
                "pure-waveform-view-code",
                "pure_waveform",
                "Pure (mV)",
                "tab:green",
                False,
                None,
            ),
            (
                "noise-waveform-code",
                "noise-waveform-view-code",
                "noise_waveform",
                "Noise (mV)",
                "tab:red",
                False,
                None,
            ),
            (
                "analog-waveform-code",
                "analog-waveform-view-code",
                "analog_waveform",
                "Analog (mV)",
                "tab:purple",
                False,
                None,
            ),
            (
                "digitized-waveform-code",
                "digitized-waveform-view-code",
                "digitized_waveform",
                "ADC code",
                "tab:brown",
                True,
                None,
            ),
            (
                "encoded-waveform-code",
                "encoded-waveform-view-code",
                "encoded_waveform",
                "Retained ADC code",
                "tab:pink",
                True,
                "encoded_waveform.spec.suppression_code",
            ),
        )
        all_cell_ids = tuple(cell.id for cell in self.notebook.cells)
        for (
            construction_id,
            view_id,
            product_name,
            title,
            color,
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
                    {"title", "color", "step"}
                    | ({"suppression_code"} if suppression else set()),
                )
                title_node = keywords["title"]
                self.assertIsInstance(title_node, ast.Constant)
                assert isinstance(title_node, ast.Constant)
                self.assertEqual(title_node.value, title)
                color_node = keywords["color"]
                self.assertIsInstance(color_node, ast.Constant)
                assert isinstance(color_node, ast.Constant)
                self.assertEqual(color_node.value, color)
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
        self.assertEqual(NOTEBOOK_PATH.read_bytes(), self.notebook_bytes)

        for executed in (first_notebook, second_notebook):
            original_cells = executed.cells[:-1]
            self.assertEqual(len(original_cells), len(self.notebook.cells))
            for committed_cell, fresh_cell in zip(
                self.notebook.cells,
                original_cells,
                strict=True,
            ):
                self.assertEqual(fresh_cell.cell_type, committed_cell.cell_type)
                self.assertEqual(fresh_cell.id, committed_cell.id)
                self.assertEqual(fresh_cell.source, committed_cell.source)
                self.assertEqual(fresh_cell.metadata, {})
                if fresh_cell.cell_type == "code":
                    self.assertEqual(
                        fresh_cell.execution_count,
                        committed_cell.execution_count,
                    )
                    self.assertEqual(
                        fresh_cell.outputs,
                        committed_cell.outputs,
                    )

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
        self.assertEqual(first["shapes"], [[2, 3, 5000]] * 7)
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
        expected_titles = [
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
                "d1d3b5339e0fcb4c3c0e03253b22f8dc221aa4e97bc681afe46d5bfa5be89be7",
                "f2bdd4298c2f1d55b94bd4a348b4ebda6b2a1eb6a318113f50d79baa646aee84",
                "ab16cea7c2b3a6b8e8845270b265e27dfa0f555ad093cabc9dfff7fffb5773b1",
                "75a2ebfd917d1fd7ceed29119f00c11df9eb3cbda0cf2c6d64bd3b54451c5e9d",
                "0ed3238c8138eb0ab31e0ac44866c3cd48382c9d1bb5b3e8a99508d2adb7e46b",
                "ed05e0fe82dbfa86239a6dd08da0fbf5dfd6ef180845f81e5a3e4b1070e4f7fc",
                "edb6703f748fa19aeebd8222c8be84a588a43ab716316a543b714b6a70a7aab0",
            ],
        )
        expected_example_digests = [
            [
                "8cd46f0f4ff164b63fc0068e22d7705cab82adbee84c804849925f436d247fc7",
                "df666d1bba4f9d1c0d09805352fd1936375a6ec274abd7c923dd123eac817542",
            ],
            [
                "ee009c5b1ba7033257ec762d569e97cd197482af3eb8d53ddbf9fbabea89318a",
                "31f86a483f59402737b2264644608062f3f2a0f826afad9e5544fc793f39616b",
            ],
            [
                "4bf076c249d208d705035c5cee8ea53bfafb129cb3138c8034c1a179329a3987",
                "8819e225bea2a0d0ecdb237061b1bf3645b0dbe7a978220168f23fc9b74f29fd",
            ],
            [
                "f3cffacc384d011ea81a2237ce41bf3c1e1e595e2bb529f3a1a171fa07c0cc95",
                "9ba350ae85a00d72cd88acdfe541377bc354fa5d61072c9927b2656daf32d269",
            ],
            [
                "914021735ec507834b73db6c26f657e319aed363aeb71ba08c43088f3415cd47",
                "357f80bdcfe8c526ca992fd2892da2f8648c1331fae221282b1a296c4deca368",
            ],
            [
                "5f59bff515434ec03c9bc989aacf29bd67590a3369b26503660bbcc3646b858f",
                "ac486a404b38f9bfa1b307762c403cb30d45b3c1aca4b3182159fe28d28e0062",
            ],
            [
                "8685ecfced88b781d855f3780aee57097411ca6883d36cd846cddb798d292089",
                "2128bfafe3584226b715ca21bf963eb69166540b9e1724acd281baad6a83543a",
            ],
        ]
        self.assertEqual(first["example_digests"], expected_example_digests)
        self.assertTrue(
            all(
                example_0 != example_1
                for example_0, example_1 in first["example_digests"]
            )
        )
        self.assertTrue(first["source_unchanged"])
        self.assertEqual(
            first["channel_labels"],
            ["sensor-0", "sensor-1", "sensor-2"],
        )
        self.assertEqual(
            first["deposits"],
            [
                [0, 0, 100, 1],
                [0, 0, 3700, 4],
                [0, 1, 1300, 2],
                [0, 2, 2500, 3],
                [1, 0, 800, 2],
                [1, 1, 2000, 4],
                [1, 1, 3500, 1],
                [1, 2, 2900, 3],
            ],
        )
        self.assertEqual(first["time_size"], 5000)
        self.assertEqual(first["time_scale"], 2.0)
        self.assertEqual(first["frequency_size"], 2501)
        self.assertEqual(first["frequency_scale"], 0.1)
        self.assertEqual(first["delayed_offsets"], list(range(1, 251)))
        self.assertEqual(first["delayed_shape"], [250])
        self.assertEqual(first["delayed_dtype"], "torch.float64")
        self.assertEqual(first["delayed_unit"], "dimensionless")
        self.assertEqual(first["delayed_sum"], 0.14999999999999997)
        self.assertEqual(first["delayed_first"], 0.0020602220776112065)
        self.assertEqual(first["delayed_last"], 7.448286214784992e-05)
        self.assertEqual(
            first["delayed_digest"],
            "a49cd5c4c14d606904a321aa48ba7a8d861f3f37ea03ecdbc0d4492d4d780b70",
        )
        self.assertEqual(first["delayed_conditioning"], [])
        self.assertEqual(first["delayed_operation"], ["OffsetAxis"])
        self.assertEqual(first["delayed_relative_to"], "TimeAxis")
        self.assertEqual(first["generations"], 2)
        self.assertEqual(
            first["descendant_positions"],
            [[0, 0, 3736], [0, 2, 2548]],
        )
        self.assertEqual(first["descendant_count"], 2)
        self.assertEqual(first["descendant_total"], 2.0)
        self.assertEqual(
            first["earliest_source"],
            {
                "0:0": 100,
                "0:1": 1300,
                "0:2": 2500,
                "1:0": 800,
                "1:1": 2000,
                "1:2": 2900,
            },
        )
        for example, channel, sample in first["descendant_positions"]:
            self.assertGreater(
                sample,
                first["earliest_source"][f"{example}:{channel}"],
            )
            self.assertLess(sample, first["time_size"])
        self.assertTrue(first["charge_integer"])
        self.assertEqual(first["pulse_support_ns"], 2020.27)
        self.assertEqual(first["pulse_coefficient_count"], 1011)
        self.assertEqual(first["pulse_offset_count"], 1011)
        self.assertAlmostEqual(first["pulse_minimum"], -14.5912372, places=5)
        self.assertTrue(first["pulse_finite"])
        self.assertEqual(first["psd_shape"], [2501])
        self.assertEqual(first["psd_dtype"], "torch.float32")
        self.assertEqual(first["psd_device"], "cpu")
        self.assertEqual(first["psd_unit"], "millivolt ** 2")
        self.assertEqual(
            first["psd_digest"],
            "928921a257e50cbe1720358a8b929249e5bff9383cb3c63be1a67769c4c09a47",
        )
        self.assertEqual(first["psd_sum"], 11.251378314314934)
        self.assertEqual(first["psd_nonzero"], 625)
        self.assertEqual(first["psd_zero"], 1876)
        self.assertEqual(first["psd_dc"], 0.0)
        self.assertTrue(first["psd_positive_prefix"])
        self.assertTrue(first["psd_zero_suffix"])
        self.assertEqual(first["psd_conditioning"], [])
        self.assertEqual(first["psd_operation"], ["FrequencyAxis"])
        self.assertEqual(
            first["noise_lane_digests"],
            [
                "4aea755e7c8c309e8afac1f9e731cfab9c8310d78850d4150588033ce5f61763",
                "cec5be6d2b2fa2eb797667cc12c4de62b4bf86039bb9b7c12dd996f377cf80cf",
                "b5171e9fdac933624bffacc51c59a915e6868c1204841107202105b21970f702",
                "8e131d07b8d5b5b0ac16c29b301ae0851eb49bb778891a317c43ff8900e248b7",
                "a13129ebcf73d93b744079d95cf97969787f340a529eb6ef75a836061bf79391",
                "d20164cd6371dd45ec1b3217b1e5a93794aba3c93ea1cb1dea9626c192a65d0b",
            ],
        )
        self.assertEqual(len(set(first["noise_lane_digests"])), 6)
        self.assertEqual(first["input_minimum"], -80.0)
        self.assertEqual(first["input_maximum"], 20.0)
        self.assertEqual(first["bit_depth"], 12)
        self.assertEqual(first["analog_gain"], 1.0)
        self.assertEqual(first["zle_policy"], [2500, 2800, 3, 25, 50])
        self.assertEqual(first["suppression_code"], -1)
        self.assertEqual(
            first["encoded_support"],
            [
                [
                    [[3724, 4371]],
                    [[1349, 1817]],
                    [[2530, 3147]],
                ],
                [
                    [[863, 1368]],
                    [[2027, 2601], [3640, 3855]],
                    [[2967, 3503]],
                ],
            ],
        )
        self.assertEqual(first["encoded_plot_gaps"], [[True] * 6])

        self.assertEqual(first["figure_count"], 7)
        self.assertEqual(first["figure_axes"], [6] * 7)
        self.assertEqual(first["line_counts"], [[1] * 6] * 7)
        expected_colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
        ]
        self.assertEqual(
            first["colors"],
            [[[color]] * 6 for color in expected_colors],
        )
        self.assertEqual(len(set(expected_colors)), 7)
        self.assertEqual(
            first["drawstyles"],
            [
                [["steps-post"]] * 6,
                [["steps-post"]] * 6,
                [["default"]] * 6,
                [["default"]] * 6,
                [["default"]] * 6,
                [["steps-post"]] * 6,
                [["steps-post"]] * 6,
            ],
        )
        self.assertEqual(first["alphas"], [[[0.72]] * 6] * 7)
        self.assertEqual(first["linewidths"], [[[0.9]] * 6] * 7)
        self.assertEqual(first["axis_legends"], [[False] * 6] * 7)
        self.assertEqual(first["figure_legends"], [0] * 7)
        self.assertEqual(
            first["figure_titles"],
            expected_titles,
        )
        self.assertEqual(
            first["axis_titles"],
            [["Example 0", "Example 1", "", "", "", ""]] * 7,
        )
        self.assertEqual(
            first["xlabels"],
            [["", "", "", "", "Time (ns)", "Time (ns)"]] * 7,
        )
        self.assertEqual(
            first["ylabels"],
            [["sensor-0", "", "sensor-1", "", "sensor-2", ""]] * 7,
        )
        for layout in first["layout_bounds"]:
            _, _, figure_width, figure_height = layout["figure"]
            self.assertEqual(len(layout["axes"]), 6)
            self.assertEqual(len(layout["text"]), 8)
            for x, y, width, height in (
                layout["axes"] + layout["text"]
            ):
                self.assertGreaterEqual(x, 0.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(x + width, figure_width)
                self.assertLessEqual(y + height, figure_height)
        self.assertEqual(
            tuple(zip(first["units"], first["figure_titles"])),
            tuple(zip(expected_units, expected_titles)),
        )
        self.assertEqual(first["adc_min"], 341)
        self.assertEqual(first["adc_max"], 3692)
        self.assertEqual(
            first["adc_example_extrema"],
            [[341, 3692], [896, 3656]],
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
            self.assertEqual(
                [_decoded_png_hash(display) for display in displays],
                list(OUTPUT_HASHES.values()),
            )
            for display in displays:
                self.assertEqual(display.output_type, "display_data")
                self.assertEqual(
                    set(display.data),
                    {"image/png", "text/plain"},
                )
                self.assertEqual(display.data["text/plain"], FIGURE_TEXT)
                self.assertEqual(display.metadata, {})


if __name__ == "__main__":
    unittest.main()
