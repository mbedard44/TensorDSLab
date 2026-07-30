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


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "demos" / "readout.ipynb"
SUMMARY_PREFIX = "READOUT_DEMO_SUMMARY="
SOURCE_PROJECTION_SHA256 = (
    "921b5f9fee0e3260867689642cb35224f2b1facd9623d42d3404a3887ee47fbf"
)
COMMITTED_NOTEBOOK_SHA256 = (
    "5f423a6c90d093e10af5d2f8e7f5dcfe1070cf369d195b28b27f2d58c66be57b"
)
FIGURE_TEXT = "<Figure size 1300x850 with 6 Axes>"
OUTPUT_HASHES = {
    "photoelectrons-view-code": (
        "3f5814fa5c83b856ecdb5a15def832251cd0a5cff91d3dfbda21bd44bceb9139"
    ),
    "charge-view-code": (
        "680c6231d009a9d6c590ee060bd5ddd1d4aad79950411e677a1e3da53cc2854c"
    ),
    "pure-waveform-view-code": (
        "924a95d4c97e00d3cc6fde320e03b548cf323d24c0ddc8e7285560fa35809a1f"
    ),
    "noise-waveform-view-code": (
        "69bf7451031461ad9a3baad9c70f6b53b7f733a1df3a7c5213e0849bae64e08b"
    ),
    "analog-waveform-view-code": (
        "74f5b46839ad9ee3263caba3fac21705fb1e3606a2e25e0d688cb1c655b040c3"
    ),
    "digitized-waveform-view-code": (
        "fdd07c0a2a187aebe7b10239f297efafa96ba9ced98377278c626dc5d12479f0"
    ),
    "encoded-waveform-view-code": (
        "dc7df1661bd9e5fca09a886ebae8d84f56b07c010f62066890a7a98f64e52e0d"
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
    "photo_charge_equal": bool(torch.equal(
        photoelectrons.tensor.to(charge.tensor.dtype),
        charge.tensor,
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
            hashlib.sha256(self.notebook_bytes).hexdigest(),
            COMMITTED_NOTEBOOK_SHA256,
        )
        self.assertEqual(len(self.notebook_bytes), 876216)
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
            tuple(range(1, 24)),
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
            "One Product call processes both independent examples "
            "and all three sensors together",
            markdown_source,
        )
        self.assertIn("independent waveform realizations", markdown_source)
        self.assertIn("not adjacent windows or sequential state", markdown_source)
        self.assertIn(
            "illustrative rather than calibrated",
            markdown_source,
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
                "18f7bcafed30c6aae5507efe928e899a16e7a0d5c5fe070116f1026feb74aceb",
                "177d956b3c0a0119548a1ad47cd92904f0d53400c4536373d67b97da8896d865",
                "1f493e35435eb2d72ff01334a6c0d0028766813d9ed8eabe6feebbd673314844",
                "94646193ffdf7ef6dd7afd0822d9850a396e80aa0b6c9a3590da3302cb143524",
                "c4e2a4210cef98ccfcc4e20090e5b456b05db2bd276ab3b0c75ddbaa2d9f02cf",
                "8b25c2ac35d68db9b183d7ab0b82427167527e0ff4137a1a6d727af062c316f8",
            ],
        )
        expected_example_digests = [
            [
                "8cd46f0f4ff164b63fc0068e22d7705cab82adbee84c804849925f436d247fc7",
                "df666d1bba4f9d1c0d09805352fd1936375a6ec274abd7c923dd123eac817542",
            ],
            [
                "66113e0e8952d8e3906217ca6efee0ade16884ff2a2ec095192faf0b72dda5b4",
                "31f86a483f59402737b2264644608062f3f2a0f826afad9e5544fc793f39616b",
            ],
            [
                "7233c98dd4237a425703855271ecd85fa4a78440ffbf34f1b6fb8b3593481f45",
                "8819e225bea2a0d0ecdb237061b1bf3645b0dbe7a978220168f23fc9b74f29fd",
            ],
            [
                "0f8798c98ec544dcbcfd57579d0f6ee9c1af2e24e10caf6832919f438be51208",
                "c3feda4109ef0c6cd99dd9a62fbf5d5e793e8048c179955ac2d4c14e393ec0e0",
            ],
            [
                "e625e37451c37424d345e470ca6426543e56a19dd2bfdf570ad5c3660bad39ad",
                "270d00c4f683027354b06bd7174232804f317bedeeeed7a7de48c06e9805b168",
            ],
            [
                "4bc693b1c8870cecc1827afcf6da88883908b5bc202250ba99c9beaa0d1d4780",
                "a7c1373e511a624054d66069ba639abd609ab30d0c9e20e4c9ca1d69bdf4f693",
            ],
            [
                "00644cce41594031fa44018ed27362c60234154e3dddafb6b6b517c87ca8335d",
                "52256c0c105e70cb2aa2f06d7ddc332325ce5ab3d641823292bd188dcee378a4",
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
        self.assertTrue(first["photo_charge_equal"])
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
                [
                    [[280, 362], [3721, 4261]],
                    [[1363, 1749]],
                    [[2541, 3085]],
                ],
                [
                    [[873, 1267]],
                    [[2030, 2570]],
                    [[2924, 3425]],
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
