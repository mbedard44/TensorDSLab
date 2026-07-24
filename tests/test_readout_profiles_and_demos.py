import ast
from dataclasses import fields, is_dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nbclient import NotebookClient
import nbformat
import numpy as np
from pint import Quantity
import torch
from tensor_core import NonnegativeFloat, PositiveInteger, Threefry4x32

import tensor_dslab
import tensor_dslab.readout as readout
from tensor_dslab import (
    AnalogWaveform,
    AnalogWaveformConfig,
    ChannelAxis,
    Charge,
    ChargeConfig,
    DarkCountConfig,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    PureWaveform,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    VetoPduPulseConfig,
    quantities,
    quantity,
    simulate_readout,
)
from tensor_dslab.common.units import _REGISTRY
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    PsdNoiseRuntime,
)
from tensor_dslab.readout.profiles import ds20k_veto
from tensor_dslab.readout.pure_waveform.runtime.prepare import (
    prepare_pure_waveform,
)
from tensor_dslab.readout.runtime.prepare import prepare_readout
from tensor_dslab.readout.runtime.sampling import SamplingRuntime


_PRODUCTS = (
    Photoelectrons,
    Charge,
    PureWaveform,
    NoiseWaveform,
    AnalogWaveform,
    DigitizedWaveform,
)
_FREQUENCY_LEFT_EDGES_MHZ = (
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    40.0,
    62.5,
)
_POWER_DENSITY_MV2_PER_HZ = (
    4.0e-8,
    7.0e-8,
    6.0e-8,
    3.0e-8,
    7.0e-9,
    1.0e-9,
    2.0e-10,
    5.0e-11,
    0.0,
)


def _literal_config() -> ReadoutConfig:
    return ReadoutConfig(
        charge=ChargeConfig(
            dark_count=DarkCountConfig(rate=quantity(100.0, "kHz")),
        ),
        pure_waveform=PureWaveformConfig(
            model=VetoPduPulseConfig(
                gaussian_center=quantity(232.89, "ns"),
                gaussian_width=quantity(507.72, "ns"),
                edge_offset_1=quantity(-81.92, "ns"),
                edge_width_1=quantity(147.28, "ns"),
                edge_offset_2=quantity(-176.50, "ns"),
                edge_width_2=quantity(45.69, "ns"),
                support_time=quantity(2020.27, "ns"),
                peak_voltage_per_photoelectron=quantity(14.5912372, "mV"),
            )
        ),
        noise_waveform=NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges=quantities(
                    _FREQUENCY_LEFT_EDGES_MHZ,
                    "MHz",
                ),
                frequency_stop=quantity(250.0, "MHz"),
                power_density=quantities(
                    _POWER_DENSITY_MV2_PER_HZ,
                    "mV ** 2 / Hz",
                ),
            )
        ),
        analog_waveform=AnalogWaveformConfig(),
        digitized_waveform=DigitizedWaveformConfig(
            bit_depth=PositiveInteger(16),
            input_minimum=quantity(-3900.0, "mV"),
            input_maximum=quantity(100.0, "mV"),
            analog_gain_db=NonnegativeFloat(3.5218),
        ),
    )


def _source() -> Photoelectrons:
    axes = (
        ExampleAxis(count=2),
        ChannelAxis(labels=("veto-0", "veto-1", "veto-2", "veto-3")),
        SampleAxis.from_period(period=quantity(2.0, "ns"), count=5000),
    )
    shape = tuple(axis.size for axis in axes)
    counts = torch.zeros(shape, dtype=torch.int64, device="cpu")
    counts[0, 0, 100] = 1
    counts[0, 0, 1300] = 2
    counts[0, 0, 2500] = 3
    counts[0, 0, 3700] = 4
    return Photoelectrons(tensor=counts, axes=axes)


def _quantity_signature(value: Quantity) -> tuple[str, tuple[float, ...]]:
    magnitude = value.magnitude
    if isinstance(magnitude, np.ndarray):
        values = tuple(float(item) for item in magnitude)
    else:
        values = (float(magnitude),)
    return str(value.units), values


def _config_signature(config: ReadoutConfig) -> tuple[object, ...]:
    assert config.charge is not None
    assert config.charge.dark_count is not None
    assert config.pure_waveform is not None
    assert type(config.pure_waveform.model) is VetoPduPulseConfig
    assert config.noise_waveform is not None
    assert type(config.noise_waveform.model) is PsdNoiseConfig
    assert config.analog_waveform is not None
    assert config.digitized_waveform is not None
    pulse = config.pure_waveform.model
    psd = config.noise_waveform.model
    adc = config.digitized_waveform
    return (
        type(config),
        type(config.charge),
        _quantity_signature(config.charge.dark_count.rate),
        config.charge.timing_jitter,
        config.charge.correlated_avalanches,
        config.charge.smearing,
        type(config.pure_waveform),
        type(pulse),
        *(
            _quantity_signature(getattr(pulse, name))
            for name in (
                "gaussian_center",
                "gaussian_width",
                "edge_offset_1",
                "edge_width_1",
                "edge_offset_2",
                "edge_width_2",
                "support_time",
                "peak_voltage_per_photoelectron",
            )
        ),
        type(config.noise_waveform),
        type(psd),
        _quantity_signature(psd.frequency_left_edges),
        _quantity_signature(psd.frequency_stop),
        _quantity_signature(psd.power_density),
        type(config.analog_waveform),
        config.analog_waveform.saturation,
        type(adc),
        adc.bit_depth.value,
        _quantity_signature(adc.input_minimum),
        _quantity_signature(adc.input_maximum),
        adc.analog_gain_db.value,
    )


def _runtime_signature(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            value.dtype,
            value.device,
            tuple(value.shape),
            tuple(value.reshape(-1).tolist()),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (field.name, _runtime_signature(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    return value


def _config_nodes(config: ReadoutConfig) -> dict[tuple[str, ...], object]:
    nodes: dict[tuple[str, ...], object] = {}

    def visit(value: object, path: tuple[str, ...]) -> None:
        value_type = type(value)
        if not (
            is_dataclass(value)
            and not isinstance(value, type)
            and value_type.__module__.startswith("tensor_dslab.readout")
            and value_type.__name__.endswith("Config")
        ):
            return
        nodes[path] = value
        for field in fields(value):
            visit(getattr(value, field.name), (*path, field.name))

    visit(config, ())
    return nodes


def _source_construction_ast(
    statements: list[ast.stmt],
) -> tuple[str, ...]:
    selected: list[ast.stmt] = []
    assigned_names = {"axes", "shape", "counts", "photoelectrons"}
    for statement in statements:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in assigned_names
        ):
            selected.append(statement)
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Subscript)
            and isinstance(statement.targets[0].value, ast.Name)
            and statement.targets[0].value.id == "counts"
        ):
            selected.append(statement)
    return tuple(
        ast.dump(statement, include_attributes=False)
        for statement in selected
    )


_EXPECTED_SOURCE_CONSTRUCTION_AST = _source_construction_ast(
    ast.parse(
        """
axes = (
    ExampleAxis(count=2),
    ChannelAxis(labels=("veto-0", "veto-1", "veto-2", "veto-3")),
    SampleAxis.from_period(period=quantity(2.0, "ns"), count=5000),
)
shape = tuple(axis.size for axis in axes)
counts = torch.zeros(shape, dtype=torch.int64, device="cpu")
counts[0, 0, 100] = 1
counts[0, 0, 1300] = 2
counts[0, 0, 2500] = 3
counts[0, 0, 3700] = 4
photoelectrons = Photoelectrons(tensor=counts, axes=axes)
"""
    ).body
)
_EXPECTED_NOTEBOOK_CODE_SHA256 = (
    "456656c129e68863bd7158a11824e1cd8c44607a2f7dc969b393fb0ce6b53ac0"
)
_EXPECTED_NOTEBOOK_SHA256 = (
    "f2ac47c6914c579d4a5559d31eef0091904cee3f0814d9bae1be9536f00f66dd"
)


class ReadoutProfilesAndDemosTest(unittest.TestCase):
    def test_live_api_and_parity_records_classify_the_profile_exactly(self) -> None:
        api = Path("docs/api.md").read_text()
        parity = Path("docs/parity.md").read_text()
        overview = Path("docs/overview.md").read_text()
        normalized_api = " ".join(api.split())
        normalized_parity = " ".join(parity.split())

        for required in (
            "from tensor_dslab.readout.profiles import ds20k_veto",
            "provisional demonstration profile, not an approved run calibration",
            "does not select source axes, channel identities, a sample grid",
            "not re-exported from `tensor_dslab` or `tensor_dslab.readout`",
            "parity.md#maintenance-9-provisional-ds20k-veto-profile",
        ):
            self.assertIn(required, normalized_api)
        self.assertIn("[Public API](api.md)", overview)

        for value in (
            "232.89 ns",
            "507.72 ns",
            "-81.92 ns",
            "147.28 ns",
            "-176.50 ns",
            "45.69 ns",
            "2020.27 ns",
            "14.5912372 mV",
            "100 kHz",
            "[0, 250] MHz",
            "62.5 MHz",
            "0.5 mV",
            "16",
            "[-3900, 100] mV",
            "3.5218 dB",
            "2 ns",
            "5000",
            "10000 ns",
            "200 ns",
            "2600 ns",
            "5000 ns",
            "7400 ns",
            "veto-0",
            "veto-3",
        ):
            self.assertIn(f"`{value}`", parity)
        for required in (
            "## Maintenance 9 Provisional DS20k Veto Profile",
            "provisional demonstration profile, not an approved run calibration",
            "existing Veto **numerical parity** boundary",
            "**Deferred** calibration choice used illustratively",
            "**Deferred** IV-DSLab-like demonstration shape",
            "**Not applicable** to detector calibration",
            "**Not applicable** to donor parity",
            "not by `ds20k_veto()`",
            "Later promotion or replacement of a provisional value requires",
            "Maintenance 10 refines only the provisional digitizer",
        ):
            self.assertIn(required, normalized_parity)

    def test_environment_script_contract_and_fake_execution(self) -> None:
        script = Path("create_environment.sh").resolve()
        self.assertFalse(Path("demos/create_environment.sh").exists())
        self.assertEqual(script.stat().st_mode & 0o777, 0o755)
        self.assertTrue(script.stat().st_mode & 0o111)
        source = script.read_text()
        self.assertTrue(source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        for required in (
            '${CONDA_EXE:-conda}',
            "tensor_dslab",
            "--no-default-packages",
            "--override-channels",
            "--channel conda-forge",
            '"python=3.14.6"',
            '"${repository_root}[demos]"',
            'repository_root="${script_directory}"',
            "--disable-pip-version-check",
            "--no-input",
            "conda activate",
            "python demos/readout.py",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "--user",
            "conda init",
            "env remove",
            "conda config",
            "jupyter kernelspec",
            "ipykernel install",
            "simulate_readout",
            "torch.cuda",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn('"${script_directory}/.."', source)
        self.assertNotIn(str(Path.cwd()), source)

        syntax = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_conda = temporary / "conda"
            log = temporary / "conda.log"
            fake_conda.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"${FAKE_CONDA_LOG}\"\n"
                "if [[ \"${1:-}\" == env && \"${2:-}\" == list ]]; then\n"
                "    echo '# conda environments:'\n"
                "    if [[ -n \"${FAKE_EXISTING_ENV:-}\" ]]; then\n"
                "        printf '%s  /fake/envs/%s\\n' "
                "\"${FAKE_EXISTING_ENV}\" \"${FAKE_EXISTING_ENV}\"\n"
                "    fi\n"
                "    exit 0\n"
                "fi\n"
                "if [[ \"${1:-}\" == create || \"${1:-}\" == run ]]; then\n"
                "    exit 0\n"
                "fi\n"
                "exit 91\n"
            )
            fake_conda.chmod(0o755)
            environment = {
                **os.environ,
                "CONDA_EXE": str(fake_conda),
                "FAKE_CONDA_LOG": str(log),
            }

            for arguments, expected_name in (
                ((), "tensor_dslab"),
                (("implementation-evidence",), "implementation-evidence"),
            ):
                with self.subTest(arguments=arguments):
                    log.unlink(missing_ok=True)
                    completed = subprocess.run(
                        [str(script), *arguments],
                        check=False,
                        capture_output=True,
                        cwd=temporary,
                        env=environment,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    commands = log.read_text().splitlines()
                    self.assertEqual(commands[0], "env list")
                    self.assertEqual(
                        commands[1],
                        " ".join(
                            (
                                "create",
                                "--yes",
                                "--name",
                                expected_name,
                                "--no-default-packages",
                                "--override-channels",
                                "--channel",
                                "conda-forge",
                                "python=3.14.6",
                                "pip",
                            )
                        ),
                    )
                    self.assertEqual(
                        commands[2],
                        " ".join(
                            (
                                "run",
                                "--name",
                                expected_name,
                                "python",
                                "-m",
                                "pip",
                                "install",
                                "--disable-pip-version-check",
                                "--no-input",
                                f"{Path.cwd()}[demos]",
                            )
                        ),
                    )
                    self.assertTrue(
                        commands[3].startswith(
                            f"run --name {expected_name} python -c "
                        )
                    )
                    self.assertFalse(
                        any(
                            forbidden in command
                            for command in commands
                            for forbidden in (
                                " activate ",
                                " init ",
                                " config ",
                                "env remove",
                                "kernelspec",
                                "readout.py",
                            )
                        )
                    )
                    self.assertIn(
                        f"conda activate {expected_name}",
                        completed.stdout,
                    )
                    self.assertIn(f"cd {Path.cwd()}", completed.stdout)
                    self.assertIn(
                        "python demos/readout.py",
                        completed.stdout,
                    )

            log.unlink(missing_ok=True)
            malformed = subprocess.run(
                [str(script), "one", "two"],
                check=False,
                capture_output=True,
                cwd=temporary,
                env=environment,
                text=True,
            )
            self.assertEqual(malformed.returncode, 2)
            self.assertFalse(log.exists())

            existing_environment = {
                **environment,
                "FAKE_EXISTING_ENV": "implementation-evidence",
            }
            existing = subprocess.run(
                [str(script), "implementation-evidence"],
                check=False,
                capture_output=True,
                cwd=temporary,
                env=existing_environment,
                text=True,
            )
            self.assertNotEqual(existing.returncode, 0)
            self.assertEqual(log.read_text().splitlines(), ["env list"])

    def test_demo_sources_are_explicitly_cpu_only(self) -> None:
        script_source = Path("demos/readout.py").read_text()
        notebook = nbformat.read("demos/readout.ipynb", as_version=4)
        notebook_source = "\n".join(cell.source for cell in notebook.cells)
        for source in (script_source, notebook_source):
            self.assertIn(
                'counts = torch.zeros(shape, dtype=torch.int64, device="cpu")',
                source,
            )
            self.assertIn('field.tensor.device.type == "cpu"', source)
            self.assertNotIn("torch.Generator", source)
            self.assertNotIn("torch.randint", source)
            self.assertNotIn("torch.cuda", source)
            self.assertNotIn('device="cuda"', source)
            self.assertNotIn("device='cuda'", source)

    def test_profile_surface_is_precise_public_and_import_inert(self) -> None:
        import tensor_dslab.readout.profiles as profiles

        self.assertEqual(profiles.__all__, ("ds20k_veto",))
        self.assertIs(profiles.ds20k_veto, ds20k_veto)
        self.assertEqual(ds20k_veto.__module__, "tensor_dslab.readout.profiles")
        self.assertEqual(
            ds20k_veto.__doc__,
            "Return a fresh provisional DS20k Veto demonstration profile.",
        )
        self.assertFalse(hasattr(tensor_dslab, "ds20k_veto"))
        self.assertFalse(hasattr(readout, "ds20k_veto"))

        path = Path("tensor_dslab/readout/profiles.py")
        tree = ast.parse(path.read_text(), filename=str(path))
        self.assertEqual(ast.get_docstring(tree, clean=False), profiles.__doc__)
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                for node in tree.body
            )
        )
        source = path.read_text()
        for forbidden in (
            "matplotlib",
            "nbclient",
            "nbformat",
            "ipykernel",
            "torch",
            "open(",
            "getenv",
            "environ",
            "socket",
        ):
            self.assertNotIn(forbidden, source)

        code = (
            "import sys, tensor_dslab, tensor_dslab.readout as readout; "
            "before=set(sys.modules); "
            "import tensor_dslab.readout.profiles as profiles; "
            "assert profiles.__all__ == ('ds20k_veto',); "
            "assert not hasattr(tensor_dslab, 'ds20k_veto'); "
            "assert not hasattr(readout, 'ds20k_veto'); "
            "forbidden=('matplotlib','ipykernel','nbclient','nbformat'); "
            "assert not any(name in sys.modules for name in forbidden); "
            "assert set(sys.modules)-before == {'tensor_dslab.readout.profiles'}"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_profile_content_is_exact_fresh_and_quantity_owned(self) -> None:
        first = ds20k_veto()
        second = ds20k_veto()
        literal = _literal_config()
        self.assertIs(type(first), ReadoutConfig)
        self.assertEqual(_config_signature(first), _config_signature(literal))
        self.assertEqual(_config_signature(second), _config_signature(literal))
        self.assertIsNot(first, second)

        assert first.charge is not None and second.charge is not None
        assert first.charge.dark_count is not None
        assert second.charge.dark_count is not None
        assert first.pure_waveform is not None and second.pure_waveform is not None
        assert type(first.pure_waveform.model) is VetoPduPulseConfig
        assert type(second.pure_waveform.model) is VetoPduPulseConfig
        assert first.noise_waveform is not None and second.noise_waveform is not None
        assert type(first.noise_waveform.model) is PsdNoiseConfig
        assert type(second.noise_waveform.model) is PsdNoiseConfig
        first_quantities = (
            first.charge.dark_count.rate,
            first.pure_waveform.model.gaussian_center,
            first.pure_waveform.model.gaussian_width,
            first.pure_waveform.model.edge_offset_1,
            first.pure_waveform.model.edge_width_1,
            first.pure_waveform.model.edge_offset_2,
            first.pure_waveform.model.edge_width_2,
            first.pure_waveform.model.support_time,
            first.pure_waveform.model.peak_voltage_per_photoelectron,
            first.noise_waveform.model.frequency_left_edges,
            first.noise_waveform.model.frequency_stop,
            first.noise_waveform.model.power_density,
        )
        second_quantities = (
            second.charge.dark_count.rate,
            second.pure_waveform.model.gaussian_center,
            second.pure_waveform.model.gaussian_width,
            second.pure_waveform.model.edge_offset_1,
            second.pure_waveform.model.edge_width_1,
            second.pure_waveform.model.edge_offset_2,
            second.pure_waveform.model.edge_width_2,
            second.pure_waveform.model.support_time,
            second.pure_waveform.model.peak_voltage_per_photoelectron,
            second.noise_waveform.model.frequency_left_edges,
            second.noise_waveform.model.frequency_stop,
            second.noise_waveform.model.power_density,
        )
        for left, right in zip(first_quantities, second_quantities):
            self.assertIsNot(left, right)
            self.assertEqual(_quantity_signature(left), _quantity_signature(right))
        for vector in (
            first.noise_waveform.model.frequency_left_edges,
            first.noise_waveform.model.power_density,
        ):
            self.assertIs(type(vector.magnitude), np.ndarray)
            self.assertFalse(vector.magnitude.flags.writeable)
        self.assertFalse(
            np.shares_memory(
                first.noise_waveform.model.frequency_left_edges.magnitude,
                second.noise_waveform.model.frequency_left_edges.magnitude,
            )
        )
        self.assertFalse(
            np.shares_memory(
                first.noise_waveform.model.power_density.magnitude,
                second.noise_waveform.model.power_density.magnitude,
            )
        )

    def test_profile_returns_a_fresh_complete_config_tree(self) -> None:
        first = _config_nodes(ds20k_veto())
        second = _config_nodes(ds20k_veto())
        expected = {
            (): ReadoutConfig,
            ("charge",): ChargeConfig,
            ("charge", "dark_count"): DarkCountConfig,
            ("pure_waveform",): PureWaveformConfig,
            ("pure_waveform", "model"): VetoPduPulseConfig,
            ("noise_waveform",): NoiseWaveformConfig,
            ("noise_waveform", "model"): PsdNoiseConfig,
            ("analog_waveform",): AnalogWaveformConfig,
            ("digitized_waveform",): DigitizedWaveformConfig,
        }
        self.assertEqual(
            {path: type(node) for path, node in first.items()},
            expected,
        )
        self.assertEqual(
            {path: type(node) for path, node in second.items()},
            expected,
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertIsNot(first[path], second[path])

    def test_factory_has_no_external_or_execution_side_effect(self) -> None:
        environment = os.environ.copy()
        with (
            patch("builtins.open", side_effect=AssertionError("file access")),
            patch.object(socket, "socket", side_effect=AssertionError("network")),
            patch.object(_REGISTRY, "define", side_effect=AssertionError("registry")),
            patch.object(torch, "empty", side_effect=AssertionError("tensor")),
            patch.object(torch, "tensor", side_effect=AssertionError("tensor")),
            patch.object(torch, "zeros", side_effect=AssertionError("tensor")),
            patch.object(torch, "ones", side_effect=AssertionError("tensor")),
            patch.object(torch, "randint", side_effect=AssertionError("rng")),
        ):
            result = ds20k_veto()
        self.assertIs(type(result), ReadoutConfig)
        self.assertEqual(os.environ, environment)

    def test_profile_preparation_matches_literal_polarity_and_psd(self) -> None:
        profile = ds20k_veto()
        literal = _literal_config()
        source = _source()
        sampling = SamplingRuntime(
            sample_count=5000,
            sample_period_ps=2000,
            sample_dimension=2,
        )

        assert profile.noise_waveform is not None
        assert type(profile.noise_waveform.model) is PsdNoiseConfig
        psd = profile.noise_waveform.model
        edges = tuple(float(value) for value in psd.frequency_left_edges.magnitude)
        densities = tuple(float(value) for value in psd.power_density.magnitude)
        stop = float(psd.frequency_stop.magnitude)
        integral = math.fsum(
            (right - left) * density
            for left, right, density in zip(
                edges,
                edges[1:] + (stop,),
                densities,
            )
        )
        self.assertEqual(integral, 0.255125)
        self.assertEqual(edges[-1], 62_500_000.0)
        self.assertEqual(densities[-1], 0.0)
        self.assertEqual(stop, 250_000_000.0)

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                profile_requested, profile_runtime = prepare_readout(
                    source,
                    products=_PRODUCTS,
                    config=profile,
                    rng=Threefry4x32(seed=17),
                    floating_dtype=dtype,
                )
                literal_requested, literal_runtime = prepare_readout(
                    source,
                    products=_PRODUCTS,
                    config=literal,
                    rng=Threefry4x32(seed=17),
                    floating_dtype=dtype,
                )
                self.assertEqual(profile_requested, literal_requested)
                self.assertEqual(
                    _runtime_signature(profile_runtime),
                    _runtime_signature(literal_runtime),
                )
                assert profile_runtime.pure_waveform is not None
                assert literal_runtime.pure_waveform is not None
                self.assertIsNot(
                    profile_runtime.pure_waveform.kernel,
                    literal_runtime.pure_waveform.kernel,
                )
                self.assertEqual(
                    float(profile_runtime.pure_waveform.kernel.min()),
                    float(torch.tensor(-14.5912372, dtype=dtype)),
                )
                assert profile_runtime.noise_waveform is not None
                assert literal_runtime.noise_waveform is not None
                self.assertIs(
                    type(profile_runtime.noise_waveform.model),
                    PsdNoiseRuntime,
                )
                self.assertIs(
                    type(literal_runtime.noise_waveform.model),
                    PsdNoiseRuntime,
                )
                profile_powers = profile_runtime.noise_waveform.model
                literal_powers = literal_runtime.noise_waveform.model
                assert type(profile_powers) is PsdNoiseRuntime
                assert type(literal_powers) is PsdNoiseRuntime
                self.assertIsNot(
                    profile_powers.represented_powers_mv2,
                    literal_powers.represented_powers_mv2,
                )
                prepared_rms = math.sqrt(
                    float(profile_powers.represented_powers_mv2.sum())
                )
                self.assertAlmostEqual(prepared_rms, 0.5031, places=4)

        assert profile.pure_waveform is not None
        for dtype in (torch.float32, torch.float64):
            direct = prepare_pure_waveform(
                profile.pure_waveform,
                sampling=sampling,
                floating_dtype=dtype,
                device=torch.device("cpu"),
            )
            self.assertEqual(
                float(direct.kernel.min()),
                float(torch.tensor(-14.5912372, dtype=dtype)),
            )

    def test_public_profile_and_literal_results_are_exact_and_repeatable(self) -> None:
        source = _source()
        source_snapshot = source.tensor.clone()
        results = (
            simulate_readout(
                source,
                products=_PRODUCTS,
                config=ds20k_veto(),
                rng=Threefry4x32(seed=17),
                floating_dtype=torch.float32,
            ),
            simulate_readout(
                source,
                products=_PRODUCTS,
                config=_literal_config(),
                rng=Threefry4x32(seed=17),
                floating_dtype=torch.float32,
            ),
            simulate_readout(
                source,
                products=_PRODUCTS,
                config=ds20k_veto(),
                rng=Threefry4x32(seed=17),
                floating_dtype=torch.float32,
            ),
        )
        self.assertTrue(torch.equal(source.tensor, source_snapshot))
        for result in results:
            self.assertEqual(tuple(result.fields), _PRODUCTS)
            self.assertIs(result.field(Photoelectrons), source)
            for product in _PRODUCTS:
                field = result.field(product)
                self.assertEqual(field.shape, source.shape)
                self.assertEqual(field.axes, source.axes)
                self.assertTrue(
                    all(
                        field_axis is source_axis
                        for field_axis, source_axis in zip(field.axes, source.axes)
                    )
                )
                self.assertEqual(field.tensor.device, source.tensor.device)
            self.assertIs(result.tensor(Photoelectrons).dtype, torch.int64)
            for product in (
                Charge,
                PureWaveform,
                NoiseWaveform,
                AnalogWaveform,
            ):
                self.assertIs(result.tensor(product).dtype, torch.float32)
            self.assertIs(result.tensor(DigitizedWaveform).dtype, torch.int32)
            self.assertTrue(
                torch.equal(
                    result.tensor(AnalogWaveform),
                    result.tensor(PureWaveform) + result.tensor(NoiseWaveform),
                )
            )
        for product in _PRODUCTS:
            self.assertTrue(
                torch.equal(
                    results[0].tensor(product),
                    results[1].tensor(product),
                )
            )
            self.assertTrue(
                torch.equal(
                    results[0].tensor(product),
                    results[2].tensor(product),
                )
            )

    def test_selected_request_retains_exactly_digitized_waveform(self) -> None:
        source = _source()
        result = simulate_readout(
            source,
            products=(DigitizedWaveform,),
            config=ds20k_veto(),
            rng=Threefry4x32(seed=17),
        )
        self.assertEqual(tuple(result.fields), (DigitizedWaveform,))
        self.assertEqual(result.field(DigitizedWaveform).axes, source.axes)
        self.assertIs(result.tensor(DigitizedWaveform).dtype, torch.int32)

    def test_repository_root_script_uses_only_public_workflow(self) -> None:
        path = Path("demos/readout.py")
        tree = ast.parse(path.read_text(), filename=str(path))
        self.assertTrue(ast.get_docstring(tree, clean=False))
        source = path.read_text()
        for forbidden in (
            ".runtime",
            "readout.runtime",
            "_prepare",
            "_produce",
            "_validate",
            "open(",
            "requests",
            "urllib",
            "socket",
            "matplotlib",
        ):
            self.assertNotIn(forbidden, source)
        completed = subprocess.run(
            [sys.executable, "-B", str(path)],
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for name in (
            "Photoelectrons",
            "Charge",
            "PureWaveform",
            "NoiseWaveform",
            "AnalogWaveform",
            "DigitizedWaveform",
            "selected products: ('DigitizedWaveform',)",
        ):
            self.assertIn(name, completed.stdout)

    def test_script_and_notebook_source_and_manual_config_are_exact(self) -> None:
        script_path = Path("demos/readout.py")
        script_tree = ast.parse(
            script_path.read_text(),
            filename=str(script_path),
        )
        main = next(
            node
            for node in script_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        requested_index = next(
            index
            for index, statement in enumerate(main.body)
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "requested"
            )
        )
        script_source_statements = main.body[:requested_index]

        notebook = nbformat.read("demos/readout.ipynb", as_version=4)
        notebook_source_cell = next(
            cell
            for cell in notebook.cells
            if cell.cell_type == "code"
            and "photoelectrons = Photoelectrons(" in cell.source
        )
        notebook_source_tree = ast.parse(
            notebook_source_cell.source,
            filename="demos/readout.ipynb:source",
        )

        self.assertEqual(
            _source_construction_ast(script_source_statements),
            _EXPECTED_SOURCE_CONSTRUCTION_AST,
        )
        self.assertEqual(
            _source_construction_ast(notebook_source_tree.body),
            _EXPECTED_SOURCE_CONSTRUCTION_AST,
        )

        expected_shape = (2, 4, 5000)
        expected_counts = torch.zeros(
            expected_shape,
            dtype=torch.int64,
            device="cpu",
        )
        expected_counts[0, 0, 100] = 1
        expected_counts[0, 0, 1300] = 2
        expected_counts[0, 0, 2500] = 3
        expected_counts[0, 0, 3700] = 4
        produced_counts: list[torch.Tensor] = []
        for label, statements in (
            ("script", script_source_statements),
            ("notebook", notebook_source_tree.body),
        ):
            with self.subTest(label=label):
                namespace: dict[str, object] = {
                    "torch": torch,
                    "ExampleAxis": ExampleAxis,
                    "ChannelAxis": ChannelAxis,
                    "SampleAxis": SampleAxis,
                    "quantity": quantity,
                    "Photoelectrons": Photoelectrons,
                }
                global_rng_before = torch.random.get_rng_state().clone()
                module = ast.fix_missing_locations(
                    ast.Module(body=statements, type_ignores=[])
                )
                exec(
                    compile(
                        module,
                        f"demos/readout:{label}-source",
                        "exec",
                    ),
                    namespace,
                )
                self.assertTrue(
                    torch.equal(
                        torch.random.get_rng_state(),
                        global_rng_before,
                    )
                )
                self.assertEqual(namespace["shape"], expected_shape)
                counts = namespace["counts"]
                assert isinstance(counts, torch.Tensor)
                self.assertTrue(torch.equal(counts, expected_counts))
                source = namespace["photoelectrons"]
                self.assertIs(type(source), Photoelectrons)
                assert type(source) is Photoelectrons
                self.assertIs(source.tensor, counts)
                self.assertEqual(source.shape, expected_shape)
                self.assertEqual(
                    source.axis(ChannelAxis).labels,
                    ("veto-0", "veto-1", "veto-2", "veto-3"),
                )
                self.assertEqual(source.axis(SampleAxis).step, 2000)
                self.assertEqual(source.axis(SampleAxis).count, 5000)
                self.assertEqual(
                    source.axis(SampleAxis).stop_time.magnitude,
                    10_000_000,
                )
                self.assertEqual(
                    tuple(
                        tuple(int(item) for item in coordinate)
                        for coordinate in torch.nonzero(source.tensor).tolist()
                    ),
                    (
                        (0, 0, 100),
                        (0, 0, 1300),
                        (0, 0, 2500),
                        (0, 0, 3700),
                    ),
                )
                self.assertEqual(
                    tuple(
                        int(source.tensor[0, 0, index])
                        for index in (100, 1300, 2500, 3700)
                    ),
                    (1, 2, 3, 4),
                )
                self.assertEqual(int(source.tensor.sum()), 10)
                produced_counts.append(counts)
        self.assertTrue(torch.equal(produced_counts[0], produced_counts[1]))

        manual_cell = next(
            cell
            for cell in notebook.cells
            if cell.cell_type == "code"
            and "manual_config = ReadoutConfig(" in cell.source
        )
        manual_namespace: dict[str, object] = {
            "ReadoutConfig": ReadoutConfig,
            "ChargeConfig": ChargeConfig,
            "DarkCountConfig": DarkCountConfig,
            "PureWaveformConfig": PureWaveformConfig,
            "VetoPduPulseConfig": VetoPduPulseConfig,
            "NoiseWaveformConfig": NoiseWaveformConfig,
            "PsdNoiseConfig": PsdNoiseConfig,
            "AnalogWaveformConfig": AnalogWaveformConfig,
            "DigitizedWaveformConfig": DigitizedWaveformConfig,
            "PositiveInteger": PositiveInteger,
            "NonnegativeFloat": NonnegativeFloat,
            "quantity": quantity,
            "quantities": quantities,
        }
        exec(
            compile(
                manual_cell.source,
                "demos/readout.ipynb:manual-config",
                "exec",
            ),
            manual_namespace,
        )
        manual_config = manual_namespace["manual_config"]
        self.assertIs(type(manual_config), ReadoutConfig)
        assert type(manual_config) is ReadoutConfig
        self.assertEqual(
            _config_signature(manual_config),
            _config_signature(ds20k_veto()),
        )

    def test_notebook_is_executed_and_has_exact_public_narrative(self) -> None:
        notebook = nbformat.read("demos/readout.ipynb", as_version=4)
        self.assertEqual(len(notebook.cells), 23)
        self.assertEqual(
            tuple(cell.cell_type for cell in notebook.cells),
            ("markdown", "code") * 11 + ("markdown",),
        )
        opening = notebook.cells[0].source
        self.assertIn("./create_environment.sh", opening)
        self.assertNotIn("./demos/create_environment.sh", opening)
        self.assertIn("conda activate tensor_dslab", opening)
        self.assertIn("before opening this notebook", opening.lower())

        code_cells = tuple(
            cell for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertEqual(
            hashlib.sha256(
                "\n\n".join(cell.source for cell in code_cells).encode()
            ).hexdigest(),
            _EXPECTED_NOTEBOOK_CODE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(Path("demos/readout.ipynb").read_bytes()).hexdigest(),
            _EXPECTED_NOTEBOOK_SHA256,
        )
        first_code = code_cells[0].source
        for required in (
            "import platform",
            "import sys",
            "import torch",
            "default_tensor = torch.empty(())",
            'print("Python executable:", sys.executable)',
            'print("Python version:", platform.python_version())',
            'print("PyTorch version:", torch.__version__)',
            'assert platform.python_version() == "3.14.6"',
            'assert torch.__version__.split("+", maxsplit=1)[0] == "2.13.0"',
            'assert default_tensor.device.type == "cpu"',
        ):
            self.assertIn(required, first_code)
        executable_source = "\n".join(cell.source for cell in code_cells)
        for forbidden in (
            "conda ",
            "create_environment.sh",
            "pip install",
            "subprocess",
            "os.system",
            "sys.executable =",
            "torch.cuda",
            'device="cuda"',
            "device='cuda'",
        ):
            self.assertNotIn(forbidden, executable_source)

        self.assertEqual(
            tuple(cell.execution_count for cell in code_cells),
            tuple(range(1, 12)),
        )
        outputs = tuple(
            output
            for cell in code_cells
            for output in cell.outputs
        )
        self.assertEqual(len(outputs), 7)
        self.assertEqual(
            sum(
                output.output_type == "display_data"
                and "image/png" in output.data
                for output in outputs
            ),
            1,
        )
        self.assertFalse(
            any(output.output_type == "error" for output in outputs)
        )
        for cell in code_cells:
            self.assertNotIn("execution", cell.metadata)
        source = "\n".join(cell.source for cell in notebook.cells)
        for required in (
            "provisional demonstration profile",
            "manual_config = ReadoutConfig(",
            "profile_config = ds20k_veto()",
            'counts = torch.zeros(shape, dtype=torch.int64, device="cpu")',
            "counts[0, 0, 100] = 1",
            "counts[0, 0, 1300] = 2",
            "counts[0, 0, 2500] = 3",
            "counts[0, 0, 3700] = 4",
            "SampleAxis.from_period(period=quantity(2.0, \"ns\"), count=5000)",
            (
                "assert photoelectrons.axis(SampleAxis).stop_time.magnitude "
                "== 10_000_000"
            ),
            "products=(DigitizedWaveform,)",
            "assert torch.equal(analog_trace, recomposed_trace)",
            'plot_axes[2].plot(time_ns.numpy(), analog_trace.numpy(), color="tab:green")',
            "for panel_index in (0, 2, 3):",
            "charge_axis.vlines(",
            'color="black"',
            'charge_axis.set_ylabel("Charge [PE]", color="black")',
            'charge_axis.tick_params(axis="y", colors="black")',
            'charge_axis.spines["right"].set_color("black")',
            "source plus seeded dark counts",
            "Pure [mV]",
            "Noise [mV]",
            "Analog [mV]",
            "ADC code",
            "Time [ns]",
            "plt.show()",
            "plt.close(figure)",
            "experimental_config = replace(",
            "FIL/TensorG4DS",
            "TensorML",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "torch.Generator",
            "torch.randint",
            "torch.where",
            "source_generator",
            "draws =",
            "plot_axes[2].legend",
            "recomposed_trace.numpy()",
            'label="AnalogWaveform"',
            "purple",
            "tab:purple",
            "savefig",
            "read_csv",
            "read_json",
            "open(",
            "tensor_dslab.readout.runtime",
            "_prepare",
            "_produce",
            "_validate",
            "import tensor_g4ds",
            "import tensor_ml",
        ):
            self.assertNotIn(forbidden, source)

    def test_temporary_notebook_executes_with_display_and_no_plot_file(self) -> None:
        repository_files_before = frozenset(
            path.relative_to(Path.cwd())
            for path in Path.cwd().rglob("*")
            if path.is_file()
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            notebook_path = temporary / "readout.ipynb"
            shutil.copyfile("demos/readout.ipynb", notebook_path)
            notebook = nbformat.read(notebook_path, as_version=4)
            with patch.dict(
                os.environ,
                {
                    "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                    "MPLCONFIGDIR": str(temporary / "matplotlib"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            ):
                executed = NotebookClient(
                    notebook,
                    timeout=240,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(Path.cwd())}},
                ).execute()
            display_outputs = tuple(
                output
                for cell in executed.cells
                if cell.cell_type == "code"
                for output in cell.outputs
                if output.output_type == "display_data"
            )
            self.assertTrue(display_outputs)
            self.assertTrue(
                any(
                    "image/png" in output.data
                    or "image/svg+xml" in output.data
                    for output in display_outputs
                )
            )
            self.assertFalse(
                any(
                    path.suffix.lower() in (".png", ".svg", ".pdf")
                    for path in temporary.rglob("*")
                    if path.is_file()
                )
            )
        repository_files_after = frozenset(
            path.relative_to(Path.cwd())
            for path in Path.cwd().rglob("*")
            if path.is_file()
        )
        self.assertEqual(repository_files_after, repository_files_before)


if __name__ == "__main__":
    unittest.main()
