"""Package topology, facade, privacy, and import-isolation evidence."""

import importlib
import ast
import inspect
import pathlib
import subprocess
import sys
import unittest

import tensor_dslab
import tensor_dslab.common as common
import tensor_dslab.readout as readout
import tensor_dslab.readout.charge as charge
import tensor_dslab.readout.pure_waveform as pure


ROOT_EXPORTS = (
    "Afterpulse",
    "AnalogSaturationConfig",
    "AnalogWaveform",
    "AnalogWaveformConfig",
    "ChannelAxis",
    "Charge",
    "ChargeConfig",
    "DarkCountRate",
    "DelayedCrosstalk",
    "DigitizedWaveform",
    "DigitizedWaveformConfig",
    "DirectCrosstalk",
    "ExampleAxis",
    "NoiseWaveform",
    "NoiseWaveformConfig",
    "Photoelectrons",
    "PsdNoiseConfig",
    "Pulse",
    "PureWaveform",
    "PureWaveformConfig",
    "QuantityKernel",
    "ReadoutCollection",
    "ReadoutConfig",
    "SampleAxis",
    "SmearingWidth",
    "TimingJitter",
    "WhiteNoiseConfig",
    "ZeroNoiseConfig",
    "quantities",
    "quantity",
    "simulate_readout",
)
COMMON_EXPORTS = (
    "ChannelAxis",
    "ExampleAxis",
    "QuantityKernel",
    "SampleAxis",
    "quantities",
    "quantity",
)
READOUT_EXPORTS = tuple(
    name
    for name in ROOT_EXPORTS
    if name not in (*COMMON_EXPORTS, "ChannelAxis", "ExampleAxis", "SampleAxis")
)
CHARGE_EXPORTS = (
    "Afterpulse",
    "Charge",
    "ChargeConfig",
    "DarkCountRate",
    "DelayedCrosstalk",
    "DirectCrosstalk",
    "SmearingWidth",
    "TimingJitter",
)
PURE_EXPORTS = ("Pulse", "PureWaveform", "PureWaveformConfig")


class PackageContractTest(unittest.TestCase):
    def test_exact_facades(self) -> None:
        self.assertEqual(tensor_dslab.__all__, ROOT_EXPORTS)
        self.assertEqual(common.__all__, COMMON_EXPORTS)
        self.assertEqual(readout.__all__, READOUT_EXPORTS)
        self.assertEqual(charge.__all__, CHARGE_EXPORTS)
        self.assertEqual(pure.__all__, PURE_EXPORTS)

    def test_export_identity_is_shared_across_facades(self) -> None:
        for name in common.__all__:
            self.assertIs(getattr(tensor_dslab, name), getattr(common, name))
        for name in readout.__all__:
            self.assertIs(getattr(tensor_dslab, name), getattr(readout, name))

    def test_required_and_retired_production_paths(self) -> None:
        required = (
            "tensor_dslab/common/axis.py",
            "tensor_dslab/common/kernel.py",
            "tensor_dslab/readout/runtime/kernel.py",
            "tensor_dslab/readout/charge/runtime/counts.py",
            "tensor_dslab/readout/charge/runtime/branching.py",
            "tensor_dslab/readout/charge/runtime/prepare.py",
            "tensor_dslab/readout/pure_waveform/runtime/prepare.py",
        )
        for path in required:
            with self.subTest(required=path):
                self.assertTrue(pathlib.Path(path).is_file())

        retired = (
            "tensor_dslab/common/axes.py",
            "tensor_dslab/readout/charge/runtime/effects",
            "tensor_dslab/readout/charge/effects",
            "tensor_dslab/readout/_random.py",
            "tensor_dslab/readout/_rng.py",
        )
        for path in retired:
            with self.subTest(retired=path):
                self.assertFalse(pathlib.Path(path).exists())

    def test_runtime_packages_export_nothing(self) -> None:
        for name in (
            "tensor_dslab.readout.runtime",
            "tensor_dslab.readout.charge.runtime",
            "tensor_dslab.readout.pure_waveform.runtime",
        ):
            module = importlib.import_module(name)
            self.assertFalse(hasattr(module, "__all__") and module.__all__)

    def test_fresh_process_import_isolation(self) -> None:
        code = (
            "import sys, tensor_dslab; "
            "print('tensor_g4ds' in sys.modules, 'tensor_ml' in sys.modules, "
            "'matplotlib' in sys.modules, 'IPython' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False False False False")

    def test_public_values_have_intentional_docstrings(self) -> None:
        for name in ROOT_EXPORTS:
            value = getattr(tensor_dslab, name)
            if inspect.isclass(value):
                self.assertTrue(value.__dict__.get("__doc__"), name)
            else:
                self.assertTrue(value.__doc__, name)

    def test_every_production_module_has_an_own_module_docstring(self) -> None:
        for path in pathlib.Path("tensor_dslab").rglob("*.py"):
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(), filename=str(path))
                self.assertTrue(ast.get_docstring(tree), path)


for _name in ROOT_EXPORTS:
    def _identity_case(
        self: PackageContractTest,
        name: str = _name,
    ) -> None:
        value = getattr(tensor_dslab, name)
        self.assertEqual(value.__name__, name)

    setattr(PackageContractTest, f"test_export_{_name}", _identity_case)
