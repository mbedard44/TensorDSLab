from pathlib import Path
import ast
import importlib
import unittest

import tensor_dslab
import tensor_dslab.analog_waveform
import tensor_dslab.charge
import tensor_dslab.common
import tensor_dslab.digitized_waveform
import tensor_dslab.noise_waveform
import tensor_dslab.photoelectrons
import tensor_dslab.pure_waveform


class PackageContractTests(unittest.TestCase):
    def test_exact_facade_and_tree(self) -> None:
        expected = (
            "Afterpulse", "AfterpulseSpec", "AnalogGain", "AnalogGainSpec",
            "AnalogMaximum", "AnalogMaximumSpec", "AnalogMinimum",
            "AnalogMinimumSpec", "AnalogWaveform", "AnalogWaveformConfig",
            "AnalogWaveformKernels", "AnalogWaveformSpec", "BitDepth",
            "BitDepthSpec", "ChannelAxis", "Charge", "ChargeConfig",
            "ChargeKernels", "ChargeSpec", "DarkCountRate",
            "DarkCountRateSpec", "DelayedCrosstalk", "DelayedCrosstalkSpec",
            "DigitizedWaveform", "DigitizedWaveformConfig",
            "DigitizedWaveformKernels", "DigitizedWaveformSpec",
            "DirectCrosstalk", "DirectCrosstalkSpec", "ExampleAxis",
            "FrequencyAxis", "InputMaximum", "InputMaximumSpec",
            "InputMinimum", "InputMinimumSpec", "NoiseWaveform",
            "NoiseWaveformConfig", "NoiseWaveformKernels",
            "NoiseWaveformSpec", "Photoelectrons", "PhotoelectronsSpec",
            "PowerSpectralDensity", "PowerSpectralDensitySpec",
            "PulseResponse", "PulseResponseSpec", "PureWaveform",
            "PureWaveformConfig", "PureWaveformKernels", "PureWaveformSpec",
            "QuantityAxis", "QuantityFieldSpec", "QuantityKernelSpec",
            "SmearingWidth", "SmearingWidthSpec", "TimeAxis",
            "TimingJitter", "TimingJitterSpec", "WhiteNoiseRms",
            "WhiteNoiseRmsSpec", "quantity", "unit_registry",
        )
        self.assertEqual(tensor_dslab.__all__, expected)
        self.assertEqual(
            (
                tensor_dslab.common.__all__,
                tensor_dslab.photoelectrons.__all__,
                tensor_dslab.charge.__all__,
                tensor_dslab.pure_waveform.__all__,
                tensor_dslab.noise_waveform.__all__,
                tensor_dslab.analog_waveform.__all__,
                tensor_dslab.digitized_waveform.__all__,
            ),
            (
                ("ChannelAxis", "ExampleAxis", "FrequencyAxis", "QuantityAxis",
                 "QuantityFieldSpec", "QuantityKernelSpec", "TimeAxis",
                 "quantity", "unit_registry"),
                ("Photoelectrons", "PhotoelectronsSpec"),
                ("Afterpulse", "AfterpulseSpec", "Charge", "ChargeConfig",
                 "ChargeKernels", "ChargeSpec", "DarkCountRate",
                 "DarkCountRateSpec", "DelayedCrosstalk",
                 "DelayedCrosstalkSpec", "DirectCrosstalk",
                 "DirectCrosstalkSpec", "SmearingWidth",
                 "SmearingWidthSpec", "TimingJitter", "TimingJitterSpec"),
                ("PulseResponse", "PulseResponseSpec", "PureWaveform",
                 "PureWaveformConfig", "PureWaveformKernels",
                 "PureWaveformSpec"),
                ("NoiseWaveform", "NoiseWaveformConfig",
                 "NoiseWaveformKernels", "NoiseWaveformSpec",
                 "PowerSpectralDensity", "PowerSpectralDensitySpec",
                 "WhiteNoiseRms", "WhiteNoiseRmsSpec"),
                ("AnalogMaximum", "AnalogMaximumSpec", "AnalogMinimum",
                 "AnalogMinimumSpec", "AnalogWaveform",
                 "AnalogWaveformConfig", "AnalogWaveformKernels",
                 "AnalogWaveformSpec"),
                ("AnalogGain", "AnalogGainSpec", "BitDepth", "BitDepthSpec",
                 "DigitizedWaveform", "DigitizedWaveformConfig",
                 "DigitizedWaveformKernels", "DigitizedWaveformSpec",
                 "InputMaximum", "InputMaximumSpec", "InputMinimum",
                 "InputMinimumSpec"),
            ),
        )
        for required in (
            "tensor_dslab/common/alignment.py",
            "tensor_dslab/common/field.py",
            "tensor_dslab/charge/runtime/random.py",
            "tensor_dslab/noise_waveform/runtime/random.py",
        ):
            self.assertTrue(Path(required).is_file(), required)
        self.assertFalse(Path("tensor_dslab/readout").exists())
        self.assertFalse(Path("demos/readout.py").exists())
        self.assertFalse(Path(".agents").exists())
        for package in (
            "analog_waveform",
            "charge",
            "digitized_waveform",
            "noise_waveform",
            "photoelectrons",
            "pure_waveform",
        ):
            runtime = importlib.import_module(
                f"tensor_dslab.{package}.runtime"
            )
            self.assertFalse(hasattr(runtime, "__all__"))
            self.assertIsNotNone(runtime.__file__)
            assert runtime.__file__ is not None
            runtime_source = Path(runtime.__file__).read_text(encoding="utf-8")
            runtime_tree = ast.parse(runtime_source)
            self.assertFalse(
                any(
                    isinstance(node, (ast.Import, ast.ImportFrom))
                    for node in runtime_tree.body
                )
            )

    def test_retired_names_absent(self) -> None:
        for name in (
            "ReadoutConfig",
            "ReadoutCollection",
            "SampleAxis",
            "simulate_readout",
            "QuantityKernel",
            "quantities",
        ):
            self.assertFalse(hasattr(tensor_dslab, name), name)
