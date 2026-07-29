from pathlib import Path
import ast
import importlib
import unittest

import tensor_dslab
import tensor_dslab.analog_waveform
import tensor_dslab.charge
import tensor_dslab.common
import tensor_dslab.digitized_waveform
import tensor_dslab.encoded_waveform
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
            "DirectCrosstalk", "DirectCrosstalkSpec", "EncodedWaveform",
            "EncodedWaveformConfig", "EncodedWaveformKernels",
            "EncodedWaveformSpec", "ExampleAxis",
            "FrequencyAxis", "InputMaximum", "InputMaximumSpec",
            "InputMinimum", "InputMinimumSpec", "NoiseWaveform",
            "NoiseWaveformConfig", "NoiseWaveformKernels",
            "NoiseWaveformSpec", "PostTriggerSamples",
            "PostTriggerSamplesSpec", "Photoelectrons", "PhotoelectronsSpec",
            "PowerSpectralDensity", "PowerSpectralDensitySpec",
            "PulseResponse", "PulseResponseSpec", "PreTriggerSamples",
            "PreTriggerSamplesSpec", "PureWaveform",
            "PureWaveformConfig", "PureWaveformKernels", "PureWaveformSpec",
            "QuantityAxis", "QuantityFieldSpec", "QuantityKernelSpec",
            "ReleaseThresholdCode", "ReleaseThresholdCodeSpec",
            "RequiredTimeOverSamples", "RequiredTimeOverSamplesSpec",
            "SmearingWidth", "SmearingWidthSpec", "TimeAxis",
            "TimingJitter", "TimingJitterSpec", "TriggerThresholdCode",
            "TriggerThresholdCodeSpec", "WhiteNoiseRms",
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
                tensor_dslab.encoded_waveform.__all__,
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
                ("EncodedWaveform", "EncodedWaveformConfig",
                 "EncodedWaveformKernels", "EncodedWaveformSpec",
                 "PostTriggerSamples", "PostTriggerSamplesSpec",
                 "PreTriggerSamples", "PreTriggerSamplesSpec",
                 "ReleaseThresholdCode", "ReleaseThresholdCodeSpec",
                 "RequiredTimeOverSamples", "RequiredTimeOverSamplesSpec",
                 "TriggerThresholdCode", "TriggerThresholdCodeSpec"),
            ),
        )
        for required in (
            "tensor_dslab/common/alignment.py",
            "tensor_dslab/common/field.py",
            "tensor_dslab/common/requirements/__init__.py",
            "tensor_dslab/common/requirements/axis.py",
            "tensor_dslab/common/requirements/capacity.py",
            "tensor_dslab/common/requirements/collection.py",
            "tensor_dslab/common/requirements/config.py",
            "tensor_dslab/common/requirements/field.py",
            "tensor_dslab/common/requirements/kernel.py",
            "tensor_dslab/common/requirements/tensor.py",
            "tensor_dslab/common/requirements/unit.py",
            "tensor_dslab/charge/runtime/random.py",
            "tensor_dslab/noise_waveform/runtime/random.py",
            "tensor_dslab/encoded_waveform/runtime/prepare.py",
            "tensor_dslab/encoded_waveform/runtime/produce.py",
            "tensor_dslab/encoded_waveform/runtime/validate.py",
        ):
            self.assertTrue(Path(required).is_file(), required)
        self.assertFalse(Path("tensor_dslab/readout").exists())
        self.assertFalse(Path("demos/readout.py").exists())
        self.assertFalse(Path(".agents").exists())
        for package in (
            "analog_waveform",
            "charge",
            "digitized_waveform",
            "encoded_waveform",
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
        package_files = tuple(
            path
            for path in Path("tensor_dslab").rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
        self.assertEqual(len(package_files), 73)
        self.assertEqual(
            sum(path.suffix == ".py" for path in package_files),
            72,
        )
        test_paths = tuple(
            path for path in Path("tests").rglob("*.py")
        )
        self.assertEqual(len(test_paths), 25)
        test_methods = tuple(
            node
            for path in test_paths
            for node in ast.walk(
                ast.parse(path.read_text(encoding="utf-8"))
            )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
        self.assertEqual(len(test_methods), 70)

    def test_private_requirement_and_kernel_collection_ownership(self) -> None:
        requirement_root = importlib.import_module(
            "tensor_dslab.common.requirements"
        )
        self.assertEqual(requirement_root.__all__, ())
        requirement_functions = {
            "axis.py": (
                "require_supported_coordinates",
                "require_supported_integer_coordinates",
                "require_coordinate_scale",
                "require_regular_coordinates",
            ),
            "capacity.py": (
                "require_tensor_capacity",
                "require_address_capacity",
            ),
            "collection.py": (
                "require_admitted_member_types",
                "require_exact_member_types",
                "require_member_count",
            ),
            "config.py": (
                "require_config_components",
                "require_prepared_config",
                "require_prepared_sources",
            ),
            "field.py": (
                "require_exact_field_spec",
                "require_fresh_product",
            ),
            "kernel.py": (
                "require_exact_kernel_spec",
                "require_no_operation_axes",
                "require_no_conditioning_axis_type",
                "require_operation_axis_count",
                "require_operation_axes_type",
                "require_nonempty_operation_extents",
                "require_operation_target_count",
                "require_offset_bounds",
                "require_operation_row_total",
            ),
            "tensor.py": (
                "require_exact_dtype",
                "require_dtype_in",
                "require_floating_dtype",
                "require_signed_integer_dtype",
                "require_negative_representable_suppression_code",
                "require_encoded_values",
                "require_finite",
                "require_nonnegative",
                "require_positive",
                "require_values_between",
            ),
            "unit.py": ("require_unit_compatible",),
        }
        requirements_path = Path("tensor_dslab/common/requirements")
        self.assertEqual(
            tuple(path.name for path in sorted(requirements_path.glob("*.py"))),
            (
                "__init__.py",
                "axis.py",
                "capacity.py",
                "collection.py",
                "config.py",
                "field.py",
                "kernel.py",
                "tensor.py",
                "unit.py",
            ),
        )
        for filename, expected in requirement_functions.items():
            tree = ast.parse(
                (requirements_path / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(
                tuple(
                    node.name
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name.startswith("require_")
                ),
                expected,
            )
        for name in (
            "require_finite",
            "require_prepared_config",
            "require_tensor_capacity",
        ):
            self.assertFalse(hasattr(tensor_dslab, name))
            self.assertFalse(hasattr(tensor_dslab.common, name))

        owners = (
            (tensor_dslab.ChargeKernels, "charge"),
            (tensor_dslab.PureWaveformKernels, "pure_waveform"),
            (tensor_dslab.NoiseWaveformKernels, "noise_waveform"),
            (tensor_dslab.AnalogWaveformKernels, "analog_waveform"),
            (tensor_dslab.DigitizedWaveformKernels, "digitized_waveform"),
            (tensor_dslab.EncodedWaveformKernels, "encoded_waveform"),
        )
        for collection_type, package in owners:
            with self.subTest(collection=collection_type.__name__):
                self.assertEqual(
                    collection_type.__module__,
                    f"tensor_dslab.{package}.kernel",
                )
                config_source = Path(
                    f"tensor_dslab/{package}/config.py"
                ).read_text(encoding="utf-8")
                config_tree = ast.parse(config_source)
                self.assertFalse(
                    any(
                        isinstance(node, ast.ClassDef)
                        and node.name == collection_type.__name__
                        for node in config_tree.body
                    )
                )

        alignment_tree = ast.parse(
            Path("tensor_dslab/common/alignment.py").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            tuple(
                node.name
                for node in alignment_tree.body
                if isinstance(node, ast.FunctionDef)
            ),
            (
                "align_source",
                "prepare_sources",
                "kernel_dimensions",
                "prepare_kernel",
            ),
        )
        for path in Path("tensor_dslab").rglob("*.py"):
            if "common/requirements/" in path.as_posix():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            self.assertFalse(
                any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("require_")
                    for node in ast.walk(tree)
                ),
                path,
            )
        charge_kernel = Path("tensor_dslab/charge/kernel.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_offsets", charge_kernel)
        self.assertNotIn("_require_probability_kernel", charge_kernel)

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
