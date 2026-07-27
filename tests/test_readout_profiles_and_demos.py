"""Provisional profile and CPU script contract evidence."""

import hashlib
import inspect
from pathlib import Path
import runpy
from typing import cast
import unittest
from unittest import mock

import nbformat  # pyright: ignore[reportMissingImports]
import torch

from tensor_dslab import (
    ChannelAxis,
    DarkCountRate,
    ExampleAxis,
    PsdNoiseConfig,
    Pulse,
    ReadoutConfig,
    SampleAxis,
    quantities,
)
from tensor_dslab.readout.profiles import ds20k_veto


READOUT_NOTEBOOK_SHA256 = (
    "c37e089c336d5c7254e64a12f8d920bb9fa0eea013d91ef77cfd4bbb83c301f5"
)


class ReadoutProfileContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_axis = SampleAxis(start=0, step=2000, count=5000)
        self.channel_axis = ChannelAxis(labels=("a", "b"))
        self.example_axis = ExampleAxis(count=2)

    def _profile(self) -> ReadoutConfig:
        return ds20k_veto(
            sample_axis=self.sample_axis,
            channel_axis=self.channel_axis,
            example_axis=self.example_axis,
        )

    def test_exact_profile_signature(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(ds20k_veto).parameters),
            ("sample_axis", "channel_axis", "example_axis"),
        )
        self.assertEqual(
            inspect.signature(ds20k_veto).parameters["sample_axis"].default,
            inspect.Parameter.empty,
        )

    def test_profile_values_and_global_dark_rate_are_exact(self) -> None:
        profile = self._profile()
        charge = profile.charge
        digitized = profile.digitized_waveform
        assert charge is not None and charge.dark_counts is not None
        assert digitized is not None
        self.assertIs(type(charge.dark_counts), DarkCountRate)
        self.assertEqual(float(charge.dark_counts.tensor), 100_000.0)
        self.assertEqual(charge.correlated_avalanche_generations.value, 0)
        self.assertEqual(digitized.bit_depth.value, 16)
        self.assertEqual(float(digitized.input_minimum.magnitude), -3900.0)
        self.assertEqual(float(digitized.input_maximum.magnitude), 100.0)

    def test_profile_pulse_is_signed_literal_and_capped_to_available_samples(self) -> None:
        pure = self._profile().pure_waveform
        assert pure is not None
        pulse = pure.pulse
        self.assertIs(type(pulse), Pulse)
        self.assertEqual(pulse.operation_axes[0].offsets, tuple(range(pulse.tensor.numel())))
        self.assertLessEqual(pulse.tensor.numel(), self.sample_axis.count)
        self.assertAlmostEqual(float(torch.amin(pulse.tensor)), -14.5912372)
        self.assertTrue(torch.all(pulse.tensor <= 0))

    def test_short_profile_pulse_is_exact_full_support_prefix(self) -> None:
        full = self._profile().pure_waveform
        short = ds20k_veto(
            sample_axis=SampleAxis(start=0, step=2000, count=64)
        ).pure_waveform
        assert full is not None and short is not None
        self.assertTrue(
            torch.equal(short.pulse.tensor, full.pulse.tensor[:64])
        )

    def test_every_call_returns_a_fresh_complete_tree(self) -> None:
        left = self._profile()
        right = self._profile()
        assert left.charge is not None and right.charge is not None
        assert left.charge.dark_counts is not None and right.charge.dark_counts is not None
        assert left.pure_waveform is not None and right.pure_waveform is not None
        self.assertIsNot(left, right)
        self.assertIsNot(left.charge, right.charge)
        self.assertIsNot(left.charge.dark_counts, right.charge.dark_counts)
        self.assertIsNot(left.pure_waveform.pulse, right.pure_waveform.pulse)
        self.assertNotEqual(
            left.pure_waveform.pulse.tensor.untyped_storage().data_ptr(),
            right.pure_waveform.pulse.tensor.untyped_storage().data_ptr(),
        )
        assert left.noise_waveform is not None and right.noise_waveform is not None
        assert isinstance(left.noise_waveform.model, PsdNoiseConfig)
        assert isinstance(right.noise_waveform.model, PsdNoiseConfig)
        assert left.analog_waveform is not None and right.analog_waveform is not None
        assert left.digitized_waveform is not None and right.digitized_waveform is not None
        self.assertIsNot(left.noise_waveform, right.noise_waveform)
        self.assertIsNot(left.noise_waveform.model, right.noise_waveform.model)
        self.assertIsNot(left.analog_waveform, right.analog_waveform)
        self.assertIsNot(left.digitized_waveform, right.digitized_waveform)
        self.assertIsNot(
            left.noise_waveform.model.frequency_left_edges,
            right.noise_waveform.model.frequency_left_edges,
        )
        self.assertIsNot(
            left.noise_waveform.model.power_density,
            right.noise_waveform.model.power_density,
        )
        self.assertIsNot(
            left.pure_waveform.pulse.operation_axes,
            right.pure_waveform.pulse.operation_axes,
        )
        self.assertIsNot(
            left.pure_waveform.pulse.operation_axes[0],
            right.pure_waveform.pulse.operation_axes[0],
        )

    def test_omitted_optional_axes_are_admitted_for_global_profile(self) -> None:
        profile = ds20k_veto(sample_axis=self.sample_axis)
        assert profile.charge is not None and profile.charge.dark_counts is not None
        assert profile.pure_waveform is not None
        self.assertEqual(profile.charge.dark_counts.conditioning_axes, ())
        self.assertEqual(profile.pure_waveform.pulse.conditioning_axes, ())

    def test_profile_rejects_conditioning_on_an_omitted_axis(self) -> None:
        global_pulse = ds20k_veto(
            sample_axis=self.sample_axis
        ).pure_waveform
        assert global_pulse is not None
        conditioned = Pulse(
            quantity=quantities(
                torch.stack(
                    (
                        global_pulse.pulse.tensor,
                        global_pulse.pulse.tensor,
                    )
                ),
                "mV",
            ),
            conditioning_axes=(self.channel_axis,),
            operation_axes=global_pulse.pulse.operation_axes,
        )
        with mock.patch(
            "tensor_dslab.readout.profiles._veto_pulse",
            return_value=conditioned,
        ):
            with self.assertRaises(ValueError):
                ds20k_veto(sample_axis=self.sample_axis)
            admitted = ds20k_veto(
                sample_axis=self.sample_axis,
                channel_axis=self.channel_axis,
            )
        assert admitted.pure_waveform is not None
        self.assertIs(
            admitted.pure_waveform.pulse.conditioning_axes[0],
            self.channel_axis,
        )

    def test_public_script_runs_cpu_only(self) -> None:
        with mock.patch("builtins.print"):
            namespace = runpy.run_path("demos/readout.py", run_name="not_main")
            namespace["main"]()

    def test_notebook_builds_an_independent_equivalent_config(self) -> None:
        notebook = nbformat.read("demos/readout.ipynb", as_version=4)
        code = {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}
        self.assertEqual(code["config"].count("ds20k_veto("), 1)
        for constructor in (
            "ReadoutConfig(",
            "ChargeConfig(",
            "DarkCountRate(",
            "Pulse(",
            "PureWaveformConfig(",
            "PsdNoiseConfig(",
            "DigitizedWaveformConfig(",
        ):
            self.assertIn(constructor, code["config"])
        namespace: dict[str, object] = {}
        with mock.patch("builtins.print"):
            for cell_id in ("imports", "source", "config"):
                exec(
                    compile(code[cell_id], f"<{cell_id}>", "exec"),
                    namespace,
                )
        profile = cast(ReadoutConfig, namespace["profile_config"])
        manual = cast(ReadoutConfig, namespace["manual_config"])
        assert profile.pure_waveform is not None
        assert manual.pure_waveform is not None
        assert profile.charge is not None
        assert manual.charge is not None
        assert profile.charge.dark_counts is not None
        assert manual.charge.dark_counts is not None
        self.assertIsNot(profile, manual)
        self.assertTrue(
            torch.equal(
                profile.pure_waveform.pulse.tensor,
                manual.pure_waveform.pulse.tensor,
            )
        )
        self.assertTrue(
            torch.equal(
                profile.charge.dark_counts.tensor,
                manual.charge.dark_counts.tensor,
            )
        )

    def test_notebook_is_executed_privacy_safe_and_timestamp_free(self) -> None:
        notebook = nbformat.read("demos/readout.ipynb", as_version=4)
        code = tuple(cell for cell in notebook.cells if cell.cell_type == "code")
        outputs = tuple(output for cell in code for output in cell.outputs)
        self.assertEqual(
            tuple(cell.execution_count for cell in code),
            tuple(range(1, len(code) + 1)),
        )
        self.assertEqual(len(outputs), 6)
        self.assertEqual(
            sum(
                output.output_type == "display_data"
                and "image/png" in output.get("data", {})
                for output in outputs
            ),
            1,
        )
        self.assertFalse(
            any(
                output.output_type == "error"
                for output in outputs
            )
        )
        payload = Path("demos/readout.ipynb").read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            READOUT_NOTEBOOK_SHA256,
        )
        text = payload.decode()
        for forbidden in (
            "/Users/",
            "/private/",
            "/scratch/",
            "iopub.execute_input",
            "shell.execute_reply",
        ):
            self.assertNotIn(forbidden, text)


for _count in range(2, 14):
    def _availability_case(
        self: ReadoutProfileContractTest,
        count: int = _count,
    ) -> None:
        axis = SampleAxis(start=0, step=2000, count=count)
        pure = ds20k_veto(sample_axis=axis).pure_waveform
        assert pure is not None
        pulse = pure.pulse
        self.assertEqual(pulse.tensor.numel(), count)
        self.assertEqual(pulse.operation_axes[0].size, count)

    setattr(ReadoutProfileContractTest, f"test_profile_availability_{_count:02d}", _availability_case)
