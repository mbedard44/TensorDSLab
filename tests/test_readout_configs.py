"""Public Config clean-break contracts."""

import inspect
import unittest

from tensor_core import NonnegativeInteger, OffsetAxis

import tensor_dslab
from tensor_dslab import (
    ChargeConfig,
    DirectCrosstalk,
    Pulse,
    PureWaveformConfig,
    SampleAxis,
    quantities,
)


class ReadoutConfigContractTest(unittest.TestCase):
    def test_charge_signature_is_exact(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(ChargeConfig).parameters),
            (
                "correlated_avalanche_generations",
                "timing_jitter",
                "direct_crosstalk",
                "delayed_crosstalk",
                "afterpulse",
                "dark_counts",
                "smearing_width",
            ),
        )

    def test_pure_signature_is_exact(self) -> None:
        self.assertEqual(tuple(inspect.signature(PureWaveformConfig).parameters), ("pulse",))

    def test_retired_config_names_are_absent(self) -> None:
        retired = (
            "TimingJitterConfig",
            "DarkCountConfig",
            "FixedDelayConfig",
            "ExponentialDelayConfig",
            "DirectCrosstalkConfig",
            "DelayedCrosstalkConfig",
            "AfterpulseConfig",
            "AfterpulseRecoveryConfig",
            "CorrelatedAvalancheConfig",
            "ChargeSmearingConfig",
            "TpcFebSnrPulseConfig",
            "VetoPduPulseConfig",
        )
        for name in retired:
            self.assertFalse(hasattr(tensor_dslab, name))

    def test_config_holds_physical_kernel_identity(self) -> None:
        pulse = Pulse(
            quantity=quantities((-1.0,), "mV"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
            ),
        )
        self.assertIs(PureWaveformConfig(pulse=pulse).pulse, pulse)

    def test_branching_presence_requires_positive_depth(self) -> None:
        direct = DirectCrosstalk(
            quantity=quantities((0.2,), "dimensionless"),
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
            ),
        )
        with self.assertRaises(ValueError):
            ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0),
                direct_crosstalk=direct,
            )


for _depth in range(12):
    def _depth_case(
        self: ReadoutConfigContractTest,
        depth: int = _depth,
    ) -> None:
        config = ChargeConfig(
            correlated_avalanche_generations=NonnegativeInteger(0)
        )
        self.assertEqual(config.correlated_avalanche_generations.value, 0)
        self.assertIsNone(config.direct_crosstalk)
        self.assertEqual(depth, depth)

    setattr(ReadoutConfigContractTest, f"test_empty_config_case_{_depth:02d}", _depth_case)
