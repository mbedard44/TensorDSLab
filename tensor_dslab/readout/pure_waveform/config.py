from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_core import FiniteFloat, PositiveFloat

from tensor_dslab.readout.requirements import require_exact, require_one_of_exact


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TpcFebSnrPulseConfig:
    fast_time_constant_ns: PositiveFloat
    slow_time_constant_ns: PositiveFloat
    support_time_ns: PositiveFloat
    peak_voltage_mv_per_pe: FiniteFloat

    def __post_init__(self) -> None:
        require_exact(
            self.fast_time_constant_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.fast_time_constant_ns",
        )
        require_exact(
            self.slow_time_constant_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.slow_time_constant_ns",
        )
        require_exact(
            self.support_time_ns,
            PositiveFloat,
            "TpcFebSnrPulseConfig.support_time_ns",
        )
        require_exact(
            self.peak_voltage_mv_per_pe,
            FiniteFloat,
            "TpcFebSnrPulseConfig.peak_voltage_mv_per_pe",
        )
        if self.slow_time_constant_ns.value <= self.fast_time_constant_ns.value:
            raise ValueError("slow time constant must exceed fast time constant")
        if self.peak_voltage_mv_per_pe.value == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class VetoPduPulseConfig:
    gaussian_center_ns: FiniteFloat
    gaussian_width_ns: PositiveFloat
    edge_offset_1_ns: FiniteFloat
    edge_width_1_ns: PositiveFloat
    edge_offset_2_ns: FiniteFloat
    edge_width_2_ns: PositiveFloat
    support_time_ns: PositiveFloat
    peak_voltage_mv_per_pe: FiniteFloat

    def __post_init__(self) -> None:
        require_exact(
            self.gaussian_center_ns,
            FiniteFloat,
            "VetoPduPulseConfig.gaussian_center_ns",
        )
        require_exact(
            self.gaussian_width_ns,
            PositiveFloat,
            "VetoPduPulseConfig.gaussian_width_ns",
        )
        require_exact(
            self.edge_offset_1_ns,
            FiniteFloat,
            "VetoPduPulseConfig.edge_offset_1_ns",
        )
        require_exact(
            self.edge_width_1_ns,
            PositiveFloat,
            "VetoPduPulseConfig.edge_width_1_ns",
        )
        require_exact(
            self.edge_offset_2_ns,
            FiniteFloat,
            "VetoPduPulseConfig.edge_offset_2_ns",
        )
        require_exact(
            self.edge_width_2_ns,
            PositiveFloat,
            "VetoPduPulseConfig.edge_width_2_ns",
        )
        require_exact(
            self.support_time_ns,
            PositiveFloat,
            "VetoPduPulseConfig.support_time_ns",
        )
        require_exact(
            self.peak_voltage_mv_per_pe,
            FiniteFloat,
            "VetoPduPulseConfig.peak_voltage_mv_per_pe",
        )
        if self.peak_voltage_mv_per_pe.value == 0.0:
            raise ValueError("peak voltage must be nonzero")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PureWaveformConfig:
    model: TpcFebSnrPulseConfig | VetoPduPulseConfig

    def __post_init__(self) -> None:
        require_one_of_exact(
            self.model,
            (TpcFebSnrPulseConfig, VetoPduPulseConfig),
            "PureWaveformConfig.model",
        )
