from __future__ import annotations

from dataclasses import dataclass
from typing import final

from tensor_core import (
    TensorCollection,
    TensorField,
    require_field_types,
    require_same_axes,
    require_same_device,
)

from tensor_dslab.common import SamplingConfig
from tensor_dslab.readout.analog_waveform import (
    AnalogWaveform,
    AnalogWaveformConfig,
)
from tensor_dslab.readout.charge import Charge, ChargeConfig
from tensor_dslab.readout.digitized_waveform import (
    DigitizedWaveform,
    DigitizedWaveformConfig,
)
from tensor_dslab.readout.noise_waveform import (
    NoiseWaveform,
    NoiseWaveformConfig,
)
from tensor_dslab.readout.photoelectrons import Photoelectrons
from tensor_dslab.readout.pure_waveform import (
    PureWaveform,
    PureWaveformConfig,
)
from tensor_dslab.readout._requirements import (
    _require_exact,
    _require_optional_exact,
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutConfig:
    sampling: SamplingConfig
    charge: ChargeConfig | None = None
    pure_waveform: PureWaveformConfig | None = None
    noise_waveform: NoiseWaveformConfig | None = None
    analog_waveform: AnalogWaveformConfig | None = None
    digitized_waveform: DigitizedWaveformConfig | None = None

    def __post_init__(self) -> None:
        _require_exact(self.sampling, SamplingConfig, "ReadoutConfig.sampling")
        _require_optional_exact(
            self.charge,
            ChargeConfig,
            "ReadoutConfig.charge",
        )
        _require_optional_exact(
            self.pure_waveform,
            PureWaveformConfig,
            "ReadoutConfig.pure_waveform",
        )
        _require_optional_exact(
            self.noise_waveform,
            NoiseWaveformConfig,
            "ReadoutConfig.noise_waveform",
        )
        _require_optional_exact(
            self.analog_waveform,
            AnalogWaveformConfig,
            "ReadoutConfig.analog_waveform",
        )
        _require_optional_exact(
            self.digitized_waveform,
            DigitizedWaveformConfig,
            "ReadoutConfig.digitized_waveform",
        )


@final
class ReadoutCollection(TensorCollection):
    __slots__ = ()

    @classmethod
    def accepted_field_types(cls) -> frozenset[type[TensorField]]:
        return frozenset(
            {
                Photoelectrons,
                Charge,
                PureWaveform,
                NoiseWaveform,
                AnalogWaveform,
                DigitizedWaveform,
            }
        )

    def _require(self) -> None:
        if not self.field_types:
            raise ValueError("ReadoutCollection must be nonempty")
        require_field_types(
            self,
            required=frozenset(),
            optional=self.accepted_field_types(),
        )

        fields = tuple(self.fields.values())
        require_same_axes(*fields)
        require_same_device(*fields)

        floating_dtypes = {
            field.tensor.dtype
            for field in fields
            if field.tensor.is_floating_point()
        }
        if len(floating_dtypes) > 1:
            raise ValueError("readout floating fields must share one dtype")
