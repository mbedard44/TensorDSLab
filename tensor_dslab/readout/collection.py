from __future__ import annotations

from typing import final

from tensor_core import (
    TensorCollection,
    TensorField,
    require_field_types,
    require_same_axes,
    require_same_device,
    require_same_dtype,
)

from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.pure_waveform.field import PureWaveform


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

        floating_fields = tuple(
            field
            for field in fields
            if type(field)
            in (Charge, PureWaveform, NoiseWaveform, AnalogWaveform)
        )
        require_same_dtype(*floating_fields)
