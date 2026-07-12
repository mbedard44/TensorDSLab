from __future__ import annotations

from tensor_core import IdSequence, TensorAxisId, TensorFieldId

EXAMPLE_AXIS_ID = TensorAxisId("example")
CHANNEL_AXIS_ID = TensorAxisId("channel")
SAMPLE_AXIS_ID = TensorAxisId("sample")

REQUIRED_AXIS_IDS = IdSequence(
    (
        EXAMPLE_AXIS_ID,
        CHANNEL_AXIS_ID,
        SAMPLE_AXIS_ID,
    )
)

READOUT_PHOTOELECTRONS_FIELD_ID = TensorFieldId("readout.photoelectrons")
READOUT_CHARGE_FIELD_ID = TensorFieldId("readout.charge")
READOUT_PURE_WAVEFORM_FIELD_ID = TensorFieldId("readout.waveform.pure")
READOUT_NOISE_WAVEFORM_FIELD_ID = TensorFieldId("readout.waveform.noise")
READOUT_ANALOG_WAVEFORM_FIELD_ID = TensorFieldId("readout.waveform.analog")
READOUT_DIGITIZED_WAVEFORM_FIELD_ID = TensorFieldId(
    "readout.waveform.digitized"
)

READOUT_FIELD_IDS = IdSequence(
    (
        READOUT_PHOTOELECTRONS_FIELD_ID,
        READOUT_CHARGE_FIELD_ID,
        READOUT_PURE_WAVEFORM_FIELD_ID,
        READOUT_NOISE_WAVEFORM_FIELD_ID,
        READOUT_ANALOG_WAVEFORM_FIELD_ID,
        READOUT_DIGITIZED_WAVEFORM_FIELD_ID,
    )
)
