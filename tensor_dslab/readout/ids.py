from __future__ import annotations

from tensor_core import IdSequence, TensorAxisId, TensorFieldId

READOUT_EXAMPLE_AXIS_ID = TensorAxisId("example")
READOUT_CHANNEL_AXIS_ID = TensorAxisId("channel")
READOUT_SAMPLE_AXIS_ID = TensorAxisId("sample")

READOUT_REQUIRED_AXIS_IDS = IdSequence(
    (
        READOUT_EXAMPLE_AXIS_ID,
        READOUT_CHANNEL_AXIS_ID,
        READOUT_SAMPLE_AXIS_ID,
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
