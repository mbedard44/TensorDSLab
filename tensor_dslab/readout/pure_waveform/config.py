"""Public configuration record for pure waveform rendering."""

from dataclasses import dataclass
from typing import final

from tensor_dslab.readout.pure_waveform.kernel import Pulse


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PureWaveformConfig:
    """Select one literal sampled pulse kernel."""

    pulse: Pulse
    __hash__ = None  # pyright: ignore[reportAssignmentType]
