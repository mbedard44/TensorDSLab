"""Raw-ZLE coefficient collection and staged configuration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, final

import torch

from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.config import require_config_components
from tensor_dslab.encoded_waveform.field import EncodedWaveformSpec
from tensor_dslab.encoded_waveform.kernel import EncodedWaveformKernels


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class EncodedWaveformConfig:
    """Describe one dense raw-ZLE waveform encoding."""

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    spec: EncodedWaveformSpec[Any]
    kernels: EncodedWaveformKernels
    _is_prepared: bool = field(default=False, init=False, repr=False)
    _source_specs: tuple[QuantityFieldSpec[Any], ...] = field(
        default=(), init=False, repr=False
    )
    _working_dtype: torch.dtype | None = field(
        default=None, init=False, repr=False
    )
    _time_dimension: int | None = field(default=None, init=False, repr=False)
    _kernel_dimensions: tuple[tuple[int, ...] | None, ...] = field(
        default=(None, None, None, None, None),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        require_config_components(
            spec=self.spec,
            kernels=self.kernels,
            spec_type=EncodedWaveformSpec,
            kernels_type=EncodedWaveformKernels,
            field="EncodedWaveformConfig",
        )
