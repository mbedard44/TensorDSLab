"""Charge coefficient collection and transformation configuration."""

from dataclasses import dataclass, field as dataclass_field
from typing import Any, ClassVar, final

import torch
from tensor_core import NonnegativeInteger

from tensor_dslab.charge.field import ChargeSpec
from tensor_dslab.charge.kernel import ChargeKernels
from tensor_dslab.common import QuantityFieldSpec
from tensor_dslab.common.requirements.config import require_config_components


@final
@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class ChargeConfig:
    """Describe one Charge transformation and its prepared policy facts."""

    __hash__: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    spec: ChargeSpec[Any]
    kernels: ChargeKernels
    correlated_avalanche_generations: NonnegativeInteger
    _is_prepared: bool = dataclass_field(default=False, init=False, repr=False)
    _source_specs: tuple[QuantityFieldSpec[Any], ...] = dataclass_field(
        default=(), init=False, repr=False
    )
    _source_dimensions: tuple[tuple[int, ...], ...] = dataclass_field(
        default=(), init=False, repr=False
    )
    _source_scales: tuple[float, ...] = dataclass_field(
        default=(), init=False, repr=False
    )
    _working_dtype: torch.dtype | None = dataclass_field(
        default=None, init=False, repr=False
    )
    _kernel_dimensions: tuple[tuple[int, ...] | None, ...] = dataclass_field(
        default=(None, None, None, None, None, None),
        init=False,
        repr=False,
    )
    _temporal_dimension: int | None = dataclass_field(
        default=None, init=False, repr=False
    )
    _temporal_step_seconds: float | None = dataclass_field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        require_config_components(
            spec=self.spec,
            kernels=self.kernels,
            spec_type=ChargeSpec,
            kernels_type=ChargeKernels,
            field="ChargeConfig",
        )
        if type(self.correlated_avalanche_generations) is not NonnegativeInteger:
            raise TypeError(
                "ChargeConfig.correlated_avalanche_generations must be exact NonnegativeInteger"
            )
        enabled = any(
            member is not None
            for member in (
                self.kernels.direct_crosstalk,
                self.kernels.delayed_crosstalk,
                self.kernels.afterpulse,
            )
        )
        if enabled != (self.correlated_avalanche_generations.value > 0):
            raise ValueError(
                "branching kernels are present exactly when generations are positive"
            )
