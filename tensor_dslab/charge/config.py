"""Charge coefficient collection and transformation configuration."""

from dataclasses import dataclass, field as dataclass_field
from typing import Any, ClassVar, final

import torch
from tensor_core import NonnegativeInteger, TensorCollection, TensorKernel

from tensor_dslab.charge.field import ChargeSpec
from tensor_dslab.charge.kernel import (
    Afterpulse,
    DarkCountRate,
    DelayedCrosstalk,
    DirectCrosstalk,
    SmearingWidth,
    TimingJitter,
)
from tensor_dslab.common import QuantityFieldSpec


@final
class ChargeKernels(TensorCollection[TensorKernel[Any]]):
    """Hold the optional physical coefficient set for Charge."""

    __slots__ = ()

    def _require(self) -> None:
        admitted = {
            TimingJitter,
            DirectCrosstalk,
            DelayedCrosstalk,
            Afterpulse,
            DarkCountRate,
            SmearingWidth,
        }
        if any(type(member) not in admitted for member in self.members.values()):
            raise TypeError("ChargeKernels contains an unsupported member")

    def _optional[T](self, member_type: type[T]) -> T | None:
        return self.members.get(member_type)  # type: ignore[return-value]

    @property
    def timing_jitter(self) -> TimingJitter | None:
        return self._optional(TimingJitter)

    @property
    def direct_crosstalk(self) -> DirectCrosstalk | None:
        return self._optional(DirectCrosstalk)

    @property
    def delayed_crosstalk(self) -> DelayedCrosstalk | None:
        return self._optional(DelayedCrosstalk)

    @property
    def afterpulse(self) -> Afterpulse | None:
        return self._optional(Afterpulse)

    @property
    def dark_count_rate(self) -> DarkCountRate | None:
        return self._optional(DarkCountRate)

    @property
    def smearing_width(self) -> SmearingWidth | None:
        return self._optional(SmearingWidth)


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
        if type(self.spec) is not ChargeSpec:
            raise TypeError("ChargeConfig.spec must be exact ChargeSpec")
        if type(self.kernels) is not ChargeKernels:
            raise TypeError("ChargeConfig.kernels must be exact ChargeKernels")
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
