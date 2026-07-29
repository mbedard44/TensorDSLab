"""Validate completed Charge relationships."""

from tensor_dslab.charge.config import ChargeConfig
from tensor_dslab.charge.field import Charge
from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.common.requirements.field import require_fresh_product


def validate_charge(*, product: Charge, sources: tuple, config: ChargeConfig) -> None:
    if type(product) is not Charge:
        raise TypeError("product must be exact Charge")
    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="ChargeConfig",
    )
    if product.spec is not config.spec:
        raise ValueError("Charge must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
