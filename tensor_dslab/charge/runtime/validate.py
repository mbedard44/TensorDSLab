"""Validate completed Charge relationships."""

from tensor_dslab.charge.config import ChargeConfig
from tensor_dslab.charge.field import Charge
from tensor_dslab.common.alignment import (
    require_fresh_product,
    require_prepared_sources,
)


def validate_charge(*, product: Charge, sources: tuple, config: ChargeConfig) -> None:
    if type(product) is not Charge:
        raise TypeError("product must be exact Charge")
    if not config._is_prepared or product.spec is not config.spec:
        raise ValueError("Charge must retain the prepared output Spec")
    require_prepared_sources(sources, source_specs=config._source_specs)
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
