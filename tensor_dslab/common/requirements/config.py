"""Requirements for Product Config composition and prepared provenance."""

from typing import Any

import torch
from tensor_core import TensorField

from tensor_dslab.common.field import QuantityFieldSpec


def require_config_components(
    *,
    spec: object,
    spec_type: type,
    kernels: object,
    kernels_type: type,
    field: str,
) -> None:
    """Require exact Product Spec and Kernels component types."""

    if type(spec) is not spec_type:
        raise TypeError(f"{field}.spec must be exactly {spec_type.__name__}")
    if type(kernels) is not kernels_type:
        raise TypeError(f"{field}.kernels must be exactly {kernels_type.__name__}")


def require_prepared_config(
    *,
    is_prepared: bool,
    working_dtype: object,
    field: str,
) -> None:
    """Require one Config to carry completed preparation state."""

    if type(is_prepared) is not bool:
        raise TypeError(f"{field}._is_prepared must be exactly bool")
    if not is_prepared or type(working_dtype) is not torch.dtype:
        raise ValueError(f"{field} must be prepared")


def require_prepared_sources(
    sources: tuple[TensorField[Any], ...],
    *,
    source_specs: tuple[QuantityFieldSpec[Any], ...],
) -> None:
    """Bind staged execution to exact ordered prepared source Specs."""

    if type(sources) is not tuple or len(sources) != len(source_specs):
        raise ValueError("sources do not match the prepared source count")
    for index, (source, prepared) in enumerate(zip(sources, source_specs)):
        if not isinstance(source, TensorField):
            raise TypeError(f"sources[{index}] must be a TensorField")
        if not isinstance(source.spec, QuantityFieldSpec):
            raise TypeError(
                f"sources[{index}].spec must be a QuantityFieldSpec"
            )
        if source.spec != prepared:
            raise ValueError(
                f"sources[{index}].spec differs from prepared provenance"
            )
