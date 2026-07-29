"""Validate completed EncodedWaveform relationships and exact support."""

import torch

from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.common.requirements.field import require_fresh_product
from tensor_dslab.digitized_waveform.field import DigitizedWaveform
from tensor_dslab.encoded_waveform.config import EncodedWaveformConfig
from tensor_dslab.encoded_waveform.field import EncodedWaveform


def _lane_values(
    tensor: torch.Tensor,
    dimensions: tuple[int, ...],
    *,
    shape: tuple[int, ...],
    time_dimension: int,
) -> torch.Tensor:
    """Broadcast one validator policy Kernel to one value per lane."""

    non_time_dimensions = tuple(
        dimension
        for dimension in range(len(shape))
        if dimension != time_dimension
    )
    target = [1] * len(non_time_dimensions)
    for source_dimension, output_dimension in enumerate(dimensions):
        target[non_time_dimensions.index(output_dimension)] = tensor.shape[
            source_dimension
        ]
    non_time_shape = tuple(shape[dimension] for dimension in non_time_dimensions)
    return tensor.reshape(target).expand(non_time_shape).reshape(-1, 1)


def _expected_support(
    source: torch.Tensor,
    *,
    trigger: torch.Tensor,
    release: torch.Tensor,
    required: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct exact support independently for Product validation."""

    lane_count, time_count = source.shape
    trigger_mask = source <= trigger
    release_mask = source <= release
    prefix = torch.cat(
        (
            torch.zeros(
                (lane_count, 1),
                dtype=torch.int64,
                device=source.device,
            ),
            trigger_mask.to(torch.int64).cumsum(dim=1),
        ),
        dim=1,
    )
    ends = torch.arange(
        1,
        time_count + 1,
        dtype=torch.int64,
        device=source.device,
    ).reshape(1, -1).expand(lane_count, -1)
    starts = torch.clamp(ends - required, min=0)
    qualifies = (
        (ends >= required)
        & (prefix[:, 1:] - prefix.gather(1, starts) == required)
    )
    release_starts = release_mask & ~torch.cat(
        (
            torch.zeros(
                (lane_count, 1),
                dtype=torch.bool,
                device=source.device,
            ),
            release_mask[:, :-1],
        ),
        dim=1,
    )
    component = release_starts.to(torch.int64).cumsum(dim=1)
    component = component + (
        torch.arange(
            lane_count,
            dtype=torch.int64,
            device=source.device,
        ).reshape(-1, 1)
        * (time_count + 1)
    )
    first = torch.full(
        (lane_count * (time_count + 1),),
        time_count,
        dtype=torch.int64,
        device=source.device,
    )
    locations = qualifies.nonzero(as_tuple=False)
    first.scatter_reduce_(
        0,
        component[qualifies],
        locations[:, 1] - required[locations[:, 0], 0] + 1,
        reduce="amin",
        include_self=True,
    )
    raw = release_mask & (
        torch.arange(
            time_count,
            dtype=torch.int64,
            device=source.device,
        ).reshape(1, -1)
        >= first[component]
    )
    transitions = torch.diff(
        torch.cat(
            (
                torch.zeros(
                    (lane_count, 1),
                    dtype=torch.int64,
                    device=source.device,
                ),
                raw.to(torch.int64),
                torch.zeros(
                    (lane_count, 1),
                    dtype=torch.int64,
                    device=source.device,
                ),
            ),
            dim=1,
        ),
        dim=1,
    )
    raw_starts = (transitions == 1).nonzero(as_tuple=False)
    raw_ends = (transitions == -1).nonzero(as_tuple=False)
    padded_starts = raw_starts[:, 1] - torch.minimum(
        pre[raw_starts[:, 0], 0],
        raw_starts[:, 1],
    )
    padded_ends = raw_ends[:, 1] + torch.minimum(
        post[raw_ends[:, 0], 0],
        time_count - raw_ends[:, 1],
    )
    events = torch.zeros(
        (lane_count, time_count + 1),
        dtype=torch.int64,
        device=source.device,
    )
    increments = torch.ones(
        raw_starts.shape[0],
        dtype=torch.int64,
        device=source.device,
    )
    events.index_put_(
        (raw_starts[:, 0], padded_starts),
        increments,
        accumulate=True,
    )
    events.index_put_(
        (raw_ends[:, 0], padded_ends),
        -increments,
        accumulate=True,
    )
    return events.cumsum(dim=1)[:, :time_count] > 0


def validate_encoded_waveform(
    *,
    product: EncodedWaveform,
    sources: tuple,
    config: EncodedWaveformConfig,
) -> None:
    """Validate exact raw-ZLE values and direct Product relationships."""

    if type(product) is not EncodedWaveform:
        raise TypeError("product must be exact EncodedWaveform")
    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="EncodedWaveformConfig",
    )
    if product.spec is not config.spec:
        raise ValueError("EncodedWaveform must retain the prepared output Spec")
    if type(sources) is not tuple or len(sources) != 1:
        raise ValueError("EncodedWaveform requires exactly one source")
    if type(sources[0]) is not DigitizedWaveform:
        raise TypeError(
            "EncodedWaveform source must be exactly DigitizedWaveform"
        )
    require_prepared_sources(sources, source_specs=config._source_specs)
    source = sources[0]
    if (
        product.spec.axes != source.spec.axes
        or product.tensor.shape != source.tensor.shape
        or product.tensor.device != source.tensor.device
        or product.tensor.dtype is not source.tensor.dtype
        or product.spec.unit != source.spec.unit
    ):
        raise ValueError(
            "EncodedWaveform must retain source geometry, device, dtype, and unit"
        )

    time_dimension = config._time_dimension
    if time_dimension is None:
        raise ValueError("EncodedWaveformConfig has incomplete prepared facts")
    time_count = source.tensor.shape[time_dimension]
    non_time_dimensions = tuple(
        dimension
        for dimension in range(source.tensor.ndim)
        if dimension != time_dimension
    )
    order = (*non_time_dimensions, time_dimension)
    lane_count = source.tensor.numel() // time_count if time_count else 0
    if time_count == 0 or lane_count == 0:
        if not torch.equal(product.tensor, source.tensor):
            raise ValueError("empty EncodedWaveform must preserve source values")
    else:
        lane_source = source.tensor.permute(order).reshape(
            lane_count,
            time_count,
        )
        lane_product = product.tensor.permute(order).reshape(
            lane_count,
            time_count,
        )
        dimensions = config._kernel_dimensions
        assert all(value is not None for value in dimensions)
        support = _expected_support(
            lane_source,
            trigger=_lane_values(
                config.kernels.trigger_threshold_code.tensor,
                dimensions[0],  # type: ignore[arg-type]
                shape=config.spec.shape,
                time_dimension=time_dimension,
            ),
            release=_lane_values(
                config.kernels.release_threshold_code.tensor,
                dimensions[1],  # type: ignore[arg-type]
                shape=config.spec.shape,
                time_dimension=time_dimension,
            ),
            required=_lane_values(
                config.kernels.required_time_over_samples.tensor,
                dimensions[2],  # type: ignore[arg-type]
                shape=config.spec.shape,
                time_dimension=time_dimension,
            ),
            pre=_lane_values(
                config.kernels.pre_trigger_samples.tensor,
                dimensions[3],  # type: ignore[arg-type]
                shape=config.spec.shape,
                time_dimension=time_dimension,
            ),
            post=_lane_values(
                config.kernels.post_trigger_samples.tensor,
                dimensions[4],  # type: ignore[arg-type]
                shape=config.spec.shape,
                time_dimension=time_dimension,
            ),
        )
        suppression = torch.tensor(
            config.spec.suppression_code,
            dtype=product.tensor.dtype,
            device=product.tensor.device,
        )
        expected = torch.where(support, lane_source, suppression)
        if not torch.equal(lane_product, expected):
            raise ValueError(
                "EncodedWaveform values do not match exact raw-ZLE support"
            )
    require_fresh_product(
        product,
        sources=sources,
        kernels=tuple(config.kernels.members.values()),
    )
