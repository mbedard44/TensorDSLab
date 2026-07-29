"""Execute dense tensor-native raw-ZLE waveform encoding."""

import math

import torch

from tensor_dslab.common.requirements.config import (
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.digitized_waveform.field import DigitizedWaveform
from tensor_dslab.encoded_waveform.config import EncodedWaveformConfig


def _lane_values(
    tensor: torch.Tensor,
    dimensions: tuple[int, ...],
    *,
    shape: tuple[int, ...],
    time_dimension: int,
) -> torch.Tensor:
    """Broadcast one aligned policy Kernel to one value per lane."""

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


def _support(
    source: torch.Tensor,
    *,
    trigger: torch.Tensor,
    release: torch.Tensor,
    required: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Construct exact dense raw-ZLE support without host state."""

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
    counts = prefix[:, 1:] - prefix.gather(1, starts)
    qualifying = (ends >= required) & (counts == required)

    previous_release = torch.cat(
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
    local_components = (
        release_mask & ~previous_release
    ).to(torch.int64).cumsum(dim=1)
    lane_offsets = (
        torch.arange(
            lane_count,
            dtype=torch.int64,
            device=source.device,
        ).reshape(-1, 1)
        * (time_count + 1)
    )
    global_components = local_components + lane_offsets
    first_start = torch.full(
        (lane_count * (time_count + 1),),
        time_count,
        dtype=torch.int64,
        device=source.device,
    )
    qualifying_indices = qualifying.nonzero(as_tuple=False)
    qualifying_components = global_components[qualifying]
    qualifying_starts = (
        qualifying_indices[:, 1]
        - required[qualifying_indices[:, 0], 0]
        + 1
    )
    first_start.scatter_reduce_(
        0,
        qualifying_components,
        qualifying_starts,
        reduce="amin",
        include_self=True,
    )
    sample_indices = torch.arange(
        time_count,
        dtype=torch.int64,
        device=source.device,
    ).reshape(1, -1)
    raw = (
        release_mask
        & (sample_indices >= first_start[global_components])
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
    start_extension = torch.minimum(
        pre[raw_starts[:, 0], 0],
        raw_starts[:, 1],
    )
    end_extension = torch.minimum(
        post[raw_ends[:, 0], 0],
        time_count - raw_ends[:, 1],
    )
    padded_starts = raw_starts[:, 1] - start_extension
    padded_ends = raw_ends[:, 1] + end_extension
    events = torch.zeros(
        (lane_count, time_count + 1),
        dtype=torch.int64,
        device=source.device,
    )
    ones = torch.ones(
        raw_starts.shape[0],
        dtype=torch.int64,
        device=source.device,
    )
    events.index_put_(
        (raw_starts[:, 0], padded_starts),
        ones,
        accumulate=True,
    )
    events.index_put_(
        (raw_ends[:, 0], padded_ends),
        -ones,
        accumulate=True,
    )
    return events.cumsum(dim=1)[:, :time_count] > 0


def produce_encoded_waveform(
    *,
    sources: tuple,
    config: EncodedWaveformConfig,
) -> torch.Tensor:
    """Produce one dense sentinel-coded waveform from one digitized source."""

    require_prepared_config(
        is_prepared=config._is_prepared,
        working_dtype=config._working_dtype,
        field="EncodedWaveformConfig",
    )
    if config._time_dimension is None or any(
        dimensions is None for dimensions in config._kernel_dimensions
    ):
        raise ValueError("EncodedWaveformConfig has incomplete prepared facts")
    if type(sources) is not tuple or len(sources) != 1:
        raise ValueError("EncodedWaveform requires exactly one source")
    if type(sources[0]) is not DigitizedWaveform:
        raise TypeError(
            "EncodedWaveform source must be exactly DigitizedWaveform"
        )
    require_prepared_sources(sources, source_specs=config._source_specs)

    source = sources[0].tensor
    time_dimension = config._time_dimension
    assert time_dimension is not None
    time_count = source.shape[time_dimension]
    non_time_dimensions = tuple(
        dimension
        for dimension in range(source.ndim)
        if dimension != time_dimension
    )
    order = (*non_time_dimensions, time_dimension)
    inverse = tuple(order.index(dimension) for dimension in range(source.ndim))
    lane_count = math.prod(source.shape[dimension] for dimension in non_time_dimensions)
    lane_source = source.permute(order).reshape(lane_count, time_count)
    if lane_count == 0 or time_count == 0:
        return source.clone().contiguous()

    dimensions = config._kernel_dimensions
    assert all(value is not None for value in dimensions)
    trigger = _lane_values(
        config.kernels.trigger_threshold_code.tensor,
        dimensions[0],  # type: ignore[arg-type]
        shape=config.spec.shape,
        time_dimension=time_dimension,
    )
    release = _lane_values(
        config.kernels.release_threshold_code.tensor,
        dimensions[1],  # type: ignore[arg-type]
        shape=config.spec.shape,
        time_dimension=time_dimension,
    )
    required = _lane_values(
        config.kernels.required_time_over_samples.tensor,
        dimensions[2],  # type: ignore[arg-type]
        shape=config.spec.shape,
        time_dimension=time_dimension,
    )
    pre = _lane_values(
        config.kernels.pre_trigger_samples.tensor,
        dimensions[3],  # type: ignore[arg-type]
        shape=config.spec.shape,
        time_dimension=time_dimension,
    )
    post = _lane_values(
        config.kernels.post_trigger_samples.tensor,
        dimensions[4],  # type: ignore[arg-type]
        shape=config.spec.shape,
        time_dimension=time_dimension,
    )
    support = _support(
        lane_source,
        trigger=trigger,
        release=release,
        required=required,
        pre=pre,
        post=post,
    )
    suppression = torch.tensor(
        config.spec.suppression_code,
        dtype=source.dtype,
        device=source.device,
    )
    encoded = torch.where(support, lane_source, suppression)
    moved_shape = tuple(source.shape[dimension] for dimension in order)
    return encoded.reshape(moved_shape).permute(inverse).contiguous()
