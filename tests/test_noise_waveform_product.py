from __future__ import annotations

from collections.abc import Iterable
from itertools import permutations
import math
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    CounterRng,
    NonnegativeFloat,
    PositiveFloat,
    PositiveInteger,
    RngKey,
    TensorAxis,
    Threefry4x32,
    logical_positions,
)

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    SampleAxis,
    SamplingConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.noise_waveform._produce import (
    _prepare_psd_powers,
    _produce_noise_waveform,
)


SEEDS = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
AXIS_ORDERS = tuple(
    permutations((ExampleAxis, ChannelAxis, SampleAxis))
)


class _FailingRng(CounterRng):
    __slots__ = ()

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        raise AssertionError(
            f"unexpected RNG request: {key=}, {quantum=}, {block=}"
        )


def _sampling(*, count: int, period_ps: int = 1_000) -> SamplingConfig:
    return SamplingConfig(
        sample_period_ps=PositiveInteger(period_ps),
        sample_count=PositiveInteger(count),
    )


def _axes(
    sampling: SamplingConfig,
    *,
    order: tuple[type[TensorAxis], ...] = (
        ExampleAxis,
        ChannelAxis,
        SampleAxis,
    ),
    examples: int = 2,
    channels: int = 3,
    label_prefix: str = "original",
) -> tuple[TensorAxis, ...]:
    available: dict[type[TensorAxis], TensorAxis] = {
        ExampleAxis: ExampleAxis(
            coordinates=tuple(f"{label_prefix}-example-{index}" for index in range(examples))
        ),
        ChannelAxis: ChannelAxis(
            coordinates=tuple(f"{label_prefix}-channel-{index}" for index in range(channels))
        ),
        SampleAxis: sampling.build_axis(),
    }
    return tuple(available[axis_type] for axis_type in order)


def _photoelectrons(
    sampling: SamplingConfig,
    *,
    order: tuple[type[TensorAxis], ...] = (
        ExampleAxis,
        ChannelAxis,
        SampleAxis,
    ),
    examples: int = 2,
    channels: int = 3,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
    label_prefix: str = "original",
    fill_offset: int = 0,
) -> Photoelectrons:
    axes = _axes(
        sampling,
        order=order,
        examples=examples,
        channels=channels,
        label_prefix=label_prefix,
    )
    shape = tuple(axis.size for axis in axes)
    values = torch.arange(
        math.prod(shape),
        dtype=torch.int64,
        device=device,
    ).reshape(shape)
    values = values + fill_offset
    if noncontiguous:
        backing = torch.empty((*shape, 2), dtype=torch.int64, device=device)
        view = backing[..., 0]
        view.copy_(values)
        values = view
    return Photoelectrons(tensor=values, axes=axes)


def _zero_config() -> NoiseWaveformConfig:
    return NoiseWaveformConfig(model=ZeroNoiseConfig())


def _white_config(rms_mv: float = 1.0) -> NoiseWaveformConfig:
    return NoiseWaveformConfig(
        model=WhiteNoiseConfig(rms_mv=PositiveFloat(rms_mv))
    )


def _flat_psd_config(
    *,
    density: float = 2.0e-9,
    stop_hz: float = 500_000_000.0,
) -> NoiseWaveformConfig:
    return NoiseWaveformConfig(
        model=PsdNoiseConfig(
            frequency_left_edges_hz=(NonnegativeFloat(0.0),),
            frequency_stop_hz=PositiveFloat(stop_hz),
            power_density_mv2_per_hz=(NonnegativeFloat(density),),
        )
    )


def _sample_last(field: NoiseWaveform) -> torch.Tensor:
    return field.tensor.movedim(field.dimension_of(SampleAxis), -1)


def _psd_normals(
    model: PsdNoiseConfig,
    *,
    seed: int,
    row_count: int,
    frequency_count: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    positions = logical_positions(
        (row_count, frequency_count),
        device=device,
    )[:, 1:]
    return Threefry4x32(seed=seed).gaussian(
        mean=0.0,
        standard_deviation=1.0,
        key=model.rng_key,
        positions=positions,
        dtype=dtype,
        quantum=0,
        ordinal=0,
        count=2,
    )


def _independent_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()


def _round(value: float, dtype: torch.dtype) -> float:
    return float(torch.tensor(value, dtype=dtype))


def _reference_psd_powers(
    config: PsdNoiseConfig,
    *,
    sampling: SamplingConfig,
    dtype: torch.dtype,
) -> tuple[float, ...]:
    sample_count = sampling.sample_count.value
    sample_rate = 1.0e12 / sampling.sample_period_ps.value
    spacing = sample_rate / sample_count
    nyquist = sample_rate / 2.0
    frequency_count = sample_count // 2 + 1
    target_left = (0.0,) + tuple(
        (index - 0.5) * spacing for index in range(1, frequency_count)
    )
    target_right = target_left[1:] + (nyquist,)
    source_left = tuple(item.value for item in config.frequency_left_edges_hz)
    source_right = source_left[1:] + (config.frequency_stop_hz.value,)
    density = tuple(item.value for item in config.power_density_mv2_per_hz)
    integrated = tuple(
        math.fsum(
            source_power
            * max(
                0.0,
                min(source_stop, target_stop)
                - max(source_start, target_start),
            )
            for source_start, source_stop, source_power in zip(
                source_left,
                source_right,
                density,
            )
        )
        for target_start, target_stop in zip(target_left, target_right)
    )
    return (0.0,) + tuple(_round(power, dtype) for power in integrated[1:])


def _delta(dtype: torch.dtype, scale: float, length: int) -> float:
    return (
        64
        * torch.finfo(dtype).eps
        * max(1, math.ceil(math.log2(length)))
        * abs(scale)
    )


class NoiseProductBranchTest(unittest.TestCase):
    def test_every_model_dtype_axis_order_and_noncontiguous_source(self) -> None:
        sampling = _sampling(count=5)
        configs = (_zero_config(), _white_config(), _flat_psd_config())
        for order in AXIS_ORDERS:
            for dtype in (torch.float32, torch.float64):
                source = _photoelectrons(
                    sampling,
                    order=order,
                    examples=2,
                    channels=2,
                    noncontiguous=True,
                )
                self.assertFalse(source.tensor.is_contiguous())
                source_values = source.tensor.clone()
                for config in configs:
                    with self.subTest(order=order, dtype=dtype, model=type(config.model)):
                        result = _produce_noise_waveform(
                            source,
                            sampling=sampling,
                            config=config,
                            rng=Threefry4x32(seed=17),
                            floating_dtype=dtype,
                        )
                        self.assertIs(type(result), NoiseWaveform)
                        self.assertIs(result.axes, source.axes)
                        for actual, expected in zip(result.axes, source.axes):
                            self.assertIs(actual, expected)
                        self.assertEqual(result.shape, source.shape)
                        self.assertEqual(result.tensor.device, source.tensor.device)
                        self.assertIs(result.tensor.dtype, dtype)
                        self.assertIs(result.tensor.layout, torch.strided)
                        self.assertFalse(result.tensor.requires_grad)
                        self.assertIsNone(result.tensor.grad_fn)
                        self.assertTrue(bool(torch.all(torch.isfinite(result.tensor))))
                        self.assertTrue(_independent_storage(result.tensor, source.tensor))
                self.assertTrue(torch.equal(source.tensor, source_values))

    def test_zero_is_fresh_exact_rng_inert_and_never_calls_rng(self) -> None:
        sampling = _sampling(count=8)
        source = _photoelectrons(sampling)
        state = torch.random.get_rng_state().clone()
        with patch(
            "tensor_dslab.readout.noise_waveform._produce.logical_positions",
            side_effect=AssertionError("zero noise must not build positions"),
        ) as positions:
            first = _produce_noise_waveform(
                source,
                sampling=sampling,
                config=_zero_config(),
                rng=_FailingRng(seed=0),
                floating_dtype=torch.float32,
            )
            second = _produce_noise_waveform(
                source,
                sampling=sampling,
                config=_zero_config(),
                rng=_FailingRng(seed=(1 << 64) - 1),
                floating_dtype=torch.float32,
            )
        positions.assert_not_called()
        self.assertTrue(torch.equal(first.tensor, torch.zeros_like(first.tensor)))
        self.assertTrue(torch.equal(first.tensor, second.tensor))
        self.assertTrue(_independent_storage(first.tensor, second.tensor))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_rng_dtype_sampling_and_device_fail_before_rng(self) -> None:
        sampling = _sampling(count=8)
        source = _photoelectrons(sampling)
        for config in (_zero_config(), _white_config(), _flat_psd_config()):
            for invalid_rng in (None, object(), True):
                with self.subTest(
                    model=type(config.model),
                    invalid_rng=invalid_rng,
                ):
                    with self.assertRaises(TypeError):
                        _produce_noise_waveform(
                            source,
                            sampling=sampling,
                            config=config,
                            rng=invalid_rng,  # type: ignore[arg-type]
                            floating_dtype=torch.float32,
                        )
        failing_rng = _FailingRng(seed=0)
        with self.assertRaises(TypeError):
            _produce_noise_waveform(
                source,
                sampling=sampling,
                config=_white_config(),
                rng=failing_rng,
                floating_dtype=torch.float16,
            )
        with self.assertRaises(ValueError):
            _produce_noise_waveform(
                source,
                sampling=_sampling(count=8, period_ps=2_000),
                config=_white_config(),
                rng=failing_rng,
                floating_dtype=torch.float32,
            )

        meta_source = _photoelectrons(sampling, device="meta")
        with self.assertRaises(ValueError):
            _produce_noise_waveform(
                meta_source,
                sampling=sampling,
                config=_zero_config(),
                rng=failing_rng,
                floating_dtype=torch.float32,
            )

    def test_white_matches_finite_lattice_equation_without_demeaning(self) -> None:
        sampling = _sampling(count=7)
        source = _photoelectrons(sampling, examples=1, channels=1)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                rms_input = 0.1
                represented_rms = _round(rms_input, dtype)
                result = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=_white_config(rms_input),
                    rng=Threefry4x32(seed=0x0123456789ABCDEF),
                    floating_dtype=dtype,
                )
                model = _white_config(rms_input).model
                assert type(model) is WhiteNoiseConfig
                positions = logical_positions(
                    source.shape,
                    device=source.tensor.device,
                )
                expected = Threefry4x32(
                    seed=0x0123456789ABCDEF
                ).gaussian(
                    mean=0.0,
                    standard_deviation=represented_rms,
                    key=model.rng_key,
                    positions=positions,
                    dtype=dtype,
                    quantum=0,
                    ordinal=0,
                    count=1,
                )
                self.assertTrue(torch.equal(result.tensor, expected))
                self.assertNotEqual(float(torch.mean(result.tensor)), 0.0)

    def test_white_is_repeatable_stream_isolated_and_source_value_independent(self) -> None:
        sampling = _sampling(count=9)
        first_source = _photoelectrons(sampling, fill_offset=0)
        second_source = _photoelectrons(
            sampling,
            fill_offset=10_000,
            label_prefix="renamed",
        )
        first = _produce_noise_waveform(
            first_source,
            sampling=sampling,
            config=_white_config(),
            rng=Threefry4x32(seed=99),
            floating_dtype=torch.float64,
        )
        repeated = _produce_noise_waveform(
            first_source,
            sampling=sampling,
            config=_white_config(),
            rng=Threefry4x32(seed=99),
            floating_dtype=torch.float64,
        )
        relabeled = _produce_noise_waveform(
            second_source,
            sampling=sampling,
            config=_white_config(),
            rng=Threefry4x32(seed=99),
            floating_dtype=torch.float64,
        )
        self.assertTrue(torch.equal(first.tensor, repeated.tensor))
        self.assertTrue(torch.equal(first.tensor, relabeled.tensor))
        positions = logical_positions(first_source.shape, device="cpu")
        psd_model = _flat_psd_config().model
        assert type(psd_model) is PsdNoiseConfig
        other_stream = Threefry4x32(seed=99).gaussian(
            mean=0.0,
            standard_deviation=1.0,
            key=psd_model.rng_key,
            positions=positions,
            dtype=torch.float64,
            quantum=0,
            ordinal=0,
            count=1,
        )
        self.assertFalse(torch.equal(first.tensor, other_stream))

    def test_float32_stochastic_products_ignore_ambient_cpu_autocast(self) -> None:
        sampling = _sampling(count=8)
        source = _photoelectrons(sampling)
        for config in (_white_config(), _flat_psd_config()):
            expected = _produce_noise_waveform(
                source,
                sampling=sampling,
                config=config,
                rng=Threefry4x32(seed=71),
                floating_dtype=torch.float32,
            )
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                actual = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=config,
                    rng=Threefry4x32(seed=71),
                    floating_dtype=torch.float32,
                )
            self.assertIs(actual.tensor.dtype, torch.float32)
            self.assertTrue(torch.equal(actual.tensor, expected.tensor))

    def test_white_normal_range_and_conservative_upper_bound(self) -> None:
        sampling = _sampling(count=2)
        source = _photoelectrons(sampling, examples=1, channels=1)
        for dtype, guard in ((torch.float32, 8.0), (torch.float64, 16.0)):
            finfo = torch.finfo(dtype)
            with self.subTest(dtype=dtype, boundary="tiny"):
                accepted = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=_white_config(finfo.tiny),
                    rng=Threefry4x32(seed=0),
                    floating_dtype=dtype,
                )
                self.assertTrue(bool(torch.all(torch.isfinite(accepted.tensor))))
            with self.subTest(dtype=dtype, boundary="subnormal"):
                with self.assertRaises(ValueError):
                    _produce_noise_waveform(
                        source,
                        sampling=sampling,
                        config=_white_config(finfo.tiny / 2.0),
                        rng=_FailingRng(seed=0),
                        floating_dtype=dtype,
                    )
            maximum_rms = float(torch.tensor(finfo.max / guard, dtype=dtype))
            with self.subTest(dtype=dtype, boundary="maximum"):
                accepted = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=_white_config(maximum_rms),
                    rng=Threefry4x32(seed=0),
                    floating_dtype=dtype,
                )
                self.assertTrue(bool(torch.all(torch.isfinite(accepted.tensor))))
            rejected_rms = float(
                torch.nextafter(
                    torch.tensor(maximum_rms, dtype=dtype),
                    torch.tensor(math.inf, dtype=dtype),
                )
            )
            with self.subTest(dtype=dtype, boundary="above maximum"):
                with self.assertRaises(ValueError):
                    _produce_noise_waveform(
                        source,
                        sampling=sampling,
                        config=_white_config(rejected_rms),
                        rng=_FailingRng(seed=0),
                        floating_dtype=dtype,
                    )


class PsdPreparationTest(unittest.TestCase):
    def test_odd_even_cells_overlap_fsum_conservation_and_one_rounding(self) -> None:
        for count in (5, 6):
            sampling = _sampling(count=count)
            sample_rate = 1.0e12 / sampling.sample_period_ps.value
            spacing = sample_rate / count
            edges = (0.0, spacing / 2.0, 1.75 * spacing, 0.45 * sample_rate)
            model = PsdNoiseConfig(
                frequency_left_edges_hz=tuple(NonnegativeFloat(value) for value in edges),
                frequency_stop_hz=PositiveFloat(sample_rate / 2.0),
                power_density_mv2_per_hz=tuple(
                    NonnegativeFloat(value) for value in (1.0e-9, 2.0e-9, 0.0, 3.0e-9)
                ),
            )
            for dtype in (torch.float32, torch.float64):
                with self.subTest(count=count, dtype=dtype):
                    expected = _reference_psd_powers(
                        model,
                        sampling=sampling,
                        dtype=dtype,
                    )
                    actual = _prepare_psd_powers(
                        model,
                        sampling=sampling,
                        dtype=dtype,
                    )
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual[0], 0.0)

            source_left = edges
            source_right = edges[1:] + (sample_rate / 2.0,)
            density = (1.0e-9, 2.0e-9, 0.0, 3.0e-9)
            full_power = math.fsum(
                value * (right - left)
                for left, right, value in zip(source_left, source_right, density)
            )
            dc_power = density[0] * (spacing / 2.0)
            binary64_retained = _reference_psd_powers(
                model,
                sampling=sampling,
                dtype=torch.float64,
            )
            self.assertAlmostEqual(sum(binary64_retained[1:]), full_power - dc_power)

    def test_coverage_dc_only_above_nyquist_and_rounded_zero_reject_before_rng(self) -> None:
        sampling = _sampling(count=4)
        source = _photoelectrons(sampling)
        invalid_models = (
            PsdNoiseConfig(
                frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                frequency_stop_hz=PositiveFloat(400_000_000.0),
                power_density_mv2_per_hz=(NonnegativeFloat(1.0e-9),),
            ),
            PsdNoiseConfig(
                frequency_left_edges_hz=(
                    NonnegativeFloat(0.0),
                    NonnegativeFloat(125_000_000.0),
                ),
                frequency_stop_hz=PositiveFloat(500_000_000.0),
                power_density_mv2_per_hz=(
                    NonnegativeFloat(1.0e-9),
                    NonnegativeFloat(0.0),
                ),
            ),
            PsdNoiseConfig(
                frequency_left_edges_hz=(
                    NonnegativeFloat(0.0),
                    NonnegativeFloat(500_000_000.0),
                ),
                frequency_stop_hz=PositiveFloat(600_000_000.0),
                power_density_mv2_per_hz=(
                    NonnegativeFloat(0.0),
                    NonnegativeFloat(1.0e-9),
                ),
            ),
            PsdNoiseConfig(
                frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                frequency_stop_hz=PositiveFloat(500_000_000.0),
                power_density_mv2_per_hz=(NonnegativeFloat(4.0e-59),),
            ),
        )
        for model in invalid_models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    _produce_noise_waveform(
                        source,
                        sampling=sampling,
                        config=NoiseWaveformConfig(model=model),
                        rng=_FailingRng(seed=0),
                        floating_dtype=torch.float32,
                    )

    def test_nonfinite_accumulation_guard_rejects_before_rng(self) -> None:
        sampling = _sampling(count=4)
        source = _photoelectrons(sampling)
        config = _flat_psd_config()
        original_fsum = math.fsum
        call_count = 0

        def fail_final_fsum(values: Iterable[float]) -> float:
            nonlocal call_count
            call_count += 1
            materialized = tuple(values)
            if call_count == sampling.sample_count.value // 2 + 2:
                return math.inf
            return original_fsum(materialized)

        with patch(
            "tensor_dslab.readout.noise_waveform._produce.math.fsum",
            side_effect=fail_final_fsum,
        ):
            with self.assertRaises(ValueError):
                _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=config,
                    rng=_FailingRng(seed=0),
                    floating_dtype=torch.float32,
                )

    def test_finite_accumulation_guard_accepts_limit_and_rejects_nextafter_before_rng(
        self,
    ) -> None:
        sampling = _sampling(count=4)
        source = _photoelectrons(sampling)
        config = _flat_psd_config()
        original_fsum = math.fsum
        final_call = sampling.sample_count.value // 2 + 2

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                normal_guard = 8.0 if dtype is torch.float32 else 16.0
                limit = torch.finfo(dtype).max / (
                    sampling.sample_count.value * normal_guard
                )
                above_limit = math.nextafter(limit, math.inf)
                self.assertTrue(math.isfinite(limit))
                self.assertLessEqual(
                    sampling.sample_count.value * normal_guard * limit,
                    torch.finfo(dtype).max,
                )
                self.assertGreater(
                    sampling.sample_count.value * normal_guard * above_limit,
                    torch.finfo(dtype).max,
                )

                accepted_call_count = 0

                def fsum_at_limit(values: Iterable[float]) -> float:
                    nonlocal accepted_call_count
                    accepted_call_count += 1
                    materialized = tuple(values)
                    if accepted_call_count == final_call:
                        return limit
                    return original_fsum(materialized)

                with patch(
                    "tensor_dslab.readout.noise_waveform._produce.math.fsum",
                    side_effect=fsum_at_limit,
                ):
                    accepted = _produce_noise_waveform(
                        source,
                        sampling=sampling,
                        config=config,
                        rng=Threefry4x32(seed=0),
                        floating_dtype=dtype,
                    )
                self.assertTrue(bool(torch.all(torch.isfinite(accepted.tensor))))
                self.assertEqual(accepted_call_count, final_call)

                rejected_call_count = 0

                def fsum_above_limit(values: Iterable[float]) -> float:
                    nonlocal rejected_call_count
                    rejected_call_count += 1
                    materialized = tuple(values)
                    if rejected_call_count == final_call:
                        return above_limit
                    return original_fsum(materialized)

                with patch(
                    "tensor_dslab.readout.noise_waveform._produce.math.fsum",
                    side_effect=fsum_above_limit,
                ):
                    with self.assertRaises(ValueError):
                        _produce_noise_waveform(
                            source,
                            sampling=sampling,
                            config=config,
                            rng=_FailingRng(seed=0),
                            floating_dtype=dtype,
                        )
                self.assertEqual(rejected_call_count, final_call)


class PsdSynthesisTest(unittest.TestCase):
    def test_small_odd_even_reference_coefficients_dc_and_zero_power(self) -> None:
        for count in (5, 6):
            sampling = _sampling(count=count)
            source = _photoelectrons(sampling, examples=1, channels=1)
            sample_rate = 1.0e12 / sampling.sample_period_ps.value
            spacing = sample_rate / count
            frequency_count = count // 2 + 1
            target_boundaries = (0.0,) + tuple(
                (index - 0.5) * spacing for index in range(1, frequency_count)
            ) + (sample_rate / 2.0,)
            densities = tuple(
                0.0 if index == 2 else 2.0e-9
                for index in range(frequency_count)
            )
            model = PsdNoiseConfig(
                frequency_left_edges_hz=tuple(
                    NonnegativeFloat(value) for value in target_boundaries[:-1]
                ),
                frequency_stop_hz=PositiveFloat(target_boundaries[-1]),
                power_density_mv2_per_hz=tuple(
                    NonnegativeFloat(value) for value in densities
                ),
            )
            for dtype in (torch.float32, torch.float64):
                powers = _prepare_psd_powers(model, sampling=sampling, dtype=dtype)
                normals = _psd_normals(
                    model,
                    seed=3,
                    row_count=1,
                    frequency_count=frequency_count,
                    dtype=dtype,
                )
                real = normals[..., 0]
                imaginary = normals[..., 1]
                captured_coefficients: list[torch.Tensor] = []
                original_irfft = torch.fft.irfft

                def capture_irfft(input: torch.Tensor, **kwargs: object) -> torch.Tensor:
                    captured_coefficients.append(input.clone())
                    return original_irfft(input, **kwargs)

                with patch(
                    "tensor_dslab.readout.noise_waveform._produce.torch.fft.irfft",
                    side_effect=capture_irfft,
                ):
                    result = _produce_noise_waveform(
                        source,
                        sampling=sampling,
                        config=NoiseWaveformConfig(model=model),
                        rng=Threefry4x32(seed=3),
                        floating_dtype=dtype,
                    )
                self.assertEqual(len(captured_coefficients), 1)
                coefficients = captured_coefficients[0]
                self.assertEqual(coefficients.shape, (1, frequency_count))
                self.assertEqual(complex(coefficients[0, 0]), 0j)
                self.assertEqual(complex(coefficients[0, 2]), 0j)
                interior_count = (count - 1) // 2
                scales = torch.tensor(
                    [count / 2.0 * math.sqrt(power) for power in powers[1 : interior_count + 1]],
                    dtype=dtype,
                )
                expected_interior = torch.complex(
                    real[:, :interior_count] * scales,
                    imaginary[:, :interior_count] * scales,
                )
                tolerance_scale = max(
                    float(torch.max(torch.abs(expected_interior))),
                    torch.finfo(dtype).tiny,
                )
                tolerance = (
                    64
                    * torch.finfo(dtype).eps
                    * max(1, math.ceil(math.log2(count)))
                    * tolerance_scale
                )
                self.assertTrue(
                    torch.allclose(
                        coefficients[:, 1 : interior_count + 1],
                        expected_interior,
                        rtol=0.0,
                        atol=tolerance,
                    )
                )
                if count % 2 == 0:
                    expected_nyquist = real[:, -1] * torch.tensor(
                        count * math.sqrt(powers[-1]),
                        dtype=dtype,
                    )
                    self.assertTrue(
                        torch.allclose(
                            coefficients[:, -1].real,
                            expected_nyquist,
                            rtol=0.0,
                            atol=tolerance,
                        )
                    )
                    self.assertTrue(torch.equal(coefficients[:, -1].imag, torch.zeros(1, dtype=dtype)))
                reconstructed = torch.fft.rfft(
                    _sample_last(result),
                    n=count,
                    dim=-1,
                    norm="backward",
                )
                self.assertTrue(
                    torch.allclose(reconstructed, coefficients, rtol=0.0, atol=tolerance)
                )
                mean_bound = (
                    64
                    * torch.finfo(dtype).eps
                    * max(1, math.ceil(math.log2(count)))
                    * max(float(torch.max(torch.abs(result.tensor))), torch.finfo(dtype).tiny)
                )
                self.assertLessEqual(abs(float(torch.mean(result.tensor))), mean_bound)

    def test_odd_terminal_imaginary_and_isolated_cosine_sine_bases(self) -> None:
        sample_count = 5
        sampling = _sampling(count=sample_count)
        source = _photoelectrons(sampling, examples=1, channels=1)
        config = _flat_psd_config()
        model = config.model
        assert type(model) is PsdNoiseConfig

        for requested_dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=requested_dtype):
                powers = _prepare_psd_powers(
                    model,
                    sampling=sampling,
                    dtype=requested_dtype,
                )
                normals = _psd_normals(
                    model,
                    seed=19,
                    row_count=1,
                    frequency_count=sample_count // 2 + 1,
                    dtype=requested_dtype,
                )[0]
                real = normals[:, 0]
                imaginary = normals[:, 1]
                result = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=config,
                    rng=Threefry4x32(seed=19),
                    floating_dtype=requested_dtype,
                )

                cosine = torch.tensor(
                    tuple(
                        math.fsum(
                            math.sqrt(powers[frequency])
                            * float(real[frequency - 1])
                            * math.cos(
                                math.tau * frequency * index / sample_count
                            )
                            for frequency in range(1, len(powers))
                        )
                        for index in range(sample_count)
                    ),
                    dtype=requested_dtype,
                )
                sine = torch.tensor(
                    tuple(
                        math.fsum(
                            -math.sqrt(powers[frequency])
                            * float(imaginary[frequency - 1])
                            * math.sin(
                                math.tau * frequency * index / sample_count
                            )
                            for frequency in range(1, len(powers))
                        )
                        for index in range(sample_count)
                    ),
                    dtype=requested_dtype,
                )
                expected = cosine + sine
                terminal_sine = torch.tensor(
                    tuple(
                        -math.sqrt(powers[-1])
                        * float(imaginary[-1])
                        * math.sin(
                            math.tau
                            * (len(powers) - 1)
                            * index
                            / sample_count
                        )
                        for index in range(sample_count)
                    ),
                    dtype=requested_dtype,
                )
                reference_scale = max(
                    float(torch.max(torch.abs(expected))),
                    torch.finfo(requested_dtype).tiny,
                )
                tolerance = (
                    64
                    * torch.finfo(requested_dtype).eps
                    * max(1, math.ceil(math.log2(sample_count)))
                    * reference_scale
                )
                self.assertTrue(
                    torch.allclose(
                        _sample_last(result).reshape(sample_count),
                        expected,
                        rtol=0.0,
                        atol=tolerance,
                    )
                )
                self.assertTrue(torch.equal(expected, cosine + sine))
                self.assertGreater(float(torch.max(torch.abs(terminal_sine))), 0.0)

    def test_two_sample_psd_is_real_nyquist_only(self) -> None:
        sample_count = 2
        sampling = _sampling(count=sample_count)
        source = _photoelectrons(sampling, examples=1, channels=1)
        config = _flat_psd_config()
        model = config.model
        assert type(model) is PsdNoiseConfig

        for requested_dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=requested_dtype):
                powers = _prepare_psd_powers(
                    model,
                    sampling=sampling,
                    dtype=requested_dtype,
                )
                normals = _psd_normals(
                    model,
                    seed=29,
                    row_count=1,
                    frequency_count=2,
                    dtype=requested_dtype,
                )
                normal_real_value = float(normals[0, 0, 0])
                normal_imaginary_value = float(normals[0, 0, 1])
                self.assertNotEqual(normal_imaginary_value, 0.0)
                captured_coefficients: list[torch.Tensor] = []
                original_irfft = torch.fft.irfft

                def capture_irfft(
                    input: torch.Tensor,
                    **kwargs: object,
                ) -> torch.Tensor:
                    captured_coefficients.append(input.clone())
                    return original_irfft(input, **kwargs)

                with patch(
                    "tensor_dslab.readout.noise_waveform._produce.torch.fft.irfft",
                    side_effect=capture_irfft,
                ):
                    result = _produce_noise_waveform(
                        source,
                        sampling=sampling,
                        config=config,
                        rng=Threefry4x32(seed=29),
                        floating_dtype=requested_dtype,
                    )

                self.assertEqual(len(captured_coefficients), 1)
                coefficients = captured_coefficients[0]
                self.assertEqual(coefficients.shape, (1, 2))
                self.assertEqual(complex(coefficients[0, 0]), 0j)
                self.assertEqual(float(coefficients[0, 1].imag), 0.0)
                expected_nyquist = torch.tensor(
                    sample_count * math.sqrt(powers[1]) * normal_real_value,
                    dtype=requested_dtype,
                )
                expected_output = torch.tensor(
                    (
                        math.sqrt(powers[1]) * normal_real_value,
                        -math.sqrt(powers[1]) * normal_real_value,
                    ),
                    dtype=requested_dtype,
                )
                reference_scale = max(
                    abs(float(expected_nyquist)),
                    float(torch.max(torch.abs(expected_output))),
                    torch.finfo(requested_dtype).tiny,
                )
                tolerance = (
                    64
                    * torch.finfo(requested_dtype).eps
                    * max(1, math.ceil(math.log2(sample_count)))
                    * reference_scale
                )
                self.assertTrue(
                    torch.allclose(
                        coefficients[0, 1].real,
                        expected_nyquist,
                        rtol=0.0,
                        atol=tolerance,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        _sample_last(result).reshape(sample_count),
                        expected_output,
                        rtol=0.0,
                        atol=tolerance,
                    )
                )

    def test_psd_repeatability_no_crop_no_normalization_and_row_independence(self) -> None:
        sampling = _sampling(count=8)
        source = _photoelectrons(sampling, examples=3, channels=2)
        first = _produce_noise_waveform(
            source,
            sampling=sampling,
            config=_flat_psd_config(),
            rng=Threefry4x32(seed=5),
            floating_dtype=torch.float64,
        )
        second = _produce_noise_waveform(
            source,
            sampling=sampling,
            config=_flat_psd_config(),
            rng=Threefry4x32(seed=5),
            floating_dtype=torch.float64,
        )
        rows = _sample_last(first).reshape(-1, 8)
        self.assertTrue(torch.equal(first.tensor, second.tensor))
        self.assertEqual(rows.shape, (6, 8))
        self.assertFalse(torch.equal(rows[0], rows[1]))
        row_power = torch.mean(rows.square(), dim=-1)
        self.assertGreater(float(torch.std(row_power)), 0.0)
        self.assertLessEqual(
            float(torch.max(torch.abs(torch.mean(rows, dim=-1)))),
            1.0e-12,
        )


class NoiseStatisticalContractTest(unittest.TestCase):
    def test_frozen_white_ensemble(self) -> None:
        sampling = _sampling(count=32)
        source = _photoelectrons(sampling, examples=64, channels=32)
        for dtype in (torch.float32, torch.float64):
            values_by_seed = tuple(
                _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=_white_config(1.0),
                    rng=Threefry4x32(seed=seed),
                    floating_dtype=dtype,
                ).tensor.reshape(-1).to(dtype=torch.float64)
                for seed in SEEDS
            )
            values = torch.cat(values_by_seed)
            pair_products = torch.cat(
                tuple(items[0::2] * items[1::2] for items in values_by_seed)
            )
            sample_count = values.numel()
            covariance_count = pair_products.numel()
            sigma = _round(1.0, dtype)
            observed_mean = float(torch.mean(values))
            observed_square = float(torch.mean(values.square()))
            observed_covariance = float(torch.mean(pair_products))
            self.assertLessEqual(
                abs(observed_mean),
                6 * sigma / math.sqrt(sample_count) + _delta(dtype, sigma, 1),
            )
            self.assertLessEqual(
                abs(observed_square - sigma**2),
                6 * sigma**2 * math.sqrt(2 / sample_count)
                + _delta(dtype, sigma**2, 1),
            )
            self.assertLessEqual(
                abs(observed_covariance),
                6 * sigma**2 / math.sqrt(covariance_count)
                + _delta(dtype, sigma**2, 1),
            )

    def test_frozen_odd_even_psd_ensembles(self) -> None:
        for count in (31, 32):
            sampling = _sampling(count=count)
            source = _photoelectrons(sampling, examples=64, channels=64)
            model = _flat_psd_config().model
            assert type(model) is PsdNoiseConfig
            for dtype in (torch.float32, torch.float64):
                powers = _reference_psd_powers(
                    model,
                    sampling=sampling,
                    dtype=dtype,
                )
                rows_by_seed = tuple(
                    _sample_last(
                        _produce_noise_waveform(
                            source,
                            sampling=sampling,
                            config=NoiseWaveformConfig(model=model),
                            rng=Threefry4x32(seed=seed),
                            floating_dtype=dtype,
                        )
                    ).reshape(-1, count).to(dtype=torch.float64)
                    for seed in SEEDS
                )
                rows = torch.cat(rows_by_seed)
                row_count = rows.shape[0]
                covariance = tuple(
                    math.fsum(
                        power * math.cos(2.0 * math.pi * index * lag / count)
                        for index, power in enumerate(powers)
                    )
                    for lag in range(3)
                )
                variance = covariance[0]
                for lag in range(3):
                    observed = float(torch.mean(rows[:, 0] * rows[:, lag]))
                    standard_error = math.sqrt(
                        (variance**2 + covariance[lag] ** 2) / row_count
                    )
                    self.assertLessEqual(
                        abs(observed - covariance[lag]),
                        8 * standard_error
                        + _delta(dtype, max(variance, abs(covariance[lag])), count),
                    )
                self.assertAlmostEqual(
                    covariance[1] / variance,
                    -1.0 / (count - 1),
                    places=6,
                )

                observed_coefficients = torch.cat(
                    tuple(
                        torch.fft.rfft(
                            seed_rows,
                            n=count,
                            dim=-1,
                            norm="backward",
                        )
                        for seed_rows in rows_by_seed
                    )
                )
                coefficient = observed_coefficients[:, 3]
                target_component_variance = count**2 * powers[3] / 4.0
                component_se = target_component_variance * math.sqrt(2.0 / row_count)
                covariance_se = target_component_variance / math.sqrt(row_count)
                magnitude_se = 2.0 * target_component_variance / math.sqrt(row_count)
                real_square = float(torch.mean(coefficient.real.square()))
                imaginary_square = float(torch.mean(coefficient.imag.square()))
                component_covariance = float(
                    torch.mean(coefficient.real * coefficient.imag)
                )
                magnitude_square = float(torch.mean(torch.abs(coefficient).square()))
                component_bound = 8 * component_se + _delta(
                    dtype,
                    target_component_variance,
                    count,
                )
                self.assertLessEqual(
                    abs(real_square - target_component_variance),
                    component_bound,
                )
                self.assertLessEqual(
                    abs(imaginary_square - target_component_variance),
                    component_bound,
                )
                self.assertLessEqual(
                    abs(component_covariance),
                    8 * covariance_se
                    + _delta(dtype, target_component_variance, count),
                )
                self.assertLessEqual(
                    abs(magnitude_square - 2 * target_component_variance),
                    8 * magnitude_se
                    + _delta(dtype, 2 * target_component_variance, count),
                )
                if count % 2 == 0:
                    nyquist = observed_coefficients[:, count // 2]
                    target_nyquist_variance = count**2 * powers[count // 2]
                    nyquist_se = target_nyquist_variance * math.sqrt(2.0 / row_count)
                    self.assertTrue(torch.equal(nyquist.imag, torch.zeros_like(nyquist.imag)))
                    self.assertLessEqual(
                        abs(float(torch.mean(nyquist.real.square())) - target_nyquist_variance),
                        8 * nyquist_se
                        + _delta(dtype, target_nyquist_variance, count),
                    )

                cross_products = torch.cat(
                    tuple(
                        seed_rows[0::2, 0] * seed_rows[1::2, 0]
                        for seed_rows in rows_by_seed
                    )
                )
                cross_count = cross_products.numel()
                self.assertLessEqual(
                    abs(float(torch.mean(cross_products))),
                    8 * variance / math.sqrt(cross_count)
                    + _delta(dtype, variance, count),
                )
                parseval = float(torch.mean(torch.mean(rows.square(), dim=-1)))
                interior = powers[1 : (count - 1) // 2 + 1]
                parseval_se = math.sqrt(
                    (
                        math.fsum(power**2 for power in interior)
                        + (2.0 * powers[count // 2] ** 2 if count % 2 == 0 else 0.0)
                    )
                    / row_count
                )
                self.assertLessEqual(
                    abs(parseval - variance),
                    8 * parseval_se + _delta(dtype, variance, count),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_all_models_and_same_backend_repeatability(self) -> None:
        sampling = _sampling(count=32)
        source = _photoelectrons(
            sampling,
            examples=8,
            channels=8,
            device="cuda",
            noncontiguous=True,
        )
        for config in (_zero_config(), _white_config(), _flat_psd_config()):
            for dtype in (torch.float32, torch.float64):
                first = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=config,
                    rng=Threefry4x32(seed=11),
                    floating_dtype=dtype,
                )
                second = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=config,
                    rng=Threefry4x32(seed=11),
                    floating_dtype=dtype,
                )
                self.assertTrue(torch.equal(first.tensor, second.tensor))
                self.assertEqual(first.tensor.device.type, "cuda")
                self.assertTrue(bool(torch.all(torch.isfinite(first.tensor))))


if __name__ == "__main__":
    unittest.main()
