from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    FiniteFloat,
    NonnegativeFloat,
    PositiveFloat,
    PositiveInteger,
    TensorAxis,
    require_same_dtype,
)

from tensor_dslab import (
    AnalogSaturationConfig,
    AnalogWaveform,
    AnalogWaveformConfig,
    ChannelAxis,
    Charge,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    ExampleAxis,
    NoiseWaveform,
    PureWaveform,
    PureWaveformConfig,
    SampleAxis,
    SamplingConfig,
    TpcFebSnrPulseConfig,
    VetoPduPulseConfig,
)
from tensor_dslab.readout.analog_waveform._produce import (
    _produce_analog_waveform,
)
from tensor_dslab.readout.analog_waveform.field import (
    _require_valid_values as require_valid_analog,
)
from tensor_dslab.readout.digitized_waveform._produce import (
    _produce_digitized_waveform,
)
from tensor_dslab.readout.digitized_waveform.field import (
    _require_valid_values as require_valid_digitized,
)
from tensor_dslab.readout.pure_waveform._produce import (
    _produce_pure_waveform,
)
from tensor_dslab.readout.pure_waveform.field import (
    _require_valid_values as require_valid_pure,
)


PulseModel = TpcFebSnrPulseConfig | VetoPduPulseConfig


def _sampling(*, period_ps: int = 8_000, count: int = 8) -> SamplingConfig:
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
    start_ps: int = 0,
    period_ps: int | None = None,
) -> tuple[TensorAxis, ...]:
    exact_period = (
        sampling.sample_period_ps.value if period_ps is None else period_ps
    )
    available: dict[type[TensorAxis], TensorAxis] = {
        ExampleAxis: ExampleAxis(coordinates=("example-0",)),
        ChannelAxis: ChannelAxis(coordinates=("channel-0",)),
        SampleAxis: SampleAxis(
            coordinates=tuple(
                f"{start_ps + index * exact_period}ps"
                for index in range(sampling.sample_count.value)
            )
        ),
    }
    return tuple(available[axis_type] for axis_type in order)


def _tensor_from_samples(
    samples: list[float],
    axes: tuple[TensorAxis, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
    requires_grad: bool = False,
) -> torch.Tensor:
    sample_dimension = next(
        index for index, axis in enumerate(axes) if type(axis) is SampleAxis
    )
    shape = tuple(axis.size for axis in axes)
    source = torch.tensor(samples, dtype=dtype, device=device)
    source_shape = [1] * len(axes)
    source_shape[sample_dimension] = len(samples)
    values = source.reshape(source_shape).expand(shape)
    if noncontiguous:
        backing = torch.empty((*shape, 2), dtype=dtype, device=device)
        noncontiguous_values = backing[..., 0]
        noncontiguous_values.copy_(values)
        values = noncontiguous_values
    else:
        values = values.clone()
    if requires_grad:
        values.requires_grad_()
    return values


def _charge(
    samples: list[float],
    sampling: SamplingConfig,
    *,
    dtype: torch.dtype,
    axes: tuple[TensorAxis, ...] | None = None,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
    requires_grad: bool = False,
) -> Charge:
    exact_axes = _axes(sampling) if axes is None else axes
    return Charge(
        tensor=_tensor_from_samples(
            samples,
            exact_axes,
            dtype=dtype,
            device=device,
            noncontiguous=noncontiguous,
            requires_grad=requires_grad,
        ),
        axes=exact_axes,
    )


def _pure(
    samples: list[float],
    sampling: SamplingConfig,
    *,
    dtype: torch.dtype,
    axes: tuple[TensorAxis, ...] | None = None,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
    requires_grad: bool = False,
) -> PureWaveform:
    exact_axes = _axes(sampling) if axes is None else axes
    return PureWaveform(
        tensor=_tensor_from_samples(
            samples,
            exact_axes,
            dtype=dtype,
            device=device,
            noncontiguous=noncontiguous,
            requires_grad=requires_grad,
        ),
        axes=exact_axes,
    )


def _noise(
    samples: list[float],
    sampling: SamplingConfig,
    *,
    dtype: torch.dtype,
    axes: tuple[TensorAxis, ...] | None = None,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
    requires_grad: bool = False,
) -> NoiseWaveform:
    exact_axes = _axes(sampling) if axes is None else axes
    return NoiseWaveform(
        tensor=_tensor_from_samples(
            samples,
            exact_axes,
            dtype=dtype,
            device=device,
            noncontiguous=noncontiguous,
            requires_grad=requires_grad,
        ),
        axes=exact_axes,
    )


def _analog(
    samples: list[float],
    sampling: SamplingConfig,
    *,
    dtype: torch.dtype,
    axes: tuple[TensorAxis, ...] | None = None,
    device: torch.device | str = "cpu",
    noncontiguous: bool = False,
    requires_grad: bool = False,
) -> AnalogWaveform:
    exact_axes = _axes(sampling) if axes is None else axes
    return AnalogWaveform(
        tensor=_tensor_from_samples(
            samples,
            exact_axes,
            dtype=dtype,
            device=device,
            noncontiguous=noncontiguous,
            requires_grad=requires_grad,
        ),
        axes=exact_axes,
    )


def _tpc_config(
    *,
    support_time_ns: float = 3_000.0,
    peak_mv: float = -7.0,
) -> PureWaveformConfig:
    return PureWaveformConfig(
        model=TpcFebSnrPulseConfig(
            fast_time_constant_ns=PositiveFloat(83.0),
            slow_time_constant_ns=PositiveFloat(383.0),
            support_time_ns=PositiveFloat(support_time_ns),
            peak_voltage_mv_per_pe=FiniteFloat(peak_mv),
        )
    )


def _veto_config() -> PureWaveformConfig:
    return PureWaveformConfig(
        model=VetoPduPulseConfig(
            gaussian_center_ns=FiniteFloat(232.89),
            gaussian_width_ns=PositiveFloat(507.72),
            edge_offset_1_ns=FiniteFloat(-81.92),
            edge_width_1_ns=PositiveFloat(147.28),
            edge_offset_2_ns=FiniteFloat(-176.50),
            edge_width_2_ns=PositiveFloat(45.69),
            support_time_ns=PositiveFloat(2020.27),
            peak_voltage_mv_per_pe=FiniteFloat(-14.5912372),
        )
    )


def _raw_pulse(time_ns: float, model: PulseModel) -> float:
    if isinstance(model, TpcFebSnrPulseConfig):
        return math.exp(-time_ns / model.slow_time_constant_ns.value) - math.exp(
            -time_ns / model.fast_time_constant_ns.value
        )
    x = time_ns - model.gaussian_center_ns.value
    gaussian = math.exp(
        -(x**2) / (2.0 * model.gaussian_width_ns.value**2)
    ) / math.sqrt(2.0 * math.pi * model.gaussian_width_ns.value**2)
    first_edge = 1.0 + math.erf(
        (x - model.edge_offset_1_ns.value)
        / (math.sqrt(2.0) * model.edge_width_1_ns.value)
    )
    second_edge = 1.0 + math.erf(
        (x - model.edge_offset_2_ns.value)
        / (math.sqrt(2.0) * model.edge_width_2_ns.value)
    )
    return gaussian * first_edge * second_edge


def _reference_coefficients(
    sampling: SamplingConfig,
    config: PureWaveformConfig,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    period_ns = sampling.sample_period_ps.value / 1000.0
    model = config.model
    sample = 0
    raw: list[float] = []
    while sample * period_ns < model.support_time_ns.value:
        raw.append(_raw_pulse(sample * period_ns, model))
        sample += 1
    normalization = max(abs(value) for value in raw)
    retained = raw[: sampling.sample_count.value]
    return torch.tensor(
        [
            value / normalization * model.peak_voltage_mv_per_pe.value
            for value in retained
        ],
        dtype=dtype,
        device=device,
    )


def _reference_pure(
    charge: torch.Tensor,
    axes: tuple[TensorAxis, ...],
    sampling: SamplingConfig,
    config: PureWaveformConfig,
) -> torch.Tensor:
    sample_dimension = next(
        index for index, axis in enumerate(axes) if type(axis) is SampleAxis
    )
    sample_last = charge.movedim(sample_dimension, -1)
    coefficients = _reference_coefficients(
        sampling,
        config,
        dtype=charge.dtype,
        device=charge.device,
    )
    samples: list[torch.Tensor] = []
    for output_index in range(sample_last.shape[-1]):
        value = sample_last[..., output_index] * coefficients[0]
        for coefficient_index in range(
            1,
            min(output_index + 1, coefficients.shape[0]),
        ):
            value = value + (
                sample_last[..., output_index - coefficient_index]
                * coefficients[coefficient_index]
            )
        samples.append(value)
    return torch.stack(samples, dim=-1).movedim(-1, sample_dimension)


def _adc_config(
    *,
    bit_depth: int = 12,
    input_min_mv: float = -1_000.0,
    input_max_mv: float = 1_000.0,
    gain_db: float = 0.0,
) -> DigitizedWaveformConfig:
    return DigitizedWaveformConfig(
        bit_depth=PositiveInteger(bit_depth),
        input_min_mv=FiniteFloat(input_min_mv),
        input_max_mv=FiniteFloat(input_max_mv),
        analog_gain_db=NonnegativeFloat(gain_db),
    )


def _guarded_adc_reference(
    analog: torch.Tensor,
    config: DigitizedWaveformConfig,
) -> torch.Tensor:
    maximum_code = (1 << config.bit_depth.value) - 1
    gain = 10.0 ** (config.analog_gain_db.value / 20.0)
    span = config.input_max_mv.value - config.input_min_mv.value

    def scalar(value: float | int) -> torch.Tensor:
        return torch.tensor(
            value,
            dtype=analog.dtype,
            device=analog.device,
        )

    zero = scalar(0.0)
    maximum = scalar(maximum_code)
    lower = scalar(config.input_min_mv.value / gain)
    upper = scalar(config.input_max_mv.value / gain)
    gained = analog * scalar(gain)
    clipped = torch.clamp(
        gained,
        min=scalar(config.input_min_mv.value),
        max=scalar(config.input_max_mv.value),
    )
    scaled = (
        (clipped - scalar(config.input_min_mv.value))
        / scalar(span)
        * maximum
    )
    interior = torch.clamp(scaled, min=zero, max=maximum).to(torch.int32)
    selected = torch.where(
        analog <= lower,
        torch.zeros_like(interior),
        torch.where(
            analog >= upper,
            torch.full_like(interior, maximum_code),
            interior,
        ),
    )
    return selected


def _independent_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()


class DeterministicWaveformProductsTest(unittest.TestCase):
    def test_tpc_pure_waveform_matches_binary64_reference(self) -> None:
        sampling = _sampling(count=375)
        config = _tpc_config()
        checkpoint_indices = (0, 1, 10, 20, 30, 40, 100, 200, 374)
        # Audited donors:
        # Projects/iv-dslab-main_db_PB/src/dselec/waveform.py
        # SHA-256 5eb5b29e6958184e520b2151877a678f6d98cdbe6e53cbf9d1b4c4e64e0f82b5
        # Projects/iv-dslab-main_db_PB/data/config_files/dselec.ini
        # SHA-256 fd42244bb4405dc328496efb8043fff522584a1922b811246670ac0e940e1c64
        # The checkpoints map the reviewed TPC equation and calibration onto
        # 8 ns left-edge samples for a unit PE-equivalent impulse, in mV.
        # TensorDSLab intentionally uses sampled-extremum normalization, a
        # signed -7 mV peak exactly once, half-open support, same-length causal
        # convolution, no fractional-bin correction, and no final inversion.
        donor_checkpoints = (
            -0.0,
            -0.9716642705837991,
            -5.867982169282606,
            -7.0,
            -6.534142697669049,
            -5.628032736687342,
            -1.6887803569736373,
            -0.20924694424980722,
            -0.005523714943124405,
        )
        for dtype, rtol, atol in (
            (torch.float32, 2e-5, 2e-6),
            (torch.float64, 1e-12, 1e-12),
        ):
            with self.subTest(dtype=dtype):
                charge = _charge(
                    [1.0] + [0.0] * 374,
                    sampling,
                    dtype=dtype,
                )
                result = _produce_pure_waveform(
                    charge,
                    sampling=sampling,
                    config=config,
                )
                expected = _reference_pure(
                    charge.tensor,
                    charge.axes,
                    sampling,
                    config,
                )
                torch.testing.assert_close(result.tensor, expected, rtol=rtol, atol=atol)
                actual_checkpoints = result.tensor.flatten()[
                    torch.tensor(checkpoint_indices)
                ]
                torch.testing.assert_close(
                    actual_checkpoints,
                    torch.tensor(donor_checkpoints, dtype=dtype),
                    rtol=1e-4,
                    atol=1e-5,
                )
                self.assertEqual(float(result.tensor.flatten()[0]), 0.0)
                self.assertEqual(float(result.tensor.flatten()[20]), -7.0)

    def test_veto_pure_waveform_matches_binary64_reference(self) -> None:
        sampling = _sampling(count=253)
        config = _veto_config()
        checkpoint_indices = (0, 1, 10, 20, 30, 40, 60, 100, 200, 252)
        # The same audited source paths and full hashes recorded immediately
        # above map the Veto equation onto 8 ns left-edge samples for a unit
        # PE-equivalent impulse, in mV. TensorDSLab uses sampled-extremum
        # normalization, signed -14.5912372 mV once, half-open support,
        # same-length causal convolution, and no fractional-bin correction.
        donor_checkpoints = (
            -0.24075045235202222,
            -0.35123563117845097,
            -3.3868647444864486,
            -8.278146774157653,
            -11.733649674571378,
            -13.904378836032057,
            -14.15149422089209,
            -8.647433307201961,
            -0.42994977958983494,
            -0.033844777632089235,
        )
        for dtype, rtol, atol in (
            (torch.float32, 2e-5, 2e-6),
            (torch.float64, 1e-12, 1e-12),
        ):
            with self.subTest(dtype=dtype):
                charge = _charge(
                    [1.0] + [0.0] * 252,
                    sampling,
                    dtype=dtype,
                )
                result = _produce_pure_waveform(
                    charge,
                    sampling=sampling,
                    config=config,
                )
                expected = _reference_pure(
                    charge.tensor,
                    charge.axes,
                    sampling,
                    config,
                )
                torch.testing.assert_close(result.tensor, expected, rtol=rtol, atol=atol)
                actual_checkpoints = result.tensor.flatten()[
                    torch.tensor(checkpoint_indices)
                ]
                torch.testing.assert_close(
                    actual_checkpoints,
                    torch.tensor(donor_checkpoints, dtype=dtype),
                    rtol=1e-4,
                    atol=1e-5,
                )
                self.assertEqual(
                    float(torch.min(result.tensor)),
                    float(torch.tensor(-14.5912372, dtype=dtype)),
                )

    def test_pure_waveform_support_is_left_closed_right_open(self) -> None:
        sampling = _sampling(count=4)
        config = _tpc_config(support_time_ns=16.0, peak_mv=-2.0)
        charge = _charge([1.0, 0.0, 0.0, 0.0], sampling, dtype=torch.float64)
        result = _produce_pure_waveform(
            charge,
            sampling=sampling,
            config=config,
        )
        torch.testing.assert_close(
            result.tensor.flatten(),
            torch.tensor([0.0, -2.0, 0.0, 0.0], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_pure_waveform_normalizes_over_complete_support_before_record_crop(
        self,
    ) -> None:
        sampling = _sampling(count=4)
        config = _tpc_config()
        charge = _charge([1.0, 0.0, 0.0, 0.0], sampling, dtype=torch.float64)
        result = _produce_pure_waveform(
            charge,
            sampling=sampling,
            config=config,
        )
        expected = _reference_pure(charge.tensor, charge.axes, sampling, config)
        torch.testing.assert_close(result.tensor, expected, rtol=1e-12, atol=1e-12)
        self.assertLess(abs(float(result.tensor.flatten()[-1])), 7.0)

    def test_pure_waveform_is_causal_same_length_and_zero_baseline(self) -> None:
        sampling = _sampling(count=8)
        config = _tpc_config(support_time_ns=32.0)
        charge = _charge(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sampling,
            dtype=torch.float64,
        )
        result = _produce_pure_waveform(
            charge,
            sampling=sampling,
            config=config,
        )
        self.assertEqual(result.shape, charge.shape)
        self.assertEqual(float(result.tensor.flatten()[0]), 0.0)
        self.assertEqual(float(result.tensor.flatten()[1]), 0.0)
        zero_charge = _charge([0.0] * 8, sampling, dtype=torch.float64)
        zero_result = _produce_pure_waveform(
            zero_charge,
            sampling=sampling,
            config=config,
        )
        torch.testing.assert_close(
            zero_result.tensor,
            torch.zeros_like(zero_result.tensor),
            rtol=0.0,
            atol=0.0,
        )

    def test_pure_waveform_handles_alternate_axis_order_and_noncontiguous_input(
        self,
    ) -> None:
        sampling = _sampling(count=8)
        axes = (
            sampling.build_axis(),
            ExampleAxis(coordinates=("example-0", "example-1")),
            ChannelAxis(coordinates=("channel-0", "channel-1")),
        )
        backing = torch.zeros((8, 2, 2, 2), dtype=torch.float32)
        values = backing[..., 0]
        values[0, 0, 0] = 1.0
        values[1, 0, 1] = 2.0
        values[2, 1, 0] = 3.0
        values[0, 1, 1] = 4.0
        values[3, 1, 1] = 1.0
        charge = Charge(
            tensor=values,
            axes=axes,
        )
        self.assertFalse(charge.tensor.is_contiguous())
        config = _tpc_config(support_time_ns=32.0)
        result = _produce_pure_waveform(
            charge,
            sampling=sampling,
            config=config,
        )
        expected = _reference_pure(
            charge.tensor,
            charge.axes,
            sampling,
            config,
        )
        torch.testing.assert_close(result.tensor, expected, rtol=2e-5, atol=2e-6)
        for example_index in range(2):
            for channel_index in range(2):
                torch.testing.assert_close(
                    result.tensor[:, example_index, channel_index],
                    expected[:, example_index, channel_index],
                    rtol=2e-5,
                    atol=2e-6,
                )
        self.assertIs(result.axes, charge.axes)
        for result_axis, source_axis in zip(result.axes, charge.axes):
            self.assertIs(result_axis, source_axis)

    def test_pure_waveform_reuses_axes_is_fresh_and_preserves_autograd(self) -> None:
        sampling = _sampling(count=6)
        config = _tpc_config(support_time_ns=32.0)
        for dtype, rtol, atol in (
            (torch.float32, 2e-5, 2e-6),
            (torch.float64, 1e-12, 1e-12),
        ):
            with self.subTest(dtype=dtype):
                samples = [0.5, 1.0, 0.25, 0.0, 0.75, 0.2]
                charge = _charge(
                    samples,
                    sampling,
                    dtype=dtype,
                    requires_grad=True,
                )
                original = charge.tensor.clone()
                version = charge.tensor._version
                result = _produce_pure_waveform(
                    charge,
                    sampling=sampling,
                    config=config,
                )
                self.assertIs(type(result), PureWaveform)
                self.assertIs(result.axes, charge.axes)
                self.assertEqual(result.shape, charge.shape)
                self.assertIs(result.tensor.dtype, dtype)
                self.assertEqual(result.tensor.device, charge.tensor.device)
                for result_axis, source_axis in zip(result.axes, charge.axes):
                    self.assertIs(result_axis, source_axis)
                self.assertTrue(_independent_storage(result.tensor, charge.tensor))
                self.assertEqual(charge.tensor._version, version)
                torch.testing.assert_close(charge.tensor, original)
                self.assertIsNone(require_valid_pure(result))
                saved_result = result.tensor.clone()
                _produce_pure_waveform(charge, sampling=sampling, config=config)
                torch.testing.assert_close(result.tensor, saved_result)

                result.tensor.square().sum().backward()
                charge_gradient = charge.tensor.grad
                assert charge_gradient is not None
                actual_gradient = charge_gradient.clone()
                reference_input = original.detach().requires_grad_(True)
                reference_output = _reference_pure(
                    reference_input,
                    charge.axes,
                    sampling,
                    config,
                )
                reference_output.square().sum().backward()
                torch.testing.assert_close(
                    actual_gradient,
                    reference_input.grad,
                    rtol=rtol,
                    atol=atol,
                )

        axes = _axes(sampling)
        gradcheck_input = _tensor_from_samples(
            [0.5, 1.0, 0.25, 0.0, 0.75, 0.2],
            axes,
            dtype=torch.float64,
            requires_grad=True,
        )

        def pure_function(values: torch.Tensor) -> torch.Tensor:
            return _produce_pure_waveform(
                Charge(tensor=values, axes=axes),
                sampling=sampling,
                config=config,
            ).tensor

        self.assertTrue(
            torch.autograd.gradcheck(
                pure_function,
                (gradcheck_input,),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

    def test_sampling_relationship_rejects_size_start_and_period_mismatch(
        self,
    ) -> None:
        config = _tpc_config(support_time_ns=32.0)
        sampling = _sampling(count=4)
        mismatch_cases = (
            (
                _charge(
                    [1.0, 0.0, 0.0],
                    _sampling(count=3),
                    dtype=torch.float32,
                ),
                sampling,
            ),
            (
                _charge(
                    [1.0, 0.0, 0.0, 0.0],
                    sampling,
                    dtype=torch.float32,
                    axes=_axes(sampling, start_ps=8_000),
                ),
                sampling,
            ),
            (
                _charge(
                    [1.0, 0.0, 0.0, 0.0],
                    sampling,
                    dtype=torch.float32,
                    axes=_axes(sampling, period_ps=4_000),
                ),
                sampling,
            ),
        )
        for charge, expected_sampling in mismatch_cases:
            with self.subTest(axis=charge.axis(SampleAxis).coordinates):
                with self.assertRaises(ValueError):
                    _produce_pure_waveform(
                        charge,
                        sampling=expected_sampling,
                        config=config,
                    )

    def test_pure_waveform_rejects_out_of_range_support_before_convolution(
        self,
    ) -> None:
        sampling = _sampling(count=2)
        charge = _charge([1.0, 0.0], sampling, dtype=torch.float32)
        before = charge.tensor.clone()
        version = charge.tensor._version
        with patch(
            "tensor_dslab.readout.pure_waveform._produce.functional.conv1d",
            side_effect=AssertionError("payload convolution started"),
        ):
            with self.assertRaises(ValueError):
                _produce_pure_waveform(
                    charge,
                    sampling=sampling,
                    config=_tpc_config(support_time_ns=1e308),
                )
        self.assertEqual(charge.tensor._version, version)
        torch.testing.assert_close(charge.tensor, before)

    def test_pure_waveform_rejects_unrepresentable_peak_before_convolution(
        self,
    ) -> None:
        sampling = _sampling(count=4)
        charge = _charge([1.0, 0.0, 0.0, 0.0], sampling, dtype=torch.float32)
        before = charge.tensor.clone()
        version = charge.tensor._version
        for peak_mv in (3.5e38, 1e-50):
            with self.subTest(peak_mv=peak_mv):
                with patch(
                    "tensor_dslab.readout.pure_waveform._produce.functional.conv1d",
                    side_effect=AssertionError("payload convolution started"),
                ):
                    with self.assertRaises(ValueError):
                        _produce_pure_waveform(
                            charge,
                            sampling=sampling,
                            config=_tpc_config(
                                support_time_ns=32.0,
                                peak_mv=peak_mv,
                            ),
                        )
                self.assertEqual(charge.tensor._version, version)
                torch.testing.assert_close(charge.tensor, before)

    def test_pure_waveform_preserves_dtype_under_cpu_autocast(self) -> None:
        sampling = _sampling(count=8)
        config = _tpc_config(support_time_ns=32.0)
        charge = _charge(
            [1.0, 0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
            sampling,
            dtype=torch.float32,
        )
        expected = _reference_pure(
            charge.tensor,
            charge.axes,
            sampling,
            config,
        )
        with torch.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
        ):
            result = _produce_pure_waveform(
                charge,
                sampling=sampling,
                config=config,
            )
        self.assertIs(result.tensor.dtype, torch.float32)
        torch.testing.assert_close(
            result.tensor,
            expected,
            rtol=2e-5,
            atol=2e-6,
        )

    def test_analog_waveform_matches_unbounded_and_each_saturation_form(
        self,
    ) -> None:
        sampling = _sampling(count=5)
        pure_samples = [-3.0, -1.0, 0.0, 2.0, 5.0]
        noise_samples = [0.5, -1.0, 1.0, 2.0, -0.5]
        configs = (
            AnalogWaveformConfig(),
            AnalogWaveformConfig(
                saturation=AnalogSaturationConfig(minimum_mv=FiniteFloat(-2.0))
            ),
            AnalogWaveformConfig(
                saturation=AnalogSaturationConfig(maximum_mv=FiniteFloat(3.0))
            ),
            AnalogWaveformConfig(
                saturation=AnalogSaturationConfig(
                    minimum_mv=FiniteFloat(-2.0),
                    maximum_mv=FiniteFloat(3.0),
                )
            ),
        )
        for dtype in (torch.float32, torch.float64):
            pure = _pure(pure_samples, sampling, dtype=dtype)
            noise = _noise(noise_samples, sampling, dtype=dtype)
            for config in configs:
                with self.subTest(dtype=dtype, saturation=config.saturation):
                    result = _produce_analog_waveform(pure, noise, config=config)
                    expected = torch.add(pure.tensor, noise.tensor)
                    if config.saturation is not None:
                        minimum = (
                            None
                            if config.saturation.minimum_mv is None
                            else torch.tensor(
                                config.saturation.minimum_mv.value,
                                dtype=dtype,
                            )
                        )
                        maximum = (
                            None
                            if config.saturation.maximum_mv is None
                            else torch.tensor(
                                config.saturation.maximum_mv.value,
                                dtype=dtype,
                            )
                        )
                        expected = torch.clamp(expected, min=minimum, max=maximum)
                    torch.testing.assert_close(result.tensor, expected, rtol=0.0, atol=0.0)
                    self.assertIs(type(result), AnalogWaveform)
                    self.assertEqual(result.shape, pure.shape)
                    self.assertIs(result.tensor.dtype, dtype)
                    self.assertEqual(result.tensor.device, pure.tensor.device)
                    self.assertIsNone(require_valid_analog(result))

        pure = _pure(pure_samples, sampling, dtype=torch.float32)
        noise = _noise(noise_samples, sampling, dtype=torch.float32)
        bounded = configs[-1]
        real_clamp = torch.clamp
        with patch(
            "tensor_dslab.readout.analog_waveform._produce.torch.clamp",
            wraps=real_clamp,
        ) as clamp_call:
            _produce_analog_waveform(pure, noise, config=bounded)
        minimum = clamp_call.call_args.kwargs["min"]
        maximum = clamp_call.call_args.kwargs["max"]
        self.assertEqual(minimum.ndim, 0)
        self.assertEqual(maximum.ndim, 0)
        self.assertIs(minimum.dtype, torch.float32)
        self.assertEqual(minimum.device, pure.tensor.device)
        self.assertEqual(maximum.ndim, 0)
        self.assertIs(maximum.dtype, torch.float32)
        self.assertEqual(maximum.device, pure.tensor.device)

    def test_analog_waveform_delegates_exact_semantic_dtype_relationship(
        self,
    ) -> None:
        sampling = _sampling(count=4)
        pure = _pure([1.0, 2.0, 3.0, 4.0], sampling, dtype=torch.float64)
        noise = _noise([0.5, 0.25, 0.0, -0.25], sampling, dtype=torch.float64)
        with patch(
            "tensor_dslab.readout.analog_waveform._produce.require_same_dtype",
            wraps=require_same_dtype,
        ) as delegated:
            result = _produce_analog_waveform(
                pure,
                noise,
                config=AnalogWaveformConfig(),
            )
        delegated.assert_called_once_with(pure, noise)
        torch.testing.assert_close(
            result.tensor,
            pure.tensor + noise.tensor,
            rtol=0.0,
            atol=0.0,
        )

    def test_analog_waveform_rejects_axis_device_or_dtype_disagreement(self) -> None:
        sampling = _sampling(count=4)
        pure = _pure([0.0] * 4, sampling, dtype=torch.float32)
        mismatched_axes = _axes(
            sampling,
            start_ps=0,
            period_ps=4_000,
        )
        with self.assertRaises(ValueError):
            _produce_analog_waveform(
                pure,
                _noise(
                    [0.0] * 4,
                    sampling,
                    dtype=torch.float32,
                    axes=mismatched_axes,
                ),
                config=AnalogWaveformConfig(),
            )
        with self.assertRaises(ValueError):
            _produce_analog_waveform(
                pure,
                _noise([0.0] * 4, sampling, dtype=torch.float64),
                config=AnalogWaveformConfig(),
            )
        with self.assertRaises(ValueError):
            _produce_analog_waveform(
                pure,
                _noise(
                    [0.0] * 4,
                    sampling,
                    dtype=torch.float32,
                    device="meta",
                ),
                config=AnalogWaveformConfig(),
            )

    def test_analog_waveform_rejects_nonfinite_or_collapsed_dtype_bounds_before_addition(
        self,
    ) -> None:
        sampling = _sampling(count=4)
        pure = _pure([1.0] * 4, sampling, dtype=torch.float32)
        noise = _noise([2.0] * 4, sampling, dtype=torch.float32)
        bad_configs = (
            AnalogWaveformConfig(
                saturation=AnalogSaturationConfig(
                    maximum_mv=FiniteFloat(3.5e38)
                )
            ),
            AnalogWaveformConfig(
                saturation=AnalogSaturationConfig(
                    minimum_mv=FiniteFloat(1.0),
                    maximum_mv=FiniteFloat(1.0 + 1e-8),
                )
            ),
        )
        for config in bad_configs:
            with self.subTest(config=config):
                with patch(
                    "tensor_dslab.readout.analog_waveform._produce.torch.add",
                    side_effect=AssertionError("payload addition started"),
                ):
                    with self.assertRaises(ValueError):
                        _produce_analog_waveform(pure, noise, config=config)

    def test_analog_waveform_is_fresh_preserves_inputs_and_autograd(self) -> None:
        sampling = _sampling(count=5)
        axes = _axes(sampling, order=(SampleAxis, ExampleAxis, ChannelAxis))
        for dtype, rtol, atol in (
            (torch.float32, 2e-5, 2e-6),
            (torch.float64, 1e-12, 1e-12),
        ):
            with self.subTest(dtype=dtype):
                pure = _pure(
                    [-1.0, 0.5, 1.0, 2.0, 3.0],
                    sampling,
                    dtype=dtype,
                    axes=axes,
                    noncontiguous=True,
                    requires_grad=True,
                )
                noise = _noise(
                    [0.25, -0.25, 0.5, -0.5, 0.75],
                    sampling,
                    dtype=dtype,
                    axes=axes,
                    noncontiguous=True,
                    requires_grad=True,
                )
                self.assertFalse(pure.tensor.is_contiguous())
                self.assertFalse(noise.tensor.is_contiguous())
                pure_before = pure.tensor.clone()
                noise_before = noise.tensor.clone()
                pure_version = pure.tensor._version
                noise_version = noise.tensor._version
                config = AnalogWaveformConfig(
                    saturation=AnalogSaturationConfig(
                        minimum_mv=FiniteFloat(-10.0),
                        maximum_mv=FiniteFloat(10.0),
                    )
                )
                result = _produce_analog_waveform(pure, noise, config=config)
                self.assertIs(type(result), AnalogWaveform)
                self.assertIs(result.axes, pure.axes)
                self.assertEqual(result.shape, pure.shape)
                self.assertIs(result.tensor.dtype, dtype)
                self.assertEqual(result.tensor.device, pure.tensor.device)
                for result_axis, source_axis in zip(result.axes, pure.axes):
                    self.assertIs(result_axis, source_axis)
                self.assertTrue(_independent_storage(result.tensor, pure.tensor))
                self.assertTrue(_independent_storage(result.tensor, noise.tensor))
                self.assertEqual(pure.tensor._version, pure_version)
                self.assertEqual(noise.tensor._version, noise_version)
                torch.testing.assert_close(pure.tensor, pure_before)
                torch.testing.assert_close(noise.tensor, noise_before)
                saved_result = result.tensor.clone()
                _produce_analog_waveform(pure, noise, config=config)
                torch.testing.assert_close(result.tensor, saved_result)

                result.tensor.square().sum().backward()
                pure_gradient = pure.tensor.grad
                noise_gradient = noise.tensor.grad
                assert pure_gradient is not None
                assert noise_gradient is not None
                actual_pure_gradient = pure_gradient.clone()
                actual_noise_gradient = noise_gradient.clone()
                reference_pure = pure_before.detach().requires_grad_(True)
                reference_noise = noise_before.detach().requires_grad_(True)
                reference = torch.clamp(
                    reference_pure + reference_noise,
                    min=torch.tensor(-10.0, dtype=dtype),
                    max=torch.tensor(10.0, dtype=dtype),
                )
                reference.square().sum().backward()
                torch.testing.assert_close(
                    actual_pure_gradient,
                    reference_pure.grad,
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    actual_noise_gradient,
                    reference_noise.grad,
                    rtol=rtol,
                    atol=atol,
                )

        gradcheck_axes = _axes(sampling)
        first = _tensor_from_samples(
            [-1.0, 0.5, 1.0, 2.0, 3.0],
            gradcheck_axes,
            dtype=torch.float64,
            requires_grad=True,
        )
        second = _tensor_from_samples(
            [0.25, -0.25, 0.5, -0.5, 0.75],
            gradcheck_axes,
            dtype=torch.float64,
            requires_grad=True,
        )

        def analog_function(
            pure_values: torch.Tensor,
            noise_values: torch.Tensor,
        ) -> torch.Tensor:
            return _produce_analog_waveform(
                PureWaveform(tensor=pure_values, axes=gradcheck_axes),
                NoiseWaveform(tensor=noise_values, axes=gradcheck_axes),
                config=AnalogWaveformConfig(),
            ).tensor

        self.assertTrue(
            torch.autograd.gradcheck(
                analog_function,
                (first, second),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

    def test_digitized_waveform_matches_guarded_affine_reference(self) -> None:
        sampling = _sampling(count=7)
        axes = _axes(sampling, order=(SampleAxis, ExampleAxis, ChannelAxis))
        samples = [-500.0, -100.0, -1.0, 0.0, 1.0, 100.0, 500.0]
        config = _adc_config(
            bit_depth=12,
            input_min_mv=-400.0,
            input_max_mv=600.0,
            gain_db=3.5218,
        )
        for dtype in (torch.float32, torch.float64):
            analog = _analog(
                samples,
                sampling,
                dtype=dtype,
                axes=axes,
                noncontiguous=True,
            )
            result = _produce_digitized_waveform(analog, config=config)
            expected = _guarded_adc_reference(analog.tensor, config)
            self.assertTrue(torch.equal(result.tensor, expected))
            self.assertIs(type(result), DigitizedWaveform)
            self.assertIs(result.axes, analog.axes)
            self.assertEqual(result.shape, analog.shape)
            self.assertIs(result.tensor.dtype, torch.int32)
            self.assertEqual(result.tensor.device, analog.tensor.device)
            for result_axis, source_axis in zip(result.axes, analog.axes):
                self.assertIs(result_axis, source_axis)
            self.assertIsNone(require_valid_digitized(result, config))

    def test_digitized_waveform_endpoints_zero_code_and_gain(self) -> None:
        sampling = _sampling(count=5)
        config = _adc_config(
            bit_depth=4,
            input_min_mv=-2.0,
            input_max_mv=6.0,
            gain_db=20.0,
        )
        samples = [-1.0, -0.2, 0.0, 0.6, 1.0]
        for dtype in (torch.float32, torch.float64):
            analog = _analog(samples, sampling, dtype=dtype)
            result = _produce_digitized_waveform(analog, config=config)
            expected = _guarded_adc_reference(analog.tensor, config)
            self.assertTrue(torch.equal(result.tensor, expected))
            self.assertEqual(result.tensor.flatten()[0].item(), 0)
            self.assertEqual(result.tensor.flatten()[1].item(), 0)
            self.assertEqual(result.tensor.flatten()[2].item(), 3)
            self.assertEqual(result.tensor.flatten()[3].item(), 15)
            self.assertEqual(result.tensor.flatten()[4].item(), 15)

        endpoint_cases = (
            (torch.float32, _adc_config(bit_depth=2, input_min_mv=1.0, input_max_mv=5.0, gain_db=0.1)),
            (torch.float64, _adc_config(bit_depth=2, input_min_mv=-0.1, input_max_mv=0.3, gain_db=1.0)),
        )
        for dtype, endpoint_config in endpoint_cases:
            gain = 10.0 ** (endpoint_config.analog_gain_db.value / 20.0)
            span = endpoint_config.input_max_mv.value - endpoint_config.input_min_mv.value
            maximum_code = (1 << endpoint_config.bit_depth.value) - 1
            upper = torch.tensor(
                endpoint_config.input_max_mv.value / gain,
                dtype=dtype,
            )
            slope = torch.tensor(gain * maximum_code / span, dtype=dtype)
            intercept = torch.tensor(
                -endpoint_config.input_min_mv.value * maximum_code / span,
                dtype=dtype,
            )
            unguarded = upper * slope + intercept
            self.assertLess(float(unguarded), float(maximum_code))
            analog = _analog(
                [float(upper), float(upper)],
                _sampling(count=2),
                dtype=dtype,
            )
            guarded = _produce_digitized_waveform(analog, config=endpoint_config)
            self.assertTrue(
                torch.equal(
                    guarded.tensor,
                    torch.full_like(guarded.tensor, maximum_code),
                )
            )

    def test_digitized_waveform_rejects_nonfinite_or_collapsed_dtype_scalars_before_mapping(
        self,
    ) -> None:
        sampling = _sampling(count=2)
        analog = _analog([0.0, 1.0], sampling, dtype=torch.float32)
        configs = (
            _adc_config(
                input_min_mv=-3.4e38,
                input_max_mv=3.4e38,
            ),
            _adc_config(
                bit_depth=2,
                input_min_mv=1.0,
                input_max_mv=1.0 + 1e-8,
            ),
        )
        for config in configs:
            with self.subTest(config=config):
                with patch(
                    "tensor_dslab.readout.digitized_waveform._produce.torch.mul",
                    side_effect=AssertionError("payload mapping started"),
                ):
                    with self.assertRaises(ValueError):
                        _produce_digitized_waveform(analog, config=config)

    def test_digitized_waveform_truncates_at_code_transitions(self) -> None:
        sampling = _sampling(count=5)
        analog = _analog(
            [0.0, 0.2, 0.5, 0.9, 1.0],
            sampling,
            dtype=torch.float64,
        )
        result = _produce_digitized_waveform(
            analog,
            config=_adc_config(
                bit_depth=2,
                input_min_mv=0.0,
                input_max_mv=1.0,
            ),
        )
        self.assertTrue(
            torch.equal(
                result.tensor.flatten(),
                torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32),
            )
        )

    def test_digitized_waveform_clips_before_int32_conversion_without_wraparound(
        self,
    ) -> None:
        sampling = _sampling(count=4)
        analog = _analog(
            [-1e20, -2.0, 2.0, 1e20],
            sampling,
            dtype=torch.float32,
        )
        result = _produce_digitized_waveform(
            analog,
            config=_adc_config(
                bit_depth=16,
                input_min_mv=-1.0,
                input_max_mv=1.0,
            ),
        )
        self.assertTrue(
            torch.equal(
                result.tensor.flatten(),
                torch.tensor([0, 0, 65_535, 65_535], dtype=torch.int32),
            )
        )

    def test_digitized_waveform_is_fresh_int32_and_nondifferentiable(self) -> None:
        sampling = _sampling(count=4)
        analog = _analog(
            [-1.0, 0.0, 1.0, 2.0],
            sampling,
            dtype=torch.float64,
            requires_grad=True,
        )
        config = _adc_config(
            bit_depth=8,
            input_min_mv=-1.0,
            input_max_mv=1.0,
        )
        before = analog.tensor.clone()
        version = analog.tensor._version
        real_tensor = torch.tensor
        created: list[torch.Tensor] = []

        def tensor_spy(
            value: float,
            *,
            dtype: torch.dtype,
            device: torch.device,
        ) -> torch.Tensor:
            result = real_tensor(value, dtype=dtype, device=device)
            created.append(result)
            return result

        with patch(
            "tensor_dslab.readout.digitized_waveform._produce.torch.tensor",
            side_effect=tensor_spy,
        ):
            result = _produce_digitized_waveform(analog, config=config)
        self.assertIs(type(result), DigitizedWaveform)
        self.assertIs(result.axes, analog.axes)
        self.assertEqual(result.shape, analog.shape)
        self.assertIs(result.tensor.dtype, torch.int32)
        self.assertEqual(result.tensor.device, analog.tensor.device)
        self.assertFalse(result.tensor.requires_grad)
        self.assertIsNone(result.tensor.grad_fn)
        self.assertTrue(_independent_storage(result.tensor, analog.tensor))
        self.assertEqual(analog.tensor._version, version)
        torch.testing.assert_close(analog.tensor, before)
        self.assertGreaterEqual(len(created), 6)
        for scalar in created:
            self.assertEqual(scalar.ndim, 0)
            self.assertIs(scalar.dtype, analog.tensor.dtype)
            self.assertEqual(scalar.device, analog.tensor.device)
        gain = 10.0 ** (config.analog_gain_db.value / 20.0)
        span = config.input_max_mv.value - config.input_min_mv.value
        maximum_code = (1 << config.bit_depth.value) - 1
        expected_execution_scalars = torch.tensor(
            (
                0.0,
                maximum_code,
                gain * maximum_code / span,
                -config.input_min_mv.value * maximum_code / span,
                config.input_min_mv.value / gain,
                config.input_max_mv.value / gain,
            ),
            dtype=analog.tensor.dtype,
            device=analog.tensor.device,
        )
        torch.testing.assert_close(
            torch.stack(created[-6:]),
            expected_execution_scalars,
            rtol=0.0,
            atol=0.0,
        )
        saved_result = result.tensor.clone()
        _produce_digitized_waveform(analog, config=config)
        self.assertTrue(torch.equal(result.tensor, saved_result))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_deterministic_products_run_conditionally_on_cuda(self) -> None:
        sampling = _sampling(count=8)
        axes = _axes(sampling, order=(SampleAxis, ExampleAxis, ChannelAxis))
        for dtype, rtol, atol in (
            (torch.float32, 2e-5, 2e-6),
            (torch.float64, 1e-12, 1e-12),
        ):
            with self.subTest(dtype=dtype):
                charge = _charge(
                    [1.0] + [0.0] * 7,
                    sampling,
                    dtype=dtype,
                    axes=axes,
                    device="cuda",
                    noncontiguous=True,
                )
                pure = _produce_pure_waveform(
                    charge,
                    sampling=sampling,
                    config=_tpc_config(support_time_ns=32.0),
                )
                expected_pure = _reference_pure(
                    charge.tensor,
                    charge.axes,
                    sampling,
                    _tpc_config(support_time_ns=32.0),
                )
                torch.testing.assert_close(
                    pure.tensor,
                    expected_pure,
                    rtol=rtol,
                    atol=atol,
                )
                noise = _noise(
                    [0.0] * 8,
                    sampling,
                    dtype=dtype,
                    axes=axes,
                    device="cuda",
                    noncontiguous=True,
                )
                analog = _produce_analog_waveform(
                    pure,
                    noise,
                    config=AnalogWaveformConfig(),
                )
                digitized = _produce_digitized_waveform(
                    analog,
                    config=_adc_config(),
                )
                self.assertEqual(pure.tensor.device.type, "cuda")
                self.assertEqual(analog.tensor.device.type, "cuda")
                self.assertEqual(digitized.tensor.device.type, "cuda")
                self.assertIsNone(require_valid_pure(pure))
                self.assertIsNone(require_valid_analog(analog))
                self.assertIsNone(require_valid_digitized(digitized, _adc_config()))


if __name__ == "__main__":
    unittest.main()
