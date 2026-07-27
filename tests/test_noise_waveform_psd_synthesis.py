from collections.abc import Iterable
from itertools import permutations
import math
from typing import Any, override
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    CounterRng,
    GaussianDistribution,
    NonnegativeFloat,
    PositiveFloat,
    RngAddress,
    RngElements,
    RngKey,
    TensorAxis,
    Threefry4x32,
)

from tensor_dslab import (
    quantities,
    quantity,
    ChannelAxis,
    ExampleAxis,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    SampleAxis,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
)
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    _prepare_psd_powers as _prepare_psd_powers_prepared,
    prepare_noise_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.produce import (
    produce_noise_waveform as _produce_noise_waveform_prepared,
)
from tensor_dslab.readout.noise_waveform.runtime.validate import (
    validate_noise_waveform,
)
from tensor_dslab.readout.runtime.sampling import SamplingRuntime, prepare_sampling
from tensor_dslab.readout.runtime.keys import (
    PSD_NOISE_RNG_KEY,
    WHITE_NOISE_RNG_KEY,
)

from tests._noise_waveform_support import (
    _hz,
    _hzs,
    _densities,
    _sampling,
    _prepare_psd_powers,
    _produce_noise_waveform,
    _photoelectrons,
    _flat_psd_config,
    _sample_last,
    _psd_normals,
)

class PsdSynthesisTest(unittest.TestCase):
    def test_small_odd_even_reference_coefficients_dc_and_zero_power(self) -> None:
        for count in (5, 6):
            sampling = _sampling(count=count)
            source = _photoelectrons(sampling, examples=1, channels=1)
            sample_rate = 1.0e12 / sampling.sample_period_ps
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
                frequency_left_edges=_hzs(target_boundaries[:-1]),
                frequency_stop=_hz(target_boundaries[-1]),
                power_density=_densities(densities),
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
                    "tensor_dslab.readout.noise_waveform.runtime.produce.torch.fft.irfft",
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
                    "tensor_dslab.readout.noise_waveform.runtime.produce.torch.fft.irfft",
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
