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
    SEEDS,
    _sampling,
    _produce_noise_waveform,
    _photoelectrons,
    _zero_config,
    _white_config,
    _flat_psd_config,
    _sample_last,
    _round,
    _reference_psd_powers,
    _delta,
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
