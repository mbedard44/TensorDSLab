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
    _FailingRng,
    _hz,
    _hzs,
    _densities,
    _sampling,
    _prepare_psd_powers,
    _produce_noise_waveform,
    _photoelectrons,
    _flat_psd_config,
    _reference_psd_powers,
)

class PsdPreparationTest(unittest.TestCase):
    def test_odd_even_cells_overlap_fsum_conservation_and_one_rounding(self) -> None:
        for count in (5, 6):
            sampling = _sampling(count=count)
            sample_rate = 1.0e12 / sampling.sample_period_ps
            spacing = sample_rate / count
            edges = (0.0, spacing / 2.0, 1.75 * spacing, 0.45 * sample_rate)
            model = PsdNoiseConfig(
                frequency_left_edges=_hzs(edges),
                frequency_stop=_hz(sample_rate / 2.0),
                power_density=_densities(
                    (1.0e-9, 2.0e-9, 0.0, 3.0e-9)
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
                frequency_left_edges=_hzs((0.0,)),
                frequency_stop=_hz(400_000_000.0),
                power_density=_densities((1.0e-9,)),
            ),
            PsdNoiseConfig(
                frequency_left_edges=_hzs((0.0, 125_000_000.0)),
                frequency_stop=_hz(500_000_000.0),
                power_density=_densities((1.0e-9, 0.0)),
            ),
            PsdNoiseConfig(
                frequency_left_edges=_hzs((0.0, 500_000_000.0)),
                frequency_stop=_hz(600_000_000.0),
                power_density=_densities((0.0, 1.0e-9)),
            ),
            PsdNoiseConfig(
                frequency_left_edges=_hzs((0.0,)),
                frequency_stop=_hz(500_000_000.0),
                power_density=_densities((4.0e-59,)),
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
            if call_count == sampling.sample_count // 2 + 2:
                return math.inf
            return original_fsum(materialized)

        with patch(
            "tensor_dslab.readout.noise_waveform.runtime.prepare.math.fsum",
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
        final_call = sampling.sample_count // 2 + 2

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                normal_guard = 8.0 if dtype is torch.float32 else 16.0
                limit = torch.finfo(dtype).max / (
                    sampling.sample_count * normal_guard
                )
                above_limit = math.nextafter(limit, math.inf)
                self.assertTrue(math.isfinite(limit))
                self.assertLessEqual(
                    sampling.sample_count * normal_guard * limit,
                    torch.finfo(dtype).max,
                )
                self.assertGreater(
                    sampling.sample_count * normal_guard * above_limit,
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
                    "tensor_dslab.readout.noise_waveform.runtime.prepare.math.fsum",
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
                    "tensor_dslab.readout.noise_waveform.runtime.prepare.math.fsum",
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
