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
    AXIS_ORDERS,
    _FailingRng,
    _sampling,
    _produce_noise_waveform,
    _photoelectrons,
    _zero_config,
    _white_config,
    _flat_psd_config,
    _independent_storage,
    _round,
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
            "tensor_dslab.readout.noise_waveform.runtime.produce."
            "RngElements.from_shape",
            side_effect=AssertionError("zero noise must not build elements"),
        ) as elements:
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
        elements.assert_not_called()
        self.assertTrue(torch.equal(first.tensor, torch.zeros_like(first.tensor)))
        self.assertTrue(torch.equal(first.tensor, second.tensor))
        self.assertTrue(_independent_storage(first.tensor, second.tensor))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_private_preparer_relies_on_public_dtype_and_device_admission(
        self,
    ) -> None:
        sampling = _sampling(count=8)
        source = _photoelectrons(sampling)
        runtime = prepare_noise_waveform(
            _zero_config(),
            sampling=sampling,
            shape=source.shape,
            floating_dtype=torch.float16,
            device=source.tensor.device,
        )
        self.assertIs(runtime.floating_dtype, torch.float16)
        meta_source = _photoelectrons(sampling, device="meta")
        meta_runtime = prepare_noise_waveform(
            _zero_config(),
            sampling=sampling,
            shape=meta_source.shape,
            floating_dtype=torch.float32,
            device=meta_source.tensor.device,
        )
        self.assertEqual(meta_runtime.device.type, "meta")

    def test_white_matches_finite_lattice_equation_without_demeaning(self) -> None:
        sampling = _sampling(count=7)
        source = _photoelectrons(sampling, examples=1, channels=1)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                rms_input = 0.1
                represented_rms_mv = _round(rms_input, dtype)
                result = _produce_noise_waveform(
                    source,
                    sampling=sampling,
                    config=_white_config(rms_input),
                    rng=Threefry4x32(seed=0x0123456789ABCDEF),
                    floating_dtype=dtype,
                )
                model = _white_config(rms_input).model
                assert type(model) is WhiteNoiseConfig
                elements = RngElements.from_shape(
                    source.shape,
                    device=source.tensor.device,
                )
                expected = GaussianDistribution(
                    mean=0.0,
                    standard_deviation=represented_rms_mv,
                    dtype=dtype,
                    ordinal=0,
                    count=1,
                ).draw(
                    rng=Threefry4x32(seed=0x0123456789ABCDEF),
                    address=RngAddress.root(
                        key=WHITE_NOISE_RNG_KEY,
                        elements=elements,
                        shape=(),
                    ),
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
        elements = RngElements.from_shape(first_source.shape, device="cpu")
        psd_model = _flat_psd_config().model
        assert type(psd_model) is PsdNoiseConfig
        other_stream = GaussianDistribution(
            mean=0.0,
            standard_deviation=1.0,
            dtype=torch.float64,
            ordinal=0,
            count=1,
        ).draw(
            rng=Threefry4x32(seed=99),
            address=RngAddress.root(
                key=PSD_NOISE_RNG_KEY,
                elements=elements,
                shape=(),
            ),
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
