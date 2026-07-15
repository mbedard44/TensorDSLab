from __future__ import annotations

from enum import Enum, IntEnum
import math
import unittest
from unittest.mock import patch

import torch
from tensor_core import NonnegativeFloat, PositiveFloat, PositiveInteger

from tensor_dslab import (
    ChannelAxis,
    ExampleAxis,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    SamplingConfig,
)
from tensor_dslab.readout._random import (
    _RngStream,
    _address_words,
    _logical_positions,
    _random_block,
    _raw_word,
    _standard_normal_pair,
    _threefry4x32,
    _uniform_closed_open,
    _uniform_open_open,
)
from tensor_dslab.readout.noise_waveform._product import (
    _product_noise_waveform,
)


MASK = (1 << 32) - 1
PARITY = 0x1BD11BDA
ROTATIONS = (
    (10, 26),
    (11, 21),
    (13, 27),
    (23, 5),
    (6, 20),
    (17, 11),
    (25, 10),
    (18, 20),
)


def _rotate(value: int, distance: int) -> int:
    return ((value << distance) & MASK) | (value >> (32 - distance))


def _scalar_threefry(
    counter: tuple[int, int, int, int],
    key: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    schedule = (*key, PARITY ^ key[0] ^ key[1] ^ key[2] ^ key[3])
    values = tuple((counter[index] + key[index]) & MASK for index in range(4))
    x0, x1, x2, x3 = values
    for round_index in range(20):
        rotation_0, rotation_1 = ROTATIONS[round_index % 8]
        if round_index % 2 == 0:
            x0 = (x0 + x1) & MASK
            x1 = _rotate(x1, rotation_0) ^ x0
            x2 = (x2 + x3) & MASK
            x3 = _rotate(x3, rotation_1) ^ x2
        else:
            x0 = (x0 + x3) & MASK
            x3 = _rotate(x3, rotation_0) ^ x0
            x2 = (x2 + x1) & MASK
            x1 = _rotate(x1, rotation_1) ^ x2
        if (round_index + 1) % 4 == 0:
            injection = (round_index + 1) // 4
            x0 = (x0 + schedule[injection % 5]) & MASK
            x1 = (x1 + schedule[(injection + 1) % 5]) & MASK
            x2 = (x2 + schedule[(injection + 2) % 5]) & MASK
            x3 = (x3 + schedule[(injection + 3) % 5] + injection) & MASK
    return x0, x1, x2, x3


def _reference_address(
    *,
    seed: int,
    stream: int,
    position: int,
    quantum: int,
    ordinal: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    block, lane = divmod(ordinal, 4)
    return (
        (seed & MASK, (seed >> 32) & MASK, stream, 0x54445331),
        (position & MASK, (position >> 32) & MASK, quantum, block),
        lane,
    )


class RngStreamContractTest(unittest.TestCase):
    def test_exact_strongly_typed_stream_registry(self) -> None:
        self.assertTrue(issubclass(_RngStream, Enum))
        self.assertFalse(issubclass(_RngStream, IntEnum))
        self.assertEqual(
            tuple((member.name, member.value) for member in _RngStream),
            (
                ("NOISE_WHITE", 0x0000_0001),
                ("NOISE_PSD_COEFFICIENT", 0x0000_0002),
            ),
        )
        self.assertEqual(len(_RngStream.__members__), 2)
        self.assertNotIn(0, tuple(member.value for member in _RngStream))


class ThreefryKnownAnswerTest(unittest.TestCase):
    def test_random123_fixed_vectors_and_batched_execution(self) -> None:
        vectors = (
            (
                (0x00000000, 0x00000000, 0x00000000, 0x00000000),
                (0x00000000, 0x00000000, 0x00000000, 0x00000000),
                (0x9C6CA96A, 0xE17EAE66, 0xFC10ECD4, 0x5256A7D8),
            ),
            (
                (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
                (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
                (0x2A881696, 0x57012287, 0xF6C7446E, 0xA16A6732),
            ),
            (
                (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
                (0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89),
                (0x59CD1DBB, 0xB8879579, 0x86B5D00C, 0xAC8B6D84),
            ),
        )
        counters = torch.tensor([item[0] for item in vectors], dtype=torch.int64)
        keys = torch.tensor([item[1] for item in vectors], dtype=torch.int64)
        expected = torch.tensor([item[2] for item in vectors], dtype=torch.int64)
        self.assertTrue(torch.equal(_threefry4x32(counters, keys), expected))
        for counter, key, fixed in vectors:
            with self.subTest(counter=counter, key=key):
                self.assertEqual(_scalar_threefry(counter, key), fixed)


class AddressSchemaTest(unittest.TestCase):
    def test_packing_halves_streams_bounds_and_rollover(self) -> None:
        cases = (
            (0x0123456789ABCDEF, 1, (1 << 32) - 1, 0, 3),
            (0xFEDCBA9876543210, 2, 1 << 32, (1 << 32) - 1, 4),
            ((1 << 64) - 1, 2, (1 << 63) - 1, (1 << 32) - 1, (1 << 34) - 1),
        )
        for seed, stream, position, quantum, ordinal in cases:
            with self.subTest(
                seed=seed,
                stream=stream,
                position=position,
                quantum=quantum,
                ordinal=ordinal,
            ):
                expected = _reference_address(
                    seed=seed,
                    stream=stream,
                    position=position,
                    quantum=quantum,
                    ordinal=ordinal,
                )
                self.assertEqual(
                    _address_words(
                        seed=seed,
                        stream=stream,
                        logical_position=position,
                        source_quantum=quantum,
                        raw_word_ordinal=ordinal,
                    ),
                    expected,
                )

        key_3, counter_3, lane_3 = _address_words(
            seed=0,
            stream=1,
            logical_position=0,
            source_quantum=0,
            raw_word_ordinal=3,
        )
        key_4, counter_4, lane_4 = _address_words(
            seed=0,
            stream=1,
            logical_position=0,
            source_quantum=0,
            raw_word_ordinal=4,
        )
        self.assertEqual(key_3, key_4)
        self.assertEqual(counter_3[3], 0)
        self.assertEqual(counter_4[3], 1)
        self.assertEqual((lane_3, lane_4), (3, 0))

    def test_raw_words_match_independent_scalar_oracle(self) -> None:
        for stream in _RngStream:
            for seed, position, quantum, ordinal in (
                (0, 0, 0, 0),
                (0x0123456789ABCDEF, (1 << 32) + 7, 0, 3),
                ((1 << 64) - 1, (1 << 63) - 1, (1 << 32) - 1, (1 << 34) - 1),
            ):
                key, counter, lane = _reference_address(
                    seed=seed,
                    stream=stream.value,
                    position=position,
                    quantum=quantum,
                    ordinal=ordinal,
                )
                expected = _scalar_threefry(counter, key)[lane]
                actual = int(
                    _raw_word(
                        seed=seed,
                        stream=stream,
                        logical_position=position,
                        source_quantum=quantum,
                        raw_word_ordinal=ordinal,
                    )
                )
                self.assertEqual(actual, expected)

    def test_every_numeric_address_bound_rejects_bool_negative_and_stop(self) -> None:
        cases = (
            ("seed", 1 << 64),
            ("stream", 1 << 32),
            ("logical_position", 1 << 63),
            ("source_quantum", 1 << 32),
            ("raw_word_ordinal", 1 << 34),
        )
        baseline = {
            "seed": 0,
            "stream": 1,
            "logical_position": 0,
            "source_quantum": 0,
            "raw_word_ordinal": 0,
        }
        for field, stop in cases:
            for invalid in (True, -1, stop):
                with self.subTest(field=field, invalid=invalid):
                    arguments = dict(baseline)
                    arguments[field] = invalid
                    expected_error = TypeError if invalid is True else ValueError
                    with self.assertRaises(expected_error):
                        _address_words(**arguments)

    def test_random_block_uses_logical_position_not_storage_stride(self) -> None:
        contiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        backing = torch.empty((3, 4, 2), dtype=torch.int64)
        noncontiguous = backing[..., 0]
        noncontiguous.copy_(contiguous)
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertTrue(torch.equal(contiguous, noncontiguous))
        expected = _random_block(
            seed=7,
            stream=_RngStream.NOISE_WHITE,
            logical_positions=contiguous,
        )
        actual = _random_block(
            seed=7,
            stream=_RngStream.NOISE_WHITE,
            logical_positions=noncontiguous,
        )
        self.assertTrue(torch.equal(actual, expected))


class UniformConversionTest(unittest.TestCase):
    def test_float32_lattices_endpoints_discarded_bits_and_midpoints(self) -> None:
        zero = torch.tensor(0, dtype=torch.int64)
        maximum = torch.tensor(MASK, dtype=torch.int64)
        self.assertEqual(float(_uniform_closed_open(zero, dtype=torch.float32)), 0.0)
        self.assertEqual(
            float(_uniform_closed_open(maximum, dtype=torch.float32)),
            1.0 - 2.0**-24,
        )
        self.assertEqual(
            float(_uniform_open_open(zero, dtype=torch.float32)),
            2.0**-24,
        )
        self.assertEqual(
            float(_uniform_open_open(maximum, dtype=torch.float32)),
            1.0 - 2.0**-24,
        )

        closed_base = torch.tensor(0x12345600, dtype=torch.int64)
        self.assertTrue(
            torch.equal(
                _uniform_closed_open(closed_base, dtype=torch.float32),
                _uniform_closed_open(closed_base + 0xFF, dtype=torch.float32),
            )
        )
        self.assertFalse(
            torch.equal(
                _uniform_closed_open(closed_base, dtype=torch.float32),
                _uniform_closed_open(closed_base + 0x100, dtype=torch.float32),
            )
        )
        open_base = torch.tensor(0x12345600, dtype=torch.int64)
        self.assertTrue(
            torch.equal(
                _uniform_open_open(open_base, dtype=torch.float32),
                _uniform_open_open(open_base + 0x1FF, dtype=torch.float32),
            )
        )
        self.assertFalse(
            torch.equal(
                _uniform_open_open(open_base, dtype=torch.float32),
                _uniform_open_open(open_base + 0x200, dtype=torch.float32),
            )
        )

        below = torch.tensor(((1 << 22) - 1) << 9, dtype=torch.int64)
        above = torch.tensor((1 << 22) << 9, dtype=torch.int64)
        self.assertEqual(
            float(_uniform_open_open(below, dtype=torch.float32)),
            0.5 - 2.0**-24,
        )
        self.assertEqual(
            float(_uniform_open_open(above, dtype=torch.float32)),
            0.5 + 2.0**-24,
        )

    def test_float64_lattices_endpoints_discarded_bits_and_midpoints(self) -> None:
        zero = torch.tensor(0, dtype=torch.int64)
        maximum = torch.tensor(MASK, dtype=torch.int64)
        self.assertEqual(
            float(
                _uniform_closed_open(
                    zero,
                    word_1=zero,
                    dtype=torch.float64,
                )
            ),
            0.0,
        )
        self.assertEqual(
            float(
                _uniform_closed_open(
                    maximum,
                    word_1=maximum,
                    dtype=torch.float64,
                )
            ),
            1.0 - 2.0**-53,
        )
        self.assertEqual(
            float(
                _uniform_open_open(
                    zero,
                    word_1=zero,
                    dtype=torch.float64,
                )
            ),
            2.0**-53,
        )
        self.assertEqual(
            float(
                _uniform_open_open(
                    maximum,
                    word_1=maximum,
                    dtype=torch.float64,
                )
            ),
            1.0 - 2.0**-53,
        )

        high = torch.tensor(0x12345678, dtype=torch.int64)
        low = torch.tensor(0x9ABCD000, dtype=torch.int64)
        self.assertTrue(
            torch.equal(
                _uniform_closed_open(high, word_1=low, dtype=torch.float64),
                _uniform_closed_open(high, word_1=low + 0x7FF, dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                _uniform_open_open(high, word_1=low, dtype=torch.float64),
                _uniform_open_open(high, word_1=low + 0xFFF, dtype=torch.float64),
            )
        )

        below_m52 = (1 << 51) - 1
        above_m52 = 1 << 51
        below_high, below_low = divmod(below_m52, 1 << 20)
        above_high, above_low = divmod(above_m52, 1 << 20)
        below = _uniform_open_open(
            torch.tensor(below_high, dtype=torch.int64),
            word_1=torch.tensor(below_low << 12, dtype=torch.int64),
            dtype=torch.float64,
        )
        above = _uniform_open_open(
            torch.tensor(above_high, dtype=torch.int64),
            word_1=torch.tensor(above_low << 12, dtype=torch.int64),
            dtype=torch.float64,
        )
        self.assertEqual(float(below), 0.5 - 2.0**-53)
        self.assertEqual(float(above), 0.5 + 2.0**-53)


class BoxMullerAndPositionTest(unittest.TestCase):
    def test_exact_word_schedule_ordered_components_and_repeatability(self) -> None:
        positions = torch.tensor((0, 1, 7), dtype=torch.int64)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                first = _standard_normal_pair(
                    seed=0x0123456789ABCDEF,
                    stream=_RngStream.NOISE_WHITE,
                    logical_positions=positions,
                    dtype=dtype,
                )
                second = _standard_normal_pair(
                    seed=0x0123456789ABCDEF,
                    stream=_RngStream.NOISE_WHITE,
                    logical_positions=positions,
                    dtype=dtype,
                )
                self.assertTrue(torch.equal(first[0], second[0]))
                self.assertTrue(torch.equal(first[1], second[1]))

                words = torch.stack(
                    tuple(
                        torch.tensor(
                            [
                                _scalar_threefry(
                                    _reference_address(
                                        seed=0x0123456789ABCDEF,
                                        stream=_RngStream.NOISE_WHITE.value,
                                        position=position,
                                        quantum=0,
                                        ordinal=0,
                                    )[1],
                                    _reference_address(
                                        seed=0x0123456789ABCDEF,
                                        stream=_RngStream.NOISE_WHITE.value,
                                        position=position,
                                        quantum=0,
                                        ordinal=0,
                                    )[0],
                                )[lane]
                                for lane in range(4)
                            ],
                            dtype=torch.int64,
                        )
                        for position in positions.tolist()
                    )
                )
                if dtype is torch.float32:
                    open_value = (
                        torch.tensor(0.5, dtype=dtype)
                        + (words[:, 0] >> 9).to(dtype)
                    ) * torch.tensor(2.0**-23, dtype=dtype)
                    closed_value = (words[:, 1] >> 8).to(dtype) * torch.tensor(
                        2.0**-24,
                        dtype=dtype,
                    )
                else:
                    open_mantissa = words[:, 0] * (1 << 20) + (words[:, 1] >> 12)
                    closed_mantissa = words[:, 2] * (1 << 21) + (words[:, 3] >> 11)
                    open_value = (
                        torch.tensor(0.5, dtype=dtype) + open_mantissa.to(dtype)
                    ) * torch.tensor(2.0**-52, dtype=dtype)
                    closed_value = closed_mantissa.to(dtype) * torch.tensor(
                        2.0**-53,
                        dtype=dtype,
                    )
                radius = torch.sqrt(torch.tensor(-2.0, dtype=dtype) * torch.log(open_value))
                angle = torch.tensor(math.tau, dtype=dtype) * closed_value
                expected = (radius * torch.cos(angle), radius * torch.sin(angle))
                tolerance = 64 * torch.finfo(dtype).eps * max(
                    float(torch.max(torch.abs(expected[0]))),
                    float(torch.max(torch.abs(expected[1]))),
                    torch.finfo(dtype).tiny,
                )
                self.assertTrue(torch.allclose(first[0], expected[0], rtol=0.0, atol=tolerance))
                self.assertTrue(torch.allclose(first[1], expected[1], rtol=0.0, atol=tolerance))

    def test_finite_lattice_radial_cutoffs(self) -> None:
        positions = torch.tensor((0,), dtype=torch.int64)
        zero_words = torch.zeros((1, 4), dtype=torch.int64)
        with patch(
            "tensor_dslab.readout._random._random_block",
            return_value=zero_words,
        ):
            float32_pair = _standard_normal_pair(
                seed=0,
                stream=_RngStream.NOISE_WHITE,
                logical_positions=positions,
                dtype=torch.float32,
            )
            float64_pair = _standard_normal_pair(
                seed=0,
                stream=_RngStream.NOISE_WHITE,
                logical_positions=positions,
                dtype=torch.float64,
            )
        self.assertAlmostEqual(float(float32_pair[0][0]), 5.7681075, places=5)
        self.assertAlmostEqual(float(float64_pair[0][0]), 8.5716743, places=7)
        self.assertEqual(float(float32_pair[1][0]), 0.0)
        self.assertEqual(float(float64_pair[1][0]), 0.0)

    def test_float32_uniforms_and_normals_ignore_ambient_cpu_autocast(self) -> None:
        words = torch.tensor((0x12345678, 0x9ABCDEF0), dtype=torch.int64)
        positions = torch.tensor((0, 1, 2), dtype=torch.int64)
        expected_open = _uniform_open_open(words, dtype=torch.float32)
        expected_closed = _uniform_closed_open(words, dtype=torch.float32)
        expected_normal = _standard_normal_pair(
            seed=23,
            stream=_RngStream.NOISE_WHITE,
            logical_positions=positions,
            dtype=torch.float32,
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual_open = _uniform_open_open(words, dtype=torch.float32)
            actual_closed = _uniform_closed_open(words, dtype=torch.float32)
            actual_normal = _standard_normal_pair(
                seed=23,
                stream=_RngStream.NOISE_WHITE,
                logical_positions=positions,
                dtype=torch.float32,
            )
        self.assertIs(actual_open.dtype, torch.float32)
        self.assertIs(actual_closed.dtype, torch.float32)
        self.assertTrue(torch.equal(actual_open, expected_open))
        self.assertTrue(torch.equal(actual_closed, expected_closed))
        self.assertTrue(torch.equal(actual_normal[0], expected_normal[0]))
        self.assertTrue(torch.equal(actual_normal[1], expected_normal[1]))

    def test_arbitrary_rank_scalar_empty_and_global_rng_immutability(self) -> None:
        state = torch.random.get_rng_state().clone()
        self.assertEqual(_logical_positions((), device="cpu").shape, ())
        self.assertEqual(int(_logical_positions((), device="cpu")), 0)
        self.assertEqual(_logical_positions((2, 0, 3), device="cpu").numel(), 0)
        positions = _logical_positions((2, 3, 4), device="cpu")
        self.assertTrue(
            torch.equal(positions.reshape(-1), torch.arange(24, dtype=torch.int64))
        )
        _standard_normal_pair(
            seed=9,
            stream=_RngStream.NOISE_WHITE,
            logical_positions=positions,
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))

    def test_psd_positions_have_every_positive_frequency_and_no_dc(self) -> None:
        sampling = SamplingConfig(
            sample_period_ps=PositiveInteger(1_000),
            sample_count=PositiveInteger(6),
        )
        axes = (
            ExampleAxis(coordinates=("e0", "e1")),
            ChannelAxis(coordinates=("c0", "c1", "c2")),
            sampling.build_axis(),
        )
        photoelectrons = Photoelectrons(
            tensor=torch.zeros((2, 3, 6), dtype=torch.int64),
            axes=axes,
        )
        config = NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                frequency_stop_hz=PositiveFloat(500_000_000.0),
                power_density_mv2_per_hz=(NonnegativeFloat(2.0e-9),),
            )
        )
        captured: list[torch.Tensor] = []

        def fake_pair(
            *,
            seed: int,
            stream: _RngStream,
            logical_positions: torch.Tensor,
            dtype: torch.dtype,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.assertEqual(seed, 4)
            self.assertIs(stream, _RngStream.NOISE_PSD_COEFFICIENT)
            captured.append(logical_positions.clone())
            zeros = torch.zeros_like(logical_positions, dtype=dtype)
            return zeros, zeros

        with patch(
            "tensor_dslab.readout.noise_waveform._product._standard_normal_pair",
            side_effect=fake_pair,
        ):
            _product_noise_waveform(
                photoelectrons,
                sampling=sampling,
                config=config,
                seed=4,
                floating_dtype=torch.float64,
            )
        frequency_count = 4
        expected = torch.tensor(
            [
                [row * frequency_count + frequency for frequency in range(1, 4)]
                for row in range(6)
            ],
            dtype=torch.int64,
        )
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0], expected))
        self.assertFalse(bool(torch.any(captured[0] % frequency_count == 0)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_raw_words_and_uniforms_match_cpu_exactly(self) -> None:
        positions = torch.tensor(
            (0, 1, (1 << 32) - 1, 1 << 32, (1 << 63) - 1),
            dtype=torch.int64,
        )
        for stream in _RngStream:
            cpu_words = _random_block(
                seed=(1 << 64) - 1,
                stream=stream,
                logical_positions=positions,
            )
            cuda_words = _random_block(
                seed=(1 << 64) - 1,
                stream=stream,
                logical_positions=positions.to("cuda"),
            )
            self.assertTrue(torch.equal(cpu_words, cuda_words.cpu()))
            for dtype in (torch.float32, torch.float64):
                if dtype is torch.float32:
                    cpu_uniform = _uniform_open_open(cpu_words[:, 0], dtype=dtype)
                    cuda_uniform = _uniform_open_open(cuda_words[:, 0], dtype=dtype)
                else:
                    cpu_uniform = _uniform_open_open(
                        cpu_words[:, 0],
                        word_1=cpu_words[:, 1],
                        dtype=dtype,
                    )
                    cuda_uniform = _uniform_open_open(
                        cuda_words[:, 0],
                        word_1=cuda_words[:, 1],
                        dtype=dtype,
                    )
                self.assertTrue(torch.equal(cpu_uniform, cuda_uniform.cpu()))


if __name__ == "__main__":
    unittest.main()
