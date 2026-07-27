"""Literal Pulse convolution and downstream deterministic product evidence."""

import unittest

import torch
from tensor_core import NonnegativeInteger, OffsetAxis, Threefry4x32

from tensor_dslab import (
    AnalogWaveform,
    AnalogWaveformConfig,
    ChannelAxis,
    Charge,
    ChargeConfig,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    ExampleAxis,
    NoiseWaveformConfig,
    Photoelectrons,
    Pulse,
    PureWaveform,
    PureWaveformConfig,
    ReadoutConfig,
    SampleAxis,
    ZeroNoiseConfig,
    quantities,
    quantity,
    simulate_readout,
)
from tensor_core import NonnegativeFloat, PositiveInteger
from tensor_dslab.readout.pure_waveform.runtime.prepare import (
    prepare_pure_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.produce import (
    produce_pure_waveform,
)
from tensor_dslab.readout.runtime.sampling import prepare_sampling


def _config(coefficients: tuple[float, ...], offsets: tuple[int, ...]) -> ReadoutConfig:
    return ReadoutConfig(
        charge=ChargeConfig(
            correlated_avalanche_generations=NonnegativeInteger(0)
        ),
        pure_waveform=PureWaveformConfig(
            pulse=Pulse(
                quantity=quantities(coefficients, "mV"),
                conditioning_axes=(),
                operation_axes=(
                    OffsetAxis(relative_to=SampleAxis, offsets=offsets),
                ),
            )
        ),
        noise_waveform=NoiseWaveformConfig(model=ZeroNoiseConfig()),
        analog_waveform=AnalogWaveformConfig(),
        digitized_waveform=DigitizedWaveformConfig(
            bit_depth=PositiveInteger(12),
            input_minimum=quantity(-10, "mV"),
            input_maximum=quantity(10, "mV"),
            analog_gain_db=NonnegativeFloat(0),
        ),
    )


def _source(values: torch.Tensor) -> Photoelectrons:
    return Photoelectrons(
        tensor=values,
        axes=(
            ExampleAxis(count=values.shape[0]),
            ChannelAxis(labels=tuple(f"c{i}" for i in range(values.shape[1]))),
            SampleAxis(start=0, step=2, count=values.shape[2]),
        ),
    )


class DeterministicWaveformContractTest(unittest.TestCase):
    def test_literal_convolution_and_finite_boundary(self) -> None:
        source = _source(torch.tensor([[[2, 0, 3, 0]]], dtype=torch.int64))
        result = simulate_readout(
            source,
            products=(PureWaveform,),
            config=_config((-2.0, -1.0), (0, 1)),
            rng=Threefry4x32(seed=0),
        ).field(PureWaveform)
        self.assertEqual(result.tensor.tolist(), [[[-4.0, -2.0, -6.0, -3.0]]])

    def test_nonconsecutive_offsets_are_literal(self) -> None:
        source = _source(torch.tensor([[[1, 0, 0, 0]]], dtype=torch.int64))
        result = simulate_readout(
            source,
            products=(PureWaveform,),
            config=_config((-3.0, -1.0), (0, 3)),
            rng=Threefry4x32(seed=0),
        ).field(PureWaveform)
        self.assertEqual(result.tensor.tolist(), [[[-3.0, 0.0, 0.0, -1.0]]])

    def test_analog_is_exact_pure_plus_noise_and_digitizer_is_bounded(self) -> None:
        source = _source(torch.tensor([[[1, 0, 0, 0]]], dtype=torch.int64))
        result = simulate_readout(
            source,
            products=(PureWaveform, AnalogWaveform, DigitizedWaveform),
            config=_config((-2.0,), (0,)),
            rng=Threefry4x32(seed=0),
        )
        pure = result.field(PureWaveform)
        analog = result.field(AnalogWaveform)
        digitized = result.field(DigitizedWaveform)
        self.assertTrue(torch.equal(analog.tensor, pure.tensor))
        self.assertTrue(torch.all((digitized.tensor >= 0) & (digitized.tensor <= 4095)))

    def test_no_hidden_period_factor_or_polarity_flip(self) -> None:
        source = _source(torch.tensor([[[1, 0, 0, 0]]], dtype=torch.int64))
        result = simulate_readout(
            source,
            products=(PureWaveform,),
            config=_config((-7.0,), (0,)),
            rng=Threefry4x32(seed=0),
        ).field(PureWaveform)
        self.assertEqual(float(result.tensor[0, 0, 0]), -7.0)

    def test_literal_convolution_preserves_charge_autograd(self) -> None:
        source = _source(torch.zeros((1, 1, 4), dtype=torch.int64))
        pulse_config = _config((-2.0, -1.0), (0, 1)).pure_waveform
        assert pulse_config is not None
        runtime = prepare_pure_waveform(
            pulse_config,
            source=source,
            sampling=prepare_sampling(source),
            floating_dtype=torch.float64,
            device=torch.device("cpu"),
        )
        charge_tensor = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        charge = Charge(tensor=charge_tensor, axes=source.axes)
        result = produce_pure_waveform(charge, runtime=runtime)
        result.tensor.sum().backward()
        assert charge_tensor.grad is not None
        self.assertTrue(
            torch.equal(
                charge_tensor.grad,
                torch.tensor(
                    [[[-3.0, -3.0, -3.0, -2.0]]],
                    dtype=torch.float64,
                ),
            )
        )
        self.assertNotEqual(
            result.tensor.untyped_storage().data_ptr(),
            charge_tensor.untyped_storage().data_ptr(),
        )


for _offset in range(18):
    def _impulse_case(
        self: DeterministicWaveformContractTest,
        offset: int = _offset,
    ) -> None:
        sample_count = 20
        source = _source(torch.tensor([[[1] + [0] * (sample_count - 1)]], dtype=torch.int64))
        result = simulate_readout(
            source,
            products=(PureWaveform,),
            config=_config((-1.0,), (offset,)),
            rng=Threefry4x32(seed=offset),
        ).field(PureWaveform)
        expected = torch.zeros_like(result.tensor)
        expected[..., offset] = -1
        self.assertTrue(torch.equal(result.tensor, expected))

    setattr(
        DeterministicWaveformContractTest,
        f"test_impulse_offset_{_offset:02d}",
        _impulse_case,
    )
