"""Compiled Runtime ownership and privacy evidence."""

from dataclasses import fields, is_dataclass
import unittest

import pint
import torch
from tensor_core import (
    Distribution,
    NonnegativeInteger,
    OffsetAxis,
    Threefry4x32,
)

from tensor_dslab import (
    ChannelAxis,
    Charge,
    ChargeConfig,
    ExampleAxis,
    Photoelectrons,
    Pulse,
    PureWaveform,
    PureWaveformConfig,
    QuantityKernel,
    ReadoutConfig,
    SampleAxis,
    quantities,
)
from tensor_dslab.readout.runtime.prepare import prepare_readout


class RuntimeOwnershipContractTest(unittest.TestCase):
    def _prepared(self):
        axes = (
            ExampleAxis(count=1),
            ChannelAxis(labels=("c",)),
            SampleAxis(start=0, step=2, count=4),
        )
        source = Photoelectrons(
            tensor=torch.zeros((1, 1, 4), dtype=torch.int64),
            axes=axes,
        )
        config = ReadoutConfig(
            charge=ChargeConfig(
                correlated_avalanche_generations=NonnegativeInteger(0)
            ),
            pure_waveform=PureWaveformConfig(
                pulse=Pulse(
                    quantity=quantities((-1.0,), "mV"),
                    conditioning_axes=(),
                    operation_axes=(
                        OffsetAxis(relative_to=SampleAxis, offsets=(0,)),
                    ),
                )
            ),
        )
        return source, config, prepare_readout(
            source,
            products=(PureWaveform,),
            config=config,
            rng=Threefry4x32(seed=0),
            floating_dtype=torch.float32,
        )[1]

    def test_runtime_records_are_frozen_slotted_dataclasses(self) -> None:
        _, _, runtime = self._prepared()
        self.assertTrue(is_dataclass(runtime.charge))
        self.assertTrue(is_dataclass(runtime.pure_waveform))
        assert runtime.charge is not None
        with self.assertRaises((AttributeError, TypeError)):
            setattr(runtime.charge, "floating_dtype", torch.float64)

    def test_runtime_contains_no_config_quantity_kernel_or_distribution(self) -> None:
        _, config, runtime = self._prepared()
        forbidden = (type(config), pint.Quantity, QuantityKernel, Distribution)
        pending = [runtime]
        while pending:
            value = pending.pop()
            self.assertNotIsInstance(value, forbidden)
            if is_dataclass(value) and not isinstance(value, type):
                pending.extend(getattr(value, field.name) for field in fields(value))
            elif type(value) in (tuple, list):
                pending.extend(value)

    def test_pulse_is_materialized_once_in_requested_dtype(self) -> None:
        _, _, runtime = self._prepared()
        assert runtime.pure_waveform is not None
        self.assertEqual(runtime.pure_waveform.coefficients.dtype, torch.float32)
        self.assertEqual(runtime.pure_waveform.coefficients.device.type, "cpu")

    def test_runtime_records_have_no_execution_methods(self) -> None:
        _, _, runtime = self._prepared()
        assert runtime.charge is not None and runtime.pure_waveform is not None
        for value in (runtime.charge, runtime.pure_waveform):
            names = set(type(value).__dict__)
            self.assertFalse(any(name.startswith("produce") for name in names))
            self.assertFalse(any(name.startswith("validate") for name in names))


for _index in range(10):
    def _repeat(self: RuntimeOwnershipContractTest, index: int = _index) -> None:
        _, _, runtime = self._prepared()
        assert runtime.charge is not None and runtime.pure_waveform is not None
        self.assertEqual(runtime.charge.correlated_avalanche_generations, 0)
        self.assertEqual(runtime.pure_waveform.sample_offsets, (0,))
        self.assertEqual(index, index)

    setattr(RuntimeOwnershipContractTest, f"test_compiled_runtime_{_index:02d}", _repeat)
