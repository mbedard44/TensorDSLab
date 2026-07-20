from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from itertools import combinations, product
import math
from typing import ClassVar
import unittest
from unittest.mock import patch

import torch
from tensor_core import (
    CounterRng,
    FiniteFloat,
    NonnegativeFloat,
    NonnegativeInteger,
    PositiveFloat,
    PositiveInteger,
    Probability,
    RngKey,
    TensorField,
    Threefry4x32,
)

from tensor_dslab import (
    AfterpulseConfig,
    AnalogSaturationConfig,
    AnalogWaveform,
    AnalogWaveformConfig,
    ChannelAxis,
    Charge,
    ChargeConfig,
    ChargeSmearingConfig,
    CorrelatedAvalancheConfig,
    DarkCountConfig,
    DelayedCrosstalkConfig,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    DirectCrosstalkConfig,
    ExampleAxis,
    FixedDelayConfig,
    NoiseWaveform,
    NoiseWaveformConfig,
    Photoelectrons,
    PsdNoiseConfig,
    PureWaveform,
    PureWaveformConfig,
    ReadoutCollection,
    ReadoutConfig,
    SampleAxis,
    SamplingConfig,
    TimingJitterConfig,
    TpcFebSnrPulseConfig,
    WhiteNoiseConfig,
    ZeroNoiseConfig,
    simulate_readout,
)
from tensor_dslab.readout import simulation
from tensor_dslab.readout.analog_waveform import _produce as analog_producer
from tensor_dslab.readout.charge import _produce as charge_producer
from tensor_dslab.readout.digitized_waveform import (
    _produce as digitized_producer,
)
from tensor_dslab.readout.noise_waveform import _produce as noise_producer
from tensor_dslab.readout.pure_waveform import _produce as pure_producer
from tests.readout_fixtures import ForeignField


PRODUCT_TYPES: tuple[type[TensorField], ...] = (
    Photoelectrons,
    Charge,
    PureWaveform,
    NoiseWaveform,
    AnalogWaveform,
    DigitizedWaveform,
)
GENERATED_TYPES: tuple[type[TensorField], ...] = PRODUCT_TYPES[1:]
PREPARERS: tuple[tuple[str, str], ...] = (
    ("_prepare_charge", "charge"),
    ("_prepare_pure_waveform", "pure"),
    ("_prepare_noise_waveform", "noise"),
    ("_prepare_analog_waveform", "analog"),
    ("_prepare_digitized_waveform", "digitized"),
)
PRODUCERS: tuple[tuple[str, str], ...] = (
    ("_produce_charge", "charge"),
    ("_produce_pure_waveform", "pure"),
    ("_produce_noise_waveform", "noise"),
    ("_produce_analog_waveform", "analog"),
    ("_produce_digitized_waveform", "digitized"),
)


class _FailingRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[int] = 0

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls += 1
        raise AssertionError(
            f"unexpected RNG request: {key=}, {quantum=}, {block=}"
        )


class _DynamicFailureRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[int] = 0

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls += 1
        raise RuntimeError("custom backend rejected the genuine request")


class _RecordingRng(CounterRng):
    __slots__ = ()

    calls: ClassVar[list[tuple[RngKey, torch.Tensor, int, int]]] = []

    def _generate_block(
        self,
        *,
        key: RngKey,
        positions: torch.Tensor,
        quantum: int,
        block: int,
    ) -> torch.Tensor:
        type(self).calls.append((key, positions.clone(), quantum, block))
        return torch.zeros(
            positions.shape + (4,),
            dtype=torch.int64,
            device=positions.device,
        )

    @classmethod
    def reset(cls) -> None:
        cls.calls = []


class _OneShotProducts:
    def __init__(self, values: tuple[type[TensorField], ...]) -> None:
        self._values = values
        self.iterations = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("product request was consumed more than once")
        return iter(self._values)


def _sampling(*, count: int = 4, period_ps: int = 1_000) -> SamplingConfig:
    return SamplingConfig(
        sample_period_ps=PositiveInteger(period_ps),
        sample_count=PositiveInteger(count),
    )


def _axes(
    sampling: SamplingConfig,
    *,
    sample_first: bool = False,
) -> tuple[ExampleAxis | ChannelAxis | SampleAxis, ...]:
    example = ExampleAxis(coordinates=("example-0", "example-1"))
    channel = ChannelAxis(coordinates=("channel-0", "channel-1"))
    sample = sampling.build_axis()
    if sample_first:
        return (sample, example, channel)
    return (example, channel, sample)


def _photoelectrons(
    sampling: SamplingConfig,
    *,
    sample_first: bool = False,
    noncontiguous: bool = False,
    device: torch.device | str = "cpu",
) -> Photoelectrons:
    axes = _axes(sampling, sample_first=sample_first)
    shape = tuple(axis.size for axis in axes)
    values = torch.arange(
        math.prod(shape),
        dtype=torch.int64,
        device=device,
    ).reshape(shape)
    values = torch.remainder(values, 4)
    if noncontiguous:
        backing = torch.empty((*shape, 2), dtype=torch.int64, device=device)
        view = backing[..., 0]
        view.copy_(values)
        values = view
    return Photoelectrons(tensor=values, axes=axes)


def _pure_config(*, support_time_ns: float = 3.0) -> PureWaveformConfig:
    return PureWaveformConfig(
        model=TpcFebSnrPulseConfig(
            fast_time_constant_ns=PositiveFloat(1.0),
            slow_time_constant_ns=PositiveFloat(2.0),
            support_time_ns=PositiveFloat(support_time_ns),
            peak_voltage_mv_per_pe=FiniteFloat(-2.0),
        )
    )


def _digitized_config(
    *,
    input_min_mv: float = -20.0,
    input_max_mv: float = 20.0,
) -> DigitizedWaveformConfig:
    return DigitizedWaveformConfig(
        bit_depth=PositiveInteger(12),
        input_min_mv=FiniteFloat(input_min_mv),
        input_max_mv=FiniteFloat(input_max_mv),
        analog_gain_db=NonnegativeFloat(0.0),
    )


def _config(
    sampling: SamplingConfig,
    *,
    charge: ChargeConfig | None = None,
    pure: PureWaveformConfig | None = None,
    noise: NoiseWaveformConfig | None = None,
    analog: AnalogWaveformConfig | None = None,
    digitized: DigitizedWaveformConfig | None = None,
) -> ReadoutConfig:
    return ReadoutConfig(
        sampling=sampling,
        charge=ChargeConfig() if charge is None else charge,
        pure_waveform=_pure_config() if pure is None else pure,
        noise_waveform=(
            NoiseWaveformConfig(model=ZeroNoiseConfig())
            if noise is None
            else noise
        ),
        analog_waveform=AnalogWaveformConfig() if analog is None else analog,
        digitized_waveform=(
            _digitized_config() if digitized is None else digitized
        ),
    )


def _requested_closure(
    requested: frozenset[type[TensorField]],
) -> tuple[bool, bool, bool, bool, bool]:
    need_digitized = DigitizedWaveform in requested
    need_analog = AnalogWaveform in requested or need_digitized
    need_pure = PureWaveform in requested or need_analog
    need_noise = NoiseWaveform in requested or need_analog
    need_charge = Charge in requested or need_pure
    return need_charge, need_pure, need_noise, need_analog, need_digitized


def _occupied_bytes(tensor: torch.Tensor) -> frozenset[int]:
    if tensor.numel() == 0:
        return frozenset()
    element_size = tensor.element_size()
    storage_address = tensor.untyped_storage().data_ptr()
    addresses: set[int] = set()
    for index in product(*(range(size) for size in tensor.shape)):
        element_offset = tensor.storage_offset() + sum(
            coordinate * stride
            for coordinate, stride in zip(index, tensor.stride())
        )
        start = storage_address + element_offset * element_size
        addresses.update(range(start, start + element_size))
    return frozenset(addresses)


def _assert_no_storage_overlap(
    test: unittest.TestCase,
    left: torch.Tensor,
    right: torch.Tensor,
) -> None:
    test.assertTrue(
        _occupied_bytes(left).isdisjoint(_occupied_bytes(right)),
        msg=(
            "tensor payloads occupy overlapping storage bytes: "
            f"left(shape={tuple(left.shape)}, stride={left.stride()}), "
            f"right(shape={tuple(right.shape)}, stride={right.stride()})"
        ),
    )


def _run_recorded(
    source: Photoelectrons,
    *,
    products: tuple[type[TensorField], ...],
    config: ReadoutConfig,
) -> tuple[ReadoutCollection, simulation._ReadoutPlan, tuple[str, ...]]:
    events: list[str] = []
    plans: list[simulation._ReadoutPlan] = []
    actual_prepare_readout = simulation._prepare_readout

    def capture_plan(*args, **kwargs):  # type: ignore[no-untyped-def]
        plan = actual_prepare_readout(*args, **kwargs)
        plans.append(plan)
        return plan

    with ExitStack() as stack:
        for function_name, label in PREPARERS + PRODUCERS:
            actual = getattr(simulation, function_name)

            def record_call(
                *args,
                _actual=actual,
                _label=label,
                _kind="prepare" if function_name.startswith("_prepare") else "produce",
                **kwargs,
            ):  # type: ignore[no-untyped-def]
                events.append(f"{_kind}:{_label}")
                return _actual(*args, **kwargs)

            stack.enter_context(
                patch.object(simulation, function_name, side_effect=record_call)
            )
        stack.enter_context(
            patch.object(simulation, "_prepare_readout", side_effect=capture_plan)
        )
        result = simulate_readout(
            source,
            products=products,
            config=config,
            rng=_FailingRng(seed=0),
        )
    if len(plans) != 1:
        raise AssertionError("simulate_readout must prepare exactly one readout plan")
    return result, plans[0], tuple(events)


class ReadoutRequestAndClosureTest(unittest.TestCase):
    def test_all_63_requests_have_exact_closure_execution_and_retention(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        config = _config(sampling)
        seen = 0

        for count in range(1, len(PRODUCT_TYPES) + 1):
            for subset in combinations(PRODUCT_TYPES, count):
                requested = frozenset(subset)
                expected_closure = _requested_closure(requested)
                needed_labels = tuple(
                    label
                    for label, needed in zip(
                        ("charge", "pure", "noise", "analog", "digitized"),
                        expected_closure,
                    )
                    if needed
                )
                expected_events = tuple(
                    [f"prepare:{label}" for label in needed_labels]
                    + [f"produce:{label}" for label in needed_labels]
                )
                with self.subTest(
                    subset=tuple(field_type.__name__ for field_type in subset)
                ):
                    forward, plan, events = _run_recorded(
                        source,
                        products=subset,
                        config=config,
                    )
                    reverse, reverse_plan, reverse_events = _run_recorded(
                        source,
                        products=tuple(reversed(subset)),
                        config=config,
                    )

                    self.assertEqual(plan.requested, requested)
                    self.assertEqual(
                        (
                            plan.need_charge,
                            plan.need_pure,
                            plan.need_noise,
                            plan.need_analog,
                            plan.need_digitized,
                        ),
                        expected_closure,
                    )
                    self.assertEqual(reverse_plan.requested, requested)
                    self.assertEqual(events, expected_events)
                    self.assertEqual(reverse_events, expected_events)
                    self.assertEqual(forward.field_types, requested)
                    self.assertEqual(reverse.field_types, requested)
                    expected_order = tuple(
                        field_type
                        for field_type in PRODUCT_TYPES
                        if field_type in requested
                    )
                    self.assertEqual(tuple(forward.fields), expected_order)
                    self.assertEqual(tuple(reverse.fields), expected_order)
                    for field_type in expected_order:
                        self.assertTrue(
                            torch.equal(
                                forward.tensor(field_type),
                                reverse.tensor(field_type),
                            )
                        )
                    if Photoelectrons in requested:
                        self.assertIs(forward.field(Photoelectrons), source)
                        self.assertIs(reverse.field(Photoelectrons), source)
                    for field_type in PRODUCT_TYPES:
                        if field_type not in requested:
                            with self.assertRaises(KeyError):
                                forward.field(field_type)
                    seen += 1
        self.assertEqual(seen, 63)

    def test_product_iterable_is_consumed_once(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        requested = _OneShotProducts((Photoelectrons, AnalogWaveform))
        result = simulate_readout(
            source,
            products=requested,
            config=_config(sampling),
            rng=_FailingRng(seed=0),
        )
        self.assertEqual(requested.iterations, 1)
        self.assertEqual(
            result.field_types,
            frozenset({Photoelectrons, AnalogWaveform}),
        )

    def test_readout_collection_is_constructed_once_after_execution(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        with patch.object(
            simulation,
            "ReadoutCollection",
            wraps=ReadoutCollection,
        ) as collection_constructor:
            result = simulate_readout(
                source,
                products=PRODUCT_TYPES,
                config=_config(sampling),
                rng=_FailingRng(seed=0),
            )
        collection_constructor.assert_called_once()
        self.assertIs(type(result), ReadoutCollection)
        self.assertEqual(result.field_types, frozenset(PRODUCT_TYPES))

    def test_malformed_requests_fail_before_producers_or_rng(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        malformed: tuple[tuple[object, type[BaseException]], ...] = (
            ((), ValueError),
            ((Photoelectrons, Photoelectrons), ValueError),
            ((source,), TypeError),
            ((TensorField,), TypeError),
            ((ForeignField,), TypeError),
            ((object,), TypeError),
            ((17,), TypeError),
        )
        for products_value, error_type in malformed:
            with self.subTest(products=products_value):
                rng = _FailingRng(seed=0)
                with ExitStack() as stack:
                    producer_mocks = tuple(
                        stack.enter_context(
                            patch.object(
                                simulation,
                                function_name,
                                side_effect=AssertionError(
                                    "producer ran during request validation"
                                ),
                            )
                        )
                        for function_name, _ in PRODUCERS
                    )
                    with self.assertRaises(error_type):
                        simulate_readout(
                            source,
                            products=products_value,  # type: ignore[arg-type]
                            config=_config(sampling),
                            rng=rng,
                        )
                self.assertEqual(rng.calls, 0)
                for producer_mock in producer_mocks:
                    producer_mock.assert_not_called()

    def test_source_config_and_iterable_public_types_are_exact(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        wrong_source = Charge(
            tensor=source.tensor.to(dtype=torch.float32),
            axes=source.axes,
        )
        cases = (
            {
                "photoelectrons": wrong_source,
                "products": (Photoelectrons,),
                "config": _config(sampling),
            },
            {
                "photoelectrons": source,
                "products": (Photoelectrons,),
                "config": object(),
            },
            {
                "photoelectrons": source,
                "products": object(),
                "config": _config(sampling),
            },
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                rng = _FailingRng(seed=0)
                with ExitStack() as stack:
                    producer_mocks = tuple(
                        stack.enter_context(patch.object(simulation, name))
                        for name, _ in PRODUCERS
                    )
                    with self.assertRaises(TypeError):
                        simulate_readout(
                            arguments["photoelectrons"],  # type: ignore[arg-type]
                            products=arguments["products"],  # type: ignore[arg-type]
                            config=arguments["config"],  # type: ignore[arg-type]
                            rng=rng,
                        )
                self.assertEqual(rng.calls, 0)
                for producer_mock in producer_mocks:
                    producer_mock.assert_not_called()

    def test_every_transitive_missing_config_edge_fails_before_preparation(
        self,
    ) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        complete = _config(sampling)
        cases: tuple[tuple[type[TensorField], str], ...] = (
            (Charge, "charge"),
            (PureWaveform, "charge"),
            (PureWaveform, "pure_waveform"),
            (NoiseWaveform, "noise_waveform"),
            (AnalogWaveform, "charge"),
            (AnalogWaveform, "pure_waveform"),
            (AnalogWaveform, "noise_waveform"),
            (AnalogWaveform, "analog_waveform"),
            (DigitizedWaveform, "charge"),
            (DigitizedWaveform, "pure_waveform"),
            (DigitizedWaveform, "noise_waveform"),
            (DigitizedWaveform, "analog_waveform"),
            (DigitizedWaveform, "digitized_waveform"),
        )
        for requested, missing in cases:
            with self.subTest(requested=requested.__name__, missing=missing):
                config = replace(complete, **{missing: None})
                rng = _FailingRng(seed=0)
                with ExitStack() as stack:
                    preparation_mocks = tuple(
                        stack.enter_context(patch.object(simulation, name))
                        for name, _ in PREPARERS
                    )
                    producer_mocks = tuple(
                        stack.enter_context(patch.object(simulation, name))
                        for name, _ in PRODUCERS
                    )
                    with self.assertRaises(ValueError):
                        simulate_readout(
                            source,
                            products=(requested,),
                            config=config,
                            rng=rng,
                        )
                self.assertEqual(rng.calls, 0)
                for mock in preparation_mocks + producer_mocks:
                    mock.assert_not_called()

    def test_contextually_invalid_irrelevant_configs_are_unconsumed(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        baseline = _config(sampling)
        invalid_charge = ChargeConfig(
            smearing=ChargeSmearingConfig(
                relative_sigma=NonnegativeFloat(3.0e38)
            )
        )
        invalid_pure = _pure_config(support_time_ns=0.5)
        invalid_noise = NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                frequency_stop_hz=PositiveFloat(1.0),
                power_density_mv2_per_hz=(NonnegativeFloat(1.0),),
            )
        )
        invalid_analog = AnalogWaveformConfig(
            saturation=AnalogSaturationConfig(
                minimum_mv=FiniteFloat(1.0),
                maximum_mv=FiniteFloat(1.0 + 1.0e-8),
            )
        )
        invalid_digitized = _digitized_config(
            input_min_mv=1.0,
            input_max_mv=1.0 + 1.0e-8,
        )
        cases = (
            (NoiseWaveform, replace(baseline, charge=invalid_charge)),
            (Charge, replace(baseline, pure_waveform=invalid_pure)),
            (Charge, replace(baseline, noise_waveform=invalid_noise)),
            (PureWaveform, replace(baseline, analog_waveform=invalid_analog)),
            (AnalogWaveform, replace(baseline, digitized_waveform=invalid_digitized)),
        )
        for requested, candidate in cases:
            with self.subTest(requested=requested.__name__):
                expected = simulate_readout(
                    source,
                    products=(requested,),
                    config=baseline,
                    rng=_FailingRng(seed=0),
                )
                observed = simulate_readout(
                    source,
                    products=(requested,),
                    config=candidate,
                    rng=_FailingRng(seed=0),
                )
                self.assertTrue(
                    torch.equal(
                        expected.tensor(requested),
                        observed.tensor(requested),
                    )
                )


class ReadoutPreparationAndValidationTest(unittest.TestCase):
    def _assert_preflight_failure(
        self,
        error_type: type[BaseException],
        *,
        source: Photoelectrons,
        products: tuple[type[TensorField], ...],
        config: ReadoutConfig,
        floating_dtype: object = torch.float32,
        rng: CounterRng | object | None = None,
    ) -> None:
        source_values = source.tensor.clone()
        source_axes = source.axes
        source_stride = source.tensor.stride()
        global_rng_state = torch.random.get_rng_state().clone()
        exact_rng = _FailingRng(seed=0) if rng is None else rng
        with ExitStack() as stack:
            field_constructor_mocks = tuple(
                stack.enter_context(
                    patch.object(
                        module,
                        class_name,
                        side_effect=AssertionError(
                            "semantic field was constructed during preflight"
                        ),
                    )
                )
                for module, class_name in (
                    (charge_producer, "Charge"),
                    (pure_producer, "PureWaveform"),
                    (noise_producer, "NoiseWaveform"),
                    (analog_producer, "AnalogWaveform"),
                    (digitized_producer, "DigitizedWaveform"),
                )
            )
            producer_mocks = tuple(
                stack.enter_context(
                    patch.object(
                        simulation,
                        function_name,
                        side_effect=AssertionError(
                            "producer ran before complete preflight"
                        ),
                    )
                )
                for function_name, _ in PRODUCERS
            )
            collection_mock = stack.enter_context(
                patch.object(simulation, "ReadoutCollection")
            )
            with self.assertRaises(error_type):
                simulate_readout(
                    source,
                    products=products,
                    config=config,
                    rng=exact_rng,  # type: ignore[arg-type]
                    floating_dtype=floating_dtype,  # type: ignore[arg-type]
                )
        self.assertIs(source.axes, source_axes)
        self.assertEqual(source.tensor.stride(), source_stride)
        self.assertTrue(torch.equal(source.tensor, source_values))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_rng_state))
        for producer_mock in producer_mocks:
            producer_mock.assert_not_called()
        for field_constructor_mock in field_constructor_mocks:
            field_constructor_mock.assert_not_called()
        collection_mock.assert_not_called()
        if isinstance(exact_rng, _FailingRng):
            self.assertEqual(exact_rng.calls, 0)

    def test_real_request_source_sampling_and_dtype_failures_are_preflight(
        self,
    ) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)

        negative_values = source.tensor.clone()
        negative_values.reshape(-1)[0] = -1
        negative = Photoelectrons(tensor=negative_values, axes=source.axes)
        self._assert_preflight_failure(
            ValueError,
            source=negative,
            products=(DigitizedWaveform,),
            config=_config(sampling),
        )

        mismatched_sampling = _sampling(count=5)
        self._assert_preflight_failure(
            ValueError,
            source=source,
            products=(DigitizedWaveform,),
            config=_config(mismatched_sampling),
        )
        example, channel, _ = source.axes
        for coordinates in (
            ("1ps", "1001ps", "2001ps", "3001ps"),
            ("0ps", "2000ps", "4000ps", "6000ps"),
        ):
            with self.subTest(sample_coordinates=coordinates):
                mismatched_source = Photoelectrons(
                    tensor=source.tensor.clone(),
                    axes=(
                        example,
                        channel,
                        SampleAxis(coordinates=coordinates),
                    ),
                )
                self._assert_preflight_failure(
                    ValueError,
                    source=mismatched_source,
                    products=(DigitizedWaveform,),
                    config=_config(sampling),
                )
        self._assert_preflight_failure(
            TypeError,
            source=source,
            products=(DigitizedWaveform,),
            config=_config(sampling),
            floating_dtype=torch.int64,
        )

    def test_each_real_product_preparation_failure_precedes_all_producers(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        complete = _config(sampling)
        invalid_charge = replace(
            complete,
            charge=ChargeConfig(
                smearing=ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(3.0e38)
                )
            ),
        )
        invalid_pure = replace(
            complete,
            pure_waveform=_pure_config(support_time_ns=0.5),
        )
        invalid_white = replace(
            complete,
            noise_waveform=NoiseWaveformConfig(
                model=WhiteNoiseConfig(rms_mv=PositiveFloat(1.0e38))
            ),
        )
        invalid_psd = replace(
            complete,
            noise_waveform=NoiseWaveformConfig(
                model=PsdNoiseConfig(
                    frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                    frequency_stop_hz=PositiveFloat(1.0),
                    power_density_mv2_per_hz=(NonnegativeFloat(1.0),),
                )
            ),
        )
        invalid_analog = replace(
            complete,
            analog_waveform=AnalogWaveformConfig(
                saturation=AnalogSaturationConfig(
                    minimum_mv=FiniteFloat(1.0),
                    maximum_mv=FiniteFloat(1.0 + 1.0e-8),
                )
            ),
        )
        invalid_digitized = replace(
            complete,
            digitized_waveform=_digitized_config(
                input_min_mv=1.0,
                input_max_mv=1.0 + 1.0e-8,
            ),
        )
        for name, candidate in (
            ("charge", invalid_charge),
            ("pulse", invalid_pure),
            ("white noise", invalid_white),
            ("PSD noise", invalid_psd),
            ("analog", invalid_analog),
            ("digitized", invalid_digitized),
        ):
            with self.subTest(preparation=name):
                self._assert_preflight_failure(
                    ValueError,
                    source=source,
                    products=(DigitizedWaveform,),
                    config=candidate,
                )

    def test_successful_generated_products_run_each_deep_validator_once(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        validators = (
            charge_producer,
            pure_producer,
            noise_producer,
            analog_producer,
            digitized_producer,
        )
        with ExitStack() as stack:
            validator_mocks = tuple(
                stack.enter_context(
                    patch.object(
                        module,
                        "_require_valid_values",
                        wraps=module._require_valid_values,
                    )
                )
                for module in validators
            )
            result = simulate_readout(
                source,
                products=PRODUCT_TYPES,
                config=_config(sampling),
                rng=_FailingRng(seed=0),
            )
        for validator_mock in validator_mocks:
            validator_mock.assert_called_once()
        self.assertTrue(torch.all(torch.isfinite(result.tensor(Charge))).item())
        self.assertTrue(torch.all(result.tensor(Charge) >= 0.0).item())
        for field_type in (PureWaveform, NoiseWaveform, AnalogWaveform):
            self.assertTrue(
                torch.all(torch.isfinite(result.tensor(field_type))).item()
            )
        digitized = result.tensor(DigitizedWaveform)
        self.assertTrue(torch.all(digitized >= 0).item())
        self.assertTrue(torch.all(digitized <= 4095).item())

    def test_postcondition_failure_stops_downstream_and_collection(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        cases = (
            (
                pure_producer,
                (
                    "_produce_noise_waveform",
                    "_produce_analog_waveform",
                    "_produce_digitized_waveform",
                ),
            ),
            (
                noise_producer,
                ("_produce_analog_waveform", "_produce_digitized_waveform"),
            ),
            (analog_producer, ("_produce_digitized_waveform",)),
            (digitized_producer, ()),
        )
        for validator_module, downstream_names in cases:
            with self.subTest(module=validator_module.__name__):
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            validator_module,
                            "_require_valid_values",
                            side_effect=ValueError("forced deep postcondition"),
                        )
                    )
                    downstream = tuple(
                        stack.enter_context(patch.object(simulation, name))
                        for name in downstream_names
                    )
                    collection_mock = stack.enter_context(
                        patch.object(simulation, "ReadoutCollection")
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "forced deep postcondition",
                    ):
                        simulate_readout(
                            source,
                            products=(DigitizedWaveform,),
                            config=_config(sampling),
                            rng=_FailingRng(seed=0),
                        )
                for downstream_mock in downstream:
                    downstream_mock.assert_not_called()
                collection_mock.assert_not_called()

    def test_truth_only_ignores_floating_dtype_but_derived_closure_does_not(
        self,
    ) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        result = simulate_readout(
            source,
            products=(Photoelectrons,),
            config=ReadoutConfig(sampling=sampling),
            rng=_FailingRng(seed=0),
            floating_dtype=object(),  # type: ignore[arg-type]
        )
        self.assertIs(result.field(Photoelectrons), source)
        self._assert_preflight_failure(
            TypeError,
            source=source,
            products=(DigitizedWaveform,),
            config=_config(sampling),
            floating_dtype=object(),
        )

    def test_truth_only_rejects_non_cpu_cuda_device_before_output(self) -> None:
        sampling = _sampling()
        axes = _axes(sampling)
        source = Photoelectrons(
            tensor=torch.empty(
                tuple(axis.size for axis in axes),
                dtype=torch.int64,
                device="meta",
            ),
            axes=axes,
        )
        with patch.object(simulation, "ReadoutCollection") as collection_mock:
            with self.assertRaises(ValueError):
                simulate_readout(
                    source,
                    products=(Photoelectrons,),
                    config=ReadoutConfig(sampling=sampling),
                    rng=_FailingRng(seed=0),
                )
        collection_mock.assert_not_called()


class ReadoutRngContractTest(unittest.TestCase):
    def test_every_request_requires_nominal_counter_rng_without_dummy_draws(
        self,
    ) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        self.assertRaises(
            TypeError,
            simulate_readout,
            source,
            products=(Photoelectrons,),
            config=ReadoutConfig(sampling=sampling),
            rng=object(),
        )

        rng = _FailingRng(seed=0)
        truth = simulate_readout(
            source,
            products=(Photoelectrons,),
            config=ReadoutConfig(sampling=sampling),
            rng=rng,
        )
        deterministic = simulate_readout(
            source,
            products=PRODUCT_TYPES,
            config=_config(sampling),
            rng=rng,
        )
        self.assertIs(truth.field(Photoelectrons), source)
        self.assertEqual(deterministic.field_types, frozenset(PRODUCT_TYPES))
        self.assertEqual(rng.calls, 0)

    def test_conforming_non_threefry_rng_is_accepted_at_genuine_request(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        white = NoiseWaveformConfig(
            model=WhiteNoiseConfig(rms_mv=PositiveFloat(1.0))
        )
        _RecordingRng.reset()
        result = simulate_readout(
            source,
            products=(NoiseWaveform,),
            config=_config(sampling, noise=white),
            rng=_RecordingRng(seed=0),
        )
        self.assertIs(type(result.field(NoiseWaveform)), NoiseWaveform)
        self.assertGreater(len(_RecordingRng.calls), 0)

    def test_custom_backend_failure_occurs_once_at_first_real_request(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        white = NoiseWaveformConfig(
            model=WhiteNoiseConfig(rms_mv=PositiveFloat(1.0))
        )
        rng = _DynamicFailureRng(seed=0)
        with patch.object(simulation, "ReadoutCollection") as collection_mock:
            with self.assertRaisesRegex(RuntimeError, "custom backend rejected"):
                simulate_readout(
                    source,
                    products=(NoiseWaveform,),
                    config=_config(sampling, noise=white),
                    rng=rng,
                )
        self.assertEqual(rng.calls, 1)
        collection_mock.assert_not_called()

    def test_every_structural_noop_charge_role_participates_in_key_collisions(
        self,
    ) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        shared = RngKey(namespace=0xABCDEF01, stream=1)
        other = RngKey(namespace=0xABCDEF01, stream=2)
        delay = FixedDelayConfig(delay_ns=NonnegativeFloat(0.0))
        role_configs = (
            ChargeConfig(
                dark_count=DarkCountConfig(
                    rate_hz=NonnegativeFloat(0.0),
                    rng_key=shared,
                )
            ),
            ChargeConfig(
                timing_jitter=TimingJitterConfig(
                    sigma_ns=NonnegativeFloat(0.0),
                    rng_key=shared,
                )
            ),
            ChargeConfig(
                correlated_avalanches=CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(0),
                    direct_crosstalk=DirectCrosstalkConfig(
                        mean_offspring_per_parent=NonnegativeFloat(0.0),
                        delay=delay,
                        retained_rng_key=shared,
                        overflow_rng_key=other,
                    ),
                )
            ),
            ChargeConfig(
                correlated_avalanches=CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(0),
                    direct_crosstalk=DirectCrosstalkConfig(
                        mean_offspring_per_parent=NonnegativeFloat(0.0),
                        delay=delay,
                        retained_rng_key=other,
                        overflow_rng_key=shared,
                    ),
                )
            ),
            ChargeConfig(
                correlated_avalanches=CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(0),
                    delayed_crosstalk=DelayedCrosstalkConfig(
                        mean_offspring_per_parent=NonnegativeFloat(0.0),
                        delay=delay,
                        retained_rng_key=shared,
                        overflow_rng_key=other,
                    ),
                )
            ),
            ChargeConfig(
                correlated_avalanches=CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(0),
                    delayed_crosstalk=DelayedCrosstalkConfig(
                        mean_offspring_per_parent=NonnegativeFloat(0.0),
                        delay=delay,
                        retained_rng_key=other,
                        overflow_rng_key=shared,
                    ),
                )
            ),
            ChargeConfig(
                correlated_avalanches=CorrelatedAvalancheConfig(
                    maximum_generations=NonnegativeInteger(0),
                    afterpulse=AfterpulseConfig(
                        probability=Probability(0.0),
                        mean_delay_ns=PositiveFloat(1.0),
                        rng_key=shared,
                    ),
                )
            ),
            ChargeConfig(
                smearing=ChargeSmearingConfig(
                    relative_sigma=NonnegativeFloat(0.0),
                    rng_key=shared,
                )
            ),
        )
        noise = NoiseWaveformConfig(
            model=WhiteNoiseConfig(
                rms_mv=PositiveFloat(1.0),
                rng_key=shared,
            )
        )
        for charge in role_configs:
            with self.subTest(charge=charge):
                rng = _FailingRng(seed=0)
                with ExitStack() as stack:
                    producer_mocks = tuple(
                        stack.enter_context(patch.object(simulation, name))
                        for name, _ in PRODUCERS
                    )
                    with self.assertRaises(ValueError):
                        simulate_readout(
                            source,
                            products=(Charge, NoiseWaveform),
                            config=_config(
                                sampling,
                                charge=charge,
                                noise=noise,
                            ),
                            rng=rng,
                        )
                self.assertEqual(rng.calls, 0)
                for producer_mock in producer_mocks:
                    producer_mock.assert_not_called()

    def test_zero_absent_and_unrequested_branches_contribute_no_key(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        shared = RngKey(namespace=0xABCDEF02, stream=1)
        charge = ChargeConfig(
            dark_count=DarkCountConfig(
                rate_hz=NonnegativeFloat(0.0),
                rng_key=shared,
            )
        )
        white = NoiseWaveformConfig(
            model=WhiteNoiseConfig(
                rms_mv=PositiveFloat(1.0),
                rng_key=shared,
            )
        )
        zero = NoiseWaveformConfig(model=ZeroNoiseConfig())

        rng = _FailingRng(seed=0)
        result = simulate_readout(
            source,
            products=(Charge, NoiseWaveform),
            config=_config(sampling, charge=charge, noise=zero),
            rng=rng,
        )
        self.assertEqual(result.field_types, frozenset({Charge, NoiseWaveform}))
        self.assertEqual(rng.calls, 0)

        charge_only = simulate_readout(
            source,
            products=(Charge,),
            config=_config(sampling, charge=charge, noise=white),
            rng=_FailingRng(seed=0),
        )
        self.assertEqual(charge_only.field_types, frozenset({Charge}))

        _RecordingRng.reset()
        noise_only = simulate_readout(
            source,
            products=(NoiseWaveform,),
            config=_config(sampling, charge=charge, noise=white),
            rng=_RecordingRng(seed=0),
        )
        self.assertEqual(noise_only.field_types, frozenset({NoiseWaveform}))
        self.assertGreater(len(_RecordingRng.calls), 0)

    def test_psd_and_intra_charge_role_collisions_are_rejected(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        shared = RngKey(namespace=0xABCDEF04, stream=1)
        dark = DarkCountConfig(
            rate_hz=NonnegativeFloat(0.0),
            rng_key=shared,
        )
        psd = NoiseWaveformConfig(
            model=PsdNoiseConfig(
                frequency_left_edges_hz=(NonnegativeFloat(0.0),),
                frequency_stop_hz=PositiveFloat(500_000_000.0),
                power_density_mv2_per_hz=(NonnegativeFloat(1.0e-9),),
                rng_key=shared,
            )
        )
        cases = (
            (
                (Charge, NoiseWaveform),
                _config(
                    sampling,
                    charge=ChargeConfig(dark_count=dark),
                    noise=psd,
                ),
            ),
            (
                (Charge,),
                _config(
                    sampling,
                    charge=ChargeConfig(
                        dark_count=dark,
                        smearing=ChargeSmearingConfig(
                            relative_sigma=NonnegativeFloat(0.0),
                            rng_key=shared,
                        ),
                    ),
                ),
            ),
        )
        for requested, config in cases:
            with self.subTest(requested=requested):
                rng = _FailingRng(seed=0)
                with ExitStack() as stack:
                    producer_mocks = tuple(
                        stack.enter_context(patch.object(simulation, name))
                        for name, _ in PRODUCERS
                    )
                    with self.assertRaises(ValueError):
                        simulate_readout(
                            source,
                            products=requested,
                            config=config,
                            rng=rng,
                        )
                self.assertEqual(rng.calls, 0)
                for producer_mock in producer_mocks:
                    producer_mock.assert_not_called()


class ReadoutCompositionAndStorageTest(unittest.TestCase):
    def test_public_stochastic_results_match_direct_prepared_producers(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        charge_config = ChargeConfig(
            dark_count=DarkCountConfig(rate_hz=NonnegativeFloat(2.5e8))
        )
        noise_config = NoiseWaveformConfig(
            model=WhiteNoiseConfig(rms_mv=PositiveFloat(0.25))
        )
        self.assertIsNotNone(charge_config.dark_count)
        self.assertIs(type(noise_config.model), WhiteNoiseConfig)
        assert charge_config.dark_count is not None
        assert type(noise_config.model) is WhiteNoiseConfig
        self.assertNotEqual(
            charge_config.dark_count.rng_key,
            noise_config.model.rng_key,
        )
        seed = 0x0123_4567_89AB_CDEF
        dtype = torch.float64

        public = simulate_readout(
            source,
            products=(Charge, NoiseWaveform),
            config=_config(
                sampling,
                charge=charge_config,
                noise=noise_config,
            ),
            rng=Threefry4x32(seed=seed),
            floating_dtype=dtype,
        )
        charge_plan = charge_producer._prepare_charge(
            source,
            sampling=sampling,
            config=charge_config,
            floating_dtype=dtype,
        )
        direct_charge = charge_producer._produce_charge(
            source,
            plan=charge_plan,
            rng=Threefry4x32(seed=seed),
        )
        noise_plan = noise_producer._prepare_noise_waveform(
            source,
            sampling=sampling,
            config=noise_config,
            floating_dtype=dtype,
        )
        direct_noise = noise_producer._produce_noise_waveform(
            source,
            plan=noise_plan,
            rng=Threefry4x32(seed=seed),
        )
        self.assertTrue(
            torch.equal(public.tensor(Charge), direct_charge.tensor)
        )
        self.assertTrue(
            torch.equal(public.tensor(NoiseWaveform), direct_noise.tensor)
        )

    def test_real_composition_and_retention_are_invariant(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        config = _config(sampling)
        requests = (
            (Photoelectrons,),
            (Charge,),
            (PureWaveform,),
            (NoiseWaveform,),
            (AnalogWaveform,),
            (DigitizedWaveform,),
            (Photoelectrons, Charge),
            (PureWaveform, NoiseWaveform),
            (AnalogWaveform, Photoelectrons),
            (DigitizedWaveform, AnalogWaveform),
            PRODUCT_TYPES,
        )
        for requested in requests:
            with self.subTest(requested=requested):
                result = simulate_readout(
                    source,
                    products=requested,
                    config=config,
                    rng=_FailingRng(seed=0),
                    floating_dtype=torch.float64,
                )
                self.assertEqual(result.field_types, frozenset(requested))

        digitized_only = simulate_readout(
            source,
            products=(DigitizedWaveform,),
            config=config,
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float64,
        )
        full = simulate_readout(
            source,
            products=tuple(reversed(PRODUCT_TYPES)),
            config=config,
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float64,
        )
        self.assertTrue(
            torch.equal(
                digitized_only.tensor(DigitizedWaveform),
                full.tensor(DigitizedWaveform),
            )
        )
        analog_only = simulate_readout(
            source,
            products=(AnalogWaveform,),
            config=config,
            rng=_FailingRng(seed=0),
            floating_dtype=torch.float64,
        )
        self.assertTrue(
            torch.equal(
                analog_only.tensor(AnalogWaveform),
                full.tensor(AnalogWaveform),
            )
        )

        white_config = _config(
            sampling,
            noise=NoiseWaveformConfig(
                model=WhiteNoiseConfig(rms_mv=PositiveFloat(0.25))
            ),
        )
        noise_only = simulate_readout(
            source,
            products=(NoiseWaveform,),
            config=white_config,
            rng=Threefry4x32(seed=123),
        )
        retained_truth = simulate_readout(
            source,
            products=(Photoelectrons, NoiseWaveform),
            config=white_config,
            rng=Threefry4x32(seed=123),
        )
        self.assertTrue(
            torch.equal(
                noise_only.tensor(NoiseWaveform),
                retained_truth.tensor(NoiseWaveform),
            )
        )

    def test_source_axes_dtype_and_exact_occupied_storage_contract(self) -> None:
        sampling = _sampling()
        for sample_first in (False, True):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(sample_first=sample_first, dtype=dtype):
                    source = _photoelectrons(
                        sampling,
                        sample_first=sample_first,
                        noncontiguous=True,
                    )
                    source_values = source.tensor.clone()
                    source_stride = source.tensor.stride()
                    axes = source.axes
                    self.assertFalse(source.tensor.is_contiguous())
                    result = simulate_readout(
                        source,
                        products=PRODUCT_TYPES,
                        config=_config(sampling),
                        rng=_FailingRng(seed=0),
                        floating_dtype=dtype,
                    )
                    self.assertIs(result.field(Photoelectrons), source)
                    self.assertIs(source.axes, axes)
                    self.assertEqual(source.tensor.stride(), source_stride)
                    self.assertTrue(torch.equal(source.tensor, source_values))

                    generated = tuple(result.field(kind) for kind in GENERATED_TYPES)
                    for field_type in (Charge, NoiseWaveform, DigitizedWaveform):
                        self.assertFalse(result.tensor(field_type).requires_grad)
                    for field in generated:
                        self.assertIs(field.axes, axes)
                        self.assertTrue(
                            all(left is right for left, right in zip(field.axes, axes))
                        )
                        self.assertEqual(field.tensor.device, source.tensor.device)
                        expected_dtype = (
                            torch.int32
                            if type(field) is DigitizedWaveform
                            else dtype
                        )
                        self.assertIs(field.tensor.dtype, expected_dtype)
                        _assert_no_storage_overlap(self, field.tensor, source.tensor)
                    for left_index, left in enumerate(generated):
                        for right in generated[left_index + 1 :]:
                            _assert_no_storage_overlap(
                                self,
                                left.tensor,
                                right.tensor,
                            )

                    snapshots = {
                        type(field): field.tensor.clone() for field in generated[1:]
                    }
                    result.tensor(Charge).add_(100.0)
                    self.assertTrue(torch.equal(source.tensor, source_values))
                    for field_type, values in snapshots.items():
                        self.assertTrue(torch.equal(result.tensor(field_type), values))

    def test_no_later_write_occurs_through_a_constructed_field_alias(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        source_values = source.tensor.clone()
        captures: dict[
            type[TensorField],
            tuple[TensorField, torch.Tensor],
        ] = {}
        constructors = (
            (charge_producer, "Charge", Charge),
            (pure_producer, "PureWaveform", PureWaveform),
            (noise_producer, "NoiseWaveform", NoiseWaveform),
            (analog_producer, "AnalogWaveform", AnalogWaveform),
            (digitized_producer, "DigitizedWaveform", DigitizedWaveform),
        )
        with ExitStack() as stack:
            constructor_mocks = []
            for module, name, field_type in constructors:
                def construct_and_capture(
                    *args,
                    _field_type=field_type,
                    **kwargs,
                ):  # type: ignore[no-untyped-def]
                    field = _field_type(*args, **kwargs)
                    captures[_field_type] = (field, field.tensor.clone())
                    return field

                constructor_mocks.append(
                    stack.enter_context(
                        patch.object(
                            module,
                            name,
                            side_effect=construct_and_capture,
                        )
                    )
                )
            result = simulate_readout(
                source,
                products=PRODUCT_TYPES,
                config=_config(sampling),
                rng=_FailingRng(seed=0),
            )
        self.assertEqual(set(captures), set(GENERATED_TYPES))
        for constructor_mock in constructor_mocks:
            constructor_mock.assert_called_once()
        for field_type, (field, values_at_construction) in captures.items():
            self.assertIs(result.field(field_type), field)
            self.assertTrue(torch.equal(field.tensor, values_at_construction))
        self.assertTrue(torch.equal(source.tensor, source_values))

    def test_public_composition_preserves_pure_and_analog_autograd(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling)
        charge_values = source.tensor.to(dtype=torch.float64).requires_grad_()
        differentiable_charge = Charge(tensor=charge_values, axes=source.axes)
        with patch.object(
            simulation,
            "_produce_charge",
            return_value=differentiable_charge,
        ) as charge_call:
            result = simulate_readout(
                source,
                products=(PureWaveform, AnalogWaveform),
                config=_config(sampling),
                rng=_FailingRng(seed=0),
                floating_dtype=torch.float64,
            )
        charge_call.assert_called_once()
        pure = result.tensor(PureWaveform)
        analog = result.tensor(AnalogWaveform)
        self.assertTrue(pure.requires_grad)
        self.assertTrue(analog.requires_grad)
        gradient = torch.autograd.grad(analog.sum(), charge_values)[0]
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.any(gradient != 0.0).item())

    def test_global_rng_source_and_host_staging_surfaces_remain_unchanged(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling, noncontiguous=True)
        source_values = source.tensor.clone()
        global_rng = torch.random.get_rng_state().clone()
        with (
            patch.object(
                torch,
                "Generator",
                side_effect=AssertionError("Generator"),
            ) as generator,
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("cpu")) as cpu,
            patch.object(
                torch.Tensor,
                "numpy",
                side_effect=AssertionError("numpy"),
            ) as numpy,
        ):
            result = simulate_readout(
                source,
                products=PRODUCT_TYPES,
                config=_config(sampling),
                rng=_FailingRng(seed=0),
            )
        generator.assert_not_called()
        cpu.assert_not_called()
        numpy.assert_not_called()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_rng))
        self.assertTrue(torch.equal(source.tensor, source_values))
        self.assertIs(result.field(Photoelectrons), source)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class ReadoutSimulationCudaTest(unittest.TestCase):
    def test_deterministic_full_closure_device_axes_and_storage(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling, noncontiguous=True, device="cuda")
        source_values = source.tensor.clone()
        result = simulate_readout(
            source,
            products=PRODUCT_TYPES,
            config=_config(sampling),
            rng=_FailingRng(seed=0),
        )
        self.assertIs(result.field(Photoelectrons), source)
        self.assertTrue(torch.equal(source.tensor, source_values))
        for field_type in GENERATED_TYPES:
            field = result.field(field_type)
            self.assertEqual(field.tensor.device.type, "cuda")
            self.assertIs(field.axes, source.axes)
            _assert_no_storage_overlap(self, field.tensor, source.tensor)

    def test_stochastic_noise_uses_same_cuda_device(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling, device="cuda")
        config = _config(
            sampling,
            noise=NoiseWaveformConfig(
                model=WhiteNoiseConfig(rms_mv=PositiveFloat(0.5))
            ),
        )
        result = simulate_readout(
            source,
            products=(NoiseWaveform,),
            config=config,
            rng=Threefry4x32(seed=17),
        )
        noise = result.field(NoiseWaveform)
        self.assertEqual(noise.tensor.device.type, "cuda")
        self.assertIs(noise.axes, source.axes)
        _assert_no_storage_overlap(self, noise.tensor, source.tensor)

    def test_key_collision_fails_before_cuda_execution(self) -> None:
        sampling = _sampling()
        source = _photoelectrons(sampling, device="cuda")
        shared = RngKey(namespace=0xABCDEF03, stream=1)
        config = _config(
            sampling,
            charge=ChargeConfig(
                dark_count=DarkCountConfig(
                    rate_hz=NonnegativeFloat(0.0),
                    rng_key=shared,
                )
            ),
            noise=NoiseWaveformConfig(
                model=WhiteNoiseConfig(
                    rms_mv=PositiveFloat(0.5),
                    rng_key=shared,
                )
            ),
        )
        rng = _FailingRng(seed=0)
        with ExitStack() as stack:
            producer_mocks = tuple(
                stack.enter_context(patch.object(simulation, name))
                for name, _ in PRODUCERS
            )
            with self.assertRaises(ValueError):
                simulate_readout(
                    source,
                    products=(Charge, NoiseWaveform),
                    config=config,
                    rng=rng,
                )
        self.assertEqual(rng.calls, 0)
        for producer_mock in producer_mocks:
            producer_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
