"""Focused raw-ZLE Product, lifecycle, parity, and boundary evidence."""

import itertools
import unittest
from unittest.mock import patch
from typing import Any, override

import torch
from torch.overrides import TorchFunctionMode
from tensor_core import (
    CountCoordinates,
    LabelCoordinates,
    OffsetAxis,
    OffsetCoordinates,
    RegularCoordinates,
    TensorField,
    TensorKernel,
)

from tensor_dslab import (
    BitDepth,
    BitDepthSpec,
    ChannelAxis,
    DigitizedWaveform,
    DigitizedWaveformConfig,
    DigitizedWaveformKernels,
    DigitizedWaveformSpec,
    EncodedWaveform,
    EncodedWaveformConfig,
    EncodedWaveformKernels,
    EncodedWaveformSpec,
    ExampleAxis,
    PostTriggerSamples,
    PostTriggerSamplesSpec,
    PreTriggerSamples,
    PreTriggerSamplesSpec,
    ReleaseThresholdCode,
    ReleaseThresholdCodeSpec,
    RequiredTimeOverSamples,
    RequiredTimeOverSamplesSpec,
    TimeAxis,
    TriggerThresholdCode,
    TriggerThresholdCodeSpec,
    unit_registry,
)
from tests._product_support import analog_config, digitized_config


CPU = torch.device("cpu")
U = unit_registry
type PolicyValue = int | tuple[int, ...] | tuple[tuple[int, ...], ...]
DONOR_IDENTITIES = (
    "src/dselec/zle.py",
    "_find_zle_intervals",
    "c06b5e9cdf35ec41e487518e3b1b0baa0c957899645bbd9ac2479c902bb1b304",
    "tests/test_zle.py",
    "ab85ec0f4deff32c1a3bdba81a6a7617c12f9aaf73d17b5e0dfa6eb6424ed187",
    "data/config_files/dselec.ini",
    "fd42244bb4405dc328496efb8043fff522584a1922b811246670ac0e940e1c64",
)


class NoHostMaterializationMode(TorchFunctionMode):
    """Record forbidden Tensor host/list boundaries during production."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __torch_function__(
        self,
        func,
        types,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if func in (torch.Tensor.detach, torch.Tensor.cpu, torch.Tensor.tolist):
            self.calls.append(func.__name__)
        return func(*args, **({} if kwargs is None else kwargs))


class _ImpostorDigitized(TensorField[DigitizedWaveformSpec[Any]]):
    """Carry valid digitized state without the required semantic Product role."""

    __slots__ = ()
    @override
    def _require(self) -> None:
        pass


def _axes(
    *,
    examples: int = 1,
    channels: tuple[str, ...] = ("a",),
    time_count: int,
    time_start: int = 0,
    time_step: int = 1,
) -> tuple:
    """Construct compact application-neutral encoded-waveform axes."""
    return (
        ExampleAxis(coordinates=CountCoordinates(count=examples)),
        ChannelAxis(coordinates=LabelCoordinates(labels=channels)),
        TimeAxis(
            coordinates=RegularCoordinates(
                start=time_start,
                step=time_step,
                count=time_count,
            ),
            coordinate_scale=2.0,
            unit=U.Unit("ns"),
        ),
    )


def _source(
    values: torch.Tensor,
    *,
    axes: tuple | None = None,
    dtype: torch.dtype = torch.int32,
) -> DigitizedWaveform:
    """Build one exact digitized source fixture."""
    selected_axes = axes if axes is not None else _axes(
        examples=values.shape[0],
        channels=tuple(f"c{index}" for index in range(values.shape[1])),
        time_count=values.shape[2],
    )
    spec = DigitizedWaveformSpec(
        axes=selected_axes, device=CPU, dtype=dtype, unit=U.Unit("")
    )
    return DigitizedWaveform(tensor=values.to(dtype), spec=spec)


def _policy(
    spec_type: type,
    kernel_type: type[TensorKernel[Any]],
    value: PolicyValue,
    *,
    conditioning_axes: tuple = (),
) -> TensorKernel[Any]:
    """Construct one literal int64 raw-ZLE policy Kernel."""

    spec = spec_type(
        conditioning_axes=conditioning_axes, operation_axes=(),
        device=CPU, dtype=torch.int64,
    )
    return kernel_type(tensor=torch.tensor(value, dtype=torch.int64), spec=spec)


def _config(
    source: DigitizedWaveform,
    *,
    trigger: PolicyValue = 950,
    release: PolicyValue = 970,
    required: PolicyValue = 3,
    pre: PolicyValue = 2,
    post: PolicyValue = 3,
    suppression_code: int = -1,
    conditioning_axes: tuple = (),
) -> EncodedWaveformConfig:
    """Construct one exact encoded-waveform punchcard."""

    spec = EncodedWaveformSpec(
        axes=source.spec.axes, device=source.spec.device,
        dtype=source.spec.dtype, unit=source.spec.unit,
        suppression_code=suppression_code,
    )
    return EncodedWaveformConfig(
        spec=spec,
        kernels=EncodedWaveformKernels(
            members=(
                _policy(TriggerThresholdCodeSpec, TriggerThresholdCode, trigger,
                        conditioning_axes=conditioning_axes),
                _policy(ReleaseThresholdCodeSpec, ReleaseThresholdCode, release,
                        conditioning_axes=conditioning_axes),
                _policy(
                    RequiredTimeOverSamplesSpec, RequiredTimeOverSamples, required,
                    conditioning_axes=conditioning_axes,
                ),
                _policy(PreTriggerSamplesSpec, PreTriggerSamples, pre,
                        conditioning_axes=conditioning_axes),
                _policy(PostTriggerSamplesSpec, PostTriggerSamples, post,
                        conditioning_axes=conditioning_axes),
            )
        ),
    )


def _raw_support(
    values: tuple[int, ...],
    *,
    trigger: int,
    release: int,
    required: int,
    pre: int,
    post: int,
) -> tuple[bool, ...]:
    """Reproduce the cited IV-DSLab raw recurrence as a test-only oracle."""

    support = [False] * len(values)
    component_start = 0
    while component_start < len(values):
        if values[component_start] > release:
            component_start += 1
            continue
        component_end = component_start
        while (
            component_end < len(values)
            and values[component_end] <= release
        ):
            component_end += 1
        qualifying_start: int | None = None
        run = 0
        for index in range(component_start, component_end):
            run = run + 1 if values[index] <= trigger else 0
            if run >= required:
                qualifying_start = index - required + 1
                break
        if qualifying_start is not None:
            start = max(0, qualifying_start - pre)
            end = min(len(values), component_end + post)
            for index in range(start, end):
                support[index] = True
        component_start = component_end
    return tuple(support)


def _encoded_values(
    values: tuple[int, ...],
    *,
    trigger: int = 950,
    release: int = 970,
    required: int = 3,
    pre: int = 2,
    post: int = 3,
    suppression_code: int = -1,
) -> tuple[int, ...]:
    """Execute the public Product for one independent lane."""

    source = _source(torch.tensor(values).reshape(1, 1, -1))
    result = EncodedWaveform.create(
        sources=(source,),
        config=_config(
            source,
            trigger=trigger,
            release=release,
            required=required,
            pre=pre,
            post=post,
            suppression_code=suppression_code,
        ),
    )
    return tuple(int(value) for value in result.tensor.reshape(-1))


class EncodedWaveformTests(unittest.TestCase):
    def test_exact_policy_leaves_geometry_domains_collection_and_exports(
        self,
    ) -> None:
        cases = (
            (TriggerThresholdCodeSpec, TriggerThresholdCode, 0, -1),
            (ReleaseThresholdCodeSpec, ReleaseThresholdCode, 0, -1),
            (RequiredTimeOverSamplesSpec, RequiredTimeOverSamples, 1, 0),
            (PreTriggerSamplesSpec, PreTriggerSamples, 0, -1),
            (PostTriggerSamplesSpec, PostTriggerSamples, 0, -1),
        )
        members: list[TensorKernel[Any]] = []
        for spec_type, kernel_type, accepted, rejected in cases:
            with self.subTest(kernel=kernel_type.__name__):
                kernel = _policy(spec_type, kernel_type, accepted)
                self.assertIs(type(kernel.spec), spec_type)
                self.assertIs(kernel.dtype, torch.int64)
                self.assertEqual(kernel.spec.operation_axes, ())
                members.append(kernel)
                with self.assertRaises(ValueError):
                    _policy(spec_type, kernel_type, rejected)
                with self.assertRaises(TypeError):
                    spec_type(
                        conditioning_axes=(),
                        operation_axes=(),
                        device=CPU,
                        dtype=torch.int32,
                    )
        kernels = EncodedWaveformKernels(members=tuple(members))
        self.assertIs(kernels.trigger_threshold_code, members[0])
        self.assertIs(kernels.release_threshold_code, members[1])
        self.assertIs(kernels.required_time_over_samples, members[2])
        self.assertIs(kernels.pre_trigger_samples, members[3])
        self.assertIs(kernels.post_trigger_samples, members[4])
        with self.assertRaises(ValueError):
            EncodedWaveformKernels(members=tuple(members[:-1]))

        time = _axes(time_count=3)[2]
        with self.assertRaises(TypeError):
            TriggerThresholdCodeSpec(
                conditioning_axes=(time,),
                operation_axes=(),
                device=CPU,
                dtype=torch.int64,
            )
        operation = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(0,)),
            relative_to=TimeAxis,
        )
        with self.assertRaises(ValueError):
            TriggerThresholdCodeSpec(
                conditioning_axes=(),
                operation_axes=(operation,),
                device=CPU,
                dtype=torch.int64,
            )

        import tensor_dslab
        import tensor_dslab.encoded_waveform as encoded

        self.assertEqual(len(tensor_dslab.__all__), 75)
        self.assertEqual(
            encoded.__all__,
            (
                "EncodedWaveform",
                "EncodedWaveformConfig",
                "EncodedWaveformKernels",
                "EncodedWaveformSpec",
                "PostTriggerSamples",
                "PostTriggerSamplesSpec",
                "PreTriggerSamples",
                "PreTriggerSamplesSpec",
                "ReleaseThresholdCode",
                "ReleaseThresholdCodeSpec",
                "RequiredTimeOverSamples",
                "RequiredTimeOverSamplesSpec",
                "TriggerThresholdCode",
                "TriggerThresholdCodeSpec",
            ),
        )

    def test_signed_specs_suppression_and_intrinsic_field_values(self) -> None:
        axes = _axes(time_count=3)
        for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                spec = EncodedWaveformSpec(
                    axes=axes,
                    device=CPU,
                    dtype=dtype,
                    unit=U.Unit(""),
                    suppression_code=-1,
                )
                field = EncodedWaveform(
                    tensor=torch.tensor(
                        [[[-1, 0, 1]]],
                        dtype=dtype,
                    ),
                    spec=spec,
                )
                self.assertIs(field.dtype, dtype)
                moved = spec.to(dtype=dtype)
                self.assertIs(type(moved), EncodedWaveformSpec)
                self.assertEqual(moved.suppression_code, -1)
                source = _source(
                    torch.tensor((5, 0, 5)).reshape(1, 1, 3),
                    axes=axes,
                    dtype=dtype,
                )
                created = EncodedWaveform.create(
                    sources=(source,),
                    config=_config(
                        source,
                        trigger=0,
                        release=0,
                        required=1,
                        pre=0,
                        post=0,
                    ),
                )
                self.assertIs(created.dtype, dtype)
                self.assertEqual(
                    tuple(int(value) for value in created.tensor.flatten()),
                    (-1, 0, -1),
                )
        with self.assertRaises(TypeError):
            EncodedWaveformSpec(
                axes=axes,
                device=CPU,
                dtype=torch.uint8,
                unit=U.Unit(""),
                suppression_code=-1,
            )
        for sentinel, error in (
            (True, TypeError),
            (0, ValueError),
            (-129, ValueError),
        ):
            with self.subTest(sentinel=sentinel):
                with self.assertRaises(error):
                    EncodedWaveformSpec(
                        axes=axes,
                        device=CPU,
                        dtype=torch.int8,
                        unit=U.Unit(""),
                        suppression_code=sentinel,
                    )
        spec = EncodedWaveformSpec(
            axes=axes,
            device=CPU,
            dtype=torch.int16,
            unit=U.Unit(""),
            suppression_code=-7,
        )
        with self.assertRaises(ValueError):
            EncodedWaveform(
                tensor=torch.tensor([[[-7, -2, 0]]], dtype=torch.int16),
                spec=spec,
            )

    def test_digitized_signed_dtype_and_bit_depth_capacity_preflight(
        self,
    ) -> None:
        base = digitized_config()
        for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                spec = base.spec.to(dtype=dtype)
                self.assertIs(type(spec), DigitizedWaveformSpec)
                self.assertIs(spec.dtype, dtype)
        source_spec = base.spec.to(dtype=torch.int8)
        input_spec = analog_config().spec
        high = DigitizedWaveformConfig(
            spec=source_spec,
            kernels=base.kernels,
        )
        with self.assertRaisesRegex(ValueError, "not representable"):
            DigitizedWaveform.prepare(
                source_specs=(input_spec,),
                config=high,
            )
        bit = BitDepth(
            tensor=torch.tensor(7, dtype=torch.int16),
            spec=BitDepthSpec(
                conditioning_axes=(),
                operation_axes=(),
                device=CPU,
                dtype=torch.int16,
            ),
        )
        low = DigitizedWaveformConfig(
            spec=source_spec,
            kernels=DigitizedWaveformKernels(
                members=(
                    bit,
                    base.kernels.input_minimum,
                    base.kernels.input_maximum,
                    base.kernels.analog_gain,
                )
            ),
        )
        prepared = DigitizedWaveform.prepare(
            source_specs=(input_spec,),
            config=low,
        )
        self.assertIs(prepared.spec.dtype, torch.int8)

    def test_preparation_relationships_and_staged_source_binding(self) -> None:
        source = _source(torch.full((1, 1, 5), 1000, dtype=torch.int32))
        config = _config(source)
        self.assertFalse(config._is_prepared)
        self.assertEqual(config._source_specs, ())
        self.assertIsNone(config._working_dtype)
        self.assertIsNone(config._time_dimension)
        self.assertEqual(config._kernel_dimensions, (None,) * 5)
        with self.assertRaises(TypeError):
            hash(config)
        prepared = EncodedWaveform.prepare(source_specs=(source.spec,), config=config)
        self.assertIsNot(prepared, config)
        self.assertTrue(prepared._is_prepared)
        self.assertIs(prepared._source_specs[0], source.spec)
        self.assertIs(prepared._working_dtype, torch.int32)
        self.assertEqual(prepared._time_dimension, 2)
        self.assertEqual(prepared._kernel_dimensions, ((),) * 5)

        with self.assertRaises(ValueError):
            EncodedWaveform.prepare(source_specs=(), config=config)
        with self.assertRaises(ValueError):
            EncodedWaveform.prepare(source_specs=(source.spec, source.spec), config=config)
        with self.assertRaisesRegex(TypeError, "DigitizedWaveformSpec"):
            EncodedWaveform.prepare(source_specs=(analog_config().spec,), config=config)
        with self.assertRaisesRegex(ValueError, "ReleaseThresholdCode"):
            EncodedWaveform.prepare(
                source_specs=(source.spec,),
                config=_config(source, trigger=10, release=9),
            )
        bad_step_source = _source(source.tensor, axes=_axes(time_count=5, time_step=2))
        with self.assertRaisesRegex(ValueError, "step 1"):
            EncodedWaveform.prepare(source_specs=(bad_step_source.spec,), config=_config(bad_step_source))
        count_time_axes = (
            source.spec.axes[0],
            source.spec.axes[1],
            TimeAxis(
                coordinates=CountCoordinates(count=5),
                coordinate_scale=2.0,
                unit=U.Unit("ns"),
            ),
        )
        count_time_source = _source(source.tensor, axes=count_time_axes)
        with self.assertRaisesRegex(TypeError, "RegularCoordinates"):
            EncodedWaveform.prepare(source_specs=(count_time_source.spec,), config=_config(count_time_source))
        missing_time_axes = source.spec.axes[:2]
        missing_time_source = DigitizedWaveform(
            tensor=torch.zeros((1, 1), dtype=torch.int32),
            spec=DigitizedWaveformSpec(
                axes=missing_time_axes,
                device=CPU,
                dtype=torch.int32,
                unit=U.Unit(""),
            ),
        )
        with self.assertRaisesRegex(ValueError, "exactly one TimeAxis"):
            EncodedWaveform.prepare(source_specs=(missing_time_source.spec,), config=_config(missing_time_source))

        product = EncodedWaveform.create(sources=(source,), config=config)
        module = "tensor_dslab.encoded_waveform"
        malformed_calls = (
            ("prepare", f"{module}.runtime.prepare.prepare_encoded_waveform",
             "source_specs", source.spec, {"config": config}),
            ("create", f"{module}.field.EncodedWaveform.prepare",
             "sources", source, {"config": config}),
            ("produce", f"{module}.runtime.produce.produce_encoded_waveform",
             "sources", source, {"config": prepared}),
            ("validate", f"{module}.runtime.validate.validate_encoded_waveform",
             "sources", source, {"product": product, "config": prepared}),
        )
        for method, target, key, admitted, context in malformed_calls:
            for malformed in ([admitted], (object(),)):
                kwargs = {key: malformed, **context}
                with self.subTest(method=method, malformed=malformed):
                    with patch(target) as downstream:
                        with self.assertRaises(TypeError):
                            getattr(EncodedWaveform, method)(**kwargs)
                        downstream.assert_not_called()

        mismatched_specs = (
            EncodedWaveformSpec(
                axes=(
                    source.spec.axes[1],
                    source.spec.axes[0],
                    source.spec.axes[2],
                ),
                device=CPU,
                dtype=torch.int32,
                unit=U.Unit(""),
                suppression_code=-1,
            ),
            EncodedWaveformSpec(
                axes=_axes(time_count=6),
                device=CPU,
                dtype=torch.int32,
                unit=U.Unit(""),
                suppression_code=-1,
            ),
            EncodedWaveformSpec(
                axes=source.spec.axes,
                device=CPU,
                dtype=torch.int16,
                unit=U.Unit(""),
                suppression_code=-1,
            ),
            EncodedWaveformSpec(
                axes=source.spec.axes,
                device=CPU,
                dtype=torch.int32,
                unit=U.Unit("percent"),
                suppression_code=-1,
            ),
        )
        for spec in mismatched_specs:
            with self.subTest(mismatch=spec):
                mismatched = EncodedWaveformConfig(
                    spec=spec,
                    kernels=config.kernels,
                )
                with self.assertRaises((TypeError, ValueError)):
                    EncodedWaveform.prepare(
                        source_specs=(source.spec,),
                        config=mismatched,
                    )

        changed = _source(
            source.tensor,
            axes=_axes(time_count=5, time_start=1),
        )
        with self.assertRaisesRegex(ValueError, "prepared provenance"):
            EncodedWaveform.produce(
                sources=(changed,),
                config=prepared,
            )
        with self.assertRaisesRegex(TypeError, "DigitizedWaveform"):
            EncodedWaveform.produce(
                sources=(
                    _ImpostorDigitized(
                        tensor=source.tensor,
                        spec=source.spec,
                    ),
                ),
                config=prepared,
            )
        equal_distinct = _source(source.tensor.clone(), axes=source.spec.axes)
        result = EncodedWaveform.produce(
            sources=(equal_distinct,),
            config=prepared,
        )
        self.assertEqual(result.shape, source.tensor.shape)

    def test_worked_acceptance_example_and_complete_lifecycle(self) -> None:
        values = (
            1000, 1000, 1000, 1000, 948, 945, 940, 960, 975, 1000,
            1000, 949, 947, 944, 965, 980, 1000, 1000, 949, 948,
        )
        expected = (
            -1, -1, 1000, 1000, 948, 945, 940, 960, 975, 1000,
            1000, 949, 947, 944, 965, 980, 1000, 1000, -1, -1,
        )
        source = _source(torch.tensor(values).reshape(1, 1, -1))
        source_before = source.tensor.clone()
        config = _config(source)
        mode = NoHostMaterializationMode()
        with mode:
            product = EncodedWaveform.create(sources=(source,), config=config)
        self.assertEqual(mode.calls, [])
        self.assertEqual(tuple(int(v) for v in product.tensor.flatten()), expected)
        self.assertTrue(torch.equal(source.tensor, source_before))
        self.assertTrue(product.tensor.is_contiguous())
        self.assertNotEqual(
            product.tensor.untyped_storage(),
            source.tensor.untyped_storage(),
        )
        replay = EncodedWaveform.create(sources=(source,), config=config)
        self.assertTrue(torch.equal(product.tensor, replay.tensor))
        self.assertNotEqual(
            product.tensor.untyped_storage(),
            replay.tensor.untyped_storage(),
        )

    def test_trigger_release_retrigger_padding_and_boundary_laws(self) -> None:
        cases = (
            ((10, 1, 1, 10), 3, 0, 0, (-1, -1, -1, -1)),
            ((10, 1, 1, 1), 3, 0, 0, (-1, 1, 1, 1)),
            ((1, 1, 1, 5, 7, 10), 3, 0, 0, (1, 1, 1, 5, 7, -1)),
            ((10, 1, 1, 1, 10), 3, 2, 2, (10, 1, 1, 1, 10)),
            ((1, 1, 1), 3, 0, 0, (1, 1, 1)),
            ((1, 1, 10), 99, 99, 99, (-1, -1, -1)),
        )
        for values, required, pre, post, expected in cases:
            with self.subTest(
                values=values,
                required=required,
                pre=pre,
                post=post,
            ):
                actual = _encoded_values(
                    values,
                    trigger=1,
                    release=7,
                    required=required,
                    pre=pre,
                    post=post,
                )
                self.assertEqual(actual, expected)
        touching = _encoded_values(
            (10, 1, 1, 1, 10, 10, 1, 1, 1, 10),
            trigger=1,
            release=1,
            required=3,
            pre=0,
            post=3,
        )
        self.assertEqual(touching, (-1, 1, 1, 1, 10, 10, 1, 1, 1, 10))

    def test_lane_example_conditioning_permutation_and_independence(self) -> None:
        axes = _axes(
            examples=2,
            channels=("a", "b"),
            time_count=6,
        )
        values = torch.tensor(
            (
                ((10, 1, 1, 1, 10, 10), (10, 2, 2, 2, 10, 10)),
                ((10, 3, 3, 3, 10, 10), (10, 4, 4, 4, 10, 10)),
            ),
            dtype=torch.int32,
        )
        source = _source(values, axes=axes)
        conditioning = (
            ChannelAxis(
                coordinates=LabelCoordinates(labels=("b", "a"))
            ),
            ExampleAxis(
                coordinates=OffsetCoordinates(offsets=(1, 0))
            ),
        )
        config = _config(
            source,
            trigger=((4, 2), (3, 1)),
            release=((4, 2), (3, 1)),
            required=((3, 3), (3, 3)),
            pre=((0, 0), (0, 0)),
            post=((0, 0), (0, 0)),
            conditioning_axes=conditioning,
        )
        result = EncodedWaveform.create(sources=(source,), config=config)
        expected = torch.where(
            values == 10,
            torch.full_like(values, -1),
            values,
        )
        self.assertTrue(torch.equal(result.tensor, expected))
        changed_values = values.clone()
        changed_values[1, 1] = 99
        changed = EncodedWaveform.create(
            sources=(_source(changed_values, axes=axes),),
            config=config,
        )
        self.assertTrue(torch.equal(changed.tensor[0], result.tensor[0]))
        self.assertTrue(torch.equal(changed.tensor[1, 0], result.tensor[1, 0]))

        time_first_axes = (axes[2], axes[0], axes[1])
        time_first_values = values.permute(2, 0, 1).contiguous()
        time_first = _source(
            time_first_values,
            axes=time_first_axes,
        )
        time_first_result = EncodedWaveform.create(
            sources=(time_first,),
            config=_config(
                time_first,
                trigger=4,
                release=4,
                required=3,
                pre=0,
                post=0,
            ),
        )
        self.assertEqual(time_first_result.spec.axes, time_first_axes)
        self.assertTrue(
            torch.equal(
                time_first_result.tensor,
                torch.where(
                    time_first_values == 10,
                    torch.full_like(time_first_values, -1),
                    time_first_values,
                ),
            )
        )

    def test_exhaustive_short_state_parity_and_donor_identities(self) -> None:
        self.assertEqual(
            DONOR_IDENTITIES,
            (
                "src/dselec/zle.py",
                "_find_zle_intervals",
                "c06b5e9cdf35ec41e487518e3b1b0baa0c957899645bbd9ac2479c902bb1b304",
                "tests/test_zle.py",
                "ab85ec0f4deff32c1a3bdba81a6a7617c12f9aaf73d17b5e0dfa6eb6424ed187",
                "data/config_files/dselec.ini",
                "fd42244bb4405dc328496efb8043fff522584a1922b811246670ac0e940e1c64",
            ),
        )
        relation_values = (0, 2, 4)
        for length in range(10):
            sequences = tuple(
                itertools.product(relation_values, repeat=length)
            )
            values_tensor = torch.tensor(
                sequences,
                dtype=torch.int32,
            ).reshape(len(sequences), 1, length)
            source = _source(values_tensor)
            for required, pre, post in (
                (1, 0, 0),
                (2, 1, 1),
                (3, 2, 1),
            ):
                with self.subTest(
                    length=length,
                    required=required,
                    pre=pre,
                    post=post,
                ):
                    expected_rows = []
                    for values in sequences:
                        support = _raw_support(
                            values,
                            trigger=1,
                            release=3,
                            required=required,
                            pre=pre,
                            post=post,
                        )
                        expected_rows.append(
                            tuple(
                                value if retained else -7
                                for value, retained in zip(values, support)
                            )
                        )
                    expected = torch.tensor(
                        expected_rows,
                        dtype=torch.int32,
                    ).reshape(len(sequences), 1, length)
                    actual = EncodedWaveform.create(
                        sources=(source,),
                        config=_config(
                            source,
                            trigger=1,
                            release=3,
                            required=required,
                            pre=pre,
                            post=post,
                            suppression_code=-7,
                        ),
                    )
                    self.assertTrue(torch.equal(actual.tensor, expected))

    def test_zero_time_large_counts_and_configurable_retained_extremes(
        self,
    ) -> None:
        empty = _source(
            torch.empty((2, 2, 0), dtype=torch.int16),
            axes=_axes(
                examples=2,
                channels=("a", "b"),
                time_count=0,
            ),
            dtype=torch.int16,
        )
        encoded = EncodedWaveform.create(
            sources=(empty,),
            config=_config(
                empty,
                required=torch.iinfo(torch.int64).max,
                pre=torch.iinfo(torch.int64).max,
                post=torch.iinfo(torch.int64).max,
                suppression_code=-9,
            ),
        )
        self.assertEqual(encoded.tensor.shape, (2, 2, 0))
        self.assertIs(encoded.tensor.dtype, torch.int16)
        self.assertNotEqual(
            encoded.tensor.untyped_storage(),
            empty.tensor.untyped_storage(),
        )
        maximum_count = torch.iinfo(torch.int64).max
        self.assertEqual(
            _encoded_values(
                (10, 1, 10),
                trigger=1,
                release=1,
                required=maximum_count,
                pre=maximum_count,
                post=maximum_count,
                suppression_code=-9,
            ),
            (-9, -9, -9),
        )
        self.assertEqual(
            _encoded_values(
                (10, 1, 10),
                trigger=1,
                release=1,
                required=1,
                pre=maximum_count,
                post=maximum_count,
                suppression_code=-9,
            ),
            (10, 1, 10),
        )
        extreme = _encoded_values(
            (0, 0, 32767),
            trigger=32767,
            release=32767,
            required=1,
            pre=0,
            post=0,
            suppression_code=-9,
        )
        self.assertEqual(extreme, (0, 0, 32767))

    def test_validation_rejects_value_storage_and_relationship_mutants(
        self,
    ) -> None:
        source = _source(
            torch.tensor((1000, 1, 1, 1, 1000)).reshape(1, 1, -1)
        )
        config = EncodedWaveform.prepare(
            source_specs=(source.spec,),
            config=_config(
                source,
                trigger=1,
                release=1,
                required=3,
                pre=0,
                post=0,
            ),
        )
        field = EncodedWaveform.produce(
            sources=(source,),
            config=config,
        )
        EncodedWaveform.validate(
            product=field,
            sources=(source,),
            config=config,
        )
        wrong = field.tensor.clone()
        wrong[..., 0] = 0
        with self.assertRaisesRegex(ValueError, "raw-ZLE support"):
            EncodedWaveform.validate(
                product=EncodedWaveform(tensor=wrong, spec=config.spec),
                sources=(source,),
                config=config,
            )
        alias_source = _source(
            torch.ones((1, 1, 5), dtype=torch.int32)
        )
        alias_config = EncodedWaveform.prepare(
            source_specs=(alias_source.spec,),
            config=_config(
                alias_source,
                trigger=1,
                release=1,
                required=1,
                pre=0,
                post=0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "storage"):
            EncodedWaveform.validate(
                product=EncodedWaveform(
                    tensor=alias_source.tensor,
                    spec=alias_config.spec,
                ),
                sources=(alias_source,),
                config=alias_config,
            )
