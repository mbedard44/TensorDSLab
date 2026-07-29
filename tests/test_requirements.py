import ast
from dataclasses import dataclass
from pathlib import Path
import unittest
from typing import Any, override

import pint
import torch
from tensor_core import (
    Coordinates,
    CountCoordinates,
    LabelCoordinates,
    OffsetAxis,
    OffsetCoordinates,
    RegularCoordinates,
    TensorAxis,
    TensorCollection,
    TensorField,
    TensorKernel,
    TensorKernelSpec,
)

from tensor_dslab import (
    ChannelAxis as ProductChannelAxis,
    ChargeSpec,
    Photoelectrons,
    PhotoelectronsSpec,
    PulseResponse,
    unit_registry,
)
from tensor_dslab.common.requirements.axis import (
    require_coordinate_scale,
    require_regular_coordinates,
    require_supported_coordinates,
    require_supported_integer_coordinates,
)
from tensor_dslab.common.requirements.capacity import (
    require_address_capacity,
    require_tensor_capacity,
)
from tensor_dslab.common.requirements.collection import (
    require_admitted_member_types,
    require_exact_member_types,
    require_member_count,
)
from tensor_dslab.common.requirements.config import (
    require_config_components,
    require_prepared_config,
    require_prepared_sources,
)
from tensor_dslab.common.requirements.field import (
    require_exact_field_spec,
    require_fresh_product,
)
from tensor_dslab.common.requirements.kernel import (
    require_exact_kernel_spec,
    require_no_operation_axes,
    require_nonempty_operation_extents,
    require_offset_bounds,
    require_operation_axes_type,
    require_operation_axis_count,
    require_operation_row_total,
    require_operation_target_count,
)
from tensor_dslab.common.requirements.tensor import (
    require_dtype_in,
    require_exact_dtype,
    require_finite,
    require_floating_dtype,
    require_nonnegative,
    require_positive,
    require_signed_integer_dtype,
    require_values_between,
)
from tensor_dslab.common.requirements.unit import require_unit_compatible
from tests._product_support import (
    charge_config,
    noise_config,
    pure_config,
    source,
)


@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class _AlienCoordinates(Coordinates[int]):
    count: int

    @property
    @override
    def size(self) -> int:
        return self.count

    @override
    def coordinate_at(self, index: int) -> int:
        return index

    @override
    def index_of(self, coordinate: int) -> int:
        return coordinate

    @override
    def _window(self, *, start_index: int, count: int):
        return type(self)(count=count)


@dataclass(slots=True)
class _TensorValue:
    tensor: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype


class _LiteralSpec[C: tuple, O: tuple](TensorKernelSpec[C, O]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        pass


class _LiteralKernel(TensorKernel[_LiteralSpec[Any, Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        pass


class _LiteralField(TensorField[Any]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        pass


class _LiteralCollection(TensorCollection[TensorKernel[Any]]):
    __slots__ = ()

    @override
    def _require(self) -> None:
        pass


@dataclass(frozen=True, slots=True, eq=False, repr=False, kw_only=True)
class ChannelAxis(TensorAxis[int]):
    """Represent a distinct same-name semantic target role."""

    coordinates: Coordinates[int]

    @override
    def _require(self) -> None:
        pass


class RequirementTests(unittest.TestCase):
    def test_introduced_and_moved_definitions_have_docstrings(self) -> None:
        requirements_path = Path("tensor_dslab/common/requirements")
        requirement_modules = (
            "__init__.py",
            "axis.py",
            "capacity.py",
            "collection.py",
            "config.py",
            "field.py",
            "kernel.py",
            "tensor.py",
            "unit.py",
        )
        self.assertEqual(
            tuple(path.name for path in sorted(requirements_path.glob("*.py"))),
            requirement_modules,
        )
        for filename in requirement_modules:
            with self.subTest(module=filename):
                path = requirements_path / filename
                tree = ast.parse(path.read_text(encoding="utf-8"))
                self.assertTrue(ast.get_docstring(tree), path)
                for definition in tree.body:
                    if isinstance(
                        definition,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    ):
                        self.assertTrue(
                            ast.get_docstring(definition),
                            f"{path}:{definition.name}",
                        )

        moved_collections = (
            ("analog_waveform", "AnalogWaveformKernels"),
            ("charge", "ChargeKernels"),
            ("digitized_waveform", "DigitizedWaveformKernels"),
            ("noise_waveform", "NoiseWaveformKernels"),
            ("pure_waveform", "PureWaveformKernels"),
        )
        for package, class_name in moved_collections:
            with self.subTest(collection=class_name):
                path = Path(f"tensor_dslab/{package}/kernel.py")
                tree = ast.parse(path.read_text(encoding="utf-8"))
                definitions = tuple(
                    definition
                    for definition in tree.body
                    if isinstance(definition, ast.ClassDef)
                    and definition.name == class_name
                )
                self.assertEqual(len(definitions), 1, path)
                self.assertTrue(
                    ast.get_docstring(definitions[0]),
                    f"{path}:{class_name}",
                )

    def test_axis_representation_scale_and_regular_requirements(self) -> None:
        count = CountCoordinates(count=2)
        labels = LabelCoordinates(labels=("a", "b"))
        regular_coordinates = RegularCoordinates(start=0, step=1, count=2)
        offsets = OffsetCoordinates(offsets=(0, 1))
        require_supported_coordinates(count)
        require_supported_coordinates(labels)
        require_supported_coordinates(regular_coordinates)
        require_supported_coordinates(offsets)
        require_supported_integer_coordinates(count)
        require_supported_integer_coordinates(regular_coordinates)
        require_supported_integer_coordinates(offsets)
        with self.assertRaises(TypeError):
            require_supported_coordinates(_AlienCoordinates(count=2))
        with self.assertRaises(TypeError):
            require_supported_integer_coordinates(
                LabelCoordinates(labels=("a", "b"))  # type: ignore[arg-type]
            )
        for scale in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    require_coordinate_scale(scale)
        with self.assertRaises(TypeError):
            require_coordinate_scale(1)
        require_coordinate_scale(1.0)
        regular = RegularCoordinates(start=3, step=2, count=2)
        require_regular_coordinates(regular, start=3, step=2)
        with self.assertRaises(ValueError):
            require_regular_coordinates(regular, start=2, step=2)
        with self.assertRaises(ValueError):
            require_regular_coordinates(regular, start=3, step=1)
        with self.assertRaises(TypeError):
            require_regular_coordinates(
                CountCoordinates(count=2),
                start=0,
                step=1,
            )

    def test_unit_requirement_is_pure_and_fail_closed(self) -> None:
        unit = unit_registry.Unit("ns")
        unit_type = type(unit)
        require_unit_compatible(unit, target="s", field="unit")
        self.assertIs(type(unit), unit_type)
        with self.assertRaises(ValueError):
            require_unit_compatible(unit, target="Hz", field="unit")
        foreign = pint.UnitRegistry(cache_folder=None)
        with self.assertRaises(ValueError):
            require_unit_compatible(
                foreign.Unit("ns"),
                target="s",
                field="unit",
            )
        derived_type = type(
            "DerivedUnit",
            (unit_type,),
            {"__slots__": ()},
        )
        derived = derived_type("ns")
        self.assertIs(derived._REGISTRY, unit_registry)
        with self.assertRaises(TypeError):
            require_unit_compatible(
                derived,
                target="s",
                field="unit",
            )
        with self.assertRaises(TypeError):
            require_unit_compatible(
                "ns",  # type: ignore[arg-type]
                target="s",
                field="unit",
            )

    def test_tensor_dtype_and_value_requirements_are_nonmutating(self) -> None:
        value = _TensorValue(
            tensor=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        )
        original = value.tensor.clone()
        require_exact_dtype(value, torch.float32)
        require_dtype_in(value, (torch.float32, torch.float64))
        require_floating_dtype(value)
        require_finite(value)
        require_nonnegative(value)
        require_values_between(value, minimum=0, maximum=2)
        torch.testing.assert_close(value.tensor, original, rtol=0, atol=0)
        require_positive(
            _TensorValue(tensor=torch.tensor([1.0], dtype=torch.float64))
        )
        require_signed_integer_dtype(
            _TensorValue(tensor=torch.tensor([1], dtype=torch.int16))
        )
        with self.assertRaises(TypeError):
            require_exact_dtype(value, torch.float64)
        with self.assertRaises(TypeError):
            require_floating_dtype(
                _TensorValue(tensor=torch.tensor([1], dtype=torch.int64))
            )
        with self.assertRaises(TypeError):
            require_signed_integer_dtype(value)
        for tensor, requirement in (
            (torch.tensor([float("inf")]), require_finite),
            (torch.tensor([-1.0]), require_nonnegative),
            (torch.tensor([0.0]), require_positive),
        ):
            with self.subTest(requirement=requirement.__name__):
                with self.assertRaises(ValueError):
                    requirement(_TensorValue(tensor=tensor))
        with self.assertRaises(ValueError):
            require_values_between(value, minimum=1, maximum=2)

    def test_field_spec_and_fresh_storage_requirements(self) -> None:
        live = source()
        require_exact_field_spec(live, type(live.spec))
        with self.assertRaises(TypeError):
            require_exact_field_spec(live, ChargeSpec)
        derived_spec_type = type(
            "DerivedPhotoelectronsSpec",
            (PhotoelectronsSpec,),
            {"__slots__": ()},
        )
        derived_spec = derived_spec_type(
            axes=live.spec.axes,
            device=live.spec.device,
            dtype=live.spec.dtype,
            unit=live.spec.unit,
        )
        derived_field = _LiteralField(
            tensor=live.tensor.clone(),
            spec=derived_spec,
        )
        with self.assertRaises(TypeError):
            require_exact_field_spec(
                derived_field,
                PhotoelectronsSpec,
            )
        fresh = Photoelectrons(
            tensor=live.tensor.clone(),
            spec=live.spec,
        )
        require_fresh_product(fresh, sources=(live,), kernels=())
        with self.assertRaises(ValueError):
            require_fresh_product(live, sources=(live,), kernels=())

    def test_kernel_geometry_and_non_temporal_row_totals(self) -> None:
        operation = OffsetAxis(
            coordinates=OffsetCoordinates(offsets=(0, 1)),
            relative_to=ProductChannelAxis,
        )
        spec = _LiteralSpec(
            conditioning_axes=(),
            operation_axes=(operation,),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        kernel = _LiteralKernel(
            tensor=torch.tensor([0.25, 0.75], dtype=torch.float64),
            spec=spec,
        )
        require_exact_kernel_spec(kernel, _LiteralSpec)
        require_operation_axis_count(spec, minimum=1, maximum=1)
        require_operation_axes_type(spec, OffsetAxis)
        require_nonempty_operation_extents(spec)
        require_operation_target_count(
            spec,
            relative_to=ProductChannelAxis,
            count=1,
        )
        require_offset_bounds(
            spec,
            relative_to=ProductChannelAxis,
            minimum=0,
            inclusive=True,
        )
        require_operation_row_total(
            kernel,
            exact=1.0,
            tolerance=1.0e-12,
        )
        require_operation_row_total(
            kernel,
            maximum=1.0,
            tolerance=1.0e-12,
        )
        with self.assertRaises(ValueError):
            require_operation_row_total(
                kernel,
                exact=0.5,
                tolerance=1.0e-12,
            )
        derived_spec_type = type(
            "DerivedLiteralSpec",
            (_LiteralSpec,),
            {"__slots__": ()},
        )
        derived_spec = derived_spec_type(
            conditioning_axes=(),
            operation_axes=(operation,),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        derived_kernel = _LiteralKernel(
            tensor=torch.tensor([0.25, 0.75], dtype=torch.float64),
            spec=derived_spec,
        )
        with self.assertRaises(TypeError):
            require_exact_kernel_spec(derived_kernel, _LiteralSpec)
        impostor_target = _LiteralSpec(
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(
                    coordinates=OffsetCoordinates(offsets=(0, 1)),
                    relative_to=ChannelAxis,
                ),
            ),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        self.assertEqual(
            ChannelAxis.__name__,
            ProductChannelAxis.__name__,
        )
        self.assertIsNot(ChannelAxis, ProductChannelAxis)
        with self.assertRaises(ValueError):
            require_operation_target_count(
                impostor_target,
                relative_to=ProductChannelAxis,
                count=1,
            )
        with self.assertRaises(ValueError):
            require_operation_row_total(
                kernel,
                maximum=0.5,
                tolerance=1.0e-12,
            )
        no_operation = _LiteralSpec(
            conditioning_axes=(),
            operation_axes=(),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        require_no_operation_axes(no_operation)
        wrong_axis = _LiteralSpec(
            conditioning_axes=(),
            operation_axes=(
                ChannelAxis(
                    coordinates=CountCoordinates(count=2),
                ),
            ),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with self.assertRaises(TypeError):
            require_operation_axes_type(wrong_axis, OffsetAxis)
        empty = _LiteralSpec(
            conditioning_axes=(),
            operation_axes=(
                OffsetAxis(
                    coordinates=OffsetCoordinates(offsets=()),
                    relative_to=ProductChannelAxis,
                ),
            ),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with self.assertRaises(ValueError):
            require_nonempty_operation_extents(empty)

    def test_collection_requirements_use_exact_member_types(self) -> None:
        pulse = pure_config().kernels.pulse_response
        collection = _LiteralCollection(members=(pulse,))
        require_admitted_member_types(
            collection,
            admitted=(PulseResponse,),
        )
        require_exact_member_types(
            collection,
            required=(PulseResponse,),
        )
        require_member_count(collection, minimum=1, maximum=1)
        derived_type = type(
            "DerivedPulseResponse",
            (PulseResponse,),
            {"__slots__": ()},
        )
        derived = derived_type(tensor=pulse.tensor, spec=pulse.spec)
        with self.assertRaises(TypeError):
            require_admitted_member_types(
                _LiteralCollection(members=(derived,)),
                admitted=(PulseResponse,),
            )
        with self.assertRaises(ValueError):
            require_exact_member_types(collection, required=())
        with self.assertRaises(ValueError):
            require_member_count(collection, maximum=0)

    def test_config_component_preparation_and_source_requirements(self) -> None:
        config = charge_config()
        require_config_components(
            spec=config.spec,
            spec_type=type(config.spec),
            kernels=config.kernels,
            kernels_type=type(config.kernels),
            field="ChargeConfig",
        )
        with self.assertRaises(TypeError):
            require_config_components(
                spec=config.spec,
                spec_type=type(config.spec),
                kernels=noise_config().kernels,
                kernels_type=type(config.kernels),
                field="ChargeConfig",
            )
        require_prepared_config(
            is_prepared=True,
            working_dtype=torch.float64,
            field="Config",
        )
        for is_prepared, dtype in (
            (False, torch.float64),
            (True, None),
        ):
            with self.subTest(
                is_prepared=is_prepared,
                working_dtype=dtype,
            ):
                with self.assertRaises(ValueError):
                    require_prepared_config(
                        is_prepared=is_prepared,
                        working_dtype=dtype,
                        field="Config",
                    )
        live = source()
        equal = Photoelectrons(
            tensor=live.tensor.clone(),
            spec=type(live.spec)(
                axes=live.spec.axes,
                device=live.spec.device,
                dtype=live.spec.dtype,
                unit=live.spec.unit,
            ),
        )
        self.assertIsNot(equal.spec, live.spec)
        self.assertEqual(equal.spec, live.spec)
        require_prepared_sources((equal,), source_specs=(live.spec,))
        with self.assertRaises(ValueError):
            require_prepared_sources(
                (live,),
                source_specs=(charge_config().spec,),
            )

    def test_capacity_requirements_preserve_exact_bounds(self) -> None:
        require_tensor_capacity(
            (2, 3),
            dtype=torch.float64,
            field="tensor",
        )
        require_address_capacity(
            (2, 3),
            address_shape=(4,),
            field="address",
        )
        with self.assertRaises((ValueError, OverflowError)):
            require_tensor_capacity(
                (1 << 62, 2),
                dtype=torch.float64,
                field="tensor",
            )
        with self.assertRaises((ValueError, OverflowError)):
            require_address_capacity(
                (1 << 62, 2),
                address_shape=(2,),
                field="address",
            )
