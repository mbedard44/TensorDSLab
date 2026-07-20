from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from tensor_core import CounterRng, TensorField

from tensor_dslab.readout._requirements import _require_sampling
from tensor_dslab.readout.analog_waveform._produce import (
    _AnalogWaveformPlan,
    _prepare_analog_waveform,
    _produce_analog_waveform,
)
from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.charge._produce import (
    _ChargePlan,
    _prepare_charge,
    _produce_charge,
)
from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.collection import ReadoutCollection
from tensor_dslab.readout.config import ReadoutConfig
from tensor_dslab.readout.digitized_waveform._produce import (
    _DigitizedWaveformPlan,
    _prepare_digitized_waveform,
    _produce_digitized_waveform,
)
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform
from tensor_dslab.readout.noise_waveform._produce import (
    _NoiseWaveformPlan,
    _prepare_noise_waveform,
    _produce_noise_waveform,
)
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.photoelectrons.field import (
    Photoelectrons,
    _require_valid_values as _require_valid_photoelectrons,
)
from tensor_dslab.readout.pure_waveform._produce import (
    _PureWaveformPlan,
    _prepare_pure_waveform,
    _produce_pure_waveform,
)
from tensor_dslab.readout.pure_waveform.field import PureWaveform


_PRODUCT_TYPES: tuple[type[TensorField], ...] = (
    Photoelectrons,
    Charge,
    PureWaveform,
    NoiseWaveform,
    AnalogWaveform,
    DigitizedWaveform,
)


@dataclass(frozen=True, slots=True)
class _ReadoutPlan:
    requested: frozenset[type[TensorField]]
    need_charge: bool
    need_pure: bool
    need_noise: bool
    need_analog: bool
    need_digitized: bool
    charge: _ChargePlan | None
    pure: _PureWaveformPlan | None
    noise: _NoiseWaveformPlan | None
    analog: _AnalogWaveformPlan | None
    digitized: _DigitizedWaveformPlan | None


def _require_requested_products(
    products: Iterable[type[TensorField]],
) -> frozenset[type[TensorField]]:
    requested_items = tuple(products)
    if not requested_items:
        raise ValueError("products must contain at least one product type")
    for index, item in enumerate(requested_items):
        if not any(item is accepted for accepted in _PRODUCT_TYPES):
            raise TypeError("products must contain exact recognized product classes")
        if any(item is previous for previous in requested_items[:index]):
            raise ValueError("products must not contain duplicate product types")
    return frozenset(requested_items)


def _require_config_closure(
    config: ReadoutConfig,
    *,
    need_charge: bool,
    need_pure: bool,
    need_noise: bool,
    need_analog: bool,
    need_digitized: bool,
) -> None:
    required = (
        (need_charge, config.charge, "Charge"),
        (need_pure, config.pure_waveform, "PureWaveform"),
        (need_noise, config.noise_waveform, "NoiseWaveform"),
        (need_analog, config.analog_waveform, "AnalogWaveform"),
        (need_digitized, config.digitized_waveform, "DigitizedWaveform"),
    )
    for needed, product_config, name in required:
        if needed and product_config is None:
            raise ValueError(f"{name} requires its product configuration")


def _require_unique_rng_keys(
    *,
    charge: _ChargePlan | None,
    noise: _NoiseWaveformPlan | None,
) -> None:
    roles = (
        (() if charge is None else charge.rng_roles)
        + (() if noise is None else noise.rng_roles)
    )

    for index, (role, key) in enumerate(roles):
        for previous_role, previous_key in roles[:index]:
            if key == previous_key:
                raise ValueError(
                    "distinct stochastic roles require distinct RNG keys: "
                    f"{previous_role} and {role}"
                )


def _prepare_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> _ReadoutPlan:
    requested = _require_requested_products(products)
    if type(photoelectrons) is not Photoelectrons:
        raise TypeError("photoelectrons must be exactly Photoelectrons")
    if type(config) is not ReadoutConfig:
        raise TypeError("config must be exactly ReadoutConfig")
    if not isinstance(rng, CounterRng):
        raise TypeError("rng must be a CounterRng")

    need_digitized = DigitizedWaveform in requested
    need_analog = AnalogWaveform in requested or need_digitized
    need_pure = PureWaveform in requested or need_analog
    need_noise = NoiseWaveform in requested or need_analog
    need_charge = Charge in requested or need_pure
    _require_config_closure(
        config,
        need_charge=need_charge,
        need_pure=need_pure,
        need_noise=need_noise,
        need_analog=need_analog,
        need_digitized=need_digitized,
    )

    _require_sampling(photoelectrons, config.sampling)
    device = photoelectrons.tensor.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("readout simulation supports only CPU and CUDA")
    _require_valid_photoelectrons(photoelectrons)
    if need_charge or need_noise:
        if floating_dtype is not torch.float32 and floating_dtype is not torch.float64:
            raise TypeError("floating_dtype must be torch.float32 or torch.float64")

    charge_plan: _ChargePlan | None = None
    pure_plan: _PureWaveformPlan | None = None
    noise_plan: _NoiseWaveformPlan | None = None
    analog_plan: _AnalogWaveformPlan | None = None
    digitized_plan: _DigitizedWaveformPlan | None = None

    if need_charge:
        charge_config = config.charge
        if charge_config is None:
            raise RuntimeError("required Charge configuration disappeared")
        charge_plan = _prepare_charge(
            photoelectrons,
            sampling=config.sampling,
            config=charge_config,
            floating_dtype=floating_dtype,
        )
    if need_pure:
        pure_config = config.pure_waveform
        if pure_config is None:
            raise RuntimeError("required PureWaveform configuration disappeared")
        pure_plan = _prepare_pure_waveform(
            photoelectrons,
            sampling=config.sampling,
            config=pure_config,
            floating_dtype=floating_dtype,
            device=device,
        )
    if need_noise:
        noise_config = config.noise_waveform
        if noise_config is None:
            raise RuntimeError("required NoiseWaveform configuration disappeared")
        noise_plan = _prepare_noise_waveform(
            photoelectrons,
            sampling=config.sampling,
            config=noise_config,
            floating_dtype=floating_dtype,
        )
    if need_analog:
        analog_config = config.analog_waveform
        if analog_config is None:
            raise RuntimeError("required AnalogWaveform configuration disappeared")
        analog_plan = _prepare_analog_waveform(
            config=analog_config,
            floating_dtype=floating_dtype,
            device=device,
        )
    if need_digitized:
        digitized_config = config.digitized_waveform
        if digitized_config is None:
            raise RuntimeError("required DigitizedWaveform configuration disappeared")
        digitized_plan = _prepare_digitized_waveform(
            config=digitized_config,
            floating_dtype=floating_dtype,
            device=device,
        )

    _require_unique_rng_keys(
        charge=charge_plan,
        noise=noise_plan,
    )
    return _ReadoutPlan(
        requested=requested,
        need_charge=need_charge,
        need_pure=need_pure,
        need_noise=need_noise,
        need_analog=need_analog,
        need_digitized=need_digitized,
        charge=charge_plan,
        pure=pure_plan,
        noise=noise_plan,
        analog=analog_plan,
        digitized=digitized_plan,
    )


def simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype = torch.float32,
) -> ReadoutCollection:
    plan = _prepare_readout(
        photoelectrons,
        products=products,
        config=config,
        rng=rng,
        floating_dtype=floating_dtype,
    )

    charge: Charge | None = None
    pure: PureWaveform | None = None
    noise: NoiseWaveform | None = None
    analog: AnalogWaveform | None = None
    digitized: DigitizedWaveform | None = None

    if plan.need_charge:
        assert plan.charge is not None
        charge = _produce_charge(photoelectrons, plan=plan.charge, rng=rng)
    if plan.need_pure:
        assert charge is not None and plan.pure is not None
        pure = _produce_pure_waveform(charge, plan=plan.pure)
    if plan.need_noise:
        assert plan.noise is not None
        noise = _produce_noise_waveform(photoelectrons, plan=plan.noise, rng=rng)
    if plan.need_analog:
        assert pure is not None and noise is not None and plan.analog is not None
        analog = _produce_analog_waveform(pure, noise, plan=plan.analog)
    if plan.need_digitized:
        assert analog is not None and plan.digitized is not None
        digitized = _produce_digitized_waveform(analog, plan=plan.digitized)

    available: tuple[TensorField | None, ...] = (
        photoelectrons,
        charge,
        pure,
        noise,
        analog,
        digitized,
    )
    retained = tuple(
        field
        for field in available
        if field is not None and type(field) in plan.requested
    )
    return ReadoutCollection(fields=retained)
