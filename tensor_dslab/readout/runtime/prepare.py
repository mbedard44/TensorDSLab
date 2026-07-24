from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import final

import torch
from tensor_core import CounterRng, TensorField

from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.analog_waveform.runtime.prepare import (
    AnalogWaveformRuntime,
    prepare_analog_waveform,
)
from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.charge.runtime.prepare import (
    ChargeRuntime,
    prepare_charge,
)
from tensor_dslab.readout.collection import ReadoutCollection
from tensor_dslab.readout.config import ReadoutConfig
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform
from tensor_dslab.readout.digitized_waveform.runtime.prepare import (
    DigitizedWaveformRuntime,
    prepare_digitized_waveform,
)
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.noise_waveform.runtime.prepare import (
    NoiseWaveformRuntime,
    prepare_noise_waveform,
)
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.photoelectrons.runtime.validate import (
    validate_photoelectrons,
)
from tensor_dslab.readout.pure_waveform.field import PureWaveform
from tensor_dslab.readout.pure_waveform.runtime.prepare import (
    PureWaveformRuntime,
    prepare_pure_waveform,
)
from tensor_dslab.readout.runtime.sampling import prepare_sampling


@final
@dataclass(frozen=True, slots=True)
class ReadoutRuntime:
    charge: ChargeRuntime | None
    pure_waveform: PureWaveformRuntime | None
    noise_waveform: NoiseWaveformRuntime | None
    analog_waveform: AnalogWaveformRuntime | None
    digitized_waveform: DigitizedWaveformRuntime | None


def _require_requested_products(
    products: Iterable[type[TensorField]],
) -> frozenset[type[TensorField]]:
    requested_items = tuple(products)
    if not requested_items:
        raise ValueError("products must contain at least one product type")
    accepted_types = ReadoutCollection.accepted_field_types()
    for index, item in enumerate(requested_items):
        if not any(item is accepted for accepted in accepted_types):
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


def prepare_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype,
) -> tuple[frozenset[type[TensorField]], ReadoutRuntime]:
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

    sampling = prepare_sampling(photoelectrons)
    device = photoelectrons.tensor.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("readout simulation supports only CPU and CUDA")
    validate_photoelectrons(photoelectrons)
    if need_charge or need_noise:
        if floating_dtype is not torch.float32 and floating_dtype is not torch.float64:
            raise TypeError("floating_dtype must be torch.float32 or torch.float64")

    charge_runtime: ChargeRuntime | None = None
    pure_runtime: PureWaveformRuntime | None = None
    noise_runtime: NoiseWaveformRuntime | None = None
    analog_runtime: AnalogWaveformRuntime | None = None
    digitized_runtime: DigitizedWaveformRuntime | None = None

    if need_charge:
        charge_config = config.charge
        if charge_config is None:
            raise RuntimeError("required Charge configuration disappeared")
        charge_runtime = prepare_charge(
            charge_config,
            photoelectrons=photoelectrons,
            sampling=sampling,
            floating_dtype=floating_dtype,
        )
    if need_pure:
        pure_config = config.pure_waveform
        if pure_config is None:
            raise RuntimeError("required PureWaveform configuration disappeared")
        pure_runtime = prepare_pure_waveform(
            pure_config,
            sampling=sampling,
            floating_dtype=floating_dtype,
            device=device,
        )
    if need_noise:
        noise_config = config.noise_waveform
        if noise_config is None:
            raise RuntimeError("required NoiseWaveform configuration disappeared")
        noise_runtime = prepare_noise_waveform(
            noise_config,
            sampling=sampling,
            shape=photoelectrons.shape,
            floating_dtype=floating_dtype,
            device=device,
        )
    if need_analog:
        analog_config = config.analog_waveform
        if analog_config is None:
            raise RuntimeError("required AnalogWaveform configuration disappeared")
        analog_runtime = prepare_analog_waveform(
            config=analog_config,
            floating_dtype=floating_dtype,
            device=device,
        )
    if need_digitized:
        digitized_config = config.digitized_waveform
        if digitized_config is None:
            raise RuntimeError("required DigitizedWaveform configuration disappeared")
        digitized_runtime = prepare_digitized_waveform(
            config=digitized_config,
            floating_dtype=floating_dtype,
            device=device,
        )

    return requested, ReadoutRuntime(
        charge=charge_runtime,
        pure_waveform=pure_runtime,
        noise_waveform=noise_runtime,
        analog_waveform=analog_runtime,
        digitized_waveform=digitized_runtime,
    )
