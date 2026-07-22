from __future__ import annotations

from collections.abc import Iterable

import torch
from tensor_core import CounterRng, TensorField

from tensor_dslab.readout.analog_waveform.field import AnalogWaveform
from tensor_dslab.readout.analog_waveform.runtime.produce import (
    produce_analog_waveform,
)
from tensor_dslab.readout.analog_waveform.runtime.validate import (
    validate_analog_waveform,
)
from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.charge.runtime.produce import produce_charge
from tensor_dslab.readout.charge.runtime.validate import validate_charge
from tensor_dslab.readout.collection import ReadoutCollection
from tensor_dslab.readout.config import ReadoutConfig
from tensor_dslab.readout.digitized_waveform.field import DigitizedWaveform
from tensor_dslab.readout.digitized_waveform.runtime.produce import (
    produce_digitized_waveform,
)
from tensor_dslab.readout.digitized_waveform.runtime.validate import (
    validate_digitized_waveform,
)
from tensor_dslab.readout.noise_waveform.field import NoiseWaveform
from tensor_dslab.readout.noise_waveform.runtime.produce import (
    produce_noise_waveform,
)
from tensor_dslab.readout.noise_waveform.runtime.validate import (
    validate_noise_waveform,
)
from tensor_dslab.readout.photoelectrons.field import Photoelectrons
from tensor_dslab.readout.pure_waveform.field import PureWaveform
from tensor_dslab.readout.pure_waveform.runtime.produce import (
    produce_pure_waveform,
)
from tensor_dslab.readout.pure_waveform.runtime.validate import (
    validate_pure_waveform,
)
from tensor_dslab.readout.runtime.prepare import prepare_readout


def simulate_readout(
    photoelectrons: Photoelectrons,
    *,
    products: Iterable[type[TensorField]],
    config: ReadoutConfig,
    rng: CounterRng,
    floating_dtype: torch.dtype = torch.float32,
) -> ReadoutCollection:
    requested, runtime = prepare_readout(
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

    if runtime.charge is not None:
        charge = produce_charge(
            photoelectrons,
            runtime=runtime.charge,
            rng=rng,
        )
        validate_charge(charge, source=photoelectrons, runtime=runtime.charge)
    if runtime.pure_waveform is not None:
        assert charge is not None
        pure = produce_pure_waveform(charge, runtime=runtime.pure_waveform)
        validate_pure_waveform(pure, source=charge)
    if runtime.noise_waveform is not None:
        noise = produce_noise_waveform(
            photoelectrons,
            runtime=runtime.noise_waveform,
            rng=rng,
        )
        validate_noise_waveform(
            noise,
            source=photoelectrons,
            runtime=runtime.noise_waveform,
        )
    if runtime.analog_waveform is not None:
        assert pure is not None and noise is not None
        analog = produce_analog_waveform(
            pure,
            noise,
            runtime=runtime.analog_waveform,
        )
        validate_analog_waveform(analog, pure=pure, noise=noise)
    if runtime.digitized_waveform is not None:
        assert analog is not None
        digitized = produce_digitized_waveform(
            analog,
            runtime=runtime.digitized_waveform,
        )
        validate_digitized_waveform(
            digitized,
            source=analog,
            maximum_code=runtime.digitized_waveform.maximum_code,
        )

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
        if field is not None and type(field) in requested
    )
    return ReadoutCollection(fields=retained)
