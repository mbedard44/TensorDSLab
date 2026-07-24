"""Run the public provisional DS20k Veto readout demonstration."""

import torch
from tensor_core import Threefry4x32

from tensor_dslab import (
    AnalogWaveform,
    ChannelAxis,
    Charge,
    DigitizedWaveform,
    ExampleAxis,
    NoiseWaveform,
    Photoelectrons,
    PureWaveform,
    SampleAxis,
    quantity,
    simulate_readout,
)
from tensor_dslab.readout.profiles import ds20k_veto


def main() -> None:
    """Construct synthetic truth and print two public readout results."""

    source_generator = torch.Generator(device="cpu")
    source_generator.manual_seed(11)

    axes = (
        ExampleAxis(count=2),
        ChannelAxis(
            labels=(
                "veto-0",
                "veto-1",
                "veto-2",
                "veto-3",
            )
        ),
        SampleAxis.from_period(
            period=quantity(2.0, "ns"),
            count=1280,
        ),
    )
    shape = tuple(axis.size for axis in axes)
    draws = torch.randint(
        low=0,
        high=512,
        size=shape,
        dtype=torch.int64,
        generator=source_generator,
    )
    counts = torch.where(draws < 2, draws + 1, torch.zeros_like(draws))
    photoelectrons = Photoelectrons(tensor=counts, axes=axes)

    requested = (
        Photoelectrons,
        Charge,
        PureWaveform,
        NoiseWaveform,
        AnalogWaveform,
        DigitizedWaveform,
    )
    readout = simulate_readout(
        photoelectrons,
        products=requested,
        config=ds20k_veto(),
        rng=Threefry4x32(seed=17),
        floating_dtype=torch.float32,
    )

    assert readout.field(Photoelectrons) is photoelectrons
    assert readout.accepted_field_types() == frozenset(requested)
    assert tuple(type(field) for field in readout.fields.values()) == requested
    for field in readout.fields.values():
        assert field.axes == axes
        assert field.tensor.shape == shape
        assert field.tensor.device.type == "cpu"
        print(
            type(field).__name__,
            f"shape={tuple(field.tensor.shape)}",
            f"dtype={field.tensor.dtype}",
            f"axes={tuple(type(axis).__name__ for axis in field.axes)}",
            f"device={field.tensor.device}",
        )

    digitized_only = simulate_readout(
        photoelectrons,
        products=(DigitizedWaveform,),
        config=ds20k_veto(),
        rng=Threefry4x32(seed=17),
    )
    assert tuple(digitized_only.fields) == (DigitizedWaveform,)
    print("selected products:", tuple(field.__name__ for field in digitized_only.fields))


if __name__ == "__main__":
    main()
