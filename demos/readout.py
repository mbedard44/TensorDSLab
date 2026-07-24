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
            count=5000,
        ),
    )
    shape = tuple(axis.size for axis in axes)
    counts = torch.zeros(shape, dtype=torch.int64, device="cpu")
    counts[0, 0, 100] = 1
    counts[0, 0, 1300] = 2
    counts[0, 0, 2500] = 3
    counts[0, 0, 3700] = 4
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
