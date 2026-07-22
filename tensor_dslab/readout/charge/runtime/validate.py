from __future__ import annotations

import torch

from tensor_dslab.readout.charge.field import Charge
from tensor_dslab.readout.charge.runtime.prepare import ChargeRuntime
from tensor_dslab.readout.photoelectrons.field import Photoelectrons


def validate_charge(
    charge: Charge,
    *,
    source: Photoelectrons,
    runtime: ChargeRuntime,
) -> None:
    if not bool(torch.all(torch.isfinite(charge.tensor) & (charge.tensor >= 0)).item()):
        raise RuntimeError("Charge production produced an invalid terminal value")
    if charge.axes is not source.axes or charge.shape != source.shape:
        raise RuntimeError("Charge output must preserve source axes and shape")
    if charge.tensor.dtype is not runtime.floating_dtype:
        raise RuntimeError("Charge output must use the prepared floating dtype")
    if charge.tensor.device != source.tensor.device:
        raise RuntimeError("Charge output must preserve the source device")
    if (
        charge.tensor.untyped_storage().data_ptr()
        == source.tensor.untyped_storage().data_ptr()
    ):
        raise RuntimeError("Charge output must have fresh storage")
