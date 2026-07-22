from __future__ import annotations

import torch

from tensor_dslab.readout.photoelectrons.field import Photoelectrons


def validate_photoelectrons(photoelectrons: Photoelectrons) -> None:
    if bool(torch.any(photoelectrons.tensor < 0).item()):
        raise ValueError("Photoelectrons values must be nonnegative")
