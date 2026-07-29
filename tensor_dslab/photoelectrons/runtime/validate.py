"""Private completed-source validation."""

import torch

from tensor_dslab.photoelectrons.field import Photoelectrons


def validate_photoelectrons(*, product: Photoelectrons) -> None:
    """Require the complete admitted photoelectron source domain."""

    if type(product) is not Photoelectrons:
        raise TypeError("product must be exact Photoelectrons")
    tensor = product.tensor
    if tensor.dtype is not torch.int64:
        raise ValueError("Photoelectrons must use torch.int64")
    if bool((tensor < 0).any()) or bool((tensor > (1 << 53) - 1).any()):
        raise ValueError("Photoelectrons values must be in [0, 2**53 - 1]")
