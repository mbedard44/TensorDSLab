"""Private completed-source validation."""

from tensor_dslab.common.requirements.field import require_exact_field_spec
from tensor_dslab.common.requirements.tensor import require_values_between
from tensor_dslab.photoelectrons.field import Photoelectrons
from tensor_dslab.photoelectrons.field import PhotoelectronsSpec


def validate_photoelectrons(*, product: Photoelectrons) -> None:
    """Require the complete admitted photoelectron source domain."""

    if type(product) is not Photoelectrons:
        raise TypeError("product must be exact Photoelectrons")
    require_exact_field_spec(product, PhotoelectronsSpec)
    require_values_between(
        product,
        minimum=0,
        maximum=(1 << 53) - 1,
    )
