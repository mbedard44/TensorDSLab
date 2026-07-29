"""Private fixed Charge RNG roles and addresses."""

from tensor_core import RngAddress, RngElements, RngKey

_NAMESPACE = 0x54445331
DARK_COUNT_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0003)
TIMING_JITTER_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0004)
DIRECT_CROSSTALK_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0005)
DELAYED_CROSSTALK_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0006)
AFTERPULSE_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0007)
CHARGE_SMEARING_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0008)


def point_address(elements: RngElements, *, key: RngKey) -> RngAddress:
    return RngAddress.root(key=key, elements=elements, shape=(), quantum=0)


def timing_address(elements: RngElements, *, shape: tuple[int, ...]) -> RngAddress:
    return RngAddress.root(
        key=TIMING_JITTER_KEY, elements=elements, shape=shape, quantum=0
    )


def branching_address(
    elements: RngElements,
    *,
    key: RngKey,
    generations: int,
    generation: int,
) -> RngAddress:
    return RngAddress.root(
        key=key, elements=elements, shape=(generations,), quantum=0
    ).select(generation)
