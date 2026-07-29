"""Private fixed noise RNG roles and addresses."""

from tensor_core import RngAddress, RngElements, RngKey

_NAMESPACE = 0x54445331
WHITE_NOISE_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0001)
PSD_NOISE_KEY = RngKey(namespace=_NAMESPACE, stream=0x0000_0002)


def white_noise_address(elements: RngElements) -> RngAddress:
    return RngAddress.root(key=WHITE_NOISE_KEY, elements=elements, shape=(), quantum=0)


def psd_noise_address(elements: RngElements) -> RngAddress:
    return RngAddress.root(key=PSD_NOISE_KEY, elements=elements, shape=(), quantum=0)
