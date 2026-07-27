"""Private fixed readout RNG namespace and role keys."""

from tensor_core import RngKey


RNG_NAMESPACE = 0x54445331

WHITE_NOISE_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0001)
PSD_NOISE_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0002)
DARK_COUNT_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0003)
TIMING_JITTER_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0004)
DIRECT_CROSSTALK_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0005)
DELAYED_CROSSTALK_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0006)
AFTERPULSE_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0007)
CHARGE_SMEARING_RNG_KEY = RngKey(namespace=RNG_NAMESPACE, stream=0x0000_0008)
