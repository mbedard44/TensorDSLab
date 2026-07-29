import unittest

from tensor_dslab.charge.runtime.random import (
    AFTERPULSE_KEY,
    CHARGE_SMEARING_KEY,
    DARK_COUNT_KEY,
    DELAYED_CROSSTALK_KEY,
    DIRECT_CROSSTALK_KEY,
    TIMING_JITTER_KEY,
)
from tensor_dslab.noise_waveform.runtime.random import (
    PSD_NOISE_KEY,
    WHITE_NOISE_KEY,
)


class RngIdentityTests(unittest.TestCase):
    def test_exact_private_role_streams(self) -> None:
        keys = (
            WHITE_NOISE_KEY,
            PSD_NOISE_KEY,
            DARK_COUNT_KEY,
            TIMING_JITTER_KEY,
            DIRECT_CROSSTALK_KEY,
            DELAYED_CROSSTALK_KEY,
            AFTERPULSE_KEY,
            CHARGE_SMEARING_KEY,
        )
        self.assertEqual(tuple(key.namespace for key in keys), (0x54445331,) * 8)
        self.assertEqual(tuple(key.stream for key in keys), tuple(range(1, 9)))
