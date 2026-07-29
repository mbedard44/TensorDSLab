from pathlib import Path
import unittest

import tensor_core


class TensorCoreAdoptionTests(unittest.TestCase):
    def test_exact_dependency_and_no_readout_package(self) -> None:
        dependency = Path("pyproject.toml").read_text()
        self.assertIn(
            "19bfae35fbc773b55cac7bcd659dda57c4dee6d6",
            dependency,
        )
        self.assertEqual(len(tensor_core.__all__), 34)
        self.assertFalse(Path("tensor_dslab/readout").exists())
