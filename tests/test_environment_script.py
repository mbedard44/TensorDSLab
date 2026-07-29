from pathlib import Path
import unittest


class EnvironmentScriptTests(unittest.TestCase):
    def test_core_only_smoke_and_exact_dependency(self) -> None:
        text = Path("create_environment.sh").read_text()
        self.assertIn("0.2.0", text)
        self.assertIn("0.22.0", text)
        self.assertIn("19bfae35fbc773b55cac7bcd659dda57c4dee6d6", text)
        self.assertIn('"${repository_root}[demos]"', text)
        self.assertNotIn("SampleAxis", text)
