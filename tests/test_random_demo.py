"""Executed addressed-distribution notebook evidence."""

import ast
import hashlib
from pathlib import Path
import unittest

import nbformat  # pyright: ignore[reportMissingImports]
import torch


NOTEBOOK = Path("demos/random.ipynb")
NOTEBOOK_SHA256 = "3a680a948965db6c64aef9fee8a4ccb9efc42cb96569d9c47f5b0811f2efce98"
CODE_IDS = ("imports", "frontier", "rate", "address", "sample", "chunks", "plot")


def _read():
    return nbformat.read(NOTEBOOK, as_version=4)


def _code(notebook):
    return tuple(cell for cell in notebook.cells if cell.cell_type == "code")


class AddressedRandomDemoTest(unittest.TestCase):
    def test_structure_outputs_and_plot_are_frozen(self) -> None:
        code = _code(_read())
        self.assertEqual(tuple(cell.id for cell in code), CODE_IDS)
        self.assertEqual(tuple(cell.execution_count for cell in code), tuple(range(1, 8)))
        self.assertFalse(
            any(
                output.output_type == "error"
                for cell in code
                for output in cell.outputs
            )
        )
        png = tuple(
            output.data["image/png"]
            for cell in code
            for output in cell.outputs
            if output.output_type == "display_data" and "image/png" in output.data
        )
        self.assertEqual(len(png), 1)
        self.assertGreater(len(png[0]), 20_000)

    def test_narrative_and_public_tensorcore_surface_are_exact(self) -> None:
        notebook = _read()
        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        for phrase in (
            "./create_environment.sh",
            "conda activate tensor_dslab",
            "unsupported implementation detail",
            "Poisson splitting/superposition",
            "root capacity",
        ):
            self.assertIn(phrase, markdown)
        source = "\n".join(cell.source for cell in _code(notebook))
        for name in (
            "RngElements",
            "RngAddress",
            "Threefry4x32",
            "PoissonDistribution",
            "DELAYED_CROSSTALK_RNG_KEY",
        ):
            self.assertIn(name, source)
        self.assertNotIn("ProbabilityKernel", source)
        self.assertNotIn("RngPositions", source)

    def test_no_private_tensorcore_or_environment_mutation(self) -> None:
        source = "\n".join(cell.source for cell in _code(_read()))
        tree = ast.parse(source)
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any(module and module.startswith("tensor_core.") for module in imports)
        )
        for forbidden in ("conda ", "pip ", "subprocess", "torch.cuda"):
            self.assertNotIn(forbidden, source)

    def test_actual_code_replays_and_preserves_global_rng(self) -> None:
        notebook = _read()
        snapshots = []
        global_before = torch.random.get_rng_state().clone()
        for _ in range(2):
            namespace: dict[str, object] = {}
            for cell in _code(notebook):
                if cell.id == "plot":
                    continue
                exec(compile(cell.source, f"<{cell.id}>", "exec"), namespace)
            sampled = namespace["sampled_avalanches"]
            repeated = namespace["repeated_avalanches"]
            chunked = namespace["chunked_avalanches"]
            assert type(sampled) is torch.Tensor
            assert type(repeated) is torch.Tensor
            assert type(chunked) is torch.Tensor
            self.assertTrue(torch.equal(sampled, repeated))
            self.assertTrue(torch.equal(sampled, chunked))
            snapshots.append(sampled.clone())
        self.assertTrue(torch.equal(snapshots[0], snapshots[1]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))

    def test_notebook_is_privacy_safe_and_cpu_only(self) -> None:
        payload = NOTEBOOK.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), NOTEBOOK_SHA256)
        text = payload.decode()
        for forbidden in (
            "/Users/",
            "/private/",
            "/scratch/",
            "torch.cuda",
            "iopub.execute_input",
            "shell.execute_reply",
        ):
            self.assertNotIn(forbidden, text)
        source = "\n".join(cell.source for cell in _code(_read()))
        self.assertIn('device="cpu"', source)


for _index in range(7):
    def _cell_case(self: AddressedRandomDemoTest, index: int = _index) -> None:
        cell = _code(_read())[index]
        self.assertEqual(cell.id, CODE_IDS[index])
        self.assertEqual(cell.execution_count, index + 1)

    setattr(AddressedRandomDemoTest, f"test_executed_cell_{_index:02d}", _cell_case)
