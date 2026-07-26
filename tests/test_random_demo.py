import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import nbformat
from nbclient import NotebookClient
import torch


_NOTEBOOK = Path("demos/random.ipynb")
_EXPECTED_CODE_IDS = (
    "imports",
    "frontier",
    "elements",
    "address",
    "words",
    "kernel-and-rate",
    "poisson",
    "chunks",
    "plot",
)


def _read_notebook():
    return nbformat.read(_NOTEBOOK, as_version=4)


def _code_cells(notebook):
    return tuple(cell for cell in notebook.cells if cell.cell_type == "code")


def _normalized_execution(notebook) -> tuple[tuple[int, str], ...]:
    normalized = []
    for cell in _code_cells(notebook):
        outputs = deepcopy(cell.outputs)
        for output in outputs:
            metadata = output.get("metadata")
            if isinstance(metadata, dict):
                metadata.pop("execution", None)
        normalized.append(
            (
                cell.execution_count,
                json.dumps(outputs, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(normalized)


def _execute(notebook):
    replay = deepcopy(notebook)
    for cell in _code_cells(replay):
        cell.execution_count = None
        cell.outputs = []
    return NotebookClient(
        replay,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()


class AddressedRandomDemoTest(unittest.TestCase):
    def test_notebook_structure_execution_counts_and_outputs_are_frozen(self) -> None:
        notebook = _read_notebook()
        code = _code_cells(notebook)
        self.assertEqual(tuple(cell.id for cell in code), _EXPECTED_CODE_IDS)
        self.assertEqual(
            tuple(cell.execution_count for cell in code),
            tuple(range(1, 10)),
        )
        self.assertTrue(all(cell.outputs for cell in code))
        self.assertFalse(
            any(
                output.output_type == "error"
                for cell in code
                for output in cell.outputs
            )
        )
        pngs = tuple(
            output.data["image/png"]
            for cell in code
            for output in cell.outputs
            if output.output_type == "display_data"
            and "image/png" in output.data
        )
        self.assertEqual(len(pngs), 1)
        self.assertGreater(len(pngs[0]), 40_000)

    def test_narrative_uses_public_addressed_surface_and_warns_on_private_key(
        self,
    ) -> None:
        notebook = _read_notebook()
        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        for phrase in (
            "./create_environment.sh",
            "conda activate tensor_dslab",
            "pure mapping",
            "not the public floating-point `Charge` TensorField",
            "unsupported implementation detail",
            "users do not configure production role keys",
            "Poisson splitting/superposition",
            "not a source-total Poisson",
            "root capacity",
            "global RNG state remains unchanged",
        ):
            self.assertIn(phrase, markdown)
        source = "\n".join(cell.source for cell in _code_cells(notebook))
        for name in (
            "RngElements",
            "RngAddress",
            "Threefry4x32",
            "CounterRng",
            "ProbabilityKernel",
            "PoissonDistribution",
            "DELAYED_CROSSTALK_RETAINED_RNG_KEY",
        ):
            self.assertIn(name, source)
        self.assertNotIn("RngPositions", source)

    def test_code_has_no_private_tensorcore_or_environment_mutation(self) -> None:
        notebook = _read_notebook()
        source = "\n".join(cell.source for cell in _code_cells(notebook))
        tree = ast.parse(source)
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any(
                module is not None
                and module.startswith("tensor_core.")
                for module in imports
            )
        )
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("MultinomialDistribution", called_names)
        for forbidden in (
            "conda ",
            "pip ",
            "subprocess",
            "os.environ",
            "torch.cuda",
            "_values",
            "_upper_bound",
            "_permute",
            "manual_seed",
        ):
            self.assertNotIn(forbidden, source)
        self.assertEqual(source.count("0x54445331"), 0)
        self.assertNotRegex(source, r"0x0000_000[0-9A-F]")

    def test_frozen_word_rate_and_sample_tables_are_exact(self) -> None:
        notebook = _read_notebook()
        by_id = {cell.id: cell for cell in _code_cells(notebook)}
        words = "\n".join(
            output.data.get("text/plain", "")
            for output in by_id["words"].outputs
            if hasattr(output, "data")
        )
        for row in (
            "2797997635,  824971383, 2664289792",
            "4088737205, 1368442777, 2447842931",
            "2294399230, 1607253094, 3446694057",
        ):
            self.assertIn(row, words)
        rate = by_id["kernel-and-rate"].outputs[0].data["text/plain"]
        self.assertEqual(
            rate,
            "tensor([0.6600, 0.5800, 1.1200, 0.5200, 0.6000, 0.2400], dtype=torch.float64)",
        )
        expected_sample = "tensor([1, 2, 1, 1, 0, 0])"
        self.assertEqual(
            by_id["poisson"].outputs[0].data["text/plain"],
            expected_sample,
        )
        self.assertEqual(
            by_id["chunks"].outputs[0].data["text/plain"],
            expected_sample,
        )

    def test_actual_source_execution_proves_repeat_chunk_and_global_rng_identity(
        self,
    ) -> None:
        notebook = _read_notebook()
        namespace: dict[str, object] = {}
        global_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            del temporary
            for cell in _code_cells(notebook):
                if cell.id == "plot":
                    continue
                exec(compile(cell.source, f"<{cell.id}>", "exec"), namespace)
        sampled = cast_tensor(namespace["sampled_avalanches"])
        repeated = cast_tensor(namespace["repeated_avalanches"])
        chunked = cast_tensor(namespace["chunked_avalanches"])
        self.assertIs(type(sampled), torch.Tensor)
        self.assertTrue(torch.equal(sampled, repeated))
        self.assertTrue(torch.equal(sampled, chunked))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        self.assertEqual(sampled.device.type, "cpu")

    def test_immediate_replays_are_byte_identical_to_committed_execution(self) -> None:
        committed = _read_notebook()
        first = _execute(committed)
        second = _execute(committed)
        self.assertEqual(
            _normalized_execution(first),
            _normalized_execution(second),
        )
        self.assertEqual(
            _normalized_execution(first),
            _normalized_execution(committed),
        )

    def test_notebook_is_privacy_safe_and_plot_is_cpu_only(self) -> None:
        text = _NOTEBOOK.read_text()
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/private/", text)
        self.assertNotIn("/scratch/", text)
        self.assertNotIn("token", text.lower())
        self.assertNotIn("job_id", text.lower())
        plot_source = next(
            cell.source
            for cell in _code_cells(_read_notebook())
            if cell.id == "plot"
        )
        self.assertIn('to(device="cpu").tolist()', plot_source)
        self.assertNotIn("cuda", plot_source.lower())


def cast_tensor(value: object) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError("expected an exact torch.Tensor")
    return value


if __name__ == "__main__":
    unittest.main()
