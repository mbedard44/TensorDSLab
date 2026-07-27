"""Offline contract evidence for the project environment creator."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
from typing import TypedDict, cast
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "create_environment.sh"
_TENSORCORE_COMMIT = "78d0891bf6c0fefbcad4abe09980867c54202a9e"


class _CommandRecord(TypedDict):
    argv: list[str]
    cwd: str


class EnvironmentScriptTest(unittest.TestCase):
    def _fake_conda(self, directory: Path) -> tuple[Path, Path]:
        executable = directory / "fake-conda"
        log = directory / "commands.jsonl"
        executable.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                from pathlib import Path
                import sys

                log = Path(os.environ["FAKE_CONDA_LOG"])
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({{
                        "argv": sys.argv[1:],
                        "cwd": os.getcwd(),
                    }}) + "\\n")

                if sys.argv[1:] == ["env", "list"]:
                    print("# conda environments:")
                    existing = os.environ.get("FAKE_EXISTING_NAME")
                    if existing:
                        print(f"{{existing}} /fake/envs/{{existing}}")
                elif (
                    os.environ.get("FAKE_FAIL_SMOKE") == "1"
                    and sys.argv[1:5] == ["run", "--name", sys.argv[3], "python"]
                    and "-c" in sys.argv[5:]
                ):
                    raise SystemExit(7)
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable, log

    def _run(
        self,
        directory: Path,
        *arguments: str,
        existing: str | None = None,
        fail_smoke: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], list[_CommandRecord]]:
        executable, log = self._fake_conda(directory)
        environment = os.environ.copy()
        environment["CONDA_EXE"] = str(executable)
        environment["FAKE_CONDA_LOG"] = str(log)
        if existing is not None:
            environment["FAKE_EXISTING_NAME"] = existing
        if fail_smoke:
            environment["FAKE_FAIL_SMOKE"] = "1"
        completed = subprocess.run(
            ["bash", str(_SCRIPT), *arguments],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        records = (
            [
                cast(_CommandRecord, json.loads(line))
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            if log.exists()
            else []
        )
        return completed, records

    def test_default_environment_commands_and_smoke_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            completed, records = self._run(directory)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["argv"], ["env", "list"])
        self.assertEqual(
            records[1]["argv"],
            [
                "create",
                "--yes",
                "--name",
                "tensor_dslab",
                "--no-default-packages",
                "--override-channels",
                "--channel",
                "conda-forge",
                "python=3.14.6",
                "pip",
            ],
        )
        self.assertEqual(
            records[2]["argv"],
            [
                "run",
                "--name",
                "tensor_dslab",
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                f"{_ROOT}[demos]",
            ],
        )

        smoke_arguments = records[3]["argv"]
        self.assertEqual(smoke_arguments[:5], [
            "run",
            "--name",
            "tensor_dslab",
            "python",
            "-c",
        ])
        smoke_code = smoke_arguments[5]
        self.assertIn('version("tensor-core") == "0.21.0"', smoke_code)
        self.assertIn(_TENSORCORE_COMMIT, smoke_code)
        self.assertIn('read_text("direct_url.json")', smoke_code)
        self.assertIn('"vcs_info"', smoke_code)
        self.assertIn("site.getsitepackages()", smoke_code)
        self.assertIn("not module_path.is_relative_to(repository_root)", smoke_code)
        self.assertIn(
            'SampleAxis.from_period(period=quantity(2, "ns"), count=8)',
            smoke_code,
        )
        self.assertIn("ds20k_veto(sample_axis=sample_axis)", smoke_code)
        self.assertEqual(smoke_arguments[6], str(_ROOT))

        smoke_cwd = Path(str(records[3]["cwd"]))
        self.assertNotEqual(smoke_cwd, _ROOT)
        self.assertFalse(smoke_cwd.is_relative_to(_ROOT))
        self.assertFalse(smoke_cwd.exists())
        self.assertIn("conda activate tensor_dslab", completed.stdout)
        self.assertIn(f"cd {_ROOT}", completed.stdout)
        self.assertIn("python demos/readout.py", completed.stdout)

    def test_alternate_environment_name_is_used_consistently(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, records = self._run(
                Path(raw_directory),
                "role-private",
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for record in records[1:]:
            arguments = record["argv"]
            self.assertEqual(
                arguments[arguments.index("--name") + 1],
                "role-private",
            )
        self.assertIn("conda activate role-private", completed.stdout)

    def test_existing_environment_is_refused_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, records = self._run(
                Path(raw_directory),
                "occupied",
                existing="occupied",
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            [record["argv"] for record in records],
            [["env", "list"]],
        )
        self.assertIn("already exists", completed.stderr)

    def test_failed_smoke_also_cleans_external_temporary_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            completed, records = self._run(
                Path(raw_directory),
                fail_smoke=True,
            )

        self.assertEqual(completed.returncode, 7)
        self.assertEqual(len(records), 4)
        smoke_cwd = Path(str(records[-1]["cwd"]))
        self.assertFalse(smoke_cwd.exists())

    def test_usage_and_nonempty_name_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            too_many, records = self._run(directory, "one", "two")
        self.assertEqual(too_many.returncode, 2)
        self.assertEqual(records, [])
        self.assertIn("usage:", too_many.stderr)

        with tempfile.TemporaryDirectory() as raw_directory:
            empty, records = self._run(Path(raw_directory), "")
        self.assertEqual(empty.returncode, 2)
        self.assertEqual(records, [])
        self.assertIn("must not be empty", empty.stderr)

    def test_script_has_safe_static_contract_and_executable_mode(self) -> None:
        mode = stat.S_IMODE(_SCRIPT.stat().st_mode)
        self.assertEqual(mode, 0o755)
        subprocess.run(
            ["bash", "-n", str(_SCRIPT)],
            check=True,
            cwd=_ROOT,
        )
        source = _SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("PYTHONPATH", source)
        self.assertNotIn("conda activate", "\n".join(
            line for line in source.splitlines()
            if not line.startswith("printf ")
        ))
        self.assertNotIn("jupyter kernelspec", source)
        self.assertNotIn("conda init", source)
        self.assertNotIn("--editable", source)
        self.assertNotIn(" -e ", source)
        self.assertNotIn("--force", source)
        self.assertIn('mktemp -d "/tmp/tensor-dslab-smoke.', source)
        self.assertIn("trap cleanup_smoke_directory EXIT", source)


if __name__ == "__main__":
    unittest.main()
