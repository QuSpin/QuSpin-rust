"""Smoke tests: every script in ``python/examples`` runs to completion.

These guard against the examples silently breaking when the Python API
changes. They only assert the script exits cleanly and prints something
recognisable, not on the exact output.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("example*.py"))


def test_examples_directory_is_not_empty():
    assert EXAMPLES, f"no example scripts found under {EXAMPLES_DIR}"


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(script):
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=script.parent,
    )
    assert result.returncode == 0, (
        f"{script.name} exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip(), f"{script.name} produced no output"
