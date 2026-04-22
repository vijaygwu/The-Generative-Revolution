"""Checks that companion notebooks stay committed with cleared outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def test_notebooks_have_cleared_outputs() -> None:
    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert notebook_paths, "Expected at least one notebook in notebooks/"

    for path in notebook_paths:
        payload = json.loads(path.read_text())
        for idx, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("outputs", []) == [], f"{path.name} cell {idx} has outputs"
            assert (
                cell.get("execution_count") is None
            ), f"{path.name} cell {idx} has execution_count={cell.get('execution_count')}"


def main() -> None:
    print("Running notebook hygiene tests...")
    test_notebooks_have_cleared_outputs()
    print("All notebook hygiene tests passed.")


if __name__ == "__main__":
    main()
