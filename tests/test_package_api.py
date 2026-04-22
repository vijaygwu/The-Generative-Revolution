"""Smoke tests for the installable package wrapper and CLI dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_repo_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import the_generative_revolution as tgr
from the_generative_revolution.cli import run_named_demo


def test_package_exports_core_symbols() -> None:
    assert hasattr(tgr, "VAE")
    assert hasattr(tgr, "DDPM")
    assert hasattr(tgr, "compute_fid")
    assert tgr.__version__ == "0.1.0"


def test_cli_dispatch_function() -> None:
    result = run_named_demo("product-imaging", seed=0)
    assert result["workflow"] == "retail_product_imaging_smoke_test"
    assert result["sampler_mode"] == "classifier_free_guided_ddim"
    assert result["sampler_steps"] == 6


def test_python_m_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "the_generative_revolution", "anomaly-screening", "--seed", "0"],
        cwd=_repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["workflow"] == "industrial_anomaly_screening_smoke_test"
    assert payload["demo_mode"] == "untrained_flow_with_split_calibration"
    assert payload["triage_sample_shape"] == [4, 6]


def main() -> None:
    print("Running package API tests...")
    test_package_exports_core_symbols()
    test_cli_dispatch_function()
    test_python_m_entrypoint()
    print("All package API tests passed.")


if __name__ == "__main__":
    main()
