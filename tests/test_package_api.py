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


def test_package_version_import_stays_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import the_generative_revolution as tgr; "
                "print(json.dumps({'version': tgr.__version__, 'torch_loaded': 'torch' in sys.modules}))"
            ),
        ],
        cwd=_repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload == {"version": "0.1.0", "torch_loaded": False}


def test_cli_dispatch_function() -> None:
    result = run_named_demo("product-imaging", seed=0)
    assert result["workflow"] == "retail_product_imaging_smoke_test"
    assert result["sampler_mode"] == "classifier_free_guided_ddim"
    assert result["sampler_steps"] == 6


def test_package_scoped_demo_module() -> None:
    from the_generative_revolution.examples.multimodal_creative_assistant import (
        run_demo as run_assistant_demo,
    )

    result = run_assistant_demo(seed=0)
    assert result["workflow"] == "multimodal_creative_assistant"


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


def test_examples_wrappers_run_from_repo_root() -> None:
    wrappers = [
        (
            "examples/product_imaging_diffusion.py",
            "retail_product_imaging_smoke_test",
        ),
        (
            "examples/anomaly_screening_flow.py",
            "industrial_anomaly_screening_smoke_test",
        ),
        (
            "examples/multimodal_creative_assistant.py",
            "multimodal_creative_assistant",
        ),
    ]

    for script_path, workflow in wrappers:
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["workflow"] == workflow


def main() -> None:
    print("Running package API tests...")
    test_package_exports_core_symbols()
    test_package_version_import_stays_lazy()
    test_cli_dispatch_function()
    test_package_scoped_demo_module()
    test_python_m_entrypoint()
    test_examples_wrappers_run_from_repo_root()
    print("All package API tests passed.")


if __name__ == "__main__":
    main()
