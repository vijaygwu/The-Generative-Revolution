"""Checks that saved expected artifacts stay in sync with the practitioner demos."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_repo_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from the_generative_revolution.examples.anomaly_screening_flow import run_demo as run_flow_demo
from the_generative_revolution.examples.multimodal_creative_assistant import (
    run_demo as run_assistant_demo,
)
from the_generative_revolution.examples.product_imaging_diffusion import (
    run_demo as run_product_demo,
)

EXPECTED_DIR = _repo_root / "artifacts" / "expected"


def _load_expected(name: str) -> dict[str, object]:
    return json.loads((EXPECTED_DIR / name).read_text())


def _assert_close(actual, expected, path="root") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), (path, type(actual), type(expected))
        assert set(actual.keys()) == set(expected.keys()), (path, actual.keys(), expected.keys())
        for key in expected:
            _assert_close(actual[key], expected[key], path=f"{path}.{key}")
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), (path, type(actual), type(expected))
        assert len(actual) == len(expected), (path, len(actual), len(expected))
        for idx, (a_item, e_item) in enumerate(zip(actual, expected)):
            _assert_close(a_item, e_item, path=f"{path}[{idx}]")
        return

    if isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6), (
            path,
            actual,
            expected,
        )
        return

    assert actual == expected, (path, actual, expected)


def test_product_imaging_expected_artifact() -> None:
    _assert_close(
        run_product_demo(seed=0),
        _load_expected("product_imaging_diffusion.json"),
        path="product_imaging_diffusion",
    )


def test_anomaly_screening_expected_artifact() -> None:
    _assert_close(
        run_flow_demo(seed=0),
        _load_expected("anomaly_screening_flow.json"),
        path="anomaly_screening_flow",
    )


def test_multimodal_creative_assistant_expected_artifact() -> None:
    _assert_close(
        run_assistant_demo(seed=0),
        _load_expected("multimodal_creative_assistant.json"),
        path="multimodal_creative_assistant",
    )


def main() -> None:
    print("Running expected artifact sync tests...")
    test_product_imaging_expected_artifact()
    test_anomaly_screening_expected_artifact()
    test_multimodal_creative_assistant_expected_artifact()
    print("All expected artifact sync tests passed.")


if __name__ == "__main__":
    main()
