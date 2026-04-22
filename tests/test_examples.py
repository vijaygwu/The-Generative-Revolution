"""Smoke tests for practitioner-facing companion examples."""

from __future__ import annotations

import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from the_generative_revolution.examples.anomaly_screening_flow import run_demo as run_flow_demo
from the_generative_revolution.examples.multimodal_creative_assistant import (
    run_demo as run_assistant_demo,
)
from the_generative_revolution.examples.product_imaging_diffusion import (
    run_demo as run_product_demo,
)


def test_product_imaging_demo() -> None:
    result = run_product_demo(seed=0)
    assert result["workflow"] == "retail_product_imaging_smoke_test"
    assert result["train_loss"] > 0
    assert result["sampler_mode"] == "classifier_free_guided_ddim"
    assert result["conditioning_hook_gap"] > 0
    assert 0.0 < result["conditioning_dropout_rate"] < 1.0
    assert result["guidance_scale"] == 2.0
    assert result["sampler_steps"] == 6
    assert result["sample_stats"]["std"] > 0
    assert result["guidance_hook_stats"]["std"] > 0


def test_anomaly_screening_demo() -> None:
    result = run_flow_demo(seed=0)
    assert result["workflow"] == "industrial_anomaly_screening_smoke_test"
    assert result["demo_mode"] == "untrained_flow_with_split_calibration"
    assert result["threshold_source"] == "calibration_batch_5th_percentile"
    assert result["split_sizes"] == {"calibration": 256, "reference": 256, "anomaly": 256}
    assert result["anomaly_mean_log_prob"] < result["reference_mean_log_prob"]
    assert 0.0 <= result["reference_flag_rate"] <= 1.0
    assert 0.0 <= result["anomaly_flag_rate"] <= 1.0
    assert result["anomaly_flag_rate"] > result["reference_flag_rate"]
    assert result["triage_sample_shape"] == [4, 6]


def test_multimodal_creative_assistant_demo() -> None:
    result = run_assistant_demo(seed=0)
    assert result["workflow"] == "multimodal_creative_assistant"
    assert 0 <= result["retrieved_asset_index"] < 4
    assert result["retrieval_margin"] > 0
    assert result["unique_codes_used"] > 0
    assert result["guidance_shift_norm"] > 0
    assert result["fid_proxy"] >= 0


def main() -> None:
    print("Running practitioner example tests...")
    test_product_imaging_demo()
    test_anomaly_screening_demo()
    test_multimodal_creative_assistant_demo()
    print("All practitioner example tests passed.")


if __name__ == "__main__":
    main()
