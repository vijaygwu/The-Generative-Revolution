"""Split-aware anomaly-screening smoke test built on the flow companion code."""

from __future__ import annotations

import json

import torch

from ..flows import RealNVP


def run_demo(seed: int = 0) -> dict[str, object]:
    """Exercise flow scoring and split-aware thresholding without a training loop."""
    torch.manual_seed(seed)

    model = RealNVP(dim=6, num_layers=4, hidden_dim=64)
    calibration_batch = torch.randn(256, 6)
    reference_batch = torch.randn(256, 6)
    anomaly_batch = torch.randn(256, 6)
    anomaly_batch[:, 0] += 3.0
    anomaly_batch[:, 1] -= 2.0

    calibration_scores = model.log_prob(calibration_batch)
    reference_scores = model.log_prob(reference_batch)
    anomaly_scores = model.log_prob(anomaly_batch)
    threshold = float(torch.quantile(calibration_scores, 0.05).item())

    reference_flag_rate = float((reference_scores < threshold).float().mean().item())
    anomaly_flag_rate = float((anomaly_scores < threshold).float().mean().item())
    triage_samples = model.sample(4)

    return {
        "workflow": "industrial_anomaly_screening_smoke_test",
        "demo_mode": "untrained_flow_with_split_calibration",
        "threshold_source": "calibration_batch_5th_percentile",
        "split_sizes": {"calibration": 256, "reference": 256, "anomaly": 256},
        "threshold": threshold,
        "calibration_mean_log_prob": float(calibration_scores.mean().item()),
        "reference_mean_log_prob": float(reference_scores.mean().item()),
        "anomaly_mean_log_prob": float(anomaly_scores.mean().item()),
        "reference_flag_rate": reference_flag_rate,
        "anomaly_flag_rate": anomaly_flag_rate,
        "triage_sample_shape": list(triage_samples.shape),
    }


def main() -> None:
    result = run_demo()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
