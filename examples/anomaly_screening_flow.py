"""Toy anomaly-screening workflow built on the flow companion code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.flows import RealNVP


def run_demo(seed: int = 0) -> dict[str, object]:
    """Score normal vs shifted observations with a flow-based detector."""
    torch.manual_seed(seed)

    model = RealNVP(dim=6, num_layers=4, hidden_dim=64)
    reference_batch = torch.randn(256, 6)
    anomaly_batch = reference_batch.clone()
    anomaly_batch[:, 0] += 3.0
    anomaly_batch[:, 1] -= 2.0

    reference_scores = model.log_prob(reference_batch)
    anomaly_scores = model.log_prob(anomaly_batch)
    threshold = float(torch.quantile(reference_scores, 0.05).item())

    reference_flag_rate = float((reference_scores < threshold).float().mean().item())
    anomaly_flag_rate = float((anomaly_scores < threshold).float().mean().item())
    triage_samples = model.sample(4)

    return {
        "workflow": "industrial_anomaly_screening",
        "threshold": threshold,
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
