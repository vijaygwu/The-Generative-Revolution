"""Toy multimodal creative-assistant workflow for Chapter 25 concepts."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from ..advanced_generative import TinyConditionedDenoiser, classifier_free_guidance, quantize
from ..metrics import compute_fid


def run_demo(seed: int = 0) -> dict[str, object]:
    """Run a synthetic retrieval-guidance-evaluation loop."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    asset_bank = torch.tensor(
        [
            [0.90, 0.15, 0.10, 0.75, 0.35, 0.20],
            [0.20, 0.85, 0.55, 0.10, 0.70, 0.65],
            [0.60, 0.40, 0.80, 0.30, 0.20, 0.75],
            [0.75, 0.30, 0.25, 0.85, 0.55, 0.15],
        ],
        dtype=torch.float32,
    )
    prompt_embedding = torch.tensor(
        [0.82, 0.24, 0.18, 0.78, 0.42, 0.12], dtype=torch.float32
    )

    retrieval_scores = F.cosine_similarity(asset_bank, prompt_embedding.unsqueeze(0), dim=1)
    retrieved_index = int(torch.argmax(retrieval_scores).item())
    retrieved_condition = asset_bank[retrieved_index].unsqueeze(0).repeat(2, 1)

    latent = torch.randn(2, 4, 8, 8)
    codebook = torch.randn(16, 4)
    quantized_latent, code_indices = quantize(latent, codebook)

    denoiser = TinyConditionedDenoiser(channels=4, cond_dim=6)
    guided_latent = classifier_free_guidance(
        denoiser,
        quantized_latent,
        torch.ones(2, dtype=torch.long),
        condition=retrieved_condition,
        null_condition=torch.zeros_like(retrieved_condition),
        guidance_scale=1.5,
    )

    real_features = asset_bank.numpy()
    fake_features = (asset_bank + 0.05 * torch.randn_like(asset_bank)).numpy()
    fid = compute_fid(real_features, fake_features)

    return {
        "workflow": "multimodal_creative_assistant",
        "retrieved_asset_index": retrieved_index,
        "retrieval_margin": float(
            (retrieval_scores.max() - retrieval_scores.mean()).item()
        ),
        "unique_codes_used": int(torch.unique(code_indices).numel()),
        "guidance_shift_norm": float((guided_latent - quantized_latent).norm().item()),
        "fid_proxy": float(fid),
    }


def main() -> None:
    result = run_demo()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
