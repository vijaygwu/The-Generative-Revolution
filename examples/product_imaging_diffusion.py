"""Toy retail product-imaging smoke test built on the diffusion companion code.

This example intentionally does not train a useful model. It runs a single
conditioned loss pass plus a short DDIM sampling loop so readers can verify the
conditioning and guidance interfaces without external data or long runtimes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.advanced_generative import classifier_free_guidance
from src.diffusion import DDPM, SimpleUNet


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _conditioned_loss(
    ddpm: DDPM,
    x0: torch.Tensor,
    condition: torch.Tensor,
    null_condition: torch.Tensor,
    drop_probability: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train with occasional null-conditioning to mimic classifier-free guidance."""
    x0 = x0.to(ddpm.device)
    condition = condition.to(ddpm.device)
    null_condition = null_condition.to(ddpm.device)

    t = torch.randint(0, ddpm.T, (x0.shape[0],), device=ddpm.device)
    x_t, noise = ddpm.forward_diffusion(x0, t)
    drop_mask = torch.rand(x0.shape[0], device=ddpm.device) < drop_probability
    if drop_mask.numel() > 1:
        if not bool(drop_mask.any()):
            drop_mask[0] = True
        elif bool(drop_mask.all()):
            drop_mask[0] = False
    train_condition = torch.where(drop_mask[:, None], null_condition, condition)
    noise_pred = ddpm.model(x_t, t, train_condition)
    loss = torch.mean((noise_pred - noise) ** 2)
    return loss, x_t, t, drop_mask


@torch.no_grad()
def _sample_guided_ddim(
    ddpm: DDPM,
    condition: torch.Tensor,
    null_condition: torch.Tensor,
    *,
    steps: int = 6,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """Sample with DDIM updates and classifier-free guidance."""
    condition = condition.to(ddpm.device)
    null_condition = null_condition.to(ddpm.device)
    shape = (condition.shape[0], 3, 32, 32)
    timesteps = ddpm._ddim_timesteps(steps)
    x = torch.randn(shape, device=ddpm.device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((shape[0],), t, device=ddpm.device, dtype=torch.long)
        noise_pred = classifier_free_guidance(
            ddpm.model,
            x,
            t_batch,
            condition=condition,
            null_condition=null_condition,
            guidance_scale=guidance_scale,
        )

        alpha_bar_t = ddpm.alpha_bars[t]
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = ddpm.alpha_bars[t_prev]
            x = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + torch.sqrt(1 - alpha_bar_prev) * noise_pred
            )
        else:
            x = x0_pred

    return x


def run_demo(seed: int = 0) -> dict[str, object]:
    """Run an untrained conditioned diffusion smoke test for the retail workflow."""
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Each row is a tiny attribute vector: background, lighting, crop, style.
    conditions = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.8],
            [0.7, 0.3, 0.9, 0.1],
            [0.4, 0.6, 0.1, 0.9],
            [0.9, 0.2, 0.5, 0.4],
        ],
        dtype=torch.float32,
        device=device,
    )
    null_condition = torch.zeros_like(conditions)

    backbone = SimpleUNet(in_ch=3, out_ch=3, ch=8, time_dim=32, cond_dim=4).to(device)
    ddpm = DDPM(backbone, T=12, device=device)

    catalog_batch = torch.rand(4, 3, 32, 32, device=device) * 2 - 1
    loss, x_t, t, drop_mask = _conditioned_loss(
        ddpm,
        catalog_batch,
        conditions,
        null_condition,
    )
    loss_value = float(loss.item())

    conditional_noise = backbone(x_t, t, conditions)
    unconditional_noise = backbone(x_t, t, null_condition)
    conditioning_hook_gap = float(
        (conditional_noise - unconditional_noise).abs().mean().item()
    )

    guidance_scale = 2.0
    sampler_steps = 6
    sampler_output = _sample_guided_ddim(
        ddpm,
        condition=conditions[:2],
        null_condition=null_condition[:2],
        steps=sampler_steps,
        guidance_scale=guidance_scale,
    )
    guided_prediction = classifier_free_guidance(
        backbone,
        x_t[:2],
        t[:2],
        condition=conditions[:2],
        null_condition=null_condition[:2],
        guidance_scale=guidance_scale,
    )

    return {
        "workflow": "retail_product_imaging_smoke_test",
        "train_loss": loss_value,
        "sampler_mode": "classifier_free_guided_ddim",
        "conditioning_hook_gap": conditioning_hook_gap,
        "conditioning_dropout_rate": float(drop_mask.float().mean().item()),
        "guidance_scale": guidance_scale,
        "sampler_steps": sampler_steps,
        "sample_stats": _tensor_stats(sampler_output),
        "guidance_hook_stats": _tensor_stats(guided_prediction),
    }


def main() -> None:
    result = run_demo()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
