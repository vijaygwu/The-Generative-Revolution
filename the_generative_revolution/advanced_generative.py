from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize(z: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BCHW latent tensor against a [K, C] codebook."""
    if z.ndim != 4:
        raise ValueError("z must have shape [B, C, H, W]")
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [K, C]")
    if z.shape[1] != codebook.shape[1]:
        raise ValueError("latent channel count must match codebook embedding size")

    batch, channels, height, width = z.shape
    z_flat = z.permute(0, 2, 3, 1).reshape(-1, channels)

    distances = (
        z_flat.pow(2).sum(dim=1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * z_flat @ codebook.T
    )
    indices = distances.argmin(dim=1)
    z_q = codebook[indices].view(batch, height, width, channels).permute(0, 3, 1, 2)
    z_q = z + (z_q - z).detach()
    return z_q, indices.view(batch, height, width)


def classifier_free_guidance(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    null_condition: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Blend conditional and unconditional predictions with CFG."""
    noise_cond = model(x_t, t, condition)
    noise_uncond = model(x_t, t, null_condition)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


class MaskedConv2d(nn.Conv2d):
    """Spatial-only masked convolution with zero padding only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        mask_type: str = "A",
        **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")

        k_h, k_w = self.weight.shape[-2:]
        if k_h % 2 == 0 or k_w % 2 == 0:
            raise ValueError("MaskedConv2d requires odd kernel dimensions")
        if self.padding_mode != "zeros":
            raise ValueError(
                "MaskedConv2d only supports padding_mode='zeros'; "
                "non-zero padding modes need explicit pre-padding."
            )

        mask = torch.ones_like(self.weight)
        mask[:, :, k_h // 2, k_w // 2 + 1 :] = 0
        mask[:, :, k_h // 2 + 1 :, :] = 0
        if mask_type == "A":
            mask[:, :, k_h // 2, k_w // 2] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.conv2d(
            x,
            masked_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class TinyConditionedDenoiser(nn.Module):
    """Minimal denoiser used for companion smoke tests and manuscript snippets."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, channels)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        del t
        bias = self.cond_proj(cond).view(x_t.size(0), x_t.size(1), 1, 1)
        return x_t + bias
