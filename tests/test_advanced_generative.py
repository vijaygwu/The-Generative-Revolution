from __future__ import annotations

import os
import sys

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.advanced_generative import (
    MaskedConv2d,
    TinyConditionedDenoiser,
    classifier_free_guidance,
    quantize,
)


def test_quantize_returns_expected_shapes() -> None:
    torch.manual_seed(0)
    z = torch.randn(2, 4, 3, 3)
    codebook = torch.randn(8, 4)

    z_q, indices = quantize(z, codebook)

    assert z_q.shape == z.shape
    assert indices.shape == (2, 3, 3)
    assert torch.isfinite(z_q).all()


def test_classifier_free_guidance_matches_formula() -> None:
    torch.manual_seed(0)
    model = TinyConditionedDenoiser(channels=4, cond_dim=3)
    x_t = torch.randn(2, 4, 8, 8)
    t = torch.tensor([5, 5])
    condition = torch.tensor([[1.0, 0.0, 0.5], [0.2, 0.8, 0.1]])
    null_condition = torch.zeros_like(condition)
    guidance_scale = 4.0

    guided = classifier_free_guidance(
        model, x_t, t, condition, null_condition, guidance_scale
    )
    noise_cond = model(x_t, t, condition)
    noise_uncond = model(x_t, t, null_condition)

    assert torch.allclose(
        guided,
        noise_uncond + guidance_scale * (noise_cond - noise_uncond),
    )


def test_masked_conv2d_preserves_spatial_shape() -> None:
    torch.manual_seed(0)
    layer = MaskedConv2d(4, 6, kernel_size=3, padding=1, mask_type="A")
    x = torch.randn(2, 4, 8, 8)

    y = layer(x)

    assert y.shape == (2, 6, 8, 8)


def test_masked_conv2d_rejects_nonzero_padding_modes() -> None:
    try:
        MaskedConv2d(
            1,
            1,
            kernel_size=3,
            padding=1,
            mask_type="A",
            padding_mode="reflect",
        )
    except ValueError as exc:
        assert "padding_mode='zeros'" in str(exc)
    else:
        raise AssertionError("MaskedConv2d should reject non-zero padding modes")


def test_masked_conv2d_type_a_blocks_current_and_future_pixels() -> None:
    layer = MaskedConv2d(1, 1, kernel_size=3, padding=1, mask_type="A", bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)

    base = torch.zeros(1, 1, 5, 5)
    base[0, 0, 2, 1] = 1.0

    leaked = base.clone()
    leaked[0, 0, 2, 2] = 5.0
    leaked[0, 0, 2, 3] = 7.0
    leaked[0, 0, 3, 2] = 11.0

    past_shifted = base.clone()
    past_shifted[0, 0, 1, 2] = 13.0

    base_center = layer(base)[0, 0, 2, 2]
    leaked_center = layer(leaked)[0, 0, 2, 2]
    past_center = layer(past_shifted)[0, 0, 2, 2]

    assert torch.allclose(base_center, leaked_center)
    assert not torch.allclose(base_center, past_center)


def test_masked_conv2d_type_b_keeps_current_but_blocks_future_pixels() -> None:
    layer = MaskedConv2d(1, 1, kernel_size=3, padding=1, mask_type="B", bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)

    base = torch.zeros(1, 1, 5, 5)
    base[0, 0, 2, 2] = 1.0

    future_shifted = base.clone()
    future_shifted[0, 0, 2, 3] = 5.0
    future_shifted[0, 0, 3, 2] = 7.0

    current_shifted = base.clone()
    current_shifted[0, 0, 2, 2] = 9.0

    base_center = layer(base)[0, 0, 2, 2]
    future_center = layer(future_shifted)[0, 0, 2, 2]
    current_center = layer(current_shifted)[0, 0, 2, 2]

    assert torch.allclose(base_center, future_center)
    assert not torch.allclose(base_center, current_center)


def main() -> None:
    print("Running advanced generative tests...")
    test_quantize_returns_expected_shapes()
    test_classifier_free_guidance_matches_formula()
    test_masked_conv2d_preserves_spatial_shape()
    test_masked_conv2d_rejects_nonzero_padding_modes()
    test_masked_conv2d_type_a_blocks_current_and_future_pixels()
    test_masked_conv2d_type_b_keeps_current_but_blocks_future_pixels()
    print("All advanced generative tests passed.")


if __name__ == "__main__":
    main()
