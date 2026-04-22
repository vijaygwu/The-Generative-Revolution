from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.diffusion import DDPM, ResBlock, SimpleUNet, SinusoidalPosEmbed, train_diffusion


def test_sinusoidal_pos_embed_shape() -> None:
    torch.manual_seed(0)
    embed = SinusoidalPosEmbed(dim=64)
    t = torch.arange(4, dtype=torch.float32)

    out = embed(t)

    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_sinusoidal_pos_embed_odd_dim_shape() -> None:
    torch.manual_seed(0)
    embed = SinusoidalPosEmbed(dim=63)
    t = torch.arange(4, dtype=torch.float32)

    out = embed(t)

    assert out.shape == (4, 63)
    assert torch.isfinite(out).all()


def test_resblock_shape() -> None:
    torch.manual_seed(0)
    block = ResBlock(8, 16, 32)
    x = torch.randn(2, 8, 16, 16)
    t_emb = torch.randn(2, 32)

    out = block(x, t_emb)

    assert out.shape == (2, 16, 16, 16)
    assert torch.isfinite(out).all()


def test_simple_unet_shape() -> None:
    torch.manual_seed(0)
    model = SimpleUNet(in_ch=3, out_ch=3, ch=8, time_dim=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,), dtype=torch.long)

    out = model(x, t)

    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_simple_unet_small_channel_count() -> None:
    torch.manual_seed(0)
    model = SimpleUNet(in_ch=3, out_ch=3, ch=4, time_dim=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,), dtype=torch.long)

    out = model(x, t)

    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_ddpm_core_paths() -> None:
    torch.manual_seed(0)
    model = SimpleUNet(in_ch=3, out_ch=3, ch=8, time_dim=32)
    ddpm = DDPM(model, T=10, device="cpu")
    x0 = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,), dtype=torch.long)

    xt, noise = ddpm.forward_diffusion(x0, t)
    loss = ddpm.loss(x0)
    samples = ddpm.sample((2, 3, 32, 32))
    ddim_samples = ddpm.sample_ddim((2, 3, 32, 32), steps=5)

    assert xt.shape == x0.shape
    assert noise.shape == x0.shape
    assert torch.isfinite(xt).all()
    assert torch.isfinite(noise).all()

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    assert samples.shape == (2, 3, 32, 32)
    assert ddim_samples.shape == (2, 3, 32, 32)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(ddim_samples).all()


def test_ddim_timestep_schedule_includes_endpoints() -> None:
    model = SimpleUNet(in_ch=3, out_ch=3, ch=8, time_dim=32)
    ddpm = DDPM(model, T=10, device="cpu")

    timesteps = ddpm._ddim_timesteps(steps=6)
    single_step_timesteps = ddpm._ddim_timesteps(steps=1)

    assert timesteps[0] == 9
    assert timesteps[-1] == 0
    assert single_step_timesteps == [9, 0]


def test_ddim_reuses_fixed_initial_noise_deterministically() -> None:
    torch.manual_seed(0)
    model = SimpleUNet(in_ch=3, out_ch=3, ch=8, time_dim=32)
    ddpm = DDPM(model, T=10, device="cpu")
    initial_noise = torch.randn(2, 3, 32, 32)

    first = ddpm.sample_ddim((2, 3, 32, 32), steps=5, initial_noise=initial_noise)
    second = ddpm.sample_ddim((2, 3, 32, 32), steps=5, initial_noise=initial_noise)

    assert torch.equal(first, second)


def test_train_diffusion_accepts_unlabeled_and_labeled_batches() -> None:
    torch.manual_seed(0)
    model = SimpleUNet(in_ch=1, out_ch=1, ch=8, time_dim=32)
    x = torch.randn(4, 1, 8, 8)

    unlabeled = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    labeled = DataLoader(TensorDataset(x, torch.zeros(4)), batch_size=2, shuffle=False)

    train_diffusion(model, unlabeled, epochs=1, device="cpu")
    train_diffusion(model, labeled, epochs=1, device="cpu")


def main() -> None:
    print("Running diffusion tests...")
    test_sinusoidal_pos_embed_shape()
    test_sinusoidal_pos_embed_odd_dim_shape()
    test_resblock_shape()
    test_simple_unet_shape()
    test_simple_unet_small_channel_count()
    test_ddpm_core_paths()
    test_ddim_timestep_schedule_includes_endpoints()
    test_ddim_reuses_fixed_initial_noise_deterministically()
    test_train_diffusion_accepts_unlabeled_and_labeled_batches()
    print("All diffusion tests passed.")


if __name__ == "__main__":
    main()
