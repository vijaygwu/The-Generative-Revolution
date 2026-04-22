"""Offline smoke tests for the Book 3 generative-model runtime stack."""

from __future__ import annotations

import importlib
import os
import sys

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.diffusion import DDPM, SimpleUNet, SinusoidalPosEmbed
from src.flows import RealNVP
from src.gan import Discriminator, Generator
from src.vae import VAE, vae_loss


def test_runtime_dependencies_import() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("torch", "numpy", "scipy"):
        module = importlib.import_module(module_name)
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def test_vae_runtime_path() -> None:
    torch.manual_seed(0)
    model = VAE()
    x = torch.rand(2, 1, 28, 28)

    x_recon, mu, logvar = model(x)
    loss, recon, kl = vae_loss(x_recon, x, mu, logvar)

    assert x_recon.shape == (2, 784)
    assert mu.shape == (2, 20)
    assert logvar.shape == (2, 20)
    assert loss.ndim == 0
    assert recon.ndim == 0
    assert kl.ndim == 0
    assert torch.isfinite(x_recon).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()
    assert torch.isfinite(loss)


def test_gan_runtime_path() -> None:
    torch.manual_seed(0)
    generator = Generator()
    discriminator = Discriminator()

    z = torch.randn(2, 100)
    fake = generator(z)
    scores = discriminator(fake)

    assert fake.shape == (2, 1, 28, 28)
    assert scores.shape == (2, 1)
    assert torch.isfinite(fake).all()
    assert torch.isfinite(scores).all()


def test_flow_runtime_path() -> None:
    torch.manual_seed(0)
    model = RealNVP(dim=6)
    x = torch.randn(4, 6)

    log_prob = model.log_prob(x)
    samples = model.sample(3)

    assert log_prob.shape == (4,)
    assert samples.shape == (3, 6)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(samples).all()


def test_diffusion_runtime_path() -> None:
    torch.manual_seed(0)

    embed = SinusoidalPosEmbed(32)
    t_embed = embed(torch.arange(2, dtype=torch.float32))
    assert t_embed.shape == (2, 32)
    assert torch.isfinite(t_embed).all()

    model = SimpleUNet(in_ch=3, out_ch=3, ch=8, time_dim=32)
    ddpm = DDPM(model, T=10, device="cpu")

    x = torch.randn(2, 3, 32, 32)
    loss = ddpm.loss(x)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def main() -> None:
    print("Running Book 3 runtime smoke tests...")
    versions = test_runtime_dependencies_import()
    print(
        "  - runtime imports succeeded for "
        + ", ".join(f"{name}={version}" for name, version in versions.items())
    )
    test_vae_runtime_path()
    print("  - VAE forward path and loss computation work")
    test_gan_runtime_path()
    print("  - GAN generator/discriminator forward paths work")
    test_flow_runtime_path()
    print("  - RealNVP log-probability and sampling work")
    test_diffusion_runtime_path()
    print("  - diffusion embedding, U-Net, and DDPM loss path work")
    print("\nAll Book 3 runtime smoke tests passed.")


if __name__ == "__main__":
    main()
