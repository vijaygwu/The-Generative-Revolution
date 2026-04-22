from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.vae import ConvVAE, VAE, train_vae, vae_loss


def test_vae_forward_shapes() -> None:
    torch.manual_seed(0)
    model = VAE()
    x = torch.rand(4, 1, 28, 28)

    x_recon, mu, logvar = model(x)

    assert x_recon.shape == (4, 784)
    assert mu.shape == (4, 20)
    assert logvar.shape == (4, 20)
    assert torch.isfinite(x_recon).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()


def test_vae_sample_shapes() -> None:
    torch.manual_seed(0)
    model = VAE()
    samples = model.sample(3)

    assert samples.shape == (3, 784)
    assert torch.isfinite(samples).all()


def test_vae_loss_outputs() -> None:
    torch.manual_seed(0)
    model = VAE()
    x = torch.rand(4, 1, 28, 28)

    x_recon, mu, logvar = model(x)
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

    assert total_loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kl_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)
    assert total_loss.item() >= recon_loss.item()


def test_vae_supports_nondefault_input_dim() -> None:
    torch.manual_seed(0)
    model = VAE(input_dim=16, hidden_dim=8, latent_dim=4)
    x = torch.rand(3, 1, 4, 4)

    x_recon, mu, logvar = model(x)
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

    assert x_recon.shape == (3, 16)
    assert mu.shape == (3, 4)
    assert logvar.shape == (3, 4)
    assert torch.isfinite(total_loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)


def test_conv_vae_forward_shapes() -> None:
    torch.manual_seed(0)
    model = ConvVAE()
    x = torch.rand(2, 1, 28, 28)

    x_recon, mu, logvar = model(x)

    assert x_recon.shape == (2, 1, 28, 28)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)
    assert torch.isfinite(x_recon).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()


def test_train_vae_accepts_unlabeled_and_labeled_batches() -> None:
    torch.manual_seed(0)
    model = VAE(input_dim=16, hidden_dim=8, latent_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(6, 1, 4, 4)

    unlabeled = DataLoader(TensorDataset(x), batch_size=3, shuffle=False)
    labeled = DataLoader(TensorDataset(x, torch.zeros(6)), batch_size=3, shuffle=False)

    train_vae(model, unlabeled, optimizer, epochs=1, device="cpu")
    train_vae(model, labeled, optimizer, epochs=1, device="cpu")


def main() -> None:
    print("Running VAE tests...")
    test_vae_forward_shapes()
    test_vae_sample_shapes()
    test_vae_loss_outputs()
    test_vae_supports_nondefault_input_dim()
    test_conv_vae_forward_shapes()
    test_train_vae_accepts_unlabeled_and_labeled_batches()
    print("All VAE tests passed.")


if __name__ == "__main__":
    main()
