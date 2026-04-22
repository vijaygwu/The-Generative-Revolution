from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.gan import Discriminator, Generator, WGANCritic, gradient_penalty, train_gan


def test_generator_output_shape() -> None:
    torch.manual_seed(0)
    model = Generator()
    z = torch.randn(4, 100)

    imgs = model(z)

    assert imgs.shape == (4, 1, 28, 28)
    assert torch.isfinite(imgs).all()


def test_discriminator_output_shape() -> None:
    torch.manual_seed(0)
    disc = Discriminator()
    imgs = torch.randn(4, 1, 28, 28)

    scores = disc(imgs)

    assert scores.shape == (4, 1)
    assert torch.isfinite(scores).all()


def test_wgan_critic_output_shape() -> None:
    torch.manual_seed(0)
    critic = WGANCritic()
    imgs = torch.randn(4, 1, 28, 28)

    scores = critic(imgs)

    assert scores.shape == (4, 1)
    assert torch.isfinite(scores).all()


def test_gradient_penalty_is_finite_and_nonnegative() -> None:
    torch.manual_seed(0)
    critic = WGANCritic()
    generator = Generator()

    z = torch.randn(4, 100)
    fake = generator(z).detach()
    real = torch.randn(4, 1, 28, 28)

    penalty = gradient_penalty(critic, real, fake)

    assert penalty.ndim == 0
    assert torch.isfinite(penalty)
    assert penalty.item() >= 0.0


def test_gradient_penalty_does_not_backprop_into_generator() -> None:
    torch.manual_seed(0)
    critic = WGANCritic()
    generator = Generator()

    z = torch.randn(4, 100)
    fake = generator(z)
    real = torch.randn(4, 1, 28, 28)

    penalty = gradient_penalty(critic, real, fake)
    penalty.backward()

    assert all(param.grad is None for param in generator.parameters())


def test_gradient_penalty_backpropagates_through_critic() -> None:
    torch.manual_seed(0)
    critic = WGANCritic()
    generator = Generator()

    z = torch.randn(4, 100)
    fake = generator(z)
    real = torch.randn(4, 1, 28, 28)

    penalty = gradient_penalty(critic, real, fake)
    penalty.backward()

    assert any(param.grad is not None for param in critic.parameters())


def test_train_gan_smoke() -> None:
    torch.manual_seed(0)
    generator = Generator()
    discriminator = Discriminator()

    x = torch.rand(8, 1, 28, 28)
    y = torch.zeros(8)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    train_gan(generator, discriminator, dataloader, epochs=1)


def test_train_gan_respects_generator_latent_dim() -> None:
    torch.manual_seed(0)
    generator = Generator(latent_dim=16)
    discriminator = Discriminator()

    x = torch.rand(8, 1, 28, 28)
    y = torch.zeros(8)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    train_gan(generator, discriminator, dataloader, epochs=1)


def test_train_gan_accepts_unlabeled_batches() -> None:
    torch.manual_seed(0)
    generator = Generator(latent_dim=16)
    discriminator = Discriminator()

    x = torch.rand(8, 1, 28, 28)
    dataloader = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)

    train_gan(generator, discriminator, dataloader, epochs=1)


def main() -> None:
    print("Running GAN tests...")
    test_generator_output_shape()
    test_discriminator_output_shape()
    test_wgan_critic_output_shape()
    test_gradient_penalty_is_finite_and_nonnegative()
    test_gradient_penalty_does_not_backprop_into_generator()
    test_gradient_penalty_backpropagates_through_critic()
    test_train_gan_smoke()
    test_train_gan_respects_generator_latent_dim()
    test_train_gan_accepts_unlabeled_batches()
    print("All GAN tests passed.")


if __name__ == "__main__":
    main()
