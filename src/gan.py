import math

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Simple generator for MNIST-like images."""

    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        flat_dim = math.prod(img_shape)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, flat_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    """Simple discriminator for MNIST-like images that returns logits."""

    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        flat_dim = math.prod(img_shape)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        return self.model(img)


def train_gan(generator, discriminator, dataloader, epochs=100, device=None,
              d_steps=1):
    if device is None:
        device = next(generator.parameters()).device

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.train()
    discriminator.train()

    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    latent_dim = generator.latent_dim

    def extract_inputs(batch):
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    for epoch in range(epochs):
        for batch in dataloader:
            real_imgs = extract_inputs(batch).to(device)
            batch_size = real_imgs.size(0)

            # Match standard [0, 1] tensors to the generator's tanh output in [-1, 1].
            if real_imgs.min().item() >= 0 and real_imgs.max().item() <= 1:
                real_imgs = 2 * real_imgs - 1

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Discriminator: d_steps updates per generator step (Algorithm 2's k)
            for _ in range(d_steps):
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_imgs = generator(z).detach()

                d_loss_real = criterion(discriminator(real_imgs), real_labels)
                d_loss_fake = criterion(discriminator(fake_imgs), fake_labels)
                d_loss = d_loss_real + d_loss_fake

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            for param in discriminator.parameters():
                param.requires_grad_(False)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            g_loss = criterion(discriminator(fake_imgs), real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            for param in discriminator.parameters():
                param.requires_grad_(True)


class WGANCritic(nn.Module):
    """Critic (discriminator) for WGAN-GP."""

    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        flat_dim = math.prod(img_shape)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        return self.model(img)


def gradient_penalty(critic, real, fake):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, *([1] * (real.dim() - 1)), device=real.device)
    interpolated = epsilon * real + (1 - epsilon) * fake.detach()
    interpolated.requires_grad_(True)

    critic_interpolated = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
