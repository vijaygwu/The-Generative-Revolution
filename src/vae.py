import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Autoencoder for binarized MNIST-like images."""

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        if x_flat.size(1) != self.input_dim:
            raise ValueError(
                f"expected flattened input_dim={self.input_dim}, got {x_flat.size(1)}"
            )
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def binarize_observations(x, threshold=0.5):
    """Threshold intensities so BCE matches a Bernoulli observation model."""
    return (x >= threshold).to(dtype=x.dtype)


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    x_flat = x.view(x.size(0), -1)
    if x_recon.shape == x.shape:
        recon_target = x
    elif x_recon.shape == x_flat.shape:
        recon_target = x_flat
    else:
        raise ValueError(
            f"x_recon shape {tuple(x_recon.shape)} must match input shape "
            f"{tuple(x.shape)} or flattened input shape {tuple(x_flat.shape)}"
        )
    recon_loss = F.binary_cross_entropy(x_recon, recon_target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_vae(model, dataloader, optimizer, epochs, device, beta=1.0):
    requested_device = torch.device(device)
    model_device = next(model.parameters()).device
    same_type = model_device.type == requested_device.type
    same_index = (
        requested_device.index is None
        or model_device.index == requested_device.index
    )
    if not (same_type and same_index):
        raise ValueError(
            "move the model to the target device before creating the optimizer; "
            f"model is on {model_device}, requested device is {requested_device}"
        )
    target_device = model_device
    model.train()

    def extract_inputs(batch):
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            data = extract_inputs(batch).to(target_device)
            data = binarize_observations(data)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(data)
            loss, recon, kl = vae_loss(x_recon, data, mu, logvar, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")


class ConvVAE(nn.Module):
    """Convolutional VAE for 28x28 binarized images."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
