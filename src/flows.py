import math

import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP."""

    def __init__(self, dim, hidden_dim=256, flip_mask=False):
        super().__init__()
        self.dim = dim
        mask = torch.cat([torch.ones(dim // 2), torch.zeros(dim - dim // 2)])
        if flip_mask:
            mask = 1 - mask
        self.register_buffer(
            "mask",
            mask,
        )
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )

    def forward(self, z, reverse=False):
        z_masked = z * self.mask
        st = self.net(z_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            x = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
            return x, s.sum(dim=-1)
        else:
            z_out = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
            return z_out, -s.sum(dim=-1)


class RealNVP(nn.Module):
    """RealNVP normalizing flow."""

    def __init__(self, dim, num_layers=6, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [
                AffineCouplingLayer(dim, hidden_dim, flip_mask=(i % 2 == 1))
                for i in range(num_layers)
            ]
        )

    def log_prob(self, x):
        z, log_det = x, 0
        for layer in reversed(self.layers):
            z, ld = layer(z, reverse=True)
            log_det += ld
        log_pz = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, num_samples):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.dim, device=device)
        for layer in self.layers:
            z, _ = layer(z, reverse=False)
        return z
