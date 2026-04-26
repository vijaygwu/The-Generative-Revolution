import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if self.dim < 2:
            raise ValueError("SinusoidalPosEmbed requires dim >= 2")
        device = t.device
        half_dim = self.dim // 2
        if half_dim == 0:
            raise ValueError("SinusoidalPosEmbed requires at least one sine/cosine pair")
        if half_dim == 1:
            emb = torch.ones(1, device=device)
        else:
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] != self.dim:
            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1))
            else:
                emb = emb[:, : self.dim]
        return emb


def _group_norm(channel_count, max_groups=8):
    for groups in range(min(max_groups, channel_count), 0, -1):
        if channel_count % groups == 0:
            return nn.GroupNorm(groups, channel_count)
    raise ValueError(f"Could not find a valid GroupNorm divisor for {channel_count} channels")


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = _group_norm(in_ch)
        self.norm2 = _group_norm(out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """Simplified U-Net for diffusion models."""

    def __init__(self, in_ch=3, out_ch=3, ch=64, time_dim=256, cond_dim=None):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(ch),
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, time_dim) if cond_dim is not None else None

        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch * 2, time_dim)
        self.down2 = ResBlock(ch * 2, ch * 4, time_dim)
        self.pool = nn.MaxPool2d(2)

        self.bot = ResBlock(ch * 4, ch * 4, time_dim)

        # Runtime-safe widths for the actual skip concatenations.
        self.up2 = ResBlock(ch * 6, ch * 2, time_dim)
        self.up1 = ResBlock(ch * 3, ch, time_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_out = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x, t, cond=None):
        if x.ndim != 4:
            raise ValueError("SimpleUNet expects input shaped (batch, channels, height, width)")
        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError(
                "SimpleUNet expects height and width divisible by 4 "
                "because it uses two 2x downsampling stages"
            )

        t_emb = self.time_embed(t)
        if self.cond_proj is not None:
            if cond is None:
                cond = torch.zeros(
                    x.size(0),
                    self.cond_proj.in_features,
                    device=x.device,
                    dtype=t_emb.dtype,
                )
            else:
                cond = cond.to(x.device, dtype=t_emb.dtype)
            t_emb = t_emb + self.cond_proj(cond)

        x1 = self.conv_in(x)
        x2 = self.pool(self.down1(x1, t_emb))
        x3 = self.pool(self.down2(x2, t_emb))

        x3 = self.bot(x3, t_emb)

        x = self.upsample(x3)
        x = self.up2(torch.cat([x, x2], dim=1), t_emb)
        x = self.upsample(x)
        x = self.up1(torch.cat([x, x1], dim=1), t_emb)

        return self.conv_out(x)


class DDPM:
    """Denoising Diffusion Probabilistic Model."""

    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device=None):
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.T = T

        self.betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    def loss(self, x0):
        x0 = x0.to(self.device)
        # Python indices 0..T-1 map to mathematical diffusion steps 1..T.
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        xt, noise = self.forward_diffusion(x0, t)
        noise_pred = self.model(xt, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            alpha_bar_prev = (
                self.alpha_bars[t - 1]
                if t > 0
                else torch.tensor(1.0, device=self.device, dtype=self.alpha_bars.dtype)
            )

            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
            )

            # When t == 0, this is the final reverse step from x_1 back to x_0.
            if t > 0:
                noise = torch.randn_like(x)
                beta_tilde = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                sigma = torch.sqrt(beta_tilde)
                x = mean + sigma * noise
            else:
                x = mean

        return x

    def _ddim_timesteps(self, steps):
        # Returned indices live in 0..T-1 and therefore map to math timesteps 1..T.
        # The terminal x_0 update is handled separately with alpha_bar_prev = 1.
        steps = max(1, min(int(steps), self.T))
        timesteps = torch.linspace(
            self.T - 1, 0, steps, device=self.betas.device
        ).round().long()
        deduped = []
        for t in timesteps.tolist():
            if not deduped or t != deduped[-1]:
                deduped.append(int(t))
        if deduped[0] != self.T - 1:
            deduped.insert(0, self.T - 1)
        if deduped[-1] != 0:
            deduped.append(0)
        return deduped

    @torch.no_grad()
    def sample_ddim(self, shape, steps=50, initial_noise=None):
        """Generate samples with the eta=0 DDIM update."""
        timesteps = self._ddim_timesteps(steps)
        shape = tuple(shape)
        if initial_noise is None:
            x = torch.randn(shape, device=self.device)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} does not match requested shape {shape}"
                )
            x = initial_noise.to(self.device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)

            alpha_bar_t = self.alpha_bars[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(
                alpha_bar_t
            )

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alpha_bars[t_prev]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            x = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + torch.sqrt(1 - alpha_bar_prev) * noise_pred
            )

        return x


def train_diffusion(model, dataloader, epochs=100, lr=1e-4, device=None):
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.train()
    ddpm = DDPM(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def extract_inputs(batch):
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = extract_inputs(batch).to(device)
            optimizer.zero_grad()
            loss = ddpm.loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")
