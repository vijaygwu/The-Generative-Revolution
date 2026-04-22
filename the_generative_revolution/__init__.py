"""Package-scoped exports for the Book 3 companion."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

_LAZY_MODULES = {
    "MaskedConv2d": "advanced_generative",
    "TinyConditionedDenoiser": "advanced_generative",
    "classifier_free_guidance": "advanced_generative",
    "quantize": "advanced_generative",
    "VAE": "vae",
    "ConvVAE": "vae",
    "vae_loss": "vae",
    "train_vae": "vae",
    "Generator": "gan",
    "Discriminator": "gan",
    "WGANCritic": "gan",
    "gradient_penalty": "gan",
    "train_gan": "gan",
    "AffineCouplingLayer": "flows",
    "RealNVP": "flows",
    "SinusoidalPosEmbed": "diffusion",
    "ResBlock": "diffusion",
    "SimpleUNet": "diffusion",
    "DDPM": "diffusion",
    "train_diffusion": "diffusion",
    "compute_fid": "metrics",
}


def __getattr__(name: str):
    """Lazy import companion symbols at the package boundary."""
    if name in _LAZY_MODULES:
        module_name = _LAZY_MODULES[name]
        import importlib

        module = importlib.import_module(f".{module_name}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .advanced_generative import (
        MaskedConv2d,
        TinyConditionedDenoiser,
        classifier_free_guidance,
        quantize,
    )
    from .diffusion import DDPM, ResBlock, SimpleUNet, SinusoidalPosEmbed, train_diffusion
    from .flows import AffineCouplingLayer, RealNVP
    from .gan import Discriminator, Generator, WGANCritic, gradient_penalty, train_gan
    from .metrics import compute_fid
    from .vae import ConvVAE, VAE, train_vae, vae_loss


__all__ = [
    "AffineCouplingLayer",
    "MaskedConv2d",
    "classifier_free_guidance",
    "compute_fid",
    "ConvVAE",
    "DDPM",
    "Discriminator",
    "Generator",
    "gradient_penalty",
    "RealNVP",
    "ResBlock",
    "SimpleUNet",
    "SinusoidalPosEmbed",
    "TinyConditionedDenoiser",
    "train_diffusion",
    "train_gan",
    "train_vae",
    "quantize",
    "VAE",
    "vae_loss",
    "WGANCritic",
    "__version__",
]
