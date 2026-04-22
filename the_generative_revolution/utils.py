"""Small shared helpers for the Book 3 companion repo."""

from __future__ import annotations


def count_parameters(model) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
