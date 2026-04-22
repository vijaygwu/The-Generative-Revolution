from __future__ import annotations

import os
import sys

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.flows import AffineCouplingLayer, RealNVP


def test_affine_coupling_forward_reverse() -> None:
    torch.manual_seed(0)
    layer = AffineCouplingLayer(dim=6)
    z = torch.randn(5, 6)

    x, log_det = layer(z, reverse=False)
    z_recovered, inv_log_det = layer(x, reverse=True)

    assert x.shape == (5, 6)
    assert log_det.shape == (5,)
    assert z_recovered.shape == (5, 6)
    assert inv_log_det.shape == (5,)
    assert torch.isfinite(x).all()
    assert torch.isfinite(log_det).all()
    assert torch.isfinite(z_recovered).all()
    assert torch.isfinite(inv_log_det).all()
    assert torch.allclose(z_recovered, z, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        log_det + inv_log_det,
        torch.zeros_like(log_det),
        atol=1e-5,
        rtol=1e-5,
    )


def test_realnvp_log_prob_shape() -> None:
    torch.manual_seed(0)
    model = RealNVP(dim=6)
    x = torch.randn(5, 6)

    log_prob = model.log_prob(x)

    assert log_prob.shape == (5,)
    assert torch.isfinite(log_prob).all()


def test_realnvp_sample_shape() -> None:
    torch.manual_seed(0)
    model = RealNVP(dim=6)

    samples = model.sample(7)

    assert samples.shape == (7, 6)
    assert torch.isfinite(samples).all()


def test_realnvp_alternates_masks() -> None:
    model = RealNVP(dim=6, num_layers=4)

    first_mask = model.layers[0].mask
    second_mask = model.layers[1].mask

    assert torch.equal(first_mask, 1 - second_mask)
    assert not torch.equal(first_mask, second_mask)


def main() -> None:
    print("Running flow tests...")
    test_affine_coupling_forward_reverse()
    test_realnvp_log_prob_shape()
    test_realnvp_sample_shape()
    test_realnvp_alternates_masks()
    print("All flow tests passed.")


if __name__ == "__main__":
    main()
