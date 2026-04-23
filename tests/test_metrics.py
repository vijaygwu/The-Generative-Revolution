from __future__ import annotations

import os
import sys

import numpy as np
from scipy import linalg

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.metrics import compute_fid
from the_generative_revolution.metrics import compute_fid as package_compute_fid


def test_compute_fid_returns_float() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(size=(16, 8))
    fake = rng.normal(size=(16, 8))

    fid = compute_fid(real, fake)

    assert isinstance(fid, float)
    assert np.isfinite(fid)


def test_compute_fid_identical_features_near_zero() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(size=(16, 8))

    fid = compute_fid(real, real.copy())

    assert np.isfinite(fid)
    assert abs(fid) < 1e-6


def test_compute_fid_identical_low_rank_features_clamps_to_zero() -> None:
    real = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    for helper in (compute_fid, package_compute_fid):
        fid = helper(real, real.copy())
        assert fid == 0.0


def test_compute_fid_rejects_rank_mismatch() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(size=(16, 8))
    fake = rng.normal(size=(16, 7))

    try:
        compute_fid(real, fake)
        assert False, "Expected ValueError for mismatched feature dimension"
    except ValueError as exc:
        assert "feature dimensions must match" in str(exc)


def test_compute_fid_rejects_too_few_samples() -> None:
    real = np.array([[0.1, 0.2, 0.3]])
    fake = np.array([[0.2, 0.1, 0.4]])

    try:
        compute_fid(real, fake)
        assert False, "Expected ValueError for too few samples"
    except ValueError as exc:
        assert "at least two samples" in str(exc)


def test_compute_fid_supports_single_feature_inputs() -> None:
    real = np.array([[0.0], [1.0], [2.0], [3.0]])
    fake = np.array([[0.5], [1.5], [2.5], [3.5]])

    fid = compute_fid(real, fake)

    assert isinstance(fid, float)
    assert np.isfinite(fid)


def test_compute_fid_rejects_materially_complex_covariance_root() -> None:
    real = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    fake = np.array([[1.0, 0.0], [0.0, -1.0], [0.5, -0.5]])

    original_sqrtm = linalg.sqrtm

    def fake_sqrtm(_):
        return np.array([[1.0, 1e-3j], [0.0, 1.0]])

    linalg.sqrtm = fake_sqrtm
    try:
        try:
            compute_fid(real, fake)
            assert False, "Expected ValueError for non-negligible imaginary component"
        except ValueError as exc:
            assert "imaginary component" in str(exc)
    finally:
        linalg.sqrtm = original_sqrtm


def main() -> None:
    print("Running metric tests...")
    test_compute_fid_returns_float()
    test_compute_fid_identical_features_near_zero()
    test_compute_fid_identical_low_rank_features_clamps_to_zero()
    test_compute_fid_rejects_rank_mismatch()
    test_compute_fid_rejects_too_few_samples()
    test_compute_fid_supports_single_feature_inputs()
    test_compute_fid_rejects_materially_complex_covariance_root()
    print("All metric tests passed.")


if __name__ == "__main__":
    main()
