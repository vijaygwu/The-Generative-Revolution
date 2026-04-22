import numpy as np
from scipy import linalg


def _stable_sqrtm(cov_prod, sigma_r, sigma_f, eps):
    covmean = linalg.sqrtm(cov_prod)

    if np.isfinite(covmean).all():
        return covmean

    offset = np.eye(sigma_r.shape[0], dtype=sigma_r.dtype) * eps
    covmean = linalg.sqrtm((sigma_r + offset) @ (sigma_f + offset))
    if not np.isfinite(covmean).all():
        raise ValueError("FID covariance square root is not finite even after jitter")
    return covmean


def compute_fid(real_features, fake_features, eps=1e-6):
    """Compute Fréchet Inception Distance between two feature sets."""

    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("real_features and fake_features must be 2D arrays")
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("feature dimensions must match")
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError("need at least two samples per feature set")

    mu_r = real_features.mean(axis=0)
    mu_f = fake_features.mean(axis=0)

    sigma_r = np.atleast_2d(np.cov(real_features, rowvar=False))
    sigma_f = np.atleast_2d(np.cov(fake_features, rowvar=False))

    diff = mu_r - mu_f
    mean_term = diff @ diff

    cov_prod = sigma_r @ sigma_f
    covmean = _stable_sqrtm(cov_prod, sigma_r, sigma_f, eps)

    if np.iscomplexobj(covmean):
        max_imag = float(np.max(np.abs(covmean.imag), initial=0.0))
        if max_imag > 1e-6:
            raise ValueError(
                "FID covariance square root has a non-negligible imaginary component"
            )
        covmean = covmean.real

    cov_term = np.trace(sigma_r + sigma_f - 2 * covmean)
    fid = float(mean_term + cov_term)
    if fid < 0 and abs(fid) < 1e-9:
        fid = 0.0
    if fid < 0:
        raise ValueError("FID is negative beyond floating-point tolerance")
    return fid
