"""
    distances/mmd.py — Method A: Maximum Mean Discrepancy.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ..data import ModelResponseData
from .base import validate_distance_matrix


class MMDDistance:
    """Maximum Mean Discrepancy distance between model response distributions.

    Works with unpaired data (no query_id required). For paired data, ignores
    pairing structure and treats each response independently.

    Parameters
    ----------
    kernel : str
        'rbf' (Gaussian) or 'linear'. Default 'rbf'.
    bandwidth : float or None
        RBF bandwidth (sigma). None = median heuristic.
    """

    def __init__(self, kernel='rbf', bandwidth=None):
        if kernel not in ('rbf', 'linear'):
            raise ValueError(f"kernel must be 'rbf' or 'linear', got '{kernel}'")
        self.kernel = kernel
        self.bandwidth = bandwidth

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        model_names = data.model_names
        m = len(model_names)

        # Get 2D embeddings per model
        embeddings = _get_2d_embeddings(data)

        # Pre-compute bandwidth if needed
        if self.kernel == 'rbf' and self.bandwidth is None:
            bandwidth = _median_bandwidth(embeddings, model_names)
        else:
            bandwidth = self.bandwidth

        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                X = embeddings[model_names[i]]
                Y = embeddings[model_names[j]]
                d = _mmd_squared(X, Y, self.kernel, bandwidth)
                # MMD^2 can be slightly negative due to estimation; clamp
                d = max(0.0, d)
                d = np.sqrt(d)
                D[i, j] = d
                D[j, i] = d

        return validate_distance_matrix(D)


def _get_2d_embeddings(data):
    """Flatten paired 3D arrays to 2D if needed."""
    embeddings = {}
    for model in data.model_names:
        arr = data.response_embeddings[model]
        if arr.ndim == 3:
            # (n_queries, n_replicates, embed_dim) -> (n_queries * n_replicates, embed_dim)
            embeddings[model] = arr.reshape(-1, arr.shape[-1])
        else:
            embeddings[model] = arr
    return embeddings


def _median_bandwidth(embeddings, model_names):
    """Compute median heuristic bandwidth from all pairwise distances."""
    all_data = np.concatenate([embeddings[m] for m in model_names], axis=0)
    # Subsample if too large
    n = len(all_data)
    if n > 2000:
        idx = np.random.choice(n, 2000, replace=False)
        all_data = all_data[idx]
    dists = cdist(all_data, all_data, 'sqeuclidean')
    # Median of off-diagonal
    mask = ~np.eye(len(dists), dtype=bool)
    median_sq = np.median(dists[mask])
    return np.sqrt(median_sq / 2.0)  # sigma such that k(x,y) = exp(-||x-y||^2 / (2*sigma^2))


def _rbf_kernel(X, Y, bandwidth):
    """RBF (Gaussian) kernel matrix."""
    sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-sq_dists / (2.0 * bandwidth ** 2))


def _mmd_squared(X, Y, kernel, bandwidth=None):
    """Compute unbiased MMD^2 estimate between samples X and Y.

    Parameters
    ----------
    X : (n, d)
    Y : (m, d)
    kernel : str
    bandwidth : float or None

    Returns
    -------
    float : unbiased MMD^2 estimate
    """
    n = len(X)
    m = len(Y)

    if kernel == 'linear':
        # MMD^2_linear = ||mean(X) - mean(Y)||^2
        # For unbiased: use the U-statistic form
        mean_X = X.mean(axis=0)
        mean_Y = Y.mean(axis=0)
        return float(np.sum((mean_X - mean_Y) ** 2))

    elif kernel == 'rbf':
        K_XX = _rbf_kernel(X, X, bandwidth)
        K_YY = _rbf_kernel(Y, Y, bandwidth)
        K_XY = _rbf_kernel(X, Y, bandwidth)

        # Unbiased estimator: exclude diagonal for same-sample terms
        np.fill_diagonal(K_XX, 0.0)
        np.fill_diagonal(K_YY, 0.0)

        term_XX = K_XX.sum() / (n * (n - 1)) if n > 1 else 0.0
        term_YY = K_YY.sum() / (m * (m - 1)) if m > 1 else 0.0
        term_XY = K_XY.sum() / (n * m)

        return float(term_XX + term_YY - 2 * term_XY)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")
