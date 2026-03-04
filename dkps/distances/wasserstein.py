"""
    distances/wasserstein.py — Method B: Wasserstein distance variants.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ..data import ModelResponseData
from .base import validate_distance_matrix


def _require_pot():
    try:
        import ot
        return ot
    except ImportError:
        raise ImportError(
            "WassersteinDistance requires the POT package. "
            "Install it with: pip install POT"
        )


class WassersteinDistance:
    """Wasserstein (Earth Mover's) distance between model response distributions.

    Works with unpaired data. For paired data, ignores pairing structure.

    Parameters
    ----------
    variant : str
        'sliced' (fast, approximate), 'exact' (linear program), or 'sinkhorn' (regularized).
        Default 'sliced'.
    n_projections : int
        Number of random projections for sliced variant. Default 100.
    reg : float
        Regularization for Sinkhorn variant. Default 0.1.
    """

    def __init__(self, variant='sliced', n_projections=100, reg=0.1):
        if variant not in ('sliced', 'exact', 'sinkhorn'):
            raise ValueError(f"variant must be 'sliced', 'exact', or 'sinkhorn', got '{variant}'")
        self.variant = variant
        self.n_projections = n_projections
        self.reg = reg

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        ot = _require_pot()

        model_names = data.model_names
        m = len(model_names)

        embeddings = _get_2d_embeddings(data)

        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                X = embeddings[model_names[i]]
                Y = embeddings[model_names[j]]
                d = self._compute(X, Y, ot)
                D[i, j] = d
                D[j, i] = d

        return validate_distance_matrix(D)

    def _compute(self, X, Y, ot):
        n = len(X)
        m = len(Y)
        a = np.ones(n) / n
        b = np.ones(m) / m

        if self.variant == 'sliced':
            return float(ot.sliced_wasserstein_distance(
                X, Y, a, b,
                n_projections=self.n_projections,
            ))

        elif self.variant == 'exact':
            M = cdist(X, Y, 'sqeuclidean')
            return float(np.sqrt(ot.emd2(a, b, M)))

        elif self.variant == 'sinkhorn':
            M = cdist(X, Y, 'sqeuclidean')
            # Normalize cost matrix for numerical stability
            M = M / M.max() if M.max() > 0 else M
            result = ot.sinkhorn2(a, b, M, self.reg)
            val = float(result[0]) if hasattr(result, '__len__') else float(result)
            return np.sqrt(max(0.0, val))


def _get_2d_embeddings(data):
    """Flatten paired 3D arrays to 2D if needed."""
    embeddings = {}
    for model in data.model_names:
        arr = data.response_embeddings[model]
        if arr.ndim == 3:
            embeddings[model] = arr.reshape(-1, arr.shape[-1])
        else:
            embeddings[model] = arr
    return embeddings
