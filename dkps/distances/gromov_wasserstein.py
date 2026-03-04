"""
    distances/gromov_wasserstein.py — Method D: Gromov-Wasserstein distance.
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
            "GromovWassersteinDistance requires the POT package. "
            "Install it with: pip install POT"
        )


class GromovWassersteinDistance:
    """Gromov-Wasserstein distance between model response distributions.

    Compares internal structure (pairwise distances within each distribution)
    rather than pointwise correspondence. Invariant to rotations/reflections.

    Works with unpaired data.

    Parameters
    ----------
    reg : float
        Entropic regularization. 0 = exact (slower). Default 0.01.
    max_iter : int
        Maximum iterations. Default 1000.
    loss : str
        Loss function: 'square_loss' or 'kl_loss'. Default 'square_loss'.
    """

    def __init__(self, reg=0.01, max_iter=1000, loss='square_loss'):
        self.reg = reg
        self.max_iter = max_iter
        self.loss = loss

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        ot = _require_pot()

        model_names = data.model_names
        m = len(model_names)

        embeddings = _get_2d_embeddings(data)

        # Pre-compute intra-distribution distance matrices
        intra_dists = {}
        for model in model_names:
            X = embeddings[model]
            intra_dists[model] = cdist(X, X, 'sqeuclidean')

        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                C1 = intra_dists[model_names[i]]
                C2 = intra_dists[model_names[j]]
                d = self._compute(C1, C2, ot)
                D[i, j] = d
                D[j, i] = d

        return validate_distance_matrix(D)

    def _compute(self, C1, C2, ot):
        n = len(C1)
        m = len(C2)
        p = np.ones(n) / n
        q = np.ones(m) / m

        if self.reg > 0:
            gw_dist, log = ot.gromov.entropic_gromov_wasserstein2(
                C1, C2, p, q,
                loss_fun=self.loss,
                epsilon=self.reg,
                max_iter=self.max_iter,
                log=True,
            )
        else:
            gw_dist, log = ot.gromov.gromov_wasserstein2(
                C1, C2, p, q,
                loss_fun=self.loss,
                log=True,
            )

        return float(np.sqrt(max(0.0, gw_dist)))


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
