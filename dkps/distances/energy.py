"""
    distances/energy.py — Method C: Energy distance.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ..data import ModelResponseData
from .base import validate_distance_matrix


class EnergyDistance:
    """Energy distance between model response distributions.

    Parameter-free. Works with unpaired data. For paired data, ignores
    pairing structure and treats each response independently.

    Energy distance: E(X,Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    where X,X' are iid from P and Y,Y' are iid from Q.
    """

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        model_names = data.model_names
        m = len(model_names)

        # Get 2D embeddings per model
        embeddings = _get_2d_embeddings(data)

        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                X = embeddings[model_names[i]]
                Y = embeddings[model_names[j]]
                d = _energy_distance(X, Y)
                D[i, j] = d
                D[j, i] = d

        return validate_distance_matrix(D)


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


def _energy_distance(X, Y):
    """Compute energy distance between samples X and Y.

    Parameters
    ----------
    X : (n, d)
    Y : (m, d)

    Returns
    -------
    float : energy distance (non-negative)
    """
    D_XY = cdist(X, Y, 'euclidean')
    D_XX = cdist(X, X, 'euclidean')
    D_YY = cdist(Y, Y, 'euclidean')

    n = len(X)
    m = len(Y)

    term_XY = D_XY.mean()

    # Exclude diagonal (self-distances = 0) for unbiased estimate
    if n > 1:
        term_XX = D_XX.sum() / (n * (n - 1))
    else:
        term_XX = 0.0

    if m > 1:
        term_YY = D_YY.sum() / (m * (m - 1))
    else:
        term_YY = 0.0

    e_sq = 2 * term_XY - term_XX - term_YY
    # Clamp to non-negative (can be slightly negative due to estimation)
    return float(np.sqrt(max(0.0, e_sq)))
