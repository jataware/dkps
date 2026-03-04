"""
    distances/soft_paired.py — Method E: Soft-Paired Optimal Transport distance.
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
            "SoftPairedOTDistance requires the POT package. "
            "Install it with: pip install POT"
        )


class SoftPairedOTDistance:
    """Soft-Paired OT distance using query embeddings to define transport cost.

    Uses query similarity to bias the transport plan: responses to similar queries
    are cheaper to transport to each other, creating a "soft pairing" without
    requiring exact query alignment.

    Requires query_embeddings in ModelResponseData.

    Parameters
    ----------
    reg : float
        Entropic regularization. Default 0.1.
    query_weight : float
        Weight of query similarity in the cost matrix. Default 0.5.
        Cost = (1 - query_weight) * response_dist + query_weight * query_dist
    """

    def __init__(self, reg=0.1, query_weight=0.5):
        self.reg = reg
        self.query_weight = query_weight

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        ot = _require_pot()

        if data.query_embeddings is None:
            raise ValueError(
                "Soft-paired distance requires query information "
                "('query' or 'query_embedding' column)"
            )

        model_names = data.model_names
        m = len(model_names)

        # Get 2D embeddings
        resp_embs = _get_2d_embeddings(data, data.response_embeddings)
        query_embs = _get_2d_embeddings(data, data.query_embeddings)

        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                X_resp = resp_embs[model_names[i]]
                Y_resp = resp_embs[model_names[j]]
                X_query = query_embs[model_names[i]]
                Y_query = query_embs[model_names[j]]
                d = self._compute(X_resp, Y_resp, X_query, Y_query, ot)
                D[i, j] = d
                D[j, i] = d

        return validate_distance_matrix(D)

    def _compute(self, X_resp, Y_resp, X_query, Y_query, ot):
        n = len(X_resp)
        m = len(Y_resp)
        a = np.ones(n) / n
        b = np.ones(m) / m

        # Response distance cost
        M_resp = cdist(X_resp, Y_resp, 'sqeuclidean')
        # Normalize
        if M_resp.max() > 0:
            M_resp = M_resp / M_resp.max()

        # Query distance cost
        M_query = cdist(X_query, Y_query, 'sqeuclidean')
        if M_query.max() > 0:
            M_query = M_query / M_query.max()

        # Combined cost
        M = (1 - self.query_weight) * M_resp + self.query_weight * M_query

        result = ot.sinkhorn2(a, b, M, self.reg)
        val = float(result[0]) if hasattr(result, '__len__') else float(result)
        return np.sqrt(max(0.0, val))


def _get_2d_embeddings(data, emb_dict):
    """Flatten paired 3D arrays to 2D if needed."""
    embeddings = {}
    for model in data.model_names:
        arr = emb_dict[model]
        if arr.ndim == 3:
            embeddings[model] = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 1:
            embeddings[model] = arr.reshape(1, -1)
        else:
            embeddings[model] = arr
    return embeddings
