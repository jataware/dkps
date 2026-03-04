"""
    distances/paired.py — Method 0: Paired DKPS distance (extracted from original).
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

from ..data import ModelResponseData
from .base import validate_distance_matrix


class PairedDistance:
    """Paired distance: flatten query-aligned embeddings, compute Euclidean/other metric.

    This is the original DKPS distance method. It requires paired data where all models
    have responded to the same set of queries (identified by query_id).

    Parameters
    ----------
    metric : str
        Distance metric (default 'euclidean'). Any scipy pdist metric.
    response_agg_fn : callable or None
        How to aggregate replicates. None = take first replicate.
    response_agg_axis : int
        Axis for replicate aggregation (default 1).
    """

    def __init__(self, metric='euclidean', response_agg_fn=None, response_agg_axis=1):
        self.metric = metric
        self.response_agg_fn = response_agg_fn
        self.response_agg_axis = response_agg_axis

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        if not data.paired:
            raise ValueError("PairedDistance requires paired data (with query_id alignment)")

        # Aggregate replicates -> {model: (n_queries, embed_dim)}
        agg = data.aggregate_replicates(fn=self.response_agg_fn, axis=self.response_agg_axis)

        # Stack -> (n_models, n_queries, embed_dim)
        model_names = data.model_names
        X = np.stack([agg[m] for m in model_names])
        n_models, n_queries, embed_dim = X.shape

        # Flatten -> (n_models, n_queries * embed_dim)
        X_flat = X.reshape(n_models, -1)

        if self.metric == 'euclidean':
            D = pairwise_distances(X_flat, metric='euclidean') / np.sqrt(n_queries)
        else:
            D = squareform(pdist(X_flat, metric=self.metric)) / np.sqrt(n_queries)

        return validate_distance_matrix(D)
