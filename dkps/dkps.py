import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from graspologic.embed import ClassicalMDS

from scipy.spatial.distance import pdist, squareform


def _normalize_to_unit_median(D):
    """Scale a distance matrix so its off-diagonal median equals 1."""
    med = np.median(D[np.triu_indices_from(D, k=1)])
    return D / med if med > 0 else D


class DataKernelPerspectiveSpace:
    def __init__(
            self,
            response_distribution_fn=None,
            response_distribution_axis=1,
            metric_cmds='euclidean',
            n_components_cmds=None,
            n_elbows_cmds=2,
            dissimilarity="precomputed",
        ):

        self.response_distribution_fn   = response_distribution_fn
        self.response_distribution_axis = response_distribution_axis
        self.metric_cmds                = metric_cmds
        self.n_components_cmds          = n_components_cmds
        self.n_elbows_cmds              = n_elbows_cmds
        self.dissimilarity              = dissimilarity

    @staticmethod
    def _partition_queries(df):
        """
        Analyze query overlap between models.

        Returns
        -------
        model_names : list
        shared_queries : set of query_ids shared by all models
        alpha : float
            Fraction of total queries that are shared.
        """
        model_names = sorted(df['model_id'].unique())

        query_sets = [set(df.loc[df['model_id'] == m, 'query_id'].unique()) for m in model_names]
        shared_queries = set.intersection(*query_sets) if query_sets else set()
        all_queries = set.union(*query_sets) if query_sets else set()
        alpha = len(shared_queries) / len(all_queries) if all_queries else 1.0

        return model_names, shared_queries, alpha

    def _aggregate_embedding(self, emb_array):
        """Aggregate replicate embeddings to a single vector."""
        if emb_array.ndim == 1:
            return emb_array
        if self.response_distribution_fn is None:
            return emb_array[0]
        return self.response_distribution_fn(emb_array, axis=self.response_distribution_axis)

    def _compute_paired_distances(self, df, model_names, shared_queries):
        """
        Compute paired distance matrix using shared queries.

        Assumes all models answered all shared queries.

        For each pair (i, k), distance = ||emb_i - emb_k||_F / sqrt(n_shared).
        """
        n = len(model_names)
        shared = sorted(shared_queries)
        assert len(shared) > 0, 'no shared queries'

        # Build (n_models, n_shared, emb_dim) matrix
        vecs = {}
        for m in model_names:
            df_m = df[df['model_id'] == m].set_index('query_id')
            assert shared_queries <= set(df_m.index), f'model {m} missing shared queries'
            vecs[m] = np.stack([self._aggregate_embedding(df_m.loc[q, 'embedding']) for q in shared])

        # Pairwise distances: ||flat_i - flat_k|| / sqrt(n_shared)
        X_flat = np.stack([vecs[m].ravel() for m in model_names])
        dist = pairwise_distances(X_flat, metric='euclidean') / np.sqrt(len(shared))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        return dist

    def _compute_unpaired_distances(self, df, model_names):
        """
        Compute unpaired (linear MMD) distance matrix.

        For each pair (i, k), distance = ||mean(emb_i) - mean(emb_k)||.
        """
        n = len(model_names)

        # Compute mean embedding per model
        means = {}
        for m in model_names:
            embs = df[df['model_id'] == m]['embedding'].values
            agg = np.stack([self._aggregate_embedding(e) for e in embs])
            means[m] = agg.mean(axis=0)

        dist = np.zeros((n, n))
        for i in range(n):
            for k in range(i + 1, n):
                d = np.linalg.norm(means[model_names[i]] - means[model_names[k]])
                dist[i, k] = dist[k, i] = d

        return dist

    def _fit_transform_legacy(self, data, return_dict=True):
        """Original fit_transform path for fully-paired dict-of-arrays input."""
        assert isinstance(data, dict),                                  'data must be a dict'
        assert all([isinstance(x, np.ndarray) for x in data.values()]), 'all values must be numpy arrays'
        assert all([x.ndim == 3 for x in data.values()]),               'all arrays must be 3D - np.array(n_queries, n_replicates, embedding_dim)'
        assert len(set([x.shape for x in data.values()])) == 1,         'all arrays must have the same shape'

        # aggregate over replicates -> (n_models, n_queries, embedding_dim)
        if self.response_distribution_fn is None:
            X = np.stack([v[:,0] for v in data.values()])
        else:
            X = np.stack([self.response_distribution_fn(v, axis=self.response_distribution_axis) for k,v in data.items()])

        n_models, n_queries, embedding_dim = X.shape

        # flatten -> (n_models, n_queries * embedding_dim)
        X_flat = X.reshape(len(X), -1)

        if self.metric_cmds == 'euclidean':
            dist_matrix = pairwise_distances(X_flat, metric='euclidean') / np.sqrt(n_queries)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
        else:
            dist_matrix = squareform(pdist(X_flat, metric=self.metric_cmds)) / np.sqrt(n_queries)

        cmds_embds = ClassicalMDS(n_components=self.n_components_cmds, n_elbows=self.n_elbows_cmds, dissimilarity=self.dissimilarity).fit_transform(dist_matrix)

        if return_dict:
            return {key: cmds_embds[i] for i, key in enumerate(data.keys())}
        else:
            return cmds_embds

    def fit_transform(self, data, return_dict=True):
        """
        Embed models into a low-dimensional space via DKPS.

        Parameters
        ----------
        data : dict or pd.DataFrame
            - dict: {model_name: np.array(n_queries, n_replicates, embedding_dim)}
              Legacy fully-paired format. All arrays must have the same shape.
            - DataFrame with columns: model_id, query_id, embedding
              Supports both paired and unpaired queries.
        return_dict : bool
            If True, return dict {model_name: embedding}. If False, return array.

        Returns
        -------
        dict or np.ndarray
        """
        # Legacy dict path: use original code exactly for backward compatibility
        if isinstance(data, dict):
            return self._fit_transform_legacy(data, return_dict=return_dict)

        # DataFrame path
        assert isinstance(data, pd.DataFrame), 'data must be a dict or DataFrame'
        for col in ('model_id', 'query_id', 'embedding'):
            assert col in data.columns, f'DataFrame must have column: {col}'

        df = data.copy()
        model_names, shared_queries, alpha = self._partition_queries(df)
        has_shared = len(shared_queries) > 0
        has_unshared = alpha < 1.0

        # Compute distance components
        if has_shared:
            paired_dist = self._compute_paired_distances(df, model_names, shared_queries)
        if has_unshared or not has_shared:
            unpaired_dist = self._compute_unpaired_distances(df, model_names)

        # Combine (normalize each to unit median before interpolating)
        if has_shared and has_unshared:
            paired_dist = _normalize_to_unit_median(paired_dist)
            unpaired_dist = _normalize_to_unit_median(unpaired_dist)
            dist_matrix = alpha * paired_dist + (1.0 - alpha) * unpaired_dist
        elif has_shared:
            dist_matrix = paired_dist
        else:
            dist_matrix = unpaired_dist

        # Store fitted attributes
        self.dist_matrix_ = dist_matrix
        self.model_names_ = model_names

        # CMDS embedding
        cmds_embds = ClassicalMDS(
            n_components=self.n_components_cmds,
            n_elbows=self.n_elbows_cmds,
            dissimilarity=self.dissimilarity,
        ).fit_transform(dist_matrix)

        if return_dict:
            return {m: cmds_embds[i] for i, m in enumerate(model_names)}
        else:
            return cmds_embds
