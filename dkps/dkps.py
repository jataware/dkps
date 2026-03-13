import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from graspologic.embed import ClassicalMDS

from scipy.spatial.distance import pdist, squareform

from .coverage import estimate_coverage_weights


class DataKernelPerspectiveSpace:
    def __init__(
            self,
            response_distribution_fn=None,
            response_distribution_axis=1,
            metric_cmds='euclidean',
            n_components_cmds=None,
            n_elbows_cmds=2,
            dissimilarity="precomputed",
            alpha=None,
            coverage_correction=True,
            d_pca=None,
            kde_bandwidth=None,
        ):

        self.response_distribution_fn   = response_distribution_fn
        self.response_distribution_axis = response_distribution_axis
        self.metric_cmds                = metric_cmds
        self.n_components_cmds          = n_components_cmds
        self.n_elbows_cmds              = n_elbows_cmds
        self.dissimilarity              = dissimilarity
        self.alpha                      = alpha
        self.coverage_correction        = coverage_correction
        self.d_pca                      = d_pca
        self.kde_bandwidth              = kde_bandwidth

    @staticmethod
    def _convert_legacy_input(data):
        """Convert dict of 3D arrays to DataFrame with (model_id, query_id, embedding)."""
        assert isinstance(data, dict),                                  'data must be a dict'
        assert all([isinstance(x, np.ndarray) for x in data.values()]), 'all values must be numpy arrays'
        assert all([x.ndim == 3 for x in data.values()]),               'all arrays must be 3D - np.array(n_queries, n_replicates, embedding_dim)'
        assert len(set([x.shape for x in data.values()])) == 1,         'all arrays must have the same shape'

        rows = []
        for model_id, arr in data.items():
            n_queries = arr.shape[0]
            for q in range(n_queries):
                rows.append({
                    'model_id': model_id,
                    'query_id': f'q_{q}',
                    'embedding': arr[q],  # (n_replicates, embedding_dim)
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _partition_queries(df):
        """
        Analyze query overlap between models.

        Returns
        -------
        model_names : list
        query_sets : dict {model_id: set of query_ids}
        shared_queries_dict : dict {(model_i, model_k): set of shared query_ids}
        alpha_matrix : np.ndarray (n_models, n_models)
            Fraction of queries shared between each pair.
        """
        model_names = sorted(df['model_id'].unique())
        n = len(model_names)

        query_sets = {}
        for m in model_names:
            query_sets[m] = set(df.loc[df['model_id'] == m, 'query_id'].unique())

        shared_queries_dict = {}
        alpha_matrix = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                mi, mk = model_names[i], model_names[k]
                shared = query_sets[mi] & query_sets[mk]
                shared_queries_dict[(mi, mk)] = shared
                total = len(query_sets[mi] | query_sets[mk])
                alpha_matrix[i, k] = len(shared) / total if total > 0 else 1.0

        return model_names, query_sets, shared_queries_dict, alpha_matrix

    def _aggregate_embedding(self, emb_array):
        """Aggregate replicate embeddings to a single vector."""
        if emb_array.ndim == 1:
            return emb_array
        if self.response_distribution_fn is None:
            return emb_array[0]
        return self.response_distribution_fn(emb_array, axis=self.response_distribution_axis)

    def _compute_paired_distances(self, df, model_names, shared_queries_dict):
        """
        Compute paired distance matrix using shared queries.

        For each pair (i, k), distance = ||emb_i - emb_k||_F / sqrt(n_shared).
        """
        n = len(model_names)
        dist = np.zeros((n, n))

        for i in range(n):
            for k in range(i + 1, n):
                mi, mk = model_names[i], model_names[k]
                shared = sorted(shared_queries_dict[(mi, mk)])
                if len(shared) == 0:
                    dist[i, k] = dist[k, i] = np.nan
                    continue

                df_i = df[df['model_id'] == mi].set_index('query_id')
                df_k = df[df['model_id'] == mk].set_index('query_id')

                vecs_i = np.stack([self._aggregate_embedding(df_i.loc[q, 'embedding']) for q in shared])
                vecs_k = np.stack([self._aggregate_embedding(df_k.loc[q, 'embedding']) for q in shared])

                diff = vecs_i - vecs_k  # (n_shared, emb_dim)
                d = np.linalg.norm(diff) / np.sqrt(len(shared))
                dist[i, k] = dist[k, i] = d

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

    @staticmethod
    def _normalize_to_unit_median(D):
        """Scale distance matrix so off-diagonal median = 1."""
        off_diag = D[np.triu_indices_from(D, k=1)]
        if len(off_diag) == 0:
            return D.copy()
        med = np.median(off_diag)
        if med == 0:
            return D.copy()
        return D / med

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
        model_names, query_sets, shared_queries_dict, alpha_matrix = self._partition_queries(df)
        n = len(model_names)

        # Check if fully paired (all alpha == 1)
        fully_paired = np.all(alpha_matrix[np.triu_indices(n, k=1)] == 1.0)

        # Compute distance components
        has_shared = np.any(alpha_matrix[np.triu_indices(n, k=1)] > 0)
        has_unshared = np.any(alpha_matrix[np.triu_indices(n, k=1)] < 1.0)

        paired_dist = None
        unpaired_dist = None

        if has_shared:
            paired_dist = self._compute_paired_distances(df, model_names, shared_queries_dict)
        if has_unshared or not has_shared:
            unpaired_dist = self._compute_unpaired_distances(df, model_names)

        # Coverage correction for unpaired component
        if self.coverage_correction and unpaired_dist is not None and has_unshared:
            embs_by_model = {}
            for m in model_names:
                embs = df[df['model_id'] == m]['embedding'].values
                embs_by_model[m] = np.stack([self._aggregate_embedding(e) for e in embs])

            cov_weights = estimate_coverage_weights(
                embs_by_model, d_pca=self.d_pca, kde_bandwidth=self.kde_bandwidth
            )

            # Adjust unpaired distances: w * d + (1 - w) * median(d)
            med_unpaired = np.median(unpaired_dist[np.triu_indices(n, k=1)]) if n > 1 else 1.0
            for i in range(n):
                for k in range(i + 1, n):
                    mi, mk = model_names[i], model_names[k]
                    w = cov_weights.get((mi, mk), 1.0)
                    adjusted = w * unpaired_dist[i, k] + (1 - w) * med_unpaired
                    unpaired_dist[i, k] = unpaired_dist[k, i] = adjusted

        # Determine effective alpha per pair
        if self.alpha is not None:
            eff_alpha = np.full((n, n), self.alpha)
        else:
            eff_alpha = alpha_matrix.copy()

        # Normalize each component to unit median
        if paired_dist is not None:
            paired_norm = self._normalize_to_unit_median(paired_dist)
        if unpaired_dist is not None:
            unpaired_norm = self._normalize_to_unit_median(unpaired_dist)

        # Combine
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for k in range(i + 1, n):
                a = eff_alpha[i, k]
                d = 0.0
                if paired_dist is not None and not np.isnan(paired_norm[i, k]):
                    d += a * paired_norm[i, k]
                else:
                    a = 0.0  # no shared queries for this pair
                if unpaired_dist is not None:
                    d += (1.0 - a) * unpaired_norm[i, k]
                dist_matrix[i, k] = dist_matrix[k, i] = d

        # Symmetrize and zero diagonal
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0.0)

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
