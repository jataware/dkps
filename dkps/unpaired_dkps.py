import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from graspologic.embed import ClassicalMDS


class DoubleKernelDKPS:
    """
    Double-kernel DKPS: extends DKPS to partially-observed (model x query) designs.

    Uses a product kernel k_Q * k_R where k_Q weights response comparisons by
    query similarity. Reduces to standard paired DKPS when query_kernel='delta'
    and all models share the same queries.

    For each model pair (i, k):

        D^2(i,k) = A_ii + A_kk - 2 * A_ik

        A_ab = sum_{j in Q_a, l in Q_b} k_Q(q_j, q_l) * k_R(x_aj, x_bl) / Z_ab

    where Z_ab = sum k_Q(q_j, q_l) is the query kernel mass.

    Parameters
    ----------
    query_kernel : str or callable
        'rbf' (default) — RBF kernel on query embeddings.
        'delta' — 1 if same query_id, 0 otherwise. Recovers standard paired DKPS.
        callable(Q_a, Q_b) -> (n_a, n_b) matrix.
    response_kernel : str or callable
        'linear' (default) — dot product on response embeddings.
        'rbf' — RBF kernel.
        callable(X_a, X_b) -> (n_a, n_b) matrix.
    query_bandwidth : float or 'median'
        Bandwidth for RBF query kernel.
    response_bandwidth : float or 'median'
        Bandwidth for RBF response kernel.
    task_filter : str or None
        None (default) — use all queries.
        'shared' — restrict to queries from tasks shared by both models.
        'unshared' — restrict to queries from tasks unique to each model.
        Requires task_id column in data.
    n_components_cmds : int or None
        Embedding dimension for ClassicalMDS.
    n_elbows_cmds : int
        Number of elbows for automatic dimension selection.
    """

    def __init__(
        self,
        query_kernel='rbf',
        response_kernel='linear',
        query_bandwidth='median',
        response_bandwidth='median',
        task_filter=None,
        n_components_cmds=None,
        n_elbows_cmds=2,
    ):
        self.query_kernel = query_kernel
        self.response_kernel = response_kernel
        self.query_bandwidth = query_bandwidth
        self.response_bandwidth = response_bandwidth
        self.task_filter = task_filter
        self.n_components_cmds = n_components_cmds
        self.n_elbows_cmds = n_elbows_cmds

    def fit(self, data):
        """Compute pairwise distance matrix from partially-observed response data."""
        model_data, model_names = self._prepare_model_data(data)

        if self.query_kernel == 'rbf':
            all_q = np.vstack([md['query_emb'] for md in model_data.values()])
            self.query_bandwidth_ = self._resolve_bandwidth(all_q, self.query_bandwidth)

        if self.response_kernel == 'rbf':
            all_r = np.vstack([md['emb'] for md in model_data.values()])
            self.response_bandwidth_ = self._resolve_bandwidth(all_r, self.response_bandwidth)

        n = len(model_names)
        dist_sq = np.zeros((n, n))

        if self.task_filter is not None:
            for i in range(n):
                for k in range(i + 1, n):
                    d2 = self._pairwise_dist_sq_filtered(
                        model_data[model_names[i]], model_data[model_names[k]])
                    dist_sq[i, k] = dist_sq[k, i] = max(d2, 0.0) if np.isfinite(d2) else np.nan
        else:
            a_self = np.array([
                self._compute_A(model_data[m], model_data[m]) for m in model_names
            ])
            for i in range(n):
                for k in range(i + 1, n):
                    a_ik = self._compute_A(model_data[model_names[i]], model_data[model_names[k]])
                    d2 = a_self[i] + a_self[k] - 2 * a_ik
                    dist_sq[i, k] = dist_sq[k, i] = max(d2, 0.0)

        self.dist_matrix_ = np.sqrt(np.nan_to_num(dist_sq, nan=np.nan))
        self.model_names_ = model_names
        return self

    def fit_transform(self, data):
        """Fit and embed models into low-dimensional space via ClassicalMDS."""
        self.fit(data)
        cmds = ClassicalMDS(
            n_components=self.n_components_cmds,
            n_elbows=self.n_elbows_cmds,
            dissimilarity='precomputed',
        ).fit_transform(self.dist_matrix_)
        return {m: cmds[i] for i, m in enumerate(self.model_names_)}

    def _prepare_model_data(self, data):
        assert isinstance(data, pd.DataFrame), 'data must be a DataFrame'
        for col in ('model_id', 'query_id', 'embedding'):
            assert col in data.columns, f'missing column: {col}'
        if self.query_kernel not in ('delta',) and not callable(self.query_kernel):
            assert 'query_embedding' in data.columns, \
                'query_embedding column required when query_kernel is not delta'
        if self.task_filter is not None:
            assert 'task_id' in data.columns, \
                'task_id column required when task_filter is set'

        assert not data.duplicated(['model_id', 'query_id']).any(), \
            'duplicate (model_id, query_id) pairs'

        model_names = sorted(data['model_id'].unique())
        model_data = {}
        for name in model_names:
            sub = data[data['model_id'] == name].sort_values('query_id')
            md = {
                'emb': np.stack(sub['embedding'].values),
                'query_ids': sub['query_id'].values,
            }
            if 'query_embedding' in data.columns:
                md['query_emb'] = np.stack(sub['query_embedding'].values)
            if 'task_id' in data.columns:
                md['task_ids'] = sub['task_id'].values
            model_data[name] = md

        return model_data, model_names

    def _resolve_bandwidth(self, X, method):
        if isinstance(method, (int, float)):
            return float(method)
        if method == 'median':
            dists = pdist(X)
            return float(np.median(dists)) if len(dists) > 0 else 1.0
        if callable(method):
            return float(method(X))
        raise ValueError(f'unknown bandwidth method: {method}')

    def _filter_to_tasks(self, md, tasks):
        """Return a subset of model data restricted to the given task set."""
        mask = np.isin(md['task_ids'], list(tasks))
        if not mask.any():
            return None
        out = {'emb': md['emb'][mask], 'query_ids': md['query_ids'][mask]}
        if 'query_emb' in md:
            out['query_emb'] = md['query_emb'][mask]
        if 'task_ids' in md:
            out['task_ids'] = md['task_ids'][mask]
        return out

    def _pairwise_dist_sq_filtered(self, md_a, md_b):
        """Compute D^2 using only shared or unshared task queries."""
        tasks_a = set(md_a['task_ids'])
        tasks_b = set(md_b['task_ids'])

        if self.task_filter == 'shared':
            keep_a = tasks_a & tasks_b
            keep_b = keep_a
        elif self.task_filter == 'unshared':
            keep_a = tasks_a - tasks_b
            keep_b = tasks_b - tasks_a
        else:
            raise ValueError(f'unknown task_filter: {self.task_filter}')

        if not keep_a or not keep_b:
            return np.nan

        fa = self._filter_to_tasks(md_a, keep_a)
        fb = self._filter_to_tasks(md_b, keep_b)
        if fa is None or fb is None:
            return np.nan

        a_ii = self._compute_A(fa, fa)
        a_kk = self._compute_A(fb, fb)
        a_ik = self._compute_A(fa, fb)
        return a_ii + a_kk - 2 * a_ik

    def _compute_A(self, data_a, data_b):
        """Compute A_ab = sum(K_Q * K_R) / sum(K_Q)."""
        if self.query_kernel == 'delta':
            ids_a, ids_b = data_a['query_ids'], data_b['query_ids']
            K_Q = (ids_a[:, None] == ids_b[None, :]).astype(np.float64)
        elif self.query_kernel == 'rbf':
            sq = cdist(data_a['query_emb'], data_b['query_emb'], 'sqeuclidean')
            K_Q = np.exp(-sq / (2 * self.query_bandwidth_ ** 2))
        elif callable(self.query_kernel):
            K_Q = np.asarray(self.query_kernel(data_a['query_emb'], data_b['query_emb']))
        else:
            raise ValueError(f'unknown query_kernel: {self.query_kernel}')

        emb_a, emb_b = data_a['emb'], data_b['emb']
        if self.response_kernel == 'linear':
            K_R = emb_a @ emb_b.T
        elif self.response_kernel == 'rbf':
            sq = cdist(emb_a, emb_b, 'sqeuclidean')
            K_R = np.exp(-sq / (2 * self.response_bandwidth_ ** 2))
        elif callable(self.response_kernel):
            K_R = np.asarray(self.response_kernel(emb_a, emb_b))
        else:
            raise ValueError(f'unknown response_kernel: {self.response_kernel}')

        Z = K_Q.sum()
        if Z == 0:
            return np.nan
        return (K_Q * K_R).sum() / Z
