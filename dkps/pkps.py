"""High-level PKPS / DKPS estimators with a fit / predict / update API.

Pipeline (matching the paper):

  1. embed queries and responses (optional -- precomputed embeddings are accepted),
  2. PCA-reduce each channel (fixed dim, or Zhu-Ghodsi elbow selection),
  3. product-kernel affinities  A_ab = sum_{j,l} k_Q(u_j, u_l) k_R(x_aj, x_bl) / sum_{j,l} k_Q(u_j, u_l),
  4. squared distances          D^2_ab = A_aa + A_bb - 2 A_ab,
  5. classical MDS on D  ->  model representations Z,
  6. per-task k-NN regression from Z onto observed sample scores.

`DKPS` is the paired special case: k_Q is the delta kernel on query ids, which on a
fully paired design recovers the standard DKPS distance (mean per-query squared
distance under a linear response kernel).

Records are exchanged as JSON-friendly tables: a list of dicts, a dict of lists, a
DataFrame, or a JSON string of either. Fit/update rows carry
(model_id, task_id, query_id, query|query_embedding, response|response_embedding, score);
predict rows carry (model_id[, task_id]).
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.extmath import randomized_svd
from graspologic.embed import ClassicalMDS, select_dimension


QUERY_DEFAULTS = dict(kernel='rbf', bandwidth='median', pca_dim='elbow', pca_n_elbows=2)
RESPONSE_DEFAULTS = dict(kernel='linear', bandwidth='median', pca_dim='elbow', pca_n_elbows=2)
MDS_DEFAULTS = dict(dim=None, n_elbows=2)
EMBEDDING_DEFAULTS = dict(provider='google', model=None, api_key=None)

_PROVIDER_ENV = {
    'google': 'GEMINI_API_KEY',
    'jina': 'JINA_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
    'huggingface': 'HF_TOKEN',
}


class _FrozenPCA:
    """PCA basis learned at fit time and frozen for out-of-sample updates.

    dim='elbow' selects the dimension at the n_elbows-th Zhu-Ghodsi elbow of the
    singular value spectrum; dim=int fixes it; dim=None disables reduction.
    """

    def __init__(self, dim='elbow', n_elbows=2, max_components=128, random_state=0):
        self.dim = dim
        self.n_elbows = n_elbows
        self.max_components = max_components
        self.random_state = random_state
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        if self.dim is None:
            return self
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        k = int(min(self.max_components, n - 1, d))
        if k < 1:
            self.mean_ = None
            return self
        _, S, Vt = randomized_svd(X - self.mean_, n_components=k,
                                  random_state=self.random_state)
        if self.dim == 'elbow':
            if k < 2 or len(S) < 2:
                dim = k
            else:
                elbows, _ = select_dimension(S, n_elbows=self.n_elbows)
                dim = int(elbows[-1]) if len(elbows) else k
        else:
            dim = int(self.dim)
        dim = max(1, min(dim, k))
        self.components_ = Vt[:dim]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.components_ is None:
            return X
        return (X - self.mean_) @ self.components_.T


def _median_bandwidth(X, cap=2000, random_state=0):
    X = np.asarray(X, dtype=np.float64)
    if len(X) > cap:
        rng = np.random.default_rng(random_state)
        X = X[rng.choice(len(X), cap, replace=False)]
    d = pdist(X)
    return float(np.median(d)) if len(d) else 1.0


def _kernel_matrix(kernel, bandwidth, Xa, Xb, ids_a=None, ids_b=None):
    """Kernel matrix between two row sets. 'delta' compares ids; the rest compare
    embedding vectors."""
    if kernel == 'delta':
        return (np.asarray(ids_a)[:, None] == np.asarray(ids_b)[None, :]).astype(np.float64)
    if kernel == 'rbf':
        sq = cdist(Xa, Xb, 'sqeuclidean')
        return np.exp(-sq / (2.0 * bandwidth ** 2))
    if kernel == 'linear':
        return Xa @ Xb.T
    if kernel == 'cosine':
        na = np.clip(np.linalg.norm(Xa, axis=1, keepdims=True), 1e-12, None)
        nb = np.clip(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-12, None)
        return np.clip((Xa / na) @ (Xb / nb).T, 0.0, None)
    if callable(kernel):
        return np.asarray(kernel(Xa, Xb))
    raise ValueError(f'unknown kernel: {kernel}')


def _parse_records(records, required, aliases=()):
    """Normalize records (DataFrame / list of dicts / dict of lists / JSON string of
    either) into a DataFrame with the required columns present."""
    if isinstance(records, str):
        records = json.loads(records)
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame(records)
    elif isinstance(records, (list, tuple)):
        df = pd.DataFrame(list(records))
    else:
        raise TypeError('records must be a DataFrame, list of dicts, dict of lists, '
                        'or a JSON string of either')
    for old, new in aliases:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    for col in required:
        if col not in df.columns:
            raise ValueError(f'records missing required column: {col}')
    return df


class PKPS:
    """Product Kernel Perspective Space estimator.

    Parameters
    ----------
    query_kwargs : dict, optional
        Query channel configuration (defaults in QUERY_DEFAULTS):
          kernel : 'rbf' | 'delta' | 'linear' | 'cosine' | callable
          bandwidth : 'median' or float (RBF kernels)
          pca_dim : 'elbow' (Zhu-Ghodsi selection) | int | None (no reduction)
          pca_n_elbows : int, elbow index used when pca_dim='elbow'
    response_kwargs : dict, optional
        Response channel configuration; same keys (defaults in RESPONSE_DEFAULTS).
    mds_kwargs : dict, optional
        Classical MDS on the model distance matrix:
          dim : int or None (Zhu-Ghodsi selection)
          n_elbows : int, elbow index used when dim is None
        The MDS acts on the joint product-kernel distance matrix, so there is a
        single spec rather than one per channel.
    embedding_kwargs : dict, optional
        Text embedding backend, used only when fit/update rows carry raw text
        instead of precomputed embeddings:
          provider : see dkps.embed.embed_api ('google', 'jina', ...)
          model : provider-specific model name (None = provider default)
          api_key : API key (None = read from the provider's environment variable)

    Attributes (after fit)
    ----------------------
    model_names_ : list of model ids
    task_names_ : list of task ids
    dist_matrix_ : (n, n) PKPS distance matrix
    embedding_ : dict model_id -> MDS coordinates
    sample_scores_ : DataFrame (model x task) of observed sample scores
    """

    def __init__(self, query_kwargs=None, response_kwargs=None, mds_kwargs=None,
                 embedding_kwargs=None):
        self.query_kwargs = {**QUERY_DEFAULTS, **(query_kwargs or {})}
        self.response_kwargs = {**RESPONSE_DEFAULTS, **(response_kwargs or {})}
        self.mds_kwargs = {**MDS_DEFAULTS, **(mds_kwargs or {})}
        self.embedding_kwargs = {**EMBEDDING_DEFAULTS, **(embedding_kwargs or {})}
        self._fitted = False

    # ------------------------------------------------------------------
    # fit / update / predict
    # ------------------------------------------------------------------
    def fit(self, records):
        """Fit from response-level records.

        Each record: model_id, task_id, query_id, a response (text under
        'response', or a vector under 'response_embedding'/'embedding'), an
        optional query (text under 'query', or vector under 'query_embedding';
        required unless the query kernel is 'delta'), and an optional per-response
        'score'.
        """
        df = self._ingest(records)
        self._raw = df
        self._fit_reducers(df)
        self._model_data = {m: self._reduce_model(df[df['model_id'] == m])
                            for m in sorted(df['model_id'].unique())}
        self._A = {}
        self._recompute_pairs(set(self._model_data))
        self._finalize()
        self._fitted = True
        return self

    def update(self, records):
        """Add records (new models, or new responses for existing models) and
        refresh the fit.

        The PCA bases and kernel bandwidths learned at fit time are frozen, so
        only affinities involving updated models are recomputed; the MDS and the
        sample-score table are refreshed. Re-run fit() to relearn the reducers.
        """
        assert self._fitted, 'call fit() before update()'
        df = self._ingest(records)
        touched = set(df['model_id'].unique())
        merged = pd.concat([self._raw, df], ignore_index=True)
        dup = merged.duplicated(['model_id', 'task_id', 'query_id'], keep='last')
        if dup.any():
            warnings.warn(f'update: {int(dup.sum())} (model, task, query) rows '
                          'replaced by their newer version', stacklevel=2)
            merged = merged[~dup]
        self._raw = merged.reset_index(drop=True)
        for m in touched:
            self._model_data[m] = self._reduce_model(self._raw[self._raw['model_id'] == m])
        self._recompute_pairs(touched)
        self._finalize()
        return self

    def predict(self, records=None, k=5):
        """Predict scores for (model, task) pairs via per-task k-NN over the MDS
        representation, trained on the other models' observed sample scores (the
        target model's own cell is never used).

        records : None (all cells without an observed score), or records with
        'model_id' and optionally 'task_id' (a model without task_id requests
        every task). Returns a list of dicts with 'score_hat' added.
        """
        assert self._fitted, 'call fit() before predict()'
        pairs = self._parse_pairs(records)
        Z = np.stack([self.embedding_[m] for m in self.model_names_])
        idx = {m: i for i, m in enumerate(self.model_names_)}
        S = self.sample_scores_
        out = []
        by_task = {}
        for m, t in pairs:
            by_task.setdefault(t, []).append(m)
        preds = {}
        for t, ms in by_task.items():
            obs = S[t].dropna()
            for m in ms:
                train = obs.drop(m, errors='ignore')
                if len(train) < 1:
                    preds[(m, t)] = np.nan
                    continue
                rows = [idx[o] for o in train.index]
                mdl = KNeighborsRegressor(n_neighbors=min(k, len(train)),
                                          weights='distance')
                mdl.fit(Z[rows], train.values)
                preds[(m, t)] = float(mdl.predict(Z[idx[m]][None])[0])
        for m, t in pairs:
            out.append({'model_id': m, 'task_id': t, 'score_hat': preds[(m, t)]})
        return out

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def _needs_query_channel(self):
        return self.query_kwargs['kernel'] != 'delta'

    def _ingest(self, records):
        df = _parse_records(records, required=('model_id', 'query_id'),
                            aliases=[('embedding', 'response_embedding')])
        if 'task_id' not in df.columns:
            df['task_id'] = '_task'
        if 'score' not in df.columns:
            df['score'] = np.nan
        df['score'] = pd.to_numeric(df['score'], errors='coerce')

        if 'response_embedding' not in df.columns:
            if 'response' not in df.columns:
                raise ValueError("records need 'response_embedding' vectors or "
                                 "'response' text")
            df['response_embedding'] = list(self._embed_texts(df['response'].tolist()))
        if self._needs_query_channel() and 'query_embedding' not in df.columns:
            if 'query' not in df.columns:
                raise ValueError("records need 'query_embedding' vectors or 'query' "
                                 "text (query kernel is not 'delta')")
            df['query_embedding'] = list(self._embed_texts(df['query'].tolist()))

        keep = ['model_id', 'task_id', 'query_id', 'response_embedding', 'score']
        if 'query_embedding' in df.columns:
            keep.append('query_embedding')
        df = df[keep].copy()
        assert not df.duplicated(['model_id', 'task_id', 'query_id']).any(), \
            'duplicate (model_id, task_id, query_id) rows'
        return df

    def _embed_texts(self, texts):
        """Embed texts through dkps.embed, deduplicating identical strings."""
        from .embed import embed_api
        provider = self.embedding_kwargs['provider']
        api_key = self.embedding_kwargs.get('api_key')
        if api_key and provider in _PROVIDER_ENV:
            os.environ[_PROVIDER_ENV[provider]] = api_key
        uniq = list(dict.fromkeys(texts))
        kwargs = {}
        if self.embedding_kwargs.get('model'):
            kwargs['model'] = self.embedding_kwargs['model']
        E = embed_api(provider, uniq, **kwargs)
        lookup = {t: E[i] for i, t in enumerate(uniq)}
        return [lookup[t] for t in texts]

    # ------------------------------------------------------------------
    # reduction + kernels
    # ------------------------------------------------------------------
    def _fit_reducers(self, df):
        R = np.stack([np.asarray(e, dtype=np.float64) for e in df['response_embedding']])
        self._response_pca = _FrozenPCA(self.response_kwargs['pca_dim'],
                                        self.response_kwargs['pca_n_elbows']).fit(R)
        Rr = self._response_pca.transform(R)
        self._response_bandwidth = self._resolve_bandwidth(
            self.response_kwargs, Rr) if self.response_kwargs['kernel'] == 'rbf' else None
        if 'query_embedding' in df.columns:
            Q = np.stack([np.asarray(e, dtype=np.float64) for e in df['query_embedding']])
            self._query_pca = _FrozenPCA(self.query_kwargs['pca_dim'],
                                         self.query_kwargs['pca_n_elbows']).fit(Q)
            Qr = self._query_pca.transform(Q)
            self._query_bandwidth = self._resolve_bandwidth(
                self.query_kwargs, Qr) if self.query_kwargs['kernel'] == 'rbf' else None
        else:
            self._query_pca = None
            self._query_bandwidth = None

    @staticmethod
    def _resolve_bandwidth(spec, X):
        bw = spec['bandwidth']
        if isinstance(bw, (int, float)):
            return float(bw)
        if bw == 'median':
            return _median_bandwidth(X)
        if callable(bw):
            return float(bw(X))
        raise ValueError(f'unknown bandwidth: {bw}')

    def _reduce_model(self, sub):
        sub = sub.sort_values(['task_id', 'query_id'])
        X = np.stack([np.asarray(e, dtype=np.float64) for e in sub['response_embedding']])
        md = {
            'query_ids': sub['query_id'].values,
            'X': self._response_pca.transform(X),
        }
        if self._query_pca is not None:
            Q = np.stack([np.asarray(e, dtype=np.float64) for e in sub['query_embedding']])
            md['Q'] = self._query_pca.transform(Q)
        return md

    def _affinity(self, a, b):
        K_Q = _kernel_matrix(self.query_kwargs['kernel'], self._query_bandwidth,
                             a.get('Q'), b.get('Q'), a['query_ids'], b['query_ids'])
        K_R = _kernel_matrix(self.response_kwargs['kernel'], self._response_bandwidth,
                             a['X'], b['X'])
        Z = K_Q.sum()
        return float((K_Q * K_R).sum() / Z) if Z > 0 else np.nan

    def _recompute_pairs(self, touched):
        names = sorted(self._model_data)
        for i, m1 in enumerate(names):
            for m2 in names[i:]:
                if m1 in touched or m2 in touched or (m1, m2) not in self._A:
                    self._A[(m1, m2)] = self._affinity(self._model_data[m1],
                                                       self._model_data[m2])
        stale = [p for p in self._A if p[0] not in self._model_data
                 or p[1] not in self._model_data]
        for p in stale:
            del self._A[p]

    def _finalize(self):
        names = sorted(self._model_data)
        n = len(names)
        D2 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a_ij = self._A[(names[i], names[j])]
                d2 = self._A[(names[i], names[i])] + self._A[(names[j], names[j])] \
                    - 2 * a_ij
                D2[i, j] = D2[j, i] = max(d2, 0.0)
        if not np.isfinite(D2).all():
            raise ValueError('non-finite PKPS distances (zero query-kernel mass for '
                             'some model pair); increase the query bandwidth')
        D = np.sqrt(D2)
        self.model_names_ = names
        self.dist_matrix_ = D
        if n >= 3:
            coords = ClassicalMDS(n_components=self.mds_kwargs['dim'],
                                  n_elbows=self.mds_kwargs['n_elbows'],
                                  dissimilarity='precomputed').fit_transform(D)
        else:
            coords = D[:, :1]
        self.embedding_ = {m: coords[i] for i, m in enumerate(names)}
        scored = self._raw.dropna(subset=['score'])
        self.sample_scores_ = (scored.groupby(['model_id', 'task_id'])['score']
                               .mean().unstack()
                               .reindex(index=names))
        self.task_names_ = list(self.sample_scores_.columns)

    # ------------------------------------------------------------------
    # predict input
    # ------------------------------------------------------------------
    def _parse_pairs(self, records):
        if records is None:
            S = self.sample_scores_
            return [(m, t) for m in self.model_names_ for t in self.task_names_
                    if pd.isna(S.at[m, t])]
        df = _parse_records(records, required=('model_id',))
        pairs = []
        for _, r in df.iterrows():
            m = r['model_id']
            if m not in self.embedding_:
                raise KeyError(f'unknown model_id: {m!r} (fit() or update() with its '
                               'responses first)')
            if 'task_id' in df.columns and pd.notna(r.get('task_id')):
                pairs.append((m, r['task_id']))
            else:
                pairs.extend((m, t) for t in self.task_names_)
        return pairs


class DKPS(PKPS):
    """Paired special case of PKPS: the query kernel is the delta kernel on query
    ids, so responses are compared only across identical queries. On a fully
    paired design with a linear response kernel this recovers the standard DKPS
    distance (per-query mean squared distance). Queries never need embedding, so
    fit/update records only require response embeddings (or text)."""

    def __init__(self, response_kwargs=None, mds_kwargs=None, embedding_kwargs=None):
        super().__init__(query_kwargs=dict(kernel='delta', pca_dim=None),
                         response_kwargs=response_kwargs, mds_kwargs=mds_kwargs,
                         embedding_kwargs=embedding_kwargs)
