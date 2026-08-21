"""High-level PKPS / DKPS estimators with a fit / predict / update API.

Pipeline (matching the paper):

  1. embed queries and responses (optional -- precomputed embeddings are accepted),
  2. PCA-reduce each channel (fixed dim, or Zhu-Ghodsi elbow selection),
  3. product-kernel affinities  A_ab = sum_{j,l} k_Q(u_j, u_l) k_R(x_aj, x_bl) / sum_{j,l} k_Q(u_j, u_l),
  4. squared distances          D^2_ab = A_aa + A_bb - 2 A_ab,
  5. classical MDS on D  ->  model representations Z,
  6. per-task k-NN regression from Z onto the other models' scores.

`DKPS` is the paired special case: k_Q is the delta kernel on query ids, which on a
fully paired design recovers the standard DKPS distance (mean per-query squared
distance under a linear response kernel).

Records are exchanged as JSON-friendly tables: a list of dicts, a dict of lists, a
DataFrame, or a JSON string of either. Fit/update rows carry
(model_id, task_id, query_id, query|query_embedding, response|response_embedding, score
[, reference_score]); predict rows carry (model_id[, task_id]).
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import KNeighborsRegressor
from graspologic.embed import ClassicalMDS

from .preprocessing import FrozenPCA, Whitener
from .records import parse_records, merge_records, parse_pairs, pairs_to_records, ScoreTable


QUERY_DEFAULTS = dict(kernel='rbf', bandwidth='median', bandwidth_ref=None, bandwidth_mult=1.0,
                      bandwidth_grid=(0.03, 0.1, 0.3, 1.0, 3.0), cv_k=8,
                      pca_dim='elbow', pca_n_elbows=2)
RESPONSE_DEFAULTS = dict(kernel='linear', bandwidth='median', bandwidth_ref=None, bandwidth_mult=1.0,
                         pca_dim='elbow', pca_n_elbows=2)
MDS_DEFAULTS = dict(dim=None, n_elbows=2)
EMBEDDING_DEFAULTS = dict(provider='google', model=None, api_key=None)

_PROVIDER_ENV = {
    'google': 'GEMINI_API_KEY',
    'jina': 'JINA_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
    'huggingface': 'HF_TOKEN',
}


def family(model_id):
    """Default model-family function: the developer prefix ('openai_gpt-4' -> 'openai',
    'allenai/olmo-3' -> 'allenai')."""
    m = str(model_id)
    return m.split('/')[0] if '/' in m else m.split('_')[0]


def _median_distance(X, cap=5000, random_state=0):
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


def _lomo_knn_error(Z, y, k):
    """Leave-one-model-out k-NN error of predicting y from Z (inverse-distance weights);
    the bandwidth-selection criterion."""
    D2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    errs = []
    for a in range(len(y)):
        nbr = np.argsort(D2[a])[:k]
        w = 1.0 / (np.sqrt(D2[a][nbr]) + 1e-9)
        errs.append(abs(np.average(y[nbr], weights=w) - y[a]))
    return errs


class PKPS:
    """Product Kernel Perspective Space estimator.

    Parameters
    ----------
    query_kwargs : dict, optional
        Query channel configuration (defaults in QUERY_DEFAULTS):
          kernel : 'rbf' | 'delta' | 'linear' | 'cosine' | callable
          bandwidth : 'median' | 'cv' | float. 'median' uses bandwidth_mult x the
              reference scale; 'cv' picks, from bandwidth_grid x the reference scale, the
              bandwidth whose representation best predicts the observed sample scores
              under leave-one-model-out k-NN (cv_k neighbours) -- the paper's protocol.
          bandwidth_ref : reference scale for 'median'/'cv' (None = median pairwise
              distance of the (reduced) query embeddings; the paper passes the
              within-domain median).
          bandwidth_mult, bandwidth_grid, cv_k : see above
          pca_dim : 'elbow' (Zhu-Ghodsi selection) | int | None (no reduction)
          pca_n_elbows : int, elbow index used when pca_dim='elbow'
    response_kwargs : dict, optional
        Response channel configuration; same keys minus the CV ones
        (defaults in RESPONSE_DEFAULTS).
    mds_kwargs : dict, optional
        Classical MDS on the model distance matrix:
          dim : int (capped at n_models - 1; the paper uses 8) or None (Zhu-Ghodsi)
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
    model_names_, task_names_ : ids
    dist_matrix_ : (n, n) PKPS distance matrix at the selected bandwidth
    embedding_ : dict model_id -> MDS coordinates
    query_bandwidth_ : selected query bandwidth (None for non-RBF kernels)
    sample_scores_ : DataFrame (model x task) of observed sample scores
    reference_scores_ : DataFrame of cell-level reference scores (NaN where not given)
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

        Each record: model_id, task_id, query_id, a response (text under 'response', or
        a vector under 'response_embedding'/'embedding'), a query (text under 'query',
        or vector under 'query_embedding'; required unless the query kernel is
        'delta'), an optional per-response 'score', and an optional cell-level
        'reference_score' (a known benchmark score for that (model, task), e.g. from a
        fully evaluated cache; used as the regression target in place of the sample
        score when present).
        """
        df = self._ingest(records)
        self._raw = df
        self._fit_reducers(df)
        self._model_data = {m: self._reduce_model(df[df['model_id'] == m])
                            for m in sorted(df['model_id'].unique())}
        self._A = {s: {} for s in self._sigmas}
        self._recompute_pairs(set(self._model_data))
        self._finalize()
        self._fitted = True
        return self

    def update(self, records):
        """Add records (new models, or new responses for existing models) and refresh
        the fit.

        The PCA bases and kernel bandwidth scale learned at fit time are frozen, so only
        affinities involving updated models are recomputed; the bandwidth selection
        (if 'cv'), the MDS and the score tables are refreshed. Re-run fit() to relearn
        the reducers.
        """
        assert self._fitted, 'call fit() before update()'
        df = self._ingest(records)
        touched = set(df['model_id'].unique())
        self._raw, n_rep = merge_records(self._raw, df)
        if n_rep:
            warnings.warn(f'update: {n_rep} (model, task, query) rows replaced by their '
                          'newer version', stacklevel=2)
        for m in touched:
            self._model_data[m] = self._reduce_model(self._raw[self._raw['model_id'] == m])
        self._recompute_pairs(touched)
        self._finalize()
        return self

    def predict(self, records=None, k=5, holdout='model', family_fn=None, whiten=False,
                target='auto', min_train=None):
        """Predict scores for (model, task) pairs by per-task k-NN regression
        (inverse-distance weights) over the MDS representation.

        records : None (every cell without an observed sample score), or records with
            'model_id' and optionally 'task_id' (a model without task_id requests every task).
        holdout : 'model' -- train on every other model with a score on the task (leave-one-
            model-out); 'family' -- additionally exclude models from the target's family
            (leave-one-family-out, the paper's query-efficiency protocol).
        family_fn : model_id -> family label (default: developer prefix).
        whiten : route targets through the logit / standardize / two-way-bias Whitener
            fitted on the observed score matrix, regress the residual, and invert (the
            paper's completion protocol).
        target : 'auto' (reference score where given, else sample score), 'sample', or
            'reference'.
        min_train : minimum support per task (default 3 when holdout='family' or whiten,
            else 1); fewer -> NaN. Under holdout='model' this counts the task's observed
            models (the target included, if observed); under 'family' it counts the
            training models outside the target's family.
        Returns a list of dicts with 'score_hat' added.
        """
        assert self._fitted, 'call fit() before predict()'
        if min_train is None:
            min_train = 3 if (holdout == 'family' or whiten) else 1
        fam = family_fn or family
        pairs = self._parse_pairs(records)
        names = self.model_names_
        idx = {m: i for i, m in enumerate(names)}
        Z = np.stack([self.embedding_[m] for m in names])
        Y = self._target_matrix(target)                        # (n_models x n_tasks)
        obs = np.isfinite(Y)
        if whiten:
            w = Whitener().fit(Y, obs)
            T = w.transform(Y)
        else:
            T = Y
        tcol = {t: j for j, t in enumerate(self.task_names_)}
        fams = np.array([fam(m) for m in names])
        preds = {}
        for m, t in pairs:
            if t not in tcol:
                preds[(m, t)] = np.nan
                continue
            j = tcol[t]
            i = idx[m]
            tr = obs[:, j].copy()
            tr[i] = False
            if holdout == 'family':
                tr &= fams != fams[i]
            elif holdout != 'model':
                raise ValueError(f'unknown holdout: {holdout}')
            rows = np.where(tr)[0]
            support = len(rows) if holdout == 'family' else int(obs[:, j].sum())
            if len(rows) < 1 or support < min_train:
                preds[(m, t)] = np.nan
                continue
            mdl = KNeighborsRegressor(n_neighbors=min(k, len(rows)), weights='distance')
            r = float(mdl.fit(Z[rows], T[rows, j]).predict(Z[i][None])[0])
            if whiten:
                R = np.full_like(Y, np.nan); R[i, j] = r
                r = float(w.inverse_transform(R)[i, j])
            preds[(m, t)] = r
        return pairs_to_records(pairs, preds)

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def _needs_query_channel(self):
        return self.query_kwargs['kernel'] != 'delta'

    def _ingest(self, records):
        df = parse_records(records, required=('model_id', 'query_id'),
                           aliases=[('embedding', 'response_embedding')])
        if 'task_id' not in df.columns:
            df['task_id'] = '_task'
        for col in ('score', 'reference_score', 'sample_score'):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'response_embedding' not in df.columns:
            if 'response' not in df.columns:
                raise ValueError("records need 'response_embedding' vectors or 'response' text")
            df['response_embedding'] = list(self._embed_texts(df['response'].tolist()))
        if self._needs_query_channel() and 'query_embedding' not in df.columns:
            if 'query' not in df.columns:
                raise ValueError("records need 'query_embedding' vectors or 'query' text "
                                 "(query kernel is not 'delta')")
            df['query_embedding'] = list(self._embed_texts(df['query'].tolist()))

        keep = ['model_id', 'task_id', 'query_id', 'response_embedding', 'score',
                'reference_score', 'sample_score']
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
        self._response_pca = FrozenPCA(self.response_kwargs['pca_dim'],
                                       self.response_kwargs['pca_n_elbows']).fit(R)
        self._response_bandwidth = None
        if self.response_kwargs['kernel'] == 'rbf':
            bw = self.response_kwargs['bandwidth']
            if bw == 'cv':
                raise ValueError("bandwidth='cv' is only supported on the query channel")
            self._response_bandwidth = self._scale(self.response_kwargs,
                                                   self._response_pca.transform(R)) \
                if not isinstance(bw, (int, float)) else float(bw)

        self._query_pca = None
        self._sigmas = [None]
        if 'query_embedding' in df.columns:
            Q = np.stack([np.asarray(e, dtype=np.float64) for e in df['query_embedding']])
            self._query_pca = FrozenPCA(self.query_kwargs['pca_dim'],
                                        self.query_kwargs['pca_n_elbows']).fit(Q)
            if self.query_kwargs['kernel'] == 'rbf':
                bw = self.query_kwargs['bandwidth']
                if isinstance(bw, (int, float)):
                    self._sigmas = [float(bw)]
                else:
                    ref = self._scale(self.query_kwargs, self._query_pca.transform(Q))
                    if bw == 'median':
                        self._sigmas = [ref]
                    elif bw == 'cv':
                        self._sigmas = [float(ref * g) for g in self.query_kwargs['bandwidth_grid']]
                    else:
                        raise ValueError(f'unknown query bandwidth: {bw}')

    @staticmethod
    def _scale(spec, X):
        """Reference bandwidth scale: bandwidth_mult x (bandwidth_ref or median distance)."""
        ref = spec['bandwidth_ref']
        if ref is None:
            ref = _median_distance(X)
        elif callable(ref):
            ref = float(ref(X))
        return float(spec['bandwidth_mult']) * float(ref)

    def _reduce_model(self, sub):
        sub = sub.sort_values(['task_id', 'query_id'])
        X = np.stack([np.asarray(e, dtype=np.float64) for e in sub['response_embedding']])
        md = {'query_ids': sub['query_id'].values, 'X': self._response_pca.transform(X)}
        if self._query_pca is not None:
            Q = np.stack([np.asarray(e, dtype=np.float64) for e in sub['query_embedding']])
            md['Q'] = self._query_pca.transform(Q)
        return md

    def _affinities(self, a, b):
        """{sigma: A_ab} for every candidate query bandwidth. The response kernel and the
        query squared distances are bandwidth-independent, so they are computed once."""
        K_R = _kernel_matrix(self.response_kwargs['kernel'], self._response_bandwidth,
                             a['X'], b['X'])
        qk = self.query_kwargs['kernel']
        if qk == 'rbf':
            q_sq = cdist(a['Q'], b['Q'], 'sqeuclidean')
            out = {}
            for s in self._sigmas:
                K_Q = np.exp(-q_sq / (2.0 * s * s))
                Zm = K_Q.sum()
                out[s] = float((K_Q * K_R).sum() / Zm) if Zm > 0 else np.nan
            return out
        K_Q = _kernel_matrix(qk, None, a.get('Q'), b.get('Q'), a['query_ids'], b['query_ids'])
        Zm = K_Q.sum()
        return {None: float((K_Q * K_R).sum() / Zm) if Zm > 0 else np.nan}

    def _recompute_pairs(self, touched):
        names = sorted(self._model_data)
        A0 = self._A[self._sigmas[0]]
        for i, m1 in enumerate(names):
            for m2 in names[i:]:
                if m1 in touched or m2 in touched or (m1, m2) not in A0:
                    for s, v in self._affinities(self._model_data[m1], self._model_data[m2]).items():
                        self._A[s][(m1, m2)] = v
        for s in self._sigmas:
            for p in [p for p in self._A[s] if p[0] not in self._model_data
                      or p[1] not in self._model_data]:
                del self._A[s][p]

    def _dist_matrix(self, s, names):
        n = len(names)
        A = self._A[s]
        D2 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d2 = A[(names[i], names[i])] + A[(names[j], names[j])] - 2 * A[(names[i], names[j])]
                D2[i, j] = D2[j, i] = max(d2, 0.0) if np.isfinite(d2) else np.nan
        D = np.sqrt(D2)
        if np.isnan(D).any():
            mx = np.nanmax(D)
            if not np.isfinite(mx):
                raise ValueError('all PKPS distances undefined (zero query-kernel mass); '
                                 'increase the query bandwidth')
            warnings.warn('model pairs with zero query-kernel mass: their distance is set to '
                          'the maximum observed distance', stacklevel=3)
            D = np.nan_to_num(D, nan=mx)
        return D

    def _mds(self, D):
        n = len(D)
        if n < 3:
            return D[:, :1]
        dim = self.mds_kwargs['dim']
        return ClassicalMDS(n_components=None if dim is None else min(int(dim), n - 1),
                            n_elbows=self.mds_kwargs['n_elbows'],
                            dissimilarity='precomputed').fit_transform(D)

    def _finalize(self):
        names = sorted(self._model_data)
        self.model_names_ = names
        table = ScoreTable(self._raw, models=names)
        self.sample_scores_ = table.sample_
        self.task_names_ = table.tasks
        ref = self._raw.dropna(subset=['reference_score']).groupby(['model_id', 'task_id'])[
            'reference_score'].first().unstack()
        self.reference_scores_ = ref.reindex(index=names, columns=self.task_names_)

        Ds = {s: self._dist_matrix(s, names) for s in self._sigmas}
        Zs = {s: self._mds(D) for s, D in Ds.items()}
        if len(self._sigmas) > 1:                                   # CV bandwidth selection
            Y = self.sample_scores_.to_numpy(dtype=float)
            k = int(self.query_kwargs['cv_k'])
            best_s, best_e = None, np.inf
            for s in self._sigmas:
                errs = []
                for j in range(Y.shape[1]):
                    rows = np.where(np.isfinite(Y[:, j]))[0]
                    if len(rows) < k + 1:
                        continue
                    errs += _lomo_knn_error(Zs[s][rows], Y[rows, j], k)
                e = float(np.mean(errs)) if errs else np.inf
                if e < best_e:
                    best_e, best_s = e, s
            if best_s is None:                                      # no task had enough models
                best_s = self._sigmas[len(self._sigmas) // 2]
        else:
            best_s = self._sigmas[0]
        self.query_bandwidth_ = best_s
        self.dist_matrix_ = Ds[best_s]
        self.embedding_ = {m: Zs[best_s][i] for i, m in enumerate(names)}
        self.embeddings_by_bandwidth_ = {s: {m: Z[i] for i, m in enumerate(names)}
                                         for s, Z in Zs.items()}

    def _target_matrix(self, target):
        S = self.sample_scores_.to_numpy(dtype=float)
        R = self.reference_scores_.to_numpy(dtype=float)
        if target == 'sample':
            return S
        if target == 'reference':
            return R
        if target == 'auto':
            return np.where(np.isfinite(R), R, S)
        raise ValueError(f'unknown target: {target}')

    def _parse_pairs(self, records):
        S = self.sample_scores_
        default = ((m, t) for m in self.model_names_ for t in self.task_names_
                   if pd.isna(S.at[m, t]))
        return parse_pairs(records, self.model_names_, self.task_names_, default)


class DKPS(PKPS):
    """Paired special case of PKPS.

    By default the query kernel is the delta kernel on query ids, so responses are
    compared only across identical queries; on a fully paired design with a linear
    response kernel this recovers the standard DKPS distance (per-query mean squared
    distance), and queries never need embedding. Passing sigma_ratio (e.g. 0.01, the
    paper's "delta limit") instead uses an RBF query kernel at sigma_ratio x the
    reference query scale, which requires query embeddings.
    """

    def __init__(self, response_kwargs=None, mds_kwargs=None, embedding_kwargs=None,
                 sigma_ratio=None, bandwidth_ref=None):
        if sigma_ratio is None:
            qk = dict(kernel='delta', pca_dim=None)
        else:
            qk = dict(kernel='rbf', bandwidth='median', bandwidth_mult=float(sigma_ratio),
                      bandwidth_ref=bandwidth_ref, pca_dim=None)
        super().__init__(query_kwargs=qk, response_kwargs=response_kwargs,
                         mds_kwargs=mds_kwargs, embedding_kwargs=embedding_kwargs)
