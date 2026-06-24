#!/usr/bin/env python
"""
helm_doublekernel.py

Investigate DoubleKernelDKPS on real HELM data (MATH benchmark).

The MATH data is fully paired: 95 models x 7 subjects (tasks), every model
answered every instance of every subject. We artificially induce *task-level*
missingness (drop each (model, task) pair with prob 1 - obs_prob) and ask the
same question as the synthetic experiment: does the *combined* product-kernel
estimator predict held-out per-subject scores better than paired/unpaired when
task overlap is sparse?

Data ingredients
----------------
- response embeddings : exports/math_google_embeddings.parquet
    google gemini-embedding-001, 3072-dim, one row per (model, subject, instance)
- query text + scores : data/math.tsv  (joined on dataset, model, instance_id)
- query embeddings    : TF-IDF(query text) -> TruncatedSVD, offline & independent
                        of the responses being compared (no circularity).

Evaluation
----------
For each obs_prob and seed:
  1. sample observation mask over (model, task)
  2. build long-form observed response table
  3. for each estimator (paired/unpaired/combined): DoubleKernelDKPS -> MDS ->
     KNN regression predicting held-out (model, task) scores -> RMSE
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsRegressor
from graspologic.embed import ClassicalMDS

from dkps.unpaired_dkps import DoubleKernelDKPS


class SubsampleMedianBandwidth:
    """Median pairwise distance over a random row subsample (bounds pdist cost/memory).

    A picklable callable (works with joblib/loky) usable as query/response bandwidth.
    """
    def __init__(self, max_n=5000, seed=0):
        self.max_n = max_n
        self.seed = seed

    def __call__(self, X):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_n:
            X = X[rng.choice(len(X), self.max_n, replace=False)]
        d = pdist(X)
        return float(np.median(d)) if len(d) else 1.0


# Both kernels RBF (query enables cross-task bridging; response RBF for kernel
# consistency). Subsampled-median bandwidth keeps the high-obs_prob jobs cheap.
_BW = SubsampleMedianBandwidth(max_n=5000)
_KERNELS = dict(query_kernel='rbf', response_kernel='rbf',
                query_bandwidth=_BW, response_bandwidth=_BW)
ESTIMATORS = {
    'rbf_paired':   dict(**_KERNELS, task_filter='shared'),
    'rbf_unpaired': dict(**_KERNELS, task_filter='unshared'),
    'rbf_combined': dict(**_KERNELS),
}


def load_helm_math(parquet_path, tsv_path, query_dim=64, response_dim=None,
                   l2_normalize=True, query_source='mean_response', seed=0):
    """Build the per-row response table, query embeddings, and score matrix.

    query_source:
      'mean_response' — query embedding = mean over models of the (real, cached)
                        google response embedding for that instance, SVD-reduced.
                        Uses real embeddings, no scores (no leakage), no API call.
      'tfidf'         — TF-IDF over query text, SVD-reduced (offline baseline).
    """
    emb = pd.read_parquet(parquet_path)  # dataset, model, instance_id, response, embedding
    meta = pd.read_csv(tsv_path, sep='\t')[['dataset', 'model', 'instance_id', 'query', 'score']]

    df = emb.merge(meta, on=['dataset', 'model', 'instance_id'], how='inner')
    df['task_id'] = df['dataset']
    df['model_id'] = df['model']
    df['query_id'] = df['dataset'] + '::' + df['instance_id'].astype(str)

    X_raw = np.stack(df['embedding'].values).astype(np.float64)  # raw google response emb

    X = X_raw
    if response_dim is not None and response_dim < X.shape[1]:
        X = TruncatedSVD(n_components=response_dim, random_state=seed).fit_transform(X)
    if l2_normalize:
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    df['embedding'] = list(X)

    # --- Query embeddings (RBF query kernel input) ---
    if query_source == 'google':
        # Real google embeddings of the bare per-instance problem text (model-
        # specific presentation stripped); see embed_math_queries.py. One vector
        # per instance, keyed by query_id.
        qpath = Path('exports/math_query_google_embeddings.parquet')
        assert qpath.exists(), f'{qpath} missing — run embed_math_queries.py first'
        qdf = pd.read_parquet(qpath)
        Qraw = np.stack(qdf['query_embedding'].values).astype(np.float64)
        qd = min(query_dim, Qraw.shape[1], Qraw.shape[0] - 1)
        Qd = TruncatedSVD(n_components=qd, random_state=seed).fit_transform(Qraw)
        qmap = {qid: vec for qid, vec in zip(qdf['query_id'].values, Qd)}
        df['query_embedding'] = [qmap[qid] for qid in df['query_id']]
    elif query_source == 'mean_response':
        # Per-instance mean of the raw google response embeddings (real, cached).
        tmp = pd.DataFrame(X_raw)
        tmp['query_id'] = df['query_id'].values
        means = tmp.groupby('query_id').mean()
        Qd = TruncatedSVD(
            n_components=min(query_dim, means.shape[1], means.shape[0] - 1),
            random_state=seed).fit_transform(means.values)
        qmap = {qid: vec for qid, vec in zip(means.index.values, Qd)}
        df['query_embedding'] = [qmap[qid] for qid in df['query_id']]
    elif query_source == 'tfidf':
        q = df[['query_id', 'query']].drop_duplicates('query_id').reset_index(drop=True)
        tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words='english')
        Q = tfidf.fit_transform(q['query'].fillna('').astype(str))
        Qd = TruncatedSVD(n_components=min(query_dim, Q.shape[1] - 1),
                          random_state=seed).fit_transform(Q)
        qmap = {qid: vec for qid, vec in zip(q['query_id'], Qd)}
        df['query_embedding'] = [qmap[qid] for qid in df['query_id']]
    else:
        raise ValueError(f'unknown query_source: {query_source}')

    # Ground-truth score matrix: mean score per (model, task).
    models = sorted(df['model_id'].unique())
    tasks = sorted(df['task_id'].unique())
    m_idx = {m: i for i, m in enumerate(models)}
    t_idx = {t: i for i, t in enumerate(tasks)}
    score_mat = np.full((len(models), len(tasks)), np.nan)
    g = df.groupby(['model_id', 'task_id'])['score'].mean()
    for (m, t), s in g.items():
        score_mat[m_idx[m], t_idx[t]] = s

    cols = ['model_id', 'task_id', 'query_id', 'embedding', 'query_embedding', 'score']
    return df[cols].reset_index(drop=True), score_mat, models, tasks


def sample_observed(n_models, n_tasks, obs_prob, rng):
    """Boolean (model, task) mask; guarantee >=1 obs per row and per column."""
    observed = rng.random((n_models, n_tasks)) < obs_prob
    for i in range(n_models):
        if not observed[i].any():
            observed[i, rng.integers(n_tasks)] = True
    for k in range(n_tasks):
        if not observed[:, k].any():
            observed[rng.integers(n_models), k] = True
    return observed


def predict_scores_knn(embeddings, score_mat, observed, k=5):
    preds = np.full_like(score_mat, np.nan)
    n_tasks = score_mat.shape[1]
    for t in range(n_tasks):
        train = observed[:, t]
        test = ~observed[:, t]
        if train.sum() < 1 or test.sum() < 1:
            continue
        k_eff = min(k, int(train.sum()))
        knn = KNeighborsRegressor(n_neighbors=k_eff, weights='distance')
        knn.fit(embeddings[train], score_mat[train, t])
        preds[test, t] = knn.predict(embeddings[test])
    return preds


def rmse_heldout(preds, score_mat, observed):
    held = ~observed & np.isfinite(preds) & np.isfinite(score_mat)
    if held.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((score_mat[held] - preds[held]) ** 2)))


def fit_predict(df_obs, score_mat, observed, models, n_components, k=5, **dkps_kwargs):
    est = DoubleKernelDKPS(**dkps_kwargs)
    est.fit(df_obs)
    dist = est.dist_matrix_.copy()

    # Reindex distance matrix to the full model list (observed-only models present).
    name2row = {m: i for i, m in enumerate(est.model_names_)}
    present = [m for m in models if m in name2row]
    if np.any(np.isnan(dist)):
        mx = np.nanmax(dist)
        if not np.isfinite(mx):
            return np.nan
        dist = np.nan_to_num(dist, nan=mx)

    nc = min(n_components, len(present) - 1)
    if nc < 1:
        return np.nan
    Z = ClassicalMDS(n_components=nc, dissimilarity='precomputed').fit_transform(dist)

    # Map embeddings back to full-model array (all models observed >=1 task, so all present).
    full = np.zeros((len(models), Z.shape[1]))
    for m, i in zip(present, [name2row[m] for m in present]):
        full[models.index(m)] = Z[i]

    preds = predict_scores_knn(full, score_mat, observed, k=k)
    return rmse_heldout(preds, score_mat, observed)


def run_one(df, groups, score_mat, models, tasks, obs_prob, query_obs_prob,
            seed, n_components, k, n_models_use=None):
    """One (obs_prob, query_obs_prob, n_models_use, seed) replicate, all estimators.

    Task-level sparsity: each (model,task) observed w.p. obs_prob.
    Query-level sparsity: within an observed (model,task), each instance is kept
    w.p. query_obs_prob (>=1 guaranteed). query_obs_prob=1.0 keeps all instances.
    Model count: if n_models_use is set, restrict to a random subset of models.
    """
    rng = np.random.default_rng(seed)
    if n_models_use is not None and n_models_use < len(models):
        sel = np.sort(rng.choice(len(models), n_models_use, replace=False))
        models = [models[i] for i in sel]
        score_mat = score_mat[sel]
    observed = sample_observed(len(models), len(tasks), obs_prob, rng)

    keep_idx = []
    for i in range(len(models)):
        for t in range(len(tasks)):
            if not observed[i, t]:
                continue
            idx = groups.get((models[i], tasks[t]))
            if idx is None or len(idx) == 0:
                continue
            if query_obs_prob < 1.0:
                m = rng.random(len(idx)) < query_obs_prob
                if not m.any():
                    m[rng.integers(len(idx))] = True
                idx = idx[m]
            keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx) if keep_idx else np.array([], dtype=int)
    df_obs = df.iloc[keep_idx].drop(columns=['score']).reset_index(drop=True)

    rows = []
    for name, kw in ESTIMATORS.items():
        rmse = fit_predict(df_obs, score_mat, observed, models, n_components, k=k, **kw)
        rows.append(dict(estimator=name, obs_prob=obs_prob,
                         query_obs_prob=query_obs_prob, n_models=len(models),
                         seed=seed, rmse=rmse))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default='exports/math_google_embeddings.parquet')
    ap.add_argument('--tsv', default='data/math.tsv')
    ap.add_argument('--sweep',
                    choices=['task_parity', 'query_sparsity', 'n_models'],
                    default='task_parity')
    ap.add_argument('--obs_probs', type=float, nargs='+',
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--query_obs_probs', type=float, nargs='+',
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument('--n_models_values', type=int, nargs='+',
                    default=[10, 20, 40, 60, 80, 95])
    ap.add_argument('--fixed_obs_prob', type=float, default=0.3,
                    help='task obs_prob held fixed during query_sparsity / n_models sweeps')
    ap.add_argument('--n_seeds', type=int, default=25)
    ap.add_argument('--n_components', type=int, default=8)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--query_dim', type=int, default=64)
    ap.add_argument('--query_source', choices=['google', 'mean_response', 'tfidf'],
                    default='google')
    ap.add_argument('--response_dim', type=int, default=256)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-doublekernel')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('loading HELM math data ...')
    df, score_mat, models, tasks = load_helm_math(
        args.parquet, args.tsv, query_dim=args.query_dim,
        response_dim=args.response_dim, query_source=args.query_source)
    print(f'  query kernel source: {args.query_source}')
    print(f'  {len(models)} models, {len(tasks)} tasks, {len(df)} response rows')
    print(f'  score range [{np.nanmin(score_mat):.3f}, {np.nanmax(score_mat):.3f}]')

    # Precompute row indices per (model, task) once (shared, read-only).
    groups = {k: g.index.to_numpy()
              for k, g in df.groupby(['model_id', 'task_id'])}

    if args.sweep == 'task_parity':
        x_col = 'obs_prob'
        specs = [dict(obs_prob=p, query_obs_prob=1.0, n_models_use=None)
                 for p in args.obs_probs]
        print(f'  sweep: task_parity over obs_prob={args.obs_probs}')
    elif args.sweep == 'query_sparsity':
        x_col = 'query_obs_prob'
        specs = [dict(obs_prob=args.fixed_obs_prob, query_obs_prob=q, n_models_use=None)
                 for q in args.query_obs_probs]
        print(f'  sweep: query_sparsity over query_obs_prob={args.query_obs_probs} '
              f'(obs_prob fixed at {args.fixed_obs_prob})')
    else:  # n_models
        x_col = 'n_models'
        specs = [dict(obs_prob=args.fixed_obs_prob, query_obs_prob=1.0, n_models_use=n)
                 for n in args.n_models_values]
        print(f'  sweep: n_models over {args.n_models_values} '
              f'(obs_prob fixed at {args.fixed_obs_prob})')

    jobs = [delayed(run_one)(df, groups, score_mat, models, tasks,
                             seed=s, n_components=args.n_components, k=args.k, **spec)
            for spec in specs for s in range(args.n_seeds)]
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(jobs)
    res = pd.DataFrame([r for sub in nested for r in sub])
    res.to_csv(outdir / f'{args.sweep}.csv', index=False)

    summary = (res.groupby([x_col, 'estimator'])['rmse']
               .agg(['mean', 'std', 'count']).reset_index())
    print(summary.to_string(index=False))
    summary.to_csv(outdir / f'{args.sweep}_summary.csv', index=False)
    print(f'wrote {outdir}/{args.sweep}.csv')


if __name__ == '__main__':
    main()
