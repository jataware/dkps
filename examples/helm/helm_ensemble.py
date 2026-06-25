#!/usr/bin/env python
"""
helm_ensemble.py

Complementarity of DKPS-combined and matrix completion, across the full sweep.
For each sweep point we collect per-held-out-pair (true, dkps, mc) over seeds,
tune the mixing weight alpha on half the seeds, and evaluate
  pred = alpha*dkps + (1-alpha)*mc
on the other half (held-out alpha -> honest ensemble line). Emits a tidy CSV
with combined / matcomplete / ensemble MAE per sweep point.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from graspologic.embed import ClassicalMDS

from dkps.unpaired_dkps import DoubleKernelDKPS, pca_reduce_elbow
from dkps.baselines import matrix_completion_predict
import helm_doublekernel as H


def one_seed(resp_X, Qu, qid_code, model_id, task_id, query_id, groups,
             score_mat, models, tasks, obs_prob, seed, mds_dim, predictor, k,
             n_models_use=None):
    rng = np.random.default_rng(seed)
    if n_models_use is not None and n_models_use < len(models):
        sel = np.sort(rng.choice(len(models), n_models_use, replace=False))
        models = [models[i] for i in sel]
        score_mat = score_mat[sel]
    observed = H.sample_observed(len(models), len(tasks), obs_prob, rng)
    keep = []
    for i in range(len(models)):
        for t in range(len(tasks)):
            if observed[i, t]:
                idx = groups.get((models[i], tasks[t]))
                if idx is not None and len(idx):
                    keep.append(idx)
    keep = np.concatenate(keep)

    resp_red = pca_reduce_elbow(resp_X[keep])
    obs_codes = qid_code[keep]
    uniq = np.unique(obs_codes)
    q_red = pca_reduce_elbow(Qu[uniq])
    code2vec = {c: q_red[j] for j, c in enumerate(uniq)}
    df_obs = pd.DataFrame({
        'model_id': model_id[keep], 'task_id': task_id[keep],
        'query_id': query_id[keep], 'embedding': list(resp_red),
        'query_embedding': [code2vec[c] for c in obs_codes]})

    est = DoubleKernelDKPS(**H.make_estimators()['rbf_combined'])
    est.fit(df_obs)
    dist = est.dist_matrix_.copy()
    name2row = {m: i for i, m in enumerate(est.model_names_)}
    present = [m for m in models if m in name2row]
    if np.any(np.isnan(dist)):
        dist = np.nan_to_num(dist, nan=np.nanmax(dist))
    if len(present) < 3:
        return np.empty((0, 3))
    Z = ClassicalMDS(n_components=min(mds_dim, len(present) - 1),
                     dissimilarity='precomputed').fit_transform(dist)
    full = np.zeros((len(models), Z.shape[1]))
    for m in present:
        full[models.index(m)] = Z[name2row[m]]

    dkps = H.predict_from_embedding(full, score_mat, observed, predictor=predictor,
                                    bias_decomp=True, logit=True, k=k)
    mc = matrix_completion_predict(score_mat, observed)
    held = (~observed & np.isfinite(dkps) & np.isfinite(mc) & np.isfinite(score_mat))
    return np.column_stack([score_mat[held], dkps[held], mc[held]])


def eval_point(data, obs_prob, n_models_use, x_col, x_val, n_seeds, **kw):
    """Run seeds, split into tune/eval halves, tune alpha on tune, eval on eval."""
    parts = Parallel(n_jobs=kw.pop('n_jobs'), verbose=0)(
        delayed(one_seed)(*data, obs_prob, s, kw['mds_dim'], kw['predictor'],
                          kw['k'], n_models_use=n_models_use)
        for s in range(n_seeds))
    tune = np.vstack([parts[s] for s in range(n_seeds) if s % 2 == 0])
    ev = np.vstack([parts[s] for s in range(n_seeds) if s % 2 == 1])
    alphas = np.linspace(0, 1, 51)
    yt, dt, mt = tune.T
    a = alphas[int(np.argmin([np.abs(yt - (al * dt + (1 - al) * mt)).mean()
                              for al in alphas]))]
    ye, de, me = ev.T
    rows = [
        dict(sweep=x_col, x=x_val, estimator='combined', mae=np.abs(ye - de).mean()),
        dict(sweep=x_col, x=x_val, estimator='matcomplete', mae=np.abs(ye - me).mean()),
        dict(sweep=x_col, x=x_val, estimator='ensemble',
             mae=np.abs(ye - (a * de + (1 - a) * me)).mean(), alpha=a),
    ]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--obs_probs', type=float, nargs='+',
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--n_models_values', type=int, nargs='+',
                    default=[10, 20, 40, 60, 80, 95])
    ap.add_argument('--fixed_obs_prob', type=float, default=0.3)
    ap.add_argument('--n_seeds', type=int, default=30)
    ap.add_argument('--mds_dim', type=int, default=8)
    ap.add_argument('--predictor', default='knn')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-doublekernel-google')
    args = ap.parse_args()

    (resp_X, Qu, qid_code, model_id, task_id, query_id,
     score_mat, models, tasks, groups, _row_score) = H.load_helm_math(
        args.parquet, args.tsv, query_source='google')
    # reorder to one_seed's signature (groups before score_mat/models/tasks)
    data = (resp_X, Qu, qid_code, model_id, task_id, query_id,
            groups, score_mat, models, tasks)
    kw = dict(mds_dim=args.mds_dim, predictor=args.predictor, k=args.k)

    rows = []
    for p in args.obs_probs:
        rows += eval_point(data, p, None, 'obs_prob', p, args.n_seeds,
                           n_jobs=args.n_jobs, **kw)
        print(f'obs_prob={p} done')
    for n in args.n_models_values:
        rows += eval_point(data, args.fixed_obs_prob, n, 'n_models', n, args.n_seeds,
                           n_jobs=args.n_jobs, **kw)
        print(f'n_models={n} done')

    df = pd.DataFrame(rows)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.outdir) / 'ensemble_sweep.csv', index=False)
    print(df.to_string(index=False))
    print(f'wrote {args.outdir}/ensemble_sweep.csv')


if __name__ == '__main__':
    main()
