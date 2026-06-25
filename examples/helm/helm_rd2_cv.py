#!/usr/bin/env python
"""
helm_rd2_cv.py  --  RD2 missing tasks with the unified per-run CV (query
bandwidth + predictor + ensemble weight all selected by cross-validation).
Sweeps task_parity (obs_prob) or n_models; emits combined / matcomplete /
ensemble held-out MAE.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dkps.unpaired_dkps import pca_reduce_elbow
import helm_doublekernel as H


def one(resp_X, Qu, qid_code, model_id, task_id, query_id, groups, score_mat,
        models, tasks, obs_prob, n_models_use, seed, suite=False, query_med=None):
    rng = np.random.default_rng(seed)
    if n_models_use and n_models_use < len(models):
        sel = np.sort(rng.choice(len(models), n_models_use, replace=False))
        models = [models[i] for i in sel]
        score_mat = score_mat[sel]
    observed = H.sample_observed(len(models), len(tasks), obs_prob, rng) & np.isfinite(score_mat)
    keep = [groups[(models[i], tasks[t])] for i in range(len(models)) for t in range(len(tasks))
            if observed[i, t] and (models[i], tasks[t]) in groups]
    if not keep:
        return []
    keep = np.concatenate(keep)
    codes = qid_code[keep]
    if suite:
        # response/query embeddings are already block-diagonal (linear k_R) / full Google
        # query space; re-PCA would destroy the block structure, so use them directly.
        df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                           'query_id': query_id[keep], 'embedding': list(resp_X[keep]),
                           'query_embedding': list(Qu[codes])})
        out = H.cv_predict_all(df, score_mat, observed, models, seed=seed,
                               response_kernel='linear', query_med=query_med)
    else:
        rr = pca_reduce_elbow(resp_X[keep])
        uq = np.unique(codes)
        qr = pca_reduce_elbow(Qu[uq])
        c2v = {c: qr[j] for j, c in enumerate(uq)}
        df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                           'query_id': query_id[keep], 'embedding': list(rr),
                           'query_embedding': [c2v[c] for c in codes]})
        out = H.cv_predict_all(df, score_mat, observed, models, seed=seed)
    held = (~observed) & np.isfinite(score_mat)
    rows = []
    for meth in ('combined', 'matcomplete', 'ensemble'):
        p = out[meth]
        h = held & np.isfinite(p)
        mae = float(np.mean(np.abs(p[h] - score_mat[h]))) if h.any() else np.nan
        rows.append(dict(obs_prob=obs_prob, n_models=len(models), seed=seed,
                         method=meth, mae=mae, predictor=out['predictor'], alpha=out['alpha']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--sweep', choices=['task_parity', 'n_models'], default='task_parity')
    ap.add_argument('--obs_probs', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 80, 95])
    ap.add_argument('--fixed_obs_prob', type=float, default=0.3)
    ap.add_argument('--n_seeds', type=int, default=20)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-rd2cv')
    args = ap.parse_args()

    suite = args.dataset == 'suite'
    query_med = None
    if suite:
        data = H.load_suite()
        resp_X, Qu, qid_code, model_id, task_id, query_id, score_mat, models, tasks, groups, _, query_med = data
    elif args.dataset == 'pooled':
        data = H.load_pooled(('math', 'wmt_14'))
        resp_X, Qu, qid_code, model_id, task_id, query_id, score_mat, models, tasks, groups, _ = data
    else:
        cfg = H.DATASETS[args.dataset]
        data = H.load_helm_math(cfg['parquet'], cfg['tsv'], query_source='google',
                                query_parquet=cfg['query_parquet'], score_col=cfg['score_col'])
        resp_X, Qu, qid_code, model_id, task_id, query_id, score_mat, models, tasks, groups, _ = data

    if args.sweep == 'task_parity':
        specs = [(p, None) for p in args.obs_probs]
        xcol = 'obs_prob'
    else:
        specs = [(args.fixed_obs_prob, n) for n in args.n_models_values]
        xcol = 'n_models'
    jobs = [delayed(one)(resp_X, Qu, qid_code, model_id, task_id, query_id, groups,
                         score_mat, models, tasks, p, nm, s, suite, query_med)
            for (p, nm) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for sub in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) for r in sub])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'rd2_{args.dataset}_{args.sweep}.csv', index=False)
    print(res.pivot_table(index=xcol, columns='method', values='mae').round(4).to_string())
    print('predictor picks:', res[res.method == 'combined'].predictor.value_counts().to_dict())


if __name__ == '__main__':
    main()
