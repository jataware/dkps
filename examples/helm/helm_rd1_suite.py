#!/usr/bin/env python
"""
helm_rd1_suite.py  --  joint RD1: query-efficient evaluation across the whole
heterogeneous suite (MATH + WMT + med_qa + legalbench).

Every model answers only m queries per task; we estimate every (model, task)
full-benchmark score. PKPS builds ONE joint perspective from the sparse responses
(block-diagonal linear k_R, within-domain RBF k_Q) and LOFO-regresses each task's
full score onto it -- borrowing strength across queries and tasks. Baselines:
  - sample : the m-query mean (no denoising).
  - irt    : 1PL Rasch per binary task (math/med_qa/legalbench; not WMT).
All tasks are observed (only queries are missing), so matrix completion does not
apply -- this is PKPS's regime, not MC's.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from graspologic.embed import ClassicalMDS

import helm_doublekernel as H
from helm_qselect import family, _lofo_regress, max_dense_block
from baselines import irt_fit_difficulties, irt_estimate_ability, irt_predict


def run_seed(data, m, seed, n_models=None, p_task=1.0, mds_dim=12, predictor='knn'):
    (resp_X, Qu, qid_code, model_id, task_id, query_id,
     score_mat, models_all, tasks, groups, row_score, qmed) = data
    rng = np.random.default_rng(seed)
    # subsample the cohort (n_models) and the observed cells (coverage p_task); query
    # efficiency scores the OBSERVED cells, so unobserved cells are simply not measured.
    if n_models is not None and n_models < len(models_all):
        sel = np.sort(rng.choice(len(models_all), n_models, replace=False))
        models = [models_all[i] for i in sel]; score_mat = score_mat[sel]
    else:
        models = list(models_all)
    fams = {mm: family(mm) for mm in models}
    binary = {t: bool(np.all(np.isin(np.unique(row_score[task_id == t]), [0.0, 1.0]))) for t in tasks}
    obs = (rng.random((len(models), len(tasks))) < p_task) if p_task < 1.0 \
        else np.ones((len(models), len(tasks)), bool)
    for i in range(len(models)):
        if not obs[i].any(): obs[i, rng.integers(len(tasks))] = True
    for t in range(len(tasks)):
        if not obs[:, t].any(): obs[rng.integers(len(models)), t] = True

    # sample m queries per OBSERVED (model, task); collect their rows
    keep, sample_mat = [], np.full((len(models), len(tasks)), np.nan)
    for i, mm in enumerate(models):
        for t, tt in enumerate(tasks):
            if not obs[i, t]:
                continue
            idx = groups.get((mm, tt))
            if idx is None or not len(idx):
                continue
            take = idx if len(idx) <= m else rng.choice(idx, m, replace=False)
            keep.append(take)
            sample_mat[i, t] = row_score[take].mean()
    keep = np.concatenate(keep)

    # joint perspective from the sparse responses
    codes = qid_code[keep]
    df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                       'query_id': query_id[keep], 'embedding': list(resp_X[keep]),
                       'query_embedding': list(Qu[codes])})
    bw = H.SubsampleMedianBandwidth(max_n=5000)
    est = H.ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel='linear',
                                          query_bandwidth=bw, response_bandwidth=bw)
    # PKPS uses the RBF query kernel at the median bandwidth; DKPS is its delta limit, which
    # dist_matrices reaches as the bandwidth -> 0 (distinct queries stop bridging, so only
    # exact query matches contribute -- standard paired DKPS). One call returns both.
    SIG_DELTA = qmed * 1e-2
    names, Ds = est.dist_matrices(df, [qmed, SIG_DELTA])
    Z = H._mds_full(Ds[qmed], names, models, mds_dim)         # PKPS
    Zd = H._mds_full(Ds[SIG_DELTA], names, models, mds_dim)   # DKPS
    n2r = {mm: i for i, mm in enumerate(models)}

    # PKPS / DKPS perspective regression per task (LOFO ignores the model's own samples)
    def _regress(Zp):
        out = np.full((len(models), len(tasks)), np.nan)
        if Zp is not None:
            for t in range(len(tasks)):
                out[:, t] = _lofo_regress(Zp, n2r, models,
                                          {mm: score_mat[i, t] for i, mm in enumerate(models)},
                                          fams, predictor=predictor)
        return out
    pk_mat = _regress(Z)
    dk_mat = _regress(Zd)
    # ensemble: blend the model's own m-query sample with the cross-model perspective.
    # Fit one shrinkage weight per seed (grid) -- pkps alone discards the direct estimate.
    fin = np.isfinite(pk_mat) & np.isfinite(sample_mat) & np.isfinite(score_mat)
    grid = np.linspace(0, 1, 21)
    if fin.any():
        a = grid[int(np.argmin([np.mean(np.abs(np.clip((1 - g) * pk_mat[fin] + g * sample_mat[fin], 0, 1)
                                                - score_mat[fin])) for g in grid]))]
    else:
        a = 1.0
    ens_mat = np.where(np.isfinite(pk_mat),
                       np.clip((1 - a) * pk_mat + a * sample_mat, 0, 1), sample_mat)

    rows = []
    for t, tt in enumerate(tasks):
        yfull = score_mat[:, t]
        preds = {'sample': sample_mat[:, t], 'dkps': dk_mat[:, t], 'pkps': pk_mat[:, t],
                 'ens': ens_mat[:, t]}
        # IRT per binary task
        if binary[tt]:
            t_rows = keep[task_id[keep] == tt]
            mods_t = sorted(set(model_id[t_rows]))
            mi = {mm: i for i, mm in enumerate(mods_t)}
            qs_t = sorted(set(query_id[t_rows]))
            qi = {q: j for j, q in enumerate(qs_t)}
            S = np.full((len(mods_t), len(qs_t)), np.nan)
            for r in t_rows:
                S[mi[model_id[r]], qi[query_id[r]]] = row_score[r]
            amask = np.isfinite(S)
            irt = np.full(len(models), np.nan)
            F, Q = max_dense_block(amask)
            if len(F) >= 2 and len(Q) >= 1:
                beta_Q, _ = irt_fit_difficulties(np.nan_to_num(S[np.ix_(F, Q)], nan=0.0))
                for mm in mods_t:
                    a = amask[mi[mm], Q]
                    if a.any():
                        irt[n2r[mm]] = irt_predict(
                            irt_estimate_ability(S[mi[mm], Q[a]], beta_Q[a]), beta_Q)
            preds['irt'] = irt
        obs_t = obs[:, t]
        for meth, p in preds.items():
            h = obs_t & np.isfinite(p) & np.isfinite(yfull)
            mae = float(np.mean(np.abs(p[h] - yfull[h]))) if h.any() else np.nan
            rows.append(dict(m=m, n_models=len(models), p_task=p_task, seed=seed, task=tt,
                             dataset=tt.split(':')[0], method=meth, mae=mae))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=['budget', 'n_models', 'coverage'], default='budget')
    ap.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 93])
    ap.add_argument('--coverages', type=float, nargs='+', default=[0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument('--fixed_m', type=int, default=8)
    ap.add_argument('--n_seeds', type=int, default=8)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-rd1')
    args = ap.parse_args()

    data = H.load_suite()
    print(f'suite: {len(data[7])} models, {len(data[8])} tasks')
    # query-efficiency panels: vary p_query (m, full coverage / all models), the cohort n
    # (fixed m, full coverage), or task coverage p_task (fixed m, all models).
    if args.sweep == 'budget':
        specs = [(m, None, 1.0) for m in args.budgets]
    elif args.sweep == 'n_models':
        specs = [(args.fixed_m, n, 1.0) for n in args.n_models_values]
    else:
        specs = [(args.fixed_m, None, p) for p in args.coverages]
    jobs = [delayed(run_seed)(data, m, s, n_models=n, p_task=p)
            for (m, n, p) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for sub in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) for r in sub])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'rd1_suite_{args.sweep}.csv', index=False)
    xcol = {'budget': 'm', 'n_models': 'n_models', 'coverage': 'p_task'}[args.sweep]
    print(f'overall MAE by {xcol}:')
    print(res.pivot_table(index=xcol, columns='method', values='mae').round(4).to_string())


if __name__ == '__main__':
    main()
