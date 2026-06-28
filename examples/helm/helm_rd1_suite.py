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


def _cv_bandwidth(Zs, sample_mat, obs, k=8):
    """Choose the query bandwidth whose perspective best predicts the OBSERVED (noisy) scores
    under leave-one-model-out kNN. Leakage-free: it only uses observed sample means, never the
    held-out full-eval scores we report, so it is not circular with the evaluation metric."""
    best_sg, best_err = None, np.inf
    for sg, Z in Zs.items():
        if Z is None:
            continue
        errs = []
        for t in range(sample_mat.shape[1]):
            idx = np.where(obs[:, t] & np.isfinite(sample_mat[:, t]))[0]
            if len(idx) < k + 1:
                continue
            Zt, yt = Z[idx], sample_mat[idx, t]
            D2 = ((Zt[:, None] - Zt[None]) ** 2).sum(-1)
            np.fill_diagonal(D2, np.inf)
            for a in range(len(idx)):
                nbr = np.argsort(D2[a])[:k]
                w = 1.0 / (np.sqrt(D2[a][nbr]) + 1e-9)
                errs.append(abs(np.average(yt[nbr], weights=w) - yt[a]))
        e = float(np.mean(errs)) if errs else np.inf
        if e < best_err:
            best_err, best_sg = e, sg
    return best_sg


def run_seed(data, m, seed, n_models=None, p_task=1.0, n_paired=None, mds_dim=12, predictor='knn',
             dump_cells=False):
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

    # pairing experiment: a fixed budget m per cell split into n_paired SHARED queries (the
    # same query_ids for every model) and m-n_paired UNIQUE queries (resampled per model).
    # rho = n_paired/m slides from fully paired (DKPS works) to fully unpaired (DKPS collapses).
    # per task: n_paired shared query_ids (all models), plus DISJOINT blocks of (m-n_paired)
    # unique query_ids per model (cycling only if the pool is exhausted), so unpaired really
    # means non-overlapping -- otherwise small pools leak coincidental pairing to DKPS.
    task_want = None
    if n_paired is not None:
        u = max(m - n_paired, 0)
        task_want = {}
        for tt in tasks:
            pool = sorted(set(query_id[task_id == tt]))
            shared = set(pool[:n_paired])
            ns = np.array(pool[n_paired:], dtype=object)
            ns = ns[rng.permutation(len(ns))] if len(ns) else ns
            assign = {}
            for i in range(len(models)):
                uq = set(ns[(i * u + np.arange(u)) % len(ns)]) if (u and len(ns)) else set()
                assign[i] = shared | uq
            task_want[tt] = assign

    # sample m queries per OBSERVED (model, task); collect their rows and query depth
    keep, sample_mat = [], np.full((len(models), len(tasks)), np.nan)
    ncnt = np.zeros((len(models), len(tasks)))
    for i, mm in enumerate(models):
        for t, tt in enumerate(tasks):
            if not obs[i, t]:
                continue
            idx = groups.get((mm, tt))
            if idx is None or not len(idx):
                continue
            if task_want is not None:
                take = idx[np.isin(query_id[idx], list(task_want[tt][i]))]
                if not len(take):
                    take = idx[:1]
            else:
                take = idx if len(idx) <= m else rng.choice(idx, m, replace=False)
            keep.append(take)
            sample_mat[i, t] = row_score[take].mean()
            ncnt[i, t] = len(take)
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
    # PKPS picks its RBF query bandwidth by cross-validation over a grid (best at predicting the
    # OBSERVED scores, leave-one-model-out); DKPS is the fixed delta limit. The grid spans
    # near-delta to broad, so CV can shrink toward DKPS when pairing is abundant.
    SIG_DELTA = qmed * 1e-2
    SIG_GRID = list(qmed * np.array([0.03, 0.1, 0.3, 1.0, 3.0]))
    names, Ds = est.dist_matrices(df, SIG_GRID + [SIG_DELTA])
    Zs = {sg: H._mds_full(Ds[sg], names, models, mds_dim) for sg in SIG_GRID}
    sig_star = _cv_bandwidth(Zs, sample_mat, obs)
    Z = Zs.get(sig_star)                                       # PKPS (CV bandwidth)
    if Z is None:
        Z = H._mds_full(Ds[SIG_GRID[2]], names, models, mds_dim)
    Zd = H._mds_full(Ds[SIG_DELTA], names, models, mds_dim)   # DKPS (delta limit)
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
    # ensemble: precision blend of the cell's own m-query sample against the PKPS prior, PER
    # cell by its query depth -- weight on the sample = prior_err/(prior_err + sample_var/m).
    # Leak-free: the weight uses only observed samples, query counts, and the PKPS/sample
    # disagreement, never the held-out full scores. Mirrors the unified estimator's shrinkage.
    nqc = np.maximum(ncnt, 1)
    pcl = np.clip(np.where(np.isfinite(pk_mat), pk_mat, 0.5), 1e-3, 1 - 1e-3)
    sig_cell = pcl * (1 - pcl) / nqc                       # per-cell sample noise variance
    ce = obs & np.isfinite(pk_mat) & np.isfinite(sample_mat)
    if ce.sum() >= 5:
        err = (pk_mat[ce] - sample_mat[ce]) ** 2 - sig_cell[ce]
        prior_err = max(float(np.sum(nqc[ce] * err) / np.sum(nqc[ce])), 1e-6)
    else:
        prior_err = 1e-6
    w_samp = prior_err / (prior_err + sig_cell)            # weight on the own sample
    ens_mat = np.where(ce, np.clip(w_samp * sample_mat + (1 - w_samp) * pk_mat, 0, 1),
                       np.where(np.isfinite(pk_mat), pk_mat, sample_mat))

    rows, cells = [], []
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
            rows.append(dict(m=m, n_paired=(-1 if n_paired is None else n_paired),
                             n_models=len(models), p_task=p_task, seed=seed, task=tt,
                             dataset=tt.split(':')[0], method=meth, mae=mae))
            # per-(model, task) errors for the conditional (per-model / per-task) analysis
            if dump_cells:
                for i in np.where(h)[0]:
                    cells.append(dict(m=m, n_paired=(-1 if n_paired is None else n_paired),
                                      seed=seed, task=tt, dataset=tt.split(':')[0], method=meth,
                                      model=models[i], family=fams[models[i]],
                                      true=float(yfull[i]), pred=float(p[i]),
                                      abs_err=float(abs(p[i] - yfull[i]))))
    return cells if dump_cells else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=['budget', 'n_models', 'coverage', 'pairing', 'breakdown'],
                    default='budget')
    ap.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 93])
    ap.add_argument('--coverages', type=float, nargs='+', default=[0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument('--pairing_budget', type=int, default=4)
    ap.add_argument('--pairing_budgets', type=int, nargs='+', default=[2, 4, 8],
                    help='pairing-cliff robustness: one full rho-curve per budget')
    ap.add_argument('--n_paired_values', type=int, nargs='+', default=[0, 4],
                    help='breakdown: which pairing levels to dump per-cell (default unpaired/paired)')
    ap.add_argument('--fixed_m', type=int, default=8)
    ap.add_argument('--n_seeds', type=int, default=16)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-rd1')
    args = ap.parse_args()

    data = H.load_suite()
    print(f'suite: {len(data[7])} models, {len(data[8])} tasks')
    # pairing cliff (headline): one full rho-curve per budget (robustness). Query-efficiency
    # panels vary p_query (m), cohort n, or task coverage p_task. breakdown dumps per-cell
    # errors at fixed pairing levels for the conditional analysis. (m, n_models, p_task, n_paired)
    dump = args.sweep == 'breakdown'
    if args.sweep == 'pairing':
        specs = [(b, None, 1.0, k) for b in args.pairing_budgets for k in range(b + 1)]
    elif args.sweep == 'breakdown':
        specs = [(args.pairing_budget, None, 1.0, k) for k in args.n_paired_values]
    elif args.sweep == 'budget':
        specs = [(m, None, 1.0, None) for m in args.budgets]
    elif args.sweep == 'n_models':
        specs = [(args.fixed_m, n, 1.0, None) for n in args.n_models_values]
    else:
        specs = [(args.fixed_m, None, p, None) for p in args.coverages]
    jobs = [delayed(run_seed)(data, m, s, n_models=n, p_task=p, n_paired=k, dump_cells=dump)
            for (m, n, p, k) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for sub in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) for r in sub])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'rd1_suite_{args.sweep}.csv', index=False)
    if dump:
        print('per-cell rows:', len(res))
        print(res.groupby(['n_paired', 'method'])['abs_err'].mean().round(4).to_string())
    else:
        xcol = {'budget': 'm', 'n_models': 'n_models', 'coverage': 'p_task', 'pairing': 'm'}[args.sweep]
        print(f'overall MAE by {xcol}:')
        print(res.pivot_table(index=xcol, columns='method', values='mae').round(4).to_string())


if __name__ == '__main__':
    main()
