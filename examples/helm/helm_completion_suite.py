#!/usr/bin/env python
"""Matrix completion on the heterogeneous suite (Result 2): a fraction p_task of (model,task)
cells is evaluated in full (true scores); predict the MISSING cells. Baseline = low-rank
matrix completion; the PKPS embedding completes from the joint perspective; ensemble blends
them. MAE over the MISSING cells only -- no sample line (a missing cell was never run).
Levers: cohort n, task coverage p_task, number of tasks T."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import helm_doublekernel as H
from helm_rd2_cv import load
from helm_unified import _perspective, mc_crossfit


def _mae(P, mask, full):
    m = mask & np.isfinite(P)
    return float(np.mean(np.abs(P[m] - full[m]))) if m.any() else np.nan


def trial(data, qmed, n_models, n_tasks, p_task, seed):
    rX, Qu, qc, mid, tid, qid, full, mods, tasks, grp, rs = data[:11]
    rng = np.random.default_rng(seed)
    msel = (np.sort(rng.choice(len(mods), min(n_models, len(mods)), replace=False))
            if n_models else np.arange(len(mods)))
    tsel = (np.sort(rng.choice(len(tasks), min(n_tasks, len(tasks)), replace=False))
            if n_tasks else np.arange(len(tasks)))
    mods_s = [mods[i] for i in msel]
    full_s = full[np.ix_(msel, tsel)]
    fin0 = np.isfinite(full_s)
    # observe p_task of the (existing) cells in full
    O = (rng.random(full_s.shape) < p_task) & fin0
    for i in range(O.shape[0]):
        if not O[i].any() and fin0[i].any(): O[i, np.where(fin0[i])[0][0]] = True
    for t in range(O.shape[1]):
        if not O[:, t].any() and fin0[:, t].any(): O[np.where(fin0[:, t])[0][0], t] = True

    # the score is the full-eval (true) mean over ALL of a cell's queries; the perspective only
    # needs a representative response sample, so cap it for tractability.
    K_PERSP = 16
    keep, samp = [], np.full(full_s.shape, np.nan)
    for ii, mi in enumerate(msel):
        for tt, ti in enumerate(tsel):
            if O[ii, tt] and (mods[mi], tasks[ti]) in grp:
                idx = grp[(mods[mi], tasks[ti])]
                samp[ii, tt] = rs[idx].mean()                        # full-eval -> true score
                keep.append(idx if len(idx) <= K_PERSP else rng.choice(idx, K_PERSP, replace=False))
    if not keep:
        return None
    keep = np.concatenate(keep)
    Z = _perspective(rX, Qu, qc, mid, tid, qid, keep, mods_s, True, qmed)
    if Z is None:
        return None
    pk = H.predict_from_embedding_all(Z, samp, O, predictor='knn', k=5)
    mc = mc_crossfit(samp, O, rng)
    # ensemble weight fit on observed cells (both channels cross-fit), applied to the missing
    oo = O & np.isfinite(pk) & np.isfinite(mc) & np.isfinite(samp)
    grid = np.linspace(0, 1, 21)
    a = grid[int(np.argmin([np.mean(np.abs(np.clip(g * pk[oo] + (1 - g) * mc[oo], 0, 1) - samp[oo]))
                            for g in grid]))] if oo.sum() >= 5 else 0.5
    ens = np.clip(a * pk + (1 - a) * mc, 0, 1)
    mis = (~O) & fin0
    return dict(n_models=len(mods_s), n_tasks=len(tsel), p_task=p_task, seed=seed,
                mc=_mae(mc, mis, full_s), pkps=_mae(pk, mis, full_s), ens=_mae(ens, mis, full_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=['n_models', 'coverage', 'n_tasks'], default='coverage')
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 93])
    ap.add_argument('--coverages', type=float, nargs='+', default=[0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument('--n_tasks_values', type=int, nargs='+', default=[4, 8, 12, 18])
    ap.add_argument('--fixed_p', type=float, default=0.5)
    ap.add_argument('--n_seeds', type=int, default=8)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-unified')
    args = ap.parse_args()
    data, qmed, suite = load('suite')
    if args.sweep == 'n_models':
        specs = [(n, None, args.fixed_p) for n in args.n_models_values]
    elif args.sweep == 'coverage':
        specs = [(None, None, p) for p in args.coverages]
    else:
        specs = [(None, T, args.fixed_p) for T in args.n_tasks_values]
    jobs = [delayed(trial)(data, qmed, n, T, p, s) for (n, T, p) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for r in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) if r])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'completion_suite_{args.sweep}.csv', index=False)
    xcol = {'n_models': 'n_models', 'coverage': 'p_task', 'n_tasks': 'n_tasks'}[args.sweep]
    print(res.groupby(xcol)[['mc', 'pkps', 'ens']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
