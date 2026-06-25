#!/usr/bin/env python
"""
helm_unified.py  --  one problem: estimate the full (model x task) score matrix from
observations where each cell has m >= 0 queries. RD1 is the m>0 (denoise) face, RD2
the m=0 (complete) face; here they are unified. Per cell the score channel is the
own noisy sample (observed) or matrix completion (missing); PKPS adds the item-level
embedding. Two blend weights are tuned honestly: the observed (denoising) weight by
splitting each cell's queries in half, the missing (completion) weight by held-out
observed cells. MAE is over EVERY cell vs the true full score.

Levers: task coverage (p), queries per observed cell (q, =0 noiseless/full), cohort.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dkps.unpaired_dkps import pca_reduce_elbow
from dkps.baselines import matrix_completion_predict
import helm_doublekernel as H
from helm_rd2_cv import load

GRID = np.linspace(0, 1, 21)


def _perspective(rX, Qu, qc, mid, tid, qid, keep, mods, suite, qmed):
    codes = qc[keep]
    if suite:
        df = pd.DataFrame({'model_id': mid[keep], 'task_id': tid[keep], 'query_id': qid[keep],
                           'embedding': list(rX[keep]), 'query_embedding': list(Qu[codes])})
        rk, med = 'linear', qmed
    else:
        rr = pca_reduce_elbow(rX[keep]); uq = np.unique(codes); qr = pca_reduce_elbow(Qu[uq])
        from scipy.spatial.distance import pdist
        c2v = {c: qr[j] for j, c in enumerate(uq)}
        df = pd.DataFrame({'model_id': mid[keep], 'task_id': tid[keep], 'query_id': qid[keep],
                           'embedding': list(rr), 'query_embedding': [c2v[c] for c in codes]})
        rk, med = 'rbf', float(np.median(pdist(qr if len(qr) <= 400 else qr[:400])))
    bw = H.SubsampleMedianBandwidth()
    est = H.ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel=rk,
                                          query_bandwidth=bw, response_bandwidth=bw)
    names, Ds = est.dist_matrices(df, [med])
    return H._mds_full(Ds[med], names, mods, 8)


def trial(data, suite, qmed, p, q, n_models, seed):
    rX, Qu, qc, mid, tid, qid, full, mods, tasks, grp, rs = data[:11]
    rng = np.random.default_rng(seed)
    if n_models and n_models < len(mods):
        sel = np.sort(rng.choice(len(mods), n_models, replace=False))
        mods = [mods[i] for i in sel]; full = full[sel]
    O = H.sample_observed(len(mods), len(tasks), p, rng) & np.isfinite(full)
    keep, samp = [], np.full_like(full, np.nan)
    ncnt = np.zeros_like(full)                                          # queries per observed cell
    s1 = np.full_like(full, np.nan); s2 = np.full_like(full, np.nan)    # half-splits (observed)
    for i in range(len(mods)):
        for t in range(len(tasks)):
            if O[i, t] and (mods[i], tasks[t]) in grp:
                idx = grp[(mods[i], tasks[t])]
                take = idx if (q is None or len(idx) <= q) else rng.choice(idx, q, replace=False)
                keep.append(take); samp[i, t] = rs[take].mean(); ncnt[i, t] = len(take)
                if len(take) >= 2:
                    h = rng.permutation(len(take))
                    s1[i, t] = rs[take[h[:len(take) // 2]]].mean()
                    s2[i, t] = rs[take[h[len(take) // 2:]]].mean()
    keep = np.concatenate(keep)
    Z = _perspective(rX, Qu, qc, mid, tid, qid, keep, mods, suite, qmed)
    if Z is None:
        return None
    pk = H.predict_from_embedding_all(Z, samp, O, predictor='knn', k=5)
    mc = matrix_completion_predict(samp, O)

    obs = O & np.isfinite(samp)
    # Channels: own sample (observed only, direct but noisy), MC (its value on observed
    # cells just echoes the sample, so it only adds signal on MISSING cells), and PKPS
    # (cross-model, available everywhere). We ensemble each where it carries independent
    # signal: missing cells = CV blend(MC, PKPS); observed cells = precision blend(sample,
    # PKPS) -- the PKPS weight rises with noise and falls as queries grow.

    # missing-cell completion weight (MC vs PKPS), held-out observed cells vs their sample
    oi = np.argwhere(obs); a_mis = 0.5
    if len(oi) >= 8:
        v = oi[rng.choice(len(oi), max(1, len(oi) // 4), replace=False)]
        cvo = O.copy(); cvo[v[:, 0], v[:, 1]] = False
        pk2 = H.predict_from_embedding_all(Z, samp, cvo, predictor='knn', k=5)
        mc2 = matrix_completion_predict(samp, cvo)
        vt = samp[v[:, 0], v[:, 1]]; p2 = pk2[v[:, 0], v[:, 1]]; m2 = mc2[v[:, 0], v[:, 1]]
        hh = np.isfinite(vt) & np.isfinite(p2) & np.isfinite(m2)
        if hh.any():
            a_mis = GRID[int(np.argmin([np.mean(np.abs(np.clip(a * p2[hh] + (1 - a) * m2[hh], 0, 1) - vt[hh]))
                                        for a in GRID]))]

    # observed-cell PKPS weight by precision. Half-splits give the full-sample noise
    # (sigma_q^2 = E[(s1-s2)^2]/4) and PKPS's error incl. bias (E[(pk-s2)^2]-E[(s1-s2)^2]/2);
    # w_pk = sigma_q^2/(sigma_q^2 + pkps_err) (q/2 -> q correction baked in).
    hv = obs & np.isfinite(s1) & np.isfinite(s2) & np.isfinite(pk)
    if hv.sum() >= 5:
        ess = np.mean((s1[hv] - s2[hv]) ** 2)
        sig_q = ess / 4.0
        pkps_err = max(np.mean((pk[hv] - s2[hv]) ** 2) - ess / 2.0, 1e-6)
        w_pk = float(np.clip(sig_q / (sig_q + pkps_err), 0.0, 1.0))
    else:  # q=1: no split; use p(1-p)/m vs E[(pk-sample)^2]
        hp = obs & np.isfinite(pk)
        nv = np.mean(pk[hp] * (1 - pk[hp]) / np.maximum(ncnt[hp], 1)) if hp.any() else 0.0
        tv = np.mean((pk[hp] - samp[hp]) ** 2) if hp.any() else 1.0
        w_pk = float(np.clip(nv / tv, 0.0, 1.0)) if tv > 1e-9 else 0.0

    score = np.where(obs, samp, mc)
    ens = np.where(obs, np.clip((1 - w_pk) * samp + w_pk * pk, 0, 1),
                   np.clip(a_mis * pk + (1 - a_mis) * mc, 0, 1))
    fin = np.isfinite(full)
    def mae(P, m): mm = m & fin & np.isfinite(P); return float(np.mean(np.abs(P[mm] - full[mm])))
    return dict(p=p, q=(q or 0), n_models=len(mods), seed=seed,
                score_all=mae(score, np.ones_like(O)), pkps_all=mae(pk, np.ones_like(O)),
                mc_all=mae(mc, np.ones_like(O)), ens_all=mae(ens, np.ones_like(O)),
                score_obs=mae(score, obs), ens_obs=mae(ens, obs),
                score_mis=mae(score, ~O), ens_mis=mae(ens, ~O),
                w_pk=w_pk, a_mis=a_mis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--sweep', choices=['queries', 'coverage', 'n_models'], default='queries')
    ap.add_argument('--queries', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--coverages', type=float, nargs='+', default=[0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 93])
    ap.add_argument('--fixed_p', type=float, default=0.5)
    ap.add_argument('--fixed_q', type=int, default=8)
    ap.add_argument('--n_seeds', type=int, default=12)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-unified')
    args = ap.parse_args()
    data, qmed, suite = load(args.dataset)

    if args.sweep == 'queries':
        specs = [(args.fixed_p, q, None) for q in args.queries]; xcol = 'q'
    elif args.sweep == 'coverage':
        specs = [(p, args.fixed_q, None) for p in args.coverages]; xcol = 'p'
    else:
        specs = [(args.fixed_p, args.fixed_q, n) for n in args.n_models_values]; xcol = 'n_models'
    jobs = [delayed(trial)(data, suite, qmed, p, q, nm, s)
            for (p, q, nm) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for r in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) if r])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'unified_{args.dataset}_{args.sweep}.csv', index=False)
    print(res.groupby(xcol)[['mc_all', 'pkps_all', 'score_all', 'ens_all', 'score_obs', 'ens_obs',
                             'score_mis', 'ens_mis', 'w_pk', 'a_mis']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
