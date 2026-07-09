"""Cost-aware routing: offload flagship queries to cheap substitutes.

Reframing of mimicry routing with an expensive target: the flagship f* is
the suite's best model by anchor mean score. For each evaluation query, a
router proposes a substitute r and a confidence d (predicted substitution
quality); sweeping an offload budget (fraction of queries served by the
substitute, lowest-d first) traces realized-score-retention vs. savings.
Savings per offloaded query = 1 - (substitute price / flagship price),
reported parametrically as the offload fraction.

Routers (substitute choice + query ordering):
    qa       : nearest to flagship in the query-localized geometry; order by
               that distance (label-free, response-based)
    static   : same with uniform weights (query ordering constant per pair
               structure -> ordering by distance of the chosen substitute)
    task*    : nearest in the task-indicator geometry (hidden labels)
    cascade  : leaderboard cascade -- per task, substitute = anchor-best
               non-flagship model; order tasks by historical score gap
               (uses task labels; what practitioners do today)
    profile  : substitute = nearest anchor score profile to the flagship;
               constant ordering (query-independent)
    random   : random substitute, random order (expectation over draws)

Metric: mean realized score of the policy at each offload fraction,
relative to all-flagship. LOQO: eval-query rows never enter geometries,
profiles, or anchor score means.

Run from repo root: pixi run python -m experiments.routing.run_cost_routing
"""

import os

import numpy as np
import pandas as pd

from .evaluate import stratified_split
from .geometry import pairwise_query_dist_tensor, rbf_weights, weighted_dist_matrix
from .run_helm import RESULTS, load_paired_core

FRACS = np.linspace(0.0, 1.0, 11)
SIGMA_FRACTION = 0.25


def parse_params(name):
    """Parameter count from the model name (compute-cost proxy for open
    models); None for closed/unsized models (the expensive tier)."""
    import re
    n = name.lower()
    m = re.findall(r'(\d+)x(\d+(?:\.\d+)?)b', n)
    if m:
        return float(m[0][0]) * float(m[0][1])
    m = re.findall(r'(\d+(?:\.\d+)?)b(?![a-z0-9])', n)
    if m:
        return max(float(x) for x in m)
    return None


def policy_curve(order, sub_scores, flag_scores):
    """Mean realized score at each offload fraction (lowest-order first)."""
    q = len(order)
    rank = np.argsort(order)
    out = []
    for f in FRACS:
        k = int(round(f * q))
        take = np.zeros(q, dtype=bool)
        take[rank[:k]] = True
        out.append(float(np.where(take, sub_scores, flag_scores).mean()))
    return out


def run_helm(seeds=5, pools=(('le13b', 13.0), ('le40b', 40.0), ('all', None))):
    X, U, cats, models, S = load_paired_core(with_scores=True)
    P = pairwise_query_dist_tensor(X)
    n, m = S.shape
    sizes = [parse_params(mm) for mm in models]
    rows = []
    for pool_name, cap in pools:
      for seed in range(seeds):
        rng = np.random.default_rng(seed)
        a_idx, e_idx = stratified_split(cats, 0.2, rng)
        flag = int(np.argmax(S[:, a_idx].mean(axis=1)))
        if cap is None:
            cand = np.array([i for i in range(n) if i != flag])
        else:
            cand = np.array([i for i in range(n)
                             if i != flag and sizes[i] is not None
                             and sizes[i] <= cap])
        flag_scores = S[flag, e_idx]

        # anchor statistics
        med_i = rng.integers(0, len(a_idx), 20000)
        med_k = rng.integers(0, len(a_idx), 20000)
        keep = med_i != med_k
        med = float(np.median(np.linalg.norm(
            U[a_idx][med_i[keep]] - U[a_idx][med_k[keep]], axis=1)))
        tasks = sorted(set(cats))
        t_of = {t: np.flatnonzero(cats[a_idx] == t) for t in tasks}
        task_mean = {t: S[:, a_idx[t_of[t]]].mean(axis=1) for t in tasks}
        prof = np.stack([task_mean[t] for t in tasks], axis=1)     # (n, T)

        # --- qa: localized distance flagship <-> candidates, per eval query
        W = rbf_weights(U[a_idx], U[e_idx], SIGMA_FRACTION * med)
        D_qa = weighted_dist_matrix(P[a_idx], W)                   # (q, n, n)
        d_qa = D_qa[:, flag, :][:, cand]                           # (q, n-1)
        pick_qa = cand[d_qa.argmin(axis=1)]
        conf_qa = d_qa.min(axis=1)

        # --- static
        D_st = weighted_dist_matrix(P[a_idx], np.ones(len(a_idx)))
        d_st = D_st[flag, cand]
        pick_st = np.full(len(e_idx), cand[int(np.argmin(d_st))])
        # order queries by the qa-agnostic proxy: constant -> random tiebreak
        conf_st = rng.random(len(e_idx))

        # --- task* (hidden labels): per-task geometry
        pick_tk = np.empty(len(e_idx), dtype=int)
        conf_tk = np.empty(len(e_idx))
        for t in tasks:
            Dt = weighted_dist_matrix(P[a_idx],
                                      (cats[a_idx] == t).astype(float))
            dt = Dt[flag, cand]
            sel = cats[e_idx] == t
            pick_tk[sel] = cand[int(np.argmin(dt))]
            conf_tk[sel] = dt.min()

        # --- cascade: per-task best substitute by anchor score; order by gap
        pick_cs = np.empty(len(e_idx), dtype=int)
        conf_cs = np.empty(len(e_idx))
        for t in tasks:
            tm = task_mean[t]
            best = cand[int(np.argmax(tm[cand]))]
            gap = float(tm[flag] - tm[best])
            sel = cats[e_idx] == t
            pick_cs[sel] = best
            conf_cs[sel] = gap

        # --- profile: nearest score profile to flagship (constant)
        dp = np.linalg.norm(prof[cand] - prof[flag], axis=1)
        pick_pf = np.full(len(e_idx), cand[int(np.argmin(dp))])
        conf_pf = rng.random(len(e_idx))

        # --- random
        pick_rd = cand[rng.integers(0, len(cand), len(e_idx))]
        conf_rd = rng.random(len(e_idx))

        for name, pick, conf in (('qa', pick_qa, conf_qa),
                                 ('task*', pick_tk, conf_tk),
                                 ('cascade', pick_cs, conf_cs),
                                 ('static', pick_st, conf_st),
                                 ('profile', pick_pf, conf_pf),
                                 ('random', pick_rd, conf_rd)):
            sub_scores = S[pick, e_idx]
            curve = policy_curve(conf, sub_scores, flag_scores)
            for f, v in zip(FRACS, curve):
                rows.append((pool_name, seed, name, float(f), v,
                             float(flag_scores.mean())))
    return pd.DataFrame(rows, columns=['pool', 'seed', 'method',
                                       'offload_frac', 'policy_score',
                                       'flagship_score'])


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = run_helm()
    df.to_parquet(os.path.join(RESULTS, 'cost_routing_helm.parquet'))
    df = df.assign(retention=df.policy_score / df.flagship_score)
    for pool, sub in df.groupby('pool'):
        piv = sub.groupby(['method', 'offload_frac'])['retention'].mean().unstack()
        piv.to_csv(os.path.join(RESULTS, f'cost_routing_helm_{pool}.csv'))
        print(f'\n== pool {pool}: score retention (%) by offload fraction ==')
        print((100 * piv).round(2).to_string())
        print('max offload at >= 99% retention:')
        for meth in piv.index:
            okm = piv.loc[meth][piv.loc[meth] >= 0.99]
            print(f'  {meth:8s} {(okm.index.max() if len(okm) else 0.0):.0%}')


if __name__ == '__main__':
    main()
