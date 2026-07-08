"""Query routing on the (unpaired) EEE suite: mimicry and score routing.

The paired per-query tensor does not exist here, so the query-localized
geometry is the rank-one-localized PKPS comparison, which factorizes under a
linear response kernel: A_{ii'}(q*) = <phi_i(q*), phi_i'(q*)> with

    phi_i(q*) = sum_j w_j x_ij / sum_j w_j,   w_j = k_Q(u_j, u_{q*})

over model i's OWN anchor pool, so D(i, i'; q*) = ||phi_i - phi_i'||. The
sigma -> inf limit is the global mean-embedding geometry (static); a task
indicator weight gives the task-localized intermediate.

Evaluations, both restricted to each eval query's responders (the models
that answered it -- realized outcomes exist only there):
  mimicry : pick the candidate nearest to the target in the geometry;
            error = ||x_{r,q*} - x_{t,q*}|| (targets with >= 2 candidates).
  score   : pick argmax of the kernel-regressed own-score estimate
            s_i(q*) = sum_j w_j s_ij / sum_j w_j; outcome = realized score.

Eval-query rows are excluded from every anchor pool (no leakage).

Run from repo root: pixi run python -m experiments.routing.run_eee
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'examples', 'helm'))
from pipeline import loaders as H  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FRACTIONS = (0.125, 0.25, 0.5, 1.0, 2.0)


def load_eee_rows():
    helm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'examples', 'helm')
    cwd = os.getcwd()
    os.chdir(helm_dir)
    try:
        (resp_X, Qu, qid_code, model_id, task_id, query_id,
         score_mat, models, tasks, groups, row_score, qmed) = H.load_eee()
    finally:
        os.chdir(cwd)
    m2i = {mm: i for i, mm in enumerate(models)}
    rows = pd.DataFrame({
        'model': [m2i[mm] for mm in model_id],
        'task': task_id, 'query': query_id, 'code': qid_code,
        'score': row_score})
    return resp_X.astype(np.float32), Qu.astype(np.float32), rows, models, qmed


def localized_stats(X, scores, w, model_of, n):
    """Per-model weighted mean embedding and weighted mean score."""
    denom = np.bincount(model_of, weights=w, minlength=n)
    phi = np.zeros((n, X.shape[1]))
    np.add.at(phi, model_of, X * w[:, None])
    s = np.bincount(model_of, weights=w * scores, minlength=n)
    ok = denom > 1e-12
    phi[ok] /= denom[ok, None]
    s[ok] /= denom[ok]
    return phi, s, ok


def build_pair_overlap(X, Qu, rows, anchor_mask):
    """For each model pair, the anchor queries BOTH answered: query embeddings
    and realized pairwise response distances. This is all a response-free
    direct-regression router can train on."""
    pair_u, pair_e = {}, {}
    a = rows[anchor_mask]
    for (_, _), g in a.groupby(['task', 'query']):
        ms = g['model'].to_numpy()
        if len(ms) < 2:
            continue
        Xg = X[g.index.to_numpy()]
        u = Qu[g['code'].iloc[0]]
        for i in range(len(ms)):
            for k in range(i + 1, len(ms)):
                key = (min(ms[i], ms[k]), max(ms[i], ms[k]))
                e = float(np.linalg.norm(Xg[i] - Xg[k]))
                pair_u.setdefault(key, []).append(u)
                pair_e.setdefault(key, []).append(e)
    return ({k: np.stack(v) for k, v in pair_u.items()},
            {k: np.asarray(v) for k, v in pair_e.items()})


def rf_predict(pair_u, pair_e, grand_mean, t_model, c_model, u_star, sigma):
    """Kernel regression of realized pairwise error on the query embedding,
    over the pair's shared anchor queries; sigma=None -> uniform (rf-static)."""
    key = (min(t_model, c_model), max(t_model, c_model))
    if key not in pair_e:
        return grand_mean
    e = pair_e[key]
    if sigma is None:
        return float(e.mean())
    d2 = ((pair_u[key] - u_star) ** 2).sum(axis=1)
    w = np.exp(-(d2 - d2.min()) / (2.0 * sigma ** 2))
    return float((w * e).sum() / w.sum())


def run_seed(X, Qu, rows, n, qmed, seed, n_eval=300):
    rng = np.random.default_rng(seed)
    resp = rows.groupby(['task', 'query'])['model'].nunique()
    eligible = resp[resp >= 3].reset_index()[['task', 'query']]
    # stratified sample of eval queries, proportional to each task's share
    eval_keys = set()
    for t_name, g in eligible.groupby('task'):
        k = min(len(g), max(1, int(round(n_eval * len(g) / len(eligible)))))
        take = g.iloc[rng.choice(len(g), size=k, replace=False)]
        eval_keys.update(zip(take['task'], take['query']))
    is_eval_row = np.array([(t, q) in eval_keys
                            for t, q in zip(rows['task'], rows['query'])])

    anchor = ~is_eval_row
    Xa = X[anchor]
    sa = rows['score'].to_numpy()[anchor]
    model_a = rows['model'].to_numpy()[anchor]
    task_a = rows['task'].to_numpy()[anchor]
    ua = Qu[rows['code'].to_numpy()[anchor]]

    pair_u, pair_e = build_pair_overlap(X, Qu, rows, anchor)
    grand_mean = float(np.concatenate(list(pair_e.values())).mean())
    rf_sigmas = {'rf-0.5x': 0.5 * qmed, 'rf-1x': 1.0 * qmed, 'rf-2x': 2.0 * qmed,
                 'rf-static': None}

    sigmas = {f'qa-{f:g}x': f * qmed for f in FRACTIONS}

    mim_rows, sc_rows = [], []
    edf = rows[is_eval_row]
    for (t_name, q), g in edf.groupby(['task', 'query']):
        resp_models = g['model'].to_numpy()
        Xq = X[g.index.to_numpy()]
        u_star = Qu[g['code'].iloc[0]]
        s_real = g['score'].to_numpy()

        # geometries: per-sigma localized, task-localized, static
        geoms = {}
        d2 = ((ua - u_star) ** 2).sum(axis=1)
        for name, s in sigmas.items():
            w = np.exp(-(d2 - d2.min()) / (2.0 * s ** 2))
            geoms[name] = localized_stats(Xa, sa, w, model_a, n)
        geoms['task'] = localized_stats(Xa, sa, (task_a == t_name).astype(float), model_a, n)
        geoms['static'] = localized_stats(Xa, sa, np.ones(len(Xa)), model_a, n)

        # ---- mimicry among responders
        E = np.sqrt(((Xq[:, None] - Xq[None, :]) ** 2).sum(-1))   # (r, r)
        r = len(resp_models)
        for ti in range(r):
            cand = [c for c in range(r) if c != ti]
            errs = E[ti, cand]
            for name, (phi, _, ok) in geoms.items():
                D = np.linalg.norm(phi[resp_models[cand]] - phi[resp_models[ti]], axis=1)
                D[~ok[resp_models[cand]]] = np.inf
                mim_rows.append((seed, t_name, q, name, float(errs[int(np.argmin(D))])))
            for name, s in rf_sigmas.items():
                pred = np.array([rf_predict(pair_u, pair_e, grand_mean,
                                            resp_models[ti], resp_models[c], u_star, s)
                                 for c in cand])
                mim_rows.append((seed, t_name, q, name, float(errs[int(np.argmin(pred))])))
            mim_rows.append((seed, t_name, q, 'random', float(errs.mean())))
            mim_rows.append((seed, t_name, q, 'oracle', float(errs.min())))

        # ---- score routing among responders
        for name, (_, shat, ok) in geoms.items():
            v = shat[resp_models].copy()
            v[~ok[resp_models]] = -np.inf
            sc_rows.append((seed, t_name, q, name, float(s_real[int(np.argmax(v))])))
        sc_rows.append((seed, t_name, q, 'random', float(s_real.mean())))
        sc_rows.append((seed, t_name, q, 'oracle', float(s_real.max())))

    mim = pd.DataFrame(mim_rows, columns=['seed', 'task', 'query', 'method', 'error'])
    sc = pd.DataFrame(sc_rows, columns=['seed', 'task', 'query', 'method', 'score'])
    return mim, sc


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, Qu, rows, models, qmed = load_eee_rows()
    n = len(models)
    print(f'{len(rows)} rows, {n} models, qmed {qmed:.3f}')

    mims, scs = [], []
    for s in range(5):
        m, c = run_seed(X, Qu, rows, n, qmed, s)
        mims.append(m); scs.append(c)
    mim, sc = pd.concat(mims, ignore_index=True), pd.concat(scs, ignore_index=True)
    mim.to_parquet(os.path.join(RESULTS, 'eee_mimicry.parquet'))
    sc.to_parquet(os.path.join(RESULTS, 'eee_score_routing.parquet'))

    def summarize(df, col, ascending):
        per_seed = df.groupby(['method', 'seed'])[col].mean().unstack()
        out = pd.DataFrame({
            f'mean_{col}': per_seed.mean(axis=1),
            'se': per_seed.std(axis=1) / np.sqrt(per_seed.shape[1])})
        return out.sort_values(f'mean_{col}', ascending=ascending)

    print('\n== mimicry (lower is better) ==')
    ms = summarize(mim, 'error', True); print(ms.round(4).to_string())
    ms.to_csv(os.path.join(RESULTS, 'eee_mimicry_summary.csv'))
    print('\n== score routing (higher is better) ==')
    ss = summarize(sc, 'score', False); print(ss.round(4).to_string())
    ss.to_csv(os.path.join(RESULTS, 'eee_score_summary.csv'))


if __name__ == '__main__':
    main()
