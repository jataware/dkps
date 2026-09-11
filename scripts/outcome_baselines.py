"""Outcome-only baselines for the QUENCH figure: predictors of a system's true
full-500 resolve rate that see graded probe outcomes on the q20 panel but no
traces.

  count_lookup     : average true rate of reference systems that got the same
                     number of the m probes right ("average score by n correct").
  irt_2pl_random   : two-parameter IRT (per-item difficulty and slope) fit on
                     the references' panel outcomes; posterior over the target's
                     ability from its m probe outcomes; prediction = posterior
                     mean of a fitted ability -> full-500 score map. Probes random.
  irt_2pl_informative : same model, but the m probes are the items with the
                     highest Fisher information averaged over the reference
                     population (one fixed panel for everyone).
  irt_2pl_adaptive : same model, probes chosen one at a time per target to
                     maximise expected information under the current posterior.

All predictors use leave-one-LLM-out references (same rule as quench.py) and
are scored on all systems. Random-probe numbers average over draws (all 20
single tasks at m=1, all draws of the full panel at m=20).

Usage:  python scripts/outcome_baselines.py [--ms 1,2,3,5,10,20] [--draws 100]
                                             [--ridge-a 10] [--out figures/outcome_baselines.json]
"""
import argparse
import json
import os
import re

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import expit, logsumexp

GRID = np.linspace(-16, 16, 1601)       # 0.02 spacing; include broad geometry priors


# ---------------------------------------------------------------- data ----
def load_panel(labels_path='data/leaderboard/verified_labels.json',
               judge_dir='data/judge/structured-qspec'):
    """Systems, panel tasks, true full-500 rates y, panel outcomes B (M x Q),
    and the leave-one-LLM-out reference mask `allowed` (M x M)."""
    labels = json.load(open(labels_path))
    systems = sorted(s for s in os.listdir(judge_dir) if 'resolved' in labels.get(s, {}))
    q20 = sorted(f[:-5] for f in os.listdir(os.path.join(judge_dir, systems[0])))
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    B = np.array([[q in set(labels[s]['resolved']) for q in q20] for s in systems], float)

    def llm_tag(s):
        m = re.search(r'^\s+model_display:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
        return m.group(1).strip().strip('"\'') if m else None
    tags = [llm_tag(s) for s in systems]
    allowed = np.array([[j != i and not (tags[i] and tags[j] == tags[i])
                         for j in range(len(systems))] for i in range(len(systems))])
    return systems, q20, y, B, allowed


def llm_groups(systems, labels_path='data/leaderboard/verified_labels.json'):
    """Group id per system: systems sharing a non-empty model_display tag share a
    group; untagged systems are singletons. Used for group-wise CV splits."""
    labels = json.load(open(labels_path))
    ids, seen = [], {}
    for s in systems:
        m = re.search(r'^\s+model_display:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
        tag = m.group(1).strip().strip('"\'') if m else None
        key = tag if tag else f'__solo__{s}'
        ids.append(seen.setdefault(key, len(seen)))
    return np.array(ids)


# ------------------------------------------------- metadata baselines ----
SCAFFOLDS = ('sweagent', 'swe-agent', 'openhands', 'agentless', 'autocoderover',
             'moatless', 'composio', 'marscode', 'lingma', 'gru', 'masai',
             'codeact', 'tools', 'epam', 'solver', 'aime', 'blackbox',
             'devlo', 'emergent', 'nemotron', 'trae', 'refact', 'zencoder')
UNINFORMATIVE_MODEL_TAGS = {None, '', 'Multiple', 'Undisclosed', 'None'}


def scaffold_of(system):
    """Scaffold keyword from the submission name (same heuristic as pillars.py)."""
    low = system.lower()
    return next((h for h in SCAFFOLDS if h in low), None)


def group_mean_baseline(y, groups, exclude):
    """Predict each system as the mean true rate of the other systems in its
    group, restricted to references not excluded for it; fall back to the
    mean of its allowed references when the group is empty or untagged.
    Returns (MAE over all systems, MAE over covered systems only, MAE of the
    fallback on those same covered systems, number covered)."""
    preds, is_cov = [], []
    for i in range(len(y)):
        ok = exclude[i] & np.array([g is not None and g == groups[i] for g in groups])
        covered = groups[i] is not None and ok.any()
        preds.append(y[ok].mean() if covered else y[exclude[i]].mean()); is_cov.append(covered)
    preds, is_cov = np.array(preds), np.array(is_cov)
    fallback = np.array([y[exclude[i]].mean() for i in range(len(y))])
    return (float(np.abs(preds - y).mean()), float(np.abs(preds - y)[is_cov].mean()),
            float(np.abs(fallback - y)[is_cov].mean()), int(is_cov.sum()))


# ------------------------------------------------------- count lookup ----
def count_lookup(B, y, allowed, i, cols):
    """Mean true rate of allowed references with the same number correct on
    `cols` as system i; widen the tolerance until some reference matches."""
    k = B[i, cols].sum()
    ks = B[:, cols].sum(1)
    for tol in range(len(cols) + 1):
        ok = allowed[i] & (np.abs(ks - k) <= tol)
        if ok.any():
            return y[ok].mean()


# ----------------------------------------------------------- 2PL IRT ----
def sigmoid(x):
    return expit(x)


def two_pl_objective(p, B_ref, ridge=1.0, ridge_a=10.0):
    """Penalized Bernoulli negative log likelihood and its analytic gradient."""
    n, Q = B_ref.shape
    th, b, log_a = p[:n], p[n:n + Q], p[n + Q:]
    a = np.exp(log_a)
    z = a[None] * (th[:, None] - b[None])
    loss = (np.logaddexp(0, z) - B_ref * z).sum()
    loss += 0.5 * ridge * (th @ th + b @ b) + 0.5 * ridge_a * (log_a @ log_a)
    R = expit(z) - B_ref
    grad = np.concatenate([(R * a[None]).sum(1) + ridge * th,
                           -(R * a[None]).sum(0) + ridge * b,
                           (R * z).sum(0) + ridge_a * log_a])
    return float(loss), grad


def fit_2pl(B_ref, ridge=1.0, ridge_a=10.0, *, n_starts=1, seed=0,
            maxiter=3000, ftol=1e-12, gtol=1e-5, return_diagnostics=False):
    """MAP fit of P(i solves q) = sigmoid(a_q * (theta_i - b_q)) on the
    reference outcome matrix. Priors: N(0,1) on theta and b, N(0, 1/ridge_a) on
    log a (so slope 1, the Rasch value, is the prior centre).
    Returns theta (n,), b (Q,), a (Q,)."""
    B_ref = np.asarray(B_ref, dtype=float)
    if B_ref.ndim != 2 or min(B_ref.shape) < 1 or not np.isin(B_ref, [0, 1]).all():
        raise ValueError('B_ref must be a nonempty binary matrix, without missing values')
    if ridge <= 0 or ridge_a <= 0 or n_starts < 1:
        raise ValueError('positive regularization and at least one start required')
    n, Q = B_ref.shape

    def unpack(p):
        return p[:n], p[n:n + Q], np.exp(p[n + Q:])

    rng = np.random.default_rng(seed)
    starts = [np.zeros(n + 2 * Q)]
    for _ in range(n_starts - 1):
        pr = (B_ref.sum(1) + 0.5) / (Q + 1)
        pq = (B_ref.sum(0) + 0.5) / (n + 1)
        th = np.log(pr / (1 - pr)); th -= th.mean()
        b = -np.log(pq / (1 - pq)); b -= b.mean()
        starts.append(np.concatenate([th, b, np.zeros(Q)])
                      + rng.normal(0, 0.1, n + 2 * Q))
    runs, results = [], []
    # Broad numerical guardrails; report any active boundary in diagnostics.
    bounds = [(None, None)] * (n + Q) + [(-6, 6)] * Q
    for start in starts:
        kwargs = dict(fun=two_pl_objective, args=(B_ref, ridge, ridge_a),
                      jac=True, method='L-BFGS-B', bounds=bounds)
        options = dict(maxiter=maxiter, maxfun=5 * maxiter, maxls=50,
                       maxcor=20, ftol=ftol, gtol=gtol)
        res = minimize(x0=start, options=options, **kwargs)
        iterations, evaluations = int(res.nit), int(res.nfev)
        polished = False
        if not res.success or np.max(np.abs(res.jac)) > 1e-3:
            polished = True
            options['ftol'] = min(ftol, 1e-14)
            res = minimize(x0=res.x, options=options, **kwargs)
            iterations += int(res.nit); evaluations += int(res.nfev)
        runs.append(dict(success=bool(res.success), message=str(res.message),
                         objective=float(res.fun), iterations=iterations, evaluations=evaluations,
                         initial_objective=two_pl_objective(start, B_ref, ridge, ridge_a)[0],
                         gradient_inf=float(np.max(np.abs(res.jac))),
                         polished=polished,
                         log_slope_boundary=bool(np.any(np.abs(res.x[n + Q:]) >= 5.999))))
        results.append(res)
        if len(runs) == n_starts and n_starts > 1 \
                and np.ptp([r['objective'] for r in runs]) > 1e-4:
            # Distinct local solutions: verify the better basin with extra starts.
            best_x = results[int(np.argmin([r['objective'] for r in runs]))].x
            starts.extend([best_x + rng.normal(0, .3, len(best_x)) for _ in range(2)])
    valid = [i for i, r in enumerate(runs) if r['success']
             and np.isfinite(r['objective']) and r['gradient_inf'] <= 1e-3
             and not r['log_slope_boundary']]
    if not valid:
        raise RuntimeError(f'2PL optimizer failed stationarity/convergence checks: {runs}')
    best = min(valid, key=lambda i: runs[i]['objective'])
    params = unpack(results[best].x)
    diagnostics = dict(runs=runs, best_start=best,
                       objective_spread=float(np.ptp([runs[i]['objective'] for i in valid])),
                       best_agreement_count=sum(abs(runs[i]['objective'] - runs[best]['objective']) < 1e-4
                                                for i in valid),
                       n_references=n, n_fit_items=Q, ridge=ridge, ridge_a=ridge_a)
    return (*params, diagnostics) if return_diagnostics else params


class ItemModel:
    """A fitted 2PL model for one target's reference pool, plus the map from
    ability to full-500 score and the population prior on ability."""

    def __init__(self, B_ref, y_ref, ridge_a, *, ridge=1.0,
                 candidate_cols=None, fit_options=None):
        self.theta, self.b, self.a, self.fit_diagnostics = fit_2pl(
            B_ref, ridge=ridge, ridge_a=ridge_a, return_diagnostics=True,
            **(fit_options or {}))
        if candidate_cols is not None:
            cols = np.asarray(candidate_cols, dtype=int)
            if cols.ndim != 1 or len(cols) == 0 or len(set(cols)) != len(cols) \
                    or np.any(cols < 0) or np.any(cols >= len(self.b)):
                raise ValueError('candidate_cols must be distinct valid fitted-item indices')
            # Expose only allowed candidate tasks to posterior and adaptive selection.
            self.b, self.a = self.b[cols], self.a[cols]
        self.mu, self.sd = self.theta.mean(), max(float(self.theta.std()), 1e-3)
        # score map: y ~ sigmoid(c1 * theta + c0), least squares on references
        c = least_squares(lambda p: sigmoid(p[0] * self.theta + p[1]) - y_ref, [1.0, 0.0]).x
        self.score_of_theta = sigmoid(c[0] * GRID + c[1])
        # item response curves on the grid, P[grid, item]
        self.P_grid = sigmoid(self.a[None] * (GRID[:, None] - self.b[None]))

    def posterior(self, cols, outcomes, mu=None, sd=None, log_prior=None):
        """Posterior weights over GRID after observing `outcomes` on `cols`.
        The prior is the population normal unless (mu, sd) is given."""
        mu = self.mu if mu is None else mu
        sd = self.sd if sd is None else max(float(sd), 1e-3)
        logp = (-0.5 * ((GRID - mu) / sd) ** 2 if log_prior is None
                else np.array(log_prior, dtype=float, copy=True))
        for q, o in zip(cols, outcomes):
            z = self.a[q] * (GRID - self.b[q])
            logp -= np.logaddexp(0, -z if o else z)
        return np.exp(logp - logsumexp(logp))

    def predict(self, cols, outcomes, mu=None, sd=None):
        return float(self.posterior(cols, outcomes, mu, sd) @ self.score_of_theta)

    def item_information(self, weights):
        """Expected Fisher information per item under ability weights on GRID."""
        return weights @ (self.a[None] ** 2 * self.P_grid * (1 - self.P_grid))

    def informative_order(self):
        """Items ranked by information averaged over the reference abilities."""
        P = sigmoid(self.a[None] * (self.theta[:, None] - self.b[None]))
        return np.argsort(-(self.a[None] ** 2 * P * (1 - P)).mean(0))

    def adaptive_path(self, outcomes_all, mu=None, sd=None, return_posteriors=False):
        """Greedy adaptive testing: at each step pick the unused item with the
        highest expected information under the current posterior, observe it
        (from the target's full outcome row), and continue. Starts from the
        population prior unless (mu, sd) is given. Returns the item order and
        the prediction after each step (and the posterior after each step if
        requested)."""
        used, preds, posts = [], [], []
        for _ in range(len(self.b)):
            info = self.item_information(self.posterior(used, outcomes_all[used], mu, sd))
            info[used] = -np.inf
            used.append(int(np.argmax(info)))
            w = self.posterior(used, outcomes_all[used], mu, sd)
            preds.append(float(w @ self.score_of_theta)); posts.append(w)
        return (used, preds, posts) if return_posteriors else (used, preds)


# --------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ms', default='1,2,3,5,10,20')
    ap.add_argument('--draws', type=int, default=100, help='random probe draws per m (1 < m < 20)')
    ap.add_argument('--ridge-a', type=float, default=10.0, help='2PL prior precision on log-slope')
    ap.add_argument('--out', default='figures/outcome_baselines.json')
    args = ap.parse_args()

    systems, q20, y, B, allowed = load_panel()
    M, Q = B.shape
    ms = [int(x) for x in args.ms.split(',')]
    print(f'{M} systems x {Q} panel tasks; fitting {M} leave-one-LLM-out 2PL models')
    models = [ItemModel(B[allowed[i]], y[allowed[i]], args.ridge_a) for i in range(M)]

    rng = np.random.default_rng(0)
    def draws_for(m):
        if m == Q:
            return [np.arange(Q)]
        if m == 1:
            return [np.array([q]) for q in range(Q)]
        return [rng.choice(Q, m, replace=False) for _ in range(args.draws)]

    # adaptive paths and informative orders are per target, independent of m
    adaptive = [models[i].adaptive_path(B[i]) for i in range(M)]
    info_order = [models[i].informative_order() for i in range(M)]

    labels = json.load(open('data/leaderboard/verified_labels.json'))
    model_tags = [None if (t := re.search(r'^\s+model_display:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M))
                  is None or t.group(1).strip().strip('"\'') in UNINFORMATIVE_MODEL_TAGS
                  else t.group(1).strip().strip('"\'') for s in systems]
    scaffolds = [scaffold_of(s) for s in systems]
    loo = ~np.eye(M, dtype=bool)                                     # exclude only the target itself
    scaf = group_mean_baseline(y, scaffolds, allowed)    # figure's rule: LLM siblings excluded
    modl = group_mean_baseline(y, model_tags, loo)       # needs LLM siblings: leave-one-out only
    for name, (mae, mae_cov, mae_fb, n) in (('same-scaffold mean (LLM-out)', scaf), ('same-model mean (leave-one-out)', modl)):
        print(f'{name}: MAE {mae:.4f} over all {M} (fallback = mean of references); '
              f'on the {n} covered systems {mae_cov:.4f} vs {mae_fb:.4f} for the fallback')

    out = {'m': ms, 'ridge_a': args.ridge_a, 'draws': args.draws,
           'constant_mean_mae': float(np.mean([abs(y[i] - y[allowed[i]].mean()) for i in range(M)])),
           'same_scaffold_mean': dict(zip(('mae', 'mae_covered', 'fallback_mae_covered', 'n_covered'), scaf)),
           'same_model_mean': dict(zip(('mae', 'mae_covered', 'fallback_mae_covered', 'n_covered'), modl)),
           'count_lookup': [], 'irt_2pl_random': [], 'irt_2pl_informative': [], 'irt_2pl_adaptive': []}
    print(f"{'m':>3} {'count lookup':>13} {'2PL random':>11} {'2PL informative':>16} {'2PL adaptive':>13}")
    for m in ms:
        e_count, e_rand = [], []
        for cols in draws_for(m):
            e_count.append(np.mean([abs(count_lookup(B, y, allowed, i, cols) - y[i]) for i in range(M)]))
            e_rand.append(np.mean([abs(models[i].predict(cols, B[i, cols]) - y[i]) for i in range(M)]))
        e_info = np.mean([abs(models[i].predict(info_order[i][:m], B[i, info_order[i][:m]]) - y[i]) for i in range(M)])
        e_adapt = np.mean([abs(adaptive[i][1][m - 1] - y[i]) for i in range(M)])
        for key, v in (('count_lookup', np.mean(e_count)), ('irt_2pl_random', np.mean(e_rand)),
                       ('irt_2pl_informative', e_info), ('irt_2pl_adaptive', e_adapt)):
            out[key].append(float(v))
        print(f'{m:3d} {np.mean(e_count):13.4f} {np.mean(e_rand):11.4f} {e_info:16.4f} {e_adapt:13.4f}')

    out['informative_panel_example'] = [q20[q] for q in info_order[0][:10]]
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
