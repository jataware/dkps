"""Paper figures for the routing paper.

fig1_concept   two query-localized geometries (same models, two queries):
               the candidate nearest the flagship changes with the query
fig2_contract  certified volume vs tolerance at alpha = 10%, and the
               flagship-bill savings the certified volume buys
fig3_ablations pairing overlap, anchor cache density, candidate count,
               flagship rank

Reads the lever-sweep decision parquets (figs 2-3) and runs one geometry
pass for the concept figure. Outputs results/figures/*.{png,pdf}.

Run from repo root: pixi run python -m experiments.routing.fig_paper
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lever_sweep2 import _prep, batched_stats2, gate2
from .run_cost_routing import parse_params
from .run_helm import RESULTS

FIGDIR = os.path.join(RESULTS, 'figures')
C = {'pairdev': '#d62728', 'qa': '#1f77b4', 'oracle': '0.45'}


def _save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {name}')


# ---------------------------------------------------------------- concept
def fig_concept():
    from .run_combined import load_combined
    X, Qu, rows, n, names = load_combined()
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    _, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    flag = ranked['helm'][0]
    pool = {i for i in range(n) if suite_of[i] == 'helm'
            and sizes[i] is not None and sizes[i] <= 13.0}

    rng = np.random.default_rng(0)
    helm = rows[rows.suite == 'helm']
    resp = helm.groupby('query')['model'].agg(set)
    full = [q for q, ms in resp.items() if pool | {flag} <= ms]
    tasks = helm.drop_duplicates('query').set_index('query')['task']
    cand_q = list(rng.choice(full, size=40, replace=False))

    is_hold = rows['query'].isin(cand_q).to_numpy()
    anchor_m = ~is_hold
    Xa = X[anchor_m]
    sa = rows['score'].to_numpy()[anchor_m]
    model_a = rows['model'].to_numpy()[anchor_m]
    ua = Qu[rows['code'].to_numpy()[anchor_m]]
    groups = [np.flatnonzero(model_a == m) for m in range(n)]
    i = rng.integers(0, len(ua), 20000)
    k = rng.integers(0, len(ua), 20000)
    kp = i != k
    med = float(np.median(np.linalg.norm(ua[i[kp]] - ua[k[kp]], axis=1)))
    hold_groups = list(rows[is_hold].groupby('query'))
    Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in hold_groups])
    D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
    D2 = D2 - D2.min(axis=1, keepdims=True)
    W = np.exp(-D2 / (2.0 * (0.25 * med) ** 2)).astype(np.float32)
    phi, _, _, ok = batched_stats2(Xa, sa, groups, n, W)

    mods = sorted(pool) + [flag]
    picks, order = [], []
    for gi, (q, g) in enumerate(hold_groups):
        d = np.linalg.norm(phi[gi][sorted(pool)] - phi[gi][flag], axis=1)
        picks.append(sorted(pool)[int(np.argmin(d))])
        order.append((q, tasks[q]))
    # two queries from different datasets with different nearest candidates,
    # such that the pick stays nearest under the shared 2-D projection
    def dset(task):
        return task.split(':')[1]

    def project(a, b):
        Pa = phi[a][mods] - phi[a][mods].mean(axis=0, keepdims=True)
        Pb = phi[b][mods] - phi[b][mods].mean(axis=0, keepdims=True)
        P = np.concatenate([Pa, Pb])
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        P2 = P @ Vt[:2].T
        return P2[:len(mods)], P2[len(mods):]

    def faithful(pts, gi):
        d2 = np.linalg.norm(pts[:-1] - pts[-1], axis=1)
        return mods[int(np.argmin(d2))] == picks[gi]

    qa_i, qb_i, A, B = None, None, None, None
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            if dset(order[a][1]) == dset(order[b][1]) \
                    or picks[a] == picks[b]:
                continue
            A, B = project(a, b)
            if faithful(A, a) and faithful(B, b):
                qa_i, qb_i = a, b
                break
        if qa_i is not None:
            break
    assert qa_i is not None, 'no projection-faithful query pair found'

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, pts, gi in ((axes[0], A, qa_i), (axes[1], B, qb_i)):
        near = mods.index(picks[gi])
        ax.scatter(pts[:-1, 0], pts[:-1, 1], s=42, color='0.62',
                   zorder=2, label='candidate models')
        ax.scatter(*pts[-1], marker='*', s=340, color=C['pairdev'],
                   zorder=4, label='flagship')
        ax.scatter(*pts[near], s=130, facecolors='none',
                   edgecolors=C['qa'], linewidths=2.2, zorder=3,
                   label='nearest candidate')
        ax.set_title(f'query from {dset(order[gi][1])}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc='lower left', frameon=False, fontsize=8)
    fig.suptitle('The same cached models, localized to two queries: '
                 'the substitute nearest the flagship changes', y=1.02)
    _save(fig, 'fig1_concept')


# ---------------------------------------------------------------- contract
def _vol_curve(sub, eps_grid, alpha=0.10):
    out = []
    for eps in eps_grid:
        per_seed = []
        for _, ss in sub.groupby('seed'):
            cc = ss[ss.split == 'cal']['conf'].to_numpy()
            ce = ss[ss.split == 'eval']['conf'].to_numpy()
            cc, ce = (np.where(np.isfinite(x), x, 1e6) for x in (cc, ce))
            off = gate2(cc, ss[ss.split == 'cal']['dev'].to_numpy(),
                        ce, eps, alpha)
            dv = ss[ss.split == 'eval']['dev'].to_numpy()
            per_seed.append((off.mean(),
                             (dv[off] > eps).mean() if off.any() else 0.0))
        out.append(np.mean(per_seed, axis=0))
    return np.array(out)


def fig_contract():
    df = pd.read_parquet(os.path.join(RESULTS, 'lever2_conf.parquet'))
    d1 = df[df.variant == 'r1']
    eps_grid = np.arange(0.15, 0.75, 0.05)
    rd_dev = d1[(d1.pool == 'le13b') & (d1.method == 'random')
                & (d1.split == 'eval')]['dev'].median()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for pool, ls in (('le13b', '-'), ('all', '--')):
        sub = d1[d1.pool == pool]
        for meth, key in (('pairdev', 'pairdev'), ('qa', 'qa')):
            v = _vol_curve(sub[sub.method == meth], eps_grid)
            ax.plot(eps_grid, 100 * v[:, 0], ls, color=C[key], lw=2,
                    label=f'{meth} ({pool})')
        v = _vol_curve(sub[sub.method == 'oracle-conf'], eps_grid)
        ax.plot(eps_grid, 100 * v[:, 0], ls, color=C['oracle'], lw=1.2,
                label=f'oracle ({pool})')
    ax.axvline(rd_dev, color='k', lw=0.7, ls=':')
    ax.text(rd_dev + 0.01, 6, 'median deviation of\na random model swap',
            fontsize=7.5)
    ax.set_xlabel(r'tolerance $\varepsilon$ (embedding deviation)')
    ax.set_ylabel(r'certified traffic volume (%)  [$\alpha$ = 10%]')
    ax.set_title('Contract frontier: volume certifiable at\n'
                 r'P(deviation > $\varepsilon$) $\leq$ 10%')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    sub = d1[d1.pool == 'le13b']
    for meth, key in (('pairdev', 'pairdev'), ('qa', 'qa')):
        v = _vol_curve(sub[sub.method == meth], eps_grid)
        for R, lsr in ((25, '-'), (10, ':')):
            ax.plot(eps_grid, 100 * v[:, 0] * (1 - 1 / R), lsr,
                    color=C[key], lw=2 if R == 25 else 1.2,
                    label=f'{meth}, {R}x price ratio')
    ax.axvline(rd_dev, color='k', lw=0.7, ls=':')
    ax.set_xlabel(r'tolerance $\varepsilon$')
    ax.set_ylabel('flagship bill saved (%)')
    ax.set_title('Money: savings from certified offloading\n'
                 r'($\leq$13B substitutes; score-free)')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig2_contract')


# --------------------------------------------------------------- ablations
def fig_ablations():
    df2 = pd.read_parquet(os.path.join(RESULTS, 'lever2_conf.parquet'))
    dfp = pd.read_parquet(os.path.join(RESULTS, 'lever2_pairing.parquet'))
    df1 = pd.read_parquet(os.path.join(RESULTS, 'lever_decisions.parquet'))
    d1 = df2[df2.variant == 'r1']
    EPS = np.array([0.5])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))

    ax = axes[0, 0]                                # pairing overlap
    xs = [100, 50, 0]
    for pool, ls in (('le13b', '-'), ('all', '--')):
        for meth, key in (('pairdev', 'pairdev'), ('qa', 'qa')):
            ys = []
            for al in ('aligned', 'independent', 'disjoint'):
                sub = dfp[(dfp.pool == pool) & (dfp.variant == al)
                          & (dfp.method == meth)]
                ys.append(100 * _vol_curve(sub, EPS)[0, 0]
                          if len(sub) else 0.0)
            ax.plot(xs, ys, ls, marker='o', color=C[key], lw=2,
                    label=f'{meth} ({pool})')
    ax.annotate('pairdev inadmissible\n(no shared anchors)', (0, 2),
                fontsize=7.5, ha='left', xytext=(8, 18),
                arrowprops=dict(arrowstyle='-', lw=0.7))
    ax.set_xlabel('flagship-candidate anchor overlap (%)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(a) Pairing: guarantees need shared anchors\n'
                 '(HELM, fixed 50% cache per model)')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]                                # anchor cache density
    for pool, ls in (('le13b', '-'), ('all', '--')):
        sub = df1[(df1.pool == pool) & (df1.method == 'qa-0.25x')]
        xs, ys = [], []
        for af, ss in sub.groupby('anchor_frac'):
            xs.append(100 * af)
            ys.append(100 * _vol_curve(ss, EPS)[0, 0])
        ax.plot(xs, ys, ls, marker='o', color=C['qa'], lw=2,
                label=f'qa ({pool})')
    ax.set_xlabel('cached responses per model (% of full cache)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(b) Cache density: volume keeps paying\nfor history')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]                                # candidate scaling
    ks, vol, odev = [], [], []
    for pool, K in [(f'k{k}', k) for k in (2, 4, 8, 16, 32, 64)] \
            + [('all', 137)]:
        sub = d1[(d1.pool == pool) & (d1.method == 'qa')]
        if not len(sub):
            continue
        ks.append(K)
        vol.append(100 * _vol_curve(sub, EPS)[0, 0])
        odev.append(d1[(d1.pool == pool) & (d1.method == 'oracle-pick')
                       & (d1.split == 'eval')]['dev'].mean())
    ax.plot(ks, vol, '-o', color=C['qa'], lw=2, label='qa certified volume')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('candidate pool size (unrestricted, random subsets)')
    ax.set_ylabel('certified volume (%)', color=C['qa'])
    ax2 = ax.twinx()
    ax2.plot(ks, odev, '-s', color=C['oracle'], lw=1.5,
             label='oracle-pick deviation')
    ax2.set_ylabel('hindsight-best deviation', color=C['oracle'])
    ax.set_title('(c) Candidate scaling: volume saturates,\n'
                 'the floor keeps falling')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]                                # flagship rank
    width = 0.35
    ranks = ('r1', 'r5', 'rmed')
    lbl = ('best', '5th best', 'median')
    for j, (pool, col) in enumerate((('le13b', '#7f9cbd'),
                                     ('all', '#4a6d8c'))):
        ys = []
        for rk in ranks:
            sub = df2[(df2.pool == pool) & (df2.variant == rk)
                      & (df2.method == 'qa')]
            ys.append(100 * _vol_curve(sub, EPS)[0, 0])
        ax.bar(np.arange(3) + (j - 0.5) * width, ys, width, color=col,
               label=f'pool {pool}')
    ax.set_xticks(range(3))
    ax.set_xticklabels(lbl)
    ax.set_xlabel('deprecated target (rank by mean score)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(d) Target rank: non-extreme targets certify\n'
                 'more traffic when the pool is rich')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(r'Ablations at contract ($\varepsilon$=0.5, '
                 r'$\alpha$=10%), 3-5 seeds', y=1.0)
    fig.tight_layout()
    _save(fig, 'fig3_ablations')


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig_contract()
    fig_ablations()
    fig_concept()


if __name__ == '__main__':
    main()
