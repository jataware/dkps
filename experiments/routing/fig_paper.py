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
C = {'pairdev': '#e11d48', 'qa': '#2563eb', 'oracle': '#94a3b8',
     'flag': '#e11d48', 'pickA': '#2563eb', 'pickB': '#f59e0b',
     'grey': '#9ca3af', 'slate1': '#7f9cbd', 'slate2': '#4a6d8c'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#334155',
    'axes.labelcolor': '#1e293b',
    'axes.titlecolor': '#0f172a',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'legend.frameon': False,
    'legend.fontsize': 8,
    'grid.color': '#cbd5e1',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.5,
    'figure.facecolor': 'white',
})


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
            and sizes[i] is not None and sizes[i] <= 40.0} - {flag}

    rng = np.random.default_rng(0)
    helm = rows[rows.suite == 'helm']
    resp = helm.groupby('query')['model'].agg(set)
    full = [q for q, ms in resp.items() if pool | {flag} <= ms]
    tasks = helm.drop_duplicates('query').set_index('query')['task']
    cand_q = list(rng.choice(full, size=120, replace=False))

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

    def legibility(A, B, iA, iB):
        # smallest separation among the highlighted relations, in span units
        span = max(np.ptp(np.concatenate([A, B]), axis=0))
        gaps = []
        for pts in (A, B):
            gaps.append(np.linalg.norm(pts[iA] - pts[iB]))
            gaps += [np.linalg.norm(pts[j] - pts[-1]) for j in (iA, iB)]
        return min(gaps) / span

    best = (-1.0, None)
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            if dset(order[a][1]) == dset(order[b][1]) \
                    or picks[a] == picks[b]:
                continue
            Pa, Pb = project(a, b)
            if not (faithful(Pa, a) and faithful(Pb, b)):
                continue
            ia, ib = mods.index(picks[a]), mods.index(picks[b])
            score = legibility(Pa, Pb, ia, ib)
            if score > best[0]:
                best = (score, (a, b, Pa, Pb))
    assert best[1] is not None, 'no projection-faithful query pair found'
    qa_i, qb_i, A, B = best[1]

    mA, mB = picks[qa_i], picks[qb_i]
    iA, iB = mods.index(mA), mods.index(mB)

    def short(m):
        s = names[m].split(':', 1)[1]
        return s if len(s) <= 26 else s[:24] + '…'

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, pts, gi in ((axes[0], A, qa_i), (axes[1], B, qb_i)):
        near = mods.index(picks[gi])
        others = [j for j in range(len(mods) - 1) if j not in (iA, iB)]
        ax.scatter(pts[others, 0], pts[others, 1], s=26, color=C['grey'],
                   alpha=0.55, linewidths=0, zorder=2,
                   label='cached models')
        # dashed link flagship -> this query's nearest
        ax.plot([pts[-1, 0], pts[near, 0]], [pts[-1, 1], pts[near, 1]],
                ls=(0, (2, 3)), color='#64748b', lw=1.1, zorder=3)
        for j, key, m, off in ((iA, 'pickA', mA, (9, 9)),
                               (iB, 'pickB', mB, (9, -14))):
            is_near = j == near
            ax.scatter(*pts[j], s=120 if is_near else 90, color=C[key],
                       edgecolors='white', linewidths=1.2, zorder=5)
            if is_near:
                ax.scatter(*pts[j], s=340, facecolors='none',
                           edgecolors=C[key], linewidths=1.6, zorder=4)
            ax.annotate(short(m), pts[j], textcoords='offset points',
                        xytext=off, fontsize=7.5, color=C[key],
                        fontweight='bold' if is_near else 'normal')
        ax.scatter(*pts[-1], marker='*', s=430, color=C['flag'],
                   edgecolors='white', linewidths=0.8, zorder=6,
                   label='flagship')
        ax.annotate(short(flag), pts[-1], textcoords='offset points',
                    xytext=(-10, -14), fontsize=7.5, color=C['flag'],
                    ha='right')
        ax.set_title(f'query from {dset(order[gi][1])}')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color('#e2e8f0')
        ax.margins(0.10)
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], marker='*', ls='', ms=15, color=C['flag'],
                      label='flagship'),
        mlines.Line2D([], [], marker='o', ls='', ms=8, color=C['pickA'],
                      label=f'nearest for the {dset(order[qa_i][1])} query'),
        mlines.Line2D([], [], marker='o', ls='', ms=8, color=C['pickB'],
                      label=f'nearest for the {dset(order[qb_i][1])} query'),
        mlines.Line2D([], [], marker='o', ls='', ms=6, color=C['grey'],
                      alpha=0.55, label='other cached models'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('One response cache, two queries: the localized geometry '
                 'reranks the substitutes', y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
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
    for j, (pool, col) in enumerate((('le13b', C['slate1']),
                                     ('all', C['slate2']))):
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
