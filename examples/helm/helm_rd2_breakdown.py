#!/usr/bin/env python
"""
helm_rd2_breakdown.py

RD2 secondary figure: held-out (model, task) MAE broken down by task and by
model, for PKPS-combined vs matrix completion vs their ensemble, at a fixed
task-observation probability.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graspologic.embed import ClassicalMDS

from dkps.unpaired_dkps import ProductKernelPerspectiveSpace, pca_reduce_elbow
from dkps.baselines import matrix_completion_predict
import helm_doublekernel as H

METHODS = ['combined', 'matcomplete', 'ensemble']
COLORS = {'combined': '#1d4ed8', 'matcomplete': '#7c3aed', 'ensemble': '#15803d'}
LABELS = {'combined': 'PKPS', 'matcomplete': 'matrix completion', 'ensemble': 'ensemble'}


def one_seed(resp_X, Qu, qid_code, model_id, task_id, query_id, groups, score_mat,
             models, tasks, obs_prob, seed, k=5, suite=False, query_med=None):
    rng = np.random.default_rng(seed)
    observed = H.sample_observed(len(models), len(tasks), obs_prob, rng) & np.isfinite(score_mat)
    keep = []
    for i in range(len(models)):
        for t in range(len(tasks)):
            if observed[i, t]:
                idx = groups.get((models[i], tasks[t]))
                if idx is not None and len(idx):
                    keep.append(idx)
    keep = np.concatenate(keep)
    codes = qid_code[keep]
    if suite:
        df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                           'query_id': query_id[keep], 'embedding': list(resp_X[keep]),
                           'query_embedding': list(Qu[codes])})
        out = H.cv_predict_all(df, score_mat, observed, models, k=k, seed=seed,
                               response_kernel='linear', query_med=query_med)
    else:
        resp_red = pca_reduce_elbow(resp_X[keep])
        uq = np.unique(codes)
        qr = pca_reduce_elbow(Qu[uq])
        c2v = {c: qr[j] for j, c in enumerate(uq)}
        df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                           'query_id': query_id[keep], 'embedding': list(resp_red),
                           'query_embedding': [c2v[c] for c in codes]})
        out = H.cv_predict_all(df, score_mat, observed, models, k=k, seed=seed)
    comb, mc, ens = out['combined'], out['matcomplete'], out['ensemble']
    held = (~observed) & np.isfinite(score_mat) & np.isfinite(comb) & np.isfinite(mc)
    rows = []
    for i in range(len(models)):
        for t in range(len(tasks)):
            if held[i, t]:
                rows.append(dict(model=models[i], task=tasks[t],
                                 combined=abs(comb[i, t] - score_mat[i, t]),
                                 matcomplete=abs(mc[i, t] - score_mat[i, t]),
                                 ensemble=abs(ens[i, t] - score_mat[i, t])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--obs_prob', type=float, default=0.3)
    ap.add_argument('--n_seeds', type=int, default=30)
    ap.add_argument('--replot', action='store_true')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-rd2-breakdown')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    csv = outdir / f'breakdown_{args.dataset}.csv'
    if args.replot and csv.exists():
        df = pd.read_csv(csv)
    else:
        suite = args.dataset == 'suite'
        query_med = None
        if suite:
            data = H.load_suite()
            (resp_X, Qu, qid_code, model_id, task_id, query_id,
             score_mat, models, tasks, groups, row_score, query_med) = data
        elif args.dataset == 'pooled':
            data = H.load_pooled(('math', 'wmt_14'))
            (resp_X, Qu, qid_code, model_id, task_id, query_id,
             score_mat, models, tasks, groups, row_score) = data
        else:
            cfg = H.DATASETS[args.dataset]
            data = H.load_helm_math(cfg['parquet'], cfg['tsv'], query_source='google',
                                    query_parquet=cfg['query_parquet'], score_col=cfg['score_col'])
            (resp_X, Qu, qid_code, model_id, task_id, query_id,
             score_mat, models, tasks, groups, row_score) = data
        parts = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(one_seed)(resp_X, Qu, qid_code, model_id, task_id, query_id, groups,
                              score_mat, models, tasks, args.obs_prob, s, suite=suite,
                              query_med=query_med)
            for s in range(args.n_seeds))
        df = pd.DataFrame([r for sub in parts for r in sub])
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)

    # task labels: strip dataset prefix
    df['task_short'] = df['task'].str.replace(r'^.*[:=]', '', regex=True)

    by_model = df.groupby('model')[METHODS].mean()
    # shared x-range across both rows
    lim = max(np.quantile(np.abs(by_model['matcomplete'] - by_model['combined']), 0.99),
              np.quantile(np.abs(by_model['matcomplete'] - by_model['ensemble']), 0.99))
    bins = np.linspace(-lim, lim, 26)

    def diff_hist(ax, ref, name, color):
        diff = (by_model['matcomplete'] - by_model[ref]).values   # >0 => ref better
        ax.hist(np.clip(diff, -lim, lim), bins=bins, color=color, alpha=0.8)
        ax.axvline(0, color='k', lw=1)
        ax.axvline(diff.mean(), color='#b91c1c', lw=1.5, ls='--',
                   label=f'mean {diff.mean():+.3f}')
        ax.set_xlim(-lim, lim)
        ax.set_ylabel('# models')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        ax.text(0.97, 0.92, f'{name} lower for {(diff > 0).mean():.0%} of models',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.grid(axis='y', alpha=0.25, lw=0.6)

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[1, 1])

    # (a) MAE by task
    by_task = df.groupby('task_short')[METHODS].mean()
    x = np.arange(len(by_task))
    w = 0.26
    for j, m in enumerate(METHODS):
        ax_a.bar(x + (j - 1) * w, by_task[m], w, color=COLORS[m], label=LABELS[m])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(by_task.index, rotation=30, ha='right', fontsize=8)
    ax_a.set_ylabel('held-out MAE')
    ax_a.set_title('(a) MAE by task', loc='left', fontsize=10)
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.grid(axis='y', alpha=0.25, lw=0.6)

    # (b) paired per-model error differences (matrix completion minus method)
    diff_hist(ax_b1, 'combined', 'PKPS', COLORS['combined'])
    ax_b1.set_title('(b) per-model error diff: matrix completion vs PKPS', loc='left', fontsize=10)
    diff_hist(ax_b2, 'ensemble', 'ensemble', COLORS['ensemble'])
    ax_b2.set_title('(c) per-model error diff: matrix completion vs ensemble', loc='left', fontsize=10)
    ax_b2.set_xlabel(r'per-model $\overline{|\mathrm{err}_{MC}|} - \overline{|\mathrm{err}_{\bullet}|}$'
                     r'  ($>0$: method better)')

    fig.suptitle(f'RD2 missing tasks ({args.dataset.upper()}, obs_prob={args.obs_prob}): '
                 f'error breakdown', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig_rd2_breakdown_{args.dataset}.{ext}', dpi=200, bbox_inches='tight')
    print(by_task.round(4).to_string())
    print(f'wrote {outdir}/fig_rd2_breakdown_{args.dataset}.png')


if __name__ == '__main__':
    main()
