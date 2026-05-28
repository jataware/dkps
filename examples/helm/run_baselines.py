#!/usr/bin/env python
"""
    run_baselines.py — IRT and APW baselines for DKPS rebuttal.

    Runs LOFO evaluation on per-item score matrices (no embeddings needed).

    Usage:
        python run_baselines.py --dataset "math:subject=counting_and_probability"
        python run_baselines.py --dataset "med_qa"
        python run_baselines.py --dataset "wmt_14:language_pair=ru-en" --score_col meteor
        python run_baselines.py --dataset "legalbench:subset=abercrombie"
"""

import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint
from joblib import Parallel, delayed

from baselines import (
    irt_fit_difficulties, irt_estimate_ability, irt_predict,
    apw_fit, apw_predict,
)


def model2family(model):
    return model.split('_')[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str, required=True)
    parser.add_argument('--score_col',    type=str, default='score')
    parser.add_argument('--sample',       type=float, default=None,
                        help='Subsample fraction of instances (for WMT)')
    parser.add_argument('--n_lofo',       type=int, default=20)
    parser.add_argument('--n_samples',    type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--outdir',       type=str, default='results-baselines')
    parser.add_argument('--n_jobs',       type=int, default=-1)
    parser.add_argument('--seed',         type=int, default=123)
    args = parser.parse_args()

    args.inpath  = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    args.outpath = Path(args.outdir) / f'{args.dataset.replace(":", "-")}.tsv'
    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    return args


def build_score_matrix(df, model_names, instance_ids, score_col='score'):
    """
    Build score matrix S[i,j] = score of model i on instance j.

    Returns
    -------
    S : (n_models, M) ndarray
    """
    model_idx = {m: i for i, m in enumerate(model_names)}
    inst_idx  = {q: j for j, q in enumerate(instance_ids)}

    S = np.full((len(model_names), len(instance_ids)), np.nan)
    rows = df['model'].map(model_idx).values
    cols = df['instance_id'].map(inst_idx).values
    S[rows, cols] = df[score_col].values

    assert not np.isnan(S).any(), 'Score matrix has NaNs'
    return S


def run_one_lofo(lofo_seed, heldout_family, S, model_names, families,
                 n_samples_list, is_wmt):
    """
    One LOFO iteration for a specific held-out family.
    """
    rng = np.random.default_rng(lofo_seed)
    n_models, M = S.shape

    heldout_mask = np.array([f == heldout_family for f in families])
    ref_mask       = ~heldout_mask

    ref_idx     = np.where(ref_mask)[0]
    heldout_idx = np.where(heldout_mask)[0]

    S_ref = S[ref_idx]
    y_true_all = S.mean(axis=1)  # true benchmark score per model

    # -- IRT: fit on reference scores --
    if is_wmt:
        threshold = np.median(S_ref)
        S_ref_bin = (S_ref >= threshold).astype(float)
        S_bin     = (S >= threshold).astype(float)
    else:
        S_ref_bin = S_ref
        S_bin     = S

    t1 = time.time()
    beta, _ = irt_fit_difficulties(S_ref_bin)
    t_irt_fit = time.time() - t1

    # For WMT: learn a linear map from IRT predicted probability to raw score
    # using reference models (so no data leakage)
    irt_cal_slope, irt_cal_intercept = 1.0, 0.0
    if is_wmt:
        ref_irt_preds = np.array([
            irt_predict(
                irt_estimate_ability(S_bin[ri], beta),
                beta
            ) for ri in ref_idx
        ])
        ref_raw_scores = y_true_all[ref_idx]
        # simple linear regression for calibration
        if np.std(ref_irt_preds) > 1e-8:
            irt_cal_slope = np.cov(ref_raw_scores, ref_irt_preds)[0, 1] / np.var(ref_irt_preds)
            irt_cal_intercept = ref_raw_scores.mean() - irt_cal_slope * ref_irt_preds.mean()
        else:
            irt_cal_slope, irt_cal_intercept = 0.0, ref_raw_scores.mean()

    # -- APW: fit for each m (K-Medoids depends on m) --
    t2 = time.time()
    apw_fits = {}
    for m in n_samples_list:
        if m >= M:
            continue
        t_apw_start = time.time()
        anchor_idx, weights = apw_fit(S_ref, m, random_state=lofo_seed)
        apw_fits[m] = (anchor_idx, weights)
        if lofo_seed == 0:
            print(f'  [seed={lofo_seed}] APW fit m={m}, M={M}: {time.time() - t_apw_start:.1f}s')
    t_apw = time.time() - t2

    if lofo_seed == 0:
        print(f'  [seed={lofo_seed}] IRT fit: {t_irt_fit:.1f}s, APW fit: {t_apw:.1f}s (M={M})')

    # -- predict for each held-out model --
    results = []
    for hi in heldout_idx:
        target_model = model_names[hi]
        y_act  = float(y_true_all[hi])
        p_null = float(y_true_all[ref_idx].mean())

        for m in n_samples_list:
            if m > M:
                continue

            # random query subset (same for IRT and sample score)
            query_idx = rng.choice(M, size=min(m, M), replace=False)

            # sample score
            p_sample = float(S[hi, query_idx].mean())

            # IRT
            scores_m = S_bin[hi, query_idx]
            beta_m   = beta[query_idx]
            theta_hat = irt_estimate_ability(scores_m, beta_m)
            p_irt = irt_predict(theta_hat, beta)
            if is_wmt:
                p_irt = float(np.clip(
                    irt_cal_slope * p_irt + irt_cal_intercept, 0.0, 1.0
                ))

            # APW
            if m in apw_fits:
                anchor_idx, weights = apw_fits[m]
                p_apw = apw_predict(S[hi], anchor_idx, weights)
            else:
                p_apw = float(S[hi].mean())  # m >= M, use all

            results.append({
                'seed':         lofo_seed,
                'n_samples':    m,
                'mode':         'family',
                'target_model': target_model,
                'y_act':        y_act,
                'p_null':       p_null,
                'p_sample':     p_sample,
                'p_irt':        p_irt,
                'p_apw':        p_apw,
            })

    return results


def main():
    args = parse_args()

    # -- load data --
    rprint(f'[blue]Loading {args.inpath} for {args.dataset}[/blue]')
    df_all = pd.read_csv(args.inpath, sep='\t')
    df = df_all[df_all.dataset == args.dataset].copy()

    if args.score_col != 'score':
        rprint(f'[yellow]Using {args.score_col} as score column[/yellow]')
        df['score'] = df[args.score_col]

    # optional subsampling (for WMT consistency with DKPS runs)
    if args.sample:
        rng = np.random.default_rng(args.seed)
        uids = df.instance_id.unique()
        keep = rng.choice(uids, int(len(uids) * args.sample), replace=False)
        df = df[df.instance_id.isin(keep)]

    df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

    # drop models with missing values (wmt_14 edge case)
    if 'wmt_14' in args.dataset:
        df = df[df.model != 'ai21_jamba-instruct']

    model_names  = list(df.model.unique())
    instance_ids = list(df.instance_id.unique())
    families     = [model2family(m) for m in model_names]
    family_list  = sorted(set(families))
    is_wmt       = 'wmt_14' in args.dataset

    rprint(f'[green]{len(model_names)} models, {len(instance_ids)} instances, '
           f'{len(family_list)} families[/green]')

    # build score matrix
    S = build_score_matrix(df, model_names, instance_ids)
    rprint(f'[green]Score matrix: {S.shape}[/green]')

    # -- run LOFO (deterministic: iterate over all families × n_lofo seeds) --
    jobs = []
    for heldout_family in family_list:
        for lofo_seed in range(args.n_lofo):
            jobs.append(delayed(run_one_lofo)(
                lofo_seed, heldout_family, S, model_names, families,
                args.n_samples, is_wmt,
            ))

    rprint(f'[blue]Running {len(jobs)} LOFO iterations ({len(family_list)} families x {args.n_lofo} seeds) ...[/blue]')
    all_results = Parallel(n_jobs=args.n_jobs, verbose=10)(jobs)
    res = sum(all_results, [])

    df_res = pd.DataFrame(res)

    # clip predictions
    for c in ['p_irt', 'p_apw']:
        df_res[c] = df_res[c].clip(0, 1)

    # compute errors
    for c in ['p_null', 'p_sample', 'p_irt', 'p_apw']:
        df_res[f'e_{c[2:]}'] = np.abs(df_res[c] - df_res['y_act'])

    # -- summary --
    rprint('\n[bold cyan]MAE summary:[/bold cyan]')
    summary = df_res.groupby('n_samples').agg({
        'e_null':   'mean',
        'e_sample': 'mean',
        'e_irt':    'mean',
        'e_apw':    'mean',
    }).rename(columns={
        'e_null':   'Pop. Mean',
        'e_sample': 'Sample',
        'e_irt':    'IRT',
        'e_apw':    'APW',
    })
    print(summary)

    # -- save --
    df_res.to_csv(args.outpath, sep='\t', index=False)
    rprint(f'[green]Saved {len(df_res)} rows to {args.outpath}[/green]')


if __name__ == '__main__':
    main()
