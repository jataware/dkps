#!/usr/bin/env python
"""
    run_combined.py — DKPS + IRT/APW combined methods.

    Implements:
      - DKPS+IRT:  append IRT ability θ̂ as extra feature to DKPS coordinates
      - APW→DKPS:  use APW K-Medoids query selection instead of random for DKPS
      - APW→DKPS+IRT: both combined

    Usage:
        python run_combined.py --dataset "math:subject=counting_and_probability" --embed_provider google
        python run_combined.py --dataset "legalbench:subset=abercrombie"  --embed_model onehot
        python run_combined.py --dataset "med_qa"                        --embed_model onehot
        python run_combined.py --dataset "wmt_14:language_pair=ru-en"    --embed_provider google --score_col meteor --sample 0.2
"""

import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from dkps.embed import embed_api
from dkps.dkps import DataKernelPerspectiveSpace as DKPS
from utils import onehot_embedding
from baselines import (
    irt_fit_difficulties, irt_estimate_ability, irt_predict,
    apw_fit,
)


def model2family(model):
    return model.split('_')[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, required=True)
    parser.add_argument('--score_col',      type=str, default='score')
    parser.add_argument('--embed_provider', type=str, default='google')
    parser.add_argument('--embed_model',    type=str, default=None)
    parser.add_argument('--sample',         type=float, default=None)
    parser.add_argument('--n_lofo',         type=int, default=20)
    parser.add_argument('--n_samples',      type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--n_components',   type=int, default=8)
    parser.add_argument('--outdir',         type=str, default='results-combined')
    parser.add_argument('--n_jobs',         type=int, default=-1)
    parser.add_argument('--seed',           type=int, default=123)
    args = parser.parse_args()

    args.inpath  = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    args.outpath = Path(args.outdir) / f'{args.dataset.replace(":", "-")}.tsv'
    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    return args


def build_score_matrix(df, model_names, instance_ids):
    """S[i,j] = score of model i on instance j."""
    model_idx = {m: i for i, m in enumerate(model_names)}
    inst_idx  = {q: j for j, q in enumerate(instance_ids)}
    S = np.full((len(model_names), len(instance_ids)), np.nan)
    rows = df['model'].map(model_idx).values
    cols = df['instance_id'].map(inst_idx).values
    S[rows, cols] = df['score'].values
    assert not np.isnan(S).any(), 'Score matrix has NaNs'
    return S


def build_embedding_dict(df, model_names, instance_ids, query_mask):
    """Build DKPS embedding dict for selected queries."""
    sel_ids = instance_ids[query_mask]
    df_sel  = df[df.instance_id.isin(sel_ids)]

    emb_dict = {}
    for model in model_names:
        sub = df_sel[df_sel.model == model].sort_values('instance_id')
        embs = np.vstack(sub.embedding.values)
        emb_dict[model] = embs[:, None]  # (m, 1, emb_dim)
    return emb_dict


def run_one_lofo(lofo_seed, heldout_family, S, S_bin, df, model_names,
                 instance_ids, families, n_samples_list, n_components,
                 is_wmt):
    rng = np.random.default_rng(lofo_seed)
    n_models, M = S.shape

    heldout_mask = np.array([f == heldout_family for f in families])
    ref_mask       = ~heldout_mask
    ref_idx        = np.where(ref_mask)[0]
    heldout_idx    = np.where(heldout_mask)[0]

    S_ref     = S[ref_idx]
    S_ref_bin = S_bin[ref_idx]
    y_true    = S.mean(axis=1)

    # IRT: fit item difficulties on reference models
    t1 = time.time()
    beta, _ = irt_fit_difficulties(S_ref_bin)
    t_irt_fit = time.time() - t1

    ref_model_names = [model_names[i] for i in ref_idx]

    results = []

    for m in n_samples_list:
        if m > M:
            continue

        # --- Query selection ---
        # Random
        rand_qidx = rng.choice(M, size=m, replace=False)

        # APW K-Medoids
        t2 = time.time()
        apw_qidx, _ = apw_fit(S_ref, m, random_state=lofo_seed)
        t_apw = time.time() - t2

        if lofo_seed == 0:
            print(f'  [seed={lofo_seed}] m={m}, M={M}: IRT fit {t_irt_fit:.1f}s, APW fit {t_apw:.1f}s')

        for qsel_name, qidx in [('rand', rand_qidx), ('apw', apw_qidx)]:
            # Build query mask
            qmask = np.zeros(M, dtype=bool)
            qmask[qidx] = True

            # DKPS embeddings on selected queries
            all_models_for_dkps = list(ref_model_names)  # will add target below
            # We need to compute DKPS per target model (since target is included)

            for hi in heldout_idx:
                target_model = model_names[hi]
                y_act  = float(y_true[hi])
                p_null = float(y_true[ref_idx].mean())
                p_sample = float(S[hi, qidx].mean())

                # DKPS: embed reference + target (shuffle to avoid CMDS ordering artifacts)
                models_for_dkps = list(ref_model_names) + [target_model]
                np.random.default_rng(lofo_seed).shuffle(models_for_dkps)
                emb_dict = build_embedding_dict(df, models_for_dkps,
                                                instance_ids, qmask)

                P = DKPS(n_components_cmds=n_components)
                P = P.fit_transform(emb_dict, return_dict=True)

                n_cmp = min(n_components, len(ref_model_names) - 1)

                X_train = np.vstack([P[rm][:n_cmp] for rm in ref_model_names])
                y_train = np.array([y_true[ref_idx[i]] for i in range(len(ref_idx))])
                X_test  = P[target_model][:n_cmp].reshape(1, -1)

                # --- Variant 1: plain DKPS (for comparison) ---
                lr = LinearRegression().fit(X_train, y_train)
                p_dkps = float(lr.predict(X_test)[0])

                # --- Variant 2: DKPS + IRT θ̂ ---
                # IRT abilities for reference models
                theta_train = np.array([
                    irt_estimate_ability(S_bin[ref_idx[i], qidx], beta[qidx])
                    for i in range(len(ref_idx))
                ])
                # IRT ability for target
                theta_test = irt_estimate_ability(S_bin[hi, qidx], beta[qidx])

                # --- Variant 2a: IRT alone ---
                p_irt = irt_predict(theta_test, beta)

                # --- Variant 2b: DKPS + IRT θ̂ ---
                X_train_irt = np.column_stack([X_train, theta_train])
                X_test_irt  = np.append(X_test.ravel(), theta_test).reshape(1, -1)

                lr_irt = LinearRegression().fit(X_train_irt, y_train)
                p_dkps_irt = float(lr_irt.predict(X_test_irt)[0])

                # --- Ensemble: α * p_method + (1-α) * p_sample, α = (M-m)/M ---
                alpha = (M - m) / M
                p_ens_dkps     = np.clip(alpha * p_dkps     + (1 - alpha) * p_sample, 0, 1)
                p_ens_dkps_irt = np.clip(alpha * p_dkps_irt + (1 - alpha) * p_sample, 0, 1)

                results.append({
                    'seed':           lofo_seed,
                    'n_samples':      m,
                    'query_sel':      qsel_name,
                    'target_model':   target_model,
                    'y_act':          y_act,
                    'p_null':         p_null,
                    'p_sample':       p_sample,
                    'p_irt':          np.clip(p_irt, 0, 1),
                    'p_dkps':         np.clip(p_dkps, 0, 1),
                    'p_dkps_irt':     np.clip(p_dkps_irt, 0, 1),
                    'p_ens_dkps':     p_ens_dkps,
                    'p_ens_dkps_irt': p_ens_dkps_irt,
                })

    return results


def main():
    args = parse_args()

    # -- Load data --
    rprint(f'[blue]Loading {args.inpath} for {args.dataset}[/blue]')
    df_all = pd.read_csv(args.inpath, sep='\t')
    if '=ALL' in args.dataset:
        df = df_all.copy()
    else:
        df = df_all[df_all.dataset == args.dataset].copy()

    if args.score_col != 'score':
        rprint(f'[yellow]Using {args.score_col} as score column[/yellow]')
        df['score'] = df[args.score_col]

    if args.sample:
        rng = np.random.default_rng(args.seed)
        uids = df.instance_id.unique()
        keep = rng.choice(uids, int(len(uids) * args.sample), replace=False)
        df = df[df.instance_id.isin(keep)]

    df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

    if 'wmt_14' in args.dataset:
        df = df[df.model != 'ai21_jamba-instruct']

    # -- Compute embeddings --
    rprint('[blue]Computing embeddings ...[/blue]')
    if args.embed_model == 'onehot':
        df = onehot_embedding(df, dataset=args.dataset)
    else:
        df['embedding'] = list(embed_api(
            provider   = args.embed_provider,
            input_strs = [str(xx) for xx in df.response.values],
            model      = args.embed_model,
        ))

    model_names  = list(df.model.unique())
    instance_ids = np.array(list(df.groupby('model').instance_id.first().values
                                 if False else df.instance_id.unique()))
    # ensure consistent ordering
    instance_ids = np.array(sorted(df.instance_id.unique()))
    df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

    families    = [model2family(m) for m in model_names]
    family_list = sorted(set(families))
    is_wmt      = 'wmt_14' in args.dataset

    rprint(f'[green]{len(model_names)} models, {len(instance_ids)} instances, '
           f'{len(family_list)} families[/green]')

    # -- Build score matrix --
    S = build_score_matrix(df, model_names, instance_ids)

    if is_wmt:
        threshold = np.median(S)
        S_bin = (S >= threshold).astype(float)
    else:
        S_bin = S.copy()

    # -- Run LOFO (deterministic: iterate over all families × n_lofo seeds) --
    jobs = []
    for heldout_family in family_list:
        for seed in range(args.n_lofo):
            jobs.append(delayed(run_one_lofo)(
                seed, heldout_family, S, S_bin, df, model_names,
                instance_ids, families, args.n_samples, args.n_components,
                is_wmt,
            ))

    rprint(f'[blue]Running {len(jobs)} LOFO iterations ({len(family_list)} families x {args.n_lofo} seeds) ...[/blue]')
    all_results = Parallel(n_jobs=args.n_jobs, verbose=10)(jobs)
    res = sum(all_results, [])

    df_res = pd.DataFrame(res)

    # Compute errors
    pred_cols = ['p_null', 'p_sample', 'p_irt', 'p_dkps', 'p_dkps_irt',
                 'p_ens_dkps', 'p_ens_dkps_irt']
    for c in pred_cols:
        df_res[f'e_{c[2:]}'] = np.abs(df_res[c] - df_res['y_act'])

    # -- Summary --
    rprint('\n[bold cyan]MAE summary:[/bold cyan]')
    summary = df_res.groupby(['query_sel', 'n_samples']).agg({
        'e_null':         'mean',
        'e_sample':       'mean',
        'e_irt':          'mean',
        'e_dkps':         'mean',
        'e_dkps_irt':     'mean',
        'e_ens_dkps':     'mean',
        'e_ens_dkps_irt': 'mean',
    }).rename(columns={
        'e_null':         'Pop. Mean',
        'e_sample':       'Sample',
        'e_irt':          'IRT',
        'e_dkps':         'DKPS',
        'e_dkps_irt':     'DKPS+IRT',
        'e_ens_dkps':     'Ens(DKPS)',
        'e_ens_dkps_irt': 'Ens(DKPS+IRT)',
    })
    print(summary)

    # -- Save --
    df_res.to_csv(args.outpath, sep='\t', index=False)
    rprint(f'[green]Saved {len(df_res)} rows to {args.outpath}[/green]')


if __name__ == '__main__':
    main()
