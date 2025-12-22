#!/usr/bin/env python
"""
    run_dkps.py - Unified runner for DKPS model prediction
"""

import os
import importlib

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint
from tqdm import trange
from joblib import Parallel, delayed

from dkps.embed import embed_api
from utils import onehot_embedding

# --
# Helpers

def model2family(model):
    return model.split('_')[0]


def predict_null(df, mode='model'):
    """ average score of other models / families """
    assert mode in ['model', 'family']

    out = {}
    for model in df.model.unique():
        if mode == 'model':
            sel = df.model != model
        elif mode == 'family':
            sel = df.model.apply(model2family) != model2family(model)

        out[model] = df.score[sel].mean()

    return out


def _rel_err(act, pred):
    return np.abs(pred - act) / act

def _abs_err(act, pred):
    return np.abs(pred - act)

err_fns = {
    "abs" : _abs_err,
    "rel" : _rel_err,
}

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner',         type=str,   default='dkps', choices=['dkps', 'qselect'])
    parser.add_argument('--dataset',        type=str,   default='math:subject=algebra')
    parser.add_argument('--score_col',      type=str,   default='score')
    parser.add_argument('--embed_provider', type=str,   default='jina')
    parser.add_argument('--embed_model',    type=str,   default=None)
    parser.add_argument('--err_fn',         type=str,   default='abs')
    parser.add_argument('--outdir',         type=str,   default='results')
    parser.add_argument('--sample',         type=float)
    parser.add_argument('--seed',           type=int,   default=123)
    parser.add_argument('--n_replicates',   type=int,   default=128)
    parser.add_argument('--n_jobs',         type=int,   default=-2)
    args = parser.parse_args()

    if args.embed_model == 'jina':
        assert os.environ.get('JINA_API_KEY') is not None, 'JINA_API_KEY is not set'
    elif args.embed_model == 'google':
        assert os.environ.get('GEMINI_API_KEY') is not None, 'GEMINI_API_KEY is not set'
    elif args.embed_model == 'jlai_tei':
        print('... jlai_tei requires some manual setup ... talk to @bkj ...')

    args.inpath = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    args.outpath = args.outdir / f'run-{args.runner}-{args.dataset}-{args.score_col}.tsv'

    return args

args = parse_args()

# --
# Load runner

runner = importlib.import_module(f'runners.{args.runner}')

# --
# Load data

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv(args.inpath, sep='\t')
df = df[df.dataset == args.dataset]

if args.sample:
    rng           = np.random.default_rng(args.seed)
    uinstance_ids = df.instance_id.unique()
    keep          = rng.choice(uinstance_ids, int(len(uinstance_ids) * args.sample), replace=False)
    df            = df[df.instance_id.isin(keep)]

df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

if args.score_col != 'score':
    print(f'{args.score_col} -> score')
    df['score'] = df[args.score_col]

# --
# QC

print(f'{len(df.response.unique())} / {df.shape[0]} responses are unique')
_instance_ids = df.groupby('model').instance_id.apply(list)
assert all([_instance_ids.iloc[0] == _instance_ids.iloc[i] for i in range(len(_instance_ids))]), 'instance_ids are not the same for each model'

# --
# Get embeddings

if args.embed_model == 'onehot':
    df = onehot_embedding(df, dataset=args.dataset)
else:
    df['embedding'] = list(embed_api(
        provider   = args.embed_provider,
        input_strs = [str(xx) for xx in df.response.values],
        model      = args.embed_model
    ))

# --
# Run

model_names  = df.model.unique()
instance_ids = df.instance_id.unique()
y_acts       = df.groupby('model').score.mean().to_dict()

modes     = ['model', 'family']
pred_null = {mode: predict_null(df, mode=mode) for mode in modes}

# --
# Run

runner_kwargs = runner.setup(df, model_names, args)

jobs = []
for iter in trange(args.n_replicates):
    rng = np.random.default_rng(iter)
    for n_samples in [2, 4, 8, 16, 32, 64, 128, 256, 512, len(instance_ids)]:
        if n_samples > len(instance_ids):
            continue

        instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
        df_sample           = df[df.instance_id.isin(instance_ids_sample)]
        jobs.append(delayed(runner.run_one)(
            df_sample    = df_sample,
            n_samples    = n_samples,
            mode         = 'family',
            seed         = iter,
            y_acts       = y_acts,
            pred_null    = pred_null,
            **runner_kwargs
        ))

res    = sum(Parallel(n_jobs=args.n_jobs, verbose=10)(jobs), [])
df_res = pd.DataFrame(res)

# --
# Clip predictions to [0, 1]

rprint('[yellow] Assumption - all metrics are bounded between 0 and 1[/yellow]')
dkps_cols = [c for c in df_res.columns if c.startswith('p_')]
rprint(f'[yellow]clipping DKPS columns to (0, 1) - {dkps_cols}[/yellow]')
for c in dkps_cols:
    df_res[c] = df_res[c].clip(0, 1)

# --
# Compute errors

for c in dkps_cols:
    df_res[c.replace('p_', 'e_')] = err_fns[args.err_fn](df_res.y_act, df_res[c])

# --
# Save

df_res.to_csv(args.outpath, sep='\t', index=False)
