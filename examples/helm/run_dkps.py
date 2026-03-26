#!/usr/bin/env python
"""
    run_dkps.py - Unified runner for DKPS model prediction
"""

import os
import importlib

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rich import print as rprint
from tqdm import trange
from joblib import Parallel, delayed

from dkps.embed import embed_api
from utils import onehot_embedding, make_experiment_path

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


# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir',         type=str,   default='results')
    
    parser.add_argument('--runner',         type=str,   default='dkps', choices=['dkps', 'qselect'])
    parser.add_argument('--embed_provider', type=str,   default='google')
    parser.add_argument('--embed_model',    type=str,   default=None)
    parser.add_argument('--dataset',        type=str,   default='math:subject=algebra')
    parser.add_argument('--score_col',      type=str,   default='score')
    
    parser.add_argument('--n_replicates',   type=int,   default=512)
    
    parser.add_argument('--sample',         type=float)
    parser.add_argument('--seed',           type=int,   default=123)
    parser.add_argument('--n_jobs',         type=int,   default=-1)
    args = parser.parse_args()

    args.inpath  = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    
    exp_path = make_experiment_path(args.embed_provider, args.embed_model, args.dataset, args.score_col, args.n_replicates)
    args.outpath = Path(args.outdir) / exp_path / args.runner / 'results-202603-modelvar.tsv'
    args.outpath.parent.mkdir(parents=True, exist_ok=True)

    rprint(f'[blue]outpath: {args.outpath}[/blue]')
    return args

args = parse_args()

# --
# Load runner

runner = importlib.import_module(f'runners.{args.runner}')

# --
# Load data

rprint('[blue]loading data ...[/blue]')

def load_data(inpath, dataset, use_all=False):
    df_all = pd.read_csv(inpath, sep='\t')

    if use_all:
        datasets = list(df_all.dataset.unique())
        print(datasets)
    else:
        datasets = [dataset]
    
    out = []
    for dataset in tqdm(datasets, desc='Loading data'):
        df = df_all[df_all.dataset == dataset]
        print(dataset, df.shape[0])

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
            df = onehot_embedding(df, dataset=dataset)
        else:
            df['embedding'] = list(embed_api(
                provider   = args.embed_provider,
                input_strs = [str(xx) for xx in df.response.values],
                model      = args.embed_model
            ))
        
        out.append(df)
    
    return pd.concat(out).reset_index(drop=True)

use_all = '=ALL' in args.dataset
df = load_data(args.inpath, args.dataset, use_all=use_all)

# --
# Run

model_names  = df.model.unique()
instance_ids = df.instance_id.unique()
y_acts       = df.groupby('model').score.mean().to_dict()

# <<
if 'wmt_14' in args.dataset:
    df = df[df.model != 'ai21_jamba-instruct'] # missing values


_instance_ids = df.groupby('model').instance_id.apply(list)
assert all([_instance_ids.iloc[0] == _instance_ids.iloc[i] for i in range(len(_instance_ids))]), 'instance_ids are not the same for each model'

assert (df.model.value_counts() == len(instance_ids)).all()
# >>

modes     = ['model', 'family']
pred_null = {mode: predict_null(df, mode=mode) for mode in modes}

# --
# Run

runner_kwargs = runner.setup(df, model_names, args)

# jobs = []
# for iter in trange(args.n_replicates):
#     rng = np.random.default_rng(iter)
#     # >>
#     # for n_samples in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, len(instance_ids)]:
#     # --
#     for n_samples in [1, 2, 4, 8, 16, 32, 64, 128]:
#     # <<
#         if n_samples > len(instance_ids):
#             continue
            
#         instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
#         df_sample           = df[df.instance_id.isin(instance_ids_sample)]
        
#         jobs.append(delayed(runner.run_one)(
#             df_sample    = df_sample,
#             n_samples    = n_samples,
#             mode         = 'family',
#             seed         = iter,
#             y_acts       = y_acts,
#             pred_null    = pred_null,
#             **runner_kwargs
#         ))

n_data_seeds  = 10
n_model_seeds = 100
n_samples = 5

jobs = []
for data_seed in trange(n_data_seeds):
    rng = np.random.default_rng(data_seed)
    instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
    
    for model_seed in trange(n_model_seeds):
        df_sample = df[df.instance_id.isin(instance_ids_sample)]
        
        jobs.append(delayed(runner.run_one)(
            df_sample    = df_sample,
            n_samples    = n_samples,
            mode         = 'family',
            seed         = data_seed,
            y_acts       = y_acts,
            pred_null    = pred_null,
            **runner_kwargs
        ))

jobs   = [jobs[i] for i in np.random.permutation(len(jobs))]
res    = sum(Parallel(backend='loky', n_jobs=args.n_jobs, verbose=10)(jobs), [])
df_res = pd.DataFrame(res)

# --
# Clip predictions to [0, 1]

rprint('[yellow] Assumption - all metrics are bounded between 0 and 1[/yellow]')
dkps_cols = [c for c in df_res.columns if c.startswith('p_')]
rprint(f'[yellow]clipping DKPS columns to (0, 1) - {dkps_cols}[/yellow]')
for c in dkps_cols:
    df_res[c] = df_res[c].clip(0, 1)

# --
# Save

df_res.to_csv(args.outpath, sep='\t', index=False)


data_seed = 0
rng = np.random.default_rng(data_seed)
instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
df_sample = df[df.instance_id.isin(instance_ids_sample)]
for replicate in range(100):
    for target_model in model_names:
        train_models = sample(model_names, 20)
        # ... do DKPS and compute error...

# groupby target model and compute [0, 25, 50, 75, 100] percentile error across replicates
# then average across models