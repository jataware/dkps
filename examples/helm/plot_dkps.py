#!/usr/bin/env python
"""
    helm.plot_dkps
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint

from utils import dkps_df, onehot_embedding
from dkps.embed import embed_api

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, default='math:subject=algebra')
    parser.add_argument('--score_col',      type=str, default='score')
    parser.add_argument('--embed_provider', type=str, default='jina', choices=['jina', 'google', 'jlai_tei'])
    parser.add_argument('--embed_model',    type=str, default=None)
    parser.add_argument('--sample',         type=float)
    parser.add_argument('--seed',           type=int, default=123)
    args = parser.parse_args()
    
    args.tsv_path = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    args.plot_dir = Path('plots') / args.dataset.replace(':', '-')
    
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    
    return args

args = parse_args()
np.random.seed(args.seed)

# --
# IO

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv(args.tsv_path, sep='\t')
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

# BAD_MODELS  = open('bad_models.txt').read().splitlines()

# --
# QC

print(f'{len(df.response.unique())} / {df.shape[0]} responses are unique')
instance_ids = df.groupby('model').instance_id.apply(list)
assert all([instance_ids.iloc[0] == instance_ids.iloc[i] for i in range(len(instance_ids))]), 'instance_ids are not the same for each model'

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

model2score = df.groupby('model').score.mean().to_dict()

# --
# Plot 1.a - whole DKPS

P = dkps_df(df, n_components_cmds=2)
P = np.vstack([P[m] for m in model2score.keys()])

_ = plt.scatter(P[:, 0], P[:,1], c=model2score.values(), cmap='viridis')
_ = plt.xticks([])
_ = plt.yticks([])
_ = plt.xlabel('DKPS-0')
_ = plt.ylabel('DKPS-1')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.title(f'DKPS - {args.dataset}')
_ = plt.colorbar(label='Score')
_ = plt.savefig(args.plot_dir / 'dkps.png')
_ = plt.close()

# # --
# # Plot 1.b - whole DKPS, bad models removed

# thresh     = np.percentile(list(model2score.values()), 10)
# BAD_MODELS = [m for m in model2score.keys() if model2score[m] < thresh]

# _model2score = {m:model2score[m] for m in model2score.keys() if m not in BAD_MODELS}
# P = dkps_df(df[~df.model.isin(BAD_MODELS)], n_components_cmds=2)
# P = np.vstack([P[m] for m in _model2score.keys()])

# _ = plt.scatter(P[:, 0], P[:,1], c=_model2score.values(), cmap='viridis')
# _ = plt.xticks([])
# _ = plt.yticks([])
# _ = plt.xlabel('DKPS-0')
# _ = plt.ylabel('DKPS-1')
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.title(f'DKPS - {args.dataset}')
# _ = plt.colorbar(label='Score')
# _ = plt.savefig(args.plot_dir / 'dkps-excl.png')
# _ = plt.close()

# --
# Plot 2 - grid, varying number of instances and models

uinstance_ids = np.random.choice(df.instance_id.unique(), size=min(50, len(df.instance_id.unique())), replace=False)
umodels       = np.random.permutation(df.model.unique())

fig, axes = plt.subplots(2, 3, figsize=(12, 10))

Ps = {}
for c, n_instances in enumerate([2, 10, len(uinstance_ids)]):
    _instance_ids = uinstance_ids[:n_instances]
    for r, n_models in enumerate([20, len(umodels)]):
        _models      = umodels[:n_models]
        _model2score = {m:model2score[m] for m in _models}
        
        df_sub = df[df.instance_id.isin(_instance_ids)]
        df_sub = df_sub[df_sub.model.isin(_models)]
        P_sub  = dkps_df(df_sub, n_components_cmds=2)
        P_sub  = np.vstack([P_sub[m] for m in _model2score.keys()])

        ax = axes[r, c]

        scatter = ax.scatter(P_sub[:, 0], P_sub[:, 1], c=list(_model2score.values()), cmap='viridis')
        _ = ax.set_xticks([])
        _ = ax.set_yticks([])
        _ = ax.set_xlabel('DKPS-0')
        _ = ax.set_ylabel('DKPS-1')
        _ = ax.grid('both', alpha=0.25, c='gray')
        _ = ax.set_title(f'n_models={n_models} | n_instances={n_instances}')

_ = plt.suptitle(f'DKPS - {args.dataset}')
_ = plt.tight_layout()

# Add colorbar to the figure
cbar = plt.colorbar(scatter, ax=axes, shrink=0.8, aspect=20)
cbar.set_label('Score')

_ = plt.savefig(args.plot_dir / 'dkps-grid.png')
_ = plt.close()
