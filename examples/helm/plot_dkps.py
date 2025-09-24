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
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, default='math:subject=algebra')
    parser.add_argument('--score_col',      type=str, default='score')
    parser.add_argument('--embed_provider', type=str, default='jina', choices=['jina', 'google', 'jlai_tei'])
    parser.add_argument('--embed_model',    type=str, default=None)
    parser.add_argument('--seed',           type=int,   default=123)
    args = parser.parse_args()

    if args.embed_model == 'jina':
        assert os.environ.get('JINA_API_KEY') is not None, 'JINA_API_KEY is not set'
    elif args.embed_model == 'google':
        assert os.environ.get('GEMINI_API_KEY') is not None, 'GEMINI_API_KEY is not set'
    elif args.embed_model == 'jlai_tei':
        print('... jlai_tei requires some manual setup ... talk to @bkj ...')
    
    args.tsv_path = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    args.plot_dir = Path('plots')
    
    args.plot_dir.mkdir(parents=True, exist_ok=True)
        
    return args

args = parse_args()
np.random.seed(args.seed)

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv(args.tsv_path, sep='\t')
df = df[df.dataset == args.dataset]
df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

if args.score_col != 'score':
    print(f'{args.score_col} -> score')
    df['score'] = df[args.score_col]

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

# --
# Run DKPS

model2score = df.groupby('model').score.mean().to_dict()

P = dkps_df(df, n_components_cmds=2)
P = np.vstack([P[m] for m in model2score.keys()])

# --
# Plotting

_ = plt.scatter(P[:, 0], P[:,1], c=model2score.values(), cmap='viridis')

_ = plt.xticks([])
_ = plt.yticks([])
_ = plt.xlabel('DKPS-0')
_ = plt.ylabel('DKPS-1')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.title(f'DKPS - {args.dataset}')
_ = plt.colorbar(label='Score')
_ = plt.savefig(args.plot_dir / f'{args.dataset}-dkps.png')
_ = plt.close()

# <<
# [BKJ 2025-09-19 - I don't think this is useful ...]
# # Plot first DKPS dimension vs performance
# z = P[:,0]
# _ = plt.scatter(z, model2score.values(), c=P[:,1], cmap='inferno')
# _ = plt.xlabel('DKPS-0')
# _ = plt.ylabel('Score')
# _ = plt.colorbar(label='DKPS-1')
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.savefig(args.plot_dir / f'{args.dataset}-dkps0-vs-score.png')
# _ = plt.close()

# # Plot second DKPS dimension vs performance
# z = P[:,1]
# _ = plt.scatter(z, model2score.values(), c=P[:,0], cmap='inferno')
# _ = plt.xlabel('DKPS-1')
# _ = plt.ylabel('Score')
# _ = plt.colorbar(label='DKPS-0')
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.savefig(args.plot_dir / f'{args.dataset}-dkps1-vs-score.png')
# _ = plt.close()
# >>

# --
# 2x2 Plot

uinstance_ids = np.random.choice(df.instance_id.unique(), size=50, replace=False)
umodels       = np.random.permutation(df.model.unique())

fig, axes = plt.subplots(2, 3, figsize=(12, 10))

Ps = {}
for c, n_instances in enumerate([2, 10, 50]):
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

_ = plt.savefig(args.plot_dir / f'{args.dataset}-dkps-grid.png')
_ = plt.close()
