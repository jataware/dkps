#!/usr/bin/env python
"""
    joint.plot_dkps
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint

from utils import dkps_df
from dkps.embed import embed_api

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default='math:subject=algebra')
    parser.add_argument('--score_col', type=str, default='score')
    args = parser.parse_args()
    
    args.tsv_path = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    args.plot_dir = Path('plots')
    
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    
    return args

args = parse_args()

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

df['embedding'] = list(embed_api('jina', [str(xx) for xx in df.response.values]))

# --
# Run DKPS

model2score = df.groupby('model').score.mean().to_dict()

P = dkps_df(df, n_components_cmds=2)
P = np.row_stack([P[m] for m in model2score.keys()])

# --
# Plotting

_ = plt.scatter(P[:, 0], P[:,1], c=model2score.values(), cmap='viridis')

_ = plt.xticks([])
_ = plt.yticks([])
_ = plt.xlabel('DKPS-0')
_ = plt.ylabel('DKPS-1')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.title(f'DKPS - {args.dataset}')
_ = plt.colorbar()
_ = plt.savefig(args.plot_dir / f'{args.dataset}-dkps.png')
_ = plt.close()

# Plot first DKPS dimension vs performance
z = P[:,0]
_ = plt.scatter(z, model2score.values(), c=P[:,1], cmap='inferno')
_ = plt.xlabel('DKPS-0')
_ = plt.ylabel('Score')
_ = plt.colorbar()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.savefig(args.plot_dir / f'{args.dataset}-dkps0-vs-score.png')
_ = plt.close()

# Plot second DKPS dimension vs performance
z = P[:,1]
_ = plt.scatter(z, model2score.values(), c=P[:,0], cmap='inferno')
_ = plt.xlabel('DKPS-1')
_ = plt.ylabel('Score')
_ = plt.colorbar()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.savefig(args.plot_dir / f'{args.dataset}-dkps1-vs-score.png')
_ = plt.close()
