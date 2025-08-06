#!/usr/bin/env python
"""
    examples/helm/math/plot_dkps.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich import print as rprint

from dkps.dkps import DataKernelPerspectiveSpace
from dkps.embed import embed_api

# --
# Helpers

def dkps_df(df, **kwargs):
    model_names  = df.model.unique()
    instance_ids = df.instance_id.unique()
    
    embedding_dict = {}
    for model_name in model_names:
        sub = df[df.model == model_name]
        assert (sub.instance_id.values == instance_ids).all(), f'instance_ids are not the same for model {model_name}'
        embedding_dict[model_name] = np.row_stack(sub.embedding.values)
    
    # <<
    # Adding extra dimension because we only have one replicate
    embedding_dict = {k:v[:,None] for k,v in embedding_dict.items()}
    # >>
    
    return DataKernelPerspectiveSpace(**kwargs).fit_transform(embedding_dict, return_dict=True)


# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='math:subject=algebra')
    parser.add_argument('--inpath',  type=str, default='math.tsv')
    parser.add_argument('--score',   type=str, default='score')
    return parser.parse_args()

args = parse_args()

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv(args.inpath, sep='\t')
df = df[df.dataset == args.dataset]
assert df.shape[0] > 0, f'no data for dataset {args.dataset}'

df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

if args.score != 'score':
    print('{args.score} -> score')
    df['score'] = df[args.score]

# <<
# drop models w/ zero score?
y_acts = df.groupby('model').score.mean().to_dict()
for model, score in y_acts.items():
    if score == 0:
        df = df[df.model != model]

df = df.reset_index(drop=True)
# >>

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

os.makedirs('plots', exist_ok=True)

_ = plt.scatter(P[:, 0], P[:,1], c=model2score.values(), cmap='viridis')

_ = plt.xticks([])
_ = plt.yticks([])
_ = plt.xlabel('DKPS-0')
_ = plt.ylabel('DKPS-1')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.title(f'DKPS - {args.dataset}')
_ = plt.colorbar()
_ = plt.savefig(f'plots/{args.dataset}-dkps.png')
_ = plt.close()

# Plot first DKPS dimension vs performance
z = P[:,0]
_ = plt.scatter(z, model2score.values(), c=P[:,1], cmap='inferno')
_ = plt.xlabel('DKPS-0 (flipped, z-scored)')
_ = plt.ylabel('Score')
_ = plt.colorbar()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.savefig(f'plots/{args.dataset}-dkps0-vs-score.png')
_ = plt.close()

# Plot second DKPS dimension vs performance
z = P[:,1]
_ = plt.scatter(z, model2score.values(), c=P[:,0], cmap='inferno')
_ = plt.xlabel('DKPS-1 (flipped, z-scored)')
_ = plt.ylabel('Score')
_ = plt.colorbar()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.savefig(f'plots/{args.dataset}-dkps1-vs-score.png')
_ = plt.close()
