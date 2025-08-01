#!/usr/bin/env python
"""
    examples/helm/wmt_14/plot_dkps.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich import print as rprint

from dkps.dkps import DataKernelPerspectiveSpace
from dkps.embed import embed_google

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
# Config

os.makedirs('plots', exist_ok=True)
dataset = 'wmt_14'

# --
# IO

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv('wmt_14.tsv', sep='\t')

# df = df[df.dataset == dataset]
df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

# --
# QC

print(f'{len(df.response.unique())} / {df.shape[0]} responses are unique')

# make sure all instance_ids are the same for each model
instance_ids = df.groupby('model').instance_id.apply(list)
assert all([instance_ids.iloc[0] == instance_ids.iloc[i] for i in range(len(instance_ids))]), 'instance_ids are not the same for each model'

# --
# Get embeddings

df['embedding'] = list(embed_google([str(xx) for xx in df.response.values]))

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
_ = plt.title(f'DKPS - {dataset}')
_ = plt.colorbar()
_ = plt.savefig('plots/dkps.png')
_ = plt.close()

# Plot first DKPS dimension vs performance
z = -1 * P[:,0]
z = z - z.mean()
z = z / z.std()

_ = plt.scatter(z, model2score.values(), c=P[:,1], cmap='inferno')
_ = plt.xlim(np.percentile(z, 2), z.max() + 0.1)
_ = plt.xlabel('DKPS-0 (flipped, z-scored)')
_ = plt.ylabel('BLEU-4')
_ = plt.colorbar()
_ = plt.savefig(f'plots/dkps0-vs-score.png')
_ = plt.close()

# Plot second DKPS dimension vs performance
z = -1 * P[:,1]
z = z - z.mean()
z = z / z.std()

_ = plt.scatter(z, model2score.values(), c=P[:,0], cmap='inferno')
_ = plt.xlabel('DKPS-1 (flipped, z-scored)')
_ = plt.ylabel('BLEU-4')
_ = plt.colorbar()
_ = plt.savefig(f'plots/dkps1-vs-score.png')
_ = plt.close()
