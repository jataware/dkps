#!/usr/bin/env python
"""
    examples/helm/wmt_14/run_dkps.py
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from rich import print as rprint

from dkps.dkps import DataKernelPerspectiveSpace
from dkps.embed import embed_google

# --
# Config

dataset    = 'wmt_14'
FIG_PATH   = f'fig-{dataset}.png'
USE_CACHE  = True
CACHE_PATH = f'.cache/{dataset}/embedding_dict.pkl'
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

# --
# IO

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv('wmt_14.tsv', sep='\t')
df.dataset.unique()

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

input_strs = [str(xx) for xx in df.response.values]

if USE_CACHE and os.path.exists(CACHE_PATH):
    rprint('[green]loading embeddings ...[/green]')
    embedding_dict = pickle.load(open(CACHE_PATH, 'rb'))
else:
    rprint('[blue]computing embeddings ...[/blue]')
    all_embeddings = embed_google(input_strs)

    embedding_dict = {}
    for model in df.model.unique():
        embedding_dict[model] = all_embeddings[(df.model == model).values]

    if USE_CACHE:
        pickle.dump(embedding_dict, open(CACHE_PATH, 'wb'))

# --
# Run DKPS

dkps = DataKernelPerspectiveSpace().fit_transform(embedding_dict)

model_names = list(embedding_dict.keys())
model2score = df.groupby('model').score.mean().to_dict()
_ = plt.scatter(dkps[:, 0], dkps[:,1], c=[model2score[xx] for xx in model_names], cmap='viridis')

# for i, model_name in enumerate(model_names):
#     _ = plt.annotate(
#         model_name,(dkps[i, 0], dkps[i, 1]), ha='center', fontsize=8
#     )

_ = plt.xticks([])
_ = plt.yticks([])
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.title(f'DKPS - {dataset}')
_ = plt.colorbar()
_ = plt.savefig(FIG_PATH)
_ = plt.close()

breakpoint()
import numpy as np
_ = plt.scatter(np.log10(dkps[:,0]), [model2score[xx] for xx in model_names])
_ = plt.savefig(f'tmp.png')
_ = plt.close()