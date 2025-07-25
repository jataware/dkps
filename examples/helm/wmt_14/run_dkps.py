#!/usr/bin/env python
"""
    examples/helm/wmt_14/run_dkps.py
"""

import os
import pickle
from turtle import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich import print as rprint

from joblib import Parallel, delayed

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

# Adding extra dimension because we only have one replicate
embedding_dict = {k:v[:,None] for k,v in embedding_dict.items()}

# --
# Run DKPS

model_names = list(embedding_dict.keys())

dkps = DataKernelPerspectiveSpace().fit_transform(embedding_dict, return_dict=False)

breakpoint()

# --
# Plotting

model2score = df.groupby('model').score.mean().to_dict()
_ = plt.scatter(dkps[:, 0], dkps[:,1], c=[model2score[xx] for xx in model_names], cmap='viridis')

_ = plt.xticks([])
_ = plt.yticks([])
_ = plt.xlabel('DKPS-0')
_ = plt.ylabel('DKPS-1')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.title(f'DKPS - {dataset}')
_ = plt.colorbar()
_ = plt.savefig(FIG_PATH)
_ = plt.close()

# Plot first DKPS dimension vs performance
z = -1 * dkps[:,0]
z = z - z.mean()
z = z / z.std()

_ = plt.scatter(z, [model2score[xx] for xx in model_names], c=dkps[:,1], cmap='inferno')
_ = plt.xlim(np.percentile(z, 2), z.max() + 0.1)
_ = plt.xlabel('DKPS-0 (flipped, z-scored)')
_ = plt.ylabel('BLEU-4')
_ = plt.colorbar()
_ = plt.savefig(f'tmp-0.png')
_ = plt.close()

# Plot second DKPS dimension vs performance
z = -1 * dkps[:,1]
z = z - z.mean()
z = z / z.std()

_ = plt.scatter(z, [model2score[xx] for xx in model_names], c=dkps[:,0], cmap='inferno')
_ = plt.xlabel('DKPS-1 (flipped, z-scored)')
_ = plt.ylabel('BLEU-4')
_ = plt.colorbar()
_ = plt.savefig(f'tmp-1.png')
_ = plt.close()

# --
# Efficient estimation

breakpoint()

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor

model_names = list(embedding_dict.keys())
instance_scores = df.pivot(index='model', columns='instance_id', values='score')
assert all(instance_scores.index.values == list(embedding_dict.keys()))
instance_ids    = instance_scores.columns.values
model_scores    = instance_scores.mean(axis=1).values


def compute_metrics(act, pred, prefix=None):
    out = {
        "err" : np.mean(np.abs(pred - act) / act),
        "spr" : spearmanr(pred, act)[0],
    }

    if prefix is not None:
        out = {f"{prefix}_{k}" : v for k,v in out.items()}

    return out

def get_splits(model_names, mode='loo'):
    if mode == 'loo':
        sels = LeaveOneOut().split(np.arange(len(model_names)))
        sels = list([sel[0] for sel in sels])
    
    elif mode == 'lgo':
        groups = [m.split('_')[0] for m in model_names]
        
        sels = []
        for group in groups:
            sel = np.array([i for i, g in enumerate(groups) if g != group])
            sels.append(sel)
            
    return sels

def predict_null(model_names, y, feats, uids, mode='loo'):
    sels = get_splits(model_names, mode=mode)
    return np.array([y[sel].mean() for sel in sels])

def predict_mu(model_names, y, feats, uids, mode='loo'):
    sels = get_splits(model_names, mode=mode)
    return np.array([y[sel].mean() for sel in sels])


def predict_rf():
    pass

def predict_lr():
    pass

def predict_knn():
    pass




def run_one(model_names, n_records, seed=None):
    rng = np.random.default_rng(123 + seed) if seed else np.random
    
    idxs_sample           = rng.choice(len(uids), size=n_records, replace=False)
    uids_sample           = uids[idxs_sample]
    embedding_dict_sample = {k:v[idxs_sample] for k,v in embedding_dict.items()}
    
    model_names
    y
    
    # predictions
    all_feats = {
        "dkps_2" : DataKernelPerspectiveSpace(n_components_cmds=2).fit_transform(embedding_dict_sample, return_dict=True),
        "dkps_8" : DataKernelPerspectiveSpace(n_components_cmds=8).fit_transform(embedding_dict_sample, return_dict=True),
    }

    
    rf2    = RandomForestRegressor(n_estimators=512, oob_score=True, random_state=234 + iter).fit(dkps2, y)    
    rf8    = RandomForestRegressor(n_estimators=512, oob_score=True, random_state=234 + iter).fit(dkps8, y)
    
    p_lr2  = cross_val_predict(LinearRegression(), dkps2, y, cv=LeaveOneOut())
    p_lr8  = cross_val_predict(LinearRegression(), dkps8, y, cv=LeaveOneOut())
    
    p_knn2  = cross_val_predict(KNeighborsRegressor(n_neighbors=1), dkps2, y, cv=LeaveOneOut())
    p_knn8  = cross_val_predict(KNeighborsRegressor(n_neighbors=1), dkps8, y, cv=LeaveOneOut())
    
    # mu - average of scores for each model
    p_mu = df[df.instance_id.isin(uids_sample)].groupby('model').score.mean()
    assert all(p_mu.index == model_names)
    
    p_null_f = y.mean() # slight cheat - should be doing LOO
    
    
    # metrics
    e_rf2_dkps = np.mean(np.abs(rf2.oob_prediction_ - y) / y)
    e_rf8_dkps = np.mean(np.abs(rf8.oob_prediction_ - y) / y)
    e_lr2_dkps = np.mean(np.abs(p_lr2 - y) / y)
    e_lr8_dkps = np.mean(np.abs(p_lr8 - y) / y)
    e_knn2_dkps = np.mean(np.abs(p_knn2 - y) / y)
    e_knn8_dkps = np.mean(np.abs(p_knn8 - y) / y)
    e_mu       = np.mean(np.abs(p_mu - y) / y)
    e_null     = np.mean(np.abs(p_null_f - y) / y)
    
    r_rf2_dkps = spearmanr(rf2.oob_prediction_, y)[0]
    r_rf8_dkps = spearmanr(rf8.oob_prediction_, y)[0]
    r_lr2_dkps = spearmanr(p_lr2, y)[0]
    r_lr8_dkps = spearmanr(p_lr8, y)[0]
    r_knn2_dkps = spearmanr(p_knn2, y)[0]
    r_knn8_dkps = spearmanr(p_knn8, y)[0]
    r_mu       = spearmanr(p_mu, y)[0]
    r_null      = 0
    
    return {
        "iter"      : iter,
        "n_records" : n_records,
        
        "e_rf2_dkps" : e_rf2_dkps,
        "e_rf8_dkps" : e_rf8_dkps,
        "e_lr2_dkps" : e_lr2_dkps,
        "e_lr8_dkps" : e_lr8_dkps,
        "e_knn2_dkps" : e_knn2_dkps,
        "e_knn8_dkps" : e_knn8_dkps,
        "e_mu"       : e_mu,
        "e_null"     : e_null,
        
        "r_rf2_dkps" : r_rf2_dkps,
        "r_rf8_dkps" : r_rf8_dkps,
        "r_lr2_dkps" : r_lr2_dkps,
        "r_lr8_dkps" : r_lr8_dkps,
        "r_knn2_dkps" : r_knn2_dkps,
        "r_knn8_dkps" : r_knn8_dkps,
        "r_mu"       : r_mu,
        "r_null"     : r_null,
    }

jobs = []
for iter in range(10):
    for n_records in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256]:
        jobs.append(delayed(run_one)(iter=iter, n_records=n_records))

res = Parallel(n_jobs=-1, verbose=10)(jobs)

df_res = pd.DataFrame(res)

tmp = df_res.groupby(['n_records']).mean()
tmp = tmp.reset_index()

# plot error
for c in df_res.columns:
    if 'e_' in c:
        # _ = plt.scatter(df_res.n_records, df_res[c], alpha=0.25, s=8)
        _ = plt.plot(tmp.n_records, tmp[c], label=f'{c}')

_ = plt.legend()
_ = plt.xlabel('n_records')
_ = plt.ylabel('mean(abs(pred - y) / y)')
_ = plt.title('DKPS vs. null')
_ = plt.xscale('log')
# _ = plt.yscale('log')
_ = plt.savefig('res0.png')
_ = plt.close()

# plot corr
for c in df_res.columns:
    if 'r_' in c:
        # _ = plt.scatter(df_res.n_records, df_res[c], alpha=0.25, s=8)
        _ = plt.plot(tmp.n_records, tmp[c], label=f'{c}')

_ = plt.legend()
_ = plt.xlabel('n_records')
_ = plt.ylabel('spearmanr(pred, y)')
_ = plt.title('DKPS vs. null')
_ = plt.xscale('log')
# _ = plt.yscale('log')
_ = plt.savefig('res1.png')
_ = plt.close()
