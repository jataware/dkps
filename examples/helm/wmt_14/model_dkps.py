#!/usr/bin/env python
"""
    examples/helm/wmt_14/model_dkps.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from rich import print as rprint
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor

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

def model2family(model):
    return model.split('_')[0]

# --
# IO

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv('wmt_14.tsv', sep='\t')
df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

df['family'] = df.model.apply(model2family)

# --
# QC

print(f'{len(df.response.unique())} / {df.shape[0]} responses are unique')
instance_ids = df.groupby('model').instance_id.apply(list)
assert all([instance_ids.iloc[0] == instance_ids.iloc[i] for i in range(len(instance_ids))]), 'instance_ids are not the same for each model'

# --
# Get embeddings

df['embedding'] = list(embed_google([str(xx) for xx in df.response.values]))


DROP_OUTLIERS = True
if DROP_OUTLIERS:
    print('starting with', len(set(df.model.values)), 'models')
    model_scores = df.groupby('model').score.mean()
    bad_models   = model_scores[model_scores <= 0.15].index
    print('dropping', len(bad_models), 'models')
    df           = df[~df.model.isin(bad_models)].reset_index(drop=True)
    print('ending with', len(set(df.model.values)), 'models')

breakpoint()

# --
# Run DKPS

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

feats = dkps_df(df, n_components_cmds=2)


# sanity check
# model2score = df.groupby('model').score.mean().to_dict()
# _ = plt.scatter(
#     [feats[k][0] for k in model2score.keys()], 
#     [feats[k][1] for k in model2score.keys()], 
#     c=[model2score[xx] for xx in model2score.keys()], 
#     cmap='viridis'
# )
# _ = plt.xticks([])
# _ = plt.yticks([])
# _ = plt.xlabel('DKPS-0')
# _ = plt.ylabel('DKPS-1')
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.title(f'DKPS - {dataset}')
# _ = plt.colorbar()
# _ = plt.savefig('model-tmp.png')
# _ = plt.close()

# --
# Efficient estimation

def compute_metrics(pred, target, suffix=None):
    assert pred.keys() == target.keys()
    
    _pred   = np.array([pred[k] for k in pred.keys()])
    _target = np.array([target[k] for k in target.keys()])
    
    out = {
        "err" : np.mean(np.abs(_pred - _target) / _target),
        "spr" : spearmanr(_pred, _target)[0],
    }
    
    if suffix is not None:
        out = {f"{k}_{suffix}" : v for k,v in out.items()}
    
    return out


def predict_null(df, mode='model'):
    """ average score of other models / families """
    assert mode in ['model', 'family']
    
    out = {}
    for model in df.model.unique():
        if mode == 'model':
            sel = df.model != model
        elif mode == 'family':
            sel = df.family != model2family(model)
        
        out[model] = df.score[sel].mean()
    
    return out


def predict_lr_dkps(feats, target, mode='model'):
    """ linear regression on DKPS features - leave-one-out over model OR family """
    assert mode in ['model', 'family']
    
    out = {}
    for model in feats.keys():
        if mode == 'model':
            train_models = [m for m in feats.keys() if m != model]
        elif mode == 'family':
            train_models = [m for m in feats.keys() if model2family(m) != model2family(model)]
        
        X_train = np.row_stack([feats[m] for m in train_models])
        y_train = np.array([target[m] for m in train_models])
        lr      = LinearRegression().fit(X_train, y_train)
        pred    = lr.predict(feats[model][None])
        
        out[model] = float(pred[0])
    
    return out


def run_one(df_sample, target, n_records, mode='model', seed=None):    
    # predictions
    all_feats = {
        "dkps_2" : dkps_df(df_sample, n_components_cmds=2),
        "dkps_8" : dkps_df(df_sample, n_components_cmds=8),
    }
    
    # pred_sample - average of scores on the sampled instances
    pred_sample = df_sample.groupby('model').score.mean().to_dict()
    
    # pred_lr{k} - linear regression on dkps_{k}
    pred_lr2    = predict_lr_dkps(all_feats['dkps_2'], target, mode=mode)
    pred_lr8    = predict_lr_dkps(all_feats['dkps_8'], target, mode=mode)
    
    return {
        "seed"      : seed,
        "n_records" : n_records,
        "mode"      : mode,
        
        **compute_metrics(pred_sample, target, suffix='sample'),
        **compute_metrics(pred_lr2, target,    suffix='lr2'),
        **compute_metrics(pred_lr8, target,    suffix='lr8'),
    }


# act - actual performance on entire benchmark
target = df.groupby('model').score.mean().to_dict()

# pred_null - average of scores on all instances for other models / families
pred_null = {
    "model"  : predict_null(df, mode='model'),
    "family" : predict_null(df, mode='family'),
}

np.random.seed(123)

instance_ids = df.instance_id.unique()

jobs = []
for seed in trange(10):
    for n_records in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for mode in ['model', 'family']:
            instance_ids_sample = np.random.choice(instance_ids, size=n_records, replace=False)
            df_sample           = df[df.instance_id.isin(instance_ids_sample)]
            jobs.append(delayed(run_one)(df_sample=df_sample, target=target, n_records=n_records, seed=seed, mode=mode))

res    = Parallel(n_jobs=-1, verbose=10)(jobs)
df_res = pd.DataFrame(res)
df_avg = df_res.groupby(['n_records', 'mode']).mean().reset_index()
breakpoint()

def plot_one(df_avg, pred_null, mode='model'):
    # plot error
    sub = df_avg[df_avg['mode'] == mode]
    for c in sub.columns:
        if 'err_' in c:
            # _ = plt.scatter(df_res.n_records, df_res[c], alpha=0.25, s=8)
            _ = plt.plot(sub.n_records, sub[c], label=f'{c}')
    
    _ = plt.axhline(np.mean(list(pred_null[mode].values())), c='gray', ls='--', label='null')
    _ = plt.legend()
    _ = plt.xlabel('n_records')
    _ = plt.ylabel('mean(abs(pred - y) / y)')
    _ = plt.title(f'DKPS vs. null - LOO {mode}')
    _ = plt.xscale('log')
    _ = plt.grid('both', alpha=0.25, c='gray')
    _ = plt.savefig(f'err-{mode}.png')
    _ = plt.close()


plot_one(df_avg, pred_null, mode='model')
plot_one(df_avg, pred_null, mode='family')

z = df_avg.groupby('n_records').apply(lambda x: x[x['mode'] == 'family'].err_lr2.iloc[0] - x[x['mode'] == 'model'].err_lr2.iloc[0])
z.values

# !! This is probably wrong - models in the family should be excluded from DKPS as well.



# if family - exclude all other models in family from _everything_
# if model  - 