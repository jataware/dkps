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
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from dkps.dkps import DataKernelPerspectiveSpace
from dkps.embed import embed_google

# --
# Helpers

def model2family(model):
    return model.split('_')[0]


def rel_err(act, pred):
    return np.abs(pred - act) / act

    
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
df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

# --
# QC

print(f'{len(df.response.unique())} / {df.shape[0]} responses are unique')
_instance_ids = df.groupby('model').instance_id.apply(list)
assert all([_instance_ids.iloc[0] == _instance_ids.iloc[i] for i in range(len(_instance_ids))]), 'instance_ids are not the same for each model'

# --
# Get embeddings

df['embedding'] = list(embed_google([str(xx) for xx in df.response.values]))

DROP_OUTLIERS = False
if DROP_OUTLIERS:
    print('starting with', len(set(df.model.values)), 'models')
    model_scores = df.groupby('model').score.mean()
    bad_models   = model_scores[model_scores <= 0.15].index
    print('dropping', len(bad_models), 'models')
    df           = df[~df.model.isin(bad_models)].reset_index(drop=True)
    print('ending with', len(set(df.model.values)), 'models')

# --
# Run

model_names  = df.model.unique()
instance_ids = df.instance_id.unique()
y_acts       = df.groupby('model').score.mean().to_dict()

modes = ['model', 'family']

pred_null = {mode: predict_null(df, mode=mode) for mode in modes}
err_null  = {
    mode : {
        model_name: rel_err(act=y_acts[model_name], pred=pred_null[mode][model_name]) for model_name in model_names
    } for mode in modes
}

# --
# Simple - DKPS w/ more than one example

def run_one(df_sample, n_samples, mode, seed):
    out = []
    model_names = df_sample.model.unique()
    for target_model in model_names:
        
        # split data
        assert mode in ['model', 'family']
        if mode == 'model':
            train_models = np.array([m for m in model_names if m != target_model])
        elif mode == 'family':
            target_family = model2family(target_model)
            train_models  = np.array([m for m in model_names if model2family(m) != target_family])
        
        df_train = df_sample[df_sample.model.isin(train_models)]
        df_test  = df_sample[df_sample.model == target_model]
        
        # compute DKPS embeddings + get labels
        P       = dkps_df(pd.concat([df_train, df_test]).reset_index(drop=True), n_components_cmds=2, dissimilarity="euclidean") # [!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TESTING]
        X_train = np.row_stack([P[m] for m in train_models])
        X_test  = np.row_stack([P[target_model]])
        y_train = np.array([y_acts[m] for m in train_models])
        y_test  = y_acts[target_model]

        # average score over the `n_samples` evaluated
        p_sample = df_test.score.mean()

        # linear regression on DKPS embeddings        
        lr         = LinearRegression().fit(X_train, y_train)
        p_lr_dkps2 = float(lr.predict(X_test)[0])
        
        # knn on DKPS embeddings
        knn         = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
        p_knn_dkps2 = float(knn.predict(X_test)[0])
        
        out.append({
            "seed"         : seed,
            "n_samples"    : n_samples,
            "mode"         : mode,
            "target_model" : target_model,
            
            "y_act"        : y_test,
            "p_null"       : pred_null[mode][target_model],
            "p_sample"     : p_sample,
            "p_dkps2"      : p_lr_dkps2,
            "p_knn_dkps2"  : p_knn_dkps2,
        })
    
    return out


jobs = []
for iter in trange(32):
    rng = np.random.default_rng(iter)
    for n_samples in [1, 2, 4, 8, 16, 32, 64, 128]:
        instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
        df_sample           = df[df.instance_id.isin(instance_ids_sample)]
        
        jobs.append(delayed(run_one)(df_sample=df_sample, n_samples=n_samples, mode='family', seed=iter))

res    = sum(Parallel(n_jobs=-1, verbose=10)(jobs), [])
df_res = pd.DataFrame(res)

# --
# Post-processing

# compute errors - abs(pred - act) / act
for c in df_res.columns:
    if 'p_' in c:
        df_res[c.replace('p_', 'e_')] = rel_err(df_res.y_act, df_res[c])

df_per_model = df_res.groupby(['target_model', 'mode', 'n_samples']).agg({
    'y_act'       : 'mean', # noop - they're all the same
    'e_null'      : 'mean',
    'e_sample'    : 'mean',
    'e_dkps2'     : 'mean',
    'e_knn_dkps2' : 'mean',
}).reset_index()

df_avg = df_res.groupby(['mode', 'n_samples']).agg({
    'y_act'       : 'median', # noop - they're all the same
    'e_null'      : 'median',
    'e_sample'    : 'median',
    'e_dkps2'     : 'median',
    'e_knn_dkps2' : 'median',
}).reset_index()

breakpoint()

# --
# Plot

# Plot performance averaged over models
_ = plt.plot(df_avg.n_samples, df_avg.e_null, label='null')
_ = plt.plot(df_avg.n_samples, df_avg.e_sample, label='sample')
_ = plt.plot(df_avg.n_samples, df_avg.e_dkps2, label='dkps2')
_ = plt.plot(df_avg.n_samples, df_avg.e_knn_dkps2, label='knn')
_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.savefig('err2.png')
_ = plt.close()


# Plot gain over null, per model
df_per_model['dkps2_gain']  = df_per_model.e_dkps2 - df_per_model.e_null
df_per_model['sample_gain'] = df_per_model.e_sample - df_per_model.e_null
df_per_model['knn_gain']    = df_per_model.e_knn_dkps2 - df_per_model.e_null

for model in model_names:
    sub = df_per_model[df_per_model.target_model == model]
    _ = plt.plot(sub.n_samples, sub.dkps2_gain, c='red', alpha=0.1)
    _ = plt.plot(sub.n_samples, sub.sample_gain, c='blue', alpha=0.1)
    _ = plt.plot(sub.n_samples, sub.knn_gain, c='green', alpha=0.1)

_ = plt.plot(df_per_model.groupby('n_samples').dkps2_gain.mean(), label='dkps2', c='red', linewidth=5)
_ = plt.plot(df_per_model.groupby('n_samples').sample_gain.mean(), label='sample', c='blue', linewidth=5)
_ = plt.plot(df_per_model.groupby('n_samples').knn_gain.mean(), label='knn', c='green', linewidth=5)

_ = plt.legend()
_ = plt.ylim(-0.75, 0.75)
_ = plt.axhline(0, c='black')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.savefig(f'err_by_model2.png')
_ = plt.close()