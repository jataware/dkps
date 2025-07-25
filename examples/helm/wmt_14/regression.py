#!/usr/bin/env python
"""
    examples/helm/wmt_14/model_dkps.py
"""

import os
from tkinter.constants import FALSE
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from rich import print as rprint
from tqdm import tqdm
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
_instance_ids = df.groupby('model').instance_id.apply(list)
assert all([_instance_ids.iloc[0] == _instance_ids.iloc[i] for i in range(len(_instance_ids))]), 'instance_ids are not the same for each model'

# --
# Get embeddings

df['embedding'] = list(embed_google([str(xx) for xx in df.response.values]))
breakpoint()

# DROP_OUTLIERS = False
# if DROP_OUTLIERS:
#     print('starting with', len(set(df.model.values)), 'models')
#     model_scores = df.groupby('model').score.mean()
#     bad_models   = model_scores[model_scores <= 0.15].index
#     print('dropping', len(bad_models), 'models')
#     df           = df[~df.model.isin(bad_models)].reset_index(drop=True)
#     print('ending with', len(set(df.model.values)), 'models')

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

def ez_dkps(X_train, X_test, **kwargs):
    dist_matrix = squareform(pdist(np.row_stack([X_train, X_test]), metric='euclidean'))
    cmds_embds  = ClassicalMDS(**kwargs).fit_transform(dist_matrix)
    return cmds_embds[:-1], cmds_embds[[-1]]

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

model_names  = df.model.unique()
instance_ids = df.instance_id.unique()
y_acts       = df.groupby('model').score.mean().to_dict()

pred_null = {
    "model"  : predict_null(df, mode='model'),
    "family" : predict_null(df, mode='family'),
}

def rel_err(act, pred):
    return np.abs(pred - act) / act

err_null = {
    "model"  : {model_name: rel_err(y_acts[model_name], pred_null['model'][model_name]) for model_name in model_names},
    "family" : {model_name: rel_err(y_acts[model_name], pred_null['family'][model_name]) for model_name in model_names},
}

rprint(err_null['model'])
print(sorted(err_null['model'].values()))

# --
# Simple - DKPS w/ one example

from tqdm import trange
from scipy.spatial.distance import pdist, squareform
from graspologic.embed import ClassicalMDS

mode = 'family'

out = []
for _ in trange(32):
    
    instance_ids_sample = np.random.choice(instance_ids, size=1, replace=False)
    df_sample           = df[df.instance_id.isin(instance_ids_sample)]
    
    for target_model in model_names:
        
        if mode == 'model':
            train_models = np.array([m for m in model_names if m != target_model])
        elif mode == 'family':
            train_models = np.array([m for m in model_names if model2family(m) != model2family(target_model)])
        else:
            raise ValueError(f'mode must be one of "model" or "family", got {mode}')
        
        df_train = df_sample[df_sample.model.isin(train_models)]
        df_test  = df_sample[df_sample.model == target_model]
        
        X_train = np.row_stack(df_train.embedding.values)
        X_test  = np.row_stack(df_test.embedding.values)
        y_train = np.array([y_acts[m] for m in df_train.model.values])
        
        p_sample = df_test.score.mean()
        
        # knn on features
        knn    = KNeighborsRegressor(n_neighbors=1, metric='cosine').fit(X_train, y_train)
        p_orig = float(knn.predict(X_test)[0])
        
        # lr on embeddings
        P_train, P_test = ez_dkps(X_train, X_test, n_components=2, n_elbows=None)
        lr              = LinearRegression().fit(P_train, y_train)
        p_lr_dkps2      = float(lr.predict(P_test)[0])
        
        out.append({
            "target_model" : target_model,
            "y_act"        : y_acts[target_model],
            "p_null"       : pred_null[mode][target_model],
            "p_sample"     : p_sample,
            "p_orig"       : p_orig,
            "p_dkps2"      : p_lr_dkps2,
        })

df1 = pd.DataFrame(out)

for c in df1.columns:
    if 'p_' in c:
        df1[c.replace('p_', 'e_')] = rel_err(df1.y_act, df1[c])


df1['e_null'].mean(), df1['e_sample'].mean(), df1['e_orig'].mean(), df1['e_dkps2'].mean()

z = df1.groupby('target_model').agg({
    'y_act'   : 'mean',
    'p_dkps2' : 'mean',
})

_ = plt.scatter(z.y_act, z.p_dkps2)
plt.show()

# --

# --
# Simple - DKPS w/ more than one example

from tqdm import trange
from scipy.spatial.distance import pdist, squareform
from graspologic.embed import ClassicalMDS


def run_one(df, n_samples, mode, seed):
    rng = np.random.default_rng(seed)
    
    instance_ids_sample = rng.choice(instance_ids, size=2, replace=False)
    df_sample           = df[df.instance_id.isin(instance_ids_sample)]
    
    for target_model in model_names:
        
        if mode == 'model':
            train_models = np.array([m for m in model_names if m != target_model])
        elif mode == 'family':
            train_models = np.array([m for m in model_names if model2family(m) != model2family(target_model)])
        else:
            raise ValueError(f'mode must be one of "model" or "family", got {mode}')
        
        df_train = df_sample[df_sample.model.isin(train_models)]
        df_test  = df_sample[df_sample.model == target_model]
        y_train  = np.array([y_acts[m] for m in train_models])
        y_test   = y_acts[target_model]
        
        # lr on embeddings
        P       = dkps_df(pd.concat([df_train, df_test]).reset_index(drop=True), n_components_cmds=2)
        P_train = np.row_stack([P[m] for m in train_models])
        P_test  = np.row_stack([P[target_model]])
        
        lr              = LinearRegression().fit(P_train, y_train)
        p_lr_dkps2      = float(lr.predict(P_test)[0])
        
        return {
            "seed"         : seed,
            "n_samples"    : n_samples,
            "mode"         : mode,
            "target_model" : target_model,
            
            "y_act"        : y_test,
            "p_null"       : pred_null[mode][target_model],
            "p_sample"     : p_sample,
            # "p_orig"       : p_orig,
            "p_dkps2"      : p_lr_dkps2,
        }

jobs = []
for _ in trange(32):
    jobs.append(delayed(run_one)(df, 2, 'family'))
out = Parallel(n_jobs=-1)(jobs)


mode = 'family'

out = []
for _ in trange(32):
    


df1 = pd.DataFrame(out)

for c in df1.columns:
    if 'p_' in c:
        df1[c.replace('p_', 'e_')] = rel_err(df1.y_act, df1[c])


rprint(df1['e_null'].mean(), df1['e_sample'].mean(), df1['e_dkps2'].mean())

z = df1.groupby('target_model').agg({
    'y_act'   : 'mean',
    'p_dkps2' : 'mean',
})

_ = plt.scatter(z.y_act, z.p_dkps2)
plt.show()



# --

def plot_one(df_avg, pred_null, mode='model'):
    # plot error
    sub = df_avg[df_avg['mode'] == mode]
    for c in sub.columns:
        if 'err_' in c:
            # _ = plt.scatter(df_res.n_records, df_res[c], alpha=0.25, s=8)
            _ = plt.plot(sub.n_records, sub[c], label=f'{c}')
    
    err_null = pred_null[mode][target_model]
    err_null = np.abs(err_null - y_acts[target_model]) / y_acts[target_model]
    
    _ = plt.axhline(err_null, c='gray', ls='--', label='null')
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