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
        dist_matrix = squareform(pdist(np.row_stack([X_train, X_test]), metric='euclidean'))
        cmds_embds  = ClassicalMDS(n_components=2, n_elbows=None).fit_transform(dist_matrix)
        
        lr         = LinearRegression().fit(cmds_embds[:-1], y_train)
        p_lr_dkps2 = float(lr.predict(cmds_embds[[-1]])[0])
        
        out.append({
            "target_model" : target_model,
            "y_act"        : y_acts[target_model],
            "p_null"       : pred_null[mode][target_model],
            "p_sample"     : p_sample,
            "p_orig"       : p_orig,
            "p_dkps2"      : p_lr_dkps2,
        })




z = pd.DataFrame(out)

z.e_0.mean(), z.e_1.mean(), z.e_null.mean()



# --
# Efficient estimation


def predict_lr_dkps(feats, target_model, train_models, y_acts):
    """ linear regression on DKPS features - leave-one-out over model OR family """
    
    X_train = np.row_stack([feats[m] for m in train_models])
    y_train = np.array([y_acts[m] for m in train_models])
    
    lr      = LinearRegression().fit(X_train, y_train)
    pred    = lr.predict(feats[target_model][None])
        
    return float(pred[0])

def predict_knn_dkps(feats, target_model, train_models, y_acts):
    X_train = np.row_stack([feats[m] for m in train_models])
    y_train = np.array([y_acts[m] for m in train_models])
    
    knn     = KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)
    pred    = knn.predict(feats[target_model][None])
    return float(pred[0])

def run_one(df_sample, target_model, train_models, y_acts, n_records, mode, seed):    
    # predictions
    all_feats = {
        "dkps_2" : dkps_df(df_sample, n_components_cmds=2),
        "dkps_8" : dkps_df(df_sample, n_components_cmds=8),
    }
    
    y_act = y_acts[target_model]
    
    # pred_lr{k} - linear regression on dkps_{k}
    pred_lr2    = predict_lr_dkps(all_feats['dkps_2'], target_model, train_models, y_acts)
    pred_lr8    = predict_lr_dkps(all_feats['dkps_8'], target_model, train_models, y_acts)
    
    pred_knn2   = predict_knn_dkps(all_feats['dkps_2'], target_model, train_models, y_acts)
    pred_knn8   = predict_knn_dkps(all_feats['dkps_8'], target_model, train_models, y_acts)
    
    return {
        "seed"         : seed,
        "n_records"    : n_records,
        "target_model" : target_model,
        "mode"         : mode,
        
        "err_lr2"      : float(np.abs(pred_lr2 - y_act) / y_act),
        "err_lr8"      : float(np.abs(pred_lr8 - y_act) / y_act),
        
        "err_knn2"     : float(np.abs(pred_knn2 - y_act) / y_act),
        "err_knn8"     : float(np.abs(pred_knn8 - y_act) / y_act),
    }


# act - actual performance on entire benchmark
y_acts = df.groupby('model').score.mean().to_dict()

# pred_null - average of scores on all instances for other models / families


np.random.seed(123)

model_names  = df.model.unique()
instance_ids = df.instance_id.unique()

jobs = []
for seed in trange(10):
    for n_records in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for mode in ['model', 'family']:
            instance_ids_sample = np.random.choice(instance_ids, size=n_records, replace=False)
            df_sample           = df[df.instance_id.isin(instance_ids_sample)]
            
            # <<
            # target_model = np.random.choice(model_names, size=1, replace=False)
            # --
            target_model = 'google_gemini-1.5-pro-001'
            # >>
            
            # exclude models
            if mode == 'model':
                train_models = [m for m in model_names if m != target_model]
            elif mode == 'family':
                train_models = [m for m in model_names if model2family(m) != model2family(target_model)]
            
            df_sample = df_sample[(df_sample.model == target_model) | df_sample.model.isin(train_models)]
            
            jobs.append(delayed(run_one)(
                df_sample     = df_sample,
                target_model  = target_model,
                train_models  = train_models,
                
                y_acts        = y_acts,
                n_records     = n_records,
                seed          = seed,
                mode          = mode,
            ))



# <<



res    = Parallel(n_jobs=-1, verbose=10)(jobs)
breakpoint()

df_res = pd.DataFrame(res)
df_avg = df_res.groupby(['n_records', 'mode', 'target_model']).mean().reset_index()

# --
# Baselines

out = []
for row in tqdm(df_avg.itertuples(), total=len(df_avg)):
    _y_act  = y_acts[row.target_model]
    _scores = df[df.model == row.target_model].score.values
    
    errs = [
        np.abs(np.random.choice(_scores, size=row.n_records, replace=False).mean() - _y_act) / _y_act
        for _ in range(10000)
    ]
    out.append({
        "err_sample" : np.mean(errs),
        "err_sample10" : np.percentile(errs, 10),
        "err_sample50" : np.percentile(errs, 50),
        "err_sample90" : np.percentile(errs, 90),
    })

df_avg = pd.concat([df_avg, pd.DataFrame(out)], axis=1)

pred_null = {
    "model"  : predict_null(df, mode='model'),
    "family" : predict_null(df, mode='family'),
}

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