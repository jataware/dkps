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
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from dkps.dkps import DataKernelPerspectiveSpace
from dkps.embed import embed_google

# --
# Helpers

def model2family(model):
    return model.split('_')[0]


def rel_err(act, pred):
    return np.abs(pred - act)#  / act

    
def dkps_df(df, data='embedding', **kwargs):
    model_names  = df.model.unique()
    instance_ids = df.instance_id.unique()
    
    embedding_dict = {}
    for model_name in model_names:
        sub = df[df.model == model_name]
        assert (sub.instance_id.values == instance_ids).all(), f'instance_ids are not the same for model {model_name}'
        if data == 'embedding':
            embedding_dict[model_name] = np.row_stack(sub.embedding.values)
        elif data == 'score':
            embedding_dict[model_name] = sub.score.values[None]
        else:
            raise ValueError(f'data must be either "embedding" or "score", got {data}')
    
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

dataset = 'med_dialog,subset=icliniq'
METRIC  = 'score'

# --
# IO

rprint('[blue]loading data ...[/blue]')
df = pd.read_csv('med_dialog.tsv', sep='\t')

df = df[df.dataset == dataset]
df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

df['score'] = df[METRIC]

# --
# QC

print(f'{len(df.response.unique())} / {df.shape[0]} responses are unique')
_instance_ids = df.groupby('model').instance_id.apply(list)
assert all([_instance_ids.iloc[0] == _instance_ids.iloc[i] for i in range(len(_instance_ids))]), 'instance_ids are not the same for each model'

# --
# Get embeddings

df['embedding'] = list(embed_google([str(xx) for xx in df.response.values]))

# DROP_OUTLIERS = False
# if DROP_OUTLIERS:
#     print('starting with', len(set(df.model.values)), 'models')
#     model_scores = df.groupby('model').score.mean()
#     bad_models   = model_scores[model_scores <= 0.15].index
#     print('dropping', len(bad_models), 'models')
#     df           = df[~df.model.isin(bad_models)].reset_index(drop=True)
#     print('ending with', len(set(df.model.values)), 'models')

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
    
    S_all = df_sample.pivot(index='model', columns='instance_id', values='score').values
    
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
        P       = dkps_df(pd.concat([df_train, df_test]).reset_index(drop=True), data='embedding', n_components_cmds=2)
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
        
        # knn on scores
        S_train = S_all[np.in1d(model_names, train_models)]
        S_test  = S_all[model_names == target_model]
        sknn    = KNeighborsRegressor(n_neighbors=3).fit(S_train, y_train)
        p_sknn  = float(sknn.predict(S_test)[0])
        
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
            "p_sknn"       : p_sknn,
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

breakpoint()

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
    'e_sknn'      : 'mean',
}).reset_index()


df_avg = df_res.groupby(['mode', 'n_samples']).agg({
    'y_act'       : lambda x: np.median(x),
    'e_null'      : lambda x: np.median(x),
    'e_sample'    : lambda x: np.median(x),
    'e_dkps2'     : lambda x: np.median(x),
    'e_knn_dkps2' : lambda x: np.median(x),
    'e_sknn'      : lambda x: np.median(x),
}).reset_index()


# --
# Plot

# plot median error over models
_ = plt.plot(df_avg.n_samples, df_avg.e_null, label='null', c='black')
_ = plt.plot(df_avg.n_samples, df_avg.e_sample, label='sample', c='blue')
_ = plt.plot(df_avg.n_samples, df_avg.e_dkps2, label='dkps2', c='red')
_ = plt.plot(df_avg.n_samples, df_avg.e_knn_dkps2, label='knn', c='green')
_ = plt.plot(df_avg.n_samples, df_avg.e_sknn, label='sknn', c='orange')

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.savefig(f'plots/{METRIC}-err.png')
_ = plt.close()



# plot gain over null, per model
df_per_model['dkps2_gain']  = df_per_model.e_dkps2 - df_per_model.e_null
df_per_model['sample_gain'] = df_per_model.e_sample - df_per_model.e_null
df_per_model['knn_gain']    = df_per_model.e_knn_dkps2 - df_per_model.e_null
df_per_model['sknn_gain']   = df_per_model.e_sknn - df_per_model.e_null

for model in model_names:
    sub = df_per_model[df_per_model.target_model == model]
    _ = plt.plot(sub.n_samples, sub.dkps2_gain, c='red', alpha=0.1)
    _ = plt.plot(sub.n_samples, sub.sample_gain, c='blue', alpha=0.1)
    _ = plt.plot(sub.n_samples, sub.knn_gain, c='green', alpha=0.1)
    _ = plt.plot(sub.n_samples, sub.sknn_gain, c='orange', alpha=0.1)

_ = plt.plot(df_per_model.groupby('n_samples').dkps2_gain.median(), label='dkps2', c='red', linewidth=5)
_ = plt.plot(df_per_model.groupby('n_samples').sample_gain.median(), label='sample', c='blue', linewidth=5)
_ = plt.plot(df_per_model.groupby('n_samples').knn_gain.median(), label='knn', c='green', linewidth=5)
_ = plt.plot(df_per_model.groupby('n_samples').sknn_gain.median(), label='sknn', c='orange', linewidth=5)

_ = plt.legend()
_ = plt.ylim(-0.2, 0.2)
_ = plt.axhline(0, c='black')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.savefig(f'plots/{METRIC}-err_by_model.png')
_ = plt.close()

breakpoint()


# if you're trying to determine whether the new model is in the top 10% of models, how well do you do vs sampling?
t = np.percentile(df.groupby('model').score.mean().values, 90)

tmp = []
for n_samples in df_res.n_samples.unique():
    sub = df_res[df_res.n_samples == n_samples]
    tmp.append({
        'n_samples'     : n_samples,
        'auc_null'      : metrics.roc_auc_score(sub.y_act > t, sub.p_null),
        'auc_sample'    : metrics.roc_auc_score(sub.y_act > t, sub.p_sample),
        'auc_dkps2'     : metrics.roc_auc_score(sub.y_act > t, sub.p_dkps2),
        'auc_knn_dkps2' : metrics.roc_auc_score(sub.y_act > t, sub.p_knn_dkps2),
        'auc_sknn'      : metrics.roc_auc_score(sub.y_act > t, sub.p_sknn),
    })

df_f1 = pd.DataFrame(tmp)

_ = plt.plot(df_f1.n_samples, df_f1.auc_null, label='null', c='black')
_ = plt.plot(df_f1.n_samples, df_f1.auc_sample, label='sample', c='blue')
_ = plt.plot(df_f1.n_samples, df_f1.auc_dkps2, label='dkps2', c='red')
_ = plt.plot(df_f1.n_samples, df_f1.auc_knn_dkps2, label='knn', c='green')
_ = plt.plot(df_f1.n_samples, df_f1.auc_sknn, label='sknn', c='orange')

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
plt.show()


# if you're trying to determine which of two models is better
# only makes sense to do within a family
from tqdm import tqdm

tmp = []
for seed in df_res.seed.unique():
    for n_samples in df_res.n_samples.unique():
        sub = df_res[(df_res.seed == seed) & (df_res.n_samples == n_samples)]
        
        o_act       = sub.y_act.values[None,] > sub.y_act.values[:,None]
        o_sample    = sub.p_sample.values[None,] > sub.p_sample.values[:,None]
        o_dkps2     = sub.p_dkps2.values[None,] > sub.p_dkps2.values[:,None]
        o_knn_dkps2 = sub.p_knn_dkps2.values[None,] > sub.p_knn_dkps2.values[:,None]
        o_sknn      = sub.p_sknn.values[None,] > sub.p_sknn.values[:,None]
        
        family = np.array([model2family(m) for m in sub.target_model.values])
        mask   = family[None,] == family[:,None]
        
        c_sample    = (o_act == o_sample)[mask].sum() / mask.sum()
        c_dkps2     = (o_act == o_dkps2)[mask].sum() / mask.sum()
        c_knn_dkps2 = (o_act == o_knn_dkps2)[mask].sum() / mask.sum()
        c_sknn      = (o_act == o_sknn)[mask].sum() / mask.sum()
        
        tmp.append({
            'seed'        : seed,
            'n_samples'   : n_samples,
            'c_sample'    : c_sample,
            'c_dkps2'     : c_dkps2,
            'c_knn_dkps2' : c_knn_dkps2,
            'c_sknn'      : c_sknn,
        })


tmp = pd.DataFrame(tmp)

(tmp.c_dkps2 > tmp.c_sample).groupby(tmp.n_samples).mean()
(tmp.c_knn_dkps2 > tmp.c_sample).groupby(tmp.n_samples).mean()

tmp = tmp.groupby('n_samples').agg({
    'c_sample'    : lambda x: np.mean(x),
    'c_dkps2'     : lambda x: np.mean(x),
    'c_knn_dkps2' : lambda x: np.mean(x),
    'c_sknn'      : lambda x: np.mean(x),
}).reset_index()

_ = plt.plot(tmp.n_samples, tmp.c_sample, label='sample', c='blue')
_ = plt.plot(tmp.n_samples, tmp.c_dkps2, label='dkps2', c='red')
_ = plt.plot(tmp.n_samples, tmp.c_knn_dkps2, label='knn', c='green')
_ = plt.plot(tmp.n_samples, tmp.c_sknn, label='sknn', c='orange')
_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
plt.show()



df_res['family'] = df_res.target_model.apply(model2family)
tmp = df_res[df_res.family == 'google'].groupby(['target_model', 'n_samples']).agg({
    'y_act'       : 'mean',
    'p_null'      : 'mean',
    'p_sample'    : 'mean',
    'p_dkps2'     : 'mean',
    'p_knn_dkps2' : 'mean',
}).reset_index()

model = tmp.target_model.unique()[2]
sub = tmp[tmp.target_model == model]
_ = plt.plot(sub.n_samples, sub.p_null, label='null', c='black')
_ = plt.plot(sub.n_samples, sub.p_sample, label='sample', c='blue')
_ = plt.plot(sub.n_samples, sub.p_dkps2, label='dkps2', c='red')
_ = plt.plot(sub.n_samples, sub.p_knn_dkps2, label='knn', c='green')
_ = plt.axhline(sub.y_act.mean(), c='black')
_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')

plt.show()