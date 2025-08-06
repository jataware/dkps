#!/usr/bin/env python
"""
    joint.model_dkps
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint
from tqdm import trange
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from utils import dkps_df
from dkps.embed import embed_api

# --
# Helpers

def model2family(model):
    return model.split('_')[0]


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


def _rel_err(act, pred):
    return np.abs(pred - act) / act

def _abs_err(act, pred):
    return np.abs(pred - act)

err_fns = {
    "abs" : _abs_err,
    "rel" : _rel_err,
}

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, default='math:subject=algebra')
    parser.add_argument('--score_col',      type=str, default='score')
    parser.add_argument('--embed_provider', type=str, default='jina')
    parser.add_argument('--embed_model',    type=str, default=None)
    parser.add_argument('--err_fn',         type=str, default='abs')
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
_instance_ids = df.groupby('model').instance_id.apply(list)
assert all([_instance_ids.iloc[0] == _instance_ids.iloc[i] for i in range(len(_instance_ids))]), 'instance_ids are not the same for each model'

# --
# Get embeddings

df['embedding'] = list(embed_api(
    provider   = args.embed_provider, 
    input_strs = [str(xx) for xx in df.response.values],
    model      = args.embed_model
))

# --
# Run

model_names  = df.model.unique()
instance_ids = df.instance_id.unique()
y_acts       = df.groupby('model').score.mean().to_dict()

modes     = ['model', 'family']
pred_null = {mode: predict_null(df, mode=mode) for mode in modes}

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

        y_train = np.array([y_acts[m] for m in train_models])
        y_test  = y_acts[target_model]

        # average score over the `n_samples` evaluated
        p_sample = df_test.score.mean()

        # knn on scores
        S_train = S_all[np.isin(model_names, train_models)]
        S_test  = S_all[model_names == target_model]
        sknn    = KNeighborsRegressor(n_neighbors=3).fit(S_train, y_train)
        p_3nn_score  = float(sknn.predict(S_test)[0])
        
        # lr on DKPS embeddings of varying dimension
        p_lr_dkps = {}
        for n_components_cmds in [2, 4, 8, 16]:
            P = dkps_df(
                pd.concat([df_train, df_test]).reset_index(drop=True),
                n_components_cmds=n_components_cmds,
            )
            X_train = np.vstack([P[m] for m in train_models])
            X_test  = np.vstack([P[target_model]])

            # linear regression on DKPS embeddings        
            lr = LinearRegression().fit(X_train, y_train)
            p_lr_dkps[f'p_lr_dkps{n_components_cmds}'] = float(lr.predict(X_test)[0])

        out.append({
            "seed"         : seed,
            "n_samples"    : n_samples,
            "mode"         : mode,
            "target_model" : target_model,
            
            "y_act"        : y_test,
            "p_null"       : pred_null[mode][target_model],
            "p_sample"     : p_sample,
            "p_3nn_score"  : p_3nn_score,
            
            **p_lr_dkps,
        })
    
    return out

n_replicates = 32

outpath = f'results/{args.dataset}-{args.score_col}-res.tsv'
# if os.path.exists(outpath):
#     df_res = pd.read_csv(outpath, sep='\t')
# else:
jobs = []
for iter in trange(n_replicates):
    rng = np.random.default_rng(iter)
    for n_samples in [1, 2, 4, 8, 16, 32, 64, 128]:
        if n_samples > len(instance_ids):
            continue
        
        instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
        df_sample           = df[df.instance_id.isin(instance_ids_sample)]
        jobs.append(delayed(run_one)(df_sample=df_sample, n_samples=n_samples, mode='family', seed=iter))

res    = sum(Parallel(n_jobs=-1, verbose=10)(jobs), [])
df_res = pd.DataFrame(res)

# compute errors - abs(pred - act) / act
for c in df_res.columns:
    if 'p_' in c:
        df_res[c.replace('p_', 'e_')] = err_fns[args.err_fn](df_res.y_act, df_res[c])

df_res.to_csv(outpath, sep='\t', index=False)

# --
# Plot

# COLORS = ['black', 'blue', 'red', 'green', 'orange']
cnames = [c for c in df_res.columns if 'e_' in c]

df_avg = df_res.groupby(['mode', 'n_samples']).agg({
    'y_act' : lambda x: np.mean(x),
    **{c: lambda x: np.mean(x) for c in cnames},
}).reset_index()

for i,c in enumerate(cnames):
    _ = plt.plot(df_avg.n_samples, df_avg[c], label=c)#, c=COLORS[i])


# # <<
# # Add error bars to show 95% CI of mean
# # [TODO] double check this
# df_ci = df_res.groupby(['mode', 'n_samples']).agg({
#     **{c: lambda x: 1.96 * np.std(x) / np.sqrt(len(x)) for c in cnames},  # 95% CI = 1.96 * SE
# }).reset_index()

# # Merge with averages to get the CI values
# df_plot = pd.merge(df_avg, df_ci, on=['mode', 'n_samples'], suffixes=('', '_ci'))

# # Plot with error bars
# plt.figure(figsize=(10, 6))
# for i, c in enumerate(cnames):
#     plt.errorbar(
#         df_plot.n_samples, 
#         df_plot[c], 
#         yerr=df_plot[f"{c}_ci"],
#         label=c, 
#         # c=COLORS[i],
#         capsize=4,
#         marker='o',
#         markersize=5,
#         linewidth=2,
#         elinewidth=1,
#     )
# # >> 

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel(f'error (mean over {n_replicates} runs x {len(model_names)} models)')
_ = plt.xlabel('n_samples')
_ = plt.title(f'{args.dataset} - {args.score_col}')
_ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err-big.png')
_ = plt.close()

# --
# # plot gain over null, per model
# fine, but I don't really care

# df_per_model = df_res.groupby(['target_model', 'mode', 'n_samples']).agg({
#     'y_act'       : 'mean', # noop - they're all the same
#     'e_null'      : 'mean',
#     'e_sample'    : 'mean',
#     'e_dkps2'     : 'mean',
#     'e_knn_dkps2' : 'mean',
#     'e_sknn'      : 'mean',
# }).reset_index()

# df_per_model['dkps2_gain']  = df_per_model.e_dkps2 - df_per_model.e_null
# df_per_model['sample_gain'] = df_per_model.e_sample - df_per_model.e_null
# df_per_model['knn_gain']    = df_per_model.e_knn_dkps2 - df_per_model.e_null
# df_per_model['sknn_gain']   = df_per_model.e_sknn - df_per_model.e_null

# for model in model_names:
#     sub = df_per_model[df_per_model.target_model == model]
#     _ = plt.plot(sub.n_samples, sub.dkps2_gain, c='red', alpha=0.1)
#     _ = plt.plot(sub.n_samples, sub.sample_gain, c='blue', alpha=0.1)
#     _ = plt.plot(sub.n_samples, sub.knn_gain, c='green', alpha=0.1)
#     _ = plt.plot(sub.n_samples, sub.sknn_gain, c='orange', alpha=0.1)

# _ = plt.plot(df_per_model.groupby('n_samples').dkps2_gain.median(), label='dkps2', c='red', linewidth=5)
# _ = plt.plot(df_per_model.groupby('n_samples').sample_gain.median(), label='sample', c='blue', linewidth=5)
# _ = plt.plot(df_per_model.groupby('n_samples').knn_gain.median(), label='knn', c='green', linewidth=5)
# _ = plt.plot(df_per_model.groupby('n_samples').sknn_gain.median(), label='sknn', c='orange', linewidth=5)

# _ = plt.legend()
# _ = plt.ylim(-0.2, 0.2)
# _ = plt.axhline(0, c='black')
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.xscale('log')
# _ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err-by-model.png')
# _ = plt.close()


# # if you're trying to determine whether the new model is in the top 10% of models, how well do you do vs sampling?
# fine, can revisit later

# t = np.percentile(list(y_acts.values()), 90)

# tmp = []
# for n_samples in df_res.n_samples.unique():
#     sub = df_res[df_res.n_samples == n_samples]
#     tmp.append({
#         'n_samples'     : n_samples,
#         'auc_null'      : metrics.roc_auc_score(sub.y_act > t, sub.p_null),
#         'auc_sample'    : metrics.roc_auc_score(sub.y_act > t, sub.p_sample),
#         'auc_dkps2'     : metrics.roc_auc_score(sub.y_act > t, sub.p_lr_dkps2),
#         'auc_knn_dkps2' : metrics.roc_auc_score(sub.y_act > t, sub.p_3nn_dkps2),
#         'auc_sknn'      : metrics.roc_auc_score(sub.y_act > t, sub.p_3nn_score),
#     })

# df_f1 = pd.DataFrame(tmp)

# _ = plt.plot(df_f1.n_samples, df_f1.auc_null, label='null', c='black')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_sample, label='sample', c='blue')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_dkps2, label='dkps2', c='red')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_knn_dkps2, label='knn', c='green')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_sknn, label='sknn', c='orange')

# _ = plt.legend()
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.xscale('log')
# _ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-auc.png')
# _ = plt.close()


# if you're trying to determine which of two models is better
# only makes sense to do within a family
# [TODO] this makes sense but need to double check

# from tqdm import tqdm

# tmp = []
# for seed in df_res.seed.unique():
#     for n_samples in df_res.n_samples.unique():
#         sub = df_res[(df_res.seed == seed) & (df_res.n_samples == n_samples)]
        
#         o_act       = sub.y_act.values[None,] > sub.y_act.values[:,None]
#         o_sample    = sub.p_sample.values[None,] > sub.p_sample.values[:,None]
#         o_dkps2     = sub.p_lr_dkps2.values[None,] > sub.p_lr_dkps2.values[:,None]
#         o_knn_dkps2 = sub.p_3nn_dkps2.values[None,] > sub.p_3nn_dkps2.values[:,None]
#         o_sknn      = sub.p_3nn_score.values[None,] > sub.p_3nn_score.values[:,None]
        
#         family = np.array([model2family(m) for m in sub.target_model.values])
#         mask   = family[None,] == family[:,None]
        
#         c_sample    = (o_act == o_sample)[mask].sum() / mask.sum()
#         c_dkps2     = (o_act == o_dkps2)[mask].sum() / mask.sum()
#         c_knn_dkps2 = (o_act == o_knn_dkps2)[mask].sum() / mask.sum()
#         c_sknn      = (o_act == o_sknn)[mask].sum() / mask.sum()
        
#         tmp.append({
#             'seed'        : seed,
#             'n_samples'   : n_samples,
#             'c_sample'    : c_sample,
#             'c_dkps2'     : c_dkps2,
#             'c_knn_dkps2' : c_knn_dkps2,
#             'c_sknn'      : c_sknn,
#         })


# tmp = pd.DataFrame(tmp)

# tmp = tmp.groupby('n_samples').agg({
#     'c_sample'    : lambda x: np.mean(x),
#     'c_dkps2'     : lambda x: np.mean(x),
#     'c_knn_dkps2' : lambda x: np.mean(x),
#     'c_sknn'      : lambda x: np.mean(x),
# }).reset_index()

# _ = plt.plot(tmp.n_samples, tmp.c_sample, label='sample', c='blue')
# _ = plt.plot(tmp.n_samples, tmp.c_dkps2, label='dkps2', c='red')
# _ = plt.plot(tmp.n_samples, tmp.c_knn_dkps2, label='knn', c='green')
# _ = plt.plot(tmp.n_samples, tmp.c_sknn, label='sknn', c='orange')
# _ = plt.legend()
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.xscale('log')
# _ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-win.png')
# _ = plt.close()