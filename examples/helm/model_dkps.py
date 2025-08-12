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
    parser.add_argument('--outdir',         type=str, default='results')
    args = parser.parse_args()

    if args.embed_model == 'jina':
        assert os.environ.get('JINA_API_KEY') is not None, 'JINA_API_KEY is not set'
    elif args.embed_model == 'google':
        assert os.environ.get('GEMINI_API_KEY') is not None, 'GEMINI_API_KEY is not set'
    elif args.embed_model == 'jlai_tei':
        print('... jlai_tei requires some manual setup ... talk to @bkj ...')
    
    args.inpath = Path('data') / f'{args.dataset.split(":")[0]}.tsv'
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    return args

args = parse_args()

rprint('[blue]loading data ...[/blue]')

df = pd.read_csv(args.inpath, sep='\t')
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

def run_one(df_sample, n_samples, mode, seed, dkps_mode='is', df_full=None):
    assert dkps_mode in ['is', 'oos']
    
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
        if df_full is not None:
            df_train_full = df_full[df_full.model.isin(train_models)]

        y_train = np.array([y_acts[m] for m in train_models])
        y_test  = y_acts[target_model]

        # average score over the `n_samples` evaluated
        p_sample = df_test.score.mean()

        # knn on scores
        S_train      = S_all[np.isin(model_names, train_models)]
        S_test       = S_all[model_names == target_model]
        sknn         = KNeighborsRegressor(n_neighbors=3).fit(S_train, y_train)
        p_3nn_score  = float(sknn.predict(S_test)[0])
        
        # lr on DKPS embeddings of varying dimension
        p_lr_dkps = {}
        for n_components_cmds in [4, 8]:
            if dkps_mode == 'is':
                P = dkps_df(
                    df                = pd.concat([df_train, df_test]).reset_index(drop=True),
                    n_components_cmds = n_components_cmds,
                )
            elif dkps_mode == 'oos':
                P = dkps_df(
                    df                = pd.concat([df_train_full, df_test]).reset_index(drop=True),
                    n_components_cmds = n_components_cmds,
                    oos               = [target_model],
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

N_REPLICATES = 32

outpath = args.outdir / f'{args.dataset}-{args.score_col}-res-oos.tsv'

jobs = []
for iter in trange(N_REPLICATES):
    rng = np.random.default_rng(iter)
    for n_samples in [1, 2, 4, 8, 16, 32, 64, 128]:
        if n_samples > len(instance_ids):
            continue
        
        instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
        df_sample           = df[df.instance_id.isin(instance_ids_sample)]
        
        # jobs.append(delayed(run_one)(df_sample=df_sample, n_samples=n_samples, mode='family', seed=iter))
        jobs.append(delayed(run_one)(df_sample=df_sample, n_samples=n_samples, mode='family', seed=iter, dkps_mode='oos', df_full=df))

res    = sum(Parallel(n_jobs=-1, verbose=10)(jobs), [])
df_res = pd.DataFrame(res)

# compute errors - abs(pred - act) / act
for c in df_res.columns:
    if 'p_' in c:
        df_res[c.replace('p_', 'e_')] = err_fns[args.err_fn](df_res.y_act, df_res[c])

df_res.to_csv(outpath, sep='\t', index=False)

