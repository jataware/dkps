#!/usr/bin/env python
"""
    helm.model_dkps
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
from sklearn.metrics import r2_score

from utils import make_embedding_dict, onehot_embedding
from dkps.embed import embed_api
from dkps.dkps import DataKernelPerspectiveSpace as DKPS

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
    parser.add_argument('--dataset',        type=str,   default='math:subject=algebra')
    parser.add_argument('--score_col',      type=str,   default='score')
    parser.add_argument('--embed_provider', type=str,   default='jina')
    parser.add_argument('--embed_model',    type=str,   default=None)
    parser.add_argument('--err_fn',         type=str,   default='abs')
    parser.add_argument('--outdir',         type=str,   default='results')
    parser.add_argument('--sample',         type=float)
    parser.add_argument('--seed',           type=int,   default=123)
    parser.add_argument('--n_replicates',   type=int,   default=128)
    parser.add_argument('--n_jobs',         type=int,   default=-2)
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

if args.sample:
    rng           = np.random.default_rng(args.seed)
    uinstance_ids = df.instance_id.unique()
    keep          = rng.choice(uinstance_ids, int(len(uinstance_ids) * args.sample), replace=False)
    df            = df[df.instance_id.isin(keep)]

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

if args.embed_model == 'onehot':
    df = onehot_embedding(df, dataset=args.dataset)
else:
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
    
    embedding_dict = make_embedding_dict(df_sample)
    
    for target_model in model_names:
        
        # split data
        assert mode in ['model', 'family']
        if mode == 'model':
            train_models = np.array([m for m in model_names if m != target_model])
        elif mode == 'family':
            target_family = model2family(target_model)
            train_models  = np.array([m for m in model_names if model2family(m) != target_family])
        
        y_test  = y_acts[target_model]

        # average score over the `n_samples` evaluated
        p_sample = df_sample[df_sample.model == target_model].score.mean()
        
        # lr on DKPS embeddings of varying dimension
        res = {}
        for n_components_cmds in [8]:
            for n_models in [20, 50, len(train_models)]:
                if n_models != len(train_models):
                    _lr_suffix  = f'lr_dkps__n_components_cmds={n_components_cmds}__n_models={n_models}'
                    _knn_suffix = f'knn5_dkps__n_components_cmds={n_components_cmds}__n_models={n_models}'
                else:
                    _lr_suffix  = f'lr_dkps__n_components_cmds={n_components_cmds}__n_models=ALL'
                    _knn_suffix = f'knn5_dkps__n_components_cmds={n_components_cmds}__n_models=ALL'

                _train_models = np.random.choice(train_models, size=n_models, replace=False)
                
                # --
                # dkps w/ target model
                
                _embedding_dict = {k:embedding_dict[k] for k in (set(_train_models) | set([target_model]))}
                P = DKPS(n_components_cmds=n_components_cmds)
                P = P.fit_transform(_embedding_dict, return_dict=True)
                
                _X_train = np.vstack([P[m] for m in _train_models])
                _y_train = np.array([y_acts[m] for m in _train_models])
                _X_test  = np.vstack([P[target_model]])

                # linear regression on DKPS embeddings        
                lr = LinearRegression().fit(_X_train, _y_train)
                
                # goodness of fit metrics (for regression testing - these are "bad")
                lr_pred_train = lr.predict(_X_train)
                res['trans-er_' + _lr_suffix] = ((lr_pred_train - _y_train) ** 2).mean()
                res['trans-ss_' + _lr_suffix] = ((_y_train.mean() - _y_train) ** 2).mean()
                res['trans-r2_' + _lr_suffix] = r2_score(_y_train, lr_pred_train)
                
                res['p_' + _lr_suffix] = float(lr.predict(_X_test)[0])
                
                # knn regression on DKPS embeddings
                knn = KNeighborsRegressor(n_neighbors=5).fit(_X_train, _y_train)
                res['p_' + _knn_suffix] = float(knn.predict(_X_test)[0])
                
                del P, lr, _X_train, _y_train, lr_pred_train
        
        out.append({
            "seed"         : seed,
            "n_samples"    : n_samples,
            "mode"         : mode,
            "target_model" : target_model,
            
            "y_act"        : y_test,
            "p_null"       : pred_null[mode][target_model],
            "p_sample"     : p_sample,
            
            **res,
        })
    
    return out


outpath = args.outdir / f'{args.dataset}-{args.score_col}-res.tsv'

jobs = []
for iter in trange(args.n_replicates):
    rng = np.random.default_rng(iter)
    for n_samples in [2, 4, 8, 16, 32, 64, 128, 256, 512, len(instance_ids)]:
        if n_samples > len(instance_ids):
            continue
        
        instance_ids_sample = rng.choice(instance_ids, size=n_samples, replace=False)
        df_sample           = df[df.instance_id.isin(instance_ids_sample)]
        jobs.append(delayed(run_one)(df_sample=df_sample, n_samples=n_samples, mode='family', seed=iter))

res    = sum(Parallel(n_jobs=args.n_jobs, verbose=10)(jobs), [])
df_res = pd.DataFrame(res)

# compute errors - abs(pred - act) / act
for c in df_res.columns:
    if 'p_' in c:
        assert c.startswith('p_')
        assert 'p_' not in c[2:]
        
        df_res[c.replace('p_', 'e_')] = err_fns[args.err_fn](df_res.y_act, df_res[c])


df_res.to_csv(outpath, sep='\t', index=False)

