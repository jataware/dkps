"""
    helm.model_dkps_analysis
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint

rprint('[yellow] Assumption - all metrics are bounded between 0 and 1[/yellow]')

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default='math:subject=intermediate_algebra')
    parser.add_argument('--score_col', type=str, default='score')
    parser.add_argument('--outdir',    type=str, default='results')
    args = parser.parse_args()
    
    args.tsv_path = Path(args.outdir) / f'{args.dataset}-{args.score_col}-res.tsv'
    args.plot_dir = Path('plots') / args.dataset.replace(':', '-')
    
    args.plot_dir.mkdir(parents=True, exist_ok=True)
        
    return args

args   = parse_args()

# --
# IO

df_res = pd.read_csv(args.tsv_path, sep='\t')
model_names  = df_res.target_model.unique()
n_replicates = df_res.seed.nunique()

# <<
# Hotfix

dkps_cols = [c for c in df_res.columns if 'p_' in c]
rprint(f'[yellow]clipping DKPS columns to (0, 1) - {dkps_cols}[/yellow]')
for c in dkps_cols:
    df_res[c] = df_res[c].clip(0, 1)

for c in dkps_cols:
    df_res[c.replace('p_', 'e_')] = np.abs(df_res[c] - df_res.y_act)

# >>

# alias the run with all models
df_res['p_lr_dkps8'] = df_res['p_lr_dkps8__n_components_cmds=8__n_models=ALL']

# compute interpolation
max_samples        = df_res.n_samples.max()
df_res['p_interp'] = (df_res.n_samples * df_res.p_sample + (max_samples - df_res.n_samples) * df_res.p_lr_dkps8) / max_samples
df_res['e_interp'] = np.abs(df_res.p_interp - df_res.y_act)

if any([xx in args.dataset for xx in ['med_qa', 'legalbench']]):
    df_res = df_res[df_res.n_samples > 2]

# --
# Plot gain (average per model)

gain_model = df_res.groupby(['n_samples', 'target_model']).apply(lambda x: (x.e_sample - x.e_interp).mean()).reset_index(name='gain')

for target_model in gain_model.target_model.unique():
    sub = gain_model[gain_model.target_model == target_model]
    _ = plt.scatter(sub.n_samples * np.random.uniform(0.9, 1.1), sub.gain, label=target_model, alpha=0.05, c='black')

_ = plt.axhline(0, c='black')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel('err_sample - err_interp')
_ = plt.xlabel('Number of queries (m)')
_ = plt.title(f'{args.dataset} \n Performance gain (average per model)')
_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-bymodel.png')
_ = plt.close()

# --
# Plot gain (per replicate)

for target_model in df_res.target_model.unique():
    sub = df_res[df_res.target_model == target_model]
    _ = plt.scatter(sub.n_samples * np.random.uniform(0.9, 1.1), sub.e_sample - sub.e_interp, label=target_model, alpha=0.05, c='black', s=2)

_ = plt.axhline(0, c='black')
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel('err_sample - err_interp')
_ = plt.xlabel('Number of queries (m)')
_ = plt.title(f'{args.dataset} \n Performance gain (per replicate)')
_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-byreplicate.png')
_ = plt.close()

# --
# Plot error vs number of queries

_cols = [
    {
        "colname"   : "e_null",
        "label"     : "Population Mean",
        "c"         : "black",
        "linestyle" : "-",
        "plots"     : [0, 1],
    },
    {
        "colname"   : "e_sample",
        "label"     : "Sample Mean",
        "c"         : "green",
        "linestyle" : "-",
        "plots"     : [0, 1],
    },
    {
        "colname"   : "e_lr_dkps8__n_components_cmds=8__n_models=20",
        "label"     : "DKPS(d=8, n_models=20)",
        "c"         : "red",
        "linestyle" : ":",
        "plots"     : [0],
    },
    {
        "colname"   : "e_lr_dkps8__n_components_cmds=8__n_models=50",
        "label"     : "DKPS(d=8, n_models=50)",
        "c"         : "red",
        "linestyle" : "--",
        "plots"     : [0],
    },
    {
        "colname"   : "e_lr_dkps8__n_components_cmds=8__n_models=ALL",
        "label"     : "DKPS(d=8, n_models=ALL)",
        "c"         : "red",
        "linestyle" : "-",
        "plots"     : [0, 1],
    },
    {
        "colname"   : "e_interp",
        "label"     : "interp(e_sample+e_lr_dkps8)",
        "c"         : "blue",
        "linestyle" : "-",
        "plots"     : [1],
    },
]
    
df_avg = df_res.groupby(['mode', 'n_samples']).agg({
    'y_act' : lambda x: np.mean(x),
    **{c['colname']: lambda x: np.mean(x) for c in _cols},
}).reset_index()

# 0th version
for c in _cols:
    if 0 not in c['plots']: continue
    _ = plt.plot(df_avg.n_samples, df_avg[c['colname']], label=c['label'], c=c['c'], linestyle=c['linestyle'], lw=2)

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel('$MAE(\hat{y}, y)$')
_ = plt.xlabel('Number of queries (m)')
_ = plt.title(f'{args.dataset}')

_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-simple-0.png')
_ = plt.close()


# 1st version
for c in _cols:
    if 1 not in c['plots']: continue
    _ = plt.plot(df_avg.n_samples, df_avg[c['colname']], label=c['label'], c=c['c'], linestyle=c['linestyle'], lw=2)

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel('$MAE(\hat{y}, y)$')
_ = plt.xlabel('Number of queries (m)')
_ = plt.title(f'{args.dataset}')

_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-simple-1.png')
_ = plt.close()