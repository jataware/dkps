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




n_samples_values = sorted(df_res.n_samples.unique())
n_plots = len(n_samples_values)
n_cols = min(4, n_plots)
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
if n_plots == 1:
    axes = np.array([axes])

axes = axes.flatten()

for idx, n_samples in enumerate(n_samples_values):
    sub = df_res[df_res.n_samples == n_samples]
    axes[idx].scatter(
        sub['e_lr_dkps8__n_components_cmds=8__n_models=ALL--er'],
        sub['e_lr_dkps8__n_components_cmds=8__n_models=ALL'],
        s=2,
        alpha=0.25,
    )
    
    median_er = np.percentile(sub['e_lr_dkps8__n_components_cmds=8__n_models=ALL--er'], 50)
    axes[idx].axvline(median_er, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    median_er = np.percentile(sub['e_lr_dkps8__n_components_cmds=8__n_models=ALL--er'], 75)
    axes[idx].axvline(median_er, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    axes[idx].set_title(f'n_samples={n_samples}')
    axes[idx].set_xlabel('Training Error')
    axes[idx].set_ylabel('Test Error')
    axes[idx].grid(alpha=0.25)

# Hide any unused subplots
for idx in range(n_plots, len(axes)):
    axes[idx].axis('off')

_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-error-grid.png')
_ = plt.close()

breakpoint()

sub = df_res[df_res.n_samples == 8]
_er = sub['e_lr_dkps8__n_components_cmds=8__n_models=20--lo']
_p  = sub['e_lr_dkps8__n_components_cmds=8__n_models=20']

z10 = _p.groupby(_er.round(2)).apply(lambda x: np.percentile(x, 10))
z25 = _p.groupby(_er.round(2)).apply(lambda x: np.percentile(x, 25))
z50 = _p.groupby(_er.round(2)).apply(lambda x: np.percentile(x, 50))
z75 = _p.groupby(_er.round(2)).apply(lambda x: np.percentile(x, 75))
z90 = _p.groupby(_er.round(2)).apply(lambda x: np.percentile(x, 90))
_ = plt.scatter(z10.index, z10.values, label='10%', c='red')
_ = plt.scatter(z25.index, z25.values, label='25%', c='orange')
_ = plt.scatter(z50.index, z50.values, label='50%', c='green')
_ = plt.scatter(z75.index, z75.values, label='75%', c='blue')
_ = plt.scatter(z90.index, z90.values, label='90%', c='purple')
_ = plt.legend()
_ = plt.show()

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
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-bymodel-clip.png')
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
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-byreplicate-clip.png')
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
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-simple-0-clip.png')
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
_ = plt.savefig(args.plot_dir / f'{args.score_col}-err-simple-1-clip.png')
_ = plt.close()