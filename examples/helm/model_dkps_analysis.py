import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default='math:subject=intermediate_algebra')
    parser.add_argument('--score_col', type=str, default='score')
    parser.add_argument('--outdir',    type=str, default='results')
    args = parser.parse_args()
    
    args.tsv_path = Path(args.outdir) / f'{args.dataset}-{args.score_col}-res.tsv'
    args.plot_dir = Path('plots')
    
    args.plot_dir.mkdir(parents=True, exist_ok=True)
        
    return args

args   = parse_args()

df_res = pd.read_csv(args.tsv_path, sep='\t')
model_names  = df_res.target_model.unique()
n_replicates = df_res.seed.nunique()

rprint(f'{len(model_names)} models, {n_replicates} replicates')

# --
# Plot (more detailed)

# # COLORS = ['black', 'blue', 'red', 'green', 'orange']
# cnames = [c for c in df_res.columns if 'e_' in c]

# df_avg = df_res.groupby(['mode', 'n_samples']).agg({
#     'y_act' : lambda x: np.mean(x),
#     **{c: lambda x: np.mean(x) for c in cnames},
# }).reset_index()

# for i,c in enumerate(cnames):
#     _ = plt.plot(df_avg.n_samples, df_avg[c], label=c)#, c=COLORS[i])


# # # <<
# # # Add error bars to show 95% CI of mean
# # # [TODO] double check this
# # df_ci = df_res.groupby(['mode', 'n_samples']).agg({
# #     **{c: lambda x: 1.96 * np.std(x) / np.sqrt(len(x)) for c in cnames},  # 95% CI = 1.96 * SE
# # }).reset_index()

# # # Merge with averages to get the CI values
# # df_plot = pd.merge(df_avg, df_ci, on=['mode', 'n_samples'], suffixes=('', '_ci'))

# # # Plot with error bars
# # plt.figure(figsize=(10, 6))
# # for i, c in enumerate(cnames):
# #     plt.errorbar(
# #         df_plot.n_samples, 
# #         df_plot[c], 
# #         yerr=df_plot[f"{c}_ci"],
# #         label=c, 
# #         # c=COLORS[i],
# #         capsize=4,
# #         marker='o',
# #         markersize=5,
# #         linewidth=2,
# #         elinewidth=1,
# #     )
# # # >> 

# _ = plt.legend()
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.xscale('log')
# _ = plt.ylabel(f'error (mean over {n_replicates} runs x {len(model_names)} models)')
# _ = plt.xlabel('n_samples')
# _ = plt.title(f'{args.dataset} - {args.score_col}')
# _ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err.png')
# _ = plt.close()


# --
# Plot (Simple)

_cols = [
    {
        "colname" : "e_null",
        "label"   : "Population Mean",
        "color"   : "black",
    },
    {
        "colname" : "e_sample",
        "label"   : "Sample Mean",
        "color"   : "green",
    },
    {
        "colname" : "e_lr_dkps8",
        "label"   : "DKPS(d=8)",
        "color"   : "red",
    }
]

df_avg = df_res.groupby(['mode', 'n_samples']).agg({
    'y_act' : lambda x: np.mean(x),
    **{c['colname']: lambda x: np.mean(x) for c in _cols},
}).reset_index()

for c in _cols:
    _ = plt.plot(df_avg.n_samples, df_avg[c['colname']], label=c['label'], c=c['color'], lw=3)

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel('$MAE(\hat{y}, y)$')
_ = plt.xlabel('Number of queries (m)')
_ = plt.title(f'{args.dataset}')

_ = plt.tight_layout()
_ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err-simple.png')
_ = plt.close()