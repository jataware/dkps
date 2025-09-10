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
# Plot (Simple)

if args.dataset == 'med_qa':
    df_res = df_res[df_res.n_samples > 2]


# <<
max_samples = df_res.n_samples.max()
df_res['p_interp'] = (df_res.n_samples * df_res.p_sample + (max_samples - df_res.n_samples) * df_res.p_lr_dkps8) / max_samples
df_res['e_interp'] = np.abs(df_res.p_interp - df_res.y_act)
# >>

# <<
z = df_res.groupby(['n_samples', 'target_model']).apply(lambda x: (x.e_sample - x.e_interp).mean()).reset_index(name='gain')

for target_model in z.target_model.unique():
    sub = z[z.target_model == target_model]
    _ = plt.scatter(sub.n_samples * np.random.uniform(0.9, 1.1), sub.gain, label=target_model, alpha=0.05, c='black')

_ = plt.axhline(0, c='black')
# _ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel('err_sample - err_interp')
_ = plt.xlabel('Number of queries (m)')
_ = plt.title(f'{args.dataset} - Gain by model')
_ = plt.tight_layout()
_ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err-bymodel.png')
_ = plt.close()
# >>


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
    },
    {
        "colname" : "e_interp",
        "label"   : "interp(e_sample+e_lr_dkps8)",
        "color"   : "blue",
    },
]
    
df_avg = df_res.groupby(['mode', 'n_samples']).agg({
    'y_act' : lambda x: np.mean(x),
    **{c['colname']: lambda x: np.mean(x) for c in _cols},
}).reset_index()


for c in _cols:
    _ = plt.plot(df_avg.n_samples, df_avg[c['colname']], label=c['label'], c=c['color'], lw=3, marker='o')

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel('$MAE(\hat{y}, y)$')
_ = plt.xlabel('Number of queries (m)')
_ = plt.title(f'{args.dataset}')

_ = plt.tight_layout()
_ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err-simple.png')
_ = plt.close()