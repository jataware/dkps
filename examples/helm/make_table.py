"""
Generate summary tables from DKPS results.

Shows performance at both dataset split level and full dataset level.
Uses DataFrame concatenation for computing dataset-level metrics.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rich import print as rprint

from utils import make_experiment_path

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser(description='Generate summary tables from DKPS results')
    parser.add_argument('--results_dir',    type=str, default='results')
    parser.add_argument('--tables_dir',     type=str, default='tables', help='Output directory for tables')
    parser.add_argument('--embed_provider', type=str, default='google', help='Embedding provider (e.g., google, jina, local)')
    parser.add_argument('--embed_model',    type=str, default=None, help='Embedding model (e.g., onehot)')
    parser.add_argument('--n_models',       type=str, default='ALL', choices=['20', '50', 'ALL'], help='Number of models used in DKPS')
    parser.add_argument('--n_samples',      type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64], help='Sample sizes to include in table')
    return parser.parse_args()

args = parse_args()

rprint('[yellow]Assumption - all metrics are bounded between 0 and 1[/yellow]')
rprint('[yellow]Assumption - metrics are averages of per-sample metrics[/yellow]')

pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# --
# Configuration

RESULTS_DIR = Path(args.results_dir)
TABLES_DIR = Path(args.tables_dir)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
RUNNER = 'dkps'
N_SAMPLES_TO_SHOW = args.n_samples
DKPS_COL = f'p_lr_dkps__n_components_cmds=8__n_models={args.n_models}'

# Use make_experiment_path to get the embed directory prefix
# (passing dummy values for dataset/score_col since we just need the embed part)
_exp_path = make_experiment_path(args.embed_provider, args.embed_model, 'dummy', 'dummy')
EMBED_DIR = _exp_path.parts[0]  # e.g., 'embed-google' or 'embed-local-onehot'

# --
# Load results

tsv_paths = list(RESULTS_DIR.glob(f'{EMBED_DIR}/**/{RUNNER}/results-202603.tsv'))
rprint(f'[green]Found {len(tsv_paths)} result files for {EMBED_DIR}[/green]')


def parse_path(path: Path) -> dict:
    """Parse result path to extract experiment metadata.

    Path format: results/{embed_provider}/{dataset}/{score_col}/{n_replicates}/{runner}/results.tsv
    """
    parts = path.parts
    runner_idx = parts.index(RUNNER)

    return {
        'embed_provider': parts[runner_idx - 4],
        'dataset_split': parts[runner_idx - 3],
        'score_col': parts[runner_idx - 2],
        'n_replicates': int(parts[runner_idx - 1]),
    }


dfs = []
for tsv_path in tqdm(tsv_paths, desc='Loading results'):
    df_tmp = pd.read_csv(tsv_path, sep='\t')
    meta   = parse_path(tsv_path)
    for k, v in meta.items():
        df_tmp[k] = v
    
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

# Compute max samples per dataset split (used for interpolation)
df['n_dataset_split'] = df.groupby('dataset_split').n_samples.transform('max')

# Parse dataset and split from dataset_split
df['dataset'] = df.dataset_split.apply(lambda x: x.split('-')[0])
df['split']   = df.dataset_split.apply(lambda x: '-'.join(x.split('-')[1:]) if '-' in x else '')

print(df[['dataset_split', 'n_dataset_split']].drop_duplicates().sort_values('dataset_split'))
# Some of the math datasets only have ~ 32 samples

# --
# Compute errors

pred_cols = [c for c in df.columns if c.startswith('p_')]
for p_col in pred_cols:
    e_col = f'e_{p_col[2:]}'
    df[e_col] = np.abs(df[p_col] - df.y_act)

# --
# Compute interpolation prediction

print(df.columns)

p_lr_dkps = df[DKPS_COL]
df['p_interp'] = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * p_lr_dkps) / df.n_dataset_split
df['e_interp'] = np.abs(df.p_interp - df.y_act)

for alpha in np.linspace(0, 1, 11):
    df[f'p_interp_alpha={alpha:.1f}'] = (alpha * df.p_sample + (1 - alpha) * p_lr_dkps)
    df[f'e_interp_alpha={alpha:.1f}'] = np.abs(df[f'p_interp_alpha={alpha:.1f}'] - df.y_act)

for _n_cmp in [1, 2, 4, 8, 16, 32]:
    c = f'p_lr_dkps__n_components_cmds={_n_cmp}__n_models={args.n_models}'
    df[f'p_interp_n_components_cmds={_n_cmp}'] = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * df[c]) / df.n_dataset_split
    df[f'e_interp_n_components_cmds={_n_cmp}'] = np.abs(df[c] - df.y_act)

df['p_interp_knn1'] = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * df['p_knn1_dkps__n_components_cmds=8__n_models=ALL']) / df.n_dataset_split
df['e_interp_knn1'] = np.abs(df['p_interp_knn1'] - df.y_act)

df['p_interp_knn9'] = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * df['p_knn9_dkps__n_components_cmds=8__n_models=ALL']) / df.n_dataset_split
df['e_interp_knn9'] = np.abs(df['p_interp_knn9'] - df.y_act)


# Alias for convenience
df['p_lr_dkps'] = df[DKPS_COL]
df['e_lr_dkps'] = np.abs(df['p_lr_dkps'] - df.y_act)


# ==============================================================================
# Table 1: Per dataset split
# ==============================================================================

rprint('\n[bold cyan]Table 1: Performance by dataset split[/bold cyan]')

df_sub = df[df.n_samples.isin(N_SAMPLES_TO_SHOW)]

tab_split = df_sub.groupby(['dataset', 'split', 'n_samples']).agg({
    'e_null': 'mean',
    'e_sample': 'mean',
    'e_lr_dkps': 'mean',
    'e_interp': 'mean',
    # "e_interp_alpha=0.0": "mean",
    # "e_interp_alpha=0.1": "mean",
    # "e_interp_alpha=0.2": "mean",
    # "e_interp_alpha=0.3": "mean",
    # "e_interp_alpha=0.4": "mean",
    # "e_interp_alpha=0.5": "mean",
    # "e_interp_alpha=0.6": "mean",
    # "e_interp_alpha=0.7": "mean",
    # "e_interp_alpha=0.8": "mean",
    # "e_interp_alpha=0.9": "mean",
    # "e_interp_alpha=1.0": "mean",
    
    'e_lr_dkps__n_components_cmds=1__n_models=ALL': 'mean',
    'e_lr_dkps__n_components_cmds=2__n_models=ALL': 'mean',
    'e_lr_dkps__n_components_cmds=4__n_models=ALL': 'mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL': 'mean',
    'e_lr_dkps__n_components_cmds=16__n_models=ALL': 'mean',
    'e_lr_dkps__n_components_cmds=32__n_models=ALL': 'mean',
    'e_interp_n_components_cmds=1': 'mean',
    'e_interp_n_components_cmds=2': 'mean',
    'e_interp_n_components_cmds=4': 'mean',
    'e_interp_n_components_cmds=8': 'mean',
    'e_interp_n_components_cmds=16': 'mean',
    'e_interp_n_components_cmds=32': 'mean',
    
    # 'e_knn1_dkps__n_components_cmds=8__n_models=ALL' : 'mean',
    # 'e_knn9_dkps__n_components_cmds=8__n_models=ALL' : 'mean',
    
    # 'e_interp_knn1': 'mean',
    # 'e_interp_knn9': 'mean',
}).rename(columns={
    'e_null': 'Population Mean',
    'e_sample': 'Sample Mean',
    'e_lr_dkps': 'DKPS (LR ; n_comp=8 - default)',
    'e_interp': 'Interp (LR ; m/M ; n_comp=8 - default)',
    # "e_interp_alpha=0.0": "Interp (alpha=0.0)",
    # "e_interp_alpha=0.1": "Interp (alpha=0.1)",
    # "e_interp_alpha=0.2": "Interp (alpha=0.2)",
    # "e_interp_alpha=0.3": "Interp (alpha=0.3)",
    # "e_interp_alpha=0.4": "Interp (alpha=0.4)",
    # "e_interp_alpha=0.5": "Interp (alpha=0.5)",
    # "e_interp_alpha=0.6": "Interp (alpha=0.6)",
    # "e_interp_alpha=0.7": "Interp (alpha=0.7)",
    # "e_interp_alpha=0.8": "Interp (alpha=0.8)",
    # "e_interp_alpha=0.9": "Interp (alpha=0.9)",
    # "e_interp_alpha=1.0": "Interp (alpha=1.0)",
    
    'e_lr_dkps__n_components_cmds=1__n_models=ALL': 'DKPS (n_comp=1)',
    'e_lr_dkps__n_components_cmds=2__n_models=ALL': 'DKPS (n_comp=2)',
    'e_lr_dkps__n_components_cmds=4__n_models=ALL': 'DKPS (n_comp=4)',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL': 'DKPS (n_comp=8)',
    'e_lr_dkps__n_components_cmds=16__n_models=ALL': 'DKPS (n_comp=16)',
    'e_lr_dkps__n_components_cmds=32__n_models=ALL': 'DKPS (n_comp=32)',
    'e_interp_n_components_cmds=1': 'Interp (m/M ; n_comp=1)',
    'e_interp_n_components_cmds=2': 'Interp (m/M ; n_comp=2)',
    'e_interp_n_components_cmds=4': 'Interp (m/M ; n_comp=4)',
    'e_interp_n_components_cmds=8': 'Interp (m/M ; n_comp=8)',
    'e_interp_n_components_cmds=16': 'Interp (m/M ; n_comp=16)',
    'e_interp_n_components_cmds=32': 'Interp (m/M ; n_comp=32)',
    
    # 'e_knn1_dkps__n_components_cmds=8__n_models=ALL' : "DKPS (KNN-1)",
    # 'e_knn9_dkps__n_components_cmds=8__n_models=ALL' : "DKPS (KNN-sqrt(n_models))",
    
    # 'e_interp_knn1': "Interp (m/M ; KNN-1)",
    # 'e_interp_knn9': "Interp (m/M ; KNN-sqrt(n_models))",
}).reset_index()

print(tab_split)
outpath_split = TABLES_DIR / f'table-v3-{EMBED_DIR}-n_models={args.n_models}-by_dataset_split.tsv'
tab_split.to_csv(outpath_split, sep='\t', index=False)
rprint(f'[green]Saved to {outpath_split}[/green]')

raise Exception('Stop here')

# ==============================================================================
# Table 2: Per dataset (aggregated across splits)
# ==============================================================================

raise Exception('Skipping table 2 for all datasets')

rprint('\n[bold cyan]Table 2: Performance by dataset[/bold cyan]')

# Vectorized approach: compute weighted sums, then aggregate
df_sub = df[df.n_samples.isin(N_SAMPLES_TO_SHOW)].copy()

# Weight is the split size (n_dataset_split)
weight_col = 'n_dataset_split'
pred_cols = ['y_act', 'p_null', 'p_sample', 'p_lr_dkps', 'p_interp']

# Compute weighted values
for col in pred_cols:
    df_sub[f'{col}_w'] = df_sub[col] * df_sub[weight_col]

# Aggregate: sum of weighted values and sum of weights per (dataset, n_samples, seed, target_model)
group_cols = ['dataset', 'n_samples', 'seed', 'target_model']
agg_dict = {f'{col}_w': 'sum' for col in pred_cols}
agg_dict[weight_col] = 'sum'

df_agg = df_sub.groupby(group_cols).agg(agg_dict).reset_index()

# Compute weighted averages
for col in pred_cols:
    df_agg[col] = df_agg[f'{col}_w'] / df_agg[weight_col]

# Compute interp2: interpolation of the weighted averages (not weighted average of interpolations)
# n_dataset = total instances across all splits (sum of weights)
# n_dataset = df_agg[weight_col]
# df_agg['p_interp2'] = (df_agg['n_samples'] * df_agg['p_sample'] + (n_dataset - df_agg['n_samples']) * df_agg['p_lr_dkps']) / n_dataset

# Compute errors at dataset level
df_agg['e_null']    = np.abs(df_agg['p_null'] - df_agg['y_act'])
df_agg['e_sample']  = np.abs(df_agg['p_sample'] - df_agg['y_act'])
df_agg['e_lr_dkps'] = np.abs(df_agg['p_lr_dkps'] - df_agg['y_act'])
df_agg['e_interp']  = np.abs(df_agg['p_interp'] - df_agg['y_act'])
# df_agg['e_interp2'] = np.abs(df_agg['p_interp2'] - df_agg['y_act'])

# df_agg['dkps_gain']    = df_agg['e_sample'] - df_agg['e_lr_dkps']
# df_agg['interp_gain']  = df_agg['e_sample'] - df_agg['e_interp']
# df_agg['interp2_gain'] = df_agg['e_sample'] - df_agg['e_interp2']

# Final aggregation: mean across seeds and models
tab_dataset = df_agg.groupby(['dataset', 'n_samples']).agg({
    'y_act': 'mean',
    'e_null': 'mean',
    'e_sample': 'mean',
    'e_lr_dkps': 'mean',
    'e_interp': 'mean',
    # 'e_interp2': 'mean',
    # 'dkps_gain': 'mean',
    # 'interp_gain': 'mean',
    # 'interp2_gain': 'mean',
}).reset_index()

# Exclude n_samples=64 for math (not enough samples in some splits)
tab_dataset = tab_dataset[~((tab_dataset.dataset == 'math') & (tab_dataset.n_samples == 64))]

print(tab_dataset)
outpath_dataset = TABLES_DIR / f'table-v2-{EMBED_DIR}-n_models={args.n_models}-by_dataset.tsv'
tab_dataset.to_csv(outpath_dataset, sep='\t', index=False)
rprint(f'[green]Saved to {outpath_dataset}[/green]')

print('\n[Transposed view]')
print(tab_dataset.set_index(['dataset', 'n_samples']).T)
