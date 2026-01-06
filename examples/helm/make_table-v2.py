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
RUNNER = 'dkps'
N_SAMPLES_TO_SHOW = args.n_samples
DKPS_COL = f'p_lr_dkps__n_components_cmds=8__n_models={args.n_models}'

# Use make_experiment_path to get the embed directory prefix
# (passing dummy values for dataset/score_col since we just need the embed part)
_exp_path = make_experiment_path(args.embed_provider, args.embed_model, 'dummy', 'dummy')
EMBED_DIR = _exp_path.parts[0]  # e.g., 'embed-google' or 'embed-local-onehot'

# --
# Load results

tsv_paths = list(RESULTS_DIR.glob(f'{EMBED_DIR}/**/{RUNNER}/results-v2.tsv'))
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

p_lr_dkps = df[DKPS_COL]
df['p_interp'] = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * p_lr_dkps) / df.n_dataset_split
df['e_interp'] = np.abs(df.p_interp - df.y_act)

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
}).rename(columns={
    'e_null': 'Population Mean',
    'e_sample': 'Sample Mean',
    'e_lr_dkps': 'DKPS',
    'e_interp': 'Interp',
}).reset_index()

print(tab_split)
outpath_split = f'table-v2-{EMBED_DIR}-n_models={args.n_models}-by_dataset_split.tsv'
tab_split.to_csv(outpath_split, sep='\t', index=False)
rprint(f'[green]Saved to {outpath_split}[/green]')


# ==============================================================================
# Table 2: Per dataset (aggregated across splits)
# ==============================================================================

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
outpath_dataset = f'table-v2-{EMBED_DIR}-n_models={args.n_models}-by_dataset.tsv'
tab_dataset.to_csv(outpath_dataset, sep='\t', index=False)
rprint(f'[green]Saved to {outpath_dataset}[/green]')

print('\n[Transposed view]')
print(tab_dataset.set_index(['dataset', 'n_samples']).T)
