"""
Generate summary tables from QSelect results.

QSelect uses holdout validation to select query sets, so R² is available
for query selection. This script shows performance at both dataset split
level and full dataset level.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rich import print as rprint

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser(description='Generate summary tables from QSelect results')
    parser.add_argument('--results_dir',    type=str, default='results')
    parser.add_argument('--embed_provider', type=str, default='google', help='Filter by embedding provider (e.g., google, jina, local-onehot)')
    parser.add_argument('--n_models',       type=str, default='ALL', choices=['20', '50', 'ALL'], help='Number of models used in DKPS')
    parser.add_argument('--n_samples',      type=int, nargs='+', default=[1, 4, 16, 64], help='Sample sizes to include in table')
    return parser.parse_args()

args = parse_args()

rprint('[yellow]Assumption - all metrics are bounded between 0 and 1[/yellow]')
rprint('[yellow]Assumption - metrics are averages of per-sample metrics[/yellow]')

pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# --
# Configuration

RESULTS_DIR = Path(args.results_dir)
RUNNER = 'qselect'
N_SAMPLES_TO_SHOW = args.n_samples
DKPS_COL = f'p_lr_dkps__n_components_cmds=8__n_models={args.n_models}'
R2_COL = f'r2_lr_dkps__n_components_cmds=8__n_models={args.n_models}'

# --
# Load results

tsv_paths = list(RESULTS_DIR.glob(f'embed-{args.embed_provider}/**/{RUNNER}/results.tsv'))
rprint(f'[green]Found {len(tsv_paths)} result files for embed_provider={args.embed_provider}[/green]')


def parse_path(path: Path) -> dict:
    """Parse result path to extract experiment metadata.

    Path format: results/{embed_provider}/{dataset}/{score_col}/{n_replicates}/{runner}/results.tsv
    """
    parts = path.parts
    runner_idx = parts.index(RUNNER)

    return {
        'embed_provider': parts[runner_idx - 4],
        'dataset_split': parts[runner_idx - 3],  # e.g., 'math-subject=algebra'
        'score_col': parts[runner_idx - 2],
        'n_replicates': int(parts[runner_idx - 1]),
    }


dfs = []
for tsv_path in tqdm(tsv_paths, desc='Loading results'):
    df_tmp = pd.read_csv(tsv_path, sep='\t')

    # Add metadata from path
    meta = parse_path(tsv_path)
    for k, v in meta.items():
        df_tmp[k] = v

    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

# Parse dataset and split from dataset_split
df['dataset'] = df.dataset_split.apply(lambda x: x.split('-')[0])
df['split'] = df.dataset_split.apply(lambda x: '-'.join(x.split('-')[1:]) if '-' in x else '')

# Compute max samples per dataset split (used for interpolation)
df['n_dataset_split'] = df.groupby('dataset_split').n_samples.transform('max')

# --
# Clip predictions to [0, 1] and compute errors

pred_cols = [c for c in df.columns if c.startswith('p_')]
rprint(f'[yellow]Clipping prediction columns to (0, 1): {pred_cols}[/yellow]')

for c in pred_cols:
    df[c] = df[c].clip(0, 1)
    df[c.replace('p_', 'e_')] = np.abs(df[c] - df.y_act)

# --
# Compute interpolation prediction

p_lr_dkps = df[DKPS_COL]
df['p_interp'] = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * p_lr_dkps) / df.n_dataset_split
df['e_interp'] = np.abs(df.p_interp - df.y_act)

# Alias for convenience
df['p_lr_dkps'] = df[DKPS_COL]
df['e_lr_dkps'] = np.abs(df['p_lr_dkps'] - df.y_act)

# --
# Select best query sets based on R²
# QSelect always has R² columns for holdout validation

df['r2_interp'] = df[R2_COL]

# For each (dataset_split, n_samples), find seeds with max R²
_df_r2_mean = df.groupby(['dataset_split', 'n_samples', 'seed'])[R2_COL].mean().reset_index()
_df_r2_best = _df_r2_mean[_df_r2_mean.groupby(['dataset_split', 'n_samples'])[R2_COL].transform(lambda x: x == x.max())]
_df_r2_best = _df_r2_best[['dataset_split', 'n_samples', 'seed']].copy()
_df_r2_best['_is_best'] = True

df = pd.merge(df, _df_r2_best, on=['dataset_split', 'n_samples', 'seed'], how='left')
df['_is_best'] = df['_is_best'].fillna(False)

# Create "_best" columns (using aliases)
for c in ['e_interp', 'e_lr_dkps', 'p_interp', 'p_lr_dkps']:
    df[c + '_best'] = df[c].where(df['_is_best'], np.nan)


# ==============================================================================
# Table 1: Per dataset split
# ==============================================================================

rprint('\n[bold cyan]Table 1: Performance by dataset split (QSelect)[/bold cyan]')

df_sub = df[df.n_samples.isin(N_SAMPLES_TO_SHOW)]

tab_split = df_sub.groupby(['dataset', 'split', 'n_samples']).agg({
    'e_null': 'mean',
    'e_sample': 'mean',
    'e_lr_dkps': 'mean',
    'e_lr_dkps_best': 'mean',
    'e_interp': 'mean',
    'e_interp_best': 'mean',
})

tab_split[tab_split > 9999] = np.nan

# Compute best vs avg differences
tab_split['e_dkps_best_vs_avg'] = tab_split['e_lr_dkps_best'] - tab_split['e_lr_dkps']
tab_split['e_interp_best_vs_avg'] = tab_split['e_interp_best'] - tab_split['e_interp']

# Rename columns
tab_split = tab_split.rename(columns={
    'e_null': 'Population Mean',
    'e_sample': 'Sample Mean',
    'e_lr_dkps': 'DKPS (avg)',
    'e_lr_dkps_best': 'DKPS (best)',
    'e_interp': 'Interp (avg)',
    'e_interp_best': 'Interp (best)',
    'e_dkps_best_vs_avg': 'DKPS Δ(best-avg)',
    'e_interp_best_vs_avg': 'Interp Δ(best-avg)',
})

tab_split = tab_split.reset_index()
print(tab_split)
outpath_split = f'table_qselect-v2-{args.embed_provider}-n_models={args.n_models}-by_dataset_split.tsv'
tab_split.to_csv(outpath_split, sep='\t', index=False)
rprint(f'[green]Saved to {outpath_split}[/green]')


# ==============================================================================
# Table 2: Per dataset (aggregated across splits using concatenation)
# ==============================================================================

rprint('\n[bold cyan]Table 2: Performance by dataset (QSelect, concatenated across splits)[/bold cyan]')

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

# Compute errors at dataset level
df_agg['e_null']    = np.abs(df_agg['p_null'] - df_agg['y_act'])
df_agg['e_sample']  = np.abs(df_agg['p_sample'] - df_agg['y_act'])
df_agg['e_lr_dkps'] = np.abs(df_agg['p_lr_dkps'] - df_agg['y_act'])
df_agg['e_interp']  = np.abs(df_agg['p_interp'] - df_agg['y_act'])

# Final aggregation: mean across seeds and models
tab_dataset = df_agg.groupby(['dataset', 'n_samples']).agg({
    'e_null': 'mean',
    'e_sample': 'mean',
    'e_lr_dkps': 'mean',
    'e_interp': 'mean',
})

# Also compute best query set metrics (vectorized)
df_best = df[df._is_best & df.n_samples.isin(N_SAMPLES_TO_SHOW)].copy()
if len(df_best) > 0:
    pred_cols_best = ['y_act', 'p_lr_dkps', 'p_interp']

    for col in pred_cols_best:
        df_best[f'{col}_w'] = df_best[col] * df_best[weight_col]

    agg_dict_best = {f'{col}_w': 'sum' for col in pred_cols_best}
    agg_dict_best[weight_col] = 'sum'

    # Note: best has only one seed per (dataset_split, n_samples), so no seed in groupby
    df_agg_best = df_best.groupby(['dataset', 'n_samples', 'target_model']).agg(agg_dict_best).reset_index()

    for col in pred_cols_best:
        df_agg_best[col] = df_agg_best[f'{col}_w'] / df_agg_best[weight_col]

    df_agg_best['e_lr_dkps_best'] = np.abs(df_agg_best['p_lr_dkps'] - df_agg_best['y_act'])
    df_agg_best['e_interp_best']  = np.abs(df_agg_best['p_interp'] - df_agg_best['y_act'])

    tab_dataset_best = df_agg_best.groupby(['dataset', 'n_samples']).agg({
        'e_lr_dkps_best': 'mean',
        'e_interp_best': 'mean',
    })

    tab_dataset = pd.concat([tab_dataset, tab_dataset_best], axis=1)

tab_dataset[tab_dataset > 9999] = np.nan

# Compute differences
if 'e_lr_dkps_best' in tab_dataset.columns:
    tab_dataset['e_dkps_best_vs_avg'] = tab_dataset['e_lr_dkps_best'] - tab_dataset['e_lr_dkps']
    tab_dataset['e_interp_best_vs_avg'] = tab_dataset['e_interp_best'] - tab_dataset['e_interp']

# Rename columns
rename_map = {
    'e_null': 'Population Mean',
    'e_sample': 'Sample Mean',
    'e_lr_dkps': 'DKPS (avg)',
    'e_interp': 'Interp (avg)',
    'e_lr_dkps_best': 'DKPS (best)',
    'e_interp_best': 'Interp (best)',
    'e_dkps_best_vs_avg': 'DKPS Δ(best-avg)',
    'e_interp_best_vs_avg': 'Interp Δ(best-avg)',
}
tab_dataset = tab_dataset.rename(columns={k: v for k, v in rename_map.items() if k in tab_dataset.columns})

tab_dataset = tab_dataset.reset_index()
print(tab_dataset)
outpath_dataset = f'table_qselect-v2-{args.embed_provider}-n_models={args.n_models}-by_dataset.tsv'
tab_dataset.to_csv(outpath_dataset, sep='\t', index=False)
rprint(f'[green]Saved to {outpath_dataset}[/green]')

# --
# Transposed view for easier reading
print('\n[Transposed view]')
print(tab_dataset.set_index(['dataset', 'n_samples']).T)
