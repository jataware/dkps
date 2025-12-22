import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from rich import print as rprint

rprint('[yellow] Assumption - all metrics are bounded between 0 and 1[/yellow]')
rprint('[yellow] Assumption - metrics are averages of per-sample metrics[/yellow]')

pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# Use holdout results from helm.bak
tsv_paths = glob('../helm.bak/results-20251202/*-res--holdout.tsv')

# --
# IO

df = []
for tsv_path in tqdm(tsv_paths):
    df_tmp          = pd.read_csv(tsv_path, sep='\t')
    df_tmp['_path'] = tsv_path
    df.append(df_tmp)

df = pd.concat(df)

def _parse_path(path):
    # Extract dataset-split from path like 'math:subject=algebra-score-res--holdout.tsv'
    filename = path.split('/')[-1]
    # Remove '-res--holdout.tsv' suffix, then remove '-score' or '-meteor' suffix
    base = filename.replace('-res--holdout.tsv', '')
    parts = base.rsplit('-', 1)  # Split on last hyphen to separate score_col
    return parts[0]

df['dataset_split']   = df._path.apply(_parse_path)
df['dataset']         = df.dataset_split.apply(lambda x: x.split(':')[0] if ':' in x else x)
df['split']           = df.dataset_split.apply(lambda x: x.split(':')[1] if ':' in x else x)

df['n_dataset_split'] = df.groupby('dataset_split').n_samples.transform('max')

# --
# Compute interpolation

p_lr_dkps = df['p_lr_dkps__n_components_cmds=8__n_models=ALL']

df['p_interp'] = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * p_lr_dkps) / df.n_dataset_split
df['e_interp'] = np.abs(df.p_interp - df.y_act)

df['r2_interp'] = df['r2_lr_dkps__n_components_cmds=8__n_models=ALL']

# --
# Select top 10% of query sets based on R²

r2_col = 'r2_lr_dkps__n_components_cmds=8__n_models=ALL'

# For each (dataset_split, n_samples), find seeds in the top 10% by R²
_df_r2_mean = df.groupby(['dataset_split', 'n_samples', 'seed'])[r2_col].mean().reset_index()
_df_r2_best = _df_r2_mean[_df_r2_mean.groupby(['dataset_split', 'n_samples'])[r2_col].transform(lambda x: x == x.max())]
_df_r2_best.columns = ['dataset_split', 'n_samples', 'seed', '_r2_best']
df_best = pd.merge(df, _df_r2_best, on=['dataset_split', 'n_samples', 'seed'], how='left')

# Create "_best" columns for best query set selection
for c in [
    'e_interp',
    'p_interp',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL',
    'p_lr_dkps__n_components_cmds=8__n_models=ALL',
]:
    df_best[c + '_best'] = df_best[c]
    df_best.loc[df_best['_r2_best'].isna(), c + '_best'] = np.nan

del df

# --
# per split

tab = df_best[df_best.n_samples.isin([1, 4, 16, 64])].groupby(['dataset', 'split', 'n_samples']).agg({
    'e_null'                                            : 'mean',
    'e_sample'                                          : 'mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL'      : 'mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best' : 'mean',
    'e_interp'                                          : 'mean',
    'e_interp_best'                                     : 'mean',
})
tab[tab > 9999] = np.nan

tab['e_dkps_best_vs_avg']   = tab['e_lr_dkps__n_components_cmds=8__n_models=ALL_best'] - tab['e_lr_dkps__n_components_cmds=8__n_models=ALL']
tab['e_interp_best_vs_avg'] = tab['e_interp_best'] - tab['e_interp']

tab.rename(columns={
    'e_null'                                            : 'Population Mean',
    'e_sample'                                          : 'Sample Mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL'      : 'DKPS (avg)',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best' : 'DKPS (top10%)',
    'e_interp'                                          : 'Interp (avg)',
    'e_interp_best'                                     : 'Interp (top10%)',
    'e_dkps_best_vs_avg'                                : 'DKPS top10%-avg',
    'e_interp_best_vs_avg'                              : 'Interp top10%-avg',
}, inplace=True)

tab = tab.reset_index()
print(tab)

tab.to_csv('table_qselect-by_dataset_split.tsv', sep='\t')

# --
# per dataset

split_sizes = df_best[['dataset_split', 'n_dataset_split']].drop_duplicates()
split_sizes = dict(zip(split_sizes.dataset_split.values, split_sizes.n_dataset_split.values))

def _compute_pred(sub, split_sizes):
    split_sizes_arr = np.array([split_sizes[dataset_split] for dataset_split in sub.dataset_split.values])

    methods = [
        'y_act',
        'p_null',
        'p_sample',
        'p_lr_dkps__n_components_cmds=8__n_models=ALL',
        'p_lr_dkps__n_components_cmds=8__n_models=ALL_best',
        'p_interp',
        'p_interp_best',
    ]

    preds = {method: sub[method].values @ split_sizes_arr / split_sizes_arr.sum() for method in methods}
    errs  = {('e_' + method[2:]): np.abs(preds[method] - preds['y_act']) for method in methods}
    return pd.Series(errs)

# [BUG] for small datasets, the max number may be less than these
#       so when we average across datasets, we may drop subsets
df_sub = df_best[df_best.n_samples.isin([1, 4, 16, 64])]

df_dataset = df_sub.groupby(['dataset', 'n_samples', 'seed', 'target_model']).apply(
    lambda x: _compute_pred(x, split_sizes), include_groups=False
)
df_dataset.reset_index(inplace=True)

tab_dataset = df_dataset.groupby(['dataset', 'n_samples']).agg({
    'e_null'                                       : 'mean',
    'e_sample'                                     : 'mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL' : 'mean',
    'e_interp'                                     : 'mean',
})


df_sub_best = df_best[df_best._r2_best.notna()]
df_sub_best = df_sub_best[df_sub_best.n_samples.isin([1, 4, 16, 64])]
df_sub_best = df_sub_best.groupby(['dataset', 'n_samples', 'target_model']).apply(
    lambda x: _compute_pred(x, split_sizes), include_groups=False
)
df_sub_best.reset_index(inplace=True)

tab_dataset_best = df_sub_best.groupby(['dataset', 'n_samples']).agg({
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best' : 'mean',
    'e_interp_best'                                     : 'mean',
})

assert (tab_dataset.index == tab_dataset_best.index).all()
tab_dataset = pd.concat([tab_dataset, tab_dataset_best], axis=1)

tab_dataset[tab_dataset > 9999] = np.nan

tab_dataset['e_dkps_best_vs_avg']   = tab_dataset['e_lr_dkps__n_components_cmds=8__n_models=ALL_best'] - tab_dataset['e_lr_dkps__n_components_cmds=8__n_models=ALL']
tab_dataset['e_interp_best_vs_avg'] = tab_dataset['e_interp_best'] - tab_dataset['e_interp']

tab_dataset.rename(columns={
    'e_null'                                            : 'Population Mean',
    'e_sample'                                          : 'Sample Mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL'      : 'DKPS (avg)',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best' : 'DKPS (top10%)',
    'e_interp'                                          : 'Interp (avg)',
    'e_interp_best'                                     : 'Interp (top10%)',
    'e_dkps_best_vs_avg'                                : 'DKPS top10%-avg',
    'e_interp_best_vs_avg'                              : 'Interp top10%-avg',
}, inplace=True)


tab_dataset = tab_dataset.reset_index()
print(tab_dataset.T)

tab_dataset.T.to_csv('table_qselect-by_dataset.tsv', sep='\t')
