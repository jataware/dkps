import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from rich import print as rprint

rprint('[yellow] Assumption - all metrics are bounded between 0 and 1[/yellow]')
rprint('[yellow] Assumption - metrics are averages of per-sample metrics[/yellow]')

pd.set_option('display.float_format', lambda x: f'{x:.3f}')


tsv_paths = glob('results/*-res.tsv')

# --
# IO

df = []
for tsv_path in tqdm(tsv_paths):
    df_tmp          = pd.read_csv(tsv_path, sep='\t')
    df_tmp['_path'] = tsv_path
    df.append(df_tmp)

df = pd.concat(df)

def _parse_path(path):
    return '-'.join(path.split('/')[-1].split('-')[:-2])

df['dataset_split']   = df._path.apply(_parse_path)
df['dataset']         = df.dataset_split.apply(lambda x: x.split(':')[0] if ':' in x else x)
df['split']           = df.dataset_split.apply(lambda x: x.split(':')[1] if ':' in x else x)

df['n_dataset_split'] = df.groupby('dataset_split').n_samples.transform('max')

dataset_sizes = df[['dataset', 'dataset_split', 'n_dataset_split']].drop_duplicates()
dataset_sizes = dataset_sizes.groupby('dataset').n_dataset_split.sum()
dataset_sizes = dataset_sizes.to_dict()
df['n_dataset'] = df.dataset.apply(lambda x: dataset_sizes[x])

df['weight'] = df.n_dataset_split / df.n_dataset

# --

# <<
# Hotfix
dkps_cols = [c for c in df.columns if 'p_' in c]
rprint(f'[yellow]clipping DKPS columns to (0, 1) - {dkps_cols}[/yellow]')
for c in dkps_cols:
    df[c] = df[c].clip(0, 1)

for c in dkps_cols:
    df[c.replace('p_', 'e_')] = np.abs(df[c] - df.y_act)

# >>

p_lr_dkps = df['p_lr_dkps__n_components_cmds=8__n_models=ALL']

df['p_interp']        = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * p_lr_dkps) / df.n_dataset_split
df['e_interp']        = np.abs(df.p_interp - df.y_act)

df['r2_interp']       = df['r2_lr_dkps__n_components_cmds=8__n_models=ALL']

# <<
r2_col              = 'r2_lr_dkps__n_components_cmds=8__n_models=ALL'
_df_r2_mean         = df.groupby(['dataset_split', 'n_samples', 'seed'])[r2_col].mean().reset_index()
_df_r2_best         = _df_r2_mean[_df_r2_mean.groupby(['dataset_split', 'n_samples'])[r2_col].transform(lambda x: x == x.max())]
_df_r2_best.columns = ['dataset_split', 'n_samples', 'seed', '_r2_best']
df_best             = pd.merge(df, _df_r2_best, on=['dataset_split', 'n_samples', 'seed'], how='left')

for c in [
    'e_interp', 
    'e_lr_dkps__n_components_cmds=8__n_models=ALL',
    
    'p_interp',
    'p_lr_dkps__n_components_cmds=8__n_models=ALL',
]:
    df_best[c + '_best'] = df_best[c]
    df_best.loc[df_best['_r2_best'].isna(), c + '_best'] = np.nan

del df
# >>

# --
# per split

tab = df_best[df_best.n_samples.isin([1, 4, 16, 64])].groupby(['dataset', 'split', 'n_samples']).agg({
    'e_null'                                            : 'mean',
    'e_sample'                                          : 'mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL'      : 'mean',
    'e_interp'                                          : 'mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best' : 'mean',
    'e_interp_best'                                     : 'mean',
})
tab[tab > 9999] = np.nan

tab['e_interp_best_vs_avg'] = tab['e_interp_best'] - tab['e_interp']
tab['e_lr_dkps__n_components_cmds=8__n_models=ALL_best_vs_avg'] = tab['e_lr_dkps__n_components_cmds=8__n_models=ALL_best'] - tab['e_lr_dkps__n_components_cmds=8__n_models=ALL']

tab.rename(columns={
    'e_null'                                                   : 'Population Mean',
    'e_sample'                                                 : 'Sample Mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL'             : 'DKPS(d=8, n_models=ALL)',
    'e_interp'                                                 : 'Interp(DKPS(d=8, n_models=ALL), Sample Mean)',
    
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best'        : 'DKPS(d=8, n_models=ALL, best=True)',
    'e_interp_best'                                            : 'Interp(DKPS(d=8, n_models=ALL, best=True), Sample Mean)',
    'e_interp_best_vs_avg'                                     : 'Interp(DKPS(d=8, n_models=ALL, best=True), Sample Mean) - Interp(DKPS(d=8, n_models=ALL), Sample Mean)',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best_vs_avg' : 'DKPS(d=8, n_models=ALL, best=True) - DKPS(d=8, n_models=ALL)',
}, inplace=True)

tab = tab.reset_index()
print(tab)

tab.to_csv('table-by_dataset_split.tsv', sep='\t')

# --
# per dataset

split_sizes = df_best[['dataset_split', 'n_dataset_split']].drop_duplicates()
split_sizes = dict(zip(split_sizes.dataset_split.values, split_sizes.n_dataset_split.values))

def _compute_pred(sub, split_sizes):
    split_sizes = np.array([split_sizes[dataset_split] for dataset_split in sub.dataset_split.values])
    
    methods = [
        'y_act', 
        'p_null', 
        'p_sample', 
        'p_lr_dkps__n_components_cmds=8__n_models=ALL', 
        'p_interp',
        
        'p_lr_dkps__n_components_cmds=8__n_models=ALL_best',
        'p_interp_best',
    ]
    
    preds = {method:sub[method].values @ split_sizes / split_sizes.sum() for method in methods}
    # replace '^p_' with '^e_'
    errs  = {('e_' + method[2:]):np.abs(preds[method] - preds['y_act']) for method in methods}
    return pd.Series(errs)

# takes a while ...
# [BUG] for small datasets, the max number may be less than these
#       so when we average across datasets, we may drop subsets
df_sub = df_best[df_best.n_samples.isin([1, 4, 16, 64])]

df_dataset = df_sub.groupby(['dataset', 'n_samples', 'seed', 'target_model']).apply(lambda x: _compute_pred(x, split_sizes))
df_dataset.reset_index(inplace=True)

tab_dataset = df_dataset.groupby(['dataset', 'n_samples']).agg({
    'e_null'                                            : np.nanmean,
    'e_sample'                                          : np.nanmean,
    'e_lr_dkps__n_components_cmds=8__n_models=ALL'      : np.nanmean,
    'e_interp'                                          : np.nanmean,
})


df_sub_best = df_best[df_best._r2_best.notna()]
df_sub_best = df_sub_best[df_sub_best.n_samples.isin([1, 4, 16, 64])]
df_sub_best = df_sub_best.groupby(['dataset', 'n_samples', 'target_model']).apply(lambda x: _compute_pred(x, split_sizes))
df_sub_best.reset_index(inplace=True)

tab_dataset_best = df_sub_best.groupby(['dataset', 'n_samples']).agg({
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best' : np.nanmean,
    'e_interp_best'                                     : np.nanmean,
})

assert (tab_dataset.index == tab_dataset_best.index).all()
tab_dataset = pd.concat([tab_dataset, tab_dataset_best], axis=1)

tab_dataset[tab_dataset > 9999] = np.nan

tab_dataset['e_interp_best_vs_avg'] = tab_dataset['e_interp_best'] - tab_dataset['e_interp']
tab_dataset['e_lr_dkps__n_components_cmds=8__n_models=ALL_best_vs_avg'] = tab_dataset['e_lr_dkps__n_components_cmds=8__n_models=ALL_best'] - tab_dataset['e_lr_dkps__n_components_cmds=8__n_models=ALL']

tab_dataset.rename(columns={
    'e_null'                                                   : 'Population Mean',
    'e_sample'                                                 : 'Sample Mean',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL'             : 'DKPS(d=8, n_models=ALL)',
    'e_interp'                                                 : 'Interp(DKPS(d=8, n_models=ALL), Sample Mean)',
    
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best'        : 'DKPS(d=8, n_models=ALL, best=True)',
    'e_interp_best'                                            : 'Interp(DKPS(d=8, n_models=ALL, best=True), Sample Mean)',
    'e_interp_best_vs_avg'                                     : 'Interp(DKPS(d=8, n_models=ALL, best=True), Sample Mean) - Interp(DKPS(d=8, n_models=ALL), Sample Mean)',
    'e_lr_dkps__n_components_cmds=8__n_models=ALL_best_vs_avg' : 'DKPS(d=8, n_models=ALL, best=True) - DKPS(d=8, n_models=ALL)',
}, inplace=True)


tab_dataset = tab_dataset.reset_index()
print(tab_dataset.T)
# print(tab_dataset.T.to_latex())

tab_dataset.T.to_csv('table-by_dataset.tsv', sep='\t')