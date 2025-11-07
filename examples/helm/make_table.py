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

df['dataset_split'] = df._path.apply(_parse_path)
df['dataset']       = df.dataset_split.apply(lambda x: x.split(':')[0] if ':' in x else x)
df['split']         = df.dataset_split.apply(lambda x: x.split(':')[1] if ':' in x else x)

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


df['p_lr_dkps8'] = df['p_lr_dkps8__n_components_cmds=8__n_models=ALL']

df['n_dataset_split'] = df.groupby('dataset_split').n_samples.transform(max)
df['p_interp']        = (df.n_samples * df.p_sample + (df.n_dataset_split - df.n_samples) * df.p_lr_dkps8) / df.n_dataset_split
df['e_interp']        = np.abs(df.p_interp - df.y_act)

# --
# per split

tab = df[df.n_samples.isin([1, 4, 16, 64])].groupby(['dataset', 'split', 'n_samples']).agg({
    'e_null'                                        : 'mean',
    'e_sample'                                      : 'mean',
    'e_lr_dkps8__n_components_cmds=8__n_models=ALL' : 'mean',
    'e_interp'                                      : 'mean',
})
tab[tab > 9999] = np.nan
tab.rename(columns={
    'e_null'                                        : 'Population Mean',
    'e_sample'                                      : 'Sample Mean',
    'e_lr_dkps8__n_components_cmds=8__n_models=ALL' : 'DKPS(d=8, n_models=ALL)',
    'e_interp'                                      : 'Interp(DKPS(d=8, n_models=ALL), Sample Mean)',
}, inplace=True)

tab = tab.reset_index()
print(tab)
tab.to_csv('table-by_dataset_split.tsv', sep='\t')

# --
# per dataset

split_sizes = df[['dataset_split', 'n_dataset_split']].drop_duplicates()
split_sizes = dict(zip(split_sizes.dataset_split.values, split_sizes.n_dataset_split.values))

def _compute_pred(sub, split_sizes):
    split_sizes = np.array([split_sizes[dataset_split] for dataset_split in sub.dataset_split.values])
    
    methods = ['y_act', 'p_null', 'p_sample', 'p_lr_dkps8__n_components_cmds=8__n_models=ALL', 'p_interp']
    preds   = {method:sub[method].values @ split_sizes / split_sizes.sum() for method in methods}
    
    errs = {method.replace('p_', 'e_'):np.abs(preds[method] - preds['y_act']) for method in methods}
    return pd.Series(errs)

# takes a while ...
df_sub = df[df.n_samples.isin([1, 4, 16, 64])]

df_dataset = df_sub.groupby(['dataset', 'n_samples', 'seed', 'target_model']).apply(lambda x: _compute_pred(x, split_sizes))
df_dataset.reset_index(inplace=True)


tab_dataset = df_dataset.groupby(['dataset', 'n_samples']).agg({
    'e_null'      : 'mean',
    'e_sample'    : 'mean',
    'e_lr_dkps8__n_components_cmds=8__n_models=ALL' : 'mean',
    'e_interp'    : 'mean',
})

tab_dataset[tab_dataset > 9999] = np.nan
tab_dataset.rename(columns={
    'e_null'                                        : 'Population Mean',
    'e_sample'                                      : 'Sample Mean',
    'e_lr_dkps8__n_components_cmds=8__n_models=ALL' : 'DKPS(d=8, n_models=ALL)',
    'e_interp'                                      : 'Interp(DKPS(d=8, n_models=ALL), Sample Mean)',
}, inplace=True)

tab_dataset = tab_dataset.reset_index()
print(tab_dataset.T)
print(tab_dataset.T.to_latex())

tab_dataset.T.to_csv('table-by_dataset.tsv', sep='\t')