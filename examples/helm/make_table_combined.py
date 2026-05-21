"""
Generate summary tables from baseline (IRT, APW) and combined (DKPS+IRT, Ens) results.

Reads from results-baselines/ and results-combined/, joins on shared keys,
and produces tables in the same style as make_table.py.

Usage:
    python make_table_combined.py
    python make_table_combined.py --n_samples 1 4 16 64
    python make_table_combined.py --query_sel rand
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint

pd.set_option('display.max_rows', 120)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')


def parse_parent_dataset(dataset):
    """'math-subject=algebra' -> 'math', 'legalbench-subset=proa' -> 'legalbench', 'med_qa' -> 'med_qa'"""
    if '-' in dataset and '=' in dataset:
        return dataset.split('-')[0]
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate tables from baseline + combined results')
    parser.add_argument('--baselines_dir', type=str, default='results-baselines')
    parser.add_argument('--combined_dir',  type=str, default='results-combined')
    parser.add_argument('--tables_dir',    type=str, default='tables-combined')
    parser.add_argument('--n_samples',     type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--query_sel',     type=str, default='rand', choices=['rand', 'apw', 'both'],
                        help='Which query selection to show from combined results')
    return parser.parse_args()


args = parse_args()

TABLES_DIR = Path(args.tables_dir)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# --
# Load baselines

baselines_dir = Path(args.baselines_dir)
baseline_paths = sorted(baselines_dir.glob('*.tsv'))
rprint(f'[green]Found {len(baseline_paths)} baseline result files[/green]')

dfs_base = []
for p in baseline_paths:
    df_tmp = pd.read_csv(p, sep='\t')
    df_tmp['dataset'] = p.stem  # e.g. "math-subject=counting_and_probability"
    dfs_base.append(df_tmp)

if dfs_base:
    df_base = pd.concat(dfs_base, ignore_index=True)
else:
    df_base = pd.DataFrame()

# --
# Load combined

combined_dir = Path(args.combined_dir)
combined_paths = sorted(combined_dir.glob('*.tsv'))
rprint(f'[green]Found {len(combined_paths)} combined result files[/green]')

dfs_comb = []
for p in combined_paths:
    df_tmp = pd.read_csv(p, sep='\t')
    df_tmp['dataset'] = p.stem
    dfs_comb.append(df_tmp)

if dfs_comb:
    df_comb = pd.concat(dfs_comb, ignore_index=True)
else:
    df_comb = pd.DataFrame()

# --
# Report coverage

all_datasets = sorted(set(
    list(df_base.dataset.unique() if len(df_base) else []) +
    list(df_comb.dataset.unique() if len(df_comb) else [])
))

rprint(f'\n[bold cyan]Datasets found:[/bold cyan]')
for ds in all_datasets:
    has_base = ds in df_base.dataset.values if len(df_base) else False
    has_comb = ds in df_comb.dataset.values if len(df_comb) else False
    status = []
    if has_base: status.append('baselines')
    if has_comb: status.append('combined')
    marker = '[green]OK[/green]' if (has_base and has_comb) else '[yellow]partial[/yellow]'
    rprint(f'  {marker} {ds}: {", ".join(status)}')

# ==============================================================================
# Table 1: Baselines only (IRT, APW) — per dataset
# ==============================================================================

if len(df_base):
    rprint('\n[bold cyan]Table 1: Baselines (IRT, APW) — MAE by dataset[/bold cyan]')

    df_sub = df_base[df_base.n_samples.isin(args.n_samples)]

    tab_base = df_sub.groupby(['dataset', 'n_samples']).agg({
        'e_null':   'mean',
        'e_sample': 'mean',
        'e_irt':    'mean',
        'e_apw':    'mean',
    }).rename(columns={
        'e_null':   'Pop. Mean',
        'e_sample': 'Sample',
        'e_irt':    'IRT',
        'e_apw':    'APW',
    }).reset_index()

    print(tab_base)

    outpath = TABLES_DIR / 'table-baselines-by_dataset.tsv'
    tab_base.to_csv(outpath, sep='\t', index=False)
    rprint(f'[green]Saved to {outpath}[/green]')

# ==============================================================================
# Table 2: Combined (DKPS, DKPS+IRT, Ensembles) — per dataset
# ==============================================================================

if len(df_comb):
    rprint('\n[bold cyan]Table 2: Combined methods — MAE by dataset[/bold cyan]')

    df_sub = df_comb[df_comb.n_samples.isin(args.n_samples)]

    if args.query_sel != 'both':
        df_sub = df_sub[df_sub.query_sel == args.query_sel]

    group_cols = ['dataset', 'n_samples']
    if args.query_sel == 'both':
        group_cols.append('query_sel')

    # Random query selection results
    df_rand = df_sub[df_sub.query_sel == 'rand']
    tab_rand = df_rand.groupby(group_cols).agg({
        'e_null':         'mean',
        'e_sample':       'mean',
        'e_irt':          'mean',
        'e_dkps':         'mean',
        'e_dkps_irt':     'mean',
        'e_ens_dkps':     'mean',
        'e_ens_dkps_irt': 'mean',
    }).rename(columns={
        'e_null':         'Pop. Mean',
        'e_sample':       'Sample',
        'e_irt':          'IRT',
        'e_dkps':         'DKPS',
        'e_dkps_irt':     'DKPS+IRT',
        'e_ens_dkps':     'Ens(DKPS)',
        'e_ens_dkps_irt': 'Ens(DKPS+IRT)',
    }).reset_index()

    # APW query selection — p_sample under apw is the APW prediction
    df_apw = df_sub[df_sub.query_sel == 'apw']
    tab_apw = df_apw.groupby(group_cols).agg({
        'e_sample': 'mean',
        'e_dkps':   'mean',
    }).rename(columns={
        'e_sample': 'APW',
        'e_dkps':   'APW->DKPS',
    }).reset_index()

    tab_comb = tab_rand.merge(tab_apw, on=group_cols, how='left')

    print(tab_comb)

    suffix = f'-qsel={args.query_sel}' if args.query_sel != 'both' else ''
    outpath = TABLES_DIR / f'table-combined{suffix}-by_dataset.tsv'
    tab_comb.to_csv(outpath, sep='\t', index=False)
    rprint(f'[green]Saved to {outpath}[/green]')

# ==============================================================================
# Table 3: Joined — baselines + combined side by side
# ==============================================================================

if len(df_base) and len(df_comb):
    rprint('\n[bold cyan]Table 3: All methods — MAE by dataset[/bold cyan]')

    # Combined: rand query selection
    df_c = df_comb[df_comb.n_samples.isin(args.n_samples)]
    df_c_rand = df_c[df_c.query_sel == 'rand']
    agg_rand = df_c_rand.groupby(['dataset', 'n_samples']).agg({
        'e_null':         'mean',
        'e_sample':       'mean',
        'e_irt':          'mean',
        'e_dkps':         'mean',
        'e_dkps_irt':     'mean',
        'e_ens_dkps':     'mean',
        'e_ens_dkps_irt': 'mean',
    }).reset_index()

    # Combined: APW query selection
    df_c_apw = df_c[df_c.query_sel == 'apw']
    agg_apw = df_c_apw.groupby(['dataset', 'n_samples']).agg({
        'e_sample': 'mean',
        'e_dkps':   'mean',
    }).rename(columns={
        'e_sample': 'e_apw',
        'e_dkps':   'e_apw_dkps',
    }).reset_index()

    tab_all = agg_rand.merge(agg_apw, on=['dataset', 'n_samples'], how='left')

    tab_all = tab_all.rename(columns={
        'e_null':         'Pop. Mean',
        'e_sample':       'Sample',
        'e_irt':          'IRT',
        'e_apw':          'APW',
        'e_apw_dkps':     'APW->DKPS',
        'e_dkps':         'DKPS',
        'e_dkps_irt':     'DKPS+IRT',
        'e_ens_dkps':     'Ens(DKPS)',
        'e_ens_dkps_irt': 'Ens(DKPS+IRT)',
    }).sort_values(['dataset', 'n_samples'])

    print(tab_all)

    outpath = TABLES_DIR / 'table-all-methods-by_dataset.tsv'
    tab_all.to_csv(outpath, sep='\t', index=False)
    rprint(f'[green]Saved to {outpath}[/green]')

    # --
    # Grand average across datasets (for each n_samples)

    rprint('\n[bold cyan]Grand average across datasets:[/bold cyan]')
    method_cols = ['Pop. Mean', 'Sample', 'IRT', 'APW', 'APW->DKPS', 'DKPS', 'DKPS+IRT', 'Ens(DKPS)', 'Ens(DKPS+IRT)']
    existing_cols = [c for c in method_cols if c in tab_all.columns]
    grand_avg = tab_all.groupby('n_samples')[existing_cols].mean().reset_index()
    print(grand_avg)

    outpath = TABLES_DIR / 'table-all-methods-grand_avg.tsv'
    grand_avg.to_csv(outpath, sep='\t', index=False)
    rprint(f'[green]Saved to {outpath}[/green]')

# ==============================================================================
# Table 4: Aggregated by parent dataset (math, legalbench, wmt_14, med_qa)
# ==============================================================================

if len(df_comb):
    rprint('\n[bold cyan]Table 4: All methods — MAE by parent dataset[/bold cyan]')

    METHOD_COLS = ['Pop. Mean', 'Sample', 'IRT', 'APW', 'APW->DKPS', 'DKPS', 'DKPS+IRT', 'Ens(DKPS)', 'Ens(DKPS+IRT)']

    # Use the per-split table (tab_all if available, else tab_comb)
    if 'tab_all' in dir():
        tab_src = tab_all.copy()
    else:
        tab_src = tab_comb.copy()

    tab_src['parent_dataset'] = tab_src['dataset'].apply(parse_parent_dataset)

    existing_cols = [c for c in METHOD_COLS if c in tab_src.columns]
    tab_parent = tab_src.groupby(['parent_dataset', 'n_samples'])[existing_cols].mean().reset_index()
    tab_parent = tab_parent.sort_values(['parent_dataset', 'n_samples'])

    print(tab_parent)

    outpath = TABLES_DIR / 'table-all-methods-by_parent_dataset.tsv'
    tab_parent.to_csv(outpath, sep='\t', index=False)
    rprint(f'[green]Saved to {outpath}[/green]')

# ==============================================================================
# Summary
# ==============================================================================

rprint(f'\n[bold green]Done. Tables saved to {TABLES_DIR}/[/bold green]')
