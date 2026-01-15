#!/usr/bin/env python3
"""Concatenate results.tsv and results-1.tsv files, handling missing columns."""

import argparse
import pandas as pd
from pathlib import Path
from rich import print as rprint

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='results')
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)

# Find all results.tsv files
results_files = list(RESULTS_DIR.glob('**/qselect/results.tsv'))
rprint(f'[green]Found {len(results_files)} results.tsv files[/green]')

results_files = [
    Path("results/embed-google/math-subject=number_theory/score/1024/qselect/results.tsv"),
    Path("results/embed-google/math-subject=precalculus/score/1024/qselect/results.tsv"),
    Path("results/embed-jina/wmt_14-language_pair=hi-en/meteor/1024/qselect/results.tsv"),
]
for results_path in results_files:
    dir_path = results_path.parent
    results1_path = dir_path / 'results-1.tsv'
    output_path = dir_path / 'results-v2.tsv'

    if not results1_path.exists():
        rprint(f'[yellow]Skipping {dir_path} (no results-1.tsv)[/yellow]')
        continue

    rprint(f'[cyan]Processing {dir_path}[/cyan]')

    # Read both files
    df0 = pd.read_csv(results_path, sep='\t')
    df1 = pd.read_csv(results1_path, sep='\t')

    # Drop n_models=50 columns from both
    cols_to_drop = [c for c in df0.columns if 'n_models=50' in c]
    if cols_to_drop:
        rprint(f'  Dropping columns: {cols_to_drop}')
        df0 = df0.drop(columns=cols_to_drop, errors='ignore')

    cols_to_drop = [c for c in df1.columns if 'n_models=50' in c]
    if cols_to_drop:
        df1 = df1.drop(columns=cols_to_drop, errors='ignore')

    # Concatenate - pandas handles missing columns by filling with NaN
    df_combined = pd.concat([df0, df1], ignore_index=True)

    # Save
    df_combined.to_csv(output_path, sep='\t', index=False)
    rprint(f'  -> {output_path} ({len(df0)} + {len(df1)} = {len(df_combined)} rows)')

rprint('[green]Done![/green]')
