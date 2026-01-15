"""Collect unique target models from all dkps results files."""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()

RESULTS_DIR = Path('results')

# Find all dkps results files
tsv_paths = list(RESULTS_DIR.glob('**/dkps/results.tsv'))
console.print(f'Found {len(tsv_paths)} result files', style='green')

# Collect models and n_samples per dataset_split
models_by_split = defaultdict(set)
n_queries_by_split = {}
all_models = set()

for tsv_path in tsv_paths:
    # Parse dataset_split from path
    parts = tsv_path.parts
    runner_idx = parts.index('dkps')
    dataset_split = parts[runner_idx - 3]

    df = pd.read_csv(tsv_path, sep='\t', usecols=['target_model', 'n_samples'])
    models = set(df['target_model'].unique())
    max_n_samples = df['n_samples'].max()

    models_by_split[dataset_split].update(models)
    all_models.update(models)

    # Track max n_samples (number of queries) per split
    if dataset_split not in n_queries_by_split:
        n_queries_by_split[dataset_split] = max_n_samples
    else:
        n_queries_by_split[dataset_split] = max(n_queries_by_split[dataset_split], max_n_samples)

console.print(f'\nTotal unique models: {len(all_models)}', style='bold')

# Group models by family
def model2family(model):
    return model.split('_')[0]

models_by_family = defaultdict(list)
for m in all_models:
    models_by_family[model2family(m)].append(m)

# Table 1: Models by family
table1 = Table(title='Models by Family', show_lines=True)
table1.add_column('Family', style='cyan', no_wrap=True)
table1.add_column('Count', style='magenta', justify='right')
table1.add_column('Models', style='white')

for family in sorted(models_by_family.keys()):
    models = sorted(models_by_family[family])
    # Strip family prefix from model names for readability
    short_names = [m.replace(family + '_', '') for m in models]
    table1.add_row(family, str(len(models)), ', '.join(short_names))

console.print(table1)

# Table 2: Missing models per dataset_split
table2 = Table(title='Missing Models by Dataset Split', show_lines=True)
table2.add_column('Dataset Split', style='cyan', no_wrap=True)
table2.add_column('Present', style='green', justify='right')
table2.add_column('Missing', style='red', justify='right')
table2.add_column('Missing Models (by family)', style='white')

for dataset_split in sorted(models_by_split.keys()):
    present = models_by_split[dataset_split]
    missing = all_models - present

    if missing:
        # Group missing by family
        missing_by_family = defaultdict(list)
        for m in missing:
            missing_by_family[model2family(m)].append(m.replace(model2family(m) + '_', ''))

        missing_str = '; '.join(
            f"{fam}: {', '.join(sorted(ms))}"
            for fam, ms in sorted(missing_by_family.items())
        )
    else:
        missing_str = '-'

    table2.add_row(
        dataset_split,
        str(len(present)),
        str(len(missing)),
        missing_str
    )

console.print(table2)

# Table 3: Number of queries per dataset_split
table3 = Table(title='Number of Queries per Dataset Split', show_lines=True)
table3.add_column('Dataset Split', style='cyan', no_wrap=True)
table3.add_column('N Queries', style='magenta', justify='right')

for dataset_split in sorted(n_queries_by_split.keys()):
    table3.add_row(dataset_split, str(n_queries_by_split[dataset_split]))

console.print(table3)
