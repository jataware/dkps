"""
    plot_dkps_compare_embeddings.py - Compare DKPS performance across embedding methods
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default='math:subject=algebra')
    parser.add_argument('--score_col', type=str, default='score')
    parser.add_argument('--results_root', type=str, default='results')
    parser.add_argument('--outdir',    type=str, default='plots')
    args = parser.parse_args()

    args.results_root = Path(args.results_root)
    args.outdir = Path(args.outdir) / 'compare_embeddings'
    args.outdir.mkdir(parents=True, exist_ok=True)

    return args

args = parse_args()

# --
# Helpers

# Default model names for each provider (when no model specified in dir name)
DEFAULT_MODELS = {
    'google': 'gemini-embedding-001',
    'jina': 'jina-embeddings-v3',
    'openrouter': 'all-minilm-l6-v2',
    'litellm': 'text-embedding-3-small',
    'huggingface': 'multilingual-e5-large',
    'sentence-transformers': 'all-MiniLM-L6-v2',
}

def extract_model_name(embed_dir_name):
    """Extract just the model name from directory like 'embed-provider-model_name'"""
    name = embed_dir_name.replace('embed-', '')

    # Check if it's just a provider name (no model specified)
    if name in DEFAULT_MODELS:
        return DEFAULT_MODELS[name]

    # Otherwise, extract model from 'provider-model' format
    for provider in DEFAULT_MODELS.keys():
        if name.startswith(f'{provider}-'):
            model = name[len(provider)+1:]
            # Convert underscores back to slashes for HF-style names
            model = model.replace('_', '/')
            # Get just the model name (last part after /)
            if '/' in model:
                model = model.split('/')[-1]
            return model

    # Fallback: return as-is
    return name

# --
# Find all embedding result directories

embed_dirs = sorted(args.results_root.glob('embed-*'))
if not embed_dirs:
    print(f"No embedding results found in {args.results_root}/embed-*")
    exit(1)

print(f"Found {len(embed_dirs)} embedding methods:")
for d in embed_dirs:
    print(f"  - {d.name} -> {extract_model_name(d.name)}")

# --
# Load results from each embedding method

results = {}
for embed_dir in embed_dirs:
    tsv_path = embed_dir / f'run-dkps-{args.dataset}-{args.score_col}.tsv'
    if not tsv_path.exists():
        print(f"  Warning: {tsv_path} not found, skipping")
        continue

    df = pd.read_csv(tsv_path, sep='\t')

    # Compute interpolation (same as plot_dkps.py)
    df['p_lr_dkps'] = df['p_lr_dkps__n_components_cmds=8__n_models=ALL']
    max_samples = df.n_samples.max()
    df['p_lr_interp'] = (df.n_samples * df.p_sample + (max_samples - df.n_samples) * df.p_lr_dkps) / max_samples
    df['e_lr_interp'] = np.abs(df.p_lr_interp - df.y_act)

    # Extract model name from directory
    model_name = extract_model_name(embed_dir.name)
    results[model_name] = df

if not results:
    print("No valid results found!")
    exit(1)

# --
# Plot: Error vs n_samples, one line per embedding method (interpolation only)

# Colors matching plot_dkps.py style
COLORS = [
    'red',      # primary DKPS color
    'blue',     # interpolation color
    'purple',
    'orange',
    'brown',
    'pink',
    'olive',
    'cyan',
]

plt.figure(figsize=(10, 6))

for i, (model_name, df) in enumerate(results.items()):
    df_avg = df.groupby('n_samples').agg({
        'e_lr_interp': 'mean',
        'e_sample': 'mean',
    }).reset_index()

    color = COLORS[i % len(COLORS)]

    plt.plot(
        df_avg.n_samples,
        df_avg.e_lr_interp,
        label=model_name,
        c=color,
        lw=2,
        marker='o',
        markersize=4,
    )

# Add sample mean baseline from first result (should be same across embeddings)
first_df = list(results.values())[0]
df_avg_baseline = first_df.groupby('n_samples').agg({'e_sample': 'mean'}).reset_index()
plt.plot(
    df_avg_baseline.n_samples,
    df_avg_baseline.e_sample,
    label='Sample Mean',
    c='green',
    lw=2,
    linestyle='-',
)

plt.legend(loc='upper right')
plt.grid('both', alpha=0.25, c='gray')
plt.xscale('log')
plt.ylabel(r'$MAE(\hat{y}, y)$')
plt.xlabel('Number of queries (m)')
plt.title(f'{args.dataset}')

plt.tight_layout()
outpath = args.outdir / f'{args.dataset.replace(":", "-")}-{args.score_col}-compare-embeddings.png'
plt.savefig(outpath, dpi=150)
plt.close()

print(f"\nSaved plot to {outpath}")

# --
# Print summary table

print("\n" + "="*60)
print("Summary: Mean interpolation error at different sample sizes")
print("="*60)

# Collect data for table
table_data = []
for model_name, df in results.items():
    row = {'model': model_name}
    for n in [8, 32, 128, 512]:
        sub = df[df.n_samples == n]
        if len(sub) > 0:
            row[f'n={n}'] = sub.e_lr_interp.mean()
    table_data.append(row)

df_summary = pd.DataFrame(table_data)
print(df_summary.to_string(index=False, float_format='%.4f'))
