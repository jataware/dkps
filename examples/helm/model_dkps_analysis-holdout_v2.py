"""
    helm.model_dkps_analysis
    
    Don't re-use seeds across models -- LOO style.  
    It looks a little weak - I think because individual models are noisy, so a training set w/ good R2
        may be a good embedding, but still not always work well on a given model.
    Correspones to model_dkps_holdout_v2.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint
from scipy.stats import rankdata

rprint('[yellow] Assumption - all metrics are bounded between 0 and 1[/yellow]')

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default='legalbench:subset=abercrombie')
    parser.add_argument('--score_col', type=str, default='score')
    parser.add_argument('--outdir',    type=str, default='results')
    args = parser.parse_args()
    
    args.tsv_path = Path(args.outdir) / f'{args.dataset}-{args.score_col}-res--holdout-v2.tsv'
    args.plot_dir = Path('plots-v3') / args.dataset.replace(':', '-')
    
    args.plot_dir.mkdir(parents=True, exist_ok=True)
        
    return args


args   = parse_args()

# --
# IO

df_res = pd.read_csv(args.tsv_path, sep='\t')
model_names  = df_res.target_model.unique()
n_replicates = df_res.seed.nunique()

# <<
# Hotfix

dkps_cols = [c for c in df_res.columns if 'p_' in c]
rprint(f'[yellow]clipping DKPS columns to (0, 1) - {dkps_cols}[/yellow]')
for c in dkps_cols:
    df_res[c] = df_res[c].clip(0, 1)

for c in dkps_cols:
    df_res[c.replace('p_', 'e_')] = np.abs(df_res[c] - df_res.y_act)

# >>

# alias the run with all models
df_res['p_lr_dkps']  = df_res['p_lr_dkps__n_components_cmds=8__n_models=ALL']
df_res['e_lr_dkps']  = df_res['e_lr_dkps__n_components_cmds=8__n_models=ALL']
df_res['r2_lr_dkps'] = df_res['r2_lr_dkps__n_components_cmds=8__n_models=ALL']

# compute interpolation
max_samples         = df_res.n_samples.max()
df_res['p_interp']  = (df_res.n_samples * df_res.p_sample + (max_samples - df_res.n_samples) * df_res.p_lr_dkps) / max_samples
df_res['e_interp']  = np.abs(df_res.p_interp - df_res.y_act)
df_res['r2_interp'] = df_res['r2_lr_dkps']

if any([xx in args.dataset for xx in ['med_qa', 'legalbench']]):
    df_res = df_res[df_res.n_samples > 2]

# <<<<<<<<<<<<<<<<<

breakpoint()

# For each target_model, compute mean e_interp and e_interp at highest r2
model_stats = []

n_samples = 4
e_col  = 'e_lr_dkps__n_components_cmds=8__n_models=ALL'
r2_col = 'r2_lr_dkps__n_components_cmds=8__n_models=ALL'

for target_model in df_res.target_model.unique():
    model_data = df_res[(df_res.target_model == target_model) & (df_res.n_samples == n_samples)]
    
    # Mean e_interp
    mean_e_interp = model_data[e_col].mean()
    
    # e_interp at highest r2
    best_r2_idx = model_data[r2_col].idxmax()
    e_interp_at_best_r2 = model_data.loc[best_r2_idx, e_col]
    
    # Compute percentile of e_interp_at_best_r2
    percentile_e_interp_at_best_r2 = (model_data[e_col] < e_interp_at_best_r2).mean()
    
    model_stats.append({
        'target_model': target_model,
        'mean_e_interp': mean_e_interp,
        'e_interp_at_best_r2': e_interp_at_best_r2,
        'percentile_e_interp_at_best_r2': percentile_e_interp_at_best_r2
    })

df_model_stats = pd.DataFrame(model_stats)
print(df_model_stats)

(df_model_stats.e_interp_at_best_r2 < df_model_stats.mean_e_interp).mean()

plt.scatter(df_model_stats.mean_e_interp, df_model_stats.e_interp_at_best_r2)
plt.xlabel('Mean e_interp')
plt.ylabel('e_interp at best R²')
plt.title('Mean e_interp vs e_interp at best R²')

# Add slope 1 line
min_val = min(df_model_stats.mean_e_interp.min(), df_model_stats.e_interp_at_best_r2.min())
max_val = max(df_model_stats.mean_e_interp.max(), df_model_stats.e_interp_at_best_r2.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
plt.legend()

plt.show()

# -------------------------------------------------------
# PLOT 1 - R2 vs Error by n_models

plt.close()

# Define consistent colors for n_samples
n_samples_list = sorted(df_avg.n_samples.unique())
colors = plt.cm.plasma(np.linspace(0, 1, len(n_samples_list)))
color_map = {n_samples: colors[i] for i, n_samples in enumerate(n_samples_list)}

# Define markers for n_models
markers = {'20': 'o', '50': 's', 'ALL': '^'}

for n_samples in n_samples_list:
    sub = df_avg[df_avg.n_samples == n_samples]
    color = color_map[n_samples]
    
    # Plot scatter points
    _ = plt.scatter(1 - sub[f'r2_lr_dkps__n_components_cmds=8__n_models=20'], sub[f'e_lr_dkps__n_components_cmds=8__n_models=20'], marker=markers['20'], color=color, s=2, alpha=0.05)
    # _ = plt.scatter(1 - sub[f'r2_lr_dkps__n_components_cmds=8__n_models=50'], sub[f'e_lr_dkps__n_components_cmds=8__n_models=50'], marker=markers['50'], color=color, s=2, alpha=0.5)
    _ = plt.scatter(1 - sub[f'r2_lr_dkps__n_components_cmds=8__n_models=ALL'], sub[f'e_lr_dkps__n_components_cmds=8__n_models=ALL'], marker=markers['ALL'], color=color, s=2, alpha=0.05)
    
    # Add polynomial fit for n_models=20
    x_20 = 1 - sub[f'r2_lr_dkps__n_components_cmds=8__n_models=20'].values
    y_20 = sub[f'e_lr_dkps__n_components_cmds=8__n_models=20'].values
    if len(x_20) > 1:
        log_x_20 = np.log(x_20)
        log_y_20 = np.log(y_20)
        poly_20 = np.polyfit(log_x_20, log_y_20, 1)
        x_fit_20 = np.linspace(x_20.min(), x_20.max(), 100)
        log_x_fit_20 = np.log(x_fit_20)
        log_y_fit_20 = np.polyval(poly_20, log_x_fit_20)
        y_fit_20 = np.exp(log_y_fit_20)
        _ = plt.plot(x_fit_20, y_fit_20, color=color, linestyle='-', linewidth=1)
    
    # Add polynomial fit for n_models=ALL
    x_all = 1 - sub[f'r2_lr_dkps__n_components_cmds=8__n_models=ALL'].values
    y_all = sub[f'e_lr_dkps__n_components_cmds=8__n_models=ALL'].values
    if len(x_all) > 1:
        log_x_all = np.log(x_all)
        log_y_all = np.log(y_all)
        poly_all  = np.polyfit(log_x_all, log_y_all, 1)
        x_fit_all = np.linspace(x_all.min(), x_all.max(), 100)
        log_x_fit_all = np.log(x_fit_all)
        log_y_fit_all = np.polyval(poly_all, log_x_fit_all)
        y_fit_all = np.exp(log_y_fit_all)
        _ = plt.plot(x_fit_all, y_fit_all, color=color, linestyle='-', linewidth=1)

# Create separate legends for colors (n_samples) and markers (n_models)
from matplotlib.lines import Line2D
color_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[n_samples], markersize=8, label=f'n_samples={n_samples}') for n_samples in n_samples_list]
marker_handles = [Line2D([0], [0], marker=markers[key], color='w', markerfacecolor='gray', markersize=8, label=f'n_models={key}') for key in ['20', 'ALL']]

first_legend = plt.legend(handles=color_handles, title='n_samples', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.gca().add_artist(first_legend)
plt.legend(handles=marker_handles, title='n_models', bbox_to_anchor=(1.01, 0), loc='upper left')

plt.xlabel('1 - R²')
plt.ylabel('Error')
plt.title(f'{args.dataset}')

_ = plt.yscale('log')
_ = plt.xscale('log')
_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-r2-vs-error-by-nmodels.png', bbox_inches='tight')
_ = plt.close()



# -------------------------------------------------------
# PLOT 2 - R2 vs Error by n_models

# Define consistent colors for n_samples with better contrast
n_samples_list = sorted(df_avg.n_samples.unique())
# Use a perceptually uniform colormap with better contrast
colors = plt.cm.tab10(np.linspace(0, 1, min(len(n_samples_list), 10)))
if len(n_samples_list) > 10:
    colors = plt.cm.tab20(np.linspace(0, 1, len(n_samples_list)))

color_map = {n_samples: colors[i] for i, n_samples in enumerate(n_samples_list)}

for n_samples in n_samples_list[::-1]:
    sub = df_avg[df_avg.n_samples == n_samples]
    color = color_map[n_samples]
    
    # Plot scatter points for n_models=ALL only
    _ = plt.scatter(1 - sub[f'r2_lr_dkps__n_components_cmds=8__n_models=ALL'], sub[f'e_lr_dkps__n_components_cmds=8__n_models=ALL'], color=color, s=10, alpha=0.05)
    
    # # Add polynomial fit for n_models=ALL
    x_all = 1 - sub[f'r2_lr_dkps__n_components_cmds=8__n_models=ALL'].values
    y_all = sub[f'e_lr_dkps__n_components_cmds=8__n_models=ALL'].values
    # if len(x_all) > 1:
    #     log_x_all = np.log(x_all)
    #     log_y_all = np.log(y_all)
    #     poly_all  = np.polyfit(log_x_all, log_y_all, 1)
    #     x_fit_all = np.linspace(x_all.min(), x_all.max(), 100)
    #     log_x_fit_all = np.log(x_fit_all)
    #     log_y_fit_all = np.polyval(poly_all, log_x_fit_all)
    #     y_fit_all = np.exp(log_y_fit_all)
    #     _ = plt.plot(x_fit_all, y_fit_all, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Mark the location of min R² (max 1-R²) for this n_samples
    r2_values = sub[f'r2_lr_dkps__n_components_cmds=8__n_models=ALL'].values
    if len(r2_values) > 0:
        min_r2_idx = np.argmax(r2_values)
        min_r2_x   = x_all[min_r2_idx]
        min_r2_y   = y_all[min_r2_idx]
        _ = plt.scatter(min_r2_x, min_r2_y, marker='x', color=color, s=100, linewidths=2, zorder=5)
    
    # Mark the mean error for this n_samples
    if len(y_all) > 0:
        mean_y = np.mean(y_all)
        mean_x = np.mean(x_all)
        _ = plt.scatter(mean_x, mean_y, marker='o', color=color, s=100, linewidths=2, zorder=5)

# Create legend for colors (n_samples)
from matplotlib.lines import Line2D
color_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[n_samples], markersize=8, label=f'n_samples={n_samples}') for n_samples in n_samples_list]

plt.legend(handles=color_handles, title='n_samples', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.xlabel('1 - R²')
plt.ylabel('Error')
plt.title(f'{args.dataset}')

_ = plt.yscale('log')
_ = plt.xscale('log')
_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-r2-vs-error-by-nmodels-v2.png', bbox_inches='tight')
_ = plt.close()

# -------------------------------------------------------
# PLOT 3 - R2 vs Error by n_models

# Histogram of error distribution for n_samples=32 across different n_models
fig, ax = plt.subplots(figsize=(10, 6))

n_models_values = [20, 50, 'ALL']
color_map = plt.cm.viridis(np.linspace(0, 1, len(n_models_values)))

for n_models, color in zip(n_models_values, color_map):
    _suffix  = f'lr_dkps__n_components_cmds=8__n_models={n_models}'
    e_col    = 'e_' + _suffix
    r2_col   = 'r2_' + _suffix
    
    sub = df_avg[df_avg.n_samples == 32]
    
    errors = sub[e_col].values
    r2_values = sub[r2_col].values
    
    # Plot histogram of errors
    _ = ax.hist(errors, bins=30, alpha=0.5, color=color, label=f'n_models={n_models}')
    
    # Find error corresponding to max R²
    if len(r2_values) > 0:
        max_r2_idx = np.argmax(r2_values)
        max_r2_value = r2_values[max_r2_idx]
        error_at_max_r2 = errors[max_r2_idx]
        
        # Calculate percentile of max R² value
        percentile = (rankdata(errors, method='average')[max_r2_idx] / len(errors)) * 100
        
        _ = ax.axvline(error_at_max_r2, color=color, linestyle='--', linewidth=2, alpha=0.7, 
                      label=f'n_models={n_models} max R² (p{percentile:.1f})')
    
    # Add mean error line
    mean_error = np.mean(errors)
    _ = ax.axvline(mean_error, color=color, linestyle='-', linewidth=2, alpha=0.7)

_ = ax.set_title(f'{args.dataset} - Error Distribution for n_samples=32 by n_models')
_ = ax.set_xlabel('Error')
_ = ax.set_ylabel('Frequency')
_ = ax.legend()
_ = ax.grid(alpha=0.25)

_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-error-histogram-nsamples32-by-nmodels.png', bbox_inches='tight')
_ = plt.close()

# -------------------------------------------------------
# PLOT 4 - R2 vs Error by n_samples

# Grid of scatter plots showing R2 vs Error for all n_samples
n_samples_values = sorted(df_avg.n_samples.unique())
n_plots = len(n_samples_values)
n_cols  = min(4, n_plots)
n_rows  = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
if n_plots == 1:
    axes = np.array([axes])

axes = axes.flatten()

_suffix  = 'lr_dkps__n_components_cmds=8__n_models=ALL'
e_col    = 'e_' + _suffix
r2_col   = 'r2_' + _suffix

for ax, n_samples in zip(axes, n_samples_values):
    sub = df_avg[df_avg.n_samples == n_samples]
    
    err = sub[e_col]
    r2  = sub[r2_col]
    
    _ = ax.scatter(1 - r2, err, alpha=0.3, s=10, color='blue')
    
    # Highlight point with highest R2 in red
    max_r2_idx = r2.idxmax()
    _ = ax.scatter(1 - r2.loc[max_r2_idx], err.loc[max_r2_idx], color='red', s=50, zorder=5, edgecolors='black', linewidths=1)
    
    # Add polynomial fit
    x = (1 - r2).values
    y = err.values
    if len(x) > 1:
        log_x = np.log(x)
        log_y = np.log(y)
        poly = np.polyfit(log_x, log_y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        log_x_fit = np.log(x_fit)
        log_y_fit = np.polyval(poly, log_x_fit)
        y_fit = np.exp(log_y_fit)
        _ = ax.plot(x_fit, y_fit, color='red', linestyle='-', linewidth=2)
    
    _ = ax.set_title(f'n_samples={n_samples}')
    _ = ax.set_xlabel('1 - R²')
    _ = ax.set_ylabel('Error')
    _ = ax.set_xscale('log')
    _ = ax.set_yscale('log')
    _ = ax.grid(alpha=0.25)

# Hide unused subplots
for i in range(n_plots, len(axes)):
    axes[i].set_visible(False)

_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-r2-vs-error-grid-by-nsamples.png', bbox_inches='tight')
_ = plt.close()

# -------------------------------------------------------
# PLOT 5 - Error as a function of n_samples

# Compute average error across all runs for each n_samples
e_col  = 'e_lr_dkps__n_components_cmds=8__n_models=ALL'
r2_col = 'r2_lr_dkps__n_components_cmds=8__n_models=ALL'

avg_error_by_nsamples = df_avg.groupby('n_samples')[e_col].mean().reset_index()

# For each n_samples, find the top 50 seeds with the best (highest) R2 and get their errors
n_top = 5
top50_r2_errors_by_nsamples = []
for n_samples in df_avg.n_samples.unique():
    sub = df_avg[df_avg.n_samples == n_samples]
    # Get top 50 seeds by R2 (or all if less than 50)
    n_top = min(n_top, len(sub))
    top_indices = sub[r2_col].nlargest(n_top).index
    for idx in top_indices:
        top50_r2_errors_by_nsamples.append({
            'n_samples': n_samples,
            'error': sub.loc[idx, e_col]
        })

top50_r2_errors_by_nsamples = pd.DataFrame(top50_r2_errors_by_nsamples)

# Plot
_ = plt.figure(figsize=(10, 6))

# Create violin plot for error distribution at each n_samples
n_samples_list = sorted(df_avg.n_samples.unique())
positions = np.arange(len(n_samples_list))

# Prepare data for violin plot (all runs)
violin_data_all = [df_avg[df_avg.n_samples == ns][e_col].values for ns in n_samples_list]

# Create violin plot for all runs
parts_all = plt.violinplot(violin_data_all, positions=positions, widths=0.7, showmeans=False, showmedians=False)

# Style the violin plot for all runs
for pc in parts_all['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.5)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

# Prepare data for violin plot (top 50 runs)
violin_data_top50 = [top50_r2_errors_by_nsamples[top50_r2_errors_by_nsamples.n_samples == ns]['error'].values for ns in n_samples_list]

# Create violin plot for top 50 runs
parts_top50 = plt.violinplot(violin_data_top50, positions=positions, widths=0.7, showmeans=False, showmedians=False)

# Style the violin plot for top 50 runs
for pc in parts_top50['bodies']:
    pc.set_facecolor('red')
    pc.set_alpha(0.5)
    pc.set_edgecolor('darkred')
    pc.set_linewidth(1)

# Plot average on top
avg_positions = [positions[n_samples_list.index(ns)] for ns in avg_error_by_nsamples.n_samples]
_ = plt.plot(avg_positions, avg_error_by_nsamples[e_col], marker='o', label='Average error (all runs)', linewidth=2, zorder=3, color='blue')

# Add dummy patches for the legend
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='blue', label='Average error (all runs)', linewidth=2, markersize=8),
    Patch(facecolor='lightblue', edgecolor='black', alpha=0.5, label='All runs distribution'),
    Patch(facecolor='red', edgecolor='darkred', alpha=0.5, label=f'Top {n_top} best R² runs distribution')
]

_ = plt.xticks(positions, n_samples_list)
_ = plt.xlabel('Number of samples')
_ = plt.ylabel('Error')
_ = plt.title(f'{args.dataset} - Error vs n_samples')
_ = plt.legend(handles=legend_elements)
# _ = plt.grid(alpha=0.25)
# _ = plt.yscale('log')
_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-error-vs-nsamples.png', bbox_inches='tight')
_ = plt.close()



# -------------------------------------------------------
# PLOT 6 - Error distribution by model with best seed per model

# Filter to n_samples=32 to match other plots
_n_samples = 8
df_res_n = df_res[df_res.n_samples == _n_samples]

# Get unique models
models = sorted(df_res_n.target_model.unique())

# Prepare data for violin plot
violin_data = []
best_seed_errors = []
avg_errors = []

for model in models:
    model_data = df_res_n[df_res_n.target_model == model]
    errors = model_data['e_lr_dkps'].values
    violin_data.append(errors)
    
    # Find the best seed for this specific model based on R²
    best_seed_for_model = model_data.loc[model_data['r2_lr_dkps'].idxmax(), 'seed']
    
    # Get the error for the best seed for this model
    best_seed_model_data = model_data[model_data.seed == best_seed_for_model]
    if len(best_seed_model_data) > 0:
        best_seed_error = best_seed_model_data['e_lr_dkps'].values[0]
    else:
        best_seed_error = np.nan
    best_seed_errors.append(best_seed_error)
    
    # Get the average error for this model
    avg_error = errors.mean()
    avg_errors.append(avg_error)

# Sort models by average error
sorted_indices = np.argsort(avg_errors)
models = [models[i] for i in sorted_indices]
violin_data = [violin_data[i] for i in sorted_indices]
best_seed_errors = [best_seed_errors[i] for i in sorted_indices]
avg_errors = [avg_errors[i] for i in sorted_indices]

# Create figure
plt.figure(figsize=(14, 8))

# Create positions for violin plots
positions = np.arange(len(models))

# Create violin plot
parts = plt.violinplot(violin_data, positions=positions, widths=0.7, showmeans=False, showmedians=False)

# Style the violin plot
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.5)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

# Plot best seed errors as points
_ = plt.scatter(positions, best_seed_errors, color='red', s=50, zorder=5, label='Best seed per model (highest R²)')

# Plot average errors as points
_ = plt.scatter(positions, avg_errors, color='blue', s=50, zorder=5, label='Average error')

# Customize plot
_ = plt.xticks(positions, models, rotation=45, ha='right')
_ = plt.xlabel('Model')
_ = plt.ylabel('Error')
_ = plt.title(f'{args.dataset} - Error Distribution by Model (n_samples={_n_samples})')
_ = plt.legend()
_ = plt.grid(alpha=0.25, axis='y')
_ = plt.tight_layout()
_ = plt.savefig(args.plot_dir / f'{args.score_col}-error-by-model-violin.png', bbox_inches='tight')
_ = plt.close()
