"""
    helm.model_dkps_analysis
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint
from scipy.stats import rankdata
from sklearn.utils.fixes import platform

rprint('[yellow] Assumption - all metrics are bounded between 0 and 1[/yellow]')

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default='legalbench:subset=abercrombie')
    parser.add_argument('--score_col', type=str, default='score')
    parser.add_argument('--outdir',    type=str, default='results')
    args = parser.parse_args()
    
    args.tsv_path = Path(args.outdir) / f'{args.dataset}-{args.score_col}-res--holdout.tsv'
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
df_res['er_lr_dkps'] = df_res['er_lr_dkps__n_components_cmds=8__n_models=ALL']
df_res['r2_lr_dkps'] = df_res['r2_lr_dkps__n_components_cmds=8__n_models=ALL']

# compute interpolation
max_samples            = df_res.n_samples.max()
df_res['p_interp']     = (df_res.n_samples * df_res.p_sample + (max_samples - df_res.n_samples) * df_res.p_lr_dkps) / max_samples
df_res['e_interp']     = np.abs(df_res.p_interp - df_res.y_act)
df_res['er_interp'] = df_res['er_lr_dkps']
df_res['r2_interp'] = df_res['r2_lr_dkps']

if any([xx in args.dataset for xx in ['med_qa', 'legalbench']]):
    df_res = df_res[df_res.n_samples > 2]

# <<<<<<<<<<<<<<<<<

df_avg = df_res.groupby(['seed', 'n_samples']).agg({
    f'e_lr_dkps__n_components_cmds=8__n_models=20'   : np.nanmean,
    f'r2_lr_dkps__n_components_cmds=8__n_models=20'  : np.nanmean,
    f'e_lr_dkps__n_components_cmds=8__n_models=50'   : np.nanmean,
    f'r2_lr_dkps__n_components_cmds=8__n_models=50'  : np.nanmean,
    f'e_lr_dkps__n_components_cmds=8__n_models=ALL'  : np.nanmean,
    f'r2_lr_dkps__n_components_cmds=8__n_models=ALL' : np.nanmean,
}).reset_index()

# -------------------------------------------------------
# Upper left

def make_upper_left(n_samples=4):
    
    e_col      = 'e_lr_dkps__n_components_cmds=8__n_models=ALL'
    r2_col     = 'r2_lr_dkps__n_components_cmds=8__n_models=ALL'
    df_avg_sub = df_avg[df_avg.n_samples == n_samples]
    
    # Extract x and y values
    x_all = 1 - df_avg_sub[r2_col].values
    y_all = df_avg_sub[e_col].values
    
    # Plot scatter points
    _ = plt.scatter(x_all, y_all, c='black', alpha=0.25)
    
    # Add polynomial fit if we have enough points
    if len(x_all) > 1:
        poly_all  = np.polyfit(x_all, y_all, 1)
        x_fit_all = np.linspace(x_all.min(), x_all.max(), 100)
        y_fit_all = np.polyval(poly_all, x_fit_all)
        _ = plt.plot(x_fit_all, y_fit_all, linestyle='-', linewidth=3, color='green', label='Linear Fit')
    
    # Mark the location of max R² (min 1-R²) for this n_samples
    if len(x_all) > 0:
        max_r2_idx = np.argmin(x_all)
        max_r2_x   = x_all[max_r2_idx]
        max_r2_y   = y_all[max_r2_idx]
        _ = plt.scatter(max_r2_x, max_r2_y, marker='x', color='red', s=100, linewidths=2, zorder=5, label=f'Max R² Query Set ({max_r2_y:.2f})')
    
    # Mark the mean error for this n_samples
    if len(y_all) > 0:
        mean_y = np.mean(y_all)
        _ = plt.axhline(mean_y, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean Error ({mean_y:.2f})')
    
    _ = plt.xlabel('1 - R²')
    _ = plt.ylabel('Error')
    _ = plt.title(f'{args.dataset} - n_samples={n_samples}')
    _ = plt.legend()
    _ = plt.tight_layout()
    _ = plt.savefig(args.plot_dir / f'{args.score_col}-upper-left-n_samples={n_samples}.png', bbox_inches='tight')
    _ = plt.close()


make_upper_left(n_samples=8)


def make_upper_center(n_samples=4):
    n_models_values = [20, 50, 'ALL']
    color_map = plt.cm.viridis(np.linspace(0, 1, len(n_models_values)))

    for n_models, color in zip(n_models_values, color_map):
        _suffix  = f'lr_dkps__n_components_cmds=8__n_models={n_models}'
        e_col    = 'e_' + _suffix
        r2_col   = 'r2_' + _suffix
        
        df_avg_sub = df_avg[df_avg.n_samples == n_samples]
        
        errors = df_avg_sub[e_col].values
        r2_values = df_avg_sub[r2_col].values
        
        # Plot histogram of errors
        _ = plt.hist(errors, bins=30, alpha=0.5, color=color, label=f'n_models={n_models}')
        
        # Find error corresponding to max R²
        if len(r2_values) > 0:
            max_r2_idx = np.argmax(r2_values)
            max_r2_value = r2_values[max_r2_idx]
            error_at_max_r2 = errors[max_r2_idx]
            
            # Calculate percentile of max R² value
            from scipy.stats import rankdata
            percentile = (rankdata(errors, method='average')[max_r2_idx] / len(errors)) * 100
            
            _ = plt.axvline(error_at_max_r2, color=color, linestyle='--', linewidth=2, alpha=0.7, 
                        label=f'n_models={n_models} max R² (err={error_at_max_r2:.2f}, {int(percentile)}th percentile)')
        
        # Add mean error line
        mean_error = np.mean(errors)
        _ = plt.axvline(mean_error, color=color, linestyle='-', linewidth=2, alpha=0.7)

    _ = plt.title(f'{args.dataset} - Error Distribution for n_samples={n_samples} by n_models')
    _ = plt.xlabel('Error')
    _ = plt.ylabel('Frequency')
    _ = plt.legend()
    _ = plt.grid(alpha=0.25)
    _ = plt.tight_layout()
    _ = plt.savefig(args.plot_dir / f'{args.score_col}-upper-center-n_samples={n_samples}.png', bbox_inches='tight')
    _ = plt.close()

make_upper_center(n_samples=8)


def make_upper_right(n_samples=4, n_components_cmds=8):
    # Define consistent colors for n_samples
    n_samples_list = sorted(df_avg.n_samples.unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(n_samples_list)))
    color_map = {n_samples: colors[i] for i, n_samples in enumerate(n_samples_list)}

    # Define markers and line styles for n_models
    n_models_values = [20, 'ALL']
    markers = {20: 'o', 'ALL': '^'}
    linestyles = {20: '-', 'ALL': '--'}

    for n_samples in n_samples_list:
        sub = df_avg[df_avg.n_samples == n_samples]
        color = color_map[n_samples]
        
        for n_models in n_models_values:
            _suffix = f'lr_dkps__n_components_cmds={n_components_cmds}__n_models={n_models}'
            r2_col = 'r2_' + _suffix
            e_col = 'e_' + _suffix
            
            # Plot scatter points
            _ = plt.scatter(1 - sub[r2_col], sub[e_col], marker=markers[n_models], color=color, s=2, alpha=0.05)
            
            # Add polynomial fit
            x_data = 1 - sub[r2_col].values
            y_data = sub[e_col].values
            if len(x_data) > 1:
                log_x_data = np.log(x_data)
                log_y_data = np.log(y_data)
                poly = np.polyfit(log_x_data, log_y_data, 1)
                x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                log_x_fit = np.log(x_fit)
                log_y_fit = np.polyval(poly, log_x_fit)
                y_fit = np.exp(log_y_fit)
                _ = plt.plot(x_fit, y_fit, color=color, linestyle=linestyles[n_models], linewidth=1.5)

    # Create separate legends for colors (n_samples) and markers (n_models)
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[n_samples], markersize=8, label=f'n_samples={n_samples}') for n_samples in n_samples_list]
    marker_handles = [Line2D([0], [0], marker=markers[n_models], color='w', markerfacecolor='gray', markersize=8, label=f'n_models={n_models}') for n_models in n_models_values]
    line_handles = [Line2D([0], [0], color='gray', linestyle=linestyles[n_models], linewidth=1.5, label=f'n_models={n_models} fit') for n_models in n_models_values]

    first_legend = plt.legend(handles=color_handles, title='n_samples', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.gca().add_artist(first_legend)
    second_legend = plt.legend(handles=marker_handles, title='n_models (markers)', bbox_to_anchor=(1.01, 0.5), loc='upper left')
    plt.gca().add_artist(second_legend)
    _ = plt.legend(handles=line_handles, title='n_models (lines)', bbox_to_anchor=(1.01, 0), loc='upper left')

    _ = plt.xlabel('1 - R²')
    _ = plt.ylabel('Error')
    _ = plt.title(f'{args.dataset} - R² vs Error by n_models')
    _ = plt.yscale('log')
    _ = plt.xscale('log')
    _ = plt.grid(alpha=0.25)
    _ = plt.tight_layout()
    _ = plt.savefig(args.plot_dir / f'{args.score_col}-upper-right.png', bbox_inches='tight')
    _ = plt.close()


def make_upper_right_kde(n_samples=4, n_components_cmds=8):
    # Define consistent colors for n_samples
    n_samples_list = sorted(df_avg.n_samples.unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(n_samples_list)))
    color_map = {n_samples: colors[i] for i, n_samples in enumerate(n_samples_list)}

    # Define markers and line styles for n_models
    n_models_values = [20, 'ALL']
    linestyles = {20: '-', 'ALL': '--'}

    for n_samples in n_samples_list:
        sub = df_avg[df_avg.n_samples == n_samples]
        color = color_map[n_samples]
        
        for n_models in n_models_values:
            _suffix = f'lr_dkps__n_components_cmds={n_components_cmds}__n_models={n_models}'
            r2_col = 'r2_' + _suffix
            e_col = 'e_' + _suffix
            
            # Get data points
            x_data = 1 - sub[r2_col].values
            y_data = sub[e_col].values
            
            if len(x_data) > 1:
                # Fit 2D Gaussian to the point cloud
                from scipy.stats import gaussian_kde
                
                # Create log-transformed data for better visualization
                log_x_data = np.log(x_data)
                log_y_data = np.log(y_data)
                
                # Fit KDE
                xy = np.vstack([log_x_data, log_y_data])
                kde = gaussian_kde(xy)
                
                # Create grid for contour plot
                x_min, x_max = log_x_data.min(), log_x_data.max()
                y_min, y_max = log_y_data.min(), log_y_data.max()
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                xx, yy = np.meshgrid(
                    np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 100),
                    np.linspace(y_min - 0.1*y_range, y_max + 0.1*y_range, 100)
                )
                
                # Evaluate KDE on grid
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = kde(positions).reshape(xx.shape)
                
                # Transform back to original scale for plotting
                xx_orig = np.exp(xx)
                yy_orig = np.exp(yy)
                
                # Plot contours
                _ = plt.contour(xx_orig, yy_orig, zz, levels=3, colors=color, linestyles=linestyles[n_models], linewidths=1.5, alpha=0.7)
                
                # Add line of best fit
                poly = np.polyfit(log_x_data, log_y_data, 1)
                x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                log_x_fit = np.log(x_fit)
                log_y_fit = np.polyval(poly, log_x_fit)
                y_fit = np.exp(log_y_fit)
                _ = plt.plot(x_fit, y_fit, color=color, linestyle=linestyles[n_models], linewidth=1, alpha=0.5)

    # Create legends
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color=color_map[n_samples], linewidth=2, label=f'n_samples={n_samples}') for n_samples in n_samples_list]
    line_handles = [Line2D([0], [0], color='gray', linestyle=linestyles[n_models], linewidth=1.5, label=f'n_models={n_models}') for n_models in n_models_values]

    first_legend = plt.legend(handles=color_handles, title='n_samples', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.gca().add_artist(first_legend)
    _ = plt.legend(handles=line_handles, title='n_models', bbox_to_anchor=(1.01, 0.5), loc='upper left')

    _ = plt.xlabel('1 - R²')
    _ = plt.ylabel('Error')
    _ = plt.title(f'{args.dataset} - R² vs Error by n_models (Gaussian KDE)')
    _ = plt.yscale('log')
    _ = plt.xscale('log')
    _ = plt.grid(alpha=0.25)
    _ = plt.tight_layout()
    _ = plt.savefig(args.plot_dir / f'{args.score_col}-upper-right-kde.png', bbox_inches='tight')
    _ = plt.close()

make_upper_right()
make_upper_right_kde()
