import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dkps.synthetic import generate_synthetic_data
from dkps.unpaired_dkps import UnpairedDKPS

from .block1 import offdiag_mse


PRESETS = {
    'default': dict(
        n_models=6,
        n_queries=60,
        sweep_alphas=(0.0, 0.1, 0.2, 0.5, 0.8, 1.0),
        n_sweep_seeds=12,
        seed=0,
    ),
    'quick': dict(
        n_models=5,
        n_queries=24,
        sweep_alphas=(0.0, 0.25, 0.5, 1.0),
        n_sweep_seeds=3,
        seed=0,
    ),
}


ESTIMATOR_COLORS = {
    'paired_only': '#0f766e',
    'unpaired_only': '#c2410c',
    'combined': '#1d4ed8',
    'strict_nonshared_only': '#7c3aed',
}


def _finalize_figure(fig):
    fig.tight_layout()
    return fig


def _center_rows(X):
    X = np.asarray(X, dtype=float)
    return X - np.mean(X, axis=0, keepdims=True)


def _aggregate(df, group_cols, value_col):
    grouped = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped['sem'] = grouped['std'].fillna(0.0) / np.sqrt(grouped['count'].clip(lower=1))
    return grouped


def _model_names(data):
    return sorted(data['model_id'].unique())


def _stack_column(df, column):
    return np.stack(df[column].values) if len(df) else np.empty((0, 2))


def _mean_vectors_by_model(data, column, mask=None):
    model_names = _model_names(data)
    vector_dim = np.asarray(data[column].iloc[0]).shape[0]
    means = []
    counts = []

    for model_name in model_names:
        sub = data[data['model_id'] == model_name]
        if mask is not None:
            sub = sub[mask(sub)]
        counts.append(len(sub))
        if len(sub):
            means.append(np.mean(np.stack(sub[column].values), axis=0))
        else:
            means.append(np.full(vector_dim, np.nan))

    return model_names, np.stack(means), np.asarray(counts, dtype=int)


def _pairwise_distances_from_vectors(X):
    X = np.asarray(X, dtype=float)
    n = len(X)
    dist = np.full((n, n), np.nan, dtype=float)
    finite = np.isfinite(X).all(axis=1)

    for i in range(n):
        if finite[i]:
            dist[i, i] = 0.0
        for j in range(i + 1, n):
            if finite[i] and finite[j]:
                value = float(np.linalg.norm(X[i] - X[j]))
                dist[i, j] = value
                dist[j, i] = value

    return dist


def _strict_nonshared_distance(data):
    _, mean_embeddings, counts = _mean_vectors_by_model(
        data,
        'embedding',
        mask=lambda sub: ~sub['is_paired'],
    )
    return _pairwise_distances_from_vectors(mean_embeddings), counts


def _evaluate_estimators(data, use_coverage=False):
    specs = {
        'paired_only': dict(mode='paired', query_kernel='constant', use_coverage=False),
        'unpaired_only': dict(mode='unpaired', query_kernel='constant', use_coverage=use_coverage),
        'combined': dict(mode='combined', query_kernel='constant', use_coverage=use_coverage),
    }
    estimators = {}
    for name, kwargs in specs.items():
        estimators[name] = UnpairedDKPS(
            coverage_mode='oracle',
            n_components_cmds=2,
            **kwargs,
        ).fit(data)
    return estimators


def _run_case(*, n_models, n_queries, alpha, s, t, d_sep, seed, pi_paired=None, pi_unpaired=None):
    data, dist_gt, metadata = generate_synthetic_data(
        d_act=2,
        d_obs=2,
        n_models=n_models,
        n_queries=n_queries,
        alpha=alpha,
        s=s,
        t=t,
        d_sep=d_sep,
        pi_paired=pi_paired,
        pi_unpaired=pi_unpaired,
        random_state=seed,
        return_metadata=True,
    )
    estimators = _evaluate_estimators(data, use_coverage=False)
    strict_nonshared_dist, strict_counts = _strict_nonshared_distance(data)

    metrics = {
        name: offdiag_mse(estimator.dist_matrix_, dist_gt)
        for name, estimator in estimators.items()
    }
    metrics['strict_nonshared_only'] = offdiag_mse(strict_nonshared_dist, dist_gt)

    return {
        'data': data,
        'dist_gt': dist_gt,
        'metadata': metadata,
        'estimators': estimators,
        'strict_nonshared_dist': strict_nonshared_dist,
        'strict_nonshared_counts': strict_counts,
        'metrics': metrics,
    }


def _run_alpha_sweep(*, n_models, n_queries, alphas, n_seeds):
    rows = []
    balanced = np.array([0.5, 0.5], dtype=float)

    for alpha in alphas:
        for seed in range(n_seeds):
            case = _run_case(
                n_models=n_models,
                n_queries=n_queries,
                alpha=alpha,
                s=np.inf,
                t=np.inf,
                d_sep=2.0,
                seed=seed,
                pi_paired=balanced,
                pi_unpaired=balanced,
            )
            for estimator_name, mse in case['metrics'].items():
                if estimator_name == 'strict_nonshared_only':
                    continue
                rows.append({
                    'alpha': float(case['metadata']['alpha_actual']),
                    'seed': seed,
                    'estimator': estimator_name,
                    'mse': mse,
                })

    return pd.DataFrame(rows)


def _run_unpaired_baseline_sweep(*, n_models, n_queries, alphas, n_seeds):
    rows = []
    pi_paired = np.array([1.0, 0.0], dtype=float)
    pi_unpaired = np.array([0.0, 1.0], dtype=float)

    for alpha in alphas:
        for seed in range(n_seeds):
            case = _run_case(
                n_models=n_models,
                n_queries=n_queries,
                alpha=alpha,
                s=np.inf,
                t=np.inf,
                d_sep=4.0,
                seed=seed,
                pi_paired=pi_paired,
                pi_unpaired=pi_unpaired,
            )
            rows.append({
                'alpha': float(case['metadata']['alpha_actual']),
                'seed': seed,
                'estimator': 'unpaired_only',
                'mse': case['metrics']['unpaired_only'],
                'mean_nonshared_queries_per_model': float(np.mean(case['strict_nonshared_counts'])),
            })
            rows.append({
                'alpha': float(case['metadata']['alpha_actual']),
                'seed': seed,
                'estimator': 'strict_nonshared_only',
                'mse': case['metrics']['strict_nonshared_only'],
                'mean_nonshared_queries_per_model': float(np.mean(case['strict_nonshared_counts'])),
            })

    return pd.DataFrame(rows)


def _plot_exact_recovery(case, output_path):
    data = case['data']
    metadata = case['metadata']
    model_names = metadata['model_names']
    model_a, model_b = model_names[:2]

    paired_a = data[(data['model_id'] == model_a) & (data['is_paired'])].sort_values('query_id')
    paired_b = data[(data['model_id'] == model_b) & (data['is_paired'])].sort_values('query_id')
    Xa = _stack_column(paired_a, 'embedding')
    Xb = _stack_column(paired_b, 'embedding')

    _, mean_embeddings, _ = _mean_vectors_by_model(data, 'embedding')
    offsets = np.asarray(metadata['model_offsets'], dtype=float)
    centered_offsets = _center_rows(offsets)
    centered_means = _center_rows(mean_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax = axes[0]
    n_plot = min(len(Xa), 12)
    if n_plot:
        indices = np.linspace(0, len(Xa) - 1, n_plot, dtype=int)
        for idx in indices:
            x0, y0 = Xa[idx]
            dx, dy = Xb[idx] - Xa[idx]
            ax.arrow(x0, y0, dx, dy, color='#94a3b8', alpha=0.45, width=0.01, length_includes_head=True)
        ax.scatter(Xa[indices, 0], Xa[indices, 1], s=40, color='#0f766e', label=model_a)
        ax.scatter(Xb[indices, 0], Xb[indices, 1], s=40, color='#1d4ed8', label=model_b)
    ax.set_title('Shared-query response pairs')
    ax.set_xlabel('response dim 1')
    ax.set_ylabel('response dim 2')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[1]
    ax.scatter(centered_offsets[:, 0], centered_offsets[:, 1], s=90, facecolors='none', edgecolors='black', linewidths=1.3, label='true offsets')
    ax.scatter(centered_means[:, 0], centered_means[:, 1], s=55, color='#c2410c', marker='x', linewidths=1.8, label='centered model means')
    for i, model_name in enumerate(model_names):
        ax.plot(
            [centered_offsets[i, 0], centered_means[i, 0]],
            [centered_offsets[i, 1], centered_means[i, 1]],
            color='#c2410c',
            alpha=0.35,
            linewidth=1.0,
        )
        ax.text(centered_offsets[i, 0], centered_offsets[i, 1], model_name.replace('model_', 'm'), fontsize=8)
    ax.set_title('Exact geometry recovery after centering')
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_aspect('equal', adjustable='box')

    fig = _finalize_figure(fig)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_unpaired_mean_mismatch(case, output_path):
    data = case['data']
    metadata = case['metadata']
    model_names = metadata['model_names']
    chosen_models = model_names[:3]

    _, mean_embeddings, _ = _mean_vectors_by_model(data, 'embedding')
    offsets = np.asarray(metadata['model_offsets'], dtype=float)
    centered_offsets = _center_rows(offsets)
    centered_means = _center_rows(mean_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax = axes[0]
    palette = ['#0f766e', '#1d4ed8', '#c2410c']
    for color, model_name in zip(palette, chosen_models):
        sub = data[data['model_id'] == model_name]
        X = _stack_column(sub, 'embedding')
        mean = np.mean(X, axis=0)
        ax.scatter(X[:, 0], X[:, 1], s=18, alpha=0.35, color=color, label=model_name)
        ax.scatter(mean[0], mean[1], s=120, color=color, marker='X', edgecolors='black', linewidths=0.5)
    ax.set_title('Unpaired response clouds')
    ax.set_xlabel('response dim 1')
    ax.set_ylabel('response dim 2')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[1]
    ax.scatter(centered_offsets[:, 0], centered_offsets[:, 1], s=90, facecolors='none', edgecolors='black', linewidths=1.3, label='true offsets')
    ax.scatter(centered_means[:, 0], centered_means[:, 1], s=55, color='#c2410c', marker='x', linewidths=1.8, label='centered model means')
    for i, model_name in enumerate(model_names):
        ax.arrow(
            centered_offsets[i, 0],
            centered_offsets[i, 1],
            centered_means[i, 0] - centered_offsets[i, 0],
            centered_means[i, 1] - centered_offsets[i, 1],
            color='#c2410c',
            alpha=0.35,
            width=0.006,
            length_includes_head=True,
        )
        ax.text(centered_offsets[i, 0], centered_offsets[i, 1], model_name.replace('model_', 'm'), fontsize=8)
    ax.set_title('Finite-sample query-mean mismatch')
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_aspect('equal', adjustable='box')

    fig = _finalize_figure(fig)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_alpha_sweep(df, output_path):
    summary = _aggregate(df, ['alpha', 'estimator'], 'mse')

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for estimator in ['paired_only', 'unpaired_only', 'combined']:
        curve = summary[summary['estimator'] == estimator].sort_values('alpha')
        color = ESTIMATOR_COLORS[estimator]
        ax.plot(curve['alpha'], curve['mean'], marker='o', linewidth=2, color=color, label=estimator)
        ax.fill_between(
            curve['alpha'],
            curve['mean'] - curve['sem'],
            curve['mean'] + curve['sem'],
            color=color,
            alpha=0.18,
        )
    ax.set_title('No-coverage alpha sweep in 2D')
    ax.set_xlabel('paired fraction alpha')
    ax.set_ylabel('off-diagonal MSE')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)

    fig = _finalize_figure(fig)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_mismatch_case(case, output_path):
    data = case['data']
    metadata = case['metadata']

    unique_queries = data.drop_duplicates('query_id')
    paired_queries = unique_queries[unique_queries['is_paired']]
    unpaired_queries = unique_queries[~unique_queries['is_paired']]

    _, all_means, _ = _mean_vectors_by_model(data, 'embedding')
    _, nonshared_means, _ = _mean_vectors_by_model(
        data,
        'embedding',
        mask=lambda sub: ~sub['is_paired'],
    )
    offsets = np.asarray(metadata['model_offsets'], dtype=float)

    centered_offsets = _center_rows(offsets)
    centered_all_means = _center_rows(all_means)
    finite_nonshared = np.isfinite(nonshared_means).all(axis=1)
    centered_nonshared = np.full_like(nonshared_means, np.nan, dtype=float)
    if finite_nonshared.any():
        centered_nonshared[finite_nonshared] = _center_rows(nonshared_means[finite_nonshared])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax = axes[0]
    X_paired = _stack_column(paired_queries, 'query_vec')
    X_unpaired = _stack_column(unpaired_queries, 'query_vec')
    if len(X_paired):
        ax.scatter(X_paired[:, 0], X_paired[:, 1], s=45, color='#1d4ed8', alpha=0.55, label='paired query ids')
    if len(X_unpaired):
        ax.scatter(X_unpaired[:, 0], X_unpaired[:, 1], s=45, color='#c2410c', alpha=0.55, label='unpaired query ids')
    ax.set_title('Query locations used by the generator')
    ax.set_xlabel('query dim 1')
    ax.set_ylabel('query dim 2')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_aspect('equal', adjustable='box')

    ax = axes[1]
    ax.scatter(centered_offsets[:, 0], centered_offsets[:, 1], s=90, facecolors='none', edgecolors='black', linewidths=1.3, label='true offsets')
    ax.scatter(centered_all_means[:, 0], centered_all_means[:, 1], s=55, color='#c2410c', marker='x', linewidths=1.8, label='all-query means')
    finite = np.isfinite(centered_nonshared).all(axis=1)
    ax.scatter(
        centered_nonshared[finite, 0],
        centered_nonshared[finite, 1],
        s=65,
        color='#7c3aed',
        marker='^',
        label='nonshared-only means',
    )
    ax.set_title('All-query vs nonshared-only geometry')
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_aspect('equal', adjustable='box')

    fig = _finalize_figure(fig)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_unpaired_baseline_sweep(df, output_path):
    summary = _aggregate(df, ['alpha', 'estimator'], 'mse')

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    for estimator in ['unpaired_only', 'strict_nonshared_only']:
        curve = summary[summary['estimator'] == estimator].sort_values('alpha')
        color = ESTIMATOR_COLORS[estimator]
        ax.plot(curve['alpha'], curve['mean'], marker='o', linewidth=2, color=color, label=estimator)
        ax.fill_between(
            curve['alpha'],
            curve['mean'] - curve['sem'],
            curve['mean'] + curve['sem'],
            color=color,
            alpha=0.18,
        )
    ax.set_title('Current unpaired vs strict nonshared-only')
    ax.set_xlabel('paired fraction alpha')
    ax.set_ylabel('off-diagonal MSE')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)

    ax = axes[1]
    counts = _aggregate(df.drop_duplicates(['alpha', 'seed']), ['alpha'], 'mean_nonshared_queries_per_model')
    ax.plot(counts['alpha'], counts['mean'], marker='o', linewidth=2, color='#475569')
    ax.fill_between(
        counts['alpha'],
        counts['mean'] - counts['sem'],
        counts['mean'] + counts['sem'],
        color='#94a3b8',
        alpha=0.2,
    )
    ax.set_title('Available nonshared queries per model')
    ax.set_xlabel('paired fraction alpha')
    ax.set_ylabel('mean nonshared queries / model')
    ax.grid(alpha=0.25, linewidth=0.6)

    fig = _finalize_figure(fig)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_distance_heatmaps(case, output_path):
    matrices = {
        'ground truth': case['dist_gt'],
        'paired_only': case['estimators']['paired_only'].dist_matrix_,
        'unpaired_only': case['estimators']['unpaired_only'].dist_matrix_,
        'combined': case['estimators']['combined'].dist_matrix_,
    }

    vmax = max(np.nanmax(matrix) for matrix in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.6), constrained_layout=True)

    for ax, (title, matrix) in zip(axes.ravel(), matrices.items()):
        im = ax.imshow(matrix, cmap='viridis', vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('model index')
        ax.set_ylabel('model index')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label('distance')

    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _write_report(output_dir, *, exact_case, unpaired_case, alpha_sweep_df, mismatch_case, baseline_sweep_df, heatmap_case):
    summary_alpha = _aggregate(alpha_sweep_df, ['alpha', 'estimator'], 'mse')
    baseline_summary = _aggregate(baseline_sweep_df, ['alpha', 'estimator'], 'mse')

    def _lookup(summary_df, alpha, estimator):
        row = summary_df[
            (summary_df['alpha'] == alpha) &
            (summary_df['estimator'] == estimator)
        ]
        return float(row.iloc[0]['mean']) if len(row) else np.nan

    notes = [
        '# 2D Walkthrough',
        'This walkthrough keeps `d_act=d_obs=2` so the latent model offsets, query locations, and observed responses can all be plotted directly. The point is not to reproduce the paper figures; it is to show why the current estimators behave the way they do.',
        '## 1. Exact shared-query cancellation',
        '![Exact recovery](plots/01_exact_recovery.png)',
        (
            'Configuration: `alpha=1`, `s=t=inf`, balanced query distribution, and no extra observed noise dimensions. '
            f'Observed MSEs: paired=`{exact_case["metrics"]["paired_only"]:.3e}`, '
            f'unpaired=`{exact_case["metrics"]["unpaired_only"]:.3e}`, '
            f'combined=`{exact_case["metrics"]["combined"]:.3e}`.'
        ),
        (
            'What to notice: the left-panel arrows are nearly identical because every shared query satisfies '
            '`x_ij - x_kj = v_i - v_k` in this noiseless regime. The right panel centers the geometry and shows '
            'that the model means recover the latent offsets exactly up to translation.'
        ),
        '## 2. Fully unpaired, noiseless responses are still not exact at finite sample size',
        '![Unpaired mismatch](plots/02_unpaired_mean_mismatch.png)',
        (
            'Configuration: `alpha=0`, `s=t=inf`, balanced query distribution. '
            f'Observed MSEs: paired=`{unpaired_case["metrics"]["paired_only"]}`, '
            f'unpaired=`{unpaired_case["metrics"]["unpaired_only"]:.3e}`, '
            f'combined=`{unpaired_case["metrics"]["combined"]:.3e}`.'
        ),
        (
            'What to notice: there are no shared queries to cancel the query term. With the constant query kernel and coverage off, '
            'the unpaired estimator reduces to distances between model-wise mean embeddings. The arrows in the right panel are finite-sample '
            'query-mean mismatch, not observation noise.'
        ),
        '## 3. Alpha sweep without coverage',
        '![Alpha sweep](plots/03_alpha_sweep_no_coverage.png)',
        (
            'This sweep uses the same 2D noiseless setting as above, with balanced query distributions and no coverage correction. '
            f'At `alpha=0`, the unpaired MSE is `{_lookup(summary_alpha, 0.0, "unpaired_only"):.3e}`. '
            f'At `alpha=0.5`, the combined MSE is `{_lookup(summary_alpha, 0.5, "combined"):.3e}`. '
            f'At `alpha=1`, all three estimators collapse to roughly `{_lookup(summary_alpha, 1.0, "combined"):.3e}`.'
        ),
        (
            'What to notice: `paired_only` becomes almost exact as soon as some shared queries exist. '
            '`combined` stays close to `unpaired_only` at small alpha because the paired component only receives weight `alpha*`, '
            'so the unpaired part still dominates.'
        ),
        '## 4. The current `mode="unpaired"` uses all observed queries',
        '![Mismatch case](plots/04_query_mixture_and_means.png)',
        (
            'This case uses two separated query components: paired queries come from one component and unpaired queries from the other. '
            'The left panel shows the generator geometry. The right panel compares centered true offsets, all-query means, and nonshared-only means.'
        ),
        (
            'This is the subtle implementation detail behind the earlier surprise: in the current estimator, `mode="unpaired"` means '
            '"set the paired weight to zero," not "throw away shared queries." So the all-query means keep getting more stable as alpha rises, '
            'even though a literal nonshared-only baseline would have fewer and fewer samples.'
        ),
        '## 5. Current unpaired baseline vs strict nonshared-only baseline',
        '![Strict baseline comparison](plots/05_unpaired_all_vs_strict_nonshared.png)',
        (
            f'With the separated query components above, the current unpaired estimator reaches MSE '
            f'`{_lookup(baseline_summary, 1.0, "unpaired_only"):.3e}` at `alpha=1`, while the strict nonshared-only baseline is undefined there '
            'because there are no nonshared queries left.'
        ),
        (
            'What to notice: the current unpaired curve improves with alpha because it is allowed to use shared queries. '
            'The strict nonshared-only curve is the closer match to the literal phrase "unpaired-only," and it becomes data-starved as alpha increases.'
        ),
        '## 6. Distance heatmaps at an intermediate alpha',
        '![Distance heatmaps](plots/06_distance_heatmaps_alpha_0_2.png)',
        (
            'This heatmap snapshot uses `alpha=0.2` in the balanced, noiseless setting. '
            'It makes the same point as the sweep plot in a more concrete way: at low alpha, the combined estimator still looks much more like '
            'the unpaired estimator than the paired one.'
        ),
    ]

    report_path = output_dir / 'walkthrough.md'
    report_path.write_text('\n\n'.join(notes) + '\n')
    return report_path


def run_walkthrough(
        output_dir,
        *,
        n_models=6,
        n_queries=60,
        sweep_alphas=(0.0, 0.1, 0.2, 0.5, 0.8, 1.0),
        n_sweep_seeds=12,
        seed=0,
    ):
    """
    Generate a narrative 2D walkthrough of the current synthetic estimator behavior.

    The walkthrough focuses on the settings that are easiest to reason about by eye:
    no extra observed dimensions, no coverage correction, and 2D latent / observed
    geometry. It writes plots, CSVs, and a markdown report under `output_dir`.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    balanced = np.array([0.5, 0.5], dtype=float)

    exact_case = _run_case(
        n_models=n_models,
        n_queries=n_queries,
        alpha=1.0,
        s=np.inf,
        t=np.inf,
        d_sep=2.0,
        seed=seed,
        pi_paired=balanced,
        pi_unpaired=balanced,
    )
    unpaired_case = _run_case(
        n_models=n_models,
        n_queries=n_queries,
        alpha=0.0,
        s=np.inf,
        t=np.inf,
        d_sep=2.0,
        seed=seed + 1,
        pi_paired=balanced,
        pi_unpaired=balanced,
    )
    mismatch_case = _run_case(
        n_models=n_models,
        n_queries=n_queries,
        alpha=0.5,
        s=np.inf,
        t=np.inf,
        d_sep=4.0,
        seed=seed + 2,
        pi_paired=np.array([1.0, 0.0], dtype=float),
        pi_unpaired=np.array([0.0, 1.0], dtype=float),
    )
    heatmap_case = _run_case(
        n_models=n_models,
        n_queries=n_queries,
        alpha=0.2,
        s=np.inf,
        t=np.inf,
        d_sep=2.0,
        seed=seed + 3,
        pi_paired=balanced,
        pi_unpaired=balanced,
    )

    alpha_sweep_df = _run_alpha_sweep(
        n_models=n_models,
        n_queries=n_queries,
        alphas=sweep_alphas,
        n_seeds=n_sweep_seeds,
    )
    baseline_sweep_df = _run_unpaired_baseline_sweep(
        n_models=n_models,
        n_queries=n_queries,
        alphas=sweep_alphas,
        n_seeds=n_sweep_seeds,
    )

    case_metrics_df = pd.DataFrame([
        {'case': 'exact', **exact_case['metrics']},
        {'case': 'fully_unpaired', **unpaired_case['metrics']},
        {'case': 'mismatch', **mismatch_case['metrics']},
        {'case': 'heatmap_alpha_0_2', **heatmap_case['metrics']},
    ])
    case_metrics_df.to_csv(output_dir / 'case_metrics.csv', index=False)
    alpha_sweep_df.to_csv(output_dir / 'alpha_sweep_no_coverage.csv', index=False)
    baseline_sweep_df.to_csv(output_dir / 'unpaired_all_vs_strict_nonshared.csv', index=False)

    _plot_exact_recovery(exact_case, plots_dir / '01_exact_recovery.png')
    _plot_unpaired_mean_mismatch(unpaired_case, plots_dir / '02_unpaired_mean_mismatch.png')
    _plot_alpha_sweep(alpha_sweep_df, plots_dir / '03_alpha_sweep_no_coverage.png')
    _plot_mismatch_case(mismatch_case, plots_dir / '04_query_mixture_and_means.png')
    _plot_unpaired_baseline_sweep(baseline_sweep_df, plots_dir / '05_unpaired_all_vs_strict_nonshared.png')
    _plot_distance_heatmaps(heatmap_case, plots_dir / '06_distance_heatmaps_alpha_0_2.png')

    report_path = _write_report(
        output_dir,
        exact_case=exact_case,
        unpaired_case=unpaired_case,
        alpha_sweep_df=alpha_sweep_df,
        mismatch_case=mismatch_case,
        baseline_sweep_df=baseline_sweep_df,
        heatmap_case=heatmap_case,
    )

    return {
        'output_dir': output_dir,
        'plots_dir': plots_dir,
        'report_path': report_path,
        'case_metrics': case_metrics_df,
        'alpha_sweep': alpha_sweep_df,
        'baseline_sweep': baseline_sweep_df,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate a 2D DKPS / Unpaired-DKPS walkthrough.')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='default')
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/unpaired/results/walkthrough'))
    args = parser.parse_args()

    config = PRESETS[args.preset]
    results = run_walkthrough(args.output_dir, **config)
    print(f'wrote walkthrough to {results["output_dir"]}')
    print(f'report: {results["report_path"]}')


if __name__ == '__main__':
    main()
