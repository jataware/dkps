import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dkps.synthetic import generate_synthetic_data
from dkps.unpaired_dkps import UnpairedDKPS

from .block1 import offdiag_mse
from .plots import save_block0_plots


PRESETS = {
    'paper': dict(
        n_models=20,
        d_act=10,
        n_queries=200,
        alpha_values=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
        latent_noise_stds=(0.0, 0.1, 0.25, 0.5, 1.0),
        obs_noise_stds=(0.0, 0.1, 0.25, 0.5, 1.0),
        extra_noise_dims=(0, 5, 20, 40),
        n_seeds=20,
    ),
    'quick': dict(
        n_models=4,
        d_act=3,
        n_queries=24,
        alpha_values=(0.0, 0.5, 1.0),
        latent_noise_stds=(0.0, 0.25, 1.0),
        obs_noise_stds=(0.0, 0.25, 1.0),
        extra_noise_dims=(0, 2, 6),
        n_seeds=2,
    ),
}


def max_offdiag_abs_error(dist_est, dist_gt):
    dist_est = np.asarray(dist_est, dtype=float)
    dist_gt = np.asarray(dist_gt, dtype=float)
    assert dist_est.shape == dist_gt.shape, 'dist_est and dist_gt must have the same shape'

    mask = ~np.eye(dist_gt.shape[0], dtype=bool)
    mask &= np.isfinite(dist_est)
    if not np.any(mask):
        return np.nan
    return float(np.max(np.abs(dist_est[mask] - dist_gt[mask])))


def _std_to_precision(noise_std):
    return np.inf if noise_std == 0 else float(1.0 / noise_std)


def _evaluate_estimators(data, dist_gt, estimator_specs):
    rows = []
    for estimator_name, kwargs in estimator_specs.items():
        estimator = UnpairedDKPS(**kwargs).fit(data)
        rows.append({
            'estimator': estimator_name,
            'mse': offdiag_mse(estimator.dist_matrix_, dist_gt),
            'max_abs_error': max_offdiag_abs_error(estimator.dist_matrix_, dist_gt),
        })
    return rows


def run_block0_sweeps(
        n_models=20,
        d_act=10,
        n_queries=200,
        alpha_values=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
        latent_noise_stds=(0.0, 0.1, 0.25, 0.5, 1.0),
        obs_noise_stds=(0.0, 0.1, 0.25, 0.5, 1.0),
        extra_noise_dims=(0, 5, 20, 40),
        n_seeds=20,
    ):
    """
    Sanity-check sweeps around the exact-recovery regime.

    Exact finite-sample recovery in this synthetic setup occurs when:
    - all queries are paired (`alpha = 1`)
    - latent and observation noise are zero (`s = t = inf`)
    - there are no appended pure-noise dimensions (`d_obs = d_act`)

    The sweeps below vary one condition at a time away from that baseline so we
    can see how quickly the distance estimates deviate from the true
    `||v_i - v_k||` distances.
    """
    rows = []
    no_mismatch_pi = np.array([0.5, 0.5], dtype=float)

    alpha_estimators = {
        'paired_only': dict(mode='paired', query_kernel='constant', use_coverage=False),
        'combined': dict(mode='combined', query_kernel='constant', use_coverage=False),
        'unpaired_only': dict(mode='unpaired', query_kernel='constant', use_coverage=False),
    }
    paired_estimator = {
        'paired_only': dict(mode='paired', query_kernel='constant', use_coverage=False),
    }

    for seed in range(n_seeds):
        for alpha in alpha_values:
            data, dist_gt, metadata = generate_synthetic_data(
                d_act=d_act,
                d_obs=d_act,
                n_models=n_models,
                n_queries=n_queries,
                alpha=alpha,
                s=np.inf,
                t=np.inf,
                pi_paired=no_mismatch_pi,
                pi_unpaired=no_mismatch_pi,
                random_state=seed,
                return_metadata=True,
            )
            for result in _evaluate_estimators(data, dist_gt, alpha_estimators):
                rows.append({
                    'study': 'alpha',
                    'x_value': float(alpha),
                    'alpha': float(metadata['alpha_actual']),
                    'latent_noise_std': 0.0,
                    'obs_noise_std': 0.0,
                    'extra_noise_dims': 0,
                    'seed': seed,
                    'n_models': n_models,
                    'd_act': d_act,
                    'd_obs': d_act,
                    'n_queries': n_queries,
                    's': np.inf,
                    't': np.inf,
                    **result,
                })

        for latent_noise_std in latent_noise_stds:
            s = _std_to_precision(latent_noise_std)
            data, dist_gt, metadata = generate_synthetic_data(
                d_act=d_act,
                d_obs=d_act,
                n_models=n_models,
                n_queries=n_queries,
                alpha=1.0,
                s=s,
                t=np.inf,
                random_state=seed,
                return_metadata=True,
            )
            for result in _evaluate_estimators(data, dist_gt, paired_estimator):
                rows.append({
                    'study': 'latent_noise',
                    'x_value': float(latent_noise_std),
                    'alpha': float(metadata['alpha_actual']),
                    'latent_noise_std': float(latent_noise_std),
                    'obs_noise_std': 0.0,
                    'extra_noise_dims': 0,
                    'seed': seed,
                    'n_models': n_models,
                    'd_act': d_act,
                    'd_obs': d_act,
                    'n_queries': n_queries,
                    's': s,
                    't': np.inf,
                    **result,
                })

        for obs_noise_std in obs_noise_stds:
            t = _std_to_precision(obs_noise_std)
            data, dist_gt, metadata = generate_synthetic_data(
                d_act=d_act,
                d_obs=d_act,
                n_models=n_models,
                n_queries=n_queries,
                alpha=1.0,
                s=np.inf,
                t=t,
                random_state=seed,
                return_metadata=True,
            )
            for result in _evaluate_estimators(data, dist_gt, paired_estimator):
                rows.append({
                    'study': 'obs_noise',
                    'x_value': float(obs_noise_std),
                    'alpha': float(metadata['alpha_actual']),
                    'latent_noise_std': 0.0,
                    'obs_noise_std': float(obs_noise_std),
                    'extra_noise_dims': 0,
                    'seed': seed,
                    'n_models': n_models,
                    'd_act': d_act,
                    'd_obs': d_act,
                    'n_queries': n_queries,
                    's': np.inf,
                    't': t,
                    **result,
                })

        for extra_noise_dim in extra_noise_dims:
            data, dist_gt, metadata = generate_synthetic_data(
                d_act=d_act,
                d_obs=d_act + int(extra_noise_dim),
                n_models=n_models,
                n_queries=n_queries,
                alpha=1.0,
                s=np.inf,
                t=np.inf,
                random_state=seed,
                return_metadata=True,
            )
            for result in _evaluate_estimators(data, dist_gt, paired_estimator):
                rows.append({
                    'study': 'extra_noise_dims',
                    'x_value': float(extra_noise_dim),
                    'alpha': float(metadata['alpha_actual']),
                    'latent_noise_std': 0.0,
                    'obs_noise_std': 0.0,
                    'extra_noise_dims': int(extra_noise_dim),
                    'seed': seed,
                    'n_models': n_models,
                    'd_act': d_act,
                    'd_obs': d_act + int(extra_noise_dim),
                    'n_queries': n_queries,
                    's': np.inf,
                    't': np.inf,
                    **result,
                })

    return pd.DataFrame(rows)


def save_block0_results(results_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'block0_results.csv', index=False)
    plot_dir = output_dir / 'plots'
    save_block0_plots(plot_dir, results_df)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run Block 0 exact-recovery sensitivity sweeps.')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='paper')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f'experiments/unpaired/results/block0/{args.preset}')
    results_df = run_block0_sweeps(**PRESETS[args.preset])
    save_block0_results(results_df, output_dir)

    print(f'wrote outputs to {output_dir}')
    print(f'  rows: {len(results_df)}')


if __name__ == '__main__':
    main()
