import numpy as np
import pandas as pd

from .synthetic import generate_synthetic_data
from .unpaired_dkps import UnpairedDKPS


def offdiag_mse(dist_est, dist_gt):
    dist_est = np.asarray(dist_est, dtype=float)
    dist_gt = np.asarray(dist_gt, dtype=float)
    assert dist_est.shape == dist_gt.shape, 'dist_est and dist_gt must have the same shape'

    mask = ~np.eye(dist_gt.shape[0], dtype=bool)
    mask &= np.isfinite(dist_est)
    if not np.any(mask):
        return np.nan
    return float(np.mean((dist_est[mask] - dist_gt[mask]) ** 2))


def _make_theory_rbf_bandwidth(dist_matrix, model_names, s, t, floor=1e-8):
    model_index = {model_name: i for i, model_name in enumerate(model_names)}
    response_var = (1.0 / s ** 2) + (1.0 / t ** 2)

    def bandwidth(model_i, model_k, _estimator):
        dist = dist_matrix[model_index[model_i], model_index[model_k]]
        sigma_sq = response_var / max(dist ** 2, floor)
        return float(np.sqrt(max(sigma_sq, floor)))

    return bandwidth


def run_s0_synthetic(
        n_models=20,
        d_act=10,
        d_obs=50,
        m_totals=(200, 500, 1000),
        alphas=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
        s=2.0,
        t=2.0,
        d_sep=2.0,
        coverage_modes=('oracle', 'pca'),
        n_seeds=50,
    ):
    rows = []
    for coverage_mode in coverage_modes:
        for m_total in m_totals:
            for alpha in alphas:
                for seed in range(n_seeds):
                    data, dist_gt, metadata = generate_synthetic_data(
                        d_act=d_act,
                        d_obs=d_obs,
                        n_models=n_models,
                        n_queries=m_total,
                        alpha=alpha,
                        s=s,
                        t=t,
                        d_sep=d_sep,
                        random_state=seed,
                        return_metadata=True,
                    )

                    estimator_specs = {
                        'paired_only': dict(mode='paired', query_kernel='constant', use_coverage=False),
                        'unpaired_only': dict(mode='unpaired', query_kernel='constant', use_coverage=True),
                        'combined': dict(mode='combined', query_kernel='constant', use_coverage=True),
                    }

                    for estimator_name, kwargs in estimator_specs.items():
                        estimator = UnpairedDKPS(
                            coverage_mode=coverage_mode,
                            n_components_cmds=2,
                            **kwargs,
                        )
                        estimator.fit(data)
                        rows.append({
                            'experiment': 's0',
                            'coverage_mode': coverage_mode,
                            'estimator': estimator_name,
                            'seed': seed,
                            'm_total': m_total,
                            'alpha_requested': alpha,
                            'alpha_actual': metadata['alpha_actual'],
                            'n_paired': metadata['n_paired'],
                            'n_unpaired': metadata['n_unpaired'],
                            'mse': offdiag_mse(estimator.dist_matrix_, dist_gt),
                        })

    return pd.DataFrame(rows)


def run_s1_exchange_rate(
        n_models=20,
        d_act=10,
        d_obs=50,
        epsilons=np.linspace(0.1, 1.0, 10),
        m_p_values=(10, 25, 50, 100),
        m_u_grid=(10, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000),
        s=2.0,
        t=2.0,
        d_sep=1.0,
        coverage_modes=('oracle', 'pca'),
        n_seeds=50,
        return_search=False,
    ):
    summary_rows = []
    search_rows = []

    for coverage_mode in coverage_modes:
        for epsilon in epsilons:
            pi_paired = np.array([0.5 + epsilon / 2.0, 0.5 - epsilon / 2.0], dtype=float)
            pi_unpaired = np.array([0.5 - epsilon / 2.0, 0.5 + epsilon / 2.0], dtype=float)
            theory_exchange_rate = np.inf if epsilon == 0 else 1.0 / (epsilon ** 2)

            for m_p in m_p_values:
                for seed in range(n_seeds):
                    candidate_rows = []
                    for m_u in sorted(m_u_grid):
                        n_queries = m_p + m_u
                        alpha = m_p / n_queries
                        data, dist_gt, metadata = generate_synthetic_data(
                            d_act=d_act,
                            d_obs=d_obs,
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

                        paired_only = UnpairedDKPS(
                            mode='paired',
                            query_kernel='constant',
                            use_coverage=False,
                            coverage_mode=coverage_mode,
                            n_components_cmds=2,
                        ).fit(data)

                        combined = UnpairedDKPS(
                            mode='combined',
                            query_kernel='constant',
                            use_coverage=True,
                            coverage_mode=coverage_mode,
                            n_components_cmds=2,
                        ).fit(data)

                        row = {
                            'experiment': 's1_search',
                            'coverage_mode': coverage_mode,
                            'seed': seed,
                            'epsilon': float(epsilon),
                            'm_p': int(m_p),
                            'm_u': int(m_u),
                            'alpha_actual': metadata['alpha_actual'],
                            'paired_only_mse': offdiag_mse(paired_only.dist_matrix_, dist_gt),
                            'combined_mse': offdiag_mse(combined.dist_matrix_, dist_gt),
                            'theory_exchange_rate': theory_exchange_rate,
                        }
                        candidate_rows.append(row)
                        search_rows.append(row)

                    matched = next(
                        (row for row in candidate_rows if np.isfinite(row['combined_mse']) and row['combined_mse'] <= row['paired_only_mse']),
                        None,
                    )

                    summary_rows.append({
                        'experiment': 's1',
                        'coverage_mode': coverage_mode,
                        'seed': seed,
                        'epsilon': float(epsilon),
                        'm_p': int(m_p),
                        'matched_m_u': np.nan if matched is None else int(matched['m_u']),
                        'exchange_rate': np.nan if matched is None else float(matched['m_u'] / m_p),
                        'paired_only_mse': np.nan if matched is None else matched['paired_only_mse'],
                        'combined_mse': np.nan if matched is None else matched['combined_mse'],
                        'theory_exchange_rate': theory_exchange_rate,
                    })

    summary_df = pd.DataFrame(summary_rows)
    search_df = pd.DataFrame(search_rows)
    if return_search:
        return summary_df, search_df
    return summary_df


def run_s2_query_kernel(
        n_models=20,
        d_act=10,
        d_obs=50,
        n_queries=1000,
        d_seps=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
        alphas=(0.0,),
        s=2.0,
        t=2.0,
        coverage_modes=('oracle', 'pca'),
        n_seeds=50,
        pi_paired=(0.5, 0.5),
        pi_unpaired=(0.5, 0.5),
    ):
    rows = []

    for coverage_mode in coverage_modes:
        for d_sep in d_seps:
            for alpha in alphas:
                for seed in range(n_seeds):
                    data, dist_gt, metadata = generate_synthetic_data(
                        d_act=d_act,
                        d_obs=d_obs,
                        n_models=n_models,
                        n_queries=n_queries,
                        alpha=alpha,
                        s=s,
                        t=t,
                        d_sep=d_sep,
                        pi_paired=np.asarray(pi_paired, dtype=float),
                        pi_unpaired=np.asarray(pi_unpaired, dtype=float),
                        random_state=seed,
                        return_metadata=True,
                    )

                    linear_mmd = UnpairedDKPS(
                        mode='combined',
                        query_kernel='constant',
                        use_coverage=False,
                        coverage_mode=coverage_mode,
                        n_components_cmds=2,
                    ).fit(data)

                    coverage_adjusted_linear = UnpairedDKPS(
                        mode='combined',
                        query_kernel='constant',
                        use_coverage=True,
                        coverage_mode=coverage_mode,
                        n_components_cmds=2,
                    ).fit(data)

                    # Use a plug-in estimate from data rather than the synthetic ground truth
                    # so the nonlinear query kernel does not get oracle access to dist_gt.
                    theory_bandwidth = _make_theory_rbf_bandwidth(
                        dist_matrix=coverage_adjusted_linear.dist_matrix_,
                        model_names=coverage_adjusted_linear.model_names_,
                        s=s,
                        t=t,
                    )

                    estimators = {
                        'linear_mmd': linear_mmd,
                        'coverage_adjusted_linear': coverage_adjusted_linear,
                        'coverage_adjusted_rbf': UnpairedDKPS(
                            mode='combined',
                            query_kernel='rbf',
                            use_coverage=True,
                            query_bandwidth=theory_bandwidth,
                            coverage_mode=coverage_mode,
                            n_components_cmds=2,
                        ).fit(data),
                    }

                    for estimator_name, estimator in estimators.items():
                        rows.append({
                            'experiment': 's2',
                            'coverage_mode': coverage_mode,
                            'estimator': estimator_name,
                            'seed': seed,
                            'd_sep': float(d_sep),
                            'alpha_requested': float(alpha),
                            'alpha_actual': metadata['alpha_actual'],
                            'mse': offdiag_mse(estimator.dist_matrix_, dist_gt),
                        })

    return pd.DataFrame(rows)
