import numpy as np
import pandas as pd
from graspologic.embed import select_dimension

from dkps.dkps import DataKernelPerspectiveSpace
from dkps.synthetic import generate_synthetic_data
from dkps.unpaired_dkps import UnpairedDKPS
from dkps.unpaired_dkps_factored import UnpairedDKPS as FactoredUnpairedDKPS
from experiments.unpaired.block0 import run_block0_sweeps, save_block0_results
from experiments.unpaired.block1 import run_s0_synthetic, run_s1_exchange_rate, run_s2_query_kernel
from experiments.unpaired.plots import save_block0_plots, save_block1_plots
from experiments.unpaired.walkthrough import run_walkthrough


def _paired_dataframe(data):
    return data[['model_id', 'query_id', 'embedding']].copy()


def test_generate_synthetic_data_pairing_structure():
    data, dist_gt, metadata = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=10,
        alpha=0.3,
        s=2.0,
        t=2.0,
        random_state=0,
        return_metadata=True,
    )

    assert len(data) == 40
    assert dist_gt.shape == (4, 4)
    assert metadata['n_paired'] == 3
    assert metadata['n_unpaired'] == 7

    paired = data[data['is_paired']]
    unpaired = data[~data['is_paired']]

    paired_counts = paired.groupby('query_id')['model_id'].nunique()
    unpaired_counts = unpaired.groupby('query_id')['model_id'].nunique()

    assert (paired_counts == 4).all()
    assert (unpaired_counts == 1).all()


def test_unpaired_dkps_paired_mode_matches_dkps_distance():
    data, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=8,
        alpha=1.0,
        s=2.0,
        t=2.0,
        random_state=1,
        return_metadata=True,
    )

    paired_df = _paired_dataframe(data)

    dkps = DataKernelPerspectiveSpace(n_components_cmds=2)
    dkps.fit_transform(paired_df)

    unpaired = UnpairedDKPS(mode='paired', n_components_cmds=2)
    unpaired.fit(data)

    np.testing.assert_allclose(unpaired.dist_matrix_, dkps.dist_matrix_, atol=1e-12)


def test_unpaired_constant_kernel_matches_mean_embedding_distance():
    data, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=12,
        alpha=0.0,
        s=2.0,
        t=2.0,
        random_state=2,
        return_metadata=True,
    )

    estimator = UnpairedDKPS(
        mode='unpaired',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    ).fit(data)

    model_names = sorted(data['model_id'].unique())
    mean_embeddings = np.stack([
        np.mean(np.stack(data.loc[data['model_id'] == model_name, 'embedding'].values), axis=0)
        for model_name in model_names
    ])
    expected = np.linalg.norm(mean_embeddings[:, None, :] - mean_embeddings[None, :, :], axis=-1)

    np.testing.assert_allclose(estimator.dist_matrix_, expected, atol=1e-12)


def test_combined_mode_matches_boundaries():
    data_unpaired, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=10,
        alpha=0.0,
        s=2.0,
        t=2.0,
        random_state=3,
        return_metadata=True,
    )
    combined_unpaired = UnpairedDKPS(
        mode='combined',
        query_kernel='constant',
        use_coverage=True,
        n_components_cmds=2,
    ).fit(data_unpaired)
    unpaired_only = UnpairedDKPS(
        mode='unpaired',
        query_kernel='constant',
        use_coverage=True,
        n_components_cmds=2,
    ).fit(data_unpaired)
    np.testing.assert_allclose(combined_unpaired.dist_matrix_, unpaired_only.dist_matrix_, atol=1e-12)

    data_paired, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=10,
        alpha=1.0,
        s=2.0,
        t=2.0,
        random_state=4,
        return_metadata=True,
    )
    combined_paired = UnpairedDKPS(
        mode='combined',
        query_kernel='constant',
        use_coverage=True,
        n_components_cmds=2,
    ).fit(data_paired)
    paired_only = UnpairedDKPS(
        mode='paired',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    ).fit(data_paired)
    np.testing.assert_allclose(combined_paired.dist_matrix_, paired_only.dist_matrix_, atol=1e-12)


def test_pca_query_space_runs():
    data, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=6,
        n_models=4,
        n_queries=10,
        alpha=0.5,
        s=2.0,
        t=2.0,
        random_state=5,
        return_metadata=True,
    )

    estimator = UnpairedDKPS(
        mode='combined',
        query_kernel='constant',
        use_coverage=True,
        coverage_mode='pca',
        n_components_cmds=2,
    ).fit(data)

    assert estimator.query_feature_transformer_ is not None
    assert estimator.dist_matrix_.shape == (4, 4)
    assert np.isfinite(estimator.dist_matrix_).all()

    X = np.stack(data['embedding'].values)
    elbows, _ = select_dimension(X - np.mean(X, axis=0, keepdims=True), n_elbows=estimator.n_elbows_cmds)
    expected_components = int(elbows[-1]) if elbows else min(data['query_vec'].iloc[0].shape[0], X.shape[0], X.shape[1])
    expected_components = max(1, min(expected_components, X.shape[0], X.shape[1]))
    assert estimator.coverage_pca_components_ == expected_components
    assert estimator.query_feature_dim_ == expected_components


def test_rbf_query_kernel_with_default_median_bandwidth_runs():
    data, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=6,
        n_models=4,
        n_queries=10,
        alpha=0.0,
        s=2.0,
        t=2.0,
        random_state=6,
        return_metadata=True,
    )

    estimator = UnpairedDKPS(
        mode='unpaired',
        query_kernel='rbf',
        use_coverage=True,
        coverage_mode='oracle',
        n_components_cmds=2,
    ).fit(data)

    assert estimator.dist_matrix_.shape == (4, 4)
    assert np.isfinite(estimator.dist_matrix_).all()


def test_invalid_query_kernel_raises():
    data, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=6,
        n_models=4,
        n_queries=10,
        alpha=0.0,
        s=2.0,
        t=2.0,
        random_state=7,
        return_metadata=True,
    )

    estimator = UnpairedDKPS(
        mode='unpaired',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    )
    estimator.query_kernel = 'not-a-kernel'

    try:
        estimator.fit(data)
    except ValueError as exc:
        assert 'unsupported query_kernel' in str(exc)
    else:
        raise AssertionError('expected ValueError for unsupported query_kernel')


def test_factored_paired_mode_matches_dkps_distance():
    data, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=8,
        alpha=1.0,
        s=2.0,
        t=2.0,
        random_state=8,
        return_metadata=True,
    )

    paired_df = _paired_dataframe(data)

    dkps = DataKernelPerspectiveSpace(n_components_cmds=2)
    dkps.fit_transform(paired_df)

    factored = FactoredUnpairedDKPS(mode='paired', n_components_cmds=2).fit(data)
    np.testing.assert_allclose(factored.dist_matrix_, dkps.dist_matrix_, atol=1e-12)


def test_factored_unpaired_constant_kernel_uses_only_unpaired_queries():
    data, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=12,
        alpha=0.5,
        s=2.0,
        t=2.0,
        random_state=9,
        return_metadata=True,
    )

    estimator = FactoredUnpairedDKPS(
        mode='unpaired',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    ).fit(data)

    model_names = sorted(data['model_id'].unique())
    mean_embeddings = np.stack([
        np.mean(
            np.stack(
                data.loc[
                    (data['model_id'] == model_name) & (~data['is_paired']),
                    'embedding',
                ].values
            ),
            axis=0,
        )
        for model_name in model_names
    ])
    expected = np.linalg.norm(mean_embeddings[:, None, :] - mean_embeddings[None, :, :], axis=-1)

    np.testing.assert_allclose(estimator.dist_matrix_, expected, atol=1e-12)


def test_factored_combined_matches_boundaries():
    data_unpaired, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=10,
        alpha=0.0,
        s=2.0,
        t=2.0,
        random_state=10,
        return_metadata=True,
    )
    combined_unpaired = FactoredUnpairedDKPS(
        mode='combined',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    ).fit(data_unpaired)
    unpaired_only = FactoredUnpairedDKPS(
        mode='unpaired',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    ).fit(data_unpaired)
    np.testing.assert_allclose(combined_unpaired.dist_matrix_, unpaired_only.dist_matrix_, atol=1e-12)

    data_paired, _, _ = generate_synthetic_data(
        d_act=3,
        d_obs=5,
        n_models=4,
        n_queries=10,
        alpha=1.0,
        s=2.0,
        t=2.0,
        random_state=11,
        return_metadata=True,
    )
    combined_paired = FactoredUnpairedDKPS(
        mode='combined',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    ).fit(data_paired)
    paired_only = FactoredUnpairedDKPS(
        mode='paired',
        query_kernel='constant',
        use_coverage=False,
        n_components_cmds=2,
    ).fit(data_paired)
    np.testing.assert_allclose(combined_paired.dist_matrix_, paired_only.dist_matrix_, atol=1e-12)


def test_factored_rejects_partially_shared_queries():
    data = pd.DataFrame([
        {
            'model_id': 'a',
            'query_id': 'shared_partial',
            'query_vec': np.array([0.0, 0.0]),
            'embedding': np.array([0.0, 0.0]),
        },
        {
            'model_id': 'b',
            'query_id': 'shared_partial',
            'query_vec': np.array([0.0, 0.0]),
            'embedding': np.array([1.0, 0.0]),
        },
        {
            'model_id': 'c',
            'query_id': 'c_only',
            'query_vec': np.array([1.0, 0.0]),
            'embedding': np.array([0.0, 1.0]),
        },
    ])

    try:
        FactoredUnpairedDKPS(mode='combined', n_components_cmds=2).fit(data)
    except AssertionError as exc:
        assert 'exactly one model or in all models' in str(exc)
    else:
        raise AssertionError('expected AssertionError for partially shared query ids')


def test_block1_runners_smoke():
    s0 = run_s0_synthetic(
        n_models=4,
        d_act=2,
        d_obs=4,
        m_totals=(12,),
        alphas=(0.0, 0.5, 1.0),
        s=2.0,
        t=2.0,
        coverage_modes=('oracle',),
        n_seeds=1,
    )
    assert set(s0['estimator']) == {'paired_only', 'unpaired_only', 'combined'}
    assert len(s0) == 9

    s1_summary, s1_search = run_s1_exchange_rate(
        n_models=4,
        d_act=2,
        d_obs=4,
        epsilons=(0.5,),
        m_p_values=(2,),
        m_u_grid=(2, 4),
        s=2.0,
        t=2.0,
        coverage_modes=('oracle',),
        n_seeds=1,
        return_search=True,
    )
    assert len(s1_summary) == 1
    assert len(s1_search) == 2

    s2 = run_s2_query_kernel(
        n_models=4,
        d_act=2,
        d_obs=4,
        n_queries=12,
        d_seps=(0.0, 1.0),
        alphas=(0.0,),
        s=2.0,
        t=2.0,
        coverage_modes=('oracle',),
        n_seeds=1,
    )
    assert set(s2['estimator']) == {
        'linear_mmd',
        'coverage_adjusted_linear',
        'coverage_adjusted_rbf',
    }
    assert len(s2) == 6


def test_block1_plotting_smoke(tmp_path):
    s0 = run_s0_synthetic(
        n_models=4,
        d_act=2,
        d_obs=4,
        m_totals=(12,),
        alphas=(0.0, 1.0),
        coverage_modes=('oracle',),
        n_seeds=1,
    )
    s1_summary, _ = run_s1_exchange_rate(
        n_models=4,
        d_act=2,
        d_obs=4,
        epsilons=(0.5,),
        m_p_values=(2,),
        m_u_grid=(2, 4),
        coverage_modes=('oracle',),
        n_seeds=1,
        return_search=True,
    )
    s2 = run_s2_query_kernel(
        n_models=4,
        d_act=2,
        d_obs=4,
        n_queries=12,
        d_seps=(0.0, 1.0),
        alphas=(0.0,),
        coverage_modes=('oracle',),
        n_seeds=1,
    )

    saved = save_block1_plots(tmp_path, s0_df=s0, s1_summary_df=s1_summary, s2_df=s2)
    assert {path.name for path in saved} == {
        's0_mse_vs_alpha.png',
        's1_exchange_rate.png',
        's2_mse_vs_d_sep.png',
    }
    assert all(path.exists() for path in saved)


def test_block0_runner_and_plotting_smoke(tmp_path):
    results = run_block0_sweeps(
        n_models=4,
        d_act=2,
        n_queries=8,
        alpha_values=(0.0, 1.0),
        latent_noise_stds=(0.0, 1.0),
        obs_noise_stds=(0.0, 1.0),
        extra_noise_dims=(0, 2),
        n_seeds=1,
    )

    assert set(results['study']) == {'alpha', 'latent_noise', 'obs_noise', 'extra_noise_dims'}

    exact_rows = results[
        (results['study'] == 'alpha') &
        (results['estimator'] == 'paired_only') &
        (results['alpha'] == 1.0)
    ]
    assert len(exact_rows) == 1
    assert exact_rows.iloc[0]['max_abs_error'] < 1e-12

    plot_paths = save_block0_plots(tmp_path / 'plots-only', results)
    assert {path.name for path in plot_paths} == {
        'block0_mse.png',
        'block0_max_abs_error.png',
    }
    assert all(path.exists() for path in plot_paths)

    output_dir = save_block0_results(results, tmp_path / 'results')
    assert (output_dir / 'block0_results.csv').exists()
    assert (output_dir / 'plots' / 'block0_mse.png').exists()
    assert (output_dir / 'plots' / 'block0_max_abs_error.png').exists()


def test_walkthrough_smoke(tmp_path):
    results = run_walkthrough(
        tmp_path / 'walkthrough',
        n_models=4,
        n_queries=12,
        sweep_alphas=(0.0, 0.5, 1.0),
        n_sweep_seeds=1,
        seed=0,
    )

    assert (results['output_dir'] / 'walkthrough.md').exists()
    assert (results['output_dir'] / 'case_metrics.csv').exists()
    assert (results['output_dir'] / 'alpha_sweep_no_coverage.csv').exists()
    assert (results['output_dir'] / 'unpaired_all_vs_strict_nonshared.csv').exists()
    assert {path.name for path in results['plots_dir'].iterdir()} == {
        '01_exact_recovery.png',
        '02_unpaired_mean_mismatch.png',
        '03_alpha_sweep_no_coverage.png',
        '04_query_mixture_and_means.png',
        '05_unpaired_all_vs_strict_nonshared.png',
        '06_distance_heatmaps_alpha_0_2.png',
    }
