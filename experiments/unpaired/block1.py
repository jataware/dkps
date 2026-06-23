import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor

from dkps.synthetic import generate_benchmark_data
from dkps.unpaired_dkps import DoubleKernelDKPS


ESTIMATORS = {
    'rbf_paired':    dict(query_kernel='rbf', task_filter='shared'),
    'rbf_unpaired':  dict(query_kernel='rbf', task_filter='unshared'),
    'rbf_combined':  dict(query_kernel='rbf'),
}


def predict_scores_knn(embeddings, scores, observed, k=5):
    """Predict held-out scores using KNN in embedding space."""
    n_models, n_tasks = scores.shape
    predictions = np.full_like(scores, np.nan)

    for task_idx in range(n_tasks):
        train_mask = observed[:, task_idx]
        test_mask = ~observed[:, task_idx]
        n_train = train_mask.sum()
        if n_train < 1 or test_mask.sum() < 1:
            continue

        k_eff = min(k, n_train)
        knn = KNeighborsRegressor(n_neighbors=k_eff, weights='distance')
        knn.fit(embeddings[train_mask], scores[train_mask, task_idx])
        predictions[test_mask, task_idx] = knn.predict(embeddings[test_mask])

    return predictions


def evaluate_predictions(predictions, scores, observed):
    """Compute RMSE on held-out (model, task) pairs."""
    held_out = ~observed & np.isfinite(predictions)
    if held_out.sum() == 0:
        return np.nan

    y_true = scores[held_out]
    y_pred = predictions[held_out]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_and_predict(data, scores, observed, k_neighbors=5, n_components=None, **dkps_kwargs):
    """Fit DoubleKernelDKPS, embed models, predict held-out scores."""
    est = DoubleKernelDKPS(**dkps_kwargs)
    est.fit(data)

    dist = est.dist_matrix_.copy()
    if np.any(np.isnan(dist)):
        max_dist = np.nanmax(dist)
        if not np.isfinite(max_dist):
            return np.nan
        dist = np.nan_to_num(dist, nan=max_dist)

    from graspologic.embed import ClassicalMDS
    if n_components is None:
        n_components = min(10, len(est.model_names_) - 1)
    n_components = min(n_components, len(est.model_names_) - 1)
    embeddings = ClassicalMDS(
        n_components=n_components, dissimilarity='precomputed',
    ).fit_transform(dist)

    predictions = predict_scores_knn(embeddings, scores, observed, k=k_neighbors)
    return evaluate_predictions(predictions, scores, observed)


def _run_estimators(data, scores, observed, experiment, k_neighbors, n_components, extra_fields):
    """Run all estimators on one dataset, return list of result rows."""
    rows = []
    for est_name, est_kwargs in ESTIMATORS.items():
        rmse = _fit_and_predict(
            data, scores, observed,
            k_neighbors=k_neighbors,
            n_components=n_components,
            **est_kwargs,
        )
        row = {'experiment': experiment, 'estimator': est_name, 'rmse': rmse}
        row.update(extra_fields)
        rows.append(row)
    return rows


_DEFAULTS = dict(
    d_latent=5, d_obs=20, score_noise=0.1, response_noise=0.5,
    task_spread=1.0, query_spread=0.5,
)


def _make_runner(experiment_name, sweep_param, default_gen_kwargs):
    """Factory for experiment functions that sweep a single parameter."""

    def run_experiment(k_neighbors=5, n_seeds=50, **kwargs):
        sweep_values = kwargs.pop(sweep_param)
        gen_kwargs = {**_DEFAULTS, **default_gen_kwargs, **kwargs}

        def _run(value, seed):
            gen_kwargs[sweep_param] = value
            n_components = gen_kwargs.get('d_latent', _DEFAULTS['d_latent'])
            data, scores, observed, _, _ = generate_benchmark_data(
                random_state=seed, **gen_kwargs,
            )
            return _run_estimators(
                data, scores, observed, experiment_name, k_neighbors,
                n_components,
                {'seed': seed, sweep_param: value},
            )

        params = [(v, s) for v in sweep_values for s in range(n_seeds)]
        nested = Parallel(n_jobs=-1, verbose=10)(
            delayed(_run)(v, s) for v, s in params)
        return pd.DataFrame([r for sub in nested for r in sub])

    run_experiment.__name__ = f'run_exp_{experiment_name}'
    run_experiment.__doc__ = f'Sweep {sweep_param} for experiment {experiment_name}.'
    return run_experiment


# The six experiments
run_exp_n_models = _make_runner('n_models', 'n_models', dict(
    n_tasks=20, n_queries_per_task=10, obs_prob=0.3, query_obs_prob=1.0))

run_exp_n_tasks = _make_runner('n_tasks', 'n_tasks', dict(
    n_models=100, n_queries_per_task=10, obs_prob=0.3, query_obs_prob=1.0))

run_exp_task_parity = _make_runner('task_parity', 'obs_prob', dict(
    n_models=100, n_tasks=20, n_queries_per_task=10, query_obs_prob=1.0))

run_exp_query_sparsity = _make_runner('query_sparsity', 'query_obs_prob', dict(
    n_models=100, n_tasks=20, n_queries_per_task=10, obs_prob=0.3))

run_exp_task_spread = _make_runner('task_spread', 'task_spread', dict(
    n_models=100, n_tasks=20, n_queries_per_task=10, obs_prob=0.3, query_obs_prob=1.0))


def run_exp_noise_x_queries(
        response_noises=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        n_queries_values=(3, 10, 50),
        n_models=100,
        n_tasks=20,
        obs_prob=0.3,
        query_obs_prob=1.0,
        k_neighbors=5,
        n_seeds=50,
        **gen_kwargs,
    ):
    """Interaction: response noise x queries per task (combined estimator only)."""
    kw = {**_DEFAULTS, **gen_kwargs}

    def _run(response_noise, n_queries_per_task, seed):
        n_components = kw.get('d_latent', _DEFAULTS['d_latent'])
        data, scores, observed, _, _ = generate_benchmark_data(
            n_models=n_models, n_tasks=n_tasks,
            n_queries_per_task=n_queries_per_task,
            obs_prob=obs_prob, query_obs_prob=query_obs_prob,
            response_noise=response_noise,
            random_state=seed, **{k: v for k, v in kw.items() if k != 'response_noise'},
        )
        rmse = _fit_and_predict(
            data, scores, observed,
            k_neighbors=k_neighbors, n_components=n_components,
            query_kernel='rbf',
        )
        return {
            'experiment': 'noise_x_queries',
            'estimator': f'nq={n_queries_per_task}',
            'seed': seed,
            'response_noise': response_noise,
            'n_queries_per_task': n_queries_per_task,
            'rmse': rmse,
        }

    params = [(rn, nq, s)
              for rn in response_noises
              for nq in n_queries_values
              for s in range(n_seeds)]
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(_run)(rn, nq, s) for rn, nq, s in params)
    return pd.DataFrame(results)
