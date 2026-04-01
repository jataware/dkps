import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def generate_synthetic_data(
        d_act,
        d_obs,
        n_models,
        n_queries,
        alpha,
        s,
        t,
        d_sep=2.0,
        pi_paired=None,
        pi_unpaired=None,
        random_state=None,
        return_metadata=False,
    ):
    """
    Generate the synthetic data described in Block I of `unpaired_dkps_iclr_plan.md`.

    Returns
    -------
    data : pd.DataFrame
        Columns include `model_id`, `query_id`, `is_paired`, `query_vec`, and `embedding`.
    dist_gt : np.ndarray
        Ground-truth pairwise distance matrix between model offsets.
    metadata : dict, optional
        Returned only when `return_metadata=True`.
    """
    assert d_act > 0, 'd_act must be positive'
    assert d_obs >= d_act, 'd_obs must be at least d_act'
    assert n_models >= 2, 'n_models must be at least 2'
    assert n_queries >= 1, 'n_queries must be positive'
    assert 0.0 <= alpha <= 1.0, 'alpha must be in [0, 1]'
    assert s > 0, 's must be positive'
    assert t > 0, 't must be positive'

    rng = np.random.default_rng(random_state)
    model_names = [f'model_{i:02d}' for i in range(n_models)]

    if pi_paired is None:
        pi_paired = np.array([1.0, 0.0], dtype=float)
    else:
        pi_paired = np.asarray(pi_paired, dtype=float)

    if pi_unpaired is None:
        pi_unpaired = np.array([0.0, 1.0], dtype=float)
    else:
        pi_unpaired = np.asarray(pi_unpaired, dtype=float)

    assert pi_paired.shape == (2,), 'pi_paired must have shape (2,)'
    assert pi_unpaired.shape == (2,), 'pi_unpaired must have shape (2,)'
    assert np.isclose(pi_paired.sum(), 1.0), 'pi_paired must sum to 1'
    assert np.isclose(pi_unpaired.sum(), 1.0), 'pi_unpaired must sum to 1'
    assert np.all(pi_paired >= 0), 'pi_paired must be non-negative'
    assert np.all(pi_unpaired >= 0), 'pi_unpaired must be non-negative'

    mu_components = np.array([
        [d_sep / 2.0] + [0.0] * (d_act - 1),
        [-d_sep / 2.0] + [0.0] * (d_act - 1),
    ])

    model_offsets = rng.normal(0.0, 1.0, size=(n_models, d_act))

    n_paired = int(round(alpha * n_queries))
    n_paired = min(max(n_paired, 0), n_queries)
    n_unpaired = n_queries - n_paired

    rows = []

    for paired_idx in range(n_paired):
        component = rng.choice(2, p=pi_paired)
        query_vec = rng.multivariate_normal(mu_components[component], np.eye(d_act))
        query_id = f'paired_{paired_idx:05d}'

        for model_idx, model_name in enumerate(model_names):
            latent_mean = rng.normal(query_vec + model_offsets[model_idx], 1.0 / s, size=d_act)
            active_response = rng.normal(latent_mean, 1.0 / t, size=d_act)
            observation_noise = rng.normal(0.0, 1.0, size=d_obs - d_act)
            embedding = np.hstack([active_response, observation_noise])

            rows.append({
                'model_id': model_name,
                'model_idx': model_idx,
                'query_id': query_id,
                'is_paired': True,
                'query_component': int(component),
                'query_vec': query_vec,
                'latent_mean_vec': latent_mean,
                'active_response_vec': active_response,
                'embedding': embedding,
            })

    for model_idx, model_name in enumerate(model_names):
        for unpaired_idx in range(n_unpaired):
            component = rng.choice(2, p=pi_unpaired)
            query_vec = rng.multivariate_normal(mu_components[component], np.eye(d_act))
            query_id = f'{model_name}_unpaired_{unpaired_idx:05d}'

            latent_mean = rng.normal(query_vec + model_offsets[model_idx], 1.0 / s, size=d_act)
            active_response = rng.normal(latent_mean, 1.0 / t, size=d_act)
            observation_noise = rng.normal(0.0, 1.0, size=d_obs - d_act)
            embedding = np.hstack([active_response, observation_noise])

            rows.append({
                'model_id': model_name,
                'model_idx': model_idx,
                'query_id': query_id,
                'is_paired': False,
                'query_component': int(component),
                'query_vec': query_vec,
                'latent_mean_vec': latent_mean,
                'active_response_vec': active_response,
                'embedding': embedding,
            })

    data = pd.DataFrame(rows)
    data = data.sort_values(['model_id', 'query_id']).reset_index(drop=True)
    dist_gt = squareform(pdist(model_offsets))

    metadata = {
        'd_act': d_act,
        'd_obs': d_obs,
        'n_models': n_models,
        'n_queries': n_queries,
        'alpha_requested': alpha,
        'alpha_actual': n_paired / n_queries,
        'n_paired': n_paired,
        'n_unpaired': n_unpaired,
        's': s,
        't': t,
        'd_sep': d_sep,
        'pi_paired': pi_paired,
        'pi_unpaired': pi_unpaired,
        'model_offsets': model_offsets,
        'model_names': model_names,
        'mu_components': mu_components,
    }

    if return_metadata:
        return data, dist_gt, metadata
    return data, dist_gt
