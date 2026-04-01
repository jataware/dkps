import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def generate_data(
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
    seed=None,
):
    """
    Generate synthetic data for unpaired DKPS experiments.

    Parameters
    ----------
    d_act       : int    — dimensionality of active (signal) subspace
    d_obs       : int    — total observed dimensionality (d_act + noise dims)
    n_models    : int    — number of models
    n_queries   : int    — total query budget per model
    alpha       : float  — fraction of queries that are paired (shared across models)
    s           : float  — query-response coupling (inverse std of latent mean)
    t           : float  — observation noise (inverse std around latent mean)
    d_sep       : float  — separation between the two mixture component means
    pi_paired   : array  — mixture weights for paired queries; default [1, 0]
    pi_unpaired : array  — mixture weights for unpaired queries; default [0, 1]
    seed        : int    — random seed

    Returns
    -------
    data : pd.DataFrame
        Columns: model_id, query_id, embedding, query_vec
    dist_gt : np.ndarray, shape (n_models, n_models)
        Ground truth pairwise distance matrix ||v_i - v_k||
    model_offsets : np.ndarray, shape (n_models, d_act)
        True model offset vectors
    """
    rng = np.random.default_rng(seed)

    assert d_obs >= d_act, 'd_obs must be >= d_act'
    assert 0.0 <= alpha <= 1.0, 'alpha must be in [0, 1]'

    # Mixture component means separated along first axis
    mu_c = np.zeros((2, d_act))
    mu_c[0, 0] = d_sep / 2
    mu_c[1, 0] = -d_sep / 2

    if pi_paired is None:
        pi_paired = np.array([1.0, 0.0])
    else:
        pi_paired = np.asarray(pi_paired, dtype=float)

    if pi_unpaired is None:
        pi_unpaired = np.array([0.0, 1.0])
    else:
        pi_unpaired = np.asarray(pi_unpaired, dtype=float)

    # Model offsets — drawn once, fixed across queries
    # v_i ~ N(0, I_{d_act})
    model_offsets = rng.standard_normal((n_models, d_act))

    n_paired = int(alpha * n_queries)
    n_unpaired = n_queries - n_paired

    rows = []

    def _sample_query(pi):
        # q_j ~ pi_1 * N(mu_1, I_{d_act}) + pi_2 * N(mu_2, I_{d_act})
        c = rng.choice(2, p=pi)
        return rng.standard_normal(d_act) + mu_c[c]

    def _sample_response(query_vec, model_idx):
        mu_ij = rng.normal(query_vec + model_offsets[model_idx], 1.0 / s)
        r_ij = rng.normal(mu_ij, 1.0 / t)
        noise = rng.standard_normal(d_obs - d_act)
        return np.concatenate([r_ij, noise])

    # Paired queries: same query for all models
    for j in range(n_paired):
        query_vec = _sample_query(pi_paired)
        query_id = f"p_{j}"
        for model_idx in range(n_models):
            rows.append({
                "model_id":  f"model_{model_idx:02d}",
                "query_id":  query_id,
                "embedding": _sample_response(query_vec, model_idx),
                "query_vec": query_vec,
            })

    # Unpaired queries: each model gets its own independent query draw
    for model_idx in range(n_models):
        for j in range(n_unpaired):
            query_vec = _sample_query(pi_unpaired)
            query_id = f"u_{model_idx}_{j}"
            rows.append({
                "model_id":  f"model_{model_idx:02d}",
                "query_id":  query_id,
                "embedding": _sample_response(query_vec, model_idx),
                "query_vec": query_vec,
            })

    data = pd.DataFrame(rows)
    dist_gt = squareform(pdist(model_offsets))

    return data, dist_gt, model_offsets
