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
        interaction_strength=0.0,
        n_replicates=1,
        pi_paired=None,
        pi_unpaired=None,
        random_state=None,
        return_metadata=False,
    ):
    """
    Generate the synthetic data described in Block I of `unpaired_dkps_iclr_plan.md`.

    The synthetic construction has three layers:

    1. Each model `i` gets a latent offset vector `v_i` in `R^{d_act}`.
    2. Each query draws a latent query vector `q` from one of two Gaussian
       components separated by `d_sep`.
    3. The observed response embedding is built as

           x_ij = [response_ij, observation_noise_ij]

       where the active response dimensions are centered at `q + v_i` up to the
       `s` and `t` noise scales, and any extra observed dimensions
       `d_obs - d_act` are pure noise.

    Under this setup, the ground-truth model distance matrix is the pairwise
    Euclidean distance between the latent model offsets `v_i`.

    Parameters
    ----------
    d_act : int
        Number of active dimensions in the latent model/query interaction.
        Query vectors and model offsets both live in this space.
    d_obs : int
        Observed embedding dimension. Must satisfy `d_obs >= d_act`.
        When `d_obs > d_act`, the remaining dimensions are independent noise
        dimensions that do not carry model signal.
    n_models : int
        Number of models to simulate.
    n_queries : int
        Total number of queries observed per model, counting both paired and
        unpaired queries.
    alpha : float
        Requested paired-query fraction in `[0, 1]`. The generator creates
        `round(alpha * n_queries)` globally shared query ids and uses model-
        specific query ids for the rest.
    s : float
        Latent-response noise scale. The intermediate latent mean is drawn as

            latent_mean ~ N(query_vec + model_offset, (1 / s)^2 I).

        Larger `s` means less latent noise; `s = np.inf` recovers the noiseless
        latent mean `query_vec + model_offset`.
    t : float
        Observation noise scale on the active response dimensions:

            active_response ~ N(latent_mean, (1 / t)^2 I).

        Larger `t` means less observation noise; `t = np.inf` makes the active
        response equal to the latent mean.
    d_sep : float, default=2.0
        Separation between the two query-distribution components along the first
        active dimension. Larger values make paired and unpaired query regimes
        more distributionally distinct when `pi_paired` and `pi_unpaired` differ.
    n_replicates : int, default=1
        Number of independent response replicates per (model, query) pair.
        Each replicate shares the same query vector and model offset but draws
        independent latent-mean and observation noise.  When > 1, each row in
        the returned DataFrame carries a `replicate_id` column.
    pi_paired : array-like of shape (2,), optional
        Mixture weights over the two query components for globally shared
        queries. Defaults to `[1.0, 0.0]`, meaning paired queries come entirely
        from the first component.
    pi_unpaired : array-like of shape (2,), optional
        Mixture weights over the two query components for model-specific
        unpaired queries. Defaults to `[0.0, 1.0]`, meaning unpaired queries
        come entirely from the second component.
    random_state : int or numpy.random.Generator-compatible seed, optional
        Seed used for reproducible synthetic draws.
    return_metadata : bool, default=False
        If `True`, also return a metadata dictionary containing realized paired
        counts, model offsets, query-component means, and other generator state
        useful for analysis.

    Returns
    -------
    data : pd.DataFrame
        Long-form response table with one row per observed `(model, query)`
        pair. Columns include:

        - `model_id`: model name
        - `query_id`: query identifier; shared across models only for paired queries
        - `is_paired`: whether the row comes from the shared-query subset
        - `query_component`: which of the two query-mixture components was used
        - `query_vec`: latent query vector in `R^{d_act}`
        - `latent_mean_vec`: latent mean before observation noise
        - `active_response_vec`: noisy active response in `R^{d_act}`
        - `embedding`: final observed response in `R^{d_obs}`
    dist_gt : np.ndarray
        Ground-truth pairwise distance matrix between latent model offsets.
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
    assert n_replicates >= 1, 'n_replicates must be at least 1'
    assert interaction_strength >= 0, 'interaction_strength must be non-negative'

    rng = np.random.default_rng(random_state)
    model_names = [f'model_{i:02d}' for i in range(n_models)]

    if pi_paired is not None:
        pi_paired = np.asarray(pi_paired, dtype=float)
        assert pi_paired.shape == (2,), 'pi_paired must have shape (2,)'
        assert np.isclose(pi_paired.sum(), 1.0), 'pi_paired must sum to 1'
        assert np.all(pi_paired >= 0), 'pi_paired must be non-negative'

    if pi_unpaired is not None:
        pi_unpaired = np.asarray(pi_unpaired, dtype=float)
        assert pi_unpaired.shape == (2,), 'pi_unpaired must have shape (2,)'
        assert np.isclose(pi_unpaired.sum(), 1.0), 'pi_unpaired must sum to 1'
        assert np.all(pi_unpaired >= 0), 'pi_unpaired must be non-negative'

    mu_components = np.array([
        [d_sep / 2.0] + [0.0] * (d_act - 1),
        [-d_sep / 2.0] + [0.0] * (d_act - 1),
    ])

    model_offsets = rng.normal(0.0, 1.0, size=(n_models, d_act))

    n_paired = int(round(alpha * n_queries))
    n_paired = min(max(n_paired, 0), n_queries)
    n_unpaired = n_queries - n_paired

    def _sample_components(rng, n, pi=None):
        """Sample component assignments for n queries.
        If pi is None, draw a random mixture proportion p ~ U(0,1)
        and assign floor(p*n) to component 0, rest to component 1."""
        if pi is not None:
            return rng.choice(2, size=n, p=pi)
        p = rng.uniform(0.0, 1.0)
        n0 = int(round(p * n))
        n0 = min(max(n0, 0), n)
        components = np.zeros(n, dtype=int)
        components[n0:] = 1
        rng.shuffle(components)
        return components

    rows = []

    # Paired queries: random mixture proportion for the shared set
    paired_components = _sample_components(rng, n_paired, pi_paired)
    for paired_idx in range(n_paired):
        component = paired_components[paired_idx]
        query_vec = rng.multivariate_normal(mu_components[component], np.eye(d_act))
        query_id = f'paired_{paired_idx:05d}'

        for model_idx, model_name in enumerate(model_names):
            for rep in range(n_replicates):
                signal = query_vec + model_offsets[model_idx] + interaction_strength * model_offsets[model_idx] * query_vec
                latent_mean = rng.normal(signal, 1.0 / s, size=d_act)
                active_response = rng.normal(latent_mean, 1.0 / t, size=d_act)
                observation_noise = rng.normal(0.0, 1.0, size=d_obs - d_act)
                embedding = np.hstack([active_response, observation_noise])

                row = {
                    'model_id': model_name,
                    'model_idx': model_idx,
                    'query_id': query_id,
                    'is_paired': True,
                    'query_component': int(component),
                    'query_vec': query_vec,
                    'latent_mean_vec': latent_mean,
                    'active_response_vec': active_response,
                    'embedding': embedding,
                }
                if n_replicates > 1:
                    row['replicate_id'] = rep
                rows.append(row)

    # Unpaired queries: each model draws its own random mixture proportion
    for model_idx, model_name in enumerate(model_names):
        model_components = _sample_components(rng, n_unpaired, pi_unpaired)
        for unpaired_idx in range(n_unpaired):
            component = model_components[unpaired_idx]
            query_vec = rng.multivariate_normal(mu_components[component], np.eye(d_act))
            query_id = f'{model_name}_unpaired_{unpaired_idx:05d}'

            for rep in range(n_replicates):
                signal = query_vec + model_offsets[model_idx] + interaction_strength * model_offsets[model_idx] * query_vec
                latent_mean = rng.normal(signal, 1.0 / s, size=d_act)
                active_response = rng.normal(latent_mean, 1.0 / t, size=d_act)
                observation_noise = rng.normal(0.0, 1.0, size=d_obs - d_act)
                embedding = np.hstack([active_response, observation_noise])

                row = {
                    'model_id': model_name,
                    'model_idx': model_idx,
                    'query_id': query_id,
                    'is_paired': False,
                    'query_component': int(component),
                    'query_vec': query_vec,
                    'latent_mean_vec': latent_mean,
                    'active_response_vec': active_response,
                    'embedding': embedding,
                }
                if n_replicates > 1:
                    row['replicate_id'] = rep
                rows.append(row)

    data = pd.DataFrame(rows)
    data = data.sort_values(['model_id', 'query_id']).reset_index(drop=True)

    # Ground truth: ||mu_i - mu_k|| where mu_i = E_q[f_i(q)] under the
    # uniform reference distribution over both components (p=0.5).
    # With interaction: mu_i = E[q](1 + gamma*v_i) + v_i.
    # E[q] under uniform mixture = 0.5*mu_1 + 0.5*mu_2 = 0 (symmetric).
    eq_ref = 0.5 * mu_components[0] + 0.5 * mu_components[1]
    mean_responses = np.array([
        eq_ref + v + interaction_strength * v * eq_ref
        for v in model_offsets
    ])
    dist_gt = squareform(pdist(mean_responses))

    metadata = {
        'd_act': d_act,
        'd_obs': d_obs,
        'n_models': n_models,
        'n_queries': n_queries,
        'n_replicates': n_replicates,
        'alpha_requested': alpha,
        'alpha_actual': n_paired / n_queries,
        'n_paired': n_paired,
        'n_unpaired': n_unpaired,
        'interaction_strength': interaction_strength,
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
