"""
Synthetic data generation for unpaired DKPS experiments.

Models are defined by mean vectors in R^p. Queries are random orthogonal
projections selecting subspaces. Responses are the projected means plus noise.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist


def generate_synthetic_models(n_models=20, p=50, seed=None):
    """
    Generate synthetic model means and ground-truth distance matrix.

    Parameters
    ----------
    n_models : int
    p : int
        Embedding dimension.
    seed : int or None

    Returns
    -------
    mus : np.ndarray (n_models, p)
        Model mean vectors ~ N(0, I_p).
    ground_truth_distances : np.ndarray (n_models, n_models)
        D*(i,k) = ||mu_i - mu_k||.
    """
    rng = np.random.default_rng(seed)
    mus = rng.standard_normal((n_models, p))
    gt = squareform(pdist(mus, metric='euclidean'))
    return mus, gt


def generate_query_projections(n_queries, p, seed=None):
    """
    Generate random orthogonal projection matrices for queries.

    Each query selects a random subspace of R^p via an orthogonal projection.

    Parameters
    ----------
    n_queries : int
    p : int
    seed : int, None, or np.random.Generator

    Returns
    -------
    projections : list of np.ndarray, each (p, p)
    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    projections = []
    for _ in range(n_queries):
        # Random subspace dimension between p//4 and 3p//4
        d_sub = rng.integers(p // 4, 3 * p // 4 + 1)
        
        # Random orthonormal basis for subspace
        A = rng.standard_normal((p, d_sub))
        Q, _ = np.linalg.qr(A)
        
        # Projection matrix P = Q Q^T
        P = Q @ Q.T
        projections.append(P)
    
    return projections


def sample_responses(mus, m_total, alpha, seed, noise_scale=0.1):
    """
    Sample model responses for a mix of shared and private queries.

    Shared projections are the same for all models. Private projections are
    generated independently per model.

    Parameters
    ----------
    mus : np.ndarray (n_models, p)
    m_total : int
        Total queries per model.
    alpha : float
        Fraction of shared queries.
    seed : int or None
    noise_scale : float
        Std dev of response noise.

    Returns
    -------
    df : pd.DataFrame with columns (model_id, query_id, embedding)
    """
    rng = np.random.default_rng(seed)
    n_models, p = mus.shape

    m_shared = int(round(alpha * m_total))
    m_private = m_total - m_shared

    # Shared projections (same for all models)
    shared_projections = generate_query_projections(m_shared, p, seed=rng)

    # Private projections (fresh per model)
    private_projections = [[] for _ in range(n_models)]
    if m_private > 0:
        private_projections = [
            generate_query_projections(m_private, p, seed=rng)
            for _ in range(n_models)
        ]

    # Generate responses
    rows = []
    for i in range(n_models):
        model_id = f'model_{i}'
        for j, P in enumerate(shared_projections):
            response = P @ mus[i] + rng.normal(0, noise_scale, p)
            rows.append({'model_id': model_id, 'query_id': f'shared_{j:03d}', 'embedding': response})
        
        for j, P in enumerate(private_projections[i]):
            response = P @ mus[i] + rng.normal(0, noise_scale, p)
            rows.append({'model_id': model_id, 'query_id': f'private_{i:03d}_{j:03d}', 'embedding': response})

    return pd.DataFrame(rows)
