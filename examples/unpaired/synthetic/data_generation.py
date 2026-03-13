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

    Returns
    -------
    projections : list of np.ndarray, each (p, p)
    """
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


def sample_responses(mus, projections, m_total, alpha, seed,
                     query_distribution_kl=0.0, noise_scale=0.1):
    """
    Sample model responses for a mix of shared and private queries.

    Parameters
    ----------
    mus : np.ndarray (n_models, p)
    projections : list of projection matrices
    m_total : int
        Total queries per model.
    alpha : float
        Fraction of shared queries.
    seed : int or None
    query_distribution_kl : float
        Controls query distribution mismatch. 0 = all models draw private
        queries from same pool. Higher values shift the private query pool
        per model.
    noise_scale : float
        Std dev of response noise.

    Returns
    -------
    df : pd.DataFrame with columns (model_id, query_id, embedding)
    """
    rng = np.random.default_rng(seed)
    n_models, p = mus.shape
    n_total_projections = len(projections)

    m_shared = int(round(alpha * m_total))
    m_private = m_total - m_shared

    # Select shared query indices
    shared_indices = rng.choice(n_total_projections, size=min(m_shared, n_total_projections), replace=False)

    rows = []
    for i in range(n_models):
        # Shared queries
        for idx in shared_indices:
            P = projections[idx]
            response = P @ mus[i] + rng.normal(0, noise_scale, p)
            rows.append({
                'model_id': f'model_{i}',
                'query_id': f'shared_{idx}',
                'embedding': response,
            })

        # Private queries
        if m_private > 0:
            if query_distribution_kl == 0.0:
                # All models draw from the remaining pool
                remaining = [j for j in range(n_total_projections) if j not in shared_indices]
                if len(remaining) == 0:
                    remaining = list(range(n_total_projections))
                priv_indices = rng.choice(remaining, size=m_private, replace=True)
            elif np.isinf(query_distribution_kl):
                # Maximum mismatch: each model gets entirely different queries
                # Shift base index by model, wrapping around
                base = (i * m_private) % n_total_projections
                priv_indices = [(base + j) % n_total_projections for j in range(m_private)]
            else:
                # Intermediate mismatch: shift the distribution per model
                # Higher KL = more shift between models
                remaining = [j for j in range(n_total_projections) if j not in shared_indices]
                if len(remaining) == 0:
                    remaining = list(range(n_total_projections))
                n_rem = len(remaining)
                # Create per-model weights over remaining queries
                shift = query_distribution_kl * i / n_models
                weights = np.array([
                    np.exp(-query_distribution_kl * ((j / n_rem - shift) % 1.0))
                    for j in range(n_rem)
                ])
                weights /= weights.sum()
                priv_indices = rng.choice(remaining, size=m_private, replace=True, p=weights)

            for j, idx in enumerate(priv_indices):
                P = projections[idx]
                response = P @ mus[i] + rng.normal(0, noise_scale, p)
                rows.append({
                    'model_id': f'model_{i}',
                    'query_id': f'private_{i}_{j}',
                    'embedding': response,
                })

    return pd.DataFrame(rows)
