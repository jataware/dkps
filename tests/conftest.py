"""
    conftest.py — Synthetic data fixtures for DKPS tests.

    All fixtures use pre-computed embeddings (no text, no API calls).
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def cluster_models_df(rng):
    """3 clusters x 5 models, Gaussian blobs in R^20, 50 paired queries.

    Ground truth: models within same cluster are closer.
    Returns DataFrame with columns: model, query_id, response_embedding.
    Also attaches .attrs['cluster_labels'] = {model: cluster_idx}.
    """
    n_clusters = 3
    models_per_cluster = 5
    n_queries = 50
    embed_dim = 20
    cluster_sep = 5.0

    # Cluster centers
    centers = rng.randn(n_clusters, embed_dim) * cluster_sep

    rows = []
    cluster_labels = {}
    model_idx = 0
    for c in range(n_clusters):
        for _ in range(models_per_cluster):
            model_name = f"model_{model_idx}"
            cluster_labels[model_name] = c
            # Model offset within cluster
            model_offset = centers[c] + rng.randn(embed_dim) * 0.5
            for q in range(n_queries):
                # Response = model offset + query-specific noise
                emb = model_offset + rng.randn(embed_dim) * 0.3
                rows.append({
                    'model': model_name,
                    'query_id': f'q{q}',
                    'response_embedding': emb,
                })
            model_idx += 1

    df = pd.DataFrame(rows)
    df.attrs['cluster_labels'] = cluster_labels
    df.attrs['n_clusters'] = n_clusters
    return df


@pytest.fixture
def manifold_models_df(rng):
    """20 models on a smooth 1-D curve in R^20, parameterized by theta in [0,1].

    Ground truth: theta ordering should be recovered.
    Returns DataFrame with columns: model, query_id, response_embedding.
    Also attaches .attrs['theta'] = {model: theta_value}.
    """
    n_models = 20
    n_queries = 50
    embed_dim = 20

    # Random direction for the manifold
    direction = rng.randn(embed_dim)
    direction = direction / np.linalg.norm(direction)

    # Secondary direction for curvature
    perp = rng.randn(embed_dim)
    perp = perp - np.dot(perp, direction) * direction
    perp = perp / np.linalg.norm(perp)

    thetas = {}
    rows = []
    for i in range(n_models):
        theta = i / (n_models - 1)
        model_name = f"model_{i}"
        thetas[model_name] = theta

        # Position on curve: linear + sine for curvature
        center = theta * direction * 10 + np.sin(theta * np.pi) * perp * 3

        for q in range(n_queries):
            emb = center + rng.randn(embed_dim) * 0.2
            rows.append({
                'model': model_name,
                'query_id': f'q{q}',
                'response_embedding': emb,
            })

    df = pd.DataFrame(rows)
    df.attrs['theta'] = thetas
    return df


@pytest.fixture
def unpaired_models_df(rng):
    """Same cluster structure as cluster_models_df but no query_id,
    varying n_responses per model (50-200).

    Returns DataFrame with columns: model, response_embedding.
    Also attaches .attrs['cluster_labels'].
    """
    n_clusters = 3
    models_per_cluster = 5
    embed_dim = 20
    cluster_sep = 5.0

    centers = rng.randn(n_clusters, embed_dim) * cluster_sep

    rows = []
    cluster_labels = {}
    model_idx = 0
    for c in range(n_clusters):
        for _ in range(models_per_cluster):
            model_name = f"model_{model_idx}"
            cluster_labels[model_name] = c
            model_offset = centers[c] + rng.randn(embed_dim) * 0.5
            n_responses = rng.randint(50, 201)
            for _ in range(n_responses):
                emb = model_offset + rng.randn(embed_dim) * 0.3
                rows.append({
                    'model': model_name,
                    'response_embedding': emb,
                })
            model_idx += 1

    df = pd.DataFrame(rows)
    df.attrs['cluster_labels'] = cluster_labels
    df.attrs['n_clusters'] = n_clusters
    return df


@pytest.fixture
def simple_paired_df(rng):
    """Minimal paired DataFrame: 3 models, 10 queries."""
    n_models = 3
    n_queries = 10
    embed_dim = 8

    rows = []
    for m in range(n_models):
        center = rng.randn(embed_dim) * (m + 1)
        for q in range(n_queries):
            emb = center + rng.randn(embed_dim) * 0.1
            rows.append({
                'model': f'model_{m}',
                'query_id': f'q{q}',
                'response_embedding': emb,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def simple_unpaired_df(rng):
    """Minimal unpaired DataFrame: 3 models, varying samples."""
    embed_dim = 8
    rows = []
    for m in range(3):
        center = rng.randn(embed_dim) * (m + 1)
        n = 20 + m * 10
        for _ in range(n):
            emb = center + rng.randn(embed_dim) * 0.1
            rows.append({
                'model': f'model_{m}',
                'response_embedding': emb,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def query_paired_df(rng):
    """DataFrame with query embeddings for soft-paired tests."""
    n_models = 3
    n_queries = 10
    embed_dim = 8
    query_dim = 4

    rows = []
    query_embs = {f'q{q}': rng.randn(query_dim) for q in range(n_queries)}

    for m in range(n_models):
        center = rng.randn(embed_dim) * (m + 1)
        for q in range(n_queries):
            emb = center + rng.randn(embed_dim) * 0.1
            rows.append({
                'model': f'model_{m}',
                'query_id': f'q{q}',
                'response_embedding': emb,
                'query_embedding': query_embs[f'q{q}'],
            })
    return pd.DataFrame(rows)
