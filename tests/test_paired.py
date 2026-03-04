"""Tests for PairedDistance (Method 0)."""

import numpy as np
import pandas as pd
import pytest

from dkps.data import ModelResponseData
from dkps.distances.paired import PairedDistance


def _make_paired_df(rng, n_models=2, n_queries=10, n_replicates=1, embed_dim=8):
    """Helper to build a paired DataFrame with optional replicates."""
    rows = []
    for m in range(n_models):
        center = rng.randn(embed_dim) * (m + 1)
        for q in range(n_queries):
            for r in range(n_replicates):
                emb = center + rng.randn(embed_dim) * 0.1
                rows.append({
                    'model': f'model_{m}',
                    'query_id': f'q{q}',
                    'response_embedding': emb,
                })
    return pd.DataFrame(rows)


class TestPairedDistance:
    def test_basic(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        dist_fn = PairedDistance()
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0)
        assert np.all(D >= 0)

    def test_unpaired_raises(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = PairedDistance()
        with pytest.raises(ValueError, match="requires paired data"):
            dist_fn(mrd)

    def test_different_metrics(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        for metric in ['euclidean', 'cosine', 'cityblock']:
            dist_fn = PairedDistance(metric=metric)
            D = dist_fn(mrd)
            assert D.shape == (3, 3)
            assert np.all(np.isfinite(D))

    def test_distance_matrix_properties(self, cluster_models_df):
        mrd = ModelResponseData.from_dataframe(cluster_models_df)
        dist_fn = PairedDistance()
        D = dist_fn(mrd)
        # Symmetric
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        # Zero diagonal
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-12)
        # Non-negative
        assert np.all(D >= -1e-12)
        # No NaN/Inf
        assert np.all(np.isfinite(D))

    def test_with_mean_aggregation(self, rng):
        df = _make_paired_df(rng, n_models=2, n_queries=10, n_replicates=3, embed_dim=8)
        mrd = ModelResponseData.from_dataframe(df)
        dist_fn = PairedDistance(response_agg_fn=np.mean)
        D = dist_fn(mrd)
        assert D.shape == (2, 2)

    def test_translation_invariance(self, simple_paired_df):
        """Adding a constant to all embeddings shouldn't change distances."""
        mrd1 = ModelResponseData.from_dataframe(simple_paired_df)
        dist_fn = PairedDistance()
        D1 = dist_fn(mrd1)

        # Shift all embeddings
        df2 = simple_paired_df.copy()
        shift = np.ones(8) * 100
        df2['response_embedding'] = df2['response_embedding'].apply(lambda e: np.asarray(e) + shift)
        mrd2 = ModelResponseData.from_dataframe(df2)
        D2 = dist_fn(mrd2)

        np.testing.assert_allclose(D1, D2, atol=1e-10)
