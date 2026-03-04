"""Tests for MMDDistance (Method A)."""

import numpy as np
import pandas as pd
import pytest

from dkps.data import ModelResponseData
from dkps.distances.mmd import MMDDistance


def _make_unpaired_df(embeddings_dict):
    """Build unpaired DataFrame from {model: (n, d)} arrays."""
    rows = []
    for model, embs in embeddings_dict.items():
        for i in range(len(embs)):
            rows.append({'model': model, 'response_embedding': embs[i]})
    return pd.DataFrame(rows)


class TestMMDDistance:
    def test_basic_unpaired(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = MMDDistance(kernel='rbf')
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0)
        assert np.all(D >= 0)

    def test_basic_paired_treated_as_unpaired(self, simple_paired_df):
        """MMD works on paired data by flattening."""
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        dist_fn = MMDDistance(kernel='rbf')
        D = dist_fn(mrd)
        assert D.shape == (3, 3)

    def test_linear_kernel(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = MMDDistance(kernel='linear')
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.all(np.isfinite(D))

    def test_linear_mmd_mean_difference(self, rng):
        """For linear kernel, MMD = ||mean(X) - mean(Y)||."""
        mean_a = np.zeros(10)
        mean_b = np.ones(10) * 3.0
        X = mean_a + rng.randn(100, 10) * 0.01  # tight cluster
        Y = mean_b + rng.randn(100, 10) * 0.01

        df = _make_unpaired_df({'a': X, 'b': Y})
        mrd = ModelResponseData.from_dataframe(df)
        dist_fn = MMDDistance(kernel='linear')
        D = dist_fn(mrd)

        expected = np.linalg.norm(X.mean(0) - Y.mean(0))
        np.testing.assert_allclose(D[0, 1], expected, rtol=0.01)

    def test_distance_matrix_properties(self, cluster_models_df):
        mrd = ModelResponseData.from_dataframe(cluster_models_df)
        dist_fn = MMDDistance()
        D = dist_fn(mrd)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-12)
        assert np.all(D >= -1e-12)
        assert np.all(np.isfinite(D))

    def test_variable_sample_sizes(self, unpaired_models_df):
        mrd = ModelResponseData.from_dataframe(unpaired_models_df)
        dist_fn = MMDDistance()
        D = dist_fn(mrd)
        assert D.shape == (15, 15)
        assert np.all(np.isfinite(D))

    def test_custom_bandwidth(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = MMDDistance(kernel='rbf', bandwidth=1.0)
        D = dist_fn(mrd)
        assert D.shape == (3, 3)

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="kernel must be"):
            MMDDistance(kernel='polynomial')
