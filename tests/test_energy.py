"""Tests for EnergyDistance (Method C)."""

import numpy as np
import pandas as pd
import pytest

from dkps.data import ModelResponseData
from dkps.distances.energy import EnergyDistance


def _make_unpaired_df(embeddings_dict):
    """Build unpaired DataFrame from {model: (n, d)} arrays."""
    rows = []
    for model, embs in embeddings_dict.items():
        for i in range(len(embs)):
            rows.append({'model': model, 'response_embedding': embs[i]})
    return pd.DataFrame(rows)


class TestEnergyDistance:
    def test_basic_unpaired(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = EnergyDistance()
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0)
        assert np.all(D >= 0)

    def test_basic_paired(self, simple_paired_df):
        """Energy works on paired data by flattening."""
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        dist_fn = EnergyDistance()
        D = dist_fn(mrd)
        assert D.shape == (3, 3)

    def test_distance_matrix_properties(self, cluster_models_df):
        mrd = ModelResponseData.from_dataframe(cluster_models_df)
        dist_fn = EnergyDistance()
        D = dist_fn(mrd)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-12)
        assert np.all(D >= -1e-12)
        assert np.all(np.isfinite(D))

    def test_variable_sample_sizes(self, unpaired_models_df):
        mrd = ModelResponseData.from_dataframe(unpaired_models_df)
        dist_fn = EnergyDistance()
        D = dist_fn(mrd)
        assert D.shape == (15, 15)
        assert np.all(np.isfinite(D))

    def test_identical_distributions_zero(self, rng):
        """Energy distance between identical distributions should be ~0."""
        X = rng.randn(100, 10)
        df = _make_unpaired_df({'a': X.copy(), 'b': X.copy()})
        mrd = ModelResponseData.from_dataframe(df)
        dist_fn = EnergyDistance()
        D = dist_fn(mrd)
        assert D[0, 1] < 1e-10

    def test_distant_distributions(self, rng):
        """Far apart distributions should have large energy distance."""
        X = rng.randn(100, 10)
        Y = rng.randn(100, 10) + 100
        df = _make_unpaired_df({'a': X, 'b': Y})
        mrd = ModelResponseData.from_dataframe(df)
        dist_fn = EnergyDistance()
        D = dist_fn(mrd)
        assert D[0, 1] > 1.0

    def test_translation_invariance(self, rng):
        """Energy distance should be translation-invariant."""
        X = rng.randn(50, 10)
        Y = rng.randn(50, 10) + 5
        df1 = _make_unpaired_df({'a': X, 'b': Y})
        mrd1 = ModelResponseData.from_dataframe(df1)

        shift = np.ones(10) * 100
        df2 = _make_unpaired_df({'a': X + shift, 'b': Y + shift})
        mrd2 = ModelResponseData.from_dataframe(df2)

        dist_fn = EnergyDistance()
        D1 = dist_fn(mrd1)
        D2 = dist_fn(mrd2)
        np.testing.assert_allclose(D1, D2, atol=1e-10)
