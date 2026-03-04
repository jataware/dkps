"""Tests for WassersteinDistance (Method B)."""

import numpy as np
import pandas as pd
import pytest

from dkps.data import ModelResponseData
from dkps.distances.wasserstein import WassersteinDistance


def _make_unpaired_df(embeddings_dict):
    """Build unpaired DataFrame from {model: (n, d)} arrays."""
    rows = []
    for model, embs in embeddings_dict.items():
        for i in range(len(embs)):
            rows.append({'model': model, 'response_embedding': embs[i]})
    return pd.DataFrame(rows)


@pytest.fixture
def _check_pot():
    pytest.importorskip("ot", reason="POT package not installed")


@pytest.mark.usefixtures("_check_pot")
class TestWassersteinDistance:
    def test_sliced(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = WassersteinDistance(variant='sliced')
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0)
        assert np.all(D >= 0)

    def test_exact(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = WassersteinDistance(variant='exact')
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.all(np.isfinite(D))

    def test_sinkhorn(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = WassersteinDistance(variant='sinkhorn', reg=0.1)
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.all(np.isfinite(D))

    def test_distance_matrix_properties(self, cluster_models_df):
        mrd = ModelResponseData.from_dataframe(cluster_models_df)
        dist_fn = WassersteinDistance(variant='sliced')
        D = dist_fn(mrd)
        np.testing.assert_allclose(D, D.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-10)
        assert np.all(D >= -1e-10)
        assert np.all(np.isfinite(D))

    def test_variable_sample_sizes(self, unpaired_models_df):
        mrd = ModelResponseData.from_dataframe(unpaired_models_df)
        dist_fn = WassersteinDistance(variant='sliced')
        D = dist_fn(mrd)
        assert D.shape == (15, 15)
        assert np.all(np.isfinite(D))

    def test_identical_distributions_zero(self, rng):
        X = rng.randn(50, 5)
        df = _make_unpaired_df({'a': X.copy(), 'b': X.copy()})
        mrd = ModelResponseData.from_dataframe(df)
        dist_fn = WassersteinDistance(variant='exact')
        D = dist_fn(mrd)
        assert D[0, 1] < 1e-8

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="variant must be"):
            WassersteinDistance(variant='invalid')
