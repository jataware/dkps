"""Tests for GromovWassersteinDistance (Method D)."""

import numpy as np
import pytest

from dkps.data import ModelResponseData
from dkps.distances.gromov_wasserstein import GromovWassersteinDistance


@pytest.fixture
def _check_pot():
    pytest.importorskip("ot", reason="POT package not installed")


@pytest.mark.usefixtures("_check_pot")
class TestGromovWassersteinDistance:
    def test_basic(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = GromovWassersteinDistance(reg=0.1)
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T, atol=1e-6)
        assert np.allclose(np.diag(D), 0, atol=1e-6)
        assert np.all(D >= -1e-6)

    def test_distance_matrix_properties(self, rng):
        data = {
            'a': rng.randn(30, 8),
            'b': rng.randn(30, 8) + 3,
            'c': rng.randn(30, 8) + 6,
        }
        mrd = ModelResponseData.from_dict(data)
        dist_fn = GromovWassersteinDistance(reg=0.1)
        D = dist_fn(mrd)
        assert np.all(np.isfinite(D))
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_rotation_invariance(self, rng):
        """GW distance should be invariant to orthogonal transforms."""
        X = rng.randn(30, 5)
        Y = rng.randn(30, 5) + 2

        # Random orthogonal matrix
        Q, _ = np.linalg.qr(rng.randn(5, 5))

        data1 = {'a': X, 'b': Y}
        data2 = {'a': X @ Q, 'b': Y @ Q}

        mrd1 = ModelResponseData.from_dict(data1)
        mrd2 = ModelResponseData.from_dict(data2)

        dist_fn = GromovWassersteinDistance(reg=0.1)
        D1 = dist_fn(mrd1)
        D2 = dist_fn(mrd2)

        np.testing.assert_allclose(D1, D2, atol=0.1)

    def test_variable_sample_sizes(self, rng):
        data = {
            'a': rng.randn(20, 5),
            'b': rng.randn(40, 5) + 3,
            'c': rng.randn(60, 5) + 6,
        }
        mrd = ModelResponseData.from_dict(data)
        dist_fn = GromovWassersteinDistance(reg=0.1)
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.all(np.isfinite(D))
