"""Tests for SoftPairedOTDistance (Method E)."""

import numpy as np
import pytest

from dkps.data import ModelResponseData
from dkps.distances.soft_paired import SoftPairedOTDistance


@pytest.fixture
def _check_pot():
    pytest.importorskip("ot", reason="POT package not installed")


@pytest.mark.usefixtures("_check_pot")
class TestSoftPairedOTDistance:
    def test_basic(self, query_paired_df):
        mrd = ModelResponseData.from_dataframe(query_paired_df)
        dist_fn = SoftPairedOTDistance(reg=0.1)
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T, atol=1e-6)
        assert np.allclose(np.diag(D), 0, atol=1e-6)

    def test_no_query_embeddings_raises(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        dist_fn = SoftPairedOTDistance()
        with pytest.raises(ValueError, match="requires query information"):
            dist_fn(mrd)

    def test_distance_matrix_properties(self, query_paired_df):
        mrd = ModelResponseData.from_dataframe(query_paired_df)
        dist_fn = SoftPairedOTDistance(reg=0.1)
        D = dist_fn(mrd)
        assert np.all(np.isfinite(D))
        assert np.all(D >= -1e-6)

    def test_query_weight_effect(self, query_paired_df):
        """Different query weights should produce different distances."""
        mrd = ModelResponseData.from_dataframe(query_paired_df)
        D1 = SoftPairedOTDistance(reg=0.1, query_weight=0.1)(mrd)
        D2 = SoftPairedOTDistance(reg=0.1, query_weight=0.9)(mrd)
        # They should differ (not guaranteed but likely with random data)
        assert not np.allclose(D1, D2, atol=1e-6)
