"""Tests for HybridDistance (Method F)."""

import numpy as np
import pytest

from dkps.data import ModelResponseData
from dkps.distances.hybrid import HybridDistance
from dkps.distances.paired import PairedDistance
from dkps.distances.energy import EnergyDistance


class TestHybridDistance:
    def test_basic(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        dist_fn = HybridDistance(
            paired_method='paired',
            unpaired_method='energy',
            alpha=0.5,
        )
        D = dist_fn(mrd)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T, atol=1e-10)
        assert np.allclose(np.diag(D), 0, atol=1e-10)

    def test_alpha_extremes(self, simple_paired_df):
        """alpha=1 → pure paired, alpha=0 → pure unpaired."""
        mrd = ModelResponseData.from_dataframe(simple_paired_df)

        D_paired = PairedDistance()(mrd)
        D_energy = EnergyDistance()(mrd)

        # Normalize as hybrid does
        D_paired_n = D_paired / D_paired.max() if D_paired.max() > 0 else D_paired
        D_energy_n = D_energy / D_energy.max() if D_energy.max() > 0 else D_energy

        D_a1 = HybridDistance(alpha=1.0)(mrd)
        D_a0 = HybridDistance(alpha=0.0, unpaired_method='energy')(mrd)

        # alpha=1 should be proportional to paired
        np.testing.assert_allclose(D_a1, D_paired_n, atol=1e-10)
        # alpha=0 should be proportional to energy
        np.testing.assert_allclose(D_a0, D_energy_n, atol=1e-10)

    def test_with_instance_methods(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        dist_fn = HybridDistance(
            paired_method=PairedDistance(),
            unpaired_method=EnergyDistance(),
            alpha=0.3,
        )
        D = dist_fn(mrd)
        assert D.shape == (3, 3)

    def test_distance_matrix_properties(self, cluster_models_df):
        mrd = ModelResponseData.from_dataframe(cluster_models_df)
        dist_fn = HybridDistance(alpha=0.5)
        D = dist_fn(mrd)
        assert np.all(np.isfinite(D))
        assert np.all(D >= -1e-10)
        np.testing.assert_allclose(D, D.T, atol=1e-10)
