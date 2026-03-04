"""
    test_synthetic.py — End-to-end: all methods on same synthetic data.

    Validates that distance methods recover known structure (clusters, orderings).

    Run with -s to see metric values:
        pixi run pytest tests/test_synthetic.py -v -s
"""

import numpy as np
import pytest
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans

from dkps import DKPS


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def cluster_ari(result_dict, cluster_labels, n_clusters):
    """Compute ARI: KMeans on DKPS embeddings vs true cluster labels."""
    model_names = list(result_dict.keys())
    X = np.vstack([result_dict[m] for m in model_names])
    true = [cluster_labels[m] for m in model_names]

    if X.shape[1] < 1:
        return 0.0

    pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    return adjusted_rand_score(true, pred)


def cluster_silhouette(result_dict, cluster_labels):
    """Compute silhouette score on DKPS embeddings."""
    model_names = list(result_dict.keys())
    X = np.vstack([result_dict[m] for m in model_names])
    true = [cluster_labels[m] for m in model_names]

    if len(set(true)) < 2 or X.shape[0] < 3:
        return 0.0

    return silhouette_score(X, true)


def ordering_kendall_tau(result_dict, theta_dict):
    """Compute Kendall tau between 1st MDS component ordering and true theta."""
    model_names = list(result_dict.keys())
    X = np.vstack([result_dict[m] for m in model_names])
    true_theta = [theta_dict[m] for m in model_names]

    # Use first component
    comp1 = X[:, 0]
    tau, pvalue = kendalltau(comp1, true_theta)
    return abs(tau), pvalue  # sign can flip


def _report(metric_name, method, value, threshold=None, extra=None):
    """Print a metric value for visibility with -s flag."""
    parts = [f"  {method:15s}  {metric_name} = {value:.4f}"]
    if threshold is not None:
        parts.append(f" (threshold: {threshold})")
    if extra:
        parts.append(f"  {extra}")
    print("\n" + "".join(parts))


# ──────────────────────────────────────────────────────────────────────
# Cluster recovery tests
# ──────────────────────────────────────────────────────────────────────

PAIRED_METHODS = ['paired']
UNPAIRED_METHODS = ['mmd', 'energy']

# Conditionally add OT methods
try:
    import ot
    UNPAIRED_METHODS += ['wasserstein']
    HAS_POT = True
except ImportError:
    HAS_POT = False


class TestClusterRecovery:
    @pytest.mark.parametrize("method", PAIRED_METHODS)
    def test_paired_cluster_recovery(self, cluster_models_df, method):
        labels = cluster_models_df.attrs['cluster_labels']
        n_clusters = cluster_models_df.attrs['n_clusters']

        dkps = DKPS(distance=method, n_components=5)
        result = dkps.fit_transform(cluster_models_df)

        ari = cluster_ari(result, labels, n_clusters)
        sil = cluster_silhouette(result, labels)
        _report("ARI", method, ari, threshold=0.8)
        _report("Silhouette", method, sil)
        assert ari > 0.8, f"Method '{method}' ARI = {ari:.4f}, expected > 0.8"

    @pytest.mark.parametrize("method", UNPAIRED_METHODS)
    def test_unpaired_cluster_recovery(self, cluster_models_df, method):
        labels = cluster_models_df.attrs['cluster_labels']
        n_clusters = cluster_models_df.attrs['n_clusters']

        # Remove query_id for unpaired methods
        df = cluster_models_df.drop(columns=['query_id'])

        dkps = DKPS(distance=method, n_components=5)
        result = dkps.fit_transform(df)

        ari = cluster_ari(result, labels, n_clusters)
        sil = cluster_silhouette(result, labels)
        _report("ARI", method, ari, threshold=0.8)
        _report("Silhouette", method, sil)
        assert ari > 0.8, f"Method '{method}' ARI = {ari:.4f}, expected > 0.8"

    @pytest.mark.parametrize("method", UNPAIRED_METHODS)
    def test_unpaired_variable_sizes(self, unpaired_models_df, method):
        labels = unpaired_models_df.attrs['cluster_labels']
        n_clusters = unpaired_models_df.attrs['n_clusters']

        dkps = DKPS(distance=method, n_components=5)
        result = dkps.fit_transform(unpaired_models_df)

        ari = cluster_ari(result, labels, n_clusters)
        sil = cluster_silhouette(result, labels)
        _report("ARI", method, ari, threshold=0.7, extra="(variable sizes)")
        _report("Silhouette", method, sil, extra="(variable sizes)")
        assert ari > 0.7, f"Method '{method}' ARI = {ari:.4f}, expected > 0.7"


# ──────────────────────────────────────────────────────────────────────
# 1-D ordering tests
# ──────────────────────────────────────────────────────────────────────

class TestOrderingRecovery:
    @pytest.mark.parametrize("method", PAIRED_METHODS)
    def test_paired_ordering(self, manifold_models_df, method):
        thetas = manifold_models_df.attrs['theta']

        dkps = DKPS(distance=method, n_components=5)
        result = dkps.fit_transform(manifold_models_df)

        tau, pvalue = ordering_kendall_tau(result, thetas)
        _report("Kendall τ", method, tau, threshold=0.7, extra=f"p={pvalue:.2e}")
        assert tau > 0.7, f"Method '{method}' Kendall τ = {tau:.4f}, expected > 0.7"

    @pytest.mark.parametrize("method", UNPAIRED_METHODS)
    def test_unpaired_ordering(self, manifold_models_df, method):
        thetas = manifold_models_df.attrs['theta']
        df = manifold_models_df.drop(columns=['query_id'])

        dkps = DKPS(distance=method, n_components=5)
        result = dkps.fit_transform(df)

        tau, pvalue = ordering_kendall_tau(result, thetas)
        _report("Kendall τ", method, tau, threshold=0.7, extra=f"p={pvalue:.2e}")
        assert tau > 0.7, f"Method '{method}' Kendall τ = {tau:.4f}, expected > 0.7"


# ──────────────────────────────────────────────────────────────────────
# Distance matrix basic properties (all methods)
# ──────────────────────────────────────────────────────────────────────

class TestDistanceMatrixProperties:
    @pytest.mark.parametrize("method", PAIRED_METHODS + UNPAIRED_METHODS)
    def test_symmetric_nonneg_zero_diag(self, simple_paired_df, method):
        if method in UNPAIRED_METHODS:
            df = simple_paired_df.drop(columns=['query_id'])
        else:
            df = simple_paired_df

        dkps = DKPS(distance=method)
        D = dkps.distance_matrix(df)

        max_asymmetry = np.max(np.abs(D - D.T))
        max_diag = np.max(np.abs(np.diag(D)))
        min_offdiag = D[np.triu_indices_from(D, k=1)].min()
        _report("max asymmetry", method, max_asymmetry)
        _report("max |diag|", method, max_diag)
        _report("min off-diag", method, min_offdiag)

        assert D.shape[0] == D.shape[1]
        np.testing.assert_allclose(D, D.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-10)
        assert np.all(D >= -1e-10)
        assert np.all(np.isfinite(D))


# ──────────────────────────────────────────────────────────────────────
# Cross-method agreement
# ──────────────────────────────────────────────────────────────────────

class TestCrossMethodAgreement:
    def test_paired_vs_unpaired_ranking(self, cluster_models_df):
        """Paired and unpaired methods should agree on distance ranking."""
        dkps_paired = DKPS(distance='paired')
        D_paired = dkps_paired.distance_matrix(cluster_models_df)

        df_unpaired = cluster_models_df.drop(columns=['query_id'])
        dkps_energy = DKPS(distance='energy')
        D_energy = dkps_energy.distance_matrix(df_unpaired)

        # Extract upper triangle
        idx = np.triu_indices_from(D_paired, k=1)
        rho, pvalue = spearmanr(D_paired[idx], D_energy[idx])
        _report("Spearman ρ", "paired↔energy", rho, threshold=0.7, extra=f"p={pvalue:.2e}")
        assert rho > 0.7, f"Spearman ρ = {rho:.4f}, expected > 0.7"

    @pytest.mark.parametrize("method", UNPAIRED_METHODS)
    def test_unpaired_vs_paired_ranking(self, cluster_models_df, method):
        """Each unpaired method should rank-correlate with paired."""
        dkps_paired = DKPS(distance='paired')
        D_paired = dkps_paired.distance_matrix(cluster_models_df)

        df_unpaired = cluster_models_df.drop(columns=['query_id'])
        dkps_other = DKPS(distance=method)
        D_other = dkps_other.distance_matrix(df_unpaired)

        idx = np.triu_indices_from(D_paired, k=1)
        rho, pvalue = spearmanr(D_paired[idx], D_other[idx])
        _report("Spearman ρ", f"paired↔{method}", rho, threshold=0.7, extra=f"p={pvalue:.2e}")
        assert rho > 0.7, f"Spearman ρ(paired, {method}) = {rho:.4f}, expected > 0.7"
