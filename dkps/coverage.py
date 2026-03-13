import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde


def select_pca_dim(explained_variance_ratio, max_dim=50):
    """Elbow detection on scree plot via maximum second-derivative."""
    evr = np.asarray(explained_variance_ratio)[:max_dim]
    if len(evr) <= 2:
        return len(evr)
    diffs = np.diff(evr)
    elbows = np.diff(diffs)  # second derivative
    # elbow = point of maximum curvature (most negative second derivative
    # means the explained variance is dropping off fastest)
    return int(np.argmin(elbows)) + 2  # +2 because of double diff offset


def estimate_coverage_weights(embeddings_by_model, d_pca=None, kde_bandwidth=None):
    """
    Estimate pairwise coverage weights for unpaired distance correction.

    Parameters
    ----------
    embeddings_by_model : dict {model_id: np.ndarray (n_queries, emb_dim)}
    d_pca : int or None
        Number of PCA dimensions. None = auto via scree elbow.
    kde_bandwidth : float or None
        KDE bandwidth. None = Scott's rule (scipy default).

    Returns
    -------
    weights : dict {(model_i, model_k): float}
        Pairwise harmonic-mean coverage weights in [0, 1].
    """
    model_ids = list(embeddings_by_model.keys())
    n_models = len(model_ids)

    # Pool all embeddings for PCA
    all_embs = np.vstack([embeddings_by_model[m] for m in model_ids])

    # PCA dimensionality reduction
    max_components = min(all_embs.shape[0], all_embs.shape[1], 50)
    pca = PCA(n_components=max_components)
    pca.fit(all_embs)

    if d_pca is None:
        d_pca = select_pca_dim(pca.explained_variance_ratio_)
    d_pca = min(d_pca, max_components)

    # Project each model's embeddings into PCA space
    projected = {}
    for m in model_ids:
        projected[m] = pca.transform(embeddings_by_model[m])[:, :d_pca]

    # Fit per-model KDE in PCA space
    kdes = {}
    for m in model_ids:
        data_t = projected[m].T  # (d_pca, n_queries)
        if kde_bandwidth is not None:
            kdes[m] = gaussian_kde(data_t, bw_method=kde_bandwidth)
        else:
            kdes[m] = gaussian_kde(data_t)  # Scott's rule

    # Pairwise harmonic-mean weights
    weights = {}
    for i in range(n_models):
        for k in range(i + 1, n_models):
            mi, mk = model_ids[i], model_ids[k]
            # Evaluate density of model_i's queries under model_k's KDE, and vice versa
            # Use model_i's query points
            pts_i = projected[mi].T
            pts_k = projected[mk].T

            # p_k(x) for x ~ p_i  and  p_i(x) for x ~ p_k
            pk_at_i = kdes[mk](pts_i)  # (n_i,)
            pi_at_i = kdes[mi](pts_i)  # (n_i,)
            pi_at_k = kdes[mi](pts_k)  # (n_k,)
            pk_at_k = kdes[mk](pts_k)  # (n_k,)

            # Harmonic mean of density ratios, averaged over both query sets
            # w_ik = mean of min(p_i/p_k, p_k/p_i) over pooled queries
            eps = 1e-300
            ratio_i = np.minimum(pk_at_i / (pi_at_i + eps), pi_at_i / (pk_at_i + eps))
            ratio_k = np.minimum(pi_at_k / (pk_at_k + eps), pk_at_k / (pi_at_k + eps))
            w = 0.5 * (np.mean(ratio_i) + np.mean(ratio_k))
            w = float(np.clip(w, 0.0, 1.0))

            weights[(mi, mk)] = w
            weights[(mk, mi)] = w

    return weights
