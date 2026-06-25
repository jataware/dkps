"""Score-only baselines for held-out (model, task) score prediction.

Uses only the observed score matrix (no response/query embeddings), to contrast
with the embedding-based DoubleKernelDKPS estimators. Returns a prediction
matrix with values at the held-out (unobserved) entries and NaN elsewhere;
evaluate with the same held-out MAE as the DKPS estimators.
"""
import numpy as np


def matrix_completion_predict(score_mat, observed, rank=2, lam=0.1, n_init=5,
                              max_iter=50, clip_eps=1e-3, logit=True,
                              random_state=0, fill_observed=False):
    """Logit-space rank-2 bias-decomposed ALS matrix completion (BenchPress).

    Papailiopoulos, 'You Don't Need to Run Every Eval' (arXiv:2606.24020):
      1. logit-transform scores in [0,1] (clipped away from endpoints)
      2. standardize each task column (zero mean, unit variance) on observed
      3. fit  x_hat_mb = xbar + (model offset) + (task offset) + (U V^T)_mb
         over observed entries via ALS (closed-form ridge updates, lam),
         ensemble-averaged over random inits; rank R=2
      4. invert standardization + logit; clip to [0,1]

    Returns predictions at unobserved entries (NaN at observed).
    """
    M = np.asarray(score_mat, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    nM, nB = M.shape
    preds = np.full_like(M, np.nan)
    if not (~mask).any() and not fill_observed:
        return preds

    # 1. logit transform
    if logit:
        s = np.clip(M, clip_eps, 1.0 - clip_eps)
        X = np.log(s / (1.0 - s))
    else:
        X = M.copy()

    # 2. column standardization (observed entries)
    mu = np.zeros(nB)
    sd = np.ones(nB)
    for b in range(nB):
        col = mask[:, b]
        if col.any():
            mu[b] = X[col, b].mean()
            std = X[col, b].std()
            sd[b] = std if std > 1e-8 else 1.0
    Xs = (X - mu) / sd

    # 3. bias terms (global + model + task offsets) on standardized observed
    gbar = Xs[mask].mean()
    row_bar = np.array([Xs[m, mask[m]].mean() if mask[m].any() else gbar
                        for m in range(nM)])
    col_bar = np.array([Xs[mask[:, b], b].mean() if mask[:, b].any() else gbar
                        for b in range(nB)])
    bias = row_bar[:, None] + col_bar[None, :] - gbar
    R = Xs - bias

    # 4. rank-r ALS on the residual over observed entries, ensembled over inits
    r = int(max(1, min(rank, nM - 1, nB - 1)))
    rng = np.random.default_rng(random_state)
    Ir = lam * np.eye(r)
    acc = np.zeros((nM, nB))
    for _ in range(n_init):
        U = rng.normal(0, 0.1, (nM, r))
        V = rng.normal(0, 0.1, (nB, r))
        for _ in range(max_iter):
            for m in range(nM):
                cols = np.where(mask[m])[0]
                if len(cols) == 0:
                    continue
                Vc = V[cols]
                U[m] = np.linalg.solve(Vc.T @ Vc + Ir, Vc.T @ R[m, cols])
            for b in range(nB):
                rows = np.where(mask[:, b])[0]
                if len(rows) == 0:
                    continue
                Ur = U[rows]
                V[b] = np.linalg.solve(Ur.T @ Ur + Ir, Ur.T @ R[rows, b])
        acc += bias + U @ V.T
    Xhat_std = acc / n_init

    # 5. invert standardization + logit
    Xhat = Xhat_std * sd + mu
    Phat = 1.0 / (1.0 + np.exp(-Xhat)) if logit else Xhat
    Pc = np.clip(Phat, 0.0, 1.0)
    # default: predictions only at unobserved entries. fill_observed also returns the
    # low-rank reconstruction at observed cells (a denoised estimate that uses the cell's
    # own value but regularizes it toward the fitted structure).
    preds[~mask] = Pc[~mask]
    if fill_observed:
        preds[mask] = Pc[mask]
    return preds
