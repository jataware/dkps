"""
    distances/base.py — DistanceFunction protocol and validation helpers.
"""

import numpy as np
from typing import Protocol, runtime_checkable

from ..data import ModelResponseData


@runtime_checkable
class DistanceFunction(Protocol):
    """Protocol for distance functions between model response distributions."""

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        """Compute pairwise distance matrix.

        Parameters
        ----------
        data : ModelResponseData

        Returns
        -------
        np.ndarray of shape (m, m), symmetric, non-negative, zero diagonal.
        """
        ...


def validate_distance_matrix(D: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Validate and symmetrize a distance matrix.

    Checks: square, symmetric, no NaN/Inf, zero diagonal, non-negative.

    Parameters
    ----------
    D : np.ndarray
        Candidate distance matrix.
    atol : float
        Tolerance for symmetry and diagonal checks.

    Returns
    -------
    np.ndarray
        Symmetrized distance matrix.

    Raises
    ------
    ValueError
        If any check fails.
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix is not square: shape {D.shape}")

    if not np.all(np.isfinite(D)):
        raise ValueError(
            "Distance matrix contains NaN/Inf — this usually means a distance "
            "computation diverged"
        )

    max_asymmetry = np.max(np.abs(D - D.T))
    if max_asymmetry > atol:
        raise ValueError(
            f"Distance matrix is not symmetric (max asymmetry: {max_asymmetry:.2e})"
        )

    max_diag = np.max(np.abs(np.diag(D)))
    if max_diag > atol:
        raise ValueError(
            f"Distance matrix diagonal is not zero (max: {max_diag:.2e})"
        )

    if np.any(D < -atol):
        raise ValueError("Distance matrix contains negative values")

    # Symmetrize and zero diagonal
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D
