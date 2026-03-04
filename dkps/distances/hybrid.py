"""
    distances/hybrid.py — Method F: Hybrid Paired + Unpaired distance.
"""

import numpy as np

from ..data import ModelResponseData
from .base import validate_distance_matrix


class HybridDistance:
    """Hybrid distance combining a paired and an unpaired method.

    Useful when data is partially paired: the paired method captures
    query-aligned structure, while the unpaired method captures overall
    distributional differences.

    Parameters
    ----------
    paired_method : DistanceFunction or str
        Distance method for paired component. Default 'paired'.
    unpaired_method : DistanceFunction or str
        Distance method for unpaired component. Default 'energy'.
    alpha : float
        Weight of paired component. Final distance = alpha * D_paired + (1 - alpha) * D_unpaired.
        Default 0.5.
    paired_kwargs : dict
        Extra kwargs for paired method constructor (if string).
    unpaired_kwargs : dict
        Extra kwargs for unpaired method constructor (if string).
    """

    def __init__(self, paired_method='paired', unpaired_method='energy',
                 alpha=0.5, paired_kwargs=None, unpaired_kwargs=None):
        self.alpha = alpha
        self._paired_kwargs = paired_kwargs or {}
        self._unpaired_kwargs = unpaired_kwargs or {}

        # Resolve string names to instances (deferred to avoid circular import)
        self._paired_method_spec = paired_method
        self._unpaired_method_spec = unpaired_method
        self._paired_fn = None
        self._unpaired_fn = None

    def _resolve(self):
        if self._paired_fn is not None:
            return

        from . import get_distance

        if isinstance(self._paired_method_spec, str):
            self._paired_fn = get_distance(self._paired_method_spec, **self._paired_kwargs)
        else:
            self._paired_fn = self._paired_method_spec

        if isinstance(self._unpaired_method_spec, str):
            self._unpaired_fn = get_distance(self._unpaired_method_spec, **self._unpaired_kwargs)
        else:
            self._unpaired_fn = self._unpaired_method_spec

    def __call__(self, data: ModelResponseData) -> np.ndarray:
        self._resolve()

        D_paired = self._paired_fn(data)
        D_unpaired = self._unpaired_fn(data)

        # Normalize each to [0, 1] range before combining
        max_p = D_paired.max()
        max_u = D_unpaired.max()
        if max_p > 0:
            D_paired_norm = D_paired / max_p
        else:
            D_paired_norm = D_paired
        if max_u > 0:
            D_unpaired_norm = D_unpaired / max_u
        else:
            D_unpaired_norm = D_unpaired

        D = self.alpha * D_paired_norm + (1 - self.alpha) * D_unpaired_norm

        return validate_distance_matrix(D)
