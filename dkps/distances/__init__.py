"""
    distances/ — Registry and re-exports for all distance methods.
"""

from .base import DistanceFunction, validate_distance_matrix
from .paired import PairedDistance
from .mmd import MMDDistance
from .energy import EnergyDistance
from .wasserstein import WassersteinDistance
from .gromov_wasserstein import GromovWassersteinDistance
from .soft_paired import SoftPairedOTDistance
from .hybrid import HybridDistance


# String name -> class mapping
DISTANCE_REGISTRY = {
    'paired':      PairedDistance,
    'mmd':         MMDDistance,
    'wasserstein': WassersteinDistance,
    'energy':      EnergyDistance,
    'gromov':      GromovWassersteinDistance,
    'soft_paired': SoftPairedOTDistance,
    'hybrid':      HybridDistance,
}


def get_distance(name, **kwargs):
    """Look up a distance function by name and instantiate with kwargs.

    Parameters
    ----------
    name : str
        One of: 'paired', 'mmd', 'wasserstein', 'energy', 'gromov', 'soft_paired', 'hybrid'.
    **kwargs
        Passed to the distance class constructor.

    Returns
    -------
    DistanceFunction instance
    """
    if name not in DISTANCE_REGISTRY:
        raise ValueError(
            f"Unknown distance '{name}'. Available: {list(DISTANCE_REGISTRY.keys())}"
        )
    return DISTANCE_REGISTRY[name](**kwargs)


__all__ = [
    'DistanceFunction',
    'validate_distance_matrix',
    'PairedDistance',
    'MMDDistance',
    'EnergyDistance',
    'WassersteinDistance',
    'GromovWassersteinDistance',
    'SoftPairedOTDistance',
    'HybridDistance',
    'DISTANCE_REGISTRY',
    'get_distance',
]
