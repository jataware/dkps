"""
    core.py — DKPS class: DataFrame → embed → distance → MDS.
"""

import numpy as np
import pandas as pd
from graspologic.embed import ClassicalMDS

from .data import ModelResponseData
from .distances import get_distance
from .distances.base import DistanceFunction


class DKPS:
    """Data Kernel Perspective Space — model comparison via response distributions.

    Parameters
    ----------
    distance : str or DistanceFunction
        Distance method: 'paired', 'mmd', 'wasserstein', 'energy', 'gromov',
        'soft_paired', 'hybrid'. Or pass a DistanceFunction instance directly.
    n_components : int or None
        MDS dimensions. None = auto-select via elbow.
    n_elbows : int
        Number of elbows for auto dimensionality selection.
    embed_provider : str
        Provider for embedding text (used if response_embedding column missing).
    embed_model : str or None
        Model name for embedding provider.
    response_agg_fn : callable or None
        For paired data with multiple replicates: how to aggregate.
        None = take first replicate.
    **distance_kwargs :
        Passed to the distance function constructor (e.g., kernel='rbf').
    """

    def __init__(
        self,
        distance='paired',
        n_components=None,
        n_elbows=2,
        embed_provider='sentence-transformers',
        embed_model=None,
        response_agg_fn=None,
        **distance_kwargs,
    ):
        self.n_components = n_components
        self.n_elbows = n_elbows
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.response_agg_fn = response_agg_fn

        # Pass response_agg_fn to paired distance if applicable
        if response_agg_fn is not None:
            if isinstance(distance, str) and distance == 'paired':
                distance_kwargs.setdefault('response_agg_fn', response_agg_fn)

        # Build distance function
        if isinstance(distance, str):
            self._distance_fn = get_distance(distance, **distance_kwargs)
        else:
            self._distance_fn = distance

        self._distance_name = distance if isinstance(distance, str) else type(distance).__name__

    def fit_transform(self, data, return_dict=True, **kwargs):
        """Compute model embeddings in DKPS space.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns: model, response|response_embedding,
            [query_id], [query|query_embedding].
        return_dict : bool
            If True, return {model_name: embedding_vector}.

        Returns
        -------
        dict[str, np.ndarray] or np.ndarray of shape (n_models, n_components)

        Raises
        ------
        TypeError
            If data is not a DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )

        mrd = self._prepare_data(data)
        D = self._distance_fn(mrd)

        cmds = ClassicalMDS(
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            dissimilarity='precomputed',
        )
        embeddings = cmds.fit_transform(D)

        if return_dict:
            return {name: embeddings[i] for i, name in enumerate(mrd.model_names)}
        else:
            return embeddings

    def distance_matrix(self, data, **kwargs):
        """Compute and return only the m x m distance matrix (no MDS).

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (m, m)

        Raises
        ------
        TypeError
            If data is not a DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )

        mrd = self._prepare_data(data)
        return self._distance_fn(mrd)

    def _prepare_data(self, data):
        """Convert DataFrame to ModelResponseData."""
        embed_fn = self._make_embed_fn()
        return ModelResponseData.from_dataframe(data, embed_fn=embed_fn)

    def _make_embed_fn(self):
        """Create an embedding function from configured provider."""
        def embed_fn(texts):
            from .embed import embed_api
            return embed_api(self.embed_provider, texts, model=self.embed_model)
        return embed_fn
