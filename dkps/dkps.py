import numpy as np
from graspologic.embed import ClassicalMDS
# from graspologic.embed import OmnibusEmbed

from scipy.spatial.distance import pdist, cdist, squareform

# from .utils import knn_graph


class DataKernelPerspectiveSpace:
    def __init__(
            self,
            response_distribution_fn=np.mean,
            response_distribution_axis=1,
            metric_cmds='euclidean',
            n_components_cmds=None,
            n_elbows_cmds=2,
            dissimilarity="euclidean",
        ):
        
        self.response_distribution_fn   = response_distribution_fn
        self.response_distribution_axis = response_distribution_axis
        self.metric_cmds                = metric_cmds
        self.n_components_cmds          = n_components_cmds
        self.n_elbows_cmds              = n_elbows_cmds
        self.dissimilarity              = dissimilarity
        
        self._n_queries = None
        self._X_flat     = None
        self._mds_input  = None
        self._cmds_embds = None

    def fit_transform(self, data, return_dict=True):
        """
        data: dict {model_name: np.array(n_queries, n_replicates, embedding_dim)}
        """
        
        # qc checks
        assert isinstance(data, dict),                                  'data must be a dict'
        assert all([isinstance(x, np.ndarray) for x in data.values()]), 'all values must be numpy arrays'
        assert all([x.ndim == 3 for x in data.values()]),               'all arrays must be 3D - np.array(n_queries, n_replicates, embedding_dim)'
        assert len(set([x.shape for x in data.values()])) == 1,         'all arrays must have the same shape'

        # aggregate over replicates -> (n_models, n_queries, embedding_dim)
        X = np.stack([self.response_distribution_fn(v, axis=self.response_distribution_axis) for k,v in data.items()])
        n_models, self._n_queries, embedding_dim = X.shape
        
        # flatten -> (n_models, n_queries * embedding_dim)
        self._X_flat = X.reshape(len(X), -1)

        self._mds_input  = squareform(pdist(self._X_flat, metric=self.metric_cmds)) 
        self._mds_input  = self._mds_input / np.sqrt(self._n_queries)
        self._cmds_embds = ClassicalMDS(
            n_components=self.n_components_cmds,
            n_elbows=self.n_elbows_cmds,
            dissimilarity=self.dissimilarity
        ).fit_transform(self._mds_input)
        
        if return_dict:
            return {key: self._cmds_embds[i] for i, key in enumerate(data.keys())}
        else:
            return self._cmds_embds
    
    def transform(self, q):
        # [BKJ] this looks weird, because the original implementation does distances twice
        #       hayden says this is what we want to do, though
        _oos_X         = self.response_distribution_fn(q, axis=self.response_distribution_axis)
        _oos_X_flat    = _oos_X.reshape(-1)
        sel            = ~np.isnan(_oos_X_flat) # don't use the missing values
        
        # <<
        # _oos_mds_input = cdist(self._X_flat[:,sel], _oos_X_flat[sel][None], metric=self.metric_cmds)
        # _oos_mds_input = _oos_mds_input.squeeze() / np.sqrt(self._n_queries) # [BKJ] this seems wrong? 
        # # [BKJ] ^ I think there's some issue w/ scaling going on here
        # --
        # inmputing missing values.  this should address scaling issues ...
        _oos_X_flat[~sel] = self._X_flat[:,~sel].mean(axis=0)
        _oos_mds_input = cdist(self._X_flat, _oos_X_flat[None], metric=self.metric_cmds)
        _oos_mds_input = _oos_mds_input.squeeze() / np.sqrt(self._n_queries)
        # >>
        
        # breakpoint()
        D2 = squareform(pdist(self._mds_input, metric='sqeuclidean'))
        d2 = ((self._mds_input - _oos_mds_input) ** 2).sum(axis=1)
        b  = -0.5 * (d2 - D2.mean(axis=1) - d2.mean() + D2.mean())
        return (self._cmds_embds.T @ b) / (self._cmds_embds ** 2).sum(axis=0)



# class DataKernelFunctionalSpace:
#     def __init__(
#         self, 
#         n_neighbors=None, 
#         n_components_joint=None, 
#         n_elbows_joint=2, 
#         n_components_cmds=None, 
#         n_elbows_cmds=2, 
#         metric='euclidean',
#     ):
#         self.n_neighbors        = n_neighbors
#         self.n_components_joint = n_components_joint
#         self.n_elbows_joint     = n_elbows_joint
#         self.n_components_cmds  = n_components_cmds
#         self.n_elbows_cmds      = n_elbows_cmds
#         self.metric             = metric
        
#     def fit_transform(self, X):
#         if isinstance(X, dict):
#             self.keys = list(X.keys())
#             X = list(X.values())
#         else:
#             self.keys = None
        
#         assert isinstance(X, (list, np.ndarray)), 'X must be a dict of 2-d array-like values or array-like with 2-d array-like items'
        
#         N = len(X)
#         for i,x in enumerate(X):
#             x = np.array(x)
#             assert x.ndim == 2
#             X[i] = knn_graph(x, self.n_neighbors, symmetrize=True)

            
#         dist_matrix = np.zeros((N, N))
#         for i in range(N):
#             for j in range(i+1, N):
#                 # [BKJ] - there's probably some way to broadcast this
#                 omni_embds = OmnibusEmbed(n_components=self.n_components_joint, n_elbows=self.n_elbows_joint).fit_transform([X[i], X[j]])
#                 temp_dist = np.linalg.norm(omni_embds[0] - omni_embds[1]) / np.linalg.norm( (omni_embds[0] + omni_embds[1]) / 2 )
#                 dist_matrix[i,j] = temp_dist
#                 dist_matrix[j,i] = temp_dist
                
#         cmds_embds = ClassicalMDS(self.n_components_cmds, self.n_elbows_cmds).fit_transform(dist_matrix)
        
#         if self.keys:
#             return {key: cmds_embds[i] for i, key in enumerate(self.keys)}
#         else:
#             return cmds_embds