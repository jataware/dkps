import warnings

import numpy as np
import pandas as pd
from graspologic.embed import ClassicalMDS, select_dimension
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


class UnpairedDKPS:
    def __init__(
            self,
            mode='combined',
            query_kernel='constant',
            use_coverage=False,
            coverage_mode='oracle',
            paired_weight=None,
            query_bandwidth='median',
            coverage_bandwidth='scott',
            coverage_pca_components=None,
            n_components_cmds=None,
            n_elbows_cmds=2,
            dissimilarity="precomputed",
        ):

        assert mode in {'paired', 'unpaired', 'combined'}, \
            "mode must be one of {'paired', 'unpaired', 'combined'}"
        assert query_kernel in {'constant', 'rbf'}, \
            "query_kernel must be one of {'constant', 'rbf'}"
        assert coverage_mode in {'oracle', 'pca'}, \
            "coverage_mode must be one of {'oracle', 'pca'}"
        if paired_weight is not None:
            assert 0.0 <= paired_weight <= 1.0, 'paired_weight must be in [0, 1]'

        self.mode = mode
        self.query_kernel = query_kernel
        self.use_coverage = use_coverage
        self.coverage_mode = coverage_mode
        self.paired_weight = paired_weight
        self.query_bandwidth = query_bandwidth
        self.coverage_bandwidth = coverage_bandwidth
        self.coverage_pca_components = coverage_pca_components
        self.n_components_cmds = n_components_cmds
        self.n_elbows_cmds = n_elbows_cmds
        self.dissimilarity = dissimilarity

    def _validate_input_dataframe(self, data):
        assert isinstance(data, pd.DataFrame), 'data must be a DataFrame'
        for col in ('model_id', 'query_id', 'embedding', 'query_vec'):
            assert col in data.columns, f'DataFrame must have column: {col}'

        df = data.copy()
        duplicate_pairs = df.duplicated(['model_id', 'query_id'], keep=False)
        assert not duplicate_pairs.any(), 'duplicate (model_id, query_id) rows are not supported'

        embedding_shapes = {np.asarray(emb).shape for emb in df['embedding'].values}
        query_shapes = {np.asarray(q).shape for q in df['query_vec'].values}

        assert len(embedding_shapes) == 1, 'all embeddings must have the same shape'
        assert len(query_shapes) == 1, 'all query_vec values must have the same shape'

        embedding_shape = next(iter(embedding_shapes))
        query_shape = next(iter(query_shapes))
        assert len(embedding_shape) == 1, 'each embedding must be a 1D vector'
        assert len(query_shape) == 1, 'each query_vec must be a 1D vector'
        assert query_shape[0] <= embedding_shape[0], \
            'query_vec dimensionality must not exceed embedding dimensionality'

        model_names = sorted(df['model_id'].unique())
        assert len(model_names) >= 2, 'at least two models are required'

        df['query_code'] = pd.factorize(df['query_id'])[0]

        return df, model_names, embedding_shape[0], query_shape[0]

    def _resolve_bandwidth(self, X, bandwidth, context='bandwidth'):
        if callable(bandwidth):
            value = float(bandwidth(X))
        elif isinstance(bandwidth, (int, float)):
            value = float(bandwidth)
        elif bandwidth == 'scott':
            n, d = X.shape
            scale = float(np.mean(np.std(X, axis=0, ddof=0)))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            value = scale * (max(n, 1) ** (-1.0 / (d + 4)))
        elif bandwidth == 'median':
            if len(X) < 2:
                value = 1.0
            else:
                dists = pdist(X)
                positive = dists[dists > 0]
                value = float(np.median(positive)) if len(positive) else 1.0
        else:
            raise ValueError(f'unsupported {context}: {bandwidth}')

        if not np.isfinite(value) or value <= 0:
            value = 1.0
        return value

    def _make_query_features(self, df, embedding_dim, query_dim):
        if self.coverage_mode == 'oracle':
            # Block I synthetic experiments can use the true latent query vectors directly.
            self.query_feature_transformer_ = None
            self.coverage_pca_components_ = None
            return np.stack(df['query_vec'].values)

        X = np.stack(df['embedding'].values)
        # The plan's coverage construction uses a low-dimensional PCA space derived
        # from responses. If the caller does not force a dimension, choose one from
        # the scree/elbow rule rather than assuming it equals the latent query dim.
        max_components = min(X.shape[0], X.shape[1])
        if self.coverage_pca_components is None:
            centered = X - np.mean(X, axis=0, keepdims=True)
            elbows, _ = select_dimension(centered, n_elbows=self.n_elbows_cmds)
            n_components = int(elbows[-1]) if elbows else min(query_dim, max_components)
        else:
            n_components = int(self.coverage_pca_components)
        
        n_components = max(1, min(n_components, max_components))

        pca = PCA(n_components=n_components)
        pca.fit(X)

        padded_queries = np.zeros((len(df), embedding_dim), dtype=float)
        padded_queries[:, :query_dim] = np.stack(df['query_vec'].values)
        self.query_feature_transformer_ = pca
        self.coverage_pca_components_ = n_components
        return pca.transform(padded_queries)

    def _split_query_sets(self, df, model_names):
        """
        Split query ids into the two sets used by the factored estimator:

        - paired queries shared across all models
        - unpaired queries that appear for exactly one model

        This intentionally rejects partially shared queries because they do not
        fit the paired-vs-unpaired factorization the user requested.
        """
        query_model_counts = df.groupby('query_code')['model_id'].nunique()
        n_models = len(model_names)

        invalid = query_model_counts[(query_model_counts != 1) & (query_model_counts != n_models)]
        assert len(invalid) == 0, (
            'factored estimator requires each query_id to appear in exactly one '
            'model or in all models'
        )

        paired_query_codes = sorted(query_model_counts[query_model_counts == n_models].index.tolist())
        unpaired_query_codes = set(query_model_counts[query_model_counts == 1].index.tolist())
        return paired_query_codes, unpaired_query_codes

    def _prepare_model_data(self, df, model_names, paired_query_codes, unpaired_query_codes):
        embedding_dim = np.asarray(df['embedding'].iloc[0]).shape[0]
        query_feature_dim = np.asarray(df['query_feature'].iloc[0]).shape[0]

        paired_query_code_set = set(paired_query_codes)
        model_data = {}

        for model_name in model_names:
            sub = df[df['model_id'] == model_name].sort_values('query_code')
            paired_sub = sub[sub['query_code'].isin(paired_query_code_set)].sort_values('query_code')
            unpaired_sub = sub[sub['query_code'].isin(unpaired_query_codes)].sort_values('query_code')

            assert len(paired_sub) == len(paired_query_codes), (
                f'model {model_name} is missing one or more globally paired queries'
            )

            if len(paired_sub):
                paired_embeddings = np.stack(paired_sub['embedding'].values)
                paired_query_features = np.stack(paired_sub['query_feature'].values)
            else:
                paired_embeddings = np.empty((0, embedding_dim), dtype=float)
                paired_query_features = np.empty((0, query_feature_dim), dtype=float)

            if len(unpaired_sub):
                unpaired_embeddings = np.stack(unpaired_sub['embedding'].values)
                unpaired_query_features = np.stack(unpaired_sub['query_feature'].values)
                unpaired_query_codes_arr = unpaired_sub['query_code'].to_numpy()
            else:
                unpaired_embeddings = np.empty((0, embedding_dim), dtype=float)
                unpaired_query_features = np.empty((0, query_feature_dim), dtype=float)
                unpaired_query_codes_arr = np.empty(0, dtype=int)

            model_data[model_name] = {
                'paired_embeddings'       : paired_embeddings,
                'paired_query_features'   : paired_query_features,
                'paired_query_codes'      : paired_sub['query_code'].to_numpy(),
                'unpaired_embeddings'     : unpaired_embeddings,
                'unpaired_query_features' : unpaired_query_features,
                'unpaired_query_codes'    : unpaired_query_codes_arr,
            }

        return model_data

    def _infer_paired_weight(self, paired_query_codes, model_data):
        if self.mode == 'paired':
            return 1.0
        if self.mode == 'unpaired':
            return 0.0
        if self.paired_weight is not None:
            return float(self.paired_weight)

        # In the factored estimator, alpha* mixes an explicit paired DKPS term
        # with an explicit unpaired term, so we estimate it from the observed
        # paired and unpaired sample counts directly.
        m_p = float(len(paired_query_codes))
        avg_unpaired = float(np.mean([len(xx['unpaired_query_codes']) for xx in model_data.values()]))
        denom = m_p + avg_unpaired
        if denom == 0:
            return 0.0
        return m_p / denom

    def _fit_coverage_models(self, model_data):
        if not self.use_coverage:
            return None, None

        # In the factored estimator, coverage adjustment belongs only to the
        # unpaired component, so the KDEs are fit on model-unique queries only.
        nonempty_feature_blocks = [
            xx['unpaired_query_features']
            for xx in model_data.values()
            if len(xx['unpaired_query_features'])
        ]
        if not nonempty_feature_blocks:
            return {}, None

        all_query_features = np.vstack(nonempty_feature_blocks)
        coverage_bandwidth = self._resolve_bandwidth(
            all_query_features,
            self.coverage_bandwidth,
            context='coverage_bandwidth',
        )

        coverage_models = {}
        for model_name, model_entry in model_data.items():
            query_features = model_entry['unpaired_query_features']
            if len(query_features) == 0:
                coverage_models[model_name] = None
                continue

            kde = KernelDensity(kernel='gaussian', bandwidth=coverage_bandwidth)
            kde.fit(query_features)
            coverage_models[model_name] = kde
        return coverage_models, coverage_bandwidth

    def _evaluate_density(self, coverage_models, model_name, query_features):
        if not self.use_coverage:
            return np.ones(len(query_features), dtype=float)
        if len(query_features) == 0:
            return np.zeros(0, dtype=float)

        model = coverage_models.get(model_name) if coverage_models is not None else None
        if model is None:
            return np.zeros(len(query_features), dtype=float)

        log_density = model.score_samples(query_features)
        return np.exp(log_density)

    def _resolve_query_bandwidth(self, model_i, model_k, query_features_i, query_features_k):
        if callable(self.query_bandwidth):
            value = float(self.query_bandwidth(model_i, model_k, self))
            if not np.isfinite(value) or value <= 0:
                value = 1.0
            return value

        pooled = np.vstack([query_features_i, query_features_k])
        return self._resolve_bandwidth(pooled, self.query_bandwidth, context='query_bandwidth')

    def _unpaired_query_kernel_matrix(
            self,
            density_model_a,
            density_model_b,
            query_features_a,
            query_features_b,
            coverage_models,
        ):
        """
        Build the unpaired query-kernel block between two query collections.

        This is the w(i, k, q, q') / kappa^*(q, q') part of the estimator from
        the plan, evaluated on two finite sets of query features:

        - if `query_kernel == "constant"`, every query pair starts with weight 1
          (linear MMD over unpaired queries)
        - if `query_kernel == "rbf"`, query pairs are additionally weighted by
          similarity in query space
        - if `use_coverage` is enabled, the base kernel is multiplied by the
          harmonic-mean coverage weight so regions poorly covered by either
          model contribute less

        The same helper is used for the `ii`, `kk`, and `ik` blocks of the
        unpaired estimator. The `density_model_*` arguments specify which pair
        of KDEs define the coverage weight; `query_features_a` and
        `query_features_b` specify where that pairwise kernel is evaluated.
        """
        if self.query_kernel == 'constant':
            kernel_matrix = np.ones((len(query_features_a), len(query_features_b)), dtype=float)
        elif self.query_kernel == 'rbf':
            bandwidth = self._resolve_query_bandwidth(
                density_model_a,
                density_model_b,
                query_features_a,
                query_features_b,
            )
            sq_dists = cdist(query_features_a, query_features_b, metric='sqeuclidean')
            kernel_matrix = np.exp(-sq_dists / (2.0 * bandwidth ** 2))
        else:
            raise ValueError(f'unsupported query_kernel: {self.query_kernel}')

        if not self.use_coverage:
            return kernel_matrix

        # Multiply the base query kernel by the coverage weight w(i, k, q, q').
        # This matches the identifiability correction in the plan: if either model
        # assigns low density to the relevant query region, that pair gets little
        # influence on the estimated distance.
        density_a = self._evaluate_density(coverage_models, density_model_a, query_features_a)
        density_b = self._evaluate_density(coverage_models, density_model_b, query_features_b)
        denom = density_a[:, None] + density_b[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            coverage = np.where(
                denom > 0,
                2.0 * density_a[:, None] * density_b[None, :] / denom,
                0.0,
            )
        return kernel_matrix * coverage

    def _weighted_linear_average(self, X_a, X_b, query_kernel_matrix):
        normalizer = np.sum(query_kernel_matrix)
        if normalizer <= 0:
            return np.nan
        return float(np.sum((query_kernel_matrix @ X_b) * X_a) / normalizer)

    def _paired_distance_sq(self, data_i, data_k):
        """
        Ordinary DKPS on the globally shared query set.

        This matches the current DKPS implementation's scaling:

            ||vec(X_i^P) - vec(X_k^P)||^2 / m_p

        which is the average per-query squared Euclidean difference.
        """
        X_i = data_i['paired_embeddings']
        X_k = data_k['paired_embeddings']
        if len(X_i) == 0 or len(X_k) == 0:
            return np.nan

        assert X_i.shape == X_k.shape, 'paired query blocks must have the same shape'
        diff = X_i - X_k
        return float(np.sum(diff * diff) / len(X_i))

    def _unpaired_distance_sq(self, model_i, model_k, data_i, data_k, coverage_models):
        """
        Linear-MMD-style distance on the model-unique query sets only.
        """
        X_i = data_i['unpaired_embeddings']
        X_k = data_k['unpaired_embeddings']
        if len(X_i) == 0 or len(X_k) == 0:
            return np.nan

        K_ii = self._unpaired_query_kernel_matrix(
            density_model_a   = model_i,
            density_model_b   = model_k,
            query_features_a  = data_i['unpaired_query_features'],
            query_features_b  = data_i['unpaired_query_features'],
            coverage_models   = coverage_models,
        )
        K_kk = self._unpaired_query_kernel_matrix(
            density_model_a   = model_k,
            density_model_b   = model_i,
            query_features_a  = data_k['unpaired_query_features'],
            query_features_b  = data_k['unpaired_query_features'],
            coverage_models   = coverage_models,
        )
        K_ik = self._unpaired_query_kernel_matrix(
            density_model_a   = model_i,
            density_model_b   = model_k,
            query_features_a  = data_i['unpaired_query_features'],
            query_features_b  = data_k['unpaired_query_features'],
            coverage_models   = coverage_models,
        )

        term_ii = self._weighted_linear_average(X_i, X_i, K_ii)
        term_kk = self._weighted_linear_average(X_k, X_k, K_kk)
        term_ik = self._weighted_linear_average(X_i, X_k, K_ik)

        if not np.isfinite(term_ii) or not np.isfinite(term_kk) or not np.isfinite(term_ik):
            return np.nan

        dist_sq = term_ii + term_kk - 2.0 * term_ik
        if dist_sq < 0 and abs(dist_sq) < 1e-10:
            dist_sq = 0.0
        return float(max(dist_sq, 0.0))

    def _combine_distance_sq(self, paired_sq, unpaired_sq, paired_weight):
        if self.mode == 'paired':
            return paired_sq
        if self.mode == 'unpaired':
            return unpaired_sq

        if paired_weight <= 0:
            return unpaired_sq
        if paired_weight >= 1:
            return paired_sq
        if not np.isfinite(paired_sq) or not np.isfinite(unpaired_sq):
            return np.nan
        return float(paired_weight * paired_sq + (1.0 - paired_weight) * unpaired_sq)

    def _pair_distance(self, model_i, model_k, data_i, data_k, paired_weight, coverage_models):
        paired_sq = self._paired_distance_sq(data_i, data_k)
        unpaired_sq = self._unpaired_distance_sq(model_i, model_k, data_i, data_k, coverage_models)
        dist_sq = self._combine_distance_sq(paired_sq, unpaired_sq, paired_weight)

        if not np.isfinite(dist_sq):
            return np.nan
        return float(np.sqrt(max(dist_sq, 0.0)))

    def _compute_dist_matrix(self, model_names, model_data, paired_weight, coverage_models):
        # Evaluate the factored paired+unpaired estimator for every model pair.
        
        n_models    = len(model_names)
        dist_matrix = np.zeros((n_models, n_models), dtype=float)
        
        for i, model_i in enumerate(model_names):
            for k in range(i + 1, n_models):
                model_k = model_names[k]
                
                dist_matrix[i, k] = dist_matrix[k, i] = self._pair_distance(
                    model_i         = model_i,
                    model_k         = model_k,
                    data_i          = model_data[model_i],
                    data_k          = model_data[model_k],
                    paired_weight   = paired_weight,
                    coverage_models = coverage_models,
                )
        
        return dist_matrix

    def fit(self, data):
        df, model_names, embedding_dim, query_dim = self._validate_input_dataframe(data)

        # Step 1: choose the query representation used by the unpaired coverage
        # adjustment. In Block I this is either the oracle latent query space or
        # the PCA surrogate derived from responses.
        query_features = self._make_query_features(df, embedding_dim, query_dim)
        df['query_feature'] = list(query_features)

        # Step 2: split query ids into globally paired and model-unique unpaired
        # sets, then reorganize the flat table into per-model paired/unpaired
        # blocks.
        paired_query_codes, unpaired_query_codes = self._split_query_sets(df, model_names)
        model_data = self._prepare_model_data(
            df,
            model_names,
            paired_query_codes,
            unpaired_query_codes,
        )

        # Step 3: infer the mixing weight between the paired DKPS component and
        # the unpaired component from the observed sample counts.
        paired_weight = self._infer_paired_weight(paired_query_codes, model_data)

        if len({len(xx['unpaired_query_codes']) for xx in model_data.values()}) > 1:
            warnings.warn(
                'models have differing numbers of unpaired queries; paired_weight uses the average per-model unpaired count',
                stacklevel=2,
            )

        # Step 4: fit per-model KDEs on the unpaired query sets if coverage
        # adjustment is enabled.
        coverage_models, coverage_bandwidth = self._fit_coverage_models(model_data)

        # Step 5: evaluate the factored estimator for every model pair.
        dist_matrix = self._compute_dist_matrix(
            model_names,
            model_data,
            paired_weight,
            coverage_models,
        )

        # Persist the fitted artifacts after the local computation is complete so
        # the data flow through fit() stays explicit rather than side-effectful.
        self.model_names_         = model_names
        self.embedding_dim_       = embedding_dim
        self.query_dim_           = query_dim
        self.query_feature_dim_   = query_features.shape[1]
        self.model_data_          = model_data
        self.paired_query_codes_  = paired_query_codes
        self.unpaired_query_codes_ = sorted(unpaired_query_codes)
        self.paired_weight_       = paired_weight
        self.coverage_models_     = coverage_models
        self.coverage_bandwidth_  = coverage_bandwidth
        self.dist_matrix_         = dist_matrix
        return self

    def fit_transform(self, data):
        self.fit(data)
        if not np.isfinite(self.dist_matrix_).all():
            raise ValueError('dist_matrix_ contains non-finite values; cannot run ClassicalMDS')

        cmds_embds = ClassicalMDS(
            n_components=self.n_components_cmds,
            n_elbows=self.n_elbows_cmds,
            dissimilarity=self.dissimilarity,
        ).fit_transform(self.dist_matrix_)

        self.embedding_ = cmds_embds
        return {model_name: cmds_embds[i] for i, model_name in enumerate(self.model_names_)}
