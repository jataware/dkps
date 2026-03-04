"""
    data.py — ModelResponseData: validates, groups, and embeds from DataFrame.
"""

import numpy as np
import pandas as pd


class ModelResponseData:
    """Internal container produced from a DataFrame.

    Attributes
    ----------
    model_names : list[str]
    response_embeddings : dict[str, np.ndarray]
        Paired:   {model: (n_queries, n_replicates, embed_dim)}
        Unpaired: {model: (n_responses, embed_dim)}
    query_embeddings : dict[str, np.ndarray] or None
        {model: (n_queries, embed_dim)} — when queries are available
    paired : bool
    """

    def __init__(self, model_names, response_embeddings, paired, query_embeddings=None):
        self.model_names = model_names
        self.response_embeddings = response_embeddings
        self.query_embeddings = query_embeddings
        self.paired = paired

    @classmethod
    def from_dataframe(cls, df, embed_fn=None, paired=None):
        """Build from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'model' column. Must have 'response' or 'response_embedding'.
            Optional: 'query_id', 'query', 'query_embedding'.
        embed_fn : callable or None
            Called on list of strings -> np.ndarray of shape (n, embed_dim).
            Used when 'response_embedding' column is absent.
        paired : bool or None
            If None, auto-detect from presence of 'query_id' column and whether
            all models share the same query_ids.
        """
        # --- Column existence checks ---
        if 'model' not in df.columns:
            raise ValueError("DataFrame must have a 'model' column")

        has_response = 'response' in df.columns
        has_response_emb = 'response_embedding' in df.columns
        if not has_response and not has_response_emb:
            raise ValueError("DataFrame must have 'response' or 'response_embedding' column")

        # --- Missing value checks ---
        if df['model'].isna().any():
            raise ValueError("'model' column contains missing values")

        if has_response_emb:
            null_mask = df['response_embedding'].isna()
            if null_mask.any():
                bad_rows = list(df.index[null_mask])
                raise ValueError(f"'response_embedding' contains missing values at rows {bad_rows}")

        if not has_response_emb and has_response:
            null_mask = df['response'].isna()
            if null_mask.any():
                bad_rows = list(df.index[null_mask])
                raise ValueError(f"'response' column contains missing values at rows {bad_rows}")

        # --- Model count check ---
        model_names = list(df['model'].unique())
        if len(model_names) < 2:
            raise ValueError(f"Need at least 2 models, got {len(model_names)}")

        # --- Embed responses if needed ---
        if not has_response_emb:
            if embed_fn is None:
                raise ValueError(
                    "DataFrame has no 'response_embedding' column and no embed_fn provided"
                )
            embeddings = embed_fn(df['response'].tolist())
            df = df.copy()
            df['response_embedding'] = list(embeddings)

        # --- Validate embedding dimensions ---
        _validate_embedding_dims(df, 'response_embedding')

        # --- Check for NaN/Inf in embeddings ---
        for idx, emb in enumerate(df['response_embedding']):
            arr = np.asarray(emb)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"'response_embedding' contains NaN/Inf at row {df.index[idx]}"
                )

        # --- Detect pairing ---
        has_query_id = 'query_id' in df.columns
        if paired is None:
            if has_query_id:
                # Check if all models share the same set of query_ids
                query_sets = {}
                for model in model_names:
                    model_df = df[df['model'] == model]
                    query_sets[model] = set(model_df['query_id'].dropna())
                all_same = all(qs == query_sets[model_names[0]] for qs in query_sets.values())
                paired = all_same and len(query_sets[model_names[0]]) > 0
            else:
                paired = False

        if paired and not has_query_id:
            raise ValueError("Paired distance requires 'query_id' column")

        # --- Build response_embeddings dict ---
        response_embeddings = {}
        if paired:
            # Validate query_id alignment
            query_sets = {}
            for model in model_names:
                model_df = df[df['model'] == model]
                query_sets[model] = set(model_df['query_id'])
            reference_qids = query_sets[model_names[0]]
            for model in model_names[1:]:
                missing = reference_qids - query_sets[model]
                if missing:
                    raise ValueError(
                        f"Paired mode requires all models to share the same query_ids. "
                        f"Model '{model}' is missing query_ids: {sorted(missing)}"
                    )
                extra = query_sets[model] - reference_qids
                if extra:
                    raise ValueError(
                        f"Paired mode requires all models to share the same query_ids. "
                        f"Model '{model}' has extra query_ids: {sorted(extra)}"
                    )

            # Sort by query_id for consistent alignment
            sorted_qids = sorted(reference_qids)
            for model in model_names:
                model_df = df[df['model'] == model]
                # Group by query_id to handle replicates
                groups = model_df.groupby('query_id')
                n_replicates_per_q = [len(g) for _, g in groups]
                if len(set(n_replicates_per_q)) > 1:
                    raise ValueError(
                        f"Duplicate (model, query_id) pairs found — if this is intentional "
                        f"(replicates), ensure consistent counts"
                    )

                emb_list = []
                for qid in sorted_qids:
                    q_rows = model_df[model_df['query_id'] == qid]
                    embs = np.stack([np.asarray(e) for e in q_rows['response_embedding']])
                    emb_list.append(embs)

                # Shape: (n_queries, n_replicates, embed_dim)
                response_embeddings[model] = np.stack(emb_list)

            if len(response_embeddings[model]) == 0:
                raise ValueError(f"Model '{model}' has 0 responses")
        else:
            for model in model_names:
                model_df = df[df['model'] == model]
                if len(model_df) == 0:
                    raise ValueError(f"Model '{model}' has 0 responses")
                embs = np.stack([np.asarray(e) for e in model_df['response_embedding']])
                response_embeddings[model] = embs

        # --- Build query_embeddings dict if available ---
        query_embeddings = None
        has_query_emb = 'query_embedding' in df.columns
        has_query = 'query' in df.columns

        if has_query_emb or has_query:
            if has_query_emb:
                # Check for missing values
                null_mask = df['query_embedding'].isna()
                if null_mask.any():
                    # Only error if we actually need them
                    pass
                else:
                    query_embeddings = {}
                    for model in model_names:
                        model_df = df[df['model'] == model]
                        if paired:
                            qembs = []
                            for qid in sorted_qids:
                                q_rows = model_df[model_df['query_id'] == qid]
                                # Take first row's query embedding (should be same for all replicates)
                                qembs.append(np.asarray(q_rows.iloc[0]['query_embedding']))
                            query_embeddings[model] = np.stack(qembs)
                        else:
                            query_embeddings[model] = np.stack(
                                [np.asarray(e) for e in model_df['query_embedding']]
                            )
            elif has_query and embed_fn is not None:
                # Embed queries
                unique_queries = df['query'].unique().tolist()
                query_embs_arr = embed_fn(unique_queries)
                query_emb_map = {q: query_embs_arr[i] for i, q in enumerate(unique_queries)}

                query_embeddings = {}
                for model in model_names:
                    model_df = df[df['model'] == model]
                    if paired:
                        qembs = []
                        for qid in sorted_qids:
                            q_rows = model_df[model_df['query_id'] == qid]
                            qembs.append(query_emb_map[q_rows.iloc[0]['query']])
                        query_embeddings[model] = np.stack(qembs)
                    else:
                        query_embeddings[model] = np.stack(
                            [query_emb_map[q] for q in model_df['query']]
                        )

        return cls(
            model_names=model_names,
            response_embeddings=response_embeddings,
            paired=paired,
            query_embeddings=query_embeddings,
        )

    def aggregate_replicates(self, fn=None, axis=1):
        """For paired data: collapse replicates along axis.

        Parameters
        ----------
        fn : callable or None
            Aggregation function. None = take first replicate.
        axis : int
            Axis to aggregate over (default 1 = replicate axis).

        Returns
        -------
        dict[str, np.ndarray]
            {model: (n_queries, embed_dim)}
        """
        if not self.paired:
            return self.response_embeddings

        result = {}
        for model, arr in self.response_embeddings.items():
            if fn is None:
                result[model] = arr[:, 0]
            else:
                result[model] = fn(arr, axis=axis)
        return result


def _validate_embedding_dims(df, col):
    """Check that all embeddings in a column have the same dimensionality."""
    dims_by_model = {}
    for _, row in df.iterrows():
        model = row['model']
        emb = np.asarray(row[col])
        dim = emb.shape[-1]
        if model not in dims_by_model:
            dims_by_model[model] = dim
        elif dims_by_model[model] != dim:
            raise ValueError(
                f"Inconsistent embedding dimensions within model '{model}': "
                f"got {dim} and {dims_by_model[model]}"
            )

    unique_dims = set(dims_by_model.values())
    if len(unique_dims) > 1:
        examples = list(dims_by_model.items())
        raise ValueError(
            f"Inconsistent embedding dimensions: "
            f"model '{examples[0][0]}' has dim {examples[0][1]}, "
            f"model '{examples[1][0]}' has dim {examples[1][1]}"
        )
