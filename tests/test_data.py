"""Tests for ModelResponseData."""

import numpy as np
import pandas as pd
import pytest

from dkps.data import ModelResponseData


class TestFromDataframe:
    def test_basic_paired(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        assert mrd.paired is True
        assert len(mrd.model_names) == 3
        for model in mrd.model_names:
            # (n_queries, n_replicates, embed_dim)
            assert mrd.response_embeddings[model].ndim == 3
            assert mrd.response_embeddings[model].shape[0] == 10  # n_queries
            assert mrd.response_embeddings[model].shape[1] == 1   # n_replicates

    def test_basic_unpaired(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        assert mrd.paired is False
        assert len(mrd.model_names) == 3
        for model in mrd.model_names:
            assert mrd.response_embeddings[model].ndim == 2

    def test_auto_detect_paired(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df, paired=None)
        assert mrd.paired is True

    def test_auto_detect_unpaired(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df, paired=None)
        assert mrd.paired is False

    def test_missing_model_column(self):
        df = pd.DataFrame({'response_embedding': [[1, 2], [3, 4]]})
        with pytest.raises(ValueError, match="must have a 'model' column"):
            ModelResponseData.from_dataframe(df)

    def test_missing_response_columns(self):
        df = pd.DataFrame({'model': ['a', 'b']})
        with pytest.raises(ValueError, match="must have 'response' or 'response_embedding'"):
            ModelResponseData.from_dataframe(df)

    def test_nan_in_model(self):
        df = pd.DataFrame({
            'model': ['a', None, 'b', 'b'],
            'response_embedding': [[1, 2], [3, 4], [5, 6], [7, 8]],
        })
        with pytest.raises(ValueError, match="'model' column contains missing values"):
            ModelResponseData.from_dataframe(df)

    def test_nan_in_response_embedding(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response_embedding': [[1, 2], None, [5, 6], [7, 8]],
        })
        with pytest.raises(ValueError, match="'response_embedding' contains missing values"):
            ModelResponseData.from_dataframe(df)

    def test_single_model_raises(self):
        df = pd.DataFrame({
            'model': ['a', 'a'],
            'response_embedding': [[1, 2], [3, 4]],
        })
        with pytest.raises(ValueError, match="Need at least 2 models"):
            ModelResponseData.from_dataframe(df)

    def test_inconsistent_embed_dims(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response_embedding': [[1, 2], [3, 4], [5, 6, 7], [8, 9, 10]],
        })
        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            ModelResponseData.from_dataframe(df)

    def test_mismatched_query_ids(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'query_id': ['q1', 'q2', 'q1', 'q3'],
            'response_embedding': [[1, 2], [3, 4], [5, 6], [7, 8]],
        })
        with pytest.raises(ValueError, match="missing query_ids"):
            ModelResponseData.from_dataframe(df, paired=True)

    def test_paired_requires_query_id(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response_embedding': [[1, 2], [3, 4], [5, 6], [7, 8]],
        })
        with pytest.raises(ValueError, match="Paired distance requires 'query_id'"):
            ModelResponseData.from_dataframe(df, paired=True)

    def test_nan_inf_in_embedding(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response_embedding': [[1, 2], [float('nan'), 4], [5, 6], [7, 8]],
        })
        with pytest.raises(ValueError, match="NaN/Inf"):
            ModelResponseData.from_dataframe(df)

    def test_with_query_embeddings(self, query_paired_df):
        mrd = ModelResponseData.from_dataframe(query_paired_df)
        assert mrd.query_embeddings is not None
        for model in mrd.model_names:
            assert mrd.query_embeddings[model].shape[0] == 10

    def test_embed_fn_called_when_no_embeddings(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response': ['hello', 'world', 'foo', 'bar'],
        })
        embed_calls = []

        def mock_embed(texts):
            embed_calls.append(texts)
            return np.random.randn(len(texts), 4)

        mrd = ModelResponseData.from_dataframe(df, embed_fn=mock_embed)
        assert len(embed_calls) == 1
        assert mrd.paired is False

    def test_no_embed_fn_raises(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response': ['hello', 'world', 'foo', 'bar'],
        })
        with pytest.raises(ValueError, match="no embed_fn provided"):
            ModelResponseData.from_dataframe(df)


class TestAggregateReplicates:
    def test_default_first_replicate(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        agg = mrd.aggregate_replicates()
        for model in mrd.model_names:
            # Single replicate: (n_queries, 1, embed_dim) -> (n_queries, embed_dim)
            assert agg[model].shape == (10, 8)
            np.testing.assert_array_equal(
                agg[model], mrd.response_embeddings[model][:, 0]
            )

    def test_mean_aggregation(self, simple_paired_df):
        mrd = ModelResponseData.from_dataframe(simple_paired_df)
        agg = mrd.aggregate_replicates(fn=np.mean)
        for model in mrd.model_names:
            assert agg[model].shape == (10, 8)

    def test_noop_for_unpaired(self, simple_unpaired_df):
        mrd = ModelResponseData.from_dataframe(simple_unpaired_df)
        agg = mrd.aggregate_replicates()
        assert agg is mrd.response_embeddings
