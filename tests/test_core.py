"""Tests for DKPS core orchestrator."""

import numpy as np
import pandas as pd
import pytest

from dkps import DKPS


class TestDKPSDataFrameAPI:
    def test_paired_df(self, simple_paired_df):
        dkps = DKPS(distance='paired', n_components=2)
        result = dkps.fit_transform(simple_paired_df)
        assert isinstance(result, dict)
        assert len(result) == 3
        for v in result.values():
            assert v.shape == (2,)

    def test_unpaired_df_energy(self, simple_unpaired_df):
        dkps = DKPS(distance='energy', n_components=2)
        result = dkps.fit_transform(simple_unpaired_df)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_unpaired_df_mmd(self, simple_unpaired_df):
        dkps = DKPS(distance='mmd', n_components=2)
        result = dkps.fit_transform(simple_unpaired_df)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_return_array(self, simple_paired_df):
        dkps = DKPS(distance='paired', n_components=2)
        result = dkps.fit_transform(simple_paired_df, return_dict=False)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3
        assert result.shape[1] == 2

    def test_distance_matrix_only(self, simple_paired_df):
        dkps = DKPS(distance='paired')
        D = dkps.distance_matrix(simple_paired_df)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0)

    def test_auto_detect_paired(self, simple_paired_df):
        """query_id present → paired auto-detection."""
        dkps = DKPS(distance='paired', n_components=2)
        result = dkps.fit_transform(simple_paired_df)
        assert len(result) == 3


class TestDKPSInputValidation:
    def test_dict_input_raises(self):
        """Passing a dict should raise TypeError."""
        data = {'a': np.zeros((10, 1, 8)), 'b': np.zeros((10, 1, 8))}
        dkps = DKPS(distance='paired', n_components=2)
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            dkps.fit_transform(data)

    def test_dict_input_distance_matrix_raises(self):
        """Passing a dict to distance_matrix should raise TypeError."""
        data = {'a': np.zeros((10, 1, 8)), 'b': np.zeros((10, 1, 8))}
        dkps = DKPS(distance='paired')
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            dkps.distance_matrix(data)

    def test_list_input_raises(self):
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            dkps.fit_transform([1, 2, 3])

    def test_ndarray_input_raises(self):
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            dkps.fit_transform(np.zeros((3, 3)))

    def test_string_input_raises(self):
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            dkps.fit_transform("not a dataframe")

    def test_none_input_raises(self):
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            dkps.fit_transform(None)

    def test_missing_model_column_raises(self):
        df = pd.DataFrame({'response_embedding': [[1, 2], [3, 4]]})
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(ValueError, match="must have a 'model' column"):
            dkps.fit_transform(df)

    def test_nan_in_embeddings_raises(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response_embedding': [[1, 2], [float('nan'), 4], [5, 6], [7, 8]],
        })
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(ValueError, match="NaN/Inf"):
            dkps.fit_transform(df)

    def test_single_model_raises(self):
        df = pd.DataFrame({
            'model': ['a', 'a'],
            'response_embedding': [[1, 2], [3, 4]],
        })
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(ValueError, match="Need at least 2 models"):
            dkps.fit_transform(df)

    def test_inconsistent_embed_dims_raises(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'response_embedding': [[1, 2], [3, 4], [5, 6, 7], [8, 9, 10]],
        })
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            dkps.fit_transform(df)

    def test_missing_response_columns_raises(self):
        df = pd.DataFrame({'model': ['a', 'b']})
        dkps = DKPS(distance='energy', n_components=2)
        with pytest.raises(ValueError, match="must have 'response' or 'response_embedding'"):
            dkps.fit_transform(df)

    def test_mismatched_query_ids_raises(self):
        df = pd.DataFrame({
            'model': ['a', 'a', 'b', 'b'],
            'query_id': ['q1', 'q2', 'q1', 'q3'],
            'response_embedding': [[1, 2], [3, 4], [5, 6], [7, 8]],
        })
        dkps = DKPS(distance='paired', n_components=2)
        with pytest.raises(ValueError, match="missing query_ids"):
            dkps.fit_transform(df)
