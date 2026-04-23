"""
Confirm that the new DataFrame-based DKPS produces numerically equivalent
results to the legacy dict-based implementation across a variety of input shapes.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances

from dkps.dkps import DataKernelPerspectiveSpace as NewDKPS
from dkps.legacy_dkps import DataKernelPerspectiveSpace as LegacyDKPS


def _dict_to_dataframe(data, include_replicate_id=True):
    """Convert legacy dict format to DataFrame format.

    Keys are model names, values are (n_queries, n_replicates, embedding_dim).
    Model names are sorted to match the new implementation's ordering.
    """
    rows = []
    for model_name in sorted(data.keys()):
        arr = data[model_name]
        n_queries, n_replicates, _ = arr.shape
        for q in range(n_queries):
            for r in range(n_replicates):
                row = {'model_id': model_name, 'query_id': q, 'embedding': arr[q, r]}
                if include_replicate_id:
                    row['replicate_id'] = r
                rows.append(row)
    return pd.DataFrame(rows)


def _make_data(n_models, n_queries, n_replicates, embedding_dim, seed=42):
    """Generate random test data in legacy dict format with sorted keys."""
    rng = np.random.RandomState(seed)
    # Use sorted keys so legacy insertion order matches new sorted order
    model_names = sorted([f'model_{i:02d}' for i in range(n_models)])
    data = {}
    for m in model_names:
        data[m] = rng.randn(n_queries, n_replicates, embedding_dim)
    return data


def _compare_results(legacy_result, new_result, atol=1e-12):
    """Compare dict results from legacy and new implementations."""
    assert set(legacy_result.keys()) == set(new_result.keys()), \
        f'Key mismatch: {set(legacy_result.keys())} vs {set(new_result.keys())}'
    for key in legacy_result:
        np.testing.assert_allclose(
            legacy_result[key], new_result[key], atol=atol,
            err_msg=f'Mismatch for model {key}',
        )


# -- Tests: no aggregation (fn=None, takes first replicate) --

@pytest.mark.parametrize("n_models,n_queries,n_replicates,embedding_dim", [
    (3, 10, 1, 8),      # minimal: single replicate
    (3, 10, 5, 8),      # multiple replicates, fn=None takes first
    (5, 50, 1, 32),     # more models, more queries
    (4, 5, 1, 128),     # fewer queries, high-dim embeddings
    (10, 20, 3, 16),    # many models
])
def test_no_aggregation(n_models, n_queries, n_replicates, embedding_dim):
    data = _make_data(n_models, n_queries, n_replicates, embedding_dim)
    df = _dict_to_dataframe(data)

    legacy = LegacyDKPS(n_components_cmds=2)
    new = NewDKPS(n_components_cmds=2)

    legacy_result = legacy.fit_transform(data, return_dict=True)
    new_result = new.fit_transform(df)

    _compare_results(legacy_result, new_result)


# -- Tests: with mean aggregation over replicates --

@pytest.mark.parametrize("n_models,n_queries,n_replicates,embedding_dim", [
    (3, 10, 5, 8),
    (4, 20, 3, 16),
    (3, 100, 10, 4),
])
def test_mean_aggregation(n_models, n_queries, n_replicates, embedding_dim):
    data = _make_data(n_models, n_queries, n_replicates, embedding_dim)
    df = _dict_to_dataframe(data)

    legacy = LegacyDKPS(response_distribution_fn=np.mean, n_components_cmds=2)
    new = NewDKPS(response_distribution_fn=np.mean, n_components_cmds=2)

    legacy_result = legacy.fit_transform(data, return_dict=True)
    new_result = new.fit_transform(df)

    _compare_results(legacy_result, new_result)


# -- Tests: with median aggregation --

@pytest.mark.parametrize("n_models,n_queries,n_replicates,embedding_dim", [
    (3, 15, 7, 8),
    (5, 10, 3, 12),
])
def test_median_aggregation(n_models, n_queries, n_replicates, embedding_dim):
    data = _make_data(n_models, n_queries, n_replicates, embedding_dim)
    df = _dict_to_dataframe(data)

    legacy = LegacyDKPS(response_distribution_fn=np.median, n_components_cmds=2)
    new = NewDKPS(response_distribution_fn=np.median, n_components_cmds=2)

    legacy_result = legacy.fit_transform(data, return_dict=True)
    new_result = new.fit_transform(df)

    _compare_results(legacy_result, new_result)


# -- Tests: fit_transform always returns dict --

def test_fit_transform_returns_dict():
    data = _make_data(4, 20, 1, 16)
    df = _dict_to_dataframe(data)

    new = NewDKPS(n_components_cmds=2)

    result = new.fit_transform(df)
    assert isinstance(result, dict)
    assert sorted(result.keys()) == sorted(data.keys())


# -- Tests: non-euclidean metric --

def test_cosine_metric():
    data = _make_data(3, 10, 1, 8)
    df = _dict_to_dataframe(data)

    legacy = LegacyDKPS(metric_cmds='cosine', n_components_cmds=2)
    new = NewDKPS(metric_cmds='cosine', n_components_cmds=2)

    legacy_result = legacy.fit_transform(data, return_dict=True)
    new_result = new.fit_transform(df)

    _compare_results(legacy_result, new_result)


# -- Tests: single replicate without replicate_id column --

def test_single_replicate_no_replicate_column():
    data = _make_data(3, 10, 1, 8)
    df = _dict_to_dataframe(data, include_replicate_id=False)
    assert 'replicate_id' not in df.columns

    legacy = LegacyDKPS(n_components_cmds=2)
    new = NewDKPS(n_components_cmds=2)

    legacy_result = legacy.fit_transform(data, return_dict=True)
    new_result = new.fit_transform(df)

    _compare_results(legacy_result, new_result)


# -- Tests: dist_matrix_ attribute --

def test_dist_matrix_stored():
    data = _make_data(4, 15, 1, 8)
    df = _dict_to_dataframe(data)

    new = NewDKPS(n_components_cmds=2)
    new.fit_transform(df)

    assert hasattr(new, 'dist_matrix_')
    assert new.dist_matrix_.shape == (4, 4)
    np.testing.assert_allclose(new.dist_matrix_, new.dist_matrix_.T, atol=1e-15)
    np.testing.assert_array_equal(np.diag(new.dist_matrix_), 0.0)


# -- Tests: error on duplicate rows without replicate_id --

def test_duplicate_rows_without_replicate_id_raises():
    data = _make_data(3, 10, 3, 8)
    df = _dict_to_dataframe(data, include_replicate_id=False)
    assert 'replicate_id' not in df.columns

    new = NewDKPS(n_components_cmds=2)
    with pytest.raises(AssertionError, match='duplicate.*without replicate_id'):
        new.fit_transform(df)


def test_single_replicate_reducer_is_skipped():
    data = _make_data(3, 10, 1, 8)
    df = _dict_to_dataframe(data)

    def reducer(x, axis):
        raise AssertionError('reducer should not be called for singleton groups')

    baseline = NewDKPS(n_components_cmds=2)
    with_reducer = NewDKPS(response_distribution_fn=reducer, n_components_cmds=2)

    baseline_result = baseline.fit_transform(df)
    result = with_reducer.fit_transform(df)

    _compare_results(baseline_result, result)
    np.testing.assert_allclose(baseline.dist_matrix_, with_reducer.dist_matrix_, atol=1e-12)


def test_aggregate_replicates_raises_on_singleton_when_reducer_present():
    new = NewDKPS(response_distribution_fn=np.mean, n_components_cmds=2)
    emb_matrix = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(AssertionError, match='at least 2 replicates'):
        new._aggregate_replicates(emb_matrix)


def test_uneven_replicate_counts_warn_and_match_manual_distance():
    rows = []
    model_offsets = {'model_00': 0.0, 'model_01': 10.0, 'model_02': 20.0}
    replicate_counts = {
        ('model_00', 0): 3,
        ('model_00', 1): 1,
        ('model_00', 2): 2,
        ('model_00', 3): 1,
        ('model_01', 0): 1,
        ('model_01', 1): 2,
        ('model_01', 2): 3,
        ('model_01', 3): 2,
        ('model_02', 0): 2,
        ('model_02', 1): 3,
        ('model_02', 2): 1,
        ('model_02', 3): 2,
    }

    for (model_id, query_id), n_replicates in replicate_counts.items():
        offset = model_offsets[model_id]
        for replicate_id in range(n_replicates):
            rows.append({
                'model_id': model_id,
                'query_id': query_id,
                'replicate_id': replicate_id,
                'embedding': np.array([
                    offset + query_id + replicate_id,
                    offset + 2 * query_id - replicate_id,
                ]),
            })

    df = pd.DataFrame(rows)
    new = NewDKPS(response_distribution_fn=np.mean, n_components_cmds=2)

    with pytest.warns(UserWarning, match='uneven replicate counts'):
        new.fit_transform(df)

    model_names = sorted(df['model_id'].unique())
    queries = sorted(df['query_id'].unique())
    X = np.stack([
        np.stack([
            np.mean(
                np.stack(
                    df.loc[
                        (df['model_id'] == model_id) & (df['query_id'] == query_id),
                        'embedding',
                    ].values
                ),
                axis=0,
            )
            for query_id in queries
        ])
        for model_id in model_names
    ])

    expected = pairwise_distances(X.reshape(len(X), -1), metric='euclidean') / np.sqrt(len(queries))
    expected = (expected + expected.T) / 2

    np.testing.assert_allclose(new.dist_matrix_, expected, atol=1e-12)


def test_mismatched_query_sets_raise():
    df = pd.DataFrame([
        {'model_id': 'model_00', 'query_id': 0, 'embedding': np.array([0.0, 1.0])},
        {'model_id': 'model_00', 'query_id': 1, 'embedding': np.array([1.0, 2.0])},
        {'model_id': 'model_01', 'query_id': 0, 'embedding': np.array([2.0, 3.0])},
    ])

    new = NewDKPS(n_components_cmds=2)
    with pytest.raises(AssertionError, match='same set of query_ids'):
        new.fit_transform(df)


def test_duplicate_replicate_ids_raise():
    df = pd.DataFrame([
        {'model_id': 'model_00', 'query_id': 0, 'replicate_id': 0, 'embedding': np.array([0.0, 1.0])},
        {'model_id': 'model_00', 'query_id': 0, 'replicate_id': 0, 'embedding': np.array([1.0, 2.0])},
        {'model_id': 'model_01', 'query_id': 0, 'replicate_id': 0, 'embedding': np.array([2.0, 3.0])},
        {'model_id': 'model_01', 'query_id': 0, 'replicate_id': 1, 'embedding': np.array([3.0, 4.0])},
    ])

    new = NewDKPS(response_distribution_fn=np.mean, n_components_cmds=2)
    with pytest.raises(AssertionError, match='replicate_id values must be unique'):
        new.fit_transform(df)
