"""Run: python -m unittest discover -s experiments/linear_probes -p 'test_*.py'."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import RidgePath, balanced_accuracy, evaluate, grouped_folds, normalize_blocks
from run import load_embeddings


class LinearProbeTests(unittest.TestCase):
    def test_dual_matches_primal_with_unpenalized_intercept(self):
        rng = np.random.default_rng(2)
        x, test = rng.normal(size=(31, 8)), rng.normal(size=(7, 8))
        y = np.array([0] * 17 + [1] * 10 + [2] * 4)
        path = RidgePath(x @ x.T, test @ x.T, y, 3)
        weights = len(y) / (3 * np.bincount(y)[y])
        xm = np.average(x, axis=0, weights=weights)
        labels = np.eye(3)[y]
        ym = np.average(labels, axis=0, weights=weights)
        xc = x - xm
        for alpha in [1e-5, .01, 1]:
            w = np.linalg.solve(xc.T @ (weights[:, None] * xc) + len(y) * alpha * np.eye(8),
                                xc.T @ (weights[:, None] * (labels - ym)))
            np.testing.assert_allclose(path.scores(alpha), (test - xm) @ w + ym, atol=1e-9)

    def test_preprocessing_is_sample_local(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(12, 3, 9))
        a = normalize_blocks(x)
        x[6:] *= rng.normal(size=(6, 3, 9))
        np.testing.assert_array_equal(a[:6], normalize_blocks(x)[:6])
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1)

    def test_group_disjointness_and_coverage(self):
        groups = np.repeat(np.arange(20), 7)
        seen = []
        for train, test in grouped_folds(groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            for it, iv in grouped_folds(groups[train], 3):
                self.assertFalse(set(groups[train[it]]) & set(groups[train[iv]]))
                self.assertTrue(set(train[it]) <= set(train))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(groups))))

    def test_recovers_linear_signal_out_of_group(self):
        rng = np.random.default_rng(4)
        y = np.tile(np.arange(3), 20)
        groups = np.repeat(np.arange(20), 3)
        x = np.eye(3)[y] + rng.normal(scale=.03, size=(60, 3))
        result = evaluate(x @ x.T, y.astype(str), groups, alphas=[.001, .1], bootstrap_draws=100)
        self.assertEqual(result['balanced_accuracy'], 1.)
        self.assertEqual(result['balanced_accuracy_ci95'], [1., 1.])

    def test_changing_test_covariates_does_not_change_other_predictions(self):
        rng = np.random.default_rng(5)
        x, test = rng.normal(size=(30, 6)), rng.normal(size=(5, 6))
        y = np.tile(np.arange(3), 10)
        original = RidgePath(x @ x.T, test @ x.T, y, 3).scores(.01)
        test[-1] *= 100
        updated = RidgePath(x @ x.T, test @ x.T, y, 3).scores(.01)
        np.testing.assert_array_equal(original[:-1], updated[:-1])

    def test_missing_training_class_rejected(self):
        with self.assertRaises(ValueError):
            RidgePath(np.eye(3), np.ones((2, 3)), np.array([0, 0, 1]), 3)

    def test_balanced_accuracy_ignores_class_prevalence(self):
        self.assertEqual(balanced_accuracy(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]), 2), .5)

    def test_explicit_row_join_and_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'embeddings.npz'
            np.savez(p, row_id=np.array(['b', 'a']), X=np.eye(2))
            np.testing.assert_array_equal(load_embeddings(p, ['a', 'b']), np.eye(2)[::-1])
            with self.assertRaises(ValueError):
                load_embeddings(p, ['a', 'c'])


if __name__ == '__main__':
    unittest.main()
